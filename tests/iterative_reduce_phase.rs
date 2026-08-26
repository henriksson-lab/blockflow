// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// This suite states the small-state iteration shape:
//
// ```text
// state[k + 1] = update(state[k], reduce(map(block, state[k])))
// ```
//
// The op below is intentionally dull. Each block emits a local sum and the state
// value it was handed. The reducer refuses if those echoed states differ from
// the one canonical state for the substage, then replaces the state with the
// whole-volume sum. A second substage observes that the state is stable and
// stops. The arithmetic is not the point; the barrier shape is.

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::env::{ArrayEnvironment, BlockBuf, Environment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::iterate::{
    iterative_reduce_phase, IterativeReduceOp, Partial, ReduceBlock, StateUpdate, SubstageLimit,
};
use blockflow::op::Chain;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::Dtype;

const VOLUME: [usize; 3] = [8, 4, 2];
const STREAM: &str = "iterative_reduce_state";

fn pack_pair(a: u64, b: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&a.to_le_bytes());
    bytes.extend_from_slice(&b.to_le_bytes());
    bytes
}

fn unpack_pair(bytes: &[u8]) -> (u64, u64) {
    assert_eq!(bytes.len(), 16, "pair encoding is fixed width");
    let mut a = [0u8; 8];
    let mut b = [0u8; 8];
    a.copy_from_slice(&bytes[..8]);
    b.copy_from_slice(&bytes[8..]);
    (u64::from_le_bytes(a), u64::from_le_bytes(b))
}

fn pack_state(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn unpack_state(bytes: &[u8]) -> u64 {
    assert_eq!(bytes.len(), 8, "state encoding is fixed width");
    let mut raw = [0u8; 8];
    raw.copy_from_slice(bytes);
    u64::from_le_bytes(raw)
}

struct SumUntilStable {
    limit: SubstageLimit,
}

impl SumUntilStable {
    fn new() -> Self {
        Self {
            limit: SubstageLimit::of(4).expect("positive limit"),
        }
    }
}

impl IterativeReduceOp for SumUntilStable {
    fn name(&self) -> &'static str {
        "sum-until-stable"
    }

    fn limit(&self) -> SubstageLimit {
        self.limit
    }

    fn initial_state(&self, _volume: [usize; 3]) -> blockflow::Result<Vec<u8>> {
        Ok(pack_state(0))
    }

    fn map_block(
        &self,
        _substage: usize,
        state: &[u8],
        block: &ReduceBlock<'_>,
    ) -> blockflow::Result<Vec<u8>> {
        let pixels = block.pixels()?;
        let BlockBuf::Array(values) = pixels else {
            return Err(blockflow::Error::InvalidArgument(
                "the test op needs real values".to_string(),
            ));
        };
        let local = values
            .view::<f64>()?
            .iter()
            .map(|value| *value as u64)
            .sum::<u64>();
        Ok(pack_pair(local, unpack_state(state)))
    }

    fn update(
        &self,
        _substage: usize,
        state: &[u8],
        partials: &[Partial],
    ) -> blockflow::Result<StateUpdate> {
        let previous = unpack_state(state);
        let mut total = 0u64;
        let mut last = None;
        for partial in partials {
            if let Some(prior) = last {
                assert!(
                    prior <= partial.index,
                    "partials must arrive in canonical block order"
                );
            }
            last = Some(partial.index);
            let (local, echoed) = unpack_pair(&partial.bytes);
            assert_eq!(
                echoed, previous,
                "every block must see the same state for one substage"
            );
            total += local;
        }
        let next = pack_state(total);
        Ok(if total == previous {
            StateUpdate::converged(next)
        } else {
            StateUpdate::continuing(next)
        })
    }

    fn state_stream(&self) -> &'static str {
        STREAM
    }

    fn state_lifecycle(&self) -> Lifecycle {
        Lifecycle::Persistent
    }
}

fn input() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(x, y, z)| {
        (1 + x + 2 * y + 3 * z) as f64
    })
}

fn plan(op: &SumUntilStable, block: [usize; 3]) -> Decomposition {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![iterative_reduce_phase(op, grid).expect("a reduce iteration phase")],
        chain_reach: [0, 0, 0],
    }
}

fn workflow() -> Workflow {
    Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64)
}

#[test]
fn blocks_reduce_to_one_state_and_the_next_substage_sees_it() {
    let op = SumUntilStable::new();
    let plan = plan(&op, [3, 2, 2]);
    let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4])
        .expect("an environment");

    let stats = execute_phases(
        "iterative-reduce",
        &workflow(),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::IterateReduce(&op)],
    )
    .expect("a run");

    assert_eq!(stats.substages, vec![2], "one update and one quiet pass");
    assert_eq!(stats.substage_changes, vec![vec![1, 0]]);

    let bytes = env
        .read_sidecar(STREAM, 0, [0, 0, 0])
        .expect("sidecar read")
        .expect("final state was written");
    let expected = input().iter().map(|value| *value as u64).sum::<u64>();
    assert_eq!(unpack_state(&bytes), expected);
}

#[test]
fn the_same_state_is_found_at_two_cuts() {
    let op = SumUntilStable::new();
    let expected = input().iter().map(|value| *value as u64).sum::<u64>();

    for block in [[8, 4, 2], [4, 2, 2], [3, 2, 1]] {
        let plan = plan(&op, block);
        let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4])
            .expect("an environment");
        execute_phases(
            "iterative-reduce",
            &workflow(),
            &plan,
            &Hints::default(),
            &env,
            &[],
            &[PhaseWork::IterateReduce(&op)],
        )
        .expect("a run");
        let bytes = env
            .read_sidecar(STREAM, 0, [0, 0, 0])
            .expect("sidecar read")
            .expect("final state was written");
        assert_eq!(unpack_state(&bytes), expected, "block {block:?}");
    }
}
