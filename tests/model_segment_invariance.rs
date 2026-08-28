//! **The answer does not depend on how the volume was cut.**
//!
//! This is the property `blockflow` exists to keep and the one this crate is
//! most able to break, because it is the crate that decides which block owns
//! what. Two runs of the same data at different block sizes must produce the
//! same objects, with the same ids, at the same positions, with the same
//! measurements — not nearly, exactly.
//!
//! Two things make that possible and both are asserted here rather than
//! asserted in a doc comment:
//!
//! * **the id is a function of the data.** It is derived from the object's
//!   centroid in the volume, so nothing is shared between blocks to make two
//!   runs agree, and there is no counter whose value depends on visit order;
//! * **every accumulator is an integer.** Count, coordinate sums, and the
//!   per-image sum, min and max are all `u64` folded by `+`, `min` and `max`.
//!   A running mean in `f64` would agree to about fifteen digits and disagree
//!   in the last, which is exactly the "nearly" this file refuses.
//!
//! Runs against the stub backend, so what varies between runs is the
//! decomposition and nothing else. A real network is *approximately* invariant
//! under retiling, and a test that had to tolerate that could not tell a
//! decomposition bug from an inference wobble.

use std::collections::BTreeMap;
use std::sync::Arc;

use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::model_segment::stub::ThresholdBackend;
use blockflow::model_segment::InstanceSegment;
use blockflow::op::Chain;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::{Row, Table};
use blockflow::{
    fragment_phase, BlockGrid, Decomposition, Dtype, Lifecycle, PhaseWork, Region, Voxels,
};
use ndarray::Array3;

const STREAM: &str = "objects";
const VOLUME: [usize; 3] = [1, 48, 56];

// ------------------------------------------------------------- fixtures --

/// Discs of several sizes, at positions with no relation to any block grid.
///
/// Discs rather than boxes here, deliberately: a box's centroid is exact on
/// every axis and a disc's is not, so this exercises the rounding rule — round
/// once, half up, in integer arithmetic — which is the step where two cuts
/// could disagree by one voxel.
fn image() -> Voxels {
    let mut array = Array3::<f64>::zeros(VOLUME);
    let discs: [([usize; 2], f64); 6] = [
        ([7, 9], 3.5),
        ([13, 27], 4.5),
        ([22, 15], 2.5),
        ([31, 38], 5.5),
        ([39, 7], 3.0),
        ([25, 49], 4.0),
    ];
    for ([cy, cx], radius) in discs {
        let reach = radius.ceil() as usize + 1;
        for y in cy.saturating_sub(reach)..(cy + reach).min(VOLUME[1]) {
            for x in cx.saturating_sub(reach)..(cx + reach).min(VOLUME[2]) {
                let dy = y as f64 - cy as f64;
                let dx = x as f64 - cx as f64;
                if dy * dy + dx * dx <= radius * radius {
                    array[[0, y, x]] = 1.0;
                }
            }
        }
    }
    array.into()
}

/// A gradient, so that the measured sums differ per object and per voxel — a
/// constant would make an accumulation bug invisible.
fn marker() -> Voxels {
    let mut array = Array3::<f64>::zeros(VOLUME);
    for y in 0..VOLUME[1] {
        for x in 0..VOLUME[2] {
            array[[0, y, x]] = (y * 3 + x * 7 % 11) as f64;
        }
    }
    array.into()
}

// ----------------------------------------------------------------- run --

#[derive(Clone, Debug, PartialEq, Eq)]
struct OwnedRow {
    at: [usize; 3],
    values: Vec<u64>,
}

impl OwnedRow {
    fn of(row: &Row<'_>) -> Self {
        Self {
            at: row.at(),
            values: (0..row.schema().len())
                .map(|column| row.u64(column).expect("a u64 column"))
                .collect(),
        }
    }

    fn id(&self) -> u64 {
        self.values[0]
    }
}

/// Every row the run emitted, in id order, with the block that emitted it
/// deliberately dropped: which block owned an object *is* a function of the
/// decomposition, and it is the only thing here that is allowed to be.
fn run(block: [usize; 3], halo: usize) -> Vec<OwnedRow> {
    let backend = Arc::new(ThresholdBackend::new(0.5));
    let op = InstanceSegment::new(
        "segment",
        backend,
        [0, halo, halo],
        STREAM,
        Lifecycle::Persistent,
        vec![(blockflow::ImageId::supplied(0), Dtype::F64)],
    );

    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let phase = fragment_phase(&op, grid.clone()).expect("a phase");
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("a legal plan");

    let env = ArrayEnvironment::with_inputs(image(), vec![marker()], &plan, [8, 8, 8])
        .expect("an environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64);
    execute_phases(
        "segment",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&op)],
    )
    .expect("a run");

    let schema = op.schema().expect("a schema");
    let mut table = Table::new(VOLUME, schema).expect("a table");
    for core in grid.cores() {
        let bytes = env
            .read_sidecar(STREAM, 0, core.index)
            .expect("the store answers")
            .unwrap_or_else(|| panic!("block {:?} wrote no blob", core.index));
        table.write(core.index, &bytes).expect("a row blob");
    }
    table.seal().expect("a seal");

    let mut rows: Vec<OwnedRow> = table
        .query(&Region::whole(&VOLUME))
        .expect("a query")
        .iter()
        .map(OwnedRow::of)
        .collect();
    rows.sort_by_key(OwnedRow::id);
    rows
}

// --------------------------------------------------------------- tests --

/// **The claim, at five block sizes**: same objects, same ids, same positions,
/// same measurements, exactly.
///
/// The sizes are chosen to divide the volume evenly, unevenly, on one axis
/// only, and not at all — so the reference is compared against a lattice with
/// one block, with a ragged last block, and with seams through the middle of
/// several objects.
#[test]
fn the_objects_are_the_same_at_every_block_size() {
    let halo = 14; // larger than the largest disc, which is 11 across
    let reference = run(VOLUME, halo); // one block: no seam anywhere
    assert!(
        reference.len() >= 6,
        "the scene should hold six discs, and the reference found {}",
        reference.len()
    );

    for block in [
        [1, 16, 16],
        [1, 24, 28],
        [1, 10, 13],
        [1, 48, 8],
        [1, 7, 56],
    ] {
        let rows = run(block, halo);
        assert_eq!(
            rows.len(),
            reference.len(),
            "block {block:?}: {} objects against the reference's {}",
            rows.len(),
            reference.len()
        );
        for (got, want) in rows.iter().zip(reference.iter()) {
            assert_eq!(
                got, want,
                "block {block:?}: an object differs from the reference"
            );
        }
    }
}

/// The ids are what make the comparison above possible, and they are a function
/// of the data: an object's id is derived from its centroid, so it is the same
/// number in a run that never cut the volume and in one that cut it five ways.
///
/// Stated separately from the row comparison because it is the sharper claim —
/// a scheme that numbered objects in visit order would pass nothing above but
/// would also fail nothing that only compared *sets* of measurements.
#[test]
fn an_objects_id_is_the_same_number_under_every_cut() {
    let halo = 14;
    let reference: BTreeMap<u64, [usize; 3]> = run(VOLUME, halo)
        .into_iter()
        .map(|row| (row.id(), row.at))
        .collect();

    for block in [[1, 16, 16], [1, 10, 13], [1, 48, 8]] {
        for row in run(block, halo) {
            assert_eq!(
                reference.get(&row.id()),
                Some(&row.at),
                "block {block:?}: id {} is at {:?} here and elsewhere in the reference",
                row.id(),
                row.at
            );
        }
    }
}

/// A block size that is not a divisor of the volume leaves a ragged last block,
/// which is the case an off-by-one in the core test would show up in.
///
/// The volume is 48 x 56; a block edge of 13 leaves last blocks of 9 and 4.
#[test]
fn a_ragged_last_block_owns_what_it_should() {
    let halo = 14;
    let reference = run(VOLUME, halo);
    let ragged = run([1, 13, 13], halo);
    assert_eq!(ragged, reference, "a ragged lattice changed the answer");
}

/// The measurements are integers all the way through, so "the same" above means
/// bit-for-bit and not "within a tolerance".
///
/// If any accumulator were `f64`, this is the test that would start failing
/// intermittently rather than the ones above — so it is written as its own
/// claim: every value a row carries survives a round trip through `u64`
/// unchanged, which is trivially true and is exactly the point. The assertion
/// that matters is the one in the type: a column that could hold `1.0000000001`
/// would not compile here.
#[test]
fn every_column_is_an_integer_accumulator() {
    let rows = run([1, 16, 16], 14);
    assert!(!rows.is_empty(), "the scene holds objects");
    let schema = blockflow::model_segment::schema(2).expect("a schema");
    for column in schema.columns() {
        assert_eq!(
            column.kind(),
            blockflow::table::ColumnType::U64,
            "column {} is not an integer, so a seam merge over it would not associate",
            column.name()
        );
    }
    for row in rows {
        assert_eq!(row.values.len(), schema.len());
    }
}
