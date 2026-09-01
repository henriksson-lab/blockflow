// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A `Combine` may write side outputs, and that is the only place some things
// can be written from.**
//
// A `Chain::Parallel`'s sink is the one point in a chain where every branch's
// value at a voxel exists at once. A `BlockOp` downstream sees the combine's
// single answer; a branch upstream sees only itself. So anything that must read
// across the branches and emit something that is *not* an image has nowhere else
// to live.
//
// The case that forced it is the trainer of
// `docs/design/pixel-classification.md`. A feature stack is 91 branches, and the
// rows it gathers are one 91-vector per labelled voxel — a fragment, not an
// image. `docs/design/pixel-classification.md` first proposed doing this with
// `BlockOp::apply_side`, which cannot work: side outputs were a `BlockOp`
// feature and the sink of a fan-in is a `Combine`.
//
// What is asserted here is the machinery, on a sampler small enough to check by
// hand — not the feature stack, which `tests/forest_predict.rs` covers.

use ndarray::{Array2, ArrayD};

use blockflow::error::Result;
use blockflow::op::{Anchor, Chain, Combine, Output, SideBlock, Slicing, SourceInputs};
use blockflow::ops::{Arithmetic, ArithmeticCombine, VoxelwiseMapOp};
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [4, 3, 2];

/// A combine that writes the mean of its branches as an image, and **every
/// branch's value at every voxel** as a side output: one row per voxel, one
/// column per branch.
///
/// That is the shape a training-row sampler has, reduced to something a test can
/// check exactly.
struct Sampler {
    rows: usize,
    columns: usize,
}

impl Combine for Sampler {
    fn name(&self) -> &'static str {
        "sampler"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() == self.columns && inputs.iter().all(|&dtype| dtype == Dtype::F64)
    }

    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::F64
    }

    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        Ok(inputs[0])
    }

    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let views: Vec<_> = inputs
            .iter()
            .map(|input| input.view::<f64>())
            .collect::<Result<Vec<_>>>()?;
        let mut out = out.view_mut::<f64>()?;
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = views
                .iter()
                .map(|view| view.iter().nth(index).copied().unwrap_or(0.0))
                .sum::<f64>()
                / self.columns as f64;
        }
        Ok(())
    }

    /// One `rows x columns` array of `f64`.
    fn side_outputs(&self, _volume: [usize; 3]) -> Vec<Output> {
        vec![Output::new("rows", Dtype::F64, &[self.rows, self.columns])]
    }

    fn side_region(&self, which: usize, valid: &Region, _volume: [usize; 3]) -> Result<Region> {
        assert_eq!(which, 0);
        // The block's voxels, laid out in the volume's own row order.
        let start =
            valid.start[0] * VOLUME[1] * VOLUME[2] + valid.start[1] * VOLUME[2] + valid.start[2];
        let count = valid.shape.iter().product::<usize>();
        Ok(Region::new(&[start, 0], &[count, self.columns]))
    }

    fn apply_side(
        &self,
        inputs: &[&Voxels],
        _primary: &Voxels,
        block: &SideBlock<'_>,
    ) -> Result<Vec<ArrayD<f64>>> {
        assert_eq!(
            inputs.len(),
            self.columns,
            "a combine writing a side output must be handed every branch's result"
        );
        let views: Vec<_> = inputs
            .iter()
            .map(|input| input.view::<f64>())
            .collect::<Result<Vec<_>>>()?;
        let count = block.within.shape.iter().product::<usize>();
        let mut rows = Array2::<f64>::zeros((count, self.columns));
        for (row, (i, j, k)) in itertools(block.within).enumerate() {
            for (column, view) in views.iter().enumerate() {
                rows[[row, column]] = view[[i, j, k]];
            }
        }
        Ok(vec![rows.into_dyn()])
    }
}

/// The valid sub-box's indices, in row order.
fn itertools(within: &Region) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    (within.start[0]..within.start[0] + within.shape[0]).flat_map(move |i| {
        (within.start[1]..within.start[1] + within.shape[1]).flat_map(move |j| {
            (within.start[2]..within.start[2] + within.shape[2]).map(move |k| (i, j, k))
        })
    })
}

fn scaled(name: &'static str, factor: f64) -> Chain {
    Chain::op(VoxelwiseMapOp::new(name, move |value: f64| value * factor))
}

fn input() -> Voxels {
    ndarray::Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        (i * VOLUME[1] * VOLUME[2] + j * VOLUME[2] + k) as f64
    })
    .into()
}

fn sampler_chain() -> Chain {
    Chain::parallel(
        vec![
            scaled("one", 1.0),
            scaled("ten", 10.0),
            scaled("hundred", 100.0),
        ],
        Box::new(Sampler {
            rows: VOLUME.iter().product(),
            columns: 3,
        }),
    )
    .expect("a fan-in")
}

// --------------------------------------------------------- the machinery --

/// **The combine's declaration reaches the chain**, after the branches' and in
/// that order.
#[test]
fn a_combines_side_outputs_are_declared_by_the_chain_after_its_branches() {
    let chain = sampler_chain();
    let outputs = chain.side_outputs(VOLUME);
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name, "rows");
    assert_eq!(outputs[0].shape, vec![24, 3]);

    // A fan-in whose combine declares nothing still declares nothing, which is
    // what every fan-in in the crate did before this existed.
    let plain = Chain::parallel(
        vec![scaled("a", 1.0), scaled("b", 2.0)],
        Box::new(ArithmeticCombine::new("add", Arithmetic::Add)),
    )
    .unwrap();
    assert!(plain.side_outputs(VOLUME).is_empty());
}

/// **Routing lands on the combine once the branches are exhausted.**
///
/// The failure this guards is an index that walks off the branches and reports
/// "declares none" for an output the chain does declare.
#[test]
fn side_region_routes_past_the_branches_to_the_combine() {
    let chain = sampler_chain();
    let valid = Region::new(&[1, 0, 0], &[2, 3, 2]);
    let region = chain.side_region(0, &valid, VOLUME).expect("routed");
    assert_eq!(region.start, vec![6, 0]);
    assert_eq!(region.shape, vec![12, 3]);

    // And one past the end is still refused, by name.
    let err = chain
        .side_region(1, &valid, VOLUME)
        .expect_err("there is only one")
        .to_string();
    assert!(err.contains("declares 1"), "{err}");
}

/// **The combine is handed every branch's result**, which is the whole point:
/// the sampler's rows are the three branches' values at each voxel, and no
/// branch could have produced them.
#[test]
fn the_combine_is_handed_every_branch_and_writes_what_it_saw() {
    let chain = sampler_chain();
    let source = input();
    let mut primary = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let at = Anchor::whole(VOLUME);
    chain.apply(&source, &mut primary, &at).unwrap();

    let within = Region::new(&[0, 0, 0], &VOLUME);
    let regions = vec![chain.side_region(0, &within, VOLUME).unwrap()];
    let produced = chain
        .apply_side(
            &source,
            SourceInputs::new(&[]),
            &primary,
            &SideBlock {
                at: &at,
                within: &within,
                regions: &regions,
            },
        )
        .expect("the side outputs");

    assert_eq!(produced.len(), 1);
    let rows = &produced[0];
    assert_eq!(rows.shape(), &[24, 3]);
    for voxel in 0..24 {
        let value = voxel as f64;
        assert_eq!(rows[[voxel, 0]], value, "branch 0 at {voxel}");
        assert_eq!(rows[[voxel, 1]], value * 10.0, "branch 1 at {voxel}");
        assert_eq!(rows[[voxel, 2]], value * 100.0, "branch 2 at {voxel}");
    }

    // The primary is unaffected: it is still the mean the combine writes.
    let mean = primary.view::<f64>().unwrap();
    assert_eq!(mean[[0, 0, 1]], 1.0 * 111.0 / 3.0);
}

/// **A fan-in whose combine declares nothing recomputes nothing extra.**
///
/// The bargain the new hook strikes is that a combine wanting side outputs makes
/// its fan-in re-derive every branch. That must not be charged to a fan-in that
/// wants none, and the observable is the branch ops' own call count.
#[test]
fn a_combine_declaring_nothing_costs_its_fan_in_no_recomputation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let calls = Arc::new(AtomicUsize::new(0));
    let counted = {
        let calls = Arc::clone(&calls);
        Chain::op(VoxelwiseMapOp::new("counted", move |value: f64| {
            calls.fetch_add(1, Ordering::Relaxed);
            value
        }))
    };
    let chain = Chain::parallel(
        vec![counted, scaled("other", 2.0)],
        Box::new(ArithmeticCombine::new("add", Arithmetic::Add)),
    )
    .unwrap();

    let source = input();
    let at = Anchor::whole(VOLUME);
    let within = Region::new(&[0, 0, 0], &VOLUME);
    let mut primary = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain.apply(&source, &mut primary, &at).unwrap();
    let after_apply = calls.load(Ordering::Relaxed);
    assert!(after_apply > 0, "the branch never ran");

    chain
        .apply_side(
            &source,
            SourceInputs::new(&[]),
            &primary,
            &SideBlock {
                at: &at,
                within: &within,
                regions: &[],
            },
        )
        .expect("no side outputs, but it must not fail");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        after_apply,
        "asking a fan-in for side outputs it does not declare recomputed a branch"
    );
}
