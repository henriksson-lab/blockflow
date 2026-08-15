// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A second array that reaches into a window, not just into a voxel.**
//
// `Chain::Source` already lets a chain read a stored level, and `tests/
// source_leaf.rs` proves that arm correct. But a source leaf has reach **zero**
// by construction — it produces the extent it was handed — so the only thing
// downstream can do with it is combine voxelwise. An op with a window over its
// *operand* had nowhere to say so.
//
// `BlockOp::source_inputs` is that statement: per operand, a level and a
// `Reach`. `BlockOp::apply_with` is where the operand reaches the kernel. This
// file asserts, in dependency order:
//
// 1. **An operand read through a window is decomposition-invariant** — every
//    block size gives the whole-volume answer, byte for byte.
// 2. **It is the level that was named**, not the phase's own input wearing its
//    number. The kernel below is built so the two differ everywhere.
// 3. **An op that declares an operand and forgets to consume it is refused**,
//    loudly, rather than computing a complete and well-formed wrong volume.
// 4. **Two readers of one level fold to the wider reach**, because they are
//    handed one buffer.
// 5. **An operand reaching past the phase's halo is refused by name**, which is
//    the equal-reach limit stated where a caller meets it.
// 6. **A source leaf is still the reach-zero case**, so nothing that worked
//    before now declares something it did not.
//
// No assertion here is on wall-clock time.

use ndarray::Array3;

use blockflow::decomposition::{check_source_levels, Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::error::Result;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInput, SourceInputs};
use blockflow::ops::{ElementShape, Morphology, MorphologyOp, StructuringElement};
use blockflow::reach::Reach;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [16, 12, 10];
/// Written by phase 0, read by phase 1 as its input **and** by phase 2 as a
/// windowed operand. An intermediate, deliberately: level 0 is never freed, so
/// it would not exercise the lifetime the operand keeps alive.
const STORED: usize = 1;
const RADIUS: usize = 1;

// ------------------------------------------------------------- the op --

/// `out[v] = 10 * input[v] + (number of operand voxels set in the window at v)`.
///
/// **Three properties are wanted from this kernel and each is load-bearing.**
/// It reads the operand over a *window*, so a buffer supplied at reach zero is
/// too narrow and the bug shows as a wrong number rather than a panic. It reads
/// the input at the centre only, so the two arrays enter at different extents
/// and an implementation that quietly passed one for the other cannot agree.
/// And the two terms occupy different decimal places, so a failure says which
/// half was wrong instead of only that something was.
struct WindowedOperand {
    level: usize,
    radius: usize,
}

impl WindowedOperand {
    fn new(level: usize, radius: usize) -> Self {
        Self { level, radius }
    }
}

impl BlockOp for WindowedOperand {
    fn name(&self) -> &'static str {
        "windowed-operand"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.radius
    }

    /// **Refuses.** This op cannot be computed from its input alone, and saying
    /// so here is what makes the routing testable: if `Chain::apply_with` ever
    /// fell back to `apply` for an op that declared an operand, every test in
    /// this file would fail with this message rather than with a wrong number.
    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(blockflow::error::Error::InvalidArgument(
            "windowed-operand has an operand and was applied without one".to_string(),
        ))
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(
            self.level,
            Reach::symmetric([self.radius; 3]),
        )]
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let operand = sources.get(self.level)?;
        let input = input.view::<f64>()?;
        let operand = operand.view::<f64>()?;
        let mut out = out.view_mut::<f64>()?;
        let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let radius = self.radius as isize;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let mut count = 0.0_f64;
                    for di in -radius..=radius {
                        for dj in -radius..=radius {
                            for dk in -radius..=radius {
                                let (ni, nj, nk) =
                                    (i as isize + di, j as isize + dj, k as isize + dk);
                                // Clamped at the buffer edge, which at a real
                                // volume boundary is the volume edge and at a
                                // block seam is inside the halo — the same
                                // convention every neighbourhood op here uses.
                                let ni = ni.clamp(0, shape[0] as isize - 1) as usize;
                                let nj = nj.clamp(0, shape[1] as isize - 1) as usize;
                                let nk = nk.clamp(0, shape[2] as isize - 1) as usize;
                                count += operand[[ni, nj, nk]];
                            }
                        }
                    }
                    out[[i, j, k]] = 10.0 * input[[i, j, k]] + count;
                }
            }
        }
        Ok(())
    }
}

/// The same declaration with **no kernel**: it asks the plan for an operand and
/// never overrides `apply_with`. The shape of bug claim 3 is about.
struct ForgetfulOperand {
    level: usize,
}

impl BlockOp for ForgetfulOperand {
    fn name(&self) -> &'static str {
        "forgetful-operand"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.level)]
    }
}

// ------------------------------------------------------------ fixtures --

/// Separated boxes, about 7% of the volume: dense enough that the window count
/// varies across the array and sparse enough that a dilation is not everything.
fn input() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        f64::from(i % 6 < 2 && j % 5 < 2 && k % 4 < 2)
    })
}

fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
}

fn dilate() -> Chain {
    Chain::op(MorphologyOp::new("dilate", Morphology::Dilate, element()))
}

fn erode() -> Chain {
    Chain::op(MorphologyOp::new("erode", Morphology::Erode, element()))
}

/// Phase 0 dilates into level 1, phase 1 erodes into level 2, phase 2 reads
/// level 2 as its input and level 1 as its operand.
///
/// **Level 1 and level 2 differ everywhere the input has a boundary**, which is
/// what makes claim 2 checkable: an implementation handing the op its own input
/// in place of the named level produces a different array, not the same one.
fn chain() -> Chain {
    Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::op(WindowedOperand::new(STORED, RADIUS)),
    ])
}

fn whole(chain: &Chain) -> Voxels {
    let source: Voxels = input().into();
    let mut out = Voxels::zeros(
        chain.produces(Dtype::F64).unwrap(),
        chain.output_shape(VOLUME).unwrap(),
    )
    .unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .unwrap();
    out
}

fn one_phase_per_slot(chain: &Chain, grid: &BlockGrid) -> Decomposition {
    let slots = chain.slots();
    let reaches = [[1usize, 1, 1], [1, 1, 1], [RADIUS, RADIUS, RADIUS]];
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                reaches[slot],
                reaches[slot],
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [2, 2, 2],
    };
    plan.declare_dtypes(chain).unwrap();
    plan.declare_source_levels(chain).unwrap();
    plan
}

fn grids() -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
        BlockGrid::along(VOLUME, &[0], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0], 8).unwrap(),
        BlockGrid::along(VOLUME, &[1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[2], 5).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1, 2], 4).unwrap(),
    ]
}

fn run(grid: &BlockGrid) -> Array3<f64> {
    let chain = chain();
    let plan = one_phase_per_slot(&chain, grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap();
    execute("operand", &workflow, &plan, &Hints::default(), &env).expect("a run");
    env.output().view::<f64>().unwrap().to_owned()
}

// ------------------------------------------- 1. decomposition invariance --

#[test]
fn a_windowed_operand_gives_the_whole_volume_answer_under_every_decomposition() {
    let reference = run(&BlockGrid::new(VOLUME, VOLUME).unwrap());
    for grid in grids() {
        assert_eq!(run(&grid), reference, "block {:?}", grid.block());
    }
}

/// The reference is not vacuous: the operand term really varies, so an
/// implementation that returned a constant would not pass the test above.
#[test]
fn the_operand_term_is_not_constant() {
    let out = run(&BlockGrid::new(VOLUME, VOLUME).unwrap());
    let counts: Vec<f64> = out.iter().map(|value| value % 10.0).collect();
    let low = counts.iter().cloned().fold(f64::INFINITY, f64::min);
    let high = counts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(low < high, "the window count was {low} everywhere");
    assert!(high > 0.0, "no operand voxel was ever counted");
}

// ------------------------------------------------- 2. the level it names --

/// Reading the operand is not the same as reading the phase's own input.
///
/// The op is handed level 2 and names level 1. If the executor supplied the
/// input in the operand's place, the answer would be the array this test
/// computes — so asserting they *differ* is what pins the routing.
#[test]
fn the_operand_is_the_level_named_and_not_the_phases_own_input() {
    let real = run(&BlockGrid::new(VOLUME, VOLUME).unwrap());

    let level_two = whole(&Chain::sequence(vec![dilate(), erode()]));
    let mut wrong = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let op = WindowedOperand::new(STORED, RADIUS);
    let entries = [(STORED, &level_two)];
    op.apply_with(
        &level_two,
        SourceInputs::new(&entries),
        &mut wrong,
        &Anchor::whole(VOLUME),
    )
    .unwrap();

    assert_ne!(
        real,
        wrong.view::<f64>().unwrap().to_owned(),
        "level 1 and level 2 produced the same answer, so this test proves nothing"
    );
}

// --------------------------------------------- 3. a forgotten operand --

#[test]
fn an_op_that_declares_an_operand_and_never_consumes_it_is_refused() {
    let op = ForgetfulOperand { level: STORED };
    let input: Voxels = input().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let stored = input.clone();
    let entries = [(STORED, &stored)];
    let failed = op
        .apply_with(
            &input,
            SourceInputs::new(&entries),
            &mut out,
            &Anchor::whole(VOLUME),
        )
        .unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("forgetful-operand") && message.contains("apply_with"),
        "the refusal must name the op and the method it is missing: {message}"
    );
    assert!(
        message.contains(&STORED.to_string()),
        "and the level it asked for: {message}"
    );
}

// ------------------------------------------- 4. two readers, one buffer --

/// One level, two readers, different reaches: the fold keeps the wider, because
/// the executor supplies **one** buffer per level and it has to satisfy both.
#[test]
fn two_readers_of_one_level_fold_to_the_wider_reach() {
    let chain = Chain::sequence(vec![
        Chain::op(WindowedOperand::new(STORED, 1)),
        Chain::op(WindowedOperand::new(STORED, 3)),
    ]);
    let inputs = chain.source_inputs(VOLUME).unwrap();
    assert_eq!(inputs.len(), 1, "one entry per level, not per reader");
    assert_eq!(inputs[0].level, STORED);
    assert_eq!(inputs[0].reach.as_symmetric(), Some([3, 3, 3]));
}

// ------------------------------------------------ 5. the equal-reach limit --

/// An operand wider than the phase fetches is refused when the plan is checked,
/// naming both numbers and the thing that would lift the limit.
#[test]
fn an_operand_reaching_past_the_phase_halo_is_refused_by_name() {
    let chain = Chain::sequence(vec![
        dilate(),
        erode(),
        // Declares a reach of 3 on its operand while the phase below is built
        // with a halo of 1.
        Chain::op(WindowedOperand::new(STORED, 3)),
    ]);
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let slots = chain.slots();
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                [1usize, 1, 1],
                [1usize, 1, 1],
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [3, 3, 3],
    };
    plan.declare_dtypes(&chain).unwrap();
    plan.declare_source_levels(&chain).unwrap();

    let failed = check_source_levels(&chain, &plan).unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("per-input halo"),
        "the refusal must name what would lift the limit: {message}"
    );
    assert!(
        message.contains(&format!("level {STORED}")),
        "and which operand: {message}"
    );
}

/// The same plan with a halo that *does* cover the operand passes, so the guard
/// above is a real boundary rather than a blanket refusal of wide operands.
#[test]
fn an_operand_within_the_halo_is_accepted() {
    let chain = Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::op(WindowedOperand::new(STORED, 3)),
    ]);
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let slots = chain.slots();
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                [3usize, 3, 3],
                [3usize, 3, 3],
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [3, 3, 3],
    };
    plan.declare_dtypes(&chain).unwrap();
    plan.declare_source_levels(&chain).unwrap();
    check_source_levels(&chain, &plan).expect("an operand inside the halo is fetchable");
}

// -------------------------------------------- 6. the leaf is unchanged --

/// `Chain::Source` declares reach zero, stated rather than assumed — so every
/// plan that used a source leaf before this feature declares exactly what it
/// declared then.
#[test]
fn a_source_leaf_is_the_reach_zero_case() {
    let leaf = Chain::source(STORED, Dtype::F64);
    let inputs = leaf.source_inputs(VOLUME).unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].level, STORED);
    assert!(inputs[0].reach.is_none(), "a source leaf reaches nothing");
}

/// And an op that declares nothing still declares nothing, which is what keeps
/// every plan in the crate fingerprinting as it did.
#[test]
fn an_ordinary_op_declares_no_operands() {
    assert!(dilate().source_inputs(VOLUME).unwrap().is_empty());
}
