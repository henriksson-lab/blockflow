// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A K-ary reach-0 op**: `K` co-located arrays in, `K'` co-located arrays out,
// nothing read outside the position it is written to.
//
// The shape is one `BlockOp` with `K - 1` `source_inputs` and `K' - 1`
// `side_outputs`. It is not a `Combine`, and that is checkable rather than
// stylistic: the `Combine` trait has no side outputs, so a fan-in cannot write
// more than one array.
//
// What this file pins:
//
// 1. **The map is the definition**, position by position, against a reference
//    written out one expression per output.
// 2. **It mixes across the arrays.** The negative control is a matrix whose
//    diagonal is plausible on its own — an implementation that applied it to
//    each input independently would produce a scaled copy of each, which looks
//    exactly like an answer. The test requires the mixed answer.
// 3. **Decomposition invariance**, for the primary result and for every side
//    output.
// 4. **Every input is really read.** Changing one input has to move every
//    output whose row has a coefficient on it, and must not move the ones whose
//    row does not.
// 5. **Side outputs are still terminal.** They land in a named map on the
//    environment, not in `images`, so nothing can name one as an image — which is
//    unchanged by a run being able to be *handed* images.
// 6. **The cost of the shape, measured**, at `K = K' = 16` over a realistic
//    block. Reported, not asserted: see the last section for why.
// 7. **Both calls are functions of their operands.** `apply_side` computes the
//    arrays beside the primary from the arrays it is handed, so it answers on
//    its own, with no `apply_with` before it and no executor around it — and
//    refuses by image when an operand is missing rather than answering from the
//    one array it does have.

use std::time::Instant;

use ndarray::{Array3, ArrayD};

use blockflow::assemble::ImageId;
use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SideBlock, SourceInputs};
use blockflow::ops::{LinearMap, TupleOp};
use blockflow::region::Region;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [16, 12, 10];

/// Four arrays in, four out. Four rather than two because the smallest case
/// that is not a pair is what the shape exists for, and small enough that the
/// reference below is readable.
const INPUTS: usize = 4;
const OUTPUTS: usize = 4;

// ------------------------------------------------------------- fixtures --

/// Input `which`: a smooth field that is different per array and never zero, so
/// that a coefficient dropped anywhere shows up everywhere rather than in a
/// sparse set of positions.
fn input(which: usize) -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        let position = (i * 37 + j * 11 + k * 5 + which * 3) as f64;
        1.0 + (position * 0.1).sin() + which as f64 * 0.25
    })
}

/// The negative control, as a matrix.
///
/// **The diagonal is plausible and the answer is not on it.** An implementation
/// that ran the map over each input on its own — taking `M[o][o] * in[o]` and
/// ignoring the rest — would produce four slightly dimmed copies of the four
/// inputs, which is a complete, well-formed, entirely wrong result that looks
/// like an answer. The off-diagonal terms are what make the two distinguishable
/// at every position, and `mixes_across_inputs_and_not_within_them` is what
/// requires it.
fn matrix() -> LinearMap {
    let mut values = vec![0.0; OUTPUTS * INPUTS];
    for row in 0..OUTPUTS {
        values[row * INPUTS + row] = 0.9;
        values[row * INPUTS + (row + 1) % INPUTS] = 0.35;
        values[row * INPUTS + (row + 2) % INPUTS] = -0.2;
    }
    LinearMap::new("mix", OUTPUTS, INPUTS, values).expect("a map")
}

/// The images of inputs `1..K`: the arrays handed to the run.
fn source_images() -> Vec<usize> {
    (0..INPUTS - 1)
        .map(|which| ImageId::supplied(which).index())
        .collect()
}

fn side_names() -> Vec<String> {
    (1..OUTPUTS).map(|out| format!("mix.{out}")).collect()
}

fn op() -> TupleOp {
    TupleOp::new(
        "mix",
        Box::new(matrix()),
        source_images(),
        side_names(),
        Dtype::F64,
    )
    .expect("an op")
}

fn chain() -> Chain {
    Chain::op(op())
}

fn plan(chain: &Chain, grid: &BlockGrid) -> Decomposition {
    let slots = chain.slots();
    let phases = vec![PhaseDecomposition::derive(
        vec![0],
        vec![slots[0].display_name()],
        [0usize, 0, 0],
        [0usize, 0, 0],
        grid.clone(),
    )];
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [0, 0, 0],
    };
    plan.declare_dtypes(chain).expect("element types");
    plan.declare_source_images(chain).expect("source images");
    plan
}

fn grids() -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
        BlockGrid::along(VOLUME, &[0], 4).unwrap(),
        BlockGrid::along(VOLUME, &[1], 3).unwrap(),
        BlockGrid::along(VOLUME, &[2], 5).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1, 2], 4).unwrap(),
    ]
}

/// The run, returning every one of the `K'` outputs in order.
fn run_with(grid: &BlockGrid, arrays: &[Array3<f64>]) -> Vec<Array3<f64>> {
    let chain = chain();
    let plan = plan(&chain, grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let supplied: Vec<Voxels> = arrays[1..].iter().cloned().map(Voxels::from).collect();
    let env = ArrayEnvironment::with_inputs(arrays[0].clone().into(), supplied, &plan, [4, 4, 4])
        .expect("an env");
    execute("mix", &workflow, &plan, &Hints::default(), &env).expect("a run");

    let mut produced = vec![env.output().view::<f64>().unwrap().to_owned()];
    for name in side_names() {
        let held: ArrayD<f64> = env
            .side_output(&name)
            .unwrap_or_else(|| panic!("side output {name}"));
        produced.push(
            held.into_dimensionality::<ndarray::Ix3>()
                .expect("a rank-3 side output"),
        );
    }
    produced
}

fn arrays() -> Vec<Array3<f64>> {
    (0..INPUTS).map(input).collect()
}

fn run(grid: &BlockGrid) -> Vec<Array3<f64>> {
    run_with(grid, &arrays())
}

/// The map written out: for every position, the whole tuple through the matrix.
///
/// A resident reference with no plan, no blocking and no shell in it. The
/// accumulation order is the kernel's — coefficient 0 first — so the comparison
/// below is exact rather than approximate.
fn resident(arrays: &[Array3<f64>]) -> Vec<Array3<f64>> {
    let map = matrix();
    let mut produced: Vec<Array3<f64>> = (0..OUTPUTS)
        .map(|_| Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2])))
        .collect();
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                let values: Vec<f64> = arrays.iter().map(|array| array[(i, j, k)]).collect();
                for (row, value) in map.at_one_position(&values).into_iter().enumerate() {
                    produced[row][(i, j, k)] = value;
                }
            }
        }
    }
    produced
}

// ----------------------------------------- 1 and 3. the map, and invariance --

#[test]
fn the_shell_computes_the_map_under_every_decomposition() {
    let wanted = resident(&arrays());
    // Not degenerate: no two outputs are the same array, and none is a constant.
    for row in 0..OUTPUTS {
        assert!(wanted[row]
            .iter()
            .any(|&value| value != wanted[row][(0, 0, 0)]));
        for other in row + 1..OUTPUTS {
            assert_ne!(wanted[row], wanted[other], "outputs {row} and {other}");
        }
    }

    for grid in grids() {
        let produced = run(&grid);
        assert_eq!(produced.len(), OUTPUTS);
        for row in 0..OUTPUTS {
            assert_eq!(
                produced[row],
                wanted[row],
                "output {row}, block {:?}",
                grid.block()
            );
        }
    }
}

// ------------------------------------------- 2. the negative control --

/// A matrix that mixes across the inputs is not a matrix applied to each of them
/// on its own.
///
/// The wrong answer this rules out is the one that looks right: four dimmed
/// copies of the four inputs, with the same shape, the same element type and
/// values in the same range as the real answer.
#[test]
fn mixes_across_inputs_and_not_within_them() {
    let arrays = arrays();
    let produced = run(&BlockGrid::along(VOLUME, &[0, 1], 4).unwrap());

    // What an implementation that never looked at the other inputs would give.
    let per_input: Vec<Array3<f64>> = (0..OUTPUTS)
        .map(|row| arrays[row].mapv(|value| 0.9 * value))
        .collect();
    for row in 0..OUTPUTS {
        assert_ne!(produced[row], per_input[row], "output {row}");
        // and not at a single position either: the two differ *everywhere*, so
        // this is not a rounding difference at a handful of voxels.
        let same = produced[row]
            .iter()
            .zip(per_input[row].iter())
            .filter(|(&mixed, &alone)| mixed == alone)
            .count();
        assert_eq!(same, 0, "output {row} agrees with the per-input answer");
    }
}

/// Every input is read, and only where the matrix says so.
///
/// Row `o` has coefficients on inputs `o`, `o + 1` and `o + 2` and a zero on
/// `o + 3`. Perturbing input `c` must move exactly the three outputs whose row
/// weights it, and leave the fourth bit-identical — which is a much stronger
/// statement than "the answer changed" and is what catches an operand wired to
/// the wrong slot.
#[test]
fn each_input_reaches_exactly_the_outputs_whose_row_weights_it() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let reference = run(&grid);
    for changed in 0..INPUTS {
        let mut perturbed = arrays();
        perturbed[changed] = perturbed[changed].mapv(|value| value + 1.0);
        let produced = run_with(&grid, &perturbed);
        for row in 0..OUTPUTS {
            // The zero of row `row` sits on input `(row + 3) % INPUTS`.
            let weighted = changed != (row + 3) % INPUTS;
            if weighted {
                assert_ne!(
                    produced[row], reference[row],
                    "input {changed}, output {row}"
                );
            } else {
                assert_eq!(
                    produced[row], reference[row],
                    "input {changed}, output {row}"
                );
            }
        }
    }
}

// -------------------------------------- 5. a side output is still terminal --

/// A side output lands in the environment's named map and not in `images`, so
/// nothing can name one where an image goes.
///
/// **Unchanged by a run being able to be handed images.** A supplied input is an
/// array that existed before the run; a side output is written during it. The
/// two are addressed differently — an image by number, a side output by name —
/// and `Chain::source` and `SourceInput::image` take a number. So the answer to
/// "can a side output be read back as an input now" is no, and it is no for a
/// reason that has nothing to do with what changed: there is no name for it in
/// the image address space, and giving it one would be a *third* thing, an image
/// written by a phase that a later phase reads — which is what an ordinary image
/// already is.
#[test]
fn a_side_output_is_not_an_image_and_cannot_be_named_as_one() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let chain = chain();
    let plan = plan(&chain, &grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let arrays = arrays();
    let supplied: Vec<Voxels> = arrays[1..].iter().cloned().map(Voxels::from).collect();
    let env = ArrayEnvironment::with_inputs(arrays[0].clone().into(), supplied, &plan, [4, 4, 4])
        .expect("an env");
    execute("mix", &workflow, &plan, &Hints::default(), &env).expect("a run");

    // The plan holds two images the run writes — the input and the output — and
    // three it was handed. The three side outputs are in none of them.
    assert_eq!(plan.n_images(), 2);
    assert_eq!(plan.n_supplied_inputs(), INPUTS - 1);
    assert_eq!(env.side_output_names(), side_names());
    for name in side_names() {
        assert!(env.side_output(&name).is_some(), "{name}");
    }
    // and nothing in the image address space answers to a side output's name:
    // the only addresses that exist are the two the plan writes and the three it
    // was handed.
    let addressable: Vec<usize> = (0..plan.n_images())
        .chain(plan.supplied_input_images())
        .collect();
    assert_eq!(addressable.len(), 2 + (INPUTS - 1));
}

// ------------------------------------------ 7. the operands, on their own --

/// One block's operands, ready to be handed to either call.
fn operands(arrays: &[Array3<f64>]) -> (Voxels, Vec<Voxels>) {
    (
        arrays[0].clone().into(),
        arrays[1..].iter().cloned().map(Voxels::from).collect(),
    )
}

/// The regions the whole volume as one block declares for its side outputs.
fn whole_block_regions(op: &TupleOp) -> Vec<Region> {
    let whole = Region::whole(&VOLUME);
    (0..OUTPUTS - 1)
        .map(|which| op.side_region(which, &whole, VOLUME).expect("a region"))
        .collect()
}

/// `apply_side` computes the outputs beside the primary from the arrays it is
/// handed, and nothing is carried to it from the call that produced the primary.
///
/// Called **directly**, one block wide, with no executor around it: that is the
/// whole content of the operands being arguments. The answer is the resident
/// reference's rows `1..K'`, exactly — and the primary the other call produced is
/// checked in the same test, because "computes its own" is only interesting if
/// the two together are still the whole map.
#[test]
fn apply_side_computes_the_outputs_from_the_operands_it_is_handed() {
    let arrays = arrays();
    let wanted = resident(&arrays);
    let op = op();
    let (input, held) = operands(&arrays);
    let entries: Vec<(ImageId, &Voxels)> = source_images()
        .into_iter()
        .map(ImageId::from)
        .zip(held.iter())
        .collect();
    let at = Anchor::whole(VOLUME);
    let whole = Region::whole(&VOLUME);
    let regions = whole_block_regions(&op);

    let mut primary = Voxels::zeros(Dtype::F64, VOLUME).expect("a buffer");
    op.apply_with(&input, SourceInputs::new(&entries), &mut primary, &at)
        .expect("the primary");
    assert_eq!(
        primary.view::<f64>().expect("f64").to_owned(),
        wanted[0],
        "the primary"
    );

    let produced = op
        .apply_side(
            &input,
            SourceInputs::new(&entries),
            &primary,
            &SideBlock {
                at: &at,
                within: &whole,
                regions: &regions,
            },
        )
        .expect("the rest");
    assert_eq!(produced.len(), OUTPUTS - 1);
    for row in 1..OUTPUTS {
        let got = produced[row - 1]
            .clone()
            .into_dimensionality::<ndarray::Ix3>()
            .expect("a rank-3 side output");
        assert_eq!(got, wanted[row], "output {row}");
    }
}

/// Without the source inputs there is no answer, and it is refused by image.
///
/// The negative control the test above needs: `apply_side` is a function of its
/// operands, so an operand that is not there has to stop the block rather than
/// produce something from the one array it does have. It is also what the
/// refusal this shape used to carry has become — that one named a block offset
/// whose entry an earlier call had not left behind, which was a statement about
/// two calls agreeing rather than about the data, and there are no two calls to
/// agree any more.
#[test]
fn apply_side_without_the_source_inputs_is_refused_by_image() {
    let op = op();
    let (input, _held) = operands(&arrays());
    let at = Anchor::whole(VOLUME);
    let whole = Region::whole(&VOLUME);
    let regions = whole_block_regions(&op);
    let primary = Voxels::zeros(Dtype::F64, VOLUME).expect("a buffer");

    let refusal = op
        .apply_side(
            &input,
            SourceInputs::none(),
            &primary,
            &SideBlock {
                at: &at,
                within: &whole,
                regions: &regions,
            },
        )
        .map(|_| ())
        .expect_err("a refusal")
        .to_string();
    let missing = source_images()[0];
    assert!(refusal.contains(&format!("image {missing}")), "{refusal}");
}

// ----------------------------------------------------- 6. the measurement --

/// The shape's cost at `K = K' = 16`, over a block the size a plan really uses.
///
/// **Reported, not asserted.** A wall-clock threshold in a test suite fails on a
/// loaded machine and passes on a fast one, which tells a reader nothing about
/// the code; what is asserted here is the answer, and the timing is printed for
/// whoever is looking. Run with `--nocapture` to see it.
///
/// The quantity that matters is not flops. At `f32` this is two flops per byte
/// moved, so it is streaming-bound, and the number to watch is the count of
/// concurrent streams: 16 reads plus 16 writes is 32, which is at the edge of
/// what a hardware prefetcher tracks. The kernel is tiled for that reason.
#[test]
fn the_cost_of_a_sixteen_by_sixteen_map_over_one_block() {
    const K: usize = 16;
    const BLOCK: [usize; 3] = [128, 128, 32];
    let positions: usize = BLOCK.iter().product();

    let mut values = vec![0.0f64; K * K];
    for row in 0..K {
        for col in 0..K {
            values[row * K + col] = ((row * 7 + col * 13) % 11) as f64 * 0.05 - 0.25;
        }
    }
    let map = LinearMap::new("bench", K, K, values).expect("a map");

    let held: Vec<Vec<f32>> = (0..K)
        .map(|col| {
            (0..positions)
                .map(|position| ((position + col) % 251) as f32 * 0.01)
                .collect()
        })
        .collect();
    let mut written: Vec<Vec<f32>> = (0..K).map(|_| vec![0.0f32; positions]).collect();

    let inputs: Vec<&[f32]> = held.iter().map(|values| values.as_slice()).collect();
    // Several tile widths, because the whole question the shape raises is
    // whether `K` reads and `K'` writes advancing together stay ahead of the
    // prefetcher. The untiled figure — one pass over the whole block per output
    // — is the one to compare against, and it is the last entry.
    let mut measured: Vec<(usize, f64)> = Vec::new();
    for tile in [64usize, 256, 1024, 4096, positions] {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let started = Instant::now();
            {
                let mut outputs: Vec<&mut [f32]> = written
                    .iter_mut()
                    .map(|values| values.as_mut_slice())
                    .collect();
                // The tiling the shell does, done here so the number is the
                // kernel's and not the executor's.
                let mut start = 0;
                while start < positions {
                    let span = tile.min(positions - start);
                    let tile_in: Vec<&[f32]> = inputs
                        .iter()
                        .map(|slice| &slice[start..start + span])
                        .collect();
                    let mut tile_out: Vec<&mut [f32]> = outputs
                        .iter_mut()
                        .map(|slice| &mut slice[start..start + span])
                        .collect();
                    blockflow::ops::TupleKernel::apply_f32(&map, &tile_in, &mut tile_out, 0, span);
                    start += span;
                }
            }
            // `total_cmp`, not `f64::min`, by the convention this crate holds
            // everywhere a selection is made: `f64::min(-0.0, 0.0)` may return
            // either operand, so a best-of-N written with it is a statement
            // about `f64::min` rather than about the run. Elapsed times cannot
            // be NaN or signed zero, so the two agree here — which is the
            // reason to write the honest one rather than to make an exception.
            let elapsed = started.elapsed().as_secs_f64();
            if elapsed.total_cmp(&best).is_lt() {
                best = elapsed;
            }
        }
        measured.push((tile, best));
    }

    // The answer, at one position, so the loop above cannot have been optimised
    // away and the number is for work that happened.
    let probe = positions / 2;
    let at_probe: Vec<f64> = (0..K).map(|col| held[col][probe] as f64).collect();
    let wanted = map.at_one_position(&at_probe);
    for row in 0..K {
        assert!(
            (written[row][probe] as f64 - wanted[row]).abs() < 1e-3,
            "row {row}: {} against {}",
            written[row][probe],
            wanted[row]
        );
    }

    let flops = 2.0 * (K * K * positions) as f64;
    let bytes = ((K + K) * positions * std::mem::size_of::<f32>()) as f64;
    println!(
        "K=16 -> 16 f32 over {BLOCK:?} ({positions} positions), {:.2} flops/byte of image \
         traffic:",
        flops / bytes
    );
    for (tile, seconds) in &measured {
        println!(
            "  tile {tile:>7}: {:>8.3} ms  {:>6.2} Gflop/s  {:>5.2} GB/s  {:>6.2} ns/position",
            seconds * 1e3,
            flops / seconds / 1e9,
            bytes / seconds / 1e9,
            seconds * 1e9 / positions as f64
        );
    }
}
