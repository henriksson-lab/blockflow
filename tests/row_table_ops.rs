// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `ops::rows`: a table of rows in, a table of rows out
// — scaled, gathered against an image, or filtered — over a sweep of lattices.
//
// **Nothing here is compared as a set.** Every comparison is `assert_eq!` on a
// whole `Vec`, in order. A table's rows are addressed by position, so a permuted
// answer is a *different* answer that is still a well-formed table of the right
// rows, and a suite that sorted both sides before comparing would pass on
// exactly that failure. `ops::coordinates` states the same rule for the same
// reason and this suite inherits it.
//
// What each of the three is on trial for
// ---------------------------------------
// | op | the claim | the fixture that can see it fail |
// |---|---|---|
// | scale | ties round to **even** | coordinates at `1`, `3`, `5` halved: exact `.5` at both parities of the floor |
// | gather | the value is the image's at the row's own voxel, and the row is in the block that reads it | an image whose every voxel differs, so a read from the wrong block is a wrong number rather than a plausible one |
// | filter | the survivors keep their order, and are renumbered | a predicate rejecting at both ends of the list |
//
// And one thing all three are on trial for together: **the same list out
// whatever the lattice**.
//
// All three run through the executor, and the gather runs twice
// ---------------------------------------------------------------
// `ops::rows` ships a `FragmentOp` for each of the three. The gather was for a
// while the one that could not have a shell: while a `FragmentOp` read only the
// image its phase was handed, no arrangement of phases gave a gather both its
// rows and the array it must sample, and this file pinned that finding as a test
// written to stop passing. `FragmentOp::source_inputs` is what resolved it — the
// array a gather samples was never the image its phase is handed, it is a
// *second* one — and the test that pinned the two refusals has been deleted,
// which is exactly what it was written to make happen. Neither refusal was
// weakened; see `ops::rows`' header.
//
// So the gather is driven **twice**, and the two are compared:
//
// * **by hand**, block by block — each block's own rows and each block's own
//   slice of the image, straight through `gather_blob`. This is the path the
//   suite trusted before the shell existed, and it uses no executor at all;
// * **through a plan**, as a `GatherRowsOp` phase, with the executor fetching
//   the declared image, gathering the fragments and writing the answer.
//
// Asserting that the two agree, list for list, is what says the shell is plumbing
// over the kernel rather than a second implementation of it: the hand-driven path
// cannot be wrong in the same way the executor is, because it does not use it.
// Both are also compared against the definition, so "the two agree" cannot be two
// wrongs agreeing.

use ndarray::{s, Array3};

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, BlockBuf, Environment};
use blockflow::error::Result;
use blockflow::fragment::{
    append_fragment_phase, fragment_only, BlockOutput, BlockView, Coverage, FragmentOp,
    FragmentOutput, PhaseWork, SeamFold,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::rows::{
    collect_rows, gather_blob, gathered_schema, merge_rows, scaled_bound, scaled_index, ColumnTest,
    FilterRowsOp, GatherRowsOp, Limit, RowFilter, RowStreams, RowValues, ScaleRowsOp,
};
use blockflow::probes::NonZeroOp;
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::{Column, RowBuilder, Schema, Value};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [8, 8, 8];
const CHUNK: [usize; 3] = [4, 4, 4];

const SEEDED: &str = "rows.seeded";
const KEPT: &str = "rows.kept";
const SCALED: &str = "rows.scaled";
const GATHERED: &str = "rows.gathered";
const MARKED: &str = "rows.marked";
const VALUE: &str = "value";
const MARK: &str = "mark";
const IMAGE: &str = "image";

// -------------------------------------------------------------- the scene --

/// The image: **every voxel a different number**, and zero where no row is
/// wanted.
///
/// `1 + 64 i + 8 j + k` is injective over this volume and is never zero, so a
/// value read back **names the voxel it came from** — which is what makes a
/// block mix-up visible in the answer rather than only in a count. Against a
/// constant image, a gather that read the right voxel of the wrong block would
/// return the right number and this suite would pass.
///
/// The non-zero voxels are arranged to be awkward across a seam:
///
/// 1. a **crooked line** `[i, 7 - i, (3 i) % 8]`, running the wrong way on axis
///    1 against axis 0, so no cut on axis 1 leaves a block's rows contiguous in
///    the answer;
/// 2. the coordinates **1, 3, 5** on every axis, which is where the rounding
///    fixture's exact ties come from — halved they are `0.5`, `1.5`, `2.5`;
/// 3. the two **opposite corners**, so nothing treats the volume boundary as
///    special;
/// 4. `[3, 6, 6]` and `[4, 6, 6]`, either side of the seam at 4 on the slowest
///    axis: the block-boundary case, as a fixture.
fn image() -> Array3<u16> {
    let mut array = Array3::<u16>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    let mut set = |at: [usize; 3]| {
        array[at] = (1 + 64 * at[0] + 8 * at[1] + at[2]) as u16;
    };

    for i in 0..VOLUME[0] {
        set([i, 7 - i, (3 * i) % 8]);
    }
    for &i in &[1usize, 3, 5] {
        for &j in &[1usize, 3, 5] {
            for &k in &[1usize, 3, 5] {
                set([i, j, k]);
            }
        }
    }
    set([0, 0, 0]);
    set([7, 7, 7]);
    set([3, 6, 6]);
    set([4, 6, 6]);

    array
}

/// Every non-zero voxel and its value, in the canonical order — the definition
/// the runs are measured against.
fn seeded() -> Vec<RowValues> {
    let image = image();
    let mut rows = Vec::new();
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                if image[[i, j, k]] != 0 {
                    rows.push(RowValues::new(
                        [i, j, k],
                        vec![Value::F64(image[[i, j, k]] as f64)],
                    ));
                }
            }
        }
    }
    rows
}

/// The cuts the suite sweeps, and why each is here.
///
/// | cut | lattice | what it is for |
/// |---|---|---|
/// | `[8, 8, 8]` | 1 | one block: the reference path, run through the framework |
/// | `[4, 8, 8]` | 2 x 1 x 1 | slabs on the slowest axis — the one shape where block-major *is* the canonical order |
/// | `[8, 4, 8]` | 1 x 2 x 1 | axis 1 alone: the blocks interleave maximally |
/// | `[8, 8, 3]` | 1 x 1 x 3 | the fastest axis, and ragged (8 = 3 + 3 + 2) |
/// | `[4, 4, 4]` | 2 x 2 x 2 | all three axes at once |
/// | `[3, 3, 3]` | 3 x 3 x 3 | ragged on every axis, so no seam lands on a round number |
/// | `[1, 8, 8]` | 8 x 1 x 1 | one voxel thick: the finest slab lattice |
const CUTS: [[usize; 3]; 7] = [
    [8, 8, 8],
    [4, 8, 8],
    [8, 4, 8],
    [8, 8, 3],
    [4, 4, 4],
    [3, 3, 3],
    [1, 8, 8],
];

// ---------------------------------------------------------- the row source --

/// One row per non-zero voxel of the block's core, carrying that voxel's value.
///
/// A producer, not one of the ops on trial: `ops::rows` transforms rows and
/// something has to make them. `ops::coordinates` would do for the scale, but
/// its rows have **no payload column**, so a filter over them would have nothing
/// to test — and a suite whose filter has no column is a suite with no filter in
/// it.
///
/// It is also the honest statement of what a row op can and cannot be spared:
/// this producer reads the image it emits values from, so where the value wanted
/// *is* the array the rows came from, no gather is needed at all. The case the
/// consumers have is the other one — rows from one array, values from a second —
/// and that is what needs the shell that does not exist yet.
struct SeedRowsOp;

impl SeedRowsOp {
    fn schema() -> Schema {
        Schema::new(vec![Column::f64(VALUE)]).expect("one named column")
    }
}

impl FragmentOp for SeedRowsOp {
    fn name(&self) -> &'static str {
        "seed rows"
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            SEEDED.to_string(),
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut rows = RowBuilder::new(std::sync::Arc::new(Self::schema()));
        if let BlockBuf::Array(pixels) = at.pixels()? {
            let array = pixels.view::<u16>()?;
            for i in 0..at.core.shape[0] {
                for j in 0..at.core.shape[1] {
                    for k in 0..at.core.shape[2] {
                        let local = [
                            at.core.start[0] - at.read.start[0] + i,
                            at.core.start[1] - at.read.start[1] + j,
                            at.core.start[2] - at.read.start[2] + k,
                        ];
                        let value = array[local];
                        if value != 0 {
                            rows.push(
                                [
                                    at.core.start[0] + i,
                                    at.core.start[1] + j,
                                    at.core.start[2] + k,
                                ],
                                &[Value::F64(value as f64)],
                            )?;
                        }
                    }
                }
            }
        }
        Ok(BlockOutput::fragment(SEEDED.to_string(), rows.encode()))
    }
}

// ------------------------------------------------------------- the harness --

fn streams(input: &str, phase: usize, output: &str, schema: Schema) -> RowStreams {
    RowStreams::new(input, phase, output, Lifecycle::Persistent, schema).expect("two names")
}

/// The predicate, and what makes it discriminating.
///
/// `[200, 500)` over an image running 1..=512 rejects rows at **both** ends — the
/// low corner at 1 and the far corner at 512 — and in the middle. A predicate
/// that kept everything would test no predicate at all, which
/// `the_predicate_rejects_at_both_ends` asserts rather than assumes.
fn predicate() -> RowFilter {
    RowFilter::new(vec![
        ColumnTest::range(VALUE, 200.0, 500.0).expect("two bounds")
    ])
    .expect("one test")
}

fn kept_by_predicate(rows: &[RowValues]) -> Vec<RowValues> {
    rows.iter()
        .filter(|row| {
            let Value::F64(value) = row.values[0] else {
                unreachable!("the schema says the column is an f64")
            };
            (200.0..500.0).contains(&value)
        })
        .cloned()
        .collect()
}

/// Run a plan of fragment phases over the image and hand back the environment,
/// which is where the answer is: no phase here writes an image.
fn run(block: [usize; 3], ops: &[&dyn FragmentOp]) -> Result<(ArrayEnvironment, Decomposition)> {
    let plan = fragment_only(VOLUME, block, Dtype::U16, ops)?;
    let input: Voxels = image().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, CHUNK)?;
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::U16);
    let work: Vec<PhaseWork<'_>> = ops.iter().map(|op| PhaseWork::Fragments(*op)).collect();
    execute_phases(
        "rows",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &work,
    )?;
    Ok((env, plan))
}

// ------------------------------------------------------ the fixture's teeth --

/// The image is the one described: injective where it is set, and holding the
/// rows the other tests name.
#[test]
fn the_image_names_every_voxel_it_holds() {
    let image = image();
    let mut seen: Vec<u16> = image.iter().copied().filter(|value| *value != 0).collect();
    let rows = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), rows, "two rows would carry the same value");
    assert_eq!(rows, seeded().len());

    for at in [[3usize, 6, 6], [4, 6, 6], [0, 0, 0], [7, 7, 7]] {
        assert_ne!(image[at], 0, "{at:?} should be a row");
    }
    // The two either side of the seam at 4 differ by a whole block of the ramp,
    // so reading one for the other is not a near miss.
    assert_ne!(image[[3, 6, 6]], image[[4, 6, 6]]);
}

/// The predicate rejects at both ends and keeps a middle, so the filter tests
/// something — asserted rather than assumed.
#[test]
fn the_predicate_rejects_at_both_ends() {
    let all = seeded();
    let kept = kept_by_predicate(&all);
    assert!(
        !kept.is_empty(),
        "a predicate that keeps nothing tests nothing"
    );
    assert!(
        kept.len() < all.len(),
        "a predicate that keeps everything tests no predicate"
    );
    let image = image();
    assert!(
        (image[[0, 0, 0]] as f64) < 200.0,
        "the low corner is rejected"
    );
    assert!(
        (image[[7, 7, 7]] as f64) >= 500.0,
        "the far corner is rejected"
    );
}

// ------------------------------------------------ the producer, as a floor --

/// The rows the producer writes are the definition's rows, whatever the cut —
/// the floor everything else here stands on.
#[test]
fn the_seeded_rows_are_the_same_list_under_every_cut() {
    let want = seeded();
    for block in CUTS {
        let (env, _) = run(block, &[&SeedRowsOp]).expect("a run");
        assert_eq!(
            collect_rows(&env, SEEDED, 0, VOLUME, SeedRowsOp::schema()).expect("the merge"),
            want,
            "the cut {block:?} seeded a different list"
        );
    }
}

// ----------------------------------------------------------- the filter --

/// **The acceptance property for the filter**: the same ordered list out of
/// every cut, and it is the list the definition gives.
#[test]
fn every_cut_filters_to_one_list() {
    let filter = FilterRowsOp::new(
        "filter",
        streams(SEEDED, 0, KEPT, SeedRowsOp::schema()),
        predicate(),
    )
    .expect("the column exists");
    let want = kept_by_predicate(&seeded());
    assert!(!want.is_empty());

    for block in CUTS {
        let (env, _) = run(block, &[&SeedRowsOp, &filter]).expect("a run");
        assert_eq!(
            collect_rows(&env, KEPT, 1, VOLUME, SeedRowsOp::schema()).expect("the merge"),
            want,
            "the cut {block:?} filtered to a different list"
        );
    }
}

/// The filter keeps a **subsequence** and **renumbers** what survives.
///
/// Both halves are asserted because they are different claims: the first is that
/// no row moved past another, the second is that a survivor's index changed. A
/// suite that checked only the first would pass under a scheme that kept the
/// input index, which is the other possible behaviour and is not this one.
#[test]
fn filtering_keeps_a_subsequence_and_renumbers_it() {
    let filter = FilterRowsOp::new(
        "filter",
        streams(SEEDED, 0, KEPT, SeedRowsOp::schema()),
        predicate(),
    )
    .expect("the column exists");
    let (env, _) = run([4, 4, 4], &[&SeedRowsOp, &filter]).expect("a run");
    let before = collect_rows(&env, SEEDED, 0, VOLUME, SeedRowsOp::schema()).expect("the merge");
    let after = collect_rows(&env, KEPT, 1, VOLUME, SeedRowsOp::schema()).expect("the merge");

    // A subsequence: every survivor appears in the input, in order, with nothing
    // moved past anything.
    let mut cursor = before.iter();
    for row in &after {
        assert!(
            cursor.any(|earlier| earlier == row),
            "{row:?} is missing from the input or is out of order"
        );
    }

    // And renumbered: the first survivor was not the first row in.
    let first = &after[0];
    let was = before
        .iter()
        .position(|row| row == first)
        .expect("it survived");
    assert!(
        was > 0,
        "the fixture must reject something before the first survivor, or renumbering is invisible"
    );
    assert_eq!(after.iter().position(|row| row == first), Some(0));
}

/// A strict bound and a closed one are different predicates, and the difference
/// is exactly the rows sitting on the limit.
#[test]
fn a_strict_bound_and_a_closed_one_differ_on_a_row_at_the_limit() {
    let limit = image()[[3, 6, 6]] as f64;
    let count = |bound: Limit| {
        let filter = FilterRowsOp::new(
            "filter",
            streams(SEEDED, 0, KEPT, SeedRowsOp::schema()),
            RowFilter::new(vec![ColumnTest::new(
                VALUE,
                Some(bound),
                Some(Limit::AtMost(limit)),
            )
            .expect("bounds")])
            .expect("one test"),
        )
        .expect("the column exists");
        let (env, _) = run([4, 4, 4], &[&SeedRowsOp, &filter]).expect("a run");
        collect_rows(&env, KEPT, 1, VOLUME, SeedRowsOp::schema())
            .expect("the merge")
            .len()
    };
    assert_eq!(
        count(Limit::AtLeast(limit)) - count(Limit::Above(limit)),
        1,
        "exactly the one row at the limit separates the two"
    );
}

// ------------------------------------------------------------- the scale --

/// **The rounding rule, through the executor.**
///
/// The image holds rows at 1, 3 and 5 on every axis; halved those are `0.5`,
/// `1.5` and `2.5` — exact ties at **both** parities of the floor. Ties-to-even
/// sends them to 0, 2 and 2; `f64::round` would send them to 1, 2 and 3.
///
/// What this catches: `f64::round` in the scale. The `1.5` case agrees under
/// both rules and is here so the test does not over-claim; `0.5` and `2.5` are
/// the ones that fail. A fixture of random coordinates contains no tie at all
/// and would certify either rule.
#[test]
fn the_scale_rounds_ties_to_even_through_a_run() {
    // The rule, written out rather than recomputed from the implementation.
    for (coordinate, even, away) in [(1usize, 0usize, 1usize), (3, 2, 2), (5, 2, 3)] {
        assert_eq!(coordinate as f64 * 0.5, (2 * coordinate) as f64 / 4.0);
        assert_eq!(scaled_index(coordinate, 0.5).expect("a factor"), even);
        assert_eq!((coordinate as f64 * 0.5).round() as usize, away);
    }
    assert_ne!(
        [0usize, 2, 2],
        [1usize, 2, 3],
        "the two rules must disagree on this fixture or it tests nothing"
    );

    let scale = ScaleRowsOp::new(
        "scale",
        streams(SEEDED, 0, SCALED, SeedRowsOp::schema()),
        [0.5; 3],
    )
    .expect("a finite factor");
    let out = scaled_bound(VOLUME, [0.5; 3]).expect("a finite factor");

    let mut want: Vec<RowValues> = seeded()
        .into_iter()
        .map(|row| {
            RowValues::new(
                [
                    scaled_index(row.at[0], 0.5).expect("a factor"),
                    scaled_index(row.at[1], 0.5).expect("a factor"),
                    scaled_index(row.at[2], 0.5).expect("a factor"),
                ],
                row.values,
            )
        })
        .collect();
    // The canonical order is coordinate then payload bits, so the expectation is
    // sorted the same way the store sorts — and the sort is what a scale's output
    // order *is*, because a scale moves rows past each other.
    want.sort_by(|a, b| {
        let key = |row: &RowValues| {
            let Value::F64(value) = row.values[0] else {
                unreachable!("the schema says the column is an f64")
            };
            (row.at, value.to_bits())
        };
        key(a).cmp(&key(b))
    });

    // The scale collapses distinct rows onto shared coordinates, and they stay
    // rows: a scale selects nothing.
    assert!(
        want.windows(2).any(|pair| pair[0].at == pair[1].at),
        "the fixture must collapse some rows or it is not testing a scale"
    );
    assert!(want.iter().any(|row| row.at == [0, 0, 0]));
    assert!(want.iter().any(|row| row.at == [2, 2, 2]));

    for block in CUTS {
        let (env, _) = run(block, &[&SeedRowsOp, &scale]).expect("a run");
        assert_eq!(
            collect_rows(&env, SCALED, 1, out, SeedRowsOp::schema()).expect("the merge"),
            want,
            "the cut {block:?} scaled to a different list"
        );
    }
}

// ------------------------------------------------------------ the gather --

/// One block's gather, exactly as the shell will do it: this block's rows, this
/// block's slice of the image, and this block's core as the region the rows must
/// lie in.
fn gather_one_block(
    env: &ArrayEnvironment,
    image: &Array3<u16>,
    core: &Region,
    index: [usize; 3],
) -> Result<Vec<u8>> {
    let rows = env
        .read_sidecar(SEEDED, 0, index)?
        .unwrap_or_else(|| panic!("block {index:?} wrote no blob"));
    let slice: Array3<u16> = image
        .slice(s![
            core.start[0]..core.start[0] + core.shape[0],
            core.start[1]..core.start[1] + core.shape[1],
            core.start[2]..core.start[2] + core.shape[2],
        ])
        .to_owned();
    gather_blob(
        VOLUME,
        &SeedRowsOp::schema(),
        &rows,
        IMAGE,
        core,
        &Voxels::U16(slice),
        [core.start[0], core.start[1], core.start[2]],
    )
}

/// Every block's gather, merged, **with no executor anywhere in it**: the path
/// the plan-driven one is measured against.
fn gathered(block: [usize; 3]) -> Vec<RowValues> {
    let (env, _) = run(block, &[&SeedRowsOp]).expect("a run");
    let grid = BlockGrid::new(VOLUME, block).expect("a grid");
    let image = image();
    let blobs: Vec<([usize; 3], Vec<u8>)> = grid
        .cores()
        .into_iter()
        .map(|core| {
            (
                core.index,
                gather_one_block(&env, &image, &core.core, core.index).expect("its own rows"),
            )
        })
        .collect();
    merge_rows(
        VOLUME,
        gathered_schema(&SeedRowsOp::schema(), IMAGE).expect("a fresh name"),
        blobs
            .iter()
            .map(|(index, bytes)| (*index, bytes.as_slice())),
    )
    .expect("the merge")
}

/// The gather **as a phase**: rows off `input`, written by `phase`, and the
/// image it samples named as a second array — image 0, the array the run was
/// handed.
///
/// It declares `reads_pixels() == false`, so the phase it runs as reads no image
/// of its own; the one array it pays for is the one it names here.
fn gather_op(input: &str, phase: usize, schema: Schema) -> GatherRowsOp {
    GatherRowsOp::new("gather", streams(input, phase, GATHERED, schema), 0, IMAGE)
        .expect("a fresh column name")
}

/// The same gather, driven **through a plan** by the executor.
fn gathered_by_plan(block: [usize; 3]) -> Vec<RowValues> {
    let gather = gather_op(SEEDED, 0, SeedRowsOp::schema());
    let (env, _) = run(block, &[&SeedRowsOp, &gather]).expect("a run");
    collect_rows(&env, GATHERED, 1, VOLUME, gather.schema().clone()).expect("the merge")
}

/// The answer the definition gives: every seeded row, with the image's own value
/// at its own voxel appended.
fn gathered_definition() -> Vec<RowValues> {
    let image = image();
    seeded()
        .into_iter()
        .map(|row| {
            let value = Value::F64(image[row.at] as f64);
            let mut values = row.values;
            values.push(value);
            RowValues::new(row.at, values)
        })
        .collect()
}

/// Whether `at` is inside `region` — cores are half-open, which is the whole
/// reason a row on a seam belongs to exactly one block.
fn holds(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

/// **The acceptance property for the gather**: the same ordered list out of
/// every cut, out of *both* paths, and it is the list the definition gives.
///
/// The injective image is what gives this teeth — a value read from the wrong
/// block is a different number, not a plausible one. Comparing the two paths
/// against each other says the shell is plumbing over the kernel; comparing both
/// against the definition stops that being two wrongs agreeing.
#[test]
fn every_cut_gathers_one_list() {
    let want = gathered_definition();
    assert!(!want.is_empty());

    for block in CUTS {
        let by_hand = gathered(block);
        let by_plan = gathered_by_plan(block);
        assert_eq!(
            by_hand, want,
            "the cut {block:?} gathered a different list by hand"
        );
        assert_eq!(
            by_plan, want,
            "the cut {block:?} gathered a different list through a plan"
        );
        assert_eq!(
            by_plan, by_hand,
            "the cut {block:?} disagrees between the shell and the kernel driven by hand"
        );
    }
}

/// **The block boundary.** A row on a seam is read once, by the block whose core
/// starts there, and carries its own voxel's value.
///
/// Cores are half-open and tile with no overlap, so `[3, 6, 6]` and `[4, 6, 6]`
/// sit either side of the seam at 4 under `[4, 8, 8]`. A duplicate would show as
/// a longer list; a row read from the wrong side would show as the neighbour's
/// number. Asserted on the plan-driven answer, because the executor is the part
/// that could hand a block the wrong fragment.
#[test]
fn a_row_on_a_block_seam_is_read_once_by_the_block_that_holds_it() {
    let rows = gathered_by_plan([4, 8, 8]);
    let image = image();
    for at in [[3usize, 6, 6], [4, 6, 6]] {
        let found: Vec<&RowValues> = rows.iter().filter(|row| row.at == at).collect();
        assert_eq!(found.len(), 1, "{at:?} appeared {} times", found.len());
        assert_eq!(found[0].values[1], Value::F64(image[at] as f64));
    }
}

/// **The `SeamFold::PerBlock` claim, checked against the answer rather than
/// restated.**
///
/// `PerBlock` says this block's fragment is a function of this block alone. The
/// framework checks the half it can — the declaration is refused beside a
/// non-zero fragment reach, and this op's reach is `[0, 0, 0]` — and this checks
/// the other half, which is about the data: every block's own fragment holds
/// exactly the rows of its own core, each carrying its own voxel's value.
/// Nothing arrives from a neighbour and nothing is accumulated, so there is no
/// order an answer could depend on.
///
/// `SeamFold::Unordered` would be true here too and would be **worse**: the
/// executor skips its reversal check when the neighbourhood holds one fragment,
/// which is exactly this op's neighbourhood, so it would be a claim nothing
/// checks.
#[test]
fn every_blocks_gathered_fragment_is_a_function_of_its_own_core() {
    let gather = gather_op(SEEDED, 0, SeedRowsOp::schema());
    assert_eq!(gather.seam_fold(), Some(SeamFold::PerBlock));
    assert_eq!(gather.inputs()[0].reach, [0, 0, 0]);

    let block = [4, 4, 4];
    let (env, plan) = run(block, &[&SeedRowsOp, &gather]).expect("a run");
    // The image is fetched at the block's own extent and nothing wider, which is
    // what makes "its own core" the whole story rather than most of it.
    assert_eq!(plan.phases[1].source_images, vec![0]);
    assert_eq!(plan.phases[1].halo, [0, 0, 0]);

    let whole = gathered_definition();
    let grid = BlockGrid::new(VOLUME, block).expect("a grid");
    let mut accounted = 0usize;
    for core in grid.cores() {
        let bytes = env
            .read_sidecar(GATHERED, 1, core.index)
            .expect("a read")
            .unwrap_or_else(|| panic!("block {:?} wrote no fragment", core.index));
        let mine = merge_rows(
            VOLUME,
            gather.schema().clone(),
            [(core.index, bytes.as_slice())],
        )
        .expect("the merge");
        let want: Vec<RowValues> = whole
            .iter()
            .filter(|row| holds(&core.core, row.at))
            .cloned()
            .collect();
        assert_eq!(
            mine, want,
            "block {:?} gathered rows that are not its own core's",
            core.index
        );
        accounted += want.len();
    }
    assert_eq!(
        accounted,
        whole.len(),
        "the blocks together hold every row exactly once, which is what a tiling of \
         half-open cores means"
    );
}

/// A row outside the block's core is **refused**, naming it — the precondition
/// reach 0 rests on, and what stops a gather reading a real value at the wrong
/// place.
///
/// Driven by handing one block another block's rows, which is exactly what a
/// scale between the producer and the gather would do.
#[test]
fn a_gather_handed_another_blocks_rows_refuses_them() {
    let (env, _) = run([4, 8, 8], &[&SeedRowsOp]).expect("a run");
    let grid = BlockGrid::new(VOLUME, [4, 8, 8]).expect("a grid");
    let cores = grid.cores();
    let image = image();

    // Its own rows are fine.
    assert!(gather_one_block(&env, &image, &cores[0].core, cores[0].index).is_ok());
    // The next block's rows, against this block's core, are not.
    let err = gather_one_block(&env, &image, &cores[0].core, cores[1].index)
        .expect_err("those rows are outside this core");
    let text = format!("{err}");
    assert!(
        text.contains("outside"),
        "the refusal should say the row is outside the region: {text}"
    );
}

// ------------------------------- rows from one array, values from another --

/// One row per **true** voxel of the block's core, carrying a column that says
/// nothing except "a row is here".
///
/// The uninformative payload is the point. In the plan below these rows come out
/// of a *different* image from the one the gather samples, and the mark carries
/// no trace of which voxel it came from — so the only voxel identity in the
/// answer is the gathered column, and it can only have come from the second
/// array. With a producer that already carried the value, a gather that returned
/// its own input would look right.
struct MarkRowsOp;

impl MarkRowsOp {
    fn schema() -> Schema {
        Schema::new(vec![Column::f64(MARK)]).expect("one named column")
    }
}

impl FragmentOp for MarkRowsOp {
    fn name(&self) -> &'static str {
        "mark rows"
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            MARKED.to_string(),
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut rows = RowBuilder::new(std::sync::Arc::new(Self::schema()));
        if let BlockBuf::Array(pixels) = at.pixels()? {
            let array = pixels.view::<bool>()?;
            for i in 0..at.core.shape[0] {
                for j in 0..at.core.shape[1] {
                    for k in 0..at.core.shape[2] {
                        let local = [
                            at.core.start[0] - at.read.start[0] + i,
                            at.core.start[1] - at.read.start[1] + j,
                            at.core.start[2] - at.read.start[2] + k,
                        ];
                        if array[local] {
                            rows.push(
                                [
                                    at.core.start[0] + i,
                                    at.core.start[1] + j,
                                    at.core.start[2] + k,
                                ],
                                &[Value::F64(1.0)],
                            )?;
                        }
                    }
                }
            }
        }
        Ok(BlockOutput::fragment(MARKED.to_string(), rows.encode()))
    }
}

/// Three phases: a pixel phase turning the injective image into a **bool** image
/// 1 that says only *where* the rows are, a producer reading that, and the
/// gather reading **image 0** — an array no phase between it and the rows
/// touched.
fn two_array_plan(block: [usize; 3]) -> (Decomposition, GatherRowsOp) {
    let base = Decomposition {
        volume: VOLUME,
        dtype: Dtype::U16,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["marks".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new(VOLUME, block).expect("a lattice"),
        )
        // The phase changes the element type — an image of marks is `bool` — and
        // a plan that did not say so would allocate image 1 at the width of the
        // image below it.
        .with_dtype(Dtype::Bool)],
        chain_reach: [0, 0, 0],
    };
    let gather = gather_op(MARKED, 1, MarkRowsOp::schema());
    let plan = append_fragment_phase(base, &MarkRowsOp).expect("a producer phase");
    let plan = append_fragment_phase(plan, &gather).expect("a gather phase");
    (plan, gather)
}

/// **The case the shell exists for**: the rows come from one array and the
/// values from a second, and the second is named rather than inherited.
///
/// This is what folding the gather into the row producer cannot do, and it is
/// the arrangement `source_inputs` buys: the producer reads image 1 and knows
/// nothing about image 0, and the gather reads image 0 and never touches the
/// image its own phase was handed.
#[test]
fn a_gather_reads_a_second_array_the_rows_did_not_come_from() {
    let image = image();
    let want: Vec<RowValues> = seeded()
        .into_iter()
        .map(|row| {
            RowValues::new(
                row.at,
                vec![Value::F64(1.0), Value::F64(image[row.at] as f64)],
            )
        })
        .collect();
    assert!(!want.is_empty());

    for block in [[8usize, 8, 8], [4, 4, 4], [3, 3, 3]] {
        let (plan, gather) = two_array_plan(block);
        // The plan records the second array, which is what makes the executor
        // fetch it, the DAG depend on its producer and the image survive to be
        // read. A gather whose image went unrecorded would read whatever
        // `prepare` left behind.
        assert_eq!(plan.phases[2].source_images, vec![0]);
        assert_eq!(plan.phases[1].source_images, Vec::<usize>::new());

        let env = ArrayEnvironment::for_decomposition(image.clone().into(), &plan, CHUNK)
            .expect("an environment");
        let workflow = Workflow::new(
            Chain::op(NonZeroOp::new("marks", [0, 0, 0])),
            VOLUME,
            Dtype::U16,
        );
        execute_phases(
            "two arrays",
            &workflow,
            &plan,
            &Hints::default(),
            &env,
            &[],
            &[
                PhaseWork::Pixels,
                PhaseWork::Fragments(&MarkRowsOp),
                PhaseWork::Fragments(&gather),
            ],
        )
        .expect("a run");

        assert_eq!(
            collect_rows(&env, GATHERED, 2, VOLUME, gather.schema().clone()).expect("the merge"),
            want,
            "the cut {block:?} gathered a different list from the second array"
        );
    }
}

// --------------------------------------------- what the shell still refuses --

/// **A scale between the producer and the gather is refused, through a real
/// plan, by the row it moved.**
///
/// This is the precondition reach 0 rests on, in the composition that breaks it:
/// after a scale, block `B`'s fragment holds rows at coordinates that are
/// somewhere else entirely, and a gather that trusted its reach would read a
/// real value at the wrong place and return a row that looked perfectly
/// well-formed.
#[test]
fn a_scale_before_a_gather_is_refused_by_the_row_it_moved() {
    let scale = ScaleRowsOp::new(
        "scale",
        streams(SEEDED, 0, SCALED, SeedRowsOp::schema()),
        [0.5; 3],
    )
    .expect("a finite factor");
    let gather = gather_op(SCALED, 1, SeedRowsOp::schema());
    let Err(error) = run([4, 4, 4], &[&SeedRowsOp, &scale, &gather]) else {
        panic!(
            "a scaled row is no longer in the block whose fragment carries it, and a gather \
             that read anyway would answer from the wrong voxel"
        )
    };
    let text = format!("{error}");
    assert!(
        text.contains("a gather was handed a row at") && text.contains("outside on axis"),
        "the refusal names the row and the axis: {text}"
    );
    assert!(
        text.contains("scale"),
        "and names the usual cause, which is this one: {text}"
    );
}

/// A gather naming an image **nothing wrote** is refused before any block runs.
///
/// Image 1 of a fragment-only plan is allocated and never written, and an
/// unwritten image in this crate is `NaN` precisely so that an absence cannot
/// pass for a value. The framework does not let it get as far as the `NaN`: a
/// second array has to be an image some phase produced.
#[test]
fn a_gather_naming_an_image_nothing_wrote_is_refused_before_it_runs() {
    let gather = GatherRowsOp::new(
        "gather",
        streams(SEEDED, 0, GATHERED, SeedRowsOp::schema()),
        1,
        IMAGE,
    )
    .expect("a fresh column name");
    let Err(error) = run([4, 4, 4], &[&SeedRowsOp, &gather]) else {
        panic!("an image nobody wrote is not a second array, it is whatever `prepare` allocated")
    };
    let text = format!("{error}");
    assert!(
        text.contains("image 1") && text.contains("did not write"),
        "expected the refusal of an image no phase produced: {text}"
    );
}
