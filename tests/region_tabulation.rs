// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A per-region reduction over a second array, and the seam that comes with
// it.** `ops::tabulate` is the op; this is what it is accepted on.
//
// The hazard, restated because it is the content
// ----------------------------------------------
// A region of a label volume straddles block seams. Its voxels are visited in
// pieces, one per block, and the pieces are combined in a merge. Combine them
// with `f64` addition and the total depends on the order the blocks merged in —
// so the same data cut two ways gives two numbers, and neither looks wrong.
// `ops/detect.rs` deferred a weighted centroid on exactly this and named the way
// out: an accumulator whose combine associates.
//
// `ops::tabulate`'s **sum** takes that way out — a fixed-point integer
// accumulator — and the op therefore claims `SeamFold::Unordered`, which the
// executor **checks** by applying each block a second time with its
// neighbourhood reversed.
//
// The **selection** needs no way out and does not take one. `min` and `max`
// choose one of the values they are handed rather than computing a new one, and
// under a total order that is associative, commutative and idempotent in `f64`
// itself — so they are `F64` columns holding the voxel's own bits, and
// `SeamFold::Unordered` holds for them on their own account. Sections 5 and 6
// are that split: 5 is the selection being exact where a quantised one could
// not have been, 6 is the sum not having moved while it happened.
//
// What each fixture is arranged so that it *can* fail
// ---------------------------------------------------
// * the block sizes put regions across seams, and the test **asserts that they
//   did** — byte-identity across five runs that merged nothing would prove
//   nothing at all;
// * the value array is signed, so a sign error is visible, and holds a `NaN` and
//   an infinity, so the stated exclusion is exercised rather than described;
// * the label volume has a **disconnected** region, a **single-voxel** region, a
//   region spanning **every block**, and a **gap in the numbering** — the four
//   degenerate cases, in the same fixture as the ordinary ones;
// * the negative control is the same plan with an `f64` fold in the merge's
//   place, which must be refused. Without it, "the reversal check passed" could
//   mean the check never ran;
// * and the selection's fixture holds a value the fixed point cannot represent
//   at **any** scale the run admits — 50 accepted and 13 refused across the 63,
//   and every accepted one quantises it inexactly — so "the column is exact" is
//   a statement that could have come back false rather than one the scale was
//   chosen to make true.
//
// The first moment, and the fixture built to separate it
// ------------------------------------------------------
// `moment_0..2_q{n}` is `sum(value * coordinate)` per axis, and its quotient by
// `sum_q{n}` is the **weighted centroid** — the one per-region quantity the
// other columns do not determine. It is the same kind of accumulator as the sum,
// on the same fixed point, so it is accepted on the same terms; section 5 is
// what it takes for those terms to mean something for *this* column.
//
// A symmetric arrangement cannot tell a weighted centre from an unweighted one,
// so a fixture built out of one would pass an implementation that never read the
// value array. `separating_labels` and `separating_value_at` are built the other
// way: a weight that rises steeply and independently on all three axes, from a
// base that is not a dyadic rational, over two regions that every cut puts
// across a seam. The two centres then differ by at least a quarter of a voxel on
// every axis of every region, which is the assertion the byte-identity rests on.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::Result;
use blockflow::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
    PhaseWork, SeamFold,
};
use blockflow::geometry::BlockGrid;
use blockflow::log::{Event, ExecutionLog};
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::tabulate::{
    append_tabulate_phases, collect_shapes, collect_tabulation, decode_partial, region_shape,
    region_values, signed_column, tabulation_schema, FixedPoint, MergeTabulationOp, PrincipalAxes,
    RegionShape, RegionValues, TabulateValuesOp, Tally, CENTRAL as CENTRAL_COLUMN,
    DEFAULT_FRACTION_BITS, MAX as MAX_COLUMN, MAX_FRACTION_BITS, MIN as MIN_COLUMN,
    MOMENT as MOMENT_COLUMN, PAIRS, SUM as SUM_COLUMN,
};
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::table::Table;
use blockflow::voxels::Voxels;
use ndarray::Array3;

// ------------------------------------------------------------- the fixture --

const VOLUME: [usize; 3] = [12, 4, 4];
/// Twenty fraction bits: a resolution of about `9.5e-7` and a range of about
/// `+/- 8.8e12`. Every value below is a few tens, so neither end binds — which
/// is the point of stating them.
fn fixed() -> FixedPoint {
    FixedPoint::default()
}

/// The label volume, as a function of the global coordinate.
///
/// Deliberately **not** a connected-component labelling, because that is the
/// distinction this op exists for:
///
/// * `1` is a full-length line on axis 0, so it touches every block of every
///   lattice tried below;
/// * `2` is a box from 3 to 9 on axis 0, whose faces sit at coordinates that no
///   block edge of 12, 6, 4, 3 or 2 lines up with on both sides;
/// * `3` is one voxel;
/// * `5` is **two disconnected pieces**, one at each end of axis 0, which
///   `ops::detect` would report as two components and this op reports as one
///   region — the whole reason it is not that op;
/// * `4` is never written, so the numbering has a gap.
fn label_at(at: [usize; 3]) -> u64 {
    let [z, y, x] = at;
    if y == 0 && x == 0 {
        return 1;
    }
    if (3..9).contains(&z) && (1..3).contains(&y) && (1..3).contains(&x) {
        return 2;
    }
    if at == [7, 3, 3] {
        return 3;
    }
    if y == 3 && x == 0 && !(2..10).contains(&z) {
        return 5;
    }
    0
}

/// The value array, as a function of the global coordinate.
///
/// **Signed**, so a sign error cannot hide, and with a `NaN` and an infinity in
/// it, so the stated exclusion is exercised. The `NaN` at `[9, 0, 0]` is in the
/// region that spans every block and the two at `[5, 2, 2]` and `[6, 1, 1]` are
/// in the one that straddles seams, so both cases meet the merge rather than
/// staying inside one block.
fn value_at(at: [usize; 3]) -> f64 {
    if at == [9, 0, 0] || at == [5, 2, 2] {
        return f64::NAN;
    }
    if at == [6, 1, 1] {
        return f64::INFINITY;
    }
    at[0] as f64 * 3.0 - at[1] as f64 * 11.5 + at[2] as f64 * 0.125 - 7.0
}

fn labels(volume: [usize; 3], of: fn([usize; 3]) -> u64) -> Voxels {
    let mut array = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
    for (index, slot) in array.indexed_iter_mut() {
        *slot = of([index.0, index.1, index.2]) as f64;
    }
    array.into()
}

/// Writes the value array from the **global** coordinate, so that a block's
/// output is a function of where it is and not of how big it is. Image 1 has to
/// be produced by a phase — an `ArrayEnvironment` holds one input array — and
/// this is the smallest honest producer of a second one.
struct CoordinateValuesOp {
    of: fn([usize; 3]) -> f64,
}

impl BlockOp for CoordinateValuesOp {
    fn name(&self) -> &'static str {
        "values"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(&self, _input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let mut view = out.view_mut::<f64>()?;
        for (index, slot) in view.indexed_iter_mut() {
            *slot = (self.of)([
                at.offset[0] + index.0,
                at.offset[1] + index.1,
                at.offset[2] + index.2,
            ]);
        }
        Ok(())
    }
}

fn values_workflow(volume: [usize; 3], of: fn([usize; 3]) -> f64) -> Workflow {
    Workflow::new(Chain::op(CoordinateValuesOp { of }), volume, Dtype::F64)
}

/// One pixel phase over `block`, which writes the value array into image 1. The
/// two fragment phases are appended after it.
fn values_phase(volume: [usize; 3], block: [usize; 3]) -> Decomposition {
    Decomposition {
        volume,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["values".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new(volume, block).expect("a lattice"),
        )],
        chain_reach: [0, 0, 0],
    }
}

/// The three-phase plan: the value array, the per-block partials read out of
/// **image 0 and image 1**, and the merge.
fn tabulation_plan(
    volume: [usize; 3],
    block: [usize; 3],
    point: FixedPoint,
) -> (Decomposition, TabulateValuesOp, MergeTabulationOp, usize) {
    let base = values_phase(volume, block);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let tabulate =
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
            .expect("two different images");
    let merge = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        lattice,
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let (plan, rows_phase) =
        append_tabulate_phases(base, &tabulate, &merge).expect("a tabulation plan");
    (plan, tabulate, merge, rows_phase)
}

/// What one run left behind: the canonical table as **words** — which is the
/// byte-level form, since a row is its words — the decoded rows, the per-block
/// partials, the raw row blobs and the event log.
#[derive(Debug)]
struct Run {
    words: Vec<u64>,
    rows: Vec<RegionValues>,
    /// The **shape** half of the same rows, decoded by `region_shape`. A second
    /// reading of one row rather than a wider first one; see the op's header on
    /// why the label volume's measurement and the value array's are two.
    shapes: Vec<RegionShape>,
    partials: BTreeMap<[usize; 3], Vec<u8>>,
    row_blobs: BTreeMap<[usize; 3], Vec<u8>>,
    lattice_blocks: usize,
    log: Arc<ExecutionLog>,
}

fn run_at(volume: [usize; 3], block: [usize; 3], point: FixedPoint) -> Run {
    try_run(volume, block, point, label_at, value_at).expect("a run")
}

/// [`run_at`] over a named fixture, and as a `Result`.
///
/// A `Result` because the scale sweeps below need to *count* the refusals: `n`
/// is a range decision before it is a resolution one, and a sweep in which
/// nothing was ever refused would not have reached the edge it is about.
fn try_run(
    volume: [usize; 3],
    block: [usize; 3],
    point: FixedPoint,
    of_label: fn([usize; 3]) -> u64,
    of_value: fn([usize; 3]) -> f64,
) -> Result<Run> {
    let (plan, tabulate, merge, rows_phase) = tabulation_plan(volume, block, point);
    let env = ArrayEnvironment::new(labels(volume, of_label), plan.n_phases(), [4, 4, 4])
        .expect("an environment");
    let log = Arc::new(ExecutionLog::new());
    execute_phases(
        "tabulation",
        &values_workflow(volume, of_value),
        &plan,
        &Hints::default(),
        &env,
        &[log.clone()],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&tabulate),
            PhaseWork::Fragments(&merge),
        ],
    )?;

    let mut partials = BTreeMap::new();
    for key in env.sidecar_keys("partials").expect("the partial keys") {
        if key.phase != 1 {
            continue;
        }
        let bytes = env
            .read_sidecar("partials", 1, key.block)
            .expect("a read")
            .expect("a listed fragment");
        partials.insert(key.block, bytes);
    }

    // The canonical table, assembled the way a caller assembles it: every
    // block's blob into one `Table`, which orders the rows from the rows.
    let mut row_blobs = BTreeMap::new();
    let mut table = Table::new(volume, tabulation_schema(point)).expect("a table");
    fold_fragments(&env, "rows", &mut |key, bytes| {
        if key.phase != rows_phase {
            return Ok(());
        }
        row_blobs.insert(key.block, bytes.to_vec());
        table.write(key.block, bytes)
    })
    .expect("the row blobs");
    table.seal().expect("a sealed table");
    let mut words = Vec::new();
    let mut rows = Vec::new();
    let mut shapes = Vec::new();
    for row in table.scan(&Region::whole(&volume)).expect("a scan") {
        words.extend_from_slice(row.words());
        rows.push(region_values(&row, point).expect("a decoded row"));
        shapes.push(region_shape(&row).expect("a decoded shape"));
    }

    // And the caller-facing path answers the same thing, so that a consumer
    // using `collect_tabulation` is not reading a different table from the one
    // this test pins.
    let collected =
        collect_tabulation(&env, "rows", rows_phase, volume, point).expect("the collected rows");
    assert_eq!(collected, rows, "the two read paths disagree");
    let collected_shapes =
        collect_shapes(&env, "rows", rows_phase, volume, point).expect("the collected shapes");
    assert_eq!(
        collected_shapes, shapes,
        "the two shape read paths disagree"
    );

    Ok(Run {
        words,
        rows,
        shapes,
        partials,
        row_blobs,
        lattice_blocks: plan.phases[rows_phase].grid.cores().len(),
        log,
    })
}

/// The answer computed over the whole volume with no blocking anywhere: the
/// oracle, so that "every block size agrees" is not "every block size is wrong
/// the same way".
fn reference(volume: [usize; 3], point: FixedPoint) -> Vec<Tally> {
    reference_with(volume, point, label_at, value_at)
        .into_values()
        .collect()
}

/// [`reference`] over a named fixture, keyed by label — which is how the oracle
/// is looked up, since the table's canonical order is on the row's *position*.
fn reference_with(
    volume: [usize; 3],
    point: FixedPoint,
    of_label: fn([usize; 3]) -> u64,
    of_value: fn([usize; 3]) -> f64,
) -> BTreeMap<u64, Tally> {
    let mut totals: BTreeMap<u64, Tally> = BTreeMap::new();
    for z in 0..volume[0] {
        for y in 0..volume[1] {
            for x in 0..volume[2] {
                let at = [z, y, x];
                let label = of_label(at);
                if label == 0 {
                    continue;
                }
                totals
                    .entry(label)
                    .or_insert_with(|| Tally::new(label))
                    .add(at, of_value(at), point)
                    .expect("no overflow");
            }
        }
    }
    totals
}

fn row_for(rows: &[RegionValues], label: u64) -> RegionValues {
    *rows
        .iter()
        .find(|row| row.label == label)
        .unwrap_or_else(|| panic!("no row for label {label}"))
}

// ---------------------------------------- 1. decomposition invariance --

/// The acceptance criterion, and the assertion that the fixture could have
/// failed it: regions really are cut, in every run but the single-block one.
#[test]
fn a_per_region_reduction_is_byte_identical_across_block_sizes() {
    let sizes: [[usize; 3]; 5] = [[12, 4, 4], [6, 4, 4], [4, 2, 4], [3, 4, 2], [2, 2, 2]];
    let mut answers: Vec<(String, Vec<u64>)> = Vec::new();
    let point = fixed();

    for block in sizes {
        let run = run_at(VOLUME, block, point);

        // Which labels were cut. Without this the byte-identity below could be
        // the identity of five runs that never merged anything.
        let mut contributors: BTreeMap<u64, BTreeSet<[usize; 3]>> = BTreeMap::new();
        for (index, bytes) in &run.partials {
            for tally in decode_partial(bytes).expect("a partial") {
                contributors.entry(tally.label).or_default().insert(*index);
            }
        }
        let straddled: Vec<u64> = contributors
            .iter()
            .filter(|(_, who)| who.len() > 1)
            .map(|(label, _)| *label)
            .collect();
        if block == [12, 4, 4] {
            assert!(
                straddled.is_empty(),
                "the one-block lattice cannot straddle anything"
            );
        } else {
            assert!(
                !straddled.is_empty(),
                "block {block:?} put no region across a seam, so this run merges nothing"
            );
            // and specifically the region that spans the volume, whatever the cut
            assert!(
                straddled.contains(&1),
                "block {block:?} did not cut label 1, which spans axis 0"
            );
        }

        // The answer is the unblocked one, not merely a stable one. Keyed by
        // label rather than zipped, because the table's canonical order is on
        // the *position* — a row is ordered by where it is — and the oracle's
        // is on the label.
        let expected: BTreeMap<u64, Tally> = reference(VOLUME, point)
            .into_iter()
            .map(|tally| (tally.label, tally))
            .collect();
        assert_eq!(run.rows.len(), expected.len(), "block {block:?}");
        for row in &run.rows {
            let tally = expected
                .get(&row.label)
                .unwrap_or_else(|| panic!("block {block:?} invented label {}", row.label));
            assert_eq!(
                row.count, tally.count,
                "block {block:?}: label {}",
                row.label
            );
            assert_eq!(row.nonfinite, tally.nonfinite, "block {block:?}");
            assert_eq!(
                i128::from(row.sum_fixed),
                tally.sum,
                "block {block:?}: label {}",
                row.label
            );
            // The selection on its bits, which is the only comparison that
            // could catch a `-0.0` moving or a value being carried through a
            // scale on the way.
            assert_eq!(
                row.min.to_bits(),
                tally.min.unwrap_or(0.0).to_bits(),
                "block {block:?}: label {}",
                row.label
            );
            assert_eq!(
                row.max.to_bits(),
                tally.max.unwrap_or(0.0).to_bits(),
                "block {block:?}: label {}",
                row.label
            );
            assert_eq!(Some(row.at), tally.centroid());
            // The first moments, on the integers, which is the form the
            // invariance claim is about — and the quotient they derive, on its
            // bits, so that a division that moved with the cut would show.
            for axis in 0..3 {
                assert_eq!(
                    i128::from(row.moment_fixed[axis]),
                    tally.moment[axis],
                    "block {block:?}: label {} on axis {axis}",
                    row.label
                );
            }
            assert_eq!(
                row.weighted_centroid.map(|centre| centre.map(f64::to_bits)),
                tally
                    .weighted_centroid()
                    .map(|centre| centre.map(f64::to_bits)),
                "block {block:?}: label {}",
                row.label
            );
        }

        // And the shape half of the same rows, against the same oracle. The
        // whole `RegionShape` at once, on `Eq` over integers — the six second
        // moments, the coordinate sums, the count and the position are all
        // exact, so there is no comparison here that has to choose a tolerance.
        assert_eq!(run.shapes.len(), expected.len(), "block {block:?}");
        for shape in &run.shapes {
            let tally = expected
                .get(&shape.label)
                .unwrap_or_else(|| panic!("block {block:?} invented label {}", shape.label));
            assert_eq!(
                Some(*shape),
                tally.shape().expect("the oracle's shape"),
                "block {block:?}: label {}",
                shape.label
            );
            // the raw form the columns are *not* stored in comes back out of
            // them exactly, which is what says nothing was given up by centring
            assert_eq!(
                shape.second_moments_about_origin(),
                tally.second,
                "block {block:?}: label {} does not recover its moments about the volume origin",
                shape.label
            );
        }

        answers.push((format!("{block:?}"), run.words));
    }

    let (first_name, first) = &answers[0];
    for (name, words) in &answers[1..] {
        assert_eq!(
            words, first,
            "the tabulation differs between block {first_name} and block {name}; a per-region \
             reduction over a second array must not be a function of the plan"
        );
    }
}

// --------------------------------- 2. the accumulator's claim is checked --

/// The same fold as [`MergeTabulationOp`], **in `f64`**, claiming
/// `SeamFold::Unordered`.
///
/// The negative control. It is here so that "the reversal check passed" is a
/// fact about the accumulator rather than about the check never running: this op
/// sits in exactly the same place in exactly the same plan and must be refused.
struct DriftingMergeOp {
    lattice: [usize; 3],
    fold: SeamFold,
    /// Which accumulator to drift: the sum, or the **first moment**. Two
    /// controls rather than one, because they are two accumulators and "the
    /// check catches an `f64` sum" does not say it catches an `f64` moment — the
    /// moment's terms are the sum's multiplied by a coordinate, so its
    /// order-dependence is a different arrangement of bits.
    moment: bool,
}

impl FragmentOp for DriftingMergeOp {
    fn name(&self) -> &'static str {
        "drifting-merge"
    }

    /// Nothing crosses as pixels, because none are read. The partials this folds
    /// arrive on `partials` over the whole-lattice fragment reach in
    /// [`Self::inputs`]. What this control varies is [`Self::seam_fold`] and
    /// nothing else, which is what makes it a control.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own("partials", 1).with_reach(self.lattice)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(self.fold)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "drifted",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut total = 0.0f64;
        for (_, bytes) in at.fragments("partials") {
            for tally in decode_partial(bytes)? {
                total += if self.moment {
                    tally.moment[0] as f64
                } else {
                    tally.sum as f64
                };
            }
        }
        Ok(BlockOutput::fragment(
            "drifted",
            blockflow::fragment::pack_u64(&[total.to_bits()]),
        ))
    }
}

/// Four voxels of one label, whose values are `1 + 1 + 2^60 - 2^60`: `0` added
/// one at a time and `2` when the large pair goes first. One block per voxel, so
/// the merge sees four fragments and the order is a real thing.
///
/// Its **first moment** is `1*0 + 1*1 + 2^60*2 - 2^60*3`, whose exact total is
/// `1 - 2^60` — a number that needs 61 significant bits and so is not an `f64` at
/// all. The integer accumulator reports it; an `f64` one cannot.
fn wide_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 | 1 => 1.0,
        2 => (1u64 << 60) as f64,
        _ => -((1u64 << 60) as f64),
    }
}

/// The same four blocks, arranged so that the **first moment** is the
/// accumulator whose `f64` fold depends on the order.
///
/// A separate fixture and not a reuse of [`wide_value_at`], because that one
/// does not do this job: its moments are `0, 1, 2^61, -3*2^60`, and an `f64`
/// fold loses the `1` at the same step whichever end it starts from, so both
/// orders agree on `-2^60` and the reversal check has nothing to catch. That is
/// the whole reason the moment needs a control of its own — it is a *different*
/// arrangement of bits from the sum's, and a fixture that breaks one need not
/// break the other.
///
/// Here the moments are `0, 1, 3*2^59, -3*2^59`. Forwards the `1` is absorbed
/// into `3*2^59` and the pair then cancels to `0`; backwards the pair cancels
/// first and the `1` survives. `0` against `1`, and the exact answer is `1`.
fn drifting_moment_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 => 0.0,
        1 => 1.0,
        // 3 * 2^58 at coordinate 2, so the moment term is 3 * 2^59
        2 => 3.0 * (1u64 << 58) as f64,
        // -2^59 at coordinate 3, so the moment term is -3 * 2^59
        _ => -((1u64 << 59) as f64),
    }
}

fn one_label(_at: [usize; 3]) -> u64 {
    1
}

fn run_wide(
    fold: Option<(SeamFold, bool)>,
    of_value: fn([usize; 3]) -> f64,
) -> Result<Vec<([usize; 3], Vec<u8>)>> {
    // Zero fraction bits, so the range is +/- 2^63 and `2^60` fits. The
    // resolution is then 1.0, which is exactly what a caller trades away to
    // reduce values this large — and it is `FixedPoint`'s whole point that the
    // trade is theirs and is readable.
    let point = FixedPoint::bits(0).expect("zero fraction bits");
    let volume = [4usize, 1, 1];
    let base = values_phase(volume, [1, 1, 1]);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let tabulate =
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)?;
    let honest = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        lattice,
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let drifting = fold.map(|(fold, moment)| DriftingMergeOp {
        lattice,
        fold,
        moment,
    });

    let (plan, rows_phase) = append_tabulate_phases(base, &tabulate, &honest)?;
    let env = ArrayEnvironment::new(labels(volume, one_label), plan.n_phases(), [1, 1, 1])?;
    let merge: &dyn FragmentOp = match &drifting {
        Some(op) => op,
        None => &honest,
    };
    execute_phases(
        "wide",
        &values_workflow(volume, of_value),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&tabulate),
            PhaseWork::Fragments(merge),
        ],
    )?;
    let stream = if drifting.is_some() {
        "drifted"
    } else {
        "rows"
    };
    let mut blobs = Vec::new();
    fold_fragments(&env, stream, &mut |key, bytes| {
        if key.phase == rows_phase {
            blobs.push((key.block, bytes.to_vec()));
        }
        Ok(())
    })?;
    Ok(blobs)
}

#[test]
fn the_fixed_point_merge_passes_the_reversal_check_on_the_fixture_that_breaks_f64() {
    // The honest merge, over four fragments whose `f64` sum is order-dependent.
    // It runs, and the executor applied every block twice to establish that.
    let blobs = run_wide(None, wide_value_at).expect("an integer fold is order-independent");
    let point = FixedPoint::bits(0).unwrap();
    let mut table = Table::new([4, 1, 1], tabulation_schema(point)).unwrap();
    assert_eq!(blobs.len(), 4, "one blob per block, even the empty ones");
    for (block, bytes) in &blobs {
        table.write(*block, bytes).unwrap();
    }
    table.seal().unwrap();
    let rows: Vec<RegionValues> = table
        .scan(&Region::whole(&[4usize, 1, 1]))
        .unwrap()
        .map(|row| region_values(&row, point).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].count, 4);
    // `1 + 1 + 2^60 - 2^60` is 2, exactly, in every order.
    assert_eq!(rows[0].sum_fixed, 2);
    // And the selection is the voxels' own values, whose bits the reversal
    // check has just compared four times over.
    assert_eq!(rows[0].max.to_bits(), ((1u64 << 60) as f64).to_bits());
    assert_eq!(rows[0].min.to_bits(), (-((1u64 << 60) as f64)).to_bits());

    // **The first moment came through the same check.** `1*0 + 1*1 + 2^60*2 -
    // 2^60*3` is `1 - 2^60` exactly, which is not an `f64`: the accumulator is
    // integer, so the answer is the integer and the reversal check just compared
    // its bytes four times over. At zero fraction bits the scale is 1, so the
    // column word is the moment itself.
    assert_eq!(rows[0].moment_fixed[0], 1 - (1i64 << 60));
    assert_eq!(rows[0].moment_fixed[1], 0, "the volume is one voxel wide");
    assert_eq!(rows[0].moment_fixed[2], 0);

    // And the quotient, which is where the signed-value rule becomes visible:
    // `(1 - 2^60) / 2` is nowhere near the four voxels it was measured over.
    let centre = rows[0]
        .weighted_centroid
        .expect("the denominator is 2, not 0");
    assert_eq!(centre[0], (1.0 - (1u64 << 60) as f64) / 2.0);
    assert!(
        centre[0] < 0.0,
        "the weighted centre is {} and the region is z in 0..4 — a signed value array makes this \
         a ratio and not a point, which is the stated behaviour",
        centre[0]
    );
    // while the geometric centre stayed where the voxels are
    assert_eq!(rows[0].at, [2, 0, 0]);
}

/// The same claim on the fixture built to break an `f64` **moment** — see
/// [`drifting_moment_value_at`] — so that the moment's passing the reversal
/// check is a fact about a fold that had somewhere to drift to.
#[test]
fn the_fixed_point_moment_passes_the_reversal_check_on_the_fixture_that_breaks_f64() {
    let blobs = run_wide(None, drifting_moment_value_at)
        .expect("an integer moment fold is order-independent");
    let point = FixedPoint::bits(0).unwrap();
    let mut table = Table::new([4, 1, 1], tabulation_schema(point)).unwrap();
    for (block, bytes) in &blobs {
        table.write(*block, bytes).unwrap();
    }
    table.seal().unwrap();
    let rows: Vec<RegionValues> = table
        .scan(&Region::whole(&[4usize, 1, 1]))
        .unwrap()
        .map(|row| region_values(&row, point).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].count, 4);
    // `0 + 1 + 3*2^59 - 3*2^59` is 1, exactly, in every order. In `f64` it is
    // `0` one way round and `1` the other.
    assert_eq!(rows[0].moment_fixed[0], 1);
    assert_eq!(
        1.0f64 + 3.0 * (1u64 << 59) as f64 - 3.0 * (1u64 << 59) as f64,
        0.0,
        "the fixture was supposed to lose the one when the large pair goes last"
    );
    assert_eq!(
        -3.0 * (1u64 << 59) as f64 + 3.0 * (1u64 << 59) as f64 + 1.0,
        1.0,
        "and to keep it when the large pair goes first"
    );
    // The sum is `1 + 2^58` and the quotient is a tiny fraction of a voxel — a
    // weighted centre pulled almost all the way to the origin by the one voxel
    // whose value did not cancel.
    assert_eq!(rows[0].sum_fixed, 1 + (1i64 << 58));
    assert_eq!(
        rows[0].weighted_centroid,
        Some([1.0 / (1.0 + (1u64 << 58) as f64), 0.0, 0.0])
    );
}

#[test]
fn an_f64_fold_in_the_same_place_claiming_unordered_is_refused_by_name() {
    let failed = run_wide(Some((SeamFold::Unordered, false)), wide_value_at)
        .expect_err("an `f64` sum over four fragments is not order-independent")
        .to_string();
    assert!(failed.contains("drifting-merge"), "{failed}");
    assert!(failed.contains("SeamFold::Unordered"), "{failed}");
    assert!(failed.contains("opposite order"), "{failed}");
    assert!(failed.contains("fixed-point"), "{failed}");

    // The same control on the **first moment**, which is a different arrangement
    // of bits and therefore a different fact: an `f64` moment over these four
    // fragments is order-dependent too, and the check catches that as well.
    // Without this, "the reversal check passed for the moment" would rest on the
    // check having been observed to run for the sum.
    let moment_failed = run_wide(Some((SeamFold::Unordered, true)), drifting_moment_value_at)
        .expect_err("an `f64` first moment over four fragments is not order-independent")
        .to_string();
    assert!(moment_failed.contains("drifting-merge"), "{moment_failed}");
    assert!(
        moment_failed.contains("SeamFold::Unordered"),
        "{moment_failed}"
    );

    // And declared honestly they run — which is what makes the refusals above a
    // statement about the accumulators rather than about the plan shape.
    run_wide(Some((SeamFold::OrderDependent, false)), wide_value_at)
        .expect("an order-dependent sum fold is admitted");
    run_wide(
        Some((SeamFold::OrderDependent, true)),
        drifting_moment_value_at,
    )
    .expect("an order-dependent moment fold is admitted");
}

// ------------------------------------------------------ 3. ground truth --

/// Region sizes against `Scene::object_table`, which is a **ground truth**
/// rather than a second opinion: the scene knows how many voxels it gave each
/// object, and the count in the table has to be that number.
#[test]
fn region_sizes_equal_the_scene_object_voxel_counts() {
    let volume = [24usize, 24, 24];
    let scene = Scene::new(SceneSpec::clean(volume, 20250817).with_objects(12)).expect("a scene");
    let truth = scene.object_table();
    let rendered = scene.render();
    let point = fixed();

    let block = [8usize, 12, 24];
    let base = values_phase(volume, block);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let tabulate =
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
            .expect("two images");
    let merge = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        lattice,
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let (plan, rows_phase) = append_tabulate_phases(base, &tabulate, &merge).expect("a plan");

    let mut as_f64 = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
    for (index, slot) in as_f64.indexed_iter_mut() {
        *slot = rendered.labels[[index.0, index.1, index.2]] as f64;
    }
    let env = ArrayEnvironment::new(as_f64.into(), plan.n_phases(), [8, 8, 8]).expect("an env");
    execute_phases(
        "ground-truth",
        &values_workflow(volume, value_at),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&tabulate),
            PhaseWork::Fragments(&merge),
        ],
    )
    .expect("a run");
    let rows = collect_tabulation(&env, "rows", rows_phase, volume, point).expect("the rows");

    let expected: BTreeMap<u64, &blockflow::synthetic::ObjectRecord> = truth
        .iter()
        .filter(|record| record.voxels > 0)
        .map(|record| (record.id as u64, record))
        .collect();
    assert!(expected.len() > 1, "a one-object scene tests nothing");
    assert_eq!(
        rows.len(),
        expected.len(),
        "the table has a row per object that has voxels and no others"
    );
    for row in &rows {
        let record = expected.get(&row.label).unwrap_or_else(|| {
            panic!(
                "the table has a row for label {}, which the scene does not have",
                row.label
            )
        });
        assert_eq!(
            row.count, record.voxels,
            "label {} is {} voxels in the table and {} in the scene",
            row.label, row.count, record.voxels
        );
        // The centroid too, exactly: the scene measures voxel *centres* and the
        // table measures voxel *indices*, so the two differ by half a voxel and
        // by nothing else.
        for axis in 0..3 {
            assert!(
                (row.centroid[axis] + 0.5 - record.centroid[axis]).abs() < 1e-9,
                "label {} centroid on axis {axis}: {} against {}",
                row.label,
                row.centroid[axis] + 0.5,
                record.centroid[axis]
            );
        }
    }
}

// ----------------------------------------------- 4. the degenerate cases --

/// All four, in one fixture, against one run — and each is a *stated* outcome
/// rather than one discovered by whoever hits it first.
#[test]
fn the_degenerate_cases_answer_the_way_the_op_says_they_do() {
    let point = fixed();
    // Cut on axis 0 only, so that "spans every block" is literally every block
    // of the lattice rather than every block the line passes through.
    let run = run_at(VOLUME, [3, 4, 4], point);
    let labels: Vec<u64> = run.rows.iter().map(|row| row.label).collect();

    // A label in no voxel gets no row. There is no way for the op to know 4 was
    // expected: it reads voxels, not a caller's numbering. The order is the
    // table's canonical one, which is on the row's *position* — label 5 sits at
    // [6, 3, 0] and label 3 at [7, 3, 3].
    assert_eq!(labels, vec![1, 2, 5, 3], "label 4 is in no voxel");

    // A label in exactly one voxel.
    let single = row_for(&run.rows, 3);
    assert_eq!(single.count, 1);
    assert_eq!(single.at, [7, 3, 3]);
    // The selection is that voxel's value **exactly**, and the sum is its
    // quantisation — which is the same number only when the value was already on
    // the fixed point's lattice, and is why the two are asserted differently.
    assert_eq!(single.min.to_bits(), value_at([7, 3, 3]).to_bits());
    assert_eq!(single.max.to_bits(), single.min.to_bits());
    assert!(
        (single.sum - value_at([7, 3, 3])).abs() <= point.resolution(),
        "one voxel's sum is that voxel's value, to the resolution"
    );

    // A region spanning every block: twelve voxels on axis 0, one of them a
    // `NaN`, folded from every block of the lattice.
    let spanning = row_for(&run.rows, 1);
    assert_eq!(spanning.count, 12);
    assert_eq!(spanning.nonfinite, 1, "the `NaN` at [9, 0, 0]");
    let mut blocks = BTreeSet::new();
    for (index, bytes) in &run.partials {
        if decode_partial(bytes)
            .expect("a partial")
            .iter()
            .any(|tally| tally.label == 1)
        {
            blocks.insert(*index);
        }
    }
    assert_eq!(
        blocks.len(),
        run.partials.len(),
        "label 1 was supposed to touch every block"
    );

    // Negatives are exact, and a disconnected region is one region.
    let split = row_for(&run.rows, 5);
    assert_eq!(split.count, 4, "two pieces of two voxels each");
    let expected: f64 = [[0, 3, 0], [1, 3, 0], [10, 3, 0], [11, 3, 0]]
        .into_iter()
        .map(value_at)
        .sum();
    assert!(expected < 0.0, "the fixture was supposed to go negative");
    assert!((split.sum - expected).abs() <= 4.0 * point.resolution());
    assert!(split.min < 0.0 && split.min <= split.max);

    // `NaN` and infinity are set aside and counted, not folded in.
    let straddling = row_for(&run.rows, 2);
    assert_eq!(straddling.nonfinite, 2, "one `NaN` and one infinity");
    assert!(
        straddling.sum.is_finite() && straddling.max.is_finite(),
        "a non-finite value reached the reductions"
    );
    assert!(straddling.count > straddling.nonfinite);
    assert!(!straddling.all_nonfinite());
    // And precisely: the infinity at [6, 1, 1] is not the maximum. A `max` that
    // came back `+inf` because one voxel was broken is the failure the exclusion
    // rule exists to prevent, and a total order over the values alone would
    // otherwise have ranked it above every real one.
    let mut widest = f64::NEG_INFINITY;
    let mut narrowest = f64::INFINITY;
    for z in 0..VOLUME[0] {
        for y in 0..VOLUME[1] {
            for x in 0..VOLUME[2] {
                let at = [z, y, x];
                let value = value_at(at);
                if label_at(at) == 2 && value.is_finite() {
                    widest = widest.max(value);
                    narrowest = narrowest.min(value);
                }
            }
        }
    }
    assert_eq!(straddling.max.to_bits(), widest.to_bits());
    assert_eq!(straddling.min.to_bits(), narrowest.to_bits());

    // A region with no finite value at all reports `0.0` for both, and the
    // count is what says why — the same statement, from the other side.
    let mut nothing = Tally::new(11);
    nothing
        .add([0, 0, 0], f64::NAN, point)
        .expect("a non-finite value is not a refusal");
    nothing
        .add([1, 0, 0], f64::NEG_INFINITY, point)
        .expect("nor is an infinity");
    assert_eq!(nothing.nonfinite, nothing.count);
    assert_eq!(nothing.min, None);
    assert_eq!(nothing.max, None);
}

/// The rows partition: every block emits only the rows it owns, so a table
/// assembled from all of them holds each region exactly once. Without the
/// ownership rule every block would write the whole answer and the table would
/// hold every row once per block — `Table::write` has no ownership rule of its
/// own and would be right not to.
#[test]
fn every_region_is_written_by_exactly_one_block() {
    let point = fixed();
    let run = run_at(VOLUME, [3, 2, 2], point);

    let mut seen: BTreeMap<u64, Vec<[usize; 3]>> = BTreeMap::new();
    for (block, bytes) in &run.row_blobs {
        let mut table = Table::new(VOLUME, tabulation_schema(point)).expect("a table");
        table.write(*block, bytes).expect("a blob");
        table.seal().expect("a sealed table");
        for row in table.scan(&Region::whole(&VOLUME)).expect("a scan") {
            seen.entry(region_values(&row, point).expect("a row").label)
                .or_default()
                .push(*block);
        }
    }

    assert_eq!(
        run.row_blobs.len(),
        run.lattice_blocks,
        "every block writes a blob, even one with no rows — present and empty is a fact"
    );
    assert_eq!(seen.keys().copied().collect::<Vec<u64>>(), vec![1, 2, 3, 5]);
    for (label, who) in &seen {
        assert_eq!(who.len(), 1, "label {label} was written by {who:?}");
    }
}

/// The second array is read, and charged for, as a read of the image it names.
/// A second array that cost nothing in the counters would make every measurement
/// of this op a measurement of a different plan.
#[test]
fn both_arrays_are_read_and_counted_as_reads_of_their_own_images() {
    let block = [3usize, 4, 4];
    let (plan, _, _, _) = tabulation_plan(VOLUME, block, fixed());
    // Image 0 and image 1 are both the tabulating phase's operands; nothing else
    // reads image 1, and the merge reads no pixels at all.
    assert_eq!(plan.phases[1].source_images, vec![0, 1]);
    assert_eq!(plan.phases[0].source_images, Vec::<usize>::new());
    assert_eq!(plan.phases[2].source_images, Vec::<usize>::new());
    // The tabulating phase's operands are voxelwise, so it is granted no halo:
    // every voxel is read by the one block whose core holds it.
    assert_eq!(plan.phases[1].halo, [0, 0, 0]);
    assert_eq!(plan.phases[1].reach, [0, 0, 0]);
    // The merge reaches the whole lattice, which is what makes every block's
    // partial a dependency of every block's merge.
    assert_eq!(plan.phases[2].halo, VOLUME);

    // And the counters agree with the plan. Image 0 is read once per block by
    // the pixel phase and once per block as the tabulating phase's first
    // operand; image 1 once per block as its second. Nothing reads image 2 —
    // the tabulating phase declares `reads_pixels() == false` and the merge is
    // fragments to fragments.
    let run = run_at(VOLUME, block, fixed());
    let blocks = VOLUME[0] / block[0];
    assert_eq!(run.partials.len(), blocks, "every block wrote a partial");
    let mut per_image: BTreeMap<usize, usize> = BTreeMap::new();
    for event in run.log.events() {
        if let Event::RegionRead { image, .. } = event {
            *per_image.entry(image).or_default() += 1;
        }
    }
    assert_eq!(per_image.get(&0).copied().unwrap_or(0), 2 * blocks);
    assert_eq!(per_image.get(&1).copied().unwrap_or(0), blocks);
    assert_eq!(per_image.get(&2).copied().unwrap_or(0), 0);
}

/// A tabulation at one scale cannot be read as one at another. The scale is in
/// the column names, so the schema in the blob and the schema of the table
/// disagree and the write is refused rather than silently rescaled.
#[test]
fn two_fixed_point_scales_are_two_schemas_and_do_not_mix() {
    let coarse = FixedPoint::bits(4).expect("four bits");
    let run = run_at(VOLUME, [4, 4, 4], coarse);
    assert!(!run.words.is_empty(), "the coarse run produced no rows");

    let mut table = Table::new(VOLUME, tabulation_schema(fixed())).expect("a table at 20 bits");
    let (block, bytes) = run
        .row_blobs
        .iter()
        .next()
        .expect("some block wrote a blob");
    let failed = table
        .write(*block, bytes)
        .expect_err("a 4-bit blob is not a 20-bit table")
        .to_string();
    assert!(failed.contains("q4") || failed.contains("q20"), "{failed}");
}

/// The producer-side arithmetic, driven without a plan at all: the shell is thin
/// and the reduction is the part worth checking on its own.
#[test]
fn a_block_can_be_tallied_without_a_run() {
    let point = fixed();
    let op = TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
        .expect("two images");
    let read = Region::new(&[2, 0, 0], &[2, 4, 4]);
    let mut label_block = Array3::<f64>::zeros((2, 4, 4));
    let mut value_block = Array3::<f64>::zeros((2, 4, 4));
    for (index, slot) in label_block.indexed_iter_mut() {
        let at = [2 + index.0, index.1, index.2];
        *slot = label_at(at) as f64;
        value_block[[index.0, index.1, index.2]] = value_at(at);
    }
    let tallies = op
        .tally_block(
            &blockflow::env::BlockBuf::Array(label_block.into()),
            &blockflow::env::BlockBuf::Array(value_block.into()),
            &read,
            &read,
        )
        .expect("a tally");
    // z in 2..4: label 1 at y=0,x=0 for both slices, label 2 at z=3 only.
    assert_eq!(tallies[&1].count, 2);
    assert_eq!(tallies[&2].count, 4);
    assert!(!tallies.contains_key(&3));
    assert_eq!(
        tallies[&1].sum,
        point.quantise(value_at([2, 0, 0])).unwrap().unwrap()
            + point.quantise(value_at([3, 0, 0])).unwrap().unwrap()
    );
}

// ------------------------------------------- 5. the first moment, which is
// ---------------------------------- the one column the others do not give --

/// **The separating fixture, built for this and for nothing else.**
///
/// A symmetric arrangement cannot tell a weighted centroid from an unweighted
/// one — put the same weight on every voxel, or arrange the weights evenly about
/// the centre, and the two coincide, so an implementation that ignored the value
/// array entirely would pass. Every choice below is there to stop that:
///
/// * the weight is a **product of a per-axis exponential**, so it rises steeply
///   and *independently* on all three axes and the two centres separate on each
///   of them. An implementation that dropped an axis, permuted them, or divided
///   by `count` rather than by `sum` moves the answer on at least one;
/// * `1.7` is **not a dyadic rational**, so `1.7^z` is not on the fixed point's
///   lattice and the moment is a sum of genuinely quantised terms rather than of
///   integers that happened to be exact;
/// * every value is **positive**, so each region's weighted centre is a convex
///   combination of its own coordinates and therefore a point inside its
///   bounding box — which is what makes "it differs from the unweighted centre
///   by 0.25 or more on every axis" a statement about the weighting rather than
///   about a denominator near zero;
/// * the two regions split axis 1 and are whole on axes 0 and 2, so **every cut
///   on axis 0 or axis 2 puts both of them across a seam** and the merge is
///   doing real work in every run but the single-block one.
///
/// The separation it buys is a quarter of a voxel on the narrowest axis and four
/// voxels on the widest; the assertion asks for a fifth, which is under the
/// former by enough that the quantisation cannot close it and over zero by
/// enough that it is not asking whether two floats happen to be unequal.
const SEPARATING_VOLUME: [usize; 3] = [12, 4, 4];

fn separating_labels(at: [usize; 3]) -> u64 {
    if at[1] < 2 {
        1
    } else {
        2
    }
}

fn separating_value_at(at: [usize; 3]) -> f64 {
    let [z, y, x] = at;
    1.7f64.powi(z as i32) * 3.0f64.powi(y as i32) * 2.5f64.powi(x as i32)
}

/// **The acceptance criterion for the new column**, on a fixture where a broken
/// implementation has somewhere to be wrong.
///
/// Byte-identical against a whole-volume reference across five block sizes, with
/// the weighted and unweighted centres required to *differ* on every axis of
/// every region — so five runs that all reported the geometric centroid would
/// agree with each other perfectly and fail here on the first assertion.
#[test]
fn the_weighted_centroid_is_byte_identical_across_block_sizes_where_it_differs_from_the_unweighted()
{
    let point = fixed();
    let sizes: [[usize; 3]; 5] = [[12, 4, 4], [6, 4, 4], [4, 2, 4], [3, 4, 2], [2, 2, 2]];
    let mut answers: Vec<(String, Vec<u64>)> = Vec::new();

    for block in sizes {
        let run = try_run(
            SEPARATING_VOLUME,
            block,
            point,
            separating_labels,
            separating_value_at,
        )
        .expect("twenty fraction bits holds this fixture");

        // Which labels were cut. Without this the byte-identity below could be
        // the identity of five runs that never merged a moment.
        let mut contributors: BTreeMap<u64, BTreeSet<[usize; 3]>> = BTreeMap::new();
        for (index, bytes) in &run.partials {
            for tally in decode_partial(bytes).expect("a partial") {
                contributors.entry(tally.label).or_default().insert(*index);
            }
        }
        let straddled: Vec<u64> = contributors
            .iter()
            .filter(|(_, who)| who.len() > 1)
            .map(|(label, _)| *label)
            .collect();
        if block == [12, 4, 4] {
            assert!(
                straddled.is_empty(),
                "the one-block lattice cannot straddle anything"
            );
        } else {
            assert_eq!(
                straddled,
                vec![1, 2],
                "block {block:?} left a region inside one block, so this run merges less than the \
                 fixture was built to make it merge"
            );
        }

        let expected = reference_with(
            SEPARATING_VOLUME,
            point,
            separating_labels,
            separating_value_at,
        );
        assert_eq!(run.rows.len(), 2, "block {block:?}");
        for row in &run.rows {
            let tally = expected
                .get(&row.label)
                .unwrap_or_else(|| panic!("block {block:?} invented label {}", row.label));

            // The moments, on the integers — the form the invariance claim is
            // about, since the quotient is taken from them and not the reverse.
            for axis in 0..3 {
                assert_eq!(
                    i128::from(row.moment_fixed[axis]),
                    tally.moment[axis],
                    "block {block:?}: label {} on axis {axis}",
                    row.label
                );
            }
            // And the quotient, on its bits.
            let weighted = row
                .weighted_centroid
                .expect("every value in this fixture is positive, so the denominator is not zero");
            assert_eq!(
                weighted.map(f64::to_bits),
                tally
                    .weighted_centroid()
                    .expect("the oracle agrees there is one")
                    .map(f64::to_bits),
                "block {block:?}: label {}",
                row.label
            );

            // The unweighted centre is the box centre, exactly — the regions are
            // solid boxes — so the two quantities are here side by side and the
            // separation below is between two numbers a reader can check.
            let box_centre = [5.5, if row.label == 1 { 0.5 } else { 2.5 }, 1.5];
            assert_eq!(row.centroid, box_centre, "label {}", row.label);
            for axis in 0..3 {
                assert!(
                    (weighted[axis] - box_centre[axis]).abs() > 0.2,
                    "block {block:?}: label {} has weighted centre {} and box centre {} on axis \
                     {axis}. They have to differ, or this fixture cannot tell the two apart and \
                     would pass an implementation that never read the value array.",
                    row.label,
                    weighted[axis],
                    box_centre[axis]
                );
                // and the weight being positive keeps it a point in the region
                assert!(
                    weighted[axis] >= 0.0 && weighted[axis] <= (SEPARATING_VOLUME[axis] - 1) as f64,
                    "label {} left the volume on axis {axis} under a positive weight",
                    row.label
                );
            }
            // The weight rises with every coordinate, so the weighted centre is
            // on the far side of the box centre on every axis and not merely a
            // different number from it.
            for axis in 0..3 {
                assert!(
                    weighted[axis] > box_centre[axis],
                    "label {} on axis {axis}: a weight increasing in the coordinate has to pull \
                     the centre up",
                    row.label
                );
            }
        }

        answers.push((format!("{block:?}"), run.words));
    }

    let (first_name, first) = &answers[0];
    for (name, words) in &answers[1..] {
        assert_eq!(
            words, first,
            "the weighted centroid differs between block {first_name} and block {name}; a cross \
             moment of value against position must not be a function of the plan"
        );
    }
}

/// One region of four voxels, through a real run of four blocks, with **the
/// arithmetic written out** rather than taken from a helper.
///
/// Values `1, 1, 1, 5` at `z = 0, 1, 2, 3`, all dyadic and so exact at twenty
/// fraction bits:
///
/// * `sum = 1 + 1 + 1 + 5 = 8`;
/// * `moment_0 = 1*0 + 1*1 + 1*2 + 5*3 = 18`;
/// * weighted centre `= 18 / 8 = 2.25`;
/// * unweighted centre `= (0 + 1 + 2 + 3) / 4 = 1.5`, and the row sits at `2`,
///   which is that rounded half up.
///
/// The two differ by `0.75`, which is the whole point of writing them both down.
fn hand_value_at(at: [usize; 3]) -> f64 {
    if at[0] == 3 {
        5.0
    } else {
        1.0
    }
}

#[test]
fn a_hand_computed_region_reports_the_weighted_centre_the_arithmetic_says() {
    let point = fixed();
    let run = try_run(AWKWARD_VOLUME, [1, 1, 1], point, one_label, hand_value_at)
        .expect("a run of four blocks");
    assert_eq!(
        run.partials.len(),
        4,
        "four fragments, so the merge is real"
    );
    let region = row_for(&run.rows, 1);

    let one = 1i64 << 20; // one unit of value, in fixed-point steps
    assert_eq!(region.count, 4);
    assert_eq!(region.sum_fixed, 8 * one);
    assert_eq!(region.sum, 8.0);
    assert_eq!(region.moment_fixed[0], 18 * one);
    assert_eq!(region.moment[0], 18.0);
    assert_eq!(region.moment_fixed[1], 0);
    assert_eq!(region.moment_fixed[2], 0);

    assert_eq!(region.weighted_centroid, Some([2.25, 0.0, 0.0]));
    assert_eq!(region.centroid, [1.5, 0.0, 0.0]);
    assert_eq!(region.at, [2, 0, 0]);
    // 18/8 against 6/4: the two are different numbers over the same four voxels,
    // and no arrangement of `count`, `sum` and `sum_0` gives the first.
    assert_eq!(
        region.weighted_centroid.unwrap()[0] - region.centroid[0],
        0.75
    );
}

/// The degenerate denominator, through a run rather than on a tally: a region
/// whose finite values cancel has **no** weighted centroid, and the row says so
/// by carrying `None` — not a `NaN`, not the unweighted centre.
///
/// Two regions in one run, so the answer is a property of the region and not of
/// the table: the second has a perfectly good weighted centre in the same rows.
fn cancelling_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 => 3.0,
        1 => -3.0,
        _ => 1.0,
    }
}

#[test]
fn a_region_whose_values_cancel_reports_no_weighted_centroid_at_all() {
    let point = fixed();
    let run = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        point,
        two_region_labels,
        cancelling_value_at,
    )
    .expect("a run of four blocks");
    let one = 1i64 << 20;

    // Region 1 is `+3` at z = 0 and `-3` at z = 1.
    let cancelled = row_for(&run.rows, 1);
    assert_eq!(cancelled.count, 2);
    assert_eq!(cancelled.nonfinite, 0, "both values are finite");
    assert_eq!(cancelled.sum_fixed, 0, "the denominator is exactly zero");
    // The numerator is still exact, still written, and is not zero: `3*0 - 3*1`.
    assert_eq!(cancelled.moment_fixed[0], -3 * one);
    assert_eq!(cancelled.moment[0], -3.0);
    // So the quotient does not exist, and that is what the row carries.
    assert_eq!(
        cancelled.weighted_centroid, None,
        "a zero denominator has no quotient; reporting one would be inventing it"
    );
    // The geometric centre is unaffected — the absence is of one quantity, not
    // of the row.
    assert_eq!(cancelled.centroid, [0.5, 0.0, 0.0]);
    assert_eq!(cancelled.count, 2);

    // Region 2 is `1` at z = 2 and z = 3, in the same table, and has one.
    let ordinary = row_for(&run.rows, 2);
    assert_eq!(ordinary.sum_fixed, 2 * one);
    assert_eq!(ordinary.moment_fixed[0], 5 * one, "1*2 + 1*3");
    assert_eq!(ordinary.weighted_centroid, Some([2.5, 0.0, 0.0]));

    // And a region of nothing but zeros is the same fact by the other road:
    // `0/0` rather than `k/0`, and the same `None`.
    let zeros = try_run(AWKWARD_VOLUME, [2, 1, 1], point, one_label, |_| 0.0)
        .expect("a run over a zero array");
    let flat = row_for(&zeros.rows, 1);
    assert_eq!(flat.count, 4);
    assert_eq!(flat.sum_fixed, 0);
    assert_eq!(flat.moment_fixed, [0, 0, 0]);
    assert_eq!(flat.weighted_centroid, None);
    assert_eq!(flat.centroid, [1.5, 0.0, 0.0]);
}

// ------------------------------- 6. the selection carries no fixed point --

/// A value the fixed point **cannot hold at any scale this fixture admits**.
///
/// One unit in the last place above `1.0`, so `AWKWARD * 2^n` is a whole number
/// only for `n >= 52`. [`WIDE`] is `2^11`, which the op refuses at every `n >=
/// 52` because a value's range is `+/- 2^(63-n)`. The two windows do not
/// overlap, so there is no `n` at which a quantised selection would have come
/// back as the voxel's own value — which is what makes the byte-identity
/// assertion below discriminating rather than lucky.
const AWKWARD: f64 = 1.0 + f64::EPSILON;
/// See [`AWKWARD`]. Also large enough that the two voxels holding it total
/// `2^63` at 51 fraction bits, so the *sum*'s range closes that scale too.
const WIDE: f64 = 2048.0;

/// Two regions on `[4, 1, 1]`: the awkward pair, then the wide pair.
fn two_region_labels(at: [usize; 3]) -> u64 {
    if at[0] < 2 {
        1
    } else {
        2
    }
}

fn awkward_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 => AWKWARD,
        1 => -AWKWARD,
        _ => WIDE,
    }
}

const AWKWARD_VOLUME: [usize; 3] = [4, 1, 1];

/// **The selection is the voxel's own value, bit for bit, at every scale the run
/// admits** — and a quantised column would have differed at every one of them.
///
/// The second half is what makes this a test rather than a tautology. It is not
/// enough that `min` and `max` come back exact; the fixture has to be one where
/// carrying them through the fixed point *visibly* could not have. So the same
/// value is quantised and unquantised at the same scale in the same loop, and
/// the two are required to disagree.
#[test]
fn the_selection_is_the_voxels_own_value_at_every_scale_the_run_admits() {
    let mut accepted = 0usize;
    let mut refused = 0usize;
    for bits in 0..=MAX_FRACTION_BITS {
        let point = FixedPoint::bits(bits).expect("in range");
        // One block per voxel, so the awkward region is cut and its selection
        // reaches the merge rather than staying inside one block.
        let run = match try_run(
            AWKWARD_VOLUME,
            [1, 1, 1],
            point,
            two_region_labels,
            awkward_value_at,
        ) {
            Ok(run) => run,
            Err(_) => {
                refused += 1;
                continue;
            }
        };
        accepted += 1;
        let region = row_for(&run.rows, 1);
        assert_eq!(
            region.max.to_bits(),
            AWKWARD.to_bits(),
            "at {bits} fraction bits the maximum came back as {} and the voxel holds {AWKWARD}",
            region.max
        );
        assert_eq!(
            region.min.to_bits(),
            (-AWKWARD).to_bits(),
            "at {bits} fraction bits the minimum came back as {}",
            region.min
        );
        // and the discrimination: the same value through the fixed point is a
        // different number, at this very scale
        let quantised = point.value_of(
            point
                .quantise(AWKWARD)
                .expect("in range, or the run would have refused")
                .expect("finite") as i64,
        );
        assert_ne!(
            quantised.to_bits(),
            AWKWARD.to_bits(),
            "at {bits} fraction bits the fixed point holds {AWKWARD} exactly, so this scale does \
             not discriminate a quantised selection from an unquantised one"
        );
    }
    println!(
        "{accepted} scales accepted, {refused} refused, and every accepted one selects the \
              voxel's own bits"
    );
    assert!(accepted > 1, "one scale is not a sweep");
    assert!(
        refused > 0,
        "no scale in `0..={MAX_FRACTION_BITS}` was refused, so this fixture never reached the \
         range edge that closes the window a quantised selection would have needed"
    );
}

/// The same fixture, cut three ways: the selection is a function of the region's
/// voxels and not of the plan, which is the acceptance criterion the whole file
/// is about, restated for the column that stopped being an accumulator.
#[test]
fn the_selection_is_byte_identical_across_block_sizes() {
    let point = FixedPoint::default();
    let mut answers: Vec<(String, Vec<u64>)> = Vec::new();
    let mut straddled = 0usize;
    for block in [[4usize, 1, 1], [2, 1, 1], [1, 1, 1]] {
        let run = try_run(
            AWKWARD_VOLUME,
            block,
            point,
            two_region_labels,
            awkward_value_at,
        )
        .expect("twenty fraction bits holds this fixture");
        let region = row_for(&run.rows, 1);
        assert_eq!(region.max.to_bits(), AWKWARD.to_bits(), "block {block:?}");
        assert_eq!(
            region.min.to_bits(),
            (-AWKWARD).to_bits(),
            "block {block:?}"
        );
        // and the fixture could have failed: one of these cuts has to put the
        // region across a seam, or the byte-identity below is the identity of
        // three runs that merged nothing.
        let cut = run
            .partials
            .values()
            .filter(|bytes| {
                decode_partial(bytes)
                    .expect("a partial")
                    .iter()
                    .any(|tally| tally.label == 1)
            })
            .count();
        if cut > 1 {
            straddled += 1;
        }
        answers.push((format!("{block:?}"), run.words));
    }
    assert!(
        straddled > 0,
        "no cut put the awkward region across a seam, so none of these runs merged a selection"
    );
    let (first_name, first) = &answers[0];
    for (name, words) in &answers[1..] {
        assert_eq!(
            words, first,
            "the table differs between block {first_name} and block {name}"
        );
    }
}

/// `-0.0` against `0.0`, **through the executor's reversal check**.
///
/// This is acceptance criterion 1 for the selection, and it is deliberately not
/// an assertion about `least`: the merge here has four fragments, the phase
/// declares `SeamFold::Unordered`, so the executor applies every block a second
/// time with its neighbourhood reversed and requires byte-identical output.
/// A run that returns at all is that check having passed on a column that is now
/// `f64`. The fixture is the one pair that could have broken it — two values
/// that compare equal and are different bits, where `f64::min` is free to return
/// either operand — and the row that comes back says which one the total order
/// picks.
fn signed_zero_at(at: [usize; 3]) -> f64 {
    if at[0] == 0 {
        -0.0
    } else {
        0.0
    }
}

#[test]
fn a_signed_zero_survives_the_reversal_check_and_keeps_its_sign() {
    let run = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        fixed(),
        one_label,
        signed_zero_at,
    )
    .expect("a selection in `f64` is order-independent, and the executor just checked it");
    assert_eq!(
        run.partials.len(),
        4,
        "four fragments, so the order is real"
    );
    let region = row_for(&run.rows, 1);
    assert_eq!(region.count, 4);
    assert_eq!(
        region.min.to_bits(),
        (-0.0f64).to_bits(),
        "the minimum came back as {} and `-0.0` is the smaller of the two under the total order",
        region.min
    );
    assert_eq!(region.max.to_bits(), 0.0f64.to_bits());
    // The sum is unaffected either way: `-0.0` quantises to the integer zero.
    assert_eq!(region.sum_fixed, 0);
}

/// The schema says which columns carry a scale and which do not, and it says it
/// in the names — which is the fact a consumer reads to know what it is holding.
///
/// **The scale is on the accumulated *values* and on nothing else**: the sum and
/// the three first moments. Not on the selections, which are values that were
/// never scaled, and not on the coordinate sums, which are integers that never
/// needed to be.
#[test]
fn the_scale_is_on_the_accumulated_values_and_nowhere_else() {
    for bits in [0u32, 4, DEFAULT_FRACTION_BITS, MAX_FRACTION_BITS] {
        let point = FixedPoint::bits(bits).expect("in range");
        let schema = tabulation_schema(point);
        let suffix = format!("_q{bits}");
        let scaled: Vec<&str> = schema
            .columns()
            .iter()
            .map(|column| column.name())
            .filter(|name| name.ends_with(&suffix))
            .collect();
        let expected: Vec<String> = std::iter::once(format!("{SUM_COLUMN}{suffix}"))
            .chain(MOMENT_COLUMN.iter().map(|stem| format!("{stem}{suffix}")))
            .collect();
        assert_eq!(
            scaled,
            expected.iter().map(String::as_str).collect::<Vec<&str>>(),
            "at {bits} fraction bits the scale is on {scaled:?}"
        );
        assert!(schema.index_of(MIN_COLUMN).is_some());
        assert!(schema.index_of(MAX_COLUMN).is_some());
    }
}

// ---------------------------------------- 7. the sum, which did not move --

/// Every finite voxel of `label`, quantised at `point` and added as integers:
/// the sum column's definition, written out, and not the op's code.
fn quantised_total(volume: [usize; 3], point: FixedPoint, label: u64) -> i128 {
    let mut total = 0i128;
    for z in 0..volume[0] {
        for y in 0..volume[1] {
            for x in 0..volume[2] {
                let at = [z, y, x];
                if label_at(at) != label {
                    continue;
                }
                if let Some(value) = point.quantise(value_at(at)).expect("in range") {
                    total += value;
                }
            }
        }
    }
    total
}

/// **The sum is exactly what it was**, asserted two ways: against its own
/// definition for every region, and against a written-down integer for the one
/// that spans the volume — so a change to the quantiser that moved the reference
/// with it would still be caught here.
#[test]
fn the_fixed_point_sum_is_unchanged_and_is_still_the_quantised_total() {
    let point = fixed();
    let run = run_at(VOLUME, [3, 2, 2], point);
    let schema = tabulation_schema(point);
    assert_eq!(
        schema.index_of(&format!("{SUM_COLUMN}{}", point.suffix())),
        Some(3),
        "the sum is the fourth column and carries the scale, as it always did"
    );
    for row in &run.rows {
        assert_eq!(
            i128::from(row.sum_fixed),
            quantised_total(VOLUME, point, row.label),
            "label {}",
            row.label
        );
    }
    // Label 1 is `3z - 7` for z in 0..12 with the voxel at z = 9 non-finite:
    // 114 - 20 = 94, and 94 * 2^20 is the fixed-point word's value.
    let spanning = row_for(&run.rows, 1);
    assert_eq!(spanning.sum_fixed, 98_566_144);
    assert_eq!(spanning.sum, 94.0);
    assert_eq!(spanning.count, 12);
    assert_eq!(spanning.nonfinite, 1);
}

/// The scale a caller with no scale in mind can **derive** rather than pick, run
/// rather than described: a region holds at most every voxel, each finite value
/// is at most the array's peak magnitude, and quantising moves each at most half
/// a step further from zero, so no *sum* exceeds `voxels * (magnitude + 0.5)`.
///
/// **And the first moment multiplies that by the largest coordinate**, since
/// each term is a value times a coordinate — so the bound the range has to cover
/// is the sum's times `max(extent) - 1`, and the derived scale is coarser than
/// the sum alone would have asked for. That factor is the whole of what the new
/// column costs a caller who derives rather than picks, and it is stated here
/// because a derivation that covered only the sum would be refused on a moment.
fn derived_scale(volume: [usize; 3], of: fn([usize; 3]) -> f64) -> (FixedPoint, f64) {
    let mut magnitude = 0.0f64;
    for z in 0..volume[0] {
        for y in 0..volume[1] {
            for x in 0..volume[2] {
                let value = of([z, y, x]);
                if value.is_finite() {
                    magnitude = magnitude.max(value.abs());
                }
            }
        }
    }
    let voxels = (volume[0] * volume[1] * volume[2]) as f64;
    let reach = volume
        .iter()
        .map(|extent| extent.saturating_sub(1))
        .max()
        .expect("three axes")
        .max(1) as f64;
    let bound = voxels * (magnitude + 0.5) * reach;
    let mut best = FixedPoint::bits(0).expect("zero bits");
    for bits in 0..=MAX_FRACTION_BITS {
        let candidate = FixedPoint::bits(bits).expect("in range");
        if candidate.limit() > bound {
            best = candidate;
        }
    }
    (best, bound)
}

/// **The derived-scale path still works**, and the selection is the same bits at
/// the derived scale as at the default — which is the difference between the two
/// columns, made visible in one assertion.
#[test]
fn the_sum_runs_at_a_derived_scale_and_the_selection_does_not_care_which() {
    let (point, bound) = derived_scale(VOLUME, value_at);
    assert!(
        point.limit() > bound,
        "the derived scale's range {:e} does not cover the bound {bound:e}",
        point.limit()
    );
    assert!(
        point.fraction_bits() == MAX_FRACTION_BITS
            || FixedPoint::bits(point.fraction_bits() + 1)
                .expect("one more bit")
                .limit()
                <= bound,
        "one more fraction bit would still have covered the bound, so this is not the finest \
         scale the range admits"
    );
    assert_ne!(
        point.fraction_bits(),
        DEFAULT_FRACTION_BITS,
        "the derived scale equals the default, which would make this indistinguishable from \
         taking it"
    );

    let derived = run_at(VOLUME, [3, 2, 2], point);
    let oracle = reference_with(VOLUME, point, label_at, value_at);
    for row in &derived.rows {
        assert_eq!(
            i128::from(row.sum_fixed),
            quantised_total(VOLUME, point, row.label),
            "label {} at the derived {} fraction bits",
            row.label,
            point.fraction_bits()
        );
        // The moments fit at the derived scale too, which is the half of the
        // derivation the trailing factor exists for.
        for axis in 0..3 {
            assert_eq!(
                i128::from(row.moment_fixed[axis]),
                oracle[&row.label].moment[axis],
                "label {} on axis {axis} at the derived {} fraction bits",
                row.label,
                point.fraction_bits()
            );
        }
    }

    // The sum moved with the scale — it is a different integer in a differently
    // named column — and the selection did not move at all.
    let default = run_at(VOLUME, [3, 2, 2], fixed());
    assert_ne!(
        derived.rows[0].sum_fixed, default.rows[0].sum_fixed,
        "two scales gave the same fixed-point word, so the scale is not doing anything here"
    );
    for (one, other) in derived.rows.iter().zip(&default.rows) {
        assert_eq!(one.label, other.label);
        assert_eq!(
            one.min.to_bits(),
            other.min.to_bits(),
            "label {} moved its minimum with the scale",
            one.label
        );
        assert_eq!(
            one.max.to_bits(),
            other.max.to_bits(),
            "label {} moved its maximum with the scale",
            one.label
        );
    }
}

/// The sum's two refusals, both still by name: a **value** outside the column's
/// range, refused at the voxel it was read at, and a **total** outside it,
/// refused where the answer becomes a column.
#[test]
fn the_sums_out_of_range_refusals_are_both_still_named() {
    // 2048 needs 2^(63 - n) > 2048, so 52 fraction bits refuses the value.
    let by_value = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(52).expect("52 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect_err("2048 does not fit 2^11 of range")
    .to_string();
    assert!(by_value.contains("52 fraction bit"), "{by_value}");
    assert!(by_value.contains("cannot hold"), "{by_value}");

    // At 51 the value fits and the *total* does not: two voxels of 2048 scale to
    // 2^63 exactly, which is one past what a signed column holds.
    let by_total = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(51).expect("51 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect_err("4096 at 51 fraction bits is 2^63")
    .to_string();
    assert!(by_total.contains("fixed-point total"), "{by_total}");
    assert!(by_total.contains("above"), "{by_total}");
    assert!(by_total.contains("fewer fraction bits"), "{by_total}");

    // And at 50 the **moment** is the one that does not fit, which is the range
    // edge the new column brings with it: the two voxels of 2048 sit at z = 2 and
    // z = 3, so the sum is `2 * 2^61 = 2^62` and fits, while the moment is
    // `5 * 2^61` and does not. A run refused here on the sum's message would be
    // sending a caller to the wrong number.
    let by_moment = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(50).expect("50 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect_err("5 * 2^61 is above what a signed 64-bit column holds")
    .to_string();
    assert!(by_moment.contains("first moment on axis 0"), "{by_moment}");
    assert!(by_moment.contains("binds first"), "{by_moment}");
    // and one bit coarser everything fits, so 50 really is the edge and not a
    // scale that was failing for some other reason
    try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(49).expect("49 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect("5 * 2^60 fits");
}

/// The one thing `Arc` is doing here: an op is shared across workers, so it is
/// `Send + Sync`, and a test that holds one behind an `Arc` is the cheapest
/// statement of that which the compiler checks.
#[test]
fn the_ops_are_shareable_across_workers() {
    let point = fixed();
    let tabulate: Arc<dyn FragmentOp> = Arc::new(
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
            .expect("two images"),
    );
    let merge: Arc<dyn FragmentOp> = Arc::new(MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        [2, 1, 1],
        point,
        "rows",
        Lifecycle::Persistent,
    ));
    assert_eq!(tabulate.seam_fold(), Some(SeamFold::PerBlock));
    assert_eq!(merge.seam_fold(), Some(SeamFold::Unordered));
    assert!(!merge.gathers(), "the whole-lattice merge streams");
    assert!(!tabulate.reads_pixels() && !merge.reads_pixels());
}

// ------------------------------------------------ 7. the second moments, --
// ------------------------- which are the shape and are about its own centre --

/// **The shape fixture**, and every region in it is there to fail a different
/// implementation.
///
/// The hazard a second-moment test has to beat is that most objects cannot tell
/// a working implementation from several broken ones. A ball has no orientation
/// to get wrong. An **axis-aligned** object has three zero off-diagonals, so it
/// cannot distinguish a moment matrix whose off-diagonal components were
/// permuted, transposed, or dropped — every one of those is the same matrix when
/// the off-diagonals are zero. So the fixture is built the other way:
///
/// * `1` is a **one-voxel-wide line along `(1, 1, 0)`**, the whole depth of the
///   volume. Its off-diagonal `central_01` equals its diagonals and its other
///   two off-diagonals are exactly zero, so permuting the three moves the
///   answer — which is asserted, not assumed;
/// * `2` is a **slab three voxels thick about the same diagonal and five wide on
///   axis 2**, so all three of its variances are distinct and *none* of its
///   principal axes is an axis of the lattice. It is the region that pins the
///   whole decomposition rather than one direction of it;
/// * `3` is a **five-voxel cube**. Odd-sided on purpose: its centroid is a voxel,
///   so it is exactly isotropic and the degenerate answer is exact rather than
///   nearly so. It has no orientation and must say so;
/// * `4` is a **line along axis 0 alone** — the object that *cannot* discriminate
///   — and it is here so that the fixture's discrimination can be asserted as a
///   difference against it rather than claimed;
/// * `5` is a **flat five-by-five plate**, the other degeneracy: two equal
///   variances at the *top* rather than at the bottom, so its minor axis is
///   determined and its major pair is not.
///
/// Every one of them spans enough of axis 0 or axis 1 that **every cut below
/// puts every region across a seam**, which the test asserts before it asserts
/// anything else.
const SHAPE_VOLUME: [usize; 3] = [12, 12, 12];

/// The block sizes the shape fixture is cut by. The first is the whole volume,
/// which is the reference; the rest are chosen so that no region survives whole
/// in any of them — see the assertion in
/// [`the_second_moments_are_byte_identical_across_block_sizes`].
const SHAPE_BLOCKS: [[usize; 3]; 5] = [[12, 12, 12], [6, 6, 6], [5, 5, 5], [4, 12, 3], [7, 3, 12]];

fn shape_labels(at: [usize; 3]) -> u64 {
    let [z, y, x] = at;
    if x == 11 && z == y {
        return 1;
    }
    if (6..11).contains(&x) && z.abs_diff(y) <= 1 {
        return 2;
    }
    if x < 5 && (4..9).contains(&z) && y < 5 {
        return 3;
    }
    if x == 5 && y == 11 {
        return 4;
    }
    if x == 5 && (2..7).contains(&z) && (6..11).contains(&y) {
        return 5;
    }
    0
}

/// A finite, signed, axis-asymmetric value array. The shape does not read it —
/// that is the claim [`the_shape_is_a_reading_of_the_label_volume_alone`] makes
/// — so what it holds matters only in that it must not be constant, or "the
/// shape ignores the values" would be indistinguishable from "the values are all
/// the same".
fn shape_value_at(at: [usize; 3]) -> f64 {
    at[0] as f64 * 0.5 - at[1] as f64 * 0.25 + at[2] as f64 - 3.0
}

/// Every voxel non-finite. See [`the_shape_is_a_reading_of_the_label_volume_alone`].
fn nothing_finite_at(_at: [usize; 3]) -> f64 {
    f64::NAN
}

fn shape_for(shapes: &[RegionShape], label: u64) -> RegionShape {
    *shapes
        .iter()
        .find(|shape| shape.label == label)
        .unwrap_or_else(|| panic!("no shape row for label {label}"))
}

/// The angle between two unit directions, taken as **lines**: the sign of an
/// eigenvector is a convention, so two directions that differ by a sign are the
/// same axis and this reports zero for them.
fn angle_between(one: [f64; 3], other: [f64; 3]) -> f64 {
    let dot = one[0] * other[0] + one[1] * other[1] + one[2] * other[2];
    let clamped = if dot.abs().total_cmp(&1.0).is_ge() {
        1.0
    } else {
        dot.abs()
    };
    clamped.acos()
}

/// **The acceptance criterion for the six new columns.** Byte-identical across
/// five block sizes against a whole-volume reference, on a fixture where every
/// region straddles every cut.
#[test]
fn the_second_moments_are_byte_identical_across_block_sizes() {
    let point = fixed();
    let mut answers: Vec<(String, Vec<u64>)> = Vec::new();
    let oracle = reference_with(SHAPE_VOLUME, point, shape_labels, shape_value_at);
    assert_eq!(oracle.len(), 5, "the fixture is five regions");

    for block in SHAPE_BLOCKS {
        let run = try_run(SHAPE_VOLUME, block, point, shape_labels, shape_value_at).expect("a run");

        // Which labels were cut, and the assertion that makes the byte-identity
        // below mean something: five runs that merged nothing would agree
        // perfectly and prove nothing.
        let mut contributors: BTreeMap<u64, BTreeSet<[usize; 3]>> = BTreeMap::new();
        for (index, bytes) in &run.partials {
            for tally in decode_partial(bytes).expect("a partial") {
                contributors.entry(tally.label).or_default().insert(*index);
            }
        }
        if block == SHAPE_BLOCKS[0] {
            assert!(
                contributors.values().all(|who| who.len() == 1),
                "the one-block lattice cannot straddle anything"
            );
        } else {
            for label in oracle.keys() {
                assert!(
                    contributors[label].len() > 1,
                    "block {block:?} kept label {label} inside one block, so its second moments \
                     never met the merge"
                );
            }
        }

        assert_eq!(run.shapes.len(), oracle.len(), "block {block:?}");
        for shape in &run.shapes {
            let tally = &oracle[&shape.label];
            assert_eq!(
                Some(*shape),
                tally.shape().expect("the oracle's shape"),
                "block {block:?}: label {}",
                shape.label
            );
            // and the derived geometry, on the bits, because a decomposition
            // that moved with the cut would be a decomposition of a matrix that
            // moved with the cut
            let axes = shape.principal_axes().expect("a non-empty region");
            let expected = tally
                .shape()
                .expect("the oracle's shape")
                .expect("a non-empty region")
                .principal_axes()
                .expect("a non-empty region");
            assert_eq!(
                axes.variance.map(f64::to_bits),
                expected.variance.map(f64::to_bits),
                "block {block:?}: label {}",
                shape.label
            );
            assert_eq!(
                axes.axis
                    .map(|found| found.map(|direction| direction.map(f64::to_bits))),
                expected
                    .axis
                    .map(|found| found.map(|direction| direction.map(f64::to_bits))),
                "block {block:?}: label {}",
                shape.label
            );
        }
        answers.push((format!("{block:?}"), run.words));
    }

    let (first_name, first) = &answers[0];
    for (name, words) in &answers[1..] {
        assert_eq!(
            words, first,
            "the second moments differ between block {first_name} and block {name}; a region's \
             shape must not be a function of the plan"
        );
    }
}

/// **The liveness partner.** A cube has no orientation to get wrong and an
/// axis-aligned rod cannot distinguish a moment matrix whose off-diagonals were
/// permuted; this asserts that the off-axis regions *can*, which is what makes
/// the invariance test above a test.
#[test]
fn an_off_axis_region_reports_the_direction_it_is_elongated_along() {
    let point = fixed();
    let run = try_run(SHAPE_VOLUME, [5, 5, 5], point, shape_labels, shape_value_at).expect("a run");

    let diagonal = 1.0 / 2.0f64.sqrt();

    // Label 1 — the one-voxel line along (1, 1, 0), hand-computable end to end.
    let line = shape_for(&run.shapes, 1);
    assert_eq!(line.count, 12);
    assert_eq!(line.at, [6, 6, 11]);
    // sum (z - 6)^2 for z in 0..12 is 146, and z == y throughout, so the (0,1)
    // component equals it and the three touching axis 2 are exactly zero.
    assert_eq!(line.central, [146, 146, 0, 146, 0, 0]);
    let axes = line.principal_axes().expect("a region with voxels");
    let orientation = line.orientation().expect("an elongated region");
    assert!(
        // `1e-6` and not `1e-9`: two unit vectors that differ in their last
        // bit have a dot product `1 - 1e-16` and an angle of `1.4e-8`, because
        // `acos` near one has a square root in it. The assertion is about the
        // direction, not about the last bit of a cosine.
        angle_between(orientation, [diagonal, diagonal, 0.0]) < 1e-6,
        "the line along (1,1,0) reported {orientation:?}"
    );
    // Its other two variances are *exactly* zero — it is one voxel wide — and
    // the closed-form solve reports them as about `1e-7`, which is
    // [`AXIS_SEPARATION`]'s whole subject made visible: near a repeated root the
    // trigonometric form resolves the pair only to `sqrt` of the machine
    // epsilon. Relative to the major variance that is `6e-9`, an order below the
    // threshold, so the pair is correctly called degenerate.
    let noise = 1e-6 * axes.variance[0];
    assert!(axes.variance[1].abs() < noise, "{:?}", axes.variance);
    assert!(axes.variance[2].abs() < noise, "{:?}", axes.variance);
    assert!(
        axes.variance[1].abs() > 0.0,
        "the solve returned an exact zero, so this assertion is no longer about the solver's \
         resolution and the reasoning behind `AXIS_SEPARATION` has lost its evidence"
    );
    assert_eq!(axes.axis[1], None);
    assert_eq!(axes.axis[2], None);
    assert!(line.eccentricity().expect("a region with extent") > 1.0 - 1e-6);

    // **The discrimination.** Permute the three off-diagonal components — the
    // mistake an axis-aligned fixture cannot see — and the orientation moves.
    let permuted = RegionShape {
        central: [
            line.central[0],
            line.central[2],
            line.central[1],
            line.central[3],
            line.central[4],
            line.central[5],
        ],
        ..line
    };
    let moved = permuted
        .orientation()
        .expect("still elongated, just differently");
    assert!(
        angle_between(orientation, moved) > 0.5,
        "permuting the off-diagonal components moved the orientation by only {} radians, so this \
         fixture does not discriminate one",
        angle_between(orientation, moved)
    );

    // **And the same permutation on the axis-aligned rod does nothing at all**,
    // which is the assertion that says why the fixture had to be built off-axis.
    // An absence, inverted rather than left out.
    let rod = shape_for(&run.shapes, 4);
    assert_eq!(rod.central, [146, 0, 0, 0, 0, 0]);
    let rod_permuted = RegionShape {
        central: [
            rod.central[0],
            rod.central[2],
            rod.central[1],
            rod.central[3],
            rod.central[4],
            rod.central[5],
        ],
        ..rod
    };
    assert_eq!(
        rod.orientation().map(|axis| axis.map(f64::to_bits)),
        rod_permuted
            .orientation()
            .map(|axis| axis.map(f64::to_bits)),
        "the axis-aligned rod distinguished a permuted moment matrix, which it cannot do — its \
         off-diagonals are all zero"
    );
    assert!(angle_between(rod.orientation().expect("elongated"), [1.0, 0.0, 0.0]) < 1e-6);

    // Label 2 — the thick diagonal slab, whose three variances are distinct and
    // none of whose axes is an axis of the lattice. This is the region that pins
    // the whole decomposition rather than one direction of it.
    let slab = shape_for(&run.shapes, 2);
    let slab_axes = slab.principal_axes().expect("a region with voxels");
    assert!(
        slab_axes.variance[0] > slab_axes.variance[1]
            && slab_axes.variance[1] > slab_axes.variance[2],
        "the slab's variances are {:?} and were meant to be three distinct numbers",
        slab_axes.variance
    );
    let expected = [
        [diagonal, diagonal, 0.0],
        [0.0, 0.0, 1.0],
        [diagonal, -diagonal, 0.0],
    ];
    for (index, want) in expected.into_iter().enumerate() {
        let found = slab_axes.axis[index].expect("three separated eigenvalues, three directions");
        assert!(
            angle_between(found, want) < 1e-6,
            "the slab's axis {index} is {found:?} and was meant to be {want:?}"
        );
    }
    // orthonormal, which a cross-product recipe can lose if it takes the wrong
    // pair of rows
    for one in 0..3 {
        let a = slab_axes.axis[one].expect("determined");
        assert!((a[0] * a[0] + a[1] * a[1] + a[2] * a[2] - 1.0).abs() < 1e-12);
        for other in (one + 1)..3 {
            let b = slab_axes.axis[other].expect("determined");
            assert!((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs() < 1e-12);
        }
    }
    // the equivalent-ellipsoid lengths, which is what `regionprops` calls
    // `axis_major_length` and `axis_minor_length`
    assert!(slab_axes.length[0] > slab_axes.length[2]);
    for (length, variance) in slab_axes.length.into_iter().zip(slab_axes.variance) {
        assert!((length - 2.0 * (5.0 * variance).sqrt()).abs() < 1e-12);
    }
    // **The eccentricity, against its own definition.** `sqrt(1 - (minor /
    // major)^2)`, recomputed here from the two lengths the same struct reports.
    //
    // This assertion used to read `(eccentricity - (1 - r^2)).sqrt().abs() <
    // 1.0`, with the square root outside the difference rather than on
    // `1 - r^2`, and a bound of one. Since `eccentricity` *is* `sqrt(1 - r^2)`,
    // that expression was `sqrt(e - e^2)`, which is below `0.5` for every `e` in
    // `[0, 1]` — so it held for any eccentricity the code could produce, the
    // `e^2`-for-`e` confusion very much included. It could not fail and was
    // measuring nothing.
    //
    // The liveness is measured rather than argued: `1 - r^2` — the value
    // without the square root, which is the natural thing to write by mistake —
    // is asserted to be far outside the bound, so the `1e-12` is a tolerance on
    // arithmetic and not a tolerance on being wrong.
    let eccentricity = slab.eccentricity().expect("a region with extent");
    let ratio = slab_axes.length[2] / slab_axes.length[0];
    let from_the_lengths = (1.0 - ratio * ratio).sqrt();
    assert!(
        (eccentricity - from_the_lengths).abs() < 1e-12,
        "the eccentricity is {eccentricity} and sqrt(1 - (minor/major)^2) over the reported \
         lengths is {from_the_lengths}"
    );
    let unrooted = 1.0 - ratio * ratio;
    assert!(
        (eccentricity - unrooted).abs() > 1e-3,
        "the squared form is only {} from the eccentricity here, so this fixture cannot tell \
         the two apart and the assertion above is not discriminating",
        (eccentricity - unrooted).abs()
    );
    assert!(eccentricity > 0.9 && eccentricity < 1.0);
}

/// **A region with no orientation says so.** A `None`, never a `NaN` and never
/// an arbitrary pick — and the eigen*values* are reported in every case, because
/// those are always determined.
#[test]
fn a_region_with_no_orientation_reports_none_rather_than_a_number() {
    let point = fixed();
    let run = try_run(SHAPE_VOLUME, [6, 6, 6], point, shape_labels, shape_value_at).expect("a run");

    // The cube: three equal variances, three `None`s. Odd-sided, so its centroid
    // is exactly a voxel and its covariance is exactly `diag(2, 2, 2)` — the
    // variance of five consecutive integers is `(25 - 1) / 12`.
    let cube = shape_for(&run.shapes, 3);
    assert_eq!(cube.count, 125);
    let covariance = cube.covariance().expect("a region with voxels");
    for one in 0..3 {
        for other in 0..3 {
            let want = if one == other { 2.0 } else { 0.0 };
            assert!(
                (covariance[one][other] - want).abs() < 1e-12,
                "the cube's covariance is {covariance:?}"
            );
        }
    }
    let axes = cube.principal_axes().expect("a region with voxels");
    assert_eq!(
        axes.axis,
        [None, None, None],
        "a cube has no principal axis"
    );
    assert_eq!(cube.orientation(), None);
    for variance in axes.variance {
        assert!((variance - 2.0).abs() < 1e-12);
        assert!(variance.is_finite(), "a `NaN` where a `None` was promised");
    }
    for length in axes.length {
        assert!(length.is_finite());
    }
    // a ball is not eccentric, and that is a number rather than an absence:
    // the *values* are determined even where the directions are not
    assert_eq!(cube.eccentricity(), Some(0.0));

    // The plate: two equal variances at the **top**, so the minor axis is
    // determined and the major pair is not. The other degeneracy, and it is not
    // the same one — an implementation that only ever looked at the largest gap
    // would answer this one wrongly.
    let plate = shape_for(&run.shapes, 5);
    assert_eq!(plate.count, 25);
    let flat = plate.principal_axes().expect("a region with voxels");
    assert_eq!(flat.axis[0], None, "an oblate region has no major axis");
    assert_eq!(flat.axis[1], None);
    let minor = flat.axis[2].expect("the odd one out is determined");
    assert!(angle_between(minor, [0.0, 0.0, 1.0]) < 1e-6, "{minor:?}");
    assert!((flat.variance[0] - 2.0).abs() < 1e-12);
    assert!((flat.variance[1] - 2.0).abs() < 1e-12);
    assert_eq!(flat.variance[2], 0.0);
    assert_eq!(plate.orientation(), None);

    // and the prolate case for completeness: a determined major axis and an
    // undetermined minor pair — see
    // `an_off_axis_region_reports_the_direction_it_is_elongated_along`, which is
    // where the direction itself is checked.
    let line = shape_for(&run.shapes, 1);
    let prolate = line.principal_axes().expect("a region with voxels");
    assert!(prolate.axis[0].is_some());
    assert_eq!(prolate.axis[1], None);
    assert_eq!(prolate.axis[2], None);
}

/// **The shape is a reading of the label volume and of nothing else**, so it is
/// taken over *every* voxel of a region — including the voxels whose value was
/// not finite, which `sum`, `min`, `max` and the first moments must exclude.
///
/// The same fixture with every value a `NaN`. A region with no finite value at
/// all has no sum, no extremes and no weighted centre, and still has a shape,
/// bit for bit the one it had when the values were ordinary numbers. This is the
/// half of the weighted-versus-unweighted argument that is a property rather
/// than a preference.
#[test]
fn the_shape_is_a_reading_of_the_label_volume_alone() {
    let point = fixed();
    let ordinary =
        try_run(SHAPE_VOLUME, [5, 5, 5], point, shape_labels, shape_value_at).expect("a run");
    let broken = try_run(
        SHAPE_VOLUME,
        [5, 5, 5],
        point,
        shape_labels,
        nothing_finite_at,
    )
    .expect("a run over an array of `NaN`s");

    assert_eq!(ordinary.shapes, broken.shapes);
    for row in &broken.rows {
        assert!(
            row.all_nonfinite(),
            "label {} kept a finite value, so this run does not exercise the claim",
            row.label
        );
        assert_eq!(row.weighted_centroid, None);
        assert_eq!(row.sum_fixed, 0);
    }
    // and the two runs really did report different *values*, or the comparison
    // above would be of two identical runs
    assert_ne!(
        ordinary.rows.iter().map(|row| row.sum_fixed).sum::<i64>(),
        0
    );
}

// ------------------------------ 8. the fold, and the two directions it has --

/// The corner of an imaginary volume the fold fixture's partials come from:
/// `2^26`, where one voxel's `z^2` is `2^52`.
///
/// **Why the number is this large, which is the finding this fixture carries.**
/// The second moment is a sum of *non-negative integers*, each at most `L^2`. An
/// `f64` fold of such a sum is exact until it passes `2^53`, and there is no
/// cancellation available to bring it there early, so a fixture on which an
/// `f64` fold of a second moment differs from an integer one needs
/// `sum x^2 >= 2^53`. That is a whole-volume region about **2 000 voxels on a
/// side** — eight thousand million voxels, sixty-four gigabytes of `f64` before
/// the second array — and the cheapest arrangement of it is not much better:
/// `sum z^2` over `T` voxels reaching a coordinate `L` is at most `T * L`, so
/// `T >= 2^26.5`, or about a hundred million voxels. **No fixture that fits in
/// memory can separate the two fold directions on this column through a run.**
///
/// So the separation is done where the seam actually is: on the partials, at
/// [`MergeTabulationOp::fold`], which is the same function the merge phase calls
/// and the same [`Tally::merge`] underneath it. The partials are real —
/// [`TabulateValuesOp::tally_block`] takes them from real blocks placed at a real
/// global offset, which is what the fetch region is for — and the comparison is
/// on [`Tally::row_words`], which is byte for byte what the executor's reversal
/// check compares. What is not exercised here is the executor's plumbing, and
/// that is exercised on every other run in this file.
///
/// The production case is not hypothetical: above about two thousand voxels on a
/// side an `f64` fold of this column *would* drift, and out-of-core volumes are
/// what this crate is for.
const WIDE_CORNER: usize = 1 << 26;

/// One partial, taken from a real block of two voxels at `start`, carrying label
/// `1` where `carried` says so.
fn planted_partial(op: &TabulateValuesOp, start: [usize; 3], carried: [bool; 2]) -> Vec<u8> {
    let mut label_block = Array3::<f64>::zeros((1, 2, 1));
    let value_block = Array3::<f64>::zeros((1, 2, 1));
    for (index, held) in carried.into_iter().enumerate() {
        label_block[[0, index, 0]] = if held { 1.0 } else { 0.0 };
    }
    let read = Region::new(&start, &[1, 2, 1]);
    let tallies = op
        .tally_block(
            &blockflow::env::BlockBuf::Array(label_block.into()),
            &blockflow::env::BlockBuf::Array(value_block.into()),
            &read,
            &read,
        )
        .expect("a tally");
    blockflow::ops::tabulate::encode_partial(&tallies)
}

/// The three partials the two fixtures below share: `sum z^2` of `1`, `9` and
/// `2^53`, from blocks at `z = 1`, `z = 3` and `z = 2^26`.
///
/// `1` and `9` rather than `1` and `1` because only one voxel has `z^2 == 1`;
/// the pair has to add to an even multiple of `2^53`'s two-unit spacing while
/// each is below half of it, and `10` is the smallest such sum two distinct
/// squares reach.
fn wide_partials(op: &TabulateValuesOp) -> Vec<Vec<u8>> {
    vec![
        planted_partial(op, [1, 0, 0], [true, false]),
        planted_partial(op, [3, 0, 0], [true, false]),
        planted_partial(op, [WIDE_CORNER, 0, 0], [true, true]),
    ]
}

/// **The fold gives the same answer in both directions where an `f64` fold does
/// not**, on the column whose `f64` hazard no runnable fixture can reach. See
/// [`WIDE_CORNER`].
#[test]
fn the_second_moment_folds_to_one_answer_where_an_f64_fold_gives_two() {
    let point = FixedPoint::bits(0).expect("zero fraction bits");
    let op = TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
        .expect("two images");
    let partials = wide_partials(&op);

    // First: the fixture separates. An `f64` fold of these three terms gives one
    // answer forwards and another backwards, and only one of them is right.
    let terms: Vec<f64> = partials
        .iter()
        .map(|bytes| decode_partial(bytes).expect("a partial")[0].second[0] as f64)
        .collect();
    let forwards = terms.iter().fold(0.0f64, |total, term| total + term);
    let backwards = terms.iter().rev().fold(0.0f64, |total, term| total + term);
    let exact = (1i128 << 53) + 10;
    assert_ne!(
        forwards.to_bits(),
        backwards.to_bits(),
        "an `f64` fold of {terms:?} did not depend on the order, so this fixture has nothing to \
         catch"
    );
    assert_eq!(forwards, exact as f64, "forwards happens to be right");
    assert_eq!(
        backwards,
        (exact - 2) as f64,
        "backwards loses two, which is the whole point of the fixture"
    );

    // Second: the integer fold does not, and gets it right from either end.
    let merge = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        [3, 1, 1],
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let one_way = merge
        .fold(partials.iter().map(|bytes| bytes.as_slice()))
        .expect("a fold");
    let other_way = merge
        .fold(partials.iter().rev().map(|bytes| bytes.as_slice()))
        .expect("a fold");
    assert_eq!(one_way, other_way, "the integer fold moved with the order");
    assert_eq!(one_way.len(), 1);
    assert_eq!(one_way[0].count, 4);
    assert_eq!(one_way[0].second[0], exact);

    // And on the **bytes**, which is what `SeamFold::Unordered` is checked on:
    // the executor applies each block a second time with its neighbourhood
    // reversed and requires the same output blob.
    assert_eq!(
        one_way[0].row_words(point).expect("a row"),
        other_way[0].row_words(point).expect("a row"),
        "the two fold directions wrote different bytes"
    );

    // The row itself: the centring happens once, at the end, and turns a moment
    // of `2^53` about the volume origin into one of `4.50e15` about the
    // region's own centre — which is `sum (z - c)^2` for the four voxels this
    // region has, and is a number a caller can check by hand.
    let shape = one_way[0]
        .shape()
        .expect("a shape")
        .expect("a region with voxels");
    assert_eq!(shape.count, 4);
    assert_eq!(shape.at[0], (1 << 25) + 1);
    assert_eq!(shape.central[0], 4_503_599_358_935_046);
    assert_eq!(
        shape.second_moments_about_origin()[0],
        exact,
        "the raw form does not come back out of the centred one"
    );
}

/// **The negative control the survey's row is about**: the same region measured
/// about the *volume* origin rather than about its own.
///
/// Two regions, and the two halves of "either overflow by name or move the
/// answer" are one each.
#[test]
fn moments_about_the_volume_origin_overflow_by_name_where_the_regions_own_do_not() {
    let point = FixedPoint::bits(0).expect("zero fraction bits");
    let op = TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
        .expect("two images");
    // Four voxels: two at `z = 2^31` and two four voxels further on. A region
    // sixteen thousandths of the way across an axis a `usize` can address, and
    // five voxels wide.
    let edge = 1usize << 31;
    let partials = [
        planted_partial(&op, [edge, 0, 0], [true, true]),
        planted_partial(&op, [edge + 4, 0, 0], [true, true]),
    ];
    let merge = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        [2, 1, 1],
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let folded = merge
        .fold(partials.iter().map(|bytes| bytes.as_slice()))
        .expect("a fold");
    let shape = folded[0]
        .shape()
        .expect("the region's own origin holds it")
        .expect("a region with voxels");

    // **Every count is reproduced**: the two framings differ in where they are
    // measured from and in nothing else.
    assert_eq!(shape.count, 4);
    assert_eq!(shape.position, [4 * edge as u64 + 8, 2, 0]);
    // Axis 1 rounds to 1: the four voxels sit two at `y = 0` and two at `y = 1`,
    // so the exact mean is a half and the row's position rounds it half up.
    assert_eq!(shape.at, [edge + 2, 1, 0]);

    // About the region's own centre the answer is `16` and fits in a column with
    // sixty-two bits to spare.
    assert_eq!(shape.central[0], 16);

    // About the volume origin it is `1.8e19`, which is above what a signed
    // 64-bit column holds — and the refusal says so by name rather than
    // wrapping to a plausible number.
    let raw = shape.second_moments_about_origin()[0];
    assert!(
        raw >= (1i128 << 63),
        "the raw moment is {raw} and was meant to be past 2^63"
    );
    let failed = signed_column(raw)
        .expect_err("2^63 does not fit a signed 64-bit column")
        .to_string();
    assert!(failed.contains("above"), "{failed}");
    assert!(failed.contains("signed 64-bit"), "{failed}");
    assert!(failed.contains(&raw.to_string()), "{failed}");
    // and the centred one at the same place does not refuse, which is what says
    // the refusal is about the origin and not about the region
    assert!(signed_column(shape.central[0] as i128).is_ok());
}

/// The other half of the same control: a region whose moments about the volume
/// origin **fit**, and where measuring from there **moves the answer** instead of
/// refusing.
///
/// This is the failure mode that matters more, because it is silent.
#[test]
fn moments_about_the_volume_origin_move_the_orientation() {
    let point = fixed();
    let run = try_run(SHAPE_VOLUME, [5, 5, 5], point, shape_labels, shape_value_at).expect("a run");
    let line = shape_for(&run.shapes, 1);

    let raw = line.second_moments_about_origin();
    // It fits — a twelve-voxel region in a twelve-voxel volume is nowhere near
    // the range — so nothing refuses and nothing warns.
    let mut about_origin = line;
    for (slot, value) in about_origin.central.iter_mut().zip(raw) {
        *slot = i64::try_from(value).expect("a small volume's raw moments fit");
    }
    // and the counts are the same in both framings
    assert_eq!(about_origin.count, line.count);
    assert_eq!(about_origin.position, line.position);

    let honest = line.orientation().expect("an elongated region");
    // `PrincipalAxes::of` on the raw matrix directly, so that the row's own
    // half-voxel correction cannot be mistaken for the effect being measured.
    let voxels = line.count as f64;
    let mut matrix = [[0.0f64; 3]; 3];
    for (value, pair) in raw.into_iter().zip(PAIRS) {
        let entry = value as f64 / voxels;
        matrix[pair[0]][pair[1]] = entry;
        matrix[pair[1]][pair[0]] = entry;
    }
    let misplaced = PrincipalAxes::of(matrix).axis[0].expect("still has a largest eigenvalue");

    let moved = angle_between(honest, misplaced);
    assert!(
        moved > 0.5,
        "measuring from the volume origin moved the orientation by only {moved} radians. The \
         region is a line along (1,1,0) at x = 11, so about the origin its largest eigenvector \
         leans into axis 2 and the answer is the direction of the region rather than the \
         direction the region points"
    );
    // and specifically: about the origin the reported axis picks up axis 2,
    // which the region has no extent on at all
    assert!(
        misplaced[2].abs() > 0.5,
        "the misplaced axis is {misplaced:?} and was meant to lean into the axis the region does \
         not extend along"
    );
    assert!(honest[2].abs() < 1e-9);
}

/// **The second moments carry no scale, so they do not narrow the one the sum
/// runs at.** The scale sweep is the same sweep it was.
///
/// This is the claim the design turns on, run rather than argued: the first
/// moment is still the column that binds first on the *scale* axis, and the six
/// new columns are the same integers at every scale the run admits. A weighted
/// second moment would have moved this count — its bound is the sum's divided by
/// the product of *two* coordinates — and it is the reason there is not one.
#[test]
fn the_second_moments_do_not_move_the_scale_the_sum_runs_at() {
    let mut accepted = 0usize;
    let mut refused = 0usize;
    let mut first: Option<[i64; 6]> = None;
    for bits in 0..=MAX_FRACTION_BITS {
        let point = FixedPoint::bits(bits).expect("in range");
        let run = match try_run(
            AWKWARD_VOLUME,
            [1, 1, 1],
            point,
            two_region_labels,
            awkward_value_at,
        ) {
            Ok(run) => run,
            Err(_) => {
                refused += 1;
                continue;
            }
        };
        accepted += 1;
        let central = shape_for(&run.shapes, 1).central;
        match first {
            None => first = Some(central),
            Some(seen) => assert_eq!(
                central, seen,
                "the second moments moved with the fixed-point scale at {bits} fraction bits, \
                 which they cannot do — there is no scale in them"
            ),
        }
    }
    // The same 50 and 13 the selection sweep counts. The six columns landed and
    // the edge did not move, which is the whole of what "they carry no scale"
    // buys a caller.
    assert_eq!(
        (accepted, refused),
        (50, 13),
        "the admissible scale range moved when the second moments landed"
    );
    // and the refusal at the edge is still the first moment's, naming its axis
    let by_moment = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(50).expect("50 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect_err("the first moment does not fit at 50 fraction bits")
    .to_string();
    assert!(by_moment.contains("first moment on axis 0"), "{by_moment}");
    assert!(by_moment.contains("binds first"), "{by_moment}");
    assert!(
        !by_moment.contains("second moment"),
        "the second moment refused where the first one binds: {by_moment}"
    );
}

/// The six columns are named without a scale at every scale, which is what
/// [`region_shape`] reading them by index rests on.
#[test]
fn the_second_moment_columns_carry_no_accumulator_in_their_names() {
    for bits in [0u32, 8, DEFAULT_FRACTION_BITS, MAX_FRACTION_BITS] {
        let point = FixedPoint::bits(bits).expect("in range");
        let schema = tabulation_schema(point);
        for (index, name) in CENTRAL_COLUMN.into_iter().enumerate() {
            assert_eq!(
                schema.index_of(name),
                Some(12 + index),
                "at {bits} fraction bits the second moments are not where `region_shape` reads \
                 them"
            );
        }
        let scaled = schema
            .columns()
            .iter()
            .filter(|column| column.name().starts_with("central_"))
            .any(|column| column.name().contains("_q"));
        assert!(
            !scaled,
            "a second moment carries an accumulator's scale at {bits} fraction bits, which would \
             make it a quantised coordinate"
        );
    }
}
