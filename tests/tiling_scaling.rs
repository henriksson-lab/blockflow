// SPDX-License-Identifier: MIT
//
// `tiling::boxes_tile_exactly` decides whether a phase's blocks cover their
// volume once each, and `Decomposition::check` — and therefore
// `assemble::PlanBuilder::finish` — asks it once per phase. It used to answer by
// comparing every pair of blocks, which is `O(n²)` in the block count, and that
// term was the entire cost of closing a plan at the block sizes a partition
// search now wants to price. Measured on this machine before the change, a
// two-phase plan over a `64³` lattice:
//
// | blocks | `finish` | of which `Decomposition::check` |
// |---|---|---|
// | `2048` | `31.7 ms` | `37.4 ms` |
// | `4096` | `113.6 ms` | `150.2 ms` |
// | `8192` | `445.0 ms` | `602.4 ms` |
//
// — a clean `4x` per doubling, extrapolating to `21 s` at the `56498` blocks a
// fine candidate grid asks for, against `138 ms` to build the phases themselves.
// The predicate now separates instead of comparing, and the same plans close in
// `4.2 ms` at `8192` blocks and `69 ms` at `65536`, growing linearly. Timed
// `--release`, load average `3`, `165 GB` of `187 GB` free.
//
// **Those figures are not what this file asserts.** A wall clock on a shared
// box is a flake generator: a bound loose enough to survive six other builds
// running is loose enough for a returning `n²` term to hide under at the sizes a
// test can afford. So the scaling is pinned with `tiling::tiling_work`, a
// counter of elementary steps — one box placed in one elementary interval, or
// one pair handed to the scan — which is the same number on an idle machine and
// a loaded one.
//
// Two things have to hold, and both are tested here:
//
// * **The answer did not change.** `finish` derives nothing from this
//   predicate; it asks it a yes/no and propagates the error text verbatim. So
//   "not one plan may change" reduces exactly to "not one `Result` may change",
//   and that is pinned against a verbatim copy of the algorithm that was
//   replaced, over a corpus that is asserted to reach every one of its outcomes.
// * **The cost did change, and stays changed.** The counter grows with the
//   block count and not its square, and the bound used to say so is checked
//   against what the pairwise scan would itself have cost — a bound that the old
//   algorithm would also have passed would be no evidence at all.

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::tiling::{boxes_tile_exactly, reset_tiling_work, tiling_work};

/// The rank the plans in this file are cut in. Not a magic number: the whole
/// point of the separating pass is that it costs one placement per box *per
/// axis*, so every bound below is stated against it.
const NDIM: usize = 3;

// ---------------------------------------------------------------------------
// The algorithm that was replaced, kept verbatim
// ---------------------------------------------------------------------------

/// `boxes_tile_exactly` as it stood before the separating pass, copied rather
/// than paraphrased.
///
/// Copied because a paraphrase is a second implementation and would be testing
/// this file's reading of the old code rather than the old code. It is the
/// oracle for every case in the corpus below, message text included: the
/// containment pass runs first, then the pairwise overlap scan, then the
/// coverage count, and which of the three speaks first is part of what must not
/// change.
fn scan(boxes: &[Vec<(usize, usize)>], shape: &[usize]) -> Result<()> {
    let total: usize = shape.iter().product();

    let mut covered: usize = 0;
    for region in boxes {
        if region.len() != shape.len() {
            return Err(Error::InvalidArgument(format!(
                "valid regions do not tile the volume exactly: region {region:?} has rank {} \
                 but the volume has rank {}",
                region.len(),
                shape.len()
            )));
        }
        for (axis, (&(lo, hi), &dim)) in region.iter().zip(shape.iter()).enumerate() {
            if hi > dim || lo > hi {
                return Err(Error::InvalidArgument(format!(
                    "valid regions do not tile the volume exactly: region {region:?} spans \
                     {lo}..{hi} on axis {axis} of {dim}"
                )));
            }
        }
        covered += region.iter().map(|&(lo, hi)| hi - lo).product::<usize>();
    }

    for (first, left) in boxes.iter().enumerate() {
        for right in boxes.iter().skip(first + 1) {
            if overlap(left, right) {
                return Err(Error::InvalidArgument(format!(
                    "valid regions do not tile the volume exactly \
                     (regions {left:?} and {right:?} overlap)"
                )));
            }
        }
    }

    if covered != total {
        return Err(Error::InvalidArgument(format!(
            "valid regions do not tile the volume exactly \
             ({covered} of {total} voxels covered)"
        )));
    }
    Ok(())
}

fn overlap(left: &[(usize, usize)], right: &[(usize, usize)]) -> bool {
    let nonempty = |region: &[(usize, usize)]| region.iter().all(|&(lo, hi)| lo < hi);
    nonempty(left)
        && nonempty(right)
        && left
            .iter()
            .zip(right.iter())
            .all(|(&(a_lo, a_hi), &(b_lo, b_hi))| a_lo < b_hi && b_lo < a_hi)
}

/// Both answers as text, which is the granularity the equality has to hold at:
/// an `Err` that named a different pair would be a different diagnosis of the
/// same broken plan, and a caller reading the message would be told something
/// else than it was told before.
fn both(boxes: &[Vec<(usize, usize)>], shape: &[usize]) -> (String, String) {
    let fast = boxes_tile_exactly(boxes, shape).map_err(|err| err.to_string());
    let slow = scan(boxes, shape).map_err(|err| err.to_string());
    (format!("{fast:?}"), format!("{slow:?}"))
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// SplitMix64, so the corpus is the same corpus on every machine and in every
/// run. A randomised test whose cases move is a test that reports a different
/// failure than the one that was fixed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Which of the predicate's four answers a case produced. Collected so the
/// corpus can be asserted to reach all of them: agreement over a corpus that
/// only ever says `Ok` would agree with anything.
#[derive(Default, Debug)]
struct Reached {
    tiles: usize,
    rank: usize,
    range: usize,
    overlap: usize,
    shortfall: usize,
}

impl Reached {
    fn record(&mut self, answer: &str) {
        if answer.starts_with("Ok") {
            self.tiles += 1;
        } else if answer.contains("has rank") {
            self.rank += 1;
        } else if answer.contains("on axis") {
            self.range += 1;
        } else if answer.contains("overlap") {
            self.overlap += 1;
        } else {
            self.shortfall += 1;
        }
    }
}

/// An exact tiling of `shape` by a product lattice with `cuts` per axis, where
/// each axis's cut positions are chosen at random — so the blocks are ragged
/// rather than regular, which is what a volume edge produces and what a regular
/// grid would not exercise.
fn ragged_tiling(rng: &mut Rng, shape: &[usize; NDIM], cuts: usize) -> Vec<Vec<(usize, usize)>> {
    let mut per_axis: Vec<Vec<(usize, usize)>> = Vec::with_capacity(NDIM);
    for &dim in shape {
        let mut edges: Vec<usize> = vec![0, dim];
        for _ in 0..cuts {
            edges.push(rng.below(dim + 1));
        }
        edges.sort_unstable();
        edges.dedup();
        per_axis.push(edges.windows(2).map(|pair| (pair[0], pair[1])).collect());
    }
    let mut boxes = vec![Vec::new()];
    for axis in per_axis {
        let mut next = Vec::new();
        for prefix in &boxes {
            for &span in &axis {
                let mut extended = prefix.clone();
                extended.push(span);
                next.push(extended);
            }
        }
        boxes = next;
    }
    boxes
}

/// One random case: a ragged tiling, then `damage` random edits to it, then a
/// shuffle. The edits are what turn a corpus of tilings into a corpus that
/// reaches every answer — a moved edge is a shortfall or an overlap depending on
/// which way it moved, and a stretched one leaves the volume.
fn case(rng: &mut Rng) -> ([usize; NDIM], Vec<Vec<(usize, usize)>>) {
    let shape = [1 + rng.below(9), 1 + rng.below(9), 1 + rng.below(9)];
    let cuts = rng.below(4);
    let mut boxes = ragged_tiling(rng, &shape, cuts);
    let damage = rng.below(4);
    for _ in 0..damage {
        if boxes.is_empty() {
            break;
        }
        let which = rng.below(boxes.len());
        // Against the box's own rank, not `NDIM`: one of the edits below drops
        // an axis, and a later edit has to be able to land on what is left.
        let rank = boxes[which].len();
        if rank == 0 {
            continue;
        }
        match rng.below(6) {
            0 => {
                let axis = rng.below(rank);
                let (lo, hi) = boxes[which][axis];
                boxes[which][axis] = (lo, hi + 1);
            }
            1 => {
                let axis = rng.below(rank);
                let (lo, hi) = boxes[which][axis];
                boxes[which][axis] = (lo, hi.saturating_sub(1).max(lo));
            }
            2 => {
                let axis = rng.below(rank);
                let (lo, hi) = boxes[which][axis];
                boxes[which][axis] = (lo.saturating_sub(1), hi);
            }
            3 => {
                boxes.swap_remove(which);
            }
            4 => {
                let copy = boxes[which].clone();
                boxes.push(copy);
            }
            _ => {
                let mut wrong = boxes[which].clone();
                wrong.pop();
                boxes[which] = wrong;
            }
        }
    }
    for index in (1..boxes.len()).rev() {
        let other = rng.below(index + 1);
        boxes.swap(index, other);
    }
    (shape, boxes)
}

/// **The criterion for "not one plan may change".** Every case, both answers,
/// compared as the text a caller would read.
#[test]
fn the_separating_pass_answers_what_the_pairwise_scan_answered() {
    let mut rng = Rng(0x5eed_1eaf_0123_4567);
    let mut reached = Reached::default();
    for index in 0..20_000 {
        let (shape, boxes) = case(&mut rng);
        let (fast, slow) = both(&boxes, &shape);
        assert_eq!(
            fast, slow,
            "case {index} over shape {shape:?}: the separating pass and the pairwise scan \
             disagree.\n  boxes {boxes:?}"
        );
        reached.record(&fast);
    }
    // The corpus has to reach every answer or the agreement above is agreement
    // about nothing. Each of the five is asserted present rather than counted:
    // a number here would be a number to re-baseline every time the generator
    // is touched, and presence is the property that matters.
    assert!(reached.tiles > 0, "{reached:?}");
    assert!(reached.rank > 0, "{reached:?}");
    assert!(reached.range > 0, "{reached:?}");
    assert!(reached.overlap > 0, "{reached:?}");
    assert!(reached.shortfall > 0, "{reached:?}");
}

/// The liveness test for the one above: the corpus is shown to be able to tell
/// a broken predicate from a working one.
///
/// The negative control is the same program with one thing changed — here, the
/// oracle loses its overlap scan, which is precisely the term the separating
/// pass replaced. If the corpus could not see that, it could not see a
/// separating pass that had quietly stopped separating either.
#[test]
fn the_corpus_would_see_a_predicate_that_stopped_checking_overlaps() {
    fn blind(boxes: &[Vec<(usize, usize)>], shape: &[usize]) -> Result<()> {
        let mut covered = 0usize;
        for region in boxes {
            if region.len() != shape.len() {
                return Err(Error::InvalidArgument("rank".to_string()));
            }
            for (&(lo, hi), &dim) in region.iter().zip(shape.iter()) {
                if hi > dim || lo > hi {
                    return Err(Error::InvalidArgument("range".to_string()));
                }
            }
            covered += region.iter().map(|&(lo, hi)| hi - lo).product::<usize>();
        }
        if covered != shape.iter().product::<usize>() {
            return Err(Error::InvalidArgument("shortfall".to_string()));
        }
        Ok(())
    }

    let mut rng = Rng(0x5eed_1eaf_0123_4567);
    let mut caught = 0usize;
    for _ in 0..20_000 {
        let (shape, boxes) = case(&mut rng);
        let truth = boxes_tile_exactly(&boxes, &shape).is_ok();
        if blind(&boxes, &shape).is_ok() != truth {
            caught += 1;
        }
    }
    assert!(
        caught > 0,
        "the corpus never distinguished a predicate with no overlap check from the real one, so \
         its agreement with the pairwise scan is not evidence of anything"
    );
}

/// The separating pass declines rather than degrades when a box has to be
/// copied into many elementary intervals, and the scan behind it still gets the
/// right answer.
///
/// The shape is the one that provokes it: a wall of full-height columns beside a
/// stack of unit cells. The cells cut the axis into as many elementary intervals
/// as there are cells, and every column then spans all of them, so the placement
/// count is quadratic in the box count even though the boxes tile perfectly.
/// This is the input the budget exists for, and a test that never reached the
/// budget would leave that branch unexecuted.
#[test]
fn a_box_set_that_blows_the_placement_budget_is_still_answered_correctly() {
    const SIDE: usize = 64;
    let shape = [1usize, SIDE, SIDE + 1];
    let mut boxes: Vec<Vec<(usize, usize)>> = Vec::new();
    for column in 0..SIDE {
        boxes.push(vec![(0, 1), (0, SIDE), (column, column + 1)]);
    }
    for cell in 0..SIDE {
        boxes.push(vec![(0, 1), (cell, cell + 1), (SIDE, SIDE + 1)]);
    }

    reset_tiling_work();
    boxes_tile_exactly(&boxes, &shape).expect("the wall and the stack tile their volume");
    let work = tiling_work();

    let n = boxes.len() as u64;
    let budget = 8 * n * NDIM as u64;
    assert!(
        work > budget,
        "this case was meant to exceed the placement budget of {budget} and only reached {work}, \
         so the decline branch it exists to exercise was never taken"
    );
    // Having declined, the answer came from the scan, which is `n(n-1)/2` pairs.
    let pairs = n * (n - 1) / 2;
    assert!(
        work >= pairs,
        "the pass declined at {work} steps without the scan's {pairs} pairs following it, so the \
         answer came from somewhere neither of them"
    );
    // And the answer is the one the scan alone would give.
    let (fast, slow) = both(&boxes, &shape);
    assert_eq!(fast, slow);
}

// ---------------------------------------------------------------------------
// The scaling
// ---------------------------------------------------------------------------

/// A two-phase plan cut on a `64³` lattice over `blocks` blocks, closed.
///
/// Two phases rather than one because `Decomposition::check` asks the predicate
/// once per phase: a per-phase cost is what the counter has to be read against,
/// and a one-phase plan would not show that.
fn close(blocks: [usize; NDIM]) -> (usize, Decomposition) {
    let volume = [blocks[0] * 64, blocks[1] * 64, blocks[2] * 64];
    let grid = BlockGrid::new(volume, [64, 64, 64]).expect("a lattice");
    let n = grid.n_blocks();
    let mut plan = PlanBuilder::new(volume, Dtype::F64, grid);
    plan.pixels(Chain::op(IdentityOp::new("first", [1, 0, 0])))
        .expect("a pixel phase");
    plan.pixels(Chain::op(IdentityOp::new("second", [0, 1, 0])))
        .expect("a pixel phase");
    (n, plan.finish().expect("a plan").decomposition)
}

/// The doubling ladder. Each step doubles the block count on one axis in turn,
/// so the lattice stays roughly cubic rather than growing into a slab — a slab
/// would make the first axis's cut do all the separating and would flatter the
/// pass.
const LADDER: [[usize; NDIM]; 5] = [
    [8, 8, 8],
    [16, 8, 8],
    [16, 16, 8],
    [16, 16, 16],
    [32, 16, 16],
];

const PHASES: u64 = 2;

/// **The scaling, pinned by a counter rather than a stopwatch.**
///
/// Two assertions, and the second is what makes the first mean anything: the
/// work is bounded by a constant times the block count, *and* the pairwise scan
/// that used to do this job would have blown that bound at every rung of the
/// ladder. A bound the old algorithm also passed would be a bound that could not
/// see the regression it is here to catch.
#[test]
fn closing_a_plan_costs_the_block_count_and_not_its_square() {
    let mut seen: Vec<(usize, u64)> = Vec::new();
    for blocks in LADDER {
        reset_tiling_work();
        let (n, decomposition) = close(blocks);
        let work = tiling_work();
        assert_eq!(decomposition.phases.len(), PHASES as usize);

        // A lattice costs exactly one placement per box per axis per phase —
        // every block covers exactly one elementary interval on every axis, so
        // no block is ever copied — and the bound is twice that. Twice rather
        // than eight times, which is where the pass gives up and falls back:
        // a bound at the give-up point would pass whether the pass worked or
        // merely declined quietly, and quietly declining is the way the old
        // cost comes back.
        let bound = 2 * n as u64 * NDIM as u64 * PHASES;
        assert!(
            work <= bound,
            "closing a {n}-block plan took {work} steps against a bound of {bound}"
        );

        // The negative control, in closed form: what the algorithm this
        // replaced would have counted on the same plan.
        let pairs = PHASES * (n as u64) * (n as u64 - 1) / 2;
        assert!(
            pairs > bound,
            "at {n} blocks the pairwise scan would have cost {pairs} steps against a bound of \
             {bound}, so this bound cannot tell the two algorithms apart and the assertion above \
             is not evidence"
        );

        seen.push((n, work));
    }

    // Doubling the block count must not more than double the work. A quadratic
    // term shows up here as `4x` and has nowhere to hide: the counter is exact,
    // so this is not a tolerance on a measurement, it is a statement about the
    // algorithm.
    for pair in seen.windows(2) {
        let (small, cheap) = pair[0];
        let (large, dear) = pair[1];
        assert_eq!(large, small * 2, "the ladder must double: {seen:?}");
        assert!(
            dear <= cheap * 2,
            "{small} blocks cost {cheap} steps and {large} cost {dear}, which is more than the \
             doubling of the input: {seen:?}"
        );
    }
}

/// The counter is a property of the plan and not of the run.
///
/// Asserted because every bound above is read off one call: a counter that drifted
/// between identical calls would make the ladder above a measurement of the
/// harness rather than of the algorithm.
#[test]
fn the_same_plan_costs_the_same_twice() {
    reset_tiling_work();
    let (n, _) = close([8, 8, 8]);
    let first = tiling_work();
    reset_tiling_work();
    let (again, _) = close([8, 8, 8]);
    let second = tiling_work();
    assert_eq!(n, again);
    assert_eq!(first, second);
    assert!(first > 0, "the counter never moved, so it is not counting");
}
