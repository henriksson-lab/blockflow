// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The candidate ladder's granularity, and the guarantee it collides with.**
//
// `Constraints::block_candidates`' header records a sweep whose answer was
// *granularity, not anisotropy*: a per-axis candidate family wins a minority of
// cells and never by much, while a rung at three quarters of each power of two
// wins by up to 2.7x and costs nothing but entries. That sweep existed and the
// ladder it argued for did not. `decomposition::refined_ladder` is it.
//
// The collision, which is the reason this file exists
// ---------------------------------------------------
// `budget::UNOBSERVED_SHAPE_MARGIN`'s header proves that an admission margin
// costs **at most one ladder step**, and calls it arithmetic rather than luck: a
// ladder of powers of two steps by `8x` in volume, and every margin is under
// eight — `3.6` alone, and `3.5626 x 2.1 = 7.48` for the worst measured shape.
//
// **That wording does not survive a finer ladder**, and the file that proves the
// bound would go on passing without noticing: `tests/working_set_residency.rs`
// hard-codes its own powers-of-two ladder, so it cannot see a ladder a caller
// supplies. A refined step is `(4/3)^3 = 2.37x` or `(3/2)^3 = 3.375x`,
// alternating, so `3.6` already spans two of them.
//
// **The arithmetic survives; the unit it was stated in does not.** Two
// consecutive refined rungs span exactly `2.37 x 3.375 = 8.0`, because the
// refinement interleaves one rung into each octave and changes nothing else. So
// every margin under eight costs at most two refined rungs, which is at most the
// same `8x` in block volume the one-step bound guaranteed. **The bound that
// survives any spacing is the volume ratio; the step count was a proxy that
// happened to equal it while the ladder was powers of two.**
//
// Everything below is measured against the real `admission_bytes` and the real
// `price_phase`, at every budget of a sweep, on both ladders. Nothing here
// re-derives the 2.7x sweep — that is recorded where it was measured — but the
// mechanism behind it is asserted, because a mechanism nobody can reproduce is a
// number waiting to rot.

use blockflow::budget::{
    admission_bytes, FrameworkFigure, UNOBSERVED_OP_MARGIN, UNOBSERVED_SHAPE_MARGIN,
};
use blockflow::decomposition::{
    price_phase, refined_ladder, BlockLadder, Constraints, CostModel, PhaseTraffic,
};
use blockflow::geometry::BlockGrid;
use blockflow::reach::Reach;

const VOLUME: [usize; 3] = [1024, 1024, 1024];
const CONCURRENCY: u64 = 40;
/// The ladder `working_set_residency.rs` proves the one-step bound against.
const COARSE: [usize; 6] = [512, 256, 128, 64, 32, 16];

/// The working set the planner charges for one block at `edge`, or `None` where
/// no grid exists at that edge.
fn working_set(edge: usize) -> Option<f64> {
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], edge).ok()?;
    Some(
        price_phase(
            &grid,
            &Reach::symmetric([0, 0, 0]),
            1.0,
            1,
            false,
            8.0,
            &CostModel::default(),
            1.0,
            PhaseTraffic::one_in_one_out(),
        )
        .working_set_bytes_per_block,
    )
}

/// The largest rung of `ladder` that fits `budget` under `charge` — which is
/// what admission does.
///
/// **Sorted here rather than trusted from the caller.** The two ladders in this
/// file are stored in opposite orders — `COARSE` descending because that is how
/// `working_set_residency.rs` writes it, `refined_ladder` ascending because that
/// is how a ladder reads — and a helper that assumed either would silently
/// admit the *smallest* rung that fits for one of them. It did, on the first
/// run of this file, and the sweep printed a block pinned at 16 at every budget.
fn admitted(ladder: &[usize], budget: u64, charge: &dyn Fn(f64) -> u64) -> usize {
    let mut descending: Vec<usize> = ladder.to_vec();
    descending.sort_unstable_by(|a, b| b.cmp(a));
    descending
        .iter()
        .copied()
        .find(|&edge| working_set(edge).is_some_and(|ws| charge(ws) * CONCURRENCY <= budget))
        .unwrap_or_else(|| *descending.last().expect("a ladder"))
}

/// Where `edge` sits on `ladder` counted from the largest rung, so the two
/// ladders' opposite storage orders cannot make a step count mean two things.
fn rung_from_the_top(ladder: &[usize], edge: usize) -> usize {
    let mut descending: Vec<usize> = ladder.to_vec();
    descending.sort_unstable_by(|a, b| b.cmp(a));
    descending
        .iter()
        .position(|&e| e == edge)
        .expect("on the ladder")
}

fn budgets() -> Vec<u64> {
    (0..9u32)
        .map(|p| (1u64 << p) * 1024 * 1024 * 1024)
        .collect()
}

fn charges() -> Vec<(&'static str, Box<dyn Fn(f64) -> u64>)> {
    vec![
        ("today", Box::new(|ws: f64| ws.round() as u64)),
        (
            "assumed x margin",
            Box::new(|ws: f64| admission_bytes(FrameworkFigure::Assumed(ws))),
        ),
        (
            "exact(3.56x) x margin",
            Box::new(|ws: f64| admission_bytes(FrameworkFigure::Exact((ws * 3.56).round() as u64))),
        ),
    ]
}

// ------------------------------------------------------------- the ladder --

/// **What `refined_ladder` builds**, stated as the shape rather than as a list
/// that would have to be kept in step with the function.
#[test]
fn the_refined_ladder_interleaves_one_rung_into_each_octave() {
    let fine = refined_ladder(&[32, 64, 128]);
    assert_eq!(fine, vec![32, 48, 64, 96, 128]);

    // Every original rung survives, and each gains a rung at three quarters of
    // it — except the smallest, whose three-quarter rung would be below the
    // floor.
    for coarse in [64usize, 128] {
        assert!(fine.contains(&coarse), "{coarse} left the ladder");
        assert!(
            fine.contains(&(coarse * 3 / 4)),
            "{coarse}'s 3/4 rung is missing"
        );
    }
    assert!(fine.contains(&32));
    assert!(!fine.contains(&24), "24 is below the floor this caller set");

    // **The floor is not lowered**, and that is what the sweep's own ladder did:
    // its `24` is three quarters of `32` on a ladder whose floor is `16`, and it
    // lists no `12`. The smallest rung is a caller's statement about how small a
    // block they will accept.
    let from_sixteen = refined_ladder(&COARSE);
    assert_eq!(*from_sixteen.first().expect("a ladder"), 16);
    assert!(!from_sixteen.contains(&12));
    for named in [24usize, 48, 96, 192, 384] {
        assert!(
            from_sixteen.contains(&named),
            "{named} is one of the rungs the sweep named and it is not on the ladder"
        );
    }

    // **It is a fixed point, and this assertion is the inversion of one that
    // said the opposite.** It used to refine whatever it was given, so three
    // quarters of `48` was `36` and a second application gave a third ladder.
    // That non-idempotence is exactly what made the old `with_refined_ladder()`
    // a bad thing to sweep — "which ladder is this" was not answerable from the
    // value — so the rule was narrowed to *three quarters of each power of two*,
    // which is what the sweep named in the first place (`24, 48, 96, 192, 384`)
    // and is invisible on an octave ladder where every entry is a power of two.
    //
    // `3k/4` for a power of two is `3 * 2^(k-2)`, never itself a power of two,
    // so there is nothing left for a second pass to insert.
    assert_eq!(
        refined_ladder(&fine),
        fine,
        "the refinement must be a fixed point, or a setting built on it cannot be read back"
    );
    assert_eq!(refined_ladder(&from_sixteen), from_sixteen);
    assert!(!refined_ladder(&fine).contains(&36));
    let widest = |ladder: &[usize]| {
        ladder
            .windows(2)
            .map(|pair| (pair[1] as f64 / pair[0] as f64).powi(3))
            .fold(0.0f64, f64::max)
    };
    eprintln!(
        "\nwidest step: coarse {:.3}x, refined {:.3}x",
        widest(&COARSE.iter().rev().copied().collect::<Vec<_>>()),
        widest(&fine),
    );

    // And it is reachable the one documented way.
    let constraints = Constraints::default().with_ladder(BlockLadder::Refined);
    assert_eq!(constraints.block_candidates, vec![32, 48, 64, 96, 128]);
    assert_eq!(
        Constraints::default().block_candidates,
        vec![32, 64, 128],
        "the default is unmoved; every recorded parity figure was measured under it"
    );
    assert_eq!(refined_ladder(&[]), Vec::<usize>::new());
}

/// **Two consecutive refined rungs are exactly one coarse rung.** This is the
/// arithmetic the restated bound rests on, asserted rather than argued.
#[test]
fn two_refined_rungs_span_exactly_one_coarse_rung_in_volume() {
    let fine = refined_ladder(&COARSE);
    let ratio = |big: usize, small: usize| (big as f64 / small as f64).powi(3);

    let mut steps = Vec::new();
    for pair in fine.windows(2) {
        steps.push(ratio(pair[1], pair[0]));
    }
    // Alternating, and neither alone is eight.
    for step in &steps {
        assert!(
            (step - 2.37037).abs() < 1e-4 || (step - 3.375).abs() < 1e-4,
            "a refined step of {step} is neither (4/3)^3 nor (3/2)^3"
        );
        assert!(*step < 8.0, "a refined step of {step} is not finer than 8x");
    }
    // But every consecutive pair is.
    for pair in fine.windows(3) {
        let two = ratio(pair[2], pair[0]);
        assert!(
            (two - 8.0).abs() < 1e-9,
            "two refined rungs span {two}x, not 8x, so the restated bound does not hold"
        );
    }
    // The coarse ladder's own step, for the comparison.
    for pair in COARSE.windows(2) {
        assert!((ratio(pair[0], pair[1]) - 8.0).abs() < 1e-9);
    }
}

// ------------------------------------------------------------ the collision --

/// **The one-step bound is false on a refined ladder, and the volume bound that
/// replaces it is true on both.**
///
/// The negative half is the point: if the refined ladder never cost two steps at
/// any budget, this test would be asserting nothing about the collision and the
/// old wording could have stood. It does cost two, and the assertion says so
/// rather than tolerating it.
#[test]
fn a_margin_costs_two_refined_rungs_and_never_more_than_eight_times_in_volume() {
    let fine = refined_ladder(&COARSE);
    let charges = charges();
    let mut worst_fine_steps = 0usize;
    let mut worst_coarse_steps = 0usize;

    eprintln!(
        "\n{:>7} | {:>16} {:>16} | {:>16} {:>16}",
        "budget", "coarse today", "coarse charged", "fine today", "fine charged"
    );
    for budget in budgets() {
        let coarse_today = admitted(&COARSE, budget, &charges[0].1);
        let fine_today = admitted(&fine, budget, &charges[0].1);
        for (name, charge) in charges.iter().skip(1) {
            for (ladder, today, worst) in [
                (&COARSE[..], coarse_today, &mut worst_coarse_steps),
                (&fine[..], fine_today, &mut worst_fine_steps),
            ] {
                let got = admitted(ladder, budget, charge);
                let steps =
                    rung_from_the_top(ladder, got).saturating_sub(rung_from_the_top(ladder, today));
                *worst = (*worst).max(steps);

                // **The bound that survives any spacing.** Whatever the step
                // count, the correction never costs more than 8x in block
                // volume, because every margin is under eight.
                let ratio = (today as f64 / got as f64).powi(3);
                assert!(
                    ratio <= 8.0 + 1e-9,
                    "at {} GiB, {name} moved the block from {today} to {got}, which is \
                     {ratio:.3}x in volume. Every margin here is under 8x, so a correction \
                     costing more than 8x means a margin grew past the arithmetic the bound \
                     rests on — not that the ladder is too fine.",
                    budget / (1024 * 1024 * 1024)
                );
            }
        }
        eprintln!(
            "{:>5} GiB | {:>16} {:>16} | {:>16} {:>16}",
            budget / (1024 * 1024 * 1024),
            coarse_today,
            admitted(&COARSE, budget, &charges[1].1),
            fine_today,
            admitted(&fine, budget, &charges[1].1),
        );
    }

    assert_eq!(
        worst_coarse_steps, 1,
        "the coarse ladder's one-step bound is what `working_set_residency.rs` proves; if it \
         moved, that file and this one disagree"
    );
    assert_eq!(
        worst_fine_steps, 2,
        "the refined ladder must actually cost two rungs somewhere, or this test is not \
         measuring the collision it exists for and the one-step wording could have stood"
    );
    eprintln!(
        "worst correction: {worst_coarse_steps} coarse rung(s), {worst_fine_steps} refined \
         rung(s) — the same 8x in volume"
    );
}

/// **Both margins, checked against the refined spacing directly.**
///
/// `budget.rs` names two numbers and asserts each is under eight. On the coarse
/// ladder that makes each cost one step. On the refined one it makes each cost
/// at most two rungs — and `3.6` costs two, which is the specific fact that
/// breaks the old sentence.
#[test]
fn every_margin_is_under_one_coarse_rung_and_over_one_refined_rung() {
    let fine = refined_ladder(&COARSE);
    let step = |a: usize, b: usize| (a as f64 / b as f64).powi(3);
    let one_refined = step(fine[1], fine[0]).min(step(fine[2], fine[1]));
    let worst_exact = 3.5626 * UNOBSERVED_OP_MARGIN;

    for (name, margin) in [
        ("UNOBSERVED_SHAPE_MARGIN", UNOBSERVED_SHAPE_MARGIN),
        (
            "the worst measured shape x UNOBSERVED_OP_MARGIN",
            worst_exact,
        ),
    ] {
        assert!(
            margin < 8.0,
            "{name} is {margin}, which is not under one coarse rung — the bound's arithmetic \
             has stopped holding and `budget.rs`'s header needs rewriting, not this test"
        );
        assert!(
            margin > one_refined,
            "{name} is {margin}, which is under one refined rung of {one_refined:.4}x. If \
             every margin fitted in one refined rung the collision would not exist and this \
             file would be unnecessary."
        );
    }
    eprintln!(
        "\nmargins {UNOBSERVED_SHAPE_MARGIN} and {worst_exact:.4}; one refined rung is \
         {one_refined:.4}x, one coarse rung is 8x"
    );
}

// ----------------------------------------------------------- what it buys --

/// **A finer ladder never admits a smaller block than a coarse one, and
/// sometimes admits a larger one.**
///
/// This is the mechanism behind the recorded 2.7x, reproducible in one page: the
/// planner takes the largest rung that fits, so adding rungs can only move the
/// answer up. The negative half — that it *does* move, somewhere — is what keeps
/// the test from passing on a ladder refinement that did nothing.
#[test]
fn a_finer_ladder_never_admits_a_smaller_block_than_a_coarse_one() {
    let fine = refined_ladder(&COARSE);
    let mut improved = 0usize;
    let mut best_gain = 1.0f64;

    for budget in budgets() {
        for (name, charge) in charges() {
            let coarse = admitted(&COARSE, budget, &charge);
            let refined = admitted(&fine, budget, &charge);
            assert!(
                refined >= coarse,
                "at {} GiB under {name}, the refined ladder admitted {refined} where the \
                 coarse one admitted {coarse}. Refining adds rungs and never removes one, so \
                 the largest that fits cannot get smaller.",
                budget / (1024 * 1024 * 1024)
            );
            if refined > coarse {
                improved += 1;
                best_gain = best_gain.max((refined as f64 / coarse as f64).powi(3));
            }
        }
    }

    assert!(
        improved > 0,
        "the refined ladder never once admitted a larger block, so it buys nothing here and \
         the sweep it was built from does not reproduce"
    );
    eprintln!(
        "\nrefined ladder admitted a larger block in {improved} of {} (budget, charge) cells; \
         best gain {best_gain:.2}x in volume",
        budgets().len() * 3
    );
}

/// The finer ladder costs entries, and the search is exponential in them. Stated
/// as a number a caller can act on rather than as a warning.
#[test]
fn the_refined_ladder_squares_the_per_phase_search_factor() {
    let coarse = Constraints::default().block_candidates;
    let fine = Constraints::default()
        .with_ladder(BlockLadder::Refined)
        .block_candidates;
    // `2n - 1`, not `2n`: every rung gains a partner except the floor, whose
    // partner would be below it.
    assert_eq!(fine.len(), 2 * coarse.len() - 1);
    for phases in [1usize, 2, 4] {
        let before = coarse.len().pow(phases as u32);
        let after = fine.len().pow(phases as u32);
        assert!(after > before);
        eprintln!(
            "{phases} phase(s): {before} candidate combination(s) become {after} ({:.1}x)",
            after as f64 / before as f64
        );
    }
}

// ------------------------------------------------------------- the setting --

/// **`BlockLadder` is a setting, and these are the three properties that makes
/// it one.** The method it replaced had none of them, and each failure bites
/// exactly when planners are compared rather than argued about.
#[test]
fn the_ladder_is_a_setting_a_competition_can_sweep() {
    // **Enumerable.** A competition that writes the variants out by hand stops
    // covering a new one the day it is added, so the list is the type's.
    assert_eq!(BlockLadder::ALL.len(), 2);
    assert!(BlockLadder::ALL.contains(&BlockLadder::Octave));
    assert!(BlockLadder::ALL.contains(&BlockLadder::Refined));
    assert_eq!(BlockLadder::default(), BlockLadder::Octave);

    // **Idempotent**, in the ladder and in the value: setting the same ladder
    // twice is setting it once, so a caller may set it without knowing what was
    // set before, and "which ladder is this" is answerable from the list.
    for ladder in BlockLadder::ALL {
        let once = Constraints::default().with_ladder(ladder);
        let twice = Constraints::default()
            .with_ladder(ladder)
            .with_ladder(ladder);
        assert_eq!(
            once.block_candidates,
            twice.block_candidates,
            "{} is not idempotent, so it is an operator rather than a setting",
            ladder.as_str()
        );
    }

    // **Sweepable**, which is the whole point: the variants go in a list and
    // the loop is the competition's.
    let mut seen: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for ladder in BlockLadder::ALL {
        let constraints = Constraints::default().with_ladder(ladder);
        seen.push((ladder.as_str(), constraints.block_candidates));
    }
    assert_eq!(seen[0], ("octave", vec![32, 64, 128]));
    assert_eq!(seen[1], ("refined", vec![32, 48, 64, 96, 128]));

    // **One effective list.** The ladder is applied to `block_candidates` rather
    // than stored beside it, so there is no second field to disagree with it and
    // every read site in the crate — including the ones this test does not own —
    // sees the rungs the planner will actually price.
    let refined = Constraints::default().with_ladder(BlockLadder::Refined);
    assert_eq!(
        refined.block_candidates,
        BlockLadder::Refined.rungs(&[32, 64, 128])
    );

    // And switching back is a different question rather than two compounded:
    // `Octave` over an already-refined list keeps it, because `Octave` is the
    // identity on whatever rungs it is given. A caller who wants the coarse
    // ladder states the coarse rungs; the ladder names the spacing, not the set.
    let back = refined.clone().with_ladder(BlockLadder::Octave);
    assert_eq!(
        back.block_candidates,
        vec![32, 48, 64, 96, 128],
        "`Octave` is the identity on the rungs it is given, and says so"
    );
}
