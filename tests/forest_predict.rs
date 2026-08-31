// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for `blockflow::forest` and `blockflow::ops::classify` —
// steps 3 and 4 of `docs/design/pixel-classification.md`.
//
// The bar here is different from an image op's, and it is worth saying how. A
// filter is checked against a closed form; a fitted classifier has none. So the
// properties below are of three kinds:
//
// 1. **Exact, on a hand-built forest.** A tree of three nodes has an answer
//    anyone can compute, and it pins the layout, the comparison sense and the
//    tie rule. This is where a wrong `<=` would show.
// 2. **Structural, on a fitted forest.** The invariant `Forest::new` checks —
//    every child after its parent — is what makes the walk terminate, so the
//    trainer is required to produce it and a forest that breaks it is required
//    to be refused.
// 3. **Statistical, and stated as a floor rather than a figure.** A forest
//    fitted to a separable problem must separate it. The threshold is set where
//    a broken implementation fails and a working one passes by a wide margin,
//    never tuned to what the current code happens to score.
//
// Plus the measurement step 3 asks for: `cost_per_voxel`, in nanoseconds, at a
// few tree counts and depths.

use std::sync::Arc;

use blockflow::forest::{Forest, Node, Samples, TrainingSpec, LEAF};
use blockflow::op::{Anchor, Combine};
use blockflow::ops::{Family, FeatureStack, ForestPredictor, Prediction};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

// ------------------------------------------------------------- fixtures --

fn channels(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("c{index}")).collect()
}

/// One tree, three nodes: split on channel 0 at 0.5, and two leaves. The
/// smallest forest that has a decision in it.
fn stump() -> Forest {
    Forest::new(
        vec![Node::split(0, 0.5, 1, 2), Node::leaf(0), Node::leaf(2)],
        vec![0],
        vec![1.0, 0.0, 0.0, 1.0],
        2,
        channels(1),
    )
    .unwrap()
}

/// A two-class problem separable by one feature, with the other channels pure
/// noise — so a forest that ignored `mtry` and always split on channel 0 and one
/// that draws properly both succeed, but one that never found the signal cannot.
fn separable(rows: usize, columns: usize, seed: u64) -> Samples {
    separable_with(rows, columns, seed, 0.35)
}

/// The same, with the noise on the signal channel as a parameter.
///
/// The classes sit at 0 and 1 and the noise is `spread * U[0,1)`, so the two
/// **only overlap once `spread` exceeds 1** — below that the problem is exactly
/// separable however large the noise looks, the trees are two or three nodes
/// deep, and a test that counts splits has nothing to count. Measured, after a
/// spread of 0.9 produced 32 splits across 20 trees.
fn separable_with(rows: usize, columns: usize, seed: u64, spread: f64) -> Samples {
    let mut state = seed | 1;
    let mut draw = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };
    let mut features = Vec::with_capacity(rows * columns);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let class = (row % 2) as u32;
        for column in 0..columns {
            features.push(if column == 3 {
                // The signal, with the two classes overlapping so this is not a
                // threshold anybody could read off by eye.
                class as f64 + spread * draw()
            } else {
                draw()
            });
        }
        labels.push(class);
    }
    Samples::new(features, labels, channels(columns)).unwrap()
}

// ------------------------------------------------------------ 1. exact --

/// **The comparison sense, and the layout.** `features[f] <= threshold` goes
/// left. Checked at the threshold itself, which is the one value where the two
/// senses differ and the one a fitted midpoint never lands on — so if it were
/// wrong, only this test would say so.
#[test]
fn a_stump_splits_where_it_says_it_does() {
    let forest = stump();
    let mut scratch = vec![0.0f32; forest.classes()];
    assert_eq!(forest.predict(&[0.0], &mut scratch), 0);
    assert_eq!(forest.predict(&[0.5], &mut scratch), 0, "at the threshold");
    assert_eq!(forest.predict(&[0.500001], &mut scratch), 1);
    assert_eq!(forest.predict(&[1.0], &mut scratch), 1);

    assert_eq!(forest.probability(&[0.0], 0, &mut scratch), 1.0);
    assert_eq!(forest.probability(&[1.0], 0, &mut scratch), 0.0);
    assert_eq!(forest.depth(), 2);
    assert_eq!(forest.mean_path(), 2.0);
}

/// **Ties go to the lowest class**, and the rule is a rule rather than whatever
/// the loop happened to do — two trees voting one each is the commonest case on
/// a two-class problem with an even tree count.
#[test]
fn a_tie_goes_to_the_lowest_class() {
    let forest = Forest::new(
        vec![Node::leaf(0), Node::leaf(2)],
        vec![0, 1],
        vec![1.0, 0.0, 0.0, 1.0],
        2,
        channels(1),
    )
    .unwrap();
    let mut scratch = vec![0.0f32; 2];
    assert_eq!(forest.predict(&[0.0], &mut scratch), 0);
    assert_eq!(forest.probability(&[0.0], 0, &mut scratch), 0.5);
    assert_eq!(forest.probability(&[0.0], 1, &mut scratch), 0.5);
}

/// `accumulate` adds rather than assigns, which is what lets the predictor reuse
/// one tally across a block. Stated in its documentation, so it is asserted.
#[test]
fn accumulating_adds_to_what_is_already_there() {
    let forest = stump();
    let mut tally = vec![2.0f32, 5.0];
    forest.accumulate(&[0.0], &mut tally);
    assert_eq!(tally, vec![3.0, 5.0]);
}

// ------------------------------------------------------- 2. structural --

/// **The invariant the walk's termination rests on**, checked in both
/// directions: the trainer must produce it, and a forest that breaks it must be
/// refused rather than looping forever on the first voxel.
#[test]
fn every_child_comes_after_its_parent_and_a_forest_that_breaks_that_is_refused() {
    let fitted = Forest::train(&separable(400, 8, 11), &TrainingSpec::default()).unwrap();
    let mut splits = 0;
    for (index, node) in fitted.nodes().iter().enumerate() {
        if node.feature == LEAF {
            continue;
        }
        assert!(node.left as usize > index && node.right as usize > index);
        splits += 1;
    }
    assert!(
        splits > 0,
        "the fitted forest is all leaves and proves nothing"
    );

    // A cycle: node 1 points back at node 0.
    let err = Forest::new(
        vec![Node::split(0, 0.5, 1, 1), Node::split(0, 0.5, 0, 0)],
        vec![0],
        vec![1.0, 0.0],
        2,
        channels(1),
    )
    .expect_err("a cycle must be refused")
    .to_string();
    assert!(err.contains("does not exceed its parent"), "{err}");
}

/// Each refusal is of something that would otherwise be an infinite loop, an
/// out-of-bounds read or a wrong answer at every voxel.
#[test]
fn a_malformed_forest_is_refused_with_its_reason() {
    let ok = |nodes, roots, votes, classes, names| Forest::new(nodes, roots, votes, classes, names);
    // a child past the end
    assert!(ok(
        vec![Node::split(0, 0.5, 9, 9)],
        vec![0],
        vec![1.0, 0.0],
        2,
        channels(1)
    )
    .is_err());
    // a split on a column the forest does not name
    assert!(ok(
        vec![Node::split(7, 0.5, 1, 2), Node::leaf(0), Node::leaf(0)],
        vec![0],
        vec![1.0, 0.0],
        2,
        channels(1)
    )
    .is_err());
    // a leaf whose distribution runs off the end
    assert!(ok(vec![Node::leaf(1)], vec![0], vec![1.0, 0.0], 2, channels(1)).is_err());
    // a root that is not a node
    assert!(ok(vec![Node::leaf(0)], vec![5], vec![1.0, 0.0], 2, channels(1)).is_err());
    // no trees, one class, no channels, a repeated channel name
    assert!(ok(vec![Node::leaf(0)], vec![], vec![1.0, 0.0], 2, channels(1)).is_err());
    assert!(ok(vec![Node::leaf(0)], vec![0], vec![1.0], 1, channels(1)).is_err());
    assert!(ok(vec![Node::leaf(0)], vec![0], vec![1.0, 0.0], 2, vec![]).is_err());
    assert!(ok(
        vec![Node::split(0, 0.5, 1, 2), Node::leaf(0), Node::leaf(0)],
        vec![0],
        vec![1.0, 0.0],
        2,
        vec!["a".to_string(), "a".to_string()]
    )
    .is_err());
    // a non-finite threshold
    assert!(ok(
        vec![Node::split(0, f64::NAN, 1, 2), Node::leaf(0), Node::leaf(0)],
        vec![0],
        vec![1.0, 0.0],
        2,
        channels(1)
    )
    .is_err());
}

/// The samples' own refusals.
#[test]
fn malformed_samples_are_refused() {
    assert!(Samples::new(vec![1.0], vec![0], vec![]).is_err());
    assert!(Samples::new(vec![], vec![], channels(1)).is_err());
    assert!(Samples::new(vec![1.0, 2.0], vec![0], channels(1)).is_err());
    assert!(Samples::new(vec![f64::NAN], vec![0], channels(1)).is_err());
    // one class only
    assert!(Samples::new(vec![1.0, 2.0], vec![0, 0], channels(1)).is_err());
    assert!(Forest::train(
        &separable(50, 4, 1),
        &TrainingSpec {
            trees: 0,
            ..Default::default()
        }
    )
    .is_err());
}

/// **The fit is a function of the seed and nothing else.** Two runs at one seed
/// give the same forest byte for byte; two seeds give different ones, which is
/// what says the seed is actually reaching the bagging and the `mtry` draw.
#[test]
fn training_is_reproducible_from_its_seed_and_varies_with_it() {
    let samples = separable(300, 8, 7);
    let spec = TrainingSpec {
        trees: 10,
        ..Default::default()
    };
    assert_eq!(
        Forest::train(&samples, &spec).unwrap(),
        Forest::train(&samples, &spec).unwrap()
    );
    let other = Forest::train(
        &samples,
        &TrainingSpec {
            seed: spec.seed ^ 0xffff,
            ..spec
        },
    )
    .unwrap();
    assert_ne!(Forest::train(&samples, &spec).unwrap(), other);
}

// ------------------------------------------------------ 3. statistical --

/// **A forest fitted to a separable problem separates it**, and the floor is set
/// where a broken implementation fails rather than where this one happens to
/// land.
///
/// Held out, not in-sample: a tree grown to purity scores 100% on its training
/// rows however wrong its splits are, so an in-sample figure would pass for an
/// implementation that memorised the bag and learned nothing.
#[test]
fn a_fitted_forest_classifies_held_out_rows() {
    let train = separable(600, 8, 3);
    let test = separable(400, 8, 999);
    let forest = Forest::train(&train, &TrainingSpec::default()).unwrap();
    let mut scratch = vec![0.0f32; forest.classes()];
    let correct = (0..test.rows())
        .filter(|&row| forest.predict(test.row(row), &mut scratch) == test.labels()[row])
        .count();
    let accuracy = correct as f64 / test.rows() as f64;
    assert!(
        accuracy > 0.9,
        "held-out accuracy {accuracy:.3} on a problem separable by one feature — a forest \
         that found the signal scores near 1.0 and one that did not scores near 0.5"
    );

    // And it is finding the *right* feature: channel 3 carries the signal and
    // the other seven are noise, so the fitted splits must concentrate there.
    let mut on_signal = 0;
    let mut splits = 0;
    for node in forest.nodes() {
        if node.feature != LEAF {
            splits += 1;
            if node.feature == 3 {
                on_signal += 1;
            }
        }
    }
    assert!(
        on_signal * 4 > splits,
        "only {on_signal} of {splits} splits used the one informative channel, so the \
         accuracy above is not coming from where it should"
    );
}

/// **`mtry` is honoured**, checked on *which column each tree splits on first*.
///
/// Two earlier versions of this test were wrong and both are worth recording,
/// because each failed in a different direction:
///
/// * **On held-out accuracy.** On a problem separable by one feature, a forest
///   restricted to one column per split still reaches that column at some node
///   of some tree, and both settings scored 1.000. The test would have passed
///   for an implementation that ignored `mtry` entirely.
/// * **On the share of *all* splits using the informative column.** That share
///   is diluted by the deep nodes, where the surviving rows are few and a noise
///   column beats the signal on Gini by chance. At 200 trees it came out 0.147
///   against 0.120 — indistinguishable.
///
/// The root split has neither problem. It is taken over the whole bag, where the
/// signal is strongest, and there is exactly one per tree.
///
/// **And at `mtry = 8` of 8 the informative column opens two thirds of the
/// trees, not all of them** — which is the draw being *with replacement*, as
/// `best_split` documents. Eight draws from eight columns miss a given one with
/// probability `(7/8)^8 = 0.344`, so it should open `0.656` of the trees.
/// Measured 0.630. That is the sharpest available check on the sampling scheme,
/// so it is what is asserted, in place of the round number this test first
/// claimed and failed.
#[test]
fn mtry_actually_restricts_the_columns_a_split_may_use() {
    // A spread of 1.5 makes the classes genuinely overlap — they sit at 0 and 1,
    // so anything at or below 1 is exactly separable however noisy it looks —
    // while leaving the signal much the strongest column.
    let train = separable_with(900, 8, 3, 1.5);
    let roots_on_signal = |mtry| {
        let forest = Forest::train(
            &train,
            &TrainingSpec {
                trees: 200,
                mtry: Some(mtry),
                ..Default::default()
            },
        )
        .unwrap();
        let roots: Vec<u32> = forest
            .roots_used()
            .iter()
            .map(|&root| forest.nodes()[root as usize].feature)
            .collect();
        assert_eq!(roots.len(), 200);
        assert!(
            roots.iter().all(|&column| column != LEAF),
            "a tree is a bare leaf, so it has no root split to attribute"
        );
        roots.iter().filter(|&&column| column == 3).count() as f64 / roots.len() as f64
    };
    let narrow = roots_on_signal(1);
    let wide = roots_on_signal(8);
    // **The bound on the restricted share is loose on purpose.** Its expected
    // value is an eighth, but it is a draw: measured over six seeds at 500 trees
    // it ran 0.112 to 0.164, and pinning it tighter would be pinning a
    // particular seed's luck. What the test has to separate is a restricted draw
    // from no restriction at all, and those are an eighth against nearly all.
    assert!(
        narrow < 0.3,
        "at mtry 1 the column is drawn uniformly from 8, so the informative one should \
         open something near an eighth of the trees; it opened {narrow:.3}"
    );
    let with_replacement = 1.0 - (7.0f64 / 8.0).powi(8);
    assert!(
        (with_replacement - 0.656).abs() < 0.001,
        "the arithmetic below is stated against 0.656"
    );
    assert!(
        (wide - with_replacement).abs() < 0.08,
        "at mtry 8 of 8 the informative column is offered with probability \
         {with_replacement:.3} — eight draws *with replacement* — and it is much the best \
         when offered, so it should open about that share of the trees. It opened \
         {wide:.3}. A draw without replacement would give 1.000."
    );
    assert!(
        wide > 3.0 * narrow,
        "the two settings are indistinguishable, so `mtry` is not reaching the draw"
    );
}

// ------------------------------------------------------- 4. the op shell --

/// The predictor's declarations, which is what the planner reads.
#[test]
fn the_predictor_declares_reach_zero_and_is_not_a_fold() {
    let forest = Arc::new(stump());
    let predictor = ForestPredictor::new("classify", forest, Prediction::Label).unwrap();
    for axis in 0..3 {
        assert_eq!(predictor.reach(axis, 1024), 0);
    }
    assert_eq!(predictor.produces(&[Dtype::F64]), Dtype::U32);
    assert!(predictor.accepts(&[Dtype::F64]));
    assert!(!predictor.accepts(&[Dtype::F32]), "a mixed precision stack");
    assert!(
        !predictor.accepts(&[Dtype::F64, Dtype::F64]),
        "the wrong width"
    );
    assert!(
        predictor.fold_carrier(&[Dtype::F64]).is_none(),
        "a tree walk needs every channel at once and cannot be a left fold over pairs"
    );

    let probability = ForestPredictor::new(
        "classify",
        Arc::new(stump()),
        Prediction::Probability { class: 1 },
    )
    .unwrap();
    assert_eq!(probability.produces(&[Dtype::F64]), Dtype::F64);
    // A class the forest does not have would be zero everywhere; refused.
    assert!(ForestPredictor::new(
        "classify",
        Arc::new(stump()),
        Prediction::Probability { class: 9 }
    )
    .is_err());
}

/// **The predictor over a block gives the same answers as the forest over its
/// rows.** The op shell and the kernel are one answer, which is what makes every
/// property above a property of the op too.
#[test]
fn the_op_agrees_with_the_forest_it_holds() {
    let samples = separable(400, 4, 21);
    let forest = Arc::new(
        Forest::train(
            &samples,
            &TrainingSpec {
                trees: 12,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let shape = [6usize, 5, 4];
    let voxels = shape[0] * shape[1] * shape[2];

    // One image per channel, filled from the first `voxels` sample rows.
    let inputs: Vec<Voxels> = (0..samples.columns())
        .map(|column| {
            let values: Vec<f64> = (0..voxels).map(|at| samples.row(at)[column]).collect();
            ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
                .unwrap()
                .into()
        })
        .collect();
    let borrowed: Vec<&Voxels> = inputs.iter().collect();

    let predictor = ForestPredictor::new("classify", forest.clone(), Prediction::Label).unwrap();
    let mut out = Voxels::zeros(Dtype::U32, shape).unwrap();
    predictor
        .apply(&borrowed, &mut out, &Anchor::whole(shape))
        .unwrap();
    let got = out.view::<u32>().unwrap();

    let mut scratch = vec![0.0f32; forest.classes()];
    for at in 0..voxels {
        let want = forest.predict(samples.row(at), &mut scratch);
        assert_eq!(got.as_slice().unwrap()[at], want, "voxel {at}");
    }
}

/// **The check that a wrong column order is refused at build time**, which is
/// the failure this whole workflow is most exposed to: a stack of the right
/// width in the wrong order runs to completion and is wrong everywhere.
#[test]
fn a_forest_is_refused_against_a_stack_it_was_not_trained_on() {
    let stack = FeatureStack::labkit(&[1.0, 2.0])
        .unwrap()
        .with_families(&[Family::Gaussian, Family::GradientMagnitude])
        .unwrap();
    let names = stack.channel_names().unwrap();
    let train = |names: Vec<String>| {
        let columns = names.len();
        let rows = 200;
        let mut state = 12345u64;
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for row in 0..rows {
            for _ in 0..columns {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                features.push((state >> 33) as f64 / (1u64 << 31) as f64);
            }
            labels.push((row % 2) as u32);
        }
        Arc::new(
            Forest::train(
                &Samples::new(features, labels, names).unwrap(),
                &TrainingSpec {
                    trees: 4,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    };

    // The matching stack builds.
    blockflow::ops::predict_workflow(&stack, train(names.clone()), Prediction::Label)
        .expect("a forest trained on this stack's own channels must be accepted");

    // The same names in a different order does not, and says where.
    let mut swapped = names.clone();
    swapped.swap(0, 1);
    let err = match blockflow::ops::predict_workflow(&stack, train(swapped), Prediction::Label) {
        Ok(_) => panic!("a reordered stack must be refused"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("first differing at 0"), "{err}");

    // And so does one of the wrong width.
    let mut shorter = names;
    shorter.pop();
    assert!(blockflow::ops::predict_workflow(&stack, train(shorter), Prediction::Label).is_err());
}

/// End to end at a small size: a stack, a forest fitted on samples drawn from
/// it, and one chain that runs.
#[test]
fn the_predict_workflow_builds_a_chain_that_runs() {
    let stack = FeatureStack::labkit(&[1.0])
        .unwrap()
        .with_truncate(2.0)
        .unwrap()
        .with_families(&[Family::Gaussian, Family::GradientMagnitude])
        .unwrap();
    let names = stack.channel_names().unwrap();
    assert_eq!(names.len(), 2);

    let mut state = 999u64;
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for row in 0..300 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (state >> 33) as f64 / (1u64 << 31) as f64;
        let class = (row % 2) as u32;
        features.push(class as f64 + 0.3 * noise);
        features.push(noise);
        labels.push(class);
    }
    let forest = Arc::new(
        Forest::train(
            &Samples::new(features, labels, names).unwrap(),
            &TrainingSpec {
                trees: 8,
                ..Default::default()
            },
        )
        .unwrap(),
    );

    let chain = blockflow::ops::predict_workflow(&stack, forest, Prediction::Label).unwrap();
    let volume = [12usize, 10, 8];
    assert_eq!(
        chain.reach3(&volume),
        [3, 3, 3],
        "the predictor adds nothing to the stack's reach"
    );

    let mut state = 4242u64;
    let input: ndarray::Array3<f64> =
        ndarray::Array3::from_shape_fn((volume[0], volume[1], volume[2]), |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        });
    let source: Voxels = input.into();
    let mut out = Voxels::zeros(Dtype::U32, volume).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(volume))
        .expect("the workflow chain must run");
    let labels = out.view::<u32>().unwrap();
    assert!(labels.iter().all(|&label| label < 2));
    assert!(
        labels.iter().any(|&label| label == 0) && labels.iter().any(|&label| label == 1),
        "every voxel got the same class, so this proves nothing about the classifier"
    );
}

// -------------------------------------------------------- 5. the table --

/// The measurement step 3 of the design document asks for: what one voxel costs,
/// at a few tree counts and depths.
///
/// ```text
/// cargo test --release --test forest_predict -- --ignored --nocapture
/// ```
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_the_predictor_cost() {
    let samples = separable(4000, 91, 5);
    let shape = [64usize, 64, 32];
    let voxels = shape[0] * shape[1] * shape[2];

    let inputs: Vec<Voxels> = (0..samples.columns())
        .map(|column| {
            let values: Vec<f64> = (0..voxels)
                .map(|at| samples.row(at % samples.rows())[column])
                .collect();
            ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
                .unwrap()
                .into()
        })
        .collect();
    let borrowed: Vec<&Voxels> = inputs.iter().collect();

    println!(
        "forest prediction, 91 channels, {}x{}x{}, best of 3",
        shape[0], shape[1], shape[2]
    );
    println!(
        "{:>7} {:>7} {:>7} {:>10} {:>12} {:>12} {:>12}",
        "trees", "depth", "nodes", "mean path", "visits/voxel", "ns/voxel", "ns/visit"
    );
    for (trees, max_depth) in [(10, 8), (10, 20), (50, 20), (100, 20), (200, 20)] {
        let forest = Forest::train(
            &samples,
            &TrainingSpec {
                trees,
                max_depth,
                ..Default::default()
            },
        )
        .unwrap();
        let visits = forest.trees() as f64 * forest.mean_path();
        let predictor =
            ForestPredictor::new("classify", Arc::new(forest.clone()), Prediction::Label).unwrap();
        let mut out = Voxels::zeros(Dtype::U32, shape).unwrap();
        // One untimed pass: a fresh buffer pays a page fault per page on first
        // touch, and what is wanted is the steady-state compute.
        predictor
            .apply(&borrowed, &mut out, &Anchor::whole(shape))
            .unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            predictor
                .apply(&borrowed, &mut out, &Anchor::whole(shape))
                .unwrap();
            let elapsed = started.elapsed().as_secs_f64() * 1e9 / voxels as f64;
            std::hint::black_box(out.view::<u32>().unwrap()[[0, 0, 0]]);
            best = best.min(elapsed);
        }
        println!(
            "{trees:>7} {:>7} {:>7} {:>10.2} {visits:>12.1} {best:>12.1} {:>12.2}",
            forest.depth(),
            forest.nodes().len(),
            forest.mean_path(),
            best / visits,
        );
    }
}

/// **What the stack costs against what the predictor costs**, through the
/// crate's own declarations rather than through a stopwatch — so a change to
/// either cost model shows up here.
///
/// The design document expects the predictor to dominate. It does, and by
/// enough that the ratio is worth pinning: a future change that made the
/// filters the expensive half would be a change to how this whole chain should
/// be planned.
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_the_stack_against_the_predictor() {
    let stack = FeatureStack::labkit(&[1.0, 2.0, 4.0, 8.0, 16.0]).unwrap();
    let arms = stack.branches().unwrap();
    let filters: f64 = arms.iter().map(|arm| arm.cost_per_voxel()).sum();
    for trees in [10usize, 100] {
        let samples = separable(2000, 91, 5);
        let forest = Forest::train(
            &samples,
            &TrainingSpec {
                trees,
                ..Default::default()
            },
        )
        .unwrap();
        let visits = forest.trees() as f64 * forest.mean_path();
        let predictor =
            ForestPredictor::new("classify", Arc::new(forest), Prediction::Label).unwrap();
        let cost = predictor.cost_per_voxel(arms.len());
        println!(
            "{trees:>4} trees: filters {filters:>10.0}, predictor {cost:>10.0}, \
             predictor is {:>5.1}% of the chain ({visits:.0} visits/voxel)",
            100.0 * cost / (cost + filters)
        );
    }
}

// ------------------------------------------------- 6. the whole workflow --

/// **Train and predict, end to end**, on a volume with a real structure in it.
///
/// The fixture is two textures — one smooth, one noisy — separated at a plane,
/// which is the kind of thing a pixel classifier is for and which no single
/// feature settles: the intensities have the same mean, so the classifier has to
/// use the neighbourhood features. Labels are drawn as two small sparse patches,
/// which is what a brush stroke is.
///
/// Asserted as a floor: the classifier must recover most of the volume it was
/// never shown. A broken chain scores about a half.
#[test]
fn training_on_two_brush_strokes_classifies_the_rest_of_the_volume() {
    // Long on the axis the two textures are split across, so that after
    // excluding a reach's margin from the faces *and* from the boundary plane —
    // where a neighbourhood feature legitimately mixes the two — there is still
    // an interior to score.
    let volume = [64usize, 32, 32];
    let mut state: u64 = 20260831;
    let mut draw = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    // Left half smooth, right half noisy, both with the same mean — so intensity
    // alone says nothing and only a neighbourhood feature separates them.
    let intensity =
        ndarray::Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, _, _)| {
            if i < volume[0] / 2 {
                0.5
            } else {
                draw()
            }
        });
    let input: Voxels = intensity.into();

    // Two brush strokes: a patch in each half, a few hundred voxels of 13,824.
    let mut labels = ndarray::Array3::<u32>::zeros((volume[0], volume[1], volume[2]));
    for i in 2..6 {
        for j in 2..8 {
            for k in 2..8 {
                labels[[i, j, k]] = 7;
                labels[[i + volume[0] / 2, j, k]] = 9;
            }
        }
    }
    let labelled = labels.iter().filter(|&&label| label != 0).count();
    assert!(
        labelled < volume.iter().product::<usize>() / 10,
        "not sparse"
    );
    let labels: Voxels = labels.into();

    // One sigma: the reach at two would be 17 voxels, which leaves no interior
    // in a volume this size, and the discriminating feature here — the local
    // deviation — is at the small scale anyway.
    let stack = FeatureStack::labkit(&[1.0])
        .unwrap()
        .with_truncate(2.0)
        .unwrap();
    let (forest, classes) = blockflow::ops::train_workflow(
        &stack,
        &input,
        &labels,
        0,
        &TrainingSpec {
            trees: 30,
            ..Default::default()
        },
    )
    .unwrap();

    // The labels the annotator drew come back, mapped to classes in order.
    assert_eq!(classes.labels(), &[7, 9]);
    assert_eq!(classes.label_of(0), Some(7));
    assert_eq!(classes.label_of(1), Some(9));
    assert_eq!(forest.classes(), 2);
    assert_eq!(forest.channels().len(), stack.len());

    // And predicting over the whole volume recovers the two halves.
    let chain =
        blockflow::ops::predict_workflow(&stack, Arc::new(forest), Prediction::Label).unwrap();
    let mut out = Voxels::zeros(Dtype::U32, volume).unwrap();
    chain
        .apply(&input, &mut out, &Anchor::whole(volume))
        .unwrap();
    let predicted = out.view::<u32>().unwrap();

    // Scored away from the halo-affected faces and from the boundary plane
    // itself, where a neighbourhood feature legitimately mixes the two.
    let margin = chain.reach3(&volume)[0] + 2;
    let (mut correct, mut total) = (0usize, 0usize);
    for i in margin..volume[0] - margin {
        if i.abs_diff(volume[0] / 2) < margin {
            continue;
        }
        for j in margin..volume[1] - margin {
            for k in margin..volume[2] - margin {
                let want = u32::from(i >= volume[0] / 2);
                correct += usize::from(predicted[[i, j, k]] == want);
                total += 1;
            }
        }
    }
    assert!(total > 200, "only {total} voxels scored");
    let accuracy = correct as f64 / total as f64;
    assert!(
        accuracy > 0.95,
        "recovered {accuracy:.3} of a volume it was shown {labelled} voxels of; a chain \
         that learned nothing scores about 0.5"
    );
}

// ------------------------------------------------------- 7. the oracle --

/// **The agreement oracle**: `smartcore`'s random forest, on the same rows.
///
/// The witness shape `ops::scikitimage_watershed` uses against skimage, applied
/// here — and with the same care about *what* can be asserted. Two independently
/// implemented forests do not agree voxel for voxel and it would be wrong to ask
/// them to: they bag from different generators, draw `mtry` differently (with
/// replacement here, without there), and break Gini ties differently. Any of
/// those changes which rows land in which tree, and a forest is a vote over
/// trees.
///
/// What *is* comparable is what a forest is for. Both are fitted to the same
/// rows and scored on the same held-out rows, and this crate's must be within a
/// few points of `smartcore`'s. That catches the failures worth catching — a
/// trainer that splits on the wrong side of its threshold, that ignores `mtry`,
/// that bags without replacement, that never recovers the informative column —
/// every one of which shows as a systematic gap in accuracy, not as a rounding
/// difference.
///
/// The problem is deliberately not separable: at a spread of 1.5 the two classes
/// overlap, so neither implementation reaches 1.0 and the comparison has room to
/// show a difference. On a separable problem both score 1.000 and the test would
/// pass for anything.
#[test]
fn the_fit_agrees_with_smartcore_on_held_out_accuracy() {
    use smartcore::ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    };
    use smartcore::linalg::basic::matrix::DenseMatrix;

    let train = separable_with(600, 8, 3, 1.5);
    let test = separable_with(400, 8, 999, 1.5);
    let trees = 100u16;

    let ours = Forest::train(
        &train,
        &TrainingSpec {
            trees: trees as usize,
            ..Default::default()
        },
    )
    .unwrap();
    let mut scratch = vec![0.0f32; ours.classes()];
    let our_accuracy = (0..test.rows())
        .filter(|&row| ours.predict(test.row(row), &mut scratch) == test.labels()[row])
        .count() as f64
        / test.rows() as f64;

    let rows = |samples: &Samples| -> Vec<Vec<f64>> {
        (0..samples.rows())
            .map(|row| samples.row(row).to_vec())
            .collect()
    };
    let x = DenseMatrix::from_2d_vec(&rows(&train)).unwrap();
    let y: Vec<i64> = train.labels().iter().map(|&label| label as i64).collect();
    let theirs = RandomForestClassifier::fit(
        &x,
        &y,
        RandomForestClassifierParameters::default()
            .with_n_trees(trees)
            .with_max_depth(20)
            .with_seed(7),
    )
    .unwrap();
    let predicted = theirs
        .predict(&DenseMatrix::from_2d_vec(&rows(&test)).unwrap())
        .unwrap();
    let their_accuracy = predicted
        .iter()
        .zip(test.labels())
        .filter(|(&got, &want)| got == want as i64)
        .count() as f64
        / test.rows() as f64;

    // The problem must be hard enough that a difference could show.
    assert!(
        their_accuracy < 0.99,
        "smartcore scored {their_accuracy:.3}, so the problem is separable and this \
         comparison could not distinguish two implementations"
    );
    assert!(
        (our_accuracy - their_accuracy).abs() < 0.05,
        "held-out accuracy: this crate {our_accuracy:.3}, smartcore {their_accuracy:.3}"
    );
}

/// **The whole predict workflow is decomposition-invariant**, run through the
/// executor rather than argued from the predictor's reach of zero.
///
/// This is the crate's central property applied to the thing the design document
/// is for, and it is not implied by the tests above: those run one `apply` over
/// a whole volume, where no halo exists to be wrong. Here the plan cuts, every
/// arm carries its own reach, the predictor carries none, and the classified
/// volume must come out byte-identical — a label volume, so byte-identical is
/// the only tolerance there is.
#[test]
fn the_predict_workflow_gives_the_same_labels_under_every_decomposition() {
    use blockflow::decomposition::{Decomposition, PhaseDecomposition};
    use blockflow::env::ArrayEnvironment;
    use blockflow::geometry::BlockGrid;
    use blockflow::strategy::{execute, Hints, Workflow};

    let volume = [28usize, 24, 20];
    let stack = FeatureStack::labkit(&[1.0])
        .unwrap()
        .with_truncate(2.0)
        .unwrap();
    let names = stack.channel_names().unwrap();

    let mut state = 606u64;
    let mut draw = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };
    let intensity =
        ndarray::Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, _, _)| {
            if i < volume[0] / 2 {
                0.5
            } else {
                draw()
            }
        });
    let input: Voxels = intensity.clone().into();

    let mut labels = ndarray::Array3::<u32>::zeros((volume[0], volume[1], volume[2]));
    for i in 2..5 {
        for j in 2..7 {
            for k in 2..7 {
                labels[[i, j, k]] = 1;
                labels[[i + volume[0] / 2, j, k]] = 2;
            }
        }
    }
    let (forest, _) = blockflow::ops::train_workflow(
        &stack,
        &input,
        &labels.into(),
        0,
        &TrainingSpec {
            trees: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(forest.channels(), names.as_slice());
    let forest = Arc::new(forest);

    let build =
        || blockflow::ops::predict_workflow(&stack, forest.clone(), Prediction::Label).unwrap();

    // The oracle: one `apply` over the whole array.
    let mut out = Voxels::zeros(Dtype::U32, volume).unwrap();
    build()
        .apply(&input, &mut out, &Anchor::whole(volume))
        .unwrap();
    let want = out.view::<u32>().unwrap().to_owned();
    assert!(
        want.iter().any(|&label| label == 0) && want.iter().any(|&label| label == 1),
        "one class everywhere; this would be invariant for the wrong reason"
    );

    let mut cut = 0;
    for block in [7usize, 9, 13] {
        for split_axes in [&[0usize][..], &[2][..], &[0, 1][..], &[0, 1, 2][..]] {
            let workflow = Workflow::new(build(), volume, Dtype::F64);
            let reach = workflow.chain.reach3(&volume);
            let slots = workflow.chain.slots();
            let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
            let grid = BlockGrid::along(volume, split_axes, block).unwrap();
            let mut phase =
                PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
            // **The phase changes the element type and has to say so.** The
            // stack reads `f64` and the predictor writes `u32` labels, so the
            // image this phase allocates is a quarter the width of the one it
            // reads. A plan that left this unset is refused by name — which is
            // the check earning its keep on the first chain in the crate whose
            // sink narrows.
            phase.dtype = Some(Dtype::U32);
            let decomposition = Decomposition {
                volume,
                dtype: workflow.dtype,
                phases: vec![phase],
                chain_reach: reach,
            };
            decomposition.check().unwrap();
            // `for_decomposition` and not `new`: it allocates each image at the
            // element type its phase gives it, which for this chain is a `u32`
            // label volume rather than another `f64` one.
            let env = ArrayEnvironment::for_decomposition(input.clone(), &decomposition, [4, 4, 4])
                .unwrap();
            execute(
                "classify",
                &workflow,
                &decomposition,
                &Hints::default(),
                &env,
            )
            .unwrap();
            assert_eq!(
                env.output().view::<u32>().unwrap().to_owned(),
                want,
                "block {block}, split {split_axes:?}"
            );
            cut += 1;
        }
    }
    assert_eq!(cut, 12);
}

/// **The cropped gather yields the whole volume's rows, bit for bit.**
///
/// `gather_samples` computes the feature stack over the bounding box of the
/// labelled voxels grown by the chain's reach, not over the volume. That is only
/// legitimate if the rows are *identical* to what the whole volume would have
/// given, and the argument — that a labelled voxel either sits a full reach
/// inside the crop or has a crop edge that is the volume edge — is the kind that
/// is easy to believe and easy to get wrong at a boundary. So it is checked.
///
/// Two label placements, because they exercise the two halves of the argument:
/// one stroke in the middle, where the crop is a true interior box, and one
/// touching a corner, where the crop clamps and the boundary rule has to be the
/// volume's own.
#[test]
fn the_cropped_gather_gives_the_same_rows_as_the_whole_volume() {
    let volume = [40usize, 36, 32];
    let stack = FeatureStack::labkit(&[1.0])
        .unwrap()
        .with_truncate(2.0)
        .unwrap();
    let reach = stack.reach(volume).unwrap();
    assert!(
        reach.iter().all(|&r| r > 0),
        "a reach of zero proves nothing"
    );

    let mut state = 8080u64;
    let input: Voxels = ndarray::Array3::from_shape_fn((volume[0], volume[1], volume[2]), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    })
    .into();

    for (name, origin) in [("interior", [16usize, 15, 14]), ("at a corner", [0, 0, 0])] {
        let mut labels = ndarray::Array3::<u32>::zeros((volume[0], volume[1], volume[2]));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    labels[[origin[0] + i, origin[1] + j, origin[2] + k]] = 1;
                    labels[[origin[0] + i + 6, origin[1] + j, origin[2] + k]] = 2;
                }
            }
        }
        let labels: Voxels = labels.into();
        let (cropped, classes) =
            blockflow::ops::gather_samples(&stack, &input, &labels, 0).unwrap();
        assert_eq!(classes.labels(), &[1, 2]);
        assert_eq!(cropped.rows(), 128, "{name}");

        // The oracle: the same stack over the whole volume, gathered by hand.
        let mut columns: Vec<Vec<f64>> = Vec::new();
        for arm in stack.branches().unwrap() {
            let mut out = Voxels::zeros(arm.produces(Dtype::F64).unwrap(), volume).unwrap();
            arm.apply(&input, &mut out, &Anchor::whole(volume)).unwrap();
            columns.push(out.view::<f64>().unwrap().iter().copied().collect());
        }
        let label_view = labels.view::<u32>().unwrap();
        let mut want: Vec<Vec<f64>> = Vec::new();
        for (at, &label) in label_view.iter().enumerate() {
            if label != 0 {
                want.push(columns.iter().map(|column| column[at]).collect());
            }
        }

        assert_eq!(want.len(), cropped.rows(), "{name}");
        for (row, expected) in want.iter().enumerate() {
            assert_eq!(
                cropped.row(row),
                expected.as_slice(),
                "{name}: row {row} differs between the cropped gather and the whole volume"
            );
        }
    }
}

/// And the crop is actually a crop — it is not quietly the whole volume.
///
/// Without this the equality above would pass for an implementation that had
/// stopped cropping, which is the failure mode of an optimisation guarded only
/// by a correctness test.
#[test]
fn the_gather_reads_only_the_labels_neighbourhood() {
    let volume = [64usize, 64, 64];
    let stack = FeatureStack::labkit(&[1.0])
        .unwrap()
        .with_truncate(2.0)
        .unwrap();
    let reach = stack.reach(volume).unwrap();

    // A stroke in one corner. Everything outside its neighbourhood is set to a
    // value that would make the features enormous if it were read.
    let mut intensity =
        ndarray::Array3::<f64>::from_elem((volume[0], volume[1], volume[2]), 1.0e12);
    let stroke = 8usize;
    let touched = stroke + reach[0];
    for i in 0..touched {
        for j in 0..touched {
            for k in 0..touched {
                intensity[[i, j, k]] = 0.25;
            }
        }
    }
    let input: Voxels = intensity.into();

    let mut labels = ndarray::Array3::<u32>::zeros((volume[0], volume[1], volume[2]));
    for i in 0..stroke {
        for j in 0..stroke {
            for k in 0..stroke {
                labels[[i, j, k]] = if i < stroke / 2 { 1 } else { 2 };
            }
        }
    }
    let (samples, _) = blockflow::ops::gather_samples(&stack, &input, &labels.into(), 0).unwrap();

    // Every feature comes from the constant 0.25 region, so nothing is large.
    // A gather that read the whole volume would have pulled 1e12 into the
    // windows of the voxels near the edge of the stroke.
    let largest = (0..samples.rows())
        .flat_map(|row| samples.row(row).iter().copied())
        .fold(0.0f64, |worst, value| worst.max(value.abs()));
    assert!(
        largest < 1.0e6,
        "a feature reached {largest:e}, so the gather read past the labels' \
         neighbourhood into the region that is only there to be noticed"
    );
}

/// **Are a forest's splits stable if the feature stack is `f32`?**
///
/// `docs/design/pixel-classification.md` lists this as unmeasured and both
/// reference tools store their stacks at single precision, so it is the obvious
/// way to halve the 91 live buffers a predictor's fan-in holds. What it risks is
/// specific: a forest's thresholds are midpoints between observed values, and a
/// feature that rounds across its threshold sends that voxel down the other
/// branch.
///
/// The measurement is a disagreement rate — how many voxels a forest classifies
/// differently when its features are put through `f32` and back. It is asserted
/// as a ceiling rather than reported, because the number is what decides whether
/// the change is safe and a ceiling is what a regression would break.
///
/// **Why it is small, and why that is not luck.** A threshold sits at the
/// midpoint of two *adjacent* observed values, so the distance from a feature to
/// the nearest threshold it could cross is on the order of the gap between
/// neighbouring samples — around `1/n` of the feature's range for `n` rows.
/// `f32` perturbs by `6e-8` relative. A voxel flips only when it lands within
/// that of a threshold, which is a vanishing fraction of them, and it flips to a
/// class the forest was very nearly going to give it anyway.
#[test]
fn a_forest_is_stable_when_its_features_are_narrowed_to_f32() {
    let train = separable_with(600, 8, 3, 1.5);
    let test = separable_with(2000, 8, 999, 1.5);
    let forest = Forest::train(&train, &TrainingSpec::default()).unwrap();
    let mut scratch = vec![0.0f32; forest.classes()];

    let mut narrowed = Vec::with_capacity(test.columns());
    let mut disagreements = 0usize;
    for row in 0..test.rows() {
        let full = forest.predict(test.row(row), &mut scratch);
        narrowed.clear();
        narrowed.extend(test.row(row).iter().map(|&value| value as f32 as f64));
        if forest.predict(&narrowed, &mut scratch) != full {
            disagreements += 1;
        }
    }
    let rate = disagreements as f64 / test.rows() as f64;
    println!(
        "f32 features: {disagreements} of {} classifications change ({:.4}%)",
        test.rows(),
        100.0 * rate
    );
    assert!(
        rate < 0.005,
        "narrowing the features to f32 changed {:.2}% of classifications, which is more \
         than the rounding argument predicts and would make the memory saving a trade \
         rather than a free one",
        100.0 * rate
    );

    // **The control, and it is what makes a rate of zero mean something.** A test
    // that only ever compares two precisions cannot distinguish "this forest is
    // insensitive to precision" from "this harness cannot see a change at all".
    // So the same forest is fed features quantised coarsely enough that it must
    // visibly disagree.
    //
    // The step is 1/32 and not something finer for a measured reason: at 1/512
    // it moved only 3 classifications of 2000, which is too few to establish
    // anything, and the first version of this control asserted against that and
    // failed. The gap between neighbouring samples on a 600-row fixture is
    // around 1/600 of the range, so a quantisation has to be coarser than that
    // before it starts crossing thresholds in quantity — which is the same
    // arithmetic that explains why `f32`, at 6e-8, crosses none.
    let mut coarse_disagreements = 0usize;
    for row in 0..test.rows() {
        let full = forest.predict(test.row(row), &mut scratch);
        narrowed.clear();
        narrowed.extend(
            test.row(row)
                .iter()
                .map(|&value| (value * 32.0).round() / 32.0),
        );
        if forest.predict(&narrowed, &mut scratch) != full {
            coarse_disagreements += 1;
        }
    }
    println!(
        "control, quantised to 1/32: {coarse_disagreements} of {} change",
        test.rows()
    );
    assert!(
        coarse_disagreements >= 20,
        "quantising to 1/32 changed only {coarse_disagreements} classifications, so this \
         harness cannot see a precision change and the rate above is not evidence"
    );
    assert!(
        coarse_disagreements > 10 * disagreements,
        "f32 and a 1/32 quantisation are indistinguishable here"
    );
}
