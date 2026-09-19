// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **One statement of when an image may be freed.**
//
// The rule — an image is freed after its last reader, if it is `Internal` or
// released and not kept — decides how much memory a run holds, and three
// separate places need to know it: the executor, which does the freeing; the
// planner's residency walk, which prices it; and the simulator, which predicts
// it. It used to be written out in all three, with the middle one describing
// itself as "word for word the executor's rule" — an accurate confession, not a
// reassurance.
//
// Two of the three had already drifted. `Decomposition::images_dead_after`
// answers *after the phase that wrote it* for an image nothing reads; both
// residency walks matched on the reader list and treated "no readers" as *drop
// now*. Those are different rules, and they agreed only because an image enters
// the live set when its writer starts, which made the difference unreachable.
// Agreement resting on an invariant stated in another function is the failure
// mode this test exists to prevent from recurring.
//
// The tests below keep the behavioral cases that made the drift visible:
// unread outputs die after their writer, released inputs are freed after their
// last reader, and `keep_images` wins over release.

use std::collections::BTreeSet;

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::Visibility;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::Dtype;

/// The reconciliation the extraction chose, asserted rather than left to the
/// invariant that used to hide it.
///
/// An image nothing reads dies after the phase that **wrote** it — that is
/// `images_dead_after`'s answer, and now everyone's. The residency walks used to
/// answer *at the first phase boundary*, which agreed only by accident of when
/// an image enters the live set.
#[test]
fn an_unread_image_dies_after_its_writer_and_the_input_needs_releasing() {
    let volume = [8, 8, 8];
    let grid = BlockGrid::new(volume, [4, 4, 4]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    for name in ["first", "second"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [0, 0, 0])))
            .expect("a pixel phase");
    }
    let assembly = builder.finish().expect("an assembly");
    let plan = &assembly.decomposition;

    // Image 2 is the run's output: written by phase 1, read by nobody.
    let last = plan.n_images() - 1;
    assert!(
        plan.readers_of_image(last).is_empty(),
        "the output image is the unread one this test is about"
    );
    assert!(
        plan.images_dead_after(last - 1).contains(&last),
        "an unread image dies after the phase that wrote it, not before"
    );

    // The other half of the `None` arm: image 0 has no writer, so the
    // "dies after the phase that wrote it" clause cannot name it. In any plan
    // that reads its input the reader clause does, at phase 0 — which is right,
    // and is what the executor does. The guard only bites for an image 0 nobody
    // reads, which no plan this crate can build produces; it is there so that
    // the arithmetic `image - 1` never underflows and so that a degenerate plan
    // does not free the run's input at phase 0.
    assert_eq!(
        plan.readers_of_image(0).last().copied(),
        Some(0),
        "phase 0 reads image 0, so image 0 dies after phase 0 like anything else"
    );
    assert!(plan.images_dead_after(0).contains(&0));

    // But being dead is not being freeable: image 0 is `Published`, so it stays
    // until a caller releases it, and then it goes — at the phase the reader
    // clause named, not at some other one.
    let none = BTreeSet::new();
    assert!(
        !plan.images_freed_after(0, &none, &none).contains(&0),
        "the run's input is not freed behind the caller's back"
    );
    let released: BTreeSet<_> = [blockflow::assemble::ImageId::from(0)]
        .into_iter()
        .collect();
    assert!(
        plan.images_freed_after(0, &released, &none).contains(&0),
        "a released input goes after its last reader"
    );
    for phase in 1..plan.n_phases() {
        assert!(
            !plan
                .images_freed_after(phase, &released, &none)
                .contains(&0),
            "and goes once, at that phase, not again later"
        );
    }
}

/// `keep_images` wins over `release_images`, at the one site that now decides
/// it.
#[test]
fn keeping_beats_releasing_at_the_one_site() {
    let volume = [8, 8, 8];
    let grid = BlockGrid::new(volume, [4, 4, 4]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    for name in ["first", "second"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [0, 0, 0])))
            .expect("a pixel phase");
    }
    let assembly = builder.finish().expect("an assembly");
    let plan = &assembly.decomposition;

    // Image 1 is written by phase 0 and read by phase 1: internal, and freeable.
    assert_eq!(plan.image_visibility(1), Visibility::Internal);
    let none = BTreeSet::new();
    assert!(
        plan.image_freeable(1, &none, &none),
        "an internal image goes"
    );

    let kept: BTreeSet<_> = [blockflow::assemble::ImageId::from(1)]
        .into_iter()
        .collect();
    assert!(
        !plan.image_freeable(1, &none, &kept),
        "keep_images holds an internal image"
    );

    let released: BTreeSet<_> = [blockflow::assemble::ImageId::from(1)]
        .into_iter()
        .collect();
    assert!(
        !plan.image_freeable(1, &released, &kept),
        "a caller that named one image in both has contradicted itself, and the reading that \
         cannot lose data is the one taken"
    );
}
