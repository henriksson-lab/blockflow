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
// **What is checked, and why a grep.** That the predicate has one home. It is a
// two-line function, so nothing stops the next caller from writing it out again
// — and a caller who does will produce something that looks right, passes every
// test, and disagrees in a corner. A grep costs milliseconds and fails at the
// commit that introduces the second copy, when it is one line to fix. It cannot
// check that the rule is *correct*; `tests/image_lifetime.rs` and
// `tests/peak_image_bytes.rs` do that.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::Visibility;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::Dtype;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// The one function allowed to compare a visibility against `Internal` in order
/// to decide whether an image may go.
///
/// `zarr_env` is the other legitimate mention and is **not** an exemption in the
/// same sense: it matches on `Visibility` to decide where an array is *stored*,
/// which is a different question with a different answer, and it never asks
/// about freeing. It is listed by file so that a freeing decision appearing
/// there in future is still caught.
const ALLOWED: &[(&str, &str)] = &[
    (
        "decomposition.rs",
        "`image_freeable` is the rule; every other caller asks it",
    ),
    (
        "zarr_env.rs",
        "decides where an array is stored, not whether it may be freed",
    ),
];

#[test]
fn the_freeing_rule_has_exactly_one_home() {
    let root = crate_root();
    let mut offences = Vec::new();
    for path in rust_sources(&root) {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        if ALLOWED.iter().any(|&(allowed, _)| allowed == name) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        for (number, line) in text.lines().enumerate() {
            // The doc-comment mentions are the ones that *explain* the rule, and
            // explaining it is what a comment is for. Code is what drifts.
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            if code.contains("Visibility::Internal") {
                offences.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(&root).unwrap_or(&path).display(),
                    number + 1,
                    code.trim()
                ));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "a second copy of the freeing rule has appeared:\n  {}\n\n\
         Whether an image may be freed is `Decomposition::image_freeable`, and when it may be \
         is `Decomposition::images_freed_after`. Call one of those. The predicate is two lines, \
         which is exactly why it gets rewritten and exactly why the copies disagree in a corner \
         instead of failing a test.",
        offences.join("\n  ")
    );
}

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

// ------------------------------------------- every sidecar states its size --

/// **No shipped fragment stream leaves its size undeclared.**
///
/// `FragmentOutput::size` is an upper bound on what one block writes, and it is
/// what a residency figure for a barrier gather has to be built on: the gather
/// holds every contributing block's fragment at once, and under
/// `Coverage::EveryBlock` that is `n_blocks x payload` resident at one instant.
/// `SidecarSize::Unstated` is the honest rendering of "nobody said", and a
/// budget built on it would be a budget built on zero — so the rule is that no
/// shipped stream may be in that state.
///
/// **Why a grep rather than a runtime check.** Reaching every shipped
/// `FragmentOp` at run time means constructing every one of them, which is a
/// second inventory to keep in step with the first. The declaration is a literal
/// beside a literal, so a grep sees all of them, costs milliseconds, and fails
/// at the commit that adds an undeclared stream rather than at the run that
/// needed the number. What it cannot check is whether a stated bound is *true*;
/// that is checked where the bytes are, in `strategy`'s fragment write path.
///
/// Test modules are excluded, on the same rule `src/distributed/tests.rs` uses:
/// a fixture that writes an unbounded blob is testing something, and a bound on
/// it would be a bound on the test.
#[test]
fn every_shipped_fragment_stream_declares_a_size() {
    let root = crate_root();
    let mut unstated = Vec::new();
    for path in rust_sources(&root) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let shipped = match text.find("#[cfg(test)]") {
            Some(at) => &text[..at],
            None => text.as_str(),
        };
        let lines: Vec<&str> = shipped.lines().collect();
        for (number, line) in lines.iter().enumerate() {
            if !line.contains("FragmentOutput::new(") {
                continue;
            }
            // The declaration is `FragmentOutput::new(..).sized(..)`, and the
            // two may be several lines apart because the arguments carry the
            // comments that explain them.
            let window = lines[number..lines.len().min(number + 40)].join("\n");
            let ends = window.find(")]").unwrap_or(window.len());
            if !window[..ends].contains(".sized(") {
                unstated.push(format!(
                    "{}:{}",
                    path.strip_prefix(&root).unwrap_or(&path).display(),
                    number + 1
                ));
            }
        }
    }
    assert!(
        unstated.is_empty(),
        "these fragment streams declare no size:\n  {}\n\n\
         Say what one block writes at most with `FragmentOutput::sized`. \
         `SidecarSize::row_table` covers the row-table shape and takes its header from the \
         schema; `SidecarSize::block_faces` covers the six-faces shape. If the payload has no \
         ceiling that is worth stating, `PerItem` with the one-item-per-voxel bound is the \
         honest answer — it refuses nothing, and saying so is the point.",
        unstated.join("\n  ")
    );
}
