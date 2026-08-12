// SPDX-License-Identifier: MIT
//
// The boundary rule, made mechanical.
//
// `blockflow` is a general image-processing library. Domain knowledge — what
// the images are of, what the right filter size is for them, what the objects
// in them are called — belongs to the application, not here. Two rules follow,
// and they are stated in `README.md`:
//
// 1. **Everything is a parameter.** Filter sizes, sigmas, thresholds, spacings
//    and structuring elements are supplied by the caller. A value that exists
//    only because one particular pipeline needs it is a leak.
// 2. **No domain vocabulary in names or documentation.** And the useful
//    corollary, which is a good filter to apply while writing: *if an op cannot
//    be named without domain terms, it is domain logic and belongs in the
//    application.* Every one of the terms below has a general equivalent that
//    is the honest name anyway — tubeness or vesselness enhancement rather than
//    "tubify", background estimation rather than "vessel background", stripe or
//    illumination correction rather than "lightsheet correction".
//
// Why a test rather than a note in a document. This is exactly the kind of
// boundary that erodes by small increments: nobody ever decides to make the
// library domain-specific, they just write one convenient helper that mentions
// the thing they are working on, and a year later the crate cannot be used for
// anything else. A grep costs milliseconds and fails at the commit that
// introduces the leak, when it is one line to fix. It passes trivially today,
// which is the moment to add it.
//
// What it cannot do. It catches vocabulary, not shape: an op with a
// domain-specific *default* baked into a general-sounding parameter passes this
// test and still violates rule 1. That one needs review, and the grep is not a
// substitute for it — it is the part that can be automated, running so the
// review can spend its attention on the part that cannot.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Terms that must not appear anywhere in the crate, in any case.
///
/// Substrings rather than words, deliberately: `vasculature`, `vascular` and
/// `vasculature_binarize` should all trip on `vascul`. Each entry is chosen to
/// have no innocent English or Rust homograph — which is why this list does not
/// contain `cell` (`RefCell`, `OnceCell`), `plane` (image planes are general),
/// or `artery` as `art` (`quarter`, `start`, `depart`).
const FORBIDDEN: &[&str] = &[
    "vessel",
    "vascul",
    "artery",
    "arterial",
    "venule",
    "capillar",
    "brain",
    "cortex",
    "cortical",
    "neuron",
    "neurit",
    "axon",
    "dendrit",
    "synap",
    "microscop",
    "lightsheet",
    "light-sheet",
    "fluoresc",
    "histolog",
    "anatom",
    "tubify",
    "autofluo",
    "cytoplasm",
    "nuclei",
    "organoid",
];

/// `clearmap` is handled separately from the list above, because it is not
/// domain vocabulary — it is the name of the application this crate was
/// extracted from, and a small number of places have a *duty* to say it.
///
/// Those places are the ones that record provenance and licensing. `lib.rs`
/// states what may and may not move across the boundary and why; `tiling.rs`
/// explains why one predicate was reimplemented rather than taken; `error.rs`
/// and `region.rs` name what they were extracted from. Deleting those mentions
/// would not make the crate more general, it would make the licensing position
/// unverifiable.
///
/// Anywhere else, a mention means somebody has reached back across the boundary,
/// and adding a file to this list should feel like a decision.
const CLEARMAP_MAY_BE_NAMED_IN: &[&str] = &["lib.rs", "tiling.rs", "error.rs", "region.rs"];

/// One frozen exception: the order log's schema identifier.
///
/// It is a wire-format name that cross-language consumers match on exactly, and
/// the format did not change when the crate did. Renaming it would break every
/// reader for no gain — an opaque identifier's job is to be stable, not to be
/// descriptive. Pinned as a whole string so that a *new* mention next to it
/// still fails.
const FROZEN_SCHEMA_ID: &str = "clearmap-rs.block_ops.order_log";

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.join("src"), root.join("tests")];
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

/// This file names every forbidden term in order to forbid it, so it is the one
/// file the scan skips. Nothing else may be added here.
fn is_this_file(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name == "no_domain_vocabulary.rs")
}

#[test]
fn no_source_file_uses_domain_vocabulary() {
    let root = crate_root();
    let sources = rust_sources(&root);
    assert!(
        sources.len() > 10,
        "the scan found only {} files under {}; it is not looking where it thinks it is",
        sources.len(),
        root.display()
    );

    let mut offences: Vec<String> = Vec::new();
    for path in &sources {
        if is_this_file(path) {
            continue;
        }
        let text = fs::read_to_string(path).expect("source file is readable");
        for (number, line) in text.lines().enumerate() {
            let lowered = line.to_lowercase();
            for term in FORBIDDEN {
                if lowered.contains(term) {
                    offences.push(format!(
                        "{}:{}: {term:?} in: {}",
                        path.strip_prefix(&root).unwrap_or(path).display(),
                        number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        offences.is_empty(),
        "blockflow is a general image-processing library and must not name a \
         particular domain. Found:\n{}\n\nIf the code is genuinely \
         domain-specific it belongs in the application crate, as an adapter \
         implementing this crate's traits. If it is general, it has a general \
         name — tubeness enhancement, background estimation, stripe correction.",
        offences.join("\n")
    );
}

#[test]
fn the_application_this_was_extracted_from_is_named_only_where_provenance_requires() {
    let root = crate_root();
    let allowed: BTreeSet<&str> = CLEARMAP_MAY_BE_NAMED_IN.iter().copied().collect();

    let mut offences: Vec<String> = Vec::new();
    for path in rust_sources(&root) {
        if is_this_file(&path) {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if allowed.contains(name) {
            continue;
        }
        let text = fs::read_to_string(&path).expect("source file is readable");
        for (number, line) in text.lines().enumerate() {
            if !line.to_lowercase().contains("clearmap") {
                continue;
            }
            if line.contains(FROZEN_SCHEMA_ID) {
                continue;
            }
            offences.push(format!(
                "{}:{}: {}",
                path.strip_prefix(&root).unwrap_or(&path).display(),
                number + 1,
                line.trim()
            ));
        }
    }

    assert!(
        offences.is_empty(),
        "only the files that record provenance may name the application this \
         crate was extracted from ({}). Found:\n{}\n\nA mention anywhere else \
         means something has reached back across the boundary.",
        CLEARMAP_MAY_BE_NAMED_IN.join(", "),
        offences.join("\n")
    );
}

#[test]
fn the_readme_states_both_rules() {
    // The grep enforces rule 2. Rule 1 — everything is a parameter — cannot be
    // checked mechanically, so the least this test can do is fail if the place
    // a reader would look for it stops saying it.
    let readme = fs::read_to_string(crate_root().join("README.md")).expect("README.md exists");
    let lowered = readme.to_lowercase();
    for phrase in ["everything is a parameter", "no domain vocabulary"] {
        assert!(
            lowered.contains(phrase),
            "README.md no longer states {phrase:?}; the rule the test enforces \
             must stay written down where somebody reading the crate will see it"
        );
    }
}
