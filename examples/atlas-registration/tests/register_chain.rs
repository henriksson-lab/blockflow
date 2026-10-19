// SPDX-License-Identifier: MIT
// Original work for this crate.

// Reading transform parameter files: what a file is allowed to say, how a chain
// is followed on disk, and which file in a directory is *the* transform.
//
// Nothing here runs transformix; `tests/register_transformix.rs` pins the
// arithmetic. This file is about the refusals and the reading around it.

use std::path::{Path, PathBuf};

use blockflow_atlas_registration::{
    transform_file, ElastixTransform, ParameterMap, RegisterParams, Transform,
};

// ------------------------------------------------------------- a scratch dir --

/// A directory under `TMPDIR`, removed when the guard drops.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "atlas-registration-register-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("the scratch directory is created");
        Self { root }
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.root.join(name);
        std::fs::write(&path, text).expect("the file is written");
        path
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// ------------------------------------------------------------- the fixtures --

/// The geometry entries every transform parameter file carries.
const GEOMETRY: &str = "\
(Size 20 24 28)
(Index 0 0 0)
(Spacing 2.0 1.0 1.0)
(Origin 0.5 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
";

fn translation(by: [f64; 3], initial: &str) -> String {
    format!(
        "(Transform \"TranslationTransform\")\n\
         (NumberOfParameters 3)\n\
         (TransformParameters {} {} {})\n\
         (InitialTransformParametersFileName \"{initial}\")\n\
         (HowToCombineTransforms \"Compose\")\n{GEOMETRY}",
        by[0], by[1], by[2]
    )
}

// ------------------------------------------------------------- the refusals --

/// Each of these is a `(Transform "...")` elastix ships and this crate does not
/// evaluate. **Named**, because the whole point of the refusal is that the
/// reader is told which transform was not applied.
const REFUSED: &[&str] = &[
    "SimilarityTransform",
    "MultiBSplineTransformWithNormal",
    "SplineKernelTransform",
    "WeightedCombinationTransform",
    "DeformationFieldTransform",
    "BSplineStackTransform",
    "AffineLogTransform",
    "",
];

#[test]
fn every_unsupported_transform_is_refused_by_name() {
    for keyword in REFUSED {
        let text = format!(
            "(Transform \"{keyword}\")\n(NumberOfParameters 3)\n\
             (TransformParameters 1 2 3)\n\
             (InitialTransformParametersFileName \"NoInitialTransform\")\n{GEOMETRY}"
        );
        let map = ParameterMap::parse(&text).expect("the file parses");
        let error = Transform::from_map(&map).unwrap_err().to_string();
        assert!(
            error.contains(keyword) || keyword.is_empty(),
            "the refusal of `{keyword}` does not name it: {error}"
        );
        assert!(
            error.contains("not supported here"),
            "the refusal of `{keyword}` is not the refusal: {error}"
        );
    }
}

/// The inverted assertion: what an **identity fallback** would have looked like.
///
/// This is the failure being engineered against, written down so that removing
/// the refusal makes a test fail rather than makes a test disappear. If
/// `Transform::from_map` ever returns `Ok` for an unsupported keyword, this
/// says so.
#[test]
fn an_unknown_transform_is_never_silently_the_identity() {
    let text = format!(
        "(Transform \"SimilarityTransform\")\n(NumberOfParameters 7)\n\
         (TransformParameters 0 0 0 1 2 3 1)\n\
         (InitialTransformParametersFileName \"NoInitialTransform\")\n{GEOMETRY}"
    );
    assert!(
        ElastixTransform::from_texts(&[&text]).is_err(),
        "an unsupported transform was accepted, which means some point list will be returned \
         unchanged and called registered"
    );
}

#[test]
fn a_bspline_of_the_wrong_order_is_refused_by_value() {
    let coefficients = vec!["0.0"; 3 * 4 * 4 * 4].join(" ");
    for order in [1usize, 2, 4, 5] {
        let text = format!(
            "(Transform \"BSplineTransform\")\n(NumberOfParameters {})\n\
             (TransformParameters {coefficients})\n\
             (InitialTransformParametersFileName \"NoInitialTransform\")\n\
             (GridSize 4 4 4)\n(GridIndex 0 0 0)\n(GridSpacing 1 1 1)\n\
             (GridOrigin 0 0 0)\n(GridDirection 1 0 0 0 1 0 0 0 1)\n\
             (BSplineTransformSplineOrder {order})\n{GEOMETRY}",
            3 * 4 * 4 * 4
        );
        let error = ElastixTransform::from_texts(&[&text])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("{order}")),
            "order {order}: {error}"
        );
    }
}

#[test]
fn a_coefficient_count_that_does_not_match_the_grid_is_refused() {
    let text = format!(
        "(Transform \"BSplineTransform\")\n(NumberOfParameters 12)\n\
         (TransformParameters 0 0 0 0 0 0 0 0 0 0 0 0)\n\
         (InitialTransformParametersFileName \"NoInitialTransform\")\n\
         (GridSize 4 4 4)\n(GridIndex 0 0 0)\n(GridSpacing 1 1 1)\n\
         (GridOrigin 0 0 0)\n(GridDirection 1 0 0 0 1 0 0 0 1)\n\
         (BSplineTransformSplineOrder 3)\n{GEOMETRY}"
    );
    let error = ElastixTransform::from_texts(&[&text])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("192"),
        "the message names the count it wanted: {error}"
    );
}

#[test]
fn an_unknown_combination_rule_is_refused() {
    let outer = translation([1.0, 0.0, 0.0], "TransformParameters.0.txt")
        .replace("\"Compose\"", "\"Multiply\"");
    let inner = translation([0.0, 1.0, 0.0], "NoInitialTransform");
    let error = ElastixTransform::from_texts(&[&outer, &inner])
        .unwrap_err()
        .to_string();
    assert!(error.contains("Multiply"), "{error}");
}

// ------------------------------------------------------------- the chain --

/// The chain is followed through `InitialTransformParametersFileName` on disk,
/// and the **order comes from the pointers**.
///
/// The files are named so that the pointer order and the numeric order disagree:
/// `TransformParameters.0.txt` names `TransformParameters.7.txt` as its initial.
/// A reader that sorted by name would build the chain the other way round, and
/// the two answers are different because the two translations are.
#[test]
fn the_chain_order_comes_from_the_pointers_and_not_the_names() {
    let scratch = Scratch::new("pointers");
    scratch.write(
        "TransformParameters.7.txt",
        &translation([100.0, 0.0, 0.0], "NoInitialTransform"),
    );
    scratch.write(
        "TransformParameters.0.txt",
        &translation([0.0, 1.0, 0.0], "TransformParameters.7.txt"),
    );
    let transform =
        ElastixTransform::read(scratch.path().join("TransformParameters.0.txt")).expect("reads");
    assert_eq!(transform.chain().len(), 2);
    let params = RegisterParams::default();
    assert_eq!(
        transform.apply([0.0, 0.0, 0.0], &params),
        [100.0, 1.0, 0.0],
        "both links applied"
    );
}

#[test]
fn a_chain_that_points_at_itself_is_refused_rather_than_hanging() {
    let scratch = Scratch::new("cycle");
    let path = scratch.write(
        "TransformParameters.0.txt",
        &translation([1.0, 0.0, 0.0], "TransformParameters.0.txt"),
    );
    let error = ElastixTransform::read(&path).unwrap_err().to_string();
    assert!(error.contains("cycle"), "{error}");
}

#[test]
fn a_missing_initial_transform_file_names_the_path() {
    let scratch = Scratch::new("missing");
    let path = scratch.write(
        "TransformParameters.1.txt",
        &translation([1.0, 0.0, 0.0], "TransformParameters.0.txt"),
    );
    let error = ElastixTransform::read(&path).unwrap_err().to_string();
    assert!(error.contains("TransformParameters.0.txt"), "{error}");
}

// ------------------------------------------------- which file is the transform --

/// The highest-numbered file is the transform, compared as **numbers**: sorted
/// as strings, `TransformParameters.9.txt` would come after
/// `TransformParameters.10.txt`.
#[test]
fn the_highest_numbered_file_is_the_transform_from_ten_files_up() {
    let scratch = Scratch::new("order");
    for number in 0..=10 {
        scratch.write(
            &format!("TransformParameters.{number}.txt"),
            &translation([number as f64, 0.0, 0.0], "NoInitialTransform"),
        );
    }
    // Files that are not transform parameters are not candidates.
    scratch.write("elastix.log", "not a transform");
    scratch.write("TransformParameters.5.log", "not a transform either");

    let chosen = transform_file(scratch.path()).unwrap();
    assert!(chosen.ends_with("TransformParameters.10.txt"), "{chosen:?}");
    let transform = ElastixTransform::from_directory(scratch.path()).expect("reads");
    assert_eq!(
        transform.apply([0.0, 0.0, 0.0], &RegisterParams::default()),
        [10.0, 0.0, 0.0]
    );
}

#[test]
fn an_unnumbered_transform_file_is_refused_by_name() {
    let scratch = Scratch::new("unnumbered");
    scratch.write(
        "TransformParameters.final.txt",
        &translation([1.0, 0.0, 0.0], "NoInitialTransform"),
    );
    let error = transform_file(scratch.path()).unwrap_err().to_string();
    assert!(error.contains("TransformParameters.final.txt"), "{error}");
}

#[test]
fn a_directory_with_no_transform_is_refused() {
    let scratch = Scratch::new("empty");
    scratch.write("elastix.log", "nothing here");
    assert!(transform_file(scratch.path()).is_err());
}

// ------------------------------------------------------------ decomposition --

/// **It decomposes, and it is resident.**
///
/// Every point's image is a function of that point alone, so cutting the list
/// anywhere changes nothing. Asserted rather than assumed because it is the
/// property a parallel caller would rely on, and because it is the reason point
/// transforms have no `blockflow::Assembly`: there is nothing for a lattice to
/// carry.
#[test]
fn splitting_the_point_list_changes_nothing() {
    let scratch = Scratch::new("split");
    scratch.write(
        "TransformParameters.0.txt",
        &translation([1.5, -2.5, 0.25], "NoInitialTransform"),
    );
    let transform =
        ElastixTransform::read(scratch.path().join("TransformParameters.0.txt")).expect("reads");
    let params = RegisterParams::default();

    let mut points = Vec::new();
    let mut state = 12_345u64;
    for _ in 0..1000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let value = |shift: u32| ((state >> shift) % 2000) as f64 / 8.0 - 125.0;
        points.push([value(33), value(21), value(11)]);
    }
    let whole = transform.apply_all(&points, &params);
    for cut in [1usize, 2, 3, 7, 64, 999, 1000] {
        let mut pieces = Vec::new();
        for chunk in points.chunks(cut) {
            pieces.extend(transform.apply_all(chunk, &params));
        }
        assert_eq!(pieces, whole, "cut into chunks of {cut}");
    }
}
