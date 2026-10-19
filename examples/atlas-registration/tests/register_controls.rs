// SPDX-License-Identifier: MIT
// Original work for this crate.
//
// The negative controls for the point transforms.
//
// A coordinate transform is the case where a positive result proves least: every
// one of the four mistakes below returns three finite numbers per point, in the
// right order, in the right range, and looking exactly like coordinates. Nothing
// downstream can tell. So each is built here — **the same program with one thing
// changed** — and each is measured: it must reproduce every count and move the
// answer.
//
// | control | the one thing changed | what it moves |
// |---|---|---|
// | [`the_transform_inverted`] | the affine replaced by its own inverse | 1000 of 1000 points, mean displacement **25.41** |
// | [`the_matrix_axes_permuted`] | `(TransformParameters ...)` read column-major instead of row-major | 1000 of 1000, mean **5.14**, and the two agree exactly where the matrix is symmetric |
// | [`the_index_physical_round_trip_dropped`] | index mode with the geometry replaced by the unit one | 1000 of 1000, mean **14.04** |
// | [`the_chain_applied_in_the_wrong_order`] | the two links of a chain swapped | 1000 of 1000, mean **5.1** and it is the *same* 5.1 for every point |
//
// Each control is constructed from the **public API and a different parameter
// file**, never by editing the module: a control that needed a hook in the code
// would be testing the hook.
//
// The fifth, which is not one of the four
// ----------------------------------------
// [`a_transform_that_is_not_applied_at_all`] is the identity control, and it is
// the one the module's refusals exist for — 1000 of 1000, mean **11.12**. It is
// here because the other four all assume the transform ran, and because it is the
// scale every other number on this fixture should be read against: the four
// mistakes above are between a third and twice as large as *not registering at
// all*, which is what says none of them would be noticed by looking at the
// output.

use blockflow_atlas_registration::{
    ElastixTransform, ImageGeometry, ParameterMap, PointSpace, RegisterParams, Transform,
};

// ------------------------------------------------------------- the fixtures --

const GEOMETRY: &str = "\
(Size 64 64 64)
(Index 0 0 0)
(Spacing 2.0 3.0 5.0)
(Origin -10.0 7.0 0.5)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
";

/// The affine the recordings use, so that the control moves the same numbers the
/// positive tests pin.
const MATRIX: [[f64; 3]; 3] = [[1.1, 0.2, -0.3], [0.05, 0.9, 0.15], [-0.25, 0.1, 1.05]];
const TRANSLATION: [f64; 3] = [3.5, -2.25, 7.125];
const CENTRE: [f64; 3] = [4.0, -6.0, 11.0];

fn affine_text(matrix: [[f64; 3]; 3], translation: [f64; 3], initial: &str) -> String {
    let mut values = Vec::new();
    for row in matrix {
        for value in row {
            values.push(format!("{value}"));
        }
    }
    for value in translation {
        values.push(format!("{value}"));
    }
    format!(
        "(Transform \"AffineTransform\")\n\
         (NumberOfParameters 12)\n\
         (TransformParameters {})\n\
         (InitialTransformParametersFileName \"{initial}\")\n\
         (HowToCombineTransforms \"Compose\")\n\
         (CenterOfRotationPoint {} {} {})\n{GEOMETRY}",
        values.join(" "),
        CENTRE[0],
        CENTRE[1],
        CENTRE[2]
    )
}

fn translation_text(by: [f64; 3], initial: &str) -> String {
    format!(
        "(Transform \"TranslationTransform\")\n\
         (NumberOfParameters 3)\n\
         (TransformParameters {} {} {})\n\
         (InitialTransformParametersFileName \"{initial}\")\n\
         (HowToCombineTransforms \"Compose\")\n{GEOMETRY}",
        by[0], by[1], by[2]
    )
}

/// A thousand points spread over a volume the size of a resampled atlas-scale
/// stack, from a fixed generator so the counts below are reproducible.
fn points() -> Vec<[f64; 3]> {
    let mut state = 987_654_321u64;
    (0..1000)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let value = |shift: u32| ((state >> shift) % 480) as f64 / 10.0;
            [value(33), value(21), value(11)]
        })
        .collect()
}

fn truth() -> ElastixTransform {
    ElastixTransform::from_texts(&[&affine_text(MATRIX, TRANSLATION, "NoInitialTransform")])
        .expect("the reference transform parses")
}

/// How many of the two answers differ, and by how much on average.
///
/// Both numbers, because "it moved" is the weak half: a control that moved every
/// point by `1e-12` would satisfy an inequality and mean nothing.
fn compare(ours: &[[f64; 3]], theirs: &[[f64; 3]]) -> (usize, f64) {
    assert_eq!(
        ours.len(),
        theirs.len(),
        "a control changed the point count"
    );
    let mut moved = 0usize;
    let mut total = 0.0f64;
    for (left, right) in ours.iter().zip(theirs) {
        if left != right {
            moved += 1;
        }
        let mut squared = 0.0;
        for axis in 0..3 {
            let difference = left[axis] - right[axis];
            squared += difference * difference;
        }
        total += squared.sqrt();
    }
    (moved, total / ours.len() as f64)
}

fn rounded(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

// ------------------------------------------------------------ the controls --

/// **The transform inverted.** The registration applied backwards.
///
/// The inverse affine is written into a parameter file and evaluated by the same
/// evaluator, so the only thing that changed is the transform. Every point moves,
/// and the answer is still a plausible coordinate everywhere — which is the
/// reason this control exists rather than an argument that it is unnecessary.
#[test]
fn the_transform_inverted() {
    // y = A(x - c) + t + c  =>  x = A^-1(y - t - c) + c, which is the affine
    // with matrix A^-1, translation -A^-1 t and the same centre.
    let inverse = invert(MATRIX);
    let inverse_translation = [
        -(inverse[0][0] * TRANSLATION[0]
            + inverse[0][1] * TRANSLATION[1]
            + inverse[0][2] * TRANSLATION[2]),
        -(inverse[1][0] * TRANSLATION[0]
            + inverse[1][1] * TRANSLATION[1]
            + inverse[1][2] * TRANSLATION[2]),
        -(inverse[2][0] * TRANSLATION[0]
            + inverse[2][1] * TRANSLATION[1]
            + inverse[2][2] * TRANSLATION[2]),
    ];
    let control = ElastixTransform::from_texts(&[&affine_text(
        inverse,
        inverse_translation,
        "NoInitialTransform",
    )])
    .expect("the inverted transform parses");
    let params = RegisterParams::default();
    let points = points();
    let forward = truth().apply_all(&points, &params);
    let backward = control.apply_all(&points, &params);

    // The control is a real inverse: composing the two returns the input.
    for (point, image) in points.iter().zip(&forward) {
        let back = control.apply(*image, &params);
        for axis in 0..3 {
            assert!((back[axis] - point[axis]).abs() < 1e-9, "not the inverse");
        }
    }
    let (moved, mean) = compare(&forward, &backward);
    assert_eq!(moved, 1000, "every point must move");
    assert_eq!(rounded(mean), 25.41, "the mean displacement");
    // And the wrong answer is still in range: nothing downstream could reject it.
    for point in &backward {
        assert!(point.iter().all(|value| value.is_finite()));
    }
}

/// **The axes permuted.** `(TransformParameters ...)` read column-major instead
/// of row-major, which is the transpose of the linear part — a permutation of
/// how the three input axes reach the three output axes.
///
/// It is the mistake with the highest prior in a reader of this format, because
/// `(Direction ...)` in the *same file* really is column-major, and one reading
/// applied to both is the natural error.
#[test]
fn the_matrix_axes_permuted() {
    let mut transposed = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            transposed[row][column] = MATRIX[column][row];
        }
    }
    let control = ElastixTransform::from_texts(&[&affine_text(
        transposed,
        TRANSLATION,
        "NoInitialTransform",
    )])
    .expect("the transposed transform parses");
    let params = RegisterParams::default();
    let points = points();
    let (moved, mean) = compare(
        &truth().apply_all(&points, &params),
        &control.apply_all(&points, &params),
    );
    assert_eq!(moved, 1000);
    assert_eq!(rounded(mean), 5.14, "the mean displacement");

    // The liveness of the control: with a **symmetric** matrix the two readings
    // are the same transform and nothing moves. So the control detects the
    // reading and not merely a difference between two files.
    let symmetric = [[1.1, 0.2, -0.3], [0.2, 0.9, 0.15], [-0.3, 0.15, 1.05]];
    let straight =
        ElastixTransform::from_texts(&[&affine_text(symmetric, TRANSLATION, "NoInitialTransform")])
            .expect("parses");
    let mut flipped = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            flipped[row][column] = symmetric[column][row];
        }
    }
    let mirrored =
        ElastixTransform::from_texts(&[&affine_text(flipped, TRANSLATION, "NoInitialTransform")])
            .expect("parses");
    assert_eq!(
        straight.apply_all(&points, &params),
        mirrored.apply_all(&points, &params),
        "a symmetric matrix reads the same both ways"
    );
}

/// **The index/physical round trip dropped.** Index mode with the file's
/// geometry replaced by the unit one — which is what an implementation does
/// when it treats a voxel index as a physical coordinate.
///
/// That shortcut is free only for volumes with unit spacing and zero origin,
/// where the two spaces are the same numbers. This control measures what it
/// costs the moment they are not, and the answer is a **scale error**, not an
/// offset — which is why it survives every sanity check downstream.
#[test]
fn the_index_physical_round_trip_dropped() {
    let transform =
        ElastixTransform::from_texts(&[&affine_text(MATRIX, TRANSLATION, "NoInitialTransform")])
            .expect("parses");
    let control = transform.clone().with_geometry(ImageGeometry::unit());
    let params = RegisterParams::in_space(PointSpace::Index);
    let points = points();
    let (moved, mean) = compare(
        &transform.apply_all(&points, &params),
        &control.apply_all(&points, &params),
    );
    assert_eq!(moved, 1000);
    assert_eq!(rounded(mean), 14.04, "the mean displacement");

    // And the inverted assertion: with the unit geometry on *both* sides the
    // shortcut is free. `Spacing 1 1 1`, `Origin 0 0 0`.
    let unit_file = affine_text(MATRIX, TRANSLATION, "NoInitialTransform")
        .replace("(Spacing 2.0 3.0 5.0)", "(Spacing 1.0 1.0 1.0)")
        .replace("(Origin -10.0 7.0 0.5)", "(Origin 0.0 0.0 0.0)");
    let unit = ElastixTransform::from_texts(&[&unit_file]).expect("parses");
    assert_eq!(
        unit.apply_all(&points, &params),
        unit.clone()
            .with_geometry(ImageGeometry::unit())
            .apply_all(&points, &params),
        "with a unit geometry the round trip is the identity and dropping it is free"
    );
}

/// **The chain applied in the wrong order.** The same two transforms, composed
/// the other way round.
///
/// Built by swapping which file names the other as its initial, so both chains
/// are two links of the same two transforms and the only difference is the order.
/// An affine and a translation do not commute unless the affine's linear part is
/// the identity, and the control asserts that too — otherwise it would be a test
/// of the fixture rather than of the fold.
#[test]
fn the_chain_applied_in_the_wrong_order() {
    let shift = [10.0, 20.0, 30.0];
    let right = ElastixTransform::from_texts(&[
        &affine_text(MATRIX, TRANSLATION, "TransformParameters.0.txt"),
        &translation_text(shift, "NoInitialTransform"),
    ])
    .expect("parses");
    let wrong = ElastixTransform::from_texts(&[
        &translation_text(shift, "TransformParameters.0.txt"),
        &affine_text(MATRIX, TRANSLATION, "NoInitialTransform"),
    ])
    .expect("parses");
    assert_eq!(right.chain().len(), 2);
    assert_eq!(wrong.chain().len(), 2);
    let params = RegisterParams::default();
    let points = points();
    let (moved, mean) = compare(
        &right.apply_all(&points, &params),
        &wrong.apply_all(&points, &params),
    );
    assert_eq!(moved, 1000);
    // Exactly `|A s - s|`: the difference between the two orders is the constant
    // vector `A s - s = (-4, 3, 1)`, so every point moves by the same 5.1. That it
    // is a **constant offset** rather than a spread is the point — a wrong chain
    // order on a rigid pair is invisible to any check that looks at the shape of
    // the point cloud.
    assert_eq!(rounded(mean), 5.1, "the mean displacement");

    // The liveness: with the identity in place of the affine the two orders are
    // the same transform, so the control is detecting the order and not the pair.
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let one = ElastixTransform::from_texts(&[
        &affine_text(identity, TRANSLATION, "TransformParameters.0.txt"),
        &translation_text(shift, "NoInitialTransform"),
    ])
    .expect("parses");
    let other = ElastixTransform::from_texts(&[
        &translation_text(shift, "TransformParameters.0.txt"),
        &affine_text(identity, TRANSLATION, "NoInitialTransform"),
    ])
    .expect("parses");
    let (bitwise, mean) = compare(
        &one.apply_all(&points, &params),
        &other.apply_all(&points, &params),
    );
    // Two translations commute, so the control finds **nothing to measure**: the
    // mean displacement is 8.2e-16, five thousand million times smaller than the
    // 5.1 above and at the scale of one rounding of a number near 60.
    assert!(
        mean < 1e-14,
        "the order is not readable off two translations: {mean:e}"
    );
    // And a fact worth keeping rather than tolerancing away: they are not
    // bit-identical. `x + t + s` and `x + s + t` are different roundings of the
    // same real number, and 155 of the 1000 rows differ in their last bits. It is
    // why the control above compares displacements and not equality, and it is the
    // reason a future caller must not expect chain reassociation to be free.
    assert_eq!(bitwise, 155, "float addition is not associative");
}

/// **The transform not applied at all.** The control the refusals exist for.
///
/// A reader that treated an unknown `(Transform "...")` as the identity would
/// return this. It is the largest of the five displacements and the only one a
/// reader might mistake for "registration is off", so the number is here beside
/// the others.
#[test]
fn a_transform_that_is_not_applied_at_all() {
    let params = RegisterParams::default();
    let points = points();
    let (moved, mean) = compare(&truth().apply_all(&points, &params), &points);
    assert_eq!(moved, 1000);
    assert_eq!(rounded(mean), 11.12, "the mean displacement");
}

/// Invert a 3x3 matrix by cofactors, for the inversion control. Independent of
/// the module, which does not expose one.
fn invert(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut inverse = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let (r0, r1) = ((column + 1) % 3, (column + 2) % 3);
            let (c0, c1) = ((row + 1) % 3, (row + 2) % 3);
            inverse[row][column] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / determinant;
        }
    }
    inverse
}

/// The refusals are not silent, and a caller cannot get an identity out of one.
///
/// The inverted assertion for the whole file: every control above assumes the
/// transform ran. This is the assertion that says a transform which cannot run
/// does not quietly become one that did.
#[test]
fn a_refusal_is_never_an_identity() {
    let text = format!(
        "(Transform \"WeightedCombinationTransform\")\n(NumberOfParameters 3)\n\
         (TransformParameters 1 1 1)\n\
         (InitialTransformParametersFileName \"NoInitialTransform\")\n{GEOMETRY}"
    );
    let map = ParameterMap::parse(&text).expect("parses");
    assert!(Transform::from_map(&map).is_err());
    assert!(ElastixTransform::from_texts(&[&text]).is_err());
}
