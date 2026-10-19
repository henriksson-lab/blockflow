// SPDX-License-Identifier: MIT
// Original work for this crate.

// The transforms themselves: what one `TransformParameters.N.txt` evaluates to
// at a point, and how a chain of them composes.
//
// Everything in this file was **measured against the transformix binary**, not
// read off a manual. `tests/register_transformix.rs` holds the recordings and
// the exact commands that produced them; the numbers quoted in the doc comments
// below are from those runs. Where a convention had two plausible readings, the
// probe was built to separate them — an asymmetric direction matrix rather than
// a symmetric one, anisotropic spacing rather than isotropic, a control point
// whose neighbours are all different rather than a constant grid.
//
// The transforms that are here, and the ones that are refused
// -----------------------------------------------------------
// | `(Transform "...")` | here | why |
// |---|---|---|
// | `TranslationTransform` | yes | 3 parameters, no centre |
// | `EulerTransform` | yes | 6 parameters, `ComputeZYX`, `CenterOfRotationPoint` |
// | `AffineTransform` | yes | 12 parameters, `CenterOfRotationPoint` |
// | `BSplineTransform` | yes | order 3 only |
// | `RecursiveBSplineTransform` | yes | the same evaluation — **measured**, see below |
// | everything else | **refused by name** | see [`Transform::from_map`] |
//
// A parser that silently treated an unknown transform as the identity would
// return plausible coordinates for a registration it had not applied. transformix
// itself refuses: `(Transform "NoSuchTransform")` gives
// `ERROR: error occurred while creating Transform 0. … The following component
// could not be created: NoSuchTransform`. This does the same, and the refusal
// carries the keyword.
//
// `SimilarityTransform`, `MultiBSplineTransformWithNormal`,
// `SplineKernelTransform`, `WeightedCombinationTransform`,
// `DeformationFieldTransform` and `BSplineStackTransform` are the named
// refusals: elastix ships all of them and none is evaluated here.

use blockflow::error::{Error, Result};

use crate::parameters::ParameterMap;

/// The value elastix writes for "this transform has no predecessor".
pub const NO_INITIAL_TRANSFORM: &str = "NoInitialTransform";

/// The cubic B-spline basis, on `[-2, 2]`.
///
/// Order 3 only, and [`BSpline::from_map`] refuses any other
/// `BSplineTransformSplineOrder` rather than falling back to this one.
fn cubic_basis(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        2.0 / 3.0 - t * t + t * t * t / 2.0
    } else if t < 2.0 {
        let u = 2.0 - t;
        u * u * u / 6.0
    } else {
        0.0
    }
}

/// A 3x3 matrix times a vector.
fn mul(matrix: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for row in 0..3 {
        out[row] =
            matrix[row][0] * vector[0] + matrix[row][1] * vector[1] + matrix[row][2] * vector[2];
    }
    out
}

/// Read nine numbers as a matrix, **column-major**.
///
/// This is the reading `(Direction ...)` and `(GridDirection ...)` need, and it
/// was settled by measurement rather than by convention: with
/// `(Direction 0 1 0 -1 0 0 0 0 1)`, `(Spacing 2 3 5)`,
/// `(Origin -10 7 0.5)`, transformix reports index `(1, 2, 3)` as
/// `InputPoint = (-16, 9, 15.5)`. Filling the matrix row-major gives
/// `(-4, 5, 15.5)`; filling it column-major gives `(-16, 9, 15.5)`. The probe
/// matrix is asymmetric precisely so the two readings cannot both be right.
fn matrix_column_major(values: &[f64]) -> [[f64; 3]; 3] {
    let mut matrix = [[0.0; 3]; 3];
    for column in 0..3 {
        for row in 0..3 {
            matrix[row][column] = values[column * 3 + row];
        }
    }
    matrix
}

/// Read nine numbers as a matrix, **row-major**.
///
/// `(TransformParameters ...)` of an affine is filled this way — the first three
/// numbers are the first *row*. Measured: with matrix
/// `1.1 0.2 -0.3 / 0.05 0.9 0.15 / -0.25 0.1 1.05`, translation
/// `(3.5, -2.25, 7.125)` and centre `(4, -6, 11)`, transformix maps the origin
/// to `(7.6, -4.7, 8.175)`, which is `A(x - c) + t + c` with `A` filled by rows.
fn matrix_row_major(values: &[f64]) -> [[f64; 3]; 3] {
    let mut matrix = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            matrix[row][column] = values[row * 3 + column];
        }
    }
    matrix
}

/// Invert a 3x3 matrix by cofactors.
fn invert(matrix: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let m = matrix;
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if determinant == 0.0 || !determinant.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "elastix transform: the direction matrix {matrix:?} has determinant {determinant} and \
             cannot be inverted, so there is no index-to-physical map"
        )));
    }
    let mut inverse = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let (r0, r1) = ((column + 1) % 3, (column + 2) % 3);
            let (c0, c1) = ((row + 1) % 3, (row + 2) % 3);
            inverse[row][column] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / determinant;
        }
    }
    Ok(inverse)
}

// ------------------------------------------------------------ the B-spline --

/// A cubic B-spline free-form deformation over a control point grid.
///
/// The coefficient layout, the grid index map, the support and the region
/// outside which the transform is the identity were each measured. The probe
/// that settled the layout set exactly one control point's three coefficients to
/// `(1, 2, 3)` and read the displacement back at the grid node: transformix
/// reports `(0.296296, 0.592593, 0.888889)`, which is `(1, 2, 3) * (2/3)^3`. The
/// probe that settled the support set the six coefficients along one grid row to
/// `1..=6` and swept the row.
#[derive(Debug, Clone, PartialEq)]
pub struct BSpline {
    size: [usize; 3],
    index: [isize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    /// The inverse of `GridDirection`, which is what the point map needs.
    inverse_direction: [[f64; 3]; 3],
    /// Three blocks of `size.iter().product()` coefficients, x block first.
    coefficients: Vec<f64>,
}

impl BSpline {
    /// The number of control points.
    pub fn control_points(&self) -> usize {
        self.size[0] * self.size[1] * self.size[2]
    }

    pub fn size(&self) -> [usize; 3] {
        self.size
    }

    /// A point's continuous index in the control point grid.
    ///
    /// `u = D^-1 (p - GridOrigin) / GridSpacing`, with the division **after** the
    /// direction, which is ITK's `TransformPhysicalPointToContinuousIndex`. The
    /// order matters as soon as the spacing is anisotropic, and the affine probe
    /// that pinned it used `(Spacing 2 3 5)` for that reason.
    pub fn continuous_index(&self, point: [f64; 3]) -> [f64; 3] {
        let shifted = [
            point[0] - self.origin[0],
            point[1] - self.origin[1],
            point[2] - self.origin[2],
        ];
        let rotated = mul(&self.inverse_direction, shifted);
        [
            rotated[0] / self.spacing[0],
            rotated[1] / self.spacing[1],
            rotated[2] / self.spacing[2],
        ]
    }

    /// Whether a continuous grid index is inside the region the deformation is
    /// defined on.
    ///
    /// For spline order 3 the measured region is
    /// `[GridIndex + 1, GridIndex + GridSize - 2)`, half open. The sweep that
    /// established it used `GridSize 6`, `GridSpacing 10`, `GridOrigin 0`: the
    /// displacement is zero at `x = 9.999`, nonzero from `x = 10` through
    /// `x = 39.999`, and zero again at `x = 40`. Repeating it with
    /// `(GridIndex 1 1 1)` moved both ends by exactly one grid step, which is
    /// what says the bound is relative to `GridIndex` and not to zero.
    ///
    /// Outside, the transform is the **identity** — the point is returned
    /// unchanged rather than extrapolated. That is ITK's behaviour and it is the
    /// off-grid case an implementation gets wrong by clamping the control point
    /// index instead, which produces a plausible displacement everywhere.
    pub fn inside(&self, u: [f64; 3]) -> bool {
        (0..3).all(|axis| {
            let low = self.index[axis] as f64 + 1.0;
            let high = self.index[axis] as f64 + self.size[axis] as f64 - 2.0;
            u[axis] >= low && u[axis] < high
        })
    }

    /// The displacement at a point, or zero outside the valid region.
    pub fn displacement(&self, point: [f64; 3]) -> [f64; 3] {
        let u = self.continuous_index(point);
        if !self.inside(u) {
            return [0.0; 3];
        }
        // The support of the order-3 basis is four consecutive control points per
        // axis, starting at `floor(u) - 1`. `inside` has already guaranteed every
        // one of them is a grid node, so no clamping happens here — and must not:
        // clamping is the wrong answer that looks right.
        let mut starts = [0isize; 3];
        let mut weights = [[0.0f64; 4]; 3];
        for axis in 0..3 {
            let start = u[axis].floor() as isize - 1;
            starts[axis] = start;
            for offset in 0..4 {
                weights[axis][offset] = cubic_basis(u[axis] - (start + offset as isize) as f64);
            }
        }
        let per_block = self.control_points();
        let mut displacement = [0.0f64; 3];
        for k in 0..4 {
            let gk = starts[2] + k as isize - self.index[2];
            let wk = weights[2][k];
            for j in 0..4 {
                let gj = starts[1] + j as isize - self.index[1];
                let wjk = wk * weights[1][j];
                for i in 0..4 {
                    let gi = starts[0] + i as isize - self.index[0];
                    let weight = wjk * weights[0][i];
                    if weight == 0.0 {
                        continue;
                    }
                    let flat =
                        gi as usize + self.size[0] * (gj as usize + self.size[1] * gk as usize);
                    for axis in 0..3 {
                        displacement[axis] += weight * self.coefficients[axis * per_block + flat];
                    }
                }
            }
        }
        displacement
    }

    /// Read a `BSplineTransform` from a parameter map.
    ///
    /// **Order 3 only.** `(BSplineTransformSplineOrder n)` for any other `n` is
    /// refused by value: the basis, the support width and the valid region all
    /// change with it, and an order-3 evaluation of an order-1 transform is a
    /// wrong answer rather than an approximate one.
    pub fn from_map(map: &ParameterMap) -> Result<Self> {
        let order = match map.values("BSplineTransformSplineOrder") {
            // elastix omits the entry for its own default, which is 3.
            None => 3,
            Some(_) => map.usize("BSplineTransformSplineOrder")?,
        };
        if order != 3 {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: `(BSplineTransformSplineOrder {order})`. Only the cubic basis \
                 is supported here; refusing rather than evaluating an order-{order} transform \
                 with order-3 weights."
            )));
        }
        if map.flag("UseCyclicTransform", false)? {
            return Err(Error::InvalidArgument(
                "elastix transform: `(UseCyclicTransform \"true\")` wraps the control point grid \
                 and is not supported here."
                    .to_string(),
            ));
        }
        let size = map.usizes("GridSize")?;
        if size.len() != 3 {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: `(GridSize ...)` has {} values and this reader is 3D only",
                size.len()
            )));
        }
        let size = [size[0], size[1], size[2]];
        if size.iter().any(|extent| *extent < 4) {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: `(GridSize {size:?})` — the order-3 support is four control \
                 points per axis, so a grid narrower than that has an empty valid region"
            )));
        }
        let index = map.isizes("GridIndex")?;
        if index.len() != 3 {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: `(GridIndex ...)` has {} values and three were expected",
                index.len()
            )));
        }
        let direction = map.floats_n("GridDirection", 9)?;
        let coefficients = map.floats("TransformParameters")?;
        let expected = 3 * size[0] * size[1] * size[2];
        if coefficients.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: `(GridSize {size:?})` needs {expected} coefficients and \
                 `(TransformParameters ...)` has {}",
                coefficients.len()
            )));
        }
        Ok(Self {
            size,
            index: [index[0], index[1], index[2]],
            origin: map.triple("GridOrigin")?,
            spacing: map.triple("GridSpacing")?,
            inverse_direction: invert(&matrix_column_major(&direction))?,
            coefficients,
        })
    }
}

// ------------------------------------------------------------- transforms --

/// One elastix transform, evaluated at a point.
#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    /// `(Transform "TranslationTransform")`: `y = x + t`.
    Translation { translation: [f64; 3] },
    /// `(Transform "EulerTransform")` and `(Transform "AffineTransform")`, which
    /// differ only in how the matrix is built.
    ///
    /// `y = A (x - c) + t + c`, with `c` the `CenterOfRotationPoint`. Both are
    /// held as the matrix rather than as the angles, because the angles are only
    /// a way of writing the matrix down and keeping them would leave two places
    /// for the same fact.
    Matrix {
        matrix: [[f64; 3]; 3],
        translation: [f64; 3],
        centre: [f64; 3],
    },
    /// `(Transform "BSplineTransform")` and `(Transform
    /// "RecursiveBSplineTransform")`: `y = x + d(x)`.
    ///
    /// The two keywords are one evaluation, and that was **measured** rather than
    /// assumed: the same 648-coefficient grid under both keywords produces byte
    /// identical `outputpoints.txt` from transformix. `Recursive` names how
    /// elastix computes the weights, not what it computes.
    BSpline(BSpline),
}

impl Transform {
    /// Apply the transform to one point, in **physical** coordinates.
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        match self {
            Self::Translation { translation } => [
                point[0] + translation[0],
                point[1] + translation[1],
                point[2] + translation[2],
            ],
            Self::Matrix {
                matrix,
                translation,
                centre,
            } => {
                let centred = [
                    point[0] - centre[0],
                    point[1] - centre[1],
                    point[2] - centre[2],
                ];
                let rotated = mul(matrix, centred);
                [
                    rotated[0] + translation[0] + centre[0],
                    rotated[1] + translation[1] + centre[1],
                    rotated[2] + translation[2] + centre[2],
                ]
            }
            Self::BSpline(spline) => {
                let displacement = spline.displacement(point);
                [
                    point[0] + displacement[0],
                    point[1] + displacement[1],
                    point[2] + displacement[2],
                ]
            }
        }
    }

    /// The `(Transform "...")` keyword this was built from.
    pub fn keyword(&self) -> &'static str {
        match self {
            Self::Translation { .. } => "TranslationTransform",
            Self::Matrix { .. } => "AffineTransform or EulerTransform",
            Self::BSpline(_) => "BSplineTransform",
        }
    }

    /// Build a transform from one parsed parameter file.
    ///
    /// **Refuses an unknown keyword by name.** The alternative — treating what it
    /// does not recognise as the identity — is the failure mode this whole module
    /// is arranged against: the coordinates come back looking exactly like
    /// coordinates, and the registration silently did not happen.
    pub fn from_map(map: &ParameterMap) -> Result<Self> {
        let keyword = map.string("Transform")?;
        let count = map.usize("NumberOfParameters")?;
        let parameters = map.floats("TransformParameters")?;
        if parameters.len() != count {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: `(NumberOfParameters {count})` but `(TransformParameters ...)` \
                 has {} values",
                parameters.len()
            )));
        }
        match keyword {
            "TranslationTransform" => {
                if count != 3 {
                    return Err(Error::InvalidArgument(format!(
                        "elastix transform: a 3D `TranslationTransform` has 3 parameters, not \
                         {count}"
                    )));
                }
                Ok(Self::Translation {
                    translation: [parameters[0], parameters[1], parameters[2]],
                })
            }
            "EulerTransform" => {
                if count != 6 {
                    return Err(Error::InvalidArgument(format!(
                        "elastix transform: a 3D `EulerTransform` has 6 parameters, not {count}"
                    )));
                }
                let compute_zyx = map.flag("ComputeZYX", false)?;
                let matrix = euler_matrix(parameters[0], parameters[1], parameters[2], compute_zyx);
                Ok(Self::Matrix {
                    matrix,
                    translation: [parameters[3], parameters[4], parameters[5]],
                    centre: map.triple("CenterOfRotationPoint")?,
                })
            }
            "AffineTransform" => {
                if count != 12 {
                    return Err(Error::InvalidArgument(format!(
                        "elastix transform: a 3D `AffineTransform` has 12 parameters, not {count}"
                    )));
                }
                Ok(Self::Matrix {
                    matrix: matrix_row_major(&parameters[..9]),
                    translation: [parameters[9], parameters[10], parameters[11]],
                    centre: map.triple("CenterOfRotationPoint")?,
                })
            }
            "BSplineTransform" | "RecursiveBSplineTransform" => {
                Ok(Self::BSpline(BSpline::from_map(map)?))
            }
            other => Err(Error::InvalidArgument(format!(
                "elastix transform: `(Transform \"{other}\")` is not supported here. Refusing by \
                 name rather than treating it as the identity, which would return coordinates for \
                 a registration that had not been applied. Supported: TranslationTransform, \
                 EulerTransform, AffineTransform, BSplineTransform, RecursiveBSplineTransform."
            ))),
        }
    }
}

/// The rotation matrix of a 3D `EulerTransform`.
///
/// **Measured, both ways.** With angles `(0.3, 0.5, 0.7)`, centre `(2, -3, 4)`
/// and translation `(1.5, -2.5, 3.5)`, transformix maps the origin to
/// `(-1.641247, -5.097380, 5.949039)` when `ComputeZYX` is `"false"` and to
/// `(-1.526417, -4.440980, 5.883335)` when it is `"true"`. Solving over all six
/// orderings of the three elementary rotations leaves exactly one match each:
///
/// | `ComputeZYX` | matrix |
/// |---|---|
/// | `"false"`, and **when the entry is absent** | `Rz · Rx · Ry` |
/// | `"true"` | `Rz · Ry · Rx` |
///
/// The absent case was measured separately — deleting the entry reproduces the
/// `"false"` output to the last digit — rather than inferred from the name.
fn euler_matrix(angle_x: f64, angle_y: f64, angle_z: f64, compute_zyx: bool) -> [[f64; 3]; 3] {
    let rx = [
        [1.0, 0.0, 0.0],
        [0.0, angle_x.cos(), -angle_x.sin()],
        [0.0, angle_x.sin(), angle_x.cos()],
    ];
    let ry = [
        [angle_y.cos(), 0.0, angle_y.sin()],
        [0.0, 1.0, 0.0],
        [-angle_y.sin(), 0.0, angle_y.cos()],
    ];
    let rz = [
        [angle_z.cos(), -angle_z.sin(), 0.0],
        [angle_z.sin(), angle_z.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ];
    if compute_zyx {
        matmul(&matmul(&rz, &ry), &rx)
    } else {
        matmul(&matmul(&rz, &rx), &ry)
    }
}

fn matmul(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            out[row][column] = (0..3).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    out
}

// ----------------------------------------------------------------- chaining --

/// `(HowToCombineTransforms "...")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combine {
    /// `y = T_current(T_initial(x))`. The **initial transform runs first**.
    Compose,
    /// `y = T_initial(x) + T_current(x) - x`: the two displacements added.
    Add,
}

impl Combine {
    /// elastix's own default when the entry is absent.
    ///
    /// **Measured**: deleting `(HowToCombineTransforms ...)` from a two-link
    /// chain reproduces the `"Compose"` output `(21, 61, 121)` and not the
    /// `"Add"` output `(11, 21, 31)`.
    pub const ABSENT: Self = Self::Compose;

    fn from_map(map: &ParameterMap) -> Result<Self> {
        match map.optional_string("HowToCombineTransforms")? {
            None => Ok(Self::ABSENT),
            Some("Compose") => Ok(Self::Compose),
            Some("Add") => Ok(Self::Add),
            Some(other) => Err(Error::InvalidArgument(format!(
                "elastix transform: `(HowToCombineTransforms \"{other}\")` is neither \"Compose\" \
                 nor \"Add\""
            ))),
        }
    }
}

/// One link of a chain: a transform and how it combines with its predecessor.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    transform: Transform,
    combine: Combine,
    initial: Option<String>,
}

impl Link {
    /// Read one link from a parsed parameter file.
    pub fn from_map(map: &ParameterMap) -> Result<Self> {
        let initial = match map.string("InitialTransformParametersFileName")? {
            NO_INITIAL_TRANSFORM => None,
            path => Some(path.to_string()),
        };
        Ok(Self {
            transform: Transform::from_map(map)?,
            combine: Combine::from_map(map)?,
            initial,
        })
    }

    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    pub fn combine(&self) -> Combine {
        self.combine
    }

    /// The `InitialTransformParametersFileName`, or `None` for
    /// `"NoInitialTransform"`.
    pub fn initial(&self) -> Option<&str> {
        self.initial.as_deref()
    }
}

/// A chain of transforms, **innermost first**.
///
/// `InitialTransformParametersFileName` makes the files a singly linked list
/// pointing *backwards*: the file handed to `transformix -tp` is the last
/// transform, and its initial is the one before it. The order is taken from the
/// format rather than from the filenames — `TransformParameters.1.txt` naming
/// `TransformParameters.0.txt` as its initial is a convention elastix happens to
/// follow, not a rule, and [`crate::ElastixTransform::read`] follows the pointers.
///
/// `links[0]` therefore has no initial and is applied first.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    links: Vec<Link>,
}

impl Chain {
    /// A chain from links given **outermost first**, which is the order the
    /// pointers are followed in.
    pub fn from_outermost_first(mut links: Vec<Link>) -> Result<Self> {
        if links.is_empty() {
            return Err(Error::InvalidArgument(
                "elastix transform: a chain needs at least one transform".to_string(),
            ));
        }
        links.reverse();
        if links[0].initial().is_some() {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: the innermost link still names an initial transform \
                 `{}`, so the chain was not followed to its end",
                links[0].initial().unwrap()
            )));
        }
        for link in &links[1..] {
            if link.initial().is_none() {
                return Err(Error::InvalidArgument(
                    "elastix transform: a link other than the innermost has no initial transform, \
                     so these files are not one chain"
                        .to_string(),
                ));
            }
        }
        Ok(Self { links })
    }

    /// The links, innermost first.
    pub fn links(&self) -> &[Link] {
        &self.links
    }

    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Apply the whole chain to one physical point.
    ///
    /// The fold is the measurement written down: each link either composes onto
    /// the running result or adds its own displacement, and `Add` at link `k`
    /// takes the *whole* sub-chain below it as its initial transform.
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let mut accumulated: Option<[f64; 3]> = None;
        for link in &self.links {
            accumulated = Some(match accumulated {
                None => link.transform.apply(point),
                Some(previous) => match link.combine {
                    Combine::Compose => link.transform.apply(previous),
                    Combine::Add => {
                        let current = link.transform.apply(point);
                        [
                            previous[0] + current[0] - point[0],
                            previous[1] + current[1] - point[1],
                            previous[2] + current[2] - point[2],
                        ]
                    }
                },
            });
        }
        accumulated.unwrap_or(point)
    }
}

// ------------------------------------------------------------- geometry --

/// How a point transform rounds a physical point back to a voxel index.
///
/// Two defensible rules, and transformix uses the first:
/// ITK's `Math::RoundHalfIntegerUp`, which is `floor(x + 0.5)`, against Rust's
/// `f64::round`, which is half away from zero. **Measured**: transformix reports
/// `OutputIndexFixed` of `-0.5` as `0` and of `-1.5` as `-1`, which is
/// `floor(x + 0.5)`; `f64::round` would give `-1` and `-2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexRounding {
    /// `floor(x + 0.5)` — ITK's, and transformix's.
    HalfUp,
    /// `f64::round` — half away from zero.
    HalfAwayFromZero,
}

impl IndexRounding {
    pub fn round(self, value: f64) -> f64 {
        match self {
            Self::HalfUp => (value + 0.5).floor(),
            Self::HalfAwayFromZero => value.round(),
        }
    }
}

/// The fixed image's geometry, as the transform parameter file records it.
///
/// **Used only in index mode.** With a `point` input file the transform never
/// touches `Origin`, `Spacing` or `Direction`, and that is measured rather than
/// argued: replacing `(Size 20 24 28) (Spacing 2 3 5) (Origin -10 7 0.5)` with
/// `(Size 3 4 5) (Spacing 7 0.25 13) (Origin 111 -222 333)` leaves every
/// `OutputPoint` bit for bit unchanged. It is the index/physical round trip that
/// reads them — see [`crate::PointSpace`].
#[derive(Debug, Clone, PartialEq)]
pub struct ImageGeometry {
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [[f64; 3]; 3],
    inverse_direction: [[f64; 3]; 3],
}

impl ImageGeometry {
    pub fn new(origin: [f64; 3], spacing: [f64; 3], direction: [[f64; 3]; 3]) -> Result<Self> {
        if spacing
            .iter()
            .any(|value| *value == 0.0 || !value.is_finite())
        {
            return Err(Error::InvalidArgument(format!(
                "elastix transform: `(Spacing {spacing:?})` has a zero or non-finite extent, so \
                 there is no physical-to-index map"
            )));
        }
        Ok(Self {
            origin,
            spacing,
            direction,
            inverse_direction: invert(&direction)?,
        })
    }

    /// Read the geometry from the parameter file handed to `transformix -tp`.
    pub fn from_map(map: &ParameterMap) -> Result<Self> {
        Self::new(
            map.triple("Origin")?,
            map.triple("Spacing")?,
            matrix_column_major(&map.floats_n("Direction", 9)?),
        )
    }

    /// The identity geometry: origin zero, unit spacing, identity direction.
    ///
    /// Named rather than defaulted: it is the one geometry under which voxel
    /// indices handed to a `point` file are also physical coordinates.
    pub fn unit() -> Self {
        Self {
            origin: [0.0; 3],
            spacing: [1.0; 3],
            direction: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            inverse_direction: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    pub fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    pub fn direction(&self) -> [[f64; 3]; 3] {
        self.direction
    }

    /// transformix's reading of an `index` input line: **round to an integer
    /// index first**, then convert.
    ///
    /// The rounding is not a detail. transformix reads an index point file into
    /// doubles and then rounds each coordinate to a `long` before calling
    /// `TransformIndexToPhysicalPoint` — an index is a voxel, and a fractional
    /// one is not a thing the fixed image has. Measured: the index line
    /// `0.5 1.5 2.5` comes back as `InputIndex = [ 1 2 3 ]`, and `-0.5 -1.5 -2.5`
    /// as `[ 0 -1 -2 ]`, which is `floor(x + 0.5)` and the same rule as the
    /// output side.
    ///
    /// An implementation that skipped this would carry the fractional part all the
    /// way through and land within a voxel of the right answer, everywhere —
    /// which is the silent scale error this convention hides. It is
    /// `index_physical_round_trip_dropped` in `tests/register_controls.rs`.
    pub fn index_to_physical_rounded(&self, index: [f64; 3], rounding: IndexRounding) -> [f64; 3] {
        self.index_to_physical([
            rounding.round(index[0]),
            rounding.round(index[1]),
            rounding.round(index[2]),
        ])
    }

    /// `p = Origin + D · (Spacing ⊙ index)`, taking the index as given.
    ///
    /// Measured with an asymmetric direction and anisotropic spacing: index
    /// `(1, 2, 3)` under `(Direction 0 1 0 -1 0 0 0 0 1)`, `(Spacing 2 3 5)`,
    /// `(Origin -10 7 0.5)` is `InputPoint = (-16, 9, 15.5)`.
    pub fn index_to_physical(&self, index: [f64; 3]) -> [f64; 3] {
        let scaled = [
            index[0] * self.spacing[0],
            index[1] * self.spacing[1],
            index[2] * self.spacing[2],
        ];
        let rotated = mul(&self.direction, scaled);
        [
            rotated[0] + self.origin[0],
            rotated[1] + self.origin[1],
            rotated[2] + self.origin[2],
        ]
    }

    /// `index = D^-1 (p - Origin) / Spacing`, unrounded.
    pub fn physical_to_continuous_index(&self, point: [f64; 3]) -> [f64; 3] {
        let shifted = [
            point[0] - self.origin[0],
            point[1] - self.origin[1],
            point[2] - self.origin[2],
        ];
        let rotated = mul(&self.inverse_direction, shifted);
        [
            rotated[0] / self.spacing[0],
            rotated[1] / self.spacing[1],
            rotated[2] / self.spacing[2],
        ]
    }

    /// transformix's `OutputIndexFixed`: the continuous index, rounded.
    pub fn physical_to_index(&self, point: [f64; 3], rounding: IndexRounding) -> [f64; 3] {
        let continuous = self.physical_to_continuous_index(point);
        [
            rounding.round(continuous[0]),
            rounding.round(continuous[1]),
            rounding.round(continuous[2]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_of(text: &str) -> ParameterMap {
        ParameterMap::parse(text).unwrap()
    }

    #[test]
    fn an_unknown_transform_is_refused_by_name() {
        let map = map_of(
            "(Transform \"SimilarityTransform\")\n(NumberOfParameters 7)\n\
             (TransformParameters 0 0 0 0 0 0 1)\n\
             (InitialTransformParametersFileName \"NoInitialTransform\")",
        );
        let error = Transform::from_map(&map).unwrap_err();
        let message = format!("{error}");
        assert!(message.contains("SimilarityTransform"), "{message}");
        assert!(message.contains("Refusing by name"), "{message}");
    }

    #[test]
    fn a_parameter_count_mismatch_is_refused() {
        let map = map_of(
            "(Transform \"TranslationTransform\")\n(NumberOfParameters 3)\n\
             (TransformParameters 1 2)\n\
             (InitialTransformParametersFileName \"NoInitialTransform\")",
        );
        assert!(Transform::from_map(&map).is_err());
    }

    #[test]
    fn the_matrix_readings_are_transposes_of_each_other() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rows = matrix_row_major(&values);
        let columns = matrix_column_major(&values);
        for row in 0..3 {
            for column in 0..3 {
                assert_eq!(rows[row][column], columns[column][row]);
            }
        }
    }

    #[test]
    fn the_cubic_basis_is_a_partition_of_unity() {
        for step in 0..97 {
            let u = step as f64 / 96.0;
            let total: f64 = (-1..=2).map(|k| cubic_basis(u - k as f64)).sum();
            assert!(
                (total - 1.0).abs() < 1e-12,
                "at {u} the weights sum to {total}"
            );
        }
    }

    #[test]
    fn index_rounding_separates_the_two_rules_at_negative_halves() {
        assert_eq!(IndexRounding::HalfUp.round(-0.5), 0.0);
        assert_eq!(IndexRounding::HalfAwayFromZero.round(-0.5), -1.0);
        assert_eq!(IndexRounding::HalfUp.round(-1.5), -1.0);
        assert_eq!(IndexRounding::HalfAwayFromZero.round(-1.5), -2.0);
    }

    #[test]
    fn the_unit_geometry_is_the_identity_both_ways() {
        let geometry = ImageGeometry::unit();
        let point = [1.25, -3.5, 7.0];
        assert_eq!(geometry.index_to_physical(point), point);
        assert_eq!(geometry.physical_to_continuous_index(point), point);
    }
}
