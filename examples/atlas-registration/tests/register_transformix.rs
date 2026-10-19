// SPDX-License-Identifier: MIT
// Original work for this crate.

// Point transforms against **transformix's own answers**.
//
// The transform parameter files below and the numbers beside them were produced
// by running transformix 4.900, and nothing in this file was written by hand.
// `RECORDINGS` is the evidence for the conventions in `src/register.rs`'s
// header, one case per row.
//
// How to reproduce every number here
// ----------------------------------
// ```text
// LD_LIBRARY_PATH=$ELASTIX/bin $ELASTIX/bin/transformix \
//     -def points.txt -out out/ -tp TransformParameters.<highest>.txt
// ```
//
// with the point file's first line `point` for `PointSpace::Physical` and
// `index` for `PointSpace::Index`, and the answer read out of
// `out/outputpoints.txt` — column `OutputPoint` for `Physical` and
// `OutputIndexFixed` for `Index`. With `TRANSFORMIX_DIR` set to the directory
// holding the binary, the ignored tests re-run all of it and check the
// recording is still what transformix says.
//
// **Shelling out is how the answers were obtained, not how the transform is
// computed.** The ordinary tests read `RECORDINGS` and run nothing.
//
// What each row is for
// --------------------
// The probes were built to *separate* readings rather than to confirm one. The
// direction matrices are asymmetric so that row-major and column-major cannot
// both fit; the spacing is anisotropic so that "divide then rotate" and "rotate
// then divide" cannot both fit; the B-spline row holds six different
// coefficients so that a support window off by one cannot land on the same
// answer; the rounding row is all halves, positive and negative, because that is
// the only place `floor(x + 0.5)` and `f64::round` part.
//
// The tolerance is `1e-6`, and it is transformix's: `outputpoints.txt` prints
// six decimals. Every assertion that can be exact — the identity outside a
// B-spline's valid region, the integer `OutputIndexFixed` — is asserted exactly.

use std::path::PathBuf;
use std::process::Command;

use blockflow::voxels::Voxels;
use blockflow_atlas_registration::compute::{run_with_transform_text, VolumeFitParams};
use blockflow_atlas_registration::{ElastixTransform, PointSpace, RegisterParams};
use ndarray::Array3;

/// One transformix run, recorded.
struct Recording {
    name: &'static str,
    /// The parameter files, **outermost first**: the one `-tp` was handed, then
    /// the one it names as its initial transform.
    files: &'static [&'static str],
    space: PointSpace,
    inputs: &'static [[f64; 3]],
    /// `OutputPoint` for `Physical`, `OutputIndexFixed` for `Index`.
    expected: &'static [[f64; 3]],
}

/// transformix prints six decimals and no more.
const TOLERANCE: f64 = 1e-6;

const RECORDINGS: &[Recording] = &[
    Recording {
        // the affine formula: y = A(x - c) + t + c, matrix row-major
        name: "affine",
        files: &[r#"(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters 1.1 0.2 -0.3 0.05 0.9 0.15 -0.25 0.1 1.05 3.5 -2.25 7.125)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(CenterOfRotationPoint 4.0 -6.0 11.0)
(Size 20 24 28)
(Index 0 0 0)
(Spacing 2.0 3.0 5.0)
(Origin -10.0 7.0 0.5)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [7.6, -4.7, 8.175],
            [8.2, -2.4, 11.275],
            [2.925, -1.4125, 8.1375],
            [82.6, -82.2, 25.675],
        ],
    },
    Recording {
        // index in, index out, through an asymmetric Direction and anisotropic Spacing
        name: "affine_index_rotated_direction",
        files: &[r#"(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters 1.1 0.2 -0.3 0.05 0.9 0.15 -0.25 0.1 1.05 3.5 -2.25 7.125)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(CenterOfRotationPoint 4.0 -6.0 11.0)
(Size 20 24 28)
(Index 0 0 0)
(Spacing 2.0 3.0 5.0)
(Origin -10.0 7.0 0.5)
(Direction 0 1 0 -1 0 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Index,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [5.0, -2.0, 0.0],
            [10.0, 11.0, 12.0],
        ],
        expected: &[
            [-3.0, -3.0, 2.0],
            [-1.0, 1.0, 6.0],
            [2.0, -5.0, 2.0],
            [10.0, 14.0, 17.0],
        ],
    },
    Recording {
        // y = x + t
        name: "translation",
        files: &[r#"(Transform "TranslationTransform")
(NumberOfParameters 3)
(TransformParameters 10.0 20.0 30.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [10.0, 20.0, 30.0],
            [11.0, 22.0, 33.0],
            [4.5, 24.25, 28.25],
            [110.0, -80.0, 80.0],
        ],
    },
    Recording {
        // Rz * Rx * Ry
        name: "euler_zyx_false",
        files: &[r#"(Transform "EulerTransform")
(NumberOfParameters 6)
(TransformParameters 0.3 0.5 0.7 1.5 -2.5 3.5)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(CenterOfRotationPoint 2.0 -3.0 4.0)
(ComputeZYX "false")
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [-1.641247, -5.09738, 5.949039],
            [-0.690921, -2.630806, 8.597227],
            [-8.380631, -5.890797, 8.256894],
            [144.585088, -5.268962, -27.48492],
        ],
    },
    Recording {
        // Rz * Ry * Rx
        name: "euler_zyx_true",
        files: &[r#"(Transform "EulerTransform")
(NumberOfParameters 6)
(TransformParameters 0.3 0.5 0.7 1.5 -2.5 3.5)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(CenterOfRotationPoint 2.0 -3.0 4.0)
(ComputeZYX "true")
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [-1.526417, -4.44098, 5.883335],
            [-0.247308, -2.024616, 8.437756],
            [-8.319384, -4.17793, 8.155208],
            [143.337326, -26.649317, -26.074225],
        ],
    },
    Recording {
        // the absent ComputeZYX is "false"
        name: "euler_zyx_absent",
        files: &[r#"(Transform "EulerTransform")
(NumberOfParameters 6)
(TransformParameters 0.3 0.5 0.7 1.5 -2.5 3.5)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(CenterOfRotationPoint 2.0 -3.0 4.0)
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [-1.641247, -5.09738, 5.949039],
            [-0.690921, -2.630806, 8.597227],
            [-8.380631, -5.890797, 8.256894],
            [144.585088, -5.268962, -27.48492],
        ],
    },
    Recording {
        // T_current(T_initial(x)): the initial transform runs first
        name: "chain_compose",
        files: &[
            r#"(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters 2.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 4.0 1.0 1.0 1.0)
(InitialTransformParametersFileName "TransformParameters.0.txt")
(HowToCombineTransforms "Compose")
(CenterOfRotationPoint 0.0 0.0 0.0)
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#,
            r#"(Transform "TranslationTransform")
(NumberOfParameters 3)
(TransformParameters 10.0 20.0 30.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#,
        ],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [21.0, 61.0, 121.0],
            [23.0, 67.0, 133.0],
            [10.0, 73.75, 114.0],
            [221.0, -239.0, 321.0],
        ],
    },
    Recording {
        // T_initial(x) + T_current(x) - x
        name: "chain_add",
        files: &[
            r#"(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters 2.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 4.0 1.0 1.0 1.0)
(InitialTransformParametersFileName "TransformParameters.0.txt")
(HowToCombineTransforms "Add")
(CenterOfRotationPoint 0.0 0.0 0.0)
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#,
            r#"(Transform "TranslationTransform")
(NumberOfParameters 3)
(TransformParameters 10.0 20.0 30.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#,
        ],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [11.0, 21.0, 31.0],
            [13.0, 27.0, 43.0],
            [0.0, 33.75, 24.0],
            [211.0, -279.0, 231.0],
        ],
    },
    Recording {
        // the absent HowToCombineTransforms is "Compose"
        name: "chain_combine_absent",
        files: &[
            r#"(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters 2.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 4.0 1.0 1.0 1.0)
(InitialTransformParametersFileName "TransformParameters.0.txt")
(CenterOfRotationPoint 0.0 0.0 0.0)
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#,
            r#"(Transform "TranslationTransform")
(NumberOfParameters 3)
(TransformParameters 10.0 20.0 30.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#,
        ],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.5, 4.25, -1.75],
            [100.0, -100.0, 50.0],
        ],
        expected: &[
            [21.0, 61.0, 121.0],
            [23.0, 67.0, 133.0],
            [10.0, 73.75, 114.0],
            [221.0, -239.0, 321.0],
        ],
    },
    Recording {
        // the coefficient layout: three blocks, x first, first grid axis fastest
        name: "bspline_one_control_point",
        files: &[r#"(Transform "BSplineTransform")
(NumberOfParameters 648)
(TransformParameters 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(GridSize 6 6 6)
(GridIndex 0 0 0)
(GridSpacing 10.0 10.0 10.0)
(GridOrigin 0.0 0.0 0.0)
(GridDirection 1 0 0 0 1 0 0 0 1)
(BSplineTransformSplineOrder 3)
(UseCyclicTransform "false")
(Size 50 50 50)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.0, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [15.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [20.5, 20.0, 20.0],
            [25.0, 20.0, 20.0],
            [30.0, 20.0, 20.0],
            [35.0, 20.0, 20.0],
            [39.0, 20.0, 20.0],
            [40.0, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.0, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [-1.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [20.0, 25.0, 20.0],
            [20.0, 20.0, 25.0],
            [12.5, 17.5, 23.25],
        ],
        expected: &[
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.0, 20.0, 20.0],
            [10.074074, 20.148148, 20.222222],
            [15.212963, 20.425926, 20.638889],
            [20.296296, 20.592593, 20.888889],
            [20.795213, 20.590426, 20.885639],
            [25.212963, 20.425926, 20.638889],
            [30.074074, 20.148148, 20.222222],
            [35.009259, 20.018519, 20.027778],
            [39.000074, 20.000148, 20.000222],
            [40.0, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.0, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [-1.0, 20.0, 20.0],
            [20.296296, 20.592593, 20.888889],
            [20.212963, 25.425926, 20.638889],
            [20.212963, 20.425926, 25.638889],
            [12.6115, 17.722999, 23.584499],
        ],
    },
    Recording {
        // RecursiveBSplineTransform is the same evaluation
        name: "bspline_recursive_keyword",
        files: &[r#"(Transform "RecursiveBSplineTransform")
(NumberOfParameters 648)
(TransformParameters 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(GridSize 6 6 6)
(GridIndex 0 0 0)
(GridSpacing 10.0 10.0 10.0)
(GridOrigin 0.0 0.0 0.0)
(GridDirection 1 0 0 0 1 0 0 0 1)
(BSplineTransformSplineOrder 3)
(UseCyclicTransform "false")
(Size 50 50 50)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.0, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [15.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [20.5, 20.0, 20.0],
            [25.0, 20.0, 20.0],
            [30.0, 20.0, 20.0],
            [35.0, 20.0, 20.0],
            [39.0, 20.0, 20.0],
            [40.0, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.0, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [-1.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [20.0, 25.0, 20.0],
            [20.0, 20.0, 25.0],
            [12.5, 17.5, 23.25],
        ],
        expected: &[
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.0, 20.0, 20.0],
            [10.074074, 20.148148, 20.222222],
            [15.212963, 20.425926, 20.638889],
            [20.296296, 20.592593, 20.888889],
            [20.795213, 20.590426, 20.885639],
            [25.212963, 20.425926, 20.638889],
            [30.074074, 20.148148, 20.222222],
            [35.009259, 20.018519, 20.027778],
            [39.000074, 20.000148, 20.000222],
            [40.0, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.0, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [-1.0, 20.0, 20.0],
            [20.296296, 20.592593, 20.888889],
            [20.212963, 25.425926, 20.638889],
            [20.212963, 20.425926, 25.638889],
            [12.6115, 17.722999, 23.584499],
        ],
    },
    Recording {
        // the valid region is [GridIndex + 1, GridIndex + GridSize - 2)
        name: "bspline_valid_region",
        files: &[r#"(Transform "BSplineTransform")
(NumberOfParameters 648)
(TransformParameters 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(GridSize 6 6 6)
(GridIndex 0 0 0)
(GridSpacing 10.0 10.0 10.0)
(GridOrigin 0.0 0.0 0.0)
(GridDirection 1 0 0 0 1 0 0 0 1)
(BSplineTransformSplineOrder 3)
(UseCyclicTransform "false")
(Size 50 50 50)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [-15.0, 20.0, 20.0],
            [-10.0, 20.0, 20.0],
            [-5.0, 20.0, 20.0],
            [-0.001, 20.0, 20.0],
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.999, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [10.001, 20.0, 20.0],
            [15.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [25.0, 20.0, 20.0],
            [30.0, 20.0, 20.0],
            [35.0, 20.0, 20.0],
            [39.999, 20.0, 20.0],
            [40.0, 20.0, 20.0],
            [40.001, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.999, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [55.0, 20.0, 20.0],
            [60.0, 20.0, 20.0],
        ],
        expected: &[
            [-15.0, 20.0, 20.0],
            [-10.0, 20.0, 20.0],
            [-5.0, 20.0, 20.0],
            [-0.001, 20.0, 20.0],
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.999, 20.0, 20.0],
            [10.888889, 20.0, 20.0],
            [10.889933, 20.0, 20.0],
            [16.111111, 20.0, 20.0],
            [21.333333, 20.0, 20.0],
            [26.555556, 20.0, 20.0],
            [31.777778, 20.0, 20.0],
            [37.0, 20.0, 20.0],
            [42.221178, 20.0, 20.0],
            [40.0, 20.0, 20.0],
            [40.001, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.999, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [55.0, 20.0, 20.0],
            [60.0, 20.0, 20.0],
        ],
    },
    Recording {
        // GridIndex moves both ends of the valid region by one grid step
        name: "bspline_grid_index",
        files: &[r#"(Transform "BSplineTransform")
(NumberOfParameters 648)
(TransformParameters 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(GridSize 6 6 6)
(GridIndex 1 1 1)
(GridSpacing 10.0 10.0 10.0)
(GridOrigin 0.0 0.0 0.0)
(GridDirection 1 0 0 0 1 0 0 0 1)
(BSplineTransformSplineOrder 3)
(UseCyclicTransform "false")
(Size 50 50 50)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [-15.0, 20.0, 20.0],
            [-10.0, 20.0, 20.0],
            [-5.0, 20.0, 20.0],
            [-0.001, 20.0, 20.0],
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.999, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [10.001, 20.0, 20.0],
            [15.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [25.0, 20.0, 20.0],
            [30.0, 20.0, 20.0],
            [35.0, 20.0, 20.0],
            [39.999, 20.0, 20.0],
            [40.0, 20.0, 20.0],
            [40.001, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.999, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [55.0, 20.0, 20.0],
            [60.0, 20.0, 20.0],
        ],
        expected: &[
            [-15.0, 20.0, 20.0],
            [-10.0, 20.0, 20.0],
            [-5.0, 20.0, 20.0],
            [-0.001, 20.0, 20.0],
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.999, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [10.001, 20.0, 20.0],
            [15.0, 20.0, 20.0],
            [20.055556, 20.0, 20.0],
            [25.069444, 20.0, 20.0],
            [30.083333, 20.0, 20.0],
            [35.097222, 20.0, 20.0],
            [40.110108, 20.0, 20.0],
            [40.111111, 20.0, 20.0],
            [40.112114, 20.0, 20.0],
            [45.125, 20.0, 20.0],
            [50.137886, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [55.0, 20.0, 20.0],
            [60.0, 20.0, 20.0],
        ],
    },
    Recording {
        // GridDirection is read column-major, like Direction
        name: "bspline_grid_direction",
        files: &[r#"(Transform "BSplineTransform")
(NumberOfParameters 648)
(TransformParameters 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(GridSize 6 6 6)
(GridIndex 0 0 0)
(GridSpacing 10.0 10.0 10.0)
(GridOrigin 0.0 0.0 0.0)
(GridDirection 0 1 0 -1 0 0 0 0 1)
(BSplineTransformSplineOrder 3)
(UseCyclicTransform "false")
(Size 50 50 50)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Physical,
        inputs: &[
            [-15.0, 20.0, 20.0],
            [-10.0, 20.0, 20.0],
            [-5.0, 20.0, 20.0],
            [-0.001, 20.0, 20.0],
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.999, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [10.001, 20.0, 20.0],
            [15.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [25.0, 20.0, 20.0],
            [30.0, 20.0, 20.0],
            [35.0, 20.0, 20.0],
            [39.999, 20.0, 20.0],
            [40.0, 20.0, 20.0],
            [40.001, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.999, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [55.0, 20.0, 20.0],
            [60.0, 20.0, 20.0],
        ],
        expected: &[
            [-14.041667, 20.0, 20.0],
            [-9.666667, 20.0, 20.0],
            [-5.0, 20.0, 20.0],
            [-0.001, 20.0, 20.0],
            [0.0, 20.0, 20.0],
            [5.0, 20.0, 20.0],
            [9.999, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [10.001, 20.0, 20.0],
            [15.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [25.0, 20.0, 20.0],
            [30.0, 20.0, 20.0],
            [35.0, 20.0, 20.0],
            [39.999, 20.0, 20.0],
            [40.0, 20.0, 20.0],
            [40.001, 20.0, 20.0],
            [45.0, 20.0, 20.0],
            [49.999, 20.0, 20.0],
            [50.0, 20.0, 20.0],
            [55.0, 20.0, 20.0],
            [60.0, 20.0, 20.0],
        ],
    },
    Recording {
        // OutputIndexFixed rounds by floor(x + 0.5), not half away from zero
        name: "index_rounding_halves",
        files: &[r#"(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters 1.1 0.2 -0.3 0.05 0.9 0.15 -0.25 0.1 1.05 3.5 -2.25 7.125)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(CenterOfRotationPoint 4.0 -6.0 11.0)
(Size 20 24 28)
(Index 0 0 0)
(Spacing 1.0 1.0 1.0)
(Origin 0.0 0.0 0.0)
(Direction 1 0 0 0 1 0 0 0 1)
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 3)
(Resampler "DefaultResampler")
(DefaultPixelValue 0.000000)
(ResultImageFormat "mhd")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImage "false")"#],
        space: PointSpace::Index,
        inputs: &[
            [0.5, 1.5, 2.5],
            [-0.5, -1.5, -2.5],
            [3.5, -3.5, 4.5],
            [-2.5, -4.5, -6.5],
            [7.5, 8.5, 9.5],
        ],
        expected: &[
            [8.0, -2.0, 11.0],
            [8.0, -6.0, 6.0],
            [10.0, -6.0, 12.0],
            [6.0, -9.0, 2.0],
            [15.0, 5.0, 18.0],
        ],
    },
];

fn transformix_command() -> Option<(PathBuf, Option<PathBuf>)> {
    if let Some(directory) = std::env::var_os("TRANSFORMIX_DIR") {
        let directory = PathBuf::from(directory);
        return Some((directory.join("transformix"), Some(directory)));
    }
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join("transformix");
        if candidate.is_file() {
            return Some((candidate, None));
        }
    }
    None
}

fn run_transformix(
    transformix: &(PathBuf, Option<PathBuf>),
    point_file: &std::path::Path,
    out: &std::path::Path,
    transform_file: &std::path::Path,
) -> std::process::Output {
    let mut command = Command::new(&transformix.0);
    if let Some(directory) = &transformix.1 {
        command.env("LD_LIBRARY_PATH", directory);
    }
    command
        .arg("-def")
        .arg(point_file)
        .arg("-out")
        .arg(out)
        .arg("-tp")
        .arg(transform_file)
        .output()
        .expect("transformix runs")
}

fn params(space: PointSpace) -> RegisterParams {
    RegisterParams::in_space(space)
}

fn recording(name: &str) -> &'static Recording {
    RECORDINGS
        .iter()
        .find(|recording| recording.name == name)
        .unwrap_or_else(|| panic!("no recording named {name}"))
}

fn transformed(recording: &Recording) -> Vec<[f64; 3]> {
    let transform = ElastixTransform::from_texts(recording.files)
        .unwrap_or_else(|error| panic!("{}: {error}", recording.name));
    transform.apply_all(recording.inputs, &params(recording.space))
}

// ------------------------------------------------------------------------
// The whole table, in one assertion
// ------------------------------------------------------------------------

/// Every recorded transformix answer, reproduced.
///
/// One test rather than sixteen, because the failure that matters is "a
/// convention is wrong" and it shows up as a row name either way — and because a
/// row silently *missing* from the sweep is the failure a per-case test cannot
/// see. The count is asserted with it.
#[test]
fn every_recorded_transformix_answer_is_reproduced() {
    assert_eq!(RECORDINGS.len(), 15, "a recording was added or lost");
    let mut points = 0usize;
    for recording in RECORDINGS {
        let ours = transformed(recording);
        assert_eq!(
            ours.len(),
            recording.expected.len(),
            "{}: point count",
            recording.name
        );
        for (row, (ours, theirs)) in ours.iter().zip(recording.expected).enumerate() {
            for axis in 0..3 {
                let difference = (ours[axis] - theirs[axis]).abs();
                assert!(
                    difference <= TOLERANCE,
                    "{}: row {row}, axis {axis}: this crate says {}, transformix recorded {}, \
                     difference {difference:e}",
                    recording.name,
                    ours[axis],
                    theirs[axis]
                );
            }
        }
        points += ours.len();
    }
    assert_eq!(points, 147, "the recorded points");
}

// ------------------------------------------------------------------------
// The four conventions, each asserted as the thing it establishes
// ------------------------------------------------------------------------

/// **Direction.** The transform is applied forward — as stored — and nothing
/// inverts it. Asserted as the property rather than only as numbers: running the
/// affine recording backwards through its own inverse returns the inputs, so the
/// recorded outputs are `T(x)` and not `T^-1(x)`.
///
/// The inverse is built here from the recorded matrix rather than from the
/// module, which has none: the point of the check is that `apply` is the forward
/// map, and taking the inverse from the same module would prove nothing.
#[test]
fn the_transform_is_applied_forward_as_stored() {
    let case = recording("affine");
    let ours = transformed(case);
    // (TransformParameters 1.1 0.2 -0.3 0.05 0.9 0.15 -0.25 0.1 1.05 3.5 -2.25 7.125)
    // (CenterOfRotationPoint 4.0 -6.0 11.0)
    let a = [[1.1, 0.2, -0.3], [0.05, 0.9, 0.15], [-0.25, 0.1, 1.05]];
    let t = [3.5, -2.25, 7.125];
    let c = [4.0, -6.0, 11.0];
    for (input, output) in case.inputs.iter().zip(&ours) {
        // y = A(x - c) + t + c  <=>  x = A^-1(y - t - c) + c
        let rhs = [
            output[0] - t[0] - c[0],
            output[1] - t[1] - c[1],
            output[2] - t[2] - c[2],
        ];
        let back = solve(&a, rhs);
        for axis in 0..3 {
            assert!(
                (back[axis] + c[axis] - input[axis]).abs() < 1e-9,
                "the recorded output is not the forward image of the input"
            );
        }
    }
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting, for the check
/// above. `f64::total_cmp`, never `f64::max`.
fn solve(a: &[[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let mut m = [[0.0f64; 4]; 3];
    for row in 0..3 {
        m[row][..3].copy_from_slice(&a[row]);
        m[row][3] = b[row];
    }
    for column in 0..3 {
        let pivot = (column..3)
            .max_by(|left, right| m[*left][column].abs().total_cmp(&m[*right][column].abs()))
            .expect("three rows");
        m.swap(column, pivot);
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = m[row][column] / m[column][column];
            for k in column..4 {
                m[row][k] -= factor * m[column][k];
            }
        }
    }
    [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]]
}

/// **Index versus physical.** The same transform, the same points, read once as
/// physical coordinates and once as voxel indices through a non-unit geometry,
/// gives different answers — which is what says the round trip is real and not
/// decoration.
#[test]
fn the_two_point_spaces_are_different_answers() {
    let case = recording("affine_index_rotated_direction");
    let transform = ElastixTransform::from_texts(case.files).expect("the files parse");
    let as_index = transform.apply_all(case.inputs, &params(PointSpace::Index));
    let as_physical = transform.apply_all(case.inputs, &params(PointSpace::Physical));
    let moved = as_index
        .iter()
        .zip(&as_physical)
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(moved, case.inputs.len(), "every point moves");
    // And the index answer is an integer, because `OutputIndexFixed` is.
    for point in &as_index {
        for axis in 0..3 {
            assert_eq!(
                point[axis],
                point[axis].trunc(),
                "OutputIndexFixed is integral"
            );
        }
    }
}

/// **Index rounding.** The recorded halves, and the rule that is *not*
/// transformix's, on the same points. Both are here because the axis is only
/// meaningful if the other value is reachable and different.
#[test]
fn the_index_rounding_axis_moves_the_negative_halves() {
    use blockflow_atlas_registration::IndexRounding;
    let case = recording("index_rounding_halves");
    let transform = ElastixTransform::from_texts(case.files).expect("the files parse");
    let mut theirs = params(PointSpace::Index);
    theirs.index_rounding = IndexRounding::HalfAwayFromZero;
    let ours = transform.apply_all(case.inputs, &params(PointSpace::Index));
    let other = transform.apply_all(case.inputs, &theirs);
    assert_eq!(ours.len(), other.len());
    let moved = ours.iter().zip(&other).filter(|(l, r)| l != r).count();
    assert!(
        moved > 0,
        "the two rounding rules agree everywhere, which cannot be"
    );
}

/// **The chain.** `"Compose"` and `"Add"` on the same two files are different
/// answers, and the file with no `HowToCombineTransforms` entry is `"Compose"`.
#[test]
fn the_chain_combination_rules_are_three_facts() {
    let compose = transformed(recording("chain_compose"));
    let add = transformed(recording("chain_add"));
    let absent = transformed(recording("chain_combine_absent"));
    assert_eq!(
        compose, absent,
        "an absent HowToCombineTransforms is Compose"
    );
    assert_ne!(compose, add, "Compose and Add are the same answer");
    // The order within Compose: the initial transform runs first. At the origin,
    // translation-then-affine is (21, 61, 121); affine-then-translation would be
    // (11, 21, 31), which is `Add`'s answer here and would hide the difference if
    // the fold ran the other way — so the two are checked against each other.
    assert_eq!(compose[0], [21.0, 61.0, 121.0]);
    assert_eq!(add[0], [11.0, 21.0, 31.0]);
}

/// **The chain, as a length.** The recorded two-file cases really are two links,
/// and the single-file cases really are one. A fold that silently dropped a link
/// would still reproduce a one-link case.
#[test]
fn the_chain_lengths_are_what_the_files_say() {
    for name in ["chain_compose", "chain_add", "chain_combine_absent"] {
        let case = recording(name);
        let transform = ElastixTransform::from_texts(case.files).expect("the files parse");
        assert_eq!(transform.chain().len(), 2, "{name}");
    }
    let transform = ElastixTransform::from_texts(recording("affine").files).expect("parses");
    assert_eq!(transform.chain().len(), 1);
}

/// **B-spline: off the grid.** Outside the valid region the transform is the
/// identity, *exactly* — not approximately, and not an extrapolation.
#[test]
fn outside_the_bspline_valid_region_the_point_is_returned_unchanged() {
    let case = recording("bspline_valid_region");
    let ours = transformed(case);
    let mut outside = 0usize;
    for (input, output) in case.inputs.iter().zip(&ours) {
        // The sweep runs x from -15 to 60 at y = z = 20; the valid region is
        // x in [10, 40).
        if input[0] < 10.0 || input[0] >= 40.0 {
            assert_eq!(output, input, "outside the region the point must not move");
            outside += 1;
        } else {
            assert_ne!(output, input, "inside the region the point must move");
        }
    }
    assert_eq!(outside, 14, "the sweep straddles both ends");
}

/// **B-spline: the edge control points.** `GridIndex` moves the valid region,
/// which is what says the bound is relative to it rather than to zero — and
/// which is the reading an implementation that hard-coded `[1, GridSize - 2)`
/// would get wrong on every elastix run with a non-zero `GridIndex`.
#[test]
fn the_bspline_valid_region_follows_grid_index() {
    let at_zero = recording("bspline_valid_region");
    let at_one = recording("bspline_grid_index");
    let ours_zero = transformed(at_zero);
    let ours_one = transformed(at_one);
    let region = |inputs: &[[f64; 3]], outputs: &[[f64; 3]]| -> (f64, f64) {
        let moved: Vec<f64> = inputs
            .iter()
            .zip(outputs)
            .filter(|(input, output)| input != output)
            .map(|(input, _)| input[0])
            .collect();
        (
            moved.iter().copied().fold(
                f64::INFINITY,
                |a, b| if a.total_cmp(&b).is_le() { a } else { b },
            ),
            moved.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
                if a.total_cmp(&b).is_ge() {
                    a
                } else {
                    b
                }
            }),
        )
    };
    let (low_zero, high_zero) = region(at_zero.inputs, &ours_zero);
    let (low_one, high_one) = region(at_one.inputs, &ours_one);
    assert_eq!((low_zero, high_zero), (10.0, 39.999));
    assert_eq!((low_one, high_one), (20.0, 49.999));
}

/// **B-spline: the two keywords are one evaluation.** Recorded from transformix,
/// so this is a fact about elastix and not an assumption of this crate.
#[test]
fn recursive_and_plain_bspline_are_the_same_transform() {
    assert_eq!(
        transformed(recording("bspline_one_control_point")),
        transformed(recording("bspline_recursive_keyword"))
    );
}

/// **Euler.** The two `ComputeZYX` values are different answers and the absent
/// entry is `"false"`.
#[test]
fn the_euler_rotation_order_is_three_facts() {
    let no = transformed(recording("euler_zyx_false"));
    let yes = transformed(recording("euler_zyx_true"));
    let absent = transformed(recording("euler_zyx_absent"));
    assert_eq!(no, absent, "an absent ComputeZYX is \"false\"");
    assert_ne!(no, yes, "the two orders are the same answer");
}

// ------------------------------------------------------------------------
// The liveness of the recording
// ------------------------------------------------------------------------

/// Re-run every case through the real transformix and check the recording.
///
/// This is the **liveness test** beside the recorded ones above: without it,
/// `RECORDINGS` could drift into being a record of what this crate does rather
/// than of what transformix does, and every assertion in this file would still
/// pass. Ignored by default because it needs the binary:
///
/// ```text
/// TRANSFORMIX_DIR=$ELASTIX/bin \
///     cargo test --release -p blockflow-atlas-registration --test register_transformix -- --ignored
/// ```
///
/// The parameter files are written out and their
/// `InitialTransformParametersFileName` entries rewritten to absolute paths
/// first, because elastix resolves a relative initial against the working
/// directory.
#[test]
#[ignore = "needs the transformix binary: set TRANSFORMIX_DIR to the directory holding it"]
fn the_recording_still_matches_transformix() {
    let binary_directory = std::env::var_os("TRANSFORMIX_DIR")
        .expect("TRANSFORMIX_DIR names the directory holding `transformix`");
    let binary_directory = std::path::PathBuf::from(binary_directory);
    let root = std::env::temp_dir().join(format!(
        "atlas-registration-register-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    for recording in RECORDINGS {
        let directory = root.join(recording.name);
        std::fs::create_dir_all(directory.join("out")).expect("the case directory is created");
        // `files` is outermost first; elastix numbers them innermost first.
        let count = recording.files.len();
        let mut written = Vec::new();
        for (position, text) in recording.files.iter().rev().enumerate() {
            let path = directory.join(format!("TransformParameters.{position}.txt"));
            let mut text = (*text).to_string();
            if position > 0 {
                let initial: &std::path::PathBuf = &written[position - 1];
                text = text.replace(
                    &format!(
                        "(InitialTransformParametersFileName \"TransformParameters.{}.txt\")",
                        position - 1
                    ),
                    &format!(
                        "(InitialTransformParametersFileName \"{}\")",
                        initial.display()
                    ),
                );
            }
            std::fs::write(&path, text).expect("the parameter file is written");
            written.push(path);
        }
        let header = match recording.space {
            PointSpace::Physical => "point",
            PointSpace::Index => "index",
        };
        let mut body = format!("{header}\n{}\n", recording.inputs.len());
        for point in recording.inputs {
            body.push_str(&format!("{} {} {}\n", point[0], point[1], point[2]));
        }
        let point_file = directory.join("points.txt");
        std::fs::write(&point_file, body).expect("the point file is written");
        let status = std::process::Command::new(binary_directory.join("transformix"))
            .env("LD_LIBRARY_PATH", &binary_directory)
            .arg("-def")
            .arg(&point_file)
            .arg("-out")
            .arg(directory.join("out"))
            .arg("-tp")
            .arg(&written[count - 1])
            .output()
            .expect("transformix runs");
        assert!(
            status.status.success(),
            "{}: transformix failed\n{}",
            recording.name,
            String::from_utf8_lossy(&status.stderr)
        );
        let text = std::fs::read_to_string(directory.join("out").join("outputpoints.txt"))
            .expect("outputpoints.txt is written");
        let column = match recording.space {
            PointSpace::Physical => "OutputPoint = [ ",
            PointSpace::Index => "OutputIndexFixed = [ ",
        };
        let live: Vec<[f64; 3]> = text
            .lines()
            .map(|line| {
                let start = line.find(column).expect("the column is present") + column.len();
                let rest = &line[start..];
                let end = rest.find(']').expect("the column is closed");
                let values: Vec<f64> = rest[..end]
                    .split_whitespace()
                    .map(|value| value.parse().expect("a number"))
                    .collect();
                [values[0], values[1], values[2]]
            })
            .collect();
        assert_eq!(live.len(), recording.expected.len(), "{}", recording.name);
        for (row, (live, recorded)) in live.iter().zip(recording.expected).enumerate() {
            for axis in 0..3 {
                assert!(
                    (live[axis] - recorded[axis]).abs() <= TOLERANCE,
                    "{}: row {row}, axis {axis}: the recording says {}, transformix now says {}",
                    recording.name,
                    recorded[axis],
                    live[axis]
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

struct GeneratedCase {
    name: &'static str,
    params: VolumeFitParams,
    fixed: Voxels,
    moving: Voxels,
    inputs: Vec<[f64; 3]>,
}

fn generated_translation_case() -> GeneratedCase {
    let shape = [9, 7, 5];
    let mut fixed = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    let mut moving = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    fixed[[2, 3, 1]] = 10.0;
    moving[[4, 2, 3]] = 10.0;
    GeneratedCase {
        name: "translation",
        params: VolumeFitParams::default(),
        fixed: Voxels::F64(fixed),
        moving: Voxels::F64(moving),
        inputs: vec![[2.0, 3.0, 1.0], [0.0, 0.0, 0.0], [8.0, 6.0, 4.0]],
    }
}

fn generated_affine_case() -> GeneratedCase {
    let shape = [9, 7, 5];
    let mut fixed = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    let mut moving = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    fixed[[3, 3, 2]] = 10.0;
    fixed[[5, 1, 3]] = 4.0;
    moving[[4, 2, 2]] = 10.0;
    moving[[6, 1, 3]] = 4.0;
    GeneratedCase {
        name: "affine",
        params: VolumeFitParams::default().affine(),
        fixed: Voxels::F64(fixed),
        moving: Voxels::F64(moving),
        inputs: vec![[3.0, 3.0, 2.0], [5.0, 1.0, 3.0], [0.0, 0.0, 0.0]],
    }
}

fn generated_bspline_case() -> GeneratedCase {
    let shape = [9, 7, 5];
    let mut fixed = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    let mut moving = Array3::from_elem((shape[0], shape[1], shape[2]), 0.0);
    fixed[[3, 3, 2]] = 10.0;
    moving[[4, 2, 2]] = 10.0;
    GeneratedCase {
        name: "bspline",
        params: VolumeFitParams::default().bspline(),
        fixed: Voxels::F64(fixed),
        moving: Voxels::F64(moving),
        inputs: vec![[3.0, 3.0, 2.0], [4.0, 3.0, 2.0], [3.0, 2.0, 2.0]],
    }
}

#[test]
#[ignore = "needs the transformix binary: on PATH, or TRANSFORMIX_DIR set to its directory"]
fn generated_native_files_match_transformix() {
    let transformix = transformix_command().expect("transformix on PATH or in TRANSFORMIX_DIR");
    for case in [
        generated_translation_case(),
        generated_affine_case(),
        generated_bspline_case(),
    ] {
        let (native, text) = run_with_transform_text(&case.params, &case.fixed, &case.moving, 4)
            .expect("native transform is generated");
        let root = std::env::temp_dir().join(format!(
            "atlas-registration-generated-transformix-{}-{}",
            case.name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("out")).expect("case directory");
        let transform_file = root.join("TransformParameters.0.txt");
        std::fs::write(&transform_file, text).expect("transform file");
        let mut point_body = format!("point\n{}\n", case.inputs.len());
        for point in &case.inputs {
            point_body.push_str(&format!("{} {} {}\n", point[0], point[1], point[2]));
        }
        let point_file = root.join("points.txt");
        std::fs::write(&point_file, point_body).expect("point file");
        let output = run_transformix(
            &transformix,
            &point_file,
            &root.join("out"),
            &transform_file,
        );
        assert!(
            output.status.success(),
            "{}: transformix failed on generated native transform\nstderr:\n{}\nstdout:\n{}",
            case.name,
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
        let text = std::fs::read_to_string(root.join("out").join("outputpoints.txt"))
            .expect("outputpoints.txt");
        let live: Vec<[f64; 3]> = text
            .lines()
            .map(|line| {
                let column = "OutputPoint = [ ";
                let start = line.find(column).expect("OutputPoint column") + column.len();
                let rest = &line[start..];
                let end = rest.find(']').expect("OutputPoint closes");
                let values: Vec<f64> = rest[..end]
                    .split_whitespace()
                    .map(|value| value.parse().expect("number"))
                    .collect();
                [values[0], values[1], values[2]]
            })
            .collect();
        assert_eq!(live.len(), case.inputs.len(), "{}", case.name);
        let params = RegisterParams::default();
        for (row, (got, input)) in live.iter().zip(&case.inputs).enumerate() {
            let expected = native.apply(*input, &params);
            for axis in 0..3 {
                assert!(
                    (got[axis] - expected[axis]).abs() <= TOLERANCE,
                    "{}: row {row}, axis {axis}: generated transformix output {} != native {}",
                    case.name,
                    got[axis],
                    expected[axis]
                );
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
