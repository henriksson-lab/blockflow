// SPDX-License-Identifier: MIT
// Original work for this crate.

// The CLI's pipeline on synthetic OME-Zarr: a fixed image at 2-unit voxels and a
// moving image of the same scene, shifted by a known physical offset and stored
// at 3-unit voxels with a pyramid. Registering on a 3-unit lattice must find the
// shift, write a transform `ElastixTransform` reads back, warp the moving image
// onto the fixed lattice, and carry a point table from fixed voxels to moving
// voxels.

use std::path::{Path, PathBuf};

use blockflow::iterate::SubstageLimit;
use blockflow::voxels::Voxels;
use blockflow_atlas_registration::compute::{ControlGrid, Metric, VolumeFitParams};
use blockflow_atlas_registration::pipeline::{
    register, registration_transform, transform_points, Input, PointSource, PointsConfig,
    RegisterConfig, Units,
};
use blockflow_atlas_registration::volume::{write_image, Volume};
use blockflow_atlas_registration::{transform_file, ElastixTransform, Frame, RegisterParams};
use ndarray::Array3;

/// The moving image is the fixed scene moved by this, in physical units, so
/// the fixed-to-moving transform is `p + SHIFT`.
const SHIFT: [f64; 3] = [3.0, -2.0, 4.5];
/// The scene's extent, in physical units.
const BOX: [f64; 3] = [48.0, 54.0, 60.0];

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "atlas-registration-e2e-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A smooth, asymmetric scene, so a translation is identifiable.
fn scene(p: [f64; 3]) -> f64 {
    let blobs = [
        ([20.0, 24.0, 30.0], [60.0, 80.0, 100.0], 1000.0),
        ([30.0, 36.0, 22.0], [30.0, 40.0, 30.0], 600.0),
        ([16.0, 18.0, 42.0], [25.0, 25.0, 45.0], 400.0),
    ];
    blobs
        .iter()
        .map(|(centre, width, height)| {
            let d: f64 = (0..3).map(|a| (p[a] - centre[a]).powi(2) / width[a]).sum();
            height * (-d).exp()
        })
        .sum::<f64>()
}

/// Sample the scene (moved by `shift`) on a lattice of `spacing` voxels.
fn write_scene(root: &Path, spacing: f64, shift: [f64; 3]) -> Frame {
    let shape: [usize; 3] = std::array::from_fn(|a| (BOX[a] / spacing) as usize);
    let frame = Frame::new([spacing; 3], [0.0; 3]).unwrap();
    let image = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(z, y, x)| {
        let p = frame.to_physical([z as f64, y as f64, x as f64]);
        scene(std::array::from_fn(|a| p[a] - shift[a])).round() as u16
    });
    write_image(root, &Voxels::from(image), &frame, Some("micrometer")).unwrap();
    frame
}

/// Squared differences on unit-peak images, stopping at a step or cost change
/// of 1e-3: on a lattice this small the sum of squares settles that far.
fn translation_stage() -> VolumeFitParams {
    let mut params = VolumeFitParams::default().with_metric(Metric::MeanSquares);
    params.limit = SubstageLimit::of(40).unwrap();
    params.tolerance = 1e-3;
    params
}

fn config(scratch: &Scratch, stages: Vec<VolumeFitParams>) -> RegisterConfig {
    RegisterConfig {
        fixed: Input {
            path: scratch.join("fixed.zarr"),
            channel: 0,
            level: None,
        },
        moving: Input {
            path: scratch.join("moving.zarr"),
            channel: 0,
            level: None,
        },
        resolution: Some([3.0; 3]),
        stages: stages.into_iter().map(Into::into).collect(),
        out: scratch.join("registration"),
        warped: Some(scratch.join("warped.zarr")),
        warp_source: None,
        nearest: false,
        block: 8,
        workers: 2,
        warped_stages: false,
    }
}

#[test]
fn a_known_shift_is_found_written_and_carried_to_points() {
    let scratch = Scratch::new("shift");
    let fixed_frame = write_scene(&scratch.join("fixed.zarr"), 2.0, [0.0; 3]);
    let moving_frame = write_scene(&scratch.join("moving.zarr"), 3.0, SHIFT);

    let summary = register(&config(&scratch, vec![translation_stage()])).expect("registers");
    assert_eq!(summary.lattice_shape, [16, 18, 20]);
    assert_eq!(summary.lattice_frame.scale, [3.0; 3]);
    // The fixed level is the stored 2-unit one; the moving is already at 3.
    assert_eq!((summary.fixed_level, summary.moving_level), (0, 0));

    // The written file is the transform, and it is the shift.
    let directory = scratch.join("registration");
    assert!(transform_file(&directory)
        .unwrap()
        .ends_with("TransformParameters.0.txt"));
    let transform = ElastixTransform::from_directory(&directory).unwrap();
    for probe in [[20.0, 24.0, 30.0], [30.0, 36.0, 22.0], [12.0, 30.0, 40.0]] {
        let moved = transform.apply(probe, &RegisterParams::default());
        for axis in 0..3 {
            let error = (moved[axis] - probe[axis] - SHIFT[axis]).abs();
            assert!(
                error < 0.15,
                "probe {probe:?} moved to {moved:?}: axis {axis} off by {error}"
            );
        }
    }

    // The warped moving image lies on the fixed scene.
    let warped = Volume::open(&scratch.join("warped.zarr"), 0).expect("warped image opens");
    assert_eq!(warped.shape(0), summary.lattice_shape);
    assert_eq!(warped.frame(0), summary.lattice_frame);
    assert!(warped.levels() >= 1);
    let values = warped.read(0).unwrap().widened();
    let lattice = summary.lattice_frame;
    let (mut warped_error, mut unwarped_error, mut count) = (0.0, 0.0, 0.0);
    for z in 3..13 {
        for y in 3..15 {
            for x in 3..17 {
                let p = lattice.to_physical([z as f64, y as f64, x as f64]);
                let shifted: [f64; 3] = std::array::from_fn(|a| p[a] - SHIFT[a]);
                warped_error += (values[[z, y, x]] - scene(p)).abs();
                unwarped_error += (scene(shifted) - scene(p)).abs();
                count += 1.0;
            }
        }
    }
    let (warped_error, unwarped_error) = (warped_error / count, unwarped_error / count);
    // The scene peaks at about 1000; linear interpolation of a 3-unit
    // sampling costs a few, and not registering costs far more.
    assert!(
        warped_error < 15.0 && warped_error * 5.0 < unwarped_error,
        "mean error {warped_error} warped, {unwarped_error} unwarped"
    );

    // Fixed level-0 voxels to moving level-0 voxels, through the sidecar.
    let points = scratch.join("points.csv");
    std::fs::write(&points, "id,z,y,x\n1,10,12,15\n2,4.5,20,25\n").unwrap();
    let output = scratch.join("moved.csv");
    let rows = transform_points(&PointsConfig {
        source: PointSource::Registration(directory.clone()),
        input: points,
        output: output.clone(),
        columns: ["z".into(), "y".into(), "x".into()],
        prefix: "moving_".into(),
        units: Units::Voxel,
        reverse_axes: false,
        inverse: false,
    })
    .unwrap();
    assert_eq!(rows, 2);
    let mut reader = csv::Reader::from_path(&output).unwrap();
    assert_eq!(
        reader.headers().unwrap().iter().collect::<Vec<_>>(),
        ["id", "z", "y", "x", "moving_z", "moving_y", "moving_x"]
    );
    for record in reader.records() {
        let record = record.unwrap();
        let value = |i: usize| record[i].parse::<f64>().unwrap();
        let voxel = [value(1), value(2), value(3)];
        let expected = moving_frame.to_index(std::array::from_fn(|a| {
            fixed_frame.to_physical(voxel)[a] + SHIFT[a]
        }));
        for axis in 0..3 {
            let got = value(4 + axis);
            assert!(
                (got - expected[axis]).abs() < 0.1,
                "row {}: axis {axis}: {got} vs {}",
                &record[0],
                expected[axis]
            );
        }
    }

    // Physical in, physical out is the transform itself.
    let physical = registration_transform(&directory, Units::Physical).unwrap();
    assert_eq!(
        physical.apply([20.0, 24.0, 30.0]),
        transform.apply([20.0, 24.0, 30.0], &RegisterParams::default())
    );
}

#[test]
fn an_affine_then_bspline_run_writes_a_two_link_chain() {
    let scratch = Scratch::new("chain");
    write_scene(&scratch.join("fixed.zarr"), 2.0, [0.0; 3]);
    write_scene(&scratch.join("moving.zarr"), 3.0, SHIFT);
    // Affine parameters are in voxels and the matrix entries multiply
    // coordinates of tens of voxels, so the step bound is small.
    let mut affine = translation_stage().affine();
    affine.limit = SubstageLimit::of(40).unwrap();
    affine.tolerance = 1e-3;
    affine.max_step = 0.05;
    let mut bspline = translation_stage()
        .bspline()
        .with_bspline_grid(ControlGrid::new([6, 6, 6]).unwrap());
    bspline.limit = SubstageLimit::of(40).unwrap();
    bspline.tolerance = 1e-3;
    bspline.max_step = 0.5;
    let mut config = config(&scratch, vec![affine, bspline]);
    config.warped = None;

    let summary = register(&config).expect("registers");
    assert_eq!(summary.stages.len(), 2);
    let directory = scratch.join("registration");
    assert!(summary
        .transform_file
        .ends_with("TransformParameters.1.txt"));
    let transform = ElastixTransform::from_directory(&directory).unwrap();
    assert_eq!(
        transform.chain().len(),
        2,
        "affine composed onto the B-spline"
    );
    let probe = [24.0, 28.0, 30.0];
    let moved = transform.apply(probe, &RegisterParams::default());
    for axis in 0..3 {
        let error = (moved[axis] - probe[axis] - SHIFT[axis]).abs();
        assert!(
            error < 0.2,
            "probe moved to {moved:?}: axis {axis} off by {error}"
        );
    }
}

#[test]
fn reversed_axes_are_refused_with_a_registration_directory() {
    let scratch = Scratch::new("reverse");
    let error = transform_points(&PointsConfig {
        source: PointSource::Registration(scratch.join("nowhere")),
        input: scratch.join("points.csv"),
        output: scratch.join("out.csv"),
        columns: ["z".into(), "y".into(), "x".into()],
        prefix: "moving_".into(),
        units: Units::Voxel,
        reverse_axes: true,
        inverse: false,
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("--reverse-axes"), "{error}");
}
