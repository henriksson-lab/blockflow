// SPDX-License-Identifier: MIT
// Original work for this crate.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blockflow::assemble::ImageId;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PyramidSpec, ZarrEnvironment};
use blockflow_atlas_registration::compute::{
    resident_pyramid_levels, zarr_pyramid_levels_or_build, PyramidSchedule, SpatialFrame,
    VolumeFitParams, VolumePyramidLevel, VolumePyramidStoreInput,
};
use blockflow_atlas_registration::RegisterParams;
use ndarray::Array3;

static SCRATCH_ID: AtomicU64 = AtomicU64::new(0);

struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let unique = SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "atlas-registration-zarr-{}-{name}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn gaussian(center: [f64; 3]) -> Array3<f64> {
    Array3::from_shape_fn((12, 10, 8), |(x, y, z)| {
        let dx = x as f64 - center[0];
        let dy = y as f64 - center[1];
        let dz = z as f64 - center[2];
        (-(dx * dx / 6.0 + dy * dy / 5.0 + dz * dz / 4.0)).exp()
            + 0.3
                * (-((x as f64 - 8.4).powi(2) / 5.0
                    + (y as f64 - 3.2).powi(2) / 3.0
                    + (z as f64 - 5.1).powi(2) / 2.5))
                    .exp()
    })
}

fn geometry(scale: [usize; 3]) -> SpatialFrame {
    SpatialFrame::new(
        [0.0, 0.0, 0.0],
        [scale[0] as f64, scale[1] as f64, scale[2] as f64],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .expect("geometry")
}

fn assert_voxels_same(left: &Voxels, right: &Voxels) {
    assert_eq!(left.dtype(), right.dtype());
    assert_eq!(left.shape(), right.shape());
    for (left, right) in left.widened().iter().zip(right.widened()) {
        assert_eq!(left, &right);
    }
}

#[test]
fn registration_reuses_stored_multiscale_levels_without_mutating_them() {
    let fixed_root = Scratch::new("fixed-store");
    let moving_root = Scratch::new("moving-store");
    let work_before = Scratch::new("before");
    let work_level0 = Scratch::new("level0");
    let work_level1 = Scratch::new("level1");
    let work_after = Scratch::new("after");
    let fixed = gaussian([4.2, 4.1, 3.0]);
    let moving = gaussian([4.95, 3.75, 3.4]);
    let spec = PyramidSpec::powers_of_two(2)
        .expect("spec")
        .with_chunk([4, 4, 4]);
    let fixed_pyramid =
        ZarrEnvironment::build_multiscale(fixed_root.path(), &fixed.clone().into(), &spec)
            .expect("fixed pyramid");
    let moving_pyramid =
        ZarrEnvironment::build_multiscale(moving_root.path(), &moving.clone().into(), &spec)
            .expect("moving pyramid");

    let before = ZarrEnvironment::attach_multiscale_level(work_before.path(), &[&fixed_pyramid], 1)
        .expect("before attach")
        .image(0)
        .expect("before image");
    let mut levels = Vec::new();
    for (level, work) in [(0, &work_level0), (1, &work_level1)] {
        let env = ZarrEnvironment::attach_multiscale_level(
            work.path(),
            &[&fixed_pyramid, &moving_pyramid],
            level,
        )
        .expect("level attaches");
        let scale = fixed_pyramid.scale()[level];
        levels.push(
            VolumePyramidLevel::new(
                env.image(0).expect("fixed level"),
                env.image(ImageId::supplied(0).index())
                    .expect("moving level"),
                scale,
                geometry(scale),
            )
            .expect("registration level"),
        );
    }

    let transform = resident_pyramid_levels(&VolumeFitParams::default(), &levels)
        .expect("stored pyramid registers");
    let moved = transform.apply([4.2, 4.1, 3.0], &RegisterParams::default());
    for (got, want) in moved.into_iter().zip([4.95, 3.75, 3.4]) {
        assert!((got - want).abs() < 0.12, "{got} vs {want}");
    }

    let after = ZarrEnvironment::attach_multiscale_level(work_after.path(), &[&fixed_pyramid], 1)
        .expect("after attach")
        .image(0)
        .expect("after image");
    assert_voxels_same(&before, &after);
}

#[test]
fn registration_builds_missing_zarr_optimisation_levels() {
    let fixed_store = Scratch::new("derived-fixed-store");
    let moving_store = Scratch::new("derived-moving-store");
    let work = Scratch::new("derived-work");
    let fixed = Voxels::from(gaussian([4.2, 4.1, 3.0]));
    let moving = Voxels::from(gaussian([4.95, 3.75, 3.4]));
    let schedule = PyramidSchedule::powers_of_two(2).expect("schedule");

    let levels = zarr_pyramid_levels_or_build(
        &VolumeFitParams::default(),
        VolumePyramidStoreInput::new(&fixed, fixed_store.path()),
        VolumePyramidStoreInput::new(&moving, moving_store.path()),
        &schedule,
        work.path(),
        [4, 4, 4],
    )
    .expect("missing optimisation levels build");

    assert_eq!(levels.len(), 2);
    assert_eq!(levels[0].scale, [1, 1, 1]);
    assert_eq!(levels[1].scale, [2, 2, 2]);
    assert!(fixed_store.path().join("1").join("zarr.json").is_file());
    assert!(moving_store.path().join("1").join("zarr.json").is_file());

    let transform = resident_pyramid_levels(&VolumeFitParams::default(), &levels)
        .expect("derived storage pyramid registers");
    let moved = transform.apply([4.2, 4.1, 3.0], &RegisterParams::default());
    for (got, want) in moved.into_iter().zip([4.95, 3.75, 3.4]) {
        assert!((got - want).abs() < 0.12, "{got} vs {want}");
    }
}
