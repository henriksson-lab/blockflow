// SPDX-License-Identifier: MIT
// Original work for this crate.

// Two real autofluorescence volumes from different brains, registered at a
// 50 µm lattice. A measurement, not a gate on synthetic data, so it is ignored
// by default and needs both images:
//
// ```text
// REGISTER_FIXED_ZARR=/path/to/a/autofluorescence.zarr \
// REGISTER_MOVING_ZARR=/path/to/b/autofluorescence.zarr \
//     cargo test --release -p blockflow-atlas-registration --test real_autofluorescence \
//     -- --ignored --nocapture
// ```

use std::path::PathBuf;
use std::time::Instant;

use blockflow::iterate::SubstageLimit;
use blockflow_atlas_registration::compute::{ControlGrid, Metric, VolumeFitParams};
use blockflow_atlas_registration::pipeline::{register, Input, RegisterConfig};

fn image(variable: &str) -> PathBuf {
    PathBuf::from(std::env::var(variable).unwrap_or_else(|_| panic!("{variable} is not set")))
}

fn stage(params: VolumeFitParams) -> VolumeFitParams {
    let mut params = params.with_metric(Metric::NormalizedCorrelation);
    params.limit = SubstageLimit::of(64).unwrap();
    params.tolerance = 1e-2;
    params
}

#[test]
#[ignore = "needs two OME-Zarr autofluorescence volumes: set REGISTER_FIXED_ZARR and REGISTER_MOVING_ZARR"]
fn a_deformable_stage_improves_on_the_rigid_start_between_two_brains() {
    let out = std::env::temp_dir().join(format!("atlas-registration-real-{}", std::process::id()));
    let config = |stages: Vec<VolumeFitParams>| RegisterConfig {
        fixed: Input {
            path: image("REGISTER_FIXED_ZARR"),
            channel: 0,
            level: None,
        },
        moving: Input {
            path: image("REGISTER_MOVING_ZARR"),
            channel: 0,
            level: None,
        },
        resolution: Some([50.0; 3]),
        stages: stages.into_iter().map(Into::into).collect(),
        out: out.clone(),
        warped: None,
        warp_source: None,
        nearest: false,
        block: 64,
        workers: rayon::current_num_threads(),
        warped_stages: false,
        initialisation: Default::default(),
    };
    let start = Instant::now();
    let translation = register(&config(vec![stage(VolumeFitParams::default())])).unwrap();
    let rigid_seconds = start.elapsed().as_secs_f64();
    let mut affine = stage(VolumeFitParams::default().affine());
    affine.max_step = 0.05;
    let mut bspline = stage(
        VolumeFitParams::default()
            .bspline()
            .with_bspline_grid(ControlGrid::new([6, 6, 6]).unwrap()),
    );
    bspline.max_step = 0.5;
    let start = Instant::now();
    let deformable = register(&config(vec![affine, bspline])).unwrap();
    let deformable_seconds = start.elapsed().as_secs_f64();
    let _ = std::fs::remove_dir_all(&out);

    let rigid_cost = translation.stages[0].final_cost.unwrap();
    let final_cost = deformable.stages.last().unwrap().final_cost.unwrap();
    eprintln!(
        "lattice {:?}: translation correlation {:.4} in {rigid_seconds:.1} s; affine then \
         B-spline {:.4} in {deformable_seconds:.1} s",
        translation.lattice_shape, -rigid_cost, -final_cost
    );
    assert!(
        final_cost < rigid_cost,
        "the deformable chain ({final_cost}) does not improve on the translation ({rigid_cost})"
    );
}
