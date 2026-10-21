use blockflow::{
    assemble::PlanBuilder,
    env::ArrayEnvironment,
    geometry::BlockGrid,
    op::{Anchor, BlockOp, Placement, SourceInputs},
    ops::plane_sample::{
        append_plane_sample_phase, sample_plane, trilinear, PhysicalGrid, PlaneGrid, PlaneSampleOp,
    },
    strategy::{execute_phases, Hints},
    Dtype, Voxels,
};
use ndarray::Array3;

fn oblique() -> (PhysicalGrid, PlaneGrid) {
    let s = 0.5_f64.sqrt();
    (
        PhysicalGrid::new([10., -5., 8.], [2., 0.7, 1.3]).unwrap(),
        PlaneGrid::new(
            [28., 2., 21.],
            [s, s, 0.],
            [0., 0., 1.],
            [s, -s, 0.],
            [9, 13],
            [0.8, 1.1],
        )
        .unwrap(),
    )
}
#[test]
fn analytic_ramp_and_physical_frame() {
    let (grid, plane) = oblique();
    let depth = 1.7;
    let samples = sample_plane(&grid, &plane, depth, [24, 32, 25], |q| {
        let p =
            std::array::from_fn::<_, 3, _>(|a| grid.origin()[a] + q[a] as f64 * grid.spacing()[a]);
        2. * p[0] - 3. * p[1] + 0.5 * p[2] + 7.
    })
    .unwrap();
    let s = 0.5_f64.sqrt();
    for r in 0..9 {
        for c in 0..13 {
            let dr = (r as f64 - 4.) * 0.8;
            let dc = (c as f64 - 6.) * 1.1;
            let p = [28. + s * dr + s * depth, 2. + s * dr - s * depth, 21. + dc];
            let expected = 2. * p[0] - 3. * p[1] + 0.5 * p[2] + 7.;
            assert!((samples[r * 13 + c] - expected).abs() < 1e-10);
        }
    }
    assert!(sample_plane(&grid, &plane, depth, [24, 32, 25], |_| 7.)
        .unwrap()
        .iter()
        .all(|x| (x - 7.).abs() < 1e-10));
}
#[test]
fn strict_boundaries_support_and_nonfinite() {
    let sh = [3, 4, 5];
    for p in [
        [-0.001, 0., 0.],
        [2.001, 0., 0.],
        [f64::NAN, 0., 0.],
        [0., f64::INFINITY, 0.],
    ] {
        assert!(trilinear(sh, p, |_| panic!("outside sample read source")).is_none());
    }
    assert_eq!(trilinear([0, 4, 5], [0.; 3], |_| 1.), None);
    assert_eq!(trilinear(sh, [2., 3., 4.], |_| 9.), Some(9.));
    assert_eq!(
        trilinear(sh, [0.25, 1., 1.], |q| if q[0] == 0 { 1. } else { 0. }),
        Some(0.75)
    );
    assert_eq!(
        trilinear(sh, [0.5, 1., 1.], |q| if q[0] == 0 { 1. } else { f64::NAN }),
        None
    );
    assert_eq!(
        trilinear(sh, [0., 1., 1.], |q| if q[0] == 0 { 1. } else { f64::NAN }),
        Some(1.)
    );
    let g = PhysicalGrid::new([0.; 3], [1.; 3]).unwrap();
    let p = PlaneGrid::new(
        [0.; 3],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [3, 3],
        [1.; 2],
    )
    .unwrap();
    let out = sample_plane(&g, &p, -0.01, sh, |_| 1.).unwrap();
    assert!(out.iter().all(|x| x.is_nan()));
    assert!(sample_plane(&g, &p, f64::NAN, sh, |_| 1.).is_err());
}
#[test]
fn invalid_frames_are_refused() {
    assert!(PhysicalGrid::new([0.; 3], [1., 0., 1.]).is_err());
    assert!(PhysicalGrid::new([f64::NAN, 0., 0.], [1.; 3]).is_err());
    for (normal, shape, spacing) in [
        ([2., 0., 0.], [2, 3], [1.; 2]),
        ([1., 0., 0.], [0, 3], [1.; 2]),
        ([1., 0., 0.], [2, 3], [-1., 1.]),
    ] {
        assert!(
            PlaneGrid::new([0.; 3], [0., 1., 0.], [0., 0., 1.], normal, shape, spacing).is_err()
        );
    }
}
fn planned<T: blockflow::voxels::VoxelElement>(
    input: Array3<T>,
    grid: PhysicalGrid,
    plane: PlaneGrid,
    depth: f64,
    block: [usize; 3],
) -> (Vec<f64>, usize)
where
    Voxels: From<Array3<T>>,
{
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let output = [1, plane.shape()[0], plane.shape()[1]];
    let op = PlaneSampleOp::new(grid.clone(), plane.clone(), depth).unwrap();
    let output_grid = BlockGrid::new(output, block).unwrap();
    let phase = op.phase(shape, output_grid.clone()).unwrap();
    let max_fetch = phase
        .blocks
        .iter()
        .map(|b| b.source.voxels())
        .max()
        .unwrap();
    // Inspect every contributing interpolation corner independently of the
    // affine footprint planner. Assert its actual global fetch contains it.
    for b in &phase.blocks {
        for r in b.core.start[1]..b.core.start[1] + b.core.shape[1] {
            for c in b.core.start[2]..b.core.start[2] + b.core.shape[2] {
                trilinear(shape, plane.coordinate(&grid, r, c, depth), |q| {
                    for a in 0..3 {
                        assert!(
                            q[a] >= b.source.start[a]
                                && q[a] < b.source.start[a] + b.source.shape[a]
                        );
                    }
                    1.
                });
            }
        }
    }
    let mut builder = PlanBuilder::new(shape, T::DTYPE, BlockGrid::whole(shape).unwrap());
    append_plane_sample_phase(&mut builder, op, shape, output_grid).unwrap();
    let assembly = builder.finish().unwrap();
    let env =
        ArrayEnvironment::for_decomposition(input.into(), &assembly.decomposition, [4; 3]).unwrap();
    execute_phases(
        "physical-plane-test",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .unwrap();
    let out = env.image(1);
    assert_eq!(out.dtype(), Dtype::F64);
    (
        out.view::<f64>().unwrap().iter().copied().collect(),
        max_fetch,
    )
}
#[test]
fn executor_matches_resident_across_grids_and_dtypes() {
    let (g, p) = oblique();
    let sh = [24, 32, 25];
    let input = Array3::from_shape_fn(sh, |q| (2 * q.0 + 3 * q.1 + q.2) as f32 + 0.125);
    let reference = sample_plane(&g, &p, 1.7, sh, |q| input[q] as f64).unwrap();
    for block in [[1, 9, 13], [1, 4, 5], [1, 1, 1]] {
        let (actual, max_fetch) = planned(input.clone(), g.clone(), p.clone(), 1.7, block);
        for (a, b) in actual.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-10);
        }
        assert!(max_fetch < sh.iter().product());
    }
    let support = input.mapv(|x| x > 60.);
    let reference = sample_plane(&g, &p, 1.7, sh, |q| f64::from(support[q])).unwrap();
    let (actual, _) = planned(support, g, p, 1.7, [1, 3, 4]);
    for (a, b) in actual.iter().zip(&reference) {
        assert!((a - b).abs() < 1e-10);
    }
}
#[test]
fn executor_handles_outside_and_nonfinite_samples() {
    let sh = [8, 9, 10];
    let g = PhysicalGrid::new([0.; 3], [1.; 3]).unwrap();
    let p = PlaneGrid::new(
        [0.5, 2., 3.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [9, 11],
        [1.; 2],
    )
    .unwrap();
    let input = Array3::from_shape_fn(sh, |q| if q == (1, 2, 3) { f64::NAN } else { 4. });
    let reference = sample_plane(&g, &p, 0., sh, |q| input[q]).unwrap();
    for block in [[1, 9, 11], [1, 3, 4], [1, 1, 1]] {
        let (actual, _) = planned(input.clone(), g.clone(), p.clone(), 0., block);
        for (a, b) in actual.iter().zip(&reference) {
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-10);
        }
    }
}
#[test]
fn understated_source_fetch_fails_loudly() {
    let g = PhysicalGrid::new([0.; 3], [1.; 3]).unwrap();
    let p = PlaneGrid::new(
        [4., 4., 4.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [3, 3],
        [1.; 2],
    )
    .unwrap();
    let op = PlaneSampleOp::new(g, p, 0.).unwrap();
    let input = Voxels::from(Array3::from_elem([2; 3], 1_f32));
    let mut output = Voxels::from(Array3::<f64>::zeros([1, 3, 3]));
    let mut at = Placement::same(Anchor::whole([1, 3, 3])).writing([1, 3, 3]);
    at.input = Anchor::new([0; 3], [10; 3]);
    assert!(op
        .apply_placed(&input, SourceInputs::none(), &mut output, &at)
        .unwrap_err()
        .to_string()
        .contains("fetch"));
}
