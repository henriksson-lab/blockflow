use blockflow::env::ArrayEnvironment;
use blockflow::op::{Anchor, BlockOp, SourceInputs};
use blockflow::ops::{
    append_geodesic_level_set_phases, chan_vese_level_set_into, chan_vese_level_set_step_into,
    geodesic_level_set_into, geodesic_level_set_reporting_into, geodesic_level_set_step_into,
    level_set_mask_into, signed_distance_level_set, ChanVeseLevelSetConfig, DistanceParams,
    GeodesicLevelSetConfig, GeodesicLevelSetStepOp,
};
use blockflow::strategy::{execute, Hints};
use blockflow::ImageId;
use blockflow::{BlockGrid, Dtype, PlanBuilder, Voxels};
use ndarray::Array3;

#[test]
fn signed_distance_is_negative_inside_the_mask() {
    let mask = Array3::from_shape_fn((1, 1, 5), |(_, _, k)| k == 2);

    let phi = signed_distance_level_set(mask.view(), &DistanceParams::default()).unwrap();

    assert_eq!(phi[[0, 0, 0]], 2.0);
    assert_eq!(phi[[0, 0, 1]], 1.0);
    assert_eq!(phi[[0, 0, 2]], -1.0);
    assert_eq!(phi[[0, 0, 3]], 1.0);
    assert_eq!(phi[[0, 0, 4]], 2.0);
}

#[test]
fn zero_forcing_leaves_a_flat_field_unchanged() {
    let image = Array3::<f64>::from_elem((3, 3, 3), 7.0);
    let phi = Array3::<f64>::from_elem((3, 3, 3), -2.0);
    let mut out = Array3::<f64>::zeros((3, 3, 3));

    geodesic_level_set_step_into(
        image.view(),
        phi.view(),
        GeodesicLevelSetConfig::default(),
        out.view_mut(),
    )
    .unwrap();

    assert_eq!(out, phi);
}

#[test]
fn balloon_force_moves_a_plane_by_the_configured_step() {
    let image = Array3::<f64>::zeros((1, 1, 5));
    let phi = Array3::from_shape_fn((1, 1, 5), |(_, _, k)| k as f64 - 2.0);
    let mut out = Array3::<f64>::zeros((1, 1, 5));
    let config = GeodesicLevelSetConfig::new(0.25, 1, 0.0, 1.0).unwrap();

    geodesic_level_set_step_into(image.view(), phi.view(), config, out.view_mut()).unwrap();

    assert_eq!(out[[0, 0, 2]], 0.25);
}

#[test]
fn boundary_gradients_use_clamped_faces() {
    let image = Array3::<f64>::zeros((1, 1, 3));
    let phi = Array3::from_shape_vec((1, 1, 3), vec![0.0, 1.0, 2.0]).unwrap();
    let mut out = Array3::<f64>::zeros((1, 1, 3));
    let config = GeodesicLevelSetConfig::new(0.25, 1, 0.0, 1.0).unwrap();

    geodesic_level_set_step_into(image.view(), phi.view(), config, out.view_mut()).unwrap();

    assert!((out[[0, 0, 0]] - 0.125).abs() <= 1.0e-12);
    assert!((out[[0, 0, 1]] - 1.25).abs() <= 1.0e-12);
    assert!((out[[0, 0, 2]] - 2.125).abs() <= 1.0e-12);
}

#[test]
fn centred_level_set_evolves_symmetrically() {
    let image = Array3::<f64>::zeros((5, 5, 5));
    let phi = Array3::from_shape_fn((5, 5, 5), |(i, j, k)| {
        let di = i as f64 - 2.0;
        let dj = j as f64 - 2.0;
        let dk = k as f64 - 2.0;
        (di * di + dj * dj + dk * dk).sqrt() - 1.5
    });
    let mut out = Array3::<f64>::zeros((5, 5, 5));
    let config = GeodesicLevelSetConfig::new(0.1, 1, 0.0, 0.0).unwrap();

    geodesic_level_set_step_into(image.view(), phi.view(), config, out.view_mut()).unwrap();

    assert_eq!(out[[1, 2, 2]], out[[3, 2, 2]]);
    assert_eq!(out[[2, 1, 2]], out[[2, 3, 2]]);
    assert_eq!(out[[2, 2, 1]], out[[2, 2, 3]]);
}

#[test]
fn geodesic_reporting_returns_iterations_and_largest_step_change() {
    let image = Array3::<f64>::zeros((1, 1, 5));
    let phi = Array3::from_shape_fn((1, 1, 5), |(_, _, k)| k as f64 - 2.0);
    let mut out = Array3::<f64>::zeros((1, 1, 5));
    let config = GeodesicLevelSetConfig::new(0.25, 1, 0.0, 1.0).unwrap();

    let report =
        geodesic_level_set_reporting_into(image.view(), phi.view(), config, out.view_mut())
            .unwrap();

    assert_eq!(report.iterations, 1);
    assert!((report.max_delta - 0.25).abs() <= 1.0e-12);
}

#[test]
fn repeated_evolution_matches_repeated_single_steps() {
    let image = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i + 2 * j + 3 * k) as f64);
    let phi = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| i as f64 - j as f64 + k as f64);
    let config = GeodesicLevelSetConfig::new(0.1, 2, 0.5, 0.25).unwrap();
    let mut evolved = Array3::<f64>::zeros((3, 3, 3));
    let mut once = Array3::<f64>::zeros((3, 3, 3));
    let mut twice = Array3::<f64>::zeros((3, 3, 3));

    geodesic_level_set_into(image.view(), phi.view(), config, evolved.view_mut()).unwrap();
    geodesic_level_set_step_into(image.view(), phi.view(), config, once.view_mut()).unwrap();
    geodesic_level_set_step_into(image.view(), once.view(), config, twice.view_mut()).unwrap();

    assert_eq!(evolved, twice);
}

#[test]
fn geodesic_evolution_can_reinitialize_on_a_cadence() {
    let image = Array3::<f64>::zeros((1, 1, 5));
    let phi = Array3::from_shape_vec((1, 1, 5), vec![1.0, -0.2, -0.1, 0.4, 1.0]).unwrap();
    let mut out = Array3::<f64>::zeros((1, 1, 5));
    let mut config = GeodesicLevelSetConfig::new(0.25, 1, 0.0, 0.0).unwrap();
    config.reinitialize_every = Some(1);

    geodesic_level_set_into(image.view(), phi.view(), config, out.view_mut()).unwrap();

    assert_eq!(
        out,
        Array3::from_shape_vec((1, 1, 5), vec![1.0, -1.0, -1.0, 1.0, 2.0]).unwrap()
    );
}

#[test]
fn block_op_step_matches_the_resident_step() {
    let image = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i + 2 * j + 3 * k) as f64);
    let phi = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| i as f64 - j as f64 + k as f64);
    let config = GeodesicLevelSetConfig::new(0.1, 1, 0.5, 0.25).unwrap();
    let op = GeodesicLevelSetStepOp::new("level-set-step", 7usize, config).unwrap();
    let mut expected = Array3::<f64>::zeros((3, 3, 3));
    let mut got = Voxels::F64(Array3::<f64>::zeros((3, 3, 3)));
    let input = Voxels::F64(phi.clone());
    let source = Voxels::F64(image.clone());
    let entries = [(7usize.into(), &source)];

    geodesic_level_set_step_into(image.view(), phi.view(), config, expected.view_mut()).unwrap();
    op.apply_with(
        &input,
        SourceInputs::new(&entries),
        &mut got,
        &Anchor::whole([3, 3, 3]),
    )
    .unwrap();

    assert_eq!(got.view::<f64>().unwrap(), expected.view());
    assert_eq!(op.reach(0, 3), 2);
    assert_eq!(op.source_inputs([3, 3, 3]).len(), 1);
}

#[test]
fn geodesic_builder_appends_one_phase_per_step() {
    let grid = BlockGrid::new([8, 8, 8], [4, 4, 4]).unwrap();
    let mut builder = PlanBuilder::new([8, 8, 8], Dtype::F64, grid);
    let config = GeodesicLevelSetConfig::new(0.1, 3, 0.5, 0.0).unwrap();
    let image = ImageId::supplied(0);

    let phases = append_geodesic_level_set_phases(&mut builder, image, config).unwrap();
    let assembly = builder.finish().unwrap();

    assert_eq!(phases.len(), 3);
    assert_eq!(assembly.decomposition.phases.len(), 3);
    for phase in &assembly.decomposition.phases {
        assert_eq!(phase.source_images, vec![image.index()]);
    }
}

#[test]
fn geodesic_planner_helper_matches_the_resident_reference_on_split_grids() {
    let shape = [6, 5, 4];
    let image = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (i * i + 2 * j + 3 * k) as f64 / 17.0
    });
    let phi = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        i as f64 - 0.5 * j as f64 + 0.25 * k as f64 - 2.0
    });
    let config = GeodesicLevelSetConfig::new(0.05, 2, 0.25, 0.1).unwrap();
    let source = ImageId::supplied(0);
    let mut expected = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));

    geodesic_level_set_into(image.view(), phi.view(), config, expected.view_mut()).unwrap();

    for block in [[6, 5, 4], [3, 3, 2]] {
        let grid = BlockGrid::new(shape, block).unwrap();
        let mut builder = PlanBuilder::new(shape, Dtype::F64, grid);
        append_geodesic_level_set_phases(&mut builder, source, config).unwrap();
        let assembly = builder.finish().unwrap();
        let env = ArrayEnvironment::with_inputs(
            phi.clone().into(),
            vec![image.clone().into()],
            &assembly.decomposition,
            [3, 3, 2],
        )
        .unwrap();

        execute(
            "level-set",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints::default(),
            &env,
        )
        .unwrap();

        assert_eq!(env.output().view::<f64>().unwrap(), expected.view());
    }
}

#[test]
fn geodesic_builder_refuses_reinitialization_it_cannot_express() {
    let grid = BlockGrid::new([8, 8, 8], [4, 4, 4]).unwrap();
    let mut builder = PlanBuilder::new([8, 8, 8], Dtype::F64, grid);
    let mut config = GeodesicLevelSetConfig::new(0.1, 1, 0.5, 0.0).unwrap();
    config.reinitialize_every = Some(1);

    assert!(append_geodesic_level_set_phases(&mut builder, 0usize, config).is_err());
}

#[test]
fn chan_vese_step_uses_current_inside_and_outside_means() {
    let image = Array3::from_shape_vec((1, 1, 4), vec![2.0, 2.0, 5.0, 8.0]).unwrap();
    let phi = Array3::from_shape_vec((1, 1, 4), vec![-1.0, -1.0, 1.0, 1.0]).unwrap();
    let mut out = Array3::<f64>::zeros((1, 1, 4));
    let mut config = ChanVeseLevelSetConfig::new(0.5, 1, 0.0, 1.0, 1.0).unwrap();
    config.smoothing_epsilon = 1.0;

    chan_vese_level_set_step_into(image.view(), phi.view(), config, out.view_mut()).unwrap();

    let delta = 1.0 / (std::f64::consts::PI * 2.0);
    let expected = -1.0 + 0.5 * delta * (-(2.0f64 - 2.0).powi(2) + (2.0f64 - 6.5).powi(2));
    assert!((out[[0, 0, 0]] - expected).abs() <= 1.0e-12);
}

#[test]
fn chan_vese_repeated_evolution_matches_repeated_single_steps() {
    let image = Array3::from_shape_vec((1, 1, 5), vec![1.0, 1.0, 3.0, 6.0, 6.0]).unwrap();
    let phi = Array3::from_shape_vec((1, 1, 5), vec![-1.0, -0.5, 0.5, 1.0, 1.5]).unwrap();
    let config = ChanVeseLevelSetConfig::new(0.1, 2, 0.0, 1.0, 1.0).unwrap();
    let mut evolved = Array3::<f64>::zeros((1, 1, 5));
    let mut once = Array3::<f64>::zeros((1, 1, 5));
    let mut twice = Array3::<f64>::zeros((1, 1, 5));

    chan_vese_level_set_into(image.view(), phi.view(), config, evolved.view_mut()).unwrap();
    chan_vese_level_set_step_into(image.view(), phi.view(), config, once.view_mut()).unwrap();
    chan_vese_level_set_step_into(image.view(), once.view(), config, twice.view_mut()).unwrap();

    assert_eq!(evolved, twice);
}

#[test]
fn chan_vese_refuses_a_missing_region() {
    let image = Array3::<f64>::zeros((1, 1, 2));
    let phi = Array3::<f64>::from_elem((1, 1, 2), -1.0);
    let mut out = Array3::<f64>::zeros((1, 1, 2));

    assert!(chan_vese_level_set_step_into(
        image.view(),
        phi.view(),
        ChanVeseLevelSetConfig::default(),
        out.view_mut()
    )
    .is_err());
}

#[test]
fn mask_conversion_uses_the_documented_inside_convention() {
    let phi = Array3::from_shape_vec((1, 1, 4), vec![-1.0, 0.0, 0.5, 2.0]).unwrap();
    let mut out = Array3::<bool>::from_elem((1, 1, 4), false);

    level_set_mask_into(phi.view(), out.view_mut()).unwrap();

    assert_eq!(
        out,
        Array3::from_shape_vec((1, 1, 4), vec![true, true, false, false]).unwrap()
    );
}

#[test]
fn invalid_parameters_and_non_finite_fields_are_refused() {
    assert!(GeodesicLevelSetConfig::new(0.0, 1, 0.0, 0.0).is_err());
    assert!(GeodesicLevelSetConfig::new(0.5001, 1, 0.0, 0.0).is_err());

    let image = Array3::<f64>::zeros((1, 1, 1));
    let phi = Array3::from_elem((1, 1, 1), f64::NAN);
    let mut out = Array3::<f64>::zeros((1, 1, 1));

    assert!(geodesic_level_set_step_into(
        image.view(),
        phi.view(),
        GeodesicLevelSetConfig::default(),
        out.view_mut(),
    )
    .is_err());
}
