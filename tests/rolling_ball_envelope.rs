use blockflow::{
    assemble::PlanBuilder,
    dtype::Dtype,
    env::ArrayEnvironment,
    geometry::BlockGrid,
    ops::{
        components::Connectivity,
        envelope::{
            append_rolling_ball_envelope_phases, rolling_ball_envelope, RollingBallEnvelope,
        },
    },
    sidecar::Lifecycle,
    strategy::{execute_phases, Hints},
    voxels::Voxels,
};
use ndarray::Array3;
use std::collections::VecDeque;

// Independent exhaustive metric + boundary BFS + ball-union oracle. No EDT or
// component primitive from the implementation is used here.
fn oracle(mask: &Array3<bool>, cfg: RollingBallEnvelope) -> Array3<bool> {
    let s = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let points: Vec<_> = (0..s[0])
        .flat_map(|z| (0..s[1]).flat_map(move |y| (0..s[2]).map(move |x| [z, y, x])))
        .collect();
    let d2 = |a: [usize; 3], b: [usize; 3]| -> f64 {
        (0..3)
            .map(|i| ((a[i] as f64 - b[i] as f64) * cfg.sampling()[i]).powi(2))
            .sum()
    };
    let r2 = cfg.radius().powi(2);
    let allowed = Array3::from_shape_fn(s, |p| {
        let p = [p.0, p.1, p.2];
        points.iter().all(|&q| !mask[q] || d2(p, q) > r2)
    });
    let mut outside = Array3::from_elem(s, false);
    let mut queue = VecDeque::new();
    for &p in &points {
        if allowed[p] && (0..3).any(|i| s[i] > 1 && (p[i] == 0 || p[i] + 1 == s[i])) {
            outside[p] = true;
            queue.push_back(p);
        }
    }
    while let Some(p) = queue.pop_front() {
        for offset in cfg.connectivity().offsets() {
            let q = [
                p[0] as isize + offset[0],
                p[1] as isize + offset[1],
                p[2] as isize + offset[2],
            ];
            if (0..3).all(|i| q[i] >= 0 && (q[i] as usize) < s[i]) {
                let q = q.map(|v| v as usize);
                if allowed[q] && !outside[q] {
                    outside[q] = true;
                    queue.push_back(q);
                }
            }
        }
    }
    Array3::from_shape_fn(s, |p| {
        let p = [p.0, p.1, p.2];
        !points.iter().any(|&q| outside[q] && d2(p, q) <= r2)
    })
}
fn chamber(width: usize) -> Array3<bool> {
    let mut a = Array3::from_shape_fn((1, 31, 31), |(_, y, x)| {
        (4..27).contains(&y) && (4..27).contains(&x)
    });
    for y in 10..21 {
        for x in 10..21 {
            a[[0, y, x]] = false;
        }
    }
    for y in 4..11 {
        for x in (15 - width / 2)..=(15 + width / 2) {
            a[[0, y, x]] = false;
        }
    }
    a
}
#[test]
fn narrow_mouth_seals_but_wide_mouth_admits_the_ball() {
    let cfg = RollingBallEnvelope::new(1., [1.; 3]).unwrap();
    let narrow = chamber(1);
    let wide = chamber(5);
    let n = rolling_ball_envelope(narrow.view(), cfg).unwrap();
    let w = rolling_ball_envelope(wide.view(), cfg).unwrap();
    assert!(n[[0, 15, 15]]);
    assert!(!w[[0, 15, 15]]);
    assert!(
        !rolling_ball_envelope(
            narrow.view(),
            RollingBallEnvelope::new(0., [1.; 3]).unwrap()
        )
        .unwrap()[[0, 15, 15]]
    );
    // Same voxel channel, different physical width: x pitch makes it traversable.
    let physical = rolling_ball_envelope(
        narrow.view(),
        RollingBallEnvelope::new(1., [1., 1., 4.]).unwrap(),
    )
    .unwrap();
    assert!(!physical[[0, 15, 15]]);
}
#[test]
fn agrees_with_exhaustive_oracle_and_preserves_all_input_points() {
    for s in [[5, 7, 9], [1, 13, 11]] {
        let mask = Array3::from_shape_fn(s, |(z, y, x)| {
            z > 0
                && z + 1 < s[0]
                && y > 1
                && y + 2 < s[1]
                && x > 1
                && x + 2 < s[2]
                && ((z * 17 + y * 11 + x * 7) % 9 < 5)
                || s[0] == 1 && y > 1 && y + 2 < s[1] && x > 1 && x + 2 < s[2] && (y + x) % 4 != 0
        });
        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            for r in [0., 0.9, 1.3, 2.7] {
                let cfg = RollingBallEnvelope::new(r, [2., 1., 0.7])
                    .unwrap()
                    .connecting(connectivity);
                let got = rolling_ball_envelope(mask.view(), cfg).unwrap();
                assert_eq!(
                    got,
                    oracle(&mask, cfg),
                    "shape {s:?}, radius {r}, connectivity {connectivity:?}"
                );
                assert!(mask.iter().zip(got.iter()).all(|(&m, &e)| !m || e));
            }
        }
    }
}
#[test]
fn empty_full_and_closed_cavity_have_defined_answers() {
    let cfg = RollingBallEnvelope::new(2., [1.; 3]).unwrap();
    for v in [false, true] {
        let a = Array3::from_elem((7, 9, 11), v);
        assert_eq!(rolling_ball_envelope(a.view(), cfg).unwrap(), a);
    }
    let mut a = chamber(1);
    a[[0, 4, 15]] = true;
    let z =
        rolling_ball_envelope(a.view(), RollingBallEnvelope::new(0., [1.; 3]).unwrap()).unwrap();
    assert!(z[[0, 15, 15]]);
    assert!(!z[[0, 0, 0]]);
}
#[test]
fn planned_result_matches_oracle_across_component_seams_and_distance_grids() {
    let a = Array3::from_shape_fn((9, 13, 15), |(z, y, x)| {
        (2..7).contains(&z) && (2..11).contains(&y) && (2..13).contains(&x) && !(y == 6 && x > 5)
    });
    for connectivity in [Connectivity::Faces, Connectivity::FacesEdgesAndCorners] {
        let cfg = RollingBallEnvelope::new(1.3, [1.5, 1., 0.8])
            .unwrap()
            .connecting(connectivity);
        let want = oracle(&a, cfg);
        for block in [3, 7, 20] {
            let shape = [9, 13, 15];
            let mut b = PlanBuilder::new(
                shape,
                Dtype::Bool,
                BlockGrid::new(shape, [block; 3]).unwrap(),
            );
            let p = append_rolling_ball_envelope_phases(
                &mut b,
                cfg,
                block,
                "test.envelope",
                Lifecycle::DeleteOnExit,
            )
            .unwrap();
            let assembly = b.finish().unwrap();
            let env = ArrayEnvironment::with_inputs(
                Voxels::from(a.clone()),
                vec![],
                &assembly.decomposition,
                [4; 3],
            )
            .unwrap();
            execute_phases(
                "envelope",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints::default(),
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            assert_eq!(
                env.image(p.image().unwrap().index())
                    .view::<bool>()
                    .unwrap(),
                want.view(),
                "block {block}"
            );
        }
    }
}
#[test]
fn validation_precedes_plan_mutation() {
    for (r, p) in [
        (f64::NAN, [1.; 3]),
        (-1., [1.; 3]),
        (1., [0., 1., 1.]),
        (1., [1e200, 1., 1.]),
        (1., [1e-200, 1., 1.]),
    ] {
        assert!(RollingBallEnvelope::new(r, p).is_err());
    }
    let cfg = RollingBallEnvelope::new(1., [1.; 3]).unwrap();
    let shape = [5; 3];
    for (dtype, block, stream) in [
        (Dtype::F64, 3, "s"),
        (Dtype::Bool, 0, "s"),
        (Dtype::Bool, 3, ""),
        (Dtype::Bool, 3, "bad/name"),
    ] {
        let mut b = PlanBuilder::new(shape, dtype, BlockGrid::new(shape, [3; 3]).unwrap());
        assert!(append_rolling_ball_envelope_phases(
            &mut b,
            cfg,
            block,
            stream,
            Lifecycle::DeleteOnExit
        )
        .is_err());
        assert_eq!(b.n_phases(), 0);
    }
    assert!(rolling_ball_envelope(Array3::from_elem((1, 1, 1), false).view(), cfg).is_err());
}

#[test]
fn planned_zero_radius_and_absent_distance_seeds_match_resident() {
    for a in [
        Array3::from_elem((1, 13, 15), false),
        Array3::from_elem((1, 13, 15), true),
        chamber(1),
    ] {
        for radius in [0., 1.] {
            let cfg = RollingBallEnvelope::new(radius, [1.; 3]).unwrap();
            let want = rolling_ball_envelope(a.view(), cfg).unwrap();
            let shape = [a.shape()[0], a.shape()[1], a.shape()[2]];
            let mut b =
                PlanBuilder::new(shape, Dtype::Bool, BlockGrid::new(shape, [5; 3]).unwrap());
            let phase = append_rolling_ball_envelope_phases(
                &mut b,
                cfg,
                5,
                "test.empty",
                Lifecycle::DeleteOnExit,
            )
            .unwrap();
            let assembly = b.finish().unwrap();
            let env = ArrayEnvironment::with_inputs(
                Voxels::from(a.clone()),
                vec![],
                &assembly.decomposition,
                [4; 3],
            )
            .unwrap();
            execute_phases(
                "envelope empty",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints::default(),
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            assert_eq!(
                env.image(phase.image().unwrap().index())
                    .view::<bool>()
                    .unwrap(),
                want.view()
            );
        }
    }
}
