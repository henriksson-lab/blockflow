// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **`ops::align` against transforms whose answer is known before the optimiser
// starts — and against the several places where the answer it returns is not the
// one it advertises.**
//
// The gap this file closes
// ------------------------
// What stood here was thirty-seven lines: two Gaussian blobs one voxel apart,
// `run`, and `assert!((params[0] - 1.0).abs() <= 0.25)`. The first test below
// measures what that assertion was worth. The dense moment initialiser — the
// difference of the two positive-mass centroids, computed at substage 0 before
// any Gauss-Newton step is taken — puts the parameters at
// `[0.9933, -0.9441, -8.9e-16]` on that fixture, whose worst axis error is
// **0.056 against a bound of 0.25**. The whole optimisation loop could have been
// deleted and the file would have passed. Nothing in the tree executed
// `TransformModel::Affine`, `TransformModel::BSpline`, `Metric::NCC`,
// `Metric::MutualInformation`, `solve_linear`, the trilinear gradient,
// `ControlGrid`, `PyramidSchedule`, or `SpatialFrame` at all.
//
// The oracle
// ----------
// Not a recorded output. A **known transform, applied**. One synthetic scene is
// rendered once at `[48, 40, 36]`; the fixed and moving images are two windows
// cut out of it, and the transform between them is the one this file chose when
// it cut them. `run` is then asked to find it. That single fixture exercises the
// metric, the interpolator, the parameter Jacobian and the linear solve at once,
// and it has an exact answer: for an integral shift the two windows are the same
// voxels, so the mean-squares minimum is `0` and the recovered parameters agree
// with the chosen ones **to 1e-13**.
//
// | claim | how |
// |---|---|
// | the shipped assertion was satisfied by the seed | the seed computed here, from the definition, and shown inside the shipped bound; then the fit measured to beat it by `1.0e16` in mean squares |
// | an integral translation is recovered | window arithmetic, `1.3e-13`, against a seed that is `3.4` voxels away — and identical across five block edges and two worker counts |
// | a twelve-parameter affine is recovered | a known non-symmetric `A` and `b` resampled into the fixed image, recovered to `9.9e-15` |
// | normalized correlation recovers it too, **but not at the shipped limit** | the default configuration is a hard error at 16 substages; it needs 27, and then lands `0.021` from the truth |
// | **mutual information never moves** | the returned parameters are the moment initialiser, `4.8e-5` away from it and `3.4` voxels from the answer |
// | a B-spline recovers a uniform shift as a uniform field | 192 coefficients, all 64 on the shifted axis within `0.008` of the shift — and only with a non-zero smoothness, which is recorded |
// | the coarse pyramid level seeds the level below it | a hand-built two-level stack whose coarse level answers `0.574`; the pair answers `2 x 0.574` to `1e-12`, and not the fine level's own initialiser 1.5 voxels away |
// | **an affine seed used to be reduced to its translation** | the regression for the fix in `src/ops/align.rs`, read out with a one-step run |
// | **more pyramid levels can lose a transform one level recovers** | a 3-voxel shift, exact at one level, a hard error at two |
// | **the spatial frame's spacing is validated and then dropped** | an anisotropic frame produces bit-identical parameters to a unit one |
//
// What is deliberately not asserted
// ---------------------------------
// That the optimiser is a *good* registration. It is not: a fractional
// translation of a textured volume does not converge at all under the shipped
// stopping rule, at any resolution, which
// `a_fractional_translation_of_a_textured_volume_does_not_converge` records.
// This file measures what the code does against the transforms it was handed,
// and names the boundary where that stops being an answer.
//
// The numerical kernels underneath — `solve_linear`, the trilinear gradient and
// the parameter Jacobian's chain rule — are private, and their tests are in
// `src/ops/align.rs`'s own `mod tests`, against finite differences.

use blockflow::iterate::SubstageLimit;
use blockflow::ops::align::{
    plan, pyramid_levels_from_recipe, resident, resident_pyramid_levels, run, run_with_diagnostics,
    run_with_seed_diagnostics, run_with_workers, ControlGrid, Metric, PyramidSchedule, Sampling,
    SpatialFrame, TransformModel, VolumeFitParams, VolumePyramidLevel,
};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use ndarray::Array3;

// ------------------------------------------------------------- fixtures --

/// The scene the two windows are cut out of.
///
/// Three different extents, so no two axes are interchangeable and a transposed
/// answer cannot pass. It is **larger than the window on every side** — the
/// margin is 10 voxels low and at least 12 high — because that is what makes the
/// window arithmetic below exact: a window displaced by up to five voxels, or
/// carried through an affine that drifts up to four, still reads scene voxels
/// that exist, so the fixed and moving images are related by exactly the chosen
/// transform and by nothing else. Anything smaller and the comparison would be
/// partly a statement about what happens at a clipped edge.
const SCENE: [usize; 3] = [48, 40, 36];

/// The window both images are cut at. 7680 voxels: large enough that the
/// positive-mass centroid is a stable number and that 64 B-spline control points
/// per axis are all supported by real data, small enough that the 71-substage
/// B-spline fit and the 27-substage correlation fit are each well under a second.
const WINDOW: [usize; 3] = [24, 20, 16];

/// Where the *fixed* window sits in the scene. Off-centre and unequal per axis,
/// for the same reason the extents are unequal.
const ORIGIN: [f64; 3] = [10.0, 8.0, 8.0];

/// A textured volume, and the texture is the point.
///
/// Sixty objects of radius 1.5 to 4 voxels means intensity structure on the same
/// scale as the shifts being recovered, which is what gives the metric a
/// gradient to follow and what makes the trilinear interpolator's cell
/// boundaries something the optimiser actually crosses. A single smooth blob —
/// what the file that stood here used — makes the mean-squares cost nearly
/// quadratic and cannot tell a working optimiser from a working initialiser.
///
/// Noise `0.01` rather than zero so the images are not analytically smooth; the
/// two windows are cut from **one** rendered array, so they share the noise
/// realisation and an integral shift is still an exact relation between them.
/// The seed is fixed, so the scene is a function of its shape and seed alone.
fn scene() -> Array3<f64> {
    Scene::new(
        SceneSpec::new(SCENE, 20260830)
            .with_objects(60)
            .with_radius(1.5, 4.0)
            .with_noise(0.01),
    )
    .expect("a well-formed scene specification")
    .render()
    .intensity
}

/// Trilinear sampling, written out here rather than reused from the crate,
/// because the windows this file cuts are the *ground truth* the crate's own
/// sampler is measured against. Out of range is `None` and never a silent zero —
/// every caller below either stays in range by construction or checks.
fn trilinear(volume: &Array3<f64>, point: [f64; 3]) -> Option<f64> {
    let shape = volume.shape();
    let base = [point[0].floor(), point[1].floor(), point[2].floor()];
    for axis in 0..3 {
        if base[axis] < 0.0 || base[axis] as usize + 1 >= shape[axis] {
            return None;
        }
    }
    let cell = [base[0] as usize, base[1] as usize, base[2] as usize];
    let frac = [point[0] - base[0], point[1] - base[1], point[2] - base[2]];
    let mut value = 0.0;
    for (i, wi) in [(0usize, 1.0 - frac[0]), (1, frac[0])] {
        for (j, wj) in [(0usize, 1.0 - frac[1]), (1, frac[1])] {
            for (k, wk) in [(0usize, 1.0 - frac[2]), (1, frac[2])] {
                value += wi * wj * wk * volume[[cell[0] + i, cell[1] + j, cell[2] + k]];
            }
        }
    }
    Some(value)
}

/// The window of `shape` whose local origin sits at `offset` in the scene.
fn window(volume: &Array3<f64>, shape: [usize; 3], offset: [f64; 3]) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(x, y, z)| {
        trilinear(
            volume,
            [
                x as f64 + offset[0],
                y as f64 + offset[1],
                z as f64 + offset[2],
            ],
        )
        .expect("every window this file cuts lies inside the scene")
    })
}

/// A fixed/moving pair whose answer is `shift`.
///
/// `fixed[x] = scene[x + ORIGIN]` and `moving[x] = scene[x + ORIGIN - shift]`,
/// so `moving[x + shift] = fixed[x]` and the transform the optimiser is asked
/// for — the map from a fixed-image point to the moving-image point that carries
/// its value — is exactly `x -> x + shift`. For an integral shift the two
/// windows are the same scene voxels and the relation is exact in binary.
fn shifted_pair(scene: &Array3<f64>, shift: [f64; 3]) -> (Array3<f64>, Array3<f64>) {
    (
        window(scene, WINDOW, ORIGIN),
        window(
            scene,
            WINDOW,
            [
                ORIGIN[0] - shift[0],
                ORIGIN[1] - shift[1],
                ORIGIN[2] - shift[2],
            ],
        ),
    )
}

/// A known non-symmetric linear part. Every off-diagonal is different and two of
/// them are zero in different places, so a transposed or axis-swapped answer is
/// a different matrix; the diagonal is off unity in both directions. It is
/// **near** identity because the fixed window is resampled through it and the
/// result has to stay inside the scene: the largest displacement it produces
/// over a 24-voxel window is under four voxels, against a margin of ten.
const LINEAR: [[f64; 3]; 3] = [[1.04, 0.03, 0.00], [-0.02, 0.97, 0.01], [0.00, 0.015, 1.02]];

/// The translation that goes with [`LINEAR`]. Fractional and of mixed sign on
/// all three axes, so an answer that rounded to voxels, or dropped a sign, is
/// not the answer.
const OFFSET: [f64; 3] = [0.7, -0.5, 0.35];

/// The pair whose answer is `(LINEAR, OFFSET)`: the fixed window is the scene
/// read through the affine, the moving window is the scene read straight.
///
/// The optimiser maps a fixed point `x` to `A x + b` in the *moving* window's
/// own coordinates, and `moving[y] = scene[y + ORIGIN]`, so requiring
/// `moving[A x + b] == fixed[x]` makes `fixed[x] = scene[A x + b + ORIGIN]`.
/// That is what this builds, so the answer is `(A, b)` with no change of frame
/// to undo.
fn affine_pair(scene: &Array3<f64>) -> (Array3<f64>, Array3<f64>) {
    let fixed = Array3::from_shape_fn((WINDOW[0], WINDOW[1], WINDOW[2]), |(x, y, z)| {
        let point = [x as f64, y as f64, z as f64];
        let mut mapped = ORIGIN;
        for row in 0..3 {
            for (col, value) in point.iter().enumerate() {
                mapped[row] += LINEAR[row][col] * value;
            }
            mapped[row] += OFFSET[row];
        }
        trilinear(scene, mapped).expect("the affine window lies inside the scene")
    });
    (fixed, window(scene, WINDOW, ORIGIN))
}

/// The dense moment initialiser, from its definition: the difference of the two
/// positive-mass centroids in voxel coordinates.
///
/// This is a **second implementation** of what `Moments::translation` computes
/// at substage 0, written from the definition rather than transcribed, and
/// `the_moment_initialiser_is_the_difference_of_the_positive_mass_centroids`
/// pins it against the crate's own before anything else in this file relies on
/// it. Every "did the optimiser actually run" claim below is a comparison
/// against this number.
fn moment_seed(fixed: &Array3<f64>, moving: &Array3<f64>) -> [f64; 3] {
    let centroid = |volume: &Array3<f64>| {
        let mut mass = 0.0;
        let mut sum = [0.0; 3];
        for ((x, y, z), value) in volume.indexed_iter() {
            if value.is_finite() && *value > 0.0 {
                mass += value;
                sum[0] += value * x as f64;
                sum[1] += value * y as f64;
                sum[2] += value * z as f64;
            }
        }
        assert!(
            mass > 0.0,
            "a fixture with no positive mass has no centroid"
        );
        [sum[0] / mass, sum[1] / mass, sum[2] / mass]
    };
    let fixed = centroid(fixed);
    let moving = centroid(moving);
    [
        moving[0] - fixed[0],
        moving[1] - fixed[1],
        moving[2] - fixed[2],
    ]
}

/// Mean squares of `moving[x + shift] - fixed[x]`, over the voxels where **both**
/// `left` and `right` are in range, so the two are scored on the same sample set
/// and neither can win by evaluating somewhere easier.
///
/// This is the objective the file compares a fit against its own seed with. It
/// is computed here, over the whole volume, from the definition — not read back
/// out of `VolumeFitDiagnostics::final_cost`, which is the optimiser's own
/// arithmetic and cannot witness itself.
fn contested_mean_squares(
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    left: [f64; 3],
    right: [f64; 3],
) -> (f64, f64, usize) {
    let mut left_sum = 0.0;
    let mut right_sum = 0.0;
    let mut count = 0usize;
    for ((x, y, z), value) in fixed.indexed_iter() {
        let point = [x as f64, y as f64, z as f64];
        let at = |shift: [f64; 3]| {
            trilinear(
                moving,
                [
                    point[0] + shift[0],
                    point[1] + shift[1],
                    point[2] + shift[2],
                ],
            )
        };
        if let (Some(a), Some(b)) = (at(left), at(right)) {
            left_sum += (a - value) * (a - value);
            right_sum += (b - value) * (b - value);
            count += 1;
        }
    }
    assert!(
        count > 0,
        "the two candidates overlap the moving image nowhere"
    );
    (left_sum / count as f64, right_sum / count as f64, count)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn translation_of(params: &[f64]) -> [f64; 3] {
    [params[0], params[1], params[2]]
}

/// The two Gaussians the file that stood here was built on, kept exactly as they
/// were so that the first test measures the shipped assertion and not a
/// different one.
const BLOB: [usize; 3] = [9, 7, 5];

fn gaussian(centre: [f64; 3]) -> Array3<f64> {
    Array3::from_shape_fn((BLOB[0], BLOB[1], BLOB[2]), |(x, y, z)| {
        let dx = x as f64 - centre[0];
        let dy = y as f64 - centre[1];
        let dz = z as f64 - centre[2];
        (-(dx * dx + dy * dy + dz * dz) / 3.0).exp()
    })
}

/// A wide, well-sampled Gaussian: `17 x 15 x 13` at a variance of six voxels, so
/// the blob's support is comfortably inside the box on every axis and the
/// intensity varies smoothly across many voxels.
///
/// This is the *control* fixture, not a test fixture. Its job is to hold a
/// fractional shift on a volume whose cost surface has no structure finer than
/// the trilinear cell, which is what separates "fractional shifts are outside
/// the model" from "fractional shifts of *texture* are outside the stopping
/// rule". [`BLOB`] is too small and too sharply truncated to do that job — it
/// recovers a 0.3-voxel shift only to 0.022 — which is why it is not reused for
/// it.
const SMOOTH: [usize; 3] = [17, 15, 13];

fn smooth_blob(centre: [f64; 3]) -> Array3<f64> {
    Array3::from_shape_fn((SMOOTH[0], SMOOTH[1], SMOOTH[2]), |(x, y, z)| {
        let dx = x as f64 - centre[0];
        let dy = y as f64 - centre[1];
        let dz = z as f64 - centre[2];
        (-(dx * dx + dy * dy + dz * dz) / 12.0).exp()
    })
}

/// A run that cannot move: one Gauss-Newton step clamped to a picovoxel, which
/// is far below the `1e-4` step tolerance, so the loop stops at substage 1 and
/// the parameters it returns are the ones substage 0 put there.
///
/// **This is the readout the seed claims below are made with.** There is no
/// other way to see the state after the initialiser: the state sidecar is
/// written only on convergence, and a substage limit of 1 is an error rather
/// than an early stop.
fn one_step_readout() -> VolumeFitParams {
    let mut params = VolumeFitParams::new();
    params.max_step = 1.0e-12;
    params
}

// -------------------------- claim 1: the shipped assertion and its seed --

/// **The moment initialiser is the difference of the positive-mass centroids**,
/// and this file's own copy of it agrees with the crate's.
///
/// Everything below that says "the optimiser moved" is a comparison against
/// [`moment_seed`], so if that function drifted from what substage 0 actually
/// computes, every one of those claims would be measuring the wrong distance.
/// It is pinned here, once, by running the fitter with a step clamp of `1e-12`:
/// the loop then stops at substage 1 having moved less than a picovoxel, so what
/// it returns *is* the initialiser.
///
/// The bound is twice the step clamp. One clamped step is the only thing that
/// can move the parameters at all; the doubling is for the second thing that
/// separates the two numbers, which is the order 7680 terms are accumulated in —
/// the fitter sums a block at a time with the first axis innermost and this file
/// sums in `ndarray`'s own order. The measured gap is `1.03e-12` against a clamp
/// of `1e-12`, so the summation contributes `3e-14` of it and the bound has
/// thirty times that in hand.
#[test]
fn the_moment_initialiser_is_the_difference_of_the_positive_mass_centroids() {
    let scene = scene();
    let (fixed, moving) = shifted_pair(&scene, [3.0, -2.0, 1.0]);
    let seed = moment_seed(&fixed, &moving);

    let frozen = run(
        &one_step_readout(),
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
    )
    .expect("a clamped run still converges");
    let gap = distance(translation_of(&frozen.params), seed);

    let clamp = one_step_readout().max_step;
    assert!(
        gap <= 2.0 * clamp,
        "this file's moment seed is {seed:?} and the fitter's substage-0 state is {:?}, \
         {gap:e} apart — more than the {:e} one clamped step and a reordered sum can \
         explain, so the two are not the same quantity and every seed comparison below is \
         measuring the wrong thing",
        frozen.params,
        2.0 * clamp
    );
    println!("moment seed {seed:?}; the fitter's own, frozen after substage 0, is {gap:e} away");
}

/// **The bound the shipped test asserted was already met by the seed**, so that
/// test would have passed with the entire optimisation loop deleted — and the
/// fit is measured against the seed instead, which is the comparison that says
/// the loop ran.
///
/// The fixture is the one that stood here: two Gaussians at `[4, 3, 2]` and
/// `[5, 2, 2]` in a `9 x 7 x 5` volume, whose answer is `[1, -1, 0]`, asserted
/// to `+/- 0.25` per axis. Two symmetric blobs have centroids at their centres
/// up to the box's truncation, so the difference of centroids is the answer
/// almost exactly: `[0.9933, -0.9441, -8.9e-16]`, whose **worst axis error is
/// 0.056 against a bound of 0.25**. That is asserted here rather than described,
/// so that a future change which makes the initialiser worse is visible as this
/// test failing rather than as the next one silently becoming meaningful.
///
/// What replaces it is a comparison on [`contested_mean_squares`], computed in
/// this file over the whole volume: the fit must beat its own seed. It beats it
/// by **sixteen orders of magnitude** — `5.9e-21` against `6.1e-5` — because for
/// two integral translates of one sampled Gaussian the minimum is exactly zero.
/// The assertion is `at least 1e6`, ten decades below the measured margin, so it
/// is not a threshold tuned to this fixture; anything that leaves the optimiser
/// merely near its seed fails it. The recovered shift is separately required to
/// be the answer to `1e-8` — the measured distance is `5.2e-10`, and the reason
/// it is not `1e-15` is that the loop stops on the `1e-4` step tolerance after
/// two substages rather than running to the last bit.
#[test]
fn the_shipped_translation_bound_is_met_by_the_seed_so_the_fit_is_scored_against_it() {
    let fixed = gaussian([4.0, 3.0, 2.0]);
    let moving = gaussian([5.0, 2.0, 2.0]);
    let truth = [1.0, -1.0, 0.0];

    // What the file that stood here asserted, aimed at the seed instead of at
    // the fit. It passes, and that is the finding.
    let seed = moment_seed(&fixed, &moving);
    for axis in 0..3 {
        assert!(
            (seed[axis] - truth[axis]).abs() <= 0.25,
            "the moment seed {seed:?} misses the shipped +/- 0.25 bound on axis {axis}, so \
             the assertion this test records as vacuous is not vacuous after all"
        );
    }

    let fitted = run(
        &VolumeFitParams::new(),
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        4,
    )
    .expect("fit computes");
    assert_eq!(fitted.model, TransformModel::Translation);
    let got = translation_of(&fitted.params);

    // The claim that is not the seed's: the fit is better than the seed, on an
    // objective computed in this file, over the same samples.
    let (cost_fit, cost_seed, samples) = contested_mean_squares(&fixed, &moving, got, seed);
    let margin = cost_seed / cost_fit;
    assert!(
        margin >= 1.0e6,
        "the fit scores {cost_fit:e} and its own seed scores {cost_seed:e} over {samples} \
         samples — only {margin:e} apart, so this fit is its initialiser and the loop did \
         nothing"
    );
    // and it is the right answer, to far more than the shipped bound
    assert!(
        distance(got, truth) <= 1.0e-8,
        "the recovered shift is {got:?} against a known {truth:?}"
    );
    println!(
        "seed {seed:?} already inside the shipped +/- 0.25 bound (worst axis {:.4}); the fit \
         {got:?} beats it {margin:e} to one in mean squares over {samples} samples",
        (0..3)
            .map(|axis| (seed[axis] - truth[axis]).abs())
            .fold(0.0, f64::max)
    );
}

// ----------------------------------- claim 2: a known transform, recovered --

/// **An integral translation of a textured volume is recovered to machine
/// precision, from a seed that is three and a half voxels away** — and the
/// answer does not depend on how the volume was cut into blocks or on how many
/// workers ran it.
///
/// The transform is `[3, -2, 1]`: different on every axis, mixed in sign, and
/// each component larger than the 1.5-to-4-voxel object radius, so the seed
/// cannot be near it by accident. It is integral, so the two windows are the
/// same scene voxels and the mean-squares minimum is exactly `0` — this is a
/// transform with an exact answer, not an approximately recoverable one.
///
/// The bound is `1e-11`. The measured error is `1.3e-13` in Euclidean distance,
/// which is the accumulated rounding in a Gauss-Newton solve over 7680 samples;
/// `1e-11` is seventy times that — room for a different summation order on
/// another machine — and is still eleven decades below the `3.4` voxels the seed
/// is away. A number picked to fit would have been `2e-13`.
///
/// **Block invariance is the crate's own property and this op is not exempt.**
/// Five block edges from 24 — one block — down to 5, which is eighty, agree to a
/// measured `1.8e-15`, asserted at `1e-13` because that is the scale a reordered
/// sum of 7680 terms of order one can move an answer of order three. One worker
/// against four is bit-identical, and that one *is* asserted on `==`: the
/// evidence is reduced over a list sorted by block index, so it must be.
#[test]
fn an_integral_translation_is_recovered_to_machine_precision_from_a_distant_seed() {
    let scene = scene();
    let truth = [3.0, -2.0, 1.0];
    let (fixed, moving) = shifted_pair(&scene, truth);
    let seed = moment_seed(&fixed, &moving);
    assert!(
        distance(seed, truth) > 3.0,
        "the seed {seed:?} is already at the answer {truth:?}, so this fixture cannot tell a \
         working optimiser from a working initialiser"
    );

    let fitted = run(
        &VolumeFitParams::new(),
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
    )
    .expect("the fit converges within the shipped substage limit");
    let got = translation_of(&fitted.params);
    let error = distance(got, truth);
    assert!(
        error <= 1.0e-11,
        "recovered {got:?} against a known {truth:?}: {error:e} apart"
    );
    assert_eq!(fitted.shape, WINDOW);

    // Same answer however the volume is cut, and however many workers ran it.
    let mut worst_across_blocks = 0.0f64;
    for block in [24usize, 16, 12, 8, 5] {
        let single = run_with_workers(
            &VolumeFitParams::new(),
            &Voxels::from(fixed.clone()),
            &Voxels::from(moving.clone()),
            block,
            1,
        )
        .expect("a blocked fit converges too");
        let parallel = run_with_workers(
            &VolumeFitParams::new(),
            &Voxels::from(fixed.clone()),
            &Voxels::from(moving.clone()),
            block,
            4,
        )
        .expect("a blocked fit converges under four workers too");
        assert_eq!(
            single.params, parallel.params,
            "block edge {block}: one worker and four disagree, so the evidence reduction is \
             not over a deterministic order"
        );
        worst_across_blocks = worst_across_blocks.max(distance(
            translation_of(&single.params),
            translation_of(&fitted.params),
        ));
    }
    assert!(
        worst_across_blocks <= 1.0e-13,
        "the answer moves by {worst_across_blocks:e} across block edges 24 down to 5, which is \
         more than the rounding a reordered sum of 7680 terms can explain"
    );

    // The plan is one phase whatever the block edge, which is what makes the
    // comparison above a comparison of the same computation.
    for block in [24usize, 5] {
        let decomposition = plan(&VolumeFitParams::new(), WINDOW, block).expect("a plan");
        assert_eq!(decomposition.phases.len(), 1);
        assert_eq!(decomposition.volume, WINDOW);
    }

    println!(
        "seed {seed:?} is {:.3} voxels from {truth:?}; the fit lands {error:e} away and moves \
         {worst_across_blocks:e} across five block edges",
        distance(seed, truth)
    );
}

/// **A subsampled fixed image recovers the same translation.**
///
/// `Sampling::Stride` is the only non-`All` policy there is, and nothing
/// executed it. A stride of `[3, 2, 2]` keeps one voxel in twelve — 640 of 7680
/// — which is a different sample set, a different normal-equation matrix and a
/// different sum order, so agreement is evidence that the policy selects samples
/// rather than, say, shifting the lattice it evaluates on. That last failure
/// mode is what the bound is set to catch: an off-by-one in the stride's phase
/// would move the answer by a fraction of a voxel, and the bound is `1e-11`
/// against a measured worst of `1.3e-13` — nine decades below a tenth of a
/// voxel.
#[test]
fn a_strided_sampling_policy_recovers_the_same_translation() {
    let scene = scene();
    let truth = [3.0, -2.0, 1.0];
    let (fixed, moving) = shifted_pair(&scene, truth);

    for stride in [[1usize, 1, 1], [2, 2, 2], [3, 2, 2]] {
        let params = VolumeFitParams::new().with_sampling(Sampling::Stride(stride));
        let fitted = run(
            &params,
            &Voxels::from(fixed.clone()),
            &Voxels::from(moving.clone()),
            WINDOW[0],
        )
        .expect("a strided fit converges");
        let error = distance(translation_of(&fitted.params), truth);
        assert!(
            error <= 1.0e-11,
            "stride {stride:?} recovers {:?}, {error:e} from {truth:?}",
            fitted.params
        );
        println!("stride {stride:?}: {error:e} from the answer");
    }

    // and a stride that selects nothing is refused rather than silently fitting
    // on an empty sample set
    assert!(run(
        &VolumeFitParams::new().with_sampling(Sampling::Stride([0, 1, 1])),
        &Voxels::from(fixed),
        &Voxels::from(moving),
        WINDOW[0],
    )
    .is_err());
}

/// **All twelve affine parameters are recovered from a resampled window**, to
/// `9.9e-15`.
///
/// This is the claim that separates the affine model from the translation one:
/// the seed puts the linear part at the identity and only the nine off-identity
/// numbers in [`LINEAR`] can move it there, each through its own column of the
/// parameter Jacobian. Two of those nine are exactly zero and the other seven
/// are all different, so an implementation that shared a Jacobian column between
/// two parameters, or transposed the matrix, produces a different answer.
///
/// The fixed window is the scene read *through* the affine with this file's own
/// trilinear sampler, and the crate's sampler is the same interpolation, so the
/// minimum is again a true zero rather than a resampling artefact — the measured
/// final cost is `6.2e-15`. The bound is `1e-12` per parameter, against a
/// measured worst of `9.9e-15`.
#[test]
fn a_twelve_parameter_affine_is_recovered_from_a_resampled_window() {
    let scene = scene();
    let (fixed, moving) = affine_pair(&scene);

    let diagnostics = run_with_diagnostics(
        &VolumeFitParams::new().affine(),
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
        1,
    )
    .expect("the affine fit converges within its own 64-substage limit");
    let got = &diagnostics.transform.params;
    assert_eq!(diagnostics.transform.model, TransformModel::Affine);
    assert_eq!(got.len(), 12);

    let mut expected = Vec::with_capacity(12);
    for row in LINEAR {
        expected.extend(row);
    }
    expected.extend(OFFSET);

    let worst = got
        .iter()
        .zip(&expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0, f64::max);
    assert!(
        worst <= 1.0e-12,
        "recovered {got:?} against a known {expected:?}: worst parameter is {worst:e} out"
    );

    // Liveness: the identity linear part the optimiser starts from is nowhere
    // near the answer, so the nine linear parameters were genuinely fitted.
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let seeded_distance = identity
        .iter()
        .zip(&expected[..9])
        .map(|(seed, want)| (seed - want).abs())
        .fold(0.0, f64::max);
    assert!(
        seeded_distance > 0.01,
        "the linear part the run starts from is only {seeded_distance:e} from the answer, so \
         this fixture cannot see the nine linear parameters being fitted"
    );
    println!(
        "affine recovered to {worst:e} in {} substages, final cost {:e}; the identity the run \
         starts from is {seeded_distance} away",
        diagnostics.substages, diagnostics.final_cost
    );
}

// ------------------------------------------ claim 3: the other two metrics --

/// **Normalized correlation recovers the same translation — but the shipped
/// configuration for it is a hard error**, because `with_metric` leaves the
/// substage limit at 16 and this metric needs 27.
///
/// The defect is a configuration one and it is recorded rather than fixed:
/// `VolumeFitParams::affine` and `::bspline` both raise the limit to 64 when
/// they change the model, and `with_metric` raises nothing when it changes the
/// metric — yet the normalized-correlation branch replaces the Gauss-Newton
/// Hessian with the **identity**, so its step is the raw gradient and it walks
/// to the answer instead of jumping to it. First-order convergence needs more
/// substages than second-order, and nothing in the constructor knows that.
///
/// With room to run it lands `0.021` voxels from the answer. That is not machine
/// precision and is not expected to be: this run stops on the cost tolerance,
/// with the correlation at `-0.999908` — `9.2e-5` short of a perfect one — and
/// a correlation is quadratic in the displacement near its optimum, so a
/// residual of that size *is* a displacement of order a hundredth of a voxel.
/// The bound is `0.05`, a little over twice the measured `0.0208`. What makes it
/// a claim about recovery rather than about the seed is the second assertion:
/// the seed is 164 times further from the answer than the fit is, required to be
/// at least 100.
#[test]
fn normalized_correlation_recovers_the_translation_but_not_at_the_shipped_substage_limit() {
    let scene = scene();
    let truth = [3.0, -2.0, 1.0];
    let (fixed, moving) = shifted_pair(&scene, truth);

    // The defect, recorded: the configuration a caller writes does not run.
    let shipped = VolumeFitParams::new().with_metric(Metric::NormalizedCorrelation);
    assert_eq!(shipped.limit.substages(), 16);
    let refused = run(
        &shipped,
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
    );
    let message = refused
        .expect_err("16 substages are not enough for this metric")
        .to_string();
    assert!(
        message.contains("did not converge in 16 substage"),
        "the shipped normalized-correlation configuration failed for some other reason: {message}"
    );

    let mut roomy = shipped.clone();
    roomy.limit = SubstageLimit::of(64).expect("a positive limit");
    let diagnostics = run_with_diagnostics(
        &roomy,
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
        1,
    )
    .expect("with room the correlation converges");
    let got = translation_of(&diagnostics.transform.params);
    let error = distance(got, truth);

    assert!(
        diagnostics.substages > 16,
        "it converged in {} substages, so the shipped limit of 16 was enough after all and \
         the refusal above is recording something else",
        diagnostics.substages
    );
    assert!(
        error <= 0.05,
        "normalized correlation recovered {got:?} against {truth:?}: {error} apart"
    );
    let seed = moment_seed(&fixed, &moving);
    assert!(
        distance(seed, truth) / error > 100.0,
        "the seed is {:.3} from the answer and the fit is {error}, so this is not a recovery",
        distance(seed, truth)
    );
    println!(
        "normalized correlation needs {} substages against a shipped limit of 16, and lands \
         {error:.4} voxels from {truth:?} (cost {:.6}, stopped by {:?})",
        diagnostics.substages, diagnostics.final_cost, diagnostics.converged_by
    );
}

/// **The defect, recorded: mutual information never leaves the moment
/// initialiser.** It returns the difference of centroids, three and a half
/// voxels from an answer the mean-squares metric finds to `1e-13` on the same
/// fixture.
///
/// The mechanism is a doubled division by the sample count in
/// `Evidence::finish_mutual_information`. The gradient it accumulates is already
/// divided by `self.count`, and the surrogate Hessian it then puts on the
/// diagonal is `self.count` again — so the Gauss-Newton step is the gradient
/// scaled by `1/N^2`, which at `N = 7680` is `4.8e-5`. That is **below the
/// `1e-4` step tolerance**, so the very first step declares convergence and the
/// loop stops. It is not a slow optimiser that needs more substages: with the
/// tolerance set to zero it takes a second step of the same size, the cost does
/// not change in the sixteenth significant figure, and it stops again.
///
/// This is asserted as the current behaviour rather than papered over, because
/// the fix is a scale nobody here can choose on evidence — what the surrogate
/// Hessian for a histogram metric should be is a design decision, not a typo —
/// and an unmarked metric that silently returns its input is worse than a
/// recorded one.
///
/// The bound is `1e-4`, which is the step tolerance: that is exactly how far the
/// one accepted step can carry the parameters before the loop stops. The
/// measured distance from the seed is `4.8e-5`.
#[test]
fn mutual_information_returns_the_moment_initialiser_and_never_optimises() {
    let scene = scene();
    let truth = [3.0, -2.0, 1.0];
    let (fixed, moving) = shifted_pair(&scene, truth);
    let seed = moment_seed(&fixed, &moving);

    let diagnostics = run_with_diagnostics(
        &VolumeFitParams::new().with_metric(Metric::MutualInformation),
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
        1,
    )
    .expect("it converges immediately, which is the defect");
    let got = translation_of(&diagnostics.transform.params);

    let from_seed = distance(got, seed);
    let from_truth = distance(got, truth);
    assert!(
        from_seed <= 1.0e-4,
        "mutual information moved {from_seed:e} from its initialiser {seed:?} to {got:?}; if \
         this now exceeds the step tolerance the metric has started optimising and this \
         recording of the defect is out of date"
    );
    assert!(
        from_truth > 3.0,
        "mutual information landed {from_truth} from {truth:?}, which is close enough that it \
         is doing something; the defect this test records has changed"
    );
    assert_eq!(
        diagnostics.substages, 1,
        "the loop stopped after {} substages rather than the first",
        diagnostics.substages
    );

    // The control that makes the reading a defect and not a property of the
    // fixture: the same two volumes, the same seed, mean squares.
    let by_mean_squares = run(
        &VolumeFitParams::new(),
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
    )
    .expect("mean squares converges on this pair");
    assert!(
        distance(translation_of(&by_mean_squares.params), truth) <= 1.0e-11,
        "mean squares does not recover this fixture either, so the comparison is not evidence \
         about the metric"
    );

    println!(
        "mutual information stops after {} substage at {got:?} — {from_seed:e} from its own \
         initialiser and {from_truth:.3} voxels from the answer mean squares finds to 1e-13",
        diagnostics.substages
    );
}

// ------------------------------------------------ claim 4: the B-spline --

/// **A uniform shift is recovered as a uniform B-spline displacement field**:
/// all 64 control coefficients on the shifted axis land within `0.008` of the
/// shift and all 128 on the other two within `0.002` of zero.
///
/// This is the only claim in the tree about the free-form model, and it is the
/// one shape whose right answer is known in closed form. A cubic B-spline
/// reproduces a constant displacement when every control coefficient equals it —
/// the basis is a partition of unity, which `src/ops/align.rs`'s own
/// `mod tests` asserts separately — so a rigid two-voxel shift has an exact
/// representation on any control grid, and the fitter either finds it or does
/// not. 192 parameters, from a default `ControlGrid` of `[4, 4, 4]`.
///
/// **The smoothness weight is not decoration here, and that is recorded.** With
/// `bspline_smoothness` at its default of `0` the run does not converge — not in
/// the 128 substages asserted below, and not in 400 either: the B-spline branch replaces the normal equations with their
/// diagonal, and a diagonal Gauss-Newton on 192 coupled coefficients limit-cycles.
/// The quadratic neighbour penalty adds its weight to every diagonal entry,
/// which is what damps it. So the shipped default for this model does not fit,
/// and a caller has to know to set a smoothness — asserted below rather than
/// described.
///
/// The bound is `0.02` on a measured worst deviation of `0.0078`. The reason it
/// is not tighter is the stopping rule: this fit ends on the cost tolerance, not
/// the step tolerance, with a `0.016` step still being taken. The reason it is
/// not looser is that `0.02` is one percent of the two-voxel shift, so a field
/// that had recovered half the shift, or recovered it on the wrong axis, fails.
#[test]
fn a_uniform_shift_is_recovered_as_a_uniform_bspline_displacement_field() {
    let scene = scene();
    let shift = 2.0;
    let (fixed, moving) = shifted_pair(&scene, [shift, 0.0, 0.0]);

    let mut params = VolumeFitParams::new()
        .bspline()
        .with_bspline_smoothness(1.0)
        .expect("a non-negative smoothness");
    params.limit = SubstageLimit::of(128).expect("a positive limit");

    let diagnostics = run_with_diagnostics(
        &params,
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
        1,
    )
    .expect("a smoothed B-spline fit converges");
    let got = &diagnostics.transform.params;
    assert_eq!(diagnostics.transform.bspline, ControlGrid::default());
    assert_eq!(
        got.len(),
        3 * ControlGrid::default().size.iter().product::<usize>()
    );

    let per_axis = got.len() / 3;
    let worst = |axis: usize, want: f64| {
        got[axis * per_axis..(axis + 1) * per_axis]
            .iter()
            .map(|value| (value - want).abs())
            .fold(0.0, f64::max)
    };
    assert!(
        worst(0, shift) <= 0.02,
        "the shifted axis's 64 coefficients are up to {} from the {shift}-voxel shift",
        worst(0, shift)
    );
    for axis in [1usize, 2] {
        assert!(
            worst(axis, 0.0) <= 0.02,
            "axis {axis} picked up a displacement of up to {} where the shift is zero",
            worst(axis, 0.0)
        );
    }

    // The defect, recorded: the shipped default for this model does not fit.
    let mut unsmoothed = VolumeFitParams::new().bspline();
    unsmoothed.limit = SubstageLimit::of(128).expect("a positive limit");
    assert_eq!(unsmoothed.bspline_smoothness, 0.0);
    let refused = run(
        &unsmoothed,
        &Voxels::from(fixed.clone()),
        &Voxels::from(moving.clone()),
        WINDOW[0],
    );
    assert!(
        refused.is_err(),
        "the unsmoothed B-spline converged, so the smoothness weight is no longer what makes \
         this model settle and this recording is out of date"
    );

    println!(
        "B-spline recovered a {shift}-voxel shift in {} substages: shifted axis within {:.4}, \
         the other two within {:.4} and {:.4}; the same fit with the default smoothness of 0 \
         does not converge",
        diagnostics.substages,
        worst(0, shift),
        worst(1, 0.0),
        worst(2, 0.0)
    );
}

// -------------------------------------------------- claim 5: the pyramid --

/// A hand-built two-level stack whose coarse level answers something the fine
/// level would never answer on its own.
///
/// `validate_pyramid_levels` constrains shapes, scales and geometry — it does
/// **not** require the coarse level to be a decimation of the fine one, and this
/// exploits that deliberately. The coarse level here is cut from a different
/// part of the scene and displaced by five voxels rather than three, so its
/// moment initialiser is a different number from the fine level's. Under the
/// one-step readout the two-level run therefore lands either at twice the coarse
/// level's answer or at the fine level's own, and those are **1.5 voxels apart**
/// — against the `0.007` that separates them on a genuinely decimated stack,
/// where the coarse level is the same picture at half the size and its
/// initialiser is nearly the same number. A defect that discards the seed would
/// be invisible at `0.007` and is not at `1.5`.
fn probe_levels(scene: &Array3<f64>) -> Vec<VolumePyramidLevel> {
    let (fixed, moving) = shifted_pair(scene, [3.0, -2.0, 1.0]);
    let coarse_shape = [WINDOW[0] / 2, WINDOW[1] / 2, WINDOW[2] / 2];
    // Eight voxels in, so that the five-voxel displacement below still reads
    // scene voxels that exist.
    let coarse_origin = [8.0, 6.0, 6.0];
    vec![
        VolumePyramidLevel::new(
            Voxels::from(fixed),
            Voxels::from(moving),
            [1, 1, 1],
            SpatialFrame::unit(),
        )
        .expect("a full-resolution level"),
        VolumePyramidLevel::new(
            Voxels::from(window(scene, coarse_shape, coarse_origin)),
            Voxels::from(window(
                scene,
                coarse_shape,
                [coarse_origin[0] - 5.0, coarse_origin[1], coarse_origin[2]],
            )),
            [2, 2, 2],
            SpatialFrame::new([0.0; 3], [2.0; 3], SpatialFrame::unit().direction())
                .expect("a level frame at scale two"),
        )
        .expect("a half-resolution level"),
    ]
}

/// **The coarse level's answer reaches the level below it, scaled up by the
/// level's own factor** — the seed is not discarded.
///
/// Read out with a step clamp of `1e-12`, so that every level stops after one
/// substage having moved less than a picovoxel. What comes out of a two-level
/// run is then, if the seed is carried, exactly twice what comes out of running
/// the coarse level alone, and if it is not, exactly the fine level's own
/// initialiser. On the probe stack the measured distances are `1.0e-12` and
/// `1.489`, twelve orders of magnitude apart, so the reading does not rest on
/// where a tolerance is put.
///
/// The bound is derived rather than chosen: the pair can differ from `2 x`
/// the coarse answer by one clamped step at the fine level plus twice one
/// clamped step at the coarse level, which is `3 x max_step`. Nothing else can
/// move it.
#[test]
fn the_coarse_pyramid_level_seeds_the_level_below_it() {
    let scene = scene();
    let levels = probe_levels(&scene);
    let params = one_step_readout();
    let clamp = params.max_step;

    let coarse_alone = run(
        &params,
        &levels[1].fixed,
        &levels[1].moving,
        levels[1]
            .fixed
            .shape()
            .into_iter()
            .max()
            .expect("an extent"),
    )
    .expect("the coarse level alone converges");
    let fine_alone = run(
        &params,
        &levels[0].fixed,
        &levels[0].moving,
        levels[0]
            .fixed
            .shape()
            .into_iter()
            .max()
            .expect("an extent"),
    )
    .expect("the fine level alone converges");
    let together = resident_pyramid_levels(&params, &levels).expect("the two-level run converges");

    let carried = [
        2.0 * coarse_alone.params[0],
        2.0 * coarse_alone.params[1],
        2.0 * coarse_alone.params[2],
    ];
    let to_carried = distance(translation_of(&together.params), carried);
    let to_fine_seed = distance(
        translation_of(&together.params),
        translation_of(&fine_alone.params),
    );

    assert!(
        to_carried <= 3.0 * clamp,
        "the two-level answer {:?} is {to_carried:e} from the coarse answer scaled up, \
         {carried:?}, which is more than the {} clamped steps that can separate them",
        together.params,
        3.0 * clamp
    );
    assert!(
        to_fine_seed > 1.0,
        "the fine level's own initialiser is {:?}, only {to_fine_seed} from the two-level \
         answer, so this stack cannot tell a carried seed from a discarded one",
        fine_alone.params
    );
    assert_eq!(together.shape, WINDOW, "the reported shape is level 0's");

    // The real thing, beside the probe: a decimated stack built from the
    // schedule has the shapes, cumulative scales and per-level frames the
    // validator demands, so the probe above is a legitimate stand-in for one and
    // not a shape the builder could never produce.
    let schedule = PyramidSchedule::powers_of_two(2).expect("two levels");
    let (fixed, moving) = shifted_pair(&scene, [3.0, -2.0, 1.0]);
    let built = pyramid_levels_from_recipe(
        &params,
        &fixed,
        &moving,
        &schedule,
        &schedule.recipe().expect("a decimation recipe"),
    )
    .expect("two levels build");
    assert_eq!(built.len(), 2);
    assert_eq!(built[0].scale, [1, 1, 1]);
    assert_eq!(built[1].scale, [2, 2, 2]);
    assert_eq!(built[0].fixed.shape(), WINDOW);
    assert_eq!(
        built[1].fixed.shape(),
        [WINDOW[0] / 2, WINDOW[1] / 2, WINDOW[2] / 2]
    );
    assert_eq!(built[1].geometry.spacing(), [2.0, 2.0, 2.0]);
    resident_pyramid_levels(&params, &built).expect("and the built stack runs");
    println!(
        "coarse level answers {:?}; scaled up that is {carried:?}, and the two-level run lands \
         {to_carried:e} from it and {to_fine_seed:.3} from the fine level's own seed {:?}",
        coarse_alone.params, fine_alone.params
    );
}

/// **A seed reaches substage 0 whole, not reduced to its translation** — the
/// regression for a defect that made the coarse level of an affine pyramid
/// pointless and the coarse level of a B-spline pyramid worse than pointless.
///
/// What substage 0 does is install the starting parameters: the moment
/// initialiser for an unseeded run, the caller's seed for a seeded one. It used
/// to install `params_with_translation(seed's translation)` in both cases, which
/// for an affine rebuilds the vector around an **identity** linear part and for
/// a B-spline — whose `translation_from_params` answered `[0, 0, 0]` — zeroed
/// every one of the control coefficients. The fine level of a pyramid therefore
/// threw away everything the coarse level had found except three numbers, and
/// for the B-spline it threw away all of them.
///
/// Read out with the same one-step clamp as above, on a pair whose answer is a
/// pure two-voxel shift so the seed's linear part is deliberately *wrong* and
/// cannot be arrived at by fitting. Before the fix the linear part came back as
/// the identity and the control coefficients as zeros; the assertion is that
/// they come back as what was handed in, to `1e-11`. Exactly one Gauss-Newton
/// step is taken before the loop stops and it is clamped to `1e-12`, so that is
/// the only thing that can separate seed from answer; the bound is ten times it
/// and the measured worst is `3.5e-13`. A defect that reinstalls the identity
/// misses by `0.3`, eleven decades away.
#[test]
fn a_seed_reaches_substage_zero_whole_and_is_not_reduced_to_its_translation() {
    let scene = scene();
    let (fixed, moving) = shifted_pair(&scene, [2.0, 0.0, 0.0]);
    let fixed = Voxels::from(fixed);
    let moving = Voxels::from(moving);

    // Affine: a linear part that is nowhere near identity and nowhere near the
    // answer, so neither the initialiser nor one clamped step can produce it.
    let affine = one_step_readout().affine();
    let affine_seed = vec![1.3, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.1, 2.0, -1.0, 0.5];
    let got = run_with_seed_diagnostics(
        &affine,
        &fixed,
        &moving,
        WINDOW[0],
        1,
        Some(affine_seed.clone()),
    )
    .expect("a seeded affine run converges in one step")
    .transform
    .params;
    let worst = got
        .iter()
        .zip(&affine_seed)
        .map(|(got, seed)| (got - seed).abs())
        .fold(0.0, f64::max);
    assert!(
        worst <= 1.0e-11,
        "the affine seed {affine_seed:?} came back as {got:?}: {worst:e} out. A linear part of \
         exactly [1, 0, 0, 0, 1, 0, 0, 0, 1] here is the defect, not a rounding gap"
    );

    // B-spline: every coefficient on axis 0 seeded to three voxels, an answer
    // fifty percent past the true two-voxel shift.
    let spline = one_step_readout().bspline();
    let coefficients = 3 * ControlGrid::default().size.iter().product::<usize>();
    let mut spline_seed = vec![0.0; coefficients];
    for slot in spline_seed.iter_mut().take(coefficients / 3) {
        *slot = 3.0;
    }
    let got = run_with_seed_diagnostics(
        &spline,
        &fixed,
        &moving,
        WINDOW[0],
        1,
        Some(spline_seed.clone()),
    )
    .expect("a seeded B-spline run converges in one step")
    .transform
    .params;
    let worst = got
        .iter()
        .zip(&spline_seed)
        .map(|(got, seed)| (got - seed).abs())
        .fold(0.0, f64::max);
    assert!(
        worst <= 1.0e-11,
        "the B-spline seed came back {worst:e} out; all-zero coefficients here are the defect"
    );

    // Liveness: an *unseeded* run of the same shape lands somewhere else
    // entirely, so the assertions above are not satisfied by whatever the
    // initialiser would have produced anyway.
    let unseeded = run(&one_step_readout(), &fixed, &moving, WINDOW[0])
        .expect("an unseeded run converges too");
    assert!(
        distance(translation_of(&unseeded.params), [2.0, -1.0, 0.5]) > 1.0,
        "the moment initialiser {:?} is already at the seed's translation, so this readout \
         cannot see a seed being honoured",
        unseeded.params
    );
    println!("an affine and a B-spline seed both survive substage 0 to within {worst:e}");
}

/// **The defect, recorded: adding pyramid levels can turn a transform one level
/// recovers exactly into a hard error.**
///
/// A 3-voxel shift is recovered to `1e-13` with no pyramid. The same pair, with
/// `PyramidSchedule::powers_of_two(2)`, fails at the coarse level — not with a
/// wrong answer, but with the substage limit's refusal, which is at least honest.
///
/// The mechanism is `a_fractional_translation_of_a_textured_volume_does_not_converge`
/// below: at scale two an odd shift becomes a **half-voxel** shift, and the
/// mean-squares loop does not converge on a fractional shift of a textured
/// volume at any resolution. The pyramid is not doing anything wrong; it is
/// exposing the optimiser's stopping rule to an input the caller never chose.
///
/// The parity is the whole evidence, so both halves are asserted: shifts of 2
/// and 4 — even, so integral at scale two — go through two levels intact, and
/// shifts of 1, 3 and 5 do not. A fix for the underlying non-convergence turns
/// this test's second half into a failure, which is the intent.
#[test]
fn more_pyramid_levels_can_lose_a_translation_that_one_level_recovers() {
    let scene = scene();
    let mut roomy = VolumeFitParams::new();
    roomy.limit = SubstageLimit::of(128).expect("a positive limit");

    let mut lost = Vec::new();
    let mut kept = Vec::new();
    for shift in [1.0f64, 2.0, 3.0, 4.0, 5.0] {
        let (fixed, moving) = shifted_pair(&scene, [shift, 0.0, 0.0]);

        let flat =
            resident(&roomy, &fixed, &moving).expect("one level recovers every one of these");
        let error = distance(translation_of(&flat.params), [shift, 0.0, 0.0]);
        assert!(
            error <= 1.0e-11,
            "a {shift}-voxel shift is not recovered at one level either ({error:e}), so this \
             test is not about the pyramid"
        );

        let two = resident(
            &roomy
                .clone()
                .with_pyramid(PyramidSchedule::powers_of_two(2).expect("two levels")),
            &fixed,
            &moving,
        );
        match two {
            Ok(fitted) => kept.push((
                shift,
                distance(translation_of(&fitted.params), [shift, 0.0, 0.0]),
            )),
            Err(_) => lost.push(shift),
        }
    }

    assert_eq!(
        lost,
        vec![1.0, 3.0, 5.0],
        "the shifts a second pyramid level loses have changed; kept: {kept:?}"
    );
    for (shift, error) in &kept {
        assert!(
            *error <= 1.0e-9,
            "the {shift}-voxel shift survives two levels but only to {error:e}"
        );
    }
    println!(
        "one level recovers 1..5 exactly; two levels lose {lost:?} — the odd ones, which halve \
         to a fractional shift at the coarse level — and keep {kept:?}"
    );
}

/// **The defect, recorded: a fractional translation of a textured volume does
/// not converge at all**, at any resolution, under the shipped stopping rule.
///
/// This is the mechanism behind the pyramid test above, isolated at full
/// resolution so it cannot be mistaken for a decimation artefact. Integral
/// shifts converge in four substages to `1e-15`. Every fractional shift tried
/// here runs to the substage limit, and it is not a limit that is merely too
/// small: 400 substages does not help, nor does a step clamp cut from 2 to 0.01,
/// nor a damping raised from `1e-6` to `1e5`. Loosening the tolerance to `1e-1`
/// does stop it — at `[1.4887, -0.035, -0.056]` against a truth of
/// `[1.5, 0, 0]`, with a `0.12` step still being taken — which is what says the
/// loop is limit-cycling around the answer rather than diverging from it.
///
/// The reason is in the update: it takes every Gauss-Newton step it computes.
/// There is no line search, no acceptance test against the previous cost, and
/// the Levenberg damping is a constant that never adapts — so a step that
/// increases the cost is taken anyway, and on a cost surface whose gradient is
/// discontinuous at every trilinear cell boundary that is enough to cycle.
///
/// **The control that makes this a statement about the data and not about
/// fractions:** the same fractional shifts of a *smooth* volume — a single
/// Gaussian, where the cost surface has no fine structure — converge in three
/// substages and are recovered to a measured `0.0071`, asserted at `0.05`. So
/// fractional shifts are inside the model; they are outside what this stopping
/// rule can settle on real texture.
#[test]
fn a_fractional_translation_of_a_textured_volume_does_not_converge() {
    let scene = scene();
    let mut roomy = VolumeFitParams::new();
    roomy.limit = SubstageLimit::of(128).expect("a positive limit");

    for shift in [0.25f64, 0.5, 1.5, 2.5] {
        let (fixed, moving) = shifted_pair(&scene, [shift, 0.0, 0.0]);
        let refused = run(
            &roomy,
            &Voxels::from(fixed),
            &Voxels::from(moving),
            WINDOW[0],
        );
        assert!(
            refused.is_err(),
            "a {shift}-voxel shift now converges, so this recording of the defect is out of \
             date and `more_pyramid_levels_can_lose_a_translation_that_one_level_recovers` \
             should be revisited with it"
        );
    }

    // The integral control, on the same volume and the same configuration.
    let (fixed, moving) = shifted_pair(&scene, [2.0, 0.0, 0.0]);
    let integral = run(
        &roomy,
        &Voxels::from(fixed),
        &Voxels::from(moving),
        WINDOW[0],
    )
    .expect("an integral shift of the same volume converges");
    assert!(distance(translation_of(&integral.params), [2.0, 0.0, 0.0]) <= 1.0e-11);

    // The smoothness control: the same fractions, on a volume with no texture.
    for shift in [0.3f64, 0.5, 1.5] {
        let fixed = smooth_blob([8.0, 7.0, 6.0]);
        let moving = smooth_blob([8.0 + shift, 7.0, 6.0]);
        let fitted = run(
            &roomy,
            &Voxels::from(fixed),
            &Voxels::from(moving),
            SMOOTH[0],
        )
        .expect("a fractional shift of a smooth volume converges");
        let error = distance(translation_of(&fitted.params), [shift, 0.0, 0.0]);
        assert!(
            error <= 0.05,
            "the smooth control recovers a {shift}-voxel shift only to {error}, so it is not \
             the control this test needs it to be"
        );
        println!("smooth control: a {shift}-voxel shift recovers to {error:e}");
    }
}

// --------------------------------------------- claim 6: the spatial frame --

/// **The defect, recorded: the spatial frame's spacing is validated, carried
/// into the emitted transform, and never used.** A fit under an anisotropic
/// frame produces parameters that are equal, bit for bit, to the same fit under
/// a unit frame.
///
/// `VolumeFitParams::geometry` used to document itself as "used to convert the
/// optimized voxel-space parameters to physical coordinates". No such conversion
/// exists: `transform_point` and `parameter_jacobian` take voxel coordinates and
/// return voxel coordinates, and the frame reaches only `FittedTransform`, where
/// it is metadata — and its own doc already said the parameters are in the voxel
/// frame, so the two doc comments contradicted each other — and
/// `geometry_for_scale`, where it is compared against a pyramid level's own. The
/// field's documentation now records what it does; this test records that it is
/// what it does. The one place a spacing is *acted on* is the pyramid validator,
/// which requires level `k`'s spacing to be the base spacing times the level's
/// scale — asserted here too, because it is the check that would go quiet if the
/// frame stopped being carried at all.
///
/// A spacing of `[0.4, 1.0, 3.0]` is deliberately extreme — a factor of 7.5
/// between the finest and coarsest axis. Under any conversion at all, the same
/// voxel displacement is a different physical one on each axis, so a fit that
/// used the frame could not return the same numbers. It returns `==`.
#[test]
fn the_spatial_frame_spacing_is_validated_carried_and_never_applied() {
    let scene = scene();
    let (fixed, moving) = shifted_pair(&scene, [3.0, -2.0, 1.0]);
    let fixed = Voxels::from(fixed);
    let moving = Voxels::from(moving);

    let anisotropic = SpatialFrame::new(
        [5.0, -2.0, 0.5],
        [0.4, 1.0, 3.0],
        SpatialFrame::unit().direction(),
    )
    .expect("a positive finite spacing");

    let unit = run(&VolumeFitParams::new(), &fixed, &moving, WINDOW[0]).expect("a fit");
    let framed = run(
        &VolumeFitParams::new().with_geometry(anisotropic.clone()),
        &fixed,
        &moving,
        WINDOW[0],
    )
    .expect("a fit under an anisotropic frame");

    assert_eq!(
        unit.params, framed.params,
        "the anisotropic frame changed the answer, so the frame is applied after all and this \
         recording of the defect is out of date"
    );
    // It is carried, which is the half of the contract that does hold.
    assert_eq!(framed.geometry, anisotropic);
    assert_eq!(unit.geometry, SpatialFrame::unit());

    // The frame is validated, and the validation is not vacuous.
    let direction = SpatialFrame::unit().direction();
    assert!(SpatialFrame::new([0.0; 3], [1.0, 0.0, 1.0], direction).is_err());
    assert!(SpatialFrame::new([0.0; 3], [1.0, f64::NAN, 1.0], direction).is_err());
    assert!(SpatialFrame::new([0.0; 3], [1.0, f64::INFINITY, 1.0], direction).is_err());
    assert!(SpatialFrame::new(
        [0.0; 3],
        [1.0; 3],
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    .is_err());
    // A *negative* spacing is accepted: the check is "zero or non-finite", not
    // "positive". Recorded rather than asserted as desirable — a mirrored axis
    // is a legitimate frame in some conventions, and since nothing here reads
    // the spacing it makes no difference to an answer either way.
    assert!(SpatialFrame::new([0.0; 3], [-1.0, 1.0, 1.0], direction).is_ok());

    // And the one place a spacing is acted on: a pyramid level whose frame does
    // not scale with its level is refused by name.
    let levels = probe_levels(&scene);
    let mismatched = vec![
        levels[0].clone(),
        VolumePyramidLevel::new(
            levels[1].fixed.clone(),
            levels[1].moving.clone(),
            [2, 2, 2],
            SpatialFrame::unit(),
        )
        .expect("a level with an unscaled frame"),
    ];
    let message = resident_pyramid_levels(&VolumeFitParams::new(), &mismatched)
        .expect_err("a level 1 at unit spacing is not a level at scale two")
        .to_string();
    assert!(
        message.contains("spacing[0] is 1, expected 2"),
        "the unscaled level was refused for some other reason: {message}"
    );
    println!(
        "an anisotropic frame {:?} leaves the fitted parameters bit-identical to a unit one, \
         and is carried through to the emitted transform unchanged",
        anisotropic.spacing()
    );
}

// ------------------------------------ claim 7: the configuration is guarded --

/// The constructors that carry a constraint, exercised on both sides of it.
///
/// Every one of these types was unreachable from the test tree, so an argument
/// check that had rotted into always accepting — or always refusing — would have
/// had no witness. Each assertion pairs a refusal with the nearest thing that is
/// accepted, so none of them can pass by refusing everything.
#[test]
fn the_configuration_constructors_refuse_exactly_what_they_document() {
    // A cubic B-spline needs four control points per axis to have a cell at all.
    assert!(ControlGrid::new([4, 4, 4]).is_ok());
    assert!(ControlGrid::new([3, 4, 4]).is_err());
    assert!(ControlGrid::new([4, 4, 3]).is_err());
    assert_eq!(ControlGrid::default().size, [4, 4, 4]);

    // A pyramid level must shrink something, and zero is not a factor.
    assert!(PyramidSchedule::new(vec![[2, 2, 2]]).is_ok());
    assert!(PyramidSchedule::new(vec![[2, 1, 1]]).is_ok());
    assert!(PyramidSchedule::new(vec![[1, 1, 1]]).is_err());
    assert!(PyramidSchedule::new(vec![[2, 0, 2]]).is_err());
    assert!(PyramidSchedule::powers_of_two(0).is_err());
    let schedule = PyramidSchedule::powers_of_two(3).expect("three levels");
    assert_eq!(schedule.levels(), 3);
    assert_eq!(schedule.factors(), [[2, 2, 2], [2, 2, 2]]);

    // A histogram metric needs bins to put samples in.
    assert!(VolumeFitParams::new().with_metric_bins(4).is_ok());
    assert!(VolumeFitParams::new().with_metric_bins(3).is_err());

    // A grid spacing resolved from the full-resolution shape must be positive.
    assert!(VolumeFitParams::new()
        .with_bspline_final_grid_spacing([8.0, 8.0, 8.0])
        .is_ok());
    assert!(VolumeFitParams::new()
        .with_bspline_final_grid_spacing([8.0, 0.0, 8.0])
        .is_err());
    assert!(VolumeFitParams::new()
        .with_bspline_final_grid_spacing([8.0, f64::NAN, 8.0])
        .is_err());
    assert!(VolumeFitParams::new()
        .with_bspline_smoothness(-1.0)
        .is_err());

    // A final grid spacing resolves to a grid before the run starts, and a
    // finer spacing is a bigger grid.
    let coarse = VolumeFitParams::new()
        .bspline()
        .with_bspline_final_grid_spacing([12.0, 12.0, 12.0])
        .expect("a positive spacing");
    let fine = VolumeFitParams::new()
        .bspline()
        .with_bspline_final_grid_spacing([6.0, 6.0, 6.0])
        .expect("a positive spacing");
    let grid_of = |params: &VolumeFitParams| {
        let scene = scene();
        let (fixed, moving) = shifted_pair(&scene, [1.0, 0.0, 0.0]);
        let mut one_step = params.clone();
        one_step.max_step = 1.0e-12;
        run(
            &one_step,
            &Voxels::from(fixed),
            &Voxels::from(moving),
            WINDOW[0],
        )
        .expect("a clamped run")
        .bspline
        .size
    };
    let coarse_grid = grid_of(&coarse);
    let fine_grid = grid_of(&fine);
    assert!(
        (0..3).all(|axis| fine_grid[axis] > coarse_grid[axis]),
        "a 6-voxel final spacing gave {fine_grid:?} and a 12-voxel one {coarse_grid:?}; the \
         finer spacing must resolve to the larger grid"
    );

    // The images themselves are guarded, and the guards are what stop a
    // degenerate pair from being fitted to a plausible wrong answer.
    let flat: Voxels = Array3::<f64>::from_elem((8, 8, 8), 1.0).into();
    let textured: Voxels = Array3::from_shape_fn((8, 8, 8), |(x, y, z)| (x + y + z) as f64).into();
    assert!(run(&VolumeFitParams::new(), &flat, &textured, 8).is_err());
    assert!(run(&VolumeFitParams::new(), &textured, &flat, 8).is_err());
    let thin: Voxels = Array3::from_shape_fn((8, 8, 1), |(x, y, _)| (x + y) as f64).into();
    assert!(run(&VolumeFitParams::new(), &thin, &thin, 8).is_err());
    let mismatched: Voxels =
        Array3::from_shape_fn((8, 8, 4), |(x, y, z)| (x + y + z) as f64).into();
    assert!(run(&VolumeFitParams::new(), &textured, &mismatched, 8).is_err());
    println!(
        "final grid spacings 12 and 6 resolve to {coarse_grid:?} and {fine_grid:?} control points"
    );
}
