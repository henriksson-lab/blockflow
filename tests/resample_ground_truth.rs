// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **`ops::resample`'s interpolated values, against the definition of the sample
// map and against `scipy.ndimage.zoom` — in three dimensions, at a rational
// factor, with a different factor on every axis.**
//
// The gap this file closes
// ------------------------
// `ops::resample` has two genuine value witnesses today and both are
// one-dimensional:
//
// * `linear_interpolation_is_the_weighted_mean_of_the_bracketing_voxels` and
//   `the_sample_map_is_centred_and_computed_exactly` (`src/ops/resample.rs`) —
//   a linear ramp on a `(n, 1, 1)` volume, eight hand-derived values, and the
//   two rival conventions written out as discriminators;
// * `an_integer_blend_rounds_to_nearest` — `(n, 1, 1)` again, at a factor of
//   four.
//
// `tests/resample_ops.rs` is decomposition invariance, the halo guard, the
// reach's tightness, the stated-extent path and the Zarr path; its reference is
// `ResampleOp::apply` called once, so it cannot see a wrong value.
//
// Two things follow, and both are what this file is for:
//
// 1. **A transposed axis is invisible to all of it.** `linear_core` composes
//    three per-axis tap tables, and a fixture whose other two axes are one
//    voxel long has nothing to transpose. Every fixture here has three
//    different extents, three different factors and a field that depends on all
//    three coordinates differently; the measured cost of rotating the factors
//    onto the wrong axes is `17.77` at the worst voxel of the affine fixture.
// 2. **No value witness exists at a genuinely rational factor.** `4/1` and
//    `1/2` land the coordinate on a voxel or on a half; `5/7`, `8/5` and `3/4`
//    do neither, which is where the fraction `((2o + 1) d - u) / (2u)` has to
//    be right rather than merely representable.
//
// The two claims
// --------------
//
// | claim | how |
// |---|---|
// | the map is the centred one, and the blend is exact on an affine field | `f(i, j, k) = 5i + 3j - 2k + 7` resampled must equal `f` at the mapped, clamped coordinate — 120 voxels, measured at `7.1e-15` — against `0.67` for the shift-free map, `1.74` for the corner-aligned one and `17.77` for the factors rotated onto the wrong axes |
// | the field is SciPy's | 120 voxels against `scipy.ndimage.zoom(..., order=1, mode='nearest', grid_mode=True)` 1.15.2, measured at `5.0e-16`, against `0.71` for a rotated factor and `0.42` for a nearest-neighbour pass |
//
// Why an affine field is a real test and not a triviality
// -------------------------------------------------------
// Linear interpolation reproduces an affine function **exactly** — that is what
// makes it linear — so the resampled value at output `o` is `f(x(o))` and
// nothing else. The test therefore knows the answer in closed form and does not
// need the implementation to tell it: it computes `x(o)` from the fraction the
// module's header states, clamps it, evaluates `f` there, and compares. That
// pins the coordinate map, the clamp, the blend weights and the axis each
// factor is applied to, all at once — because the three coefficients `5`, `3`
// and `-2` are different, a factor applied to the wrong axis is a different
// field.
//
// A *constant* field would be reproduced by any convention, and a field affine
// on one axis only would be reproduced by any permutation of the other two.
// `an_affine_field_cannot_be_reproduced_by_the_rival_conventions` measures that
// the fixture in use is neither.
//
// Why the comparison goes through the stated extent
// -------------------------------------------------
// `Resample::to_extent(input, output, ..)`, not `Resample::new(factor, ..)`.
// The two libraries turn a requested *factor* into an output extent
// differently — this crate takes `floor(in * up / down)` and keeps the exact
// rational `up/down` as the scale, SciPy takes `round(in * zoom)` and then
// rescales to `out / in` — so comparing at a factor would be measuring that
// disagreement rather than the sample map. Stating both extents removes the
// question: `Ratio::new(out, in)` and SciPy's effective zoom are then the same
// number, and what is left over is exactly the map. The generating script
// asserts this.
//
// Reproducing the recording
// -------------------------
// `python3 tools/reference_values.py resample`, under the versions
// `--versions` prints.

use ndarray::Array3;

use blockflow::ops::resample::{resample_linear_into_with, resample_nearest_into_with, Ratio};

/// The input volume's extents, and the output's. Six different numbers, so no
/// axis can be mistaken for another and no factor is the reciprocal of an
/// integer: `5/7`, `8/5` and `3/4`.
const FROM: [usize; 3] = [7, 5, 4];
const INTO: [usize; 3] = [5, 8, 3];

fn ratios() -> [Ratio; 3] {
    [
        Ratio::new(INTO[0], FROM[0]).expect("a positive extent"),
        Ratio::new(INTO[1], FROM[1]).expect("a positive extent"),
        Ratio::new(INTO[2], FROM[2]).expect("a positive extent"),
    ]
}

/// `scipy.ndimage.zoom(lcg_volume((7, 5, 4)), zoom=(5/7, 8/5, 3/4), order=1,
/// mode='nearest', grid_mode=True)` under SciPy 1.15.2, in C order.
const SCIPY_ZOOM: [f64; 120] = [
    0.5270281334718068,
    0.6160441279411315,
    0.4636165877183279,
    0.5973840268949668,
    0.5913392666727304,
    0.3247821159660817,
    0.6933931219081084,
    0.5555270735174417,
    0.1729356599350772,
    0.7489082994560401,
    0.5150390725582839,
    0.4394857364396255,
    0.7328421816229821,
    0.44420714974403386,
    0.5787051806847254,
    0.6451947684089342,
    0.34303130507469176,
    0.590593992670377,
    0.5590125006934006,
    0.4257863108068705,
    0.4107958517968655,
    0.4987988690535228,
    0.4980205476284028,
    0.27002816796302803,
    0.5301083286603294,
    0.7932469606399536,
    0.5219105839729311,
    0.5174879830330612,
    0.6289688427001239,
    0.44933399744331837,
    0.4952843355635802,
    0.4183007452636959,
    0.36014859862625587,
    0.4355094475050769,
    0.4237670015543701,
    0.4014221515506506,
    0.4230620605250201,
    0.4174312103539707,
    0.49807408377528184,
    0.4579421746234099,
    0.3992933716624977,
    0.6501043953001497,
    0.5116317566484213,
    0.49581051692366596,
    0.41665271669626236,
    0.5506774226824444,
    0.572290128469467,
    0.22325460910797132,
    0.340670645236969,
    0.43624594807624817,
    0.7404460012912748,
    0.4384983399262031,
    0.3811141084879637,
    0.5369051781793435,
    0.5778890910247962,
    0.31037632562220097,
    0.2741827418406806,
    0.7140119560062885,
    0.3118364345282316,
    0.26391181846459716,
    0.6785376773526272,
    0.491041449829936,
    0.4695308214674394,
    0.4714662550638119,
    0.8479913715273142,
    0.8910397508492073,
    0.24886770360171795,
    0.6615505907684565,
    0.7334950137883425,
    0.09184105197588602,
    0.48877832293510437,
    0.5781761904557546,
    0.4489115397135417,
    0.344732105731964,
    0.4603875855604803,
    0.3453501845399538,
    0.4077617526054382,
    0.3471292061110336,
    0.21286138420303655,
    0.4873136784881353,
    0.21318130667010946,
    0.21947651877999297,
    0.47245176322758214,
    0.3298814766108988,
    0.41743445731699463,
    0.5388514492660763,
    0.4203301640848317,
    0.8067351998140414,
    0.6865127366036177,
    0.4845273690919082,
    0.6244114382813375,
    0.7719468116760253,
    0.5311598031471173,
    0.45232512156168625,
    0.8269107699394226,
    0.5624363581339518,
    0.5210260172684988,
    0.4282756805419922,
    0.5540283819039664,
    0.5477308426052332,
    0.4279813490808011,
    0.6185464518765609,
    0.5591637155661979,
    0.43420585542917256,
    0.6986007068306205,
    0.3301446908464034,
    0.5002351805567742,
    0.6696252138664325,
    0.34758276169498764,
    0.5676332425326109,
    0.6775794816513856,
    0.6114779281119507,
    0.6364000413566827,
    0.7224635101854799,
    0.4994674399495124,
    0.7624177265912295,
    0.6047691027323405,
    0.3918229917685191,
    0.8550829529762269,
    0.5097380280494691,
];

// ------------------------------------------------------------- fixtures --

/// The fixture SciPy saw: a 64-bit LCG with Knuth's MMIX constants seeded at 1,
/// bits 40..63 scaled into `[0, 1)`. `the_fixture_is_the_one_scipy_saw` pins
/// three summaries of it.
fn lcg_volume(shape: [usize; 3]) -> Array3<f64> {
    let mut state: u64 = 1;
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f64 / 16_777_216.0
    })
}

/// `f(i, j, k) = 5i + 3j - 2k + 7`.
///
/// Three different coefficients, one of them negative, and a non-zero constant:
/// a factor applied to the wrong axis, a sign slip in the blend and a dropped
/// offset are each a different field.
fn affine(i: f64, j: f64, k: f64) -> f64 {
    5.0 * i + 3.0 * j - 2.0 * k + 7.0
}

fn affine_volume(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        affine(i as f64, j as f64, k as f64)
    })
}

fn linear(input: &Array3<f64>, ratios: &[Ratio; 3], into: [usize; 3]) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros((into[0], into[1], into[2]));
    resample_linear_into_with(input.view(), [0, 0, 0], [0, 0, 0], ratios, out.view_mut())
        .expect("the resampling must run");
    out
}

fn nearest(input: &Array3<f64>, ratios: &[Ratio; 3], into: [usize; 3]) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros((into[0], into[1], into[2]));
    resample_nearest_into_with(input.view(), [0, 0, 0], [0, 0, 0], ratios, out.view_mut())
        .expect("the resampling must run");
    out
}

fn worst(a: &Array3<f64>, b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

// ------------------------------------- the three candidate coordinate maps --

/// **The centred map the module states**, written here as the fraction rather
/// than as a ratio of floats: `x(o) = ((2o + 1) d - u) / (2u)`, then clamped to
/// the array.
///
/// Transcribed from `src/ops/resample.rs`'s header, not called from it, which is
/// the whole point — a test that asked the implementation where it sampled would
/// agree with any map at all.
fn centred(output: usize, ratio: Ratio, extent: usize) -> f64 {
    let up = ratio.up() as f64;
    let down = ratio.down() as f64;
    let x = ((2 * output + 1) as f64 * down - up) / (2.0 * up);
    x.clamp(0.0, (extent - 1) as f64)
}

/// The **corner-aligned** rival: `x(o) = o (n_in - 1) / (n_out - 1)`. Endpoints
/// map to endpoints; the module's header names it and says why it is not the
/// one — it is not shift-invariant, so a block and the whole volume would not
/// agree, and it divides by zero at an output extent of one.
fn corner_aligned(output: usize, into: usize, extent: usize) -> f64 {
    output as f64 * (extent - 1) as f64 / (into - 1) as f64
}

/// The **shift-free** rival: `x(o) = o d / u`, the centred map with the half
/// voxel left out at both ends. It is the one an implementation writes when it
/// reads "scale by `down/up`" literally.
fn shift_free(output: usize, ratio: Ratio, extent: usize) -> f64 {
    (output as f64 * ratio.down() as f64 / ratio.up() as f64).clamp(0.0, (extent - 1) as f64)
}

/// The affine field evaluated at whatever map is handed in — the closed-form
/// answer a linear interpolation must produce, because a linear interpolation
/// reproduces an affine function exactly.
fn affine_reference(map: impl Fn(usize, usize) -> f64) -> Array3<f64> {
    Array3::from_shape_fn((INTO[0], INTO[1], INTO[2]), |(i, j, k)| {
        affine(map(0, i), map(1, j), map(2, k))
    })
}

// ------------------------------------ claim 1: the map and the blend --

/// **A linear resampling of an affine field is that field at the mapped
/// coordinate**, and the coordinate is the centred one.
///
/// The closed form is computed in this file from the fraction the module's
/// header states. What it pins, all at once: the sample map, the clamp at both
/// ends of every axis, the blend weights, and which factor goes on which axis —
/// the last because the three coefficients of `f` are different.
///
/// The bound is `1e-14` absolute on a field whose values run from `-1` to `45`,
/// so about `2e-16` relative — a few ulp. It is not zero because the two sides
/// accumulate differently: this file evaluates `5x + 3y - 2z + 7` once at a
/// coordinate, and the implementation blends three times in `f64`. The measured
/// worst is `7.1e-15` and the test prints it.
#[test]
fn a_linear_resampling_of_an_affine_field_is_that_field_at_the_centred_coordinate() {
    let ratios = ratios();
    let got = linear(&affine_volume(FROM), &ratios, INTO);
    let want = affine_reference(|axis, output| centred(output, ratios[axis], FROM[axis]));

    let mut apart = 0.0f64;
    for ((i, j, k), value) in got.indexed_iter() {
        apart = apart.max((value - want[[i, j, k]]).abs());
    }
    assert!(
        apart < 1e-14,
        "the worst of {} voxels is {apart:e} from the closed-form answer",
        got.len()
    );
    assert_eq!(got.len(), 120, "the fixture changed shape");
    println!("120 voxels reproduce the affine field at the centred coordinate to {apart:e}");
}

/// **The liveness of the claim above.** Three rival conventions are evaluated on
/// the same fixture and measured.
///
/// All three produce an affine field too — that is exactly why the fixture has
/// to be checked rather than trusted — so what separates them is *where* they
/// sample, and each gap is printed. Measured: the rotated factors are `17.77`
/// away at the worst voxel, corner-aligned `1.74`, the shift-free map `0.67` —
/// which is the smallest of the three and still fourteen orders of magnitude
/// above the `1e-14` bound the parity test holds. The assertion is at `0.5`, so
/// a fixture on which any of them came within half a unit of `f` would fail
/// here rather than let the parity test pass for nothing.
#[test]
fn an_affine_field_cannot_be_reproduced_by_the_rival_conventions() {
    let ratios = ratios();
    let got = linear(&affine_volume(FROM), &ratios, INTO);

    let corner = affine_reference(|axis, output| corner_aligned(output, INTO[axis], FROM[axis]));
    let shift = affine_reference(|axis, output| shift_free(output, ratios[axis], FROM[axis]));
    let transposed = affine_reference(|axis, output| {
        // the same three factors, rotated onto the wrong axes
        centred(output, ratios[(axis + 1) % 3], FROM[axis])
    });

    for (name, rival) in [
        ("corner-aligned", corner),
        ("shift-free", shift),
        ("the factors rotated onto the wrong axes", transposed),
    ] {
        let gap = got
            .iter()
            .zip(rival.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            gap > 0.5,
            "{name} differs from the answer by only {gap:e}, so this fixture does not \
             discriminate it and the parity test above is not measuring the convention"
        );
        println!("{name}: worst voxel differs by {gap:.6}, against a 1e-14 parity bound");
    }
}

// ------------------------------------------- claim 2: the field is SciPy's --

/// The fixture is the volume SciPy saw.
#[test]
fn the_fixture_is_the_one_scipy_saw() {
    let volume = lcg_volume(FROM);
    assert_eq!(volume.len(), 140);
    assert_eq!(volume[[0, 0, 0]], 0.42320913076400757);
    assert_eq!(volume[[6, 4, 3]], 0.41306066513061523);
    let total: f64 = volume.iter().sum();
    assert!(
        (total - 70.80916684865952).abs() < 1e-12,
        "the fixture's sum is {total}, not the recorded 70.80916684865952"
    );
}

/// **The resampled field is `scipy.ndimage.zoom`'s**, at three different
/// rational factors on the three axes.
///
/// SciPy's `grid_mode=True` is the same centred map — `x = (o + 0.5)/z - 0.5`
/// with `z` the zoom — and `mode='nearest'` is the same clamp, so this is a
/// second program computing the same definition. It is compared through the
/// stated extent rather than through a factor; the header says why.
///
/// The bound is `1e-15` absolute on a field of values around `0.5`; the measured
/// worst is printed. Bit identity is not available and would not be honest to
/// ask for: SciPy runs its own spline path in C and this crate composes three
/// tap tables in Rust.
#[test]
fn the_resampled_field_is_scipys_zoom() {
    let got = linear(&lcg_volume(FROM), &ratios(), INTO);
    let apart = worst(&got, &SCIPY_ZOOM);
    assert!(
        apart < 1e-15,
        "the worst of 120 voxels differs from SciPy 1.15.2 by {apart:e}"
    );
    println!("120 voxels reproduce SciPy 1.15.2's zoom to {apart:e} at worst");
}

/// **The liveness of the SciPy comparison.** Two ways to be wrong that still
/// produce a plausible resampled volume of the right shape, each measured
/// against the same recording:
///
/// * the three factors rotated onto the wrong axes — the failure a
///   one-dimensional fixture cannot see at all, and the reason this file's
///   extents and factors are all different;
/// * nearest-neighbour where SciPy interpolated, which is right at every
///   coordinate that lands on a voxel and wrong in between.
#[test]
fn a_rotated_factor_and_a_nearest_pass_are_both_visible() {
    let volume = lcg_volume(FROM);
    let ratios = ratios();
    let rotated = [ratios[1], ratios[2], ratios[0]];

    for (name, field) in [
        (
            "the factors rotated onto the wrong axes",
            linear(&volume, &rotated, INTO),
        ),
        (
            "nearest where SciPy interpolated",
            nearest(&volume, &ratios, INTO),
        ),
    ] {
        let gap = worst(&field, &SCIPY_ZOOM);
        assert!(
            gap > 1e-9,
            "{name} differs from the recording by only {gap:e}, so the parity test is not \
             alive to it"
        );
        println!("{name}: worst voxel moves by {gap:e}");
    }
}
