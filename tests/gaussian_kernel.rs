// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The Gaussian's own numbers, against SciPy — the one thing about the
// smoothing that nothing else in the tree checks.**
//
// The gap this file closes
// ------------------------
// `ops::ridge` and `ops::smooth` are thoroughly tested, and none of it looks at
// what the weights *are*. What is checked today is:
//
// * `ridge::tests::the_kernel_is_normalised_symmetric_and_as_wide_as_the_radius_says`
//   — the kernel sums to one, is symmetric bit for bit, has `2r + 1` taps and a
//   non-zero outer tap;
// * `smooth::tests::the_reflected_answer_is_the_hand_written_sum_of_taps` — the
//   convolution is the literal sum of `taps[i] * samples[j]` with the reflected
//   indices spelled out, where `taps` is *read from the implementation*;
// * `ridge::tests::the_packed_and_strided_walks_agree_on_every_bit` — the fast
//   walk is the slow walk.
//
// Every one of those is a property of the convolution given a kernel, or a
// property of the kernel's shape. **A box kernel of the same width satisfies all
// of them**, and so does the sign-of-the-exponent slip `exp(-x^2 / sigma^2)`
// that leaves out the one half. `the_existing_properties_do_not_see_the_wrong_kernel`
// below builds both and measures that claim rather than asserting it in prose.
//
// So the weights get an outside witness: `scipy.ndimage`'s, which is a different
// program written by different people from the same definition.
//
// The three claims
// ----------------
//
// | claim | how |
// |---|---|
// | the 1-D weights are SciPy's | seven `(sigma, truncate)` pairs, every tap compared against `scipy.ndimage._filters._gaussian_kernel1d` 1.15.2 |
// | the separable 3-D pass is SciPy's `gaussian_filter` | 120 voxels x two boundary modes, at an **anisotropic** sigma, so a transposed axis is a failure |
// | the liveness of both | a box kernel and a half-exponent kernel measured against the same reference, and shown to pass the properties that exist today |
//
// Why a tolerance and not a digest
// --------------------------------
// `tests/distance_transform.rs` pins SciPy bit for bit, because at unit sampling
// both programs produce the correctly rounded square root of the same exact
// integer and a tolerance would have hidden that. Nothing of the sort is
// available here. SciPy forms the exponent as `-0.5 / sigma^2 * x^2` and this
// crate as `-0.5 * (x / sigma) * (x / sigma)`; the products are summed in a
// different order, and the three separable passes run in C against Rust. The
// bits will not agree and it would be dishonest to demand that they do. What is
// pinned instead is a **measured** bound, printed by the tests, with the
// discriminating alternatives measured beside it so that the bound is known to
// be far below the difference a wrong kernel makes. Measured: the weights agree
// with SciPy to `9.1e-16` relative at worst and the fields to `2.2e-16`
// absolute, against a half-exponent kernel that is `0.99` off relatively and a
// transposed sigma that moves a voxel by `2.0e-1` — fourteen to fifteen orders
// of magnitude of daylight, all of it printed by the tests below.
//
// Where SciPy and this crate genuinely differ, and why it is not a bug
// ---------------------------------------------------------------------
// The **radius** conventions are not the same function. SciPy takes
// `int(truncate * sigma + 0.5)`; this crate takes `ceil(truncate * sigma)`. They
// agree on most inputs and part company when the fractional part of
// `truncate * sigma` is in `(0.5, 1)` — at `sigma = 0.7, truncate = 3.0` SciPy
// uses radius 2 and this crate radius 3, which
// `the_radius_conventions_are_not_the_same_function` records. `ceil` is the
// wider of the two and therefore never truncates a tap SciPy kept, so it is a
// defensible reading of "truncate at this many standard deviations" and not a
// defect; but it means a parity fixture has to be chosen where the two agree, or
// the comparison measures the truncation rule instead of the weights. Every
// pair recorded below was checked for that in the generating script, which
// asserts it.
//
// Reproducing the recordings
// --------------------------
// `python3 tools/reference_values.py gaussian` prints every constant in this
// file, and `--versions` prints the versions it printed them under. Nothing here
// is a number somebody typed.

use ndarray::{Array3, ArrayView3};

use blockflow::ops::ridge::{
    gaussian_radius, gaussian_smooth_into_with, gaussian_weights, Boundary,
};

// ------------------------------------------------------- the recordings --

/// `scipy.ndimage._filters._gaussian_kernel1d(sigma, order=0, radius)` under
/// SciPy 1.15.2, at the seven `(sigma, truncate)` pairs where SciPy's radius and
/// this crate's agree. Printed by `tools/reference_values.py gaussian`.
///
/// The seven are not arbitrary: `0.5` is narrow enough that the kernel is
/// dominated by its centre tap, `2.5` wide enough that the outer taps are three
/// orders of magnitude down, `1.0` appears at two truncations so that the same
/// sigma is seen at two widths, and `1.25` gives a radius that is neither
/// `sigma` nor a whole multiple of it.
#[allow(clippy::type_complexity)]
const SCIPY_KERNELS: &[(f64, f64, &[f64])] = &[
    (
        0.5,
        4.0,
        &[
            0.00026386508273735414,
            0.10645077197359151,
            0.7865707258873422,
            0.10645077197359151,
            0.00026386508273735414,
        ],
    ),
    (
        1.0,
        4.0,
        &[
            0.00013383062461474175,
            0.0044318616200312655,
            0.05399112742070441,
            0.24197144565660073,
            0.39894346935609776,
            0.24197144565660073,
            0.05399112742070441,
            0.0044318616200312655,
            0.00013383062461474175,
        ],
    ),
    (
        1.0,
        3.0,
        &[
            0.004433048175243745,
            0.054005582622414484,
            0.2420362293761143,
            0.3990502796524549,
            0.2420362293761143,
            0.054005582622414484,
            0.004433048175243745,
        ],
    ),
    (
        1.25,
        4.0,
        &[
            0.00010706486987605864,
            0.0019072828399117412,
            0.017915739574145686,
            0.0887372390177623,
            0.2317547342039941,
            0.3191558789886203,
            0.2317547342039941,
            0.0887372390177623,
            0.017915739574145686,
            0.0019072828399117412,
            0.00010706486987605864,
        ],
    ),
    (
        1.5,
        3.0,
        &[
            0.00102838008447911,
            0.007598758135239185,
            0.03600077212843083,
            0.10936068950970002,
            0.2130055377112537,
            0.26601172486179436,
            0.2130055377112537,
            0.10936068950970002,
            0.03600077212843083,
            0.007598758135239185,
            0.00102838008447911,
        ],
    ),
    (
        2.0,
        4.0,
        &[
            6.691628957263553e-05,
            0.0004363490205067883,
            0.002215963172596555,
            0.00876430436278587,
            0.026995957967298846,
            0.06475993660472744,
            0.12098748976534904,
            0.17603575888479034,
            0.199474647864745,
            0.17603575888479034,
            0.12098748976534904,
            0.06475993660472744,
            0.026995957967298846,
            0.00876430436278587,
            0.002215963172596555,
            0.0004363490205067883,
            6.691628957263553e-05,
        ],
    ),
    (
        2.5,
        3.0,
        &[
            0.0009542270849561244,
            0.003168145492896394,
            0.008963371132443686,
            0.021609788831716978,
            0.04439586785088073,
            0.07772262497331667,
            0.11594853150070399,
            0.14739947215128454,
            0.15967594196360177,
            0.14739947215128454,
            0.11594853150070399,
            0.07772262497331667,
            0.04439586785088073,
            0.021609788831716978,
            0.008963371132443686,
            0.003168145492896394,
            0.0009542270849561244,
        ],
    ),
];

/// The anisotropic sigma the field recordings were taken at.
///
/// **Three different values, deliberately.** An isotropic sigma cannot tell
/// `kernels[0]` applied to axis 0 from `kernels[0]` applied to axis 2, so a
/// transposed axis order would reproduce SciPy exactly and the test would say
/// nothing about it. With `[1.5, 0.5, 1.0]` every permutation is a different
/// field, which `a_transposed_sigma_is_a_different_field` measures.
const FIELD_SIGMA: [f64; 3] = [1.5, 0.5, 1.0];
const FIELD_TRUNCATE: f64 = 4.0;
const FIELD_SHAPE: (usize, usize, usize) = (6, 5, 4);

/// `scipy.ndimage.gaussian_filter(lcg_volume((6, 5, 4)), sigma=(1.5, 0.5, 1.0),
/// truncate=4.0, mode='reflect')` under SciPy 1.15.2, in C order.
const SCIPY_REFLECT: [f64; 120] = [
    0.5829520848944852,
    0.6075833940035008,
    0.5777102781071881,
    0.5096563413671065,
    0.5938066327486085,
    0.5454361093672664,
    0.44589544428334893,
    0.3300295181905452,
    0.6237562876082335,
    0.542762583271999,
    0.5024250394569106,
    0.4965115426909004,
    0.5323743257257634,
    0.4775681809769808,
    0.4768025527221722,
    0.5226720491233454,
    0.5078262801237493,
    0.47651062782519205,
    0.41314551976197483,
    0.3433434620717045,
    0.5591440107454502,
    0.5934518157705919,
    0.5770340523069403,
    0.5213615023826188,
    0.551545175557827,
    0.5102873667852689,
    0.4321881485834267,
    0.34157655684929394,
    0.572175926845772,
    0.5095219741297613,
    0.4766302965704552,
    0.4673327147265829,
    0.5218501511042577,
    0.49368421917517397,
    0.5105558723344686,
    0.5546584900303362,
    0.49435111867537646,
    0.48796216384184266,
    0.4401552672786907,
    0.37087796862094746,
    0.5054000815646551,
    0.5466702491372413,
    0.5543940452433567,
    0.5258152386289634,
    0.485528065049864,
    0.4556669662011278,
    0.40933930586540657,
    0.34925412396320415,
    0.5005331532281639,
    0.46345725589757136,
    0.44363929491451554,
    0.42788068220525377,
    0.5323152006723062,
    0.5362019926516414,
    0.5659223135978606,
    0.6009575156687588,
    0.4717105805155205,
    0.5136985060935373,
    0.4977364836168609,
    0.43167725986904937,
    0.44721661940923024,
    0.48053077284437895,
    0.5148335600180073,
    0.5229971676856056,
    0.41829806720853957,
    0.4095432430608259,
    0.3918752652562607,
    0.3465172541499629,
    0.4420569969548519,
    0.4301237005148463,
    0.4251495361416282,
    0.4067890584297766,
    0.580475791607379,
    0.5976401694806655,
    0.6099501080036657,
    0.6193777673727271,
    0.4648442544447813,
    0.5555403840379135,
    0.5710395686925478,
    0.5096639619630202,
    0.41680893945267306,
    0.43347715461877934,
    0.4931123817907186,
    0.5381346118369981,
    0.36783213572730455,
    0.3896985895274344,
    0.39105782200656675,
    0.34590332460097917,
    0.40269179626033735,
    0.41272384323359357,
    0.4243018545193632,
    0.4192570541855517,
    0.6420781717173527,
    0.6479977427006955,
    0.616787025842319,
    0.591401215866521,
    0.4904921994118945,
    0.6085286488194228,
    0.6370738300178809,
    0.5756828504079757,
    0.41180779467551715,
    0.41774517443114445,
    0.49639002537065446,
    0.5667373201352826,
    0.3432215391739024,
    0.38845479153427237,
    0.3988720562527045,
    0.35209591039906535,
    0.38016295283154444,
    0.403324901941208,
    0.42864299724462473,
    0.44409324671438133,
    0.6809628248909885,
    0.6700537998554402,
    0.6028354198158382,
    0.5542065325832954,
    0.5223867209926971,
    0.6490021178754761,
    0.6770677160242222,
    0.6119099660295383,
];

/// The same call with `mode='nearest'`, which is what this crate calls
/// [`Boundary::Clamp`] and what it uses by default.
const SCIPY_NEAREST: [f64; 120] = [
    0.5419450129083311,
    0.5759756385381634,
    0.5552209313013882,
    0.476861463931124,
    0.6200846840972669,
    0.5482580738398837,
    0.440133849914684,
    0.301744166795313,
    0.6453260150478808,
    0.537596164508314,
    0.5075437952010733,
    0.5045841877188004,
    0.5693516677705547,
    0.4796189516107901,
    0.47524625831636413,
    0.5480280935589961,
    0.501369208138861,
    0.49023560849349024,
    0.4345328210066101,
    0.3427752288244918,
    0.5389701752463318,
    0.5825973376043567,
    0.5696883613745116,
    0.5054462393114054,
    0.5620783882604986,
    0.5103193951854192,
    0.42924048363161416,
    0.3243750474652555,
    0.5846738021547673,
    0.5070443449181283,
    0.4771787569833481,
    0.4671131405068673,
    0.5371691940463041,
    0.49472479027908817,
    0.5111936316300416,
    0.5677443860713749,
    0.4895533041482422,
    0.4927373227395483,
    0.44729572408863927,
    0.3633906620201293,
    0.4950688841470232,
    0.5441156931888542,
    0.5530106709370277,
    0.5191754917064038,
    0.49216511296532495,
    0.45599587880949705,
    0.4081274453420516,
    0.3375403834302637,
    0.5091967633687373,
    0.4631623600436505,
    0.44329251872299025,
    0.4238189484413833,
    0.5356465515716098,
    0.5363358003559376,
    0.5662091886738106,
    0.6066444714017252,
    0.4615140573450049,
    0.5141053699865816,
    0.49863583857024996,
    0.42038189571038365,
    0.4447491159497788,
    0.4814580153101153,
    0.5175586385485338,
    0.5261000892886333,
    0.4233622689296321,
    0.4107428080415699,
    0.3918138263860599,
    0.336931572333687,
    0.44717730372523967,
    0.4296783112408055,
    0.4246304225161936,
    0.40343109408049915,
    0.5765704388545404,
    0.596538062873861,
    0.6088572737476725,
    0.6208493528829842,
    0.4472364171241068,
    0.5549960422051686,
    0.5711999261997777,
    0.4981754411548968,
    0.4257881725417333,
    0.4405961376089679,
    0.5079898006973571,
    0.5648863983363905,
    0.37007950608308904,
    0.3940880070900622,
    0.39465440788051165,
    0.34295950807554587,
    0.4031987491582611,
    0.4091329265046499,
    0.42411197585485805,
    0.4304902986524878,
    0.6338773819411503,
    0.6436175399313999,
    0.6101232608911235,
    0.5869038191224503,
    0.4701331847061766,
    0.6112143091240814,
    0.6412935669911041,
    0.5684615501130446,
    0.4368599888807564,
    0.4389602942915137,
    0.5433889170078943,
    0.6510638733865031,
    0.3441708114213462,
    0.40218272591842685,
    0.4132788437626916,
    0.36997393229604403,
    0.37416250148221547,
    0.39089184980908986,
    0.42976290566920744,
    0.49436597425790957,
    0.6727125970734011,
    0.6586952025750566,
    0.5792693571535779,
    0.5335288999934524,
    0.5110503766391054,
    0.6657383025724852,
    0.6956466057115602,
    0.6178766415010035,
];

// ---------------------------------------------------------- the fixture --

/// The volume the field recordings were taken on, built by the same recurrence
/// on both sides so that no array has to be shipped.
///
/// A 64-bit LCG with Knuth's MMIX constants seeded at 1, taking bits 40..63 and
/// scaling into `[0, 1)`. `the_fixture_is_the_one_scipy_saw` pins three
/// summaries of it, so a generator that drifted is a failure here rather than a
/// silent comparison of a different volume.
fn lcg_volume(shape: (usize, usize, usize)) -> Array3<f64> {
    let mut state: u64 = 1;
    Array3::from_shape_fn(shape, |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f64 / 16_777_216.0
    })
}

/// A **box** kernel of the width the Gaussian would have had: the
/// plausible-but-wrong implementation this file's liveness rests on.
fn box_kernel(sigma: f64, truncate: f64) -> Vec<f64> {
    let radius = gaussian_radius(sigma, truncate);
    let taps = 2 * radius + 1;
    vec![1.0 / taps as f64; taps]
}

/// The same Gaussian with the **one half left out of the exponent** —
/// `exp(-x^2 / sigma^2)` rather than `exp(-x^2 / 2 sigma^2)`, which is the
/// slip of writing the density from memory. It is still a Gaussian, of
/// `sigma / sqrt(2)`, so it is normalised, exactly symmetric, has the declared
/// number of taps and a non-zero outer tap.
fn half_exponent_kernel(sigma: f64, truncate: f64) -> Vec<f64> {
    let radius = gaussian_radius(sigma, truncate) as isize;
    let mut weights: Vec<f64> = (-radius..=radius)
        .map(|step| {
            let ratio = step as f64 / sigma;
            (-ratio * ratio).exp()
        })
        .collect();
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

fn worst_relative(ours: &[f64], reference: &[f64]) -> f64 {
    assert_eq!(
        ours.len(),
        reference.len(),
        "the kernels are different widths"
    );
    ours.iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs() / b.abs())
        .fold(0.0f64, f64::max)
}

fn worst_absolute(ours: ArrayView3<'_, f64>, reference: &[f64]) -> f64 {
    ours.iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max)
}

fn smooth_with(
    input: ArrayView3<'_, f64>,
    kernels: &[Vec<f64>; 3],
    boundary: Boundary,
) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros(input.raw_dim());
    gaussian_smooth_into_with(input, kernels, boundary, out.view_mut())
        .expect("the smoothing must run");
    out
}

fn gaussian_kernels(sigma: [f64; 3], truncate: f64) -> [Vec<f64>; 3] {
    [
        gaussian_weights(sigma[0], truncate).expect("axis 0"),
        gaussian_weights(sigma[1], truncate).expect("axis 1"),
        gaussian_weights(sigma[2], truncate).expect("axis 2"),
    ]
}

// ------------------------------------------- claim 1: the weights are SciPy's --

/// **The taps themselves, against a different program.**
///
/// This is the only assertion in the tree about what a Gaussian weight *is*.
/// Every tap of every recorded kernel is compared, relatively, so that the outer
/// taps — which are four orders of magnitude below the centre and are exactly
/// where a truncation or a normalisation error shows — are held to the same
/// standard as the middle.
///
/// The bound is `1e-15` relative, which is a few ulp; the measured worst is
/// printed. It is not tighter because the two form the exponent differently:
/// SciPy computes `-0.5 / sigma^2 * x^2` and this crate `-0.5 * (x/sigma)^2`.
#[test]
fn the_weights_are_scipys_weights() {
    let mut worst = 0.0f64;
    let mut taps_checked = 0usize;
    for (sigma, truncate, reference) in SCIPY_KERNELS {
        let ours = gaussian_weights(*sigma, *truncate).expect("the kernel must build");
        assert_eq!(
            ours.len(),
            reference.len(),
            "sigma {sigma}, truncate {truncate}: this crate's radius is not SciPy's, so the \
             comparison would be measuring the truncation rule; see the header"
        );
        let apart = worst_relative(&ours, reference);
        assert!(
            apart < 1e-15,
            "sigma {sigma}, truncate {truncate}: worst tap differs from SciPy 1.15.2 by {apart:e} \
             relative"
        );
        worst = worst.max(apart);
        taps_checked += ours.len();
    }
    // The measured total, so a pair quietly dropped from the table is a failure
    // here rather than a smaller number nobody reads.
    assert_eq!(taps_checked, 77, "a (sigma, truncate) pair left the table");
    println!("77 taps across 7 kernels reproduce SciPy 1.15.2 to {worst:e} relative at worst");
}

/// **The liveness of the claim above, measured rather than argued.**
///
/// Two plausible wrong kernels are built — a box of the same width, and the
/// Gaussian with the one half missing from the exponent — and two things are
/// shown about each:
///
/// 1. it passes *every* property the existing tests assert: normalised to
///    within `1e-12`, exactly symmetric bit for bit, `2r + 1` taps, non-zero
///    outer tap. So nothing in the tree today distinguishes it from the real
///    kernel;
/// 2. it is far outside the `1e-15` bound `the_weights_are_scipys_weights`
///    holds — by more than ten orders of magnitude on the measured numbers.
///
/// That pair is what makes the tolerance a discriminating one rather than a
/// number chosen to make a test pass.
#[test]
fn the_existing_properties_do_not_see_the_wrong_kernel() {
    let (sigma, truncate) = (1.5, 3.0);
    let radius = gaussian_radius(sigma, truncate);
    let reference = SCIPY_KERNELS
        .iter()
        .find(|(s, t, _)| *s == sigma && *t == truncate)
        .expect("sigma 1.5 truncate 3.0 is recorded")
        .2;

    for (name, wrong) in [
        ("box", box_kernel(sigma, truncate)),
        ("half exponent", half_exponent_kernel(sigma, truncate)),
    ] {
        // (1) it satisfies everything that is asserted today.
        assert_eq!(wrong.len(), 2 * radius + 1, "{name}: width");
        for step in 0..=radius {
            assert_eq!(
                wrong[radius - step].to_bits(),
                wrong[radius + step].to_bits(),
                "{name}: symmetry, bit for bit"
            );
        }
        assert!(
            (wrong.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "{name}: normalisation"
        );
        assert!(wrong[0] > 0.0, "{name}: the outer tap is not zero");

        // (2) and it is nowhere near SciPy.
        let apart = worst_relative(&wrong, reference);
        assert!(
            apart > 1e-4,
            "{name} is only {apart:e} from the true kernel, so it is not a discriminating \
             alternative and this test is not measuring liveness"
        );
        println!(
            "{name} kernel: passes width, symmetry, normalisation and outer-tap; differs from \
             SciPy by {apart:e} relative, against the 1e-15 bound the parity test holds"
        );
    }
}

/// **Where the two programs genuinely disagree**, recorded so that a future
/// reader does not mistake it for a defect or try to "fix" it.
///
/// SciPy's radius is `int(truncate * sigma + 0.5)`; this crate's is
/// `ceil(truncate * sigma)`. The two are the same function only when the
/// fractional part of `truncate * sigma` is not in `(0.5, 1)`. `ceil` is the
/// wider, so this crate never drops a tap SciPy kept — which is why the
/// difference is a reading of the parameter rather than an error, and why the
/// parity fixtures above are all chosen where the two agree.
#[test]
fn the_radius_conventions_are_not_the_same_function() {
    let scipy_radius = |sigma: f64, truncate: f64| (truncate * sigma + 0.5) as usize;

    // Agreeing, on every pair the parity test uses.
    for (sigma, truncate, reference) in SCIPY_KERNELS {
        assert_eq!(
            gaussian_radius(*sigma, *truncate),
            scipy_radius(*sigma, *truncate)
        );
        assert_eq!(2 * gaussian_radius(*sigma, *truncate) + 1, reference.len());
    }

    // And parting company, with the case named.
    assert_eq!(gaussian_radius(0.7, 3.0), 3);
    assert_eq!(scipy_radius(0.7, 3.0), 2);
    assert!(
        gaussian_radius(0.7, 3.0) >= scipy_radius(0.7, 3.0),
        "ceil must never be the narrower of the two, which is the whole reason the difference \
         is acceptable"
    );
    println!("sigma 0.7 truncate 3.0: this crate reaches 3 voxels, SciPy 2");
}

// -------------------------------- claim 2: the separable pass is SciPy's --

/// The fixture is the volume SciPy saw. Three summaries of it, so a drifted
/// generator fails here and not as a mysterious field mismatch.
#[test]
fn the_fixture_is_the_one_scipy_saw() {
    let volume = lcg_volume(FIELD_SHAPE);
    assert_eq!(volume.len(), 120);
    assert_eq!(volume[[0, 0, 0]], 0.42320913076400757);
    assert_eq!(volume[[5, 4, 3]], 0.6174697279930115);
    let total: f64 = volume.iter().sum();
    assert!(
        (total - 59.50238960981369).abs() < 1e-12,
        "the fixture's sum is {total}, not the recorded 59.50238960981369"
    );
}

/// **The whole separable pass, against `scipy.ndimage.gaussian_filter`.**
///
/// 120 voxels at an anisotropic sigma, in both boundary modes this crate offers,
/// against SciPy's `'reflect'` and `'nearest'`. What this catches that the
/// kernel test does not: the axis a kernel is applied to, the order of the three
/// passes at the boundary, and the boundary convention itself — SciPy's
/// `'reflect'` is `d c b a | a b c d | d c b a`, which is the convention
/// `Boundary::Reflect`'s documentation states, and this is the first check that
/// the two statements describe the same thing.
///
/// The bound is `1e-15` absolute on a field whose values are order `0.5`, and
/// the measured worst is printed. A digest would have been better and is not
/// available: see the header.
#[test]
fn the_separable_pass_is_scipys_gaussian_filter() {
    let volume = lcg_volume(FIELD_SHAPE);
    let kernels = gaussian_kernels(FIELD_SIGMA, FIELD_TRUNCATE);

    for (name, boundary, reference) in [
        ("reflect", Boundary::Reflect, &SCIPY_REFLECT),
        ("nearest / clamp", Boundary::Clamp, &SCIPY_NEAREST),
    ] {
        let ours = smooth_with(volume.view(), &kernels, boundary);
        let apart = worst_absolute(ours.view(), reference);
        assert!(
            apart < 1e-15,
            "{name}: worst voxel differs from SciPy 1.15.2 by {apart:e}"
        );
        println!("{name}: 120 voxels reproduce SciPy 1.15.2 to {apart:e} at worst");
    }
}

/// **The liveness of the field comparison.** Three ways to be wrong that leave
/// the answer looking like a smoothed volume, each measured against the same
/// reference:
///
/// * the sigma transposed onto the wrong axes — which an isotropic fixture
///   could not see at all, and is why `FIELD_SIGMA` has three different values;
/// * the box kernel of the same widths;
/// * the wrong boundary mode, which is only wrong within one radius of a face
///   and is therefore the subtlest of the three.
///
/// Each is asserted to be at least a million times the `1e-15` bound the parity
/// test holds, and the measured gaps are printed.
#[test]
fn a_transposed_sigma_a_box_kernel_and_the_wrong_boundary_are_all_visible() {
    let volume = lcg_volume(FIELD_SHAPE);
    let bound = 1e-15;

    let transposed = gaussian_kernels(
        [FIELD_SIGMA[2], FIELD_SIGMA[0], FIELD_SIGMA[1]],
        FIELD_TRUNCATE,
    );
    let field = smooth_with(volume.view(), &transposed, Boundary::Reflect);
    let rotated_axes = worst_absolute(field.view(), &SCIPY_REFLECT);

    let boxes = [
        box_kernel(FIELD_SIGMA[0], FIELD_TRUNCATE),
        box_kernel(FIELD_SIGMA[1], FIELD_TRUNCATE),
        box_kernel(FIELD_SIGMA[2], FIELD_TRUNCATE),
    ];
    let field = smooth_with(volume.view(), &boxes, Boundary::Reflect);
    let box_gap = worst_absolute(field.view(), &SCIPY_REFLECT);

    let kernels = gaussian_kernels(FIELD_SIGMA, FIELD_TRUNCATE);
    let field = smooth_with(volume.view(), &kernels, Boundary::Clamp);
    let boundary_gap = worst_absolute(field.view(), &SCIPY_REFLECT);

    for (name, gap) in [
        ("sigma rotated onto the wrong axes", rotated_axes),
        ("box kernels of the same widths", box_gap),
        ("clamped where SciPy reflected", boundary_gap),
    ] {
        assert!(
            gap > bound * 1e6,
            "{name} differs from the reference by only {gap:e}, which is not enough above the \
             {bound:e} bound for the parity test to be alive to it"
        );
        println!("{name}: worst voxel moves by {gap:e}");
    }
}
