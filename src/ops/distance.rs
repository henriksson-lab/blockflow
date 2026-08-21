// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation
// and from two references' call sites, not adapted from either implementation
// of it.
//
// **The exact Euclidean distance transform**: a mask in, the distance from
// every foreground voxel to the nearest background voxel out.
// =========================================================================
//
// The operation, and why "exact" is the whole of it
// -------------------------------------------------
// `distance(v) = min over background w of |v - w|`, with the per-axis pitch
// [`DistanceParams::sampling`] scaling each component. Nothing about that
// definition is approximate, and the reason this module is worth its length is
// that almost every cheap way of computing it *is*.
//
// A **chamfer** — the two-pass forward/backward propagation of a fixed
// neighbourhood's weights — is exact along the directions its offset set can
// represent and wrong in between. On a sphere it is wrong by under half a voxel
// and looks right; on a thin oblique sheet it is wrong nearly everywhere. That
// is not a rhetorical contrast: [`chamfer_distance`] is in this module as a
// **control**, and `tests/distance_transform.rs` measures both halves of it on
// the two fixtures — 9484 of 12 167 voxels wrong on the oblique slab, 1167 and
// by under half a voxel on the centred ball. A parity test built on the ball
// alone could not tell the two implementations apart, which is why the slab is
// the fixture that ships.
//
// **Felzenszwalb–Huttenlocher** is what is implemented: seed `0` on background
// and `+inf` on foreground, then three independent 1-D passes, one per axis,
// each the lower envelope of a set of parabolas in `O(n)`. SciPy's
// `scipy.ndimage.distance_transform_edt` uses Maurer's exact dimensionality
// reduction instead. Both are *exact*, and at unit sampling both produce the
// correctly rounded square root of an exact integer, so they agree bit for bit
// rather than to a tolerance — which is what
// `the_field_reproduces_scipys_own_numbers` asserts against four recorded
// digests rather than describes.
//
// Decomposition: what is available, stated narrowly
// --------------------------------------------------
// **An exact distance transform is not halo-decomposable.** The nearest
// background voxel to an interior voxel can be arbitrarily far away, and on real
// volumes it is: a `320 x 528 x 456` mask this transform has been run over has a
// largest value of 109.508 voxels, mean 29.6 over the foreground. No declared
// reach bounds that in general, and there is no block size at which a halo makes
// a block-local transform into the volume's — a halo of 110 on a `64`-edge block
// is not a halo, it is the volume.
//
// The **separable** form is decomposable in the one sense that is available, and
// this crate's reach algebra can state exactly that sense: each pass declares
// [`AxisReach::All`](crate::reach::AxisReach::All) on the axis it sweeps and
// [`AxisReach::none`](crate::reach::AxisReach::none) on the other two. A 1-D
// lower envelope over a lane is a function of that lane alone, so the other two
// axes are free, and `Reach::is_whole_axis` on one axis drops that axis from
// `decomposition::splittable_axes` while leaving the other two cuttable. Four
// phases: three sweeps on three lattices, and a pointwise finish.
//
// **This is a same-rank whole-axis stencil, not a collapsed-axis projection.**
// The distinction matters and has been got wrong before. The reach is declared
// in the phase's own frame — `Space::phase_voxels()`, the default, no
// `.in_space(..)` — and the op does not change shape: it reads `n` voxels of the
// swept axis and writes `n`. A projection declares `All` in
// `Space::source_voxels()` and writes an axis of extent 1, and it is
// `Decomposition::check`'s fetch verification that keeps *that* honest. Nothing
// here fetches across grids, so nothing here needs it.
//
// **The whole-axis mandate, and what the guard inside `apply` is still for.**
// Two failures are possible and only one of them is a correctness failure:
//
// * A lattice that grants a **short halo** against an `AxisReach::All` axis is
//   *refused*: no block of a cut axis has a trustworthy voxel, every block stays
//   degenerate, and the tiling check in `Decomposition::check` fires by name.
//   `tests/reach_space.rs` pins that, and `tests/distance_transform.rs` provokes
//   it on this op's own plan. So a plan cannot quietly compute a block-local
//   distance transform: the declaration *mandates* that the swept axis is left
//   whole or given a whole-axis halo, and there is no third option.
// * A lattice that **cuts the swept axis with the full halo** is *accepted*, and
//   is right: every block re-reads the whole lane, so every core is trustworthy
//   and the regions tile. It is redundant, not wrong, and the cost model rather
//   than a guard is what should drive such a phase to one block per lane.
//
// So the invariant is enforced at *planning* time, and [`sweep_grid`] is the
// lattice that makes it cheap rather than merely correct.
//
// **Where that enforcement comes from, exactly**, because it is easy to credit
// the wrong machinery. This reach is stated in the phase's own frame, where
// `Reach::space().clamp_is_an_edge()` is already true, so the per-axis "consumed
// axis" grant in `BlockGeometry::derive_with` never fires for it and never had
// to: the ordinary halo arithmetic leaves every block of a short-haloed cut axis
// with no trustworthy voxel, and the tiling check reports that. The grant, and
// `Decomposition::check`'s comparison of a whole-axis claim against the *fetch*,
// exist for a reach stated in `Space::source_voxels()` — a phase whose own edges
// are interior positions of the array it reads, which is a projection and is not
// this. Nothing here fetches across grids, so nothing here needs either. What
// the two together did do is make the same mandate expressible for the other
// shape, which is why `docs/ops-survey`'s ask for a `BlockConstraint` on one
// axis is answered by a *declaration* rather than by a constraint type.
// [`DistanceSweepOp::apply`] refuses a buffer that does not span its axis from
// voxel 0 anyway, and that guard is **not** redundant — nothing above it ever
// made it so, because nothing above it runs at the same time — it is the only thing
// standing between a caller who invokes a public `BlockOp` outside any plan and
// a complete, well-formed, wrong volume. Everything above happens when a
// `Decomposition` is built; `apply` is reachable without building one. Inside a
// plan the branch never fires, which is exactly why
// `a_sweep_handed_a_partial_lane_refuses_rather_than_answering` builds the
// anchor by hand: an assertion nothing can run is not an assertion.
//
// The parameters, and why each is one
// ------------------------------------
// Rule 1 of this crate is that everything is a parameter. Two of these are
// parameters because two references disagree about them, which is the strongest
// form of the argument:
//
// | choice | verdict | why |
// |---|---|---|
// | `sampling` | **parameter** | `distance_transform_edt`'s own keyword. It reaches the field and changes it; of the two reference call sites one omits it and the other writes `[1, 1, 1]` out by hand. It is a per-axis voxel pitch, which is general, not a property of any particular image. |
// | the all-foreground value | **parameter**, and the references disagree | a mask with **no** background voxel has no nearest background voxel. SciPy returns the distance to a phantom background voxel at feature index `(-1, 0, 0)`; a second implementation returns `+inf`. Neither is documented, neither is clipping, and both readings are therefore available — see [`Unbounded`]. |
// | the pass order | **synonym** | the three 1-D passes commute — the final value is the minimum over all background voxels however the axes are ordered — and at unit sampling every intermediate is an exact integer below `2^53`, so all six orders are *bit*-identical. Asserted over all six rather than argued. |
// | the block lattice | **synonym** | a 1-D pass over a lane is a function of the lane alone and the swept axis is never cut, so a block decides where a lane is computed and never what it is. Asserted over eleven lattices anyway, from one voxel to the whole volume, because "structurally invariant" is what an untested claim always sounds like. |
// | the accumulator width | **not a choice** | `f64`, as in both references. At unit sampling it is not even a convention: every intermediate is an integer below `2^53` and the arithmetic is exact. |
// | the foreground predicate | **not this op's** | a mask arrives as a mask. Which threshold made it is the caller's, and `ops::voxelwise::Threshold` is where a caller says so. |
//
// **The squared field is where comparison is exact**, and it is public for that
// reason. [`squared_distance_transform`] is the last surface before the square
// root, and at unit sampling every value in it is an exact integer. That makes a
// threshold comparison against an integer exact with no square root at all, and
// it makes a comparison against a stored field a comparison of integers.
//
// The distinction is not theoretical. A stored `float64` field produced by
// another library's `sqrt` was found to differ from the correctly rounded value
// at **765 926 of 77 045 760** voxels, every one of them by exactly one ulp and
// every one low — the signature of a `sqrt` that is faithful but not correctly
// rounded — while the two fields' *squared* values agreed at every one of those
// 77 million voxels. So a caller who needs a bit-exact comparison against
// somebody else's field compares squares, and a caller who needs a distance
// takes the root and accepts that its last bit is a property of a libm.
//
// The all-foreground volume, in detail
// -------------------------------------
// * [`Unbounded::PhantomOrigin`] is SciPy's. Measured rather than read off a
//   doc: `distance_transform_edt` on an all-ones volume with
//   `return_indices=True` hands back a feature index of `[-1, 0, 0]` at *every*
//   voxel, and the distances are exactly `sqrt((i+1)^2 + j^2 + k^2)` — `1.0` at
//   the origin, `sqrt(17)` at `(1, 2, 3)` of a `2 x 3 x 4` volume. It is the
//   algorithm's uninitialised sentinel surfacing as a number, it is
//   undocumented, and it is deterministic in SciPy 1.15.2.
// * [`Unbounded::Infinite`] is the other reference's: a 1-D pass returns
//   all-infinite when no parabola on the lane is finite, and after three passes
//   a voxel is infinite exactly when the whole volume is foreground.
//
// Neither clips, and **the case is decidable block by block**, which is why it
// costs the plan nothing: after the three passes a voxel's squared value is
// infinite *iff the whole volume is foreground* — the axis-0 pass is infinite on
// a lane iff that lane is all foreground, the axis-1 pass iff that plane is, the
// axis-2 pass iff the volume is — so a pointwise finish that sees an infinity
// knows the global fact without being told it. [`DistanceFinishOp`] does need
// the [`Anchor`], because the phantom point is a point of the **volume** and a
// block that re-anchored to itself would produce one phantom voxel per block.
//
// What it costs, and why blocking it is worth anything
// ----------------------------------------------------
// Residency is the reason this is worth blocking at all. Per block, a sweep
// phase holds `volume[swept] x block[a] x block[b] x 8 bytes` twice, in and out:
//
// | volume | resident, whole-volume `f64` | worst per-block sweep at `block = 64` | at `block = 32` |
// |---|---|---|---|
// | `320 x 528 x 456`, 77.0 Mvoxel | 616.4 MB per image | `528 x 64 x 64 x 8 x 2` = **34.6 MB** | 8.7 MB |
// | `404 x 1304 x 3369`, 1.775 Gvoxel | **14.2 GB** per image | `3369 x 64 x 64 x 8 x 2` = **220.8 MB** | 55.2 MB |
//
// The worst phase is the sweep on the volume's **longest** axis — worth saying
// because the obvious guess is the last pass and it is not. These are not this
// module's multiplications: [`working_set_bytes`] reads them back out of
// `decomposition::price_phase` for this plan, and the test checks the table
// against it at both scales without running either.
//
// [`distance_transform`] is resident and is the sensible way to compute a field
// that fits; at the second row above it is not, and [`plan`] is.

use ndarray::{Array3, ArrayView3, Axis as NdAxis};

use crate::assemble::{Assembly, Phase, PlanBuilder};
use crate::decomposition::{price_phase, CostModel};
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, BlockOp, Chain};
use crate::reach::{AxisReach, Reach};
use crate::voxels::Voxels;
use crate::Dtype;

// ----------------------------------------------------------- the parameters --

/// What a foreground voxel's distance is when the volume has **no** background.
///
/// A parameter because the two references disagree, and the disagreement is the
/// whole of their difference: on any input with at least one background voxel
/// they are the same function bit for bit. See the module header for the
/// measurement behind each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unbounded {
    /// SciPy's: the distance to a phantom background voxel at index
    /// `(-1, 0, 0)`, which is what its uninitialised feature index evaluates to.
    PhantomOrigin,
    /// `+inf`, which is what a lower envelope over no finite parabola is.
    Infinite,
}

/// The two knobs of a distance transform.
///
/// [`Default`] is SciPy's default point — unit pitch on every axis and
/// [`Unbounded::PhantomOrigin`] — because that is the behaviour a caller who
/// passes nothing to `distance_transform_edt` gets, and a default that silently
/// differed from the reference would be the worst kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceParams {
    sampling: [f64; 3],
    unbounded: Unbounded,
}

impl Default for DistanceParams {
    fn default() -> Self {
        Self {
            sampling: [1.0, 1.0, 1.0],
            unbounded: Unbounded::PhantomOrigin,
        }
    }
}

impl DistanceParams {
    /// The per-axis voxel pitch the distance is measured in.
    pub fn sampling(&self) -> [f64; 3] {
        self.sampling
    }

    /// [`Self::sampling`], set.
    pub fn with_sampling(mut self, sampling: [f64; 3]) -> Self {
        self.sampling = sampling;
        self
    }

    /// What an all-foreground volume's voxels are.
    pub fn unbounded(&self) -> Unbounded {
        self.unbounded
    }

    /// [`Self::unbounded`], set.
    pub fn with_unbounded(mut self, unbounded: Unbounded) -> Self {
        self.unbounded = unbounded;
        self
    }

    /// The squared pitch per axis, which is what the envelope actually uses.
    ///
    /// Refuses a pitch that is not finite and positive: a zero pitch divides by
    /// zero inside the parabola intersection and a negative one is not a length.
    /// Public because it is the validation, and a caller assembling the ops by
    /// hand should be able to run it before it is too late to say so.
    pub fn squared_sampling(&self) -> Result<[f64; 3]> {
        let mut squared = [0.0f64; 3];
        for axis in 0..3 {
            let pitch = self.sampling[axis];
            if !pitch.is_finite() || pitch <= 0.0 {
                return Err(Error::InvalidArgument(format!(
                    "distance: the voxel pitch on axis {axis} is {pitch}, and a distance \
                     transform needs a finite positive length on every axis — a zero pitch \
                     divides by zero in the lower envelope and a negative one is not a length."
                )));
            }
            squared[axis] = pitch * pitch;
        }
        Ok(squared)
    }
}

// --------------------------------------------------------------- the kernel --

/// The 1-D squared distance transform: the lower envelope of the parabolas
/// `f[p] + scale_squared * (q - p)^2`.
///
/// Felzenszwalb–Huttenlocher, `O(n)`. `v` holds the indices of the parabolas on
/// the envelope and `z` the `n + 1` boundaries between them; both are handed in
/// so that a sweep over a volume allocates once rather than once per lane.
///
/// **Infinite seeds are skipped rather than pushed.** A parabola rooted at an
/// infinite `f` can never be the minimum unless every parabola is, and pushing
/// one puts `inf` into the intersection formula, where `inf - inf` is `NaN` and a
/// `NaN` boundary silently stops the envelope scan. The all-infinite lane is
/// therefore its own branch and is the only place `+inf` is written.
fn lower_envelope(f: &[f64], scale_squared: f64, v: &mut [usize], z: &mut [f64], out: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }

    // `k` is the index of the top of the envelope stack, and `-1` means the
    // stack is empty because no finite seed has been seen yet.
    let mut k: isize = -1;
    for (q, &value) in f.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if k < 0 {
            k = 0;
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            continue;
        }
        let mut boundary = intersection(f, scale_squared, q, v[k as usize]);
        // Terminates at `k == 0`: `z[0]` is `-inf` and `boundary` is finite,
        // because both seeds are finite and `q > v[k]`.
        while boundary <= z[k as usize] {
            k -= 1;
            boundary = intersection(f, scale_squared, q, v[k as usize]);
        }
        k += 1;
        v[k as usize] = q;
        z[k as usize] = boundary;
        z[k as usize + 1] = f64::INFINITY;
    }

    if k < 0 {
        // No background anywhere on this lane. The only `+inf` this function
        // writes, and the only value the finish ever has to interpret.
        for value in out.iter_mut() {
            *value = f64::INFINITY;
        }
        return;
    }

    let mut k = 0usize;
    for (q, value) in out.iter_mut().enumerate() {
        let position = q as f64;
        while z[k + 1] < position {
            k += 1;
        }
        let offset = position - v[k] as f64;
        *value = f[v[k]] + scale_squared * offset * offset;
    }
}

/// Where the parabolas rooted at `q` and `p` cross. `q > p` and both `f` values
/// are finite, so the denominator is positive and the result is finite.
///
/// Spelled as `(f + s*x^2)` differences rather than as a rearrangement, because
/// at `scale_squared == 1.0` that is operation for operation the form the
/// reference implementations use, and the two are meant to agree in the last bit
/// rather than to within a tolerance.
fn intersection(f: &[f64], scale_squared: f64, q: usize, p: usize) -> f64 {
    let qf = q as f64;
    let pf = p as f64;
    ((f[q] + scale_squared * qf * qf) - (f[p] + scale_squared * pf * pf))
        / (2.0 * scale_squared * (qf - pf))
}

/// One pass of [`lower_envelope`] over every lane of `data` parallel to `axis`.
///
/// Public because it is the whole of what one sweep does, and a caller holding a
/// resident volume who wants the passes in another order — they commute — should
/// not have to go through a `BlockOp` to get them. `scale_squared` is the
/// *square* of the pitch on `axis`.
pub fn sweep_axis(data: &mut Array3<f64>, axis: usize, scale_squared: f64) {
    let n = data.shape()[axis];
    if n == 0 {
        return;
    }
    let mut line = vec![0.0f64; n];
    let mut transformed = vec![0.0f64; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    for mut lane in data.lanes_mut(NdAxis(axis)) {
        for (slot, value) in line.iter_mut().zip(lane.iter()) {
            *slot = *value;
        }
        lower_envelope(&line, scale_squared, &mut v, &mut z, &mut transformed);
        for (slot, value) in lane.iter_mut().zip(transformed.iter()) {
            *slot = *value;
        }
    }
}

/// The seeded field: `0` where the mask is background, `+inf` where it is
/// foreground. Squared distances, so `0` is the squared distance of a background
/// voxel from itself.
pub fn seed(mask: ArrayView3<'_, bool>) -> Array3<f64> {
    mask.mapv(|foreground| if foreground { f64::INFINITY } else { 0.0 })
}

/// The **squared** distance field, resident, over a whole volume.
///
/// The three passes in axis order. This is the surface where comparison is
/// exact: at unit sampling every value here is an exact integer, so a threshold
/// against an integer needs no square root and a comparison against a stored
/// field is a comparison of integers. See the module header.
pub fn squared_distance_transform(
    mask: ArrayView3<'_, bool>,
    params: &DistanceParams,
) -> Result<Array3<f64>> {
    let scales = params.squared_sampling()?;
    let mut field = seed(mask);
    for axis in 0..3 {
        sweep_axis(&mut field, axis, scales[axis]);
    }
    Ok(field)
}

/// The distance field, resident, over a whole volume.
///
/// See the module header for what "resident" costs: on a volume that fits it is
/// the sensible way to compute the field, and on one that does not, [`plan`] is.
pub fn distance_transform(
    mask: ArrayView3<'_, bool>,
    params: &DistanceParams,
) -> Result<Array3<f64>> {
    let mut field = squared_distance_transform(mask, params)?;
    finish_in_place(&mut field, [0, 0, 0], params);
    Ok(field)
}

/// The pointwise last step: a square root, and the all-foreground rule where the
/// squared value is infinite.
///
/// `offset` is where this buffer's lower corner sits in the volume, which the
/// phantom-origin rule needs and the square root does not.
fn finish_in_place(field: &mut Array3<f64>, offset: [usize; 3], params: &DistanceParams) {
    let sampling = params.sampling();
    let unbounded = params.unbounded();
    for ((i, j, k), value) in field.indexed_iter_mut() {
        if value.is_finite() {
            *value = value.sqrt();
            continue;
        }
        *value = match unbounded {
            Unbounded::Infinite => f64::INFINITY,
            Unbounded::PhantomOrigin => {
                // The phantom background voxel sits at `(-1, 0, 0)`, so the
                // offset on axis 0 is one past the volume's own start and the
                // other two are the absolute index itself.
                let da = sampling[0] * ((offset[0] + i) as f64 + 1.0);
                let db = sampling[1] * (offset[1] + j) as f64;
                let dc = sampling[2] * (offset[2] + k) as f64;
                (da * da + db * db + dc * dc).sqrt()
            }
        };
    }
}

// ------------------------------------------------------------------ the ops --

/// The declared cost of one [`DistanceSweepOp`] that reads an `f64` field, in
/// units of one voxelwise map.
///
/// **Measured, not derived**, by [`cost_report`] — see `super::COST_MEASUREMENT`
/// for the method and for why the number is a ratio. Over a `96 x 64 x 64`
/// volume on the machine this was written on, best of 20, with the voxelwise map
/// priced in the same run as the denominator: `27.2, 28.6` on axis 0, `21.9,
/// 23.1` on axis 1, `23.7, 28.9` on axis 2. The axis barely matters — a lane
/// copy in and out makes every sweep sequential in the array whatever it is
/// sweeping — so one constant covers all three, and 25 is the middle of the
/// spread rather than the best of it.
///
/// A third repetition on a loaded machine put the same rows at `39.6, 35.4,
/// 36.4`, which is the usual one-sided contamination and is why the stored
/// number comes from the quiet runs.
pub const DISTANCE_SWEEP_COST: f64 = 25.0;

/// [`DISTANCE_SWEEP_COST`] for the one pass that reads the `bool` mask.
///
/// **About half, and the reason is the input width rather than the arithmetic**:
/// the envelope does the same work per lane either way, and the seeding pass
/// reads one byte a voxel where the others read eight. Measured at `14.5, 14.0`
/// against the field pass' `27.2, 28.6` on the same axis in the same runs.
pub const DISTANCE_SEED_SWEEP_COST: f64 = 14.0;

/// The declared cost of one [`DistanceFinishOp`]: a square root, a branch and a
/// copy. Measured at `7.3, 8.2`.
pub const DISTANCE_FINISH_COST: f64 = 7.5;

/// One separable pass: the lower envelope along `axis`, over a buffer that spans
/// that axis whole.
///
/// [`AxisReach::All`] on `axis` and nothing on the other two. That asymmetry is
/// the entire reason this operation is decomposable at all, and it is what this
/// crate's reach algebra can state and a halo cannot: the swept axis is
/// unbounded and the other two are free.
///
/// Stated in the phase's own frame and writing the shape it read — a whole-axis
/// stencil, not a projection. See the module header for why that distinction is
/// worth a paragraph.
#[derive(Debug, Clone, Copy)]
pub struct DistanceSweepOp {
    axis: usize,
    scale_squared: f64,
    /// Whether this pass reads the mask and seeds the field, or reads the
    /// previous pass' `f64`. Only the first pass seeds.
    seeds: bool,
    name: &'static str,
    cost: f64,
}

impl DistanceSweepOp {
    /// One pass, as a `BlockOp` a caller can hold on its own.
    ///
    /// `pitch` is the voxel edge on `axis`, not its square; the square is this
    /// type's business. `seeds` says whether this pass reads the `bool` mask —
    /// exactly one pass of a chain does.
    pub fn along(axis: usize, pitch: f64, seeds: bool) -> Result<Self> {
        if axis >= 3 {
            return Err(Error::InvalidArgument(format!(
                "distance: there is no axis {axis} to sweep"
            )));
        }
        if !pitch.is_finite() || pitch <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "distance: a voxel pitch of {pitch} is not a length"
            )));
        }
        Ok(Self::squared(axis, pitch * pitch, seeds))
    }

    /// [`Self::along`] from the already-squared pitch, which is what
    /// [`DistanceParams::squared_sampling`] hands back.
    fn squared(axis: usize, scale_squared: f64, seeds: bool) -> Self {
        let name = match (axis, seeds) {
            (0, true) => "distance-sweep-0-seed",
            (0, false) => "distance-sweep-0",
            (1, true) => "distance-sweep-1-seed",
            (1, false) => "distance-sweep-1",
            (_, true) => "distance-sweep-2-seed",
            _ => "distance-sweep-2",
        };
        Self {
            axis,
            scale_squared,
            seeds,
            name,
            cost: if seeds {
                DISTANCE_SEED_SWEEP_COST
            } else {
                DISTANCE_SWEEP_COST
            },
        }
    }

    /// The axis this pass sweeps.
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Whether this pass reads the mask rather than the previous pass' field.
    pub fn seeds(&self) -> bool {
        self.seeds
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for DistanceSweepOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The whole axis, stated as an integer because the trait wants a symmetric
    /// bound, and as [`AxisReach::All`] in [`Self::reach_spec`], which is the
    /// variant that *means* it.
    ///
    /// Returning `0` here for the swept axis would be caught — `Chain::reach_spec`
    /// refuses a spec wider than the bound — but returning the extent is the
    /// honest answer and not merely the one that passes.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        if axis == self.axis {
            volume_len
        } else {
            0
        }
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        let mut axes = [AxisReach::none(), AxisReach::none(), AxisReach::none()];
        axes[self.axis] = AxisReach::All;
        Reach::per_axis(axes)
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        if self.seeds {
            dtype == Dtype::Bool
        } else {
            dtype == Dtype::F64
        }
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    /// Measured; see [`DISTANCE_SWEEP_COST`] and [`DISTANCE_SEED_SWEEP_COST`].
    /// Advisory anyway: it prices a phase that is a planning barrier.
    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let shape = input.shape();
        // **Not redundant with the planner's guard, and the module header says
        // why.** A lattice that cuts the swept axis without the whole-axis halo
        // is refused when the `Decomposition` is built; this method is public
        // and is reachable without building one. A buffer that does not span the
        // swept axis whole would produce a *block-local* distance transform — a
        // complete, well-formed, wrong volume.
        if at.offset[self.axis] != 0 || shape[self.axis] != at.volume[self.axis] {
            return Err(Error::InvalidArgument(format!(
                "{}: this pass sweeps axis {} and its reach on that axis is the whole of it, so \
                 the buffer must start at 0 and span all {} voxels of it. It starts at {} and \
                 spans {}. A lattice that cuts the swept axis without granting the full halo \
                 would compute a block-local distance transform, which is a well-formed volume \
                 and the wrong one.",
                self.name, self.axis, at.volume[self.axis], at.offset[self.axis], shape[self.axis],
            )));
        }

        let mut field = if self.seeds {
            seed(input.view::<bool>()?)
        } else {
            input.view::<f64>()?.to_owned()
        };
        sweep_axis(&mut field, self.axis, self.scale_squared);
        super::shapes_agree(&shape, &out.shape(), self.name)?;
        out.view_mut::<f64>()?.assign(&field);
        Ok(())
    }
}

/// The pointwise finish: a square root, and the all-foreground rule.
///
/// Reach zero on every axis. It needs the [`Anchor`] rather than only the
/// buffer, because [`Unbounded::PhantomOrigin`] is a distance to a fixed point
/// of the **volume** and a block that re-anchored to itself would produce the
/// distance to a phantom voxel per block.
#[derive(Debug, Clone, Copy)]
pub struct DistanceFinishOp {
    params: DistanceParams,
    cost: f64,
}

impl DistanceFinishOp {
    pub fn new(params: DistanceParams) -> Self {
        Self {
            params,
            cost: DISTANCE_FINISH_COST,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for DistanceFinishOp {
    fn name(&self) -> &'static str {
        "distance-finish"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    /// `+0.0` is exactly true and is the only constant that is: a background
    /// voxel's squared distance is `+0.0` and its square root is `+0.0`, in
    /// every rounding mode.
    ///
    /// **Not `-0.0`, and the bits are compared rather than the values.**
    /// `sqrt(-0.0)` is `-0.0`, not `+0.0`, so a declaration written as
    /// `value == 0.0` would claim a constant this op does not produce — and
    /// `-0.0 == 0.0` is true, which is exactly how that mistake stays invisible.
    /// A squared distance is never a negative zero here, so the branch costs
    /// nothing and says only what is true.
    ///
    /// No other value survives: the phantom rule makes an infinite input
    /// position-dependent, and every finite non-zero one changes under the root.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value.to_bits() == 0.0f64.to_bits()).then_some(0.0)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let mut field = input.view::<f64>()?.to_owned();
        finish_in_place(&mut field, at.offset, &self.params);
        super::shapes_agree(&input.shape(), &out.shape(), "distance-finish")?;
        out.view_mut::<f64>()?.assign(&field);
        Ok(())
    }
}

// ------------------------------------------------------------------- a plan --

/// The lattice one sweep runs on: the swept axis whole, the other two cut at
/// `block`.
///
/// Stated as a function because getting it wrong is not an error the framework
/// raises — a grid that cuts the swept axis and grants the whole-axis halo still
/// produces the right answer, merely by re-reading every lane in every block —
/// so the one place the choice is made should be the one place a reader looks.
pub fn sweep_grid(volume: [usize; 3], axis: usize, block: usize) -> Result<BlockGrid> {
    if axis >= 3 {
        return Err(Error::InvalidArgument(format!(
            "distance: there is no axis {axis} to sweep"
        )));
    }
    if block == 0 {
        return Err(Error::InvalidArgument(
            "distance: a block edge of 0 cuts the volume into nothing".to_string(),
        ));
    }
    let free: Vec<usize> = (0..3).filter(|&other| other != axis).collect();
    BlockGrid::along(volume, &free, block)
}

/// The four phases — three sweeps on three lattices, then the pointwise finish —
/// **appended to a plan that already has some**.
///
/// The plan's current grid is replaced: the first sweep needs axis 0 whole, and
/// a caller who had a grid before this call gets it back by calling
/// [`PlanBuilder::regrid`] afterwards. Returns the finish's phase, which is the
/// one that writes the field.
///
/// The plan must be reading a `bool` mask when this is called; the first sweep
/// refuses anything else.
pub fn append_to(plan: &mut PlanBuilder, params: &DistanceParams, block: usize) -> Result<Phase> {
    let scales = params.squared_sampling()?;
    let volume = plan.grid().volume();
    plan.regrid(sweep_grid(volume, 0, block)?);
    plan.pixels(Chain::op(DistanceSweepOp::squared(0, scales[0], true)))?;
    for axis in 1..3 {
        plan.regrid(sweep_grid(volume, axis, block)?);
        plan.pixels(Chain::op(DistanceSweepOp::squared(
            axis,
            scales[axis],
            false,
        )))?;
    }
    plan.regrid(BlockGrid::new(volume, [block; 3])?);
    plan.pixels(Chain::op(DistanceFinishOp::new(*params)))
}

/// The whole transform as its own plan: a `bool` mask in, an `f64` field out.
pub fn plan(params: &DistanceParams, volume: [usize; 3], block: usize) -> Result<Assembly> {
    let mut builder = PlanBuilder::new(volume, Dtype::Bool, sweep_grid(volume, 0, block)?);
    append_to(&mut builder, params, block)?;
    builder.finish()
}

/// What each phase of [`plan`] holds while one block is in flight, in bytes.
///
/// **Not this module's arithmetic.** It is [`price_phase`] pointed at the plan's
/// own grid and reach, which is the same function the planner prices candidates
/// with — so the table in the module header is the planner's figure for this
/// plan rather than a multiplication written out beside it.
///
/// The width is `f64` for every phase, which is exact for the last three and an
/// over-statement for the first: its *input* is a `bool` mask, so it really
/// holds nine bytes a voxel and not sixteen. Over-stating a budget is the safe
/// direction and stating one width is what keeps this a lookup rather than a
/// second cost model.
pub fn working_set_bytes(
    params: &DistanceParams,
    volume: [usize; 3],
    block: usize,
) -> Result<Vec<f64>> {
    let assembly = plan(params, volume, block)?;
    Ok(assembly
        .decomposition
        .phases
        .iter()
        .map(|phase| {
            price_phase(
                &phase.grid,
                &phase.reach,
                0.0,
                1,
                true,
                8.0,
                &CostModel::default(),
                1.0,
            )
            .working_set_bytes_per_block
        })
        .collect())
}

// ------------------------------------------------------------- the controls --

/// The distance from every foreground voxel to the nearest background voxel, by
/// looking at every background voxel.
///
/// `O(foreground * background)` and useless on anything but a fixture, which is
/// exactly what it is for: an exact transform and an approximate one agree on
/// smooth shapes, so the fast route is only trustworthy against the definition.
/// This is the same discipline `ops::fft` landed with and for the same reason.
pub fn brute_force_distance(
    mask: ArrayView3<'_, bool>,
    params: &DistanceParams,
) -> Result<Array3<f64>> {
    let sampling = params.sampling();
    params.squared_sampling()?;
    let background: Vec<[usize; 3]> = mask
        .indexed_iter()
        .filter(|(_, &foreground)| !foreground)
        .map(|((i, j, k), _)| [i, j, k])
        .collect();

    let mut out = Array3::<f64>::zeros(mask.raw_dim());
    for ((i, j, k), value) in out.indexed_iter_mut() {
        if !mask[[i, j, k]] {
            *value = 0.0;
            continue;
        }
        if background.is_empty() {
            *value = match params.unbounded() {
                Unbounded::Infinite => f64::INFINITY,
                Unbounded::PhantomOrigin => {
                    let da = sampling[0] * (i as f64 + 1.0);
                    let db = sampling[1] * j as f64;
                    let dc = sampling[2] * k as f64;
                    (da * da + db * db + dc * dc).sqrt()
                }
            };
            continue;
        }
        let mut best = f64::INFINITY;
        for corner in &background {
            let da = sampling[0] * (i as f64 - corner[0] as f64);
            let db = sampling[1] * (j as f64 - corner[1] as f64);
            let dc = sampling[2] * (k as f64 - corner[2] as f64);
            let squared = da * da + db * db + dc * dc;
            // `total_cmp`, never `min`: a `NaN` here would be a bug and
            // `f64::min` would swallow it.
            if squared.total_cmp(&best).is_lt() {
                best = squared;
            }
        }
        *value = best.sqrt();
    }
    Ok(out)
}

/// The **chamfer** approximation: the two-pass forward/backward propagation of a
/// fixed 26-neighbour offset set, with each offset weighted by its own Euclidean
/// length.
///
/// **A control, and never reachable from [`plan`] or [`distance_transform`].**
/// "Approximately correct" distance transforms are the trap this module exists
/// to avoid: a chamfer is exact on axis-aligned and diagonal directions and
/// wrong in between, which is invisible on a sphere and visible on a thin
/// oblique sheet. It ships so that the exactness claim beside it is a
/// measurement rather than an adjective — and so that a caller who genuinely
/// wants the cheap approximation gets the one that has been measured against the
/// exact answer rather than writing a fourth one.
pub fn chamfer_distance(mask: ArrayView3<'_, bool>) -> Array3<f64> {
    let (na, nb, nc) = mask.dim();
    let mut field = mask.mapv(|foreground| if foreground { f64::INFINITY } else { 0.0 });
    let offsets: Vec<[isize; 3]> = (-1..=1isize)
        .flat_map(|a| (-1..=1isize).flat_map(move |b| (-1..=1isize).map(move |c| [a, b, c])))
        .filter(|offset| offset != &[0, 0, 0])
        .collect();
    let weight = |offset: &[isize; 3]| -> f64 {
        ((offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]) as f64).sqrt()
    };
    let forward: Vec<&[isize; 3]> = offsets
        .iter()
        .filter(|offset| {
            offset[0] < 0
                || (offset[0] == 0 && offset[1] < 0)
                || (offset[0] == 0 && offset[1] == 0 && offset[2] < 0)
        })
        .collect();
    let backward: Vec<&[isize; 3]> = offsets
        .iter()
        .filter(|offset| !forward.contains(offset))
        .collect();

    let relax = |field: &mut Array3<f64>, order: &[[usize; 3]], taps: &[&[isize; 3]]| {
        for &[i, j, k] in order {
            let mut best = field[[i, j, k]];
            for tap in taps {
                let a = i as isize + tap[0];
                let b = j as isize + tap[1];
                let c = k as isize + tap[2];
                if a < 0 || b < 0 || c < 0 {
                    continue;
                }
                let (a, b, c) = (a as usize, b as usize, c as usize);
                if a >= na || b >= nb || c >= nc {
                    continue;
                }
                let candidate = field[[a, b, c]] + weight(tap);
                if candidate.total_cmp(&best).is_lt() {
                    best = candidate;
                }
            }
            field[[i, j, k]] = best;
        }
    };

    let ascending: Vec<[usize; 3]> = (0..na)
        .flat_map(|i| (0..nb).flat_map(move |j| (0..nc).map(move |k| [i, j, k])))
        .collect();
    let descending: Vec<[usize; 3]> = ascending.iter().rev().copied().collect();
    relax(&mut field, &ascending, &forward);
    relax(&mut field, &descending, &backward);
    field
}

/// Retake [`DISTANCE_SWEEP_COST`], [`DISTANCE_SEED_SWEEP_COST`] and
/// [`DISTANCE_FINISH_COST`] on this machine.
/// See `super::COST_MEASUREMENT`.
///
/// **Measures its own denominator.** The stored table is in units of one
/// voxelwise map, and that unit is a different number of nanoseconds on every
/// machine — so this prices a threshold map in the same run, with the same
/// fixture size and the same best-of rule, and divides. What comes out is a
/// ratio, which is what the planner uses and what survives a change of machine.
pub fn cost_report() -> String {
    use std::time::Instant;

    let shape = [96usize, 64, 64];
    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let mask = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (3 * i as i64 - 5 * j as i64 + 2 * k as i64 - 20).abs() > 2
    });
    let source = Voxels::Bool(mask.clone());
    let seeded = Voxels::F64(
        squared_distance_transform(mask.view(), &DistanceParams::default())
            .expect("the transform must run"),
    );

    let mut unit = f64::INFINITY;
    for _ in 0..40 {
        let values = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            (((i * 7 + j * 13 + k * 29) % 251) as f64) / 251.0
        });
        let mut out = Array3::<f64>::zeros(values.raw_dim());
        let started = Instant::now();
        ndarray::Zip::from(&mut out)
            .and(&values)
            .for_each(|slot, &value| *slot = if value > 0.5 { 1.0 } else { 0.0 });
        let taken = started.elapsed().as_secs_f64() / voxels;
        if taken < unit {
            unit = taken;
        }
        std::hint::black_box(&out);
    }

    let mut report = format!(
        "one voxelwise map over {shape:?} = {:.3} ns/voxel, which is the unit below\n",
        unit * 1e9
    );
    let anchor = Anchor::whole(shape);
    for (what, op, input) in [
        (
            "sweep on axis 0, seeding from a mask",
            DistanceSweepOp::along(0, 1.0, true).unwrap(),
            &source,
        ),
        (
            "sweep on axis 0, from a field",
            DistanceSweepOp::along(0, 1.0, false).unwrap(),
            &seeded,
        ),
        (
            "sweep on axis 1, from a field",
            DistanceSweepOp::along(1, 1.0, false).unwrap(),
            &seeded,
        ),
        (
            "sweep on axis 2, from a field",
            DistanceSweepOp::along(2, 1.0, false).unwrap(),
            &seeded,
        ),
    ] {
        let mut best = f64::INFINITY;
        for _ in 0..20 {
            let mut out = Voxels::F64(Array3::zeros((shape[0], shape[1], shape[2])));
            let started = Instant::now();
            op.apply(input, &mut out, &anchor)
                .expect("the sweep must run");
            let taken = started.elapsed().as_secs_f64() / voxels;
            if taken < best {
                best = taken;
            }
            std::hint::black_box(&out);
        }
        report.push_str(&format!(
            "{what}: {:.3} ns/voxel = {:.1} x the map\n",
            best * 1e9,
            best / unit
        ));
    }

    let finish = DistanceFinishOp::new(DistanceParams::default());
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let mut out = Voxels::F64(Array3::zeros((shape[0], shape[1], shape[2])));
        let started = Instant::now();
        finish
            .apply(&seeded, &mut out, &anchor)
            .expect("the finish must run");
        let taken = started.elapsed().as_secs_f64() / voxels;
        if taken < best {
            best = taken;
        }
        std::hint::black_box(&out);
    }
    report.push_str(&format!(
        "finish: {:.3} ns/voxel = {:.1} x the map\n",
        best * 1e9,
        best / unit
    ));
    report
}

// -------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;

    /// Retaking the measurement. Ignored because timing in a test suite measures
    /// the machine's mood, not the code — but it is here, it runs, and it is one
    /// command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::distance
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report());
    }

    /// What can be asserted about a measured cost without measuring: the order
    /// the constants encode is the order the ops actually have. A sweep is a
    /// lower envelope and two copies per voxel; the finish is a square root and
    /// one, and both are dearer than the voxelwise map they are denominated in.
    #[test]
    fn the_declared_costs_are_in_the_order_the_ops_are() {
        assert!(DISTANCE_SWEEP_COST > DISTANCE_SEED_SWEEP_COST);
        assert!(DISTANCE_SEED_SWEEP_COST > DISTANCE_FINISH_COST);
        assert!(DISTANCE_FINISH_COST > 1.0);
        let sweep = DistanceSweepOp::along(0, 1.0, true).unwrap();
        assert_eq!(sweep.cost_per_voxel(), DISTANCE_SEED_SWEEP_COST);
        assert_eq!(
            DistanceSweepOp::along(0, 1.0, false)
                .unwrap()
                .cost_per_voxel(),
            DISTANCE_SWEEP_COST
        );
        assert_eq!(
            DistanceFinishOp::new(DistanceParams::default()).cost_per_voxel(),
            DISTANCE_FINISH_COST
        );
        // And a caller may say otherwise, because a cost is advisory and the
        // machine that will do the work is not this one.
        assert_eq!(sweep.with_cost(9.0).cost_per_voxel(), 9.0);
    }

    /// The dtypes each pass takes, which is what stops a chain being assembled
    /// with two seeding passes or none.
    #[test]
    fn only_the_seeding_pass_reads_a_mask() {
        let seeding = DistanceSweepOp::along(0, 1.0, true).unwrap();
        assert!(seeding.accepts(Dtype::Bool));
        assert!(!seeding.accepts(Dtype::F64));
        let later = DistanceSweepOp::along(1, 1.0, false).unwrap();
        assert!(!later.accepts(Dtype::Bool));
        assert!(later.accepts(Dtype::F64));
        assert_eq!(seeding.produces(Dtype::Bool), Dtype::F64);
        assert_eq!(later.produces(Dtype::F64), Dtype::F64);
        assert_eq!(seeding.axis(), 0);
        assert!(seeding.seeds());
        assert!(!later.seeds());
    }

    /// The reach, both ways it is stated, and that the two agree.
    #[test]
    fn the_swept_axis_is_the_only_one_declared() {
        for axis in 0..3usize {
            let op = DistanceSweepOp::along(axis, 1.0, false).unwrap();
            let volume = [7usize, 11, 13];
            let spec = op.reach_spec(volume);
            assert!(spec.is_whole_axis(axis, volume[axis]));
            assert_eq!(op.reach(axis, volume[axis]), volume[axis]);
            for other in (0..3).filter(|&other| other != axis) {
                assert!(!spec.is_whole_axis(other, volume[other]));
                assert_eq!(op.reach(other, volume[other]), 0);
            }
            assert!(spec.is_barrier(volume));
        }
    }
}
