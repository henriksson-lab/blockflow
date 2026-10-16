// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Space-filling synthetic tissue with exact dense labels, and a label → image
// formation model, for developing and scoring dense segmentation of volume EM.
//
// Why this exists
// ---------------
// `synthetic::SceneSpec` places isolated ellipsoids on a background. Dense EM
// segmentation needs the opposite: **every voxel belongs to a cell**, cells are
// long thin processes rather than blobs, neighbours touch across thin dark
// membranes, and the hard cases are thin necks, parallel fibres and dark
// organelles that look like membranes. This module generates that geometry with
// its ground truth, and renders it through a parameterised image formation model.
// The two halves are independent: [`render_tissue`] accepts *any* dense `u32`
// label volume (a real annotated crop as well as a generated one).
//
// Part A — geometry ([`generate_tissue`])
// --------------------------------------
// Objects are generated in a fixed order and each object's id is its place in
// that order:
//
// 1. **Fibre bundles.** Per bundle a direction, a cross-section disk packed by
//    Poisson-disk dart throwing (radius + radius + gap), and a coherent slow
//    wiggle shared by the bundle plus a small per-fibre wiggle.
// 2. **Somata.** Noise-perturbed ellipsoids (rejected while they collide with
//    what is already placed), each the root of a few neurites with its label.
// 3. **Free neurites.** Branching persistent random walks entering through a
//    face or starting inside, both ways.
//
// Neurite walks are worm-like chains: per step of length `h` the direction gets
// a Gaussian kick of size `sqrt(h / persistence)`, clamped to `max_turn * h`
// (bounded curvature). The radius varies smoothly along the arc (1-D value
// noise) and is multiplied by smooth bumps for necks (< 1) and varicosities
// (> 1). A step whose sphere would overlap a sphere of another object by more
// than `collision_tolerance` is deflected (eight tries with growing kicks) and
// the branch ends if every try collides — so processes mostly avoid each other
// instead of being carved, as in packing generators such as ConFiG/MEDUSA.
//
// **Priority.** Everything is rasterised as primitives (spheres along
// polylines, blobs for somata) and a voxel covered by several takes the
// *lowest* object id — the earliest-drawn object wins and later objects are
// carved. Because the rule is a minimum, rasterisation is order-free and runs
// plane-parallel with the same bits.
//
// **Fill.** Optionally objects first grow by `object_growth` into empty space
// (seeded watershed on the distance to them). The rest is partitioned by a
// seeded watershed of `-distance + texture·noise` restricted to the empty
// space, from Poisson-disk "glia" seeds plus one seed in every empty component
// that got none — so the result is dense and the fill cells are irregular.
//
// **Extracellular space** (optional): for a hashed `fraction` of touching label
// pairs, a gap of `width` voxels along their interface is set to label 0, taken
// from the larger cell of the pair so that thin processes are not cut.
//
// **Clean ground truth.** Finally every label is split into its 6-connected
// components (carving, the ECS and the clipping at the volume edge can all cut
// an object), fragments smaller than `min_fragment_voxels` are merged into the
// neighbour they share most faces with, and labels are renumbered `1..=n` in
// raster order of their first voxel. Skeleton nodes are kept only where they lie
// in a label descended from their own object, and split along with it.
//
// Part B — image formation ([`render_tissue`], [`form_image`])
// ------------------------------------------------------------
// In this order, all whole-volume (the membrane distance, the PSF and the
// scan-line AR(1) all need context, so region-wise rendering is not offered):
//
// 1. clean image: per-label cytoplasm level × (1 + smooth texture), myelin
//    wraps, organelles clipped to their cell and kept off its membrane,
//    extracellular level, then membranes mixed in by partial-volume coverage
//    `clamp(thickness/2 - d, 0, 1)` where `d` is the distance to a boundary
//    voxel of `ops::boundaries` (so `thickness` 2 is exactly the two boundary
//    voxels, 1 is half of each);
// 2. polarity (dark membranes, or the whole configured range inverted);
// 3. anisotropic Gaussian PSF (`ops::smooth::Gaussian`, truncated at 3σ);
// 4. per-slice xy jitter (integer or bilinear subpixel, edge clamped);
// 5. per-slice gain/offset drift (random walks along z);
// 6. curtaining: additive offsets per x column, constant over y and over runs
//    of slices;
// 7. Poisson shot noise in photons, Gaussian read noise;
// 8. scan-line correlated noise, AR(1) along x;
// 9. `u16` quantisation `round(offset + scale·v)` clamped to a range.
//
// Randomness is `synthetic`'s SplitMix64, keyed by object index or by voxel
// position, so the output is a function of the spec alone on any thread count.
// Unlike `synthetic`, `sqrt` is not the only non-trivial function: the Poisson
// sampler uses `exp`, so bit-identity across libm implementations is not
// claimed for the noise stage. Lengths are in the units of `spacing` (so with
// `spacing = [2, 1, 1]` they are xy voxels); node coordinates are in voxels with
// `synthetic`'s convention that voxel `i` covers `[i, i + 1)`.
//
// Nothing here has a dataset default: every spec is built explicitly, and the
// tests carry a documented example configuration.

use std::collections::{BTreeMap, HashMap, VecDeque};

use ndarray::{Array3, ArrayView3, Zip};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::ops::components::{label_members_into, Union};
use crate::ops::distance::{distance_transform, DistanceParams, Unbounded};
use crate::ops::smooth::Gaussian;
use crate::ops::watershed::{seeded_watershed, Separation};
use crate::synthetic::{hash3, unit_from, Stream};

// ================================================================ specs ==

/// The shape of a branching process (soma neurites and free neurites).
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessShape {
    /// Base radius range, drawn per branch root.
    pub radius: [f64; 2],
    /// Radius never goes below this, whatever necks do.
    pub min_radius: f64,
    /// Relative amplitude of the smooth radius variation along the arc (`0..1`).
    pub radius_variation: f64,
    /// Arc length over which that variation decorrelates.
    pub radius_wavelength: f64,
    /// Persistence length of the walk: larger is straighter.
    pub persistence: f64,
    /// Curvature bound: the direction changes by at most this per unit length.
    pub max_turn: f64,
    /// Length range of each branch.
    pub length: [f64; 2],
    /// Expected branch points per unit length.
    pub branch_rate: f64,
    /// Maximum number of extra branches per process (root branches excluded).
    pub max_branches: usize,
    /// Child radius as a fraction of the parent radius at the branch point.
    pub branch_radius_factor: f64,
    /// Size of the sideways kick a child direction gets (`~1` ≈ 45°).
    pub branch_spread: [f64; 2],
    /// Necks: expected per unit length, radius factor at their centre (< 1), length.
    pub neck_rate: f64,
    pub neck_factor: f64,
    pub neck_length: f64,
    /// Varicosities: expected per unit length, radius factor (> 1), length.
    pub varicosity_rate: f64,
    pub varicosity_factor: f64,
    pub varicosity_length: f64,
}

/// Free neurites.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuriteSpec {
    pub count: usize,
    /// Fraction that enter through a face of the volume; the rest start inside
    /// and grow both ways.
    pub enter_from_face: f64,
    pub shape: ProcessShape,
}

/// Cell bodies with neurites.
#[derive(Debug, Clone, PartialEq)]
pub struct SomaSpec {
    pub count: usize,
    /// Mean semi-axis range.
    pub radius: [f64; 2],
    /// Each semi-axis is the mean times a factor in `1/e ..= e` (`e >= 1`).
    pub elongation: f64,
    /// Surface perturbation: relative radius change, `0..1`.
    pub irregularity: f64,
    /// Correlation length of that perturbation, as a fraction of the mean radius.
    pub irregularity_scale: f64,
    /// Neurites per soma, inclusive range.
    pub neurites: [usize; 2],
    pub shape: ProcessShape,
}

/// Axon fascicles: near-parallel packed tubes.
#[derive(Debug, Clone, PartialEq)]
pub struct FibreSpec {
    pub bundles: usize,
    /// Fixed direction `[z, y, x]` (normalised here), or random per bundle.
    pub direction: Option<[f64; 3]>,
    /// Radius range of a bundle's cross-section disk.
    pub bundle_radius: [f64; 2],
    /// Radius range of one fibre.
    pub fibre_radius: [f64; 2],
    /// Minimum surface-to-surface gap between fibres of a bundle.
    pub gap: f64,
    /// Dart-throwing attempts per bundle.
    pub attempts: usize,
    /// Coherent bundle wiggle: amplitude and wavelength along the bundle.
    pub wiggle_amplitude: f64,
    pub wiggle_wavelength: f64,
    /// Independent wiggle amplitude per fibre (keep below `gap / 2`).
    pub own_wiggle: f64,
    /// Relative radius variation along a fibre (`0..1`).
    pub radius_variation: f64,
}

/// How empty space is filled.
#[derive(Debug, Clone, PartialEq)]
pub struct FillSpec {
    /// Objects first claim empty voxels within this distance (0 disables).
    pub object_growth: f64,
    /// Number of extra fill ("glia") seeds to attempt.
    pub seeds: usize,
    /// Fill seeds need at least this distance to any object...
    pub min_seed_distance: f64,
    /// ...and this distance to each other.
    pub seed_spacing: f64,
    /// Amplitude (length units) and scale of the noise added to the flooding
    /// cost; this is what makes fill cells irregular.
    pub texture: f64,
    pub texture_scale: f64,
}

/// Extracellular space.
#[derive(Debug, Clone, PartialEq)]
pub struct EcsSpec {
    /// Gap width in xy voxels, rounded to whole voxels, carved from the larger
    /// cell of each pair.
    pub width: f64,
    /// Fraction of touching label pairs that get a gap.
    pub fraction: f64,
}

/// Everything the tissue generator needs. There is no `Default`: start from
/// [`TissueSpec::empty`] and switch components on.
#[derive(Debug, Clone, PartialEq)]
pub struct TissueSpec {
    /// `[z, y, x]`.
    pub shape: [usize; 3],
    pub seed: u64,
    /// Physical voxel size per axis; all lengths in the specs are in its units.
    pub spacing: [f64; 3],
    pub fibres: Option<FibreSpec>,
    pub somata: Option<SomaSpec>,
    pub neurites: Option<NeuriteSpec>,
    pub fill: FillSpec,
    pub ecs: Option<EcsSpec>,
    /// Allowed fractional overlap of two objects' spheres before a step counts
    /// as a collision (`0` = none, `1` = never collide).
    pub collision_tolerance: f64,
    /// Connected pieces smaller than this are merged into a neighbour (0 keeps
    /// every piece as its own label).
    pub min_fragment_voxels: usize,
}

impl TissueSpec {
    /// No objects, fill only by the seeds given later. A starting point, not a
    /// configuration.
    pub fn empty(shape: [usize; 3], seed: u64, spacing: [f64; 3]) -> Self {
        Self {
            shape,
            seed,
            spacing,
            fibres: None,
            somata: None,
            neurites: None,
            fill: FillSpec {
                object_growth: 0.0,
                seeds: 0,
                min_seed_distance: 0.0,
                seed_spacing: 0.0,
                texture: 0.0,
                texture_scale: 1.0,
            },
            ecs: None,
            collision_tolerance: 0.0,
            min_fragment_voxels: 0,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.shape.contains(&0) {
            return Err(Error::invalid(format!(
                "tissue spec: shape {:?} has an empty axis",
                self.shape
            )));
        }
        for s in self.spacing {
            positive("spacing", s)?;
        }
        unit_interval("collision tolerance", self.collision_tolerance)?;
        if let Some(f) = &self.fibres {
            range("fibre bundle radius", f.bundle_radius)?;
            range("fibre radius", f.fibre_radius)?;
            non_negative("fibre gap", f.gap)?;
            non_negative("fibre wiggle", f.wiggle_amplitude)?;
            positive("fibre wiggle wavelength", f.wiggle_wavelength)?;
            non_negative("fibre own wiggle", f.own_wiggle)?;
            unit_interval("fibre radius variation", f.radius_variation)?;
            if let Some(d) = f.direction {
                if norm(d) <= 0.0 || !norm(d).is_finite() {
                    return Err(Error::invalid("tissue spec: fibre direction is zero"));
                }
            }
        }
        if let Some(s) = &self.somata {
            range("soma radius", s.radius)?;
            if s.elongation < 1.0 || !s.elongation.is_finite() {
                return Err(Error::invalid("tissue spec: soma elongation below 1"));
            }
            unit_interval("soma irregularity", s.irregularity)?;
            positive("soma irregularity scale", s.irregularity_scale)?;
            if s.neurites[1] < s.neurites[0] {
                return Err(Error::invalid("tissue spec: soma neurite range unordered"));
            }
            s.shape.validate()?;
        }
        if let Some(n) = &self.neurites {
            unit_interval("enter-from-face fraction", n.enter_from_face)?;
            n.shape.validate()?;
        }
        non_negative("object growth", self.fill.object_growth)?;
        non_negative("min seed distance", self.fill.min_seed_distance)?;
        non_negative("seed spacing", self.fill.seed_spacing)?;
        non_negative("fill texture", self.fill.texture)?;
        positive("fill texture scale", self.fill.texture_scale)?;
        if let Some(e) = &self.ecs {
            positive("ECS width", e.width)?;
            unit_interval("ECS fraction", e.fraction)?;
        }
        Ok(())
    }
}

impl ProcessShape {
    fn validate(&self) -> Result<()> {
        range("process radius", self.radius)?;
        positive("process min radius", self.min_radius)?;
        unit_interval("radius variation", self.radius_variation)?;
        positive("radius wavelength", self.radius_wavelength)?;
        positive("persistence", self.persistence)?;
        positive("max turn", self.max_turn)?;
        range("process length", self.length)?;
        non_negative("branch rate", self.branch_rate)?;
        positive("branch radius factor", self.branch_radius_factor)?;
        if !(self.branch_spread[0] >= 0.0 && self.branch_spread[1] >= self.branch_spread[0]) {
            return Err(Error::invalid("tissue spec: branch spread unordered"));
        }
        non_negative("neck rate", self.neck_rate)?;
        positive("neck factor", self.neck_factor)?;
        positive("neck length", self.neck_length)?;
        non_negative("varicosity rate", self.varicosity_rate)?;
        positive("varicosity factor", self.varicosity_factor)?;
        positive("varicosity length", self.varicosity_length)?;
        Ok(())
    }
}

fn positive(name: &str, v: f64) -> Result<()> {
    if v > 0.0 && v.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "tissue spec: {name} is {v}, must be positive"
        )))
    }
}

fn non_negative(name: &str, v: f64) -> Result<()> {
    if v >= 0.0 && v.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "tissue spec: {name} is {v}, must be >= 0"
        )))
    }
}

fn unit_interval(name: &str, v: f64) -> Result<()> {
    if (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "tissue spec: {name} is {v}, outside 0..=1"
        )))
    }
}

fn range(name: &str, r: [f64; 2]) -> Result<()> {
    if r[0] > 0.0 && r[1] >= r[0] && r[1].is_finite() {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "tissue spec: {name} {r:?} must be positive and ordered"
        )))
    }
}

// ============================================================== outputs ==

/// What a label is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabelKind {
    /// Label 0 (only present with extracellular space).
    Extracellular,
    Neurite,
    Fibre,
    Soma,
    Fill,
}

/// A centreline graph inside one final label.
#[derive(Debug, Clone, PartialEq)]
pub struct Skeleton {
    pub label: u32,
    /// `[z, y, x]` in voxels; voxel `i` covers `[i, i + 1)`, so the node lies in
    /// voxel `floor(node)`.
    pub nodes: Vec<[f64; 3]>,
    /// Process radius at each node, in spacing units.
    pub radii: Vec<f64>,
    /// Pairs of node indices.
    pub edges: Vec<[u32; 2]>,
}

/// A generated tissue volume and its ground truth.
#[derive(Debug, Clone, PartialEq)]
pub struct TissueTruth {
    /// Dense labels `1..=n`; 0 only where extracellular space was requested.
    pub labels: Array3<u32>,
    pub spacing: [f64; 3],
    /// Indexed by label, `kinds[0] == Extracellular`.
    pub kinds: Vec<LabelKind>,
    /// Indexed by label: the generated object (1-based, generation order) the
    /// label descends from; several labels share an origin when it was split.
    pub origins: Vec<u32>,
    /// Centrelines of neurite, soma and fibre labels, ascending by label.
    pub skeletons: Vec<Skeleton>,
}

impl TissueTruth {
    pub fn label_count(&self) -> usize {
        self.kinds.len() - 1
    }

    /// The hints the renderer uses (kinds for myelin, skeletons to orient
    /// mitochondria).
    pub fn hints(&self) -> LabelHints<'_> {
        LabelHints {
            kinds: &self.kinds,
            skeletons: &self.skeletons,
        }
    }
}

// ========================================================= small helpers ==

const SALT_FIBRE: u64 = 0x5449_5353_4649_4201;
const SALT_SOMA: u64 = 0x5449_5353_534f_4d02;
const SALT_NEURITE: u64 = 0x5449_5353_4e45_5503;
const SALT_FILL: u64 = 0x5449_5353_4649_4c04;
const SALT_ECS: u64 = 0x5449_5353_4543_5305;
const SALT_TEXTURE: u64 = 0x5449_5353_5445_5806;
const SALT_CYTO: u64 = 0x5245_4e44_4359_5407;
const SALT_RTEXTURE: u64 = 0x5245_4e44_5458_5408;
const SALT_MITO: u64 = 0x5245_4e44_4d49_5409;
const SALT_VESICLE: u64 = 0x5245_4e44_5645_530a;
const SALT_MYELIN: u64 = 0x5245_4e44_4d59_450b;
const SALT_JITTER: u64 = 0x5245_4e44_4a49_540c;
const SALT_DRIFT: u64 = 0x5245_4e44_4452_490d;
const SALT_CURTAIN: u64 = 0x5245_4e44_4355_520e;
const SALT_SHOT: u64 = 0x5245_4e44_5348_4f0f;
const SALT_READ: u64 = 0x5245_4e44_5245_4110;
const SALT_SCAN: u64 = 0x5245_4e44_5343_4111;

type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn normalise(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        [0.0, 0.0, 1.0]
    }
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn lerp(lo: f64, hi: f64, t: f64) -> f64 {
    lo + (hi - lo) * t
}

/// Approximately standard normal: four uniforms, as `synthetic` does.
fn gauss(rng: &mut Stream) -> f64 {
    (rng.unit() + rng.unit() + rng.unit() + rng.unit() - 2.0) * 1.732_050_807_568_877_2
}

/// Approximately standard normal from one hash.
fn gauss_of(hash: u64) -> f64 {
    let mut sum = 0.0;
    for quarter in 0..4 {
        sum += ((hash >> (quarter * 16)) & 0xffff) as f64 / 65536.0;
    }
    (sum - 2.0) * 1.732_050_807_568_877_2
}

fn gauss3(rng: &mut Stream) -> V3 {
    [gauss(rng), gauss(rng), gauss(rng)]
}

fn random_unit(rng: &mut Stream) -> V3 {
    for _ in 0..16 {
        let v = gauss3(rng);
        if norm(v) > 1e-6 {
            return normalise(v);
        }
    }
    [0.0, 0.0, 1.0]
}

fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// 1-D value noise in `-1..1`, unit wavelength.
fn noise1(seed: u64, t: f64) -> f64 {
    let f = t.floor();
    let i = f as i64;
    let w = smooth(t - f);
    let a = unit_from(hash3(seed, i as u64, 0, 0)) * 2.0 - 1.0;
    let b = unit_from(hash3(seed, (i + 1) as u64, 0, 0)) * 2.0 - 1.0;
    a + (b - a) * w
}

/// 3-D value noise in `-1..1` on a lattice of spacing `cell`, at physical `p`.
fn noise3(seed: u64, p: V3, cell: f64) -> f64 {
    let mut corner = [0i64; 3];
    let mut frac = [0.0; 3];
    for a in 0..3 {
        let s = p[a] / cell;
        let f = s.floor();
        corner[a] = f as i64;
        frac[a] = smooth(s - f);
    }
    let mut total = 0.0;
    for step in 0..8 {
        let o = [(step >> 2) & 1, (step >> 1) & 1, step & 1];
        let mut w = 1.0;
        for a in 0..3 {
            w *= if o[a] == 1 { frac[a] } else { 1.0 - frac[a] };
        }
        let h = hash3(
            seed,
            (corner[0] + o[0] as i64) as u64,
            (corner[1] + o[1] as i64) as u64,
            (corner[2] + o[2] as i64) as u64,
        );
        total += w * (unit_from(h) * 2.0 - 1.0);
    }
    total
}

/// Physical position of a voxel centre.
fn centre_of(z: usize, y: usize, x: usize, spacing: V3) -> V3 {
    [
        (z as f64 + 0.5) * spacing[0],
        (y as f64 + 0.5) * spacing[1],
        (x as f64 + 0.5) * spacing[2],
    ]
}

fn distance_params(spacing: V3) -> DistanceParams {
    DistanceParams::default()
        .with_sampling(spacing)
        .with_unbounded(Unbounded::Infinite)
}

// ====================================================== Part A: geometry ==

#[derive(Debug, Clone)]
enum Primitive {
    Sphere {
        centre: V3,
        radius: f64,
        id: u32,
    },
    Blob {
        centre: V3,
        radii: V3,
        irregularity: f64,
        cell: f64,
        seed: u64,
        id: u32,
    },
}

impl Primitive {
    fn id(&self) -> u32 {
        match self {
            Self::Sphere { id, .. } | Self::Blob { id, .. } => *id,
        }
    }

    /// Physical half-extent per axis.
    fn half(&self) -> V3 {
        match self {
            Self::Sphere { radius, .. } => [*radius; 3],
            Self::Blob {
                radii,
                irregularity,
                ..
            } => scale(*radii, 1.0 + irregularity),
        }
    }

    fn centre(&self) -> V3 {
        match self {
            Self::Sphere { centre, .. } | Self::Blob { centre, .. } => *centre,
        }
    }

    fn covers(&self, p: V3) -> bool {
        match self {
            Self::Sphere { centre, radius, .. } => {
                let d = sub(p, *centre);
                dot(d, d) <= radius * radius
            }
            Self::Blob {
                centre,
                radii,
                irregularity,
                cell,
                seed,
                ..
            } => {
                let d = sub(p, *centre);
                let q = [d[0] / radii[0], d[1] / radii[1], d[2] / radii[2]];
                let limit = 1.0 + irregularity * noise3(*seed, p, *cell);
                dot(q, q) <= limit * limit
            }
        }
    }
}

/// Spheres already placed, for collision tests. Every sphere is registered in
/// every cell its box meets, so a query only looks at the cells of its own box.
struct Occupancy {
    cell: f64,
    cells: HashMap<[i64; 3], Vec<u32>>,
    spheres: Vec<(V3, f64, u32)>,
}

impl Occupancy {
    fn new(cell: f64) -> Self {
        Self {
            cell: cell.max(1.0),
            cells: HashMap::new(),
            spheres: Vec::new(),
        }
    }

    fn cells_of(&self, c: V3, r: f64) -> ([i64; 3], [i64; 3]) {
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        for a in 0..3 {
            lo[a] = ((c[a] - r) / self.cell).floor() as i64;
            hi[a] = ((c[a] + r) / self.cell).floor() as i64;
        }
        (lo, hi)
    }

    fn insert(&mut self, c: V3, r: f64, owner: u32) {
        let index = self.spheres.len() as u32;
        self.spheres.push((c, r, owner));
        let (lo, hi) = self.cells_of(c, r);
        for z in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for x in lo[2]..=hi[2] {
                    self.cells.entry([z, y, x]).or_default().push(index);
                }
            }
        }
    }

    fn collides(&self, c: V3, r: f64, owner: u32, tolerance: f64) -> bool {
        let (lo, hi) = self.cells_of(c, r);
        for z in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for x in lo[2]..=hi[2] {
                    let Some(list) = self.cells.get(&[z, y, x]) else {
                        continue;
                    };
                    for &i in list {
                        let (oc, or, oo) = self.spheres[i as usize];
                        if oo == owner {
                            continue;
                        }
                        let reach = (r + or) * (1.0 - tolerance);
                        let d = sub(c, oc);
                        if dot(d, d) < reach * reach {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

struct RawSkeleton {
    object: u32,
    nodes: Vec<V3>, // physical
    radii: Vec<f64>,
    edges: Vec<[u32; 2]>,
}

impl RawSkeleton {
    fn new(object: u32) -> Self {
        Self {
            object,
            nodes: Vec::new(),
            radii: Vec::new(),
            edges: Vec::new(),
        }
    }

    fn push(&mut self, p: V3, r: f64, parent: Option<u32>) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(p);
        self.radii.push(r);
        if let Some(parent) = parent {
            self.edges.push([parent, index]);
        }
        index
    }
}

struct Builder<'s> {
    spec: &'s TissueSpec,
    extent: V3,
    kinds: Vec<LabelKind>, // by object id - 1
    primitives: Vec<Primitive>,
    skeletons: Vec<RawSkeleton>,
    occupancy: Occupancy,
}

/// One branch waiting to be walked.
struct Branch {
    position: V3,
    direction: V3,
    radius: f64,
    parent: Option<u32>,
    /// Roots place a node at their start; children start by stepping.
    place_start: bool,
}

/// Safety bound on nodes per process, far above any sensible configuration.
const MAX_NODES: usize = 200_000;

impl<'s> Builder<'s> {
    fn new(spec: &'s TissueSpec) -> Self {
        let extent = [
            spec.shape[0] as f64 * spec.spacing[0],
            spec.shape[1] as f64 * spec.spacing[1],
            spec.shape[2] as f64 * spec.spacing[2],
        ];
        let mut largest: f64 = 1.0;
        for shape in [
            spec.somata.as_ref().map(|s| &s.shape),
            spec.neurites.as_ref().map(|n| &n.shape),
        ]
        .into_iter()
        .flatten()
        {
            largest = largest.max(shape.radius[1] * shape.varicosity_factor.max(1.0));
        }
        if let Some(f) = &spec.fibres {
            largest = largest.max(f.fibre_radius[1] * (1.0 + f.radius_variation));
        }
        Self {
            spec,
            extent,
            kinds: Vec::new(),
            primitives: Vec::new(),
            skeletons: Vec::new(),
            occupancy: Occupancy::new(2.0 * largest),
        }
    }

    fn new_object(&mut self, kind: LabelKind) -> u32 {
        self.kinds.push(kind);
        self.kinds.len() as u32
    }

    fn outside(&self, p: V3, r: f64) -> bool {
        (0..3).any(|a| p[a] < -r || p[a] > self.extent[a] + r)
    }

    fn sphere(&mut self, p: V3, r: f64, id: u32) {
        self.primitives.push(Primitive::Sphere {
            centre: p,
            radius: r,
            id,
        });
        self.occupancy.insert(p, r, id);
    }

    // ------------------------------------------------------------ fibres --

    fn fibres(&mut self, spec: &FibreSpec) {
        let diagonal = norm(self.extent);
        for bundle in 0..spec.bundles {
            let mut rng = Stream::new(self.spec.seed ^ SALT_FIBRE, bundle as u64);
            let direction = match spec.direction {
                Some(d) => normalise(d),
                None => random_unit(&mut rng),
            };
            let origin = [
                rng.unit() * self.extent[0],
                rng.unit() * self.extent[1],
                rng.unit() * self.extent[2],
            ];
            let helper = if direction[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let u = normalise(cross(direction, helper));
            let v = cross(direction, u);
            let disk = lerp(spec.bundle_radius[0], spec.bundle_radius[1], rng.unit());

            // Poisson-disk packing of the cross-section.
            let mut placed: Vec<(f64, f64, f64)> = Vec::new();
            for _ in 0..spec.attempts {
                let r = lerp(spec.fibre_radius[0], spec.fibre_radius[1], rng.unit());
                let room = disk - r;
                if room < 0.0 {
                    continue;
                }
                let a = (rng.unit() * 2.0 - 1.0) * room;
                let b = (rng.unit() * 2.0 - 1.0) * room;
                if a * a + b * b > room * room {
                    continue;
                }
                let free = placed.iter().all(|&(pa, pb, pr)| {
                    let reach = r + pr + spec.gap;
                    (a - pa) * (a - pa) + (b - pb) * (b - pb) >= reach * reach
                });
                if free {
                    placed.push((a, b, r));
                }
            }

            let wiggle_seed = rng.next_u64();
            let half = 0.5 * diagonal + disk + spec.wiggle_amplitude;
            for (fibre, &(a, b, r)) in placed.iter().enumerate() {
                let id = self.new_object(LabelKind::Fibre);
                let own = Stream::new(wiggle_seed, fibre as u64 + 1).next_u64();
                let mut skeleton = RawSkeleton::new(id);
                let step = (0.5 * r * (1.0 - spec.radius_variation)).max(0.5);
                let mut parent: Option<u32> = None;
                let mut t = -half;
                while t <= half {
                    let phase = t / spec.wiggle_wavelength;
                    let coherent = [
                        spec.wiggle_amplitude * noise1(wiggle_seed, phase),
                        spec.wiggle_amplitude * noise1(wiggle_seed ^ 0x55, phase),
                    ];
                    let private = [
                        spec.own_wiggle * noise1(own, phase),
                        spec.own_wiggle * noise1(own ^ 0x55, phase),
                    ];
                    let across = add(
                        scale(u, a + coherent[0] + private[0]),
                        scale(v, b + coherent[1] + private[1]),
                    );
                    let p = add(add(origin, scale(direction, t)), across);
                    let radius = r * (1.0 + spec.radius_variation * noise1(own ^ 0xaa, phase));
                    if self.outside(p, radius) {
                        parent = None;
                    } else {
                        self.sphere(p, radius, id);
                        parent = Some(skeleton.push(p, radius, parent));
                    }
                    t += step;
                }
                self.skeletons.push(skeleton);
            }
        }
    }

    // ------------------------------------------------------------ somata --

    fn somata(&mut self, spec: &SomaSpec) {
        for soma in 0..spec.count {
            let mut rng = Stream::new(self.spec.seed ^ SALT_SOMA, soma as u64);
            let mean = lerp(spec.radius[0], spec.radius[1], rng.unit());
            let mut radii = [0.0; 3];
            for r in radii.iter_mut() {
                *r = mean * lerp(1.0 / spec.elongation, spec.elongation, rng.unit());
            }
            let mut centre = None;
            for _ in 0..32 {
                let c = [
                    rng.unit() * self.extent[0],
                    rng.unit() * self.extent[1],
                    rng.unit() * self.extent[2],
                ];
                if !self
                    .occupancy
                    .collides(c, mean, u32::MAX, self.spec.collision_tolerance)
                {
                    centre = Some(c);
                    break;
                }
            }
            let Some(centre) = centre else { continue };
            let id = self.new_object(LabelKind::Soma);
            self.primitives.push(Primitive::Blob {
                centre,
                radii,
                irregularity: spec.irregularity,
                cell: spec.irregularity_scale * mean,
                seed: rng.next_u64(),
                id,
            });
            self.occupancy.insert(centre, mean, id);
            let mut skeleton = RawSkeleton::new(id);
            let root = skeleton.push(centre, mean, None);
            let count = spec.neurites[0]
                + ((rng.unit() * (spec.neurites[1] - spec.neurites[0] + 1) as f64) as usize)
                    .min(spec.neurites[1] - spec.neurites[0]);
            let mut roots = Vec::new();
            for _ in 0..count {
                let direction = random_unit(&mut rng);
                let radius = lerp(spec.shape.radius[0], spec.shape.radius[1], rng.unit());
                roots.push(Branch {
                    position: centre,
                    direction,
                    radius,
                    parent: Some(root),
                    place_start: false,
                });
            }
            self.walk(id, &spec.shape, roots, &mut rng, &mut skeleton);
            self.skeletons.push(skeleton);
        }
    }

    // ---------------------------------------------------------- neurites --

    fn neurites(&mut self, spec: &NeuriteSpec) {
        for neurite in 0..spec.count {
            let mut rng = Stream::new(self.spec.seed ^ SALT_NEURITE, neurite as u64);
            let radius = lerp(spec.shape.radius[0], spec.shape.radius[1], rng.unit());
            let from_face = rng.unit() < spec.enter_from_face;
            let mut start = None;
            for _ in 0..16 {
                let mut p = [
                    rng.unit() * self.extent[0],
                    rng.unit() * self.extent[1],
                    rng.unit() * self.extent[2],
                ];
                let mut d = random_unit(&mut rng);
                if from_face {
                    let axis = ((rng.unit() * 3.0) as usize).min(2);
                    let high = rng.unit() < 0.5;
                    p[axis] = if high { self.extent[axis] } else { 0.0 };
                    let mut inward = [0.0; 3];
                    inward[axis] = if high { -1.0 } else { 1.0 };
                    d = normalise(add(inward, scale(random_unit(&mut rng), 0.7)));
                    if d[axis] * inward[axis] <= 0.1 {
                        d = inward;
                    }
                }
                if !self
                    .occupancy
                    .collides(p, radius, u32::MAX, self.spec.collision_tolerance)
                {
                    start = Some((p, d));
                    break;
                }
            }
            let Some((p, d)) = start else { continue };
            let id = self.new_object(LabelKind::Neurite);
            let mut skeleton = RawSkeleton::new(id);
            let root = skeleton.push(p, radius, None);
            self.sphere(p, radius, id);
            let mut roots = vec![Branch {
                position: p,
                direction: d,
                radius,
                parent: Some(root),
                place_start: false,
            }];
            if !from_face {
                roots.push(Branch {
                    position: p,
                    direction: scale(d, -1.0),
                    radius,
                    parent: Some(root),
                    place_start: false,
                });
            }
            self.walk(id, &spec.shape, roots, &mut rng, &mut skeleton);
            self.skeletons.push(skeleton);
        }
    }

    /// Walk every branch of one process, breadth first, spawning children.
    fn walk(
        &mut self,
        id: u32,
        shape: &ProcessShape,
        roots: Vec<Branch>,
        rng: &mut Stream,
        skeleton: &mut RawSkeleton,
    ) {
        let tolerance = self.spec.collision_tolerance;
        let mut queue: VecDeque<Branch> = roots.into();
        let mut spawned = 0usize;
        while let Some(branch) = queue.pop_front() {
            if skeleton.nodes.len() >= MAX_NODES {
                break;
            }
            let length = lerp(shape.length[0], shape.length[1], rng.unit());
            let radius_seed = rng.next_u64();
            let mut events: Vec<(f64, f64, f64)> = Vec::new(); // centre, half, factor
            let radius_at = |s: f64, events: &[(f64, f64, f64)]| {
                let mut r = branch.radius
                    * (1.0
                        + shape.radius_variation
                            * noise1(radius_seed, s / shape.radius_wavelength));
                for &(c, h, f) in events {
                    let t = 1.0 - (s - c).abs() / h;
                    if t > 0.0 {
                        r *= 1.0 + (f - 1.0) * smooth(t);
                    }
                }
                r.max(shape.min_radius)
            };
            let mut p = branch.position;
            let mut d = branch.direction;
            let mut parent = branch.parent;
            let mut s = 0.0;
            if branch.place_start {
                let r = radius_at(0.0, &events);
                if self.occupancy.collides(p, r, id, tolerance) || self.outside(p, r) {
                    continue;
                }
                self.sphere(p, r, id);
                parent = Some(skeleton.push(p, r, parent));
            }
            while s < length && skeleton.nodes.len() < MAX_NODES {
                let r = radius_at(s, &events);
                let h = (0.5 * r).max(0.5);
                let mut kick = scale(gauss3(rng), (h / shape.persistence).sqrt());
                let bound = shape.max_turn * h;
                let size = norm(kick);
                if size > bound {
                    kick = scale(kick, bound / size);
                }
                let mut next_d = normalise(add(d, kick));
                let r_next = radius_at(s + h, &events);
                let mut q = add(p, scale(next_d, h));
                if self.occupancy.collides(q, r_next, id, tolerance) {
                    let mut found = false;
                    for attempt in 1..=8 {
                        let trial =
                            normalise(add(next_d, scale(random_unit(rng), 0.25 * attempt as f64)));
                        let trial_q = add(p, scale(trial, h));
                        if !self.occupancy.collides(trial_q, r_next, id, tolerance) {
                            next_d = trial;
                            q = trial_q;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        break;
                    }
                }
                p = q;
                d = next_d;
                s += h;
                if self.outside(p, r_next) {
                    break;
                }
                self.sphere(p, r_next, id);
                let node = skeleton.push(p, r_next, parent);
                parent = Some(node);

                if spawned < shape.max_branches && rng.unit() < shape.branch_rate * h {
                    spawned += 1;
                    let side = random_unit(rng);
                    let perpendicular = normalise(sub(side, scale(d, dot(side, d))));
                    let spread = lerp(shape.branch_spread[0], shape.branch_spread[1], rng.unit());
                    queue.push_back(Branch {
                        position: p,
                        direction: normalise(add(d, scale(perpendicular, spread))),
                        radius: (r_next * shape.branch_radius_factor).max(shape.min_radius),
                        parent: Some(node),
                        place_start: false,
                    });
                }
                if rng.unit() < shape.neck_rate * h {
                    let half = 0.5 * shape.neck_length;
                    events.push((s + half, half, shape.neck_factor));
                }
                if rng.unit() < shape.varicosity_rate * h {
                    let half = 0.5 * shape.varicosity_length;
                    events.push((s + half, half, shape.varicosity_factor));
                }
            }
        }
    }

    // -------------------------------------------------------- rasterise --

    /// Lowest covering object id per voxel, 0 where nothing covers.
    fn rasterise(&self) -> Array3<u32> {
        let [nz, ny, nx] = self.spec.shape;
        let sp = self.spec.spacing;
        let mut by_plane: Vec<Vec<u32>> = vec![Vec::new(); nz];
        for (i, prim) in self.primitives.iter().enumerate() {
            let c = prim.centre();
            let h = prim.half();
            let Some((lo, hi)) = index_range(c[0], h[0], sp[0], nz) else {
                continue;
            };
            for plane in by_plane.iter_mut().take(hi).skip(lo) {
                plane.push(i as u32);
            }
        }
        let mut labels = vec![u32::MAX; nz * ny * nx];
        labels
            .par_chunks_mut(ny * nx)
            .enumerate()
            .for_each(|(z, plane)| {
                for &i in &by_plane[z] {
                    let prim = &self.primitives[i as usize];
                    let id = prim.id();
                    let c = prim.centre();
                    let h = prim.half();
                    let Some((ylo, yhi)) = index_range(c[1], h[1], sp[1], ny) else {
                        continue;
                    };
                    let Some((xlo, xhi)) = index_range(c[2], h[2], sp[2], nx) else {
                        continue;
                    };
                    for y in ylo..yhi {
                        for x in xlo..xhi {
                            let at = y * nx + x;
                            if plane[at] <= id {
                                continue;
                            }
                            if prim.covers(centre_of(z, y, x, sp)) {
                                plane[at] = id;
                            }
                        }
                    }
                }
            });
        for l in labels.iter_mut() {
            if *l == u32::MAX {
                *l = 0;
            }
        }
        Array3::from_shape_vec((nz, ny, nx), labels).expect("shape matches")
    }
}

/// Voxel index range `lo..hi` whose centres can lie within `half` of `c`.
fn index_range(c: f64, half: f64, spacing: f64, n: usize) -> Option<(usize, usize)> {
    let lo = ((c - half) / spacing - 0.5).ceil().max(0.0);
    let hi = ((c + half) / spacing - 0.5).floor() + 1.0;
    let hi = hi.min(n as f64);
    if hi <= lo {
        return None;
    }
    Some((lo as usize, hi as usize))
}

/// Generate a dense synthetic tissue volume with its ground truth.
pub fn generate_tissue(spec: &TissueSpec) -> Result<TissueTruth> {
    spec.validate()?;
    let mut builder = Builder::new(spec);
    if let Some(f) = &spec.fibres {
        builder.fibres(f);
    }
    if let Some(s) = &spec.somata {
        builder.somata(s);
    }
    if let Some(n) = &spec.neurites {
        builder.neurites(n);
    }
    let mut labels = builder.rasterise();
    let mut kinds = builder.kinds.clone();
    fill(spec, &mut labels, &mut kinds)?;
    if let Some(ecs) = &spec.ecs {
        extracellular(spec, ecs, &mut labels)?;
    }
    let origins_of_group = split_components(&mut labels, spec.min_fragment_voxels);

    let mut final_kinds = Vec::with_capacity(origins_of_group.len() + 1);
    final_kinds.push(LabelKind::Extracellular);
    let mut origins = vec![0u32];
    for &origin in &origins_of_group {
        final_kinds.push(kinds[origin as usize - 1]);
        origins.push(origin);
    }
    let skeletons = finish_skeletons(&builder.skeletons, &labels, &origins, spec.spacing);
    Ok(TissueTruth {
        labels,
        spacing: spec.spacing,
        kinds: final_kinds,
        origins,
        skeletons,
    })
}

/// Grow objects, then partition what is left among fill seeds.
fn fill(spec: &TissueSpec, labels: &mut Array3<u32>, kinds: &mut Vec<LabelKind>) -> Result<()> {
    let params = distance_params(spec.spacing);
    let shape = spec.shape;
    let any_object = labels.iter().any(|&l| l != 0);

    if spec.fill.object_growth > 0.0 && any_object {
        let empty = labels.mapv(|l| l == 0);
        let distance = distance_transform(empty.view(), &params)?;
        let mask = Zip::from(&distance)
            .and(&*labels)
            .map_collect(|&d, &l| l != 0 || d <= spec.fill.object_growth);
        let grown = seeded_watershed(
            distance.view(),
            labels.view(),
            Some(mask.view()),
            Separation::Adjacent,
        )?;
        *labels = grown;
    }

    let empty = labels.mapv(|l| l == 0);
    if !empty.iter().any(|&e| e) {
        return Ok(());
    }
    let mut distance = distance_transform(empty.view(), &params)?;
    distance.mapv_inplace(|d| if d.is_finite() { d } else { 0.0 });

    // Poisson-disk fill seeds.
    let mut rng = Stream::new(spec.seed ^ SALT_FILL, 0);
    let mut seeds: Vec<[usize; 3]> = Vec::new();
    let spacing2 = spec.fill.seed_spacing * spec.fill.seed_spacing;
    for _ in 0..spec.fill.seeds.saturating_mul(50) {
        if seeds.len() >= spec.fill.seeds {
            break;
        }
        let at = [
            ((rng.unit() * shape[0] as f64) as usize).min(shape[0] - 1),
            ((rng.unit() * shape[1] as f64) as usize).min(shape[1] - 1),
            ((rng.unit() * shape[2] as f64) as usize).min(shape[2] - 1),
        ];
        if !empty[at] || (any_object && distance[at] < spec.fill.min_seed_distance) {
            continue;
        }
        let p = centre_of(at[0], at[1], at[2], spec.spacing);
        let far = seeds.iter().all(|s| {
            let d = sub(p, centre_of(s[0], s[1], s[2], spec.spacing));
            dot(d, d) >= spacing2
        });
        if far {
            seeds.push(at);
        }
    }

    // One more seed, at its deepest voxel, in every empty component without one.
    let mut components = Array3::<u32>::zeros(shape);
    let count = label_members_into(shape, |at| empty[at], components.view_mut())? as usize;
    let mut seeded = vec![false; count + 1];
    for s in &seeds {
        seeded[components[*s] as usize] = true;
    }
    let mut deepest: Vec<Option<([usize; 3], f64)>> = vec![None; count + 1];
    for ((z, y, x), &c) in components.indexed_iter() {
        if c == 0 || seeded[c as usize] {
            continue;
        }
        let d = distance[[z, y, x]];
        match deepest[c as usize] {
            Some((_, best)) if best >= d => {}
            _ => deepest[c as usize] = Some(([z, y, x], d)),
        }
    }
    seeds.extend(deepest.into_iter().flatten().map(|(at, _)| at));

    let mut markers = Array3::<u32>::zeros(shape);
    for at in &seeds {
        kinds.push(LabelKind::Fill);
        markers[*at] = kinds.len() as u32;
    }
    let texture_seed = spec.seed ^ SALT_TEXTURE;
    let mut cost = Array3::<f64>::zeros(shape);
    let sp = spec.spacing;
    let texture = spec.fill.texture;
    let cell = spec.fill.texture_scale;
    let (ny, nx) = (shape[1], shape[2]);
    cost.as_slice_mut()
        .expect("standard layout")
        .par_chunks_mut(ny * nx)
        .zip(
            distance
                .as_slice()
                .expect("standard layout")
                .par_chunks(ny * nx),
        )
        .enumerate()
        .for_each(|(z, (out, dist))| {
            for y in 0..ny {
                for x in 0..nx {
                    let at = y * nx + x;
                    let mut c = -dist[at];
                    if texture > 0.0 {
                        c += texture * noise3(texture_seed, centre_of(z, y, x, sp), cell);
                    }
                    out[at] = c;
                }
            }
        });
    let flooded = seeded_watershed(
        cost.view(),
        markers.view(),
        Some(empty.view()),
        Separation::Adjacent,
    )?;
    Zip::from(&mut *labels).and(&flooded).for_each(|l, &f| {
        if *l == 0 {
            *l = f;
        }
    });
    Ok(())
}

/// Clear a gap along the interfaces of a hashed fraction of label pairs.
///
/// The gap is carved from the **larger** cell of each pair (ties: the higher
/// id), `round(width)` voxels deep and confined to that cell, so a thin process
/// next to a big cell keeps its thickness and is not cut. Growth into z counts
/// `round(spacing_z / spacing_xy)` xy steps per z step.
fn extracellular(spec: &TissueSpec, ecs: &EcsSpec, labels: &mut Array3<u32>) -> Result<()> {
    let (nz, ny, nx) = labels.dim();
    let seed = spec.seed ^ SALT_ECS;
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut size = vec![0usize; max_label + 1];
    for &l in labels.iter() {
        size[l as usize] += 1;
    }
    let gives = |a: u32, b: u32| {
        let (lo, hi) = (a.min(b), a.max(b));
        unit_from(hash3(seed, lo as u64, hi as u64, 0)) < ecs.fraction
            && (size[a as usize], a) > (size[b as usize], b)
    };
    const FACES: [(i64, i64, i64); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    let neighbour = |z: usize, y: usize, x: usize, (dz, dy, dx): (i64, i64, i64)| {
        let (zz, yy, xx) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
        (zz >= 0 && yy >= 0 && xx >= 0 && zz < nz as i64 && yy < ny as i64 && xx < nx as i64)
            .then_some([zz as usize, yy as usize, xx as usize])
    };
    let mut marks = Array3::<bool>::from_elem((nz, ny, nx), false);
    for ((z, y, x), &a) in labels.indexed_iter() {
        if a == 0 {
            continue;
        }
        marks[[z, y, x]] = FACES.iter().any(|&f| {
            neighbour(z, y, x, f).is_some_and(|at| {
                let b = labels[at];
                b != 0 && b != a && gives(a, b)
            })
        });
    }
    let passes = (ecs.width.round() as usize).saturating_sub(1);
    let z_every = ((spec.spacing[0] / spec.spacing[1]).round() as usize).max(1);
    for pass in 1..=passes {
        let grow_z = pass % z_every == 0;
        let previous = marks.clone();
        for ((z, y, x), &marked) in previous.indexed_iter() {
            if !marked {
                continue;
            }
            let l = labels[[z, y, x]];
            for &f in &FACES {
                if f.0 != 0 && !grow_z {
                    continue;
                }
                if let Some(at) = neighbour(z, y, x, f) {
                    if labels[at] == l {
                        marks[at] = true;
                    }
                }
            }
        }
    }
    Zip::from(&mut *labels).and(&marks).for_each(|l, &m| {
        if m {
            *l = 0;
        }
    });
    Ok(())
}

/// Split every label into 6-connected pieces, merge pieces smaller than
/// `min_fragment` into their most-touching neighbour (or clear them to 0 when
/// they touch only extracellular space), and renumber `1..=n` in
/// raster order. Returns, per new label (index `label - 1`), the old label of
/// its largest piece.
fn split_components(labels: &mut Array3<u32>, min_fragment: usize) -> Vec<u32> {
    let shape = labels.dim();
    let (nz, ny, nx) = shape;
    let flat = labels.as_slice().expect("standard layout").to_vec();
    let n = flat.len();
    let mut voxels = Union::new(n);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = (z * ny + y) * nx + x;
                let l = flat[i];
                if l == 0 {
                    continue;
                }
                if x + 1 < nx && flat[i + 1] == l {
                    voxels.union(i, i + 1);
                }
                if y + 1 < ny && flat[i + nx] == l {
                    voxels.union(i, i + nx);
                }
                if z + 1 < nz && flat[i + ny * nx] == l {
                    voxels.union(i, i + ny * nx);
                }
            }
        }
    }
    // Components in raster order of their first voxel.
    let mut component_of_root: HashMap<usize, u32> = HashMap::new();
    let mut component = vec![0u32; n];
    let mut size: Vec<usize> = vec![0];
    let mut old: Vec<u32> = vec![0];
    for i in 0..n {
        if flat[i] == 0 {
            continue;
        }
        let root = voxels.find(i);
        let c = *component_of_root.entry(root).or_insert_with(|| {
            size.push(0);
            old.push(flat[i]);
            size.len() as u32 - 1
        });
        component[i] = c;
        size[c as usize] += 1;
    }
    drop(voxels);
    let count = size.len() - 1;

    let mut groups = Union::new(count + 1);
    let mut dropped = vec![false; count + 1];
    if min_fragment > 0 {
        let mut contact: HashMap<(u32, u32), usize> = HashMap::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (z * ny + y) * nx + x;
                    let a = component[i];
                    if a == 0 {
                        continue;
                    }
                    let mut touch = |j: usize| {
                        let b = component[j];
                        if b != 0 && b != a {
                            *contact.entry((a, b)).or_default() += 1;
                            *contact.entry((b, a)).or_default() += 1;
                        }
                    };
                    if x + 1 < nx {
                        touch(i + 1);
                    }
                    if y + 1 < ny {
                        touch(i + nx);
                    }
                    if z + 1 < nz {
                        touch(i + ny * nx);
                    }
                }
            }
        }
        let mut best: Vec<Option<(usize, usize, u32)>> = vec![None; count + 1];
        for (&(a, b), &faces) in &contact {
            // Prefer most faces, then the larger neighbour, then the lower id.
            let key = (faces, size[b as usize], u32::MAX - b);
            let slot = &mut best[a as usize];
            if slot.map(|held| key > held).unwrap_or(true) {
                *slot = Some(key);
            }
        }
        for c in 1..=count {
            if size[c] < min_fragment {
                match best[c] {
                    Some((_, _, inv)) => groups.union(c, (u32::MAX - inv) as usize),
                    // Only extracellular space around it: it becomes that too.
                    None => dropped[c] = true,
                }
            }
        }
    }

    // Renumber groups in raster order; a group's origin is its largest piece.
    let mut final_of_root: HashMap<usize, u32> = HashMap::new();
    let mut largest: Vec<(usize, u32)> = Vec::new();
    let mut final_of_component = vec![0u32; count + 1];
    for i in 0..n {
        let c = component[i] as usize;
        if c == 0 || dropped[c] || final_of_component[c] != 0 {
            continue;
        }
        let root = groups.find(c);
        let f = *final_of_root.entry(root).or_insert_with(|| {
            largest.push((0, 0));
            largest.len() as u32
        });
        final_of_component[c] = f;
    }
    for c in 1..=count {
        let f = final_of_component[c] as usize;
        if f == 0 {
            continue;
        }
        let slot = &mut largest[f - 1];
        if size[c] > slot.0 {
            *slot = (size[c], old[c]);
        }
    }
    let out = labels.as_slice_mut().expect("standard layout");
    for i in 0..n {
        out[i] = final_of_component[component[i] as usize];
    }
    largest.into_iter().map(|(_, origin)| origin).collect()
}

/// Keep each skeleton node only where it lies in a label descended from its own
/// object, and split skeletons along label changes.
fn finish_skeletons(
    raw: &[RawSkeleton],
    labels: &Array3<u32>,
    origins: &[u32],
    spacing: V3,
) -> Vec<Skeleton> {
    let (nz, ny, nx) = labels.dim();
    let mut out: BTreeMap<u32, Skeleton> = BTreeMap::new();
    for skeleton in raw {
        let mut placed: Vec<Option<(u32, u32)>> = Vec::with_capacity(skeleton.nodes.len());
        for (node, &radius) in skeleton.nodes.iter().zip(&skeleton.radii) {
            let v = [
                node[0] / spacing[0],
                node[1] / spacing[1],
                node[2] / spacing[2],
            ];
            let inside = v[0] >= 0.0
                && v[1] >= 0.0
                && v[2] >= 0.0
                && (v[0] as usize) < nz
                && (v[1] as usize) < ny
                && (v[2] as usize) < nx;
            let label = if inside {
                labels[[v[0] as usize, v[1] as usize, v[2] as usize]]
            } else {
                0
            };
            if label == 0 || origins[label as usize] != skeleton.object {
                placed.push(None);
                continue;
            }
            let entry = out.entry(label).or_insert_with(|| Skeleton {
                label,
                nodes: Vec::new(),
                radii: Vec::new(),
                edges: Vec::new(),
            });
            entry.nodes.push(v);
            entry.radii.push(radius);
            placed.push(Some((label, entry.nodes.len() as u32 - 1)));
        }
        for &[a, b] in &skeleton.edges {
            if let (Some((la, ia)), Some((lb, ib))) = (placed[a as usize], placed[b as usize]) {
                if la == lb {
                    out.get_mut(&la).expect("present").edges.push([ia, ib]);
                }
            }
        }
    }
    out.into_values().collect()
}

// ============================================ Part B: image formation ==

/// Which way the contrast goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MembranePolarity {
    /// The configured intensities as given (dark membranes on light cytoplasm
    /// when configured so, as in EM).
    AsConfigured,
    /// The clean image reflected within the configured intensity range.
    Inverted,
}

/// Dark elongated organelles (mitochondria-like).
#[derive(Debug, Clone, PartialEq)]
pub struct MitochondriaSpec {
    /// Expected count per cell voxel.
    pub density: f64,
    /// Full length range along the local process direction.
    pub length: [f64; 2],
    /// Cross-section radius range (also capped at 0.6 × the process radius).
    pub radius: [f64; 2],
    pub intensity: f64,
}

/// Small dark spheres.
#[derive(Debug, Clone, PartialEq)]
pub struct VesicleSpec {
    pub density: f64,
    pub radius: [f64; 2],
    pub intensity: f64,
}

/// A thick dark wrap inside the boundary of some fibre labels.
#[derive(Debug, Clone, PartialEq)]
pub struct MyelinSpec {
    /// Fraction of fibre labels wrapped.
    pub fraction: f64,
    pub thickness: f64,
    pub intensity: f64,
}

/// Shot and read noise.
#[derive(Debug, Clone, PartialEq)]
pub struct ShotNoiseSpec {
    /// Expected detected photons per unit signal.
    pub photons_per_unit: f64,
    /// Gaussian read noise, signal units.
    pub read_noise: f64,
}

/// Noise correlated along the scan line (x).
#[derive(Debug, Clone, PartialEq)]
pub struct ScanlineSpec {
    pub sigma: f64,
    /// AR(1) coefficient between neighbouring x samples, `0..1`.
    pub correlation: f64,
}

/// Random walks of per-slice gain and offset along z.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftSpec {
    pub gain_step: f64,
    pub offset_step: f64,
}

/// Vertical stripes: per-column offsets constant over runs of slices.
#[derive(Debug, Clone, PartialEq)]
pub struct CurtainSpec {
    /// Standard deviation of a striped column's offset.
    pub amplitude: f64,
    /// Fraction of columns that are striped.
    pub column_fraction: f64,
    /// Run length range in slices, inclusive.
    pub run: [usize; 2],
}

/// Independent per-slice translation in y and x.
#[derive(Debug, Clone, PartialEq)]
pub struct JitterSpec {
    /// Standard deviation of the shift, voxels.
    pub sigma: f64,
    /// `false` rounds shifts to whole voxels.
    pub subpixel: bool,
}

/// Mapping to `u16`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSpec {
    pub offset: f64,
    pub scale: f64,
    pub range: [u16; 2],
}

/// The image formation model. No `Default`; see the tests for an example.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderSpec {
    pub seed: u64,
    pub polarity: MembranePolarity,
    /// Membrane thickness (spacing units) and intensity.
    pub membrane_thickness: f64,
    pub membrane_intensity: f64,
    /// Per-label cytoplasm level, uniform in this range.
    pub cytoplasm: [f64; 2],
    /// Relative texture amplitude and its correlation length.
    pub texture: f64,
    pub texture_scale: f64,
    /// Level of label 0.
    pub extracellular: f64,
    pub mitochondria: Option<MitochondriaSpec>,
    pub vesicles: Option<VesicleSpec>,
    pub myelin: Option<MyelinSpec>,
    /// PSF sigma per axis, in voxels (0 disables an axis).
    pub psf_sigma: [f64; 3],
    pub jitter: Option<JitterSpec>,
    pub drift: Option<DriftSpec>,
    pub curtaining: Option<CurtainSpec>,
    pub shot_noise: Option<ShotNoiseSpec>,
    pub scanline: Option<ScanlineSpec>,
    pub output: OutputSpec,
}

/// Optional knowledge about the labels the renderer can use.
#[derive(Debug, Clone, Copy, Default)]
pub struct LabelHints<'a> {
    /// Indexed by label; myelin needs `Fibre` entries.
    pub kinds: &'a [LabelKind],
    /// Orient mitochondria along these where available.
    pub skeletons: &'a [Skeleton],
}

/// The formed image and the nuisance parameters that were drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct FormedImage {
    pub image: Array3<u16>,
    /// Per slice `[dy, dx]`: output `(y, x)` reads input `(y - dy, x - dx)`.
    pub shifts: Vec<[f64; 2]>,
    pub gain: Vec<f64>,
    pub offset: Vec<f64>,
}

impl RenderSpec {
    fn validate(&self) -> Result<()> {
        non_negative("membrane thickness", self.membrane_thickness)?;
        if !(self.cytoplasm[1] >= self.cytoplasm[0]) {
            return Err(Error::invalid("render spec: cytoplasm range unordered"));
        }
        non_negative("texture", self.texture)?;
        positive("texture scale", self.texture_scale)?;
        for s in self.psf_sigma {
            non_negative("PSF sigma", s)?;
        }
        if let Some(m) = &self.mitochondria {
            non_negative("mitochondria density", m.density)?;
            range("mitochondria length", m.length)?;
            range("mitochondria radius", m.radius)?;
        }
        if let Some(v) = &self.vesicles {
            non_negative("vesicle density", v.density)?;
            range("vesicle radius", v.radius)?;
        }
        if let Some(m) = &self.myelin {
            unit_interval("myelin fraction", m.fraction)?;
            non_negative("myelin thickness", m.thickness)?;
        }
        if let Some(j) = &self.jitter {
            non_negative("jitter sigma", j.sigma)?;
        }
        if let Some(c) = &self.curtaining {
            unit_interval("curtain column fraction", c.column_fraction)?;
            if c.run[0] == 0 || c.run[1] < c.run[0] {
                return Err(Error::invalid(
                    "render spec: curtain run range must be >= 1 and ordered",
                ));
            }
        }
        if let Some(n) = &self.shot_noise {
            positive("photons per unit", n.photons_per_unit)?;
            non_negative("read noise", n.read_noise)?;
        }
        if let Some(s) = &self.scanline {
            non_negative("scanline sigma", s.sigma)?;
            if !(0.0..1.0).contains(&s.correlation) {
                return Err(Error::invalid(
                    "render spec: scanline correlation outside 0..1",
                ));
            }
        }
        if self.output.range[1] < self.output.range[0] {
            return Err(Error::invalid("render spec: output range unordered"));
        }
        Ok(())
    }

    /// Lowest and highest configured intensity, the range inversion reflects in.
    fn intensity_range(&self) -> (f64, f64) {
        let mut values = vec![
            self.membrane_intensity,
            self.cytoplasm[0] * (1.0 - self.texture),
            self.cytoplasm[1] * (1.0 + self.texture),
            self.extracellular,
        ];
        if let Some(m) = &self.mitochondria {
            values.push(m.intensity);
        }
        if let Some(v) = &self.vesicles {
            values.push(v.intensity);
        }
        if let Some(m) = &self.myelin {
            values.push(m.intensity);
        }
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    }
}

/// Render a dense label volume through the whole model.
pub fn render_tissue(
    labels: ArrayView3<'_, u32>,
    spacing: [f64; 3],
    hints: LabelHints<'_>,
    spec: &RenderSpec,
) -> Result<FormedImage> {
    let clean = clean_image(labels, spacing, hints, spec)?;
    form_image(clean, spec)
}

/// Stages 1–2: the noise-free, unblurred image of the labels.
pub fn clean_image(
    labels: ArrayView3<'_, u32>,
    spacing: [f64; 3],
    hints: LabelHints<'_>,
    spec: &RenderSpec,
) -> Result<Array3<f64>> {
    spec.validate()?;
    for s in spacing {
        positive("spacing", s)?;
    }
    let labels = labels.as_standard_layout();
    let (nz, ny, nx) = labels.dim();
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;

    // Distance from every voxel to the nearest boundary voxel.
    let mut boundary = Array3::<bool>::from_elem((nz, ny, nx), false);
    crate::ops::boundaries::find_boundaries_into(labels.view(), boundary.view_mut())?;
    let off_boundary = boundary.mapv(|b| !b);
    let to_boundary = distance_transform(off_boundary.view(), &distance_params(spacing))?;

    let cytoplasm: Vec<f64> = (0..=max_label)
        .map(|l| {
            lerp(
                spec.cytoplasm[0],
                spec.cytoplasm[1],
                unit_from(hash3(spec.seed ^ SALT_CYTO, l as u64, 0, 0)),
            )
        })
        .collect();
    let myelinated: Vec<bool> = (0..=max_label)
        .map(|l| match &spec.myelin {
            Some(m) => {
                hints.kinds.get(l) == Some(&LabelKind::Fibre)
                    && unit_from(hash3(spec.seed ^ SALT_MYELIN, l as u64, 0, 0)) < m.fraction
            }
            None => false,
        })
        .collect();
    let half = 0.5 * spec.membrane_thickness;
    let coverage = |d: f64| {
        if d.is_finite() {
            (half - d).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };

    let plane = ny * nx;
    let mut image = Array3::<f64>::zeros((nz, ny, nx));
    let label_slice = labels.as_slice().expect("standard layout");
    let dist_slice = to_boundary.as_slice().expect("standard layout");
    image
        .as_slice_mut()
        .expect("standard layout")
        .par_chunks_mut(plane)
        .enumerate()
        .for_each(|(z, out)| {
            for y in 0..ny {
                for x in 0..nx {
                    let at = y * nx + x;
                    let g = z * plane + at;
                    let l = label_slice[g] as usize;
                    out[at] = if l == 0 {
                        spec.extracellular
                    } else if myelinated[l]
                        && dist_slice[g] <= spec.myelin.as_ref().map_or(0.0, |m| m.thickness)
                    {
                        spec.myelin.as_ref().map_or(0.0, |m| m.intensity)
                    } else {
                        let seed = hash3(spec.seed ^ SALT_RTEXTURE, l as u64, 0, 0);
                        cytoplasm[l]
                            * (1.0
                                + spec.texture
                                    * noise3(seed, centre_of(z, y, x, spacing), spec.texture_scale))
                    };
                }
            }
        });

    organelles(
        &labels.view(),
        &to_boundary,
        &myelinated,
        spacing,
        hints,
        spec,
        half,
        &mut image,
    );

    Zip::from(&mut image)
        .and(&labels)
        .and(&to_boundary)
        .for_each(|v, &l, &d| {
            if l != 0 {
                let c = coverage(d);
                *v = (1.0 - c) * *v + c * spec.membrane_intensity;
            }
        });

    if spec.polarity == MembranePolarity::Inverted {
        let (lo, hi) = spec.intensity_range();
        image.mapv_inplace(|v| lo + hi - v);
    }
    Ok(image)
}

/// Mitochondria and vesicles, each clipped to its own label and kept off the
/// membrane and myelin, so no organelle ever crosses a cell boundary.
#[allow(clippy::too_many_arguments)]
fn organelles(
    labels: &ArrayView3<'_, u32>,
    to_boundary: &Array3<f64>,
    myelinated: &[bool],
    spacing: V3,
    hints: LabelHints<'_>,
    spec: &RenderSpec,
    membrane_half: f64,
    image: &mut Array3<f64>,
) {
    if spec.mitochondria.is_none() && spec.vesicles.is_none() {
        return;
    }
    let (nz, ny, nx) = labels.dim();
    let max_label = myelinated.len() - 1;
    let mut count = vec![0usize; max_label + 1];
    let mut lo = vec![[usize::MAX; 3]; max_label + 1];
    let mut hi = vec![[0usize; 3]; max_label + 1];
    for ((z, y, x), &l) in labels.indexed_iter() {
        let l = l as usize;
        count[l] += 1;
        for (a, c) in [z, y, x].into_iter().enumerate() {
            lo[l][a] = lo[l][a].min(c);
            hi[l][a] = hi[l][a].max(c);
        }
    }
    let mut skeletons: HashMap<u32, &Skeleton> = HashMap::new();
    for s in hints.skeletons {
        if s.nodes.len() >= 2 && !s.edges.is_empty() {
            skeletons.insert(s.label, s);
        }
    }
    let usable = |l: u32, at: [usize; 3]| {
        labels[at] == l && to_boundary[at] > membrane_half && !myelinated[l as usize]
    };
    let random_voxel = |rng: &mut Stream, l: usize| -> Option<V3> {
        for _ in 0..32 {
            let mut at = [0usize; 3];
            for a in 0..3 {
                at[a] = lo[l][a]
                    + ((rng.unit() * (hi[l][a] - lo[l][a] + 1) as f64) as usize)
                        .min(hi[l][a] - lo[l][a]);
            }
            if usable(l as u32, at) {
                return Some(centre_of(at[0], at[1], at[2], spacing));
            }
        }
        None
    };
    // An ellipsoid with semi-axis `along` on `axis` and `across` otherwise.
    let mut paint = |l: u32, c: V3, axis: V3, along: f64, across: f64, value: f64| {
        let reach = along.max(across);
        let mut range = [(0usize, 0usize); 3];
        for a in 0..3 {
            match index_range(c[a], reach, spacing[a], [nz, ny, nx][a]) {
                Some(r) => range[a] = r,
                None => return,
            }
        }
        for z in range[0].0..range[0].1 {
            for y in range[1].0..range[1].1 {
                for x in range[2].0..range[2].1 {
                    let d = sub(centre_of(z, y, x, spacing), c);
                    let t = dot(d, axis);
                    let r2 = dot(d, d) - t * t;
                    if (t * t) / (along * along) + r2 / (across * across) <= 1.0
                        && usable(l, [z, y, x])
                    {
                        image[[z, y, x]] = value;
                    }
                }
            }
        }
    };
    let draws = |rng: &mut Stream, expected: f64| {
        let whole = expected.floor();
        whole as usize + usize::from(rng.unit() < expected - whole)
    };

    for l in 1..=max_label {
        if count[l] == 0 {
            continue;
        }
        if let Some(m) = &spec.mitochondria {
            let mut rng = Stream::new(spec.seed ^ SALT_MITO, l as u64);
            for _ in 0..draws(&mut rng, m.density * count[l] as f64) {
                let length = lerp(m.length[0], m.length[1], rng.unit());
                let mut radius = lerp(m.radius[0], m.radius[1], rng.unit());
                let (centre, axis) = match skeletons.get(&(l as u32)) {
                    Some(s) => {
                        let e = s.edges
                            [((rng.unit() * s.edges.len() as f64) as usize).min(s.edges.len() - 1)];
                        let to_phys =
                            |n: [f64; 3]| [n[0] * spacing[0], n[1] * spacing[1], n[2] * spacing[2]];
                        let (a, b) = (
                            to_phys(s.nodes[e[0] as usize]),
                            to_phys(s.nodes[e[1] as usize]),
                        );
                        let node_r = s.radii[e[0] as usize];
                        radius = radius.min(0.6 * node_r);
                        let axis = normalise(sub(b, a));
                        let side = random_unit(&mut rng);
                        let side = normalise(sub(side, scale(axis, dot(side, axis))));
                        (add(a, scale(side, 0.3 * node_r * rng.unit())), axis)
                    }
                    None => match random_voxel(&mut rng, l) {
                        Some(c) => (c, random_unit(&mut rng)),
                        None => continue,
                    },
                };
                paint(l as u32, centre, axis, 0.5 * length, radius, m.intensity);
            }
        }
        if let Some(v) = &spec.vesicles {
            let mut rng = Stream::new(spec.seed ^ SALT_VESICLE, l as u64);
            for _ in 0..draws(&mut rng, v.density * count[l] as f64) {
                let radius = lerp(v.radius[0], v.radius[1], rng.unit());
                if let Some(c) = random_voxel(&mut rng, l) {
                    paint(l as u32, c, [0.0, 0.0, 1.0], radius, radius, v.intensity);
                }
            }
        }
    }
}

/// Stages 3–9 over any clean image: PSF, jitter, drift, curtaining, shot and
/// read noise, scan-line noise, quantisation.
pub fn form_image(mut image: Array3<f64>, spec: &RenderSpec) -> Result<FormedImage> {
    spec.validate()?;
    let (nz, ny, nx) = image.dim();
    let plane = ny * nx;

    if spec.psf_sigma.iter().any(|&s| s > 0.0) {
        let gaussian = Gaussian::new(spec.psf_sigma, 3.0)?;
        let mut blurred = Array3::<f64>::zeros(image.dim());
        crate::ops::ridge::gaussian_smooth_into(
            image.view(),
            gaussian.kernels(),
            blurred.view_mut(),
        )?;
        image = blurred;
    }

    let mut shifts = vec![[0.0f64; 2]; nz];
    if let Some(j) = &spec.jitter {
        for (z, shift) in shifts.iter_mut().enumerate() {
            let mut rng = Stream::new(spec.seed ^ SALT_JITTER, z as u64);
            for s in shift.iter_mut() {
                *s = j.sigma * gauss(&mut rng);
                if !j.subpixel {
                    *s = s.round();
                }
            }
        }
        let source = image.clone();
        let src = source.as_slice().expect("standard layout");
        image
            .as_slice_mut()
            .expect("standard layout")
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(z, out)| {
                let [dy, dx] = shifts[z];
                let slice = &src[z * plane..(z + 1) * plane];
                let sample = |y: f64, x: f64| {
                    let y = y.clamp(0.0, (ny - 1) as f64);
                    let x = x.clamp(0.0, (nx - 1) as f64);
                    let (y0, x0) = (y.floor() as usize, x.floor() as usize);
                    let (y1, x1) = ((y0 + 1).min(ny - 1), (x0 + 1).min(nx - 1));
                    let (fy, fx) = (y - y0 as f64, x - x0 as f64);
                    let top = slice[y0 * nx + x0] * (1.0 - fx) + slice[y0 * nx + x1] * fx;
                    let bottom = slice[y1 * nx + x0] * (1.0 - fx) + slice[y1 * nx + x1] * fx;
                    if fy == 0.0 {
                        top
                    } else {
                        top * (1.0 - fy) + bottom * fy
                    }
                };
                for y in 0..ny {
                    for x in 0..nx {
                        out[y * nx + x] = sample(y as f64 - dy, x as f64 - dx);
                    }
                }
            });
    }

    let mut gain = vec![1.0f64; nz];
    let mut offset = vec![0.0f64; nz];
    if let Some(d) = &spec.drift {
        let mut rng = Stream::new(spec.seed ^ SALT_DRIFT, 0);
        for z in 1..nz {
            gain[z] = gain[z - 1] + d.gain_step * gauss(&mut rng);
            offset[z] = offset[z - 1] + d.offset_step * gauss(&mut rng);
        }
    }

    // Per slice, per column curtain offset.
    let mut curtain = vec![
        0.0f64;
        if spec.curtaining.is_some() {
            nz * nx
        } else {
            0
        }
    ];
    if let Some(c) = &spec.curtaining {
        let seed = spec.seed ^ SALT_CURTAIN;
        let mut z = 0usize;
        let mut run = 0u64;
        let mut rng = Stream::new(seed, u64::MAX);
        while z < nz {
            let length = c.run[0]
                + ((rng.unit() * (c.run[1] - c.run[0] + 1) as f64) as usize)
                    .min(c.run[1] - c.run[0]);
            for x in 0..nx {
                let striped = unit_from(hash3(seed, x as u64, 0, 1)) < c.column_fraction;
                let value = if striped {
                    c.amplitude * gauss_of(hash3(seed, x as u64, run, 2))
                } else {
                    0.0
                };
                for zz in z..(z + length).min(nz) {
                    curtain[zz * nx + x] = value;
                }
            }
            z += length;
            run += 1;
        }
    }

    let out_spec = &spec.output;
    let mut quantised = vec![0u16; nz * plane];
    quantised
        .par_chunks_mut(plane)
        .zip(image.as_slice().expect("standard layout").par_chunks(plane))
        .enumerate()
        .for_each(|(z, (out, src))| {
            for y in 0..ny {
                let mut ar = 0.0f64;
                for x in 0..nx {
                    let at = y * nx + x;
                    let mut v = gain[z] * src[at] + offset[z];
                    if !curtain.is_empty() {
                        v += curtain[z * nx + x];
                    }
                    if let Some(n) = &spec.shot_noise {
                        let lambda = v.max(0.0) * n.photons_per_unit;
                        let photons = poisson(
                            hash3(spec.seed ^ SALT_SHOT, z as u64, y as u64, x as u64),
                            lambda,
                        );
                        v = photons / n.photons_per_unit
                            + n.read_noise
                                * gauss_of(hash3(
                                    spec.seed ^ SALT_READ,
                                    z as u64,
                                    y as u64,
                                    x as u64,
                                ));
                    }
                    if let Some(s) = &spec.scanline {
                        let e = s.sigma
                            * gauss_of(hash3(spec.seed ^ SALT_SCAN, z as u64, y as u64, x as u64));
                        ar = if x == 0 {
                            e
                        } else {
                            s.correlation * ar + (1.0 - s.correlation * s.correlation).sqrt() * e
                        };
                        v += ar;
                    }
                    let q = (out_spec.offset + out_spec.scale * v).round();
                    out[at] = q.clamp(out_spec.range[0] as f64, out_spec.range[1] as f64) as u16;
                }
            }
        });

    Ok(FormedImage {
        image: Array3::from_shape_vec((nz, ny, nx), quantised).expect("shape matches"),
        shifts,
        gain,
        offset,
    })
}

/// A Poisson sample from one hash: Knuth's product method below 30, the normal
/// approximation above.
fn poisson(hash: u64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda >= 30.0 {
        return (lambda + lambda.sqrt() * gauss_of(hash)).round().max(0.0);
    }
    let limit = (-lambda).exp();
    let mut rng = Stream::new(hash, 0);
    let mut k = 0.0;
    let mut product = rng.unit();
    while product > limit {
        k += 1.0;
        product *= rng.unit();
    }
    k
}

// ================================================================ tests ==

#[cfg(test)]
mod tests {
    use super::*;

    /// A documented example configuration (not a default): anisotropic voxels
    /// (z twice xy), one fibre bundle along x, a few somata, free neurites with
    /// necks and varicosities, glia-like fill. Lengths in xy voxels.
    fn example_tissue(shape: [usize; 3], seed: u64) -> TissueSpec {
        let process = ProcessShape {
            radius: [2.0, 5.0],
            min_radius: 1.5,
            radius_variation: 0.3,
            radius_wavelength: 20.0,
            persistence: 60.0,
            max_turn: 0.15,
            length: [80.0, 300.0],
            branch_rate: 0.01,
            max_branches: 4,
            branch_radius_factor: 0.7,
            branch_spread: [0.6, 1.4],
            neck_rate: 0.01,
            neck_factor: 0.45,
            neck_length: 6.0,
            varicosity_rate: 0.01,
            varicosity_factor: 1.6,
            varicosity_length: 8.0,
        };
        TissueSpec {
            fibres: Some(FibreSpec {
                bundles: 1,
                direction: Some([0.1, 0.2, 1.0]),
                bundle_radius: [25.0, 35.0],
                fibre_radius: [1.8, 4.0],
                gap: 1.0,
                attempts: 400,
                wiggle_amplitude: 4.0,
                wiggle_wavelength: 80.0,
                own_wiggle: 0.4,
                radius_variation: 0.15,
            }),
            somata: Some(SomaSpec {
                count: 3,
                radius: [12.0, 18.0],
                elongation: 1.3,
                irregularity: 0.2,
                irregularity_scale: 0.5,
                neurites: [2, 4],
                shape: process.clone(),
            }),
            neurites: Some(NeuriteSpec {
                count: 40,
                enter_from_face: 0.6,
                shape: process,
            }),
            fill: FillSpec {
                object_growth: 1.0,
                seeds: 60,
                min_seed_distance: 3.0,
                seed_spacing: 12.0,
                texture: 3.0,
                texture_scale: 6.0,
            },
            collision_tolerance: 0.1,
            min_fragment_voxels: 20,
            ..TissueSpec::empty(shape, seed, [2.0, 1.0, 1.0])
        }
    }

    /// A documented example of the image formation model: dark membranes and
    /// organelles on lighter, varied cytoplasm; mild anisotropic blur; FIB-SEM
    /// style nuisances. Intensities in arbitrary signal units.
    fn example_render(seed: u64) -> RenderSpec {
        RenderSpec {
            seed,
            polarity: MembranePolarity::AsConfigured,
            membrane_thickness: 2.0,
            membrane_intensity: 0.15,
            cytoplasm: [0.55, 0.85],
            texture: 0.08,
            texture_scale: 4.0,
            extracellular: 0.7,
            mitochondria: Some(MitochondriaSpec {
                density: 2e-4,
                length: [6.0, 16.0],
                radius: [1.0, 2.5],
                intensity: 0.25,
            }),
            vesicles: Some(VesicleSpec {
                density: 1e-3,
                radius: [0.7, 1.5],
                intensity: 0.3,
            }),
            myelin: Some(MyelinSpec {
                fraction: 0.3,
                thickness: 1.5,
                intensity: 0.1,
            }),
            psf_sigma: [0.5, 0.8, 0.8],
            jitter: Some(JitterSpec {
                sigma: 1.0,
                subpixel: true,
            }),
            drift: Some(DriftSpec {
                gain_step: 0.01,
                offset_step: 0.005,
            }),
            curtaining: Some(CurtainSpec {
                amplitude: 0.03,
                column_fraction: 0.1,
                run: [5, 20],
            }),
            shot_noise: Some(ShotNoiseSpec {
                photons_per_unit: 200.0,
                read_noise: 0.02,
            }),
            scanline: Some(ScanlineSpec {
                sigma: 0.02,
                correlation: 0.7,
            }),
            output: OutputSpec {
                offset: 4000.0,
                scale: 40000.0,
                range: [0, 65535],
            },
        }
    }

    fn components_per_label(labels: &Array3<u32>) -> HashMap<u32, usize> {
        let shape = labels.dim();
        let mut seen = Array3::<bool>::from_elem(shape, false);
        let mut counts: HashMap<u32, usize> = HashMap::new();
        let dims = [shape.0, shape.1, shape.2];
        for ((z, y, x), &l) in labels.indexed_iter() {
            if l == 0 || seen[[z, y, x]] {
                continue;
            }
            *counts.entry(l).or_default() += 1;
            let mut stack = vec![[z, y, x]];
            seen[[z, y, x]] = true;
            while let Some(at) = stack.pop() {
                for a in 0..3 {
                    for step in [-1i64, 1] {
                        let c = at[a] as i64 + step;
                        if c < 0 || c >= dims[a] as i64 {
                            continue;
                        }
                        let mut to = at;
                        to[a] = c as usize;
                        if !seen[to] && labels[to] == l {
                            seen[to] = true;
                            stack.push(to);
                        }
                    }
                }
            }
        }
        counts
    }

    fn small() -> TissueTruth {
        generate_tissue(&example_tissue([24, 96, 96], 3)).unwrap()
    }

    #[test]
    fn the_partition_is_dense_and_every_label_is_one_component() {
        let truth = small();
        assert!(
            truth.labels.iter().all(|&l| l != 0),
            "no ECS requested, so no zeros"
        );
        let n = truth.label_count();
        assert!(n > 20, "{n} labels");
        let counts = components_per_label(&truth.labels);
        assert_eq!(counts.len(), n, "labels are exactly 1..=n");
        for (label, c) in counts {
            assert_eq!(c, 1, "label {label} has {c} components");
            assert!(label as usize <= n);
        }
        for kind in [
            LabelKind::Fibre,
            LabelKind::Soma,
            LabelKind::Neurite,
            LabelKind::Fill,
        ] {
            assert!(truth.kinds.contains(&kind), "{kind:?} present");
        }
    }

    #[test]
    fn extracellular_space_adds_zeros_and_keeps_labels_connected() {
        let mut spec = example_tissue([24, 96, 96], 5);
        spec.ecs = Some(EcsSpec {
            width: 2.0,
            fraction: 0.5,
        });
        let truth = generate_tissue(&spec).unwrap();
        let zeros = truth.labels.iter().filter(|&&l| l == 0).count();
        let total = truth.labels.len();
        assert!(zeros > 0 && zeros < total / 3, "{zeros} of {total}");
        for (label, c) in components_per_label(&truth.labels) {
            assert_eq!(c, 1, "label {label}");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_truth() {
        let a = small();
        let b = small();
        assert_eq!(a, b);
        let c = generate_tissue(&example_tissue([24, 96, 96], 4)).unwrap();
        assert_ne!(a.labels, c.labels);
    }

    #[test]
    fn skeleton_nodes_lie_inside_their_label() {
        let truth = small();
        assert!(!truth.skeletons.is_empty());
        let mut long = 0;
        for s in &truth.skeletons {
            assert_ne!(truth.kinds[s.label as usize], LabelKind::Fill);
            for n in &s.nodes {
                let at = [n[0] as usize, n[1] as usize, n[2] as usize];
                assert_eq!(truth.labels[at], s.label);
            }
            for e in &s.edges {
                assert!((e[0] as usize) < s.nodes.len() && (e[1] as usize) < s.nodes.len());
            }
            if s.nodes.len() > 30 {
                long += 1;
            }
        }
        assert!(long > 5, "long processes survive intact: {long}");
    }

    #[test]
    fn membranes_render_darker_than_interiors() {
        let truth = small();
        let image = render_tissue(
            truth.labels.view(),
            truth.spacing,
            truth.hints(),
            &example_render(1),
        )
        .unwrap();
        let mut boundary = Array3::<bool>::from_elem(truth.labels.dim(), false);
        crate::ops::boundaries::find_boundaries_into(truth.labels.view(), boundary.view_mut())
            .unwrap();
        let interior = distance_transform(
            boundary.mapv(|b| !b).view(),
            &distance_params(truth.spacing),
        )
        .unwrap();
        let (mut on, mut n_on, mut off, mut n_off) = (0.0, 0usize, 0.0, 0usize);
        for ((at, &b), &d) in boundary.indexed_iter().zip(interior.iter()) {
            let v = image.image[at] as f64;
            if b {
                on += v;
                n_on += 1;
            } else if d >= 3.0 {
                off += v;
                n_off += 1;
            }
        }
        let (on, off) = (on / n_on as f64, off / n_off as f64);
        assert!(on + 2000.0 < off, "boundary {on} vs interior {off}");

        // Inverted polarity flips the relation.
        let mut inverted = example_render(1);
        inverted.polarity = MembranePolarity::Inverted;
        let image =
            render_tissue(truth.labels.view(), truth.spacing, truth.hints(), &inverted).unwrap();
        let mean = |want: bool| {
            let (s, c) = boundary
                .iter()
                .zip(image.image.iter())
                .filter(|(&b, _)| b == want)
                .fold((0.0, 0usize), |(s, c), (_, &v)| (s + v as f64, c + 1));
            s / c as f64
        };
        assert!(mean(true) > mean(false));
    }

    #[test]
    fn organelles_stay_inside_their_cell() {
        let truth = small();
        let mut spec = example_render(2);
        spec.mitochondria.as_mut().unwrap().density = 2e-3;
        spec.mitochondria.as_mut().unwrap().intensity = -7.0;
        spec.vesicles.as_mut().unwrap().intensity = -9.0;
        spec.membrane_thickness = 0.0;
        spec.myelin = None;
        let clean = clean_image(truth.labels.view(), truth.spacing, truth.hints(), &spec).unwrap();
        let mut boundary = Array3::<bool>::from_elem(truth.labels.dim(), false);
        crate::ops::boundaries::find_boundaries_into(truth.labels.view(), boundary.view_mut())
            .unwrap();
        let mut painted = 0;
        for (&v, &b) in clean.iter().zip(boundary.iter()) {
            if v < -6.0 {
                painted += 1;
                assert!(!b, "an organelle reached a boundary voxel");
            }
        }
        assert!(painted > 100, "{painted}");
    }

    fn quiet(seed: u64) -> RenderSpec {
        RenderSpec {
            jitter: None,
            drift: None,
            curtaining: None,
            shot_noise: None,
            scanline: None,
            ..example_render(seed)
        }
    }

    #[test]
    fn curtaining_only_adds_column_offsets() {
        let truth = small();
        let base =
            render_tissue(truth.labels.view(), truth.spacing, truth.hints(), &quiet(3)).unwrap();
        let mut spec = quiet(3);
        spec.curtaining = Some(CurtainSpec {
            amplitude: 0.05,
            column_fraction: 0.2,
            run: [3, 8],
        });
        let striped =
            render_tissue(truth.labels.view(), truth.spacing, truth.hints(), &spec).unwrap();
        let (nz, ny, nx) = base.image.dim();
        let mut moved = 0;
        for z in 0..nz {
            for x in 0..nx {
                let d0 = striped.image[[z, 0, x]] as i64 - base.image[[z, 0, x]] as i64;
                if d0.abs() > 1 {
                    moved += 1;
                }
                for y in 0..ny {
                    let d = striped.image[[z, y, x]] as i64 - base.image[[z, y, x]] as i64;
                    assert!(
                        (d - d0).abs() <= 1,
                        "difference varies along y at z {z} x {x}"
                    );
                }
            }
        }
        assert!(moved > 0);
        assert_eq!(striped.shifts, base.shifts);
    }

    #[test]
    fn integer_jitter_only_translates_slices() {
        let truth = small();
        let mut plain = quiet(4);
        plain.psf_sigma = [0.0; 3];
        let base =
            render_tissue(truth.labels.view(), truth.spacing, truth.hints(), &plain).unwrap();
        let mut spec = plain.clone();
        spec.jitter = Some(JitterSpec {
            sigma: 2.0,
            subpixel: false,
        });
        let jittered =
            render_tissue(truth.labels.view(), truth.spacing, truth.hints(), &spec).unwrap();
        let (nz, ny, nx) = base.image.dim();
        assert!(jittered.shifts.iter().any(|s| s != &[0.0, 0.0]));
        for z in 0..nz {
            let [dy, dx] = jittered.shifts[z];
            for y in 0..ny {
                for x in 0..nx {
                    let sy = (y as f64 - dy).clamp(0.0, (ny - 1) as f64) as usize;
                    let sx = (x as f64 - dx).clamp(0.0, (nx - 1) as f64) as usize;
                    assert_eq!(jittered.image[[z, y, x]], base.image[[z, sy, sx]]);
                }
            }
        }
    }

    #[test]
    fn rendering_is_deterministic() {
        let truth = small();
        let a = render_tissue(
            truth.labels.view(),
            truth.spacing,
            truth.hints(),
            &example_render(9),
        )
        .unwrap();
        let b = render_tissue(
            truth.labels.view(),
            truth.spacing,
            truth.hints(),
            &example_render(9),
        )
        .unwrap();
        assert_eq!(a, b);
        let c = render_tissue(
            truth.labels.view(),
            truth.spacing,
            truth.hints(),
            &example_render(10),
        )
        .unwrap();
        assert_ne!(a.image, c.image);
    }

    #[test]
    fn a_crop_sized_volume_is_quick() {
        let start = std::time::Instant::now();
        let mut spec = example_tissue([64, 256, 256], 11);
        spec.neurites.as_mut().unwrap().count = 600;
        spec.somata.as_mut().unwrap().count = 6;
        spec.fill.seeds = 300;
        spec.ecs = Some(EcsSpec {
            width: 1.0,
            fraction: 0.3,
        });
        let truth = generate_tissue(&spec).unwrap();
        let generated = start.elapsed();
        let image = render_tissue(
            truth.labels.view(),
            truth.spacing,
            truth.hints(),
            &example_render(11),
        )
        .unwrap();
        let total = start.elapsed();
        eprintln!(
            "64x256x256: {} labels, {} skeletons, generate {:?}, render {:?}",
            truth.label_count(),
            truth.skeletons.len(),
            generated,
            total - generated
        );
        assert_eq!(image.image.dim(), (64, 256, 256));
        assert!(generated.as_secs_f64() < 30.0);
    }

    #[test]
    fn invalid_specs_are_refused() {
        let mut spec = example_tissue([8, 8, 8], 1);
        spec.spacing = [0.0, 1.0, 1.0];
        assert!(generate_tissue(&spec).is_err());
        let mut render = example_render(1);
        render.cytoplasm = [0.9, 0.1];
        let labels = Array3::<u32>::ones((4, 4, 4));
        assert!(render_tissue(labels.view(), [1.0; 3], LabelHints::default(), &render).is_err());
    }
}
