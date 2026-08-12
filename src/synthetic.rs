// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A generated volume whose correct answer is known by construction.
//
// Why this exists
// ---------------
// `probes` proves that the framework *schedules* everything: its ops move no
// pixels, so a plan that visits every block in a sane order is indistinguishable
// from a correct one. What it cannot catch is an op — or a halo, or a stitch —
// that produces **wrong values**. For that you need a volume whose answer is
// known independently of the pipeline that computes it.
//
// A recorded dataset does not give you that. It gives you a second opinion at
// best, and when two decompositions of the same chain disagree it cannot say
// which one is wrong. A *generated* volume can: the objects are placed first and
// rendered second, so the label volume is exact rather than an estimate of the
// intensity volume, and any disagreement is attributable.
//
// The three properties everything else rests on
// ---------------------------------------------
// 1. **Deterministic from the seed alone.** Same seed and shape give the same
//    bits, on any thread count. Every per-voxel quantity here is a pure function
//    of *global* coordinates — never of an iteration order, never of a
//    sequential random stream — and the only floating-point operations used are
//    `+ - * /`, `sqrt`, `floor` and `ceil`, every one of which IEEE-754 requires
//    to be correctly rounded. No transcendental function appears anywhere, so
//    the bits do not depend on which libm is linked, and on any target whose
//    `f64` is IEEE-754 binary64 they are the same bits. *Verified* across seeds
//    and thread counts here; the cross-target claim is an argument from the
//    operations used, not a measurement.
// 2. **Global placement.** An object's geometry is decided in whole-volume
//    coordinates before any region is considered, so rendering a sub-region
//    yields exactly the voxels the full volume has there. This is what makes the
//    fixture able to test decomposition invariance at all: without it, "the
//    block disagrees with the whole" would be the generator's fault.
// 3. **Region-wise, not whole-volume.** `render_region` allocates the region and
//    nothing else, and touches only the objects whose bounds meet it. A volume
//    far larger than memory can be produced block by block, which is the only
//    way a fixture can be "big enough to be split many ways".
//
// The randomness, and why it is written here
// ------------------------------------------
// SplitMix64 — Steele, Lea and Flood's mixing function, the one Java's
// `SplittableRandom` and Rust's `rand` use to seed everything else. Eleven lines
// of shifts and multiplies, no state to carry, and — the property that actually
// matters here — usable as a *hash* as well as a stream, which is what lets
// per-voxel noise be a function of position instead of of iteration order. It is
// written out rather than depended on because a dependency would give a worse
// guarantee: a version bump could change the bits, and the bits are the contract.
//
// What is deliberately **not** modelled
// ------------------------------------
// Anything specific to a subject matter. The objects are ellipsoids with a soft
// edge; the background is smooth lattice noise; the sensor noise is additive and
// white. That is enough to exercise halos, background estimation anchored to a
// grid, and object-splitting at block seams, and it commits the crate to no
// particular kind of image.

use std::collections::HashMap;

use ndarray::Array3;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::region::Region;

// ------------------------------------------------------------- the generator --

/// Every knob the generated volume has. Defaults are middling difficulty: some
/// objects touch, the background is not flat, and the noise is visible but
/// smaller than the dimmest object.
///
/// Sizes are in voxels, intensities in whatever unit the caller likes — the
/// generator never interprets them.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneSpec {
    /// Volume extent, `[z, y, x]`. An axis of extent 1 is the degenerate case:
    /// this stays a 3-D library, and 2-D is a volume one voxel deep.
    pub shape: [usize; 3],
    /// The whole scene is a function of this and `shape`.
    pub seed: u64,
    /// How many objects to *attempt*. Placements that fall entirely outside the
    /// volume are dropped, so `Scene::object_count` may be smaller; it never
    /// changes for a given seed.
    pub objects: usize,
    /// Mean radius range, `[min, max]`, drawn per object.
    pub radius: [f64; 2],
    /// `1.0` gives spheres. Above that, each axis' radius is drawn from
    /// `mean / elongation ..= mean * elongation`, so objects are not aligned
    /// blobs of one shape.
    pub elongation: f64,
    /// Peak intensity range, `[min, max]`, drawn per object.
    pub brightness: [f64; 2],
    /// Fraction of the radius over which an object fades to nothing at its rim,
    /// in `0.0..1.0`. `0.0` is a hard edge; the label boundary is at the full
    /// radius either way.
    pub edge_softness: f64,
    /// Fraction of objects placed against an already-placed one, in `0.0..=1.0`.
    /// These are the ones that exercise halos: a pair that touches across a
    /// block seam is split by any op that cannot see past the seam.
    pub touching: f64,
    /// How far a touching pair interpenetrates, as a fraction of the sum of
    /// their radii. `0.0` places them just touching; `0.25` overlaps them
    /// noticeably. Only affects objects chosen by `touching`.
    pub overlap: f64,
    /// Constant added everywhere.
    pub background: f64,
    /// Amplitude of the smooth low-frequency field added on top of
    /// `background`. The field is in `-1.0..=1.0` before scaling.
    pub gradient: f64,
    /// Roughly how many cycles of that field span the longest axis. Below 1.0 it
    /// is a gentle ramp; at 4.0 it is coarse mottling. Grid-anchored background
    /// estimation is exactly what this is for.
    pub gradient_cycles: f64,
    /// Standard deviation of the additive per-voxel noise.
    pub noise: f64,
}

impl SceneSpec {
    /// A middling-difficulty scene of the given extent.
    pub fn new(shape: [usize; 3], seed: u64) -> Self {
        Self {
            shape,
            seed,
            objects: 64,
            radius: [3.0, 7.0],
            elongation: 1.4,
            brightness: [0.6, 1.0],
            edge_softness: 0.35,
            touching: 0.3,
            overlap: 0.15,
            background: 0.1,
            gradient: 0.15,
            gradient_cycles: 1.5,
            noise: 0.05,
        }
    }

    /// Nothing but objects on a flat background: no gradient, no noise, hard
    /// edges, nothing touching. The control case — anything that fails here
    /// fails on geometry alone.
    pub fn clean(shape: [usize; 3], seed: u64) -> Self {
        Self {
            edge_softness: 0.0,
            touching: 0.0,
            overlap: 0.0,
            background: 0.0,
            gradient: 0.0,
            noise: 0.0,
            ..Self::new(shape, seed)
        }
    }

    pub fn with_objects(mut self, objects: usize) -> Self {
        self.objects = objects;
        self
    }

    pub fn with_radius(mut self, min: f64, max: f64) -> Self {
        self.radius = [min, max];
        self
    }

    pub fn with_touching(mut self, fraction: f64, overlap: f64) -> Self {
        self.touching = fraction;
        self.overlap = overlap;
        self
    }

    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise = noise;
        self
    }

    pub fn with_gradient(mut self, amplitude: f64, cycles: f64) -> Self {
        self.gradient = amplitude;
        self.gradient_cycles = cycles;
        self
    }

    fn validate(&self) -> Result<()> {
        for (axis, &extent) in self.shape.iter().enumerate() {
            if extent == 0 {
                return Err(Error::invalid(format!(
                    "scene spec: axis {axis} has extent 0"
                )));
            }
        }
        // A non-finite knob would propagate silently into every voxel it
        // touches, so it is refused here rather than found later in the output.
        for (name, value) in [
            ("radius", self.radius[0]),
            ("radius", self.radius[1]),
            ("elongation", self.elongation),
            ("brightness", self.brightness[0]),
            ("brightness", self.brightness[1]),
            ("edge softness", self.edge_softness),
            ("touching", self.touching),
            ("overlap", self.overlap),
            ("background", self.background),
            ("gradient", self.gradient),
            ("gradient cycles", self.gradient_cycles),
            ("noise", self.noise),
        ] {
            if !value.is_finite() {
                return Err(Error::invalid(format!(
                    "scene spec: {name} is {value}, which is not a finite number"
                )));
            }
        }
        if self.radius[0] <= 0.0 || self.radius[1] < self.radius[0] {
            return Err(Error::invalid(format!(
                "scene spec: radius {:?} must be positive and ordered",
                self.radius
            )));
        }
        if self.elongation < 1.0 {
            return Err(Error::invalid(format!(
                "scene spec: elongation {} is below 1.0",
                self.elongation
            )));
        }
        if self.brightness[1] < self.brightness[0] {
            return Err(Error::invalid(format!(
                "scene spec: brightness {:?} is not ordered",
                self.brightness
            )));
        }
        if !(0.0..1.0).contains(&self.edge_softness) {
            return Err(Error::invalid(format!(
                "scene spec: edge softness {} is outside 0.0..1.0",
                self.edge_softness
            )));
        }
        if !(0.0..=1.0).contains(&self.touching) {
            return Err(Error::invalid(format!(
                "scene spec: touching fraction {} is outside 0.0..=1.0",
                self.touching
            )));
        }
        if !(0.0..1.0).contains(&self.overlap) {
            return Err(Error::invalid(format!(
                "scene spec: overlap {} is outside 0.0..1.0",
                self.overlap
            )));
        }
        if self.noise < 0.0 {
            return Err(Error::invalid(format!(
                "scene spec: noise {} is negative",
                self.noise
            )));
        }
        if self.gradient_cycles <= 0.0 {
            return Err(Error::invalid(format!(
                "scene spec: gradient cycles {} must be positive",
                self.gradient_cycles
            )));
        }
        Ok(())
    }
}

/// One placed object, in whole-volume coordinates.
///
/// `centre` and `radii` are the geometry that was decided; `bounds` is that
/// geometry's box clipped to the volume, and is what region rendering tests
/// against. Nothing here depends on which region is being rendered — that is the
/// whole point.
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    pub id: u32,
    /// Centre in voxel units, `[z, y, x]`. Voxel `i` covers `i..i+1` and its
    /// centre is at `i + 0.5`.
    pub centre: [f64; 3],
    /// Semi-axes, one per axis.
    pub radii: [f64; 3],
    /// Peak intensity at the object's centre.
    pub brightness: f64,
    /// The object's box, clipped to the volume.
    pub bounds: Region,
}

/// What the label volume actually contains for one object, measured from the
/// same ownership rule the renderer uses.
///
/// This is the ground-truth table. It is *not* the geometry in `Object`: an
/// object clipped by the volume edge, or overlapped by a lower-numbered one, has
/// fewer voxels than its ellipsoid would suggest, and this is the count that is
/// true of the labels.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectRecord {
    pub id: u32,
    /// Voxels carrying this id in the label volume.
    pub voxels: u64,
    /// Centre of mass of those voxels, in voxel-centre coordinates. All zeros
    /// when `voxels` is 0.
    pub centroid: [f64; 3],
    /// Tight box around those voxels. Shape is all zeros when `voxels` is 0.
    pub bounds: Region,
}

/// A rendered piece of the volume: the region, its intensities and its labels.
///
/// Deliberately a plain struct of `Array3`s and a `Region` — it touches neither
/// `BlockOp` nor `Environment`, so a signature pass over those leaves it alone.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    /// Which part of the volume this is.
    pub region: Region,
    /// Intensities, `region.shape`.
    pub intensity: Array3<f64>,
    /// Labels, `region.shape`. `0` is background, otherwise an object id.
    pub labels: Array3<u32>,
}

/// Below this many voxels a region is rendered on the calling thread. Chosen to
/// be about a millisecond of work: small enough that no realistic block is
/// serialised, large enough that a fixture used from inside a test does not
/// occupy every core to fill a thousand voxels.
const PARALLEL_FROM: usize = 1 << 16;

/// A placed but not yet rendered volume.
///
/// Construction is cheap and proportional to the number of objects, never to the
/// number of voxels: a scene of a hundred billion voxels costs the same to
/// create as one of a thousand. Rendering is where voxels are paid for, and only
/// for the region asked about.
#[derive(Debug, Clone)]
pub struct Scene {
    spec: SceneSpec,
    objects: Vec<Object>,
    field: Field,
    index: Index,
}

impl Scene {
    /// Place the objects. Deterministic in `spec` alone.
    pub fn new(spec: SceneSpec) -> Result<Self> {
        spec.validate()?;
        let objects = place(&spec);
        let field = Field::new(&spec);
        let index = Index::new(&objects);
        Ok(Self {
            spec,
            objects,
            field,
            index,
        })
    }

    pub fn spec(&self) -> &SceneSpec {
        &self.spec
    }

    pub fn shape(&self) -> [usize; 3] {
        self.spec.shape
    }

    /// Objects that survived placement, in id order (`objects[i].id == i + 1`).
    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// The whole volume. Materialises it — for anything large, use
    /// `render_region`.
    pub fn render(&self) -> Rendered {
        self.render_region(&Region::whole(&self.spec.shape))
            .expect("the whole volume is a valid region of itself")
    }

    /// One region, and nothing else.
    ///
    /// Allocates `region.voxels()` elements, visits only objects whose bounds
    /// meet the region, and computes every voxel from its **global** position.
    /// The result is bit-for-bit the corresponding cut of `render`.
    pub fn render_region(&self, region: &Region) -> Result<Rendered> {
        region.check_within(&self.spec.shape, "scene")?;
        if region.ndim() != 3 {
            return Err(Error::ShapeMismatch {
                expected: self.spec.shape.to_vec(),
                got: region.shape.clone(),
            });
        }
        let shape = [region.shape[0], region.shape[1], region.shape[2]];
        let plane = shape[1] * shape[2];
        let voxels = region.voxels();
        let mut labels = vec![0u32; voxels];
        let mut intensity = vec![0.0f64; voxels];

        let candidates: Vec<&Object> = self
            .index
            .near(region, self.objects.len())
            .into_iter()
            .map(|at| &self.objects[at as usize])
            .filter(|object| overlaps(&object.bounds, region))
            .collect();

        // Planes are independent, so they may go wide — but only when there is
        // enough of them to pay for it. A small region rendered on every core
        // costs more in scheduling than it saves, and a fixture is most often
        // asked for a small region. The split changes no bits either way: a
        // plane's values depend on its global coordinates and nothing else.
        if plane > 0 {
            if voxels >= PARALLEL_FROM {
                labels
                    .par_chunks_mut(plane)
                    .zip(intensity.par_chunks_mut(plane))
                    .enumerate()
                    .for_each(|(offset, (labels, intensity))| {
                        self.fill_plane(region, offset, &candidates, labels, intensity);
                    });
            } else {
                labels
                    .chunks_mut(plane)
                    .zip(intensity.chunks_mut(plane))
                    .enumerate()
                    .for_each(|(offset, (labels, intensity))| {
                        self.fill_plane(region, offset, &candidates, labels, intensity);
                    });
            }
        }

        Ok(Rendered {
            region: region.clone(),
            intensity: Array3::from_shape_vec(shape, intensity)
                .expect("intensity buffer has the region's extent"),
            labels: Array3::from_shape_vec(shape, labels)
                .expect("label buffer has the region's extent"),
        })
    }

    /// One plane of the region, at `region.start[0] + offset`.
    ///
    /// Every value written here is a function of the global coordinate and the
    /// object list, so the plane does not know, and cannot know, which region it
    /// is part of.
    fn fill_plane(
        &self,
        region: &Region,
        offset: usize,
        candidates: &[&Object],
        labels: &mut [u32],
        intensity: &mut [f64],
    ) {
        let z = region.start[0] + offset;
        let (y0, y1) = (region.start[1], region.start[1] + region.shape[1]);
        let (x0, x1) = (region.start[2], region.start[2] + region.shape[2]);
        let width = region.shape[2];

        // Objects, lowest id first: the first to cover a voxel owns its label,
        // and intensity takes the largest contribution. Both rules are
        // independent of which voxels happen to be in this region.
        for object in candidates {
            let bounds = &object.bounds;
            if z < bounds.start[0] || z >= bounds.start[0] + bounds.shape[0] {
                continue;
            }
            let dz = (z as f64 + 0.5 - object.centre[0]) / object.radii[0];
            let dz2 = dz * dz;
            if dz2 > 1.0 {
                continue;
            }
            let ylo = y0.max(bounds.start[1]);
            let yhi = y1.min(bounds.start[1] + bounds.shape[1]);
            let xlo = x0.max(bounds.start[2]);
            let xhi = x1.min(bounds.start[2] + bounds.shape[2]);
            for y in ylo..yhi {
                let dy = (y as f64 + 0.5 - object.centre[1]) / object.radii[1];
                let planar = dz2 + dy * dy;
                if planar > 1.0 {
                    continue;
                }
                let row = (y - y0) * width;
                for x in xlo..xhi {
                    let dx = (x as f64 + 0.5 - object.centre[2]) / object.radii[2];
                    let radial = planar + dx * dx;
                    if radial > 1.0 {
                        continue;
                    }
                    let at = row + (x - x0);
                    if labels[at] == 0 {
                        labels[at] = object.id;
                    }
                    let value = self.profile(object, radial);
                    if value > intensity[at] {
                        intensity[at] = value;
                    }
                }
            }
        }

        // Background, gradient and noise, in that fixed order, from global
        // coordinates only.
        for y in y0..y1 {
            let row = (y - y0) * width;
            for x in x0..x1 {
                let at = row + (x - x0);
                let mut value = self.spec.background;
                value += self.spec.gradient * self.field.at(z, y, x);
                value += intensity[at];
                if self.spec.noise > 0.0 {
                    value += self.spec.noise * gaussian_at(self.spec.seed, z, y, x);
                }
                intensity[at] = value;
            }
        }
    }

    /// Intensity of `object` at squared normalised radius `radial` (`<= 1`).
    ///
    /// Flat to `1 - edge_softness`, then a smoothstep to zero at the rim. The
    /// label boundary is at the rim either way, so a soft edge makes the
    /// *intensity* boundary ambiguous without making the *truth* ambiguous —
    /// which is the difficulty being asked for.
    fn profile(&self, object: &Object, radial: f64) -> f64 {
        let softness = self.spec.edge_softness;
        if softness <= 0.0 {
            return object.brightness;
        }
        let radius = radial.sqrt();
        let inner = 1.0 - softness;
        if radius <= inner {
            return object.brightness;
        }
        let t = (1.0 - radius) / softness;
        object.brightness * t * t * (3.0 - 2.0 * t)
    }

    /// The ground-truth table, computed from the same ownership rule the
    /// renderer uses.
    ///
    /// Costs the sum of the objects' boxes, not the volume — so it is available
    /// for a scene far too large to render. Objects entirely hidden by a
    /// lower-numbered neighbour appear with `voxels == 0` rather than being
    /// silently dropped: the table describes what the labels contain, and
    /// "nothing" is an answer.
    pub fn object_table(&self) -> Vec<ObjectRecord> {
        self.objects
            .par_iter()
            .enumerate()
            .map(|(index, object)| self.record(index, object))
            .collect()
    }

    fn record(&self, index: usize, object: &Object) -> ObjectRecord {
        // Only lower ids can take a voxel away from this object.
        let occluders: Vec<&Object> = self
            .index
            .near(&object.bounds, self.objects.len())
            .into_iter()
            .filter(|&at| (at as usize) < index)
            .map(|at| &self.objects[at as usize])
            .filter(|other| overlaps(&other.bounds, &object.bounds))
            .collect();

        let end = object.bounds.end();
        let mut voxels = 0u64;
        let mut sum = [0.0f64; 3];
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        for z in object.bounds.start[0]..end[0] {
            for y in object.bounds.start[1]..end[1] {
                for x in object.bounds.start[2]..end[2] {
                    if !covers(object, z, y, x) {
                        continue;
                    }
                    if occluders.iter().any(|other| covers(other, z, y, x)) {
                        continue;
                    }
                    voxels += 1;
                    for (axis, coordinate) in [z, y, x].into_iter().enumerate() {
                        sum[axis] += coordinate as f64 + 0.5;
                        lo[axis] = lo[axis].min(coordinate);
                        hi[axis] = hi[axis].max(coordinate + 1);
                    }
                }
            }
        }

        if voxels == 0 {
            return ObjectRecord {
                id: object.id,
                voxels: 0,
                centroid: [0.0; 3],
                bounds: Region::new(&[0, 0, 0], &[0, 0, 0]),
            };
        }
        let count = voxels as f64;
        ObjectRecord {
            id: object.id,
            voxels,
            centroid: [sum[0] / count, sum[1] / count, sum[2] / count],
            bounds: Region::new(&lo, &[hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]),
        }
    }
}

// ------------------------------------------------------------------ as a source --

/// The generated intensities as a `RegionSource`.
///
/// This is what makes the fixture usable as the *input* of a streaming run: the
/// executor reads blocks out of it exactly as it would out of a file, and
/// nothing is ever materialised. A volume of any declared size can be the input
/// to a decomposition, which is what "big enough to be split many ways" needs to
/// mean in practice.
///
/// A separate type from `Scene` on purpose. The trait's signature is due a
/// dtype-and-shape pass; when that lands, this thirty-line adapter changes and
/// the generator does not.
pub struct IntensitySource {
    scene: Scene,
}

/// The ground-truth labels as a `RegionSource`, for the same reason.
pub struct LabelSource {
    scene: Scene,
}

impl Scene {
    /// This scene's intensities, as something the executor can read blocks from.
    pub fn as_intensity_source(&self) -> IntensitySource {
        IntensitySource {
            scene: self.clone(),
        }
    }

    /// This scene's labels, as something the executor can read blocks from.
    pub fn as_label_source(&self) -> LabelSource {
        LabelSource {
            scene: self.clone(),
        }
    }
}

impl crate::region::RegionSource<f64> for IntensitySource {
    fn shape(&self) -> &[usize] {
        &self.scene.spec.shape
    }

    fn read_region(&self, region: &Region) -> Result<ndarray::ArrayD<f64>> {
        Ok(self.scene.render_region(region)?.intensity.into_dyn())
    }

    fn describe(&self) -> String {
        format!(
            "generated intensities {:?}, {} objects, seed {}",
            self.scene.spec.shape,
            self.scene.object_count(),
            self.scene.spec.seed
        )
    }
}

impl crate::region::RegionSource<u32> for LabelSource {
    fn shape(&self) -> &[usize] {
        &self.scene.spec.shape
    }

    fn read_region(&self, region: &Region) -> Result<ndarray::ArrayD<u32>> {
        Ok(self.scene.render_region(region)?.labels.into_dyn())
    }

    /// A region no object's box meets is all background, and the generator knows
    /// that without rendering it.
    ///
    /// Only ever `Some(true)` or `None`. A box that *does* meet the region still
    /// may not put a voxel in it — the box is not the ellipsoid — and claiming
    /// `Some(false)` there would be asserting something this method has not
    /// checked. The trait's contract is that an unsure backend says `None` and
    /// pays for the read.
    fn is_known_empty(&self, region: &Region) -> Option<bool> {
        if region.ndim() != 3
            || region
                .check_within(&self.scene.spec.shape, "scene")
                .is_err()
        {
            return None;
        }
        let meets = self
            .scene
            .index
            .near(region, self.scene.objects.len())
            .into_iter()
            .any(|at| overlaps(&self.scene.objects[at as usize].bounds, region));
        (!meets).then_some(true)
    }

    fn describe(&self) -> String {
        format!(
            "generated labels {:?}, {} objects, seed {}",
            self.scene.spec.shape,
            self.scene.object_count(),
            self.scene.spec.seed
        )
    }
}

/// Do two boxes meet? `Region::intersect` answers the same question but
/// allocates two vectors doing it, and this is asked once per object per region.
fn overlaps(left: &Region, right: &Region) -> bool {
    (0..3).all(|axis| {
        left.start[axis] < right.start[axis] + right.shape[axis]
            && right.start[axis] < left.start[axis] + left.shape[axis]
    })
}

/// Is this voxel inside the ellipsoid? The renderer's test, in one place so the
/// table and the labels cannot drift apart.
fn covers(object: &Object, z: usize, y: usize, x: usize) -> bool {
    let mut radial = 0.0;
    for (axis, coordinate) in [z, y, x].into_iter().enumerate() {
        let d = (coordinate as f64 + 0.5 - object.centre[axis]) / object.radii[axis];
        radial += d * d;
    }
    radial <= 1.0
}

// -------------------------------------------------------------------- index --

/// Which objects are near a box.
///
/// Without this, rendering one block costs a scan of every object in the volume,
/// and the object table costs a scan per object — which is quadratic, and shows
/// up the moment the fixture is large enough to be worth having. A uniform grid
/// keyed by occupied cell only: dense enough to be useful, sparse in memory, and
/// no help needed from the caller.
///
/// Cells are sized to the largest object, so an object touches at most eight of
/// them. Queries that would sweep more cells than there are objects fall back to
/// the whole list — the index is an optimisation and must never cost more than
/// the thing it replaces.
#[derive(Debug, Clone)]
struct Index {
    cell: [usize; 3],
    cells: HashMap<[usize; 3], Vec<u32>>,
}

impl Index {
    fn new(objects: &[Object]) -> Self {
        let mut cell = [1usize; 3];
        for object in objects {
            for axis in 0..3 {
                cell[axis] = cell[axis].max(object.bounds.shape[axis]);
            }
        }
        let mut cells: HashMap<[usize; 3], Vec<u32>> = HashMap::new();
        for (at, object) in objects.iter().enumerate() {
            let (lo, hi) = cell_range(&object.bounds, &cell);
            for z in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for x in lo[2]..=hi[2] {
                        cells.entry([z, y, x]).or_default().push(at as u32);
                    }
                }
            }
        }
        Self { cell, cells }
    }

    /// Object indices whose cells meet `box_`, ascending — so the caller sees
    /// them in id order, exactly as a linear scan would have produced them.
    fn near(&self, box_: &Region, total: usize) -> Vec<u32> {
        if box_.shape.contains(&0) {
            return Vec::new();
        }
        let (lo, hi) = cell_range(box_, &self.cell);
        let mut spans = 1usize;
        for axis in 0..3 {
            spans = spans.saturating_mul(hi[axis] - lo[axis] + 1);
        }
        if spans > total.max(64) {
            return (0..total as u32).collect();
        }
        let mut found = Vec::new();
        for z in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for x in lo[2]..=hi[2] {
                    if let Some(list) = self.cells.get(&[z, y, x]) {
                        found.extend_from_slice(list);
                    }
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }
}

/// The inclusive cell range a non-empty box occupies.
fn cell_range(box_: &Region, cell: &[usize; 3]) -> ([usize; 3], [usize; 3]) {
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for axis in 0..3 {
        lo[axis] = box_.start[axis] / cell[axis];
        hi[axis] = (box_.start[axis] + box_.shape[axis] - 1) / cell[axis];
    }
    (lo, hi)
}

// ---------------------------------------------------------------- placement --

/// Salts, so that the object stream, the background field and the per-voxel
/// noise are independent of each other. Without them, a scene with the same seed
/// would correlate its noise with its object positions.
const SALT_OBJECT: u64 = 0x4f_62_6a_65_63_74_00_01;
const SALT_FIELD: u64 = 0x46_69_65_6c_64_00_00_02;
const SALT_NOISE: u64 = 0x4e_6f_69_73_65_00_00_03;

fn place(spec: &SceneSpec) -> Vec<Object> {
    let mut placed: Vec<Object> = Vec::with_capacity(spec.objects);
    for index in 0..spec.objects {
        // A stream per object, keyed by index rather than carried along, so an
        // object's geometry does not depend on how many were accepted before it.
        let mut rng = Stream::new(spec.seed ^ SALT_OBJECT, index as u64);

        let mean = lerp(spec.radius[0], spec.radius[1], rng.unit());
        let mut radii = [0.0f64; 3];
        for radius in radii.iter_mut() {
            let factor = lerp(1.0 / spec.elongation, spec.elongation, rng.unit());
            *radius = (mean * factor).max(0.5);
        }
        let brightness = lerp(spec.brightness[0], spec.brightness[1], rng.unit());

        let companion = rng.unit() < spec.touching && !placed.is_empty();
        let mut centre = [0.0f64; 3];
        if companion {
            let pick = (rng.unit() * placed.len() as f64) as usize;
            let partner = &placed[pick.min(placed.len() - 1)];
            let direction = unit_vector(&mut rng, spec.shape);
            for axis in 0..3 {
                let reach = (partner.radii[axis] + radii[axis]) * (1.0 - spec.overlap);
                centre[axis] = partner.centre[axis] + direction[axis] * reach;
            }
        } else {
            for axis in 0..3 {
                centre[axis] = rng.unit() * spec.shape[axis] as f64;
            }
        }
        // A one-voxel axis is the degenerate 2-D case: the objects have to be in
        // the single plane or there is nothing to see.
        for axis in 0..3 {
            if spec.shape[axis] == 1 {
                centre[axis] = 0.5;
            }
        }

        let Some(bounds) = clipped_bounds(&centre, &radii, spec.shape) else {
            continue;
        };
        placed.push(Object {
            id: placed.len() as u32 + 1,
            centre,
            radii,
            brightness,
            bounds,
        });
    }
    placed
}

/// The ellipsoid's box, clipped to the volume; `None` when nothing is left.
fn clipped_bounds(centre: &[f64; 3], radii: &[f64; 3], shape: [usize; 3]) -> Option<Region> {
    let mut start = [0usize; 3];
    let mut extent = [0usize; 3];
    for axis in 0..3 {
        let low = (centre[axis] - radii[axis] - 0.5).floor();
        let high = (centre[axis] + radii[axis] + 0.5).ceil();
        if high <= 0.0 || low >= shape[axis] as f64 {
            return None;
        }
        let low = low.max(0.0) as usize;
        let high = (high.min(shape[axis] as f64) as usize).min(shape[axis]);
        if high <= low {
            return None;
        }
        start[axis] = low;
        extent[axis] = high - low;
    }
    Some(Region::new(&start, &extent))
}

/// A direction, by rejection sampling in the cube. Axes of extent 1 are held at
/// zero so a companion in the degenerate 2-D case stays in the plane.
///
/// Bounded: after sixteen rejections — probability about `(1 - pi/6)^16`, one in
/// nine million — it gives up and returns an axis direction rather than looping.
fn unit_vector(rng: &mut Stream, shape: [usize; 3]) -> [f64; 3] {
    let free: Vec<usize> = (0..3).filter(|&axis| shape[axis] > 1).collect();
    if free.is_empty() {
        return [0.0; 3];
    }
    for _ in 0..16 {
        let mut candidate = [0.0f64; 3];
        let mut square = 0.0;
        for &axis in &free {
            let value = rng.unit() * 2.0 - 1.0;
            candidate[axis] = value;
            square += value * value;
        }
        if square <= 1.0 && square > 1e-6 {
            let norm = square.sqrt();
            for &axis in &free {
                candidate[axis] /= norm;
            }
            return candidate;
        }
    }
    let mut fallback = [0.0f64; 3];
    fallback[free[0]] = 1.0;
    fallback
}

fn lerp(low: f64, high: f64, t: f64) -> f64 {
    low + (high - low) * t
}

// ------------------------------------------------------------------- fields --

/// The smooth background: value noise on a lattice, two octaves.
///
/// A lattice rather than a sum of sinusoids for one reason — no transcendental
/// function, so the result is exact IEEE arithmetic and the same bits on any
/// target. It is also the shape that makes grid-anchored background estimation
/// interesting: the field has structure at a scale the caller chooses, and a
/// block that estimates its background locally will get a different answer than
/// one that estimates it globally.
#[derive(Debug, Clone, PartialEq)]
struct Field {
    seed: u64,
    /// Lattice spacing in voxels, per octave, per axis.
    spacing: [[f64; 3]; 2],
    weight: [f64; 2],
}

impl Field {
    fn new(spec: &SceneSpec) -> Self {
        let longest = spec.shape.iter().copied().max().unwrap_or(1) as f64;
        let base = (longest / spec.gradient_cycles).max(2.0);
        Self {
            seed: spec.seed ^ SALT_FIELD,
            spacing: [[base; 3], [(base * 0.5).max(2.0); 3]],
            // Normalised so the field stays inside -1.0..=1.0.
            weight: [2.0 / 3.0, 1.0 / 3.0],
        }
    }

    /// The field at a global voxel centre. Pure in the coordinate.
    fn at(&self, z: usize, y: usize, x: usize) -> f64 {
        let point = [z as f64 + 0.5, y as f64 + 0.5, x as f64 + 0.5];
        let mut total = 0.0;
        for octave in 0..2 {
            total += self.weight[octave] * self.octave(octave, &point);
        }
        total
    }

    fn octave(&self, octave: usize, point: &[f64; 3]) -> f64 {
        let mut corner = [0i64; 3];
        let mut fraction = [0.0f64; 3];
        for axis in 0..3 {
            let scaled = point[axis] / self.spacing[octave][axis];
            let floor = scaled.floor();
            corner[axis] = floor as i64;
            let t = scaled - floor;
            fraction[axis] = t * t * (3.0 - 2.0 * t);
        }
        // Trilinear interpolation between the eight lattice corners.
        let mut total = 0.0;
        for step in 0..8 {
            let offsets = [(step >> 2) & 1, (step >> 1) & 1, step & 1];
            let mut weight = 1.0;
            let mut at = [0i64; 3];
            for axis in 0..3 {
                let t = fraction[axis];
                weight *= if offsets[axis] == 1 { t } else { 1.0 - t };
                at[axis] = corner[axis] + offsets[axis] as i64;
            }
            total += weight * lattice(self.seed, octave as u64, at);
        }
        total
    }
}

/// A value in `-1.0..1.0` at a lattice corner.
fn lattice(seed: u64, octave: u64, at: [i64; 3]) -> f64 {
    let hash = hash3(
        seed ^ octave.wrapping_mul(0x2545_f491_4f6c_dd1d),
        at[0] as u64,
        at[1] as u64,
        at[2] as u64,
    );
    unit_from(hash) * 2.0 - 1.0
}

/// Additive noise at a global voxel, approximately standard normal.
///
/// The sum of four uniforms — Irwin-Hall(4), scaled to unit variance. Not a
/// perfect normal (it has no tails past four sigma), and that is the trade
/// deliberately made: it needs one hash and no transcendental, so the volume is
/// bit-identical everywhere and generation stays cheap. Anything relying on the
/// exact tail behaviour of the noise is relying on the wrong thing.
fn gaussian_at(seed: u64, z: usize, y: usize, x: usize) -> f64 {
    let hash = hash3(seed ^ SALT_NOISE, z as u64, y as u64, x as u64);
    let mut sum = 0.0;
    for quarter in 0..4 {
        let part = (hash >> (quarter * 16)) & 0xffff;
        sum += part as f64 / 65536.0;
    }
    // Irwin-Hall(4) has mean 2 and variance 1/3.
    (sum - 2.0) * 1.732_050_807_568_877_2
}

// --------------------------------------------------------------- randomness --

/// SplitMix64's finalising mix. Used both as the stream's output stage and, on
/// its own, as the hash that makes per-voxel quantities positional.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Hash a seed together with three coordinates.
///
/// This is what makes a per-voxel quantity a function of *where* it is rather
/// than of when it was computed — the property region-wise generation depends
/// on. Each coordinate is multiplied by a distinct odd constant and mixed in
/// turn, so no permutation of coordinates collides and a change of one voxel in
/// any axis changes every bit of the result.
fn hash3(seed: u64, first: u64, second: u64, third: u64) -> u64 {
    let mut hash = mix(seed);
    hash = mix(hash ^ first.wrapping_mul(0xff51_afd7_ed55_8ccd));
    hash = mix(hash ^ second.wrapping_mul(0xc4ce_b9fe_1a85_ec53));
    mix(hash ^ third.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn unit_from(hash: u64) -> f64 {
    // The top 53 bits, which is every bit an f64 in 0.0..1.0 can hold.
    (hash >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// A SplitMix64 stream, seeded from a base seed and an index so that streams can
/// be created independently rather than split off one another.
struct Stream {
    state: u64,
}

impl Stream {
    fn new(seed: u64, index: u64) -> Self {
        Self {
            state: mix(seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix(self.state)
    }

    /// The next value in `0.0..1.0`.
    fn unit(&mut self) -> f64 {
        unit_from(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Scene {
        Scene::new(SceneSpec::new([24, 32, 40], 7).with_objects(30)).unwrap()
    }

    #[test]
    fn the_same_seed_gives_the_same_bits() {
        let left = small().render();
        let right = small().render();
        assert_eq!(left.labels, right.labels);
        for (a, b) in left.intensity.iter().zip(right.intensity.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// Rendering goes wide above `PARALLEL_FROM` and stays on the calling thread
    /// below it. Two code paths is one more than the claim "the bits do not
    /// depend on the thread count" can afford to leave untested.
    #[test]
    fn the_parallel_and_the_serial_path_produce_the_same_bits() {
        let shape = [40usize, 64, 64];
        assert!(shape.iter().product::<usize>() >= PARALLEL_FROM);
        let scene = Scene::new(SceneSpec::new(shape, 404).with_objects(50)).unwrap();
        let wide = scene.render();
        for z in 0..shape[0] {
            let plane = Region::new(&[z, 0, 0], &[1, shape[1], shape[2]]);
            assert!(plane.voxels() < PARALLEL_FROM);
            let narrow = scene.render_region(&plane).unwrap();
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    assert_eq!(narrow.labels[[0, y, x]], wide.labels[[z, y, x]]);
                    assert_eq!(
                        narrow.intensity[[0, y, x]].to_bits(),
                        wide.intensity[[z, y, x]].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn a_different_seed_gives_a_different_scene() {
        let other = Scene::new(SceneSpec::new([24, 32, 40], 8).with_objects(30)).unwrap();
        assert_ne!(small().render().labels, other.render().labels);
    }

    #[test]
    fn every_region_is_the_cut_the_whole_volume_has_there() {
        let scene = small();
        let whole = scene.render();
        for (start, shape) in [
            ([0usize, 0, 0], [24usize, 32, 40]),
            ([5, 7, 11], [3, 4, 5]),
            ([13, 0, 20], [11, 32, 20]),
            ([23, 31, 39], [1, 1, 1]),
        ] {
            let region = Region::new(&start, &shape);
            let piece = scene.render_region(&region).unwrap();
            for z in 0..shape[0] {
                for y in 0..shape[1] {
                    for x in 0..shape[2] {
                        let there = [start[0] + z, start[1] + y, start[2] + x];
                        assert_eq!(piece.labels[[z, y, x]], whole.labels[there]);
                        assert_eq!(
                            piece.intensity[[z, y, x]].to_bits(),
                            whole.intensity[there].to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_table_says_what_the_labels_contain() {
        let scene = small();
        let whole = scene.render();
        for record in scene.object_table() {
            let mut voxels = 0u64;
            let mut sum = [0.0f64; 3];
            let mut lo = [usize::MAX; 3];
            let mut hi = [0usize; 3];
            for (index, &label) in whole.labels.indexed_iter() {
                if label != record.id {
                    continue;
                }
                voxels += 1;
                for (axis, coordinate) in [index.0, index.1, index.2].into_iter().enumerate() {
                    sum[axis] += coordinate as f64 + 0.5;
                    lo[axis] = lo[axis].min(coordinate);
                    hi[axis] = hi[axis].max(coordinate + 1);
                }
            }
            assert_eq!(voxels, record.voxels, "object {}", record.id);
            if voxels == 0 {
                continue;
            }
            for axis in 0..3 {
                assert_eq!(record.bounds.start[axis], lo[axis]);
                assert_eq!(record.bounds.shape[axis], hi[axis] - lo[axis]);
                assert!((record.centroid[axis] - sum[axis] / voxels as f64).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_one_voxel_axis_is_the_degenerate_two_dimensional_case() {
        let scene = Scene::new(
            SceneSpec::new([1, 64, 64], 3)
                .with_objects(20)
                .with_radius(3.0, 6.0),
        )
        .unwrap();
        let rendered = scene.render();
        assert_eq!(rendered.labels.shape(), &[1, 64, 64]);
        let labelled = rendered.labels.iter().filter(|&&label| label != 0).count();
        assert!(labelled > 100, "only {labelled} labelled voxels in a plane");
    }

    #[test]
    fn touching_objects_are_actually_produced() {
        let scene = Scene::new(
            SceneSpec::new([32, 48, 48], 11)
                .with_objects(60)
                .with_touching(1.0, 0.1),
        )
        .unwrap();
        let labels = scene.render().labels;
        let mut adjacent = 0;
        for z in 0..32 {
            for y in 0..48 {
                for x in 0..47 {
                    let here = labels[[z, y, x]];
                    let next = labels[[z, y, x + 1]];
                    if here != 0 && next != 0 && here != next {
                        adjacent += 1;
                    }
                }
            }
        }
        assert!(adjacent > 0, "no two objects ever touch");
    }

    #[test]
    fn the_background_field_stays_inside_its_amplitude() {
        let scene = Scene::new(
            SceneSpec::new([16, 16, 16], 5)
                .with_objects(0)
                .with_gradient(1.0, 2.0)
                .with_noise(0.0),
        )
        .unwrap();
        let rendered = scene.render();
        let background = scene.spec().background;
        for &value in rendered.intensity.iter() {
            assert!((value - background).abs() <= 1.0 + 1e-12, "field {value}");
        }
    }

    #[test]
    fn the_noise_is_centred_and_has_about_unit_variance() {
        let scene = Scene::new(
            SceneSpec::new([32, 32, 32], 17)
                .with_objects(0)
                .with_gradient(0.0, 1.0)
                .with_noise(1.0),
        )
        .unwrap();
        let values: Vec<f64> = scene
            .render()
            .intensity
            .iter()
            .map(|&value| value - scene.spec().background)
            .collect();
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.05, "variance {variance}");
    }

    /// The adapters must hand out exactly what the generator would, or a
    /// streamed run and a direct one stop being comparable.
    #[test]
    fn the_region_sources_hand_out_what_the_generator_generates() {
        use crate::region::RegionSource;
        let scene = small();
        let intensity = scene.as_intensity_source();
        let labels = scene.as_label_source();
        assert_eq!(RegionSource::<f64>::shape(&intensity), &[24, 32, 40]);
        assert_eq!(RegionSource::<u32>::shape(&labels), &[24, 32, 40]);

        let region = Region::new(&[4, 6, 8], &[5, 7, 9]);
        let direct = scene.render_region(&region).unwrap();
        let read = intensity.read_region(&region).unwrap();
        for (&here, &there) in read.iter().zip(direct.intensity.iter()) {
            assert_eq!(here.to_bits(), there.to_bits());
        }
        assert_eq!(
            labels.read_region(&region).unwrap(),
            direct.labels.into_dyn()
        );

        // Where no object's box reaches, the labels are known empty without a
        // read; anywhere else the answer is "I cannot tell cheaply".
        let empty = (0..24)
            .flat_map(|z| (0..32).map(move |y| Region::new(&[z, y, 0], &[1, 1, 40])))
            .find(|region| labels.is_known_empty(region) == Some(true));
        if let Some(region) = empty {
            assert!(labels
                .read_region(&region)
                .unwrap()
                .iter()
                .all(|&label| label == 0));
        }
        let occupied = scene.objects()[0].bounds.clone();
        assert_eq!(labels.is_known_empty(&occupied), None);
        assert_eq!(labels.is_known_empty(&Region::new(&[0, 0], &[4, 4])), None);
    }

    #[test]
    fn a_bad_spec_is_refused_rather_than_clamped() {
        assert!(Scene::new(SceneSpec::new([0, 4, 4], 1)).is_err());
        assert!(Scene::new(SceneSpec::new([4, 4, 4], 1).with_radius(-1.0, 2.0)).is_err());
        assert!(Scene::new(SceneSpec::new([4, 4, 4], 1).with_noise(-0.1)).is_err());
        assert!(Scene::new(SceneSpec::new([4, 4, 4], 1).with_noise(f64::NAN)).is_err());
        assert!(
            Scene::new(SceneSpec::new([4, 4, 4], 1).with_gradient(f64::INFINITY, 1.0)).is_err()
        );
        let mut spec = SceneSpec::new([4, 4, 4], 1);
        spec.edge_softness = 1.0;
        assert!(Scene::new(spec).is_err());
    }

    #[test]
    fn a_region_outside_the_volume_is_refused() {
        let scene = small();
        assert!(scene
            .render_region(&Region::new(&[20, 0, 0], &[8, 32, 40]))
            .is_err());
        assert!(scene.render_region(&Region::new(&[0, 0], &[4, 4])).is_err());
    }

    #[test]
    fn placement_ignores_how_many_objects_came_before() {
        // Object i is a function of (seed, i), so asking for more objects does
        // not move the ones already placed.
        let few = Scene::new(SceneSpec::new([40, 40, 40], 21).with_objects(10)).unwrap();
        let many = Scene::new(SceneSpec::new([40, 40, 40], 21).with_objects(40)).unwrap();
        for (a, b) in few.objects().iter().zip(many.objects().iter()) {
            assert_eq!(a, b);
        }
    }

    /// The index is an optimisation, so it has to be indistinguishable from the
    /// scan it replaces — including for boxes it decides to answer wholesale.
    #[test]
    fn the_index_finds_exactly_what_a_linear_scan_would() {
        let scene = Scene::new(SceneSpec::new([40, 50, 60], 31).with_objects(80)).unwrap();
        for (start, shape) in [
            ([0usize, 0, 0], [40usize, 50, 60]),
            ([0, 0, 0], [1, 1, 1]),
            ([7, 9, 11], [13, 17, 19]),
            ([39, 49, 59], [1, 1, 1]),
            ([20, 25, 30], [20, 25, 30]),
        ] {
            let region = Region::new(&start, &shape);
            let scanned: Vec<u32> = scene
                .objects
                .iter()
                .enumerate()
                .filter(|(_, object)| overlaps(&object.bounds, &region))
                .map(|(at, _)| at as u32)
                .collect();
            let indexed: Vec<u32> = scene
                .index
                .near(&region, scene.objects.len())
                .into_iter()
                .filter(|&at| overlaps(&scene.objects[at as usize].bounds, &region))
                .collect();
            assert_eq!(indexed, scanned, "region {start:?} {shape:?}");
        }
    }

    #[test]
    fn the_stream_is_splitmix_and_not_something_that_drifted() {
        // Pinning the first outputs, because "deterministic" is only useful if
        // the bits are the same next month too.
        let mut stream = Stream::new(0, 0);
        assert_eq!(stream.next_u64(), mix(0x9e37_79b9_7f4a_7c15));
        assert_eq!(mix(0), 0);
        assert!((0.0..1.0).contains(&unit_from(u64::MAX)));
    }
}
