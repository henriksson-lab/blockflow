// SPDX-License-Identifier: MIT
//
// Original work for this crate. **The kernel is not**: the flood itself lives
// in `super::scikitimage_watershed`, which is a translation of scikit-image and
// carries BSD-3-Clause. This file is the shell that adapts it, and the two are
// separate files so that the notice travels with the code it belongs to.
//
// A seeded watershed: partition a cost volume into one basin per seed.
// ====================================================================
//
// What it is, and what it is not
// ------------------------------
// Given a cost volume, a set of labelled seeds and a mask, flood outward from
// every seed at once, lowest cost first, and give every masked voxel the label
// of the seed whose flood reached it. Optionally leave a one-voxel line of
// label 0 wherever two floods met ([`Separation::Line`]).
//
// Two ops already here are near neighbours and neither can stand in:
//
// * `reconstruct` is a *reconstruction* — a marker grown under a ceiling — and
//   produces a greyscale volume, not a partition. It has no notion of which
//   marker a voxel came from, which is the entire output here.
// * `components` is an *equivalence relation* over a mask, and a basin boundary
//   is not one. Two voxels of one basin need not be connected inside it: with
//   [`Separation::Line`] the line clears voxels *between* labels, so a basin is
//   routinely cut into several connected pieces. The test
//   `a_basin_is_not_a_connected_component` measures the gap rather than asserting
//   it: on a `32 x 32 x 24` fixture, **138 basins against 383 six-connected
//   components** of the same labelled voxels. That is not a rounding error, so
//   counting components of the labelled output is not a way to count basins and
//   `ops::detect`'s `Moments::count` over them is not a basin size.
//
// The cost volume is the caller's
// -------------------------------
// This op floods *up* a cost: low first. A caller who wants basins around
// **bright** structure passes `-intensity`, which is one `voxelwise` map, and a
// caller who wants a distance-transform watershed passes the distance. Building
// the cost here would be one convention baked into an op that has no reason to
// hold one; the mask is the caller's for the same reason, and is one threshold.
//
// **A barrier, and why that is the honest declaration**
// ----------------------------------------------------
// This op declares [`Reach::all`], which makes it a planning barrier: it gets a
// phase of its own with the grid collapsed to a single block
// (`decomposition::is_planning_barrier`, `decomposition::splittable_axes`).
// That is a declared property of the op, not a special case in the planner.
//
// It is declared because **the answer is a function of one global queue's pop
// order over the whole volume**, and three separate things make that so:
//
// 1. Priority is `(cost, age)` where `age` is a global counter incremented once
//    per push. Two voxels of equal cost are ordered by which flood reached the
//    queue first *anywhere in the volume*. A block that started its own counter
//    at zero would order them differently.
// 2. Ties on both keys are resolved by the queue array's internal layout, so
//    even the seeds' relative order — they all carry `age == 0` — depends on how
//    many seeds there are and what order they were inserted in. Cutting the
//    volume changes both.
// 3. A voxel's priority is raised to its source's, so a flood carries the worst
//    cost it has crossed. That history is unbounded in length: a basin can be
//    decided by a barrier hundreds of voxels away, which is a distance no halo
//    is willing to be.
//
// (1) and (3) are properties of the algorithm; (2) is a property of an
// implementation, and it is reproduced rather than replaced because the
// reference this op must agree with bit-for-bit has it. A *different* seeded
// watershed could be decomposable. This one is not, and the test
// `a_blocked_run_is_not_the_whole_volume_answer` runs one by hand and counts the
// voxels that move rather than arguing about it.
//
// What a barrier costs, as arithmetic
// -----------------------------------
// One block spanning the volume means the phase's resident set is the volume,
// and it is worth having the number rather than the adjective. Per voxel of the
// volume, while the phase runs:
//
// | array | width |
// |---|---|
// | cost (the input) | 8 B (`f64`) |
// | seeds (a source level) | 4 B (`u32`) |
// | mask (a source level) | 1 B (`bool`) |
// | labels (the output) | 4 B (`u32`) |
// | the kernel's own copy of the mask | 1 B (`bool`) |
// | | **18 B/voxel** |
//
// Only the last row is this op's; the four above it are what the framework hands
// it, and the kernel **borrows** all four rather than copying them — the cost
// slice and the label slice go in as they are, and the seeds are written into
// the output that is about to be flooded in place. The mask is the one array
// that has to be duplicated, because carving a separating line means clearing
// voxels in it and the level it came from has other readers. A naive shell that
// materialised each of the four would cost 30 B/voxel, which on a 1024^3 volume
// is 13 GiB of pure copy.
//
// **The queue sits on top of that, and it is not the small term it looks like.**
// One queued item is 32 B (`scikitimage_watershed::HEAP_ITEM_BYTES`) and the
// queue is bounded above by six pushes per *masked* voxel, i.e. **192 B per
// masked voxel** — eleven times the dense figure, so anything above about a
// tenth of the volume being masked makes the queue the larger term. It sits well
// below its bound, because it holds a flood **front** rather than a history, and
// how far below is a fact about the data rather than about the algorithm.
// `the_queue_is_a_front_not_a_history` measures it: 20.4% of the bound on a
// volume 82% masked, 5.1% at 47%, 0.04% at 18% — which is 32, 4.6 and 0.02 bytes
// per *volume* voxel respectively.
//
// So the arithmetic a machine is sized on, taking the densely-masked case as the
// one nobody should be surprised by (18 B dense + 32 B queue = **50 B/voxel**)
// and a sparse mask as the one a caller usually has (**~18 B/voxel**):
//
// | volume | sparse mask | 80% masked |
// |---|---|---|
// | 256^3 | 288 MiB | 800 MiB |
// | 512^3 | 2.3 GiB | 6.3 GiB |
// | 1024^3 | 18 GiB | 50 GiB |
// | 2048 x 2048 x 1024 | 72 GiB | 200 GiB |
//
// So the barrier is affordable to about 1024^3 on a large node with a sparse
// mask, and not beyond it. The thing a caller does about that is cut the
// *problem* rather than the op: this op is defined on whatever volume it is
// handed, so a caller with a volume too large for it runs it on a region whose
// faces fall in background — which is a decision about the data that this crate
// has no way to make, and is why there is no parameter here for it.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, SourceInput, SourceInputs};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::scikitimage_watershed::watershed_raveled_reporting_peak;
use super::shapes_agree;

/// What happens where two floods meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Separation {
    /// The meeting voxels are cleared to label 0, leaving a one-voxel line
    /// between basins. Every basin is then separated from every other, and the
    /// labelled voxels are **fewer** than the masked ones.
    Line,
    /// Every masked voxel takes a label. Basins touch, and the boundary between
    /// two of them is wherever the floods happened to meet.
    Adjacent,
}

impl Separation {
    fn carves_a_line(self) -> bool {
        matches!(self, Self::Line)
    }
}

/// Flood `cost` from `seeds`, within `mask`, into `out`.
///
/// * `cost` — lower floods first. A caller wanting basins around bright
///   structure passes the negation.
/// * `seeds` — 0 is "not a seed"; every other value is a label, and a label may
///   occupy more than one voxel.
/// * `mask` — `None` means the whole array. A seed outside the mask is dropped,
///   which is what `skimage`'s `markers * mask` does at the Python layer.
/// * `out` — one label per voxel, 0 where nothing reached.
///
/// The mask is consumed by value inside: [`Separation::Line`] carves the line by
/// clearing mask voxels as it goes, so the caller's copy is untouched.
pub fn seeded_watershed_into(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
    out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    flood(cost, seeds, mask, separation, out).map(|_| ())
}

/// [`seeded_watershed_into`], returning the largest number of items the queue
/// ever held.
///
/// The queue is the whole of this op's memory cost beyond the dense arrays, and
/// it can only be known by running — so it is returned rather than estimated,
/// and the module's arithmetic is checked against it.
pub fn seeded_watershed_into_reporting_peak(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
    out: ArrayViewMut3<'_, u32>,
) -> Result<usize> {
    flood(cost, seeds, mask, separation, out)
}

fn flood(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<usize> {
    shapes_agree(cost.shape(), seeds.shape(), "seeded_watershed_into (seeds)")?;
    if let Some(mask) = mask.as_ref() {
        shapes_agree(cost.shape(), mask.shape(), "seeded_watershed_into (mask)")?;
    }
    shapes_agree(cost.shape(), out.shape(), "seeded_watershed_into (out)")?;

    let shape = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    let line = separation.carves_a_line();

    // **Borrowed where it can be.** The kernel wants one raveled C-order slice
    // per array, which is exactly what a contiguous standard-layout view already
    // is; copying it would be a second dense array on a phase whose whole
    // difficulty is that it holds the volume. A view that is *not* contiguous —
    // a transposed or strided one — has no such slice and is materialised, which
    // is a conversion rather than an avoidable allocation.
    let copied_image;
    let image: &[f64] = match cost.as_slice() {
        Some(slice) => slice,
        None => {
            copied_image = cost.iter().copied().collect::<Vec<f64>>();
            &copied_image
        }
    };

    // `markers * mask`: a seed outside the mask is not a seed. Written straight
    // into the output, which is the array the kernel floods in place.
    match mask.as_ref() {
        Some(mask) => ndarray::Zip::from(&mut out)
            .and(seeds)
            .and(mask)
            .for_each(|slot, &label, &inside| *slot = if inside { label } else { 0 }),
        None => ndarray::Zip::from(&mut out)
            .and(seeds)
            .for_each(|slot, &label| *slot = label),
    }

    // The mask is the one array that must be owned: the kernel clears voxels in
    // it to carve the separating line, and it is handed in as a shared view of a
    // stored level that other readers must still find intact.
    let mut flags: Vec<bool> = match mask.as_ref() {
        Some(mask) => mask.iter().copied().collect(),
        None => vec![true; image.len()],
    };

    match out.as_slice_mut() {
        Some(labels) => Ok(watershed_raveled_reporting_peak(
            image, &shape, &mut flags, labels, line,
        )),
        None => {
            let mut labels: Vec<u32> = out.iter().copied().collect();
            let peak =
                watershed_raveled_reporting_peak(image, &shape, &mut flags, &mut labels, line);
            for (slot, label) in out.iter_mut().zip(labels) {
                *slot = label;
            }
            Ok(peak)
        }
    }
}

/// [`seeded_watershed_into`], allocating its own output.
pub fn seeded_watershed(
    cost: ArrayView3<'_, f64>,
    seeds: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    separation: Separation,
) -> Result<Array3<u32>> {
    let mut out = Array3::zeros(cost.raw_dim());
    seeded_watershed_into(cost, seeds, mask, separation, out.view_mut())?;
    Ok(out)
}

// ---------------------------------------------------------------- the op --

/// [`seeded_watershed_into`] as a phase: a cost volume in, one label per voxel
/// out, seeds and mask read from stored levels.
///
/// **This op is a planning barrier** and says so through [`Reach::all`]. See the
/// module documentation for the argument and for what it costs.
pub struct SeededWatershedOp {
    name: &'static str,
    seeds: usize,
    mask: Option<usize>,
    separation: Separation,
    cost: f64,
}

impl SeededWatershedOp {
    /// `seeds` is the level holding the labels to flood from.
    pub fn new(
        name: &'static str,
        seeds: impl Into<crate::assemble::Level>,
        separation: Separation,
    ) -> Self {
        Self {
            name,
            seeds: seeds.into().into(),
            mask: None,
            separation,
            cost: match separation {
                Separation::Line => WATERSHED_LINE_COST,
                Separation::Adjacent => WATERSHED_COST,
            },
        }
    }

    /// Flood only within `mask`. Without one the whole block is floodable, and
    /// every voxel of it takes a label.
    pub fn within(mut self, mask: impl Into<crate::assemble::Level>) -> Self {
        self.mask = Some(mask.into().into());
        self
    }

    pub fn seed_level(&self) -> usize {
        self.seeds
    }

    pub fn mask_level(&self) -> Option<usize> {
        self.mask
    }

    pub fn separation(&self) -> Separation {
        self.separation
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for SeededWatershedOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The whole axis. Stated as an integer here because the trait requires a
    /// symmetric bound, and as the variant that *means* it in
    /// [`Self::reach_spec`], which is what the planner reads.
    fn reach(&self, _axis: usize, volume_len: usize) -> usize {
        volume_len
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::all()
    }

    /// Both operands over the whole volume, which is the same statement the op
    /// makes about its own input — a flood consults every voxel of all three or
    /// it is a different flood.
    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let mut declared = vec![SourceInput::new(self.seeds, Reach::all())];
        if let Some(mask) = self.mask {
            declared.push(SourceInput::new(mask, Reach::all()));
        }
        declared
    }

    /// `f64` and only `f64`.
    ///
    /// The cost is compared and never combined, so any ordered type would give
    /// the same partition — but the queue is stated in `f64` and widening at the
    /// shell would be a second place the element type is decided. A caller
    /// holding something else maps it, which is the same voxelwise pass that
    /// negates it.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    /// Refuses, and the refusal is the point: a flood with no seeds has no
    /// answer, it has an empty one. See [`BlockOp::apply_with`].
    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(Error::InvalidArgument(format!(
            "{}: the seeds come from level {}, so this op has no answer from its input alone — \
             it would flood nothing and write an empty volume. It is applied through \
             `apply_with`.",
            self.name, self.seeds
        )))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let seeds = sources.get(self.seeds)?;
        if seeds.dtype() != Dtype::U32 {
            return Err(Error::InvalidArgument(format!(
                "{}: the seeds are read from level {}, which holds {}. A seed is a label and is \
                 stored as one; a float would leave 'which values are the same seed' to be \
                 decided somewhere this op cannot see.",
                self.name,
                self.seeds,
                seeds.dtype().numpy_name()
            )));
        }
        let seeds = seeds.view::<u32>()?;

        let mask = match self.mask {
            Some(level) => {
                let mask = sources.get(level)?;
                if mask.dtype() != Dtype::Bool {
                    return Err(Error::InvalidArgument(format!(
                        "{}: the floodable region is read from level {}, which holds {}. It is a \
                         yes-or-no per voxel and is stored as one.",
                        self.name,
                        level,
                        mask.dtype().numpy_name()
                    )));
                }
                Some(mask.view::<bool>()?)
            }
            None => None,
        };

        seeded_watershed_into(
            input.view::<f64>()?,
            seeds,
            mask,
            self.separation,
            out.view_mut::<u32>()?,
        )
    }

    /// **Nothing.** A constant cost block is not a constant answer: what a voxel
    /// gets is decided by the seeds and the mask, neither of which this
    /// declaration can see, and a flat cost is precisely the case where the
    /// partition is decided by tie-breaking alone.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Measured, for [`Separation::Adjacent`]; see `super::COST_MEASUREMENT`.
///
/// **Not taken through `ops::cost::measure`**, which builds one shared `f64`
/// ramp and reads the result as `f64`: this op writes `u32` and needs two
/// operands, so there is no signature to hand it through. Taken instead by
/// [`cost_report`] below, over the same 96 x 64 x 64 volume and against the same
/// `MAP_COST` unit: five runs on a busy machine gave 639, 638, 699, 700 and 1010,
/// against a voxelwise map measured at 0.63-0.75 ns/voxel in the same runs. The
/// figure stored is the middle of that, and the spread is the honest precision —
/// the crate's own note on this table applies with force here, since `statistics`
/// recalibrates from real runs anyway.
///
/// **This is two to three orders of magnitude above everything else in this
/// module's table, and the figure is real rather than a units mistake.** Every
/// voxel costs a push and a pop of a queue holding hundreds of thousands of
/// items, and a sift through that is a chain of cache misses. Two things follow
/// that a planner should have: the figure is per *masked* voxel, so a mask
/// keeping a tenth of the volume costs about a tenth of it — the constant is the
/// all-masked case, which is the one nobody should be surprised by — and no
/// block size makes it cheaper, because the op is a barrier.
pub const WATERSHED_COST: f64 = 680.0;

/// [`WATERSHED_COST`] for [`Separation::Line`], which is **4.7x** dearer and
/// measured rather than derived (2758, 2993, 3030, 3600, 3760 over the same five
/// runs).
///
/// The reason is structural, not incidental. Without a line a voxel is labelled
/// the moment it is pushed and is therefore queued exactly once; with one it is
/// not settled until it *pops*, so it is queued again by every neighbour that
/// pops before it does — up to six times. The extra queue traffic is the whole
/// difference, and it is why `Separation` is a cost input and not only a
/// behaviour switch.
pub const WATERSHED_LINE_COST: f64 = 3200.0;

/// Retake [`WATERSHED_COST`] and [`WATERSHED_LINE_COST`] on this machine. See
/// `super::COST_MEASUREMENT`.
///
/// **Measures its own denominator.** The stored table is in units of one
/// voxelwise map, and that unit is 0.71 ns/voxel on one machine and something
/// else on the next — so this prices a threshold map in the same run, with the
/// same fixture size and the same best-of rule, and divides. What comes out is a
/// ratio, which is what the planner uses and what survives a change of machine.
pub fn cost_report() -> String {
    use std::time::Instant;

    let shape = [96usize, 64, 64];
    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let cost = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (((i * 7 + j * 13 + k * 29) % 251) as f64) / 251.0
    });
    let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
    let mut label = 0u32;
    for i in (4..shape[0]).step_by(16) {
        for j in (4..shape[1]).step_by(16) {
            for k in (4..shape[2]).step_by(16) {
                label += 1;
                seeds[[i, j, k]] = label;
            }
        }
    }

    let mut unit = f64::INFINITY;
    for _ in 0..40 {
        let mut out = Array3::<f64>::zeros(cost.raw_dim());
        let started = Instant::now();
        ndarray::Zip::from(&mut out)
            .and(&cost)
            .for_each(|slot, &value| *slot = if value > 0.5 { 1.0 } else { 0.0 });
        unit = unit.min(started.elapsed().as_secs_f64() / voxels * 1e9);
        std::hint::black_box(&out);
    }

    let mut report = format!("one voxelwise map  {unit:.2} ns/voxel  = MAP_COST = 1.0\n");
    for (what, separation) in [
        ("line", Separation::Line),
        ("adjacent", Separation::Adjacent),
    ] {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut out = Array3::<u32>::zeros(cost.raw_dim());
            let started = Instant::now();
            seeded_watershed_into(cost.view(), seeds.view(), None, separation, out.view_mut())
                .expect("the fixture is well formed");
            best = best.min(started.elapsed().as_secs_f64() / voxels * 1e9);
            std::hint::black_box(&out);
        }
        report.push_str(&format!(
            "seeded watershed, {what:<9} {best:.1} ns/voxel  = {:.1} x MAP_COST\n",
            best / unit
        ));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::super::scikitimage_watershed::reference_case;
    use super::*;

    fn reference_fixture() -> (Array3<f64>, Array3<u32>, Array3<bool>) {
        let shape = (
            reference_case::SHAPE[0],
            reference_case::SHAPE[1],
            reference_case::SHAPE[2],
        );
        let source = Array3::from_shape_vec(shape, reference_case::SOURCE.to_vec())
            .expect("the fixture is 5 x 5 x 3");
        let cost = source.mapv(|value| -value);
        let mask = source.mapv(|value| value > reference_case::THRESHOLD);
        let mut seeds = Array3::<u32>::zeros(source.raw_dim());
        for (label, at) in reference_case::SEEDS {
            seeds[at] = label;
        }
        (cost, seeds, mask)
    }

    /// The whole point, through the shell rather than the kernel.
    #[test]
    fn the_shell_reproduces_the_reference_partition() {
        let (cost, seeds, mask) = reference_fixture();
        let labels = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
        )
        .unwrap();
        assert_eq!(
            labels.iter().copied().collect::<Vec<_>>(),
            reference_case::EXPECTED.to_vec()
        );
    }

    #[test]
    fn a_seed_outside_the_mask_is_not_a_seed() {
        let cost = Array3::from_shape_fn((1, 1, 5), |(_, _, k)| k as f64);
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 0]] = 1;
        seeds[[0, 0, 4]] = 2;
        let mut mask = Array3::from_elem(cost.raw_dim(), true);
        mask[[0, 0, 4]] = false;

        let labels = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Adjacent,
        )
        .unwrap();
        assert_eq!(
            labels.iter().copied().collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 0]
        );
    }

    #[test]
    fn without_a_mask_every_voxel_takes_a_label() {
        let cost = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| ((i * 3 + j * 5 + k) % 7) as f64);
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        seeds[[0, 0, 0]] = 1;
        seeds[[2, 3, 4]] = 2;
        let labels =
            seeded_watershed(cost.view(), seeds.view(), None, Separation::Adjacent).unwrap();
        assert!(labels.iter().all(|&label| label != 0));
    }

    /// The two separations are the same partition up to the line, which is what
    /// makes `Separation` a parameter rather than two ops.
    #[test]
    fn the_line_only_ever_removes_voxels() {
        let (cost, seeds, mask) = reference_fixture();
        let with_line = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
        )
        .unwrap();
        let without = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Adjacent,
        )
        .unwrap();
        let labelled_with = with_line.iter().filter(|&&label| label != 0).count();
        let labelled_without = without.iter().filter(|&&label| label != 0).count();
        assert!(
            labelled_with < labelled_without,
            "the line must clear something, or the parameter says nothing"
        );
        assert_eq!(
            labelled_without,
            mask.iter().filter(|&&inside| inside).count(),
            "without a line every masked voxel is labelled"
        );
    }

    /// The shell borrows a contiguous view and materialises one that is not. A
    /// transposed view is the second case, and it must give the same answer as
    /// the copy the caller could have made itself — otherwise the optimisation
    /// is a second implementation.
    #[test]
    fn a_view_that_has_no_slice_gives_the_same_answer_as_one_that_does() {
        let (cost, seeds, mask) = reference_fixture();
        let expected = seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
        )
        .unwrap();

        // Permuted so no array involved has a standard-layout slice, then read
        // back through the inverse permutation.
        let flip = |axes: [usize; 3]| [axes[2], axes[0], axes[1]];
        let cost = cost.clone().permuted_axes(flip([0, 1, 2]));
        let seeds = seeds.clone().permuted_axes(flip([0, 1, 2]));
        let mask = mask.clone().permuted_axes(flip([0, 1, 2]));
        assert!(
            cost.as_slice().is_none(),
            "the fixture stopped exercising the copy path"
        );
        let mut out = Array3::<u32>::zeros(cost.raw_dim());
        let strided = out.slice_mut(ndarray::s![.., .., ..;1]);
        seeded_watershed_into(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
            strided,
        )
        .unwrap();
        assert_eq!(out.permuted_axes([1, 2, 0]), expected);
    }

    #[test]
    fn a_shape_mismatch_is_refused_by_name() {
        let cost = Array3::<f64>::zeros((2, 2, 2));
        let seeds = Array3::<u32>::zeros((2, 2, 3));
        let failed = seeded_watershed(cost.view(), seeds.view(), None, Separation::Line)
            .expect_err("mismatched shapes");
        assert!(failed.to_string().contains("seeds"), "{failed}");
    }
}
