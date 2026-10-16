// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **Grey-level path openings and closings** (Talbot & Appleton 2007), with
//! the robust, gap-tolerant variant (Cokelaer, Talbot & Chanussot 2012).
//!
//! A path opening keeps a voxel at grey level `t` when it lies on some
//! *path* of at least `L` voxels that are all `>= t`, where a path is a
//! sequence in which every step is one of the few offsets of an *adjacency
//! cone*. Because the cone admits a bend at every step, the structuring
//! "element" is the family of all roughly straight, possibly curved, lines of
//! length `L` — exactly what keeps long thin bright structures (membranes in
//! inverted EM) and removes compact blobs and noise of any brightness, which a
//! straight-line opening or a disc cannot do. The closing is the dual (dark
//! structures on a bright ground, as EM membranes come out of the scope).
//!
//! # Cones
//!
//! * [`PathCones::Planar`] — the four standard 2-D cones (Talbot & Appleton)
//!   in the plane of axes 1 and 2, applied independently in every axis-0
//!   slice: along axis 1, along axis 2, and the two diagonals, each with three
//!   successor offsets.
//! * [`PathCones::Volume`] — seven 3-D cones: one per axis (nine successors
//!   `+e_a` plus any `{-1,0,1}` on the other two axes) and one per body
//!   diagonal (the seven non-zero offsets whose components lie in `{0, s_a}`
//!   for a sign vector `s`). This is the 7-orientation set of Cokelaer et al.
//!
//! The result is the supremum over cones (infimum for the closing).
//!
//! # Robust paths
//!
//! With `gaps = G > 0` a path may cross up to `G` consecutive voxels *below*
//! the level; those voxels do not count towards `L` and are not themselves
//! kept. A path still starts and ends on a voxel at or above the level.
//!
//! # Algorithm
//!
//! Threshold decomposition done incrementally, as Talbot & Appleton do: levels
//! are visited in increasing order; each voxel carries the longest path ending
//! at it and starting from it (per trailing-gap count when `G > 0`), capped at
//! `L`; a voxel whose path length drops below `L` is removed with the previous
//! level as its output, and its removal is propagated only to the voxels whose
//! lengths it can change. Removing a voxel that is on no path of length `L`
//! never shortens a voxel that is, so the set at each level is the opening of
//! the previous one — the work is proportional to the changes, not to
//! `levels x voxels`. A voxel that is on no length-`L` path even at the lowest
//! level (only possible when the volume is shorter than `L` along every cone)
//! gets the element type's bottom (top for the closing).
//!
//! # Reach
//!
//! A path through `p` with `L` counted voxels and at most `G` gaps between any
//! two spans at most `(L - 1)(G + 1)` steps, each moving at most one voxel per
//! axis; every longer path contains such a sub-path through `p`. So the
//! operator is **local** with reach `(L - 1)(G + 1)` on the cone axes (zero on
//! axis 0 for planar cones), and [`PathOpeningOp`] is an exact stencil.

use std::collections::VecDeque;

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::reach::Reach;
use crate::voxels::{VoxelElement, Voxels};

use super::shapes_agree;

/// Which family of adjacency cones paths are drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathCones {
    /// The four 2-D cones in the axis-1/axis-2 plane, per axis-0 slice.
    Planar,
    /// The seven 3-D cones (three axial, four body-diagonal).
    Volume,
}

/// Opening (bright structures kept) or closing (dark structures kept).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathOperator {
    Opening,
    Closing,
}

/// A path operator, validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PathParams {
    length: usize,
    gaps: usize,
    cones: PathCones,
    operator: PathOperator,
}

impl PathParams {
    /// A path opening keeping paths of at least `length` voxels. `length = 1`
    /// is the identity.
    pub fn opening(length: usize, cones: PathCones) -> Result<Self> {
        Self::new(length, 0, cones, PathOperator::Opening)
    }

    /// The dual closing.
    pub fn closing(length: usize, cones: PathCones) -> Result<Self> {
        Self::new(length, 0, cones, PathOperator::Closing)
    }

    pub fn new(
        length: usize,
        gaps: usize,
        cones: PathCones,
        operator: PathOperator,
    ) -> Result<Self> {
        if length == 0 {
            return Err(Error::InvalidArgument(
                "path opening: the path length counts voxels and must be at least 1".to_string(),
            ));
        }
        if length > u32::MAX as usize / 2 {
            return Err(Error::InvalidArgument(format!(
                "path opening: a path length of {length} is beyond what the length counters hold"
            )));
        }
        Ok(Self {
            length,
            gaps,
            cones,
            operator,
        })
    }

    /// Tolerate up to `gaps` consecutive voxels below the level (robust path).
    pub fn with_gaps(mut self, gaps: usize) -> Self {
        self.gaps = gaps;
        self
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn gaps(&self) -> usize {
        self.gaps
    }

    pub fn cones(&self) -> PathCones {
        self.cones
    }

    pub fn operator(&self) -> PathOperator {
        self.operator
    }

    /// `(L - 1)(G + 1)` on the axes the cones move along; see the module header.
    pub fn reach(&self, axis: usize) -> usize {
        if axis == 0 && self.cones == PathCones::Planar {
            0
        } else {
            (self.length - 1) * (self.gaps + 1)
        }
    }
}

/// One adjacency cone: successor offsets, and a lexicographic sweep order in
/// which every successor comes after its predecessor.
#[derive(Debug, Clone)]
struct Cone {
    steps: Vec<[isize; 3]>,
    /// Axes, most significant first.
    order: [usize; 3],
    /// Direction of the sweep along each axis.
    sign: [isize; 3],
}

fn cones(kind: PathCones) -> Vec<Cone> {
    match kind {
        PathCones::Planar => vec![
            Cone {
                steps: vec![[0, 1, -1], [0, 1, 0], [0, 1, 1]],
                order: [1, 2, 0],
                sign: [1, 1, 1],
            },
            Cone {
                steps: vec![[0, -1, 1], [0, 0, 1], [0, 1, 1]],
                order: [2, 1, 0],
                sign: [1, 1, 1],
            },
            Cone {
                steps: vec![[0, 1, 0], [0, 0, 1], [0, 1, 1]],
                order: [1, 2, 0],
                sign: [1, 1, 1],
            },
            Cone {
                steps: vec![[0, 1, 0], [0, 0, -1], [0, 1, -1]],
                order: [1, 2, 0],
                sign: [1, 1, -1],
            },
        ],
        PathCones::Volume => {
            let mut all = Vec::with_capacity(7);
            for axis in 0..3 {
                let others: Vec<usize> = (0..3).filter(|&b| b != axis).collect();
                let mut steps = Vec::with_capacity(9);
                for p in -1..=1 {
                    for q in -1..=1 {
                        let mut step = [0isize; 3];
                        step[axis] = 1;
                        step[others[0]] = p;
                        step[others[1]] = q;
                        steps.push(step);
                    }
                }
                all.push(Cone {
                    steps,
                    order: [axis, others[0], others[1]],
                    sign: [1, 1, 1],
                });
            }
            for sign in [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]] {
                let mut steps = Vec::with_capacity(7);
                for mask in 1..8usize {
                    let mut step = [0isize; 3];
                    for axis in 0..3 {
                        if mask & (1 << axis) != 0 {
                            step[axis] = sign[axis];
                        }
                    }
                    steps.push(step);
                }
                all.push(Cone {
                    steps,
                    order: [0, 1, 2],
                    sign,
                });
            }
            all
        }
    }
}

/// Grey-level path opening or closing of `input` into `out`.
///
/// Integers and floats; a NaN is refused, since the operator needs a total
/// order on the values it visits.
pub fn path_opening_into<T>(
    input: ArrayView3<'_, T>,
    params: &PathParams,
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement + PartialOrd,
{
    shapes_agree(input.shape(), out.shape(), "path_opening_into")?;
    let dims = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let values: Vec<T> = input.iter().copied().collect();
    if values.is_empty() {
        return Ok(());
    }
    if values.iter().any(|v| v.partial_cmp(v).is_none()) {
        return Err(Error::InvalidArgument(
            "path opening: the input holds NaN, which has no place in the level order".to_string(),
        ));
    }
    let opening = params.operator == PathOperator::Opening;
    // `before(a, b)`: level a is visited before level b.
    let before = |a: &T, b: &T| -> std::cmp::Ordering {
        let ord = a.partial_cmp(b).expect("NaN refused above");
        if opening {
            ord
        } else {
            ord.reverse()
        }
    };
    let mut sorted: Vec<usize> = (0..values.len()).collect();
    sorted.sort_by(|&a, &b| before(&values[a], &values[b]));
    let bottom = T::from_f64(if opening {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    });

    let mut result: Option<Vec<T>> = None;
    for cone in cones(params.cones) {
        let mut engine = Engine::new(dims, &cone, params);
        let single = engine.run(&values, &sorted, bottom);
        result = Some(match result {
            None => single,
            Some(mut best) => {
                for (b, s) in best.iter_mut().zip(single) {
                    if before(b, &s) == std::cmp::Ordering::Less {
                        *b = s;
                    }
                }
                best
            }
        });
    }
    let result = result.expect("every cone family has cones");
    for (slot, value) in out.iter_mut().zip(result) {
        *slot = value;
    }
    Ok(())
}

/// The incremental per-cone state. `down[p * s + g]` is the longest path
/// (counted voxels, capped at `L`) ending at `p` with `g` trailing gaps; `up`
/// the same for paths starting at `p`. `0` means no such path.
struct Engine<'a> {
    dims: [usize; 3],
    cone: &'a Cone,
    length: u32,
    s: usize,
    active: Vec<bool>,
    down: Vec<u32>,
    up: Vec<u32>,
    queue_down: VecDeque<usize>,
    queued_down: Vec<bool>,
    queue_up: VecDeque<usize>,
    queued_up: Vec<bool>,
    pending: Vec<usize>,
    scratch: Vec<u32>,
}

impl<'a> Engine<'a> {
    fn new(dims: [usize; 3], cone: &'a Cone, params: &PathParams) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        let s = params.gaps + 1;
        Self {
            dims,
            cone,
            length: params.length as u32,
            s,
            active: vec![true; n],
            down: vec![0; n * s],
            up: vec![0; n * s],
            queue_down: VecDeque::new(),
            queued_down: vec![false; n],
            queue_up: VecDeque::new(),
            queued_up: vec![false; n],
            pending: Vec::new(),
            scratch: vec![0; s],
        }
    }

    fn coords(&self, p: usize) -> [usize; 3] {
        let plane = self.dims[1] * self.dims[2];
        [
            p / plane,
            (p / self.dims[2]) % self.dims[1],
            p % self.dims[2],
        ]
    }

    fn flat(&self, c: [usize; 3]) -> usize {
        (c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]
    }

    /// The neighbour of `c` at `step` times `direction`, if inside.
    fn neighbour(&self, c: [usize; 3], step: [isize; 3], direction: isize) -> Option<usize> {
        let mut next = [0usize; 3];
        for axis in 0..3 {
            let moved = c[axis] as isize + direction * step[axis];
            if moved < 0 || moved >= self.dims[axis] as isize {
                return None;
            }
            next[axis] = moved as usize;
        }
        Some(self.flat(next))
    }

    /// Recompute `p`'s state in one direction into `scratch`; `forward` reads
    /// predecessors and `down`, otherwise successors and `up`. Returns whether
    /// it changed (and stores it).
    fn recompute(&mut self, p: usize, forward: bool) -> bool {
        let c = self.coords(p);
        let direction = if forward { -1 } else { 1 };
        let s = self.s;
        for slot in self.scratch.iter_mut() {
            *slot = 0;
        }
        let active = self.active[p];
        for &step in &self.cone.steps {
            let Some(q) = self.neighbour(c, step, direction) else {
                continue;
            };
            let field = if forward { &self.down } else { &self.up };
            let state = &field[q * s..(q + 1) * s];
            if active {
                let best = state.iter().copied().max().unwrap_or(0);
                self.scratch[0] = self.scratch[0].max(best);
            } else {
                for g in 1..s {
                    self.scratch[g] = self.scratch[g].max(state[g - 1]);
                }
            }
        }
        if active {
            self.scratch[0] = (self.scratch[0] + 1).min(self.length);
        }
        let field = if forward {
            &mut self.down
        } else {
            &mut self.up
        };
        let stored = &mut field[p * s..(p + 1) * s];
        if stored == self.scratch.as_slice() {
            false
        } else {
            stored.copy_from_slice(&self.scratch);
            true
        }
    }

    fn path_length(&self, p: usize) -> u32 {
        (self.down[p * self.s] + self.up[p * self.s]).saturating_sub(1)
    }

    /// Voxel indices in the cone's topological order.
    fn sweep(&self) -> Vec<usize> {
        let order = self.cone.order;
        let mut out = Vec::with_capacity(self.active.len());
        let at = |axis: usize, index: usize| -> usize {
            if self.cone.sign[axis] > 0 {
                index
            } else {
                self.dims[axis] - 1 - index
            }
        };
        for a in 0..self.dims[order[0]] {
            for b in 0..self.dims[order[1]] {
                for c in 0..self.dims[order[2]] {
                    let mut coords = [0usize; 3];
                    coords[order[0]] = at(order[0], a);
                    coords[order[1]] = at(order[1], b);
                    coords[order[2]] = at(order[2], c);
                    out.push(self.flat(coords));
                }
            }
        }
        out
    }

    fn enqueue(&mut self, p: usize, forward: bool) {
        if forward {
            if !self.queued_down[p] {
                self.queued_down[p] = true;
                self.queue_down.push_back(p);
            }
        } else if !self.queued_up[p] {
            self.queued_up[p] = true;
            self.queue_up.push_back(p);
        }
    }

    fn remove<T: Copy>(&mut self, p: usize, value: T, out: &mut [T]) {
        self.active[p] = false;
        out[p] = value;
        self.enqueue(p, true);
        self.enqueue(p, false);
    }

    /// Drain both queues; a change at an active voxel that leaves it short is
    /// recorded for removal.
    fn propagate(&mut self) {
        for forward in [true, false] {
            loop {
                let next = if forward {
                    self.queue_down.pop_front()
                } else {
                    self.queue_up.pop_front()
                };
                let Some(p) = next else { break };
                if forward {
                    self.queued_down[p] = false;
                } else {
                    self.queued_up[p] = false;
                }
                if !self.recompute(p, forward) {
                    continue;
                }
                if self.active[p] && self.path_length(p) < self.length {
                    self.pending.push(p);
                }
                let c = self.coords(p);
                let direction = if forward { 1 } else { -1 };
                for i in 0..self.cone.steps.len() {
                    let step = self.cone.steps[i];
                    if let Some(q) = self.neighbour(c, step, direction) {
                        self.enqueue(q, forward);
                    }
                }
            }
        }
    }

    fn run<T: Copy + PartialEq>(&mut self, values: &[T], sorted: &[usize], bottom: T) -> Vec<T> {
        let n = values.len();
        let mut out = vec![bottom; n];
        let order = self.sweep();
        for &p in &order {
            self.recompute(p, true);
        }
        for &p in order.iter().rev() {
            self.recompute(p, false);
        }
        let short: Vec<usize> = (0..n)
            .filter(|&p| self.path_length(p) < self.length)
            .collect();
        self.pending = short;

        let mut previous = bottom;
        let mut position = 0;
        while position < n {
            let level = values[sorted[position]];
            // Voxels with no length-L path at this level keep the last level
            // they survived at.
            loop {
                let batch = std::mem::take(&mut self.pending);
                if batch.is_empty() {
                    break;
                }
                for p in batch {
                    if self.active[p] && self.path_length(p) < self.length {
                        self.remove(p, previous, &mut out);
                    }
                }
                self.propagate();
            }
            // The survivors at this level drop out at the next one.
            while position < n && values[sorted[position]] == level {
                let p = sorted[position];
                if self.active[p] {
                    self.remove(p, level, &mut out);
                }
                position += 1;
            }
            if position < n {
                self.propagate();
            }
            previous = level;
        }
        out
    }
}

// ------------------------------------------------------------- the op --

/// A path opening or closing as an exact stencil [`BlockOp`]; element type
/// in, same element type out.
#[derive(Debug, Clone, PartialEq)]
pub struct PathOpeningOp {
    name: &'static str,
    params: PathParams,
    cost: f64,
}

impl PathOpeningOp {
    pub fn new(name: &'static str, params: PathParams) -> Self {
        let per_cone: usize = cones(params.cones).iter().map(|c| c.steps.len()).sum();
        let cost = PATH_OPENING_SORT_COST
            + PATH_OPENING_COST_PER_STEP * (per_cone * (params.gaps + 1)) as f64;
        Self { name, params, cost }
    }

    pub fn params(&self) -> &PathParams {
        &self.params
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

/// Placeholder per-voxel costs, in voxelwise-map units. Not measured.
pub const PATH_OPENING_SORT_COST: f64 = 30.0;
pub const PATH_OPENING_COST_PER_STEP: f64 = 6.0;

impl BlockOp for PathOpeningOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.params.reach(axis)
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([
            self.params.reach(0),
            self.params.reach(1),
            self.params.reach(2),
        ])
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        !matches!(dtype, Dtype::Bool | Dtype::F16)
    }

    fn produces(&self, input: Dtype) -> Dtype {
        input
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        fn typed<T: VoxelElement + PartialOrd>(
            input: &Voxels,
            out: &mut Voxels,
            params: &PathParams,
        ) -> Result<()> {
            path_opening_into(input.view::<T>()?, params, out.view_mut::<T>()?)
        }
        match input.dtype() {
            Dtype::U8 => typed::<u8>(input, out, &self.params),
            Dtype::U16 => typed::<u16>(input, out, &self.params),
            Dtype::U32 => typed::<u32>(input, out, &self.params),
            Dtype::U64 => typed::<u64>(input, out, &self.params),
            Dtype::I8 => typed::<i8>(input, out, &self.params),
            Dtype::I16 => typed::<i16>(input, out, &self.params),
            Dtype::I32 => typed::<i32>(input, out, &self.params),
            Dtype::I64 => typed::<i64>(input, out, &self.params),
            Dtype::F32 => typed::<f32>(input, out, &self.params),
            Dtype::F64 => typed::<f64>(input, out, &self.params),
            other => Err(Error::InvalidArgument(format!(
                "{}: a path opening orders grey levels; {other:?} is refused by `accepts`",
                self.name
            ))),
        }
    }

    /// Not declared: a constant maps to itself only where the volume is at
    /// least `L` long along some cone, which a per-block declaration cannot
    /// see.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;

    use crate::decomposition::{Decomposition, PhaseDecomposition};
    use crate::env::ArrayEnvironment;
    use crate::geometry::BlockGrid;
    use crate::op::Chain;
    use crate::strategy::{execute, Hints, Workflow};

    fn open<T: VoxelElement + PartialOrd>(input: &Array3<T>, params: &PathParams) -> Array3<T> {
        let mut out = input.clone();
        path_opening_into(input.view(), params, out.view_mut()).unwrap();
        out
    }

    fn noise(dim: (usize, usize, usize), seed: u64, levels: u64) -> Array3<u16> {
        let mut state = seed | 1;
        Array3::from_shape_fn(dim, |_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % levels) as u16
        })
    }

    /// The definition, by brute force: for every level, the binary robust path
    /// lengths over `{f >= t}` by one forward and one backward sweep per cone.
    fn reference(input: &Array3<u16>, params: &PathParams) -> Array3<u16> {
        let dims = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let values: Vec<u16> = input.iter().copied().collect();
        let mut levels = values.clone();
        levels.sort();
        levels.dedup();
        let mut out = vec![0u16; values.len()];
        for &t in &levels {
            for cone in cones(params.cones) {
                let mut engine = Engine::new(dims, &cone, params);
                for p in 0..values.len() {
                    engine.active[p] = values[p] >= t;
                }
                let order = engine.sweep();
                for &p in &order {
                    engine.recompute(p, true);
                }
                for &p in order.iter().rev() {
                    engine.recompute(p, false);
                }
                for p in 0..values.len() {
                    if engine.active[p] && engine.path_length(p) >= engine.length {
                        out[p] = out[p].max(t);
                    }
                }
            }
        }
        Array3::from_shape_vec(input.raw_dim(), out).unwrap()
    }

    #[test]
    fn every_cone_sweeps_predecessors_first() {
        for kind in [PathCones::Planar, PathCones::Volume] {
            for cone in cones(kind) {
                for step in &cone.steps {
                    let adjusted: Vec<isize> = cone
                        .order
                        .iter()
                        .map(|&axis| step[axis] * cone.sign[axis])
                        .collect();
                    let first = adjusted.iter().find(|&&v| v != 0).copied();
                    assert_eq!(first.map(|v| v > 0), Some(true), "{kind:?} {step:?}");
                }
            }
        }
        assert_eq!(cones(PathCones::Volume).len(), 7);
    }

    #[test]
    fn a_long_line_survives_and_short_segments_and_noise_do_not() {
        let mut image = Array3::<u16>::zeros((1, 20, 40));
        // a long, gently wandering bright line
        for k in 2..36 {
            image[[0, 5 + k / 12, k]] = 100;
        }
        // a short segment and isolated specks, just as bright
        for k in 5..10 {
            image[[0, 15, k]] = 100;
        }
        image[[0, 10, 20]] = 100;
        image[[0, 18, 30]] = 90;
        let params = PathParams::opening(15, PathCones::Planar).unwrap();
        let out = open(&image, &params);
        for k in 2..36 {
            assert_eq!(out[[0, 5 + k / 12, k]], 100, "line at {k}");
        }
        for k in 5..10 {
            assert_eq!(out[[0, 15, k]], 0, "segment at {k}");
        }
        assert_eq!(out[[0, 10, 20]], 0);
        assert_eq!(out[[0, 18, 30]], 0);
        assert!(out.iter().zip(image.iter()).all(|(o, i)| o <= i));
    }

    #[test]
    fn the_incremental_algorithm_matches_the_threshold_definition() {
        for (seed, gaps, kind, length) in [
            (1u64, 0usize, PathCones::Planar, 4usize),
            (2, 1, PathCones::Planar, 5),
            (3, 2, PathCones::Planar, 4),
            (4, 0, PathCones::Volume, 3),
            (5, 1, PathCones::Volume, 4),
        ] {
            let input = noise((3, 9, 11), seed, 6);
            let params = PathParams::opening(length, kind).unwrap().with_gaps(gaps);
            assert_eq!(
                open(&input, &params),
                reference(&input, &params),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn openings_are_anti_extensive_and_idempotent_and_closings_dual() {
        let input = noise((2, 12, 13), 9, 50);
        let params = PathParams::opening(5, PathCones::Planar).unwrap();
        let once = open(&input, &params);
        assert!(once.iter().zip(input.iter()).all(|(o, i)| o <= i));
        assert_eq!(open(&once, &params), once);

        let closing = PathParams::closing(5, PathCones::Planar).unwrap();
        let closed = open(&input, &closing);
        let inverted = input.mapv(|v| 1000 - v);
        let via_opening = open(&inverted, &params).mapv(|v| 1000 - v);
        assert_eq!(closed, via_opening);
    }

    #[test]
    fn a_robust_path_bridges_gaps_and_a_plain_one_does_not() {
        let mut image = Array3::<f32>::zeros((1, 5, 40));
        for k in 0..40 {
            if k % 5 != 4 {
                image[[0, 2, k]] = 1.0;
            }
        }
        let plain = PathParams::opening(20, PathCones::Planar).unwrap();
        assert!(open(&image, &plain).iter().all(|&v| v == 0.0));
        let robust = plain.with_gaps(1);
        let out = open(&image, &robust);
        assert_eq!(out, image, "members kept, gaps not filled");
    }

    #[test]
    fn volume_cones_follow_lines_across_slices() {
        let mut image = Array3::<u8>::zeros((12, 5, 5));
        for i in 0..12 {
            image[[i, 2, 2]] = 7;
        }
        let planar = PathParams::opening(6, PathCones::Planar).unwrap();
        assert!(open(&image, &planar).iter().all(|&v| v == 0));
        let volume = PathParams::opening(6, PathCones::Volume).unwrap();
        assert_eq!(open(&image, &volume), image);
    }

    #[test]
    fn the_op_run_blockwise_equals_the_whole_volume() {
        let volume = [4usize, 30, 34];
        let base = noise((volume[0], volume[1], volume[2]), 21, 40);
        for params in [
            PathParams::opening(6, PathCones::Planar).unwrap(),
            PathParams::opening(4, PathCones::Planar)
                .unwrap()
                .with_gaps(1),
            PathParams::closing(3, PathCones::Volume).unwrap(),
        ] {
            let want = open(&base, &params);
            let chain = Chain::op(PathOpeningOp::new("path-opening", params));
            let workflow = Workflow::new(chain, volume, Dtype::U16);
            let reach = workflow.chain.reach3(&volume);
            for (block, axes) in [(8usize, vec![1]), (11, vec![1, 2]), (2, vec![0])] {
                let slots = workflow.chain.slots();
                let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
                let grid = BlockGrid::along(volume, &axes, block).unwrap();
                let mut plan = Decomposition {
                    volume,
                    dtype: workflow.dtype,
                    phases: vec![PhaseDecomposition::derive(
                        (0..slots.len()).collect(),
                        names,
                        reach,
                        reach,
                        grid,
                    )],
                    chain_reach: reach,
                };
                plan.declare_dtypes(&workflow.chain).unwrap();
                plan.check().unwrap();
                let env =
                    ArrayEnvironment::new(base.clone().into(), plan.n_phases(), [4, 4, 4]).unwrap();
                execute("path", &workflow, &plan, &Hints::default(), &env).unwrap();
                let got = env.output().view::<u16>().unwrap().to_owned();
                assert_eq!(got, want, "{params:?} block {block} axes {axes:?}");
            }
        }
    }
}
