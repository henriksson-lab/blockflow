// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Signed closest distance from sparse points to the boundary of a mask, with
//! the nearest boundary voxel, without a dense distance image.
//!
//! When only a few points need a distance — detections measured against a
//! segmented layer — a dense transform (`ops::distance`) computes and writes a
//! field over the whole volume to read it at those points. This reads the mask
//! once, keeps only its boundary, and searches that.
//!
//! Two phases ([`append_point_distance_phases`]):
//!
//! 1. [`MaskBoundaryOp`]: each block lists its **boundary voxels** — inside
//!    voxels with a 6-neighbour inside the volume that is outside. Outside
//!    the volume is not outside the mask: a layer cut by the volume's edge has
//!    no surface there. Sparse: a sheet keeps a few voxels per column.
//! 2. [`PointDistanceOp`]: each block gathers the boundary lists of the blocks
//!    within `reach_blocks` and, for each point it owns (a row of a points
//!    stream), finds the nearest boundary voxel. Rows out, at the point's
//!    position: `distance` (physical units; **negative inside the mask**),
//!    `nearest_z/y/x` (voxel index) and `status` ([`STATUS_EXACT`],
//!    [`STATUS_CAPPED`], [`STATUS_NONE`]).
//!
//! # Definition
//!
//! The distance is between voxel centres, to the nearest boundary voxel, in
//! the pitch given. A boundary voxel is at distance 0, so for a layer whose
//! first voxel in a column is at `h`, a point straight above at `z` is
//! `(h - z) * pitch_z` away, the convention of a height map's vertical
//! distance. Outside, the nearest boundary voxel is the nearest inside voxel.
//!
//! # Exactness
//!
//! A point's answer is exact when the nearest voxel found is no farther than
//! the gathered region's nearest face that is not the volume's edge; then
//! nothing outside the region can be nearer. Otherwise the status is
//! `STATUS_CAPPED` (the distance found is an upper bound) — or, with no
//! boundary voxel in the region, `STATUS_NONE`, with distance 0 and the point
//! itself as the nearest voxel (tables hold no `NaN`). [`reach_for`] turns a physical cap into the block reach that
//! makes everything up to the cap exact. Ties go to the smallest `(z, y, x)`,
//! so exact answers are the same on every block grid.

use std::sync::Arc;

use crate::assemble::{Phase, PlanBuilder};
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, SeamFold,
    SourceBlocks,
};
use crate::geometry::BlockGrid;
use crate::op::SourceInput;
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::table::{encoded_schema, Column, RowBuilder, Schema, Table, Value};
use crate::Reach;

/// Which voxels of the mask image are inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskTest {
    NonZero,
    /// Exactly this value, compared as `f64` (exact for labels below 2^53).
    Equals(f64),
}

impl MaskTest {
    pub fn inside(self, value: f64) -> bool {
        match self {
            MaskTest::NonZero => value != 0.0,
            MaskTest::Equals(v) => value == v,
        }
    }
}

const BOUNDARY_MAGIC: &[u8; 4] = b"MBd1";

/// Boundary voxels of one block: its core start, then local `u16` triples.
pub fn encode_boundary(start: [usize; 3], voxels: &[[u16; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 32 + voxels.len() * 6);
    out.extend(BOUNDARY_MAGIC);
    for s in start {
        out.extend((s as u64).to_le_bytes());
    }
    out.extend((voxels.len() as u64).to_le_bytes());
    for v in voxels {
        for c in v {
            out.extend(c.to_le_bytes());
        }
    }
    out
}

/// The global voxel indices of an [`encode_boundary`] blob.
pub fn decode_boundary(bytes: &[u8]) -> Result<Vec<[usize; 3]>> {
    let bad = || Error::invalid("point distance: a boundary fragment is malformed");
    if bytes.len() < 36 || &bytes[..4] != BOUNDARY_MAGIC {
        return Err(bad());
    }
    let word = |i: usize| u64::from_le_bytes(bytes[4 + 8 * i..12 + 8 * i].try_into().unwrap());
    let start = [word(0) as usize, word(1) as usize, word(2) as usize];
    let n = word(3) as usize;
    let body = &bytes[36..];
    if body.len() != n * 6 {
        return Err(bad());
    }
    Ok(body
        .chunks_exact(6)
        .map(|c| {
            let u = |i: usize| u16::from_le_bytes([c[2 * i], c[2 * i + 1]]) as usize;
            [start[0] + u(0), start[1] + u(1), start[2] + u(2)]
        })
        .collect())
}

/// Phase 1: every block's boundary voxels (see the module docs).
pub struct MaskBoundaryOp {
    name: &'static str,
    stream: String,
    mask: SourceInput,
    test: MaskTest,
    lifecycle: Lifecycle,
}

impl MaskBoundaryOp {
    /// `mask` names the image; its reach is set here (one voxel).
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        mask: SourceInput,
        test: MaskTest,
        lifecycle: Lifecycle,
    ) -> Self {
        let mut mask = mask;
        mask.reach = Reach::symmetric([1, 1, 1]);
        Self {
            name,
            stream: stream.into(),
            mask,
            test,
            lifecycle,
        }
    }
}

/// The region `core` grown by `by` on every axis, clipped to `volume`.
fn grown(core: &Region, by: [usize; 3], volume: [usize; 3]) -> ([usize; 3], [usize; 3]) {
    let s = core.shape3();
    let lo: [usize; 3] = std::array::from_fn(|a| core.start[a].saturating_sub(by[a]));
    let hi: [usize; 3] = std::array::from_fn(|a| (core.start[a] + s[a] + by[a]).min(volume[a]));
    (lo, hi)
}

impl FragmentOp for MaskBoundaryOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.stream,
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![self.mask.clone()]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "mask boundary reads its mask through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let volume = at.volume();
        let (lo, hi) = grown(at.core, [1, 1, 1], volume);
        let mask = sources.get(self.mask.image.index())?.as_array()?.widened();
        if mask.shape() != [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]] {
            return Err(Error::invalid(
                "mask boundary: the mask block is not the core grown by one voxel",
            ));
        }
        let start: [usize; 3] = std::array::from_fn(|a| at.core.start[a]);
        let shape = at.core.shape3();
        if shape.iter().any(|&s| s > u16::MAX as usize + 1) {
            return Err(Error::invalid(
                "mask boundary: a block edge exceeds 65536 voxels",
            ));
        }
        let inside = |g: [usize; 3]| {
            self.test
                .inside(mask[[g[0] - lo[0], g[1] - lo[1], g[2] - lo[2]]])
        };
        let mut voxels = Vec::new();
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let g = [start[0] + z, start[1] + y, start[2] + x];
                    if !inside(g) {
                        continue;
                    }
                    let mut edge = false;
                    for a in 0..3 {
                        if g[a] > 0 {
                            let mut n = g;
                            n[a] -= 1;
                            edge |= !inside(n);
                        }
                        if g[a] + 1 < volume[a] {
                            let mut n = g;
                            n[a] += 1;
                            edge |= !inside(n);
                        }
                    }
                    if edge {
                        voxels.push([z as u16, y as u16, x as u16]);
                    }
                }
            }
        }
        Ok(BlockOutput::fragment(
            &self.stream,
            encode_boundary(start, &voxels),
        ))
    }
}

/// The rows [`PointDistanceOp`] writes.
pub fn point_distance_schema() -> Result<Schema> {
    Schema::new(vec![
        Column::f64("distance"),
        Column::f64("nearest_z"),
        Column::f64("nearest_y"),
        Column::f64("nearest_x"),
        Column::u64("status"),
    ])
}

/// `status`: the distance is exact.
pub const STATUS_EXACT: u64 = 0;
/// `status`: a voxel outside the gathered region may be nearer; the distance
/// is an upper bound.
pub const STATUS_CAPPED: u64 = 1;
/// `status`: no boundary voxel in the gathered region (distance 0).
pub const STATUS_NONE: u64 = 2;

/// The block reach that makes every distance up to `cap` (physical) exact.
pub fn reach_for(grid: &BlockGrid, pitch: [f64; 3], cap: f64) -> [usize; 3] {
    let block = grid.block();
    let counts = grid.blocks_per_axis();
    std::array::from_fn(|a| {
        let r = (cap / (block[a] as f64 * pitch[a])).ceil().max(0.0) as usize;
        r.min(counts[a].saturating_sub(1))
    })
}

/// One point's answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestBoundary {
    /// Signed: negative inside the mask. 0 when nothing was found.
    pub distance: f64,
    /// Voxel index of the nearest boundary voxel (the point when none).
    pub nearest: [usize; 3],
    /// [`STATUS_EXACT`], [`STATUS_CAPPED`] or [`STATUS_NONE`].
    pub status: u64,
}

/// Boundary voxels bucketed into columns of `cell` × `cell` voxels in y and x,
/// each with its z range, for nearest-voxel queries.
pub struct BoundaryIndex {
    pitch: [f64; 3],
    cell: usize,
    origin: [usize; 2],
    cells: [usize; 2],
    /// CSR over cells: `offsets[c]..offsets[c + 1]` into `points`.
    offsets: Vec<usize>,
    points: Vec<[u32; 3]>,
    z_range: Vec<[u32; 2]>,
}

impl BoundaryIndex {
    /// Voxels within the y/x window `[lo, hi)`.
    pub fn new(
        points: &[[usize; 3]],
        lo: [usize; 2],
        hi: [usize; 2],
        pitch: [f64; 3],
        cell: usize,
    ) -> Result<Self> {
        let cell = cell.max(1);
        let cells = [
            (hi[0] - lo[0]).div_ceil(cell).max(1),
            (hi[1] - lo[1]).div_ceil(cell).max(1),
        ];
        let n = cells[0] * cells[1];
        let of = |p: &[usize; 3]| ((p[1] - lo[0]) / cell) * cells[1] + (p[2] - lo[1]) / cell;
        let mut counts = vec![0usize; n + 1];
        for p in points {
            if p[1] < lo[0] || p[1] >= hi[0] || p[2] < lo[1] || p[2] >= hi[1] {
                return Err(Error::invalid(
                    "point distance: a boundary voxel lies outside the index window",
                ));
            }
            if p.iter().any(|&v| v > u32::MAX as usize) {
                return Err(Error::invalid("point distance: a coordinate exceeds u32"));
            }
            counts[of(p) + 1] += 1;
        }
        for c in 0..n {
            counts[c + 1] += counts[c];
        }
        let offsets = counts.clone();
        let mut fill = counts;
        let mut sorted = vec![[0u32; 3]; points.len()];
        let mut z_range = vec![[u32::MAX, 0u32]; n];
        for p in points {
            let c = of(p);
            sorted[fill[c]] = [p[0] as u32, p[1] as u32, p[2] as u32];
            fill[c] += 1;
            z_range[c][0] = z_range[c][0].min(p[0] as u32);
            z_range[c][1] = z_range[c][1].max(p[0] as u32);
        }
        Ok(Self {
            pitch,
            cell,
            origin: lo,
            cells,
            offsets,
            points: sorted,
            z_range,
        })
    }

    /// The nearest boundary voxel to voxel `p` (unsigned distance, ties to the
    /// smallest `(z, y, x)`), or `None` for an empty index.
    pub fn nearest(&self, p: [usize; 3]) -> Option<(f64, [usize; 3])> {
        let [vz, vy, vx] = self.pitch;
        let pf = [p[0] as f64, p[1] as f64, p[2] as f64];
        let cy = (p[1].saturating_sub(self.origin[0]) / self.cell).min(self.cells[0] - 1) as isize;
        let cx = (p[2].saturating_sub(self.origin[1]) / self.cell).min(self.cells[1] - 1) as isize;
        let step = self.cell as f64 * vy.min(vx);
        let mut best = f64::INFINITY;
        let mut best_q: Option<[u32; 3]> = None;
        let gap = |v: f64, lo: f64, hi: f64| {
            if v < lo {
                lo - v
            } else if v > hi {
                v - hi
            } else {
                0.0
            }
        };
        let max_ring = self.cells[0].max(self.cells[1]) as isize;
        for r in 0..=max_ring {
            if r >= 1 && ((r - 1) as f64 * step).powi(2) > best {
                break;
            }
            let mut visit = |iy: isize, ix: isize| {
                if iy < 0 || ix < 0 || iy >= self.cells[0] as isize || ix >= self.cells[1] as isize
                {
                    return;
                }
                let c = iy as usize * self.cells[1] + ix as usize;
                let (a, b) = (self.offsets[c], self.offsets[c + 1]);
                if a == b {
                    return;
                }
                let y0 = (self.origin[0] + iy as usize * self.cell) as f64;
                let x0 = (self.origin[1] + ix as usize * self.cell) as f64;
                let [z0, z1] = self.z_range[c];
                let lb = (gap(pf[0], z0 as f64, z1 as f64) * vz).powi(2)
                    + (gap(pf[1], y0, y0 + self.cell as f64 - 1.0) * vy).powi(2)
                    + (gap(pf[2], x0, x0 + self.cell as f64 - 1.0) * vx).powi(2);
                if lb > best {
                    return;
                }
                for q in &self.points[a..b] {
                    let d = ((q[0] as f64 - pf[0]) * vz).powi(2)
                        + ((q[1] as f64 - pf[1]) * vy).powi(2)
                        + ((q[2] as f64 - pf[2]) * vx).powi(2);
                    if d < best || (d == best && best_q.is_none_or(|b| *q < b)) {
                        best = d;
                        best_q = Some(*q);
                    }
                }
            };
            if r == 0 {
                visit(cy, cx);
                continue;
            }
            for d in -r..=r {
                visit(cy - r, cx + d);
                visit(cy + r, cx + d);
            }
            for d in -r + 1..r {
                visit(cy + d, cx - r);
                visit(cy + d, cx + r);
            }
        }
        best_q.map(|q| (best.sqrt(), [q[0] as usize, q[1] as usize, q[2] as usize]))
    }
}

/// Phase 2: the nearest boundary voxel of every point (see the module docs).
pub struct PointDistanceOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    points: (String, usize),
    boundary: (String, usize),
    mask: SourceInput,
    test: MaskTest,
    pitch: [f64; 3],
    reach_blocks: [usize; 3],
    schema: Schema,
}

impl PointDistanceOp {
    /// `points` and `boundary` are (stream, phase); `mask` the image the
    /// boundary came from (its reach is set here: zero), read for the sign.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        points: (String, usize),
        boundary: (String, usize),
        mask: SourceInput,
        test: MaskTest,
        pitch: [f64; 3],
        reach_blocks: [usize; 3],
    ) -> Result<Self> {
        if pitch.iter().any(|p| !(p.is_finite() && *p > 0.0)) {
            return Err(Error::invalid(
                "point distance: the pitch must be finite and positive",
            ));
        }
        let mut mask = mask;
        mask.reach = Reach::symmetric([0, 0, 0]);
        Ok(Self {
            name,
            stream: stream.into(),
            lifecycle,
            points,
            boundary,
            mask,
            test,
            pitch,
            reach_blocks,
            schema: point_distance_schema()?,
        })
    }
}

/// Bucket width in voxels of the boundary index.
const CELL: usize = 8;

impl FragmentOp for PointDistanceOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(&self.points.0, self.points.1),
            FragmentInput::own(&self.boundary.0, self.boundary.1).with_reach(self.reach_blocks),
        ]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.stream,
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![self.mask.clone()]
    }

    /// A barrier, so the fragment reach does not widen what the block reads:
    /// it reads its own core of the mask and its neighbours' boundary lists.
    fn barrier(&self) -> bool {
        true
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "point distance reads its mask through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let volume = at.volume();
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        let points = match at.own(&self.points.0) {
            Some(bytes) if !bytes.is_empty() => {
                let mut table = Table::new(volume, encoded_schema(bytes)?)?;
                table.write(at.index, bytes)?;
                table.seal()?;
                let at_of: Vec<[usize; 3]> = table
                    .scan(&Region::whole(&volume))?
                    .map(|row| row.at())
                    .collect();
                at_of
            }
            _ => Vec::new(),
        };
        if points.is_empty() {
            return Ok(BlockOutput::fragment(&self.stream, rows.encode()));
        }
        // the gathered region: blocks within the reach, clipped to the lattice
        let block = at.grid.block();
        let counts = at.grid.blocks_per_axis();
        let first: [usize; 3] =
            std::array::from_fn(|a| at.index[a].saturating_sub(self.reach_blocks[a]));
        let last: [usize; 3] =
            std::array::from_fn(|a| (at.index[a] + self.reach_blocks[a]).min(counts[a] - 1));
        let lo: [usize; 3] = std::array::from_fn(|a| first[a] * block[a]);
        let hi: [usize; 3] = std::array::from_fn(|a| ((last[a] + 1) * block[a]).min(volume[a]));
        let mut boundary = Vec::new();
        for (_, bytes) in at.fragments(&self.boundary.0) {
            boundary.extend(decode_boundary(bytes)?);
        }
        let index =
            BoundaryIndex::new(&boundary, [lo[1], lo[2]], [hi[1], hi[2]], self.pitch, CELL)?;
        let mask = sources.get(self.mask.image.index())?.as_array()?.widened();
        let start: [usize; 3] = std::array::from_fn(|a| at.core.start[a]);
        if mask.shape() != at.core.shape3().as_slice() {
            return Err(Error::invalid(
                "point distance: the mask block is not the core",
            ));
        }
        for p in points {
            let inside = self
                .test
                .inside(mask[[p[0] - start[0], p[1] - start[1], p[2] - start[2]]]);
            // nothing outside the gathered region is nearer than its nearest
            // face that is not the volume's edge
            let mut margin = f64::INFINITY;
            for a in 0..3 {
                if lo[a] > 0 {
                    margin = margin.min((p[a] - lo[a] + 1) as f64 * self.pitch[a]);
                }
                if hi[a] < volume[a] {
                    margin = margin.min((hi[a] - p[a]) as f64 * self.pitch[a]);
                }
            }
            let answer = match index.nearest(p) {
                Some((d, q)) => NearestBoundary {
                    distance: if inside { -d } else { d },
                    nearest: q,
                    status: if d > margin {
                        STATUS_CAPPED
                    } else {
                        STATUS_EXACT
                    },
                },
                None => NearestBoundary {
                    distance: 0.0,
                    nearest: p,
                    status: STATUS_NONE,
                },
            };
            let q = answer.nearest.map(|v| v as f64);
            rows.push(
                p,
                &[
                    Value::F64(answer.distance),
                    Value::F64(q[0]),
                    Value::F64(q[1]),
                    Value::F64(q[2]),
                    Value::U64(answer.status),
                ],
            )?;
        }
        Ok(BlockOutput::fragment(&self.stream, rows.encode()))
    }
}

/// Both phases, on the plan's current grid. `points` is (stream, phase) of a
/// row stream; `mask` the image whose boundary is measured to; `cap` the
/// physical distance up to which answers are guaranteed exact. Returns the
/// phase whose `stream` rows ([`point_distance_schema`]) hold the answers.
#[allow(clippy::too_many_arguments)]
pub fn append_point_distance_phases(
    plan: &mut PlanBuilder,
    stream: &str,
    lifecycle: Lifecycle,
    points: (String, usize),
    mask: SourceInput,
    test: MaskTest,
    pitch: [f64; 3],
    cap: f64,
) -> Result<Phase> {
    let boundary_stream = format!("{stream}.boundary");
    let boundary = plan.fragments(MaskBoundaryOp::new(
        "mask boundary voxels",
        boundary_stream.clone(),
        mask.clone(),
        test,
        Lifecycle::DeleteOnExit,
    ))?;
    let reach_blocks = reach_for(&plan.grid(), pitch, cap);
    plan.fragments(PointDistanceOp::new(
        "point to mask boundary distance",
        stream,
        lifecycle,
        points,
        (boundary_stream, boundary.index()),
        mask,
        test,
        pitch,
        reach_blocks,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::ArrayEnvironment;
    use crate::ops::rows::collect_rows;
    use crate::strategy::{execute_phases, Hints};
    use crate::voxels::Voxels;
    use crate::Dtype;
    use ndarray::Array3;

    /// Emits the given points as rows, each in the block that owns it.
    struct PointsOp {
        points: Vec<[usize; 3]>,
    }

    impl FragmentOp for PointsOp {
        fn name(&self) -> &'static str {
            "test points"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "points",
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            )]
        }
        fn seam_fold(&self) -> Option<SeamFold> {
            Some(SeamFold::PerBlock)
        }
        fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
            let mut rows = RowBuilder::new(Arc::new(Schema::new(vec![Column::u64("i")])?));
            for (i, &p) in self.points.iter().enumerate() {
                if crate::ops::owner_of(at.grid, p) == at.index {
                    rows.push(p, &[Value::U64(i as u64)])?;
                }
            }
            Ok(BlockOutput::fragment("points", rows.encode()))
        }
    }

    fn brute(mask: &Array3<bool>, p: [usize; 3], pitch: [f64; 3]) -> Option<(f64, [usize; 3])> {
        let sh = mask.shape();
        let mut best: Option<(f64, [usize; 3])> = None;
        for ((z, y, x), &m) in mask.indexed_iter() {
            if !m {
                continue;
            }
            let g = [z, y, x];
            let mut edge = false;
            for a in 0..3 {
                for s in [-1isize, 1] {
                    let v = g[a] as isize + s;
                    if v < 0 || v >= sh[a] as isize {
                        continue;
                    }
                    let mut n = g;
                    n[a] = v as usize;
                    edge |= !mask[n];
                }
            }
            if !edge {
                continue;
            }
            let d = (0..3)
                .map(|a| ((g[a] as f64 - p[a] as f64) * pitch[a]).powi(2))
                .sum::<f64>();
            if best.is_none_or(|(b, q)| d < b || (d == b && g < q)) {
                best = Some((d, g));
            }
        }
        best.map(|(d, q)| (d.sqrt(), q))
    }

    fn run(
        mask: &Array3<bool>,
        points: &[[usize; 3]],
        block: [usize; 3],
        pitch: [f64; 3],
        cap: f64,
    ) -> Vec<([usize; 3], f64, [f64; 3], u64)> {
        let volume = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
        let mut plan =
            PlanBuilder::new(volume, Dtype::Bool, BlockGrid::new(volume, block).unwrap());
        let pts = plan
            .fragments(PointsOp {
                points: points.to_vec(),
            })
            .unwrap();
        let phase = append_point_distance_phases(
            &mut plan,
            "dist",
            Lifecycle::Persistent,
            ("points".into(), pts.index()),
            SourceInput::voxelwise(0),
            MaskTest::NonZero,
            pitch,
            cap,
        )
        .unwrap();
        let assembly = plan.finish().unwrap();
        let env = ArrayEnvironment::with_inputs(
            Voxels::from(mask.clone()),
            vec![],
            &assembly.decomposition,
            [4, 4, 4],
        )
        .unwrap();
        execute_phases(
            "pd",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints::default(),
            &env,
            &[],
            &assembly.work(),
        )
        .unwrap();
        let mut rows: Vec<_> = collect_rows(
            &env,
            "dist",
            phase.index(),
            volume,
            point_distance_schema().unwrap(),
        )
        .unwrap()
        .into_iter()
        .map(|r| {
            let f = |i: usize| match r.values[i] {
                Value::F64(v) => v,
                Value::U64(v) => v as f64,
            };
            (r.at, f(0), [f(1), f(2), f(3)], f(4) as u64)
        })
        .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        rows
    }

    fn random_mask(volume: [usize; 3], seed: u64) -> Array3<bool> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as f64 / (1u64 << 31) as f64
        };
        // a wavy layer from a height plus scattered blobs
        let mut m = Array3::from_elem((volume[0], volume[1], volume[2]), false);
        let (a, b) = (next() * 6.28, next() * 6.28);
        for ((z, y, x), v) in m.indexed_iter_mut() {
            let h = volume[0] as f64 * 0.6
                + 2.0 * ((y as f64 * 0.3 + a).sin() + (x as f64 * 0.2 + b).cos());
            *v = z as f64 >= h;
        }
        for _ in 0..6 {
            let c = [
                next() * volume[0] as f64,
                next() * volume[1] as f64,
                next() * volume[2] as f64,
            ];
            for ((z, y, x), v) in m.indexed_iter_mut() {
                let d = (z as f64 - c[0]).powi(2)
                    + (y as f64 - c[1]).powi(2)
                    + (x as f64 - c[2]).powi(2);
                if d < 4.0 {
                    *v = !*v;
                }
            }
        }
        m
    }

    #[test]
    fn matches_brute_force_on_every_grid_and_signs_inside() {
        let volume = [14, 20, 18];
        let pitch = [5.0, 2.16, 2.16];
        let mask = random_mask(volume, 7);
        let mut points = Vec::new();
        for z in (0..volume[0]).step_by(3) {
            for y in (0..volume[1]).step_by(4) {
                for x in (1..volume[2]).step_by(5) {
                    points.push([z, y, x]);
                }
            }
        }
        // a cap beyond the volume: everything exact
        let reference = run(&mask, &points, volume, pitch, 1e6);
        for block in [[4, 8, 6], [14, 5, 7], [3, 20, 18]] {
            assert_eq!(run(&mask, &points, block, pitch, 1e6), reference);
        }
        assert_eq!(reference.len(), points.len());
        for (p, d, q, capped) in &reference {
            let (bd, bq) = brute(&mask, *p, pitch).unwrap();
            let sign = if mask[*p] { -1.0 } else { 1.0 };
            assert!((d - sign * bd).abs() < 1e-9, "{p:?}: {d} vs {}", sign * bd);
            assert_eq!(*q, bq.map(|v| v as f64), "{p:?}");
            assert_eq!(*capped, 0);
        }
    }

    #[test]
    fn a_small_cap_is_exact_up_to_the_cap_and_flags_the_rest() {
        let pitch = [1.0, 1.0, 1.0];
        let mut mask = Array3::from_elem((12, 32, 32), false);
        mask[[6, 2, 2]] = true;
        mask[[6, 15, 23]] = true;
        // cap 3 on 8-voxel blocks: one block of reach
        let points = vec![[6, 3, 3], [6, 2, 9], [6, 30, 30]];
        let rows = run(&mask, &points, [12, 8, 8], pitch, 3.0);
        let status: Vec<u64> = rows.iter().map(|r| r.3).collect();
        // rows in position order: [6, 2, 9], [6, 3, 3], [6, 30, 30] (whose
        // region holds no boundary voxel)
        assert_eq!(status, vec![STATUS_EXACT, STATUS_EXACT, STATUS_NONE]);
        assert_eq!((rows[0].1, rows[1].1), (7.0, 2f64.sqrt()));
        // [6, 2, 17]: its region (x from 8) holds only [6, 15, 23], 14.3 away,
        // farther than the region's lower x face (10): an upper bound
        let rows = run(&mask, &[[6, 2, 17]], [12, 8, 8], pitch, 3.0);
        assert_eq!(rows[0].3, STATUS_CAPPED);
        assert!(rows[0].1 >= brute(&mask, [6, 2, 17], pitch).unwrap().0);
    }

    #[test]
    fn boundary_ignores_the_volume_edge() {
        let mask = Array3::from_elem((4, 4, 4), true);
        let rows = run(&mask, &[[1, 1, 1]], [4, 4, 4], [1.0; 3], 10.0);
        assert_eq!(rows[0].1, 0.0);
        assert_eq!(rows[0].3, STATUS_NONE);
    }
}
