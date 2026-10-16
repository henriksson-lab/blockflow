// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A region adjacency graph with boundary-evidence edge features.**
//
// Input: a `u32` supervoxel volume (typically a watershed over-segmentation) and
// an `f64` boundary-evidence volume (membrane probability in `[0, 1]`, or any
// scalar), optionally a second `f64` intensity volume for per-node appearance.
// Output: a [`Rag`] — one [`EdgeStats`] per pair of labels that share a face
// under 6-connectivity, one [`NodeStats`] per label. It is the input
// [`super::agglomerate`] partitions.
//
// Label 0
// -------
// Label 0 is *not a region*: it is ignored (background, or the boundary voxels a
// watershed with lines emits). Faces touching 0 are not edges and 0 is not a
// node. With [`RagOptions::bridge_lines`] a 1-voxel run of 0 *sandwiched*
// between two different labels along an axis (`a | 0 | b`) is counted as a face
// between `a` and `b`, so a watershed that outputs lines still yields a
// connected graph. A bridged face samples the evidence at the line voxel itself
// (the boundary), a direct face the mean of its two voxels.
//
// Blocks and the merge
// --------------------
// A [`Rag`] is a sum of per-face and per-voxel contributions, so it is
// mergeable: [`Rag::merge`] is the exact union (sums, counts, histograms add;
// min/max fold). [`rag_block`] computes the contribution of one block under this
// convention, which makes every face count in **exactly one** block:
//
// * a voxel contributes to its node iff it is in the block's core;
// * a face (or bridge) is owned by the block whose core holds its
//   **lower** voxel (the `-axis` end), so a block looks only along `+axis`;
// * hence the caller passes a view that starts at the core's origin and extends
//   [`RagOptions::halo`] voxels past the core on the **high** side of each axis
//   (1, or 2 with bridging), clipped at the volume's end.
//
// Merging the blocks of any tiling then equals the whole-volume graph up to
// floating-point summation order. [`rag`] is exactly this over rayon z-slabs;
// [`super::rag_phases`] is the same over a planned, out-of-core decomposition,
// with each block's graph carried as [`Rag::to_bytes`] in a sidecar stream.
//
// Why not `ops::measure`'s `BoundaryMeasure` / `ContactRows`
// ----------------------------------------------------------
// `BoundaryMeasure` is the natural host and was checked first. It does not fit:
// its partials are keyed by **one** label (rows are per label, an edge's two
// sides would have to be reconciled afterwards), the label volume arrives as
// `f64`, the per-contact callback has no access to a second volume and no way to
// see the voxel *beyond* a 0 (so no line bridging), and it is driven through the
// measure planner's row streams rather than returning a value a graph algorithm
// can use directly. The face walk itself is a few lines; the edge-keyed
// accumulator is the part that matters, and that is what this module is.

use std::collections::HashMap;

use ndarray::{s, ArrayView3};
use rayon::prelude::*;

use super::agglomerate::FastHash;
use crate::error::{Error, Result};

/// What [`rag`] and [`rag_block`] accumulate and how.
#[derive(Debug, Clone, PartialEq)]
pub struct RagOptions {
    /// Physical voxel spacing `[z, y, x]`; a face normal to axis `a` has area
    /// equal to the product of the other two spacings.
    pub spacing: [f64; 3],
    /// Histogram bins per edge (0 disables the histogram and quantiles).
    pub bins: usize,
    /// Value range `[lo, hi)` the bins cover; values outside go to the end bins.
    pub range: (f64, f64),
    /// Count `a | 0 | b` along an axis as an `a`–`b` face (watershed lines).
    pub bridge_lines: bool,
}

impl RagOptions {
    /// `spacing`, a `bins`-bin histogram over `range`, no line bridging.
    pub fn new(spacing: [f64; 3], bins: usize, range: (f64, f64)) -> Self {
        Self {
            spacing,
            bins,
            range,
            bridge_lines: false,
        }
    }

    pub fn bridging_lines(mut self, bridge: bool) -> Self {
        self.bridge_lines = bridge;
        self
    }

    /// High-side halo a [`rag_block`] view needs past its core: 1, or 2 when
    /// bridging lines.
    pub fn halo(&self) -> usize {
        if self.bridge_lines {
            2
        } else {
            1
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let (lo, hi) = self.range;
        if self.bins > 0 && !(lo.is_finite() && hi.is_finite() && hi > lo) {
            return Err(Error::invalid(format!(
                "rag: histogram range must be finite with hi > lo, got {:?}",
                self.range
            )));
        }
        if self.spacing.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(Error::invalid(format!(
                "rag: spacing must be positive and finite, got {:?}",
                self.spacing
            )));
        }
        Ok(())
    }
}

/// Per-edge accumulated face statistics of the boundary evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeStats {
    /// Number of faces (bridged faces included).
    pub faces: u64,
    /// Faces per normal axis `[z, y, x]`.
    pub axis_faces: [u64; 3],
    /// Of `faces`, how many were bridged across a 0-line.
    pub bridged: u64,
    /// Physical face area (spacing-aware).
    pub area: f64,
    pub sum: f64,
    pub sum_sq: f64,
    pub min: f64,
    pub max: f64,
}

impl EdgeStats {
    fn empty() -> Self {
        Self {
            faces: 0,
            axis_faces: [0; 3],
            bridged: 0,
            area: 0.0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn mean(&self) -> f64 {
        self.sum / self.faces as f64
    }

    /// Population standard deviation of the face samples.
    pub fn std(&self) -> f64 {
        let mean = self.mean();
        (self.sum_sq / self.faces as f64 - mean * mean)
            .max(0.0)
            .sqrt()
    }

    fn fold(&mut self, other: &EdgeStats) {
        self.faces += other.faces;
        for axis in 0..3 {
            self.axis_faces[axis] += other.axis_faces[axis];
        }
        self.bridged += other.bridged;
        self.area += other.area;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

/// Per-node voxel count and (optional) intensity moments.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NodeStats {
    pub voxels: u64,
    /// Sum of the intensity volume over the node (0 when none was given).
    pub sum: f64,
    pub sum_sq: f64,
}

impl NodeStats {
    pub fn mean(&self) -> f64 {
        self.sum / self.voxels as f64
    }

    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        (self.sum_sq / self.voxels as f64 - mean * mean).max(0.0)
    }
}

/// Region adjacency graph. Edges are stored in insertion order and addressed by
/// index; [`Rag::sorted`] gives a canonical order.
#[derive(Debug, Clone)]
pub struct Rag {
    options: RagOptions,
    index: HashMap<u64, usize, FastHash>,
    keys: Vec<(u32, u32)>,
    edges: Vec<EdgeStats>,
    /// `edges.len() * bins` counts, edge-major.
    histograms: Vec<u32>,
    nodes: HashMap<u32, NodeStats, FastHash>,
}

fn key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

impl Rag {
    /// An empty graph accumulating under `options`.
    pub fn new(options: RagOptions) -> Self {
        Self {
            options,
            index: HashMap::default(),
            keys: Vec::new(),
            edges: Vec::new(),
            histograms: Vec::new(),
            nodes: HashMap::default(),
        }
    }

    pub fn options(&self) -> &RagOptions {
        &self.options
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Edge `i`: its labels `(lo, hi)` with `lo < hi`, and statistics.
    pub fn edge(&self, i: usize) -> ((u32, u32), &EdgeStats) {
        (self.keys[i], &self.edges[i])
    }

    /// Index of the edge between `a` and `b` (either order).
    pub fn find_edge(&self, a: u32, b: u32) -> Option<usize> {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        self.index.get(&key(lo, hi)).copied()
    }

    pub fn node(&self, label: u32) -> Option<&NodeStats> {
        self.nodes.get(&label)
    }

    /// Edge `i`'s histogram (`bins` counts; empty when `bins == 0`).
    pub fn histogram(&self, i: usize) -> &[u32] {
        let bins = self.options.bins;
        &self.histograms[i * bins..(i + 1) * bins]
    }

    /// Approximate `q`-quantile (`q` in `[0, 1]`) of edge `i`'s face samples,
    /// linearly interpolated inside the histogram bin and clamped to the edge's
    /// exact `[min, max]`. `NaN` without a histogram.
    pub fn quantile(&self, i: usize, q: f64) -> f64 {
        let bins = self.options.bins;
        if bins == 0 {
            return f64::NAN;
        }
        let hist = self.histogram(i);
        let total: u64 = hist.iter().map(|&c| c as u64).sum();
        let stats = &self.edges[i];
        if total == 0 {
            return f64::NAN;
        }
        let (lo, hi) = self.options.range;
        let width = (hi - lo) / bins as f64;
        let target = q.clamp(0.0, 1.0) * total as f64;
        let mut cumulative = 0.0;
        let mut value = hi;
        for (bin, &count) in hist.iter().enumerate() {
            let next = cumulative + count as f64;
            if count > 0 && next >= target {
                let fraction = (target - cumulative) / count as f64;
                value = lo + (bin as f64 + fraction) * width;
                break;
            }
            cumulative = next;
        }
        value.clamp(stats.min, stats.max)
    }

    /// All labels seen: nodes and edge endpoints, ascending.
    pub fn labels(&self) -> Vec<u32> {
        let mut labels: Vec<u32> = self.nodes.keys().copied().collect();
        for &(a, b) in &self.keys {
            if !self.nodes.contains_key(&a) {
                labels.push(a);
            }
            if !self.nodes.contains_key(&b) {
                labels.push(b);
            }
        }
        labels.sort_unstable();
        labels.dedup();
        labels
    }

    /// Edge indices ordered by `(lo, hi)` label pair.
    pub fn sorted(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.keys.len()).collect();
        order.sort_unstable_by_key(|&i| self.keys[i]);
        order
    }

    /// A weighted graph over dense node indices for [`super::agglomerate`]:
    /// `labels[n]` is node `n`'s supervoxel label, and each edge is
    /// `(u, v, weight(rag, edge_index))` in [`Rag::sorted`] order.
    pub fn graph(&self, mut weight: impl FnMut(&Rag, usize) -> f64) -> RagGraph {
        let labels = self.labels();
        let edges = self
            .sorted()
            .into_iter()
            .map(|i| {
                let (a, b) = self.keys[i];
                let u = labels.binary_search(&a).expect("endpoint is a label");
                let v = labels.binary_search(&b).expect("endpoint is a label");
                (u, v, weight(self, i))
            })
            .collect();
        RagGraph { labels, edges }
    }

    /// Add `other`'s contributions into `self`. Both must have equal options.
    pub fn merge(&mut self, other: Rag) {
        assert_eq!(
            self.options, other.options,
            "Rag::merge: graphs accumulated under different options"
        );
        if other.edges.len() > self.edges.len() {
            let mut other = other;
            std::mem::swap(self, &mut other);
            self.merge(other);
            return;
        }
        let bins = self.options.bins;
        for (i, (&(a, b), stats)) in other.keys.iter().zip(&other.edges).enumerate() {
            let at = self.slot(a, b);
            self.edges[at].fold(stats);
            let from = &other.histograms[i * bins..(i + 1) * bins];
            for (into, &count) in self.histograms[at * bins..(at + 1) * bins]
                .iter_mut()
                .zip(from)
            {
                *into = into.saturating_add(count);
            }
        }
        for (label, stats) in other.nodes {
            let node = self.nodes.entry(label).or_default();
            node.voxels += stats.voxels;
            node.sum += stats.sum;
            node.sum_sq += stats.sum_sq;
        }
    }

    /// [`Rag::merge`], refusing rather than panicking when the two graphs were
    /// accumulated under different options (as decoded fragments can be).
    pub fn try_merge(&mut self, other: Rag) -> Result<()> {
        if self.options != other.options {
            return Err(Error::invalid(format!(
                "rag: cannot merge a graph accumulated under {:?} into one under {:?}",
                other.options, self.options
            )));
        }
        self.merge(other);
        Ok(())
    }

    fn slot(&mut self, lo: u32, hi: u32) -> usize {
        let next = self.edges.len();
        let at = *self.index.entry(key(lo, hi)).or_insert(next);
        if at == next {
            self.keys.push((lo, hi));
            self.edges.push(EdgeStats::empty());
            self.histograms
                .extend(std::iter::repeat_n(0, self.options.bins));
        }
        at
    }

    #[inline]
    fn add_face(
        &mut self,
        a: u32,
        b: u32,
        axis: usize,
        value: f64,
        bridged: bool,
        cache: &mut Option<(u64, usize)>,
    ) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let k = key(lo, hi);
        let at = match *cache {
            Some((cached, at)) if cached == k => at,
            _ => {
                let at = self.slot(lo, hi);
                *cache = Some((k, at));
                at
            }
        };
        let area = self.face_area(axis);
        let edge = &mut self.edges[at];
        edge.faces += 1;
        edge.axis_faces[axis] += 1;
        edge.bridged += bridged as u64;
        edge.area += area;
        edge.sum += value;
        edge.sum_sq += value * value;
        edge.min = edge.min.min(value);
        edge.max = edge.max.max(value);
        let bins = self.options.bins;
        if bins > 0 {
            let (rlo, rhi) = self.options.range;
            let t = (value - rlo) / (rhi - rlo) * bins as f64;
            // NaN falls into bin 0 through the saturating cast.
            let bin = (t as isize).clamp(0, bins as isize - 1) as usize;
            let count = &mut self.histograms[at * bins + bin];
            *count = count.saturating_add(1);
        }
    }

    #[inline]
    fn face_area(&self, axis: usize) -> f64 {
        let s = self.options.spacing;
        match axis {
            0 => s[1] * s[2],
            1 => s[0] * s[2],
            _ => s[0] * s[1],
        }
    }
}

// --------------------------------------------------------- serialisation --
//
// A per-block `Rag` travels as a sidecar fragment (`super::rag_phases`), so it
// needs bytes. Little-endian, fixed-width, no serde — the sidecar convention is
// that the writer and the reader own the format. Edges are written in
// `(lo, hi)` order and nodes in label order, so equal graphs encode to equal
// bytes whatever order they were accumulated in.
//
// header  : b"BFRG", version u32,
//           spacing 3 x f64, bins u32, bridge_lines u32, range 2 x f64,
//           edge count u64, node count u64                          (72 B)
// edge    : lo u32, hi u32, axis_faces 3 x u64, bridged u64,
//           area, sum, sum_sq, min, max f64, then `bins` x u32       (80 + 4 bins B)
// node    : label u32, voxels u64, sum f64, sum_sq f64              (28 B)

const RAG_MAGIC: [u8; 4] = *b"BFRG";
const RAG_VERSION: u32 = 1;

/// Bytes of an encoded [`Rag`]'s header.
pub const RAG_HEADER_BYTES: usize = 72;
/// Bytes per encoded node.
pub const RAG_NODE_BYTES: usize = 28;

/// Bytes per encoded edge with a `bins`-bin histogram.
pub fn rag_edge_bytes(bins: usize) -> usize {
    80 + 4 * bins
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len());
        let Some(end) = end else {
            return Err(Error::invalid(format!(
                "rag: an encoded graph of {} bytes ends inside {what} (at byte {})",
                self.bytes.len(),
                self.at
            )));
        };
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }
    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn f64(&mut self, what: &str) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
}

impl Rag {
    /// Encode as compact little-endian bytes (see the format above). Canonical:
    /// equal graphs give equal bytes regardless of accumulation order.
    pub fn to_bytes(&self) -> Vec<u8> {
        let bins = self.options.bins;
        let mut out = Vec::with_capacity(
            RAG_HEADER_BYTES
                + self.edges.len() * rag_edge_bytes(bins)
                + self.nodes.len() * RAG_NODE_BYTES,
        );
        out.extend_from_slice(&RAG_MAGIC);
        out.extend_from_slice(&RAG_VERSION.to_le_bytes());
        for s in self.options.spacing {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(&(bins as u32).to_le_bytes());
        out.extend_from_slice(&(self.options.bridge_lines as u32).to_le_bytes());
        out.extend_from_slice(&self.options.range.0.to_le_bytes());
        out.extend_from_slice(&self.options.range.1.to_le_bytes());
        out.extend_from_slice(&(self.edges.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.nodes.len() as u64).to_le_bytes());
        for i in self.sorted() {
            let (lo, hi) = self.keys[i];
            let e = &self.edges[i];
            out.extend_from_slice(&lo.to_le_bytes());
            out.extend_from_slice(&hi.to_le_bytes());
            for count in e.axis_faces {
                out.extend_from_slice(&count.to_le_bytes());
            }
            out.extend_from_slice(&e.bridged.to_le_bytes());
            for value in [e.area, e.sum, e.sum_sq, e.min, e.max] {
                out.extend_from_slice(&value.to_le_bytes());
            }
            for count in self.histogram(i) {
                out.extend_from_slice(&count.to_le_bytes());
            }
        }
        let mut labels: Vec<u32> = self.nodes.keys().copied().collect();
        labels.sort_unstable();
        for label in labels {
            let n = &self.nodes[&label];
            out.extend_from_slice(&label.to_le_bytes());
            out.extend_from_slice(&n.voxels.to_le_bytes());
            out.extend_from_slice(&n.sum.to_le_bytes());
            out.extend_from_slice(&n.sum_sq.to_le_bytes());
        }
        out
    }

    /// The inverse of [`Rag::to_bytes`]. Refuses a wrong magic or version, a
    /// truncated or over-long buffer, an edge that is not `0 < lo < hi`, a
    /// repeated edge or node, and invalid options.
    pub fn from_bytes(bytes: &[u8]) -> Result<Rag> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4, "the magic")? != RAG_MAGIC {
            return Err(Error::invalid(
                "rag: the bytes do not start with the encoded-graph magic \"BFRG\"",
            ));
        }
        let version = r.u32("the version")?;
        if version != RAG_VERSION {
            return Err(Error::invalid(format!(
                "rag: encoded graph version {version}, this build reads {RAG_VERSION}"
            )));
        }
        let spacing = [
            r.f64("the spacing")?,
            r.f64("the spacing")?,
            r.f64("the spacing")?,
        ];
        let bins = r.u32("the bin count")? as usize;
        let bridge_lines = match r.u32("the bridging flag")? {
            0 => false,
            1 => true,
            other => {
                return Err(Error::invalid(format!(
                    "rag: bridging flag {other} is neither 0 nor 1"
                )))
            }
        };
        let range = (r.f64("the range")?, r.f64("the range")?);
        let options = RagOptions {
            spacing,
            bins,
            range,
            bridge_lines,
        };
        options.validate()?;
        let n_edges = r.u64("the edge count")? as usize;
        let n_nodes = r.u64("the node count")? as usize;
        let expected = n_edges
            .checked_mul(rag_edge_bytes(bins))
            .and_then(|e| {
                n_nodes
                    .checked_mul(RAG_NODE_BYTES)
                    .and_then(|n| e.checked_add(n))
            })
            .and_then(|body| body.checked_add(RAG_HEADER_BYTES));
        if expected != Some(bytes.len()) {
            return Err(Error::invalid(format!(
                "rag: an encoded graph declaring {n_edges} edges and {n_nodes} nodes at {bins} \
                 bins is {expected:?} bytes, and this one is {}",
                bytes.len()
            )));
        }
        let mut rag = Rag::new(options);
        rag.keys.reserve(n_edges);
        rag.edges.reserve(n_edges);
        rag.histograms.reserve(n_edges * bins);
        for _ in 0..n_edges {
            let lo = r.u32("an edge")?;
            let hi = r.u32("an edge")?;
            if lo == 0 || lo >= hi {
                return Err(Error::invalid(format!(
                    "rag: encoded edge ({lo}, {hi}) is not 0 < lo < hi"
                )));
            }
            let axis_faces = [r.u64("an edge")?, r.u64("an edge")?, r.u64("an edge")?];
            let bridged = r.u64("an edge")?;
            let [area, sum, sum_sq, min, max] = [
                r.f64("an edge")?,
                r.f64("an edge")?,
                r.f64("an edge")?,
                r.f64("an edge")?,
                r.f64("an edge")?,
            ];
            let at = rag.edges.len();
            if rag.index.insert(key(lo, hi), at).is_some() {
                return Err(Error::invalid(format!(
                    "rag: encoded edge ({lo}, {hi}) appears twice"
                )));
            }
            rag.keys.push((lo, hi));
            rag.edges.push(EdgeStats {
                faces: axis_faces.iter().sum(),
                axis_faces,
                bridged,
                area,
                sum,
                sum_sq,
                min,
                max,
            });
            for _ in 0..bins {
                let count = r.u32("a histogram")?;
                rag.histograms.push(count);
            }
        }
        rag.nodes.reserve(n_nodes);
        for _ in 0..n_nodes {
            let label = r.u32("a node")?;
            let stats = NodeStats {
                voxels: r.u64("a node")?,
                sum: r.f64("a node")?,
                sum_sq: r.f64("a node")?,
            };
            if label == 0 || rag.nodes.insert(label, stats).is_some() {
                return Err(Error::invalid(format!(
                    "rag: encoded node {label} is 0 or appears twice"
                )));
            }
        }
        Ok(rag)
    }
}

/// [`Rag::graph`]'s output: dense node numbering and a weighted edge list.
#[derive(Debug, Clone, PartialEq)]
pub struct RagGraph {
    /// Node index -> supervoxel label, ascending.
    pub labels: Vec<u32>,
    pub edges: Vec<(usize, usize, f64)>,
}

fn check_shapes(
    labels: &ArrayView3<'_, u32>,
    evidence: &ArrayView3<'_, f64>,
    intensity: &Option<ArrayView3<'_, f64>>,
) -> Result<()> {
    if labels.shape() != evidence.shape() {
        return Err(Error::invalid(format!(
            "rag: evidence shape {:?} differs from labels {:?}",
            evidence.shape(),
            labels.shape()
        )));
    }
    if let Some(intensity) = intensity {
        if intensity.shape() != labels.shape() {
            return Err(Error::invalid(format!(
                "rag: intensity shape {:?} differs from labels {:?}",
                intensity.shape(),
                labels.shape()
            )));
        }
    }
    Ok(())
}

/// One block's contribution to the RAG (see the module header for the
/// ownership convention). The views start at the core's origin; `core` is the
/// core's shape, and anything past it on the high side (up to
/// [`RagOptions::halo`]) is read-only context. Merging the results of a tiling
/// with [`Rag::merge`] gives the whole-volume graph.
pub fn rag_block(
    labels: ArrayView3<'_, u32>,
    evidence: ArrayView3<'_, f64>,
    intensity: Option<ArrayView3<'_, f64>>,
    core: [usize; 3],
    options: &RagOptions,
) -> Result<Rag> {
    options.validate()?;
    check_shapes(&labels, &evidence, &intensity)?;
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    if (0..3).any(|a| core[a] > shape[a]) {
        return Err(Error::invalid(format!(
            "rag: core {core:?} exceeds view {shape:?}"
        )));
    }
    let mut rag = Rag::new(options.clone());
    let bridge = options.bridge_lines;
    let mut node_cache: Option<(u32, NodeStats)> = None;
    let mut caches: [Option<(u64, usize)>; 3] = [None; 3];
    let flush = |rag: &mut Rag, cache: &mut Option<(u32, NodeStats)>| {
        if let Some((label, stats)) = cache.take() {
            let node = rag.nodes.entry(label).or_default();
            node.voxels += stats.voxels;
            node.sum += stats.sum;
            node.sum_sq += stats.sum_sq;
        }
    };
    for z in 0..core[0] {
        for y in 0..core[1] {
            for x in 0..core[2] {
                // SAFETY: z, y, x < core <= shape, and the neighbour reads below
                // are each bounds-checked against `shape` first.
                let a = unsafe { *labels.uget([z, y, x]) };
                if a == 0 {
                    continue;
                }
                let value = unsafe { *evidence.uget([z, y, x]) };
                // Node accumulation, run-length cached along x.
                let sample = intensity
                    .as_ref()
                    .map(|v| unsafe { *v.uget([z, y, x]) })
                    .unwrap_or(0.0);
                match &mut node_cache {
                    Some((label, stats)) if *label == a => {
                        stats.voxels += 1;
                        stats.sum += sample;
                        stats.sum_sq += sample * sample;
                    }
                    _ => {
                        flush(&mut rag, &mut node_cache);
                        node_cache = Some((
                            a,
                            NodeStats {
                                voxels: 1,
                                sum: sample,
                                sum_sq: sample * sample,
                            },
                        ));
                    }
                }
                let at = [z, y, x];
                for axis in 0..3 {
                    let mut q = at;
                    q[axis] += 1;
                    if q[axis] >= shape[axis] {
                        continue;
                    }
                    let b = unsafe { *labels.uget(q) };
                    if b == a {
                        continue;
                    }
                    if b != 0 {
                        let mean = 0.5 * (value + unsafe { *evidence.uget(q) });
                        rag.add_face(a, b, axis, mean, false, &mut caches[axis]);
                    } else if bridge {
                        let mut r = q;
                        r[axis] += 1;
                        if r[axis] >= shape[axis] {
                            continue;
                        }
                        let c = unsafe { *labels.uget(r) };
                        if c != 0 && c != a {
                            let line = unsafe { *evidence.uget(q) };
                            rag.add_face(a, c, axis, line, true, &mut caches[axis]);
                        }
                    }
                }
            }
        }
    }
    flush(&mut rag, &mut node_cache);
    Ok(rag)
}

/// The whole-volume RAG, computed as [`rag_block`] over rayon z-slabs and
/// merged. Label 0 is ignored; see the module header.
pub fn rag(
    labels: ArrayView3<'_, u32>,
    evidence: ArrayView3<'_, f64>,
    intensity: Option<ArrayView3<'_, f64>>,
    options: &RagOptions,
) -> Result<Rag> {
    options.validate()?;
    check_shapes(&labels, &evidence, &intensity)?;
    let nz = labels.shape()[0];
    let [ny, nx] = [labels.shape()[1], labels.shape()[2]];
    let slabs = (rayon::current_num_threads() * 2).clamp(1, nz.max(1));
    let step = nz.div_ceil(slabs).max(1);
    let halo = options.halo();
    let parts: Result<Vec<Rag>> = (0..nz.div_ceil(step))
        .into_par_iter()
        .map(|slab| {
            let z0 = slab * step;
            let z1 = (z0 + step).min(nz);
            let read = (z1 + halo).min(nz);
            let lv = labels.slice(s![z0..read, .., ..]);
            let ev = evidence.slice(s![z0..read, .., ..]);
            let iv = intensity.as_ref().map(|v| v.slice(s![z0..read, .., ..]));
            rag_block(lv, ev, iv, [z1 - z0, ny, nx], options)
        })
        .collect();
    let mut parts = parts?;
    // Pairwise tree reduction keeps the merges balanced.
    while parts.len() > 1 {
        parts = parts
            .into_par_iter()
            .chunks(2)
            .map(|mut pair| {
                let mut first = pair.swap_remove(0);
                if let Some(second) = pair.pop() {
                    first.merge(second);
                }
                first
            })
            .collect();
    }
    Ok(parts.pop().unwrap_or_else(|| Rag::new(options.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    /// 1 x 2 x 4: labels [1 1 2 2 / 3 3 3 3].
    fn toy() -> (Array3<u32>, Array3<f64>) {
        let labels = Array3::from_shape_vec((1, 2, 4), vec![1, 1, 2, 2, 3, 3, 3, 3]).unwrap();
        let evidence =
            Array3::from_shape_vec((1, 2, 4), vec![0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8])
                .unwrap();
        (labels, evidence)
    }

    #[test]
    fn toy_faces_and_means() {
        let (labels, evidence) = toy();
        let opts = RagOptions::new([2.0, 1.0, 3.0], 10, (0.0, 1.0));
        let g = rag(labels.view(), evidence.view(), Some(evidence.view()), &opts).unwrap();
        assert_eq!(g.edge_count(), 3);
        // 1-2: one x-face, (0.3+0.5)/2.
        let e = g.find_edge(2, 1).unwrap();
        let (k, st) = g.edge(e);
        assert_eq!(k, (1, 2));
        assert_eq!((st.faces, st.axis_faces), (1, [0, 0, 1]));
        assert!(approx(st.mean(), 0.4));
        assert!(approx(st.area, 2.0 * 1.0));
        // 1-3: two y-faces, (0.1+0.2)/2 and (0.3+0.4)/2.
        let (_, st) = g.edge(g.find_edge(1, 3).unwrap());
        assert_eq!((st.faces, st.axis_faces), (2, [0, 2, 0]));
        assert!(approx(st.mean(), 0.25));
        assert!(approx(st.min, 0.15) && approx(st.max, 0.35));
        assert!(approx(st.area, 2.0 * 2.0 * 3.0));
        // 2-3: (0.5+0.6)/2, (0.7+0.8)/2.
        let (_, st) = g.edge(g.find_edge(2, 3).unwrap());
        assert_eq!(st.faces, 2);
        assert!(approx(st.mean(), 0.65));
        let hist = g.histogram(g.find_edge(2, 3).unwrap());
        assert_eq!(hist[5], 1);
        assert_eq!(hist[7], 1);
        let median = g.quantile(g.find_edge(2, 3).unwrap(), 0.5);
        assert!((0.55..=0.75).contains(&median));
        let n3 = g.node(3).unwrap();
        assert_eq!(n3.voxels, 4);
        assert!(approx(n3.mean(), 0.5));
        assert_eq!(g.node(1).unwrap().voxels, 2);
        assert!(g.node(0).is_none());
    }

    #[test]
    fn bridging_lines() {
        // 1 0 2 along x, and 1 0 1 (same label: no edge).
        let labels = Array3::from_shape_vec((1, 2, 3), vec![1, 0, 2, 1, 0, 1]).unwrap();
        let evidence =
            Array3::from_shape_vec((1, 2, 3), vec![0.0, 0.9, 0.0, 0.0, 0.5, 0.0]).unwrap();
        let plain = RagOptions::new([1.0; 3], 4, (0.0, 1.0));
        let g = rag(labels.view(), evidence.view(), None, &plain).unwrap();
        assert_eq!(g.edge_count(), 1); // 1-2 via the y face at x=2.
        assert_eq!(g.edge(0).1.faces, 1);
        let g = rag(
            labels.view(),
            evidence.view(),
            None,
            &plain.clone().bridging_lines(true),
        )
        .unwrap();
        let (_, st) = g.edge(g.find_edge(1, 2).unwrap());
        assert_eq!((st.faces, st.bridged), (2, 1));
        assert!(approx(st.max, 0.9));
    }

    fn random_volume(
        shape: (usize, usize, usize),
        labels: u32,
        seed: u64,
    ) -> (Array3<u32>, Array3<f64>) {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Blocky labels so that faces repeat, with some zeros.
        let lab = Array3::from_shape_fn(shape, |(z, y, x)| {
            let h = ((z / 2) * 31 + (y / 3) * 17 + (x / 2) * 7) as u32;
            if (x + y + z) % 11 == 0 {
                0
            } else {
                h % labels + 1
            }
        });
        let lab = lab.mapv(|v| {
            if next() % 9 == 0 {
                (next() % labels as u64) as u32
            } else {
                v
            }
        });
        let ev = Array3::from_shape_fn(shape, |(z, y, x)| ((z * 7 + y * 3 + x) % 13) as f64 / 13.0);
        (lab, ev)
    }

    fn assert_same(a: &Rag, b: &Rag) {
        assert_eq!(a.edge_count(), b.edge_count());
        assert_eq!(a.node_count(), b.node_count());
        for i in 0..a.edge_count() {
            let ((u, v), sa) = a.edge(i);
            let j = b.find_edge(u, v).expect("edge present in both");
            let (_, sb) = b.edge(j);
            assert_eq!(sa.faces, sb.faces);
            assert_eq!(sa.axis_faces, sb.axis_faces);
            assert_eq!(sa.bridged, sb.bridged);
            assert!(approx(sa.sum, sb.sum) && approx(sa.sum_sq, sb.sum_sq));
            assert!(approx(sa.area, sb.area));
            assert_eq!((sa.min, sa.max), (sb.min, sb.max));
            assert_eq!(a.histogram(i), b.histogram(j));
        }
        for label in a.labels() {
            let (na, nb) = (a.node(label).unwrap(), b.node(label).unwrap());
            assert_eq!(na.voxels, nb.voxels);
            assert!(approx(na.sum, nb.sum));
        }
    }

    #[test]
    fn block_merge_equals_whole() {
        let shape = (9, 10, 11);
        let (lab, ev) = random_volume(shape, 12, 7);
        for bridge in [false, true] {
            let opts = RagOptions::new([1.0, 0.5, 0.25], 8, (0.0, 1.0)).bridging_lines(bridge);
            let whole =
                rag_block(lab.view(), ev.view(), Some(ev.view()), [9, 10, 11], &opts).unwrap();
            let slabs = rag(lab.view(), ev.view(), Some(ev.view()), &opts).unwrap();
            assert_same(&whole, &slabs);
            let halo = opts.halo();
            let block = [4, 3, 5];
            let mut merged = Rag::new(opts.clone());
            let dims = [shape.0, shape.1, shape.2];
            for z0 in (0..dims[0]).step_by(block[0]) {
                for y0 in (0..dims[1]).step_by(block[1]) {
                    for x0 in (0..dims[2]).step_by(block[2]) {
                        let o = [z0, y0, x0];
                        let core: [usize; 3] =
                            std::array::from_fn(|a| block[a].min(dims[a] - o[a]));
                        let end: [usize; 3] =
                            std::array::from_fn(|a| (o[a] + core[a] + halo).min(dims[a]));
                        let sl = s![z0..end[0], y0..end[1], x0..end[2]];
                        let part =
                            rag_block(lab.slice(sl), ev.slice(sl), Some(ev.slice(sl)), core, &opts)
                                .unwrap();
                        merged.merge(part);
                    }
                }
            }
            assert_same(&whole, &merged);
        }
    }

    #[test]
    fn bytes_round_trip_and_are_canonical() {
        let (lab, ev) = random_volume((5, 6, 7), 9, 3);
        let opts = RagOptions::new([2.0, 1.0, 1.0], 6, (0.0, 1.0)).bridging_lines(true);
        let g = rag(lab.view(), ev.view(), Some(ev.view()), &opts).unwrap();
        let bytes = g.to_bytes();
        assert_eq!(
            bytes.len(),
            RAG_HEADER_BYTES + g.edge_count() * rag_edge_bytes(6) + g.node_count() * RAG_NODE_BYTES
        );
        let back = Rag::from_bytes(&bytes).unwrap();
        assert_eq!(back.options(), g.options());
        assert_same(&g, &back);
        assert_eq!(back.to_bytes(), bytes);
        // Accumulated in another order (one block), same bytes up to f64 sums:
        // the edge and node sets and integer fields encode identically.
        let one = rag_block(lab.view(), ev.view(), Some(ev.view()), [5, 6, 7], &opts).unwrap();
        assert_same(&one, &back);
        // Empty graph, truncation, garbage.
        let empty = Rag::new(opts.clone());
        assert_eq!(Rag::from_bytes(&empty.to_bytes()).unwrap().edge_count(), 0);
        assert!(Rag::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(Rag::from_bytes(b"nope").is_err());
        let mut other = Rag::new(RagOptions::new([1.0; 3], 6, (0.0, 1.0)));
        assert!(other.try_merge(back).is_err());
    }

    #[test]
    fn graph_export() {
        let (labels, evidence) = toy();
        let g = rag(
            labels.view(),
            evidence.view(),
            None,
            &RagOptions::new([1.0; 3], 0, (0.0, 1.0)),
        )
        .unwrap();
        let graph = g.graph(|r, i| r.edge(i).1.mean());
        assert_eq!(graph.labels, vec![1, 2, 3]);
        assert_eq!(graph.edges.len(), 3);
        assert_eq!((graph.edges[0].0, graph.edges[0].1), (0, 1));
        assert!(g.quantile(0, 0.5).is_nan());
    }

    /// `cargo test --release --lib ops::rag::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn timing() {
        let shape = (128, 512, 512);
        // ~ 50k supervoxels: 6 x 11 x 11 bricks, jittered per row.
        let lab = Array3::from_shape_fn(shape, |(z, y, x)| {
            let bz = z / 6;
            let by = (y + 3 * bz) / 11;
            let bx = (x + 5 * by) / 11;
            (bz * 48 * 48 + by * 48 + bx) as u32 + 1
        });
        let ev = Array3::from_shape_fn(shape, |(z, y, x)| {
            ((z * 7 + y * 3 + x) % 101) as f64 / 101.0
        });
        let opts = RagOptions::new([2.0, 1.0, 1.0], 32, (0.0, 1.0));
        let t = std::time::Instant::now();
        let g = rag(lab.view(), ev.view(), Some(ev.view()), &opts).unwrap();
        eprintln!(
            "rag 128x512x512: {} nodes, {} edges in {:.2?}",
            g.node_count(),
            g.edge_count(),
            t.elapsed()
        );
    }
}
