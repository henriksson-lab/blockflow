// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The region adjacency graph and its relabelling, as planned phases.**
//
// [`super::rag::rag`] needs the whole supervoxel and evidence volumes resident.
// On a volume-EM stack (~130 x 1750 x 2300, `u32` labels plus an `f32`/`f64`
// membrane map) that is 2-4 GB per array before anything is computed, and the
// arrays live in a Zarr store anyway. This module runs the same computation out
// of core, as three pieces:
//
// 1. **Emit** — [`RagBlockOp`], a `volume -> fragments` [`FragmentOp`]. Each
//    block reads its core of the supervoxel image, the evidence image and,
//    optionally, an intensity image, each widened by the
//    [`RagOptions::halo`] the face walk needs (1, or 2 with line bridging;
//    `fragment_phase` grants it on both sides although only the high side is
//    used), runs [`super::rag::rag_block`] under its ownership convention and
//    writes the block's graph as [`Rag::to_bytes`] into a sidecar stream. A
//    face is owned by the block holding its lower voxel, so every face lands
//    in exactly one fragment and no seam reconciliation is needed: the
//    fragment is a function of this block alone ([`SeamFold::PerBlock`]).
// 2. **Reduce** — [`collect_rag`], after the run: [`fold_fragments`] streams
//    the fragments one at a time, in key order, into one [`Rag`] with
//    [`Rag::try_merge`]. Resident: the merged graph plus one fragment. In key
//    order, so the `f64` sums are reproducible for a given lattice; across
//    lattices they agree to rounding (the integer fields — faces, bridges,
//    voxel counts, histograms — and min/max exactly).
// 3. **Relabel** — [`RelabelOp`], a stencil [`BlockOp`] of reach 0 mapping
//    supervoxel ids to cluster ids through a [`LabelLookup`] built once and
//    shared. It runs in a *second* plan over the supervoxel image, because
//    the table is the output of the agglomeration, which runs on the reduced
//    graph between the two `execute_phases` calls.
//
// Supervoxel ids must be globally unique
// --------------------------------------
// A RAG node is "every voxel carrying label `l`", wherever it is. A
// supervoxel image whose ids were assigned per block (two blocks both using
// 1..n) silently merges unrelated regions. What this module needs is only that
// distinct regions carry distinct ids — not that ids are dense or that a
// region is connected. The ways to get such an image in this crate:
//
// * **Seeded watershed from globally labelled seeds.** Label the seed mask
//   (e.g. thresholded / regional minima of the membrane map) with
//   `ops::label::component_label_phases` (`LabelComponentsOp` +
//   `RelabelComponentsOp`): components are numbered from 1 in row-major order
//   of their lowest voxel over the *whole volume*, byte-identical across
//   lattices (`GlobalLabels`). A watershed flood
//   (`ops::watershed::ScalableWatershedOp`, `scalable_seeded_watershed`)
//   copies seed labels into basins, so its output inherits the uniqueness.
// * **Connected components of a foreground mask**, the same two phases, when
//   supervoxels are the components themselves.
// * **An image produced elsewhere** (attached through
//   `ZarrEnvironment::attach`) whose ids are already global.
//
// Block-local ids that were never closed across seams must not be fed here;
// `ops::label`'s header (invariant 3) says the same of its local-label image.

use std::sync::Arc;

use ndarray::{s, Array3, ArrayView3};

use super::agglomerate::LabelLookup;
use super::rag::{rag_block, rag_edge_bytes, Rag, RagOptions, RAG_HEADER_BYTES, RAG_NODE_BYTES};
use super::TypedSource;
use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, SeamFold,
    SidecarSize, SourceBlocks,
};
use crate::op::{Anchor, BlockOp, Chain, Slicing, SourceInput};
use crate::reach::Reach;
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

/// A stored image an op reads, and what it holds.
///
/// The element type is required: a supplied input (`ImageId::supplied`) is
/// written by no phase, so the reader's declaration is the only statement of
/// it, and the plan refuses one without.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RagImage {
    pub image: ImageId,
    pub dtype: Dtype,
}

impl RagImage {
    pub fn new(image: impl Into<ImageId>, dtype: Dtype) -> Self {
        Self {
            image: image.into(),
            dtype,
        }
    }

    fn source(self) -> TypedSource {
        TypedSource::new(self.image).holding(self.dtype)
    }
}

// ------------------------------------------------------------------ emit --

/// `volume -> fragments`: one block's [`Rag`] contribution, as bytes.
pub struct RagBlockOp {
    labels: RagImage,
    evidence: RagImage,
    intensity: Option<RagImage>,
    options: RagOptions,
    stream: String,
    lifecycle: Lifecycle,
}

impl RagBlockOp {
    /// Refuses invalid options, a float or bool label image, and two roles
    /// naming the same image.
    pub fn new(
        labels: RagImage,
        evidence: RagImage,
        intensity: Option<RagImage>,
        options: RagOptions,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
    ) -> Result<Self> {
        options.validate()?;
        if matches!(labels.dtype, Dtype::Bool | Dtype::F32 | Dtype::F64) {
            return Err(Error::invalid(format!(
                "rag phases: the supervoxel image holds {:?}; a label image is an integer image",
                labels.dtype
            )));
        }
        if labels.image == evidence.image || intensity.is_some_and(|i| i.image == labels.image) {
            return Err(Error::invalid(
                "rag phases: the supervoxel image is also named as evidence or intensity; the \
                 graph's edge features come from a second array",
            ));
        }
        Ok(Self {
            labels,
            evidence,
            intensity,
            options,
            stream: stream.into(),
            lifecycle,
        })
    }

    pub fn options(&self) -> &RagOptions {
        &self.options
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The graph of one block from its three buffers over `read`, of which
    /// `core` is the part this block owns. The arithmetic without the plan,
    /// so a test can drive it directly. A simulated (accounted) block gives
    /// an empty graph.
    pub fn block_rag(
        &self,
        labels: &BlockBuf,
        evidence: &BlockBuf,
        intensity: Option<&BlockBuf>,
        read: &Region,
        core: &Region,
    ) -> Result<Rag> {
        let (BlockBuf::Array(labels), BlockBuf::Array(evidence)) = (labels, evidence) else {
            return Ok(Rag::new(self.options.clone()));
        };
        let intensity = match intensity {
            Some(BlockBuf::Array(v)) => Some(v),
            Some(BlockBuf::Accounted { .. }) => return Ok(Rag::new(self.options.clone())),
            None => None,
        };
        let shape = read.shape3();
        for (what, got) in [
            ("supervoxel image", labels.shape()),
            ("evidence", evidence.shape()),
        ]
        .into_iter()
        .chain(intensity.map(|v| ("intensity", v.shape())))
        {
            if got != shape {
                return Err(Error::invalid(format!(
                    "rag phases: the {what} arrived as {got:?} for a read extent of {shape:?}"
                )));
            }
        }
        // The view starts at the core's origin and runs `halo` past its end,
        // clipped to what was read (the volume's end).
        let halo = self.options.halo();
        let mut offset = [0usize; 3];
        let mut core_shape = [0usize; 3];
        let mut end = [0usize; 3];
        for axis in 0..3 {
            offset[axis] = core.start[axis]
                .checked_sub(read.start[axis])
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "rag phases: core {core:?} starts before its read extent {read:?}"
                    ))
                })?;
            core_shape[axis] = core.shape[axis];
            end[axis] = (offset[axis] + core_shape[axis] + halo).min(shape[axis]);
        }
        let window = s![offset[0]..end[0], offset[1]..end[1], offset[2]..end[2]];
        let labels = u32_labels(labels)?;
        let evidence = f64_values(evidence);
        let intensity = intensity.map(f64_values);
        rag_block(
            labels.view().slice_move(window),
            evidence.view().slice_move(window),
            intensity.as_ref().map(|v| v.view().slice_move(window)),
            core_shape,
            &self.options,
        )
    }
}

/// Either a borrowed `u32` view or an owned conversion.
enum Labels<'a> {
    Borrowed(ArrayView3<'a, u32>),
    Owned(Array3<u32>),
}

impl Labels<'_> {
    fn view(&self) -> ArrayView3<'_, u32> {
        match self {
            Labels::Borrowed(v) => v.view(),
            Labels::Owned(a) => a.view(),
        }
    }
}

fn u32_labels(voxels: &Voxels) -> Result<Labels<'_>> {
    if let Voxels::U32(array) = voxels {
        return Ok(Labels::Borrowed(array.view()));
    }
    if matches!(voxels, Voxels::Bool(_) | Voxels::F32(_) | Voxels::F64(_)) {
        return Err(Error::invalid(format!(
            "rag phases: the supervoxel image holds {:?}; a label image is an integer image",
            voxels.dtype()
        )));
    }
    let wide = voxels.widened_i64("rag phases supervoxel image")?;
    let mut out = Array3::<u32>::zeros(wide.raw_dim());
    for (o, &v) in out.iter_mut().zip(wide.iter()) {
        *o = u32::try_from(v).map_err(|_| {
            Error::invalid(format!(
                "rag phases: supervoxel id {v} does not fit a u32 label; graph nodes are u32"
            ))
        })?;
    }
    Ok(Labels::Owned(out))
}

/// `f64` evidence, widened from whatever the image holds (copy-free for
/// `f64`). Returned owned or borrowed through `CowArray`.
fn f64_values(voxels: &Voxels) -> ndarray::CowArray<'_, f64, ndarray::Ix3> {
    match voxels {
        Voxels::F64(array) => ndarray::CowArray::from(array.view()),
        other => ndarray::CowArray::from(other.widened()),
    }
}

impl FragmentOp for RagBlockOp {
    fn name(&self) -> &'static str {
        "rag-block"
    }

    /// **Zero.** The face walk's halo is a window on the source images and is
    /// declared there ([`Self::source_inputs`]); what the block emits is for
    /// its core whatever window it read, so nothing is shrunk off it.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn cost_per_voxel(&self) -> f64 {
        2.0
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let halo = self.options.halo();
        let reach = Reach::asymmetric([(0, halo); 3]);
        let mut inputs = vec![
            self.labels.source().source_input(reach.clone()),
            self.evidence.source().source_input(reach.clone()),
        ];
        if let Some(intensity) = self.intensity {
            inputs.push(intensity.source().source_input(reach));
        }
        inputs
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        // At most three owned faces and one node per voxel looked at.
        let per_voxel = 3 * rag_edge_bytes(self.options.bins) + RAG_NODE_BYTES;
        vec![
            FragmentOutput::new(self.stream.clone(), self.lifecycle, Coverage::EveryBlock).sized(
                SidecarSize::per_read_voxel(RAG_HEADER_BYTES as u64, per_voxel as u64),
            ),
        ]
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "rag phases: a block's graph is read from the supervoxel and evidence images and \
             cannot be computed without them. It is applied through `apply_with`.",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let labels = sources.get(self.labels.image.index())?;
        let evidence = sources.get(self.evidence.image.index())?;
        let intensity = match self.intensity {
            Some(image) => Some(sources.get(image.image.index())?),
            None => None,
        };
        let rag = self.block_rag(labels, evidence, intensity, at.read, at.core)?;
        Ok(BlockOutput::fragment(self.stream.clone(), rag.to_bytes()))
    }
}

/// Where [`append_rag_phases`] put its output: the phase whose fragments hold
/// the per-block graphs, the stream, and the options they were accumulated
/// under. What [`collect_rag`] reads.
#[derive(Debug, Clone)]
pub struct RagPhases {
    pub phase: Phase,
    pub stream: String,
    pub options: RagOptions,
}

/// Append the emit phase to `builder` on its current lattice.
///
/// The phase writes no image, so it ends what the plan can do with pixels:
/// put it last, and run the relabelling as its own plan
/// ([`append_relabel_phase`]). The images may be image 0, an image an earlier
/// phase wrote, or a supplied input (`ImageId::supplied`).
pub fn append_rag_phases(
    builder: &mut PlanBuilder,
    labels: RagImage,
    evidence: RagImage,
    intensity: Option<RagImage>,
    options: &RagOptions,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
) -> Result<RagPhases> {
    let op = RagBlockOp::new(
        labels,
        evidence,
        intensity,
        options.clone(),
        stream,
        lifecycle,
    )?;
    let stream = op.stream.clone();
    let phase = builder.fragments(op)?;
    Ok(RagPhases {
        phase,
        stream,
        options: options.clone(),
    })
}

// ---------------------------------------------------------------- reduce --

/// The whole-volume graph, folded from the fragments [`append_rag_phases`]
/// left in `env`, one fragment resident at a time.
///
/// Refuses a fragment accumulated under different options (another run's
/// stream of the same name), and a stream with no fragment of this phase at
/// all — under `Coverage::EveryBlock` that is a run that did not happen, not
/// an empty graph.
pub fn collect_rag(env: &dyn Environment, phases: &RagPhases) -> Result<Rag> {
    let phase = phases.phase.index();
    let mut merged = Rag::new(phases.options.clone());
    let mut seen = 0usize;
    fold_fragments(env, &phases.stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        merged.try_merge(Rag::from_bytes(bytes)?)?;
        seen += 1;
        Ok(())
    })?;
    if seen == 0 {
        return Err(Error::invalid(format!(
            "rag phases: stream {:?} holds no fragment of phase {phase}; was the plan run in \
             this environment?",
            phases.stream
        )));
    }
    Ok(merged)
}

// --------------------------------------------------------------- relabel --

/// Supervoxel ids -> cluster ids, per voxel, through a shared [`LabelLookup`].
///
/// A stencil of reach 0: the output at a voxel is the table applied to that
/// voxel. Label 0 and labels absent from the table map to 0.
#[derive(Clone)]
pub struct RelabelOp {
    table: Arc<LabelLookup>,
}

impl RelabelOp {
    pub fn new(table: Arc<LabelLookup>) -> Self {
        Self { table }
    }

    /// From a [`super::rag::RagGraph`]'s labels and a partition of its nodes
    /// (any of `ops::agglomerate`'s solvers).
    pub fn from_partition(labels: &[u32], clusters: &[u32]) -> Result<Self> {
        Ok(Self::new(Arc::new(LabelLookup::new(labels, clusters)?)))
    }

    pub fn table(&self) -> &LabelLookup {
        &self.table
    }
}

impl BlockOp for RelabelOp {
    fn name(&self) -> &'static str {
        "rag-relabel"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::U32
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.view::<u32>()?;
        let target = out.view_mut::<u32>()?;
        self.table.apply_into(source, target)
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value >= 0.0 && value.fract() == 0.0 && value <= u32::MAX as f64)
            .then(|| self.table.get(value as u32) as f64)
    }

    fn cost_per_voxel(&self) -> f64 {
        0.5
    }
}

/// Append the relabelling as one pixel phase. The builder's current image
/// must be the supervoxel image (`u32`), e.g. image 0 of a plan built over it.
pub fn append_relabel_phase(builder: &mut PlanBuilder, op: RelabelOp) -> Result<Phase> {
    builder.pixels(Chain::op(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(shape: (usize, usize, usize)) -> (Array3<u32>, Array3<f64>) {
        let labels = Array3::from_shape_fn(shape, |(z, y, x)| {
            if (x + 2 * y + z) % 7 == 0 {
                0
            } else {
                ((z / 2) * 9 + (y / 3) * 3 + x / 3) as u32 % 11 + 1
            }
        });
        let evidence =
            Array3::from_shape_fn(shape, |(z, y, x)| ((z * 5 + y * 3 + x) % 17) as f64 / 17.0);
        (labels, evidence)
    }

    /// `block_rag` over a hand-made tiling with symmetric halos (what
    /// `fragment_phase` grants) reproduces the whole-volume graph, for
    /// non-`u32` label and non-`f64` evidence buffers too.
    #[test]
    fn block_rag_tiles_to_the_whole_graph() {
        let shape = (7, 8, 9);
        let (labels, evidence) = fixture(shape);
        for bridge in [false, true] {
            let options = RagOptions::new([2.0, 1.0, 1.0], 5, (0.0, 1.0)).bridging_lines(bridge);
            let whole =
                super::super::rag::rag(labels.view(), evidence.view(), None, &options).unwrap();
            let op = RagBlockOp::new(
                RagImage::new(0usize, Dtype::U16),
                RagImage::new(ImageId::supplied(0), Dtype::F32),
                None,
                options.clone(),
                "rag",
                Lifecycle::DeleteOnExit,
            )
            .unwrap();
            let halo = options.halo();
            let dims = [shape.0, shape.1, shape.2];
            let block = [3, 4, 4];
            let mut merged = Rag::new(options.clone());
            for z0 in (0..dims[0]).step_by(block[0]) {
                for y0 in (0..dims[1]).step_by(block[1]) {
                    for x0 in (0..dims[2]).step_by(block[2]) {
                        let start = [z0, y0, x0];
                        let core_shape: [usize; 3] =
                            std::array::from_fn(|a| block[a].min(dims[a] - start[a]));
                        let lo: [usize; 3] = std::array::from_fn(|a| start[a].saturating_sub(halo));
                        let hi: [usize; 3] =
                            std::array::from_fn(|a| (start[a] + core_shape[a] + halo).min(dims[a]));
                        let read_shape: [usize; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
                        let window = s![lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]];
                        let l: Voxels = labels.slice(window).mapv(|v| v as u16).into();
                        let e: Voxels = evidence.slice(window).mapv(|v| v as f32).into();
                        let part = op
                            .block_rag(
                                &BlockBuf::Array(l),
                                &BlockBuf::Array(e),
                                None,
                                &Region::new(&lo, &read_shape),
                                &Region::new(&start, &core_shape),
                            )
                            .unwrap();
                        let part = Rag::from_bytes(&part.to_bytes()).unwrap();
                        merged.try_merge(part).unwrap();
                    }
                }
            }
            assert_eq!(merged.edge_count(), whole.edge_count());
            assert_eq!(merged.node_count(), whole.node_count());
            for i in 0..whole.edge_count() {
                let ((a, b), w) = whole.edge(i);
                let j = merged.find_edge(a, b).unwrap();
                let (_, m) = merged.edge(j);
                assert_eq!(
                    (m.faces, m.axis_faces, m.bridged),
                    (w.faces, w.axis_faces, w.bridged)
                );
                assert_eq!(merged.histogram(j), whole.histogram(i));
                // f32 evidence: the sums agree to f32 rounding.
                assert!((m.sum - w.sum).abs() < 1e-5 * (1.0 + w.sum.abs()));
            }
        }
    }

    #[test]
    fn construction_refuses_what_it_cannot_mean() {
        let options = RagOptions::new([1.0; 3], 0, (0.0, 1.0));
        let mk = |labels: RagImage, evidence: RagImage| {
            RagBlockOp::new(
                labels,
                evidence,
                None,
                options.clone(),
                "s",
                Lifecycle::DeleteOnExit,
            )
        };
        assert!(mk(
            RagImage::new(0usize, Dtype::F32),
            RagImage::new(1usize, Dtype::F64)
        )
        .is_err());
        assert!(mk(
            RagImage::new(0usize, Dtype::U32),
            RagImage::new(0usize, Dtype::F64)
        )
        .is_err());
        assert!(RagBlockOp::new(
            RagImage::new(0usize, Dtype::U32),
            RagImage::new(1usize, Dtype::F64),
            None,
            RagOptions::new([0.0, 1.0, 1.0], 0, (0.0, 1.0)),
            "s",
            Lifecycle::DeleteOnExit
        )
        .is_err());
    }

    #[test]
    fn relabel_op_maps_through_the_table() {
        let op = RelabelOp::from_partition(&[3, 5, 9], &[1, 2, 1]).unwrap();
        let input: Voxels = Array3::from_shape_vec((1, 2, 3), vec![0u32, 3, 5, 9, 7, 3])
            .unwrap()
            .into();
        let mut out = Voxels::zeros(Dtype::U32, [1, 2, 3]).unwrap();
        op.apply(&input, &mut out, &Anchor::whole([1, 2, 3]))
            .unwrap();
        assert_eq!(
            out.view::<u32>()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 1, 0, 1]
        );
        assert_eq!(op.constant_maps_to(5.0), Some(2.0));
        assert_eq!(op.constant_maps_to(0.5), None);
        assert!(RelabelOp::from_partition(&[3, 3], &[1, 2]).is_err());
    }
}
