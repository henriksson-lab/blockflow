// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Whole-volume label and mask cleanup rules.
//!
//! These are the data-level definitions. Distributed forms should reuse these
//! rules after their component facts have been merged, rather than defining a
//! second idea of which labels touch a border or which component is small.

use std::collections::{BTreeMap, HashMap, HashSet};

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SourceBlocks,
};
use crate::op::{Anchor, BlockOp, Chain, Slicing, SourceInput};
use crate::reach::Reach;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

use super::components::{
    decode_block_flags_for, encode_block_flags, walk_seams_with, Connectivity, LabelIndex, Union,
};
use super::detect::{label_regions_into_with, moments_of_labels, LabelRegionsOp, RegionMoments};
use super::fill::{as_mask, label_background_into_with, outside_flags};
use super::shapes_agree;

/// Remove every foreground component that touches a volume face.
pub fn clear_border_into(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    clear_border_on_axes_into(mask, connectivity, [true, true, true], out)
}

/// Remove foreground components that touch a selected set of volume faces.
///
/// This is useful for 2-D images carried in a `z=1` volume: callers can clear
/// the image-plane border with `[false, true, true]` without treating the
/// singleton z faces as border contact.
pub fn clear_border_on_axes_into(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
    axes: [bool; 3],
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(mask.shape(), out.shape(), "clear_border_into")?;
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_regions_into_with(mask, connectivity, labels.view_mut())?;
    let touching = labels_touching_border_on_axes(labels.view(), count, axes)?;
    let keep: Vec<bool> = touching.into_iter().map(|touches| !touches).collect();
    rewrite_labels_by_keep(labels.view(), &keep, out)
}

/// Remove foreground components with fewer than `minimum_size` voxels.
pub fn remove_small_objects_into(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
    minimum_size: u64,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(mask.shape(), out.shape(), "remove_small_objects_into")?;
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_regions_into_with(mask, connectivity, labels.view_mut())?;
    let moments = moments_of_labels(labels.view(), count, [0, 0, 0])?;
    let keep: Vec<bool> = moments
        .iter()
        .map(|moment| moment.count >= minimum_size)
        .collect();
    rewrite_labels_by_keep(labels.view(), &keep, out)
}

/// Remove non-zero labels that touch any selected volume face.
///
/// Unlike [`clear_border_on_axes_into`], this operates on an already-labelled
/// image. That distinction matters for watershed-style object identification:
/// a foreground component can touch the image border before declumping, while
/// the final split objects inside it do not.
pub fn filter_labels_touching_border_on_axes_into(
    labels: ArrayView3<'_, u32>,
    axes: [bool; 3],
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    shapes_agree(
        labels.shape(),
        out.shape(),
        "filter_labels_touching_border_on_axes_into",
    )?;
    let border = label_set_touching_border_on_axes(labels, axes);
    for (slot, &label) in out.iter_mut().zip(labels.iter()) {
        *slot = if label == 0 || border.contains(&label) {
            0
        } else {
            label
        };
    }
    Ok(())
}

/// Fill zero-valued holes enclosed by each label inside that label's 2-D box.
///
/// This is a label-preserving rule for post-watershed cleanup. It is different
/// from binary hole filling: every label is considered independently, and only
/// background that cannot reach the label's own bounding-box edge without
/// crossing that label is assigned to the label.
pub fn fill_label_holes_2d_by_label_into(
    labels: ArrayView3<'_, u32>,
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    shapes_agree(
        labels.shape(),
        out.shape(),
        "fill_label_holes_2d_by_label_into",
    )?;
    if labels.shape()[0] != 1 {
        return Err(Error::invalid(
            "fill_label_holes_2d_by_label_into currently supports 2-D label images",
        ));
    }
    out.assign(&labels);
    let mut boxes = BTreeMap::<u32, ([usize; 2], [usize; 2])>::new();
    for ((_, y, x), &label) in labels.indexed_iter() {
        if label == 0 {
            continue;
        }
        boxes
            .entry(label)
            .and_modify(|(min, max)| {
                min[0] = min[0].min(y);
                min[1] = min[1].min(x);
                max[0] = max[0].max(y + 1);
                max[1] = max[1].max(x + 1);
            })
            .or_insert(([y, x], [y + 1, x + 1]));
    }
    for (label, (min, max)) in boxes {
        fill_label_holes_in_box(label, min, max, labels, out.view_mut());
    }
    Ok(())
}

fn fill_label_holes_in_box(
    label: u32,
    min: [usize; 2],
    max: [usize; 2],
    labels: ArrayView3<'_, u32>,
    mut out: ArrayViewMut3<'_, u32>,
) {
    let height = max[0] - min[0];
    let width = max[1] - min[1];
    if height < 3 || width < 3 {
        return;
    }
    let mut outside = vec![false; height * width];
    let mut queue = std::collections::VecDeque::<(usize, usize)>::new();
    for y in 0..height {
        for x in [0, width - 1] {
            enqueue_label_background(label, min, labels, &mut outside, &mut queue, width, y, x);
        }
    }
    for x in 0..width {
        for y in [0, height - 1] {
            enqueue_label_background(label, min, labels, &mut outside, &mut queue, width, y, x);
        }
    }
    while let Some((y, x)) = queue.pop_front() {
        for (ny, nx) in [
            y.checked_sub(1).map(|ny| (ny, x)),
            (y + 1 < height).then_some((y + 1, x)),
            x.checked_sub(1).map(|nx| (y, nx)),
            (x + 1 < width).then_some((y, x + 1)),
        ]
        .into_iter()
        .flatten()
        {
            enqueue_label_background(label, min, labels, &mut outside, &mut queue, width, ny, nx);
        }
    }
    for y in 0..height {
        for x in 0..width {
            let global = [0, min[0] + y, min[1] + x];
            if labels[global] == 0 && !outside[y * width + x] {
                out[global] = label;
            }
        }
    }
}

fn enqueue_label_background(
    label: u32,
    min: [usize; 2],
    labels: ArrayView3<'_, u32>,
    outside: &mut [bool],
    queue: &mut std::collections::VecDeque<(usize, usize)>,
    width: usize,
    y: usize,
    x: usize,
) {
    let index = y * width + x;
    if outside[index] || labels[[0, min[0] + y, min[1] + x]] == label {
        return;
    }
    outside[index] = true;
    queue.push_back((y, x));
}

const COMPONENT_MASK_MAGIC: u32 = 0x4d43_4d52;
const COMPONENT_MASK_NOUN: &str = "a component-mask reduction";
const LABEL_SIZE_COUNTS_MAGIC: u64 = 0x4c53_434e_5453_0001;
const LABEL_SIZE_KEEP_MAGIC: u64 = 0x4c53_4b45_4550_0001;

#[derive(Debug, Clone, Copy)]
enum ComponentMaskRule {
    MinimumSize { minimum_size: u64 },
    ClearBorder { volume: [usize; 3], axes: [bool; 3] },
}

impl ComponentMaskRule {
    fn keep(&self, moments: &super::detect::Moments) -> Result<bool> {
        match *self {
            Self::MinimumSize { minimum_size } => Ok(moments.count >= minimum_size),
            Self::ClearBorder { volume, axes } => {
                let Some((low, high)) = moments.bounds() else {
                    return Ok(false);
                };
                Ok((0..3)
                    .all(|axis| !axes[axis] || (low[axis] != 0 && high[axis] + 1 != volume[axis])))
            }
        }
    }
}

pub struct ApplyComponentMaskOp {
    name: &'static str,
    stream: String,
    phase: usize,
    rule: ComponentMaskRule,
    lattice: [usize; 3],
    connectivity: Connectivity,
    mask: ImageId,
    mask_dtype: Dtype,
}

impl ApplyComponentMaskOp {
    fn new(
        name: &'static str,
        stream: impl Into<String>,
        phase: impl Into<Phase>,
        rule: ComponentMaskRule,
        lattice: [usize; 3],
        mask: impl Into<ImageId>,
        mask_dtype: Dtype,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            phase: phase.into().index(),
            rule,
            lattice,
            connectivity: Connectivity::Faces,
            mask: mask.into(),
            mask_dtype,
        }
    }

    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }
}

impl FragmentOp for ApplyComponentMaskOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.phase).with_reach([0, 0, 0])]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.mask).holding(self.mask_dtype)]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(&self.stream)? {
            reports.insert(key.block, RegionMoments::decode(&bytes)?);
        }
        let index = LabelIndex::build(&reports, at.grid.blocks_per_axis(), |report| report.labels)?;
        let gathered = index.gather(
            &reports,
            |report| &report.moments,
            super::detect::Moments::EMPTY,
        );
        let mut sets = Union::new(index.total());
        walk_seams_with(
            &reports,
            at.grid.blocks_per_axis(),
            &index,
            self.connectivity,
            |report| &report.faces,
            |a, b| sets.union(a, b),
        )?;
        let mut root_moments = vec![super::detect::Moments::EMPTY; index.total()];
        for (node, moments) in gathered.iter().enumerate() {
            let root = sets.find(node);
            root_moments[root].merge(moments)?;
        }
        let root_keep: Vec<bool> = root_moments
            .iter()
            .map(|moments| self.rule.keep(moments))
            .collect::<Result<_>>()?;
        let keep = index.per_block(&mut sets, &root_keep);
        encode_block_flags(&keep, at.grid.blocks_per_axis(), COMPONENT_MASK_MAGIC)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "component-mask cleanup rewrites a declared mask source and is applied through \
             `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = sources.get(self.mask.index())? else {
            return Ok(BlockOutput::nothing());
        };
        let mask = as_mask(pixels)?;
        let mut labels = Array3::<u32>::zeros(mask.raw_dim());
        label_regions_into_with(mask.view(), self.connectivity, labels.view_mut())?;
        let keep = decode_block_flags_for(
            at.reduced,
            self.lattice,
            at.index,
            COMPONENT_MASK_MAGIC,
            COMPONENT_MASK_NOUN,
        )?;
        let mut out = Voxels::zeros(
            Dtype::Bool,
            [labels.shape()[0], labels.shape()[1], labels.shape()[2]],
        )?;
        rewrite_labels_by_keep(labels.view(), &keep, out.view_mut::<bool>()?)?;
        Ok(BlockOutput::nothing().with_pixels(BlockBuf::Array(out)))
    }
}

pub fn append_remove_small_objects_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
    minimum_size: u64,
) -> Result<Phase> {
    let stream = stream.into();
    let mask = ImageId::from(plan.n_phases());
    let mask_dtype = plan.reads();
    let labels = plan.fragments(
        LabelRegionsOp::new(
            "remove-small-objects component labelling",
            stream.clone(),
            lifecycle,
        )
        .connecting(connectivity),
    )?;
    let lattice = plan.grid().blocks_per_axis();
    plan.fragments(
        ApplyComponentMaskOp::new(
            "remove-small-objects rewrite",
            stream,
            labels,
            ComponentMaskRule::MinimumSize { minimum_size },
            lattice,
            mask,
            mask_dtype,
        )
        .connecting(connectivity),
    )
}

pub fn append_clear_border_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
) -> Result<Phase> {
    append_clear_border_on_axes_phases(plan, stream, lifecycle, connectivity, [true, true, true])
}

pub fn append_clear_border_on_axes_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
    axes: [bool; 3],
) -> Result<Phase> {
    let stream = stream.into();
    let mask = ImageId::from(plan.n_phases());
    let mask_dtype = plan.reads();
    let labels = plan.fragments(
        LabelRegionsOp::new(
            "clear-border component labelling",
            stream.clone(),
            lifecycle,
        )
        .connecting(connectivity),
    )?;
    let lattice = plan.grid().blocks_per_axis();
    let volume = plan.grid().volume();
    plan.fragments(
        ApplyComponentMaskOp::new(
            "clear-border rewrite",
            stream,
            labels,
            ComponentMaskRule::ClearBorder { volume, axes },
            lattice,
            mask,
            mask_dtype,
        )
        .connecting(connectivity),
    )
}

/// Count non-zero labels in every block and emit one sorted label/count table.
pub struct LabelSizeCountsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
}

impl LabelSizeCountsOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
        }
    }
}

impl FragmentOp for LabelSizeCountsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            Coverage::EveryBlock,
        )]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                encode_label_counts(&BTreeMap::new()),
            ));
        };
        let labels = pixels.view::<u32>()?;
        let mut counts = BTreeMap::<u32, u64>::new();
        for &label in labels.iter() {
            if label == 0 {
                continue;
            }
            let count = counts.entry(label).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                Error::invalid("label-size filter: per-block label count overflowed")
            })?;
        }
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_label_counts(&counts),
        ))
    }
}

/// Rewrite labels by a global size rule derived from block-local label counts.
pub struct ApplyLabelSizeFilterOp {
    name: &'static str,
    stream: String,
    phase: usize,
    min_size: u64,
    max_size: Option<u64>,
    labels: ImageId,
}

impl ApplyLabelSizeFilterOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        phase: impl Into<Phase>,
        labels: impl Into<ImageId>,
        min_size: u64,
        max_size: Option<u64>,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            phase: phase.into().index(),
            min_size,
            max_size,
            labels: labels.into(),
        }
    }

    fn keep_label(&self, count: u64) -> bool {
        count >= self.min_size && self.max_size.is_none_or(|limit| count <= limit)
    }
}

impl FragmentOp for ApplyLabelSizeFilterOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.phase).with_reach([0, 0, 0])]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.labels).holding(Dtype::U32)]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut totals = BTreeMap::<u32, u64>::new();
        for (_key, bytes) in at.fragments(&self.stream)? {
            for (label, count) in decode_label_counts(&bytes)? {
                let total = totals.entry(label).or_default();
                *total = total.checked_add(count).ok_or_else(|| {
                    Error::invalid("label-size filter: global label count overflowed")
                })?;
            }
        }
        let keep = totals
            .into_iter()
            .filter_map(|(label, count)| self.keep_label(count).then_some(label))
            .collect::<Vec<_>>();
        Ok(encode_label_keep_set(&keep))
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "label-size filter rewrites a declared label source and is applied through \
             `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = sources.get(self.labels.index())? else {
            return Ok(BlockOutput::nothing());
        };
        let labels = pixels.view::<u32>()?;
        let keep = decode_label_keep_set(at.reduced)?;
        let mut out = Voxels::zeros(
            Dtype::U32,
            [labels.shape()[0], labels.shape()[1], labels.shape()[2]],
        )?;
        {
            let mut out = out.view_mut::<u32>()?;
            for ((z, y, x), slot) in out.indexed_iter_mut() {
                let label = labels[[z, y, x]];
                if label != 0 && keep.contains(&label) {
                    *slot = label;
                }
            }
        }
        Ok(BlockOutput::nothing().with_pixels(BlockBuf::Array(out)))
    }
}

pub fn append_filter_labels_by_size_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    min_size: u64,
    max_size: Option<u64>,
) -> Result<Phase> {
    let stream = stream.into();
    let labels = ImageId::from(plan.n_phases());
    let counts = plan.fragments(LabelSizeCountsOp::new(
        "label-size-filter count labels",
        stream.clone(),
        lifecycle,
    ))?;
    plan.fragments(ApplyLabelSizeFilterOp::new(
        "label-size-filter rewrite",
        stream,
        counts,
        labels,
        min_size,
        max_size,
    ))
}

/// Planned label-image border cleanup after object splitting.
pub struct FilterLabelsTouchingBorderOnAxesOp {
    name: &'static str,
    labels: ImageId,
    axes: [bool; 3],
}

impl FilterLabelsTouchingBorderOnAxesOp {
    pub fn new(name: &'static str, labels: impl Into<ImageId>, axes: [bool; 3]) -> Self {
        Self {
            name,
            labels: labels.into(),
            axes,
        }
    }
}

impl FragmentOp for FilterLabelsTouchingBorderOnAxesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn cost_per_voxel(&self) -> f64 {
        2.0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(self.labels, Reach::all()).holding(Dtype::U32)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "label-border filtering rewrites a declared label source and is applied through \
             `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = sources.get(self.labels.index())? else {
            return Ok(BlockOutput::nothing());
        };
        let labels = pixels.view::<u32>()?;
        if labels.shape() != at.volume() {
            return Err(Error::invalid(format!(
                "{} needs a whole-volume label source shaped {:?}, got {:?}",
                self.name,
                at.volume(),
                labels.shape()
            )));
        }
        let border = label_set_touching_border_on_axes(labels, self.axes);
        let mut out = Voxels::zeros(Dtype::U32, at.read.shape3())?;
        {
            let mut out = out.view_mut::<u32>()?;
            for ((z, y, x), slot) in out.indexed_iter_mut() {
                let global = [
                    at.read.start[0] + z,
                    at.read.start[1] + y,
                    at.read.start[2] + x,
                ];
                let label = labels[global];
                *slot = if label == 0 || border.contains(&label) {
                    0
                } else {
                    label
                };
            }
        }
        Ok(BlockOutput::nothing().with_pixels(BlockBuf::Array(out)))
    }
}

pub fn append_filter_labels_touching_border_on_axes_phase(
    plan: &mut PlanBuilder,
    axes: [bool; 3],
) -> Result<Phase> {
    let labels = ImageId::from(plan.n_phases());
    plan.fragments(FilterLabelsTouchingBorderOnAxesOp::new(
        "filter-labels-touching-border",
        labels,
        axes,
    ))
}

/// Planned 2-D per-label hole filling over a label image.
pub struct FillLabelHoles2dByLabelOp {
    name: &'static str,
    labels: ImageId,
}

impl FillLabelHoles2dByLabelOp {
    pub fn new(name: &'static str, labels: impl Into<ImageId>) -> Self {
        Self {
            name,
            labels: labels.into(),
        }
    }
}

impl FragmentOp for FillLabelHoles2dByLabelOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn cost_per_voxel(&self) -> f64 {
        4.0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(self.labels, Reach::all()).holding(Dtype::U32)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "label-hole filling rewrites a declared label source and is applied through \
             `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = sources.get(self.labels.index())? else {
            return Ok(BlockOutput::nothing());
        };
        let labels = pixels.view::<u32>()?;
        if labels.shape() != at.volume() {
            return Err(Error::invalid(format!(
                "{} needs a whole-volume label source shaped {:?}, got {:?}",
                self.name,
                at.volume(),
                labels.shape()
            )));
        }
        let mut filled = Array3::<u32>::zeros(labels.raw_dim());
        fill_label_holes_2d_by_label_into(labels, filled.view_mut())?;
        let mut out = Voxels::zeros(Dtype::U32, at.read.shape3())?;
        {
            let mut out = out.view_mut::<u32>()?;
            for ((z, y, x), slot) in out.indexed_iter_mut() {
                let global = [
                    at.read.start[0] + z,
                    at.read.start[1] + y,
                    at.read.start[2] + x,
                ];
                *slot = filled[global];
            }
        }
        Ok(BlockOutput::nothing().with_pixels(BlockBuf::Array(out)))
    }
}

pub fn append_fill_label_holes_2d_by_label_phase(plan: &mut PlanBuilder) -> Result<Phase> {
    let labels = ImageId::from(plan.n_phases());
    plan.fragments(FillLabelHoles2dByLabelOp::new(
        "fill-label-holes-2d-by-label",
        labels,
    ))
}

fn encode_words(words: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(words));
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn decode_words(bytes: &[u8], noun: &'static str) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u64>()) {
        return Err(Error::invalid(format!(
            "label-size filter: {noun} byte length {} is not a whole number of words",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u64>())
        .map(|chunk| {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect())
}

fn encode_label_counts(counts: &BTreeMap<u32, u64>) -> Vec<u8> {
    let mut words = Vec::with_capacity(2 + counts.len() * 2);
    words.push(LABEL_SIZE_COUNTS_MAGIC);
    words.push(counts.len() as u64);
    for (&label, &count) in counts {
        words.push(u64::from(label));
        words.push(count);
    }
    encode_words(&words)
}

fn decode_label_counts(bytes: &[u8]) -> Result<Vec<(u32, u64)>> {
    let words = decode_words(bytes, "count fragment")?;
    if words.len() < 2 || words[0] != LABEL_SIZE_COUNTS_MAGIC {
        return Err(Error::invalid(
            "label-size filter: count fragment has the wrong magic",
        ));
    }
    let rows = usize::try_from(words[1]).map_err(|_| {
        Error::invalid("label-size filter: count fragment row count does not fit usize")
    })?;
    if words.len() != 2 + rows * 2 {
        return Err(Error::invalid(format!(
            "label-size filter: count fragment declares {rows} rows but has {} words",
            words.len()
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for pair in words[2..].chunks_exact(2) {
        let label = u32::try_from(pair[0])
            .map_err(|_| Error::invalid("label-size filter: label id does not fit uint32"))?;
        if label == 0 {
            return Err(Error::invalid(
                "label-size filter: count fragment contains background label 0",
            ));
        }
        out.push((label, pair[1]));
    }
    Ok(out)
}

fn encode_label_keep_set(labels: &[u32]) -> Vec<u8> {
    let mut words = Vec::with_capacity(2 + labels.len());
    words.push(LABEL_SIZE_KEEP_MAGIC);
    words.push(labels.len() as u64);
    words.extend(labels.iter().map(|&label| u64::from(label)));
    encode_words(&words)
}

fn decode_label_keep_set(bytes: &[u8]) -> Result<HashSet<u32>> {
    let words = decode_words(bytes, "keep-set reduction")?;
    if words.len() < 2 || words[0] != LABEL_SIZE_KEEP_MAGIC {
        return Err(Error::invalid(
            "label-size filter: keep-set reduction has the wrong magic",
        ));
    }
    let labels = usize::try_from(words[1]).map_err(|_| {
        Error::invalid("label-size filter: keep-set label count does not fit usize")
    })?;
    if words.len() != 2 + labels {
        return Err(Error::invalid(format!(
            "label-size filter: keep-set declares {labels} labels but has {} words",
            words.len()
        )));
    }
    let mut keep = HashSet::with_capacity(labels);
    for &word in &words[2..] {
        let label = u32::try_from(word).map_err(|_| {
            Error::invalid("label-size filter: keep-set label id does not fit uint32")
        })?;
        if label == 0 {
            return Err(Error::invalid(
                "label-size filter: keep-set contains background label 0",
            ));
        }
        keep.insert(label);
    }
    Ok(keep)
}

pub struct LabelBackgroundRegionsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
}

impl LabelBackgroundRegionsOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            connectivity: Connectivity::Faces,
        }
    }

    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }
}

impl FragmentOp for LabelBackgroundRegionsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<crate::fragment::FragmentOutput> {
        vec![crate::fragment::FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            crate::fragment::Coverage::EveryBlock,
        )
        .sized(crate::fragment::SidecarSize::component_report())]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                RegionMoments::empty().encode(),
            ));
        };
        let mask = as_mask(pixels)?;
        let mut labels = Array3::<u32>::zeros(mask.raw_dim());
        let count = label_background_into_with(mask.view(), self.connectivity, labels.view_mut())?;
        let moments = moments_of_labels(labels.view(), count, at.at.offset)?;
        let report = RegionMoments::of(labels.view(), count, moments)?;
        Ok(BlockOutput::fragment(self.stream.clone(), report.encode()))
    }
}

pub struct ApplyRemoveSmallHolesOp {
    name: &'static str,
    stream: String,
    phase: usize,
    area_threshold: u64,
    lattice: [usize; 3],
    volume: [usize; 3],
    connectivity: Connectivity,
    mask: ImageId,
    mask_dtype: Dtype,
}

impl ApplyRemoveSmallHolesOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        phase: impl Into<Phase>,
        area_threshold: u64,
        lattice: [usize; 3],
        volume: [usize; 3],
        mask: impl Into<ImageId>,
        mask_dtype: Dtype,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            phase: phase.into().index(),
            area_threshold,
            lattice,
            volume,
            connectivity: Connectivity::Faces,
            mask: mask.into(),
            mask_dtype,
        }
    }

    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }
}

impl FragmentOp for ApplyRemoveSmallHolesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.phase).with_reach([0, 0, 0])]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.mask).holding(self.mask_dtype)]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(&self.stream)? {
            reports.insert(key.block, RegionMoments::decode(&bytes)?);
        }
        let index = LabelIndex::build(&reports, at.grid.blocks_per_axis(), |report| report.labels)?;
        let gathered = index.gather(
            &reports,
            |report| &report.moments,
            super::detect::Moments::EMPTY,
        );
        let mut sets = Union::new(index.total());
        walk_seams_with(
            &reports,
            at.grid.blocks_per_axis(),
            &index,
            self.connectivity,
            |report| &report.faces,
            |a, b| sets.union(a, b),
        )?;
        let mut root_moments = vec![super::detect::Moments::EMPTY; index.total()];
        for (node, moments) in gathered.iter().enumerate() {
            let root = sets.find(node);
            root_moments[root].merge(moments)?;
        }
        let root_fill: Vec<bool> = root_moments
            .iter()
            .map(|moments| {
                let Some((low, high)) = moments.bounds() else {
                    return false;
                };
                let touches_border =
                    (0..3).any(|axis| low[axis] == 0 || high[axis] + 1 == self.volume[axis]);
                !touches_border && moments.count < self.area_threshold
            })
            .collect();
        let fill = index.per_block(&mut sets, &root_fill);
        encode_block_flags(&fill, at.grid.blocks_per_axis(), COMPONENT_MASK_MAGIC)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "small-hole cleanup rewrites a declared mask source and is applied through \
             `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = sources.get(self.mask.index())? else {
            return Ok(BlockOutput::nothing());
        };
        let mask = as_mask(pixels)?;
        let mut labels = Array3::<u32>::zeros(mask.raw_dim());
        label_background_into_with(mask.view(), self.connectivity, labels.view_mut())?;
        let fill = decode_block_flags_for(
            at.reduced,
            self.lattice,
            at.index,
            COMPONENT_MASK_MAGIC,
            COMPONENT_MASK_NOUN,
        )?;
        let mut out = Voxels::zeros(
            Dtype::Bool,
            [labels.shape()[0], labels.shape()[1], labels.shape()[2]],
        )?;
        fill_small_holes_from_labels(labels.view(), &fill, out.view_mut::<bool>()?)?;
        Ok(BlockOutput::nothing().with_pixels(BlockBuf::Array(out)))
    }
}

/// Fill background components with fewer than `area_threshold` voxels.
///
/// Components that reach the volume outside are never holes and are not filled.
pub fn remove_small_holes_into(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
    area_threshold: u64,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(mask.shape(), out.shape(), "remove_small_holes_into")?;
    let volume = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_background_into_with(mask, connectivity, labels.view_mut())?;
    let outside = outside_flags(labels.view(), count, [0, 0, 0], volume, volume);
    let sizes = label_sizes(labels.view(), count)?;
    let fill: Vec<bool> = outside
        .iter()
        .zip(sizes)
        .map(|(&touches_outside, size)| !touches_outside && size < area_threshold)
        .collect();
    fill_small_holes_from_labels(labels.view(), &fill, out)
}

pub fn append_remove_small_holes_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
    area_threshold: u64,
) -> Result<Phase> {
    let stream = stream.into();
    let mask = ImageId::from(plan.n_phases());
    let mask_dtype = plan.reads();
    let labels = plan.fragments(
        LabelBackgroundRegionsOp::new(
            "remove-small-holes background labelling",
            stream.clone(),
            lifecycle,
        )
        .connecting(connectivity),
    )?;
    let lattice = plan.grid().blocks_per_axis();
    let volume = plan.grid().volume();
    plan.fragments(
        ApplyRemoveSmallHolesOp::new(
            "remove-small-holes rewrite",
            stream,
            labels,
            area_threshold,
            lattice,
            volume,
            mask,
            mask_dtype,
        )
        .connecting(connectivity),
    )
}

/// Expand non-zero labels into nearby zero-valued voxels by Euclidean distance.
///
/// A tie is resolved by the smaller label value, then lexicographic source
/// coordinate. Existing labels are copied unchanged.
pub fn expand_labels_into<T>(
    labels: ArrayView3<'_, T>,
    distance: f64,
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: Copy + Default + PartialEq + Ord,
{
    shapes_agree(labels.shape(), out.shape(), "expand_labels_into")?;
    if !distance.is_finite() || distance < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "expand_labels_into needs a finite non-negative distance, got {distance}"
        )));
    }
    let zero = T::default();
    let radius = distance.floor() as isize;
    let limit2 = distance * distance;
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    let sources: Vec<([usize; 3], T)> = labels
        .indexed_iter()
        .filter_map(|((i, j, k), &label)| (label != zero).then_some(([i, j, k], label)))
        .collect();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let current = labels[[i, j, k]];
                if current != zero {
                    out[[i, j, k]] = current;
                    continue;
                }
                let mut best: Option<(f64, T, [usize; 3])> = None;
                for &(source, label) in &sources {
                    if (source[0] as isize - i as isize).abs() > radius
                        || (source[1] as isize - j as isize).abs() > radius
                        || (source[2] as isize - k as isize).abs() > radius
                    {
                        continue;
                    }
                    let d0 = source[0] as f64 - i as f64;
                    let d1 = source[1] as f64 - j as f64;
                    let d2 = source[2] as f64 - k as f64;
                    let distance2 = d0 * d0 + d1 * d1 + d2 * d2;
                    if distance2 > limit2 {
                        continue;
                    }
                    let candidate = (distance2, label, source);
                    if best.map(|held| candidate < held).unwrap_or(true) {
                        best = Some(candidate);
                    }
                }
                out[[i, j, k]] = best.map(|(_, label, _)| label).unwrap_or(zero);
            }
        }
    }
    Ok(())
}

pub struct ExpandLabelsOp {
    name: &'static str,
    distance: f64,
    cost: f64,
}

impl ExpandLabelsOp {
    pub fn new(name: &'static str, distance: f64) -> Result<Self> {
        validate_expand_distance(distance)?;
        Ok(Self {
            name,
            distance,
            cost: expand_labels_cost(distance),
        })
    }

    pub fn distance(&self) -> f64 {
        self.distance
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    fn radius(&self) -> usize {
        self.distance.floor() as usize
    }
}

impl BlockOp for ExpandLabelsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.radius()
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([self.radius(); 3])
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(
            dtype,
            Dtype::U8
                | Dtype::U16
                | Dtype::U32
                | Dtype::U64
                | Dtype::I8
                | Dtype::I16
                | Dtype::I32
                | Dtype::I64
        )
    }

    fn produces(&self, input: Dtype) -> Dtype {
        input
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        macro_rules! run {
            ($ty:ty) => {
                expand_labels_into(input.view::<$ty>()?, self.distance, out.view_mut::<$ty>()?)
            };
        }
        match input.dtype() {
            Dtype::U8 => run!(u8),
            Dtype::U16 => run!(u16),
            Dtype::U32 => run!(u32),
            Dtype::U64 => run!(u64),
            Dtype::I8 => run!(i8),
            Dtype::I16 => run!(i16),
            Dtype::I32 => run!(i32),
            Dtype::I64 => run!(i64),
            dtype => Err(Error::InvalidArgument(format!(
                "{} accepts integer label images, not {}",
                self.name,
                dtype.numpy_name()
            ))),
        }
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

pub fn append_expand_labels_phase(plan: &mut PlanBuilder, distance: f64) -> Result<Phase> {
    plan.pixels(Chain::op(ExpandLabelsOp::new("expand labels", distance)?))
}

fn validate_expand_distance(distance: f64) -> Result<()> {
    if !distance.is_finite() || distance < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "expand labels needs a finite non-negative distance, got {distance}"
        )));
    }
    Ok(())
}

fn expand_labels_cost(distance: f64) -> f64 {
    let width = 2.0 * distance.floor() + 1.0;
    width * width * width * EXPAND_LABELS_COST_PER_CANDIDATE
}

const EXPAND_LABELS_COST_PER_CANDIDATE: f64 = 2.0;

const OBJECT_DISTANCE_SAMPLES_MAGIC: u64 = 0x4f44_5052_5341_4d50;
const OBJECT_DISTANCE_REMOVE_MAGIC: u64 = 0x4f44_5052_524d_4f56;
const OBJECT_DISTANCE_NOUN: &str = "an object-distance pruning reduction";

pub struct ObjectDistanceSamplesOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
}

impl ObjectDistanceSamplesOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
        }
    }
}

impl FragmentOp for ObjectDistanceSamplesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<crate::fragment::FragmentOutput> {
        vec![crate::fragment::FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            crate::fragment::Coverage::EveryBlock,
        )
        .sized(crate::fragment::SidecarSize::per_read_voxel(16, 40))]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                encode_object_distance_samples(&[]),
            ));
        };
        let samples = object_distance_samples_from_voxels(pixels, at.at.offset)?;
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_object_distance_samples(&samples),
        ))
    }
}

pub struct ApplyObjectDistancePruneOp {
    name: &'static str,
    stream: String,
    phase: usize,
    distance: f64,
    labels: ImageId,
    labels_dtype: Dtype,
}

impl ApplyObjectDistancePruneOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        phase: impl Into<Phase>,
        distance: f64,
        labels: impl Into<ImageId>,
        labels_dtype: Dtype,
    ) -> Result<Self> {
        validate_expand_distance(distance)?;
        Ok(Self {
            name,
            stream: stream.into(),
            phase: phase.into().index(),
            distance,
            labels: labels.into(),
            labels_dtype,
        })
    }
}

impl FragmentOp for ApplyObjectDistancePruneOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, input: Dtype) -> Dtype {
        input
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.phase).with_reach([0, 0, 0])]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.labels).holding(self.labels_dtype)]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut by_label: HashMap<i128, Vec<[usize; 3]>> = HashMap::new();
        at.stream_fragments(&self.stream, &mut |_key, bytes| {
            for (label, position) in decode_object_distance_samples(bytes)? {
                by_label.entry(label).or_default().push(position);
            }
            Ok(())
        })?;
        let radius = self.distance.floor() as isize;
        let limit2 = self.distance * self.distance;
        let mut labels: Vec<i128> = by_label.keys().copied().collect();
        labels.sort_unstable();
        let mut remove = HashSet::new();
        for (left_index, &left) in labels.iter().enumerate() {
            for &right in &labels[left_index + 1..] {
                if remove.contains(&left) || remove.contains(&right) {
                    continue;
                }
                if labels_are_close(&by_label[&left], &by_label[&right], radius, limit2) {
                    remove.insert(right);
                }
            }
        }
        let mut remove: Vec<i128> = remove.into_iter().collect();
        remove.sort_unstable();
        encode_object_distance_remove(&remove)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "object-distance pruning rewrites a declared label source and is applied through \
             `apply_with`."
                .to_string(),
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let labels = sources.get(self.labels.index())?.as_array()?;
        let remove = decode_object_distance_remove(at.reduced)?;
        let shape = labels.shape();
        let mut out = Voxels::zeros(self.labels_dtype, [shape[0], shape[1], shape[2]])?;
        prune_labels_by_encoded_set(labels, &remove, &mut out)?;
        Ok(BlockOutput::nothing().with_pixels(BlockBuf::Array(out)))
    }
}

pub fn append_object_distance_prune_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    distance: f64,
) -> Result<Phase> {
    let stream = stream.into();
    let labels = ImageId::from(plan.n_phases());
    let labels_dtype = plan.reads();
    let samples = plan.fragments(ObjectDistanceSamplesOp::new(
        "object-distance label samples",
        stream.clone(),
        lifecycle,
    ))?;
    plan.fragments(ApplyObjectDistancePruneOp::new(
        "object-distance prune rewrite",
        stream,
        samples,
        distance,
        labels,
        labels_dtype,
    )?)
}

fn object_distance_samples_from_voxels(
    pixels: &Voxels,
    offset: [usize; 3],
) -> Result<Vec<(i128, [usize; 3])>> {
    macro_rules! run_signed {
        ($ty:ty) => {{
            let labels = pixels.view::<$ty>()?;
            let mut samples = Vec::new();
            for ((i, j, k), &label) in labels.indexed_iter() {
                if label != 0 {
                    samples.push((label as i128, [offset[0] + i, offset[1] + j, offset[2] + k]));
                }
            }
            Ok(samples)
        }};
    }
    macro_rules! run_unsigned {
        ($ty:ty) => {{
            let labels = pixels.view::<$ty>()?;
            let mut samples = Vec::new();
            for ((i, j, k), &label) in labels.indexed_iter() {
                if label != 0 {
                    let label = i128::try_from(label).map_err(|_| {
                        Error::InvalidArgument(format!(
                            "object-distance pruning cannot encode label value {label}"
                        ))
                    })?;
                    samples.push((label, [offset[0] + i, offset[1] + j, offset[2] + k]));
                }
            }
            Ok(samples)
        }};
    }
    match pixels.dtype() {
        Dtype::U8 => run_unsigned!(u8),
        Dtype::U16 => run_unsigned!(u16),
        Dtype::U32 => run_unsigned!(u32),
        Dtype::U64 => run_unsigned!(u64),
        Dtype::I8 => run_signed!(i8),
        Dtype::I16 => run_signed!(i16),
        Dtype::I32 => run_signed!(i32),
        Dtype::I64 => run_signed!(i64),
        dtype => Err(Error::InvalidArgument(format!(
            "object-distance pruning accepts integer label images, not {}",
            dtype.numpy_name()
        ))),
    }
}

fn prune_labels_by_encoded_set(
    labels: &Voxels,
    remove: &HashSet<i128>,
    out: &mut Voxels,
) -> Result<()> {
    macro_rules! run_signed {
        ($ty:ty) => {{
            let labels = labels.view::<$ty>()?;
            let mut out = out.view_mut::<$ty>()?;
            for (slot, &label) in out.iter_mut().zip(labels.iter()) {
                *slot = if label != 0 && remove.contains(&(label as i128)) {
                    0
                } else {
                    label
                };
            }
            Ok(())
        }};
    }
    macro_rules! run_unsigned {
        ($ty:ty) => {{
            let labels = labels.view::<$ty>()?;
            let mut out = out.view_mut::<$ty>()?;
            for (slot, &label) in out.iter_mut().zip(labels.iter()) {
                let encoded = i128::try_from(label).ok();
                *slot = if encoded
                    .map(|value| remove.contains(&value))
                    .unwrap_or(false)
                {
                    0
                } else {
                    label
                };
            }
            Ok(())
        }};
    }
    match labels.dtype() {
        Dtype::U8 => run_unsigned!(u8),
        Dtype::U16 => run_unsigned!(u16),
        Dtype::U32 => run_unsigned!(u32),
        Dtype::U64 => run_unsigned!(u64),
        Dtype::I8 => run_signed!(i8),
        Dtype::I16 => run_signed!(i16),
        Dtype::I32 => run_signed!(i32),
        Dtype::I64 => run_signed!(i64),
        dtype => Err(Error::InvalidArgument(format!(
            "object-distance pruning accepts integer label images, not {}",
            dtype.numpy_name()
        ))),
    }
}

fn encode_object_distance_samples(samples: &[(i128, [usize; 3])]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + samples.len() * 40);
    bytes.extend_from_slice(&OBJECT_DISTANCE_SAMPLES_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&(samples.len() as u64).to_le_bytes());
    for &(label, at) in samples {
        bytes.extend_from_slice(&label.to_le_bytes());
        for coordinate in at {
            bytes.extend_from_slice(&(coordinate as u64).to_le_bytes());
        }
    }
    bytes
}

fn decode_object_distance_samples(bytes: &[u8]) -> Result<Vec<(i128, [usize; 3])>> {
    let mut cursor = ObjectDistanceCursor::new(bytes, "object-distance label samples");
    cursor.expect_magic(OBJECT_DISTANCE_SAMPLES_MAGIC)?;
    let count = cursor.take_u64()? as usize;
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let label = cursor.take_i128()?;
        let at = [
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
        ];
        samples.push((label, at));
    }
    cursor.expect_end()?;
    Ok(samples)
}

fn encode_object_distance_remove(labels: &[i128]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(16 + labels.len() * 16);
    bytes.extend_from_slice(&OBJECT_DISTANCE_REMOVE_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&(labels.len() as u64).to_le_bytes());
    for &label in labels {
        bytes.extend_from_slice(&label.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_object_distance_remove(bytes: &[u8]) -> Result<HashSet<i128>> {
    let mut cursor = ObjectDistanceCursor::new(bytes, OBJECT_DISTANCE_NOUN);
    cursor.expect_magic(OBJECT_DISTANCE_REMOVE_MAGIC)?;
    let count = cursor.take_u64()? as usize;
    let mut labels = HashSet::with_capacity(count);
    for _ in 0..count {
        labels.insert(cursor.take_i128()?);
    }
    cursor.expect_end()?;
    Ok(labels)
}

struct ObjectDistanceCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    noun: &'static str,
}

impl<'a> ObjectDistanceCursor<'a> {
    fn new(bytes: &'a [u8], noun: &'static str) -> Self {
        Self {
            bytes,
            offset: 0,
            noun,
        }
    }

    fn expect_magic(&mut self, expected: u64) -> Result<()> {
        let found = self.take_u64()?;
        if found != expected {
            return Err(Error::InvalidArgument(format!(
                "{} has magic {found:#x}, expected {expected:#x}",
                self.noun
            )));
        }
        Ok(())
    }

    fn take_u64(&mut self) -> Result<u64> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn take_i128(&mut self) -> Result<i128> {
        let bytes = self.take(16)?;
        Ok(i128::from_le_bytes(
            bytes.try_into().expect("sixteen bytes"),
        ))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| Error::InvalidArgument(format!("{} cursor overflowed", self.noun)))?;
        if end > self.bytes.len() {
            return Err(Error::InvalidArgument(format!(
                "{} ended early at byte {}, needed {} more byte(s)",
                self.noun, self.offset, len
            )));
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn expect_end(&self) -> Result<()> {
        if self.offset != self.bytes.len() {
            return Err(Error::InvalidArgument(format!(
                "{} has {} trailing byte(s)",
                self.noun,
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }
}

fn labels_touching_border_on_axes(
    labels: ArrayView3<'_, u32>,
    count: u32,
    axes: [bool; 3],
) -> Result<Vec<bool>> {
    let mut touching = vec![false; count as usize];
    let shape = labels.shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let touches_selected_axis = (axes[0] && (i == 0 || i + 1 == shape[0]))
                    || (axes[1] && (j == 0 || j + 1 == shape[1]))
                    || (axes[2] && (k == 0 || k + 1 == shape[2]));
                if !touches_selected_axis {
                    continue;
                }
                let label = labels[[i, j, k]];
                if label == 0 {
                    continue;
                }
                *touching.get_mut(label as usize - 1).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "label {label} is outside the {count} label(s) this array was said to hold"
                    ))
                })? = true;
            }
        }
    }
    Ok(touching)
}

fn label_set_touching_border_on_axes(labels: ArrayView3<'_, u32>, axes: [bool; 3]) -> HashSet<u32> {
    let mut touching = HashSet::<u32>::new();
    let shape = labels.shape();
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let touches_selected_axis = (axes[0] && (z == 0 || z + 1 == shape[0]))
                    || (axes[1] && (y == 0 || y + 1 == shape[1]))
                    || (axes[2] && (x == 0 || x + 1 == shape[2]));
                if !touches_selected_axis {
                    continue;
                }
                let label = labels[[z, y, x]];
                if label != 0 {
                    touching.insert(label);
                }
            }
        }
    }
    touching
}

fn label_sizes(labels: ArrayView3<'_, u32>, count: u32) -> Result<Vec<u64>> {
    let mut sizes = vec![0u64; count as usize];
    for &label in labels {
        if label == 0 {
            continue;
        }
        *sizes.get_mut(label as usize - 1).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "label {label} is outside the {count} label(s) this array was said to hold"
            ))
        })? += 1;
    }
    Ok(sizes)
}

fn rewrite_labels_by_keep(
    labels: ArrayView3<'_, u32>,
    keep: &[bool],
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    for (slot, &label) in out.iter_mut().zip(labels.iter()) {
        *slot = if label == 0 {
            false
        } else {
            *keep.get(label as usize - 1).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "label {label} has no entry in a {}-label keep map",
                    keep.len()
                ))
            })?
        };
    }
    Ok(())
}

fn fill_small_holes_from_labels(
    labels: ArrayView3<'_, u32>,
    fill: &[bool],
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    for (slot, &label) in out.iter_mut().zip(labels.iter()) {
        *slot = if label == 0 {
            true
        } else {
            *fill.get(label as usize - 1).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "label {label} has no entry in a {}-label fill map",
                    fill.len()
                ))
            })?
        };
    }
    Ok(())
}

/// Labels whose distance to another object is below `distance`, grouped by
/// victim label.
///
/// This is the data-level priority rule for object-distance pruning: when two
/// labelled objects are too close, the larger label is the victim, so the answer
/// is deterministic and independent of pair enumeration order.
pub fn object_distance_prune_set<T>(labels: ArrayView3<'_, T>, distance: f64) -> Result<HashSet<T>>
where
    T: Copy + Default + Eq + Ord + std::hash::Hash,
{
    if !distance.is_finite() || distance < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "object distance pruning needs a finite non-negative distance, got {distance}"
        )));
    }
    let zero = T::default();
    let radius = distance.floor() as isize;
    let limit2 = distance * distance;
    let mut by_label: HashMap<T, Vec<[usize; 3]>> = HashMap::new();
    for ((i, j, k), &label) in labels.indexed_iter() {
        if label != zero {
            by_label.entry(label).or_default().push([i, j, k]);
        }
    }
    let mut labels_sorted: Vec<T> = by_label.keys().copied().collect();
    labels_sorted.sort();
    let mut remove = HashSet::new();
    for (left_index, &left) in labels_sorted.iter().enumerate() {
        for &right in &labels_sorted[left_index + 1..] {
            if remove.contains(&left) || remove.contains(&right) {
                continue;
            }
            if labels_are_close(&by_label[&left], &by_label[&right], radius, limit2) {
                remove.insert(right);
            }
        }
    }
    Ok(remove)
}

/// Remove labels selected by [`object_distance_prune_set`].
pub fn prune_by_object_distance_into<T>(
    labels: ArrayView3<'_, T>,
    distance: f64,
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: Copy + Default + Eq + Ord + std::hash::Hash,
{
    shapes_agree(labels.shape(), out.shape(), "prune_by_object_distance_into")?;
    let remove = object_distance_prune_set(labels, distance)?;
    let zero = T::default();
    for (slot, &label) in out.iter_mut().zip(labels.iter()) {
        *slot = if remove.contains(&label) { zero } else { label };
    }
    Ok(())
}

fn labels_are_close(left: &[[usize; 3]], right: &[[usize; 3]], radius: isize, limit2: f64) -> bool {
    left.iter().any(|a| {
        right.iter().any(|b| {
            if (a[0] as isize - b[0] as isize).abs() > radius
                || (a[1] as isize - b[1] as isize).abs() > radius
                || (a[2] as isize - b[2] as isize).abs() > radius
            {
                return false;
            }
            let d0 = a[0] as f64 - b[0] as f64;
            let d1 = a[1] as f64 - b[1] as f64;
            let d2 = a[2] as f64 - b[2] as f64;
            d0 * d0 + d1 * d1 + d2 * d2 <= limit2
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_border_removes_only_components_touching_a_face() {
        let mut mask = Array3::from_elem((5, 5, 3), false);
        mask[[0, 2, 1]] = true;
        mask[[1, 2, 1]] = true;
        mask[[3, 3, 1]] = true;
        let mut out = Array3::from_elem(mask.raw_dim(), true);
        clear_border_into(mask.view(), Connectivity::Faces, out.view_mut()).unwrap();
        assert!(!out[[0, 2, 1]]);
        assert!(!out[[1, 2, 1]]);
        assert!(out[[3, 3, 1]]);
    }

    #[test]
    fn small_objects_and_small_holes_use_connected_component_sizes() {
        let mut mask = Array3::from_elem((5, 5, 3), false);
        mask[[1, 1, 1]] = true;
        mask[[3, 2, 1]] = true;
        mask[[3, 3, 1]] = true;
        let mut out = Array3::from_elem(mask.raw_dim(), true);
        remove_small_objects_into(mask.view(), Connectivity::Faces, 2, out.view_mut()).unwrap();
        assert!(!out[[1, 1, 1]]);
        assert!(out[[3, 2, 1]]);
        assert!(out[[3, 3, 1]]);

        let mut solid = Array3::from_elem((5, 5, 3), true);
        solid[[2, 2, 1]] = false;
        solid[[0, 0, 1]] = false;
        let mut filled = Array3::from_elem(solid.raw_dim(), false);
        remove_small_holes_into(solid.view(), Connectivity::Faces, 2, filled.view_mut()).unwrap();
        assert!(filled[[2, 2, 1]], "the enclosed one-voxel hole is filled");
        assert!(!filled[[0, 0, 1]], "outside background is not a hole");
    }

    #[test]
    fn expand_labels_uses_distance_and_deterministic_ties() {
        let mut labels = Array3::from_elem((5, 1, 1), 0u16);
        labels[[1, 0, 0]] = 2;
        labels[[3, 0, 0]] = 1;
        let mut out = Array3::from_elem(labels.raw_dim(), 0u16);
        expand_labels_into(labels.view(), 1.0, out.view_mut()).unwrap();
        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![2, 2, 1, 1, 1],
            "the middle voxel is equally close to both labels and takes the smaller label"
        );
    }

    #[test]
    fn object_distance_pruning_removes_the_larger_label_in_close_pairs() {
        let mut labels = Array3::from_elem((6, 1, 1), 0u16);
        labels[[0, 0, 0]] = 4;
        labels[[2, 0, 0]] = 7;
        labels[[5, 0, 0]] = 9;
        let remove = object_distance_prune_set(labels.view(), 2.0).unwrap();
        assert!(remove.contains(&7));
        assert!(!remove.contains(&4));
        assert!(!remove.contains(&9));

        let mut pruned = Array3::from_elem(labels.raw_dim(), 0u16);
        prune_by_object_distance_into(labels.view(), 2.0, pruned.view_mut()).unwrap();
        assert_eq!(
            pruned.iter().copied().collect::<Vec<_>>(),
            vec![4, 0, 0, 0, 0, 9]
        );
        assert!(object_distance_prune_set(labels.view(), f64::NAN).is_err());
    }
}
