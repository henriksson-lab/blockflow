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
    BlockOutput, BlockView, FragmentInput, FragmentOp, PhaseView, SeamFold, SourceBlocks,
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
    shapes_agree(mask.shape(), out.shape(), "clear_border_into")?;
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count = label_regions_into_with(mask, connectivity, labels.view_mut())?;
    let touching = labels_touching_border(labels.view(), count)?;
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

const COMPONENT_MASK_MAGIC: u32 = 0x4d43_4d52;
const COMPONENT_MASK_NOUN: &str = "a component-mask reduction";

#[derive(Debug, Clone, Copy)]
enum ComponentMaskRule {
    MinimumSize { minimum_size: u64 },
    ClearBorder { volume: [usize; 3] },
}

impl ComponentMaskRule {
    fn keep(&self, moments: &super::detect::Moments) -> Result<bool> {
        match *self {
            Self::MinimumSize { minimum_size } => Ok(moments.count >= minimum_size),
            Self::ClearBorder { volume } => {
                let Some((low, high)) = moments.bounds() else {
                    return Ok(false);
                };
                Ok((0..3).all(|axis| low[axis] != 0 && high[axis] + 1 != volume[axis]))
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
            ComponentMaskRule::ClearBorder { volume },
            lattice,
            mask,
            mask_dtype,
        )
        .connecting(connectivity),
    )
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

fn labels_touching_border(labels: ArrayView3<'_, u32>, count: u32) -> Result<Vec<bool>> {
    let mut touching = vec![false; count as usize];
    let shape = labels.shape();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if i != 0
                    && j != 0
                    && k != 0
                    && i + 1 != shape[0]
                    && j + 1 != shape[1]
                    && k + 1 != shape[2]
                {
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
