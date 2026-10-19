// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Point lookups in an atlas: the region, hemisphere and distance to the brain
//! surface at each of a set of points given in atlas voxel coordinates.
//!
//! A point lookup has reach zero — the voxel a point reads depends on the
//! point alone — so the columns are computed against whole in-memory volumes,
//! and row `i` of every output is the point at row `i` of the input.

use std::sync::Arc;

use blockflow::error::{Error, Result};
use blockflow::voxels::Voxels;

use crate::ontology::Ontology;

/// How a coordinate becomes a voxel index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Rounding {
    /// To the nearest integer, halves to even. Right when voxel centres sit at
    /// integer coordinates, which is blockflow's convention. Halves to even
    /// rather than away from zero so that a resample by `0.5`, which puts an
    /// exact `.5` on every other coordinate, does not bias every such point
    /// upwards.
    #[default]
    Nearest,
    /// Downwards. Right when voxel `i` spans `[i, i + 1)`.
    Floor,
}

impl Rounding {
    /// The voxel index of `value` before any bound is applied, or `None` where
    /// the coordinate is not finite or does not fit an `i64`.
    pub fn index(self, value: f64) -> Option<i64> {
        if !value.is_finite() {
            return None;
        }
        let rounded = match self {
            Self::Nearest => value.round_ties_even(),
            Self::Floor => value.floor(),
        };
        if rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
            return None;
        }
        Some(rounded as i64)
    }
}

/// What happens to a point whose voxel falls outside the volume being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfBounds {
    /// No read happens; the point gets the lookup's sentinel.
    Invalid,
    /// The index is clamped into the volume and the read happens anyway.
    Clamp,
}

/// The voxel `point` reads in a volume of `shape`, or `None` where it has none
/// (a non-finite coordinate, or outside the volume under
/// [`OutOfBounds::Invalid`]).
pub fn voxel_index(
    point: [f64; 3],
    shape: [usize; 3],
    rounding: Rounding,
    out_of_bounds: OutOfBounds,
) -> Option<[usize; 3]> {
    let mut voxel = [0usize; 3];
    for axis in 0..3 {
        let index = rounding.index(point[axis])?;
        let len = shape[axis] as i64;
        voxel[axis] = match out_of_bounds {
            OutOfBounds::Invalid if index < 0 || index >= len => return None,
            OutOfBounds::Invalid => index as usize,
            OutOfBounds::Clamp if len == 0 => return None,
            OutOfBounds::Clamp => index.clamp(0, len - 1) as usize,
        };
    }
    Some(voxel)
}

/// The integer label at `at`, exactly.
///
/// A float volume is refused rather than truncated (a region id is an
/// integer), as is a `u64` above `i64::MAX`.
pub fn label_at(volume: &Voxels, at: [usize; 3]) -> Result<i64> {
    Ok(match volume {
        Voxels::Bool(array) => i64::from(array[at]),
        Voxels::U8(array) => i64::from(array[at]),
        Voxels::U16(array) => i64::from(array[at]),
        Voxels::U32(array) => i64::from(array[at]),
        Voxels::I8(array) => i64::from(array[at]),
        Voxels::I16(array) => i64::from(array[at]),
        Voxels::I32(array) => i64::from(array[at]),
        Voxels::I64(array) => array[at],
        Voxels::U64(array) => i64::try_from(array[at]).map_err(|_| {
            Error::invalid(format!(
                "the label volume holds {} at {at:?}, which does not fit an i64; region ids are \
                 small, so the volume is probably not an annotation",
                array[at]
            ))
        })?,
        Voxels::F32(_) | Voxels::F64(_) => {
            return Err(Error::invalid(format!(
                "a label lookup was pointed at a {} volume; a region id is an integer, so \
                 convert the annotation to an integer type first",
                volume.dtype().numpy_name()
            )))
        }
    })
}

/// The field value at `at` as `f64`.
///
/// 64-bit integer volumes are refused: values past `2^53` do not survive the
/// widening, and a distance field has no reason to be stored that way.
pub fn field_at(volume: &Voxels, at: [usize; 3]) -> Result<f64> {
    Ok(match volume {
        Voxels::Bool(array) => f64::from(u8::from(array[at])),
        Voxels::U8(array) => f64::from(array[at]),
        Voxels::U16(array) => f64::from(array[at]),
        Voxels::U32(array) => f64::from(array[at]),
        Voxels::I8(array) => f64::from(array[at]),
        Voxels::I16(array) => f64::from(array[at]),
        Voxels::I32(array) => f64::from(array[at]),
        Voxels::F32(array) => f64::from(array[at]),
        Voxels::F64(array) => array[at],
        Voxels::U64(_) | Voxels::I64(_) => {
            return Err(Error::invalid(format!(
                "a field sample was pointed at a {} volume, which cannot be read exactly as f64 \
                 (values above 2^53 collapse); store the field as float",
                volume.dtype().numpy_name()
            )))
        }
    })
}

/// The settings of a lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotateParams {
    /// Collapse every region id onto its ancestor at this level (see
    /// [`Ontology::collapse`]). `None` keeps the annotation's own ids. Setting
    /// a level requires [`Annotator::with_ontology`].
    pub level: Option<i64>,
    /// The region id of a point outside the annotation volume.
    pub annotation_invalid: i64,
    /// The hemisphere id of a point outside the hemispheres volume.
    pub hemisphere_invalid: i64,
    /// How every lookup rounds a coordinate to a voxel.
    pub rounding: Rounding,
    /// What a distance sample does outside the volume: [`OutOfBounds::Clamp`]
    /// reads the nearest edge voxel, [`OutOfBounds::Invalid`] gives `NaN`.
    pub distance_out_of_bounds: OutOfBounds,
}

impl Default for AnnotateParams {
    fn default() -> Self {
        Self {
            level: None,
            annotation_invalid: 0,
            hemisphere_invalid: -1,
            rounding: Rounding::Nearest,
            distance_out_of_bounds: OutOfBounds::Clamp,
        }
    }
}

impl AnnotateParams {
    pub fn with_level(mut self, level: Option<i64>) -> Self {
        self.level = level;
        self
    }

    pub fn with_annotation_invalid(mut self, invalid: i64) -> Self {
        self.annotation_invalid = invalid;
        self
    }

    pub fn with_hemisphere_invalid(mut self, invalid: i64) -> Self {
        self.hemisphere_invalid = invalid;
        self
    }

    pub fn with_rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }

    pub fn with_distance_out_of_bounds(mut self, out_of_bounds: OutOfBounds) -> Self {
        self.distance_out_of_bounds = out_of_bounds;
        self
    }
}

/// The atlas volumes a lookup reads. They need not share a shape: each lookup
/// is bounded by the volume it reads.
#[derive(Debug, Clone, Copy)]
pub struct AtlasVolumes<'a> {
    /// Region ids (integer).
    pub annotation: &'a Voxels,
    /// Hemisphere ids (integer), for example `0` and `255`.
    pub hemispheres: Option<&'a Voxels>,
    /// Distance to the brain surface, see [`crate::distance_to_surface`].
    pub distance_to_surface: Option<&'a Voxels>,
}

impl<'a> AtlasVolumes<'a> {
    pub fn new(annotation: &'a Voxels) -> Self {
        Self {
            annotation,
            hemispheres: None,
            distance_to_surface: None,
        }
    }

    pub fn with_hemispheres(mut self, volume: &'a Voxels) -> Self {
        self.hemispheres = Some(volume);
        self
    }

    pub fn with_distance_to_surface(mut self, volume: &'a Voxels) -> Self {
        self.distance_to_surface = Some(volume);
        self
    }
}

/// The per-point columns, in the caller's row order.
///
/// `hemisphere` and `distance_to_surface` are empty where the corresponding
/// volume was not given.
#[derive(Debug, Clone, PartialEq)]
pub struct AtlasColumns {
    /// The region id, collapsed to [`AnnotateParams::level`] if one is set.
    pub annotation: Vec<i64>,
    pub hemisphere: Vec<i64>,
    /// In the distance volume's units (atlas voxels unless it was built with a
    /// physical sampling).
    pub distance_to_surface: Vec<f64>,
}

impl AtlasColumns {
    pub fn len(&self) -> usize {
        self.annotation.len()
    }

    pub fn is_empty(&self) -> bool {
        self.annotation.is_empty()
    }
}

/// Atlas volumes and points in, per-point columns out.
#[derive(Debug, Clone, Default)]
pub struct Annotator {
    params: AnnotateParams,
    ontology: Option<Arc<Ontology>>,
}

impl Annotator {
    pub fn new(params: AnnotateParams) -> Self {
        Self {
            params,
            ontology: None,
        }
    }

    /// The region tree a collapse walks. Required when
    /// [`AnnotateParams::level`] is set.
    pub fn with_ontology(mut self, ontology: Arc<Ontology>) -> Self {
        self.ontology = Some(ontology);
        self
    }

    pub fn params(&self) -> &AnnotateParams {
        &self.params
    }

    pub fn ontology(&self) -> Option<&Ontology> {
        self.ontology.as_deref()
    }

    /// Every column the volumes allow.
    pub fn columns(&self, volumes: AtlasVolumes<'_>, points: &[[f64; 3]]) -> Result<AtlasColumns> {
        Ok(AtlasColumns {
            annotation: self.regions(volumes.annotation, points)?,
            hemisphere: match volumes.hemispheres {
                Some(volume) => self.hemispheres(volume, points)?,
                None => Vec::new(),
            },
            distance_to_surface: match volumes.distance_to_surface {
                Some(volume) => self.distances(volume, points)?,
                None => Vec::new(),
            },
        })
    }

    /// The region id at each point, collapsed if a level is set.
    ///
    /// The collapse applies only to annotated points: neither the sentinel of a
    /// point outside the volume nor the background label 0 inside it (the
    /// voxels [`crate::mask_from_annotation`] leaves out) is a region, and
    /// neither is looked up in the tree.
    pub fn regions(&self, annotation: &Voxels, points: &[[f64; 3]]) -> Result<Vec<i64>> {
        let collapse = match self.params.level {
            None => None,
            Some(level) => {
                let ontology = self.ontology.as_deref().ok_or_else(|| {
                    Error::invalid(format!(
                        "a collapse to level {level} was asked for and no region tree was \
                         given; pass one with Annotator::with_ontology"
                    ))
                })?;
                Some((ontology, level))
            }
        };
        self.labels(
            annotation,
            points,
            self.params.annotation_invalid,
            |id| match collapse {
                Some((ontology, level)) if id != 0 => ontology.collapse(id, level),
                _ => Ok(id),
            },
        )
    }

    /// The hemisphere id at each point.
    pub fn hemispheres(&self, volume: &Voxels, points: &[[f64; 3]]) -> Result<Vec<i64>> {
        self.labels(volume, points, self.params.hemisphere_invalid, Ok)
    }

    /// The field value at each point.
    ///
    /// A non-finite coordinate is refused under [`OutOfBounds::Clamp`], where
    /// every finite point has a voxel, and gives `NaN` under
    /// [`OutOfBounds::Invalid`].
    pub fn distances(&self, volume: &Voxels, points: &[[f64; 3]]) -> Result<Vec<f64>> {
        let shape = volume.shape();
        let out_of_bounds = self.params.distance_out_of_bounds;
        points
            .iter()
            .map(
                |point| match voxel_index(*point, shape, self.params.rounding, out_of_bounds) {
                    Some(voxel) => field_at(volume, voxel),
                    None if out_of_bounds == OutOfBounds::Invalid => Ok(f64::NAN),
                    None => Err(Error::invalid(format!(
                        "the coordinate {point:?} has no voxel in a {shape:?} volume"
                    ))),
                },
            )
            .collect()
    }

    fn labels(
        &self,
        volume: &Voxels,
        points: &[[f64; 3]],
        invalid: i64,
        map: impl Fn(i64) -> Result<i64>,
    ) -> Result<Vec<i64>> {
        let shape = volume.shape();
        points
            .iter()
            .map(|point| {
                match voxel_index(*point, shape, self.params.rounding, OutOfBounds::Invalid) {
                    Some(voxel) => map(label_at(volume, voxel)?),
                    None => Ok(invalid),
                }
            })
            .collect()
    }
}

/// The minimum of `values` over each `[start, end)` range: a segmented
/// reduction, such as the smallest distance to the surface along each edge's
/// polyline when the polyline points are stored in one flat buffer.
///
/// Ordered with `total_cmp`, so `-0.0` sorts below `0.0` and the answer does
/// not depend on the walk order. An empty range is refused rather than
/// answered with an infinity.
pub fn polyline_minima(values: &[f64], ranges: &[[usize; 2]]) -> Result<Vec<f64>> {
    ranges
        .iter()
        .enumerate()
        .map(|(segment, &[start, end])| {
            if start >= end {
                return Err(Error::invalid(format!(
                    "segment {segment} has the empty range [{start}, {end})"
                )));
            }
            if end > values.len() {
                return Err(Error::ShapeMismatch {
                    expected: vec![end],
                    got: vec![values.len()],
                });
            }
            Ok(values[start..end]
                .iter()
                .copied()
                .min_by(f64::total_cmp)
                .expect("a non-empty range"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// A 4 x 4 x 4 atlas whose value is `x * 100 + y * 10 + z`, so a wrong
    /// axis or a wrong rounding names itself in the answer.
    fn keyed_atlas() -> Voxels {
        Voxels::U32(Array3::from_shape_fn((4, 4, 4), |(x, y, z)| {
            (x * 100 + y * 10 + z) as u32
        }))
    }

    fn tree() -> Arc<Ontology> {
        Arc::new(
            Ontology::new([
                (0, None, None),
                (2, Some(1), Some(0)),
                (10, Some(3), Some(2)),
                (11, Some(3), Some(2)),
                (12, Some(2), Some(0)),
            ])
            .unwrap(),
        )
    }

    #[test]
    fn nearest_rounds_halves_to_even_and_floor_rounds_down() {
        for (value, floor, nearest) in [
            (0.5, 0, 0),
            (1.5, 1, 2),
            (2.5, 2, 2),
            (3.5, 3, 4),
            (-0.5, -1, 0),
            (-1.5, -2, -2),
            (2.4, 2, 2),
            (2.6, 2, 3),
        ] {
            assert_eq!(Rounding::Floor.index(value), Some(floor), "{value}");
            assert_eq!(Rounding::Nearest.index(value), Some(nearest), "{value}");
        }
    }

    #[test]
    fn a_non_finite_coordinate_has_no_voxel() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300] {
            assert_eq!(Rounding::Floor.index(value), None);
            assert_eq!(Rounding::Nearest.index(value), None);
        }
    }

    #[test]
    fn every_lookup_reads_the_same_voxel() {
        let atlas = keyed_atlas();
        let points = [[1.5, 1.5, 1.5], [2.5, 0.4, 2.6], [0.5, 2.5, 1.5]];
        let volumes = AtlasVolumes::new(&atlas).with_distance_to_surface(&atlas);
        let columns = Annotator::default().columns(volumes, &points).unwrap();
        assert_eq!(columns.annotation, vec![222, 203, 22]);
        assert_eq!(columns.distance_to_surface, vec![222.0, 203.0, 22.0]);

        let floor = Annotator::new(AnnotateParams::default().with_rounding(Rounding::Floor));
        let columns = floor.columns(volumes, &points).unwrap();
        assert_eq!(columns.annotation, vec![111, 202, 21]);
        assert_eq!(columns.distance_to_surface, vec![111.0, 202.0, 21.0]);
    }

    #[test]
    fn a_point_outside_has_no_region_and_a_clamped_distance() {
        let atlas = keyed_atlas();
        let volumes = AtlasVolumes::new(&atlas)
            .with_hemispheres(&atlas)
            .with_distance_to_surface(&atlas);
        let points = [[-1.0, 2.0, 2.0], [9.0, 2.0, 2.0], [1.0, 1.0, 1.0]];
        let columns = Annotator::default().columns(volumes, &points).unwrap();
        assert_eq!(columns.len(), 3);
        assert_eq!(columns.annotation, vec![0, 0, 111]);
        assert_eq!(columns.hemisphere, vec![-1, -1, 111]);
        assert_eq!(columns.distance_to_surface, vec![22.0, 322.0, 111.0]);

        let strict = Annotator::new(
            AnnotateParams::default().with_distance_out_of_bounds(OutOfBounds::Invalid),
        );
        let distances = strict.distances(&atlas, &points).unwrap();
        assert!(distances[0].is_nan() && distances[1].is_nan());
        assert_eq!(distances[2], 111.0);
    }

    #[test]
    fn a_non_finite_point_is_invalid_for_labels_and_refused_for_a_clamped_field() {
        let atlas = keyed_atlas();
        let annotator = Annotator::default();
        assert_eq!(
            annotator.regions(&atlas, &[[f64::NAN, 0.0, 0.0]]).unwrap(),
            vec![0]
        );
        assert!(annotator
            .distances(&atlas, &[[f64::NAN, 0.0, 0.0]])
            .is_err());
    }

    #[test]
    fn each_lookup_is_bounded_by_the_volume_it_reads() {
        let annotation = keyed_atlas();
        let hemispheres = Voxels::U8(Array3::from_elem((2, 2, 2), 7u8));
        let volumes = AtlasVolumes::new(&annotation).with_hemispheres(&hemispheres);
        let columns = Annotator::default()
            .columns(volumes, &[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
            .unwrap();
        assert_eq!(columns.annotation, vec![111, 311]);
        assert_eq!(columns.hemisphere, vec![7, -1]);
    }

    #[test]
    fn the_collapse_reaches_the_column_and_leaves_the_sentinel_alone() {
        let atlas = Voxels::U32(Array3::from_shape_fn((2, 1, 1), |(x, _, _)| 10 + x as u32));
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
        let params = AnnotateParams::default()
            .with_level(Some(2))
            .with_annotation_invalid(-3);
        let annotator = Annotator::new(params).with_ontology(tree());
        assert_eq!(annotator.regions(&atlas, &points).unwrap(), vec![2, 2, -3]);
        // Without a level the tree makes no difference.
        let plain = Annotator::default().with_ontology(tree());
        assert_eq!(plain.regions(&atlas, &points).unwrap(), vec![10, 11, 0]);
    }

    #[test]
    fn the_collapse_leaves_unannotated_voxels_alone() {
        // Label 0 inside the volume is background, not a region of the tree.
        let atlas = Voxels::U32(Array3::from_shape_fn((2, 1, 1), |(x, _, _)| 10 * x as u32));
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let annotator =
            Annotator::new(AnnotateParams::default().with_level(Some(2))).with_ontology(tree());
        assert_eq!(annotator.regions(&atlas, &points).unwrap(), vec![0, 2]);
    }

    #[test]
    fn a_collapse_without_an_ontology_is_refused() {
        let annotator = Annotator::new(AnnotateParams::default().with_level(Some(2)));
        let error = annotator
            .regions(&keyed_atlas(), &[[1.0, 1.0, 1.0]])
            .unwrap_err();
        assert!(error.to_string().contains("no region tree"), "{error}");
    }

    #[test]
    fn a_float_volume_is_refused_as_a_label_atlas() {
        let atlas = Voxels::F64(Array3::from_elem((2, 2, 2), 3.7));
        let error = Annotator::default()
            .regions(&atlas, &[[0.0, 0.0, 0.0]])
            .unwrap_err();
        assert!(
            error.to_string().contains("region id is an integer"),
            "{error}"
        );
    }

    #[test]
    fn a_64_bit_field_is_refused_rather_than_widened() {
        let field = Voxels::I64(Array3::from_elem((4, 4, 4), 1i64 << 60));
        let error = Annotator::default()
            .distances(&field, &[[0.0, 0.0, 0.0]])
            .unwrap_err();
        assert!(error.to_string().contains("2^53"), "{error}");
    }

    #[test]
    fn the_rows_keep_the_callers_order() {
        let points = [
            [3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
        ];
        assert_eq!(
            Annotator::default()
                .regions(&keyed_atlas(), &points)
                .unwrap(),
            vec![333, 0, 210, 102]
        );
    }

    #[test]
    fn the_polyline_minimum_is_over_every_point_of_the_segment() {
        // The second segment dips in the middle, below both of its ends.
        let values = vec![5.0, 4.0, 6.0, 9.0, 1.0, 9.0];
        assert_eq!(
            polyline_minima(&values, &[[0, 3], [3, 6]]).unwrap(),
            vec![4.0, 1.0]
        );
    }

    #[test]
    fn the_polyline_minimum_orders_signed_zeroes() {
        for values in [[0.0, -0.0], [-0.0, 0.0]] {
            let minimum = polyline_minima(&values, &[[0, 2]]).unwrap()[0];
            assert_eq!(minimum.to_bits(), (-0.0f64).to_bits());
        }
    }

    #[test]
    fn an_empty_or_overlong_segment_is_refused() {
        let error = polyline_minima(&[1.0, 2.0], &[[1, 1]]).unwrap_err();
        assert!(error.to_string().contains("empty range"), "{error}");
        assert!(polyline_minima(&[1.0, 2.0], &[[0, 3]]).is_err());
    }
}
