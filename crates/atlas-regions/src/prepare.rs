// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Atlas preparation: orient the atlas volumes to the sample, crop them, and
//! optionally keep one hemisphere — the reference's
//! `Annotation.prepare_annotation_files`, in memory.
//!
//! The order is the reference's: orient first, then crop, with the crop given
//! in the **oriented** frame. A hemisphere selection then narrows the crop to
//! the bounding box of one hemisphere label (plus a margin), which is how the
//! reference keeps one hemisphere (a slicing such as `z: 0..256` on the
//! left-right axis). Voxels outside the selected box are dropped, not zeroed.
//!
//! [`PreparedWindow`] records where the prepared volume sits in the source
//! atlas, so points can be carried either way.

use std::ops::Range;

use blockflow::error::{Error, Result};
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use serde_json::{json, Value};

use crate::orientation::Orientation;

/// A Python-style slice of one axis (step 1): `None` is open, a negative
/// index counts from the end, and an index past the end is clamped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AxisRange {
    pub start: Option<i64>,
    pub stop: Option<i64>,
}

impl AxisRange {
    /// The whole axis.
    pub const ALL: Self = Self {
        start: None,
        stop: None,
    };

    pub fn new(start: Option<i64>, stop: Option<i64>) -> Self {
        Self { start, stop }
    }

    /// `"a:b"`, `"a:"`, `":b"`, `":"` or `""` (the whole axis).
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() || spec == ":" {
            return Ok(Self::ALL);
        }
        let (start, stop) = spec.split_once(':').ok_or_else(|| {
            Error::invalid(format!(
                "axis range {spec:?}: expected start:stop, either side optional"
            ))
        })?;
        let bound = |text: &str| -> Result<Option<i64>> {
            let text = text.trim();
            if text.is_empty() || text == "None" {
                return Ok(None);
            }
            text.parse()
                .map(Some)
                .map_err(|error| Error::invalid(format!("axis range {spec:?}: {error}")))
        };
        Ok(Self::new(bound(start)?, bound(stop)?))
    }

    /// The voxel range on an axis of `len` voxels; an empty range is refused.
    pub fn resolve(&self, len: usize) -> Result<Range<usize>> {
        let n = len as i64;
        let clamp = |index: i64| -> usize {
            let index = if index < 0 { index + n } else { index };
            index.clamp(0, n) as usize
        };
        let start = self.start.map_or(0, clamp);
        let stop = self.stop.map_or(len, clamp);
        if stop <= start {
            return Err(Error::invalid(format!(
                "axis range {self} of an axis of {len} voxels is empty"
            )));
        }
        Ok(start..stop)
    }
}

impl std::fmt::Display for AxisRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let side = |bound: Option<i64>| bound.map(|b| b.to_string()).unwrap_or_default();
        write!(f, "{}:{}", side(self.start), side(self.stop))
    }
}

/// A crop of the oriented atlas, per axis in `[z, y, x]` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crop(pub [AxisRange; 3]);

impl Crop {
    /// No crop.
    pub const ALL: Self = Self([AxisRange::ALL; 3]);

    /// The reference's slicing, given in its `(x, y, z)` order.
    pub fn from_xyz(xyz: [AxisRange; 3]) -> Self {
        Self([xyz[2], xyz[1], xyz[0]])
    }

    /// `"z0:z1,y0:y1,x0:x1"` in `[z, y, x]` order; each part as
    /// [`AxisRange::parse`].
    pub fn parse_zyx(spec: &str) -> Result<Self> {
        Ok(Self(parse_three(spec)?))
    }

    /// `"x0:x1,y0:y1,z0:z1"` in the reference's `(x, y, z)` order.
    pub fn parse_xyz(spec: &str) -> Result<Self> {
        Ok(Self::from_xyz(parse_three(spec)?))
    }

    /// The start and shape of the crop in a volume of `shape`.
    pub fn resolve(&self, shape: [usize; 3]) -> Result<([usize; 3], [usize; 3])> {
        let mut start = [0; 3];
        let mut extent = [0; 3];
        for axis in 0..3 {
            let range = self.0[axis].resolve(shape[axis])?;
            start[axis] = range.start;
            extent[axis] = range.len();
        }
        Ok((start, extent))
    }
}

fn parse_three(spec: &str) -> Result<[AxisRange; 3]> {
    let parts = spec
        .split(',')
        .map(AxisRange::parse)
        .collect::<Result<Vec<_>>>()?;
    parts.try_into().map_err(|_| {
        Error::invalid(format!(
            "crop {spec:?}: expected three comma-separated ranges such as 0:256,:,:"
        ))
    })
}

/// Keep one hemisphere: crop to the bounding box of the voxels whose
/// hemisphere label is `value`, grown by `margin` voxels on every side.
///
/// In the shipped 25 µm Allen atlas the hemisphere volume splits the
/// left-right axis (`z` before orientation) into `255` (`z < 228`) and `0`;
/// note that `0` is a hemisphere there, not background.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HemisphereSelection {
    pub value: i64,
    pub margin: usize,
}

/// What to do to the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrepareParams {
    pub orientation: Orientation,
    /// In the oriented frame.
    pub crop: Crop,
    pub hemisphere: Option<HemisphereSelection>,
}

/// The atlas volumes, all of one shape. Only the annotation is required.
#[derive(Debug, Clone, PartialEq)]
pub struct AtlasStack {
    pub annotation: Voxels,
    pub reference: Option<Voxels>,
    pub hemispheres: Option<Voxels>,
    /// The distance of each voxel to the brain surface, in voxels as the
    /// reference ships it.
    pub distance_to_surface: Option<Voxels>,
}

impl AtlasStack {
    pub fn new(annotation: Voxels) -> Self {
        Self {
            annotation,
            reference: None,
            hemispheres: None,
            distance_to_surface: None,
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        self.annotation.shape()
    }

    fn each(&self) -> impl Iterator<Item = (&'static str, &Voxels)> {
        [
            ("annotation", Some(&self.annotation)),
            ("reference", self.reference.as_ref()),
            ("hemispheres", self.hemispheres.as_ref()),
            ("distance_to_surface", self.distance_to_surface.as_ref()),
        ]
        .into_iter()
        .filter_map(|(name, volume)| volume.map(|v| (name, v)))
    }

    fn map(&self, f: impl Fn(&Voxels) -> Result<Voxels>) -> Result<Self> {
        Ok(Self {
            annotation: f(&self.annotation)?,
            reference: self.reference.as_ref().map(&f).transpose()?,
            hemispheres: self.hemispheres.as_ref().map(&f).transpose()?,
            distance_to_surface: self.distance_to_surface.as_ref().map(&f).transpose()?,
        })
    }
}

/// Where a prepared atlas sits in its source: the source is oriented, then
/// the box `start .. start + shape` of the oriented volume is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedWindow {
    pub orientation: Orientation,
    /// The source atlas's shape, before orientation.
    pub source_shape: [usize; 3],
    /// The first kept voxel, in the oriented frame.
    pub start: [usize; 3],
    /// The prepared volume's shape.
    pub shape: [usize; 3],
}

impl PreparedWindow {
    /// The whole source, unoriented.
    pub fn identity(shape: [usize; 3]) -> Self {
        Self {
            orientation: Orientation::IDENTITY,
            source_shape: shape,
            start: [0; 3],
            shape,
        }
    }

    pub fn oriented_shape(&self) -> [usize; 3] {
        self.orientation.orient_shape(self.source_shape)
    }

    /// A point in source-atlas voxels, in prepared-atlas voxels.
    pub fn to_prepared(&self, point: [f64; 3]) -> [f64; 3] {
        let oriented = self.orientation.orient_point(point, self.source_shape);
        [0, 1, 2].map(|a| oriented[a] - self.start[a] as f64)
    }

    /// A point in prepared-atlas voxels, in source-atlas voxels.
    pub fn to_source(&self, point: [f64; 3]) -> [f64; 3] {
        let oriented = [0, 1, 2].map(|a| point[a] + self.start[a] as f64);
        self.orientation.unorient_point(oriented, self.source_shape)
    }

    /// The JSON this window is recorded as in a prepared atlas's metadata.
    pub fn to_json(&self) -> Value {
        json!({
            "orientation_xyz": self.orientation.to_xyz(),
            "orientation_source_zyx": self.orientation.source(),
            "orientation_flip_zyx": self.orientation.flip(),
            "source_shape_zyx": self.source_shape,
            "start_zyx": self.start,
            "shape_zyx": self.shape,
        })
    }

    /// [`Self::to_json`] read back.
    pub fn from_json(value: &Value) -> Result<Self> {
        let triple = |key: &str| -> Result<[i64; 3]> {
            let entries = value
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| Error::invalid(format!("prepared window has no `{key}`")))?;
            let numbers = entries
                .iter()
                .map(|entry| {
                    entry.as_i64().ok_or_else(|| {
                        Error::invalid(format!("prepared window `{key}` holds {entry}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            numbers
                .try_into()
                .map_err(|_| Error::invalid(format!("prepared window `{key}` is not three long")))
        };
        let sizes = |key: &str| -> Result<[usize; 3]> {
            let t = triple(key)?;
            if t.iter().any(|&v| v < 0) {
                return Err(Error::invalid(format!(
                    "prepared window `{key}` is negative"
                )));
            }
            Ok(t.map(|v| v as usize))
        };
        let xyz = triple("orientation_xyz")?.map(|v| v as i32);
        Ok(Self {
            orientation: Orientation::from_xyz(xyz)?,
            source_shape: sizes("source_shape_zyx")?,
            start: sizes("start_zyx")?,
            shape: sizes("shape_zyx")?,
        })
    }
}

/// Orient, crop and optionally select a hemisphere.
pub fn prepare_atlas(
    stack: &AtlasStack,
    params: &PrepareParams,
) -> Result<(AtlasStack, PreparedWindow)> {
    let source_shape = stack.shape();
    for (name, volume) in stack.each() {
        if volume.shape() != source_shape {
            return Err(Error::invalid(format!(
                "the atlas {name} volume is {:?} but the annotation is {source_shape:?}; the \
                 atlas volumes have to share one grid",
                volume.shape()
            )));
        }
    }
    let orientation = params.orientation;
    let oriented_shape = orientation.orient_shape(source_shape);
    let (mut start, mut shape) = params.crop.resolve(oriented_shape)?;

    if let Some(selection) = params.hemisphere {
        let hemispheres = stack
            .hemispheres
            .as_ref()
            .ok_or_else(|| Error::invalid("selecting a hemisphere needs the hemispheres volume"))?;
        // The bounding box is found in the source frame, on the cropped region
        // carried back, so the full oriented copy is not needed for it.
        let (box_start, box_shape) =
            hemisphere_box(hemispheres, selection, orientation, start, shape)?;
        start = box_start;
        shape = box_shape;
    }

    let window = PreparedWindow {
        orientation,
        source_shape,
        start,
        shape,
    };
    let region = Region::new(&start, &shape);
    let prepared = stack.map(|volume| {
        let oriented = orientation.orient_voxels(volume);
        if start == [0; 3] && shape == oriented_shape {
            Ok(oriented)
        } else {
            oriented.slice_region(&region)
        }
    })?;
    Ok((prepared, window))
}

/// The bounding box (oriented frame) of `selection.value` inside the oriented
/// box `start, shape`, grown by the margin and clipped to that box.
fn hemisphere_box(
    hemispheres: &Voxels,
    selection: HemisphereSelection,
    orientation: Orientation,
    start: [usize; 3],
    shape: [usize; 3],
) -> Result<([usize; 3], [usize; 3])> {
    let labels = hemispheres.widened_i64("hemispheres")?;
    let source_shape = hemispheres.shape();
    let mut low = [usize::MAX; 3];
    let mut high = [0usize; 3];
    let mut found = false;
    for ((z, y, x), &value) in labels.indexed_iter() {
        if value != selection.value {
            continue;
        }
        let p = orientation.orient_point([z as f64, y as f64, x as f64], source_shape);
        let p = p.map(|c| c as usize);
        if (0..3).any(|a| p[a] < start[a] || p[a] >= start[a] + shape[a]) {
            continue;
        }
        found = true;
        for a in 0..3 {
            low[a] = low[a].min(p[a]);
            high[a] = high[a].max(p[a]);
        }
    }
    if !found {
        return Err(Error::invalid(format!(
            "no voxel of the cropped atlas has hemisphere label {}",
            selection.value
        )));
    }
    let mut box_start = [0; 3];
    let mut box_shape = [0; 3];
    for a in 0..3 {
        let first = low[a].saturating_sub(selection.margin).max(start[a]);
        let last = (high[a] + selection.margin).min(start[a] + shape[a] - 1);
        box_start[a] = first;
        box_shape[a] = last + 1 - first;
    }
    Ok((box_start, box_shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn keyed(shape: [usize; 3]) -> Voxels {
        Voxels::I64(Array3::from_shape_fn(shape, |(z, y, x)| {
            (z * 10000 + y * 100 + x) as i64
        }))
    }

    #[test]
    fn axis_ranges_follow_python_slices() {
        assert_eq!(AxisRange::ALL.resolve(10).unwrap(), 0..10);
        assert_eq!(AxisRange::parse("2:5").unwrap().resolve(10).unwrap(), 2..5);
        assert_eq!(AxisRange::parse(":-3").unwrap().resolve(10).unwrap(), 0..7);
        assert_eq!(AxisRange::parse("-4:").unwrap().resolve(10).unwrap(), 6..10);
        assert_eq!(
            AxisRange::parse("0:256").unwrap().resolve(100).unwrap(),
            0..100
        );
        assert_eq!(
            AxisRange::parse("None:3").unwrap().resolve(10).unwrap(),
            0..3
        );
        assert!(AxisRange::parse("5:5").unwrap().resolve(10).is_err());
        assert!(AxisRange::parse("5").is_err());
        assert_eq!(
            Crop::parse_xyz(":,:,0:256").unwrap(),
            Crop([
                AxisRange::new(Some(0), Some(256)),
                AxisRange::ALL,
                AxisRange::ALL
            ])
        );
    }

    #[test]
    fn preparation_orients_then_crops() {
        let shape = [6, 5, 4];
        let stack = AtlasStack {
            hemispheres: Some(keyed(shape)),
            ..AtlasStack::new(keyed(shape))
        };
        let orientation = Orientation::from_xyz([2, -1, 3]).unwrap();
        let params = PrepareParams {
            orientation,
            crop: Crop::parse_zyx("1:4,:,-2:").unwrap(),
            hemisphere: None,
        };
        let (prepared, window) = prepare_atlas(&stack, &params).unwrap();
        // Oriented shape is [6, 4, 5]; the crop keeps z 1..4 and x 3..5.
        assert_eq!(window.oriented_shape(), [6, 4, 5]);
        assert_eq!(window.start, [1, 0, 3]);
        assert_eq!(prepared.shape(), [3, 4, 2]);
        let annotation = prepared.annotation.view::<i64>().unwrap();
        for ((z, y, x), &value) in annotation.indexed_iter() {
            let source = window.to_source([z as f64, y as f64, x as f64]);
            let s = source.map(|c| c as i64);
            assert_eq!(value, s[0] * 10000 + s[1] * 100 + s[2]);
            assert_eq!(window.to_prepared(source), [z as f64, y as f64, x as f64]);
        }
        assert_eq!(prepared.hemispheres.unwrap(), prepared.annotation);
        assert_eq!(
            PreparedWindow::from_json(&window.to_json()).unwrap(),
            window
        );
    }

    #[test]
    fn a_hemisphere_is_its_bounding_box_plus_the_margin() {
        let shape = [8, 3, 4];
        let hemispheres = Voxels::U8(Array3::from_shape_fn(
            shape,
            |(z, _, _)| {
                if z < 5 {
                    255
                } else {
                    0
                }
            },
        ));
        let stack = AtlasStack {
            hemispheres: Some(hemispheres),
            ..AtlasStack::new(keyed(shape))
        };
        let mut params = PrepareParams {
            hemisphere: Some(HemisphereSelection {
                value: 0,
                margin: 0,
            }),
            ..PrepareParams::default()
        };
        let (prepared, window) = prepare_atlas(&stack, &params).unwrap();
        assert_eq!((window.start, window.shape), ([5, 0, 0], [3, 3, 4]));
        assert_eq!(prepared.annotation.shape(), [3, 3, 4]);
        params.hemisphere = Some(HemisphereSelection {
            value: 255,
            margin: 1,
        });
        let (_, window) = prepare_atlas(&stack, &params).unwrap();
        assert_eq!((window.start, window.shape), ([0, 0, 0], [6, 3, 4]));
        // Flipping z moves the 255 side to the far end.
        params.orientation = Orientation::new([0, 1, 2], [true, false, false]).unwrap();
        let (_, window) = prepare_atlas(&stack, &params).unwrap();
        assert_eq!((window.start, window.shape), ([2, 0, 0], [6, 3, 4]));
        params.hemisphere = Some(HemisphereSelection {
            value: 7,
            margin: 0,
        });
        assert!(prepare_atlas(&stack, &params).is_err());
    }

    #[test]
    fn mismatched_volumes_are_refused() {
        let stack = AtlasStack {
            reference: Some(keyed([2, 2, 2])),
            ..AtlasStack::new(keyed([2, 2, 3]))
        };
        let error = prepare_atlas(&stack, &PrepareParams::default()).unwrap_err();
        assert!(error.to_string().contains("one grid"), "{error}");
    }
}
