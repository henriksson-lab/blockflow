// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Boundary images derived from neighbouring labels.

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Slicing};
use crate::points::Point;
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::shapes_agree;

/// One boundary voxel owned by one non-background label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelBoundaryPoint<T> {
    pub label: T,
    pub at: [usize; 3],
}

/// Mark every voxel whose 6-neighbourhood contains a different value.
///
/// The image boundary itself is not a boundary: only neighbours that exist in
/// the volume are compared. A block seam is handled by the declared reach, as
/// for every other stencil.
pub fn find_boundaries_into<T>(
    input: ArrayView3<'_, T>,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()>
where
    T: PartialEq + Copy,
{
    shapes_agree(input.shape(), out.shape(), "find_boundaries")?;
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                out[[i, j, k]] = is_boundary(input, [i, j, k], shape);
            }
        }
    }
    Ok(())
}

/// Emit all boundary voxels in lexicographic order as unit-weighted points.
///
/// This is the point/outline view of [`find_boundaries_into`]. It deliberately
/// uses the same neighbour predicate, so an image boundary and an emitted
/// outline cannot drift apart. Labels themselves are not encoded in the point
/// weight; callers that need per-label contour tables need a richer row schema.
pub fn boundary_points<T>(input: ArrayView3<'_, T>) -> Vec<Point>
where
    T: PartialEq + Copy,
{
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut points = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if is_boundary(input, [i, j, k], shape) {
                    points.push(Point::unit([i, j, k]));
                }
            }
        }
    }
    points
}

/// Emit labelled boundary voxels in deterministic `(label, coordinate)` order.
///
/// `T::default()` is the background label and is not emitted. This is the
/// contour-identity view of [`find_boundaries_into`]: the boundary predicate is
/// shared, and the only additional policy is ownership by the non-background
/// label stored at the boundary voxel.
pub fn labelled_boundary_points<T>(input: ArrayView3<'_, T>) -> Vec<LabelBoundaryPoint<T>>
where
    T: Copy + Default + Ord,
{
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let background = T::default();
    let mut points = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let label = input[at];
                if label != background && is_boundary(input, at, shape) {
                    points.push(LabelBoundaryPoint { label, at });
                }
            }
        }
    }
    points.sort();
    points
}

fn is_boundary<T>(input: ArrayView3<'_, T>, at: [usize; 3], shape: [usize; 3]) -> bool
where
    T: PartialEq + Copy,
{
    let here = input[at];
    for axis in 0..3 {
        for step in [-1isize, 1] {
            let mut neighbour = [at[0] as isize, at[1] as isize, at[2] as isize];
            neighbour[axis] += step;
            if neighbour[axis] < 0 || neighbour[axis] >= shape[axis] as isize {
                continue;
            }
            if input[[
                neighbour[0] as usize,
                neighbour[1] as usize,
                neighbour[2] as usize,
            ]] != here
            {
                return true;
            }
        }
    }
    false
}

/// A bool boundary mask from integer labels or a bool mask.
pub struct FindBoundariesOp {
    name: &'static str,
    cost: f64,
}

impl FindBoundariesOp {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            cost: super::voxelwise::MASK_COST * 6.0,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for FindBoundariesOp {
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric([1, 1, 1])
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(
            dtype,
            Dtype::Bool
                | Dtype::U8
                | Dtype::U16
                | Dtype::U32
                | Dtype::U64
                | Dtype::I8
                | Dtype::I16
                | Dtype::I32
                | Dtype::I64
        )
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let out = out.view_mut::<bool>()?;
        match input.dtype() {
            Dtype::Bool => find_boundaries_into(input.view::<bool>()?, out),
            Dtype::U8 => find_boundaries_into(input.view::<u8>()?, out),
            Dtype::U16 => find_boundaries_into(input.view::<u16>()?, out),
            Dtype::U32 => find_boundaries_into(input.view::<u32>()?, out),
            Dtype::U64 => find_boundaries_into(input.view::<u64>()?, out),
            Dtype::I8 => find_boundaries_into(input.view::<i8>()?, out),
            Dtype::I16 => find_boundaries_into(input.view::<i16>()?, out),
            Dtype::I32 => find_boundaries_into(input.view::<i32>()?, out),
            Dtype::I64 => find_boundaries_into(input.view::<i64>()?, out),
            Dtype::F16 | Dtype::F32 | Dtype::F64 => Err(Error::InvalidArgument(format!(
                "{}: boundaries compare labels for equality, so the input must be bool or integer",
                self.name
            ))),
        }
    }

    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        Some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use super::*;

    #[test]
    fn a_label_change_marks_both_sides_of_the_boundary() {
        let labels = Array3::from_shape_fn((5, 1, 1), |(i, _, _)| if i < 2 { 1u16 } else { 2 });
        let mut out = Array3::from_elem(labels.raw_dim(), false);
        find_boundaries_into(labels.view(), out.view_mut()).unwrap();
        let line: Vec<bool> = (0..5).map(|i| out[[i, 0, 0]]).collect();
        assert_eq!(line, vec![false, true, true, false, false]);
    }

    #[test]
    fn the_volume_face_is_not_a_boundary_by_itself() {
        let labels = Array3::from_elem((3, 3, 3), 9i32);
        let mut out = Array3::from_elem(labels.raw_dim(), true);
        find_boundaries_into(labels.view(), out.view_mut()).unwrap();
        assert!(out.iter().all(|&value| !value));
        assert!(boundary_points(labels.view()).is_empty());
    }

    #[test]
    fn boundary_points_are_the_outline_view_of_the_boundary_image() {
        let labels = Array3::from_shape_fn(
            (4, 3, 1),
            |(i, j, _)| {
                if i >= 2 && j == 1 {
                    5u16
                } else {
                    1
                }
            },
        );
        let mut out = Array3::from_elem(labels.raw_dim(), false);
        find_boundaries_into(labels.view(), out.view_mut()).unwrap();

        let mut from_image = Vec::new();
        for ((i, j, k), &value) in out.indexed_iter() {
            if value {
                from_image.push(Point::unit([i, j, k]));
            }
        }
        let points = boundary_points(labels.view());
        assert_eq!(points, from_image);
        assert_eq!(
            points,
            vec![
                Point::unit([1, 1, 0]),
                Point::unit([2, 0, 0]),
                Point::unit([2, 1, 0]),
                Point::unit([2, 2, 0]),
                Point::unit([3, 0, 0]),
                Point::unit([3, 1, 0]),
                Point::unit([3, 2, 0]),
            ]
        );
    }

    #[test]
    fn labelled_boundary_points_keep_non_background_contour_identity() {
        let labels = Array3::from_shape_fn((4, 3, 1), |(i, j, _)| {
            if i < 2 {
                0u16
            } else if j == 1 {
                7
            } else {
                3
            }
        });
        let points = labelled_boundary_points(labels.view());
        assert_eq!(
            points,
            vec![
                LabelBoundaryPoint {
                    label: 3,
                    at: [2, 0, 0],
                },
                LabelBoundaryPoint {
                    label: 3,
                    at: [2, 2, 0],
                },
                LabelBoundaryPoint {
                    label: 3,
                    at: [3, 0, 0],
                },
                LabelBoundaryPoint {
                    label: 3,
                    at: [3, 2, 0],
                },
                LabelBoundaryPoint {
                    label: 7,
                    at: [2, 1, 0],
                },
                LabelBoundaryPoint {
                    label: 7,
                    at: [3, 1, 0],
                },
            ]
        );
    }

    #[test]
    fn find_boundaries_op_declares_the_label_boundary_contract() {
        let op = FindBoundariesOp::new("boundaries");
        assert!(op.accepts(Dtype::Bool));
        assert!(op.accepts(Dtype::U32));
        assert!(!op.accepts(Dtype::F64));
        assert_eq!(op.produces(Dtype::U16), Dtype::Bool);
        assert_eq!(op.reach_spec([10; 3]), Reach::symmetric([1, 1, 1]));
        assert_eq!(op.constant_maps_to(7.0), Some(0.0));

        let labels: Voxels = Array3::from_shape_vec((3, 1, 1), vec![1u8, 1, 2])
            .unwrap()
            .into();
        let mut out = Voxels::zeros(Dtype::Bool, [3, 1, 1]).unwrap();
        op.apply(&labels, &mut out, &Anchor::whole([3, 1, 1]))
            .unwrap();
        let got: Vec<bool> = (0..3)
            .map(|i| out.view::<bool>().unwrap()[[i, 0, 0]])
            .collect();
        assert_eq!(got, vec![false, true, true]);
    }
}
