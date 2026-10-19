// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The brain mask of an annotation, and the distance-to-surface field built
//! from it with `blockflow::ops::distance`.

use blockflow::error::Result;
use blockflow::ops::distance::{distance_transform, DistanceParams};
use blockflow::voxels::Voxels;
use ndarray::Array3;

/// `annotation > 0`: every voxel that carries a region.
///
/// `> 0` and not `!= 0`, so a negative label (used by some atlases for
/// "unassigned") counts as outside. Compared in each element type, without
/// widening the volume to `f64` first.
pub fn mask_from_annotation(annotation: &Voxels) -> Array3<bool> {
    match annotation {
        Voxels::Bool(array) => array.clone(),
        Voxels::U8(array) => array.mapv(|value| value > 0),
        Voxels::U16(array) => array.mapv(|value| value > 0),
        Voxels::U32(array) => array.mapv(|value| value > 0),
        Voxels::U64(array) => array.mapv(|value| value > 0),
        Voxels::I8(array) => array.mapv(|value| value > 0),
        Voxels::I16(array) => array.mapv(|value| value > 0),
        Voxels::I32(array) => array.mapv(|value| value > 0),
        Voxels::I64(array) => array.mapv(|value| value > 0),
        Voxels::F32(array) => array.mapv(|value| value > 0.0),
        Voxels::F64(array) => array.mapv(|value| value > 0.0),
    }
}

/// The exact Euclidean distance from every voxel inside the brain to the
/// nearest voxel outside it, as an `F64` volume; zero outside.
///
/// Resident: the whole field is computed in memory, which is about 8 bytes
/// per voxel on top of the mask (616 MB for a 25 µm mouse atlas). For a
/// volume too large for that, build the mask and append
/// `blockflow::ops::distance::append_to` to a plan instead.
///
/// `params` sets the voxel pitch (`with_sampling`, to get the distance in
/// physical units) and the value of a volume with no outside voxel at all.
pub fn distance_to_surface(annotation: &Voxels, params: &DistanceParams) -> Result<Voxels> {
    let mask = mask_from_annotation(annotation);
    Ok(Voxels::F64(distance_transform(mask.view(), params)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use blockflow::ops::distance::Unbounded;

    #[test]
    fn the_mask_is_strictly_positive_labels() {
        let annotation =
            Voxels::I32(Array3::from_shape_vec((1, 1, 4), vec![-2, 0, 1, 700]).unwrap());
        let mask = mask_from_annotation(&annotation);
        assert_eq!(
            mask.iter().copied().collect::<Vec<_>>(),
            vec![false, false, true, true]
        );
        let float = Voxels::F32(Array3::from_shape_vec((1, 1, 3), vec![0.0, 0.5, 5.28e8]).unwrap());
        assert_eq!(
            mask_from_annotation(&float)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![false, true, true]
        );
    }

    #[test]
    fn a_single_labelled_voxel_is_one_voxel_from_the_surface() {
        let mut annotation = Array3::<u16>::zeros((5, 5, 5));
        annotation[[2, 2, 2]] = 42;
        let field =
            distance_to_surface(&Voxels::U16(annotation), &DistanceParams::default()).unwrap();
        let Voxels::F64(field) = field else {
            panic!("an f64 field");
        };
        assert_eq!(field[[2, 2, 2]], 1.0);
        assert_eq!(field.iter().filter(|value| **value != 0.0).count(), 1);
    }

    #[test]
    fn the_distance_grows_into_a_labelled_slab_and_honours_the_pitch() {
        // Labelled for x in 1..=5 of 7, background on both faces.
        let annotation = Voxels::U32(Array3::from_shape_fn((7, 2, 2), |(x, _, _)| {
            u32::from((1..=5).contains(&x)) * 9
        }));
        let field = distance_to_surface(&annotation, &DistanceParams::default()).unwrap();
        let Voxels::F64(field) = field else {
            panic!("an f64 field");
        };
        let profile: Vec<f64> = (0..7).map(|x| field[[x, 0, 0]]).collect();
        assert_eq!(profile, vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]);

        let scaled = distance_to_surface(
            &annotation,
            &DistanceParams::default().with_sampling([2.5, 1.0, 1.0]),
        )
        .unwrap();
        let Voxels::F64(scaled) = scaled else {
            panic!("an f64 field");
        };
        assert_eq!(scaled[[3, 1, 1]], 7.5);
    }

    #[test]
    fn a_volume_with_no_outside_follows_the_unbounded_rule() {
        let annotation = Voxels::U8(Array3::from_elem((2, 3, 4), 1u8));
        let params = DistanceParams::default().with_unbounded(Unbounded::Infinite);
        let Voxels::F64(field) = distance_to_surface(&annotation, &params).unwrap() else {
            panic!("an f64 field");
        };
        assert!(field.iter().all(|value| value.is_infinite()));
    }
}
