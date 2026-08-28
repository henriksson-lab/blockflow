// SPDX-License-Identifier: MIT

use blockflow::op::Chain;
use blockflow::ops::VoxelwiseMapOp;
use blockflow::pyramid::{LevelRecipe, PyramidRecipe};
use blockflow::voxels::Voxels;
use ndarray::Array3;

#[test]
fn a_recipe_decimates_level_by_level() {
    let input: Voxels =
        Array3::from_shape_fn((4, 2, 2), |(x, y, z)| (x + 10 * y + 100 * z) as f64).into();
    let recipe = PyramidRecipe::decimation(vec![[2, 1, 1], [2, 2, 1]]).unwrap();

    let levels = recipe.build_resident(&input).expect("levels build");

    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0].shape(), [4, 2, 2]);
    assert_eq!(levels[1].shape(), [2, 2, 2]);
    assert_eq!(levels[2].shape(), [1, 1, 2]);
    assert_eq!(levels[1].view::<f64>().unwrap()[[0, 0, 0]], 0.5);
}

#[test]
fn image_ops_are_applied_before_decimation() {
    let input: Voxels = Array3::from_shape_fn((4, 2, 1), |(x, _, _)| x as f64).into();
    let threshold = Chain::op(VoxelwiseMapOp::threshold("threshold", 1.5, 10.0, 0.0));
    let recipe = PyramidRecipe::new(vec![LevelRecipe::new([2, 1, 1], threshold).expect("level")]);

    let levels = recipe.build_resident(&input).expect("levels build");
    let level = levels[1].view::<f64>().unwrap();

    assert_eq!(level[[0, 0, 0]], 0.0);
    assert_eq!(level[[1, 0, 0]], 10.0);
}
