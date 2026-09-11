// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{append_remove_small_objects_phases, remove_small_objects_into, Connectivity};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [4, 4, 4];
const STREAM: &str = "remove-small-objects";

fn run(input: &Array3<bool>, block: [usize; 3], connectivity: Connectivity) -> Array3<bool> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::Bool, grid);
    append_remove_small_objects_phases(
        &mut builder,
        STREAM,
        Lifecycle::DeleteOnExit,
        connectivity,
        2,
    )
    .unwrap();
    let assembly = builder.finish().expect("remove-small-objects phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "remove small objects",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .expect("a cleanup run");
    env.output().view::<bool>().unwrap().to_owned()
}

#[test]
fn remove_small_objects_phases_size_components_after_seam_merging() {
    let mut mask = Array3::<bool>::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    mask[[1, 1, 1]] = true;
    mask[[2, 1, 1]] = true;
    mask[[0, 3, 3]] = true;

    let mut expected = Array3::<bool>::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    remove_small_objects_into(mask.view(), Connectivity::Faces, 2, expected.view_mut()).unwrap();
    assert!(expected[[1, 1, 1]]);
    assert!(expected[[2, 1, 1]]);
    assert!(!expected[[0, 3, 3]]);
    assert_eq!(run(&mask, VOLUME, Connectivity::Faces), expected);
    assert_eq!(run(&mask, [2, 4, 4], Connectivity::Faces), expected);
}

#[test]
fn remove_small_objects_phases_honour_wider_connectivity() {
    let mut mask = Array3::<bool>::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    mask[[1, 1, 1]] = true;
    mask[[2, 2, 2]] = true;

    let faces = run(&mask, [2, 2, 2], Connectivity::Faces);
    let full = run(&mask, [2, 2, 2], Connectivity::FacesEdgesAndCorners);
    assert!(!faces[[1, 1, 1]]);
    assert!(!faces[[2, 2, 2]]);
    assert!(full[[1, 1, 1]]);
    assert!(full[[2, 2, 2]]);
}
