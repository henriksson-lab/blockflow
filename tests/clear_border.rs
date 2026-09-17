// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::ops::{
    append_clear_border_on_axes_phases, append_clear_border_phases, clear_border_into,
    clear_border_on_axes_into, filter_labels_touching_border_on_axes_into, Connectivity,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use ndarray::Array3;

const VOLUME: [usize; 3] = [5, 5, 4];
const STREAM: &str = "clear-border";

fn run(input: &Array3<bool>, block: [usize; 3], connectivity: Connectivity) -> Array3<bool> {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::Bool, grid);
    append_clear_border_phases(&mut builder, STREAM, Lifecycle::DeleteOnExit, connectivity)
        .unwrap();
    let assembly = builder.finish().expect("clear-border phases");
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "clear border",
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

fn run_on_axes(
    input: &Array3<bool>,
    volume: [usize; 3],
    block: [usize; 3],
    connectivity: Connectivity,
    axes: [bool; 3],
) -> Array3<bool> {
    let grid = BlockGrid::new(volume, block).unwrap();
    let mut builder = PlanBuilder::new(volume, Dtype::Bool, grid);
    append_clear_border_on_axes_phases(
        &mut builder,
        STREAM,
        Lifecycle::DeleteOnExit,
        connectivity,
        axes,
    )
    .unwrap();
    let assembly = builder.finish().expect("clear-border phases");
    let env =
        ArrayEnvironment::for_decomposition(input.clone().into(), &assembly.decomposition, block)
            .unwrap();
    execute_phases(
        "clear border on axes",
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
fn clear_border_phases_merge_components_before_testing_volume_border() {
    let mut mask = Array3::<bool>::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    mask[[1, 2, 1]] = true;
    mask[[2, 2, 1]] = true;
    mask[[3, 2, 1]] = true;
    mask[[0, 4, 2]] = true;
    mask[[1, 4, 2]] = true;

    let mut expected = Array3::<bool>::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    clear_border_into(mask.view(), Connectivity::Faces, expected.view_mut()).unwrap();
    assert!(expected[[1, 2, 1]]);
    assert!(expected[[3, 2, 1]]);
    assert!(!expected[[0, 4, 2]]);
    assert!(!expected[[1, 4, 2]]);
    assert_eq!(run(&mask, VOLUME, Connectivity::Faces), expected);
    assert_eq!(run(&mask, [2, 5, 4], Connectivity::Faces), expected);
}

#[test]
fn clear_border_phases_honour_wider_connectivity() {
    let mut mask = Array3::<bool>::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    mask[[0, 0, 0]] = true;
    mask[[1, 1, 1]] = true;

    let faces = run(&mask, [2, 2, 2], Connectivity::Faces);
    let full = run(&mask, [2, 2, 2], Connectivity::FacesEdgesAndCorners);
    assert!(!faces[[0, 0, 0]]);
    assert!(faces[[1, 1, 1]]);
    assert!(!full[[0, 0, 0]]);
    assert!(!full[[1, 1, 1]]);
}

#[test]
fn clear_border_on_axes_can_ignore_singleton_z_for_2d_images() {
    let volume = [1, 5, 5];
    let mut mask = Array3::<bool>::from_elem(volume, false);
    mask[[0, 2, 2]] = true;
    mask[[0, 2, 3]] = true;
    mask[[0, 0, 0]] = true;
    mask[[0, 1, 0]] = true;

    let mut expected = Array3::<bool>::from_elem(volume, false);
    clear_border_on_axes_into(
        mask.view(),
        Connectivity::Faces,
        [false, true, true],
        expected.view_mut(),
    )
    .unwrap();
    assert!(expected[[0, 2, 2]]);
    assert!(expected[[0, 2, 3]]);
    assert!(!expected[[0, 0, 0]]);
    assert!(!expected[[0, 1, 0]]);

    assert_eq!(
        run_on_axes(
            &mask,
            volume,
            [1, 2, 3],
            Connectivity::Faces,
            [false, true, true]
        ),
        expected
    );
}

#[test]
fn label_border_filter_removes_only_final_labels_touching_selected_axes() {
    let volume = [1, 5, 6];
    let mut labels = Array3::<u32>::zeros(volume);
    labels[[0, 0, 0]] = 1;
    labels[[0, 1, 0]] = 1;
    labels[[0, 1, 1]] = 2;
    labels[[0, 1, 2]] = 2;
    labels[[0, 2, 1]] = 2;
    labels[[0, 3, 4]] = 3;
    labels[[0, 3, 5]] = 3;
    labels[[0, 4, 2]] = 4;

    let mut out = Array3::<u32>::zeros(volume);
    filter_labels_touching_border_on_axes_into(labels.view(), [false, true, true], out.view_mut())
        .unwrap();

    assert_eq!(out[[0, 0, 0]], 0);
    assert_eq!(out[[0, 1, 0]], 0);
    assert_eq!(out[[0, 1, 1]], 2);
    assert_eq!(out[[0, 1, 2]], 2);
    assert_eq!(out[[0, 2, 1]], 2);
    assert_eq!(out[[0, 3, 4]], 0);
    assert_eq!(out[[0, 3, 5]], 0);
    assert_eq!(out[[0, 4, 2]], 0);
}
