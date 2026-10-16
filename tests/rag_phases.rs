// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `ops::rag_phases`: the region adjacency graph built out of core, as planned
// phases, against the in-memory `ops::rag::rag` on the whole volume.
//
// * the merged graph of a blocked run has the whole-volume graph's edge set,
//   face counts (per axis, bridged), histograms, min/max and node voxel counts
//   exactly, and its sums to rounding — at several block shapes, with and
//   without line bridging, with the evidence and intensity as supplied inputs;
// * agglomerating the merged graph and relabelling through a planned
//   `RelabelOp` phase gives the volume `ops::agglomerate::relabel_volume`
//   gives in memory, and the same partition the whole-volume graph gives;
// * the same holds in a `ZarrEnvironment` (feature `zarr`).

use std::collections::BTreeSet;

use ndarray::Array3;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::ops::agglomerate::{boundary_cost, gaec, relabel_volume};
use blockflow::ops::rag::{rag, Rag, RagOptions};
use blockflow::ops::rag_phases::{
    append_rag_phases, append_relabel_phase, collect_rag, RagImage, RagPhases, RelabelOp,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [9, 13, 14];

/// Brick coordinates of a voxel; a supervoxel is one brick.
fn brick(z: usize, y: usize, x: usize) -> [usize; 3] {
    let bz = z / 3;
    let by = (y + bz) / 4;
    let bx = (x + 2 * by) / 5;
    [bz, by, bx]
}

/// The "cell" a brick belongs to: two bricks wide on y and x, so the
/// agglomeration has supervoxels to merge and boundaries to keep.
fn cell(z: usize, y: usize, x: usize) -> [usize; 3] {
    let [bz, by, bx] = brick(z, y, x);
    [bz, by / 2, bx / 2]
}

/// Supervoxels: jittered bricks, globally numbered, with a sprinkling of
/// 0-lines so bridging has something to bridge.
fn supervoxels() -> Array3<u32> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        if (x + 3 * y + 5 * z) % 13 == 0 {
            return 0;
        }
        let [bz, by, bx] = brick(z, y, x);
        (bz * 64 + by * 8 + bx) as u32 + 1
    })
}

/// Membrane evidence: high on cell boundaries, low noise elsewhere.
fn membrane() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        let here = cell(z, y, x);
        let at = [z, y, x];
        let boundary = (0..3).any(|axis| {
            [at[axis].wrapping_sub(1), at[axis] + 1]
                .into_iter()
                .any(|v| {
                    if v >= VOLUME[axis] {
                        return false;
                    }
                    let mut q = at;
                    q[axis] = v;
                    cell(q[0], q[1], q[2]) != here
                })
        });
        let noise = ((z * 7 + y * 5 + x * 3) % 23) as f64 / 23.0 * 0.1;
        if boundary {
            0.85 + noise
        } else {
            noise
        }
    })
}

fn intensity() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        ((z + 2 * y + x) % 9) as f64
    })
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

fn assert_same(expected: &Rag, got: &Rag, context: &str) {
    let edges =
        |g: &Rag| -> BTreeSet<(u32, u32)> { (0..g.edge_count()).map(|i| g.edge(i).0).collect() };
    assert_eq!(edges(expected), edges(got), "{context}: edge sets");
    assert_eq!(expected.labels(), got.labels(), "{context}: node sets");
    for i in 0..expected.edge_count() {
        let ((a, b), e) = expected.edge(i);
        let j = got.find_edge(a, b).unwrap();
        let (_, g) = got.edge(j);
        assert_eq!(
            (e.faces, e.axis_faces, e.bridged),
            (g.faces, g.axis_faces, g.bridged),
            "{context}: faces of ({a}, {b})"
        );
        assert_eq!(
            (e.min, e.max),
            (g.min, g.max),
            "{context}: min/max of ({a}, {b})"
        );
        assert!(
            close(e.sum, g.sum) && close(e.sum_sq, g.sum_sq),
            "{context}: sums"
        );
        assert!(close(e.area, g.area), "{context}: area");
        assert_eq!(
            expected.histogram(i),
            got.histogram(j),
            "{context}: histogram"
        );
    }
    for label in expected.labels() {
        let (e, g) = (expected.node(label).unwrap(), got.node(label).unwrap());
        assert_eq!(e.voxels, g.voxels, "{context}: voxels of {label}");
        assert!(
            close(e.sum, g.sum) && close(e.sum_sq, g.sum_sq),
            "{context}: node sums"
        );
    }
}

/// Plan the emit phase over image 0 (supervoxels), supplied 0 (membrane) and
/// supplied 1 (intensity, stored as `f32`).
fn rag_plan(block: [usize; 3], options: &RagOptions) -> (blockflow::assemble::Assembly, RagPhases) {
    let grid = BlockGrid::new(VOLUME, block).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U32, grid);
    let phases = append_rag_phases(
        &mut builder,
        RagImage::new(0usize, Dtype::U32),
        RagImage::new(ImageId::supplied(0), Dtype::F64),
        Some(RagImage::new(ImageId::supplied(1), Dtype::F32)),
        options,
        "rag",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    (builder.finish().unwrap(), phases)
}

fn supplied() -> Vec<Voxels> {
    vec![membrane().into(), intensity().mapv(|v| v as f32).into()]
}

fn run_rag(block: [usize; 3], options: &RagOptions) -> Rag {
    let (plan, phases) = rag_plan(block, options);
    let env = ArrayEnvironment::with_inputs(
        supervoxels().into(),
        supplied(),
        &plan.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    execute_phases(
        "rag-phases",
        &plan.workflow,
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &plan.work(),
    )
    .unwrap();
    collect_rag(&env, &phases).unwrap()
}

#[test]
fn the_merged_graph_of_a_blocked_run_is_the_whole_volume_graph() {
    let labels = supervoxels();
    let evidence = membrane();
    let values = intensity();
    for bridge in [false, true] {
        let options = RagOptions::new([2.0, 1.0, 1.0], 12, (0.0, 1.0)).bridging_lines(bridge);
        let whole = rag(
            labels.view(),
            evidence.view(),
            Some(values.view()),
            &options,
        )
        .unwrap();
        assert!(whole.edge_count() > 20, "the fixture has a real graph");
        assert!(bridge == ((0..whole.edge_count()).any(|i| whole.edge(i).1.bridged > 0)));
        for block in [[9, 13, 14], [4, 5, 6], [3, 4, 4], [2, 13, 3], [1, 2, 14]] {
            let merged = run_rag(block, &options);
            assert_same(&whole, &merged, &format!("block {block:?} bridge {bridge}"));
        }
    }
}

/// Partition with GAEC on quantised boundary costs (so sub-ulp differences in
/// the merged sums cannot reorder ties).
fn partition(graph: &Rag) -> (Vec<u32>, Vec<u32>) {
    let g = graph.graph(|r, i| {
        let c = boundary_cost(r.edge(i).1.mean(), 0.5);
        (c * 1e6).round() / 1e6
    });
    let clusters = gaec(g.labels.len(), &g.edges);
    (g.labels, clusters)
}

#[test]
fn agglomerating_and_relabelling_through_the_phases_matches_memory() {
    let labels = supervoxels();
    let options = RagOptions::new([2.0, 1.0, 1.0], 0, (0.0, 1.0));
    let whole = rag(labels.view(), membrane().view(), None, &options).unwrap();
    let merged = run_rag([4, 5, 6], &options);
    let (ids, clusters) = partition(&merged);
    let (whole_ids, whole_clusters) = partition(&whole);
    assert_eq!((&ids, &clusters), (&whole_ids, &whole_clusters));
    let distinct: BTreeSet<u32> = clusters.iter().copied().collect();
    assert!(
        distinct.len() > 1 && distinct.len() < ids.len(),
        "a non-trivial partition"
    );

    let expected = relabel_volume(labels.view(), &ids, &clusters).unwrap();

    // The relabelling is its own plan, over the supervoxel image.
    let grid = BlockGrid::new(VOLUME, [4, 5, 6]).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U32, grid);
    append_relabel_phase(
        &mut builder,
        RelabelOp::from_partition(&ids, &clusters).unwrap(),
    )
    .unwrap();
    let plan = builder.finish().unwrap();
    let env =
        ArrayEnvironment::for_decomposition(labels.into(), &plan.decomposition, [4, 4, 4]).unwrap();
    execute_phases(
        "rag-relabel",
        &plan.workflow,
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &plan.work(),
    )
    .unwrap();
    let got = env.output().view::<u32>().unwrap().to_owned();
    assert_eq!(got, expected);
}

#[test]
fn collecting_from_an_environment_that_did_not_run_is_refused() {
    let options = RagOptions::new([1.0; 3], 0, (0.0, 1.0));
    let (plan, phases) = rag_plan([4, 5, 6], &options);
    let env = ArrayEnvironment::with_inputs(
        supervoxels().into(),
        supplied(),
        &plan.decomposition,
        [4, 4, 4],
    )
    .unwrap();
    env.declare_sidecar("rag", Lifecycle::DeleteOnExit).unwrap();
    assert!(collect_rag(&env, &phases).is_err());
}

#[cfg(feature = "zarr")]
#[test]
fn the_same_graph_out_of_a_zarr_store() {
    use blockflow::zarr_env::ZarrEnvironment;
    let options = RagOptions::new([2.0, 1.0, 1.0], 8, (0.0, 1.0)).bridging_lines(true);
    let labels = supervoxels();
    let whole = rag(
        labels.view(),
        membrane().view(),
        Some(intensity().view()),
        &options,
    )
    .unwrap();
    let (plan, phases) = rag_plan([4, 5, 6], &options);
    let root = std::env::temp_dir().join(format!("blockflow-rag-phases-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let input: Voxels = labels.into();
    let extra = supplied();
    let env = ZarrEnvironment::create_with_inputs(
        &root,
        &input,
        &extra.iter().collect::<Vec<_>>(),
        [4, 5, 6],
    )
    .unwrap();
    execute_phases(
        "rag-phases-zarr",
        &plan.workflow,
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &plan.work(),
    )
    .unwrap();
    let merged = collect_rag(&env, &phases).unwrap();
    assert_same(&whole, &merged, "zarr");
    drop(env);
    let _ = std::fs::remove_dir_all(&root);
}
