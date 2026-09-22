#![cfg(feature = "model-segment")]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blockflow::assemble::PlanBuilder;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::model_segment::{stub::ThresholdBackend, InstanceSegment};
use blockflow::ops::collect_rows;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::table::Value;
use blockflow::{Dtype, Voxels};
use ndarray::Array3;

#[test]
fn materialised_labels_use_the_same_stable_ids_as_the_object_rows() {
    let volume = [1, 8, 8];
    let mut input = Array3::<u8>::zeros(volume);
    for y in 2..6 {
        for x in 3..6 {
            input[[0, y, x]] = 10;
        }
    }
    input[[0, 7, 0]] = 20;

    let segment = InstanceSegment::new(
        "threshold instances",
        Arc::new(ThresholdBackend::new(1.0)),
        [0, 4, 4],
        "instances",
        Lifecycle::Persistent,
        Vec::new(),
    )
    .writing_labels();
    let schema = segment.schema().unwrap();
    let mut builder = PlanBuilder::new(
        volume,
        Dtype::U8,
        BlockGrid::new(volume, [1, 4, 4]).unwrap(),
    );
    builder.fragments(segment).unwrap();
    let assembly = builder.finish().unwrap();
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(input),
        &assembly.decomposition,
        [1, 4, 4],
    )
    .unwrap();
    execute_phases(
        "materialised instance labels",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .unwrap();

    let rows = collect_rows(&env, "instances", 0, volume, schema).unwrap();
    let row_counts = rows
        .iter()
        .map(|row| {
            let Value::U64(id) = row.values[0] else {
                panic!("id must be an integer")
            };
            let Value::U64(count) = row.values[1] else {
                panic!("count must be an integer")
            };
            (id, count)
        })
        .collect::<BTreeMap<_, _>>();
    let labels = env.image(1);
    let mut image_counts = BTreeMap::<u64, u64>::new();
    for id in labels
        .view::<u64>()
        .unwrap()
        .iter()
        .copied()
        .filter(|id| *id != 0)
    {
        *image_counts.entry(id).or_default() += 1;
    }

    assert_eq!(row_counts, image_counts);
    assert_eq!(row_counts.keys().copied().collect::<BTreeSet<_>>().len(), 2);
}
