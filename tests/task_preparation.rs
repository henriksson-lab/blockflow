use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::*;
use blockflow::op::SourceInput;
use blockflow::ops::activity::{self, ActivityOp};
use blockflow::ops::interior::{encode_runs, LabelRun, RunLabelsOp};
use blockflow::ops::tabulate::{FixedPoint, TabulateValuesOp};
use blockflow::reach::Reach;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::{
    BlockGrid, Dtype, Error, ExecutionLog, ImageId, PlanBuilder, Region, Result, Voxels,
};
use ndarray::Array3;
use std::sync::{atomic::Ordering, Arc};

use blockflow::fragment::Unprepared as Dense;

fn stream(name: &str) -> Vec<FragmentOutput> {
    vec![FragmentOutput::new(
        name,
        Lifecycle::Persistent,
        Coverage::EveryBlock,
    )]
}
fn contains(r: &Region, p: [usize; 3]) -> bool {
    (0..3).all(|a| p[a] >= r.start[a] && p[a] < r.start[a] + r.shape[a])
}
struct Seeds(bool);
impl FragmentOp for Seeds {
    fn name(&self) -> &'static str {
        "seeds"
    }
    fn reach(&self, _: usize, _: usize) -> usize {
        0
    }
    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new("sentinel", Lifecycle::Persistent, Coverage::EveryBlock),
            FragmentOutput::new("activity", Lifecycle::Persistent, Coverage::Sparse),
        ]
    }
    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let out = BlockOutput::fragment("sentinel", vec![0]);
        Ok(if self.0 {
            out
        } else {
            out.with_fragment("activity", activity::encode(contains(at.core, [5; 3])))
        })
    }
}
struct Sample {
    phase: usize,
    invalid: bool,
}
impl FragmentOp for Sample {
    fn name(&self) -> &'static str {
        "sample"
    }
    fn reach(&self, _: usize, _: usize) -> usize {
        2
    }
    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own("activity", self.phase)]
    }
    fn outputs(&self) -> Vec<FragmentOutput> {
        stream("sum")
    }
    fn source_inputs(&self, _: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(ImageId::from(0), Reach::symmetric([2; 3]))]
    }
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }
    fn prepares(&self) -> bool {
        true
    }
    fn prepare(&self, at: &PreparationView<'_>) -> Result<BlockPreparation> {
        let Some(bytes) = at.own("activity") else {
            return Ok(BlockPreparation::Run);
        };
        if !activity::decode(bytes)? {
            return Ok(BlockPreparation::Complete {
                fragments: vec![("sum".into(), 0u64.to_le_bytes().to_vec())],
                constant: None,
            });
        }
        Ok(BlockPreparation::ReadWindows(vec![ReadWindow {
            image: 0,
            region: Region::new(if self.invalid { &[100; 3] } else { &[5; 3] }, &[2; 3]),
        }]))
    }
    fn apply(&self, _: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid("source required"))
    }
    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let region = Region::new(&[5; 3], &[2; 3]);
        let mut sum = 0u64;
        if contains(at.core, [5; 3]) {
            let (buf, r) = sources.covering(0, &region, at.read)?;
            let a = buf.as_array()?.view::<u16>()?;
            for z in 5..7 {
                for y in 5..7 {
                    for x in 5..7 {
                        sum += a[[z - r.start[0], y - r.start[1], x - r.start[2]]] as u64;
                    }
                }
            }
        }
        Ok(BlockOutput::fragment("sum", sum.to_le_bytes().to_vec()))
    }
}
fn sample_run(
    dense: bool,
    edge: usize,
    workers: usize,
    invalid: bool,
    missing: bool,
) -> Result<(u64, u64, u64)> {
    let n = [24; 3];
    let grid = BlockGrid::new(n, [edge; 3])?;
    let mut b = PlanBuilder::new(n, Dtype::U16, grid);
    let seeds = b.fragments(Seeds(missing))?;
    let op = Sample {
        phase: seeds.index(),
        invalid,
    };
    if dense {
        b.fragments(Dense(op))?;
    } else {
        b.fragments(op)?;
    }
    let a = b.finish()?;
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(Array3::from_shape_fn(n, |(z, y, x)| {
            (z * 3 + y * 2 + x) as u16
        })),
        &a.decomposition,
        [4; 3],
    )?;
    let log = Arc::new(ExecutionLog::new());
    execute_phases(
        "sample",
        &a.workflow,
        &a.decomposition,
        &Hints {
            concurrency: workers,
            prefetch_depth: 2,
            ..Hints::default()
        },
        &env,
        &[log.clone()],
        &a.work(),
    )?;
    let sum = env
        .sidecar_fragments("sum")?
        .iter()
        .map(|(_, b)| u64::from_le_bytes(b.as_slice().try_into().unwrap()))
        .sum();
    let p = log.preparation_stats();
    for e in log.events() {
        let json = blockflow::export::event_json(&e);
        blockflow::export::event_from_json(&json, 0)?;
    }
    Ok((
        sum,
        env.counters().read_voxels.load(Ordering::SeqCst),
        p.completed,
    ))
}
#[test]
fn sparse_reads_equal_dense_across_grids_and_workers() {
    for edge in [8, 12] {
        for workers in [1, 4] {
            let dense = sample_run(true, edge, workers, false, false).unwrap();
            let sparse = sample_run(false, edge, workers, false, false).unwrap();
            assert_eq!(dense.0, sparse.0);
            assert_eq!(sparse.1, 8);
            assert!(dense.1 > 10000);
            assert!(sparse.2 > 0);
        }
    }
}
#[test]
fn runtime_windows_cannot_escape_the_static_dependencies() {
    assert!(sample_run(false, 8, 4, true, false)
        .unwrap_err()
        .to_string()
        .contains("declared source footprint"));
}

struct Runs;
impl FragmentOp for Runs {
    fn name(&self) -> &'static str {
        "runs"
    }
    fn reach(&self, _: usize, _: usize) -> usize {
        0
    }
    fn outputs(&self) -> Vec<FragmentOutput> {
        stream("runs")
    }
    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let runs = if at.index == [0; 3] {
            vec![LabelRun {
                label: 7,
                at: [1, 1, 6],
                len: 5,
            }]
        } else {
            vec![]
        };
        Ok(BlockOutput::fragment("runs", encode_runs(&runs)))
    }
}
fn paint(dense: bool, workers: usize) -> (Vec<u64>, u64) {
    let mut b = PlanBuilder::new([16; 3], Dtype::U8, BlockGrid::new([16; 3], [8; 3]).unwrap());
    let runs = b.fragments(Runs).unwrap();
    let op = RunLabelsOp::new("paint", ("runs", runs.index()), 5.0);
    if dense {
        b.fragments(Dense(op)).unwrap();
    } else {
        b.fragments(op).unwrap();
    }
    let a = b.finish().unwrap();
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(Array3::from_elem([16; 3], 0u8)),
        &a.decomposition,
        [4; 3],
    )
    .unwrap();
    let log = Arc::new(ExecutionLog::new());
    execute_phases(
        "paint",
        &a.workflow,
        &a.decomposition,
        &Hints {
            concurrency: workers,
            ..Hints::default()
        },
        &env,
        &[log.clone()],
        &a.work(),
    )
    .unwrap();
    let values = env
        .output()
        .view::<u64>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    (values, log.preparation_stats().completed)
}
#[test]
fn neighbour_runs_are_not_skipped_and_empty_cores_have_real_backgrounds() {
    let dense = paint(true, 1);
    let sparse = paint(false, 4);
    assert_eq!(dense.0, sparse.0);
    assert_eq!(sparse.0.iter().filter(|v| **v == 7).count(), 5);
    assert_eq!(sparse.1, 6);
}

fn tabulate_on(
    n: [usize; 3],
    edge: usize,
    dense: bool,
    all_active: bool,
    workers: usize,
) -> (Vec<Vec<u8>>, u64) {
    let mut b = PlanBuilder::new(n, Dtype::U16, BlockGrid::new(n, [edge; 3]).unwrap());
    let activity = b
        .fragments(ActivityOp::new(
            ImageId::from(0),
            Dtype::U16,
            "mask",
            Lifecycle::Persistent,
        ))
        .unwrap();
    let op = TabulateValuesOp::new(
        "measure",
        ImageId::from(0),
        ImageId::supplied(0),
        FixedPoint::default(),
        "partials",
        Lifecycle::Persistent,
    )
    .unwrap()
    .holding(Dtype::U16, Dtype::F64)
    .with_activity("mask", activity.index());
    if dense {
        b.fragments(Dense(op)).unwrap();
    } else {
        b.fragments(op).unwrap();
    }
    let a = b.finish().unwrap();
    let labels = Array3::from_shape_fn(n, |(z, y, x)| {
        if all_active || (z == 1 && y == 1 && x == 1) {
            3u16
        } else {
            0
        }
    });
    let env = ArrayEnvironment::with_inputs(
        Voxels::from(labels),
        vec![Voxels::from(Array3::from_elem(n, 2f64))],
        &a.decomposition,
        [4; 3],
    )
    .unwrap();
    execute_phases(
        "measure",
        &a.workflow,
        &a.decomposition,
        &Hints {
            concurrency: workers,
            ..Hints::default()
        },
        &env,
        &[],
        &a.work(),
    )
    .unwrap();
    let mut rows = env.sidecar_fragments("partials").unwrap();
    rows.sort_by_key(|(key, _)| key.block);
    (
        rows.into_iter().map(|(_, b)| b).collect(),
        env.counters().read_voxels.load(Ordering::SeqCst),
    )
}
fn tabulate(dense: bool, all_active: bool, workers: usize) -> (Vec<Vec<u8>>, u64) {
    tabulate_on([16; 3], 8, dense, all_active, workers)
}
#[test]
fn mask_summary_reuses_occupancy_without_stale_results() {
    let dense = tabulate(true, false, 1);
    let sparse = tabulate(false, false, 4);
    assert_eq!(dense.0, sparse.0);
    assert!(sparse.1 < dense.1);
    let dense_full = tabulate(true, true, 1);
    let sparse_full = tabulate(false, true, 4);
    assert_eq!(dense_full.0, sparse_full.0);
    assert_eq!(dense_full.1, sparse_full.1);
}
#[test]
fn coalescing_preserves_disjoint_windows_and_images() {
    let w = |image, start| ReadWindow {
        image,
        region: Region::new(&[start; 3], &[4; 3]),
    };
    let out = coalesce_read_windows(vec![w(0, 0), w(0, 2), w(0, 16), w(1, 2)]);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].region, Region::new(&[0; 3], &[6; 3]));
}

#[test]
fn missing_metadata_falls_back_to_dense_reads() {
    let dense = sample_run(true, 8, 1, false, true).unwrap();
    let unknown = sample_run(false, 8, 4, false, true).unwrap();
    assert_eq!(dense.0, unknown.0);
    assert_eq!(dense.1, unknown.1);
    assert_eq!(unknown.2, 0);
}

/// Run explicitly in release mode: cargo test --release --test task_preparation
/// benchmark_sparse_and_dense -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn benchmark_sparse_and_dense() {
    for full in [false, true] {
        for dense in [true, false] {
            let mut timings = Vec::new();
            let mut voxels = 0;
            for _ in 0..5 {
                let started = std::time::Instant::now();
                let (_, read) = tabulate_on([128; 3], 32, dense, full, 4);
                timings.push(started.elapsed().as_secs_f64());
                voxels = read;
            }
            timings.sort_by(f64::total_cmp);
            println!(
                "activity={} execution={} median_seconds={:.6} requested_voxels={}",
                if full { "dense" } else { "sparse" },
                if dense { "reference" } else { "prepared" },
                timings[2],
                voxels
            );
        }
    }
}

#[test]
fn chunk_coalescing_limits_overfetch() {
    let w = |x| ReadWindow {
        image: 0,
        region: Region::new(&[0, 0, x], &[2, 2, 2]),
    };
    let adjacent = coalesce_read_windows_for_chunks(vec![w(0), w(3)], [64; 3]);
    assert_eq!(adjacent.len(), 1);
    assert_eq!(adjacent[0].region.shape, vec![2, 2, 5]);
    assert_eq!(
        coalesce_read_windows_for_chunks(vec![w(0), w(32)], [64; 3]).len(),
        2
    );
}

struct Completion(u8);
impl FragmentOp for Completion {
    fn name(&self) -> &'static str {
        "completion"
    }
    fn reach(&self, _: usize, _: usize) -> usize {
        0
    }
    fn source_inputs(&self, _: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(ImageId::from(0), Reach::symmetric([0; 3]))]
    }
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }
    fn outputs(&self) -> Vec<FragmentOutput> {
        stream("done")
    }
    fn writes_pixels(&self) -> bool {
        self.0 >= 2
    }
    fn background(&self) -> Option<f64> {
        Some(0.0)
    }
    fn prepares(&self) -> bool {
        true
    }
    fn prepare(&self, _: &PreparationView<'_>) -> Result<BlockPreparation> {
        Ok(BlockPreparation::Complete {
            fragments: if self.0 == 0 {
                vec![]
            } else {
                vec![("done".into(), vec![1])]
            },
            constant: if self.0 == 1 || self.0 == 3 {
                Some(7.0)
            } else {
                None
            },
        })
    }
    fn apply(&self, _: &BlockView<'_>) -> Result<BlockOutput> {
        panic!("completed tasks must not apply")
    }
}
#[test]
fn completed_tasks_preserve_output_contracts_and_write_non_background_constants() {
    for mode in 0..4 {
        let mut b = PlanBuilder::new([16; 3], Dtype::U8, BlockGrid::new([16; 3], [8; 3]).unwrap());
        b.fragments(Completion(mode)).unwrap();
        let a = b.finish().unwrap();
        let env = ArrayEnvironment::for_decomposition(
            Voxels::from(Array3::from_elem([16; 3], 0u8)),
            &a.decomposition,
            [4; 3],
        )
        .unwrap();
        let result = execute_phases(
            "complete",
            &a.workflow,
            &a.decomposition,
            &Hints {
                concurrency: 4,
                ..Hints::default()
            },
            &env,
            &[],
            &a.work(),
        );
        assert_eq!(env.counters().read_voxels.load(Ordering::SeqCst), 0);
        if mode < 3 {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap().tasks_short_circuited, 8);
            assert!(env.output().view::<u8>().unwrap().iter().all(|v| *v == 7));
            assert_eq!(env.sidecar_fragments("done").unwrap().len(), 8);
        }
    }
}
#[test]
fn standalone_worker_entry_uses_the_same_preparation() {
    let mut b = PlanBuilder::new([8; 3], Dtype::U8, BlockGrid::new([8; 3], [8; 3]).unwrap());
    b.fragments(Completion(3)).unwrap();
    let a = b.finish().unwrap();
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(Array3::from_elem([8; 3], 0u8)),
        &a.decomposition,
        [4; 3],
    )
    .unwrap();
    env.prepare(&a.decomposition).unwrap();
    env.declare_sidecar("done", Lifecycle::Persistent).unwrap();
    let graph = blockflow::graph::TaskGraph::build(&a.decomposition);
    let outcome = blockflow::strategy::execute_task_of(
        &a.workflow.chain,
        &a.decomposition,
        &graph.tasks[0],
        &a.work()[0],
        &env,
        &[],
    )
    .unwrap();
    assert!(outcome.short_circuited);
    assert_eq!(env.counters().read_voxels.load(Ordering::SeqCst), 0);
    assert!(env.output().view::<u8>().unwrap().iter().all(|v| *v == 7));
}

struct KnownZero {
    inner: ArrayEnvironment,
    known_zero: bool,
    prepared_reads: std::sync::Mutex<Vec<(usize, u32, Region)>>,
    prefetches: std::sync::atomic::AtomicU64,
}
impl Environment for KnownZero {
    fn volume(&self) -> [usize; 3] {
        self.inner.volume()
    }
    fn prepare(&self, d: &blockflow::Decomposition) -> Result<()> {
        self.inner.prepare(d)
    }
    fn read(&self, image: usize, region: &Region) -> Result<blockflow::env::BlockBuf> {
        assert!(!self.known_zero, "known-zero mask must prevent pixel reads");
        self.inner.read(image, region)
    }
    fn apply(
        &self,
        slot: &blockflow::Chain,
        input: &blockflow::env::BlockBuf,
        sources: &[(usize, blockflow::env::BlockBuf)],
        at: &blockflow::op::Placement,
    ) -> Result<blockflow::env::BlockBuf> {
        self.inner.apply(slot, input, sources, at)
    }
    fn write(
        &self,
        image: usize,
        within: &Region,
        valid: &Region,
        buf: &blockflow::env::BlockBuf,
    ) -> Result<()> {
        self.inner.write(image, within, valid, buf)
    }
    fn uniform(&self, buf: &blockflow::env::BlockBuf) -> Option<f64> {
        self.inner.uniform(buf)
    }
    fn known_uniform(&self, image: usize, _: &Region) -> Option<f64> {
        if self.known_zero && image == 0 {
            Some(0.0)
        } else {
            None
        }
    }
    fn constant(&self, d: Dtype, r: &Region, v: f64) -> Result<blockflow::env::BlockBuf> {
        self.inner.constant(d, r, v)
    }
    fn release(&self, b: &blockflow::env::BlockBuf) {
        self.inner.release(b)
    }
    fn finish(&self, image: usize) -> Result<()> {
        self.inner.finish(image)
    }
    fn counters(&self) -> &blockflow::env::EnvCounters {
        self.inner.counters()
    }
    fn sidecars(&self) -> Option<&blockflow::sidecar::Sidecars> {
        self.inner.sidecars()
    }
    fn prefetch_prepared(&self, image: usize, regions: &[(u32, Region)]) -> Result<()> {
        self.prefetches.fetch_add(1, Ordering::SeqCst);
        self.prepared_reads
            .lock()
            .unwrap()
            .extend(regions.iter().map(|(rank, r)| (image, *rank, r.clone())));
        Ok(())
    }
    fn prefetch(&self, _: usize, _: &[Region]) -> Result<()> {
        self.prefetches.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
#[test]
fn storage_metadata_gates_before_pixel_prefetch_and_reads() {
    let mut b = PlanBuilder::new(
        [16; 3],
        Dtype::U16,
        BlockGrid::new([16; 3], [8; 3]).unwrap(),
    );
    b.fragments(
        TabulateValuesOp::new(
            "known zero",
            ImageId::from(0),
            ImageId::supplied(0),
            FixedPoint::default(),
            "partial",
            Lifecycle::Persistent,
        )
        .unwrap()
        .holding(Dtype::U16, Dtype::F64),
    )
    .unwrap();
    let a = b.finish().unwrap();
    let inner = ArrayEnvironment::with_inputs(
        Voxels::from(Array3::from_elem([16; 3], 0u16)),
        vec![Voxels::from(Array3::from_elem([16; 3], 2f64))],
        &a.decomposition,
        [4; 3],
    )
    .unwrap();
    let env = KnownZero {
        inner,
        known_zero: true,
        prepared_reads: std::sync::Mutex::new(Vec::new()),
        prefetches: std::sync::atomic::AtomicU64::new(0),
    };
    let result = execute_phases(
        "known zero",
        &a.workflow,
        &a.decomposition,
        &Hints {
            concurrency: 4,
            prefetch_depth: 3,
            ..Hints::default()
        },
        &env,
        &[],
        &a.work(),
    )
    .unwrap();
    assert_eq!(result.tasks_short_circuited, 8);
    assert_eq!(env.prefetches.load(Ordering::SeqCst), 0);
}

#[test]
fn prepared_prefetch_contains_only_active_windows_with_stable_task_ranks() {
    let n = [24; 3];
    let mut b = PlanBuilder::new(n, Dtype::U16, BlockGrid::new(n, [8; 3]).unwrap());
    let seeds = b.fragments(Seeds(false)).unwrap();
    b.fragments(Sample {
        phase: seeds.index(),
        invalid: false,
    })
    .unwrap();
    let a = b.finish().unwrap();
    let inner = ArrayEnvironment::for_decomposition(
        Voxels::from(Array3::from_elem(n, 1u16)),
        &a.decomposition,
        [4; 3],
    )
    .unwrap();
    let env = KnownZero {
        inner,
        known_zero: false,
        prepared_reads: std::sync::Mutex::new(Vec::new()),
        prefetches: std::sync::atomic::AtomicU64::new(0),
    };
    execute_phases(
        "prefetch",
        &a.workflow,
        &a.decomposition,
        &Hints {
            concurrency: 4,
            prefetch_depth: 2,
            ..Hints::default()
        },
        &env,
        &[],
        &a.work(),
    )
    .unwrap();
    let graph = blockflow::graph::TaskGraph::build(&a.decomposition);
    let expected = graph
        .tasks
        .iter()
        .position(|t| t.phase == 1 && t.index == [0; 3])
        .unwrap() as u32;
    assert_eq!(
        *env.prepared_reads.lock().unwrap(),
        vec![(0, expected, Region::new(&[5; 3], &[2; 3]))]
    );
}
struct OrderSensitive(usize);
impl FragmentOp for OrderSensitive {
    fn name(&self) -> &'static str {
        "order-sensitive preparation"
    }
    fn reach(&self, _: usize, _: usize) -> usize {
        0
    }
    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own("activity", self.0).with_reach([1; 3])]
    }
    fn outputs(&self) -> Vec<FragmentOutput> {
        stream("bad")
    }
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }
    fn prepares(&self) -> bool {
        true
    }
    fn prepare(&self, at: &PreparationView<'_>) -> Result<BlockPreparation> {
        let key = at.fragments("activity").next().unwrap().0;
        Ok(BlockPreparation::Complete {
            fragments: vec![("bad".into(), vec![key.block[0] as u8])],
            constant: None,
        })
    }
    fn apply(&self, _: &BlockView<'_>) -> Result<BlockOutput> {
        panic!("bad preparation must be refused")
    }
}
#[test]
fn unordered_preparation_is_checked_in_both_fragment_orders() {
    let n = [16; 3];
    let mut b = PlanBuilder::new(n, Dtype::U16, BlockGrid::new(n, [8; 3]).unwrap());
    let seeds = b.fragments(Seeds(false)).unwrap();
    b.fragments(OrderSensitive(seeds.index())).unwrap();
    let a = b.finish().unwrap();
    let env = ArrayEnvironment::for_decomposition(
        Voxels::from(Array3::from_elem(n, 1u16)),
        &a.decomposition,
        [4; 3],
    )
    .unwrap();
    let error = execute_phases(
        "bad",
        &a.workflow,
        &a.decomposition,
        &Hints::default(),
        &env,
        &[],
        &a.work(),
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("unordered fragment iteration order"));
}
