// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What a globally consistent label volume costs, measured, on a real volume.**
//
// `ops::label` offers two ways to spend the reconciliation the merge produces —
// materialise a second `u32` image, or decorate reads of the first — and the
// question this file answers is not which is nicer but which costs less, at
// what, and under which condition the answer flips. `tests/global_labels.rs` is
// where they are shown to produce the same bytes; nothing below is evidence
// about a voxel.
//
// What is measured, and why each part of it
// ------------------------------------------
// **A real volume.** A crop of a **recorded** binary segmentation — a `bool`
// `.npy` of `404 x 1304 x 3369`, produced by a real run of a consumer of this
// crate over a real acquisition — read through `blockflow`'s own `NpySource`
// region reader. Its path is named by an environment variable and there is no
// baked-in default, because a machine-local path in a general library's test is
// the same leak the boundary rule is about. The crop is taken on **axis 0**,
// which is the file's slowest axis, so the read is contiguous and the fixture
// cost is not part of what is being compared.
//
// It has to be recorded data rather than a synthetic scene, and the reason is
// specific: the table size, the merge cost and the number of seam crossings are
// all functions of *how the objects are shaped and how they are distributed*,
// not only of the plan's geometry. A generator's objects are placed by a
// generator.
//
// **A real downstream consumer.** `ops::tabulate` — the crate's per-object
// measurement, which is the op the ops-survey index names as the reason a label
// volume is wanted at all. Both arms drive the same consumer over the same
// values, and the consumer was written before either design existed.
//
// **More than one block size**, and the lattice is the axis the answer turns on.
//
// **Read amplification counted, not modelled.** `EnvCounters::read_bytes` is the
// framework's own counter, bumped inside `Environment::read`, and both arms are
// measured with the same instrument. Nothing here multiplies a block count by a
// volume and calls it traffic.
//
// **Resident bytes measured, not modelled.** `VmHWM` from `/proc/self/status`,
// reset between arms through `/proc/self/clear_refs`, because this project has a
// working-buffer residency of a stable 59.6-65.4 bytes a voxel that no declared
// image accounts for — measured by a consumer of this crate, on a different
// chain — so a peak derived from the plan is a lower bound and not a
// measurement.
//
// The two arms, and why they are shaped the same
// -----------------------------------------------
// Both are two `execute_phases` calls over two environments, so that the only
// difference between them is what happens inside the first call:
//
// | | call 1 | between | call 2 |
// |---|---|---|---|
// | **materialise** | `LabelComponentsOp` then `RelabelComponentsOp` — a `u32` local-label image and a `u32` global-label image | nothing | the consumer over the global-label image |
// | **decorate** | `LabelComponentsOp` — a `u32` local-label image | `GlobalLabels::merge`, **once** | the consumer over the local-label image, through `RelabelledEnvironment` |
//
// The consumer plan is character for character the same in both arms: a
// `CarryOp` phase making the values image, then the two tabulation phases. The
// copy is there because `TabulateValuesOp` needs two *different* images and
// declares no element type on its operands, so a **supplied** array cannot be
// its label volume — a small gap, recorded here because it shaped the harness.
// It is paid identically by both arms, and it has a second use: it makes the
// decorated label image be read by **two** readers, which is the cost the
// decorator has and the materialiser does not.
//
// What is *not* claimed
// ----------------------
// Wall clock on a shared machine is the least trustworthy of the three signals
// and is printed with the load average that produced it. The byte counts and the
// component counts are exact. `VmHWM` is honest but is the whole process's, so
// the reset is checked and said so when it fails.
//
// What it said, on this machine
// ------------------------------
// Linux 6.8, 40 cores, 187 GiB total. `MemAvailable` 156-166 GiB throughout.
// **The one-minute load average was 20-45**: this is a shared machine with six
// other jobs on it, so every wall-clock figure is an upper bound on a quiet one,
// and the byte and component counts are exact and load-independent. Read the
// byte columns first. `--release`, `CONCURRENCY = 8`, `Connectivity::Faces`.
// The volume is `[D, 1304, 3369]` of the recording, 2.2-2.5% set.
//
// `D = 16`, **70 290 816 voxels, 23 627 components**:
//
// | blocks | arm | s | read GiB | write GiB | frag, merge | frag, consumer | peak `VmHWM` GiB | table KB |
// |---|---|---|---|---|---|---|---|---|
// | 1 | materialise | 5.23 | 0.655 | 0.524 | 0.067 | 0.017 | 1.781 | — |
// | 1 | decorate | **4.15** | **0.393** | **0.262** | 0.067 | 0.017 | 1.821 | 92.3 |
// | 4 | materialise | 3.10 | 1.440 | 0.524 | 0.170 | 0.052 | 2.666 | — |
// | 4 | decorate | **2.01** | **0.393** | **0.262** | **0.068** | 0.052 | 2.016 | 93.1 |
// | 32 | materialise | 4.87 | 8.772 | 0.524 | 2.241 | 0.357 | 4.101 | — |
// | 32 | decorate | **1.73** | **0.393** | **0.262** | **0.136** | 0.357 | 2.098 | 108.5 |
// | 256 | materialise | 23.50 | 67.427 | 0.524 | 34.864 | 3.250 | 5.227 | — |
// | 256 | decorate | **2.30** | **0.393** | **0.262** | **0.271** | 3.250 | 3.194 | 148.7 |
//
// Every byte column above is **deterministic** — two runs at load 32 and load 40
// produced them to the digit — and the wall clock is not. The fragment columns
// are split into the merge's own traffic and the consumer's, because only the
// first differs between the arms: the consumer's is `0.017 / 0.052 / 0.357 /
// 3.250` in **both**, as it must be, the two arms running the same plan over it.
//
// **There are two amplifications, not one, and they are in different
// currencies.** This was missed on the first pass, which counted pixels only.
//
// | | pixels, at 256 blocks | fragments, at 256 blocks |
// |---|---|---|
// | materialise | 67.427 GiB — `blocks x` the label image | 34.864 GiB — `(1 + blocks) x` every fragment |
// | decorate | 0.393 GiB — flat | 0.271 GiB — `2 x` every fragment, written once and read once |
//
// The pixel half is **quadratic in the block count in total bytes moved**; the
// fragment half is `(1 + blocks) x F(blocks)` with `F` itself growing, which is
// `blocks^(4/3)` on a cubically-cut lattice and much flatter on one that leaves
// the short axis whole — see
// `the_fragment_set_grows_with_the_cut_and_how_fast_depends_on_the_shape_of_it`,
// which sweeps it. Both come from the same clause: the relabelling phase declares a whole-lattice fragment
// reach, which the framework turns into *both* a whole-volume halo (the pixels)
// *and* a per-block gather of every fragment (the fragments). The total extra the
// materialising design pays at 256 blocks is **101.9 GiB** against a decorated
// total of 4.18 GiB — **25.4x the traffic**, of which two thirds is pixels and
// one third is fragments.
//
// **The read ratio is `0.60 / 0.27 / 0.04 / 0.01` at 1 / 4 / 32 / 256 blocks,
// and it is those four numbers at `D = 2` and `D = 8` as well** — identically,
// not approximately. That is what makes them a property of the two designs
// rather than three measurements.
//
// **Four arms, not two, since `barriers.md` asked what the ceiling is.** The two
// middle ones are what the framework would produce with G7 closed — a barrier
// alone, and a barrier with the reduction hoisted out of the per-block loop —
// obtained today by moving the same work out of the plan. Total bytes moved,
// pixels and fragments together, at `D = 16`:
//
// | blocks | in-plan | barrier | barrier + hoisted | merge outside the plan |
// |---|---|---|---|---|
// | 1 | 1.26 | 1.26 | 1.26 | 0.74 |
// | 4 | 2.19 | 1.40 | 1.30 | 0.77 |
// | 32 | 11.89 | 3.78 | 1.67 | 1.15 |
// | 256 | **106.07** | **39.29** | **4.70** | **4.18** |
// | | 25.4x | 9.41x | **1.13x** | 1.00 |
//
// All four agree on 23 627 components and 23 627 rows at every lattice.
//
// **And the largest cost of the per-block reduction is not traffic at all.**
// Timed directly, re-deriving the merge once per block takes **0.10 s at 1
// block, 0.14 at 4, 2.47 at 32 and 33.67 at 256** — serial here, so divide by
// the concurrency for a wall-clock contribution and it is CPU-seconds either
// way. The hoisted arm's *entire run* is 3.80 s. `ops::fill`'s header says the
// redundant union-find "is small next to the pixels"; it is small per
// invocation, at about 0.13 s, and there are `blocks` of them.
//
// **Read amplification is exactly the block count, counted.** The materialising
// arm's reads decompose to the byte as `mask + blocks x (u32 volume) +
// consumer`, and the consumer's own reads are *identical* in the two arms. At
// `D = 16`: `0.0655 + blocks x 0.2618 + 0.3273`, which is 0.655, 1.440, 8.772,
// 67.43 — the table. So what the materialising design costs over the decorated
// one in **pixels** is `4 x blocks` bytes a voxel of reads and `4` of writes.
// That is `ops::fill`'s stated cost, measured: the whole-lattice fragment reach
// is also the halo, so every block of the relabelling phase reads the whole
// label image.
//
// **The fragment half decomposes just as cleanly, and it is the worse of the
// two.** Writing every fragment once and reading it once is `2 x F(blocks)`,
// where `F` is what all the fragments weigh together — the decorated column,
// so `F` is 0.0335, 0.034, 0.068 and 0.1355 GiB. `F` **grows with the block
// count**, because the fragments are the blocks' faces and cutting more finely
// makes more face. The materialising arm pays `(1 + blocks) x F(blocks)`: `2 x
// 0.0335 = 0.067`, `5 x 0.034 = 0.170`, `33 x 0.068 = 2.244`, `257 x 0.1355 =
// 34.82`, against a measured 0.067, 0.170, 2.241, 34.864. So its fragment
// traffic grows faster than the block count and the decorated arm's does not
// grow with the block count at all. **How much faster is a property of the
// lattice's shape and not of its size** — `blocks^(4/3)` when the cut is
// cubical, far flatter when it leaves the volume's shortest axis whole. An
// earlier draft of this header said "worse than quadratic"; that was a fit to
// two points and it is wrong.
//
// **The write was never the cost — and neither was the rebuild.** At one block
// there is no amplification at all, and the materialising arm costs only
// `1.05-1.45x` (0.95x, 0.77x and 0.69x at `D = 16`, `8` and `2`; the spread is
// contention). The floor is exact in bytes and is `8` a voxel: one extra read
// and one extra write of a `u32` volume. **Everything above about `1.2x` in the
// table is the halo, not the write.**
//
// **What one decorated read costs**, by
// `one_decorated_read_costs_what_the_remap_costs_and_no_more`: at or below
// **1.6 ns a voxel**, with no block-count trend, and *it does not reliably
// resolve above zero on this machine*. At 3 repetitions and load 43 one lattice
// came back at **-0.172 ns a voxel** and at 31 repetitions and load 21 another
// came back at **-0.430**; the spread across every run is -0.4 to +2.6. The
// remap is smaller than the run-to-run variance of the 268 MB read it rides on,
// which is a result and not a failed measurement: **the honest figure is a
// bound, not a point**, and the bound is what the arithmetic below uses. Raise
// `BLOCKFLOW_LABEL_REPS` to narrow it on a quieter machine.
//
// So the break-even is a **reader count**, and it is bounded rather than
// pinned. Two ways to it, and they agree:
//
// * *from bytes.* Materialising's floor is 8 bytes a voxel. A 0.262 GiB read of
//   this volume takes 0.23-0.35 s here, so 4 bytes a voxel is 3.3-5.0 ns a
//   voxel and 8 is **7-10**. Against a remap of at most 1.6, that is
//   **R\* = 4-6 readers**.
// * *from the direct difference.* The one-block delta measured 4.6, 18.8 and
//   37.6 ns a voxel at `D = 16`, `8` and `2` — a spread of eight, on a machine
//   at load 28-36, clustering near the 16.5 an earlier quieter run of an earlier
//   harness gave. Against the same bound that is **R\* = 3-12 readers**.
//
// Call it **order ten readers at the coarsest possible cut**, and note that the
// uncertainty is entirely in the wall clock and none of it in the bytes. R*
// rises linearly with the block count: at 256 blocks materialising moves an
// extra **1028 bytes a voxel**, so it is in the hundreds.
//
// **Residency does not follow time.** At 256 blocks the materialising arm peaks
// at `5.223 GiB` against `3.183` — `1.64x`, the difference being the
// whole-volume `u32` buffer its second phase allocates per concurrent block. At
// **one** block the decorated arm is very slightly *worse* (`1.820` against
// `1.780`, `1.02x`), because the table, the label image and the remapped buffer
// are alive together and there is no amplification to pay for the other side.
// Neither figure is a modelled peak: both are `VmHWM`, reset between arms.
//
// **The table is the premise, and it holds by three orders of magnitude.**
// 92-149 KB of reconciliation against a 268 MB label volume — and the ratio
// grows with the block count, because the table gains an entry per block-local
// label while the volume does not move.
//
// **One earlier figure is superseded and is recorded rather than overwritten.**
// The first version of this harness put a `CarryOp` copy in front of the
// tabulation in both arms, because `TabulateValuesOp` declared no element type
// on its operands and so could not read a **supplied** label volume at all. That
// workaround gave the label image two readers instead of one and made the value
// array `u32` instead of `bool`, and it reported read ratios of
// `0.76 / 0.45 / 0.09 / 0.01` and a one-block time ratio of `0.82-0.85x`.
// `TabulateValuesOp::holding` removed the need for it; the ratios above are the
// measurement without it, and they are the ones to quote. The direction and the
// decomposition did not change — only the constants, and only because the
// consumer got cheaper on both sides.
//
// Why it is `#[ignore]`d
// -----------------------
// It reads a machine-local recording and allocates gigabytes. Run it with
// `--release --ignored --nocapture` and read the table.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array3;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::npy::{NpySource, OrderPolicy};
use blockflow::op::Chain;
use blockflow::ops::components::Connectivity;
use blockflow::ops::label::{
    component_label_phases, component_labelling_phase, gather_component_faces, GlobalLabels,
    LabelComponentsOp, RelabelComponentsOp, RelabelledEnvironment,
};
use blockflow::ops::tabulate::{
    collect_tabulation, tabulate_phases, FixedPoint, MergeTabulationOp, TabulateValuesOp,
};
use blockflow::ops::voxelwise::CarryOp;
use blockflow::region::{Region, RegionSource};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

/// Where the recorded volume is, as an environment variable and **with no
/// default**.
///
/// A path is machine-local, and a machine-local path written into a general
/// library's test is a leak of exactly the kind `tests/no_domain_vocabulary.rs`
/// exists to catch. Unset means "there is nothing to measure on here", which is
/// reported and is not a failure. What it must name: a C- or Fortran-ordered
/// `.npy` holding a three-dimensional `bool` array.
const FIXTURE_ENV: &str = "BLOCKFLOW_LABEL_FIXTURE";

/// How much of axis 0 to take. The other two axes are the tile's own, so the
/// only thing that varies between rows is the volume — which is what has to vary
/// for a coefficient to mean anything.
const DEPTH_ENV: &str = "BLOCKFLOW_LABEL_DEPTH";
const DEPTH: usize = 4;

/// Repetitions of the whole-volume read in
/// [`one_decorated_read_costs_what_the_remap_costs_and_no_more`]. Three by
/// default and not enough on a loaded machine; see that test.
const REPS_ENV: &str = "BLOCKFLOW_LABEL_REPS";

/// **Pinned, and stated rather than defaulted.** The machine this runs on is
/// shared, and the materialising arm's second phase holds one whole-volume `u32`
/// buffer per concurrent block, so the peak is a function of this number. Both
/// arms get the same one.
const CONCURRENCY: usize = 8;

const STREAM: &str = "components";
const PARTIALS: &str = "partials";
const ROWS: &str = "rows";
const CONNECTIVITY: Connectivity = Connectivity::Faces;

// ------------------------------------------------------------ the plumbing --

/// A field of `/proc/self/status`, in bytes. The same helper the consumer
/// crates' residency measurements use, restated rather than shared because they
/// are different crates.
fn status_field(name: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = text.lines().find(|line| line.starts_with(name))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Reset the high-water mark, and check that it took.
///
/// Linux clears `VmHWM` on a write of `5` to `/proc/self/clear_refs`. On a
/// kernel that does not, every peak below is the peak since process start and
/// the differences between rows stop meaning anything, which the caller has to
/// be told rather than quietly averaged over.
fn reset_peak_rss() -> bool {
    if std::fs::write("/proc/self/clear_refs", "5\n").is_err() {
        return false;
    }
    match (status_field("VmHWM"), status_field("VmRSS")) {
        // A tolerance rather than equality: the two are read by two separate
        // opens of `/proc/self/status` and the allocator moves between them.
        (Some(peak), Some(now)) => peak <= now + 64 * 1024 * 1024,
        _ => false,
    }
}

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse().ok())
        .unwrap_or(f64::NAN)
}

fn mem_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemAvailable"))
                .and_then(|line| line.split_whitespace().nth(1)?.parse::<u64>().ok())
                .map(|kb| kb * 1024)
        })
        .unwrap_or(0)
}

/// What the framework's own counters say one environment moved.
#[derive(Debug, Default, Clone, Copy)]
struct Traffic {
    reads: u64,
    read_bytes: u64,
    writes: u64,
    write_bytes: u64,
    sidecar_bytes: u64,
}

impl Traffic {
    fn of(env: &dyn Environment) -> Self {
        let counters = env.counters();
        Self {
            reads: counters.reads.load(Ordering::Relaxed),
            read_bytes: counters.read_bytes.load(Ordering::Relaxed),
            writes: counters.writes.load(Ordering::Relaxed),
            write_bytes: counters.write_bytes.load(Ordering::Relaxed),
            sidecar_bytes: counters.sidecar_bytes_written.load(Ordering::Relaxed)
                + counters.sidecar_bytes_read.load(Ordering::Relaxed),
        }
    }

    fn plus(self, other: Self) -> Self {
        Self {
            reads: self.reads + other.reads,
            read_bytes: self.read_bytes + other.read_bytes,
            writes: self.writes + other.writes,
            write_bytes: self.write_bytes + other.write_bytes,
            sidecar_bytes: self.sidecar_bytes + other.sidecar_bytes,
        }
    }
}

#[derive(Debug, Clone)]
struct Arm {
    seconds: f64,
    peak_rss: u64,
    #[allow(dead_code)]
    rss_after: u64,
    load: (f64, f64),
    reset: bool,
    traffic: Traffic,
    /// The labelling-and-merge half on its own: everything before the consumer
    /// plan starts. This is where the two arms differ, and keeping it apart from
    /// the consumer's own traffic is what makes the difference attributable.
    upstream: Traffic,
    components: u32,
    table_bytes: usize,
    rows: usize,
}

/// Every byte an arm moved: pixels in, pixels out, and fragments both ways.
///
/// Summed rather than compared column by column because the three designs move
/// their bytes in different currencies — the whole point of the comparison is
/// that one of them trades a pixel read for a fragment gather — and a ratio on
/// one column would flatter whichever design happens to be cheap in it.
fn total(arm: &Arm) -> u64 {
    arm.traffic.read_bytes + arm.traffic.write_bytes + arm.traffic.sidecar_bytes
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn hints() -> Hints {
    Hints {
        concurrency: CONCURRENCY,
        ..Hints::default()
    }
}

// ------------------------------------------------------------ the fixture --

fn fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(FIXTURE_ENV).ok()?);
    path.exists().then_some(path)
}

/// A crop of the recording, taken on axis 0 so that the read is contiguous.
fn crop(path: &PathBuf, depth: usize) -> (Array3<bool>, [usize; 3]) {
    let source: NpySource<bool> =
        NpySource::open(path, OrderPolicy::Either).expect("the recording opens");
    let shape = source.shape().to_vec();
    assert_eq!(shape.len(), 3, "the recording is a volume");
    let volume = [depth.min(shape[0]), shape[1], shape[2]];
    let region = Region::new(&[0, 0, 0], &volume);
    let read = source.read_region(&region).expect("the crop reads");
    let mask = read
        .into_dimensionality::<ndarray::Ix3>()
        .expect("three axes")
        .as_standard_layout()
        .to_owned();
    (mask, volume)
}

// -------------------------------------------------------------- the arms --

/// The consumer, identical in both arms: `ops::tabulate` over a **supplied**
/// label volume, with the mask as the value array.
///
/// **Supplied, which is what a real consumer is handed.** The labelling ran in
/// its own plan; this plan is given its result beside its own input. That needs
/// `TabulateValuesOp::holding`, because a supplied array is produced by no phase
/// and no fold of the plan can say what is in it.
///
/// It also means the label volume has **exactly one reader**, which matters to
/// the comparison: the decorated design pays the remap per reader, so one reader
/// is the arrangement least favourable to it.
fn consumer_plan(
    volume: [usize; 3],
    block: [usize; 3],
) -> (Decomposition, TabulateValuesOp, MergeTabulationOp) {
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let lattice = grid.blocks_per_axis();
    let fixed = FixedPoint::default();
    let tabulate = TabulateValuesOp::new(
        "tabulate",
        ImageId::supplied(0),
        0usize,
        fixed,
        PARTIALS,
        Lifecycle::DeleteOnExit,
    )
    .expect("two different images")
    .holding(Dtype::U32, Dtype::Bool);
    let merge = MergeTabulationOp::new(
        "merge",
        PARTIALS,
        0,
        lattice,
        fixed,
        ROWS,
        Lifecycle::Persistent,
    );
    let plan = tabulate_phases(grid, Dtype::Bool, &tabulate, &merge).expect("a plan");
    (plan, tabulate, merge)
}

fn run_consumer(
    plan: &Decomposition,
    tabulate: &TabulateValuesOp,
    merge: &MergeTabulationOp,
    env: &dyn Environment,
) -> usize {
    let workflow = Workflow::new(Chain::sequence(Vec::new()), plan.volume, Dtype::Bool);
    execute_phases(
        "consume",
        &workflow,
        plan,
        &hints(),
        env,
        &[],
        &[PhaseWork::Fragments(tabulate), PhaseWork::Fragments(merge)],
    )
    .expect("a run");
    collect_tabulation(env, ROWS, 1, plan.volume, FixedPoint::default())
        .expect("the rows")
        .len()
}

/// **Arm 1 — materialise.** Two fragment phases write a global label volume;
/// the consumer reads an ordinary image.
fn materialise(mask: &Array3<bool>, volume: [usize; 3], block: [usize; 3]) -> Arm {
    let reset = reset_peak_rss();
    let load_before = loadavg();
    let started = Instant::now();

    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let label =
        LabelComponentsOp::new("label", STREAM, Lifecycle::DeleteOnExit).connecting(CONNECTIVITY);
    let relabel = RelabelComponentsOp::new("relabel", STREAM, 0, &grid).connecting(CONNECTIVITY);
    let plan = component_label_phases(grid, Dtype::Bool, &label, &relabel).expect("a plan");
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
    execute_phases(
        "materialise",
        &workflow,
        &plan,
        &hints(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&relabel)],
    )
    .expect("a run");
    let labels = env.output();
    let components = labels
        .view::<u32>()
        .expect("labels")
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let first = Traffic::of(&env);
    drop(env);

    let (downstream_plan, tabulate, merge) = consumer_plan(volume, block);
    let downstream = ArrayEnvironment::with_inputs(
        mask.clone().into(),
        vec![labels],
        &downstream_plan,
        chunk(block),
    )
    .expect("environment");
    let rows = run_consumer(&downstream_plan, &tabulate, &merge, &downstream);
    let traffic = first.plus(Traffic::of(&downstream));
    drop(downstream);

    Arm {
        seconds: started.elapsed().as_secs_f64(),
        peak_rss: status_field("VmHWM").unwrap_or(0),
        rss_after: status_field("VmRSS").unwrap_or(0),
        load: (load_before, loadavg()),
        reset,
        traffic,
        upstream: first,
        components,
        table_bytes: 0,
        rows,
    }
}

/// **Arm 2 — decorate.** One fragment phase, one merge, and the consumer reads
/// the local-label image through a `RelabelledEnvironment`.
fn decorate(mask: &Array3<bool>, volume: [usize; 3], block: [usize; 3]) -> Arm {
    let reset = reset_peak_rss();
    let load_before = loadavg();
    let started = Instant::now();

    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let label =
        LabelComponentsOp::new("label", STREAM, Lifecycle::Persistent).connecting(CONNECTIVITY);
    let plan = component_labelling_phase(grid.clone(), Dtype::Bool, &label).expect("a plan");
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
    execute_phases(
        "label",
        &workflow,
        &plan,
        &hints(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label)],
    )
    .expect("a run");

    // The merge, once — not per block, and not as a phase.
    let reports: BTreeMap<_, _> =
        gather_component_faces(&env, STREAM, 0, &grid).expect("every block wrote one");
    let map = Arc::new(GlobalLabels::merge(&reports, &grid, CONNECTIVITY).expect("the merge"));
    let components = map.components();
    let table_bytes = map.table_bytes();
    drop(reports);

    let labels = env.output();
    let first = Traffic::of(&env);
    drop(env);

    let (downstream_plan, tabulate, merge) = consumer_plan(volume, block);
    let downstream = ArrayEnvironment::with_inputs(
        mask.clone().into(),
        vec![labels],
        &downstream_plan,
        chunk(block),
    )
    .expect("environment");
    let view = RelabelledEnvironment::new(&downstream, ImageId::supplied(0), map);
    let rows = run_consumer(&downstream_plan, &tabulate, &merge, &view);
    assert!(
        view.remapped_reads() > 0,
        "the consumer read nothing through the decorator"
    );
    let traffic = first.plus(Traffic::of(&downstream));
    drop(view);
    drop(downstream);

    Arm {
        seconds: started.elapsed().as_secs_f64(),
        peak_rss: status_field("VmHWM").unwrap_or(0),
        rss_after: status_field("VmRSS").unwrap_or(0),
        load: (load_before, loadavg()),
        reset,
        traffic,
        upstream: first,
        components,
        table_bytes,
        rows,
    }
}

/// The chunk shape the environment reports. Kept at the block edge clamped to
/// something small, because it is only used for the chunk counters and both arms
/// get the same.
fn chunk(block: [usize; 3]) -> [usize; 3] {
    [block[0].min(64), block[1].min(64), block[2].min(64)]
}

/// **Arm 3 — the ceiling.** The merge runs **once**, and the relabelling reads
/// its table.
///
/// This is what a barrier phase permitted to run its reduction once would
/// produce, obtained today by moving both out of the plan: one labelling phase,
/// one `GlobalLabels::merge`, then a reach-0 `CarryOp` reading the local labels
/// **through the decorator** and writing an ordinary global-label image, then the
/// consumer over that image with no decoration at all.
///
/// Its traffic is the traffic an in-plan barrier phase with a hoisted reduction
/// would have, term for term: one read of the label image, one write of the
/// global one, one gather of every fragment. What it is *not* is the same
/// schedule — it is three `execute_phases` calls where a closed G7 would be one
/// plan — so the wall clock is an upper bound and the byte columns are the
/// measurement.
fn ceiling(mask: &Array3<bool>, volume: [usize; 3], block: [usize; 3]) -> Arm {
    let reset = reset_peak_rss();
    let load_before = loadavg();
    let started = Instant::now();

    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let label =
        LabelComponentsOp::new("label", STREAM, Lifecycle::Persistent).connecting(CONNECTIVITY);
    let plan = component_labelling_phase(grid.clone(), Dtype::Bool, &label).expect("a plan");
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
    execute_phases(
        "label",
        &workflow,
        &plan,
        &hints(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label)],
    )
    .expect("a run");

    let reports = gather_component_faces(&env, STREAM, 0, &grid).expect("every block wrote one");
    let map = Arc::new(GlobalLabels::merge(&reports, &grid, CONNECTIVITY).expect("the merge"));
    let components = map.components();
    let table_bytes = map.table_bytes();
    drop(reports);
    let local = env.output();
    let mut upstream = Traffic::of(&env);
    drop(env);

    // The relabelling: reach 0, reading the table rather than rebuilding it.
    let mut builder = PlanBuilder::new(volume, Dtype::U32, grid);
    builder
        .pixels(Chain::op(CarryOp::new("relabel")))
        .expect("a pixel phase");
    let assembly = builder.finish().expect("a plan");
    let relabel_env =
        ArrayEnvironment::for_decomposition(local, &assembly.decomposition, chunk(block))
            .expect("environment");
    let view = RelabelledEnvironment::new(&relabel_env, 0usize, map);
    execute_phases(
        "relabel",
        &assembly.workflow,
        &assembly.decomposition,
        &hints(),
        &view,
        &[],
        &assembly.work(),
    )
    .expect("a run");
    assert!(
        view.remapped_reads() > 0,
        "the relabelling read nothing through the decorator"
    );
    let labels = relabel_env.output();
    upstream = upstream.plus(Traffic::of(&relabel_env));
    drop(view);
    drop(relabel_env);

    let (downstream_plan, tabulate, merge) = consumer_plan(volume, block);
    let downstream = ArrayEnvironment::with_inputs(
        mask.clone().into(),
        vec![labels],
        &downstream_plan,
        chunk(block),
    )
    .expect("environment");
    let rows = run_consumer(&downstream_plan, &tabulate, &merge, &downstream);
    let traffic = upstream.plus(Traffic::of(&downstream));
    drop(downstream);

    Arm {
        seconds: started.elapsed().as_secs_f64(),
        peak_rss: status_field("VmHWM").unwrap_or(0),
        rss_after: status_field("VmRSS").unwrap_or(0),
        load: (load_before, loadavg()),
        reset,
        traffic,
        upstream,
        components,
        table_bytes,
        rows,
    }
}

/// **Arm 4 — a barrier and nothing else.** The halo is relieved; the reduction is
/// still re-derived per block.
///
/// The point of measuring this rather than inferring it: `barriers.md` §3.3
/// claimed a barrier alone leaves most of the cost on the table, and that claim
/// was arithmetic on two columns of a table. This is the arithmetic performed by
/// a machine. It differs from [`ceiling`] in exactly one line — the merge is run
/// once per block of the lattice instead of once — which is what an in-plan
/// barrier phase does today, because `check_phase_work` gives it nowhere to put
/// a per-phase result.
///
/// It measures a second thing the composed figure could not: **the CPU of
/// re-deriving the union-find `blocks` times**, which is not in any byte column
/// and which nobody had priced.
fn barrier_only(mask: &Array3<bool>, volume: [usize; 3], block: [usize; 3]) -> Arm {
    let reset = reset_peak_rss();
    let load_before = loadavg();
    let started = Instant::now();

    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let counts = grid.blocks_per_axis();
    let blocks: usize = counts.iter().product();
    let label =
        LabelComponentsOp::new("label", STREAM, Lifecycle::Persistent).connecting(CONNECTIVITY);
    let plan = component_labelling_phase(grid.clone(), Dtype::Bool, &label).expect("a plan");
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
    execute_phases(
        "label",
        &workflow,
        &plan,
        &hints(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label)],
    )
    .expect("a run");

    // **Once per block**, gather and merge — which is what the in-plan phase
    // does, and the only thing that separates this arm from the one above.
    let merging = Instant::now();
    let mut map = None;
    for _ in 0..blocks {
        let reports = gather_component_faces(&env, STREAM, 0, &grid).expect("every block");
        map = Some(GlobalLabels::merge(&reports, &grid, CONNECTIVITY).expect("the merge"));
    }
    let merge_seconds = merging.elapsed().as_secs_f64();
    let map = Arc::new(map.expect("a lattice has at least one block"));
    let components = map.components();
    let table_bytes = map.table_bytes();
    let local = env.output();
    let mut upstream = Traffic::of(&env);
    drop(env);

    let mut builder = PlanBuilder::new(volume, Dtype::U32, grid);
    builder
        .pixels(Chain::op(CarryOp::new("relabel")))
        .expect("a pixel phase");
    let assembly = builder.finish().expect("a plan");
    let relabel_env =
        ArrayEnvironment::for_decomposition(local, &assembly.decomposition, chunk(block))
            .expect("environment");
    let view = RelabelledEnvironment::new(&relabel_env, 0usize, map);
    execute_phases(
        "relabel",
        &assembly.workflow,
        &assembly.decomposition,
        &hints(),
        &view,
        &[],
        &assembly.work(),
    )
    .expect("a run");
    let labels = relabel_env.output();
    upstream = upstream.plus(Traffic::of(&relabel_env));
    drop(view);
    drop(relabel_env);

    let (downstream_plan, tabulate, merge) = consumer_plan(volume, block);
    let downstream = ArrayEnvironment::with_inputs(
        mask.clone().into(),
        vec![labels],
        &downstream_plan,
        chunk(block),
    )
    .expect("environment");
    let rows = run_consumer(&downstream_plan, &tabulate, &merge, &downstream);
    let traffic = upstream.plus(Traffic::of(&downstream));
    drop(downstream);

    eprintln!(
        "{:>26} | re-deriving the merge {blocks} time(s) took {merge_seconds:.2} s",
        ""
    );
    Arm {
        seconds: started.elapsed().as_secs_f64(),
        peak_rss: status_field("VmHWM").unwrap_or(0),
        rss_after: status_field("VmRSS").unwrap_or(0),
        load: (load_before, loadavg()),
        reset,
        traffic,
        upstream,
        components,
        table_bytes,
        rows,
    }
}

// ------------------------------------------------------------ the sweep --

/// The lattices swept. Distinct as lattices, and the sweep spans the case with
/// no seams at all through one with hundreds of blocks — which is the axis the
/// answer turns on, since the materialising phase's halo is the whole volume and
/// therefore its read amplification is the block count.
fn blockings(volume: [usize; 3]) -> Vec<[usize; 3]> {
    // Stated as **lattices** and turned into block edges by a ceiling divide,
    // because the interesting quantity is the block count and a block edge
    // chosen by hand on an axis of 3369 produces a one-voxel remainder block and
    // a lattice one larger than intended.
    [[1usize, 1, 1], [1, 2, 2], [2, 4, 4], [4, 8, 8]]
        .into_iter()
        .map(|counts| {
            let mut block = [0usize; 3];
            for axis in 0..3 {
                let want = counts[axis].min(volume[axis]);
                block[axis] = volume[axis].div_ceil(want);
            }
            block
        })
        .collect()
}

#[test]
#[ignore = "reads a machine-local recorded volume and allocates gigabytes; run with --release --ignored --nocapture"]
fn materialising_and_decorating_a_global_label_volume_are_measured_against_each_other() {
    let Some(path) = fixture() else {
        eprintln!(
            "\n{FIXTURE_ENV} names no readable file, so there is no real volume to measure on. \
             Point it at a three-dimensional `bool` `.npy`. Nothing is asserted and nothing is \
             reported."
        );
        return;
    };
    let depth: usize = std::env::var(DEPTH_ENV)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(DEPTH);

    let (mask, volume) = crop(&path, depth);
    let voxels: usize = volume.iter().product();
    let set = mask.iter().filter(|&&v| v).count();
    eprintln!(
        "\n{}\n  volume {volume:?} = {voxels} voxels, {:.2}% set\n  MemAvailable {:.1} GiB, load {:.2}, concurrency {CONCURRENCY}",
        path.display(),
        100.0 * set as f64 / voxels as f64,
        gib(mem_available()),
        loadavg(),
    );

    eprintln!(
        "\n{:>18} {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>9} | {:>9} | {:>8} | {:>8} | {:>7}",
        "block",
        "blocks",
        "arm",
        "seconds",
        "read GiB",
        "write GiB",
        "frag merge",
        "frag cons",
        "peak GiB",
        "table KB",
        "load"
    );

    let mut seen: Vec<[usize; 3]> = Vec::new();
    for block in blockings(volume) {
        let grid = BlockGrid::new(volume, block).expect("a lattice");
        let counts = grid.blocks_per_axis();
        assert!(
            !seen.contains(&counts),
            "block {block:?} gives lattice {counts:?}, which another already gives"
        );
        seen.push(counts);
        let blocks: usize = counts.iter().product();

        let m = materialise(&mask, volume, block);
        report(block, blocks, "in-plan", &m);
        let b = barrier_only(&mask, volume, block);
        report(block, blocks, "barrier", &b);
        let c = ceiling(&mask, volume, block);
        report(block, blocks, "ceiling", &c);
        let d = decorate(&mask, volume, block);
        report(block, blocks, "decorate", &d);
        assert_eq!(
            c.components, d.components,
            "the ceiling arm found a different component count at {block:?}"
        );
        assert_eq!(
            c.rows, d.rows,
            "the ceiling arm tabulated differently at {block:?}"
        );

        // The two arms must agree about the volume they described, or the
        // comparison is between two different answers.
        assert_eq!(
            m.components, d.components,
            "the two arms found different component counts at {block:?}"
        );
        assert_eq!(
            m.rows, d.rows,
            "the two arms tabulated different row counts at {block:?}"
        );
        eprintln!(
            "{:>26} | {} components, {} rows | total GiB: in-plan {:.2}, \
             barrier {:.2}, ceiling {:.2}, decorate {:.2} | over decorate: in-plan {:.1}x, \
             barrier {:.2}x, ceiling {:.2}x",
            "",
            m.components,
            m.rows,
            gib(total(&m)),
            gib(total(&b)),
            gib(total(&c)),
            gib(total(&d)),
            total(&m) as f64 / total(&d) as f64,
            total(&b) as f64 / total(&d) as f64,
            total(&c) as f64 / total(&d) as f64,
        );
    }
    assert!(
        seen.len() >= 3,
        "a comparison that turns on the block count needs more than two block counts"
    );
}

fn report(block: [usize; 3], blocks: usize, arm: &str, cost: &Arm) {
    eprintln!(
        "{:>18} {:>7} | {:>8} | {:>8.2} | {:>8.3} | {:>8.3} | {:>9.3} | {:>9.3} | {:>8.3} | \
         {:>7.1} | {:>3.1}->{:.1}{}",
        format!("{block:?}"),
        blocks,
        arm,
        cost.seconds,
        gib(cost.traffic.read_bytes),
        gib(cost.traffic.write_bytes),
        gib(cost.upstream.sidecar_bytes),
        gib(cost.traffic.sidecar_bytes - cost.upstream.sidecar_bytes),
        gib(cost.peak_rss),
        cost.table_bytes as f64 / 1024.0,
        cost.load.0,
        cost.load.1,
        if cost.reset {
            ""
        } else {
            "  [PEAK NOT RESET: process-wide]"
        }
    );
}

// -------------------------------------------- what one decorated read costs --

/// **The marginal cost of decorating one read**, which is the quantity the
/// recommendation's condition is stated in.
///
/// The decorated design pays the remap **per reader** and the materialised one
/// pays a rewrite **once**, so "which is cheaper" is a question about how many
/// readers a label volume has. The sweep above fixes the consumer at two readers
/// of the label image; this measures the slope, so that the break-even reader
/// count can be computed rather than guessed:
///
/// ```text
///   R* = (what materialising costs beyond decorating, at one reader)
///        / (what one decorated pass costs beyond a plain one)
/// ```
///
/// Both halves are measured, on the same volume, with the same instrument. The
/// plain read is the *same* read through the *same* environment with the
/// decorator taken off, so the difference is the remap and nothing else — the
/// allocation, the copy out of the image and the counter bump are in both.
#[test]
#[ignore = "reads a machine-local recorded volume; run with --release --ignored --nocapture"]
fn one_decorated_read_costs_what_the_remap_costs_and_no_more() {
    let Some(path) = fixture() else {
        eprintln!("\n{FIXTURE_ENV} names no readable file; nothing measured.");
        return;
    };
    let depth: usize = std::env::var(DEPTH_ENV)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(DEPTH);
    let (mask, volume) = crop(&path, depth);
    let voxels: usize = volume.iter().product();
    // The difference being measured is a fraction of a second against a read of
    // a few hundred megabytes, so on a busy machine three repetitions do not
    // resolve it — the run that produced the table in the header came back
    // *negative* at the coarsest lattice at load 43. Overridable so that a
    // measurement can be made to converge rather than repeated and averaged by
    // hand.
    let reps: usize = std::env::var(REPS_ENV)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(3);

    eprintln!(
        "\n  volume {volume:?} = {voxels} voxels, load {:.2}, MemAvailable {:.1} GiB",
        loadavg(),
        gib(mem_available())
    );
    eprintln!(
        "\n{:>18} {:>7} | {:>10} | {:>10} | {:>12} | {:>10}",
        "block", "blocks", "plain s", "decorated s", "remap ns/vox", "table KB"
    );

    for block in blockings(volume) {
        let grid = BlockGrid::new(volume, block).expect("a lattice");
        let blocks: usize = grid.blocks_per_axis().iter().product();
        let label =
            LabelComponentsOp::new("label", STREAM, Lifecycle::Persistent).connecting(CONNECTIVITY);
        let plan = component_labelling_phase(grid.clone(), Dtype::Bool, &label).expect("a plan");
        let input: Voxels = mask.clone().into();
        let env =
            ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
        let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
        execute_phases(
            "label",
            &workflow,
            &plan,
            &hints(),
            &env,
            &[],
            &[PhaseWork::Fragments(&label)],
        )
        .expect("a run");
        let reports = gather_component_faces(&env, STREAM, 0, &grid).expect("every block");
        let map = Arc::new(GlobalLabels::merge(&reports, &grid, CONNECTIVITY).expect("the merge"));
        let table_bytes = map.table_bytes();

        let whole = Region::whole(&volume);
        // plain, through the undecorated environment
        let started = Instant::now();
        for _ in 0..reps {
            let buf = env.read(1, &whole).expect("a read");
            std::hint::black_box(&buf);
            env.release(&buf);
        }
        let plain = started.elapsed().as_secs_f64() / reps as f64;

        let view = RelabelledEnvironment::new(&env, 1usize, map);
        let started = Instant::now();
        for _ in 0..reps {
            let buf = view.read(1, &whole).expect("a decorated read");
            std::hint::black_box(&buf);
            view.release(&buf);
        }
        let decorated = started.elapsed().as_secs_f64() / reps as f64;

        eprintln!(
            "{:>18} {:>7} | {:>10.4} | {:>10.4} | {:>12.3} | {:>10.1}",
            format!("{block:?}"),
            blocks,
            plain,
            decorated,
            (decorated - plain) * 1e9 / voxels as f64,
            table_bytes as f64 / 1024.0,
        );
        assert!(
            view.remapped_reads() as usize == reps,
            "the decorator rewrote {} of {reps} reads",
            view.remapped_reads()
        );
    }
}

// ------------------------------------------- the acceptance bar, on real data --

/// **Byte-identical across lattices and across designs, on the recorded
/// volume.**
///
/// `tests/global_labels.rs` establishes this on a fixture built to be hard; this
/// establishes it on data nobody arranged, at a size where a block-local
/// labelling has tens of thousands of components and the seam traffic is real.
/// The whole-volume reference is the one-block *materialised* run, which has no
/// seams at all, and every other lattice and both designs are compared against
/// its bytes.
#[test]
#[ignore = "reads a machine-local recorded volume and allocates gigabytes; run with --release --ignored --nocapture"]
fn the_two_designs_agree_byte_for_byte_on_the_real_volume_at_every_lattice() {
    let Some(path) = fixture() else {
        eprintln!("\n{FIXTURE_ENV} names no readable file; nothing measured.");
        return;
    };
    let depth: usize = std::env::var(DEPTH_ENV)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(DEPTH);
    let (mask, volume) = crop(&path, depth);

    let mut want: Option<Array3<u32>> = None;
    let mut lattices = 0usize;
    for block in blockings(volume) {
        let grid = BlockGrid::new(volume, block).expect("a lattice");
        let blocks: usize = grid.blocks_per_axis().iter().product();

        // the materialising design
        let label = LabelComponentsOp::new("label", STREAM, Lifecycle::DeleteOnExit)
            .connecting(CONNECTIVITY);
        let relabel =
            RelabelComponentsOp::new("relabel", STREAM, 0, &grid).connecting(CONNECTIVITY);
        let plan =
            component_label_phases(grid.clone(), Dtype::Bool, &label, &relabel).expect("a plan");
        let input: Voxels = mask.clone().into();
        let env =
            ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
        let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
        execute_phases(
            "materialise",
            &workflow,
            &plan,
            &hints(),
            &env,
            &[],
            &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&relabel)],
        )
        .expect("a run");
        let got = env.output().view::<u32>().expect("labels").to_owned();
        drop(env);

        // the decorating design, over the same lattice
        let label =
            LabelComponentsOp::new("label", STREAM, Lifecycle::Persistent).connecting(CONNECTIVITY);
        let plan = component_labelling_phase(grid.clone(), Dtype::Bool, &label).expect("a plan");
        let input: Voxels = mask.clone().into();
        let env =
            ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
        execute_phases(
            "label",
            &workflow,
            &plan,
            &hints(),
            &env,
            &[],
            &[PhaseWork::Fragments(&label)],
        )
        .expect("a run");
        let reports = gather_component_faces(&env, STREAM, 0, &grid).expect("every block");
        let map = Arc::new(GlobalLabels::merge(&reports, &grid, CONNECTIVITY).expect("the merge"));
        let view = RelabelledEnvironment::new(&env, 1usize, map);
        let buf = view
            .read(1, &Region::whole(&volume))
            .expect("a decorated read");
        let blockflow::env::BlockBuf::Array(voxels) = buf else {
            unreachable!("an array environment answers with an array");
        };
        let lazily = voxels.view::<u32>().expect("labels").to_owned();
        drop(view);
        drop(env);

        assert_eq!(
            got, lazily,
            "the two designs disagreed at {blocks} blocks ({block:?})"
        );
        match &want {
            None => {
                assert_eq!(blocks, 1, "the reference row must be the one-block run");
                let components = got.iter().copied().max().unwrap_or(0);
                assert!(
                    components > 1000,
                    "only {components} components: the crop is too thin for this to mean much"
                );
                eprintln!("\n  reference: {blocks} block, {components} components");
                want = Some(got);
            }
            Some(want) => {
                assert_eq!(
                    &got, want,
                    "{blocks} blocks ({block:?}) disagreed with the reference"
                );
                eprintln!("  {blocks} blocks ({block:?}): byte-identical, both designs");
            }
        }
        lattices += 1;
    }
    assert!(
        lattices >= 3,
        "an invariance claim over two lattices is not one"
    );
}

/// `f64::max` by the crate's rule — `total_cmp`, never `min`/`max`.
trait TotalCmpMax {
    fn total_cmp_max(self, other: f64) -> f64;
}

impl TotalCmpMax for f64 {
    fn total_cmp_max(self, other: f64) -> f64 {
        match self.total_cmp(&other) {
            std::cmp::Ordering::Less => other,
            _ => self,
        }
    }
}

// ------------------------------------- what the fragment set costs, and why --

/// **How much the fragments weigh, swept over the block count *and the shape of
/// the cut*.**
///
/// `F(blocks)` — what every block's fragment weighs together — turned out to
/// grow with the block count, which is why the in-plan merge's traffic is
/// `(1 + blocks) x F(blocks)` and not `(1 + blocks) x` a constant. That growth
/// needs separating into two questions that are not the same question:
///
/// 1. **must the fragment set be that big?** Partly yes and it is geometry: a
///    fragment is a block's six faces, cutting more finely makes more face, and
///    no scheme that communicates block boundaries escapes it;
/// 2. **must it be re-transmitted per block?** No, and that is the whole of the
///    difference between the arms above.
///
/// The first is what this sweeps, and it sweeps the thing every figure in this
/// file has so far held fixed: **the shape of the lattice.**
///
/// What it said
/// ------------
/// `F` is **the total face area of the cut**, and it obeys one line:
///
/// > `F = sum over axes of (cuts on that axis) x (area of the face perpendicular
/// > to it)`
///
/// On `[16, 1304, 3369]` the axis-0 face is `1304 x 3369` and the axis-2 face is
/// `16 x 1304` — **sixty-six times smaller** — so `F` is very nearly a function
/// of the axis-0 cuts alone. Measured, with the number of axis-0 divisions in
/// brackets: `0.0335 (1)`, `0.0668 (2)`, `0.1334 (4)`, `0.1357 (4)`, `0.2667
/// (8)` GiB. That is linear in the divisions of the shortest axis to three
/// figures, and the `[4, 8, 8]` row sits off the block-count trend for exactly
/// that reason — it makes 256 blocks while cutting the expensive axis only four
/// ways.
///
/// **The prediction that motivated this sweep was backwards, which is the most
/// useful thing it produced.** The expectation written down first was that a
/// slab cut would carry *more* face than a cube cut at equal block count,
/// reasoning about seam planes in a cube. This volume is not a cube. At 64
/// blocks the cube cut carries **3.08x** the fragments of the slab cut, because
/// the cube cut divides the 16-voxel axis and the slab cut does not touch it.
/// The assertion below encodes the mechanism rather than the guess.
///
/// Two consequences worth having:
///
/// * **cutting the short axis is doubly expensive.** In-plan fragment traffic is
///   `(1 + blocks) x F`, and dividing the short axis raises *both* factors. At
///   482 slab blocks `(1 + n) F` is 52.4 GiB; at 512 cube blocks it is **136.8**.
/// * **"the fragments are small next to the pixels" stops being true.** At `[8,
///   8, 8]` the fragment set is **101.9%** of the label image — one transmission
///   of the fragments costs more than one transmission of the whole volume. That
///   sentence appears in `ops::fill`'s header and in earlier drafts of this file,
///   and it is a statement about coarse lattices that nobody swept. All four sweeps
/// above cut roughly cubically. A `1 x 1 x n` slab cut and an `n^(1/3)` cube cut
/// at the *same block count* are not the same amount of face, and an argument
/// that treats "the number of blocks" as the variable is resting on an unswept
/// assumption. Two rows below have equal block counts and are named so that the
/// comparison is the point rather than a coincidence.
#[test]
#[ignore = "reads a machine-local recorded volume; run with --release --ignored --nocapture"]
fn the_fragment_set_grows_with_the_cut_and_how_fast_depends_on_the_shape_of_it() {
    let Some(path) = fixture() else {
        eprintln!("\n{FIXTURE_ENV} names no readable file; nothing measured.");
        return;
    };
    let depth: usize = std::env::var(DEPTH_ENV)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(DEPTH);
    let (mask, volume) = crop(&path, depth);
    let voxels: usize = volume.iter().product();
    let image = voxels * std::mem::size_of::<u32>();

    // (name, lattice). The `cube` rows cut all three axes; the `slab` rows cut
    // one. `64` and `256` appear in both families so that block count is
    // controlled and only the shape varies.
    let lattices: [(&str, [usize; 3]); 9] = [
        ("cube 1", [1, 1, 1]),
        ("cube 8", [2, 2, 2]),
        ("cube 64", [4, 4, 4]),
        ("cube 256", [4, 8, 8]),
        ("cube 512", [8, 8, 8]),
        ("slab 4", [1, 1, 4]),
        ("slab 64", [1, 1, 64]),
        ("slab 256", [1, 1, 256]),
        ("slab 512", [1, 1, 512]),
    ];

    eprintln!(
        "\n  volume {volume:?}, label image {:.3} GiB, load {:.2}",
        gib(image as u64),
        loadavg()
    );
    eprintln!(
        "\n{:>10} {:>18} {:>8} | {:>10} | {:>9} | {:>12} | {:>14}",
        "lattice", "block", "blocks", "F GiB", "F / image", "F / n^(1/3)", "(1+n) F GiB"
    );

    let mut cube: Vec<(usize, f64)> = Vec::new();
    let mut slab: Vec<(usize, f64)> = Vec::new();
    for (name, counts) in lattices {
        let mut block = [0usize; 3];
        for axis in 0..3 {
            block[axis] = volume[axis].div_ceil(counts[axis].min(volume[axis]));
        }
        let grid = BlockGrid::new(volume, block).expect("a lattice");
        let blocks: usize = grid.blocks_per_axis().iter().product();

        let label =
            LabelComponentsOp::new("label", STREAM, Lifecycle::Persistent).connecting(CONNECTIVITY);
        let plan = component_labelling_phase(grid, Dtype::Bool, &label).expect("a plan");
        let input: Voxels = mask.clone().into();
        let env =
            ArrayEnvironment::for_decomposition(input, &plan, chunk(block)).expect("environment");
        let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
        execute_phases(
            "label",
            &workflow,
            &plan,
            &hints(),
            &env,
            &[],
            &[PhaseWork::Fragments(&label)],
        )
        .expect("a run");
        // What the executor charged for writing them, which is what one
        // transmission of the whole fragment set costs.
        let bytes = env.counters().sidecar_bytes_written.load(Ordering::Relaxed);
        drop(env);

        let f = gib(bytes);
        eprintln!(
            "{name:>10} {:>18} {blocks:>8} | {f:>10.4} | {:>8.1}% | {:>12.4} | {:>14.2}",
            format!("{block:?}"),
            100.0 * bytes as f64 / image as f64,
            f / (blocks as f64).cbrt(),
            (1 + blocks) as f64 * f,
        );
        if name.starts_with("cube") {
            cube.push((blocks, f));
        } else {
            slab.push((blocks, f));
        }
    }

    // **The growth is real**, or the `(1 + blocks) x F(blocks)` decomposition
    // would have been `(1 + blocks) x` a constant and the whole argument about
    // fine cutting would be a factor of one out.
    let (first_blocks, first) = cube[0];
    let (last_blocks, last) = *cube.last().expect("a sweep");
    assert!(
        last > first * 2.0,
        "the fragment set went {first:.4} GiB at {first_blocks} blocks to {last:.4} at \
         {last_blocks}; if it does not grow then `F` is a constant and this file's \
         decomposition is wrong"
    );

    // **And the shape of the cut matters at least as much as the count.** This
    // is the assumption every other figure in this file holds fixed, and it does
    // not hold: at 64 blocks the two families differ by 3x. The *direction* is
    // recorded rather than asserted, because the prediction that motivated this
    // sweep was backwards — see the header.
    //
    // What is asserted is the mechanism, which is the part that generalises: a
    // fragment is six faces, the total is dominated by the **largest** face, and
    // the largest face is the one perpendicular to the **shortest** axis. So
    // cutting the short axis is what costs, and a lattice that cuts it four ways
    // beats a lattice that makes four times as many blocks without touching it.
    let cube_64 = cube
        .iter()
        .find(|(blocks, _)| *blocks == 64)
        .map(|(_, f)| *f)
        .expect("a 64-block cube row");
    let slab_64 = slab
        .iter()
        .find(|(blocks, _)| *blocks == 64)
        .map(|(_, f)| *f)
        .expect("a 64-block slab row");
    eprintln!(
        "\n  at 64 blocks, block count controlled: cube {cube_64:.4} GiB, slab {slab_64:.4} GiB \
         — cube/slab {:.2}x",
        cube_64 / slab_64
    );
    assert!(
        cube_64 > slab_64 * 1.5,
        "at 64 blocks a cube cut produced {cube_64:.4} GiB of fragments and a slab cut \
         {slab_64:.4}. The cube cut is the one that divides the volume's shortest axis, so it \
         should carry far more face; if these are close then the fragment cost is not dominated \
         by the largest face and this file's account of why `F` grows is wrong"
    );
    let slab_most = slab
        .iter()
        .map(|(_, f)| *f)
        .fold(f64::MIN, f64::total_cmp_max);
    assert!(
        slab_most < cube.last().expect("a sweep").1,
        "the finest slab cut ({slab_most:.4} GiB) carries more face than the finest cube cut \
         ({:.4}), which would invert the mechanism above",
        cube.last().expect("a sweep").1
    );
}
