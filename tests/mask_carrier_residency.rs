// SPDX-License-Identifier: MIT
//
// **What the narrow carrier is worth in bytes the process actually held.**
//
// `tests/mask_carrier.rs` is about voxels: that a `Bool` sink and an `f64` one
// are the same answer. This is about the reason to prefer one — and it is a
// separate binary on purpose, because it measures the **whole process** through
// a counting global allocator and a second test running beside it would be
// measured too.
//
// Why an allocator and not a plan
// -------------------------------
// A `Decomposition` can price its images and nothing else. Everything an op
// allocates inside `BlockOp::apply`, every buffer a fan-in makes for a branch,
// every intermediate a `Chain::Sequence` threads between its children — none of
// it belongs to an image, so none of it appears in any figure a plan can
// produce. In the workload this change is for, that unpriced part is the
// *majority* of the residency: a sibling crate measures a stable 59.6 to 65.4
// bytes a voxel of it against images that price at about 14, and the one run
// ever attempted at full size was killed on its way up.
//
// So a saving that shows up only in a declared number is not a saving. This
// measures `PEAK - LIVE` across the run, in bytes, with the input array
// allocated *before* the baseline is taken so that what is reported is what the
// plan and the run added.
//
// **The measurement is differential and the control is built in**: the two runs
// are the same program with one thing changed — the carrier the sink is held in
// — over the same fixture, at the same block size, producing the same voxels
// (which `tests/mask_carrier.rs` is what proves). Any per-process noise is in
// both.
//
// What it said on this machine
// ----------------------------
// Over `128 x 128 x 96`, which is 1.57 M voxels:
//
// ```text
//            block |    sink |    all images | measured peak | difference
//   [128, 128, 96] | float64 |      60.0 MiB |     108.0 MiB |   +48.0 MiB
//   [128, 128, 96] |    bool |      18.0 MiB |      55.5 MiB |   +37.5 MiB
//     [32, 32, 32] | float64 |      60.0 MiB |      38.3 MiB |   -21.7 MiB
//     [32, 32, 32] |    bool |      18.0 MiB |      16.5 MiB |    -1.5 MiB
// ```
//
// Two things in that table are worth more than the magnitudes.
//
// **At one block the measured saving is 52.5 MiB against a priced 42.0** — the
// narrowing reaches further than the images. The extra is the fan-in's own
// branch buffers: the branch carrying the sink forward allocates at the sink's
// width, three times, and the executor's block buffers follow the same widths.
// That is the part no `Decomposition` can see, and it is the part the workload
// this change is for is mostly made of.
//
// **At `32^3` the peak is *below* the image total, in both carriers**, which is
// not a contradiction: `ImageStore::pending` allocates nothing until a phase
// writes, and an internal image is freed after its last reader, so a five-image
// plan never holds five images. The saving there is 21.8 MiB, about half the
// priced 42.0, because only about half the sink images are alive at once.
//
// The assertions are floors well below both figures rather than the figures
// themselves, because a peak over rayon's workers is not repeatable to the byte
// and a test that demanded it would be measuring the scheduler.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::Array3;

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::{from_set, CarryOp, Logic, LogicCombine, VoxelwiseMapOp, VoxelwiseMaskOp};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::Dtype;

// ------------------------------------------------------- the measurement --

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// The system allocator, counting. Nothing here allocates, so it cannot recurse.
struct Counting;

fn took(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            took(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            took(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = System.realloc(ptr, layout, new_size);
        if !new.is_null() {
            if new_size >= layout.size() {
                took(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The bytes `body` added at its worst moment, over what was already held.
fn peak_of<R>(body: impl FnOnce() -> R) -> (R, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let result = body();
    let peak = PEAK.load(Ordering::Relaxed);
    (result, peak.saturating_sub(base))
}

// ------------------------------------------------------------- the plans --

/// `tests/mask_carrier.rs`'s chain, restated: a seed and three arms, each arm's
/// verdict OR-ed into a sink the next arm carries forward. Arm 1 answers in
/// `Bool` and arms 2 and 3 in `f64`, so whichever carrier the sink is in, some
/// join is mixed.
fn sink_plan(volume: [usize; 3], carrier: Dtype, block: [usize; 3]) -> Assembly {
    let grid = BlockGrid::new(volume, block).expect("a lattice");
    let mut plan = PlanBuilder::new(volume, Dtype::F64, grid);
    let seed_holds = |value: f64| (0.25..0.35).contains(&value);
    let seed = match carrier {
        Dtype::Bool => Chain::op(VoxelwiseMaskOp::new("seed", seed_holds)),
        _ => Chain::op(VoxelwiseMapOp::new("seed", move |value| {
            from_set(seed_holds(value))
        })),
    };
    plan.pixels(seed).expect("the seed");
    let arms: Vec<Chain> = vec![
        Chain::op(VoxelwiseMaskOp::at_or_above("arm 1", 0.9)),
        Chain::op(VoxelwiseMapOp::new("arm 2", |value| {
            from_set((0.45..0.55).contains(&value))
        })),
        Chain::op(VoxelwiseMapOp::new("arm 3", |value| {
            from_set((0.65..0.75).contains(&value))
        })),
    ];
    for arm in arms {
        let join = Chain::parallel(
            vec![
                Chain::op(CarryOp::new("sink so far")),
                Chain::sequence(vec![Chain::source(0usize, Dtype::F64), arm]),
            ],
            Box::new(
                LogicCombine::new("or", Logic::Or)
                    .producing(carrier)
                    .expect("a mask carrier"),
            ),
        )
        .expect("a fan-in");
        plan.pixels(join).expect("an arm");
    }
    plan.finish().expect("a plan whose types check")
}

fn run(assembly: &Assembly, input: &Array3<f64>) {
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [64, 64, 64],
    )
    .expect("an environment typed by the plan");
    let listeners: Vec<Arc<dyn blockflow::listener::EventListener>> = Vec::new();
    execute_phases(
        "residency",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &listeners,
        &assembly.work(),
    )
    .expect("the run");
    // Touch the answer so nothing above is dead code.
    let out = env.output();
    assert_eq!(out.shape(), assembly.decomposition.volume);
}

fn priced_images(assembly: &Assembly) -> u64 {
    let plan = &assembly.decomposition;
    (0..plan.n_images())
        .map(|image| {
            plan.volume_at(image).iter().product::<usize>() as u64
                * plan.dtype_at(image).size_of() as u64
        })
        .sum()
}

/// **The narrow sink holds fewer bytes, and more of them than the plan knows
/// about.**
///
/// Two block sizes, because they answer different halves of the question. At one
/// block the images *are* the residency and the saving is the four sink images'
/// seven-eighths. At a small block the images are unchanged but every fan-in
/// allocates a buffer per branch at the branch's own declared width — see
/// `Chain::apply_placed`'s `Parallel` arm — so the branch carrying the sink
/// forward narrows too, and that part is invisible to every figure a
/// decomposition can produce.
#[test]
fn the_narrow_sink_holds_fewer_bytes_than_the_wide_one() {
    const VOLUME: [usize; 3] = [128, 128, 96];
    let voxels: u64 = VOLUME.iter().product::<usize>() as u64;
    let input = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i * 7 + j * 5 + k * 3) % 11) as f64 / 10.0
    });

    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "\n{:>16} | {:>7} | {:>13} | {:>13} | {:>10}",
        "block", "sink", "all images", "measured peak", "difference"
    );

    for block in [VOLUME, [32, 32, 32]] {
        let mut measured = Vec::new();
        for carrier in [Dtype::F64, Dtype::Bool] {
            // The plan is built inside the measurement, because a plan is part
            // of what a run holds — and it is the same size either way, so
            // including it cannot flatter the narrow case.
            let (priced, peak) = peak_of(|| {
                let assembly = sink_plan(VOLUME, carrier, block);
                let priced = priced_images(&assembly);
                run(&assembly, &input);
                priced
            });
            eprintln!(
                "{block:>16?} | {:>7} | {:>9.1} MiB | {:>9.1} MiB | {:>+6.1} MiB",
                carrier.numpy_name(),
                mib(priced),
                mib(peak as u64),
                peak as f64 / (1024.0 * 1024.0) - mib(priced),
            );
            measured.push((priced, peak as u64));
        }
        let (wide_priced, wide_peak) = measured[0];
        let (narrow_priced, narrow_peak) = measured[1];

        // The plan's own claim: four images of the sink, seven bytes a voxel
        // narrower each.
        assert_eq!(
            wide_priced - narrow_priced,
            voxels * 7 * 4,
            "at {block:?} the priced saving is not the four sink images' own"
        );

        // The claim that matters: it is a saving in bytes the process held.
        assert!(
            narrow_peak < wide_peak,
            "at {block:?} the narrow sink peaked at {narrow_peak} against the wide one's \
             {wide_peak}"
        );

        // A floor rather than the figure, because a peak over rayon's workers is
        // not repeatable to the byte and because how much of the saving *lands*
        // depends on how much of the plan is alive at once — at one block the
        // measured saving is larger than the priced one and at a small block it
        // is about half, for the reason the two table rows show. A quarter is
        // under both by a wide margin and still fails outright if the narrowing
        // stops reaching the run, which is the failure worth catching: a widen
        // inserted anywhere in the fan-in would put the eight bytes back.
        let floor = (wide_priced - narrow_priced) / 4;
        assert!(
            wide_peak - narrow_peak >= floor,
            "at {block:?} the measured saving is {} against a floor of {floor}; the plan says \
             {} was taken off the images, so something is putting the width back at run time",
            wide_peak - narrow_peak,
            wide_priced - narrow_priced
        );
    }
}
