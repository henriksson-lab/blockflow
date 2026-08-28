//! **Every object is claimed by exactly one block.**
//!
//! This is the property that replaces `cellpose-stitch`. The old pipeline
//! matched cells across tile overlaps by IoU and deduplicated the matches; this
//! one never creates the duplicate, because ownership is decided by a rule that
//! partitions the volume rather than by a comparison between neighbours.
//!
//! The rule is: a block keeps an object if and only if the object's centroid
//! lies in that block's core, half-open on every axis. Cores tile the volume
//! exactly, so "exactly one" follows — and the tests here are what say it
//! follows in the code and not only on paper.
//!
//! Everything runs against `stub::ThresholdBackend`, so a failure here is a
//! statement about this crate. There is no GPU, no model file and no feature
//! flag involved.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::model_segment::stub::ThresholdBackend;
use blockflow::model_segment::{identify, InstanceSegment};
use blockflow::op::Chain;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::{Row, Table};
use blockflow::{
    fragment_phase, BlockGrid, Decomposition, Dtype, Lifecycle, PhaseWork, Region, Voxels,
};
use ndarray::Array3;

const STREAM: &str = "objects";

// ------------------------------------------------------------- fixtures --

/// A volume with `objects` solid boxes in it, none touching another.
///
/// Boxes rather than spheres because the expected count, the expected voxel
/// total and the expected centroid are then exact integers a test can state
/// rather than approximate — which matters, since what is being checked is that
/// the *decomposition* changes none of the three.
struct Scene {
    volume: [usize; 3],
    /// `(lower corner, extent)`, in volume coordinates.
    boxes: Vec<([usize; 3], [usize; 3])>,
}

impl Scene {
    fn render(&self) -> Voxels {
        let mut array = Array3::<f64>::zeros(self.volume);
        for (corner, extent) in &self.boxes {
            for i in 0..extent[0] {
                for j in 0..extent[1] {
                    for k in 0..extent[2] {
                        array[[corner[0] + i, corner[1] + j, corner[2] + k]] = 1.0;
                    }
                }
            }
        }
        array.into()
    }

    /// A second image whose value is constant per box, so that a measurement
    /// can be checked against a number rather than against another computation.
    ///
    /// Box `n` holds `n + 1`; the background holds zero.
    fn marker(&self) -> Voxels {
        let mut array = Array3::<f64>::zeros(self.volume);
        for (index, (corner, extent)) in self.boxes.iter().enumerate() {
            for i in 0..extent[0] {
                for j in 0..extent[1] {
                    for k in 0..extent[2] {
                        array[[corner[0] + i, corner[1] + j, corner[2] + k]] = (index + 1) as f64;
                    }
                }
            }
        }
        array.into()
    }

    /// Where the centroid of each box lands, by the same rounding rule the op
    /// uses: accumulate exactly, divide once, round half up.
    fn centroids(&self) -> Vec<[usize; 3]> {
        self.boxes
            .iter()
            .map(|(corner, extent)| {
                let mut at = [0usize; 3];
                for axis in 0..3 {
                    let count = extent[axis] as u64;
                    let sum: u64 = (0..extent[axis])
                        .map(|offset| (corner[axis] + offset) as u64)
                        .sum();
                    // The op divides the whole object's per-axis sum by its
                    // voxel count; for a box those factor per axis.
                    at[axis] = ((2 * sum + count) / (2 * count)) as usize;
                }
                at
            })
            .collect()
    }

    /// Each box's lowest voxel in raster order, which is its corner — and
    /// which is what the op derives an id from.
    fn corners(&self) -> Vec<[usize; 3]> {
        self.boxes.iter().map(|(corner, _)| *corner).collect()
    }

    fn voxels_per_box(&self) -> Vec<u64> {
        self.boxes
            .iter()
            .map(|(_, extent)| (extent[0] * extent[1] * extent[2]) as u64)
            .collect()
    }
}

/// Boxes at positions chosen to exercise the seams: one well inside a block,
/// one straddling a block boundary, one at the volume's edge, one small.
fn scene() -> Scene {
    Scene {
        volume: [1, 40, 48],
        boxes: vec![
            ([0, 4, 4], [1, 6, 6]),   // inside one block at every edge tried
            ([0, 14, 14], [1, 8, 8]), // straddles the 16-block seam on both axes
            ([0, 30, 2], [1, 5, 5]),  // near the volume's lower edge in x
            ([0, 33, 41], [1, 6, 6]), // near the far corner
            ([0, 20, 30], [1, 2, 2]), // small
        ],
    }
}

// ----------------------------------------------------------------- run --

/// Run the segmentation phase at `block` and hand back every row emitted, with
/// the block that emitted it.
fn run(scene: &Scene, block: [usize; 3], halo: usize) -> Vec<([usize; 3], Vec<OwnedRow>)> {
    let backend = Arc::new(ThresholdBackend::new(0.5));
    let op = InstanceSegment::new(
        "segment",
        backend,
        [0, halo, halo],
        STREAM,
        Lifecycle::Persistent,
        vec![(blockflow::ImageId::supplied(0), Dtype::F64)],
    );

    let grid = BlockGrid::new(scene.volume, block).expect("a lattice");
    let phase = fragment_phase(&op, grid.clone()).expect("a phase");
    let plan = Decomposition {
        volume: scene.volume,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    };
    plan.check().expect("a legal plan");

    let env = ArrayEnvironment::with_inputs(scene.render(), vec![scene.marker()], &plan, [8, 8, 8])
        .expect("an environment");

    let workflow = Workflow::new(Chain::sequence(Vec::new()), scene.volume, Dtype::F64);
    execute_phases(
        "segment",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&op)],
    )
    .expect("a run");

    let schema = op.schema().expect("a schema");
    every_block(&grid)
        .into_iter()
        .map(|index| {
            let bytes = env
                .read_sidecar(STREAM, 0, index)
                .expect("the store answers")
                .unwrap_or_else(|| panic!("block {index:?} wrote no blob"));
            let mut table = Table::new(scene.volume, schema.clone()).expect("a table");
            table.write(index, &bytes).expect("a row blob");
            table.seal().expect("a seal");
            let rows = table
                .query(&Region::whole(&scene.volume))
                .expect("a query")
                .iter()
                .map(OwnedRow::of)
                .collect();
            (index, rows)
        })
        .collect()
}

/// A row, off the borrowed `Row` so it outlives the table it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
struct OwnedRow {
    at: [usize; 3],
    values: Vec<u64>,
}

impl OwnedRow {
    fn of(row: &Row<'_>) -> Self {
        Self {
            at: row.at(),
            // `len` is the columns; `width` counts the position words too.
            values: (0..row.schema().len())
                .map(|column| row.u64(column).expect("a u64 column"))
                .collect(),
        }
    }

    fn id(&self) -> u64 {
        self.values[0]
    }

    fn count(&self) -> u64 {
        self.values[1]
    }

    fn measured(&self, image: usize) -> (u64, u64, u64) {
        let base = blockflow::model_segment::measured_column(image);
        (
            self.values[base],
            self.values[base + 1],
            self.values[base + 2],
        )
    }
}

fn every_block(grid: &BlockGrid) -> Vec<[usize; 3]> {
    grid.cores().into_iter().map(|core| core.index).collect()
}

fn all_rows(emitted: &[([usize; 3], Vec<OwnedRow>)]) -> Vec<OwnedRow> {
    let mut rows: Vec<OwnedRow> = emitted
        .iter()
        .flat_map(|(_, rows)| rows.iter().cloned())
        .collect();
    rows.sort_by_key(|row| row.id());
    rows
}

// --------------------------------------------------------------- tests --

/// **The claim.** Every object is emitted exactly once, by exactly one block,
/// at every block size — including sizes whose seams cut objects in half.
#[test]
fn every_object_is_claimed_by_exactly_one_block() {
    let scene = scene();
    let expected = scene.centroids();

    for block in [[1, 16, 16], [1, 10, 10], [1, 40, 48], [1, 8, 12]] {
        let emitted = run(&scene, block, 8);

        // One row per object, and no row twice.
        let rows = all_rows(&emitted);
        assert_eq!(
            rows.len(),
            scene.boxes.len(),
            "block {block:?}: {} rows for {} objects",
            rows.len(),
            scene.boxes.len()
        );

        let ids: BTreeSet<u64> = rows.iter().map(OwnedRow::id).collect();
        assert_eq!(
            ids.len(),
            rows.len(),
            "block {block:?}: an id was emitted twice"
        );

        // And each is the object it should be, identified by its lowest voxel
        // and positioned at its centroid.
        let mut seen: BTreeMap<u64, [usize; 3]> = BTreeMap::new();
        for row in &rows {
            seen.insert(row.id(), row.at);
        }
        for (corner, centroid) in scene.corners().iter().zip(expected.iter()) {
            let id = identify(*corner, scene.volume);
            assert_eq!(
                seen.get(&id),
                Some(centroid),
                "block {block:?}: no row for the object at {corner:?}, centred at {centroid:?}"
            );
        }
    }
}

/// The owning block is the one whose **core** holds the centroid — not the one
/// whose read extent does, which is several of them.
#[test]
fn the_owning_block_is_the_one_whose_core_holds_the_centroid() {
    let scene = scene();
    let block = [1, 16, 16];
    let grid = BlockGrid::new(scene.volume, block).expect("a lattice");
    let emitted = run(&scene, block, 8);

    let cores: std::collections::BTreeMap<[usize; 3], Region> = grid
        .cores()
        .into_iter()
        .map(|core| (core.index, core.core))
        .collect();

    for (index, rows) in &emitted {
        let core = &cores[index];
        for row in rows {
            for axis in 0..3 {
                assert!(
                    row.at[axis] >= core.start[axis]
                        && row.at[axis] < core.start[axis] + core.shape[axis],
                    "block {index:?} emitted an object at {:?}, outside its core {:?}..{:?}",
                    row.at,
                    core.start,
                    core.shape
                );
            }
        }
    }
}

/// A block that owns nothing still writes a blob.
///
/// The difference between "owned nothing" and "never ran" is exactly what the
/// `Coverage::EveryBlock` declaration is for, and a phase that writes no image
/// is constrained by nothing else — the tiling check passes vacuously over it.
#[test]
fn a_block_that_owns_nothing_still_writes_a_blob() {
    let scene = scene();
    let emitted = run(&scene, [1, 8, 8], 8);

    let empty = emitted.iter().filter(|(_, rows)| rows.is_empty()).count();
    assert!(
        empty > 0,
        "this lattice was chosen to have blocks owning nothing; none did"
    );
    // `run` panics if any block wrote no blob at all, so reaching here with
    // empty blocks present is the assertion.
    assert_eq!(
        emitted.len(),
        BlockGrid::new(scene.volume, [1, 8, 8]).unwrap().n_blocks(),
        "every block of the lattice answered"
    );
}

/// **The halo is what makes a measurement a measurement of the whole object.**
///
/// With a halo larger than the objects, the owning block holds each one whole
/// and measures it whole. With no halo, an object crossing a seam is *seen* by
/// each block only as the piece inside that block's core, so each piece gets a
/// centroid of its own, is owned by that block, and is emitted as a separate
/// object — the objects fragment rather than being under-measured, and the
/// fragments outnumber the truth.
///
/// Both halves are asserted, because the second is the failure mode and a
/// warning in a doc comment is not a measurement.
///
/// The invariant that holds either way is the sharper one: **the voxel total is
/// exact under every halo**, because cores tile the volume and a piece's
/// centroid is always inside the piece. Nothing is counted twice and nothing is
/// dropped; what a short halo costs is that the pieces are not joined.
#[test]
fn the_halo_decides_whether_an_object_is_measured_whole() {
    let scene = scene();
    let expected: BTreeMap<u64, u64> = scene
        .corners()
        .iter()
        .zip(scene.voxels_per_box())
        .map(|(corner, voxels)| (identify(*corner, scene.volume), voxels))
        .collect();
    let total: u64 = scene.voxels_per_box().iter().sum();

    // A halo comfortably larger than the largest box, which is 8 across.
    let whole = all_rows(&run(&scene, [1, 16, 16], 8));
    assert_eq!(whole.len(), scene.boxes.len(), "one row per object");
    for row in &whole {
        assert_eq!(
            row.count(),
            expected[&row.id()],
            "object {} was measured over {} voxels and has {}",
            row.id(),
            row.count(),
            expected[&row.id()]
        );
    }
    assert_eq!(whole.iter().map(OwnedRow::count).sum::<u64>(), total);

    // No halo. The boxes crossing a seam come back in pieces.
    let pieces = all_rows(&run(&scene, [1, 16, 16], 0));
    assert!(
        pieces.len() > scene.boxes.len(),
        "a zero halo split no object, so this scene does not exercise the seam: \
         {} rows for {} objects",
        pieces.len(),
        scene.boxes.len()
    );
    // And every voxel is still accounted for exactly once.
    assert_eq!(
        pieces.iter().map(OwnedRow::count).sum::<u64>(),
        total,
        "cores tile the volume, so the pieces add up to the whole whatever the halo"
    );
}

/// A measured image is reduced over the object's own voxels, and the reduction
/// is the one the columns say it is.
///
/// The marker image holds a constant per object, so `sum` is that constant times
/// the voxel count and `min` and `max` are both the constant — three columns
/// checkable against a number rather than against a second implementation of the
/// same loop.
#[test]
fn a_measured_image_is_reduced_over_the_objects_own_voxels() {
    let scene = scene();
    let by_id: BTreeMap<u64, (u64, u64)> = scene
        .corners()
        .iter()
        .zip(scene.voxels_per_box())
        .enumerate()
        .map(|(index, (corner, voxels))| {
            (
                identify(*corner, scene.volume),
                ((index + 1) as u64, voxels),
            )
        })
        .collect();

    for row in all_rows(&run(&scene, [1, 16, 16], 8)) {
        let (value, voxels) = by_id[&row.id()];

        // Image 0 of the measured set is the segmented image itself, which is
        // 1.0 over every object voxel.
        let (sum, low, high) = row.measured(0);
        assert_eq!((sum, low, high), (voxels, 1, 1), "the segmented image");

        // Image 1 is the marker, constant at `value` over this object.
        let (sum, low, high) = row.measured(1);
        assert_eq!(
            (sum, low, high),
            (value * voxels, value, value),
            "the marker over object {}",
            row.id()
        );
    }
}
