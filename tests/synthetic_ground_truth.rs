// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// What a generated fixture has to prove before anything may be concluded from
// it.
//
// The crate's whole argument is that a block-decomposed run and a whole-volume
// run produce the same bits. A fixture used to test that must not itself have
// the property under test as a bug: if `render_region` and `render` could
// disagree, every failure would be ambiguous and every pass would be worthless.
// So the assertions here are on the generator, not on any pipeline:
//
// * the same seed gives the same bits, however many threads are running;
// * a region is bit-for-bit the cut the whole volume has there, including
//   regions that slice objects in half;
// * a tiling of blocks reassembles into exactly the whole volume;
// * the object table describes the label volume that actually comes out.
//
// Bits, not values. Floating point is compared through `to_bits`, because
// `a == b` is not the claim being made — "byte-identical" is, and two different
// bit patterns that compare equal would pass the weaker test.

use std::time::Instant;

use ndarray::{Array3, Axis};

use blockflow::agreement::compare_labels;
use blockflow::region::Region;
use blockflow::synthetic::{Rendered, Scene, SceneSpec};

fn scene(shape: [usize; 3], seed: u64, objects: usize) -> Scene {
    Scene::new(SceneSpec::new(shape, seed).with_objects(objects)).expect("valid spec")
}

/// Every intensity bit and every label, or the first place they differ.
fn assert_same_bits(left: &Rendered, right: &Rendered, what: &str) {
    assert_eq!(left.labels.shape(), right.labels.shape(), "{what}: shape");
    for ((index, &here), &there) in left.labels.indexed_iter().zip(right.labels.iter()) {
        assert_eq!(here, there, "{what}: label at {index:?}");
    }
    for ((index, &here), &there) in left.intensity.indexed_iter().zip(right.intensity.iter()) {
        assert_eq!(
            here.to_bits(),
            there.to_bits(),
            "{what}: intensity at {index:?} ({here} vs {there})"
        );
    }
}

#[test]
fn the_same_seed_and_shape_give_byte_identical_volumes() {
    let shape = [20, 36, 44];
    for seed in [0u64, 1, 12345, u64::MAX] {
        let first = scene(shape, seed, 40).render();
        let second = scene(shape, seed, 40).render();
        assert_same_bits(&first, &second, &format!("seed {seed}"));
    }
}

#[test]
fn different_seeds_give_different_volumes() {
    let shape = [20, 36, 44];
    let first = scene(shape, 1, 40).render();
    let second = scene(shape, 2, 40).render();
    assert_ne!(first.labels, second.labels);
}

/// The property that makes the fixture usable from a parallel executor at all.
/// Rendering is spread over planes by rayon, so a plane that depended on
/// anything but its own global coordinates would show up here as a thread-count
/// dependency.
#[test]
fn the_thread_count_does_not_change_a_single_bit() {
    // Above the size at which rendering goes wide — a volume small enough to be
    // rendered on the calling thread would prove nothing here.
    let shape = [32, 64, 64];
    let reference = scene(shape, 99, 60).render();
    for threads in [1usize, 2, 3, 7, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        let got = pool.install(|| scene(shape, 99, 60).render());
        assert_same_bits(&reference, &got, &format!("{threads} threads"));
    }
}

/// The single most important test in this file. A region generated directly must
/// equal the same region cut out of a full generation — otherwise "the block
/// disagrees with the whole" is the fixture's fault and nothing downstream can
/// be diagnosed.
#[test]
fn a_region_is_byte_identical_to_the_same_cut_of_the_whole_volume() {
    // The whole volume is large enough to be rendered in parallel and the
    // regions are small enough not to be, so this also crosses the two paths.
    let shape = [28, 48, 56];
    let scene = scene(shape, 4242, 70);
    let whole = scene.render();

    let mut regions = vec![
        // The whole thing, one voxel, and the far corner.
        Region::new(&[0, 0, 0], &shape),
        Region::new(&[0, 0, 0], &[1, 1, 1]),
        Region::new(&[27, 47, 55], &[1, 1, 1]),
        // Slabs, columns, and a box that is aligned to nothing.
        Region::new(&[0, 0, 0], &[7, 48, 56]),
        Region::new(&[14, 0, 0], &[14, 48, 56]),
        Region::new(&[0, 13, 0], &[28, 9, 56]),
        Region::new(&[3, 5, 7], &[11, 13, 17]),
        Region::new(&[13, 21, 29], &[15, 19, 23]),
    ];

    // And regions whose corner sits inside an object, so the object is cut in
    // half along all three axes at once. These are the ones that would catch a
    // renderer that resolved anything relative to the region's origin.
    let table = scene.object_table();
    let mut cut = 0;
    for record in table.iter().filter(|record| record.voxels > 8) {
        let start = [
            record.centroid[0] as usize,
            record.centroid[1] as usize,
            record.centroid[2] as usize,
        ];
        if start
            .iter()
            .zip(shape.iter())
            .any(|(&at, &end)| at + 2 > end)
        {
            continue;
        }
        regions.push(Region::new(&start, &[2, 3, 4]));
        cut += 1;
        if cut == 12 {
            break;
        }
    }
    assert!(cut >= 8, "only {cut} regions straddle an object");

    for region in &regions {
        let piece = scene.render_region(region).expect("region is inside");
        let cut = cut_out(&whole, region);
        assert_same_bits(&piece, &cut, &format!("region {region:?}"));
    }
}

/// The decomposition case: tile the volume with blocks of an awkward size,
/// generate each one on its own, and reassemble. Nothing here overlaps and
/// nothing is written twice, so the result must be the whole volume exactly.
#[test]
fn a_tiling_of_generated_blocks_reassembles_the_whole_volume() {
    let shape = [26, 52, 61];
    let scene = scene(shape, 77, 90);
    let whole = scene.render();

    for block in [[7usize, 9, 11], [26, 52, 61], [1, 52, 61], [5, 5, 5]] {
        let mut intensity = Array3::<f64>::zeros(shape);
        let mut labels = Array3::<u32>::zeros(shape);
        let mut blocks = 0;
        let mut z = 0;
        while z < shape[0] {
            let mut y = 0;
            while y < shape[1] {
                let mut x = 0;
                while x < shape[2] {
                    let extent = [
                        block[0].min(shape[0] - z),
                        block[1].min(shape[1] - y),
                        block[2].min(shape[2] - x),
                    ];
                    let region = Region::new(&[z, y, x], &extent);
                    let piece = scene.render_region(&region).expect("block is inside");
                    for (index, &value) in piece.intensity.indexed_iter() {
                        intensity[[z + index.0, y + index.1, x + index.2]] = value;
                    }
                    for (index, &value) in piece.labels.indexed_iter() {
                        labels[[z + index.0, y + index.1, x + index.2]] = value;
                    }
                    blocks += 1;
                    x += block[2];
                }
                y += block[1];
            }
            z += block[0];
        }
        let assembled = Rendered {
            region: Region::whole(&shape),
            intensity,
            labels,
        };
        assert_same_bits(&assembled, &whole, &format!("{blocks} blocks of {block:?}"));
    }
}

/// The label volume and the table are two statements about the same thing, and
/// the table is the one a consumer will trust without checking.
#[test]
fn the_object_table_agrees_with_the_label_volume() {
    let shape = [24, 30, 36];
    let scene = Scene::new(
        SceneSpec::new(shape, 555)
            .with_objects(80)
            .with_touching(0.6, 0.25),
    )
    .expect("valid spec");
    let labels = scene.render().labels;
    let table = scene.object_table();

    assert_eq!(table.len(), scene.object_count());
    let mut seen = std::collections::BTreeMap::<u32, u64>::new();
    for (_, &label) in labels.indexed_iter() {
        if label != 0 {
            *seen.entry(label).or_default() += 1;
        }
    }

    let mut total = 0u64;
    for record in &table {
        assert_eq!(
            record.voxels,
            seen.get(&record.id).copied().unwrap_or(0),
            "object {} voxel count",
            record.id
        );
        total += record.voxels;
        if record.voxels == 0 {
            continue;
        }
        // The bounding box is tight: every labelled voxel is inside it, and
        // every face of it carries at least one.
        let end = record.bounds.end();
        let mut faces = [false; 6];
        for (index, &label) in labels.indexed_iter() {
            if label != record.id {
                continue;
            }
            let at = [index.0, index.1, index.2];
            for axis in 0..3 {
                assert!(
                    at[axis] >= record.bounds.start[axis] && at[axis] < end[axis],
                    "object {} voxel {at:?} outside its box",
                    record.id
                );
                faces[axis * 2] |= at[axis] == record.bounds.start[axis];
                faces[axis * 2 + 1] |= at[axis] == end[axis] - 1;
            }
        }
        assert!(faces.iter().all(|&touched| touched), "box is not tight");
    }
    assert_eq!(
        total,
        labels.iter().filter(|&&label| label != 0).count() as u64,
        "the table accounts for every labelled voxel"
    );
    // With this much touching, some object must have lost voxels to a
    // lower-numbered neighbour, or the overlap knob is doing nothing.
    let overlapped = table
        .iter()
        .zip(scene.objects())
        .filter(|(record, object)| {
            let ideal = object.radii.iter().product::<f64>() * std::f64::consts::PI * 4.0 / 3.0;
            (record.voxels as f64) < ideal * 0.95
        })
        .count();
    assert!(overlapped > 0, "no object is clipped or overlapped at all");
}

/// The laziness claim, made falsifiable. This volume is seventeen billion voxels
/// — a hundred and thirty gigabytes as `f64` — so any implementation that
/// materialised it, or that even walked it, could not finish. What is timed is
/// one small block out of it.
#[test]
fn a_region_of_a_volume_too_large_to_materialise_costs_only_the_region() {
    let shape = [1024usize, 4096, 4096];
    let started = Instant::now();
    let scene = Scene::new(
        SceneSpec::new(shape, 8191)
            .with_objects(4000)
            .with_radius(4.0, 9.0),
    )
    .expect("valid spec");
    let placed = started.elapsed();
    assert_eq!(scene.shape(), shape);

    // Somewhere an object is known to be, so the block is not trivially empty.
    let table = scene.object_table();
    let target = table
        .iter()
        .find(|record| record.voxels > 50)
        .expect("some object is not clipped away");
    let start = [
        target.bounds.start[0].saturating_sub(4),
        target.bounds.start[1].saturating_sub(4),
        target.bounds.start[2].saturating_sub(4),
    ];
    let block = [64usize, 64, 64];
    let region = Region::new(&start, &block);

    let started = Instant::now();
    let piece = scene.render_region(&region).expect("block is inside");
    let rendered = started.elapsed();

    assert!(
        piece.labels.iter().any(|&label| label == target.id),
        "the block does not contain the object it was placed around"
    );
    assert_eq!(piece.intensity.len(), block.iter().product::<usize>());
    assert!(
        rendered.as_secs() < 10 && placed.as_secs() < 30,
        "placing took {placed:?} and rendering one 64-cubed block took {rendered:?}; \
         this test only makes sense if neither is proportional to the volume"
    );
}

/// The 2-D case is a volume one voxel deep, and everything above still holds
/// there — including region generation, which is where an axis of extent 1 tends
/// to break arithmetic.
#[test]
fn a_one_voxel_deep_volume_behaves_like_every_other() {
    let shape = [1usize, 96, 128];
    let scene = Scene::new(
        SceneSpec::new(shape, 606)
            .with_objects(50)
            .with_radius(4.0, 9.0),
    )
    .expect("valid spec");
    let whole = scene.render();
    assert!(whole.labels.iter().any(|&label| label != 0));

    for start in [[0usize, 0, 0], [0, 40, 60], [0, 95, 127]] {
        let extent = [
            1usize,
            (shape[1] - start[1]).min(17),
            (shape[2] - start[2]).min(23),
        ];
        let region = Region::new(&start, &extent);
        let piece = scene.render_region(&region).unwrap();
        assert_same_bits(&piece, &cut_out(&whole, &region), &format!("2-D {start:?}"));
    }
}

/// The comparison helper, against the truth it was written for: the labels
/// compared with themselves are exact, and a labelling that fuses touching
/// objects is reported as merges rather than as a good score.
#[test]
fn the_comparison_helper_recognises_the_truth_and_a_fusion_of_it() {
    let shape = [20, 40, 40];
    let scene = Scene::new(
        SceneSpec::new(shape, 313)
            .with_objects(60)
            .with_touching(1.0, 0.2),
    )
    .expect("valid spec");
    let truth = scene.render().labels;

    let exact = compare_labels(truth.view(), truth.view(), 0.5).unwrap();
    assert!(exact.is_exact(), "{}", exact.summary());
    assert_eq!(exact.matched.len(), exact.truth_objects);

    // A labelling that cannot tell touching objects apart: everything non-zero
    // becomes one object. Every truth object is then part of one merge.
    let fused = truth.map(|&label| u32::from(label != 0));
    let merged = compare_labels(truth.view(), fused.view(), 0.5).unwrap();
    assert!(!merged.is_exact());
    assert_eq!(merged.produced_objects, 1);
    assert_eq!(merged.merged.len(), 1);
    assert!(
        merged.merged[0].parts.len() > 1,
        "a fusion of {} objects was not reported as a merge: {}",
        merged.truth_objects,
        merged.summary()
    );
}

/// Slice a region out of a rendered volume.
fn cut_out(whole: &Rendered, region: &Region) -> Rendered {
    let mut intensity = whole.intensity.view();
    let mut labels = whole.labels.view();
    for axis in 0..3 {
        let (start, len) = (region.start[axis], region.shape[axis]);
        intensity = intensity.slice_axis_move(Axis(axis), (start..start + len).into());
        labels = labels.slice_axis_move(Axis(axis), (start..start + len).into());
    }
    Rendered {
        region: region.clone(),
        intensity: intensity.to_owned(),
        labels: labels.to_owned(),
    }
}
