// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The planned (out-of-core) fusion and the slab-by-slab pyramid writer
//! against what they replace: the fusion of every tile held whole, and
//! `ZarrEnvironment::build_multiscale` of the resident volume, byte for byte.

mod common;

use std::path::Path;

use blockflow::region::Region;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{chunk_for_block, PhysicalFrame, PyramidSpec, ZarrEnvironment};
use blockflow::Dtype;
use ndarray::{Array3, Axis};
use serde_json::json;
use tile_stitching::fuse::{FuseParams, Method};
use tile_stitching::io::write_volume_at;
use tile_stitching::planned::fuse_planned;
use tile_stitching::pyramid::{pyramid_factors, PyramidWriter};
use tile_stitching::relayout::StoredLayout;

const TILE: [usize; 3] = [30, 60, 50];

/// A 2 x 3 grid of tiles cut from the common scene, each wobbling by up to
/// five voxels per plane, with a few planes unusable, stored with small chunks
/// (so the reads cross chunk boundaries) at a z offset of 3.
fn wobbly_layout(dir: &Path) -> StoredLayout {
    let z_offset = 3usize;
    let mut tiles = Vec::new();
    let mut lowest = [0i64; 3];
    let mut entries = Vec::new();
    for row in 0..2i64 {
        for column in 0..3i64 {
            let index = (row * 3 + column) as usize;
            let base = [
                (index % 2) as i64,
                row * (TILE[1] as i64 - 12),
                column * (TILE[2] as i64 - 10),
            ];
            let wobble = |k: usize, axis: u64| {
                (common::hash(k as isize, index as isize, axis as isize) % 11) as i64 - 5
            };
            let planes = TILE[0] - z_offset;
            let slices = (0..planes)
                .map(|k| {
                    [
                        base[0] + k as i64,
                        base[1] + wobble(k, 1),
                        base[2] + wobble(k, 2),
                    ]
                })
                .collect::<Vec<_>>();
            for p in &slices {
                for a in 0..3 {
                    lowest[a] = lowest[a].min(p[a]);
                }
            }
            let status = (0..planes)
                .map(|k| {
                    if (k + index) % 9 == 4 {
                        "invalid"
                    } else {
                        "valid"
                    }
                })
                .collect::<Vec<_>>();
            let voxels = common::crop(
                [base[0] as isize, base[1] as isize, base[2] as isize],
                TILE,
                |_| [0, 0],
            );
            let path = dir.join(format!("tile_{row:02}x{column:02}.zarr"));
            ZarrEnvironment::build_multiscale(
                &path,
                &voxels,
                &PyramidSpec::new(vec![[1, 2, 2]])
                    .expect("a pyramid")
                    .with_chunk([7, 16, 13]),
            )
            .expect("a tile image");
            tiles.push(voxels);
            entries.push(json!({
                "path": path.display().to_string(),
                "shape": [planes, TILE[1], TILE[2]],
                "slices": slices,
                "status": status,
            }));
        }
    }
    let json = json!({
        "wobble_axis": 0,
        "voxel_size": [1.0, 1.0, 1.0],
        "origin": lowest,
        "z_offset": z_offset,
        "tiles": entries,
    });
    StoredLayout::from_json(&json, None).expect("a stored layout")
}

/// Every slab size, chunk shape and sweep axis gives what fusing every tile
/// held whole gives, over the whole volume and over boxes, for each method.
#[test]
fn the_planned_fusion_is_the_resident_fusion() {
    let dir = common::scratch("planned_fusion");
    let layout = wobbly_layout(&dir);
    let shape = layout.shape().expect("a shape");
    let placement = layout.fused_placement().expect("a placement");
    let (sources, dtype) = layout.tile_sources(0, 0).expect("tile sources");
    assert_eq!(dtype, Dtype::U16);
    let whole = Region::new(&[0, 0, 0], &shape);
    let boxes = [
        whole.clone(),
        Region::new(&[2, 7, 11], &[shape[0] - 5, shape[1] - 20, shape[2] - 30]),
        Region::new(&[0, shape[1] / 2 - 4, 0], &[shape[0], 9, shape[2]]),
        Region::new(&[5, 0, 37], &[1, shape[1], 1]),
    ];
    for method in [Method::Interpolation, Method::Max, Method::Mean] {
        let params = FuseParams {
            method,
            ..FuseParams::default()
        };
        for region in &boxes {
            let expected = layout
                .fuse_resident(&params, Some(region), 0, 0)
                .expect("the resident fusion");
            assert_eq!(
                layout.fuse(&params, Some(region), 0, 0).expect("planned"),
                expected,
                "{method:?} {region:?}"
            );
            let expected = expected.view::<u16>().expect("u16").to_owned();
            let region_shape = [region.shape[0], region.shape[1], region.shape[2]];
            for axis in 0..3 {
                for rows in [1, 3, 8, 1000] {
                    for chunk in [[4, 5, 6], [1000, 1, 1000], region_shape] {
                        let mut out = Array3::<u16>::zeros(region_shape);
                        let mut at = 0;
                        let stats = fuse_planned::<u16>(
                            &params,
                            &placement,
                            &sources,
                            region,
                            axis,
                            rows,
                            chunk,
                            |slab| {
                                let n = slab.len_of(Axis(axis));
                                out.slice_axis_mut(Axis(axis), ndarray::Slice::from(at..at + n))
                                    .assign(&slab);
                                at += n;
                                Ok(())
                            },
                        )
                        .expect("a planned fusion");
                        assert_eq!(at, region.shape[axis]);
                        assert_eq!(stats.slabs, region.shape[axis].div_ceil(rows));
                        assert_eq!(
                            out, expected,
                            "{method:?} {region:?} axis {axis} rows {rows} chunk {chunk:?}"
                        );
                    }
                }
            }
        }
    }
    // Reads stay inside what the region needs: a thin box reads a fraction of
    // what the whole volume does.
    let params = FuseParams::default();
    let count = |region: &Region| {
        fuse_planned::<u16>(
            &params,
            &placement,
            &sources,
            region,
            1,
            4,
            [4, 4, 4],
            |_| Ok(()),
        )
        .expect("a planned fusion")
        .voxels_read
    };
    let thin = Region::new(&[0, 0, 0], &[shape[0], shape[1], 8]);
    assert!(
        count(&thin) * 4 < count(&whole),
        "{} {}",
        count(&thin),
        count(&whole)
    );
}

/// The planned image on disk is, file for file and byte for byte, the image
/// the resident fusion and `build_multiscale` write.
#[test]
fn the_planned_image_is_the_resident_image_byte_for_byte() {
    let dir = common::scratch("planned_image");
    let layout = wobbly_layout(&dir.join("tiles"));
    let shape = layout.shape().expect("a shape");
    let params = FuseParams::default();
    let frame =
        PhysicalFrame::new([2.0, 0.5, 0.5], Some("micrometer")).with_origin([1.0, -3.0, 4.5]);
    for (name, region) in [
        ("whole", None),
        (
            "box",
            Some(Region::new(
                &[1, 9, 4],
                &[shape[0] - 2, shape[1] - 11, shape[2] - 7],
            )),
        ),
    ] {
        let resident = layout
            .fuse_resident(&params, region.as_ref(), 0, 0)
            .expect("the resident fusion");
        let reference = dir.join(format!("{name}_reference.zarr"));
        let factors = pyramid_factors(resident.shape(), 4);
        ZarrEnvironment::build_multiscale(
            &reference,
            &resident,
            &PyramidSpec::new(factors)
                .expect("a pyramid")
                .with_physical(frame.clone()),
        )
        .expect("the reference image");
        let planned = dir.join(format!("{name}_planned.zarr"));
        layout
            .fuse_to_zarr(&params, region.as_ref(), 0, 0, &planned, 4, frame.clone())
            .expect("the planned image");
        assert_same_files(&reference, &planned);
    }
}

fn files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("a directory") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path.strip_prefix(root).expect("below").to_path_buf());
            }
        }
    }
    out.sort();
    out
}

fn assert_same_files(a: &Path, b: &Path) {
    let (fa, fb) = (files(a), files(b));
    assert_eq!(fa, fb, "the same files");
    assert!(fa.len() >= 3, "{fa:?}");
    for file in &fa {
        assert!(
            std::fs::read(a.join(file)).expect("a") == std::fs::read(b.join(file)).expect("b"),
            "{} differs",
            file.display()
        );
    }
}

fn volume(dtype: Dtype, shape: [usize; 3]) -> Voxels {
    let value = |(z, y, x): (usize, usize, usize)| common::hash(z as isize, y as isize, x as isize);
    match dtype {
        Dtype::U16 => Voxels::U16(Array3::from_shape_fn(shape, |i| (value(i) % 4000) as u16)),
        Dtype::F32 => Voxels::F32(Array3::from_shape_fn(shape, |i| {
            (value(i) % 997) as f32 / 7.0
        })),
        Dtype::Bool => Voxels::Bool(Array3::from_shape_fn(shape, |i| value(i) % 3 == 0)),
        Dtype::U8 => Voxels::U8(Array3::from_shape_fn(shape, |i| (value(i) % 256) as u8)),
        other => panic!("no test volume for {other:?}"),
    }
}

/// `write_volume_at` (the slab-by-slab writer) writes what
/// `build_multiscale` writes, for shapes whose chunking cuts each axis, prime
/// extents and every element type it is used with; and so does the writer fed
/// slabs of uneven sizes along any axis.
#[test]
fn the_slab_writer_writes_what_build_multiscale_writes() {
    let dir = common::scratch("slab_writer");
    let frame =
        PhysicalFrame::new([3.0, 1.5, 1.5], Some("micrometer")).with_origin([0.5, 2.0, -1.0]);
    let cases = [
        (Dtype::U16, [37, 131, 70]),
        (Dtype::U16, [130, 211, 97]),
        (Dtype::F32, [64, 128, 96]),
        (Dtype::Bool, [20, 70, 66]),
        (Dtype::U8, [3, 300, 257]),
        (Dtype::U16, [1, 5, 7]),
    ];
    for (index, (dtype, shape)) in cases.into_iter().enumerate() {
        let data = volume(dtype, shape);
        let factors = pyramid_factors(shape, 4);
        let reference = dir.join(format!("{index}_reference.zarr"));
        let spec = if factors.is_empty() {
            PyramidSpec::new(vec![])
        } else {
            PyramidSpec::new(factors.clone())
        }
        .expect("a pyramid")
        .with_physical(frame.clone());
        ZarrEnvironment::build_multiscale(&reference, &data, &spec).expect("a reference");
        let written = dir.join(format!("{index}_written.zarr"));
        write_volume_at(
            &written,
            &data,
            frame.voxel_size,
            frame.origin,
            frame.unit.as_deref(),
            4,
        )
        .expect("written");
        assert_same_files(&reference, &written);

        if dtype != Dtype::U16 {
            continue;
        }
        let view = data.view::<u16>().expect("u16");
        for axis in 0..3 {
            let uneven = dir.join(format!("{index}_axis{axis}.zarr"));
            let mut writer =
                PyramidWriter::<u16>::create(&uneven, shape, &factors, axis, Some(frame.clone()))
                    .expect("a writer");
            assert_eq!(writer.chunk(), chunk_for_block(shape, dtype));
            let (mut at, mut step) = (0, 1);
            while at < shape[axis] {
                let stop = (at + step).min(shape[axis]);
                writer
                    .push(
                        view.slice_axis(Axis(axis), ndarray::Slice::from(at..stop))
                            .to_owned(),
                    )
                    .expect("a slab");
                at = stop;
                step = step * 2 + 1;
            }
            writer.finish().expect("finished");
            assert_same_files(&reference, &uneven);
        }
    }
}

/// A writer that was not given every row refuses to finish.
#[test]
fn an_unfinished_pyramid_is_refused() {
    let dir = common::scratch("slab_writer_unfinished");
    let mut writer =
        PyramidWriter::<u16>::create(&dir.join("out.zarr"), [8, 70, 70], &[[1, 2, 2]], 1, None)
            .expect("a writer");
    writer.push(Array3::zeros([8, 30, 70])).expect("a slab");
    let message = writer.finish().expect_err("a refusal").to_string();
    assert!(message.contains("rows"), "{message}");
}
