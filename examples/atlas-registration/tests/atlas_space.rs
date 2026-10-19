// SPDX-License-Identifier: MIT
// Original work for this crate.

//! `PointTransform::sample_frame` / `atlas_frame` and `atlas_space`'s
//! `RegisteredAtlas` on a synthetic prepared atlas and hand-written
//! registrations whose answers are known: frames, the atlas checks, the change
//! of lattice from an image to the sample (with an offset), lookups inside and
//! outside the atlas, collapses, and densities.

use std::fs;
use std::path::{Path, PathBuf};

use atlas_regions::atlas_zarr::{write_prepared_atlas, WriteAtlasParams};
use atlas_regions::{AtlasStack, Ontology, OntologyLevel, PreparedWindow, Rounding};
use blockflow::voxels::Voxels;
use blockflow_atlas_registration::atlas_space::{
    ImageFrame, ImageToSample, LookupParams, OpenParams, RegisteredAtlas,
};
use blockflow_atlas_registration::PointTransform;
use ndarray::Array3;

const SAMPLE_SHAPE: [usize; 3] = [10, 20, 24];
const SAMPLE_SCALE: [f64; 3] = [4.0, 2.0, 2.0];
const SAMPLE_TRANSLATION: [f64; 3] = [8.0, -2.0, 0.0];
const ATLAS_SHAPE: [usize; 3] = [8, 20, 24];
const ATLAS_UM: f64 = 2.0;
/// Sample physical to atlas physical: `diag(A) p + T`.
const A_DIAGONAL: [f64; 3] = [0.8, 1.0, 1.0];
const T: [f64; 3] = [8.0, -2.0, 2.0];

const TREE: &str = r#"{"msg": [{"id": 997, "name": "root", "acronym": "root",
  "color_hex_triplet": "FFFFFF", "st_level": 0, "children": [
    {"id": 8, "name": "grey", "acronym": "grey", "color_hex_triplet": "BFDAE3", "st_level": 1,
     "children": [{"id": 567, "name": "Cerebrum", "acronym": "CH",
                   "color_hex_triplet": "B0F0FF", "st_level": 2, "children": []}]}]}]}"#;

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("atlas-space-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn region_at([z, y, _]: [usize; 3]) -> u32 {
    if y < 2 {
        0
    } else if z < 4 {
        567
    } else {
        8
    }
}

struct Atlas {
    voxel_um: [f64; 3],
    bump: bool,
    distance: bool,
    hemispheres: bool,
}

impl Default for Atlas {
    fn default() -> Self {
        Self {
            voxel_um: [ATLAS_UM; 3],
            bump: false,
            distance: true,
            hemispheres: true,
        }
    }
}

/// A prepared atlas: regions by z (0 at low y), hemispheres by x, distance
/// to surface `y` voxels; `bump` changes one reference voxel.
fn write_atlas(root: &Path, shape: [usize; 3], atlas: Atlas) {
    let dims = (shape[0], shape[1], shape[2]);
    let stack = AtlasStack {
        annotation: Voxels::U32(Array3::from_shape_fn(dims, |(z, y, x)| {
            region_at([z, y, x])
        })),
        reference: Some(Voxels::U16(Array3::from_shape_fn(dims, |(z, y, x)| {
            (z * 1000 + y * 30 + x + usize::from(atlas.bump && z == 1 && y == 1 && x == 1)) as u16
        }))),
        hemispheres: atlas.hemispheres.then(|| {
            Voxels::U8(Array3::from_shape_fn(
                dims,
                |(_, _, x)| {
                    if x < 12 {
                        255
                    } else {
                        0
                    }
                },
            ))
        }),
        distance_to_surface: atlas
            .distance
            .then(|| Voxels::F64(Array3::from_shape_fn(dims, |(_, y, _)| y as f64))),
    };
    let ontology = Ontology::from_ontology_json(TREE, OntologyLevel::TreeDepth).unwrap();
    write_prepared_atlas(
        root,
        &stack,
        &PreparedWindow::identity(shape),
        &ontology,
        &WriteAtlasParams {
            voxel_size_um: atlas.voxel_um,
            name: "synthetic".into(),
            replace: true,
            workers: 2,
        },
    )
    .unwrap();
}

/// `path`, level-0 shape, scale, translation, and the unit to record (none:
/// a sidecar from before units were recorded).
type Side = (
    PathBuf,
    [usize; 3],
    [f64; 3],
    [f64; 3],
    Option<&'static str>,
);

/// One registration step, `diag(a) p + t` in physical units.
fn write_step(dir: &Path, fixed: Side, moving: Side, a: [f64; 3], t: [f64; 3]) {
    fs::create_dir_all(dir).unwrap();
    let side = |(path, shape, scale, translation, unit): &Side| {
        let mut side = serde_json::json!({"path": path.display().to_string(), "channel": 0,
            "level": 0, "level0_shape": shape,
            "level0_frame": {"scale": scale, "translation": translation}});
        if let Some(unit) = unit {
            side["unit"] = serde_json::json!(unit);
        }
        side
    };
    let json = serde_json::json!({
        "schema": 1,
        "axis_order": ["z", "y", "x"],
        "transform": "TransformParameters.0.txt",
        "direction": "fixed physical -> moving physical",
        "fixed": side(&fixed),
        "moving": side(&moving),
    });
    fs::write(
        dir.join("registration.json"),
        serde_json::to_vec_pretty(&json).unwrap(),
    )
    .unwrap();
    let ([a0, a1, a2], [t0, t1, t2]) = (a, t);
    let (shape, scale, origin) = (fixed.1, fixed.2, fixed.3);
    fs::write(
        dir.join("TransformParameters.0.txt"),
        format!(
            "(Transform \"AffineTransform\")\n(NumberOfParameters 12)\n\
             (TransformParameters {a0} 0 0 0 {a1} 0 0 0 {a2} {t0} {t1} {t2})\n\
             (InitialTransformParametersFileName \"NoInitialTransform\")\n\
             (HowToCombineTransforms \"Compose\")\n(CenterOfRotationPoint 0.0 0.0 0.0)\n\
             (Size {} {} {})\n(Index 0 0 0)\n(Spacing {} {} {})\n(Origin {} {} {})\n\
             (Direction 1 0 0 0 1 0 0 0 1)\n(FixedImageDimension 3)\n(MovingImageDimension 3)\n",
            shape[2],
            shape[1],
            shape[0],
            scale[2],
            scale[1],
            scale[0],
            origin[2],
            origin[1],
            origin[0]
        ),
    )
    .unwrap();
}

/// A bare OME-Zarr group declaring `unit` on its axes (no arrays).
fn write_unit_only(root: &Path, unit: &str, scale: [f64; 3], translation: [f64; 3]) {
    fs::create_dir_all(root).unwrap();
    let axes: Vec<_> = ["z", "y", "x"]
        .iter()
        .map(|name| serde_json::json!({"name": name, "type": "space", "unit": unit}))
        .collect();
    let json = serde_json::json!({"zarr_format": 3, "node_type": "group", "attributes": {
        "ome": {"version": "0.5", "multiscales": [{"axes": axes, "datasets": [{"path": "0",
        "coordinateTransformations": [{"type": "scale", "scale": scale},
                                      {"type": "translation", "translation": translation}]}]}]}}});
    fs::write(root.join("zarr.json"), serde_json::to_vec(&json).unwrap()).unwrap();
}

fn sample_side(dir: &Path, unit: Option<&'static str>) -> Side {
    (
        dir.join("sample.ome.zarr"),
        SAMPLE_SHAPE,
        SAMPLE_SCALE,
        SAMPLE_TRANSLATION,
        unit,
    )
}

fn atlas_side(atlas: &Path) -> Side {
    (
        atlas.to_path_buf(),
        ATLAS_SHAPE,
        [ATLAS_UM; 3],
        [0.0; 3],
        Some("micrometer"),
    )
}

/// The registration's sample voxels to atlas voxels, as the test states it.
fn expected_atlas(sample: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| {
        let physical = SAMPLE_TRANSLATION[a] + sample[a] * SAMPLE_SCALE[a];
        (A_DIAGONAL[a] * physical + T[a]) / ATLAS_UM
    })
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|axis| (a[axis] - b[axis]).abs() < 1e-6)
}

#[test]
fn the_frames_are_the_first_fixed_and_the_last_moving_images() {
    let dir = scratch("frames");
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, Atlas::default());

    // One step, the units recorded.
    let single = dir.join("single");
    write_step(
        &single,
        sample_side(&dir, Some("micrometer")),
        atlas_side(&atlas),
        A_DIAGONAL,
        T,
    );
    let transform = PointTransform::from_registration_dir(&single).unwrap();
    let sample = transform.sample_frame();
    assert_eq!(
        sample.path.as_deref(),
        Some(dir.join("sample.ome.zarr").as_path())
    );
    assert_eq!(sample.shape, SAMPLE_SHAPE);
    assert_eq!(sample.frame.scale, SAMPLE_SCALE);
    assert_eq!(sample.frame.translation, SAMPLE_TRANSLATION);
    assert_eq!(sample.unit.as_deref(), Some("micrometer"));
    let moving = transform.atlas_frame();
    assert_eq!(moving.path.as_deref(), Some(atlas.as_path()));
    assert_eq!(moving.shape, ATLAS_SHAPE);
    assert_eq!(moving.frame.scale, [ATLAS_UM; 3]);
    assert_eq!(moving.frame.translation, [0.0; 3]);

    // A chain: the first step's fixed image and the last step's moving one.
    // The units are not recorded; the sample's comes from its OME-Zarr, the
    // middle image's is unknown.
    let chain = dir.join("chain");
    let middle: Side = (
        dir.join("middle.ome.zarr"),
        [20, 20, 24],
        [2.0; 3],
        [0.0; 3],
        None,
    );
    write_step(
        &chain.join("a"),
        sample_side(&dir, None),
        middle.clone(),
        [1.0; 3],
        [0.0; 3],
    );
    write_step(
        &chain.join("b"),
        middle,
        atlas_side(&atlas),
        [1.0; 3],
        [0.0; 3],
    );
    fs::write(chain.join("chain.json"), r#"{"steps": ["a", "b"]}"#).unwrap();
    let transform = PointTransform::from_registration_dir(&chain).unwrap();
    assert_eq!(transform.steps(), 2);
    assert_eq!(transform.sample_frame().unit, None);
    write_unit_only(
        &dir.join("sample.ome.zarr"),
        "millimeter",
        SAMPLE_SCALE,
        SAMPLE_TRANSLATION,
    );
    let transform = PointTransform::from_registration_dir(&chain).unwrap();
    let sample = transform.sample_frame();
    assert_eq!(sample.unit.as_deref(), Some("millimeter"));
    assert_eq!(sample.shape, SAMPLE_SHAPE);
    assert_eq!(sample.frame.translation, SAMPLE_TRANSLATION);
    assert_eq!(
        transform.atlas_frame().path.as_deref(),
        Some(atlas.as_path())
    );
    assert_eq!(transform.atlas_frame().unit.as_deref(), Some("micrometer"));

    // The OME-Zarr frame reader agrees with the metadata written.
    let frame = ImageFrame::from_ome_zarr(&dir.join("sample.ome.zarr")).unwrap();
    assert_eq!(
        frame,
        ImageFrame::new(SAMPLE_SCALE, SAMPLE_TRANSLATION, Some("millimeter"))
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_atlas_that_is_not_the_registrations_is_refused() {
    let dir = scratch("refused");
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, Atlas::default());
    let registration = dir.join("registration");
    write_step(
        &registration,
        sample_side(&dir, None),
        atlas_side(&atlas),
        A_DIAGONAL,
        T,
    );
    let open = |atlas: &Path, check_reference: bool| {
        RegisteredAtlas::open(
            &registration,
            &OpenParams {
                atlas: Some(atlas.to_path_buf()),
                check_reference,
            },
        )
    };
    // The moving image is a prepared atlas, so it is the default.
    let registered = RegisteredAtlas::open(&registration, &OpenParams::default()).unwrap();
    assert_eq!(registered.atlas.root, atlas);

    let other = dir.join("other.ome.zarr");
    write_atlas(&other, [8, 20, 23], Atlas::default());
    let error = open(&other, true).unwrap_err().to_string();
    assert!(
        error.contains("[8, 20, 23]") && error.contains("[8, 20, 24]"),
        "{error}"
    );

    let coarse = dir.join("coarse.ome.zarr");
    let voxel_um = [ATLAS_UM, ATLAS_UM, 3.0];
    write_atlas(
        &coarse,
        ATLAS_SHAPE,
        Atlas {
            voxel_um,
            ..Atlas::default()
        },
    );
    let error = open(&coarse, false).unwrap_err().to_string();
    assert!(error.contains("µm voxels"), "{error}");

    let bumped = dir.join("bumped.ome.zarr");
    write_atlas(
        &bumped,
        ATLAS_SHAPE,
        Atlas {
            bump: true,
            ..Atlas::default()
        },
    );
    let error = open(&bumped, true).unwrap_err().to_string();
    assert!(error.contains("at 1 voxels"), "{error}");
    assert!(open(&bumped, false).is_ok());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn image_points_go_through_the_sample_to_the_atlas() {
    let dir = scratch("map");
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, Atlas::default());
    let registration = dir.join("registration");
    write_step(
        &registration,
        sample_side(&dir, Some("micrometer")),
        atlas_side(&atlas),
        A_DIAGONAL,
        T,
    );
    let registered = RegisteredAtlas::open(&registration, &OpenParams::default()).unwrap();

    // An image twice as fine as the sample, at its origin on z and y, placed
    // by a 2 µm offset on x.
    let image = ImageFrame::new([2.0, 1.0, 1.0], [8.0, -2.0, 0.0], Some("µm"));
    let to_sample = registered.image_to_sample(&image, [0.0, 0.0, 2.0]).unwrap();
    assert_eq!(
        to_sample,
        ImageToSample {
            scale: [0.5; 3],
            offset: [0.0, 0.0, 1.0]
        }
    );
    let points = [[0.0, 4.0, 0.0], [3.0, 7.0, 11.0], [19.0, 39.0, 47.0]];
    let mapped = registered.map(&to_sample, &points);
    for ((point, sample), atlas) in points.iter().zip(&mapped.sample).zip(&mapped.atlas) {
        assert!(close(*sample, to_sample.apply(*point)));
        assert!(close(*atlas, expected_atlas(*sample)));
    }
    // The last is at sample [9.5, 19.5, 24.5]: past the box.
    assert_eq!(mapped.outside_sample, 1);
    assert!(registered.frame_warning(&to_sample, [20, 40, 46]).is_none());
    assert!(registered.frame_warning(&to_sample, [20, 40, 60]).is_some());
    let unshifted = registered.map_points(&image, &points[..1]).unwrap();
    assert!(close(unshifted[0], expected_atlas([0.0, 2.0, 0.0])));

    // Units must agree, and an offset in µm needs an image in µm.
    let millimetres = ImageFrame::new([2.0, 1.0, 1.0], [0.0; 3], Some("millimeter"));
    let error = registered
        .image_to_sample(&millimetres, [0.0; 3])
        .unwrap_err()
        .to_string();
    assert!(error.contains("millimeter"), "{error}");
    let pixels = ImageFrame::new([1.0; 3], [0.0; 3], Some("pixel"));
    let mut unknown = registered.sample().clone();
    unknown.unit = None;
    assert!(ImageToSample::new(&pixels, &unknown, [0.0; 3]).is_ok());
    assert!(ImageToSample::new(&pixels, &unknown, [1.0, 0.0, 0.0]).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn lookups_round_collapse_and_read_outside_the_atlas() {
    let dir = scratch("lookup");
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, Atlas::default());
    let registration = dir.join("registration");
    write_step(
        &registration,
        sample_side(&dir, None),
        atlas_side(&atlas),
        A_DIAGONAL,
        T,
    );
    let registered = RegisteredAtlas::open(&registration, &OpenParams::default()).unwrap();
    assert_eq!(registered.shape(), ATLAS_SHAPE);
    assert_eq!(registered.voxel_length_um(), Some(ATLAS_UM));
    assert!((registered.voxel_volume_mm3() - 8e-9).abs() < 1e-18);

    let points = [
        [1.0, 5.0, 3.0],   // 567, left
        [3.5, 5.0, 13.0],  // ties to even: z 4 (8); floor: z 3 (567)
        [6.0, 1.0, 3.0],   // background
        [-3.0, 30.0, 3.0], // outside: distance clamped at y 19
    ];
    let nearest = registered
        .lookup(&points, &LookupParams::default())
        .unwrap();
    assert_eq!(nearest.region, vec![567, 8, 0, 0]);
    assert_eq!(nearest.hemisphere, Some(vec![255, 0, 255, -1]));
    assert_eq!(
        nearest.distance_to_surface_voxels,
        vec![5.0, 5.0, 1.0, 19.0]
    );
    let floor = LookupParams {
        level: None,
        rounding: Rounding::Floor,
    };
    assert_eq!(
        registered.lookup(&points, &floor).unwrap().region,
        vec![567, 567, 0, 0]
    );
    let collapsed = LookupParams {
        level: Some(1),
        rounding: Rounding::Nearest,
    };
    assert_eq!(
        registered.lookup(&points, &collapsed).unwrap().region,
        vec![8, 8, 0, 0]
    );
    assert_eq!(registered.outside(&points, Rounding::Nearest), 1);
    assert_eq!(
        registered.voxel([3.5, 5.0, 13.0], Rounding::Floor),
        Some([3, 5, 13])
    );

    // Region voxels, collapsed or not.
    let voxels = registered.region_voxels(None).unwrap();
    assert_eq!(voxels[&(567, 255)], 4 * 18 * 12);
    assert_eq!(voxels[&(0, 0)], 8 * 2 * 12);
    let collapsed = registered.region_voxels(Some(1)).unwrap();
    assert_eq!(collapsed[&(8, 0)], 8 * 18 * 12);

    // The description says the distance's unit.
    assert_eq!(
        registered.describe()["distance_to_surface_unit"],
        "atlas voxel"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_distance_is_computed_in_voxels_and_hemispheres_may_be_absent() {
    let dir = scratch("computed");
    let atlas = dir.join("atlas.ome.zarr");
    let bare = Atlas {
        distance: false,
        hemispheres: false,
        ..Atlas::default()
    };
    write_atlas(&atlas, ATLAS_SHAPE, bare);
    let registration = dir.join("registration");
    write_step(
        &registration,
        sample_side(&dir, None),
        atlas_side(&atlas),
        [1.0; 3],
        [0.0; 3],
    );
    let registered = RegisteredAtlas::open(&registration, &OpenParams::default()).unwrap();
    let lookup = registered
        .lookup(
            &[[0.0, 2.0, 5.0], [3.0, 3.0, 11.0]],
            &LookupParams::default(),
        )
        .unwrap();
    assert_eq!(lookup.hemisphere, None);
    // The nearest background voxels are at y 1, in voxels.
    assert_eq!(lookup.distance_to_surface_voxels, vec![1.0, 2.0]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn densities_count_or_weigh_the_points_inside() {
    let dir = scratch("density");
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, Atlas::default());
    let registration = dir.join("registration");
    write_step(
        &registration,
        sample_side(&dir, None),
        atlas_side(&atlas),
        [1.0; 3],
        [0.0; 3],
    );
    let registered = RegisteredAtlas::open(&registration, &OpenParams::default()).unwrap();
    let points = [
        [2.0, 10.0, 10.0],
        [2.5, 10.0, 10.0],
        [8.0, 0.0, 0.0],
        [1.0, 1.0, f64::NAN],
    ];

    let counts = registered
        .density(&points, None, Rounding::Nearest, [0; 3])
        .unwrap();
    assert_eq!(counts.dropped, 2);
    let field = counts.voxels.view::<f64>().unwrap();
    assert_eq!(field.shape(), &[8, 20, 24]);
    assert_eq!(field[[2, 10, 10]], 2.0); // 2.5 ties to 2
    assert_eq!(field.sum(), 2.0);

    let weights = [1.5, 2.0, 1.0, 1.0];
    let weighted = registered
        .density(&points, Some(&weights), Rounding::Floor, [1, 1, 1])
        .unwrap();
    let field = weighted.voxels.view::<f64>().unwrap();
    assert_eq!(field[[2, 10, 10]], 3.5);
    assert_eq!(field[[3, 10, 10]], 3.5);
    assert_eq!(field.sum(), 3.5 * 7.0);
    assert!(registered
        .density(&points, Some(&weights[..2]), Rounding::Floor, [0; 3])
        .is_err());
    let _ = fs::remove_dir_all(&dir);
}

/// The osf-cztq2 registration (autofluorescence → the 25 µm 2017 atlas) and
/// its prepared atlas. Needs both under /husky/henriksson/lightsheet-reference.
#[test]
#[ignore = "needs the osf-cztq2 registration and prepared atlas under /husky/henriksson/lightsheet-reference"]
fn the_osf_cztq2_registration_opens_against_its_prepared_atlas() {
    let root = Path::new("/husky/henriksson/lightsheet-reference");
    let registration = root.join("registration/ours/auto_to_reference");
    let atlas = root.join("atlas/ABA_25um_2017_1_2_3_z0-256.ome.zarr");
    let started = std::time::Instant::now();
    let registered = RegisteredAtlas::open(
        &registration,
        &OpenParams {
            atlas: Some(atlas),
            check_reference: true,
        },
    )
    .unwrap();
    let opened = started.elapsed().as_secs_f64();
    let sample = registered.sample();
    assert_eq!(sample.frame.scale, [25.0; 3]);
    assert_eq!(registered.voxel_length_um(), Some(25.0));
    // The sample's centre lands inside the atlas, in the brain.
    let centre = sample.shape.map(|n| (n as f64 - 1.0) / 2.0);
    let image = ImageFrame::new(sample.frame.scale, sample.frame.translation, None);
    let atlas_at = registered.map_points(&image, &[centre]).unwrap();
    let lookup = registered
        .lookup(&atlas_at, &LookupParams::default())
        .unwrap();
    eprintln!(
        "opened in {opened:.2} s; sample centre {centre:?} -> atlas {:?}: region {}, \
         hemisphere {:?}, {} voxels deep",
        atlas_at[0], lookup.region[0], lookup.hemisphere, lookup.distance_to_surface_voxels[0]
    );
    assert!(registered.voxel(atlas_at[0], Rounding::Nearest).is_some());
    assert!(lookup.region[0] > 0 && lookup.distance_to_surface_voxels[0] > 0.0);
}
