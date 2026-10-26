use blockflow::{
    voxels::Voxels,
    zarr_env::{Compression, PhysicalFrame, TileWriter},
};
use std::path::PathBuf;
fn temp() -> PathBuf {
    std::env::temp_dir().join(format!(
        "tile-writer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
#[test]
fn clipped_chunks_write_exact_native_data_and_calibrated_single_level_metadata() {
    let root = temp();
    let physical =
        PhysicalFrame::new([1., 7., 4.], Some("micrometer")).with_origin([0., -21., -16.]);
    let mut writer = TileWriter::<f64>::create(
        &root,
        [1, 7, 9],
        [1, 3, 4],
        Some(physical),
        Compression::None,
    )
    .unwrap();
    assert_eq!(writer.total_chunks(), 9);
    while let Some((p, s)) = writer.next_tile() {
        assert!(!root.join("zarr.json").exists());
        let values: Vec<_> = (0..s[1])
            .flat_map(|y| {
                (0..s[2]).map(move |x| 100. * (p[1] + y) as f64 + (p[2] + x) as f64 + 0.125)
            })
            .collect();
        writer.write_next(&values).unwrap();
    }
    assert!(writer.write_next(&[]).is_err());
    let array = writer.finish().unwrap();
    match array.read().unwrap() {
        Voxels::F64(v) => {
            for y in 0..7 {
                for x in 0..9 {
                    assert_eq!(v[[0, y, x]], 100. * y as f64 + x as f64 + 0.125);
                }
            }
        }
        _ => panic!("dtype"),
    }
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("zarr.json")).unwrap()).unwrap();
    let transforms = &metadata["attributes"]["ome"]["multiscales"][0]["datasets"][0]
        ["coordinateTransformations"];
    assert_eq!(transforms[0]["scale"], serde_json::json!([1., 7., 4.]));
    assert_eq!(
        transforms[1]["translation"],
        serde_json::json!([0., -21., -16.])
    );
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn incomplete_bad_payload_existing_roots_and_invalid_geometry_cannot_publish() {
    let root = temp();
    let mut writer =
        TileWriter::<u8>::create(&root, [1, 7, 9], [1, 3, 4], None, Compression::None).unwrap();
    let before = std::fs::read(root.join("0/zarr.json")).unwrap();
    assert!(
        TileWriter::<u8>::create(&root, [1, 7, 9], [1, 3, 4], None, Compression::None).is_err()
    );
    assert_eq!(before, std::fs::read(root.join("0/zarr.json")).unwrap());
    assert!(writer.write_next(&[1]).is_err());
    assert_eq!(writer.chunks_written(), 0);
    assert!(writer.finish().is_err());
    assert!(!root.join("zarr.json").exists());
    std::fs::remove_dir_all(root).unwrap();
    for (shape, chunk) in [
        ([0, 3, 4], [1, 2, 2]),
        ([1, 3, 4], [1, 0, 2]),
        ([usize::MAX, 2, 2], [1; 3]),
        ([1, 1, usize::MAX], [1, 1, usize::MAX]),
    ] {
        let root = temp();
        assert!(TileWriter::<f64>::create(&root, shape, chunk, None, Compression::None).is_err());
        assert!(!root.exists());
    }
}
#[test]
fn failed_backend_write_poisoning_prevents_later_finish() {
    let root = temp();
    let mut writer =
        TileWriter::<f64>::create(&root, [1, 2, 2], [1, 2, 2], None, Compression::None).unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("0/zarr.json")).unwrap()).unwrap();
    assert_eq!(
        meta["chunk_key_encoding"]["configuration"]["separator"],
        "/"
    );
    std::fs::write(root.join("0/c"), b"obstruct directory creation").unwrap();
    assert!(writer.write_next(&[1.; 4]).is_err());
    assert_eq!(writer.chunks_written(), 0);
    assert!(writer.next_tile().is_none());
    assert!(writer.finish().is_err());
    assert!(!root.join("zarr.json").exists());
    std::fs::remove_dir_all(root).unwrap();
}
