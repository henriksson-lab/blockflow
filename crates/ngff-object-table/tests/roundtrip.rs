use ngff_object_table::{
    ColumnRole, ColumnSpec, CoordinateColumn, DType, SpatialIndexSpec, TableSpec, TableWriter,
};

#[test]
fn chunked_columns_and_indexes_round_trip() {
    let scratch = tempfile::tempdir().unwrap();
    let columns = vec![
        ColumnSpec::new("label_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("area_pixels", DType::U32, ColumnRole::Measurement),
    ];
    let spatial_index = SpatialIndexSpec {
        coordinates: vec![
            CoordinateColumn {
                axis: "y".into(),
                column: "centroid_y".into(),
            },
            CoordinateColumn {
                axis: "x".into(),
                column: "centroid_x".into(),
            },
        ],
        tile_shape: vec![1024, 1024],
        grid_shape: vec![2, 2],
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
    };
    let spec = TableSpec::new(5, 2, "../../", "label_id", columns, spatial_index);
    let writer = TableWriter::create(scratch.path().join("cells"), spec).unwrap();

    writer.write_u64("label_id", 0, &[10, 11]).unwrap();
    writer.write_u64("label_id", 2, &[20, 21]).unwrap();
    writer.write_u64("label_id", 4, &[30]).unwrap();
    writer.write_f32("centroid_y", 0, &[1., 2.]).unwrap();
    writer.write_f32("centroid_y", 2, &[3., 4.]).unwrap();
    writer.write_f32("centroid_y", 4, &[5.]).unwrap();
    writer.write_f32("centroid_x", 0, &[6., 7.]).unwrap();
    writer.write_f32("centroid_x", 2, &[8., 9.]).unwrap();
    writer.write_f32("centroid_x", 4, &[10.]).unwrap();
    writer.write_u32("area_pixels", 0, &[100, 110]).unwrap();
    writer.write_u32("area_pixels", 2, &[120, 130]).unwrap();
    writer.write_u32("area_pixels", 4, &[140]).unwrap();
    writer.write_identity_index(0, &[10, 11], &[0, 1]).unwrap();
    writer.write_identity_index(2, &[20, 21], &[2, 3]).unwrap();
    writer.write_identity_index(4, &[30], &[4]).unwrap();
    writer
        .write_spatial_index(&[0, 2, 3, 5], &[2, 1, 2, 0])
        .unwrap();

    let reader = writer.finish().unwrap();
    assert_eq!(reader.read_u64("label_id", 1..4).unwrap(), [11, 20, 21]);
    assert_eq!(reader.read_f32("centroid_x", 3..5).unwrap(), [9., 10.]);
    assert_eq!(
        reader.read_identity_index(2..5).unwrap(),
        (vec![20, 21, 30], vec![2, 3, 4])
    );
    assert_eq!(
        reader.read_spatial_index().unwrap(),
        (vec![0, 2, 3, 5], vec![2, 1, 2, 0])
    );
    assert_eq!(
        reader.read_spatial_index_region(&[1, 0], &[1, 2]).unwrap(),
        (vec![3, 5], vec![2, 0])
    );
    assert_eq!(reader.occupancy_shapes(), [vec![2, 2], vec![1, 1]]);
    assert_eq!(reader.read_occupancy(0, &[1, 0], &[1, 2]).unwrap(), [2, 0]);
    assert_eq!(reader.read_occupancy(1, &[0, 0], &[1, 1]).unwrap(), [5]);
}
