use blockflow::{budget::MemoryBudget, feature_index::*};
use std::path::PathBuf;
fn dir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "bf-feature-index-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!p.exists());
    p
}
fn geometry() -> Geometry {
    Geometry {
        shape_zyx: [8, 8, 12],
        cell_zyx: [4, 4, 4],
        origin_xyz: [-5., 10., 20.],
        spacing_xyz: [0.5, 1.5, 2.],
        units: "micrometer".into(),
        max_offset_voxels: 0.5,
    }
}
fn point(g: &Geometry, x: usize, weight: f64) -> Feature {
    let at = [1, 1, x];
    Feature {
        at_zyx: at,
        position_xyz: std::array::from_fn(|i| {
            g.origin_xyz[i] + at[2 - i] as f64 * g.spacing_xyz[i]
        }),
        weight,
        scale: 1.,
    }
}
fn done<T>(x: Outcome<T>) -> T {
    match x {
        Outcome::Complete(v) => v,
        Outcome::Stopped(s) => panic!("unexpected stop {s:?}"),
    }
}
fn binding() -> String {
    "a".repeat(64)
}

#[test]
fn physical_anchor_checks_account_for_translated_world_roundoff() {
    // Includes the actual human Blaze 6b working frame; its Y half-voxel
    // coordinate previously exceeded the local-q-only floating-point bound.
    let origin = [1428.036304347826, 8006.806983082706, 20.058823529411764];
    let spacing = [39.41260869565217, 39.39396616541353, 40.11764705882353];
    let q: f64 = ((origin[1] + 0.5 * spacing[1]) - origin[1]) / spacing[1];
    assert!((q - 1.).abs() > 0.5 + 16. * f64::EPSILON * q.abs().max(1.));
    for signs in 0..8 {
        for allowance in [0., 0.5] {
            let mut g = geometry();
            g.origin_xyz =
                std::array::from_fn(|i| origin[i] * if signs & (1 << i) == 0 { 1. } else { -1. });
            g.spacing_xyz = spacing;
            g.max_offset_voxels = allowance;
            for displacement in [-allowance, 0., allowance] {
                let path = dir("world-roundoff");
                let memory = MemoryBudget::new(256 * 1024);
                let mut writer =
                    Writer::create(&path, g.clone(), binding(), 2, memory.clone()).unwrap();
                let mut p = point(&g, 1, 1.);
                let anchor = if displacement < 0. {
                    4
                } else if displacement > 0. {
                    3
                } else {
                    1
                };
                p.at_zyx = [anchor; 3];
                p.position_xyz = std::array::from_fn(|i| {
                    g.origin_xyz[i] + (anchor as f64 + displacement) * g.spacing_xyz[i]
                });
                let mut b = Budget::new(Limits::default());
                assert_eq!(
                    done(
                        writer
                            .write_cell(g.cell_of(p.at_zyx).unwrap(), [p], &mut b)
                            .unwrap()
                    ),
                    1
                );
                writer.finish().unwrap();
                let reader = Reader::open(&path, &binding(), memory).unwrap();
                for radius in [0.001, 1e-14] {
                    let found = done(reader.neighbors(p.position_xyz, radius, 2, &mut b).unwrap());
                    assert_eq!(found.points.len(), 1, "signs {signs}, allowance {allowance}, displacement {displacement}, radius {radius}");
                    assert_eq!(found.points[0].feature, p);
                }
                std::fs::remove_dir_all(path).unwrap();
            }
            for axis in 0..3 {
                for displacement in [-allowance - 1e-6, allowance + 1e-6] {
                    let path = dir("bad-world-offset");
                    let mut writer = Writer::create(
                        &path,
                        g.clone(),
                        binding(),
                        2,
                        MemoryBudget::new(256 * 1024),
                    )
                    .unwrap();
                    let mut p = point(&g, 1, 1.);
                    p.position_xyz[axis] =
                        g.origin_xyz[axis] + (1. + displacement) * g.spacing_xyz[axis];
                    assert!(writer
                        .write_cell([0, 0, 0], [p], &mut Budget::new(Limits::default()))
                        .is_err());
                    drop(writer);
                    std::fs::remove_dir_all(path).unwrap();
                }
            }
        }
    }
}

#[test]
fn unresolved_world_coordinates_are_refused_instead_of_widening_offset_bounds() {
    let mut g = geometry();
    g.origin_xyz = [1e20, -1e20, 1e20];
    g.spacing_xyz = [1., 1.5, 2.];
    let path = dir("unresolved-world");
    let mut writer = Writer::create(
        &path,
        g.clone(),
        binding(),
        2,
        MemoryBudget::new(256 * 1024),
    )
    .unwrap();
    let error = writer
        .write_cell(
            [0, 0, 0],
            [point(&g, 1, 1.)],
            &mut Budget::new(Limits::default()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("lack subvoxel precision"));
    drop(writer);
    std::fs::remove_dir_all(path).unwrap();
}
fn build(path: &std::path::Path, g: Geometry, reverse: bool) -> Metadata {
    let mut writer =
        Writer::create(path, g.clone(), binding(), 2, MemoryBudget::new(256 * 1024)).unwrap();
    let mut b = Budget::new(Limits::default());
    let mut points = vec![
        point(&g, 1, 1.),
        point(&g, 2, 2.),
        point(&g, 3, 3.),
        point(&g, 2, 1.),
    ];
    if reverse {
        points.reverse();
    }
    assert_eq!(
        done(writer.write_cell([0, 0, 0], points, &mut b).unwrap()),
        2
    );
    assert_eq!(
        done(
            writer
                .write_cell([0, 0, 1], [point(&g, 5, 1.), point(&g, 6, 2.)], &mut b)
                .unwrap()
        ),
        2
    );
    writer.finish().unwrap()
}
#[test]
fn canonical_local_retention_roundtrips_existing_table_encoding_and_radius_queries() {
    let a = dir("a");
    let z = dir("z");
    let g = geometry();
    let meta = build(&a, g.clone(), false);
    build(&z, g.clone(), true);
    assert_eq!(meta.submitted_points, 6);
    assert_eq!(meta.retained_points, 4);
    assert_eq!(meta.occupied_cells, 2);
    assert_eq!(meta.retention_events, 1);
    assert_eq!(
        std::fs::read(a.join("pages/0/0/0.rows")).unwrap(),
        std::fs::read(z.join("pages/0/0/0.rows")).unwrap()
    );
    let mem = MemoryBudget::new(256 * 1024);
    let mut r = Reader::open(&a, &binding(), mem.clone()).unwrap();
    let mut b = Budget::new(Limits::default());
    assert_eq!(done(r.next_cell(&mut b).unwrap()), Some([0, 0, 0]));
    let p = point(&g, 3, 1.);
    let q = done(r.neighbors(p.position_xyz, 3., 2, &mut b).unwrap());
    assert!(q.truncated);
    assert_eq!(q.matching_points, 4);
    assert_eq!(
        q.points
            .iter()
            .map(|p| p.feature.at_zyx[2])
            .collect::<Vec<_>>(),
        vec![3, 2]
    );
    assert_eq!(done(r.next_cell(&mut b).unwrap()), Some([0, 0, 1]));
    assert_eq!(done(r.next_cell(&mut b).unwrap()), None);
    assert!(
        b.stats.peak_planned_working_bytes <= mem.total()
            && b.stats.peak_planned_working_bytes > 65536
    );
    assert!(Reader::open(&a, &"b".repeat(64), mem).is_err());
    std::fs::remove_dir_all(a).unwrap();
    std::fs::remove_dir_all(z).unwrap();
}
#[test]
fn budget_stops_are_sticky_unsealed_results_and_corruption_are_not_empty_neighborhoods() {
    let path = dir("stops");
    let g = geometry();
    let memory = MemoryBudget::new(256 * 1024);
    let mut writer = Writer::create(&path, g.clone(), binding(), 2, memory.clone()).unwrap();
    assert!(Reader::open(&path, &binding(), memory.clone()).is_err());
    let mut b = Budget::new(Limits {
        max_work_units: 0,
        ..Limits::default()
    });
    assert!(matches!(
        writer
            .write_cell([0, 0, 0], [point(&g, 1, 1.)], &mut b)
            .unwrap(),
        Outcome::Stopped(Stop::Work)
    ));
    assert!(writer.finish().is_err());
    std::fs::remove_dir_all(&path).unwrap();
    build(&path, g.clone(), false);
    let r = Reader::open(&path, &binding(), memory.clone()).unwrap();
    let lease = memory.try_acquire(220 * 1024).unwrap();
    let mut b = Budget::new(Limits::default());
    assert!(matches!(
        r.neighbors(point(&g, 3, 1.).position_xyz, 2., 2, &mut b)
            .unwrap(),
        Outcome::Stopped(Stop::Memory)
    ));
    drop(lease);
    assert!(matches!(
        r.read_cell([0, 0, 0], &mut b).unwrap(),
        Outcome::Stopped(Stop::Memory)
    ));
    for limits in [
        Limits {
            max_read_calls: 0,
            ..Limits::default()
        },
        Limits {
            max_bytes_read: 1,
            ..Limits::default()
        },
        Limits {
            max_cell_visits: 0,
            ..Limits::default()
        },
    ] {
        let mut b = Budget::new(limits);
        assert!(matches!(
            r.neighbors(point(&g, 3, 1.).position_xyz, 2., 2, &mut b)
                .unwrap(),
            Outcome::Stopped(_)
        ));
    }
    std::fs::remove_file(path.join("pages/0/0/0.rows")).unwrap();
    let mut b = Budget::new(Limits::default());
    assert!(r
        .read_cell([0, 0, 0], &mut b)
        .unwrap_err()
        .to_string()
        .contains("missing an indexed page"));
    std::fs::remove_dir_all(path).unwrap();
}
#[test]
fn subvoxel_features_across_cell_seams_are_found_and_empty_volume_extent_costs_no_ram() {
    let mut peaks = vec![];
    for shape in [[8, 8, 12], [1_000_000_000; 3]] {
        let mut g = geometry();
        g.shape_zyx = shape;
        let path = dir("extent");
        let memory = MemoryBudget::new(256 * 1024);
        let mut w = Writer::create(&path, g.clone(), binding(), 2, memory.clone()).unwrap();
        let mut b = Budget::new(Limits::default());
        let mut p = point(&g, 3, 2.);
        p.position_xyz[0] += 0.5 * g.spacing_xyz[0];
        done(w.write_cell([0, 0, 0], [p], &mut b).unwrap());
        w.finish().unwrap();
        let r = Reader::open(&path, &binding(), memory.clone()).unwrap();
        let q = done(r.neighbors(p.position_xyz, 0.01, 2, &mut b).unwrap());
        assert_eq!(q.points.len(), 1);
        assert_eq!(q.points[0].feature, p);
        peaks.push(memory.high_water());
        std::fs::remove_dir_all(path).unwrap();
    }
    assert_eq!(peaks[0], peaks[1]);
}
#[test]
fn wrong_cell_order_invalid_offsets_and_oversized_pages_are_refused() {
    let path = dir("invalid");
    let g = geometry();
    let memory = MemoryBudget::new(256 * 1024);
    let mut w = Writer::create(&path, g.clone(), binding(), 2, memory.clone()).unwrap();
    let mut b = Budget::new(Limits::default());
    done(w.write_cell([0, 0, 1], [point(&g, 5, 1.)], &mut b).unwrap());
    assert!(w.write_cell([0, 0, 0], [point(&g, 1, 1.)], &mut b).is_err());
    assert!(w.finish().is_err());
    std::fs::remove_dir_all(&path).unwrap();
    build(&path, g, false);
    let r = Reader::open(&path, &binding(), memory).unwrap();
    std::fs::write(path.join("pages/0/0/0.rows"), vec![0u8; 1_000_000]).unwrap();
    let mut b = Budget::new(Limits::default());
    assert!(r.read_cell([0, 0, 0], &mut b).is_err());
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn tiny_radius_between_integer_anchors_is_empty_instead_of_underflowing_cell_bounds() {
    let path = dir("between");
    let mut g = geometry();
    g.max_offset_voxels = 0.;
    build(&path, g.clone(), false);
    let r = Reader::open(&path, &binding(), MemoryBudget::new(256 * 1024)).unwrap();
    let mut b = Budget::new(Limits::default());
    let mut position = point(&g, 3, 1.).position_xyz;
    position[0] += g.spacing_xyz[0] * 0.5;
    let q = done(r.neighbors(position, 0.001, 2, &mut b).unwrap());
    assert!(q.points.is_empty());
    assert!(b.stop.is_none());
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn bounded_cache_preserves_neighbors_and_stream_cursor_and_releases_for_compute() {
    let root = dir("cached");
    let g = geometry();
    build(&root, g.clone(), false);
    let memory = MemoryBudget::new(512 * 1024);
    let mut cached = Reader::open_cached(
        &root,
        &binding(),
        memory.clone(),
        ReadCacheConfig {
            max_bytes: 64 * 1024,
            directory_block_keys: 2,
        },
    )
    .unwrap();
    let plain = Reader::open(&root, &binding(), MemoryBudget::new(512 * 1024)).unwrap();
    for limit in [1, 2, 8] {
        for x in [1, 3, 6] {
            let p = point(&g, x, 1.).position_xyz;
            let a = done(
                plain
                    .neighbors(p, 3., limit, &mut Budget::new(Limits::default()))
                    .unwrap(),
            );
            let b = done(
                cached
                    .neighbors(p, 3., limit, &mut Budget::new(Limits::default()))
                    .unwrap(),
            );
            assert_eq!(a.points, b.points);
            assert_eq!(a.truncated, b.truncated);
            assert_eq!(a.matching_points, b.matching_points);
        }
    }
    let stats = cached.cache_stats();
    assert!(stats.page_hits > 0 && stats.directory_hits > 0);
    assert!(stats.resident_bytes > 0 && stats.peak_resident_bytes <= 64 * 1024);
    let p = point(&g, 3, 1.).position_xyz;
    let mut no_reads = Budget::new(Limits {
        max_read_calls: 0,
        ..Limits::default()
    });
    let warm = done(cached.neighbors(p, 3., 8, &mut no_reads).unwrap());
    assert!(!warm.points.is_empty());
    assert_eq!(no_reads.stats.read_calls, 0);
    assert_eq!(no_reads.stats.page_decodes, 0);
    assert!(no_reads.stats.work_units > 0 && no_reads.stats.cell_visits > 0);
    assert_eq!(
        done(cached.next_cell(&mut no_reads).unwrap()),
        Some([0, 0, 0])
    );
    assert_eq!(
        done(cached.next_cell(&mut no_reads).unwrap()),
        Some([0, 0, 1])
    );
    assert_eq!(done(cached.next_cell(&mut no_reads).unwrap()), None);
    let all = memory
        .try_acquire_reclaiming(memory.total())
        .expect("cache must release its opportunistic leases for compute");
    assert_eq!(cached.cache_stats().resident_bytes, 0);
    assert!(cached.cache_stats().evictions > 0);
    let mut budget = Budget::new(Limits::default());
    assert!(matches!(
        cached.read_cell([0, 0, 0], &mut budget).unwrap(),
        Outcome::Stopped(Stop::Memory)
    ));
    drop(all);
    let mut cold = Budget::new(Limits::default());
    assert!(!done(cached.neighbors(p, 3., 8, &mut cold).unwrap())
        .points
        .is_empty());
    assert!(cold.stats.read_calls > 0 && cold.stats.page_decodes > 0);
    assert!(memory.high_water() <= memory.total());
    cached.clear_cache();
    assert_eq!(memory.opportunistic_in_use(), 0);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn optional_directory_read_amplification_respects_small_byte_limits_and_cold_stops() {
    let root = dir("cache-byte-limit");
    build(&root, geometry(), false);
    for cap in [0, 1, 64 * 1024] {
        let mut r = Reader::open_cached(
            &root,
            &binding(),
            MemoryBudget::new(512 * 1024),
            ReadCacheConfig {
                max_bytes: cap,
                directory_block_keys: 128,
            },
        )
        .unwrap();
        let mut b = Budget::new(Limits {
            max_bytes_read: 24,
            ..Limits::default()
        });
        assert_eq!(done(r.next_cell(&mut b).unwrap()), Some([0, 0, 0]));
        assert_eq!(b.stats.charged_bytes_read, 24);
        assert!(matches!(
            r.next_cell(&mut b).unwrap(),
            Outcome::Stopped(Stop::ReadBytes)
        ));
        assert_eq!(b.stats.charged_bytes_read, 24);
        let mut b = Budget::new(Limits {
            max_cell_visits: 0,
            ..Limits::default()
        });
        assert!(matches!(
            r.read_cell([0, 0, 0], &mut b).unwrap(),
            Outcome::Stopped(Stop::Cells)
        ));
        assert_eq!(b.stats.read_calls, 0);
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cache_cannot_turn_a_low_memory_empty_region_into_a_false_resource_failure() {
    let root = dir("cache-empty-memory");
    let g = geometry();
    build(&root, g.clone(), false);
    for bytes in [0, 64 * 1024] {
        let memory = MemoryBudget::new(80 * 1024);
        let r = Reader::open_cached(
            &root,
            &binding(),
            memory.clone(),
            ReadCacheConfig {
                max_bytes: bytes,
                ..ReadCacheConfig::default()
            },
        )
        .unwrap();
        let mut b = Budget::new(Limits::default());
        // The valid cell at native x=9 has no page. Only the neighborhood output
        // workspace fits, not an additional page-copy/decode reservation.
        let p = point(&g, 9, 1.).position_xyz;
        let empty = done(r.neighbors(p, 0.1, 1, &mut b).unwrap());
        assert!(empty.points.is_empty() && !empty.truncated);
        assert!(b.stop.is_none());
        assert!(memory.high_water() <= 80 * 1024);
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn partitioned_queries_filter_before_caps_and_charge_excluded_points() {
    let root = dir("partitioned");
    let g = geometry();
    build(&root, g.clone(), false);
    let r = Reader::open(&root, &binding(), MemoryBudget::new(512 * 1024)).unwrap();
    let p = point(&g, 3, 1.).position_xyz;
    let mut a = Budget::new(Limits::default());
    let mut b = Budget::new(Limits::default());
    let plain = done(r.neighbors(p, 3., 4, &mut a).unwrap());
    let same = done(
        r.neighbors_by(p, 3., 1, 4, |_, d| Some((0, d)), &mut b)
            .unwrap(),
    );
    assert_eq!(plain.points, same.points);
    assert_eq!(plain.partition_counts, same.partition_counts);
    assert_eq!(
        serde_json::to_value(a.stats).unwrap(),
        serde_json::to_value(b.stats).unwrap()
    );
    let mut b = Budget::new(Limits::default());
    let selected = done(
        r.neighbors_by(
            p,
            3.,
            2,
            1,
            |p, _| Some(((p.feature.at_zyx[2] >= 4) as usize, -p.feature.weight)),
            &mut b,
        )
        .unwrap(),
    );
    assert_eq!(
        selected
            .points
            .iter()
            .map(|p| p.feature.at_zyx[2])
            .collect::<Vec<_>>(),
        vec![3, 6]
    );
    assert_eq!(selected.radius_points, 4);
    assert_eq!(selected.matching_points, 4);
    assert_eq!(selected.partition_counts, vec![2, 2]);
    assert!(selected.truncated);
    let selected = done(
        r.neighbors_by(
            p,
            3.,
            1,
            1,
            |p, d| (p.feature.at_zyx[2] >= 5).then_some((0, d)),
            &mut Budget::new(Limits::default()),
        )
        .unwrap(),
    );
    assert_eq!(selected.points[0].feature.at_zyx[2], 5);
    assert_eq!(selected.radius_points, 4);
    assert_eq!(selected.matching_points, 2);
    assert!(selected.truncated);
    let mut b = Budget::new(Limits::default());
    let none = done(r.neighbors_by(p, 3., 2, 2, |_, _| None, &mut b).unwrap());
    assert!(none.points.is_empty() && !none.truncated);
    assert_eq!(none.matching_points, 0);
    assert_eq!(none.radius_points, 4);
    assert!(b.stats.work_units > 0);
    for (bin, score) in [(2, 1.), (0, f64::NAN)] {
        assert!(r
            .neighbors_by(
                p,
                3.,
                2,
                1,
                |_, _| Some((bin, score)),
                &mut Budget::new(Limits::default())
            )
            .is_err());
    }
    let calls = std::cell::Cell::new(0);
    let mut b = Budget::new(Limits {
        max_work_units: 0,
        ..Limits::default()
    });
    assert!(matches!(
        r.neighbors_by(
            p,
            3.,
            2,
            1,
            |_, _| {
                calls.set(calls.get() + 1);
                None
            },
            &mut b
        )
        .unwrap(),
        Outcome::Stopped(Stop::Work)
    ));
    assert_eq!(calls.get(), 0);
    assert_eq!(b.stats.work_units, 0);
    std::fs::remove_dir_all(root).unwrap();
}

fn packed_build(root: &std::path::Path) -> (Metadata, Stats) {
    let g = geometry();
    let mut w = Writer::create_with_layout(
        root,
        g.clone(),
        binding(),
        2,
        MemoryBudget::new(256 * 1024),
        PageLayout::Packed,
    )
    .unwrap();
    let mut b = Budget::new(Limits::default());
    done(
        w.write_cell(
            [0, 0, 0],
            [
                point(&g, 2, 1.),
                point(&g, 3, 3.),
                point(&g, 2, 2.),
                point(&g, 1, 1.),
            ],
            &mut b,
        )
        .unwrap(),
    );
    done(
        w.write_cell([0, 0, 1], [point(&g, 6, 2.), point(&g, 5, 1.)], &mut b)
            .unwrap(),
    );
    (w.finish().unwrap(), b.stats)
}
#[test]
fn packed_payload_is_exact_legacy_pages_and_queries_preserve_keys_order_and_cursors() {
    let legacy = dir("packed-legacy");
    let packed = dir("packed-v2");
    let a = build(&legacy, geometry(), false);
    let (b, _) = packed_build(&packed);
    let mut am = serde_json::to_value(a).unwrap();
    let bm = serde_json::to_value(b).unwrap();
    am["version"] = serde_json::json!(2);
    assert_eq!(am, bm);
    assert_eq!(
        std::fs::read(legacy.join("cells.bin")).unwrap(),
        std::fs::read(packed.join("cells.bin")).unwrap()
    );
    let offsets = std::fs::read(packed.join("page_offsets.bin")).unwrap();
    let payload = std::fs::read(packed.join("pages.bin")).unwrap();
    assert_eq!(offsets.len(), 32);
    let mut end = 0;
    for i in 0..2 {
        let start = u64::from_le_bytes(offsets[i * 16..i * 16 + 8].try_into().unwrap()) as usize;
        let len = u64::from_le_bytes(offsets[i * 16 + 8..i * 16 + 16].try_into().unwrap()) as usize;
        assert_eq!(start, end);
        end = start + len;
        assert_eq!(
            &payload[start..end],
            std::fs::read(legacy.join(format!("pages/0/0/{i}.rows"))).unwrap()
        );
    }
    assert_eq!(end, payload.len());
    for cache_bytes in [0, 64 * 1024] {
        let mut readers = [&legacy, &packed].map(|p| {
            Reader::open_cached(
                p,
                &binding(),
                MemoryBudget::new(512 * 1024),
                ReadCacheConfig {
                    max_bytes: cache_bytes,
                    ..Default::default()
                },
            )
            .unwrap()
        });
        let mut results = vec![];
        for r in &mut readers {
            let mut budget = Budget::new(Limits::default());
            assert_eq!(done(r.next_cell(&mut budget).unwrap()), Some([0, 0, 0]));
            let q = done(
                r.neighbors(point(&geometry(), 3, 1.).position_xyz, 3., 3, &mut budget)
                    .unwrap(),
            );
            let parts = done(
                r.neighbors_by(
                    point(&geometry(), 3, 1.).position_xyz,
                    3.,
                    2,
                    1,
                    |p, _| Some(((p.feature.at_zyx[2] >= 4) as usize, -p.feature.weight)),
                    &mut budget,
                )
                .unwrap(),
            );
            assert_eq!(done(r.next_cell(&mut budget).unwrap()), Some([0, 0, 1]));
            assert_eq!(done(r.next_cell(&mut budget).unwrap()), None);
            if cache_bytes > 0 {
                let mut warm = Budget::new(Limits {
                    max_read_calls: 0,
                    ..Default::default()
                });
                let repeated = done(
                    r.neighbors(point(&geometry(), 3, 1.).position_xyz, 3., 3, &mut warm)
                        .unwrap(),
                );
                assert_eq!(q.points, repeated.points);
                assert_eq!(warm.stats.read_calls, 0);
                assert!(r.cache_stats().resident_bytes <= cache_bytes);
                r.clear_cache();
                assert_eq!(r.cache_stats().resident_bytes, 0);
            }
            results.push((
                q.points,
                q.partition_counts,
                q.matching_points,
                q.truncated,
                parts.points,
                parts.partition_counts,
            ));
        }
        assert_eq!(results[0], results[1]);
    }
    std::fs::remove_dir_all(legacy).unwrap();
    std::fs::remove_dir_all(packed).unwrap();
}
#[test]
fn packed_offsets_consume_explicit_read_credits_and_stop_before_page_decode() {
    let legacy = dir("packed-ledger-legacy");
    let packed = dir("packed-ledger");
    build(&legacy, geometry(), false);
    let (_, writes) = packed_build(&packed);
    let mut ledgers = vec![];
    for p in [&legacy, &packed] {
        let r = Reader::open(p, &binding(), MemoryBudget::new(256 * 1024)).unwrap();
        let mut b = Budget::new(Limits::default());
        done(r.read_cell([0, 0, 0], &mut b).unwrap());
        ledgers.push(b.stats);
    }
    assert_eq!(ledgers[1].read_calls, ledgers[0].read_calls + 1);
    assert_eq!(
        ledgers[1].charged_bytes_read,
        ledgers[0].charged_bytes_read + 16
    );
    assert_eq!(ledgers[1].page_read_calls, ledgers[0].page_read_calls);
    assert_eq!(ledgers[1].work_units, ledgers[0].work_units);
    let expected_write = std::fs::metadata(packed.join("pages.bin")).unwrap().len() + 48 + 32;
    assert_eq!(writes.charged_bytes_written, expected_write);
    assert_eq!(writes.write_calls, 2);
    for limits in [
        Limits {
            max_read_calls: 2,
            ..Default::default()
        },
        Limits {
            max_bytes_read: 63,
            ..Default::default()
        },
    ] {
        let r = Reader::open(&packed, &binding(), MemoryBudget::new(256 * 1024)).unwrap();
        let mut b = Budget::new(limits);
        assert!(matches!(
            r.read_cell([0, 0, 0], &mut b).unwrap(),
            Outcome::Stopped(_)
        ));
        assert_eq!(b.stats.page_decodes, 0);
        assert_eq!(b.stats.page_read_calls, 0);
    }
    let memory = MemoryBudget::new(1);
    let r = Reader::open(&packed, &binding(), memory.clone()).unwrap();
    let mut b = Budget::new(Limits::default());
    assert!(matches!(
        r.read_cell([0, 0, 0], &mut b).unwrap(),
        Outcome::Stopped(Stop::Memory)
    ));
    assert_eq!(b.stats.page_decodes, 0);
    assert_eq!(memory.high_water(), 0);
    std::fs::remove_dir_all(legacy).unwrap();
    std::fs::remove_dir_all(packed).unwrap();
}
#[test]
fn packed_partial_budgeted_and_failed_publication_remain_unsealed() {
    for kind in 0..3 {
        let p = dir("packed-unsealed");
        let g = geometry();
        let memory = MemoryBudget::new(256 * 1024);
        let mut w = Writer::create_with_layout(
            &p,
            g.clone(),
            binding(),
            2,
            memory.clone(),
            PageLayout::Packed,
        )
        .unwrap();
        assert!(Reader::open(&p, &binding(), memory.clone()).is_err());
        if kind == 0 {
            drop(w);
        } else if kind == 1 {
            let mut b = Budget::new(Limits {
                max_bytes_written: 1,
                ..Default::default()
            });
            assert!(matches!(
                w.write_cell([0, 0, 0], [point(&g, 1, 1.)], &mut b).unwrap(),
                Outcome::Stopped(_)
            ));
            assert!(w.finish().is_err());
            assert_eq!(std::fs::metadata(p.join("pages.bin")).unwrap().len(), 0);
        } else {
            done(
                w.write_cell(
                    [0, 0, 0],
                    [point(&g, 1, 1.)],
                    &mut Budget::new(Limits::default()),
                )
                .unwrap(),
            );
            std::fs::write(p.join("manifest.partial"), b"publication blocked").unwrap();
            assert!(w.finish().is_err());
        }
        assert!(Reader::open(&p, &binding(), memory).is_err());
        std::fs::remove_dir_all(p).unwrap();
    }
    let p = dir("packed-empty");
    let m = MemoryBudget::new(256 * 1024);
    Writer::create_with_layout(&p, geometry(), binding(), 2, m.clone(), PageLayout::Packed)
        .unwrap()
        .finish()
        .unwrap();
    let mut r = Reader::open(&p, &binding(), m).unwrap();
    assert_eq!(
        done(r.next_cell(&mut Budget::new(Limits::default())).unwrap()),
        None
    );
    std::fs::remove_dir_all(p).unwrap();
}
#[test]
fn packed_corrupt_ranges_truncation_and_wrong_cell_pages_are_errors() {
    for kind in 0..7 {
        let p = dir("packed-corrupt");
        packed_build(&p);
        match kind {
            0 => {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(p.join("page_offsets.bin"))
                    .unwrap()
                    .set_len(31)
                    .unwrap();
            }
            1 => {
                std::fs::remove_file(p.join("pages.bin")).unwrap();
            }
            2 => {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(p.join("pages.bin"))
                    .unwrap()
                    .set_len(1)
                    .unwrap();
            }
            3 | 4 => {
                let mut data = std::fs::read(p.join("page_offsets.bin")).unwrap();
                data[..8].copy_from_slice(&u64::MAX.to_le_bytes());
                if kind == 4 {
                    data[8..16].copy_from_slice(&0u64.to_le_bytes());
                }
                std::fs::write(p.join("page_offsets.bin"), data).unwrap();
            }
            5 => {
                let mut data = std::fs::read(p.join("page_offsets.bin")).unwrap();
                let first = data[..16].to_vec();
                data[16..].copy_from_slice(&first);
                std::fs::write(p.join("page_offsets.bin"), data).unwrap();
            }
            _ => {
                let mut data = std::fs::read(p.join("pages.bin")).unwrap();
                data[0] ^= 255;
                std::fs::write(p.join("pages.bin"), data).unwrap();
            }
        }
        match Reader::open(&p, &binding(), MemoryBudget::new(256 * 1024)) {
            Err(_) => assert!(kind <= 1),
            Ok(r) => assert!(
                r.read_cell(
                    if kind == 5 { [0, 0, 1] } else { [0, 0, 0] },
                    &mut Budget::new(Limits::default())
                )
                .is_err(),
                "kind {kind}"
            ),
        }
        std::fs::remove_dir_all(p).unwrap();
    }
}
