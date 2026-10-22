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
