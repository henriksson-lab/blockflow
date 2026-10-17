use std::fs;
use std::io::Write;
use std::sync::{Arc, Barrier};

use ngff_label_map::store::{self, SegmentMapStore, EDITS_FILE, SNAPSHOT_FILE};
use ngff_label_map::{EditMeta, EditOp, Error, Layout, SegmentMap};

fn initial() -> SegmentMap {
    // Fragments 1..=12 grouped in threes: segments 101, 102, 103, 104.
    SegmentMap::from_pairs((1..=12).map(|f| (f, 100 + (f - 1) / 3 + 1))).unwrap()
}

fn label_dir(root: &std::path::Path) -> std::path::PathBuf {
    let dir = root.join("labels/frag");
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"group","attributes":{"ome":{"version":"0.5"}}}"#,
    )
    .unwrap();
    dir
}

#[test]
fn create_discover_merge_undo_and_reload() {
    let temp = tempfile::tempdir().unwrap();
    let label = label_dir(temp.path());
    assert!(store::discover(&label).unwrap().is_none());
    store::create_for_label(&label, &initial()).unwrap();
    let metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(label.join("zarr.json")).unwrap()).unwrap();
    assert_eq!(
        metadata["attributes"]["ome"]["version"], "0.5",
        "other keys survive"
    );
    assert_eq!(
        metadata["attributes"]["segment_map"],
        serde_json::json!({"version": 1, "path": "segment-map"})
    );
    let store = store::discover(&label).unwrap().expect("declared");
    let mut loaded = store.load().unwrap();
    assert_eq!(loaded.revision(), 0);
    let merge = store
        .merge(
            &mut loaded,
            0,
            101,
            102,
            EditMeta {
                author: Some("t".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(merge.revision, 1);
    assert_eq!(merge.segments, vec![101, 102]);
    assert_eq!(loaded.map().get(4), 101);
    store
        .assign(&mut loaded, 1, &[7, 8], 900, EditMeta::default())
        .unwrap();
    store.undo(&mut loaded, 2, EditMeta::default()).unwrap();
    assert_eq!(loaded.map().get(7), 103);

    let reloaded = store.load().unwrap();
    assert_eq!(reloaded.revision(), 3);
    assert_eq!(reloaded.map(), loaded.map());
    assert_eq!(reloaded.state().undo_target(), Some(1));
    let ops: Vec<_> = reloaded.state().records().iter().map(|r| r.op).collect();
    assert_eq!(ops, [EditOp::Merge, EditOp::Assign, EditOp::Undo]);
    assert_eq!(reloaded.state().records()[0].author.as_deref(), Some("t"));
}

#[test]
fn snapshot_compaction_and_replay_give_the_same_map() {
    let temp = tempfile::tempdir().unwrap();
    let store = SegmentMapStore::create(temp.path().join("m"), &initial())
        .unwrap()
        .with_snapshot_interval(2);
    let mut loaded = store.load().unwrap();
    let generation = loaded.generation();
    store
        .merge(&mut loaded, 0, 101, 102, EditMeta::default())
        .unwrap();
    store
        .merge(&mut loaded, 1, 103, 104, EditMeta::default())
        .unwrap();
    assert_eq!(loaded.snapshot_revision(), 2, "compacted after two edits");
    store.undo(&mut loaded, 2, EditMeta::default()).unwrap();
    let reloaded = store.load().unwrap();
    assert_eq!(reloaded.snapshot_revision(), 2);
    assert_eq!(reloaded.revision(), 3);
    assert_eq!(reloaded.map(), loaded.map());
    assert_eq!(
        reloaded.generation(),
        generation,
        "compaction keeps the lineage"
    );
    // The log keeps the whole history even though the snapshot moved on.
    let lines = fs::read_to_string(store.dir().join(EDITS_FILE)).unwrap();
    assert_eq!(lines.lines().count(), 3);
    // Undo still reaches edits folded into the snapshot.
    let mut again = reloaded;
    store.undo(&mut again, 3, EditMeta::default()).unwrap();
    assert_eq!(*again.map(), initial());
}

#[test]
fn stale_revision_is_a_conflict_and_changes_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let store = SegmentMapStore::create(temp.path().join("m"), &initial()).unwrap();
    let mut first = store.load().unwrap();
    let mut second = store.load().unwrap();
    store
        .merge(&mut first, 0, 101, 102, EditMeta::default())
        .unwrap();
    let error = store
        .merge(&mut second, 0, 103, 104, EditMeta::default())
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::Conflict {
                expected: 0,
                current: 1
            }
        ),
        "{error}"
    );
    // The refused writer was brought up to date and can retry at the current revision.
    assert_eq!(second.revision(), 1);
    store
        .merge(&mut second, 1, 103, 104, EditMeta::default())
        .unwrap();
    assert!(store.refresh(&mut first).unwrap());
    assert_eq!(first.map(), second.map());
}

#[test]
fn concurrent_writers_at_the_same_revision_commit_exactly_once() {
    let temp = tempfile::tempdir().unwrap();
    let store = SegmentMapStore::create(temp.path().join("m"), &initial()).unwrap();
    let threads = 8;
    let barrier = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|i| {
            let store = store.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut loaded = store.load().unwrap();
                barrier.wait();
                let pair = [(101, 102), (103, 104)][i % 2];
                store.merge(&mut loaded, 0, pair.0, pair.1, EditMeta::default())
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert!(results
        .iter()
        .filter_map(|r| r.as_ref().err())
        .all(|e| matches!(
            e,
            Error::Conflict {
                expected: 0,
                current: 1
            }
        )));
    assert_eq!(store.load().unwrap().revision(), 1);
}

#[test]
fn torn_final_line_is_ignored_then_dropped() {
    let temp = tempfile::tempdir().unwrap();
    let store = SegmentMapStore::create(temp.path().join("m"), &initial()).unwrap();
    let mut loaded = store.load().unwrap();
    store
        .merge(&mut loaded, 0, 101, 102, EditMeta::default())
        .unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(store.dir().join(EDITS_FILE))
        .unwrap()
        .write_all(br#"{"revision":2,"op":"me"#)
        .unwrap();
    let mut fresh = store.load().unwrap();
    assert_eq!(fresh.revision(), 1);
    store
        .merge(&mut fresh, 1, 103, 104, EditMeta::default())
        .unwrap();
    let reloaded = store.load().unwrap();
    assert_eq!(reloaded.revision(), 2);
    assert_eq!(reloaded.map(), fresh.map());
}

#[test]
fn refresh_reads_appended_edits_and_notices_a_new_lineage() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("m");
    let store = SegmentMapStore::create(&dir, &initial()).unwrap();
    let mut reader = store.load().unwrap();
    assert!(!store.refresh(&mut reader).unwrap());
    let mut writer = store.load().unwrap();
    store
        .merge(&mut writer, 0, 101, 104, EditMeta::default())
        .unwrap();
    assert!(store.refresh(&mut reader).unwrap());
    assert_eq!(reader.map(), writer.map());
    // A pipeline re-run replaces the store wholesale.
    fs::remove_dir_all(&dir).unwrap();
    let replacement = SegmentMap::from_pairs([(1, 5)]).unwrap();
    SegmentMapStore::create(&dir, &replacement).unwrap();
    assert!(store.refresh(&mut reader).unwrap());
    assert_eq!(reader.revision(), 0);
    assert_eq!(*reader.map(), replacement);
}

#[test]
fn copy_if_missing_forks_state_and_history() {
    let temp = tempfile::tempdir().unwrap();
    let source = SegmentMapStore::create(temp.path().join("dataset/m"), &initial()).unwrap();
    let mut loaded = source.load().unwrap();
    source
        .merge(&mut loaded, 0, 101, 102, EditMeta::default())
        .unwrap();
    let copy = SegmentMapStore::new(temp.path().join("state/m"));
    assert!(copy.copy_if_missing(&source).unwrap());
    assert!(!copy.copy_if_missing(&source).unwrap());
    let mut forked = copy.load().unwrap();
    assert_eq!(forked.revision(), 1);
    assert_eq!(forked.map(), loaded.map());
    copy.undo(&mut forked, 1, EditMeta::default()).unwrap();
    assert_eq!(*forked.map(), initial());
    assert_eq!(
        source.load().unwrap().revision(),
        1,
        "the source is untouched"
    );
}

#[test]
fn snapshot_files_are_replaced_not_left_behind() {
    let temp = tempfile::tempdir().unwrap();
    let store = SegmentMapStore::create(temp.path().join("m"), &initial())
        .unwrap()
        .with_snapshot_interval(1);
    let mut loaded = store.load().unwrap();
    store
        .merge(&mut loaded, 0, 101, 102, EditMeta::default())
        .unwrap();
    let mut names: Vec<_> = fs::read_dir(store.dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, [EDITS_FILE, SNAPSHOT_FILE]);
    assert!(SegmentMapStore::create(store.dir(), &initial()).is_err());
}

#[test]
fn discovery_rejects_paths_leaving_the_label() {
    let temp = tempfile::tempdir().unwrap();
    let label = label_dir(temp.path());
    for bad in ["../x", "/abs", ""] {
        fs::write(
            label.join("zarr.json"),
            serde_json::to_vec(&serde_json::json!({
                "zarr_format": 3, "node_type": "group",
                "attributes": {"segment_map": {"version": 1, "path": bad}}
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(store::discover(&label).is_err(), "{bad:?}");
    }
}

#[test]
fn sparse_and_dense_stores_behave_the_same() {
    let temp = tempfile::tempdir().unwrap();
    let pairs: Vec<_> = (1..=12).map(|f| (f, 100 + (f - 1) / 3 + 1)).collect();
    let mut maps = Vec::new();
    for (name, layout) in [("d", Layout::Dense), ("s", Layout::Sparse)] {
        let map = SegmentMap::from_pairs_with(pairs.iter().copied(), layout).unwrap();
        let store = SegmentMapStore::create(temp.path().join(name), &map).unwrap();
        let mut loaded = store.load().unwrap();
        assert_eq!(loaded.map().is_dense(), layout == Layout::Dense);
        store
            .merge(&mut loaded, 0, 101, 103, EditMeta::default())
            .unwrap();
        store
            .assign(&mut loaded, 1, &[2, 50], 7, EditMeta::default())
            .unwrap();
        maps.push(store.load().unwrap().map().clone());
    }
    assert_eq!(maps[0], maps[1]);
}
