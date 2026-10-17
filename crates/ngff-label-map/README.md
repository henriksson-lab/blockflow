# ngff-label-map

Editable id transform layers for NGFF label images: a label image stores
*fragment* ids, and a durable, undoable map turns them into the *segment* ids
that are drawn, measured and exported.

The motivating case is EM proofreading. An automatic pipeline writes
supervoxels (fragments) and an initial grouping (a multicut). Proofreading
edits only the grouping: a merge rewrites the map entries of the smaller
segment, never voxels, and every edit can be undone. The same mechanism fits
track-id relabelling, atlas hierarchy depth or id -> class tables, which are
other `IdTransform`s.

The core has no dependencies beyond `serde`/`serde_json` and builds for
`wasm32-unknown-unknown` with `--no-default-features`. The default `fs` feature
adds the directory store.

## API

| item | purpose |
|---|---|
| `SegmentMap` | fragment -> segment. Dense table for compact ids, hash map otherwise (`Layout`). Unmapped ids map to themselves; `0 -> 0`; `u64::MAX` is reserved. |
| `SegmentMap::plan_merge(a, b)` / `merge` | small-to-large merge: the segment with fewer fragments takes the other's id; returns the survivor |
| `SegmentMap::plan_assign(fragments, s)` / `assign` | put fragments into `s` (a new id, or `0` for background). A split is an assign of one side of a cut. |
| `fragments_of(s)`, `segments_of(&[f])`, `segment_size(s)` | queries; the reverse index is built on first use |
| `with_complete(true)` | declares that every fragment has an entry (a pipeline's full grouping), so a segment id that is not also a fragment id never counts as its own fragment |
| `SegmentMapState` | map + edit log; `plan_*` then `commit`, `undo`, `undo_target`, `revision` |
| `IdTransform`, `Chain`, `FnTransform` | ids -> ids, composable |
| `remap_ids_in_place(ids, &[&dyn IdTransform])` | per-tile applier: evaluates the chain once per distinct id (run cache + local open-addressed table) |
| `snapshot::{encode, decode, decode_header}` | binary snapshot codec |
| `store::SegmentMapStore` (`fs`) | `create`, `create_if_missing`, `copy_if_missing`, `load`, `refresh`, `merge`/`assign`/`undo`/`edit` with an expected revision, `compact` |
| `store::{discover, write_attribute, create_for_label}` (`fs`) | metadata discovery and publication |

## On-disk format, version 1

```text
labels/<layer>/zarr.json            attributes.segment_map = {"version": 1, "path": "segment-map"}
labels/<layer>/segment-map/
    snapshot.bin                    the map at revision R
    edits.jsonl                     all edit records, revisions 1..N, one JSON object per line
labels/<layer>/.segment-map.lock    writer lock (sibling of the store; created on first edit)
```

`path` is relative to the label group and may not leave it. For a Zarr v2
label without `zarr.json`, the same `segment_map` key sits at the top level of
`.zattrs`.

The map at revision N is `snapshot.bin` (at revision R <= N) with records R+1..N
replayed. Revision = number of records. History is never rewritten: undo
appends the inverse of the latest edit that is not yet undone.

### `snapshot.bin`

Little-endian; `u64` unless noted.

| offset | field |
|---|---|
| 0 | magic `NGFFSMAP` (8 bytes) |
| 8 | version `u32` = 1 |
| 12 | flags `u32`: bit 0 = complete |
| 16 | revision (records already folded in) |
| 24 | generation (lineage id: new on create, kept by compaction) |
| 32 | base: fragment id of dense slot 0 |
| 40 | dense_count |
| 48 | pair_count |
| 56 | dense_count segment ids, slot i = fragment base+i, `u64::MAX` = no entry |
| ... | pair_count `(fragment, segment)` pairs sorted by fragment, outside the dense range |

A multicut result over fragments `1..=N` is the dense form with `base = 1`,
`pair_count = 0`, flag `complete`, revision 0 — `N * 8 + 56` bytes. From numpy:

```python
header = b"NGFFSMAP" + struct.pack("<IIQQQQQ", 1, 1, 0, generation, 1, len(seg), 0)
open("snapshot.bin", "wb").write(header + seg.astype("<u8").tobytes())
open("edits.jsonl", "wb").close()
```

### `edits.jsonl`

```json
{"revision":3,"op":"merge","segments":[12,40],"moves":[[40,40,12],[41,40,12]],"time_ms":1760000000000,"author":"jo"}
{"revision":4,"op":"undo","segments":[12,40],"undoes":3,"moves":[[41,12,40],[40,12,40]],"time_ms":1760000001000}
```

`op` is `merge` (`segments = [survivor, absorbed]`), `assign`
(`segments = [target]`) or `undo` (`undoes` = reverted revision). `moves` are
`[fragment, from, to]` applied in order; `from` must equal the fragment's
current segment, so a replay that does not match is reported as corrupt.

### Concurrency and durability

Writers take the lock (`File::lock`), re-read the log, refuse with
`Error::Conflict { expected, current }` unless the store is at the caller's
revision (servers answer HTTP 409), append one line and `fsync`. The snapshot
is rewritten every 64 edits by temp file + rename. A store is created in a
staging directory and renamed into place. Readers take no lock; a torn final
line from a crashed writer is ignored and dropped by the next writer.

## Publishing from a pipeline

```rust
let map = ngff_label_map::SegmentMap::from_dense(1, segment_of_fragment)?.with_complete(true);
ngff_label_map::store::create_for_label(&dataset.join("labels/fragments"), &map)?;
```

## Tests

```bash
cargo test --release -p ngff-label-map
cargo build --release -p ngff-label-map --target wasm32-unknown-unknown --no-default-features
```
