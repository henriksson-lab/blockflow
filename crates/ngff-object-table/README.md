# ngff-object-table

Reusable reading and writing of typed, spatially indexed object tables stored
inside Zarr v3 datasets.

The crate owns the native table representation shared by producers such as
Blockflow and consumers such as newvolim. It has no dependency on either. A
viewer-specific cache is outside this format and may be discarded or redesigned
without changing the stored table.

The initial representation provides:

- one typed, chunked Zarr array per column;
- fixed storage tiles independent of compute blocks;
- tile start and count arrays for spatial range reads;
- a summed occupancy pyramid for choosing useful spatial-index levels;
- sorted identity and physical-row arrays for stable-ID lookup;
- chunk-aligned incremental writes;
- root completion metadata written only by `TableWriter::finish`;
- partial column and index reads through `TableReader`.

Blockflow's `object_table` module is one producer. It externally sorts fragment
rows into this representation with bounded memory; the storage crate itself has
no dependency on Blockflow's planner or fragment types.

The metadata key is `object_table`. It is a versioned extension rather than a
claim that object tables are part of the current OME-NGFF specification.

## Converting legacy measurement CSVs

The release converter accepts the measurement CSV emitted by the Cellpose and
StarDist examples, sorts it into the native spatial layout, and builds both
indexes and the occupancy pyramid:

```bash
cargo run --release -p ngff-object-table --bin ngff-object-table-convert -- \
  table.csv output.zarr label-name HEIGHT WIDTH
```

Repeated `label_id` rows are combined because one raster label is one object:
areas are summed, centroids and mean intensity are area weighted, and intensity
extrema are merged. Verify any completed table, including every identity lookup,
with:

```bash
cargo run --release -p ngff-object-table --bin ngff-object-table-convert -- \
  --verify output.zarr
```
