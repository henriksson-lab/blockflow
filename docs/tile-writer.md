# Native images written one tile at a time

`zarr_env::TileWriter<T>` is a single-level OME-Zarr writer for callers whose
image arrives as bounded chunks rather than full-width slabs. Existing
`MultiscaleWriter` behavior and pyramid rules are unchanged.

Create a fresh root with shape/chunk in ZYX, optional `PhysicalFrame` in ZYX,
and an explicit compression choice. `next_tile()` returns the expected global
start and clipped shape. Supply exactly its row-major elements to `write_next`.
Tiles progress in natural ZYX chunk order. No whole-image, row or tile-occupancy
table is retained; shape/byte arithmetic is checked. Root creation is exclusive
and never replaces an existing image.

Every write uses the same native typed encoder as `ZarrEnvironment`. The caller
owns its tile and accounts for encoder/copy workspace and metadata; this helper
does not establish a process RSS or codec workspace bound. It does not resample,
convert dtype or build a pyramid. Different chunk layouts therefore preserve
exact supplied values, including floating NaNs and integer masks.

`finish` writes the existing calibrated single-level multiscales metadata only
after every tile succeeds. The `0/zarr.json` array metadata exists earlier;
do not publish or interpret a partially written array as a complete image.
An IO write failure poisons the writer, and an incomplete/poisoned writer cannot
finish. Bad payload lengths are rejected before IO. Failed directories remain
for diagnosis; resume is not implemented. Finish is not an fsync/power-loss
durability guarantee. Multi-image bundles need their own completion protocol.

The targeted tests in `tests/tile_writer.rs` check clipped-edge native float64
values, physical metadata, refusal to overwrite, incomplete publication,
invalid/overflowing geometry and poisoning after an obstructed backend write.
