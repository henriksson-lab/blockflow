# Physical plane sampling

Sample a prepared Zarr image and its separate support array on an oriented
physical plane, using blockflow's planned source footprints. Sampling runs use
`ZarrEnvironment::attach`; only the separate fixture-preparation command holds
small analytic arrays in memory.

From the blockflow workspace:

```bash
cargo run --release -p blockflow-physical-plane-sampling -- prepare --out /tmp/plane-fixture
cargo run --release -p blockflow-physical-plane-sampling -- sample --config /tmp/plane-fixture/plane.json --out /tmp/plane-output
```

Use a fresh output directory. Edit the prepared JSON to point at real rank-3
Zarr arrays and specify their physical frame, plane, signed depth, row/column
spacing and block sizes. Paths identify arrays (the directory with `zarr.json`),
not OME image groups. Intensity and support must already share the configured
physical grid. Support should contain bool or numeric 0/1 values. A pyramid
level needs its own actual origin and voxel spacing; these are explicit inputs,
not inferred from a level number or assumed to be micrometers.

Each run writes `intensity/level1` and `support/level1`, both f64, plus
`sampling.json` with the complete plane frame, input paths, output shape,
maximum source fetch per block and measured execution times. Output axes are
`[singleton,row,column]`; the JSON carries the oblique physical embedding.
These outputs are not mislabeled as axis-aligned global z,y,x OME images.

Out-of-domain and nonfinite contributing image values yield NaN. Finite image
samples remain independent of the support value; no mask threshold, support
normalization, depth averaging or registration acceptance is supplied. To obtain
several depths, run this command for each depth into a separate output directory.
All distances share the configuration's named unit, and positive depth follows
the supplied normal. Voxel indices denote centers, including at boundaries.

Software validation: `cargo test --release -p blockflow --test physical_plane`
compares the resident kernel with the actual block executor on several grids,
with oblique/anisotropic analytic fields, bool support, invalid values and
bounds. This example is a sampling demonstration, not tissue alignment validation.
