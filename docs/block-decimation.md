# Typed decimation of one bounded block

`zarr_env::downsample_block::<T>(&voxels, factor_zyx)` exposes the existing
native pyramid decimator without constructing a whole image or another writer.
The caller supplies the block and accounts for the returned array. Numeric
types keep the existing f64 accumulation and native conversion; bool keeps the
first sample. Nonfinite floats propagate. Incomplete far-face windows are
dropped; zero factors, empty output and wrong element types fail.

Global input block starts must align with the factor lattice. The caller owns
physical calibration and masks; invalid inputs must be represented consistently
before applying a strict mean. For numeric window means, output pitch is input
pitch times factor and the first centre shifts by (factor-1)*pitch/2. Bool's
first-sample rule has no centre shift. This helper changes no pyramid policy.

The subslice prototype uses it with bounded native tiles and the existing
TileWriter to prepare extra coarse fitting-channel levels.
