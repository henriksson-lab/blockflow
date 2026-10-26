//! A bounded block adapter to the existing native pyramid decimator.
use super::downsample_region;
use crate::{
    error::{Error, Result},
    voxels::{VoxelElement, Voxels},
};
use ndarray::Array3;

/// Downsample a resident input block using the existing pyramid's exact typed
/// rule. Factors/array axes are ZYX. Incomplete far-face windows are dropped;
/// every returned voxel uses a full window. Numeric types use the existing
/// f64 accumulation/type conversion, bool selects the window's first sample.
/// Floating nonfinite values propagate. No boundary padding or validity fill.
/// The caller owns input/output memory and must align global block starts with
/// its decimation lattice; this helper does not infer global calibration.
pub fn downsample_block<T: VoxelElement>(input: &Voxels, factor: [usize; 3]) -> Result<Array3<T>> {
    if factor.contains(&0) {
        return Err(Error::invalid("positive block decimation factors required"));
    }
    let shape = input.shape();
    let output = std::array::from_fn::<_, 3, _>(|i| shape[i] / factor[i]);
    if output.contains(&0) {
        return Err(Error::invalid("decimation block has no full output window"));
    }
    downsample_region::<T>(input, factor, &output)
}
