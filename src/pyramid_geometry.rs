// SPDX-License-Identifier: MIT
//! Coordinates of pyramid samples, with integer indices at voxel centres.

/// The position represented by a decimation window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyramidSampling {
    /// The centre of the nominal full window (also used for max pooling).
    WindowCentre,
    /// The first input sample; used by Boolean decimation.
    FirstSample,
}

/// Output pitch and first sample position along one axis.
///
/// `factor` can be a single step or a cumulative factor from level zero.
/// A clipped edge window stays on this regular lattice; its nominal centre
/// need not equal the centroid of the available samples. Callers decide the
/// edge extent policy independently. Factors must be positive.
pub fn pyramid_axis_transform(
    origin: f64,
    pitch: f64,
    factor: usize,
    sampling: PyramidSampling,
) -> (f64, f64) {
    assert!(factor > 0, "a pyramid factor must be positive");
    let offset = match sampling {
        PyramidSampling::WindowCentre => (factor as f64 - 1.0) * pitch / 2.0,
        PyramidSampling::FirstSample => 0.0,
    };
    (pitch * factor as f64, origin + offset)
}
