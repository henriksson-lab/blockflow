// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The detector's parameters, in the volume's own `[z, y, x]` order and in
//! voxels and raw intensity units.

use blockflow::error::{Error, Result};
use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::watershed::Separation;

/// The two scales of a difference of Gaussians, per axis, in voxels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DogScales {
    /// The narrow scale.
    pub sigma: [f64; 3],
    /// The wide scale; must exceed `sigma` on every axis, or the difference is
    /// the negative of a band-pass.
    pub sigma2: [f64; 3],
    /// Where each kernel is cut, in standard deviations.
    pub truncate: f64,
}

/// Which operator "the maxima" means.
///
/// The two differ on flat regions: a window maximum marks every voxel of a
/// plateau wider than the window (a uniform background included, which the
/// foreground threshold then removes), while a regional maximum marks only the
/// plateaux from which one cannot ascend. With noise inside bright objects the
/// window merges ripples closer than its radius into one seed, where the
/// regional maxima need a prominence (`h`) to do the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maxima {
    /// A voxel equal to the maximum of the box of this radius around it.
    Window { radius: [usize; 3] },
    /// The morphological regional maxima (`ops::regional`).
    Regional,
}

/// How the basins are flooded from the global seeds.
///
/// Both flood the negated filtered volume inside the foreground from the same
/// numbered seeds; they differ in what the answer is a function of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Watershed {
    /// One flood over the whole volume (`ops::watershed::ScalableWatershedOp`).
    /// A planning barrier: the phase holds the cost, mask, seeds and basins of
    /// the whole volume at once and runs on one thread.
    Whole,
    /// One flood per tile over the tile and a halo around it
    /// (`ops::tiled_watershed::TiledWatershedOp`), from the global seeds that
    /// lie in that buffer. The tile is a parameter of the answer (the block
    /// grid is not); the halo bounds how far outside its tile a basin's flood
    /// is followed. A basin part whose seed is beyond the halo is dropped
    /// (label 0) rather than given to a neighbour.
    Tiled { tile: [usize; 3], halo: [usize; 3] },
}

impl Watershed {
    /// The default tiling: `64 x 128 x 128` tiles with a 16-voxel halo, about
    /// three cell diameters. The reference this is compared against cuts
    /// `z` slabs of 50 to 100 planes overlapping by 32, i.e. 16 planes either
    /// side of a seam.
    pub const DEFAULT_TILED: Self = Self::Tiled {
        tile: [64, 128, 128],
        halo: [16, 16, 16],
    };
}

/// A half-open band on one column: kept when `min <= value < max`. `None` is
/// an open edge.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Band {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl Band {
    /// No edge at all.
    pub const OPEN: Self = Self {
        min: None,
        max: None,
    };

    pub const fn new(min: Option<f64>, max: Option<f64>) -> Self {
        Self { min, max }
    }

    pub fn keeps(&self, value: f64) -> bool {
        if self.min.is_some_and(|min| !(value >= min)) {
            return false;
        }
        if self.max.is_some_and(|max| !(value < max)) {
            return false;
        }
        true
    }

    /// Refuse a non-finite edge or an empty band, naming the column.
    pub fn validate(&self, column: &str) -> Result<()> {
        for edge in [self.min, self.max].into_iter().flatten() {
            if !edge.is_finite() {
                return Err(Error::invalid(format!(
                    "the {column} band {self:?} has a non-finite edge; leave the edge open instead"
                )));
            }
        }
        if let (Some(min), Some(max)) = (self.min, self.max) {
            if min >= max {
                return Err(Error::invalid(format!(
                    "the {column} band [{min}, {max}) is empty, so it would drop every cell; the \
                     band is half-open, `min <= value < max`"
                )));
            }
        }
        Ok(())
    }
}

/// Everything the detector decides.
#[derive(Debug, Clone, PartialEq)]
pub struct CellParams {
    /// Which voxels of the background element's bounding box belong to it.
    pub background_form: ElementShape,
    /// The background element's extent, `[z, y, x]`. Flat in `z` by default so
    /// that the background is estimated plane by plane, which keeps the
    /// estimate independent of the (usually coarser) axial sampling.
    pub background_size: [usize; 3],
    /// An optional band-pass after the background removal.
    pub dog: Option<DogScales>,
    /// The prominence below which a peak is flattened before the maxima are
    /// taken, in intensity units. `0` skips the h-maxima transform entirely.
    pub h: f64,
    pub maxima: Maxima,
    /// The foreground level, in the filtered volume's units: a maximum seeds a
    /// cell only where the filtered value is above it, and a basin grows only
    /// through voxels above it. One number for both, so that a seed can never
    /// sit outside the region its basin may grow in.
    pub threshold: f64,
    /// Whether a one-voxel line is left where two basins meet.
    pub separation: Separation,
    /// Whole-volume or tiled flood; the tile is part of the answer.
    pub watershed: Watershed,
    /// The band on `size` (basin voxels) that decides `passes_filter`.
    pub size: Band,
    /// The band on `source` (the raw maximum over the basin).
    pub intensity: Band,
}

impl Default for CellParams {
    /// Defaults for nuclear or cFos staining in 16-bit light-sheet data at
    /// roughly 2-6 µm voxels: a 10-voxel disk per plane for the background, a
    /// window of five for the maxima, a foreground of 500 counts above the
    /// background, and cells of 50 to 900 voxels. The foreground follows the
    /// reference's shipped cell-map configuration (500; 700 is only its code
    /// default). The defaults were checked against a recorded run on a
    /// 384 x 384 x 256 crop of a cleared mouse brain, at a foreground of 700
    /// (see the README). Volumes with other units, magnifications or labels
    /// need their own `threshold` and `size`.
    fn default() -> Self {
        Self {
            background_form: ElementShape::ExtentEllipsoid,
            background_size: [1, 10, 10],
            dog: None,
            h: 0.0,
            maxima: Maxima::Window { radius: [2, 2, 2] },
            threshold: 500.0,
            separation: Separation::Line,
            watershed: Watershed::DEFAULT_TILED,
            size: Band::new(Some(50.0), Some(900.0)),
            intensity: Band::OPEN,
        }
    }
}

impl CellParams {
    pub fn validate(&self) -> Result<()> {
        if !self.h.is_finite() || self.h < 0.0 {
            return Err(Error::invalid(format!(
                "the h-maxima prominence is {}; it must be finite and non-negative",
                self.h
            )));
        }
        if !self.threshold.is_finite() {
            return Err(Error::invalid(format!(
                "the foreground threshold is {}; it must be finite",
                self.threshold
            )));
        }
        if self.background_size.contains(&0) {
            return Err(Error::invalid(format!(
                "the background element {:?} has an empty axis; use 1 for a flat axis",
                self.background_size
            )));
        }
        if let Some(scales) = &self.dog {
            for axis in 0..3 {
                if !(scales.sigma[axis] > 0.0) || !(scales.sigma2[axis] > scales.sigma[axis]) {
                    return Err(Error::invalid(format!(
                        "the difference of Gaussians has sigma {:?} and sigma2 {:?}; on axis \
                         {axis} the outer scale does not exceed the inner one",
                        scales.sigma, scales.sigma2
                    )));
                }
            }
            if !(scales.truncate > 0.0) || !scales.truncate.is_finite() {
                return Err(Error::invalid(format!(
                    "the difference of Gaussians is truncated at {} sigma; it must be positive",
                    scales.truncate
                )));
            }
        }
        if self.maxima == (Maxima::Window { radius: [0, 0, 0] }) {
            return Err(Error::invalid(
                "a maxima window of one voxel marks every voxel as a maximum",
            ));
        }
        if let Watershed::Tiled { tile, halo } = self.watershed {
            for axis in 0..3 {
                if tile[axis] == 0 || halo[axis] >= tile[axis] {
                    return Err(Error::invalid(format!(
                        "the watershed tile {tile:?} with halo {halo:?}: every tile axis must be \
                         positive and larger than the halo on that axis"
                    )));
                }
            }
        }
        self.size.validate("size")?;
        self.intensity.validate("intensity")
    }

    /// The background element these parameters name.
    pub fn background_element(&self) -> Result<StructuringElement> {
        StructuringElement::from_size(self.background_form, self.background_size)
    }

    /// Whether a row with this size and source intensity passes both bands.
    pub fn passes(&self, size: f64, source: f64) -> bool {
        self.size.keeps(size) && self.intensity.keeps(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_band_is_half_open() {
        let band = Band::new(Some(20.0), Some(120.0));
        assert!(!band.keeps(19.999));
        assert!(band.keeps(20.0));
        assert!(band.keeps(119.999));
        assert!(!band.keeps(120.0));
        assert!(!band.keeps(f64::NAN));
        assert!(Band::OPEN.keeps(0.0));
        assert!(Band::new(Some(20.0), None).keeps(f64::MAX));
    }

    #[test]
    fn an_empty_band_is_refused() {
        let params = CellParams {
            size: Band::new(Some(30.0), Some(30.0)),
            ..CellParams::default()
        };
        let failed = params.validate().unwrap_err();
        assert!(failed.to_string().contains("half-open"), "{failed}");
    }

    #[test]
    fn an_inverted_band_pass_is_refused() {
        let params = CellParams {
            dog: Some(DogScales {
                sigma: [2.0; 3],
                sigma2: [1.0; 3],
                truncate: 4.0,
            }),
            ..CellParams::default()
        };
        assert!(params
            .validate()
            .unwrap_err()
            .to_string()
            .contains("outer scale"));
    }

    #[test]
    fn the_defaults_are_valid() {
        CellParams::default().validate().unwrap();
        assert_eq!(
            CellParams::default().background_element().unwrap().len(),
            80,
            "a 10 x 10 disk per plane"
        );
    }
}
