//! Per-bead measurements against the layer surfaces.

use std::io::Write;
use std::path::Path;

use blockflow::{Error, Result};
use ndarray::Array2;

/// Surfaces recovered from the region labels: first z of each region per column.
pub struct HeightMaps {
    /// Mucus top (first z labelled mucus or tissue).
    pub mucus: Array2<f32>,
    /// Epithelium apical surface (first z labelled tissue).
    pub epi: Array2<f32>,
    /// Where `[0, 0]` of the maps sits in the volume's (y, x): the maps may
    /// cover only a window of it.
    pub origin: [usize; 2],
    /// The volume's (y, x) extent.
    pub extent: [usize; 2],
}

/// Height maps at voxel resolution from the solved surface columns: the same
/// slices the region labels change at.
pub fn heights_from_columns(
    surfaces: &blockflow::ops::surfaces::SurfaceColumns,
    ny: usize,
    nx: usize,
) -> HeightMaps {
    heights_window(surfaces, [0, 0], [ny, nx], [ny, nx])
}

/// The same over the window of `shape` at `origin` of a volume `extent` wide.
pub fn heights_window(
    surfaces: &blockflow::ops::surfaces::SurfaceColumns,
    origin: [usize; 2],
    shape: [usize; 2],
    extent: [usize; 2],
) -> HeightMaps {
    HeightMaps {
        mucus: Array2::from_shape_fn((shape[0], shape[1]), |(y, x)| {
            surfaces.slice_at(0, origin[0] + y, origin[1] + x)
        }),
        epi: Array2::from_shape_fn((shape[0], shape[1]), |(y, x)| {
            surfaces.slice_at(1, origin[0] + y, origin[1] + x)
        }),
        origin,
        extent,
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: u64,
    pub z: f64,
    pub y: f64,
    pub x: f64,
    pub response: f64,
    pub raw_intensity: f64,
    /// Background-subtracted mean of the z neighbours over the peak: ~0 for a
    /// single-voxel spike, ~0.7+ for a PSF-blurred bead.
    pub z_support: f64,
    /// Same for the 4 xy neighbours.
    pub xy_support: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Bead,
    /// Below the apical surface: C-Blue also stains the epithelium itself.
    InTissue,
    /// Too dim relative to the bead population.
    Dim,
    /// Not PSF-shaped: a single-voxel noise spike or hot pixel.
    Spike,
    /// Inside a large AF488 object (probably a dying cell) — ignored.
    InLargeObject,
}

impl Class {
    /// Code stored in the object table's `class` column.
    pub fn code(self) -> u8 {
        match self {
            Class::Bead => 0,
            Class::InTissue => 1,
            Class::Dim => 2,
            Class::Spike => 3,
            Class::InLargeObject => 4,
        }
    }

    pub const ALL: [Class; 5] = [
        Class::Bead,
        Class::InTissue,
        Class::Dim,
        Class::Spike,
        Class::InLargeObject,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Class::Bead => "bead",
            Class::InTissue => "in_tissue",
            Class::Dim => "dim",
            Class::Spike => "spike",
            Class::InLargeObject => "in_large_object",
        }
    }
}

/// A person's decision from viewer review, overriding the automatic class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Manual {
    None,
    Accepted,
    Rejected,
}

impl Manual {
    pub fn code(self) -> u8 {
        match self {
            Manual::None => 0,
            Manual::Accepted => 1,
            Manual::Rejected => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Manual::None => "",
            Manual::Accepted => "accepted",
            Manual::Rejected => "rejected",
        }
    }
}

pub struct Measured {
    pub c: Candidate,
    /// Automatic class.
    pub class: Class,
    pub manual: Manual,
    pub h_mucus_um: f64,
    pub h_epi_um: f64,
    /// Positive above the apical surface.
    pub vertical_um: f64,
    /// Signed 3D distance to the nearest apical-surface point.
    pub euclidean_um: f64,
    pub mucus_thickness_um: f64,
    /// 0 at the mucus top, 1 at the epithelium.
    pub relative_depth: f64,
    /// Id of the large AF488 object the candidate sits in, or 0.
    pub large_object: u32,
}

impl Measured {
    /// Final decision: the manual one if any, else the automatic class.
    pub fn kept(&self) -> bool {
        match self.manual {
            Manual::Accepted => true,
            Manual::Rejected => false,
            Manual::None => self.class == Class::Bead,
        }
    }
}

pub struct MeasureParams {
    pub voxel: [f64; 3],
    /// Candidates more than this far below the apical surface are `InTissue`.
    pub tissue_margin_um: f64,
    pub min_z_support: f64,
    pub min_xy_support: f64,
}

/// Distances to the surfaces and the automatic class before the brightness
/// split (`Dim`) and the large-object flag, which need every candidate.
pub fn measure_one(c: &Candidate, h: &HeightMaps, p: &MeasureParams) -> Measured {
    let [ny, nx] = h.extent;
    let [oy, ox] = h.origin;
    let (wy, wx) = h.epi.dim();
    let [dz, dy, dx] = p.voxel;
    let yi = (c.y.round() as usize).min(ny - 1);
    let xi = (c.x.round() as usize).min(nx - 1);
    let he = h.epi[[yi - oy, xi - ox]] as f64;
    let hm = h.mucus[[yi - oy, xi - ox]] as f64;
    let vertical = (he - c.z) * dz;
    // nearest surface point, searched within |vertical| (the vertical
    // distance is an upper bound on the Euclidean one)
    let pz = c.z * dz;
    let (py, px) = (c.y * dy, c.x * dx);
    let r = vertical.abs() + dz;
    let (ry, rx) = ((r / dy).ceil() as isize, (r / dx).ceil() as isize);
    let mut best = f64::INFINITY;
    // the window must cover the search; a narrower one would answer differently
    debug_assert!(
        (yi as isize - ry).max(0) >= oy as isize
            && (yi as isize + ry).min(ny as isize - 1) < (oy + wy) as isize
            && (xi as isize - rx).max(0) >= ox as isize
            && (xi as isize + rx).min(nx as isize - 1) < (ox + wx) as isize
    );
    for j in (yi as isize - ry).max(oy as isize)
        ..=(yi as isize + ry)
            .min(ny as isize - 1)
            .min((oy + wy) as isize - 1)
    {
        for i in (xi as isize - rx).max(ox as isize)
            ..=(xi as isize + rx)
                .min(nx as isize - 1)
                .min((ox + wx) as isize - 1)
        {
            let sz = h.epi[[j as usize - oy, i as usize - ox]] as f64 * dz;
            let d2 =
                (sz - pz).powi(2) + (j as f64 * dy - py).powi(2) + (i as f64 * dx - px).powi(2);
            best = best.min(d2);
        }
    }
    let euclid = best.sqrt() * vertical.signum();
    let thickness = (he - hm) * dz;
    let rel = if he > hm {
        (c.z - hm) / (he - hm)
    } else {
        f64::NAN
    };
    let class = if vertical < -p.tissue_margin_um {
        Class::InTissue
    } else if c.z_support < p.min_z_support || c.xy_support < p.min_xy_support {
        Class::Spike
    } else {
        Class::Bead
    };
    Measured {
        c: c.clone(),
        class,
        manual: Manual::None,
        h_mucus_um: hm * dz,
        h_epi_um: he * dz,
        vertical_um: vertical,
        euclidean_um: euclid,
        mucus_thickness_um: thickness,
        relative_depth: rel,
        large_object: 0,
    }
}

/// Response below which a shape-passing, above-tissue candidate is `Dim`:
/// Otsu on log(response) of those candidates (0 if fewer than two).
pub fn dim_threshold(bead_responses: &[f64]) -> Result<f64> {
    if bead_responses.iter().filter(|r| **r > 0.0).count() < 2 {
        return Ok(0.0);
    }
    blockflow::ops::log_otsu_threshold(bead_responses, 128)
}

impl Class {
    pub fn from_code(code: u64) -> Result<Self> {
        Class::ALL
            .into_iter()
            .find(|c| u64::from(c.code()) == code)
            .ok_or_else(|| Error::invalid(format!("unknown bead class code {code}")))
    }
}

pub fn write_csv(rows: &[Measured], voxel: [f64; 3], path: &Path) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).map_err(Error::backend)?);
    writeln!(
        w,
        "id,kept,class,manual,z,y,x,z_um,y_um,x_um,response,raw_intensity,mucus_top_um,epi_top_um,\
         vertical_dist_um,euclidean_dist_um,mucus_thickness_um,relative_depth,z_support,xy_support,large_object"
    )
    .map_err(Error::backend)?;
    for m in rows {
        let c = &m.c;
        writeln!(
            w,
            "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.1},{:.1},{:.1},{:.2},{:.2},{:.1},{:.3},{:.3},{:.3},{}",
            c.id,
            u8::from(m.kept()),
            m.class.as_str(),
            m.manual.as_str(),
            c.z,
            c.y,
            c.x,
            c.z * voxel[0],
            c.y * voxel[1],
            c.x * voxel[2],
            c.response,
            c.raw_intensity,
            m.h_mucus_um,
            m.h_epi_um,
            m.vertical_um,
            m.euclidean_um,
            m.mucus_thickness_um,
            m.relative_depth,
            c.z_support,
            c.xy_support,
            m.large_object
        )
        .map_err(Error::backend)?;
    }
    w.flush().map_err(Error::backend)
}
