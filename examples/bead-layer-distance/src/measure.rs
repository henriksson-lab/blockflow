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
    /// Closest distance to the fitted tissue region (`bjorn-layers` 2).
    pub surface: BoundaryDistance,
    /// Closest distance to the stained epithelium (`bjorn-epithelium`).
    pub epithelium: BoundaryDistance,
}

/// A signed closest distance to a mask's boundary (blockflow
/// `ops::point_distance`): negative inside the mask.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryDistance {
    pub distance_um: f64,
    /// Voxel index (z, y, x) of the nearest boundary voxel.
    pub nearest: [f64; 3],
    /// `ops::STATUS_EXACT`, `STATUS_CAPPED` (an upper bound) or `STATUS_NONE`.
    pub status: u64,
}

impl BoundaryDistance {
    /// Not measured yet: the candidate itself, status none.
    pub fn unset(c: &Candidate) -> Self {
        Self {
            distance_um: 0.0,
            nearest: [c.z, c.y, c.x],
            status: blockflow::ops::STATUS_NONE,
        }
    }

    pub fn status_str(&self) -> &'static str {
        match self.status {
            blockflow::ops::STATUS_EXACT => "exact",
            blockflow::ops::STATUS_CAPPED => "capped",
            _ => "none",
        }
    }
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
        surface: BoundaryDistance::unset(c),
        epithelium: BoundaryDistance::unset(c),
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
         vertical_dist_um,euclidean_dist_um,mucus_thickness_um,relative_depth,z_support,xy_support,large_object,\
         surface_dist_um,surface_nearest_z_um,surface_nearest_y_um,surface_nearest_x_um,surface_status,\
         epithelium_dist_um,epithelium_nearest_z_um,epithelium_nearest_y_um,epithelium_nearest_x_um,epithelium_status"
    )
    .map_err(Error::backend)?;
    for m in rows {
        let c = &m.c;
        writeln!(
            w,
            "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.1},{:.1},{:.1},{:.2},{:.2},{:.1},{:.3},{:.3},{:.3},{},{}",
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
            m.large_object,
            [&m.surface, &m.epithelium]
                .map(|d| format!(
                    "{:.2},{:.2},{:.2},{:.2},{}",
                    d.distance_um,
                    d.nearest[0] * voxel[0],
                    d.nearest[1] * voxel[1],
                    d.nearest[2] * voxel[2],
                    d.status_str()
                ))
                .join(",")
        )
        .map_err(Error::backend)?;
    }
    w.flush().map_err(Error::backend)
}

/// Per distance (fitted surface, stained epithelium), over the counted beads:
/// count, negatives, inexact rows and quantiles, as CSV; the lines are also
/// returned for the log.
pub fn write_distance_summary(rows: &[Measured], path: &Path) -> Result<Vec<String>> {
    let mut lines = vec![
        "distance,beads,negative,capped,none,min_um,p05_um,p25_um,median_um,p75_um,p95_um,max_um"
            .to_string(),
    ];
    let kept: Vec<&Measured> = rows.iter().filter(|m| m.kept()).collect();
    for (name, of) in [
        (
            "surface",
            (|m: &Measured| m.surface) as fn(&Measured) -> BoundaryDistance,
        ),
        ("epithelium", |m: &Measured| m.epithelium),
    ] {
        let all: Vec<BoundaryDistance> = kept.iter().map(|m| of(m)).collect();
        let mut d: Vec<f64> = all
            .iter()
            .filter(|b| b.status != blockflow::ops::STATUS_NONE)
            .map(|b| b.distance_um)
            .collect();
        d.sort_by(f64::total_cmp);
        let q = |f: f64| {
            if d.is_empty() {
                f64::NAN
            } else {
                d[((d.len() - 1) as f64 * f).round() as usize]
            }
        };
        let count = |status: u64| all.iter().filter(|b| b.status == status).count();
        lines.push(format!(
            "{name},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            all.len(),
            d.iter().filter(|v| **v < 0.0).count(),
            count(blockflow::ops::STATUS_CAPPED),
            count(blockflow::ops::STATUS_NONE),
            q(0.0),
            q(0.05),
            q(0.25),
            q(0.5),
            q(0.75),
            q(0.95),
            q(1.0)
        ));
    }
    std::fs::write(path, lines.join("\n") + "\n").map_err(Error::backend)?;
    Ok(lines)
}
