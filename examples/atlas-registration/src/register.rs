// SPDX-License-Identifier: MIT
// Original work for this crate.

// Point transforms read from elastix `TransformParameters.N.txt` files.
//
// Every point's image is a function of that point and the transform alone, so
// this is resident point arithmetic: there is no lattice, no block and no plan.
// `tests/register_chain.rs` asserts that splitting a point list changes nothing,
// which is the property a parallel caller would rely on.
//
// The conventions, each measured against transformix 4.900
// --------------------------------------------------------
// `tests/register_transformix.rs` holds the recordings.
//
// **1. Direction.** An elastix transform maps the **fixed** image's physical
// space to the **moving** image's (`T : ΩF → ΩM`): the resampler walks the
// fixed grid asking where each voxel came from in the moving image, and
// `transformix -def` applies the transform as stored, with no inversion.
// [`ElastixTransform::apply`] does the same. To carry points from a sample into
// an atlas, register with the sample as the fixed image.
//
// **2. Index versus physical.** A point file's first line chooses whether its
// numbers are physical coordinates (`point`) or voxel indices (`index`), which
// are converted by `p = Origin + Direction · (Spacing ⊙ index)`. The `index`
// branch rounds twice — on the way in, and when the output is put back through
// the fixed geometry — both with `floor(x + 0.5)`, which is ITK's
// `RoundHalfIntegerUp` and not `f64::round`. With a unit geometry the two
// spaces are the same numbers; with any other geometry handing voxel indices
// to a `point` file is a silent scale error (`tests/register_controls.rs`).
//
// **3. The chain.** `InitialTransformParametersFileName` is a backwards
// pointer: the file handed to `-tp` is the *last* transform and names the one
// before it, so the order comes from the pointers, not the filenames.
// `(HowToCombineTransforms "Compose")` gives `T_current(T_initial(x))` — the
// initial runs first — and `"Add"` gives `T_initial(x) + T_current(x) - x`. An
// absent entry is `"Compose"`.
//
// **4. B-spline.** Cubic basis, four control points of support per axis
// starting at `floor(u) - 1`, over `u = GridDirection^-1 (p - GridOrigin) /
// GridSpacing`. Outside `[GridIndex + 1, GridIndex + GridSize - 2)` the
// transform is the identity; clamping the control point index instead gives a
// displacement that looks reasonable everywhere and is wrong everywhere
// outside. `RecursiveBSplineTransform` is the same evaluation.
//
// What is refused
// ---------------
// Every `(Transform "…")` other than translation, Euler, affine and the two
// B-spline keywords is refused by name, as are B-spline orders other than 3 and
// `(UseCyclicTransform "true")`. A reader that treated an unknown transform as
// the identity would hand back coordinates for a registration it had not
// applied.

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};

use crate::parameters::ParameterMap;
use crate::rescale::Rescale;
use crate::transform::{Chain, ImageGeometry, IndexRounding, Link};

/// What the numbers in a point list mean: elastix's `point` or `index`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointSpace {
    /// Physical coordinates in, `OutputPoint` out.
    Physical,
    /// Voxel indices in, `OutputIndexFixed` out — an integer, because that is
    /// what transformix prints.
    Index,
}

/// How a point list is read and written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterParams {
    /// Physical or index, at the input **and** the output: one choice in the
    /// point file format.
    pub space: PointSpace,
    /// How a physical point becomes a voxel index. Read only in
    /// [`PointSpace::Index`].
    pub index_rounding: IndexRounding,
}

impl Default for RegisterParams {
    fn default() -> Self {
        Self {
            space: PointSpace::Physical,
            index_rounding: IndexRounding::HalfUp,
        }
    }
}

impl RegisterParams {
    pub fn in_space(space: PointSpace) -> Self {
        Self {
            space,
            ..Self::default()
        }
    }
}

// -------------------------------------------------------- one registration --

/// One registration's transform: the chain, and the fixed image geometry the
/// index mode needs.
#[derive(Debug, Clone, PartialEq)]
pub struct ElastixTransform {
    chain: Chain,
    geometry: ImageGeometry,
}

impl ElastixTransform {
    /// From parameter file texts given **outermost first** — the file
    /// `transformix -tp` would be handed, then its initial, then its initial's.
    ///
    /// Pure: no filesystem, and the `InitialTransformParametersFileName` values
    /// are checked for consistency rather than followed.
    pub fn from_texts(texts: &[&str]) -> Result<Self> {
        if texts.is_empty() {
            return Err(Error::InvalidArgument(
                "elastix transform: a registration needs at least one parameter file".to_string(),
            ));
        }
        let maps = texts
            .iter()
            .map(|text| ParameterMap::parse(text))
            .collect::<Result<Vec<_>>>()?;
        Self::from_maps(&maps)
    }

    fn from_maps(maps: &[ParameterMap]) -> Result<Self> {
        let geometry = ImageGeometry::from_map(&maps[0])?;
        let links = maps
            .iter()
            .map(Link::from_map)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            chain: Chain::from_outermost_first(links)?,
            geometry,
        })
    }

    /// Follow `InitialTransformParametersFileName` from one file on disk.
    ///
    /// A relative initial path is resolved against the directory of the file
    /// that names it, so a result directory can be moved as a whole. The chain
    /// length is bounded so a file naming itself is an error rather than a hang.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        const MAX_LINKS: usize = 64;
        let mut maps = Vec::new();
        let mut next = Some(path.as_ref().to_path_buf());
        while let Some(current) = next.take() {
            if maps.len() >= MAX_LINKS {
                return Err(Error::InvalidArgument(format!(
                    "elastix transform: the chain from {} is more than {MAX_LINKS} links long, \
                     which means `InitialTransformParametersFileName` forms a cycle",
                    path.as_ref().display()
                )));
            }
            let text = fs::read_to_string(&current).map_err(|error| {
                Error::InvalidArgument(format!(
                    "elastix transform: cannot read {}: {error}",
                    current.display()
                ))
            })?;
            let map = ParameterMap::parse(&text)?;
            if let Some(initial) = Link::from_map(&map)?.initial() {
                let initial = Path::new(initial);
                next = Some(if initial.is_absolute() {
                    initial.to_path_buf()
                } else {
                    current
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join(initial)
                });
            }
            maps.push(map);
        }
        Self::from_maps(&maps)
    }

    /// The transform of a result directory: the highest-numbered
    /// `TransformParameters.N.txt`.
    pub fn from_directory(directory: impl AsRef<Path>) -> Result<Self> {
        Self::read(transform_file(directory)?)
    }

    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    pub fn geometry(&self) -> &ImageGeometry {
        &self.geometry
    }

    /// Replace the fixed image geometry, for a caller who knows a geometry the
    /// file does not record.
    pub fn with_geometry(mut self, geometry: ImageGeometry) -> Self {
        self.geometry = geometry;
        self
    }

    /// Transform one point, in the space `params` names.
    ///
    /// `Physical` is transformix's `OutputPoint`; `Index` is its
    /// `OutputIndexFixed`, the same physical answer put back through the fixed
    /// image's geometry and rounded.
    pub fn apply(&self, point: [f64; 3], params: &RegisterParams) -> [f64; 3] {
        match params.space {
            PointSpace::Physical => self.chain.apply(point),
            PointSpace::Index => {
                let rounding = params.index_rounding;
                let physical = self.geometry.index_to_physical_rounded(point, rounding);
                let transformed = self.chain.apply(physical);
                self.geometry.physical_to_index(transformed, rounding)
            }
        }
    }

    /// The whole list, **in order**: callers address points by index.
    pub fn apply_all(&self, points: &[[f64; 3]], params: &RegisterParams) -> Vec<[f64; 3]> {
        points
            .iter()
            .map(|point| self.apply(*point, params))
            .collect()
    }
}

/// The transform of a result directory: the `TransformParameters.N.txt` with
/// the largest `N`.
///
/// Numeric, not lexicographic: sorted as strings `TransformParameters.9.txt`
/// comes after `TransformParameters.10.txt`, which picks the wrong file as soon
/// as a run has ten or more. A name of that shape whose middle is not a number
/// is refused rather than skipped.
pub fn transform_file(directory: impl AsRef<Path>) -> Result<PathBuf> {
    let directory = directory.as_ref();
    let mut best: Option<(u64, PathBuf)> = None;
    let entries = fs::read_dir(directory).map_err(|error| {
        Error::InvalidArgument(format!(
            "elastix transform: cannot list {}: {error}",
            directory.display()
        ))
    })?;
    for path in entries.filter_map(|entry| entry.ok().map(|entry| entry.path())) {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !is_transform_parameter_file(name) {
            continue;
        }
        let middle = &name["TransformParameters.".len()..name.len() - ".txt".len()];
        let number = middle.parse::<u64>().map_err(|error| {
            Error::InvalidArgument(format!(
                "elastix transform: `{name}` in {} is not numbered, so there is no highest \
                 transform file: {error}",
                directory.display()
            ))
        })?;
        if best.as_ref().is_none_or(|(seen, _)| number > *seen) {
            best = Some((number, path));
        }
    }
    best.map(|(_, path)| path).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "elastix transform: no `TransformParameters.*.txt` in {}",
            directory.display()
        ))
    })
}

fn is_transform_parameter_file(name: &str) -> bool {
    name.starts_with("TransformParameters.") && name.ends_with(".txt")
}

// ------------------------------------------------------- the whole chain --

/// A point from one volume's frame to another's: a rescale into the frame the
/// registration was computed in, the registrations in order, and a rescale out.
///
/// The registrations are applied **in the order given** and each forward, as
/// stored; this type does not choose the direction of any of them.
#[derive(Debug, Clone, PartialEq)]
pub struct AtlasTransform {
    into: Rescale,
    registrations: Vec<ElastixTransform>,
    out_of: Rescale,
    params: RegisterParams,
}

impl AtlasTransform {
    pub fn new(
        into: Rescale,
        registrations: Vec<ElastixTransform>,
        out_of: Rescale,
        params: RegisterParams,
    ) -> Self {
        Self {
            into,
            registrations,
            out_of,
            params,
        }
    }

    /// The rescale alone, with no registration.
    pub fn rescale_only(rescale: Rescale) -> Self {
        Self::new(
            rescale,
            Vec::new(),
            Rescale::identity(),
            RegisterParams::default(),
        )
    }

    pub fn params(&self) -> &RegisterParams {
        &self.params
    }

    pub fn registrations(&self) -> &[ElastixTransform] {
        &self.registrations
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let mut current = self.into.apply(point);
        for registration in &self.registrations {
            current = registration.apply(current, &self.params);
        }
        self.out_of.apply(current)
    }

    /// The whole list, in order.
    pub fn apply_all(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|point| self.apply(*point)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transform_file_filter_is_the_elastix_naming() {
        assert!(is_transform_parameter_file("TransformParameters.0.txt"));
        assert!(is_transform_parameter_file("TransformParameters.10.txt"));
        assert!(!is_transform_parameter_file("TransformParameters.0.log"));
        assert!(!is_transform_parameter_file("elastix.log"));
    }

    #[test]
    fn the_default_point_space_is_physical() {
        let params = RegisterParams::default();
        assert_eq!(params.space, PointSpace::Physical);
        assert_eq!(params.index_rounding, IndexRounding::HalfUp);
    }
}
