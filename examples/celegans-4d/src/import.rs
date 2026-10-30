// SPDX-License-Identifier: MIT

//! **An Endrov lineage as Blockflow tracking tables**, published inside the
//! image it annotates.
//!
//! One detection per (nucleus, stored frame of the image) where the nucleus
//! exists by Endrov's rules ([`crate::endrov`]); one track per nucleus (a cell
//! between divisions); links that continue a cell from frame to frame and
//! join a parent's last detection to each child's first. The tables are the
//! ones `plate-timelapse-tracking` writes — the same identity graph, spatial
//! index, `tracking` extensions and link writer, called from that crate — so
//! a viewer that shows its tracks shows these:
//!
//! | output | content |
//! |---|---|
//! | `tables/<layer>` | detections, plus Endrov columns: world position, radius, keyframe/interpolated/held, bracketing keyframes, expression |
//! | `tables/<layer>-tracks` | one row per cell, plus keyframe counts and lifetime overrides; cell names in the `tracking.names` extension |
//! | `tables/<layer>-links` | continue / divide (`plate_timelapse_tracking::tables::write_links`) |
//! | `labels/<layer>` | optional: each nucleus as a sphere, value = detection id, every level of the image |
//! | `tracking/<layer>/` | summary (written last), identity state, `cells.csv`, `detections.csv`, napari exports |
//!
//! # Coordinates
//!
//! Endrov stores positions in world micrometres. Its image transform
//! (`EvStack.transformWorldImage`) is `pixel = (world − disp) / res`, where
//! integer pixel coordinates are pixel *corners* (pixel `(0,0)` is drawn with
//! its top-left corner at `disp`) and z plane `k` lies at `k·resZ + dispZ`
//! (`getClosestPlaneIndex` rounds). OME-Zarr voxel coordinates are voxel
//! *centres*, so with [`PixelOrigin::Corner`] the table's level-0 voxel
//! position is `(world − translation)/scale − ½` in x and y and
//! `(world − translation)/scale` in z, using the image's own level-0
//! transform. No axis is flipped and binning does not enter (Endrov's `resX`
//! is already per stored pixel).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use blockflow::label_pyramid::refresh_label_registry;
use blockflow::object_table::{
    write_object_rows, ColumnRole, ColumnSpec, DType, ObjectValue, TableSpec,
};
use blockflow::ome_zarr::ensure_zarr_group;
use blockflow::time_series::TimeSeriesImage;
use blockflow::track::{
    export, BorderState, Detection, DetectionSet, EndReason, Frame, LinkKind, LinkerCapabilities,
    SeriesGeometry, StartReason, Track, TrackEvents, TrackGraph, TrackLink,
};
use blockflow::{Error, Result};
use ngff_label_map::{ScopedId, TrackingIdentityState};
use plate_timelapse_tracking::series::durable_lineages;
use plate_timelapse_tracking::tables::{
    coordinate_columns, coordinates, identity_graph, index, optional, write_links, TableContext,
};
use serde_json::{json, Value};

use crate::endrov::{LeafEnd, Lineage, Sample, Source};

const ROW_CHUNK: u64 = 65536;
/// Detection id = `t_index * ID_STRIDE + cell number`: unique, stable across
/// reruns of the same lineage, and readable (`310042` is cell 42 at t 31).
pub const ID_STRIDE: u64 = 10_000;

/// Which point of an Endrov pixel its integer coordinates address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelOrigin {
    /// Endrov's convention (corner); see the module docs.
    Corner,
    /// Treat Endrov pixel coordinates as centres (no half-pixel shift).
    Centre,
}

/// Endrov world µm ↔ level-0 OME voxel `[z, y, x]`, per the module docs:
/// `voxel = (world − translation)/scale`, minus ½ in y and x for
/// [`PixelOrigin::Corner`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldMapping {
    /// Level-0 `[z, y, x]` scale (µm per voxel) and translation (µm).
    pub scale: [f64; 3],
    pub translation: [f64; 3],
    /// ½ for [`PixelOrigin::Corner`], else 0.
    pub half: f64,
}

impl WorldMapping {
    pub(crate) fn new(level0: &Grid, origin: PixelOrigin) -> Self {
        Self {
            scale: level0.scale,
            translation: level0.translation,
            half: match origin {
                PixelOrigin::Corner => 0.5,
                PixelOrigin::Centre => 0.0,
            },
        }
    }

    /// Level-0 voxel `[z, y, x]` of Endrov world `(x, y, z)`.
    pub fn voxel(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        [
            (z - self.translation[0]) / self.scale[0],
            (y - self.translation[1]) / self.scale[1] - self.half,
            (x - self.translation[2]) / self.scale[2] - self.half,
        ]
    }

    /// Endrov world `[x, y, z]` of level-0 voxel `[z, y, x]`.
    pub fn world(&self, voxel: [f64; 3]) -> [f64; 3] {
        [
            (voxel[2] + self.half) * self.scale[2] + self.translation[2],
            (voxel[1] + self.half) * self.scale[1] + self.translation[1],
            voxel[0] * self.scale[0] + self.translation[0],
        ]
    }
}

/// What to import and how.
#[derive(Debug, Clone)]
pub struct ImportParams {
    /// The multiscale image group the lineage annotates (e.g. `<rec>.ome.zarr/DIC`).
    pub image: PathBuf,
    /// `rmd.ostxml`, a `.ost` directory or a `.ost.zip`.
    pub ostxml: PathBuf,
    /// `<nuclineage id=..>`; `None` picks the only hand-annotated one.
    pub lineage: Option<String>,
    pub layer: String,
    pub leaf_end: LeafEnd,
    pub pixel_origin: PixelOrigin,
    /// Keep single-keyframe, relative-less markers (`gast`, `venc`, ...) as tracks.
    pub keep_markers: bool,
    /// Rasterise each nucleus as a sphere into `labels/<layer>`.
    pub labels: bool,
    pub label_chunk: usize,
    /// Physical distance from the stack edge below which an endpoint is "near border".
    pub border_distance: f64,
    pub overwrite: bool,
    pub jobs: usize,
}

/// Counts, for the report and the summary.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ImportCounts {
    pub nuclei_in_lineage: usize,
    pub markers: usize,
    pub cells: usize,
    pub cells_named: usize,
    pub cells_without_detections: usize,
    pub keyframes: usize,
    pub keyframes_on_image_frames: usize,
    pub detections: usize,
    pub detections_keyframe: usize,
    pub detections_interpolated: usize,
    pub detections_held: usize,
    /// Detections whose position lies outside the image stack (clamped in the table).
    pub detections_outside: usize,
    pub links: usize,
    pub divisions: usize,
    /// Sibling groups listed as children of more than one nucleus (an
    /// annotation error Endrov tolerates); only the nearest parent is linked.
    pub conflicting_parents: usize,
    pub frames: usize,
}

/// The Endrov frame id (seconds) of every stored time index, from the
/// `img2omezarr.ost.frames.frame` attribute; else `translation + i·scale`
/// from the level-0 transform.
pub fn frame_ids(
    image: &Path,
    n_frames: usize,
    t_scale: f64,
    t_translation: f64,
) -> Result<Vec<f64>> {
    let metadata: Value = read_json(&image.join("zarr.json"))?;
    if metadata
        .pointer("/attributes/ome/version")
        .and_then(Value::as_str)
        == Some("0.6")
    {
        let timeline = TimeSeriesImage::open(image)?.timeline()?;
        let ids = timeline
            .into_iter()
            .map(|frame| {
                frame.elapsed_seconds.ok_or_else(|| {
                    Error::invalid("OME-Zarr 0.6 time axis has no convertible physical unit")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if ids.len() != n_frames {
            return Err(Error::invalid(format!(
                "{} has {} coordinate times for {n_frames} stored frames",
                image.display(),
                ids.len()
            )));
        }
        return Ok(ids);
    }
    if let Some(frames) = metadata
        .pointer("/attributes/img2omezarr/ost/frames/frame")
        .and_then(Value::as_array)
    {
        let ids: Vec<f64> = frames.iter().filter_map(Value::as_f64).collect();
        if ids.len() != n_frames {
            return Err(Error::invalid(format!(
                "{} lists {} frame ids for {n_frames} stored frames",
                image.display(),
                ids.len()
            )));
        }
        return Ok(ids);
    }
    Ok((0..n_frames)
        .map(|i| t_translation + i as f64 * t_scale)
        .collect())
}

/// `(scale, translation)` of one level, in the image's axis order.
fn level_transform(transforms: &Value, rank: usize) -> (Vec<f64>, Vec<f64>) {
    let mut scale = vec![1.0; rank];
    let mut translation = vec![0.0; rank];
    fn apply(rows: &[Value], rank: usize, scale: &mut Vec<f64>, translation: &mut Vec<f64>) {
        for row in rows {
            let values = |key: &str| -> Option<Vec<f64>> {
                row.get(key)?
                    .as_array()?
                    .iter()
                    .map(Value::as_f64)
                    .collect::<Option<Vec<f64>>>()
            };
            match row.get("type").and_then(Value::as_str) {
                Some("scale") => {
                    if let Some(v) = values("scale").filter(|v| v.len() == rank) {
                        *scale = v;
                    }
                }
                Some("translation") => {
                    if let Some(v) = values("translation").filter(|v| v.len() == rank) {
                        *translation = v;
                    }
                }
                Some("sequence") => {
                    if let Some(rows) = row.get("transformations").and_then(Value::as_array) {
                        apply(rows, rank, scale, translation);
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(rows) = transforms.as_array() {
        apply(rows, rank, &mut scale, &mut translation);
    }
    (scale, translation)
}

/// Spatial `[z, y, x]` scale and translation of `level`, plus the `t` pair.
pub(crate) struct Grid {
    pub scale: [f64; 3],
    pub translation: [f64; 3],
    pub t: (f64, f64),
}

pub(crate) fn grid(image: &Path, source: &TimeSeriesImage, level: usize) -> Result<Grid> {
    let metadata = read_json(&image.join("zarr.json"))?;
    let multiscale = metadata
        .pointer("/attributes/ome/multiscales/0")
        .or_else(|| metadata.pointer("/attributes/multiscales/0"));
    let axes = multiscale
        .and_then(|multiscale| {
            multiscale
                .get("axes")
                .and_then(Value::as_array)
                .or_else(|| {
                    multiscale
                        .pointer("/coordinateSystems/0/axes")
                        .and_then(Value::as_array)
                })
        })
        .cloned()
        .unwrap_or_default();
    let names: Vec<String> = axes
        .iter()
        .map(|a| {
            a.as_str()
                .or_else(|| a.get("name").and_then(Value::as_str))
                .unwrap_or("")
                .to_ascii_lowercase()
        })
        .collect();
    let (scale, translation) = level_transform(
        &source.levels()[level].coordinate_transformations,
        names.len(),
    );
    let at = |name: &str, v: &[f64], default: f64| {
        names
            .iter()
            .position(|n| n == name)
            .map_or(default, |i| v[i])
    };
    Ok(Grid {
        scale: [
            at("z", &scale, 1.0),
            at("y", &scale, 1.0),
            at("x", &scale, 1.0),
        ],
        translation: [
            at("z", &translation, 0.0),
            at("y", &translation, 0.0),
            at("x", &translation, 0.0),
        ],
        t: (at("t", &scale, 1.0), at("t", &translation, 0.0)),
    })
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).map_err(|e| Error::backend(format!("read {}: {e}", path.display())))?,
    )
    .map_err(|e| Error::invalid(format!("{} is not JSON: {e}", path.display())))
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(Error::backend)?,
    )
    .map_err(|e| Error::backend(format!("write {}: {e}", path.display())))
}

/// Pick the lineage: the one named, else the only `<nuclineage>` with
/// keyframes that is not Endrov's model fit (`estcell`).
pub fn choose_lineage(lineages: Vec<Lineage>, wanted: Option<&str>) -> Result<Lineage> {
    let ids: Vec<String> = lineages.iter().map(|l| l.id.clone()).collect();
    if let Some(wanted) = wanted {
        return lineages
            .into_iter()
            .find(|l| l.id == wanted)
            .ok_or_else(|| {
                Error::invalid(format!("no <nuclineage id={wanted:?}>; present: {ids:?}"))
            });
    }
    let mut candidates: Vec<Lineage> = lineages
        .into_iter()
        .filter(|l| l.id != "estcell" && l.nuclei.values().any(|n| !n.keyframes.is_empty()))
        .collect();
    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => Err(Error::invalid(format!(
            "no hand-annotated lineage (with keyframes, not estcell); present: {ids:?}"
        ))),
        _ => Err(Error::invalid(format!(
            "several lineages carry keyframes; pick one with --lineage: {:?}",
            candidates.iter().map(|l| &l.id).collect::<Vec<_>>()
        ))),
    }
}

/// One detection row.
struct Row {
    id: u64,
    cell: usize,
    t_index: u64,
    /// Level-0 voxel position, clamped into the stack (object tables refuse
    /// negative coordinates); `outside` records whether clamping moved it.
    voxel: [f64; 3],
    outside: bool,
    sample: Sample,
    expression: Vec<(Option<f64>, bool)>,
}

/// One cell (Endrov nucleus) that became a track.
struct Cell {
    number: u64,
    name: String,
    rows: Vec<usize>,
}

/// Column names must be NGFF path components.
fn column_safe(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub fn run(params: &ImportParams) -> Result<(ImportCounts, Value)> {
    let image = TimeSeriesImage::open(&params.image)?;
    let extent = image.frame_shape(0)?;
    let level0 = grid(&params.image, &image, 0)?;
    let frames = frame_ids(&params.image, image.n_frames(), level0.t.0, level0.t.1)?;
    let text = crate::endrov::read_ostxml(&params.ostxml)?;
    let lineage = choose_lineage(
        crate::endrov::parse_lineages(&text)?,
        params.lineage.as_deref(),
    )?;
    let layer = params.layer.as_str();
    let destinations = [
        params.image.join("tables").join(layer),
        params.image.join("tables").join(format!("{layer}-tracks")),
        params.image.join("tables").join(format!("{layer}-links")),
        params.image.join("labels").join(layer),
        params.image.join("tracking").join(layer),
    ];
    if !params.overwrite {
        if let Some(existing) = destinations.iter().find(|d| d.exists()) {
            return Err(Error::invalid(format!(
                "{} exists; pass --overwrite to replace it",
                existing.display()
            )));
        }
    }
    if lineage.nuclei.len() as u64 >= ID_STRIDE {
        return Err(Error::invalid(format!(
            "lineage has {} nuclei; detection ids allow {}",
            lineage.nuclei.len(),
            ID_STRIDE - 1
        )));
    }

    // World to level-0 voxel.
    let mapping = WorldMapping::new(&level0, params.pixel_origin);
    let half = mapping.half;
    let voxel = |s: &Sample| -> [f64; 3] { mapping.voxel(s.x, s.y, s.z) };

    let expression_names = lineage.expression_names();
    let mut counts = ImportCounts {
        nuclei_in_lineage: lineage.nuclei.len(),
        frames: frames.len(),
        ..ImportCounts::default()
    };
    let frame_set: BTreeSet<u64> = frames.iter().map(|f| f.to_bits()).collect();
    let mut markers = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    let mut cell_of_name: BTreeMap<&str, usize> = BTreeMap::new();
    for (number, (name, nucleus)) in lineage.nuclei.iter().enumerate() {
        let number = number as u64 + 1;
        if nucleus.is_marker() && !params.keep_markers {
            let k = nucleus.keyframes[0];
            counts.markers += 1;
            markers.push(
                json!({"name": name, "frame": k.frame, "x": k.x, "y": k.y, "z": k.z, "r": k.r}),
            );
            continue;
        }
        counts.cells += 1;
        counts.cells_named += usize::from(nucleus.is_named());
        counts.keyframes += nucleus.keyframes.len();
        counts.keyframes_on_image_frames += nucleus
            .keyframes
            .iter()
            .filter(|k| frame_set.contains(&k.frame.to_bits()))
            .count();
        let child_first = lineage.child_first_frame(name);
        let cell = cells.len();
        let mut cell_rows = Vec::new();
        for (t_index, &frame) in frames.iter().enumerate() {
            let Some(sample) = nucleus.interpolate(frame, child_first, params.leaf_end) else {
                continue;
            };
            let expression = expression_names
                .iter()
                .map(|e| {
                    nucleus
                        .expression
                        .get(e)
                        .map_or((None, false), |x| (x.level(frame), x.measured_at(frame)))
                })
                .collect();
            let exact = voxel(&sample);
            let clamped: [f64; 3] =
                std::array::from_fn(|a| exact[a].clamp(0.0, (extent[a] - 1) as f64));
            cell_rows.push(rows.len());
            rows.push(Row {
                id: t_index as u64 * ID_STRIDE + number,
                cell,
                t_index: t_index as u64,
                voxel: clamped,
                outside: clamped != exact,
                sample,
                expression,
            });
        }
        if cell_rows.is_empty() {
            counts.cells_without_detections += 1;
        }
        cell_of_name.insert(name, cell);
        cells.push(Cell {
            number,
            name: name.clone(),
            rows: cell_rows,
        });
    }
    for row in &rows {
        match row.sample.source {
            Source::Keyframe => counts.detections_keyframe += 1,
            Source::Interpolated => counts.detections_interpolated += 1,
            Source::Held => counts.detections_held += 1,
        }
        counts.detections_outside += usize::from(row.outside);
    }
    counts.detections = rows.len();
    if rows.is_empty() {
        return Err(Error::invalid(format!(
            "lineage {:?} has no nucleus on any stored frame of {}",
            lineage.id,
            params.image.display()
        )));
    }

    // Blockflow's tracking structures: the timeline, the detection set, the graph.
    let geometry = SeriesGeometry::new(extent, level0.scale)?;
    let timeline: Vec<Frame> = frames
        .iter()
        .enumerate()
        .map(|(t, &f)| Frame {
            t_index: t as u64,
            elapsed_seconds: Some(f),
        })
        .collect();
    let voxel_volume = level0.scale.iter().product::<f64>();
    let detections: Vec<Detection> = rows
        .iter()
        .map(|r| Detection {
            id: r.id,
            t_index: r.t_index,
            native: r.voxel,
            registered: geometry.physical(r.voxel),
            size: 4.0 / 3.0 * std::f64::consts::PI * r.sample.r.powi(3) / voxel_volume,
        })
        .collect();
    let set = DetectionSet::new(geometry, timeline.clone(), detections)?;
    let link = |from: &Row, to: &Row| TrackLink {
        from: from.id,
        to: to.id,
        dt: to.t_index - from.t_index,
        elapsed_seconds: Some(to.sample.frame - from.sample.frame),
        cost: 0.0,
        kind: LinkKind::Continue,
    };
    let mut links = Vec::new();
    for cell in &cells {
        for pair in cell.rows.windows(2) {
            links.push(link(&rows[pair[0]], &rows[pair[1]]));
        }
    }
    // Candidate parents per child cell. A child listed under several nuclei
    // keeps the parent whose last detection is nearest in time, then space,
    // to the child's first (no use of names or lineage rules).
    let mut parents_of: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, cell) in cells.iter().enumerate() {
        if cell.rows.is_empty() {
            continue;
        }
        for child in &lineage.nuclei[&cell.name].children {
            let Some(&c) = cell_of_name.get(child.as_str()) else {
                continue;
            };
            if !cells[c].rows.is_empty() {
                parents_of.entry(c).or_default().push(index);
            }
        }
    }
    // Children with the same candidate set are siblings claimed by duplicate
    // parents; resolve them together so they stay siblings.
    let gap = |p: usize, c: usize| {
        let first = &rows[cells[c].rows[0]];
        let last = &rows[*cells[p].rows.last().expect("non-empty")];
        let d2: f64 = [
            first.sample.x - last.sample.x,
            first.sample.y - last.sample.y,
            first.sample.z - last.sample.z,
        ]
        .iter()
        .map(|d| d * d)
        .sum();
        ((first.sample.frame - last.sample.frame).abs(), d2)
    };
    let mut groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for (&c, candidates) in &parents_of {
        let mut key = candidates.clone();
        key.sort_unstable();
        key.dedup();
        groups.entry(key).or_default().push(c);
    }
    let mut conflicts = Vec::new();
    let mut child_cells: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (candidates, children) in &groups {
        let total = |p: usize| {
            children.iter().fold((0.0, 0.0), |acc, &c| {
                let g = gap(p, c);
                (acc.0 + g.0, acc.1 + g.1)
            })
        };
        let &p = candidates
            .iter()
            .min_by(|&&a, &&b| {
                let (ga, gb) = (total(a), total(b));
                ga.0.total_cmp(&gb.0).then(ga.1.total_cmp(&gb.1))
            })
            .expect("non-empty");
        if candidates.len() > 1 {
            conflicts.push(json!({
                "children": children.iter().map(|&c| cells[c].name.clone()).collect::<Vec<_>>(),
                "listed_under": candidates.iter().map(|&q| cells[q].name.clone()).collect::<Vec<_>>(),
                "linked_to": cells[p].name,
            }));
        }
        let last = *cells[p].rows.last().expect("non-empty");
        for &c in children {
            links.push(link(&rows[last], &rows[cells[c].rows[0]]));
            child_cells.entry(p).or_default().push(c);
        }
    }
    counts.conflicting_parents = conflicts.len();
    for conflict in &conflicts {
        eprintln!("warning: children listed under several parents: {conflict}");
    }
    TrackGraph::classify(&mut links);
    let graph = TrackGraph::new(
        &set,
        links,
        LinkerCapabilities {
            gaps: true,
            divisions: true,
            merges: false,
        },
    )?;
    counts.links = graph.links().len();
    counts.divisions = child_cells.values().filter(|c| c.len() > 1).count();

    // Tracks are cells, not maximal unbranched chains: a nucleus with a single
    // child (an annotation gap) stays two named tracks.
    let track_of_cell: BTreeMap<usize, u64> = cells
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.rows.first().map(|&r| (i, rows[r].id)))
        .collect();
    let mut parent_cell: BTreeMap<usize, usize> = BTreeMap::new();
    for (&p, children) in &child_cells {
        for &c in children {
            parent_cell.insert(c, p);
        }
    }
    let first_t = 0u64;
    let last_t = frames.len() as u64 - 1;
    let border = |row: &Row| {
        if geometry.border_distance(row.voxel) < params.border_distance {
            BorderState::NearBorder
        } else {
            BorderState::Interior
        }
    };
    let mut tracks: Vec<Track> = Vec::new();
    let mut cell_of_track: BTreeMap<u64, usize> = BTreeMap::new();
    for (index, cell) in cells.iter().enumerate() {
        let (Some(&first), Some(&last)) = (cell.rows.first(), cell.rows.last()) else {
            continue;
        };
        let (first, last) = (&rows[first], &rows[last]);
        let parents: Vec<u64> = parent_cell
            .get(&index)
            .map(|p| track_of_cell[p])
            .into_iter()
            .collect();
        let mut children: Vec<u64> = child_cells
            .get(&index)
            .into_iter()
            .flatten()
            .map(|c| track_of_cell[c])
            .collect();
        children.sort_unstable();
        let start_reason = if !parents.is_empty() {
            StartReason::Division
        } else if first.t_index == first_t {
            StartReason::FirstFrame
        } else {
            StartReason::Unmatched
        };
        let end_reason = if !children.is_empty() {
            EndReason::Division
        } else if last.t_index == last_t {
            EndReason::LastFrame
        } else {
            EndReason::Unmatched
        };
        cell_of_track.insert(first.id, index);
        tracks.push(Track {
            track_id: first.id,
            detections: cell.rows.iter().map(|&r| rows[r].id).collect(),
            t_start: first.t_index,
            t_end: last.t_index,
            parents,
            children,
            events: TrackEvents {
                start_reason,
                end_reason,
                start_border: border(first),
                end_border: border(last),
            },
        });
    }
    tracks.sort_by_key(|t| t.track_id);
    let track_of = TrackGraph::track_ids(&tracks);
    let lineage_of = durable_lineages(&tracks);
    let names: BTreeMap<String, String> = tracks
        .iter()
        .map(|t| {
            (
                t.track_id.to_string(),
                cells[cell_of_track[&t.track_id]].name.clone(),
            )
        })
        .collect();

    // Stage everything beside the destinations, then publish by rename.
    let work = params.image.join(format!(".{layer}-endrov-import-work"));
    if work.exists() {
        fs::remove_dir_all(&work).map_err(Error::backend)?;
    }
    fs::create_dir_all(work.join("tables")).map_err(Error::backend)?;
    let has_z = extent[0] > 1;
    let elapsed: BTreeMap<u64, Option<f64>> = timeline
        .iter()
        .map(|f| (f.t_index, f.elapsed_seconds))
        .collect();
    let context = TableContext {
        has_z,
        extent,
        spacing: level0.scale,
        frames: frames.len() as u64,
        layer,
        elapsed: &elapsed,
        track_labels: false,
    };
    write_detections(
        &work.join("tables").join(layer),
        &context,
        &rows,
        &expression_names,
        &track_of,
        &lineage_of,
        params.labels,
        &cells,
    )?;
    write_tracks(
        &work.join("tables").join(format!("{layer}-tracks")),
        &context,
        &set,
        &tracks,
        &lineage_of,
        &cells,
        &cell_of_track,
        &lineage,
        &names,
    )?;
    write_links(
        &work.join("tables").join(format!("{layer}-links")),
        &context,
        &set,
        &graph,
    )?;
    if params.labels {
        crate::labels::rasterize(
            &work.join("labels"),
            &params.image,
            &image,
            &set,
            &rows
                .iter()
                .map(|r| (r.id, r.sample.r))
                .collect::<BTreeMap<_, _>>(),
            params.label_chunk,
            params.jobs,
        )?;
    }

    // tracking/<layer>/: identity state, exports, summary.
    let tracking = work.join("tracking");
    fs::create_dir_all(&tracking).map_err(Error::backend)?;
    let time_of: BTreeMap<u64, u32> = rows.iter().map(|r| (r.id, r.t_index as u32)).collect();
    let identity_state = TrackingIdentityState::new(
        track_of
            .iter()
            .map(|(&d, &t)| (ScopedId::at_time(time_of[&d], d), t))
            .collect(),
        lineage_of.clone(),
        graph
            .links()
            .iter()
            .map(|l| {
                (
                    (
                        ScopedId::at_time(time_of[&l.from], l.from),
                        ScopedId::at_time(time_of[&l.to], l.to),
                    ),
                    l.kind.code(),
                )
            })
            .collect(),
    )
    .map_err(Error::backend)?;
    write_json(
        &tracking.join("identity-state.json"),
        &serde_json::to_value(&identity_state).map_err(Error::backend)?,
    )?;
    fs::write(
        tracking.join("tracks.csv"),
        export::napari_tracks_csv(&set, &tracks)?,
    )
    .map_err(Error::backend)?;
    write_json(&tracking.join("graph.json"), &export::napari_graph(&tracks))?;
    fs::write(
        tracking.join("cells.csv"),
        cells_csv(&cells, &rows, &lineage, &track_of_cell, &lineage_of),
    )
    .map_err(Error::backend)?;
    fs::write(
        tracking.join("detections.csv"),
        detections_csv(&rows, &cells, &expression_names, &track_of),
    )
    .map_err(Error::backend)?;
    let summary = json!({
        "schema": "blockflow.celegans-4d.endrov-lineage/v1",
        "layer": layer,
        "image": params.image,
        "source": {
            "ostxml": params.ostxml,
            "lineage_id": lineage.id,
            "format": "Endrov <nuclineage> (rmd.ostxml)",
        },
        "outputs": {
            "detections": format!("tables/{layer}"),
            "tracks": format!("tables/{layer}-tracks"),
            "links": format!("tables/{layer}-links"),
            "labels": params.labels.then(|| format!("labels/{layer}")),
            "identity_state": format!("tracking/{layer}/identity-state.json"),
            "exports": ["cells.csv", "detections.csv", "tracks.csv", "graph.json"],
        },
        "frame_shape": extent,
        "spacing": level0.scale,
        "translation": level0.translation,
        "timeline": timeline.iter().map(|f| json!({"t_index": f.t_index, "elapsed_seconds": f.elapsed_seconds})).collect::<Vec<_>>(),
        "counts": counts,
        "markers": markers,
        "conflicting_parents": conflicts,
        "semantics": {
            "lifetime": "from the first keyframe (or start=) to end= inclusive; a parent ends before its earliest child's first frame",
            "between_keyframes": "linear in x, y, z, r (Endrov Lineage.interpolatePos)",
            "after_last_keyframe": "held at the last keyframe (source = held) while the nucleus exists",
            "leaf_end": match params.leaf_end { LeafEnd::LastKeyframe => "last_keyframe", LeafEnd::Hold => "hold" },
            "expression": "linear between stored levels, first/last held outside (Endrov LineageExp.interpolateLevel)",
            "source_codes": {"0": Source::Keyframe.as_str(), "1": Source::Interpolated.as_str(), "2": Source::Held.as_str()},
            "markers": if params.keep_markers { "kept as single-detection tracks" } else { "single-keyframe nuclei without relatives are listed here, not tracked" },
        },
        "tracking": {
            "linker": "endrov-hand-annotation",
            "capabilities": {"gaps": true, "divisions": true, "merges": false},
            "coordinates": format!(
                "level-0 voxel [z, y, x] in tables: (world - translation) / scale{}, clamped into the stack (outside_stack = 1); world_* columns are Endrov micrometres",
                if half > 0.0 { ", minus 1/2 in y and x (Endrov pixel corners -> OME voxel centres)" } else { "" }
            ),
        },
        "codes": {
            "start_reason": {"0": "first_frame", "1": "unmatched", "2": "frame_gap", "3": "division", "4": "merge"},
            "end_reason": {"0": "last_frame", "1": "unmatched", "2": "frame_gap", "3": "division", "4": "merge"},
            "link_kind": {"0": "continue", "1": "divide", "2": "merge", "3": "gap"},
        },
    });
    write_json(&tracking.join("tracking-summary.json"), &summary)?;

    // Publish: replace the destinations, the tracking directory last.
    for destination in &destinations {
        if destination.exists() {
            fs::remove_dir_all(destination).map_err(Error::backend)?;
        }
    }
    ensure_zarr_group(&params.image.join("tables"))?;
    let mut moves = vec![
        (work.join("tables").join(layer), destinations[0].clone()),
        (
            work.join("tables").join(format!("{layer}-tracks")),
            destinations[1].clone(),
        ),
        (
            work.join("tables").join(format!("{layer}-links")),
            destinations[2].clone(),
        ),
    ];
    if params.labels {
        ensure_zarr_group(&params.image.join("labels"))?;
        moves.push((work.join("labels"), destinations[3].clone()));
    }
    for (from, to) in moves {
        fs::rename(&from, &to).map_err(|e| {
            Error::backend(format!(
                "publish {} -> {}: {e}",
                from.display(),
                to.display()
            ))
        })?;
    }
    if params.labels {
        let path = destinations[3].join("zarr.json");
        let mut label_json = read_json(&path)?;
        label_json["attributes"]["identity_graph"] =
            serde_json::to_value(identity_graph(layer)).map_err(Error::backend)?;
        write_json(&path, &label_json)?;
        refresh_label_registry(&params.image.join("labels"))?;
    }
    fs::create_dir_all(params.image.join("tracking")).map_err(Error::backend)?;
    fs::rename(&tracking, &destinations[4]).map_err(Error::backend)?;
    fs::remove_dir_all(&work).map_err(Error::backend)?;
    Ok((counts, summary))
}

#[allow(clippy::too_many_arguments)]
fn write_detections(
    root: &Path,
    context: &TableContext<'_>,
    rows: &[Row],
    expression_names: &[String],
    track_of: &BTreeMap<u64, u64>,
    lineage_of: &BTreeMap<u64, u64>,
    labels: bool,
    cells: &[Cell],
) -> Result<()> {
    let um = |name: &str| {
        let mut c = ColumnSpec::new(name, DType::F64, ColumnRole::Measurement);
        c.unit = Some("micrometer".into());
        c
    };
    let seconds = |name: &str| {
        let mut c = ColumnSpec::new(name, DType::F64, ColumnRole::Measurement);
        c.unit = Some("second".into());
        c
    };
    let mut columns = vec![ColumnSpec::new(
        "detection_id",
        DType::U64,
        ColumnRole::Identity,
    )];
    columns.extend(coordinate_columns(context.has_z, ""));
    columns.extend([
        ColumnSpec::new("track_id", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("lineage_id", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("t_index", DType::U64, ColumnRole::Measurement),
        seconds("elapsed_seconds"),
        ColumnSpec::new("cell_number", DType::U64, ColumnRole::Category),
        ColumnSpec::new("named", DType::U8, ColumnRole::Category),
        ColumnSpec::new("source", DType::U8, ColumnRole::Category),
        ColumnSpec::new("keyframe", DType::U8, ColumnRole::Category),
        ColumnSpec::new("outside_stack", DType::U8, ColumnRole::Category),
        seconds("key_before"),
        seconds("key_after"),
        um("world_x"),
        um("world_y"),
        um("world_z"),
        um("radius"),
    ]);
    for name in expression_names {
        let safe = column_safe(name);
        columns.push(ColumnSpec::new(
            format!("expression_{safe}"),
            DType::F64,
            ColumnRole::Intensity,
        ));
        columns.push(ColumnSpec::new(
            format!("expression_{safe}_measured"),
            DType::U8,
            ColumnRole::Category,
        ));
    }
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let track = track_of[&row.id];
        let s = &row.sample;
        let cell = &cells[row.cell];
        let mut v = vec![ObjectValue::U64(row.id)];
        v.extend(coordinates(context.has_z, row.t_index, row.voxel));
        v.extend([
            ObjectValue::U64(track),
            ObjectValue::U64(lineage_of[&track]),
            ObjectValue::U64(row.t_index),
            optional(context.elapsed.get(&row.t_index).copied().flatten()),
            ObjectValue::U64(cell.number),
            ObjectValue::U8(u8::from(!cell.name.starts_with(':'))),
            ObjectValue::U8(s.source.code()),
            ObjectValue::U8(u8::from(s.source == Source::Keyframe)),
            ObjectValue::U8(u8::from(row.outside)),
            ObjectValue::F64(s.key_before),
            optional(s.key_after),
            ObjectValue::F64(s.x),
            ObjectValue::F64(s.y),
            ObjectValue::F64(s.z),
            ObjectValue::F64(s.r),
        ]);
        for &(level, measured) in &row.expression {
            v.push(optional(level));
            v.push(ObjectValue::U8(u8::from(measured)));
        }
        values.push(v);
    }
    let mut spec = TableSpec::new(
        0,
        ROW_CHUNK,
        "../../",
        "detection_id",
        columns,
        index(context.has_z, context.extent, context.frames, ""),
    );
    let layer = context.layer;
    spec.region = labels.then(|| format!("../../labels/{layer}"));
    spec.extensions.insert(
        "tracking".into(),
        json!({
            "version": 1,
            "role": "detections",
            "track_column": "track_id",
            "time_column": "t_index",
            "elapsed_column": "elapsed_seconds",
            "tracks": format!("../{layer}-tracks"),
            "links": format!("../{layer}-links"),
            "summary": format!("../../tracking/{layer}/tracking-summary.json"),
            "label": labels.then(|| format!("../../labels/{layer}")),
            "track_label": Value::Null,
        }),
    );
    spec.extensions.insert(
        "identity_graph".into(),
        serde_json::to_value(identity_graph(layer)).expect("identity graph serializes"),
    );
    spec.extensions.insert(
        "endrov".into(),
        json!({
            "version": 1,
            "source_codes": {"0": "keyframe", "1": "interpolated", "2": "held"},
            "key_after": "NaN while held after the last keyframe",
            "expression": expression_names,
        }),
    );
    write_object_rows(root, spec, values)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_tracks(
    root: &Path,
    context: &TableContext<'_>,
    detections: &DetectionSet,
    tracks: &[Track],
    lineage_of: &BTreeMap<u64, u64>,
    cells: &[Cell],
    cell_of_track: &BTreeMap<u64, usize>,
    lineage: &Lineage,
    names: &BTreeMap<String, String>,
) -> Result<()> {
    let mut columns = vec![ColumnSpec::new(
        "track_id",
        DType::U64,
        ColumnRole::Identity,
    )];
    columns.extend(coordinate_columns(context.has_z, "start_"));
    let u64_column = |name: &str| ColumnSpec::new(name, DType::U64, ColumnRole::Measurement);
    let u8_column = |name: &str| ColumnSpec::new(name, DType::U8, ColumnRole::Measurement);
    let seconds = |name: &str| {
        let mut c = ColumnSpec::new(name, DType::F64, ColumnRole::Measurement);
        c.unit = Some("second".into());
        c
    };
    // The plate-timelapse-tracking columns, then the Endrov ones.
    columns.extend([
        u64_column("lineage_id"),
        u64_column("t_start"),
        u64_column("t_end"),
        u64_column("first_id"),
        u64_column("last_id"),
        u64_column("n_detections"),
        u64_column("n_parents"),
        u64_column("parent_track_id"),
        u64_column("n_children"),
        u8_column("start_reason"),
        u8_column("end_reason"),
        u8_column("start_near_border"),
        u8_column("end_near_border"),
        u8_column("likely_entered_fov"),
        u8_column("likely_left_fov"),
        ColumnSpec::new("cell_number", DType::U64, ColumnRole::Category),
        ColumnSpec::new("named", DType::U8, ColumnRole::Category),
        u64_column("n_keyframes"),
        seconds("first_keyframe"),
        seconds("last_keyframe"),
        seconds("override_start"),
        seconds("override_end"),
    ]);
    let mut values = Vec::with_capacity(tracks.len());
    for track in tracks {
        let first = detections
            .get(track.first_id())
            .ok_or_else(|| Error::invalid("track names an unknown detection"))?;
        let cell = &cells[cell_of_track[&track.track_id]];
        let nucleus = &lineage.nuclei[&cell.name];
        let mut v = vec![ObjectValue::U64(track.track_id)];
        v.extend(coordinates(context.has_z, track.t_start, first.native));
        v.extend([
            ObjectValue::U64(lineage_of[&track.track_id]),
            ObjectValue::U64(track.t_start),
            ObjectValue::U64(track.t_end),
            ObjectValue::U64(track.first_id()),
            ObjectValue::U64(track.last_id()),
            ObjectValue::U64(track.detections.len() as u64),
            ObjectValue::U64(track.parents.len() as u64),
            ObjectValue::U64(track.parents.first().copied().unwrap_or(0)),
            ObjectValue::U64(track.children.len() as u64),
            ObjectValue::U8(track.events.start_reason.code()),
            ObjectValue::U8(track.events.end_reason.code()),
            ObjectValue::U8(track.events.start_border.code()),
            ObjectValue::U8(track.events.end_border.code()),
            ObjectValue::U8(u8::from(track.events.likely_entered_fov())),
            ObjectValue::U8(u8::from(track.events.likely_left_fov())),
            ObjectValue::U64(cell.number),
            ObjectValue::U8(u8::from(nucleus.is_named())),
            ObjectValue::U64(nucleus.keyframes.len() as u64),
            optional(nucleus.keyframes.first().map(|k| k.frame)),
            optional(nucleus.keyframes.last().map(|k| k.frame)),
            optional(nucleus.override_start),
            optional(nucleus.override_end),
        ]);
        values.push(v);
    }
    let mut spec = TableSpec::new(
        0,
        ROW_CHUNK,
        "../../",
        "track_id",
        columns,
        index(context.has_z, context.extent, context.frames, "start_"),
    );
    spec.extensions.insert(
        "tracking".into(),
        json!({
            "version": 1,
            "role": "tracks",
            "detections": format!("../{}", context.layer),
            // Numeric columns only in a native table, so cell names ride here.
            "names": names,
        }),
    );
    write_object_rows(root, spec, values)?;
    Ok(())
}

fn csv_text(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

fn cells_csv(
    cells: &[Cell],
    rows: &[Row],
    lineage: &Lineage,
    track_of_cell: &BTreeMap<usize, u64>,
    lineage_of: &BTreeMap<u64, u64>,
) -> String {
    let opt = |v: Option<f64>| v.map(|v| v.to_string()).unwrap_or_default();
    let mut text = String::from(
        "track_id,lineage_id,cell_number,name,parent,children,n_keyframes,first_keyframe,last_keyframe,\
         override_start,override_end,t_start,t_end,n_detections\n",
    );
    for (index, cell) in cells.iter().enumerate() {
        let nucleus = &lineage.nuclei[&cell.name];
        let track = track_of_cell.get(&index).copied();
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            track.map(|t| t.to_string()).unwrap_or_default(),
            track
                .and_then(|t| lineage_of.get(&t))
                .map(|l| l.to_string())
                .unwrap_or_default(),
            cell.number,
            csv_text(&cell.name),
            csv_text(nucleus.parent.as_deref().unwrap_or("")),
            csv_text(&nucleus.children.join(";")),
            nucleus.keyframes.len(),
            opt(nucleus.keyframes.first().map(|k| k.frame)),
            opt(nucleus.keyframes.last().map(|k| k.frame)),
            opt(nucleus.override_start),
            opt(nucleus.override_end),
            cell.rows
                .first()
                .map(|&r| rows[r].t_index.to_string())
                .unwrap_or_default(),
            cell.rows
                .last()
                .map(|&r| rows[r].t_index.to_string())
                .unwrap_or_default(),
            cell.rows.len(),
        ));
    }
    text
}

fn detections_csv(
    rows: &[Row],
    cells: &[Cell],
    expression_names: &[String],
    track_of: &BTreeMap<u64, u64>,
) -> String {
    let opt = |v: Option<f64>| v.map(|v| v.to_string()).unwrap_or_default();
    let mut text = String::from(
        "detection_id,track_id,name,t_index,frame,z,y,x,world_x,world_y,world_z,radius,source,key_before,key_after",
    );
    for name in expression_names {
        text.push_str(&format!(",expression_{}", column_safe(name)));
    }
    text.push('\n');
    for r in rows {
        let s = &r.sample;
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.id,
            track_of[&r.id],
            csv_text(&cells[r.cell].name),
            r.t_index,
            s.frame,
            r.voxel[0],
            r.voxel[1],
            r.voxel[2],
            s.x,
            s.y,
            s.z,
            s.r,
            s.source.as_str(),
            s.key_before,
            opt(s.key_after),
        ));
        for &(level, _) in &r.expression {
            text.push(',');
            text.push_str(&opt(level));
        }
        text.push('\n');
    }
    text
}
