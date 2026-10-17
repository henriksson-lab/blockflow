//! Planned bead selection and measurement.
//!
//! Upstream phases (assembled in `main`): DoG response on C-Blue; a
//! robust-sigma level from a lattice sample of the response; the response
//! kept at or above that level; blockflow's exact regional maxima as rows; a
//! lattice sample of raw C-Blue for the background.
//!
//! [`BeadSelectOp`] (barrier): `reduce` sorts the maxima strongest first and
//! keeps those at least the minimum distance apart, takes the background
//! median and the surface height maps; each block then scores the PSF shape
//! of the peaks it owns, measures them against the surfaces and classifies
//! them (`Bead` / `InTissue` / `Spike`).
//!
//! [`BeadFinalOp`] (barrier): `reduce` sets the brightness split (`Dim`) from
//! every `Bead`'s response; each block applies it and looks up the large
//! AF488 object each of its candidates touches.

use std::sync::Arc;

use blockflow::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SourceBlocks,
};
use blockflow::op::SourceInput;
use blockflow::ops::surfaces::SurfaceColumns;
use blockflow::ops::{sampled_thresholds, GlobalThreshold};
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::table::{Column, RowBuilder, Schema, Table, Value};
use blockflow::{Error, ImageId, Reach, Result};
use ndarray::Array2;

use crate::measure::{measure_one, Candidate, Class, HeightMaps, MeasureParams, Measured};

// ---------------------------------------------------------------- bytes --

pub(crate) struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    pub(crate) fn new(bytes: &'a [u8], magic: &[u8; 4]) -> Result<Self> {
        if bytes.len() < 4 || &bytes[..4] != magic {
            return Err(Error::invalid("bjorn: reduction blob has the wrong magic"));
        }
        Ok(Self(&bytes[4..]))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            return Err(Error::invalid("bjorn: reduction blob truncated"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub(crate) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self
            .take(n * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

pub(crate) fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend(v.to_le_bytes());
}

pub(crate) fn put_f64(out: &mut Vec<u8>, v: f64) {
    out.extend(v.to_le_bytes());
}

/// Every row of `stream` (all blocks), through `schema`.
pub(crate) fn rows_of(at: &PhaseView<'_>, stream: &str, schema: &Schema) -> Result<Table> {
    let mut table = Table::new(at.volume(), schema.clone())?;
    at.stream_fragments(stream, &mut |key, bytes| table.write(key.block, bytes))?;
    table.seal()?;
    Ok(table)
}

/// This block's own rows of `stream`.
pub(crate) fn own_rows(at: &BlockView<'_>, stream: &str, schema: &Schema) -> Result<Table> {
    let mut table = Table::new(at.volume(), schema.clone())?;
    if let Some(bytes) = at.own(stream) {
        table.write(at.index, bytes)?;
    }
    table.seal()?;
    Ok(table)
}

// ------------------------------------------------------------- the rows --

/// One row per candidate, at its peak voxel.
pub fn bead_schema() -> Result<Schema> {
    Schema::new(vec![
        Column::u64("id"),
        Column::f64("response"),
        Column::f64("raw_intensity"),
        Column::f64("z_support"),
        Column::f64("xy_support"),
        Column::u64("class"),
        Column::u64("large_object"),
        Column::f64("mucus_top_um"),
        Column::f64("epi_top_um"),
        Column::f64("vertical_dist_um"),
        Column::f64("euclidean_dist_um"),
        Column::f64("mucus_thickness_um"),
        Column::f64("relative_depth"),
    ])
}

fn values_of(m: &Measured) -> [Value; 13] {
    [
        Value::U64(m.c.id),
        Value::F64(m.c.response),
        Value::F64(m.c.raw_intensity),
        Value::F64(m.c.z_support),
        Value::F64(m.c.xy_support),
        Value::U64(u64::from(m.class.code())),
        Value::U64(u64::from(m.large_object)),
        Value::F64(m.h_mucus_um),
        Value::F64(m.h_epi_um),
        Value::F64(m.vertical_um),
        Value::F64(m.euclidean_um),
        Value::F64(m.mucus_thickness_um),
        Value::F64(m.relative_depth),
    ]
}

/// A row back into a [`Measured`].
pub fn measured_of(at: [usize; 3], v: &[Value]) -> Result<Measured> {
    let f = |i: usize| match v[i] {
        Value::F64(x) => Ok(x),
        Value::U64(_) => Err(Error::invalid("bead row: expected f64")),
    };
    let u = |i: usize| match v[i] {
        Value::U64(x) => Ok(x),
        Value::F64(_) => Err(Error::invalid("bead row: expected u64")),
    };
    Ok(Measured {
        c: Candidate {
            id: u(0)?,
            z: at[0] as f64,
            y: at[1] as f64,
            x: at[2] as f64,
            response: f(1)?,
            raw_intensity: f(2)?,
            z_support: f(3)?,
            xy_support: f(4)?,
        },
        class: Class::from_code(u(5)?)?,
        manual: crate::measure::Manual::None,
        large_object: u(6)? as u32,
        h_mucus_um: f(7)?,
        h_epi_um: f(8)?,
        vertical_um: f(9)?,
        euclidean_um: f(10)?,
        mucus_thickness_um: f(11)?,
        relative_depth: f(12)?,
    })
}

/// Every value of a row, in schema order.
pub fn values_of_row(row: &blockflow::table::Row<'_>) -> Result<Vec<Value>> {
    (0..row.schema().len())
        .map(|i| match row.schema().columns()[i].kind() {
            blockflow::table::ColumnType::U64 => row.u64(i).map(Value::U64),
            blockflow::table::ColumnType::F64 => row.f64(i).map(Value::F64),
        })
        .collect()
}

pub fn measured_from_row(row: &blockflow::table::Row<'_>) -> Result<Measured> {
    measured_of(row.at(), &values_of_row(row)?)
}

// --------------------------------------------------------------- select --

const SELECT_MAGIC: &[u8; 4] = b"BSe1";

pub struct BeadSelectParams {
    pub min_distance_um: f64,
    pub measure: MeasureParams,
}

/// Selection, PSF shape, measurement and the automatic class (see module docs).
pub struct BeadSelectOp {
    pub peaks_stream: String,
    pub peaks_phase: usize,
    pub peaks_schema: Schema,
    pub background_stream: String,
    pub background_phase: usize,
    pub heights_stream: String,
    pub heights_phase: usize,
    /// Raw C-Blue.
    pub raw: ImageId,
    pub raw_dtype: blockflow::Dtype,
    pub rows_stream: String,
    pub params: BeadSelectParams,
    pub schema: Schema,
}

impl FragmentOp for BeadSelectOp {
    fn name(&self) -> &'static str {
        "bead selection and measurement"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(&self.peaks_stream, self.peaks_phase),
            FragmentInput::own(&self.background_stream, self.background_phase),
            FragmentInput::own(&self.heights_stream, self.heights_phase),
        ]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.rows_stream,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(self.raw, Reach::symmetric([1, 1, 1])).holding(self.raw_dtype)]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let [_, ny, nx] = at.volume();
        // maxima, strongest first, ties in raster order
        let table = rows_of(at, &self.peaks_stream, &self.peaks_schema)?;
        let mut peaks: Vec<([usize; 3], f64)> = Vec::with_capacity(table.len());
        for row in table.scan(&Region::whole(&at.volume()))? {
            peaks.push((row.at(), row.f64(0)?));
        }
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let positions: Vec<[usize; 3]> = peaks.iter().map(|p| p.0).collect();
        let kept = blockflow::classical_centers::suppress_within_distance(
            &positions,
            self.params.min_distance_um,
            self.params.measure.voxel,
        )?;
        let background =
            sampled_thresholds(at, &self.background_stream, GlobalThreshold::median())?[0];
        let surfaces = SurfaceColumns::from_stream(at, &self.heights_stream)?;
        eprintln!(
            "  bead peaks {} → {} after {} µm suppression",
            peaks.len(),
            kept.len(),
            self.params.min_distance_um
        );

        let mut out = SELECT_MAGIC.to_vec();
        put_f64(&mut out, background);
        for k in [0, 1] {
            for y in 0..ny {
                for x in 0..nx {
                    out.extend(surfaces.slice_at(k, y, x).to_le_bytes());
                }
            }
        }
        put_u64(&mut out, kept.len() as u64);
        for i in kept {
            let (p, response) = peaks[i];
            for v in p {
                put_u64(&mut out, v as u64);
            }
            put_f64(&mut out, response);
        }
        Ok(out)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "bead selection reads raw C-Blue through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let [nz, ny, nx] = at.volume();
        let mut r = Bytes::new(at.reduced, SELECT_MAGIC)?;
        let background = r.f64()?;
        let mucus = Array2::from_shape_vec((ny, nx), r.f32s(ny * nx)?).map_err(Error::backend)?;
        let epi = Array2::from_shape_vec((ny, nx), r.f32s(ny * nx)?).map_err(Error::backend)?;
        let heights = HeightMaps { mucus, epi };
        let raw = sources.get(self.raw.index())?.as_array()?.widened();
        let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
        if raw.shape() != at.read.shape3().as_slice() {
            return Err(Error::invalid(
                "bead selection: raw C-Blue block does not cover the read region",
            ));
        }
        // background-subtracted raw at a clamped volume coordinate
        let value = |z: isize, y: isize, x: isize| -> f64 {
            let g = [
                z.clamp(0, nz as isize - 1) as usize,
                y.clamp(0, ny as isize - 1) as usize,
                x.clamp(0, nx as isize - 1) as usize,
            ];
            raw[[g[0] - origin[0], g[1] - origin[1], g[2] - origin[2]]] - background
        };
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        let n = r.u64()? as usize;
        for _ in 0..n {
            let p = [r.u64()? as usize, r.u64()? as usize, r.u64()? as usize];
            let response = r.f64()?;
            if blockflow::ops::owner_of(at.grid, p) != at.index {
                continue;
            }
            let [z, y, x] = p.map(|v| v as isize);
            let peak = value(z, y, x).max(1.0);
            let c = Candidate {
                id: ((p[0] * ny + p[1]) * nx + p[2]) as u64 + 1,
                z: p[0] as f64,
                y: p[1] as f64,
                x: p[2] as f64,
                response,
                raw_intensity: value(z, y, x) + background,
                z_support: (value(z - 1, y, x) + value(z + 1, y, x)) / 2.0 / peak,
                xy_support: (value(z, y - 1, x)
                    + value(z, y + 1, x)
                    + value(z, y, x - 1)
                    + value(z, y, x + 1))
                    / 4.0
                    / peak,
            };
            rows.push(
                p,
                &values_of(&measure_one(&c, &heights, &self.params.measure)),
            )?;
        }
        Ok(BlockOutput::fragment(&self.rows_stream, rows.encode()))
    }
}

// ---------------------------------------------------------------- final --

const FINAL_MAGIC: &[u8; 4] = b"BFi1";

/// Brightness split and large-object flag (see module docs).
pub struct BeadFinalOp {
    pub select_stream: String,
    pub select_phase: usize,
    /// Stable-ID large-object labels.
    pub objects: ImageId,
    pub rows_stream: String,
    /// Fixed brightness split instead of the automatic one.
    pub min_response: Option<f64>,
    pub schema: Schema,
}

/// Object search window around a candidate: ±1 slice, ±3 px.
const OBJECT_REACH: [usize; 3] = [1, 3, 3];

impl FragmentOp for BeadFinalOp {
    fn name(&self) -> &'static str {
        "bead brightness split and object flag"
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        OBJECT_REACH[axis]
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.select_stream, self.select_phase)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.rows_stream,
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(
            self.objects,
            Reach::symmetric(OBJECT_REACH),
        )]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let threshold = match self.min_response {
            Some(t) => t,
            None => {
                let table = rows_of(at, &self.select_stream, &self.schema)?;
                let mut responses = Vec::new();
                for row in table.scan(&Region::whole(&at.volume()))? {
                    if row.u64(5)? == u64::from(Class::Bead.code()) {
                        responses.push(row.f64(1)?);
                    }
                }
                crate::measure::dim_threshold(&responses)?
            }
        };
        eprintln!("  bead response threshold {threshold:.2}");
        let mut out = FINAL_MAGIC.to_vec();
        put_f64(&mut out, threshold);
        Ok(out)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "bead final step reads the object labels through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let threshold = Bytes::new(at.reduced, FINAL_MAGIC)?.f64()?;
        let objects = sources.get(self.objects.index())?.as_array()?.widened();
        let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
        let end = at.read.end();
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        let table = own_rows(at, &self.select_stream, &self.schema)?;
        for row in table.scan(&Region::whole(&at.volume()))? {
            let mut m = measured_from_row(&row)?;
            if m.class == Class::Bead && m.c.response < threshold {
                m.class = Class::Dim;
            }
            let c = row.at();
            // first object voxel in raster order of the window
            let mut id = 0u32;
            'search: for z in c[0].saturating_sub(1)..(c[0] + 2).min(end[0]) {
                for y in c[1].saturating_sub(3)..(c[1] + 4).min(end[1]) {
                    for x in c[2].saturating_sub(3)..(c[2] + 4).min(end[2]) {
                        id = objects[[z - origin[0], y - origin[1], x - origin[2]]] as u32;
                        if id > 0 {
                            break 'search;
                        }
                    }
                }
            }
            if id > 0 {
                m.large_object = id;
                if m.class == Class::Bead {
                    m.class = Class::InLargeObject;
                }
            }
            rows.push(c, &values_of(&m))?;
        }
        Ok(BlockOutput::fragment(&self.rows_stream, rows.encode()))
    }
}

// --------------------------------------------------------------- labels --

/// Every candidate painted as a small ellipsoid labelled with its id, as a
/// `u64` label image. Rejected candidates first, kept beads last, each group
/// strongest first, so overlaps resolve as they did resident. Reads the rows
/// of the neighbouring blocks too: an ellipsoid may cross a seam.
pub struct BeadLabelsOp {
    pub rows: (String, usize),
    pub schema: Schema,
    /// Ellipsoid radii in voxels (z, y, x); at most one block.
    pub radius: [f64; 3],
}

impl FragmentOp for BeadLabelsOp {
    fn name(&self) -> &'static str {
        "bead labels"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: blockflow::Dtype) -> blockflow::Dtype {
        blockflow::Dtype::U64
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.rows.0, self.rows.1).with_reach([1, 1, 1])]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut table = Table::new(at.volume(), self.schema.clone())?;
        for (key, bytes) in at.fragments(&self.rows.0) {
            table.write(key.block, bytes)?;
        }
        table.seal()?;
        let mut rows = table
            .scan(&Region::whole(&at.volume()))?
            .map(|row| measured_from_row(&row))
            .collect::<Result<Vec<_>>>()?;
        rows.sort_by(|a, b| {
            a.kept()
                .cmp(&b.kept())
                .then(b.c.response.total_cmp(&a.c.response))
                .then(
                    (a.c.z, a.c.y, a.c.x)
                        .partial_cmp(&(b.c.z, b.c.y, b.c.x))
                        .unwrap(),
                )
        });
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let mut view = out.view_mut::<u64>()?;
            let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
            let end = at.read.end();
            let [nz, ny, nx] = at.volume();
            let r = self.radius;
            let lo = |v: f64, r: f64, o: usize| ((v - r).floor().max(0.0) as usize).max(o);
            let hi = |v: f64, r: f64, n: usize, e: usize| {
                ((v + r).ceil() as usize).min(n - 1).min(e - 1)
            };
            for m in &rows {
                let c = &m.c;
                for z in lo(c.z, r[0], origin[0])..=hi(c.z, r[0], nz, end[0]) {
                    for y in lo(c.y, r[1], origin[1])..=hi(c.y, r[1], ny, end[1]) {
                        for x in lo(c.x, r[2], origin[2])..=hi(c.x, r[2], nx, end[2]) {
                            let d = ((z as f64 - c.z) / r[0]).powi(2)
                                + ((y as f64 - c.y) / r[1]).powi(2)
                                + ((x as f64 - c.x) / r[2]).powi(2);
                            if d <= 1.0 {
                                view[[z - origin[0], y - origin[1], x - origin[2]]] = c.id;
                            }
                        }
                    }
                }
            }
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}
