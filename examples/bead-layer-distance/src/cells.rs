//! Large bright AF488 objects inside the mucus: probably dying (shed) cells,
//! to be ignored. They are only *flagged* — published as their own label
//! layer with a table — so a person can review them in a viewer.
//!
//! Planned (assembled in `main`): AF488 and C-Blue are smoothed; both are
//! sampled inside the mucus band on a sparse lattice; [`CellMaskOp`] keeps
//! voxels bright in AF488 (robust level) and not bright in C-Blue (beads are
//! bright in AF488 too); blockflow labels the components across blocks;
//! [`ObjectStatsOp`] sums each component per block; [`ObjectFilterOp`] keeps
//! components of at least the minimum volume and renames them by their first
//! voxel, which is stable across runs.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use blockflow::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SourceBlocks,
};
use blockflow::op::SourceInput;
use blockflow::ops::{sampled_thresholds, GlobalThreshold};
use blockflow::sidecar::Lifecycle;
use blockflow::table::{Column, RowBuilder, Schema, Value};
use blockflow::{Dtype, Error, ImageId, Result};

use crate::beads::{put_f64, put_u64, Bytes};

pub struct LargeObject {
    pub id: u32,
    pub voxels: u64,
    pub volume_um3: f64,
    pub centroid: [f64; 3],
    pub mean_af488: f64,
    pub mean_blue: f64,
    /// Candidates that touch it.
    pub n_beads: u32,
}

// ----------------------------------------------------------------- mask --

const MASK_MAGIC: &[u8; 4] = b"CMa1";

/// `smoothed AF488 > level && smoothed C-Blue <= level && inside the band`.
pub struct CellMaskOp {
    pub af488_samples: (String, usize),
    pub blue_samples: (String, usize),
    pub af488: ImageId,
    pub blue: ImageId,
    pub band: ImageId,
    pub k_mad: f64,
}

impl FragmentOp for CellMaskOp {
    fn name(&self) -> &'static str {
        "large AF488 object mask"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(&self.af488_samples.0, self.af488_samples.1),
            FragmentInput::own(&self.blue_samples.0, self.blue_samples.1),
        ]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        [self.af488, self.blue, self.band]
            .into_iter()
            .map(SourceInput::voxelwise)
            .collect()
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let rule = GlobalThreshold::robust_sigma(self.k_mad)?;
        let af488 = sampled_thresholds(at, &self.af488_samples.0, rule)?[0];
        let blue = sampled_thresholds(at, &self.blue_samples.0, rule)?[0];
        eprintln!("  large-object AF488 threshold {af488:.1}, bead C-Blue exclusion {blue:.1}");
        let mut out = MASK_MAGIC.to_vec();
        put_f64(&mut out, af488);
        put_f64(&mut out, blue);
        Ok(out)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "the object mask reads its images through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let mut r = Bytes::new(at.reduced, MASK_MAGIC)?;
        let (t_af, t_blue) = (r.f64()?, r.f64()?);
        let af = sources.get(self.af488.index())?.as_array()?.widened();
        let blue = sources.get(self.blue.index())?.as_array()?.widened();
        let band = sources.get(self.band.index())?.as_array()?.widened();
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let mut view = out.view_mut::<bool>()?;
            ndarray::Zip::from(&mut view)
                .and(&af)
                .and(&blue)
                .and(&band)
                .for_each(|o, &a, &b, &m| *o = a > t_af && b <= t_blue && m != 0.0);
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

// ---------------------------------------------------------------- stats --

const STATS_MAGIC: &[u8; 4] = b"CSt1";

#[derive(Clone, Copy, Default)]
struct Partial {
    count: u64,
    sum: [f64; 3],
    af488: f64,
    blue: f64,
    first: u64,
}

impl Partial {
    fn merge(&mut self, o: &Partial) {
        if self.count == 0 {
            self.first = o.first;
        }
        self.count += o.count;
        for a in 0..3 {
            self.sum[a] += o.sum[a];
        }
        self.af488 += o.af488;
        self.blue += o.blue;
        self.first = self.first.min(o.first);
    }

    fn put(&self, out: &mut Vec<u8>) {
        put_u64(out, self.count);
        for v in self.sum {
            put_f64(out, v);
        }
        put_f64(out, self.af488);
        put_f64(out, self.blue);
        put_u64(out, self.first);
    }

    fn get(r: &mut Bytes<'_>) -> Result<Self> {
        Ok(Self {
            count: r.u64()?,
            sum: [r.f64()?, r.f64()?, r.f64()?],
            af488: r.f64()?,
            blue: r.f64()?,
            first: r.u64()?,
        })
    }
}

/// Per block, per component label: voxel count, coordinate sums, raw AF488
/// and C-Blue sums and the least linear index. All sums are of integers, so
/// they are exact in any order.
pub struct ObjectStatsOp {
    pub stream: String,
    pub raw_af488: SourceInput,
    pub raw_blue: SourceInput,
}

impl FragmentOp for ObjectStatsOp {
    fn name(&self) -> &'static str {
        "large AF488 object sums"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.stream,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![self.raw_af488.clone(), self.raw_blue.clone()]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "object sums read the raw channels through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let [_, ny, nx] = at.volume();
        let labels = at.pixels()?.as_array()?;
        let labels = labels.view::<u32>()?;
        let af = sources
            .get(self.raw_af488.image.index())?
            .as_array()?
            .widened();
        let blue = sources
            .get(self.raw_blue.image.index())?
            .as_array()?
            .widened();
        let o = [at.read.start[0], at.read.start[1], at.read.start[2]];
        let mut parts: BTreeMap<u32, Partial> = BTreeMap::new();
        for ((z, y, x), &l) in labels.indexed_iter() {
            if l == 0 {
                continue;
            }
            let g = [o[0] + z, o[1] + y, o[2] + x];
            let linear = ((g[0] * ny + g[1]) * nx + g[2]) as u64;
            let p = parts.entry(l).or_insert(Partial {
                first: u64::MAX,
                ..Partial::default()
            });
            p.count += 1;
            for (sum, &v) in p.sum.iter_mut().zip(&g) {
                *sum += v as f64;
            }
            p.af488 += af[[z, y, x]];
            p.blue += blue[[z, y, x]];
            p.first = p.first.min(linear);
        }
        let mut out = STATS_MAGIC.to_vec();
        put_u64(&mut out, parts.len() as u64);
        for (l, p) in &parts {
            put_u64(&mut out, u64::from(*l));
            p.put(&mut out);
        }
        Ok(BlockOutput::fragment(&self.stream, out))
    }
}

// --------------------------------------------------------------- filter --

const FILTER_MAGIC: &[u8; 4] = b"CFi1";

pub fn object_schema() -> Result<Schema> {
    Schema::new(vec![
        Column::u64("id"),
        Column::u64("voxels"),
        Column::f64("centroid_z"),
        Column::f64("centroid_y"),
        Column::f64("centroid_x"),
        Column::f64("mean_af488"),
        Column::f64("mean_cblue"),
    ])
}

/// Keep components of at least `min_voxels`, rename them `first voxel + 1`,
/// write the renamed labels and one row per kept object (from the block that
/// holds its first voxel).
pub struct ObjectFilterOp {
    pub stats: (String, usize),
    /// Global component labels.
    pub components: ImageId,
    pub min_volume_um3: f64,
    pub voxel_volume: f64,
    pub rows_stream: String,
    pub schema: Schema,
}

impl FragmentOp for ObjectFilterOp {
    fn name(&self) -> &'static str {
        "large AF488 object filter"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        // published as is: OME-Zarr label pyramids here are uint64
        Dtype::U64
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.stats.0, self.stats.1)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.rows_stream,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.components)]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut all: BTreeMap<u64, Partial> = BTreeMap::new();
        at.stream_fragments(&self.stats.0, &mut |_key, bytes| {
            let mut r = Bytes::new(bytes, STATS_MAGIC)?;
            for _ in 0..r.u64()? {
                let l = r.u64()?;
                let p = Partial::get(&mut r)?;
                all.entry(l).or_default().merge(&p);
            }
            Ok(())
        })?;
        let kept: Vec<(u64, Partial)> = all
            .into_iter()
            .filter(|(_, p)| p.count as f64 * self.voxel_volume >= self.min_volume_um3)
            .collect();
        eprintln!("  large AF488 objects: {} kept", kept.len());
        let mut out = FILTER_MAGIC.to_vec();
        put_u64(&mut out, kept.len() as u64);
        for (l, p) in &kept {
            put_u64(&mut out, *l);
            p.put(&mut out);
        }
        Ok(out)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "the object filter reads the component labels through apply_with",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let [_, ny, nx] = at.volume();
        let mut r = Bytes::new(at.reduced, FILTER_MAGIC)?;
        let n = r.u64()? as usize;
        let mut rename: HashMap<u32, u64> = HashMap::with_capacity(n);
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        for _ in 0..n {
            let l = r.u64()? as u32;
            let p = Partial::get(&mut r)?;
            let id =
                u32::try_from(p.first + 1).map_err(|_| Error::invalid("object id exceeds u32"))?;
            rename.insert(l, u64::from(id));
            let first = [
                p.first as usize / (ny * nx),
                (p.first as usize / nx) % ny,
                p.first as usize % nx,
            ];
            if blockflow::ops::owner_of(at.grid, first) == at.index {
                let c = p.count as f64;
                rows.push(
                    first,
                    &[
                        Value::U64(u64::from(id)),
                        Value::U64(p.count),
                        Value::F64(p.sum[0] / c),
                        Value::F64(p.sum[1] / c),
                        Value::F64(p.sum[2] / c),
                        Value::F64(p.af488 / c),
                        Value::F64(p.blue / c),
                    ],
                )?;
            }
        }
        let components = sources.get(self.components.index())?.as_array()?;
        let components = components.view::<u32>()?;
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            let mut view = out.view_mut::<u64>()?;
            ndarray::Zip::from(&mut view)
                .and(&components)
                .for_each(|o, &l| *o = rename.get(&l).copied().unwrap_or(0));
        }
        Ok(BlockOutput::fragment(&self.rows_stream, rows.encode()).with_pixels(buffer))
    }
}

// ------------------------------------------------------- bead counts --

const COUNTS_MAGIC: &[u8; 4] = b"COc1";

/// The object rows with an eighth column, `n_beads`: how many bead candidates
/// touch each object (counted over every candidate row in `reduce`).
pub fn final_object_schema() -> Result<Schema> {
    let mut columns: Vec<Column> = object_schema()?.columns().to_vec();
    columns.push(Column::u64("n_beads"));
    Schema::new(columns)
}

pub struct ObjectCountsOp {
    pub objects: (String, usize),
    pub beads: (String, usize),
    pub bead_schema: Schema,
    /// Column of `large_object` in the bead rows.
    pub bead_object_column: usize,
    pub object_schema: Schema,
    pub rows_stream: String,
    pub schema: Schema,
}

impl FragmentOp for ObjectCountsOp {
    fn name(&self) -> &'static str {
        "large AF488 object bead counts"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![
            FragmentInput::own(&self.objects.0, self.objects.1),
            FragmentInput::own(&self.beads.0, self.beads.1),
        ]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            &self.rows_stream,
            Lifecycle::Persistent,
            Coverage::EveryBlock,
        )]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let beads = crate::beads::rows_of(at, &self.beads.0, &self.bead_schema)?;
        let mut counts: BTreeMap<u64, u64> = BTreeMap::new();
        for row in beads.scan(&blockflow::region::Region::whole(&at.volume()))? {
            let id = row.u64(self.bead_object_column)?;
            if id > 0 {
                *counts.entry(id).or_default() += 1;
            }
        }
        let mut out = COUNTS_MAGIC.to_vec();
        put_u64(&mut out, counts.len() as u64);
        for (id, n) in counts {
            put_u64(&mut out, id);
            put_u64(&mut out, n);
        }
        Ok(out)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut r = Bytes::new(at.reduced, COUNTS_MAGIC)?;
        let mut counts = HashMap::new();
        for _ in 0..r.u64()? {
            let id = r.u64()?;
            counts.insert(id, r.u64()?);
        }
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        let own = crate::beads::own_rows(at, &self.objects.0, &self.object_schema)?;
        for row in own.scan(&blockflow::region::Region::whole(&at.volume()))? {
            let id = row.u64(0)?;
            let mut values = Vec::with_capacity(8);
            for (i, column) in self.object_schema.columns().iter().enumerate() {
                values.push(match column.kind() {
                    blockflow::table::ColumnType::U64 => Value::U64(row.u64(i)?),
                    blockflow::table::ColumnType::F64 => Value::F64(row.f64(i)?),
                });
            }
            values.push(Value::U64(counts.get(&id).copied().unwrap_or(0)));
            rows.push(row.at(), &values)?;
        }
        Ok(BlockOutput::fragment(&self.rows_stream, rows.encode()))
    }
}

/// One final object row (with `n_beads`) back into an object.
pub fn object_of(values: &[Value], voxel: [f64; 3]) -> Result<LargeObject> {
    let f = |i: usize| match values[i] {
        Value::F64(v) => Ok(v),
        Value::U64(_) => Err(Error::invalid("object row: expected f64")),
    };
    let u = |i: usize| match values[i] {
        Value::U64(v) => Ok(v),
        Value::F64(_) => Err(Error::invalid("object row: expected u64")),
    };
    Ok(LargeObject {
        id: u(0)? as u32,
        voxels: u(1)?,
        volume_um3: u(1)? as f64 * voxel.iter().product::<f64>(),
        centroid: [f(2)?, f(3)?, f(4)?],
        mean_af488: f(5)?,
        mean_blue: f(6)?,
        n_beads: u(7)? as u32,
    })
}

pub fn write_csv(objects: &[LargeObject], voxel: [f64; 3], path: &Path) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).map_err(Error::backend)?);
    writeln!(
        w,
        "id,rejected,n_beads,voxels,volume_um3,z,y,x,z_um,y_um,x_um,mean_af488,mean_cblue"
    )
    .map_err(Error::backend)?;
    for o in objects {
        let c = o.centroid;
        writeln!(
            w,
            "{},{},{},{},{:.0},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            o.id,
            0,
            o.n_beads,
            o.voxels,
            o.volume_um3,
            c[0],
            c[1],
            c[2],
            c[0] * voxel[0],
            c[1] * voxel[1],
            c[2] * voxel[2],
            o.mean_af488,
            o.mean_blue
        )
        .map_err(Error::backend)?;
    }
    w.flush().map_err(Error::backend)
}
