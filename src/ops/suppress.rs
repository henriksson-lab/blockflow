// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Greedy suppression of nearby points, as a kernel and as a planned row op.**
//
// Walk points strongest first; keep one unless it lies within a threshold
// distance of a point already kept. The threshold is a pair rule over the two
// points' radii, so one kernel covers the shapes callers ask for:
//
// | caller wants | radius | pair | factor |
// |---|---|---|---|
// | a fixed minimum separation `d` | `Fixed(d)` | any but `Sum` | 1 |
// | non-maximum suppression of scale-space peaks | the peak's scale | `Smaller` | e.g. 0.75 |
// | drop an object centred inside a better one | the object's radius | `Larger` | e.g. 0.9 |
// | drop touching or overlapping spheres | the radius | `Sum` | 1 |
//
// The answer is the all-pairs greedy one and depends only on the priority
// order, which is total: priority descending, then coordinate ascending. A grid
// of cells as wide as the largest threshold limits each comparison to the kept
// points in the 27 neighbouring cells, so the cost is linear in the points.
// Positions are fractional voxels and may come from columns, so a fitted
// centre that moved off its row's voxel is compared where it is.
//
// The op is a barrier because a point's fate depends on kept points in any
// block: `reduce` reads every row, runs the kernel once and returns the kept
// coordinates; each block then emits its own rows that were kept. Row tables
// hold at most one row per voxel, so a coordinate names a row. The output is a
// subsequence of the input, as for `rows::FilterRowsOp`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, FragmentInput, FragmentOp, FragmentOutput, PhaseView, SeamFold,
};
use crate::ops::rows::{sealed_rows, RowStreams};
use crate::table::{RowBuilder, Schema, Value};

const KEPT_MAGIC: u64 = 0x5355_5050_4b45_5031;

/// How two radii combine into the distance below which the weaker point goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiusPair {
    Smaller,
    Larger,
    Sum,
}

impl RadiusPair {
    fn of(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Smaller => a.min(b),
            Self::Larger => a.max(b),
            Self::Sum => a + b,
        }
    }
}

/// Where a row's radius comes from, in the same physical units as `voxel_size`.
#[derive(Debug, Clone, PartialEq)]
pub enum SuppressionRadius {
    Fixed(f64),
    Column(String),
}

/// The full rule for [`SuppressRowsOp`].
#[derive(Debug, Clone, PartialEq)]
pub struct Suppression {
    /// Higher is stronger. `U64` or `F64`.
    pub priority: String,
    pub radius: SuppressionRadius,
    pub pair: RadiusPair,
    /// Multiplies the paired radius. Zero disables suppression.
    pub factor: f64,
    /// Columns holding each row's `[z, y, x]` position in voxels, for rows whose
    /// position is finer than (or moved from) the voxel the row is keyed at,
    /// such as a fitted centre. `None` uses the row's own voxel.
    pub position: Option<[String; 3]>,
    /// Physical size of a voxel, `[z, y, x]`.
    pub voxel_size: [f64; 3],
}

fn check_spacing(voxel_size: [f64; 3]) -> Result<()> {
    if voxel_size.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::invalid("voxel spacing must be finite and positive"));
    }
    Ok(())
}

/// Greedy suppression of `positions`, given in priority order (strongest
/// first): a point is dropped when it lies closer than
/// `factor * pair(r_kept, r_point)` to a kept point, distances in physical
/// units through `voxel_size`. Returns the kept indices in priority order.
pub fn suppress_points(
    positions: &[[f64; 3]],
    radii: &[f64],
    pair: RadiusPair,
    factor: f64,
    voxel_size: [f64; 3],
) -> Result<Vec<usize>> {
    if positions.len() != radii.len() {
        return Err(Error::invalid(format!(
            "suppression was given {} positions and {} radii",
            positions.len(),
            radii.len()
        )));
    }
    if !factor.is_finite() || factor < 0.0 {
        return Err(Error::invalid(
            "suppression needs a finite non-negative factor",
        ));
    }
    if radii.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return Err(Error::invalid(
            "suppression radii must be finite and non-negative",
        ));
    }
    check_spacing(voxel_size)?;
    let largest = radii.iter().copied().fold(0.0, f64::max);
    let reach = factor * pair.of(largest, largest);
    if reach == 0.0 {
        return Ok((0..positions.len()).collect());
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::invalid("suppression positions must be finite"));
    }
    // Cells in physical units, signed: a fitted centre may sit just outside the
    // volume.
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    let mut kept = Vec::new();
    for (index, &at) in positions.iter().enumerate() {
        let home: [i64; 3] =
            std::array::from_fn(|a| (at[a] * voxel_size[a] / reach).floor() as i64);
        let mut separated = true;
        'cells: for dz in -1..=1i64 {
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let key = [home[0] + dz, home[1] + dy, home[2] + dx];
                    for &other in grid.get(&key).map(Vec::as_slice).unwrap_or_default() {
                        let limit = factor * pair.of(radii[other], radii[index]);
                        let squared = (0..3)
                            .map(|axis| {
                                let d = (at[axis] - positions[other][axis]) * voxel_size[axis];
                                d * d
                            })
                            .sum::<f64>();
                        if squared < limit * limit {
                            separated = false;
                            break 'cells;
                        }
                    }
                }
            }
        }
        if separated {
            grid.entry(home).or_default().push(index);
            kept.push(index);
        }
    }
    Ok(kept)
}

/// **Rows in, the rows that survive greedy suppression out.**
///
/// A barrier: `reduce` reads every block's rows, orders them by priority
/// (descending, ties by coordinate) and runs [`suppress_points`]; each block
/// then emits the kept rows it holds, unchanged.
pub struct SuppressRowsOp {
    name: &'static str,
    rows: RowStreams,
    rule: Suppression,
    priority: usize,
    radius: Option<usize>,
    position: Option<[usize; 3]>,
}

impl SuppressRowsOp {
    /// Refuses columns the input rows do not have, and a bad rule, at
    /// construction rather than on the first block.
    pub fn new(name: &'static str, rows: RowStreams, rule: Suppression) -> Result<Self> {
        let column = |name: &str| {
            rows.schema.index_of(name).ok_or_else(|| {
                Error::invalid(format!(
                    "suppression names column {name:?}, which the input rows do not have"
                ))
            })
        };
        let priority = column(&rule.priority)?;
        let radius = match &rule.radius {
            SuppressionRadius::Fixed(r) => {
                if !r.is_finite() || *r < 0.0 {
                    return Err(Error::invalid(
                        "a fixed suppression radius must be finite and non-negative",
                    ));
                }
                None
            }
            SuppressionRadius::Column(name) => Some(column(name)?),
        };
        let position = match &rule.position {
            None => None,
            Some([z, y, x]) => Some([column(z)?, column(y)?, column(x)?]),
        };
        if !rule.factor.is_finite() || rule.factor < 0.0 {
            return Err(Error::invalid(
                "suppression needs a finite non-negative factor",
            ));
        }
        check_spacing(rule.voxel_size)?;
        Ok(Self {
            name,
            rows,
            rule,
            priority,
            radius,
            position,
        })
    }

    /// The schema of the rows this op emits: its input's, unchanged.
    pub fn schema(&self) -> &Schema {
        &self.rows.schema
    }
}

fn number(value: Value) -> f64 {
    match value {
        Value::U64(v) => v as f64,
        Value::F64(v) => v,
    }
}

fn encode_kept(kept: &[[usize; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + kept.len() * 24);
    out.extend_from_slice(&KEPT_MAGIC.to_le_bytes());
    out.extend_from_slice(&(kept.len() as u64).to_le_bytes());
    for at in kept {
        for v in at {
            out.extend_from_slice(&(*v as u64).to_le_bytes());
        }
    }
    out
}

fn decode_kept(bytes: &[u8]) -> Result<Vec<[usize; 3]>> {
    let words: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
        .collect();
    let valid = bytes.len() % 8 == 0
        && words.len() >= 2
        && words[0] == KEPT_MAGIC
        && words.len() as u64 == 2 + 3 * words[1];
    if !valid {
        return Err(Error::invalid("not a suppression result"));
    }
    Ok(words[2..]
        .chunks_exact(3)
        .map(|w| [w[0] as usize, w[1] as usize, w[2] as usize])
        .collect())
}

impl FragmentOp for SuppressRowsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        self.rows.inputs()
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        self.rows.outputs()
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let volume = at.grid.volume();
        // (keyed voxel, position, priority, radius)
        let mut points: Vec<([usize; 3], [f64; 3], f64, f64)> = Vec::new();
        at.stream_fragments(&self.rows.input, &mut |_key, bytes| {
            for row in sealed_rows(volume, &self.rows.schema, bytes)? {
                let priority = number(row.values[self.priority]);
                if !priority.is_finite() {
                    return Err(Error::invalid(format!(
                        "row at {:?} has a non-finite priority {priority}",
                        row.at
                    )));
                }
                let radius = match (&self.rule.radius, self.radius) {
                    (_, Some(column)) => number(row.values[column]),
                    (SuppressionRadius::Fixed(r), None) => *r,
                    (SuppressionRadius::Column(_), None) => {
                        unreachable!("resolved at construction")
                    }
                };
                let position = match self.position {
                    Some(columns) => columns.map(|c| number(row.values[c])),
                    None => row.at.map(|v| v as f64),
                };
                points.push((row.at, position, priority, radius));
            }
            Ok(())
        })?;
        points.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        let positions: Vec<[f64; 3]> = points.iter().map(|p| p.1).collect();
        let radii: Vec<f64> = points.iter().map(|p| p.3).collect();
        let mut kept: Vec<[usize; 3]> = suppress_points(
            &positions,
            &radii,
            self.rule.pair,
            self.rule.factor,
            self.rule.voxel_size,
        )?
        .into_iter()
        .map(|i| points[i].0)
        .collect();
        kept.sort_unstable();
        Ok(encode_kept(&kept))
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let kept = decode_kept(at.reduced)?;
        let mut out = RowBuilder::new(Arc::new(self.rows.schema.clone()));
        for row in sealed_rows(at.volume(), &self.rows.schema, self.rows.own(at))? {
            if kept.binary_search(&row.at).is_ok() {
                out.push(row.at, &row.values)?;
            }
        }
        Ok(self.rows.fragment(out.encode()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::PlanBuilder;
    use crate::dtype::Dtype;
    use crate::env::ArrayEnvironment;
    use crate::geometry::BlockGrid;
    use crate::ops::rows::{collect_rows, RowSourceOp, RowValues};
    use crate::sidecar::Lifecycle;
    use crate::strategy::{execute_phases, Hints};
    use crate::table::Column;
    use crate::voxels::Voxels;
    use ndarray::Array3;

    fn random(seed: &mut u64, m: usize) -> usize {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as usize % m
    }

    fn all_pairs(
        positions: &[[f64; 3]],
        radii: &[f64],
        pair: RadiusPair,
        factor: f64,
        voxel: [f64; 3],
    ) -> Vec<usize> {
        let mut kept: Vec<usize> = Vec::new();
        for (i, p) in positions.iter().enumerate() {
            let far = kept.iter().all(|&j| {
                let q = positions[j];
                let d2: f64 = (0..3).map(|a| ((p[a] - q[a]) * voxel[a]).powi(2)).sum();
                d2 >= (factor * pair.of(radii[j], radii[i])).powi(2)
            });
            if far {
                kept.push(i);
            }
        }
        kept
    }

    /// The grid keeps exactly what the all-pairs walk keeps, for every pair
    /// rule and with radii that vary by an order of magnitude.
    #[test]
    fn matches_all_pairs_for_every_rule() {
        let mut seed = 7u64;
        let voxel = [2.0, 0.5, 0.5];
        for pair in [RadiusPair::Smaller, RadiusPair::Larger, RadiusPair::Sum] {
            for factor in [0.0, 0.75, 1.0, 1.6] {
                // fractional and partly negative, as fitted centres can be
                let positions: Vec<[f64; 3]> = (0..600)
                    .map(|_| {
                        [
                            random(&mut seed, 400) as f64 / 10.0 - 3.0,
                            random(&mut seed, 1600) as f64 / 10.0,
                            random(&mut seed, 1600) as f64 / 10.0 - 5.0,
                        ]
                    })
                    .collect();
                let radii: Vec<f64> = (0..600)
                    .map(|_| 0.5 + random(&mut seed, 100) as f64 / 10.0)
                    .collect();
                assert_eq!(
                    suppress_points(&positions, &radii, pair, factor, voxel).unwrap(),
                    all_pairs(&positions, &radii, pair, factor, voxel),
                    "{pair:?} x {factor}"
                );
            }
        }
    }

    /// The pair rule decides, not the order alone: a small strong point next to
    /// a large weak one.
    #[test]
    fn pair_rule_decides() {
        let positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]];
        let radii = [2.0, 10.0];
        let one = [1.0; 3];
        let keep = |pair| suppress_points(&positions, &radii, pair, 1.0, one).unwrap();
        assert_eq!(keep(RadiusPair::Smaller), vec![0, 1]);
        assert_eq!(keep(RadiusPair::Larger), vec![0]);
        assert_eq!(keep(RadiusPair::Sum), vec![0]);
    }

    /// Planned over several lattices, the op keeps what the kernel keeps on the
    /// whole table, including pairs that straddle block seams.
    #[test]
    fn planned_matches_kernel_across_seams() {
        let volume = [12, 40, 40];
        let schema = Schema::new(vec![Column::f64("score"), Column::f64("radius")]).unwrap();
        let mut seed = 3u64;
        let mut rows: Vec<RowValues> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        while rows.len() < 120 {
            let at = [
                random(&mut seed, 12),
                random(&mut seed, 40),
                random(&mut seed, 40),
            ];
            if seen.insert(at) {
                rows.push(RowValues::new(
                    at,
                    vec![
                        Value::F64(random(&mut seed, 1000) as f64),
                        Value::F64(1.0 + random(&mut seed, 40) as f64 / 10.0),
                    ],
                ));
            }
        }
        let voxel = [1.0, 1.0, 1.0];
        let mut order: Vec<&RowValues> = rows.iter().collect();
        order.sort_by(|a, b| {
            number(b.values[0])
                .total_cmp(&number(a.values[0]))
                .then_with(|| a.at.cmp(&b.at))
        });
        let positions: Vec<[f64; 3]> = order.iter().map(|r| r.at.map(|v| v as f64)).collect();
        let radii: Vec<f64> = order.iter().map(|r| number(r.values[1])).collect();
        let mut expected: Vec<[usize; 3]> =
            suppress_points(&positions, &radii, RadiusPair::Larger, 0.9, voxel)
                .unwrap()
                .into_iter()
                .map(|i| order[i].at)
                .collect();
        expected.sort_unstable();
        assert!(
            expected.len() < rows.len() && expected.len() > 10,
            "fixture suppresses some"
        );

        for block in [[12, 40, 40], [4, 16, 16], [3, 7, 9]] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
            let source = builder
                .fragments(RowSourceOp::new(
                    "rows",
                    "rows",
                    Lifecycle::DeleteOnExit,
                    schema.clone(),
                    rows.clone(),
                ))
                .unwrap();
            let streams = RowStreams::new(
                "rows",
                source.index(),
                "kept",
                Lifecycle::Persistent,
                schema.clone(),
            )
            .unwrap();
            let rule = Suppression {
                priority: "score".into(),
                radius: SuppressionRadius::Column("radius".into()),
                pair: RadiusPair::Larger,
                factor: 0.9,
                voxel_size: voxel,
                position: None,
            };
            let phase = builder
                .fragments(SuppressRowsOp::new("suppress", streams, rule).unwrap())
                .unwrap();
            let assembly = builder.finish().unwrap();
            let env = ArrayEnvironment::for_decomposition(
                Voxels::from(Array3::<f64>::zeros((volume[0], volume[1], volume[2]))),
                &assembly.decomposition,
                block,
            )
            .unwrap();
            execute_phases(
                "suppress",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints::default(),
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            let got: Vec<[usize; 3]> =
                collect_rows(&env, "kept", phase.index(), volume, schema.clone())
                    .unwrap()
                    .into_iter()
                    .map(|r| r.at)
                    .collect();
            assert_eq!(got, expected, "block {block:?}");
        }
    }
}
