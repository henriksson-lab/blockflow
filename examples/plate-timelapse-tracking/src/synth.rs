// SPDX-License-Identifier: MIT

//! A synthetic multiwell time-lapse with known answers.
//!
//! Each field is a `(t, c, y, x)` `uint16` OME-Zarr image with two levels:
//! channel 0 is a nuclear stain (bright discs), channel 1 a marker whose
//! per-cell intensity oscillates over time, so "measurement over time per
//! cell" has a ground truth too. Cells take random steps, never come closer
//! than a minimum separation, appear, disappear, and leave across the border.
//! The truth for each field — every cell's position and marker level in every
//! frame — is written as CSV beside the plate.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blockflow::zarr_env::Compression;
use blockflow::zarr_layout::StorageLayout;
use blockflow::{Dtype, Error, Result};
use serde_json::json;
use zarrs::array::{Array as ZarrArray, ArraySubset};
use zarrs::filesystem::FilesystemStore;

#[derive(Clone, Debug)]
pub struct SynthSpec {
    pub rows: Vec<String>,
    pub columns: Vec<String>,
    pub fields: usize,
    pub frames: usize,
    pub size: [usize; 2],
    pub cells: usize,
    pub radius: f64,
    /// Largest per-axis step per frame, in pixels.
    pub step: f64,
    pub frame_interval_seconds: f64,
    /// Per-cell, per-frame probability of disappearing in the interior.
    pub disappear: f64,
    /// Global stage drift per frame, `[dy, dx]` pixels: frame `t` shows the
    /// scene shifted by `t · drift`. Truth positions are where cells appear
    /// in the image, so they include it.
    pub drift: [f64; 2],
    /// Per-cell, per-frame probability of dividing into two touching
    /// daughters.
    pub divide: f64,
    /// Per-frame probability that a new cell appears.
    pub appear: f64,
    pub seed: u64,
}

impl Default for SynthSpec {
    fn default() -> Self {
        Self {
            rows: vec!["A".into()],
            columns: vec!["1".into(), "2".into()],
            fields: 1,
            frames: 8,
            size: [160, 160],
            cells: 12,
            radius: 5.0,
            step: 2.0,
            frame_interval_seconds: 300.0,
            disappear: 0.02,
            appear: 0.3,
            drift: [0.0, 0.0],
            divide: 0.0,
            seed: 7,
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

#[derive(Clone, Debug)]
struct Cell {
    id: u64,
    y: f64,
    x: f64,
    marker: f64,
    phase: f64,
    /// The mother, for a daughter's first frame only.
    parent: u64,
}

/// One truth row: a cell in a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct TruthRow {
    pub cell: u64,
    pub t: u64,
    pub y: f64,
    pub x: f64,
    pub marker: f64,
    /// On a daughter's first row: the cell it divided from. 0 otherwise.
    pub parent: u64,
}

/// Write an HCS plate at `root`, and truth CSVs under `root/truth/`.
/// Returns the truth file of each field.
pub fn write_plate(root: &Path, spec: &SynthSpec) -> Result<Vec<PathBuf>> {
    if root.exists() {
        return Err(Error::invalid(format!("{} already exists", root.display())));
    }
    fs::create_dir_all(root.join("truth")).map_err(Error::backend)?;
    let mut wells = Vec::new();
    for (r, row) in spec.rows.iter().enumerate() {
        for (c, column) in spec.columns.iter().enumerate() {
            wells.push(json!({"path": format!("{row}/{column}"), "rowIndex": r, "columnIndex": c}));
        }
    }
    write_group(
        root,
        json!({"ome": {"version": "0.5", "plate": {
            "version": "0.5",
            "name": "synthetic-plate",
            "rows": spec.rows.iter().map(|n| json!({"name": n})).collect::<Vec<_>>(),
            "columns": spec.columns.iter().map(|n| json!({"name": n})).collect::<Vec<_>>(),
            "wells": wells,
            "acquisitions": [{"id": 0, "name": "synthetic"}],
            "field_count": spec.fields,
        }}}),
    )?;
    let mut truth = Vec::new();
    let mut seed = spec.seed;
    for row in &spec.rows {
        write_group(&root.join(row), json!({}))?;
        for column in &spec.columns {
            let well = root.join(row).join(column);
            write_group(
                &well,
                json!({"ome": {"version": "0.5", "well": {
                    "images": (0..spec.fields).map(|f| json!({"path": f.to_string(), "acquisition": 0})).collect::<Vec<_>>()
                }}}),
            )?;
            for field in 0..spec.fields {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let rows = write_image(&well.join(field.to_string()), spec, seed)?;
                let path = root
                    .join("truth")
                    .join(format!("{row}-{column}-f{field}-acq0.csv"));
                write_truth(&path, &rows)?;
                truth.push(path);
            }
        }
    }
    Ok(truth)
}

/// Write one `(t, c, y, x)` field image at `root` and return its truth.
pub fn write_image(root: &Path, spec: &SynthSpec, seed: u64) -> Result<Vec<TruthRow>> {
    let [height, width] = spec.size;
    let mut rng = Rng(seed);
    let separation = 3.0 * spec.radius;
    let margin = spec.radius + 1.0;
    let mut next_id = 1u64;
    let mut daughter_id = 1_000_000u64;
    let mut cells: Vec<Cell> = Vec::new();
    let mut place = |cells: &mut Vec<Cell>, rng: &mut Rng| {
        for _ in 0..200 {
            let y = rng.range(margin, height as f64 - margin);
            let x = rng.range(margin, width as f64 - margin);
            if cells
                .iter()
                .all(|c| ((c.y - y).powi(2) + (c.x - x).powi(2)).sqrt() >= separation)
            {
                cells.push(Cell {
                    id: next_id,
                    y,
                    x,
                    marker: rng.range(80.0, 400.0),
                    phase: rng.range(0.0, std::f64::consts::TAU),
                    parent: 0,
                });
                next_id += 1;
                return;
            }
        }
    };
    for _ in 0..spec.cells {
        place(&mut cells, &mut rng);
    }
    let levels = [[height, width], [height.div_ceil(2), width.div_ceil(2)]];
    let mut level_data: Vec<Vec<u16>> = levels
        .iter()
        .map(|[h, w]| vec![0u16; spec.frames * 2 * h * w])
        .collect();
    let mut truth = Vec::new();
    for t in 0..spec.frames {
        // Render.
        let plane = height * width;
        let mut nuclear = vec![0u16; plane];
        let mut marker = vec![0u16; plane];
        for (i, value) in nuclear.iter_mut().enumerate() {
            *value = 10 + ((i as u64).wrapping_mul(2654435761) % 7) as u16;
        }
        for value in marker.iter_mut() {
            *value = 5;
        }
        for cell in &cells {
            let level = cell.marker * (1.0 + 0.3 * (t as f64 * 0.7 + cell.phase).sin());
            let cy = cell.y + t as f64 * spec.drift[0];
            let cx = cell.x + t as f64 * spec.drift[1];
            if cy < 0.0 || cx < 0.0 || cy >= height as f64 || cx >= width as f64 {
                // Drifted out of view: not in this frame's image or truth.
                continue;
            }
            let y0 = (cy - spec.radius).floor().max(0.0) as usize;
            let y1 = ((cy + spec.radius).ceil() as usize).min(height - 1);
            let x0 = (cx - spec.radius).floor().max(0.0) as usize;
            let x1 = ((cx + spec.radius).ceil() as usize).min(width - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if (y as f64 - cy).powi(2) + (x as f64 - cx).powi(2)
                        <= spec.radius * spec.radius
                    {
                        nuclear[y * width + x] = 200;
                        marker[y * width + x] = level.round() as u16;
                    }
                }
            }
            truth.push(TruthRow {
                cell: cell.id,
                t: t as u64,
                y: cy,
                x: cx,
                marker: level.round(),
                parent: cell.parent,
            });
        }
        for (channel, values) in [nuclear, marker].into_iter().enumerate() {
            let base = (t * 2 + channel) * plane;
            level_data[0][base..base + plane].copy_from_slice(&values);
            let [h, w] = levels[1];
            let base1 = (t * 2 + channel) * h * w;
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0u32;
                    let mut n = 0u32;
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let (yy, xx) = (2 * y + dy, 2 * x + dx);
                        if yy < height && xx < width {
                            sum += u32::from(values[yy * width + xx]);
                            n += 1;
                        }
                    }
                    level_data[1][base1 + y * w + x] = (sum / n) as u16;
                }
            }
        }
        // Move, then divide, remove and add. A move is refused only if it ends
        // closer than the separation to another cell *and* closer than that
        // cell was before, so touching daughters can drift apart.
        for cell in cells.iter_mut() {
            cell.parent = 0;
        }
        let snapshot: Vec<(f64, f64)> = cells.iter().map(|c| (c.y, c.x)).collect();
        for index in 0..cells.len() {
            let y = cells[index].y + rng.range(-spec.step, spec.step);
            let x = cells[index].x + rng.range(-spec.step, spec.step);
            let crowded = snapshot.iter().enumerate().any(|(other, &(oy, ox))| {
                let after = ((oy - y).powi(2) + (ox - x).powi(2)).sqrt();
                let before = ((oy - cells[index].y).powi(2) + (ox - cells[index].x).powi(2)).sqrt();
                other != index && after < separation && after < before
            });
            if !crowded {
                cells[index].y = y;
                cells[index].x = x;
            }
        }
        cells.retain(|c| c.y >= 0.0 && c.x >= 0.0 && c.y < height as f64 && c.x < width as f64);
        let disappear = spec.disappear;
        cells.retain(|_| rng.unit() >= disappear);
        if spec.divide > 0.0 {
            let mut next = Vec::with_capacity(cells.len());
            for cell in cells.drain(..) {
                if rng.unit() < spec.divide {
                    let angle = rng.range(0.0, std::f64::consts::PI);
                    let half = 1.15 * spec.radius;
                    for sign in [-1.0, 1.0] {
                        daughter_id += 1;
                        next.push(Cell {
                            id: daughter_id,
                            y: cell.y + sign * half * angle.sin(),
                            x: cell.x + sign * half * angle.cos(),
                            marker: cell.marker,
                            phase: rng.range(0.0, std::f64::consts::TAU),
                            parent: cell.id,
                        });
                    }
                } else {
                    next.push(cell);
                }
            }
            cells = next;
        }
        if rng.unit() < spec.appear {
            place(&mut cells, &mut rng);
        }
    }

    let datasets: Vec<_> = (0..2)
        .map(|level| {
            let s = (1 << level) as f64;
            json!({"path": level.to_string(), "coordinateTransformations": [
                {"type": "scale", "scale": [spec.frame_interval_seconds, 1.0, 0.65 * s, 0.65 * s]}
            ]})
        })
        .collect();
    write_group(
        root,
        json!({"ome": {"version": "0.5", "multiscales": [{
            "name": "synthetic",
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": datasets
        }]}}),
    )?;
    for (level, [h, w]) in levels.iter().enumerate() {
        let dir = root.join(level.to_string());
        fs::create_dir_all(&dir).map_err(Error::backend)?;
        let shape = [spec.frames, 2, *h, *w];
        let metadata = StorageLayout::chunked(
            vec![1, 1, (*h).min(256), (*w).min(256)],
            Compression::Gzip(1),
        )
        .array_metadata(
            &shape,
            Dtype::U16,
            json!(0),
            Some(json!(["t", "c", "y", "x"])),
        )?;
        fs::write(
            dir.join("zarr.json"),
            serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
        )
        .map_err(Error::backend)?;
        let store = Arc::new(FilesystemStore::new(&dir).map_err(Error::backend)?);
        let array = ZarrArray::open(store, "/").map_err(Error::backend)?;
        let subset = ArraySubset::new_with_start_shape(
            vec![0; 4],
            shape.iter().map(|&n| n as u64).collect(),
        )
        .map_err(Error::backend)?;
        array
            .store_array_subset(&subset, &level_data[level])
            .map_err(Error::backend)?;
    }
    Ok(truth)
}

fn write_group(root: &Path, attributes: serde_json::Value) -> Result<()> {
    fs::create_dir_all(root).map_err(Error::backend)?;
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(
            &json!({"zarr_format": 3, "node_type": "group", "attributes": attributes}),
        )
        .map_err(Error::backend)?,
    )
    .map_err(Error::backend)
}

fn write_truth(path: &Path, rows: &[TruthRow]) -> Result<()> {
    let mut text = String::from("cell_id,t,y,x,marker,parent\n");
    for r in rows {
        text.push_str(&format!(
            "{},{},{:.4},{:.4},{},{}\n",
            r.cell, r.t, r.y, r.x, r.marker, r.parent
        ));
    }
    fs::write(path, text).map_err(Error::backend)
}

pub fn read_truth(path: &Path) -> Result<Vec<TruthRow>> {
    let text = fs::read_to_string(path).map_err(Error::backend)?;
    text.lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let num = |i: usize| -> Result<f64> {
                f.get(i)
                    .and_then(|v| v.parse::<f64>().ok())
                    .ok_or_else(|| Error::invalid(format!("bad truth line {line:?}")))
            };
            Ok(TruthRow {
                cell: num(0)? as u64,
                t: num(1)? as u64,
                y: num(2)?,
                x: num(3)?,
                marker: num(4)?,
                parent: if f.len() > 5 { num(5)? as u64 } else { 0 },
            })
        })
        .collect()
}
