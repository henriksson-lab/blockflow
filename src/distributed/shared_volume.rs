// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// An `Environment` over files that several processes can open at once.
//
// Why this exists
// ---------------
// The distribution design rests on one property: **a task never needs a peer's
// in-memory output.** A task depends on the previous phase's tasks covering its
// read extent, and satisfies that by *reading storage*. Every existing
// environment in this crate is in-process — `ArrayEnvironment` holds the levels
// in its own memory, `AccountingEnvironment` holds nothing — so neither can
// stand in for that shared storage across a process boundary, and a distributed
// run could not be exercised for real without one.
//
// So this is the smallest thing that is genuinely shared: one file per level,
// positional reads and writes, no locking. It is not a storage layer and does
// not want to become one — a deployment supplies its own environment through
// its own `WorkflowFactory`, and would use a chunked, compressed store.
//
// Why no lock is needed, and what that depends on
// -----------------------------------------------
// **The valid regions of one phase tile the volume exactly.** That is the
// framework's central guard, asserted by `Decomposition::check` and again by
// the executor over what was actually written. It means no two tasks of a phase
// write the same voxel — so no two writers ever touch the same byte, and
// positional writes to disjoint ranges need no coordination at all.
//
// The dependency is worth stating because it is the same hazard a chunked store
// has and solves differently: there, a block boundary *inside* a chunk makes two
// writers collide on a read-modify-write of the whole chunk, which is why block
// size must be a multiple of chunk size. Here the granularity is a byte, so the
// tiling property alone is sufficient.
//
// Durability, and where the barrier goes
// --------------------------------------
// A phase-`p+1` task reads what phase-`p` tasks wrote, possibly on another node.
// The barrier that makes that safe is **not** a global one: a worker flushes its
// level before it reports the task complete, and the coordinator does not release
// a dependent task until every dependency has been reported. Durability is
// therefore a precondition of a completion message rather than a synchronisation
// point, and no worker ever waits for another.
//
// On `f64`
// --------
// This file stores `f64`, and now says so rather than inheriting it. A block's
// element type is a tag (`voxels::Voxels`) and a level may be any of them, but
// the on-disk layout here is a flat array of eight-byte little-endian words —
// the file name says `.f64` and the offset arithmetic multiplies by `BYTES`. So
// a plan whose levels are not `f64` is **refused by `prepare`**, by name, rather
// than silently converted at the seam: a conversion would make the headline
// claim ("N workers produce byte-identical output to a single-node run") a
// statement about a different array than the one the plan describes.
//
// Widening it is a storage change — one file per level at that level's width,
// and the sentinel question `Voxels::unwritten` documents — not a buffer change.
// **Nothing in the distribution protocol is affected either way**: a message
// carries block indices, phases and regions, and never an element.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use ndarray::Array3;

use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::{BlockBuf, EnvCounters, Environment};
use crate::error::{Error, Result};
use crate::geometry::chunks_touched;
use crate::op::{Chain, Placement, SourceInputs};
use crate::region::Region;
use crate::sidecar::{FileSidecars, Sidecars};
use crate::voxels::Voxels;

const BYTES: usize = std::mem::size_of::<f64>();

/// A sentinel for "nobody wrote this", so a voxel a run missed is loud rather
/// than a convincing zero. The same choice `ArrayEnvironment` makes, and the
/// reason a byte-identical comparison between a one-worker run and an N-worker
/// run is a real check: an unwritten voxel differs from a written one.
const UNWRITTEN: f64 = f64::NAN;

#[cfg(unix)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buffer, offset)
}

#[cfg(unix)]
fn write_all_at(file: &File, buffer: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buffer, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buffer.len() {
        let read = file.seek_read(&mut buffer[done..], offset + done as u64)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read",
            ));
        }
        done += read;
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &File, buffer: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buffer.len() {
        done += file.seek_write(&buffer[done..], offset + done as u64)?;
    }
    Ok(())
}

/// One file per storage level, in a directory every worker can see.
pub struct SharedVolumes {
    dir: PathBuf,
    volume: [usize; 3],
    chunk: [usize; 3],
    levels: Vec<File>,
    counters: EnvCounters,
    /// Fragments beside the levels, in the same shared directory and for the
    /// same reason: a task's non-pixel output has to be readable by whichever
    /// process merges, and shared storage is the only thing every process here
    /// has in common. It is what lets a per-block fragment stop being an
    /// in-memory value that pins its stage to one node.
    sidecars: Sidecars,
}

impl SharedVolumes {
    /// Where level `level` lives. Named rather than numbered opaquely so a
    /// directory left behind by a run can be read by anything.
    pub fn level_path(dir: &Path, level: usize) -> PathBuf {
        dir.join(format!("level-{level}.f64"))
    }

    /// Where the sidecar fragments live. Under the job's own directory, so a
    /// worker that can see the levels can see the fragments by construction.
    pub fn sidecar_root(dir: &Path) -> PathBuf {
        dir.join("sidecars")
    }

    /// Create the level files, sized and initialised.
    ///
    /// Level 0 is the workflow input and is left at zero; levels above it are
    /// filled with the sentinel so a voxel nobody writes is detectable.
    pub fn create(
        dir: &Path,
        volume: [usize; 3],
        chunk: [usize; 3],
        n_phases: usize,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .map_err(|err| Error::backend(format!("creating {}: {err}", dir.display())))?;
        let voxels: usize = volume.iter().product();
        for level in 0..=n_phases {
            let path = Self::level_path(dir, level);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .map_err(|err| Error::backend(format!("creating {}: {err}", path.display())))?;
            let fill = if level == 0 { 0.0 } else { UNWRITTEN };
            // Written in slabs rather than one enormous buffer, so the memory
            // cost of creating a volume does not scale with the volume.
            let slab = 1usize << 16;
            let mut buffer = Vec::with_capacity(slab * BYTES);
            for _ in 0..slab {
                buffer.extend_from_slice(&fill.to_le_bytes());
            }
            let mut written = 0usize;
            while written < voxels {
                let take = slab.min(voxels - written);
                write_all_at(&file, &buffer[..take * BYTES], (written * BYTES) as u64)
                    .map_err(|err| Error::backend(format!("sizing {}: {err}", path.display())))?;
                written += take;
            }
        }
        Self::open(dir, volume, chunk, n_phases)
    }

    /// Open level files that already exist. What every worker does.
    pub fn open(
        dir: &Path,
        volume: [usize; 3],
        chunk: [usize; 3],
        n_phases: usize,
    ) -> Result<Self> {
        let mut levels = Vec::with_capacity(n_phases + 1);
        for level in 0..=n_phases {
            let path = Self::level_path(dir, level);
            levels.push(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                    .map_err(|err| {
                        Error::backend(format!(
                            "opening {}: {err}. Every worker must be able to see the job's \
                             directory; on a cluster that means a shared filesystem.",
                            path.display()
                        ))
                    })?,
            );
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            volume,
            chunk,
            levels,
            counters: EnvCounters::default(),
            sidecars: Sidecars::new(FileSidecars::at(Self::sidecar_root(dir))?),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn offset(&self, position: [usize; 3]) -> u64 {
        (((position[0] * self.volume[1]) + position[1]) * self.volume[2] + position[2]) as u64
            * BYTES as u64
    }

    fn level(&self, level: usize) -> Result<&File> {
        self.levels.get(level).ok_or_else(|| {
            Error::invalid(format!(
                "level {level} does not exist; this store has {}",
                self.levels.len()
            ))
        })
    }

    /// The whole of one level, for setting an input up or checking an output.
    pub fn read_level(&self, level: usize) -> Result<Array3<f64>> {
        let region = Region::whole(&self.volume);
        match self.read(level, &region)? {
            BlockBuf::Array(array) => Ok(array.view::<f64>()?.to_owned()),
            BlockBuf::Accounted { .. } => unreachable!("this environment holds arrays"),
        }
    }

    /// Replace the whole of one level.
    pub fn write_level(&self, level: usize, values: &Array3<f64>) -> Result<()> {
        let region = Region::whole(&self.volume);
        self.write(
            level,
            &region,
            &region,
            &BlockBuf::Array(values.clone().into()),
        )
    }

    /// One level's bytes, exactly as they are on disk.
    ///
    /// The comparison the headline verification makes: "N workers produce
    /// byte-identical output to a single-node run" is a statement about bytes,
    /// and comparing decoded arrays would quietly treat two different bit
    /// patterns for the same value — or two sentinels — as equal.
    pub fn level_bytes(&self, level: usize) -> Result<Vec<u8>> {
        let path = Self::level_path(&self.dir, level);
        std::fs::read(&path)
            .map_err(|err| Error::backend(format!("reading {}: {err}", path.display())))
    }
}

impl Environment for SharedVolumes {
    fn volume(&self) -> [usize; 3] {
        self.volume
    }

    fn prepare(&self, decomposition: &Decomposition) -> Result<()> {
        // Every level is a file of one shape, so this environment can host only
        // a plan that keeps one. Refused here rather than in the plan's guard:
        // see `AccountingEnvironment::prepare`.
        let uniform = decomposition.uniform_volume().ok_or_else(|| {
            Error::invalid(format!(
                "shared volumes hold one shape {:?} per level, and this decomposition changes \
                 shape between levels: {:?}",
                self.volume,
                (0..decomposition.n_levels())
                    .map(|level| decomposition.volume_at(level))
                    .collect::<Vec<_>>()
            ))
        })?;
        if uniform != self.volume {
            return Err(Error::invalid(format!(
                "shared volumes hold {:?}, the decomposition is over {:?}",
                self.volume, decomposition.volume
            )));
        }
        if decomposition.n_phases() + 1 != self.levels.len() {
            return Err(Error::invalid(format!(
                "shared volumes were opened for {} phases, the decomposition has {}",
                self.levels.len() - 1,
                decomposition.n_phases()
            )));
        }
        // The on-disk layout is eight-byte words, so a level of any other width
        // has no place to go. Refused here, naming the level, rather than
        // converted at the write — see this file's header.
        for level in 0..=decomposition.n_phases() {
            let dtype = decomposition.dtype_at(level);
            if dtype != Dtype::F64 {
                return Err(Error::invalid(format!(
                    "shared volumes store level {level} as eight-byte words and this \
                     decomposition writes it as {}. Storing a narrower level is a change to \
                     the file layout, not to the buffer.",
                    dtype.numpy_name()
                )));
            }
        }
        Ok(())
    }

    fn read(&self, level: usize, region: &Region) -> Result<BlockBuf> {
        region.check_within(&self.volume, "shared-volume read")?;
        let file = self.level(level)?;
        let shape = [region.shape[0], region.shape[1], region.shape[2]];
        let mut values = vec![0.0f64; shape.iter().product()];
        let run = shape[2];
        let mut raw = vec![0u8; run * BYTES];
        let mut at = 0usize;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                if run > 0 {
                    let offset =
                        self.offset([region.start[0] + i, region.start[1] + j, region.start[2]]);
                    read_exact_at(file, &mut raw, offset).map_err(|err| {
                        Error::backend(format!("reading level {level} at {offset}: {err}"))
                    })?;
                    for (slot, chunk) in raw.chunks_exact(BYTES).enumerate() {
                        values[at + slot] =
                            f64::from_le_bytes(chunk.try_into().expect("eight bytes"));
                    }
                }
                at += run;
            }
        }
        let array = Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
            .map_err(|err| Error::invalid(format!("shaping a block: {err}")))?;
        self.counters.reads.fetch_add(1, Ordering::SeqCst);
        self.counters
            .read_voxels
            .fetch_add(array.len() as u64, Ordering::SeqCst);
        self.counters
            .read_bytes
            .fetch_add((array.len() * BYTES) as u64, Ordering::SeqCst);
        self.counters
            .chunks_read
            .fetch_add(chunks_touched(region, &self.chunk), Ordering::SeqCst);
        Ok(BlockBuf::Array(array.into()))
    }

    fn apply(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
        at: &Placement,
    ) -> Result<BlockBuf> {
        let array = input.as_array()?;
        let stored = crate::env::as_source_arrays(sources)?;
        let mut out = Voxels::zeros(
            slot.produces(array.dtype())?,
            slot.placed_output_shape(array.shape(), at)?,
        )?;
        slot.apply_placed(array, SourceInputs::new(&stored), &mut out, at)?;
        self.counters.ops_applied.fetch_add(1, Ordering::SeqCst);
        self.counters.estimated_work.fetch_add(
            (array.len() as f64 * slot.cost_per_voxel()) as u64,
            Ordering::SeqCst,
        );
        Ok(BlockBuf::Array(out))
    }

    fn write(&self, level: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
        let array = buf.as_array()?.view::<f64>()?;
        valid.check_within(&self.volume, "shared-volume write")?;
        if valid.voxels() == 0 {
            return Ok(());
        }
        let file = self.level(level)?;
        let run = valid.shape[2];
        let mut raw = vec![0u8; run * BYTES];
        for i in 0..valid.shape[0] {
            for j in 0..valid.shape[1] {
                for k in 0..run {
                    let value = array[[
                        within.start[0] + i,
                        within.start[1] + j,
                        within.start[2] + k,
                    ]];
                    raw[k * BYTES..(k + 1) * BYTES].copy_from_slice(&value.to_le_bytes());
                }
                let offset = self.offset([valid.start[0] + i, valid.start[1] + j, valid.start[2]]);
                write_all_at(file, &raw, offset).map_err(|err| {
                    Error::backend(format!("writing level {level} at {offset}: {err}"))
                })?;
            }
        }
        self.counters.writes.fetch_add(1, Ordering::SeqCst);
        self.counters
            .write_voxels
            .fetch_add(valid.voxels() as u64, Ordering::SeqCst);
        self.counters
            .write_bytes
            .fetch_add((valid.voxels() * BYTES) as u64, Ordering::SeqCst);
        Ok(())
    }

    fn uniform(&self, buf: &BlockBuf) -> Option<f64> {
        match buf {
            BlockBuf::Array(array) => array.uniform(),
            BlockBuf::Accounted { uniform, .. } => *uniform,
        }
    }

    fn constant(&self, dtype: Dtype, region: &Region, value: f64) -> Result<BlockBuf> {
        Ok(BlockBuf::Array(Voxels::filled(
            dtype,
            crate::env::block_shape(region)?,
            value,
        )?))
    }

    fn release(&self, _buf: &BlockBuf) {}

    /// Everything written to `level` by *this* worker is durable.
    ///
    /// Called before a task's completion is reported, which is what makes a
    /// dependent task safe to hand to anybody. Not a barrier: it waits for this
    /// process's own writes and for nothing else.
    fn finish(&self, level: usize) -> Result<()> {
        if let Ok(file) = self.level(level) {
            file.sync_data()
                .map_err(|err| Error::backend(format!("flushing level {level}: {err}")))?;
        }
        Ok(())
    }

    fn counters(&self) -> &EnvCounters {
        &self.counters
    }

    fn chunk_shape(&self) -> [usize; 3] {
        self.chunk
    }

    fn sidecars(&self) -> Option<&Sidecars> {
        Some(&self.sidecars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "blockflow-shared-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_region_round_trips_through_a_file() {
        let dir = scratch("round-trip");
        let volume = [4, 3, 5];
        let store = SharedVolumes::create(&dir, volume, [2, 2, 2], 1).unwrap();
        let mut input = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
        for (flat, value) in input.iter_mut().enumerate() {
            *value = flat as f64;
        }
        store.write_level(0, &input).unwrap();
        assert_eq!(store.read_level(0).unwrap(), input);

        let region = Region::new(&[1, 1, 1], &[2, 2, 2]);
        let buf = store.read(0, &region).unwrap();
        store
            .write(1, &Region::new(&[0, 0, 0], &[2, 2, 2]), &region, &buf)
            .unwrap();
        let out = store.read_level(1).unwrap();
        assert_eq!(out[[1, 1, 1]], input[[1, 1, 1]]);
        assert_eq!(out[[2, 2, 2]], input[[2, 2, 2]]);
        assert!(out[[0, 0, 0]].is_nan(), "an unwritten voxel must stay loud");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_handles_on_one_directory_see_each_others_writes() {
        // The whole reason this environment exists: another process reading a
        // level a different process wrote, with no message between them.
        let dir = scratch("two-handles");
        let volume = [8, 2, 2];
        let writer = SharedVolumes::create(&dir, volume, [4, 2, 2], 1).unwrap();
        let reader = SharedVolumes::open(&dir, volume, [4, 2, 2], 1).unwrap();
        let region = Region::new(&[4, 0, 0], &[4, 2, 2]);
        let block = BlockBuf::Array(Array3::from_elem((4, 2, 2), 7.5).into());
        writer
            .write(1, &Region::new(&[0, 0, 0], &[4, 2, 2]), &region, &block)
            .unwrap();
        writer.finish(1).unwrap();
        let seen = reader.read(1, &region).unwrap();
        assert_eq!(seen, block);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn disjoint_writers_do_not_need_a_lock() {
        // The tiling property in action: several threads writing different
        // valid regions of one level, with no coordination, and every byte
        // landing where it should.
        let dir = scratch("disjoint");
        let volume = [16, 2, 2];
        let store = SharedVolumes::create(&dir, volume, [4, 2, 2], 1).unwrap();
        std::thread::scope(|scope| {
            for block in 0..4usize {
                let store = &store;
                scope.spawn(move || {
                    let region = Region::new(&[block * 4, 0, 0], &[4, 2, 2]);
                    let values = Array3::from_elem((4, 2, 2), block as f64);
                    store
                        .write(
                            1,
                            &Region::new(&[0, 0, 0], &[4, 2, 2]),
                            &region,
                            &BlockBuf::Array(values.into()),
                        )
                        .unwrap();
                });
            }
        });
        let out = store.read_level(1).unwrap();
        for block in 0..4usize {
            assert_eq!(out[[block * 4, 0, 0]], block as f64);
            assert_eq!(out[[block * 4 + 3, 1, 1]], block as f64);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
