// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Where the alignment reads its tiles from, and how many pairs it measures at
//! once under the memory cap.
//!
//! The alignment never needs a whole tile: the rigid step reads a `depth`-thick
//! slab at each facing end and the per-slice step reads the overlap cuts. So a
//! tile set is a source of **boxes** ([`TileBoxes`]): views into tiles already
//! in memory, or boxes read from the stored images ([`StoredTiles`]), so that
//! only the boxes of the pairs in flight are held.
//!
//! **The active set.** Pairs are measured in parallel, each holding a lease of
//! what it will hold (its boxes, and their `f64` copies and scratch) from the
//! process budget ([`blockflow::budget::process_budget`]) for as long as it
//! runs. So as many pairs are in flight as the memory cap leaves room for, and
//! always one; without a cap that is every worker.

use blockflow::budget::process_budget;
use blockflow::error::{Error, Result};
use blockflow::voxels::{VoxelElement, Voxels};
use blockflow::Dtype;
use ndarray::{s, ArrayView3};

use crate::io::TileInfo;
use crate::planned::TileSource;

/// One box of a tile: a view into a resident tile, or voxels read for it.
pub enum TileBox<'a, T> {
    View(ArrayView3<'a, T>),
    Read(Voxels),
}

impl<T: VoxelElement> TileBox<'_, T> {
    /// The box's voxels, in its own coordinates.
    pub fn view(&self) -> ArrayView3<'_, T> {
        match self {
            Self::View(view) => view.view(),
            Self::Read(voxels) => voxels
                .view::<T>()
                .expect("a read box holds the tiles' element type"),
        }
    }
}

/// A set of tiles the alignment reads boxes of.
pub trait TileBoxes<T>: Sync {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tile `tile`'s extent, `[z, y, x]`.
    fn shape(&self, tile: usize) -> [usize; 3];

    /// The box of `shape` at `start` of tile `tile`.
    fn read(&self, tile: usize, start: [usize; 3], shape: [usize; 3]) -> Result<TileBox<'_, T>>;

    /// What reading a box of `shape` puts in memory: nothing for a view.
    fn read_bytes(&self, shape: [usize; 3]) -> u64;

    /// Tile `tile`'s stored chunk shape and where its coordinates start in
    /// the stored image (a read decodes whole stored chunks): `None` for a
    /// view, which decodes nothing.
    fn chunking(&self, _tile: usize) -> Option<([usize; 3], [usize; 3])> {
        None
    }

    /// How many stored voxels reading the box of `shape` at `start` of tile
    /// `tile` decodes (whole stored chunks): nothing for a view.
    fn decoded(&self, tile: usize, start: [usize; 3], shape: [usize; 3]) -> u64 {
        let Some((chunk, offset)) = self.chunking(tile) else {
            return 0;
        };
        (0..3)
            .map(|a| {
                let c = chunk[a].max(1);
                let lo = (start[a] + offset[a]) / c * c;
                let hi = (start[a] + shape[a] + offset[a]).div_ceil(c) * c;
                (hi - lo) as u64
            })
            .product()
    }

    /// The most the pairs in flight may hold between them, beside the memory
    /// cap: for stored tiles, what holding every tile would take, so reading
    /// boxes never holds more than the resident alignment held.
    fn ceiling(&self) -> u64 {
        u64::MAX
    }
}

impl<T: VoxelElement> TileBoxes<T> for [ArrayView3<'_, T>] {
    fn len(&self) -> usize {
        <[ArrayView3<'_, T>]>::len(self)
    }

    fn shape(&self, tile: usize) -> [usize; 3] {
        let shape = self[tile].shape();
        [shape[0], shape[1], shape[2]]
    }

    fn read(&self, tile: usize, start: [usize; 3], shape: [usize; 3]) -> Result<TileBox<'_, T>> {
        let stop: [usize; 3] = std::array::from_fn(|a| start[a] + shape[a]);
        let extent = TileBoxes::<T>::shape(self, tile);
        if (0..3).any(|a| stop[a] > extent[a]) {
            return Err(Error::invalid(format!(
                "box {start:?} + {shape:?} is outside tile {tile} of {extent:?}"
            )));
        }
        Ok(TileBox::View(self[tile].slice_move(s![
            start[0]..stop[0],
            start[1]..stop[1],
            start[2]..stop[2]
        ])))
    }

    fn read_bytes(&self, _shape: [usize; 3]) -> u64 {
        0
    }
}

/// Tiles in their stored images, read box by box.
#[derive(Debug, Clone)]
pub struct StoredTiles {
    pub sources: Vec<TileSource>,
    /// Each tile's extent (its planes in the stitched range).
    pub shapes: Vec<[usize; 3]>,
    pub dtype: Dtype,
}

impl StoredTiles {
    /// Every tile of `tiles` (as [`crate::io::open_tile`] opens them), checked
    /// to share one element type.
    pub fn new(tiles: &[TileInfo]) -> Result<Self> {
        let first = tiles.first().ok_or_else(|| Error::invalid("no tiles"))?;
        let dtype = first.dtype;
        if let Some(index) = tiles.iter().position(|tile| tile.dtype != dtype) {
            return Err(Error::invalid(format!(
                "tile {index} holds {} and tile 0 holds {}; the tiles must share one element type",
                tiles[index].dtype.numpy_name(),
                dtype.numpy_name()
            )));
        }
        Ok(Self {
            sources: tiles.iter().map(|tile| tile.source.clone()).collect(),
            shapes: tiles.iter().map(|tile| tile.shape).collect(),
            dtype,
        })
    }
}

impl<T: VoxelElement> TileBoxes<T> for StoredTiles {
    fn len(&self) -> usize {
        self.sources.len()
    }

    fn shape(&self, tile: usize) -> [usize; 3] {
        self.shapes[tile]
    }

    fn read(&self, tile: usize, start: [usize; 3], shape: [usize; 3]) -> Result<TileBox<'_, T>> {
        let extent = self.shapes[tile];
        if (0..3).any(|a| start[a] + shape[a] > extent[a]) {
            return Err(Error::invalid(format!(
                "box {start:?} + {shape:?} is outside tile {tile} of {extent:?}"
            )));
        }
        let source = &self.sources[tile];
        let at: [usize; 3] = std::array::from_fn(|a| start[a] + source.offset[a]);
        let voxels = source.image.read_region(at, shape)?;
        if voxels.dtype() != T::DTYPE {
            return Err(Error::invalid(format!(
                "tile {tile} holds {} and was read as {}",
                voxels.dtype().numpy_name(),
                T::DTYPE.numpy_name()
            )));
        }
        Ok(TileBox::Read(voxels))
    }

    fn read_bytes(&self, shape: [usize; 3]) -> u64 {
        (shape.iter().product::<usize>() * self.dtype.size_of().max(1)) as u64
    }

    fn chunking(&self, tile: usize) -> Option<([usize; 3], [usize; 3])> {
        let source = &self.sources[tile];
        Some((source.chunk, source.offset))
    }

    fn ceiling(&self) -> u64 {
        self.shapes
            .iter()
            .map(|&shape| TileBoxes::<T>::read_bytes(self, shape))
            .sum()
    }
}

/// A box, `(start, shape)` in tile coordinates.
pub type BoxAt = ([usize; 3], [usize; 3]);

/// The smallest box holding both.
pub fn bounding(a: BoxAt, b: BoxAt) -> BoxAt {
    let start: [usize; 3] = std::array::from_fn(|d| a.0[d].min(b.0[d]));
    let shape = std::array::from_fn(|d| (a.0[d] + a.1[d]).max(b.0[d] + b.1[d]) - start[d]);
    (start, shape)
}

/// `range` cut into pieces of whole stored chunks of `tile` along `axis`,
/// each about `bytes_per_row`-times-its-rows no more than `target` bytes (one
/// chunk at least): read piece by piece, each stored chunk is still decoded
/// once. A view is one piece.
pub fn chunk_pieces<T, B: TileBoxes<T> + ?Sized>(
    tiles: &B,
    tile: usize,
    axis: usize,
    range: std::ops::Range<usize>,
    bytes_per_row: u64,
    target: u64,
) -> Vec<std::ops::Range<usize>> {
    let Some((chunk, offset)) = tiles.chunking(tile) else {
        return vec![range];
    };
    let c = chunk[axis].max(1);
    let step = c * ((target / (bytes_per_row.max(1) * c as u64)).max(1) as usize);
    let mut pieces = Vec::new();
    let mut at = range.start;
    while at < range.end {
        let stored = at + offset[axis];
        let stop = ((stored / step + 1) * step - offset[axis]).min(range.end);
        pieces.push(at..stop);
        at = stop;
    }
    pieces
}

/// A box read for one step and kept for a later one.
pub struct KeptBox<'a, T> {
    pub at: BoxAt,
    pub data: TileBox<'a, T>,
}

impl<T: VoxelElement> KeptBox<'_, T> {
    /// Whether the box holds all of `inner`.
    pub fn covers(&self, inner: BoxAt) -> bool {
        (0..3).all(|d| {
            self.at.0[d] <= inner.0[d] && inner.0[d] + inner.1[d] <= self.at.0[d] + self.at.1[d]
        })
    }

    /// `inner` (which the box covers), in its own coordinates.
    pub fn view_of(&self, inner: BoxAt) -> ArrayView3<'_, T> {
        let from: [usize; 3] = std::array::from_fn(|d| inner.0[d] - self.at.0[d]);
        self.data.view().slice_move(s![
            from[0]..from[0] + inner.1[0],
            from[1]..from[1] + inner.1[1],
            from[2]..from[2] + inner.1[2]
        ])
    }
}

/// The two boxes one pair's rigid step read and its per-slice step will cut,
/// and the lease of the memory they hold until then.
pub struct KeptPair<'a, T> {
    pub pre: KeptBox<'a, T>,
    pub post: KeptBox<'a, T>,
    pub lease: blockflow::budget::Lease,
}

/// Run `work` on every item in parallel, each holding a lease of
/// `bytes(item)` from the process budget while it runs, the items in flight
/// holding no more than `ceiling` between them, and return the answers in
/// item order.
///
/// Items are admitted in order from this thread as room frees, never from
/// inside a worker: a worker that waited for room could be one that a running
/// item's nested parallel work is waiting on. An item with nothing else in
/// flight runs whatever it costs.
pub fn admitted<I, R>(
    items: Vec<I>,
    bytes: impl Fn(&I) -> u64 + Sync,
    ceiling: u64,
    work: impl Fn(I) -> Result<R> + Sync,
) -> Result<Vec<R>>
where
    I: Send,
    R: Send,
{
    type Done<R> = (usize, std::thread::Result<Result<R>>);
    let budget = process_budget();
    let workers = rayon::current_num_threads().max(1);
    let total = items.len();
    let mut answers: Vec<Option<R>> = (0..total).map(|_| None).collect();
    let mut failure: Option<(usize, Error)> = None;
    let mut panicked = None;
    let work = &work;
    rayon::in_place_scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::channel::<Done<R>>();
        let mut queue = items.into_iter().enumerate().peekable();
        let mut in_flight = 0usize;
        // What the items in flight were admitted at, by index.
        let mut held: Vec<u64> = vec![0; total];
        let mut holding = 0u64;
        loop {
            while in_flight < workers && failure.is_none() && panicked.is_none() {
                let Some((_, item)) = queue.peek() else {
                    break;
                };
                let want = bytes(item);
                if in_flight > 0 && holding.saturating_add(want) > ceiling {
                    break;
                }
                let lease = match budget.try_acquire_reclaiming(want) {
                    Some(lease) => lease,
                    None if in_flight == 0 => blockflow::budget::Lease::unbudgeted(),
                    None => break,
                };
                let (index, item) = queue.next().expect("peeked");
                let sender = sender.clone();
                in_flight += 1;
                held[index] = want;
                holding += want;
                scope.spawn(move |_| {
                    let answer =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(item)));
                    drop(lease);
                    let _ = sender.send((index, answer));
                });
            }
            if in_flight == 0 {
                break;
            }
            let (index, answer) = wait(&receiver);
            in_flight -= 1;
            holding -= held[index];
            match answer {
                Ok(Ok(answer)) => answers[index] = Some(answer),
                Ok(Err(error)) => {
                    if failure.as_ref().is_none_or(|(first, _)| index < *first) {
                        failure = Some((index, error));
                    }
                }
                Err(payload) => {
                    panicked.get_or_insert(payload);
                }
            }
        }
    });
    if let Some(payload) = panicked {
        std::panic::resume_unwind(payload);
    }
    if let Some((_, error)) = failure {
        return Err(error);
    }
    Ok(answers
        .into_iter()
        .map(|answer| answer.expect("every item answered"))
        .collect())
}

/// The next finished item. On a thread of a pool the wait runs that pool's
/// queued work rather than parking a thread the items may need.
fn wait<T>(receiver: &std::sync::mpsc::Receiver<T>) -> T {
    loop {
        if let Ok(done) = receiver.try_recv() {
            return done;
        }
        match rayon::yield_now() {
            Some(rayon::Yield::Executed) => {}
            Some(rayon::Yield::Idle) => {
                if let Ok(done) = receiver.recv_timeout(std::time::Duration::from_micros(100)) {
                    return done;
                }
            }
            None => {
                return receiver
                    .recv()
                    .expect("a sender outlives every item in flight")
            }
        }
    }
}
