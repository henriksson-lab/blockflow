//! A backend that finds objects without a network.
//!
//! **This is the one that the tests that matter run against.** What can be
//! silently wrong in this crate is the block decomposition — ownership,
//! numbering, whether a measurement covers the whole object — and none of that
//! is a property of Cellpose or StarDist. A stub that segments deterministically
//! makes every one of those checkable with no GPU, no model file and no feature
//! flag, and makes a failure a statement about this crate rather than about
//! somebody else's weights.
//!
//! It is not a mock in the usual sense: it does real connected-component
//! labelling over a threshold, so it behaves like a segmentation model in the
//! ways the decomposition cares about — an object may straddle a seam, may be
//! partly outside the buffer, may be one voxel or ten thousand.

use crate::error::Result;
use ndarray::{Array3, ArrayView3};

use crate::model_segment::SegmentBackend;

/// Label the connected components of `tile > threshold`.
///
/// Deterministic, position-independent, and — the property the tests rest on —
/// **the same object gets the same voxels whatever tile it is shown in**, as
/// long as the tile contains it. That is what a real backend is *approximately*
/// true of and this one is exactly true of, which is the right split: a test
/// that had to tolerate the network disagreeing with itself between tiles could
/// not tell a decomposition bug from an inference wobble.
pub struct ThresholdBackend {
    threshold: f32,
    /// Which of the twenty-six neighbours count as adjacent. `false` — the
    /// default — is the six that share a face.
    diagonal: bool,
}

impl ThresholdBackend {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            diagonal: false,
        }
    }

    #[must_use]
    pub fn diagonal(mut self, diagonal: bool) -> Self {
        self.diagonal = diagonal;
        self
    }

    fn neighbours(&self) -> Vec<[isize; 3]> {
        if self.diagonal {
            let mut offsets = Vec::with_capacity(26);
            for i in -1isize..=1 {
                for j in -1isize..=1 {
                    for k in -1isize..=1 {
                        if [i, j, k] != [0, 0, 0] {
                            offsets.push([i, j, k]);
                        }
                    }
                }
            }
            offsets
        } else {
            vec![
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ]
        }
    }
}

impl SegmentBackend for ThresholdBackend {
    fn name(&self) -> &'static str {
        "threshold"
    }

    fn cost_per_voxel(&self) -> f64 {
        // A flood fill over the block, which really is a small multiple of an
        // ordinary voxelwise pass. Unlike a network, this one can honestly say so.
        2.0
    }

    fn segment(&self, tile: ArrayView3<'_, f32>, _at: &crate::Anchor) -> Result<Array3<u32>> {
        let (di, dj, dk) = tile.dim();
        let mut labels = Array3::<u32>::zeros((di, dj, dk));
        let offsets = self.neighbours();
        let mut next = 0u32;

        // Iterative flood fill: a segmentation over a whole block can be deeper
        // than a stack is willing to recurse, and a stub that overflows the
        // stack on a large tile would be a test failure about the test.
        let mut stack: Vec<[usize; 3]> = Vec::new();
        for i in 0..di {
            for j in 0..dj {
                for k in 0..dk {
                    if labels[[i, j, k]] != 0 || tile[[i, j, k]] <= self.threshold {
                        continue;
                    }
                    next += 1;
                    labels[[i, j, k]] = next;
                    stack.push([i, j, k]);
                    while let Some(at) = stack.pop() {
                        for offset in &offsets {
                            let Some(near) = step(at, *offset, (di, dj, dk)) else {
                                continue;
                            };
                            if labels[near] != 0 || tile[near] <= self.threshold {
                                continue;
                            }
                            labels[near] = next;
                            stack.push([near[0], near[1], near[2]]);
                        }
                    }
                }
            }
        }
        Ok(labels)
    }
}

/// `at + offset`, if it is inside `shape`.
fn step(at: [usize; 3], offset: [isize; 3], shape: (usize, usize, usize)) -> Option<[usize; 3]> {
    let extent = [shape.0, shape.1, shape.2];
    let mut moved = [0usize; 3];
    for axis in 0..3 {
        let value = at[axis] as isize + offset[axis];
        if value < 0 || value as usize >= extent[axis] {
            return None;
        }
        moved[axis] = value as usize;
    }
    Some(moved)
}
