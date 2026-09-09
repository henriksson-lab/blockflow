use ndarray::Array3;

use blockflow::geometry::BlockGrid;
use blockflow::synthetic::{Scene, SceneSpec};

pub fn shape3(shape: [usize; 3]) -> (usize, usize, usize) {
    (shape[0], shape[1], shape[2])
}

pub fn scene_intensity(spec: SceneSpec) -> Array3<f64> {
    Scene::new(spec)
        .expect("a valid synthetic scene")
        .render()
        .intensity
}

/// A volume whose every voxel is different, so wrong block placement is visible.
pub fn flat_ramp_f64(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn(shape3(shape), |(z, y, x)| {
        (z * shape[1] * shape[2] + y * shape[2] + x) as f64
    })
}

pub fn xorshift_u16(shape: [usize; 3], seed: u64) -> Array3<u16> {
    let mut state = seed;
    Array3::from_shape_fn(shape3(shape), |_| {
        xorshift_13_7_17(&mut state);
        (state >> 48) as u16
    })
}

pub fn modular_ramp_u16(shape: [usize; 3], domain: usize) -> Array3<u16> {
    assert!(domain > 0, "a modular ramp needs a non-empty domain");
    assert!(
        domain <= u16::MAX as usize + 1,
        "a u16 ramp domain must fit in u16"
    );
    let upper = if domain == u16::MAX as usize + 1 {
        u16::MAX
    } else {
        domain as u16 - 1
    };
    Array3::from_shape_fn(shape3(shape), |(i, j, k)| {
        (((i * 7919 + j * 104729 + k * 1299709) % domain) as u16).min(upper)
    })
}

pub fn xorshift_unit_f64(shape: [usize; 3], seed: u64) -> Array3<f64> {
    let mut state = seed;
    Array3::from_shape_fn(shape3(shape), |_| {
        xorshift_13_7_17(&mut state);
        (state >> 11) as f64 / (1u64 << 53) as f64
    })
}

pub fn sparse_xorshift_bool(shape: [usize; 3], seed: u64, threshold: u8) -> Array3<bool> {
    let mut state = seed;
    Array3::from_shape_fn(shape3(shape), |_| {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        ((state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8) < threshold
    })
}

pub fn point_mask_bool(shape: [usize; 3], points: &[[usize; 3]]) -> Array3<bool> {
    let mut mask = Array3::from_elem(shape3(shape), false);
    for &point in points {
        mask[point] = true;
    }
    mask
}

pub fn fill_box(mask: &mut Array3<bool>, low: [usize; 3], high: [usize; 3], value: bool) {
    for i in low[0]..=high[0] {
        for j in low[1]..=high[1] {
            for k in low[2]..=high[2] {
                mask[[i, j, k]] = value;
            }
        }
    }
}

pub fn every_block(grid: &BlockGrid) -> Vec<[usize; 3]> {
    grid.cores().into_iter().map(|core| core.index).collect()
}

pub fn core_of(grid: &BlockGrid, index: [usize; 3]) -> ([usize; 3], [usize; 3]) {
    let volume = grid.volume();
    let edge = grid.block();
    let mut low = [0usize; 3];
    let mut extent = [0usize; 3];
    for axis in 0..3 {
        low[axis] = index[axis] * edge[axis];
        extent[axis] = edge[axis].min(volume[axis] - low[axis]);
    }
    (low, extent)
}

pub fn core_cut<T: Copy>(volume: &Array3<T>, low: [usize; 3], extent: [usize; 3]) -> Array3<T> {
    Array3::from_shape_fn(shape3(extent), |(i, j, k)| {
        volume[[low[0] + i, low[1] + j, low[2] + k]]
    })
}

pub fn block_local_disagreements<S, T>(
    grid: &BlockGrid,
    input: &Array3<S>,
    global: &Array3<T>,
    locally: impl Fn(&Array3<S>, [usize; 3]) -> Array3<T>,
) -> Vec<[usize; 3]>
where
    S: Copy,
    T: PartialEq,
{
    let mut out = Vec::new();
    for index in every_block(grid) {
        let (low, extent) = core_of(grid, index);
        let cut = core_cut(input, low, extent);
        let local = locally(&cut, extent);
        for i in 0..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    let at = [low[0] + i, low[1] + j, low[2] + k];
                    if local[[i, j, k]] != global[at] {
                        out.push(at);
                    }
                }
            }
        }
    }
    out
}

pub fn grid_sweep(volume: [usize; 3], edges: [usize; 6]) -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(volume, volume).expect("the whole volume is one block"),
        BlockGrid::along(volume, &[0], edges[0]).expect("an axis-0 grid"),
        BlockGrid::along(volume, &[0], edges[1]).expect("a second axis-0 grid"),
        BlockGrid::along(volume, &[1], edges[2]).expect("an axis-1 grid"),
        BlockGrid::along(volume, &[2], edges[3]).expect("an axis-2 grid"),
        BlockGrid::along(volume, &[0, 1], edges[4]).expect("an axis-0/1 grid"),
        BlockGrid::along(volume, &[0, 1, 2], edges[5]).expect("a three-axis grid"),
    ]
}

pub fn standard_grid_sweep(volume: [usize; 3]) -> Vec<BlockGrid> {
    grid_sweep(volume, [4, 8, 4, 5, 4, 4])
}

pub fn sample_offset_grid_sweep(volume: [usize; 3]) -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(volume, volume).expect("the whole volume is one block"),
        BlockGrid::along(volume, &[0], 5).expect("an axis-0 grid before the samples"),
        BlockGrid::along(volume, &[0], 8).expect("an axis-0 grid on the samples"),
        BlockGrid::along(volume, &[0], 9).expect("an axis-0 grid after the samples"),
        BlockGrid::along(volume, &[1], 6).expect("an axis-1 grid"),
        BlockGrid::along(volume, &[2], 4).expect("an axis-2 grid"),
        BlockGrid::along(volume, &[0, 2], 6).expect("an axis-0/2 grid"),
        BlockGrid::along(volume, &[0, 1, 2], 6).expect("a three-axis grid"),
    ]
}

pub fn clipped_start_grid_sweep(volume: [usize; 3]) -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(volume, volume).expect("the whole volume is one block"),
        BlockGrid::along(volume, &[0], 5).expect("an axis-0 grid before the clipped start"),
        BlockGrid::along(volume, &[0], 7).expect("an axis-0 grid on the clipped start"),
        BlockGrid::along(volume, &[0], 8).expect("an axis-0 grid after the clipped start"),
        BlockGrid::along(volume, &[1], 2).expect("an axis-1 grid"),
        BlockGrid::along(volume, &[2], 3).expect("an axis-2 grid"),
        BlockGrid::along(volume, &[0, 1], 4).expect("an axis-0/1 grid"),
        BlockGrid::along(volume, &[0, 1, 2], 3).expect("a three-axis grid"),
    ]
}

pub fn modular_mask_bool(shape: [usize; 3], modulus: usize, residue: usize) -> Array3<bool> {
    assert!(modulus > 0, "a modular mask needs a non-empty modulus");
    let residue = residue % modulus;
    Array3::from_shape_fn(shape3(shape), |(z, y, x)| (z + y + x) % modulus == residue)
}

fn xorshift_13_7_17(state: &mut u64) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
}
