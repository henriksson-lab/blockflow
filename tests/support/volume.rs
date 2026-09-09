use ndarray::Array3;

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
