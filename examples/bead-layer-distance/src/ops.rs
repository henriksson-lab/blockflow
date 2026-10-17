//! Small stencil ops used to build surface costs.

use blockflow::{Anchor, BlockOp, Dtype, Result, Voxels};

/// A linear filter along z only: `out(z) = sum_i w_i * v(z + o_i)`, indices
/// clamped at the buffer ends. Output is `f32`.
///
/// z grows away from the objective, so:
/// * [`ZTapsOp::edge_cost`] — `-(v(z+k) - v(z-k))`: most negative at a
///   dark-above / bright-below boundary, i.e. a layer's upper surface;
/// * [`ZTapsOp::below`] — `v(z+k)`: what lies just under a candidate surface.
pub struct ZTapsOp {
    name: &'static str,
    taps: Vec<(isize, f64)>,
}

impl ZTapsOp {
    pub fn new(name: &'static str, taps: Vec<(isize, f64)>) -> Self {
        assert!(!taps.is_empty());
        Self { name, taps }
    }

    pub fn edge_cost(name: &'static str, k: usize) -> Self {
        let k = k as isize;
        Self::new(name, vec![(k, -1.0), (-k, 1.0)])
    }

    pub fn below(name: &'static str, k: usize) -> Self {
        Self::new(name, vec![(k as isize, 1.0)])
    }
}

impl BlockOp for ZTapsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        if axis == 0 {
            self.taps
                .iter()
                .map(|(o, _)| o.unsigned_abs())
                .max()
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F32
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let v = input.widened();
        let mut o = out.view_mut::<f32>()?;
        let nz = v.shape()[0] as isize;
        for z in 0..nz {
            let mut dst = o.index_axis_mut(ndarray::Axis(0), z as usize);
            dst.fill(0.0);
            for &(off, w) in &self.taps {
                let src = v.index_axis(ndarray::Axis(0), (z + off).clamp(0, nz - 1) as usize);
                ndarray::Zip::from(&mut dst)
                    .and(&src)
                    .for_each(|d, &s| *d += (w * s) as f32);
            }
        }
        Ok(())
    }
}
