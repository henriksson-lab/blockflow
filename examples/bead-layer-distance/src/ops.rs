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

/// A layer's upper surface as a relative edge, without the position bias of
/// a log: `-(S(z) - S(z-1)) / max((B(z) + B(z-1)) / 2 - dark, 1)`, where the
/// input `S` is the intensity smoothed at the cost scale and the source `B`
/// the same smoothed broadly. The difference straddles the boundary between
/// slices `z-1` and `z`, so the most negative value is at the first bright
/// slice; dividing by a brightness that barely changes across the edge makes
/// the contrast relative (robust to depth fading and saturation) without
/// moving it. A log, which stretches the dark side, puts the minimum one or
/// two slices above the edge. Output `f32`.
pub struct RelativeEdgeOp {
    name: &'static str,
    broad: blockflow::ImageId,
    dark: f64,
}

impl RelativeEdgeOp {
    pub fn new(name: &'static str, broad: blockflow::ImageId, dark: f64) -> Self {
        Self { name, broad, dark }
    }
}

impl BlockOp for RelativeEdgeOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        usize::from(axis == 0)
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F32
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<blockflow::op::SourceInput> {
        vec![blockflow::op::SourceInput::new(
            self.broad,
            blockflow::Reach::symmetric([1, 0, 0]),
        )]
    }

    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(blockflow::Error::invalid(
            "a relative edge needs its broad source image",
        ))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: blockflow::op::SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let s = input.widened();
        let b = sources.get(self.broad)?.widened();
        if s.shape() != b.shape() {
            return Err(blockflow::Error::invalid(
                "relative edge: the broad image block does not match",
            ));
        }
        let mut o = out.view_mut::<f32>()?;
        let nz = s.shape()[0];
        for z in 0..nz {
            let up = z.saturating_sub(1);
            ndarray::Zip::from(o.index_axis_mut(ndarray::Axis(0), z))
                .and(s.index_axis(ndarray::Axis(0), z))
                .and(s.index_axis(ndarray::Axis(0), up))
                .and(b.index_axis(ndarray::Axis(0), z))
                .and(b.index_axis(ndarray::Axis(0), up))
                .for_each(|d, &s0, &s1, &b0, &b1| {
                    *d = (-(s0 - s1) / ((b0 + b1) / 2.0 - self.dark).max(1.0)) as f32
                });
        }
        Ok(())
    }
}
