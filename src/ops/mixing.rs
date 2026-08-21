// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A **K-ary reach-0 op**: K co-located arrays in, K' co-located arrays out,
// every value read at the voxel it is written to and nowhere else.
//
// Why this is one shape rather than several
// -----------------------------------------
// The operations that want all of several arrays at once want them **at one
// voxel**. A per-voxel linear map, a per-voxel classification, an argmax over K
// arrays, a colour-space conversion, a windowed filter along a series read as K
// separate volumes — every one of them is `K` values in, `K'` values out, reach
// zero. What they need is *arity*, which is a different mechanism from a
// *window*: a window is what `Reach` describes and would need an axis to be
// stated over; arity needs nothing but somewhere for the operands to come from.
//
// So this module is a shell — [`TupleOp`] — and a kernel trait —
// [`TupleKernel`] — and the first kernel, [`LinearMap`], which is the matrix
// case: `out[o] = sum_c M[o][c] * in[c]`.
//
// Why the extra inputs are images and the extra outputs are side outputs
// ----------------------------------------------------------------------
// `BlockOp::source_inputs` names stored images a phase reads besides the one it
// is handed, each read at the block's own fetch region — which is exactly reach
// zero. `BlockOp::side_outputs` declares arrays the op writes beside its primary
// result, each with its own name, element type and rank. Neither is new here.
// What *was* missing until images could be supplied is anything for the extra
// inputs to point at that the run did not compute itself.
//
// It is a `BlockOp` and not a `Combine` for a checkable reason: the `Combine`
// trait has no side outputs, so a fan-in cannot write more than one array.
//
// Layout: K separate volumes is the right one, and not by default
// ---------------------------------------------------------------
// The K inputs arrive as K separate buffers, one per array. For this kernel that
// is the layout a vector unit wants: load a vector of **positions** from one
// input, broadcast the scalar coefficient, fused-multiply-add into the
// accumulator. No shuffles, no horizontal reductions, no transposes. The
// alternative — K values of one position adjacent in memory — would put the
// reduction *across* a vector register and need a horizontal add per position.
//
// The arithmetic is about two flops per byte at `f32`, so this is
// streaming-bound rather than compute-bound and the quantity to watch is the
// number of concurrent streams: `K` reads plus `K'` writes, which is 32 at
// `K = K' = 16` and at the edge of what a hardware prefetcher tracks. That is
// what [`TILE`] is for.

use ndarray::ArrayD;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Output, SideBlock, SourceInput, SourceInputs};
use crate::region::Region;
use crate::voxels::Voxels;

/// Positions processed per pass of the output loop.
///
/// The kernel is written output-major — for each output, accumulate over every
/// input — because that keeps one accumulator live at a time rather than `K'` of
/// them, which is what a vector unit has registers for. Done over a whole block
/// that would read every input once per output; done over a **tile** it reads
/// them once per output *from cache*, and the traffic to memory is `K` reads and
/// `K'` writes, each advancing one tile at a time.
///
/// **Measured, and the effect is large.** At `K = K' = 16` in `f32` over a
/// `[128, 128, 32]` block on a Xeon Gold 6138, one pass per output over the
/// whole block runs at 3.8–4.4 Gflop/s and the tiled form at 8.5–12.9 — a factor
/// of **2.5 to 2.8**, and it is entirely traffic: untiled, each of the 16 output
/// passes re-reads all 16 inputs from memory. Across four sweeps under different
/// machine load the figures at tiles of 64 / 256 / 1024 / 4096 were 7.5–10.3 /
/// 8.5–12.9 / 9.0–12.1 / 9.0–12.3 Gflop/s: the *absolute* number moves with what
/// else is running, the shape of the curve does not, and it is flat from a few
/// hundred positions upward. Any value in that range is the same answer; only
/// "no tiling at all" is a different one.
///
/// 1024 positions is 64 KiB of inputs and 64 KiB of outputs at `K = K' = 16` in
/// `f32`, which is a second-level cache. It is a constant rather than a
/// parameter because it is a property of the machine and not of the plan; the
/// number to change it to would come from a measurement, not from a caller, and
/// `tests/tuple_map.rs` is the measurement.
const TILE: usize = 1024;

/// What a K-ary reach-0 op computes at one position: `K` values in, `K'` out.
///
/// The kernel sees **contiguous slices of positions**, one per array, all of the
/// same length — not one position at a time. That is what lets the inner loop
/// vectorise: a per-position callback would be a function call per voxel with
/// nothing for the optimiser to widen.
///
/// Two element types rather than one generic method, because the trait is used
/// behind a `dyn` and a generic method is not object-safe. `f32` and `f64` are
/// the two an arithmetic kernel is worth writing; anything else is refused by
/// [`Self::accepts`] before a run starts rather than converted silently.
pub trait TupleKernel: Send + Sync {
    fn name(&self) -> &'static str;

    /// How many arrays it reads. Must be at least one.
    fn n_inputs(&self) -> usize;

    /// How many arrays it writes. Must be at least one.
    fn n_outputs(&self) -> usize;

    /// Which element types it will run in.
    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::F32 | Dtype::F64)
    }

    /// What it writes, given what it reads. The default hands the type on.
    fn produces(&self, input: Dtype) -> Dtype {
        input
    }

    fn cost_per_voxel(&self) -> f64;

    /// `inputs[c]` and `outputs[o]` are contiguous, and every one of them is
    /// `len` long.
    ///
    /// **`outputs[o]` receives output `from + o`, not output `o`.** The shell
    /// asks for a *window* of the outputs rather than always for the whole
    /// tuple, because the primary result and the arrays beside it are collected
    /// by two different calls and neither is meant to compute the other's. A
    /// kernel that ignored `from` would fill the second call's buffers with the
    /// first call's answers: the same count, the same shapes, the same range of
    /// values, every one of them one row wrong.
    ///
    /// `from + outputs.len()` is never more than [`Self::n_outputs`]. A kernel
    /// whose outputs are computed jointly may compute all of them and write only
    /// the window; what it may not do is write a different one.
    fn apply_f32(&self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], from: usize, len: usize);

    /// The same, in double precision.
    fn apply_f64(&self, inputs: &[&[f64]], outputs: &mut [&mut [f64]], from: usize, len: usize);
}

// ------------------------------------------------------------- the shell --

/// One [`BlockOp`] reading `K` arrays at reach 0 and writing `K'`.
///
/// Input 0 is the image the phase is handed; inputs `1..K` are the images named
/// in `sources`, read through `BlockOp::source_inputs` at the block's own fetch
/// region. Output 0 is the phase's own image; outputs `1..K'` are the side
/// outputs named in `side_names`.
///
/// # Where the outputs beside the primary are computed, and what that costs
///
/// [`BlockOp::apply_side`] is handed the operands [`BlockOp::apply_with`] is
/// handed: the buffer the op read, the [`SourceInputs`] the phase read beside
/// it, and the primary result. That is what a K-ary op needs and what it did not
/// have when this module was written — output `o` is a function of all `K`
/// inputs, and a call holding one of them could not compute a single one of
/// them.
///
/// So each call computes its own and nothing is carried between them:
/// `apply_with` asks the kernel for **output 0 alone**, `apply_side` asks it for
/// **outputs `1..K'`**, and each is one tiled pass over the `K` inputs. The
/// window is what the kernel's `from` is for — see [`TupleKernel::apply_f32`] —
/// and it is what keeps every output computed exactly once per block.
///
/// **What the split costs, measured.** The inputs are streamed twice rather than
/// once, so a block's traffic goes from `K` reads and `K'` writes to `2K` reads
/// and `K'` writes. At `K = K' = 16` in `f32` over a `[128, 128, 32]` block, one
/// fused pass over the whole tuple ran at 22.3–22.6 ms and the split at
/// 24.3–24.6 ms: **1.08–1.10x**, against the **2.5–2.8x** the tiling itself is
/// worth. The tiling is the part the split keeps — both passes are tiled, and it
/// is a pass *per output* that gives the 2.5–2.8x back, not a second pass.
///
/// It is the shape that is worth the 8–10%, not tidiness. What it replaced was a
/// per-block map from an output buffer's offset to the arrays computed for it,
/// filled by `apply_with` and drained by `apply_side`: two calls agreeing out of
/// band about which block they were on, arrays resident between them that no
/// counter knew about — in a crate whose side outputs exist because 95.2 MB was
/// counted against 158.6 MB written — and a refusal by name whenever the pair
/// came apart.
pub struct TupleOp {
    name: &'static str,
    kernel: Box<dyn TupleKernel>,
    sources: Vec<usize>,
    side_names: Vec<String>,
    /// What every array this op reads holds. One type for all of them: the
    /// kernel sees `K` slices of one element type, and a per-input type would
    /// be `K` kernels.
    dtype: Dtype,
}

impl TupleOp {
    /// `sources` are the images of inputs `1..K` and `side_names` the names of
    /// outputs `1..K'`, both in the kernel's own order.
    ///
    /// Fallible, because the two lists have to agree with the kernel's arity and
    /// a kernel asked for a value it has no coefficient for would produce a
    /// well-formed, complete, wrong volume.
    pub fn new(
        name: &'static str,
        kernel: Box<dyn TupleKernel>,
        sources: Vec<usize>,
        side_names: Vec<String>,
        dtype: Dtype,
    ) -> Result<Self> {
        if sources.len() + 1 != kernel.n_inputs() {
            return Err(Error::InvalidArgument(format!(
                "{name}: kernel {:?} reads {} array(s) and was given {} source image(s). Input 0 \
                 is the image the phase is handed, so the list names inputs 1..{}.",
                kernel.name(),
                kernel.n_inputs(),
                sources.len(),
                kernel.n_inputs()
            )));
        }
        if side_names.len() + 1 != kernel.n_outputs() {
            return Err(Error::InvalidArgument(format!(
                "{name}: kernel {:?} writes {} array(s) and was given {} side-output name(s). \
                 Output 0 is the image the phase writes, so the list names outputs 1..{}.",
                kernel.name(),
                kernel.n_outputs(),
                side_names.len(),
                kernel.n_outputs()
            )));
        }
        if !kernel.accepts(dtype) {
            return Err(Error::InvalidArgument(format!(
                "{name}: kernel {:?} does not run in {}",
                kernel.name(),
                dtype.numpy_name()
            )));
        }
        let mut seen = sources.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != sources.len() {
            return Err(Error::InvalidArgument(format!(
                "{name}: the source images {sources:?} name one image twice. One buffer is \
                 fetched per image per block, so two inputs reading it would be the same \
                 numbers under two coefficients — say so with a coefficient instead."
            )));
        }
        Ok(Self {
            name,
            kernel,
            sources,
            side_names,
            dtype,
        })
    }

    /// The `K` buffers, input 0 first, checked to be co-located and contiguous.
    fn gather<'a>(&self, input: &'a Voxels, sources: SourceInputs<'a>) -> Result<Vec<&'a Voxels>> {
        let mut held = Vec::with_capacity(self.sources.len() + 1);
        held.push(input);
        for &image in &self.sources {
            let array = sources.get(image)?;
            if array.shape() != input.shape() {
                return Err(Error::InvalidArgument(format!(
                    "{}: the image it was handed is {:?} and image {image} came back {:?}. A \
                     reach-0 operand is read at the block's own fetch region, so the two are the \
                     same extent or the positions do not correspond.",
                    self.name,
                    input.shape(),
                    array.shape()
                )));
            }
            if array.dtype() != self.dtype {
                return Err(Error::InvalidArgument(format!(
                    "{}: image {image} holds {} and this op reads {}. Every array a kernel sees \
                     is one element type.",
                    self.name,
                    array.dtype().numpy_name(),
                    self.dtype.numpy_name()
                )));
            }
            held.push(array);
        }
        Ok(held)
    }

    /// Run the kernel over every position of the block, tiled.
    ///
    /// `outputs[o]` receives output `from + o` and is of the block's extent.
    /// Both callers pass a window: `apply_with` asks for output 0, `apply_side`
    /// for outputs `1..K'`, and neither computes the other's.
    fn run(&self, inputs: &[&Voxels], outputs: &mut [&mut Voxels], from: usize) -> Result<()> {
        let len: usize = inputs[0].shape().iter().product();
        match self.dtype {
            Dtype::F32 => {
                let held: Vec<&[f32]> = inputs
                    .iter()
                    .map(|array| contiguous::<f32>(array, self.name))
                    .collect::<Result<_>>()?;
                let mut written: Vec<&mut [f32]> = Vec::with_capacity(outputs.len());
                for out in outputs.iter_mut() {
                    written.push(contiguous_mut::<f32>(out, self.name)?);
                }
                for start in (0..len).step_by(TILE) {
                    let span = TILE.min(len - start);
                    let tile_in: Vec<&[f32]> = held
                        .iter()
                        .map(|slice| &slice[start..start + span])
                        .collect();
                    let mut tile_out: Vec<&mut [f32]> = written
                        .iter_mut()
                        .map(|slice| &mut slice[start..start + span])
                        .collect();
                    self.kernel.apply_f32(&tile_in, &mut tile_out, from, span);
                }
            }
            Dtype::F64 => {
                let held: Vec<&[f64]> = inputs
                    .iter()
                    .map(|array| contiguous::<f64>(array, self.name))
                    .collect::<Result<_>>()?;
                let mut written: Vec<&mut [f64]> = Vec::with_capacity(outputs.len());
                for out in outputs.iter_mut() {
                    written.push(contiguous_mut::<f64>(out, self.name)?);
                }
                for start in (0..len).step_by(TILE) {
                    let span = TILE.min(len - start);
                    let tile_in: Vec<&[f64]> = held
                        .iter()
                        .map(|slice| &slice[start..start + span])
                        .collect();
                    let mut tile_out: Vec<&mut [f64]> = written
                        .iter_mut()
                        .map(|slice| &mut slice[start..start + span])
                        .collect();
                    self.kernel.apply_f64(&tile_in, &mut tile_out, from, span);
                }
            }
            other => {
                return Err(Error::InvalidArgument(format!(
                    "{}: no kernel lane for {}",
                    self.name,
                    other.numpy_name()
                )))
            }
        }
        Ok(())
    }
}

/// One array's positions as one contiguous slice.
///
/// Every buffer this op is handed is a fresh block, so it is in standard layout
/// and this is a borrow rather than a copy. Refused by name if it ever is not,
/// because a kernel indexing a non-contiguous buffer by position would read the
/// right number of values from the wrong places.
fn contiguous<'a, T: crate::voxels::VoxelElement>(
    array: &'a Voxels,
    name: &str,
) -> Result<&'a [T]> {
    array.view::<T>()?.to_slice().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{name}: a buffer of {:?} is not contiguous, and this kernel walks positions in \
             storage order",
            array.shape()
        ))
    })
}

fn contiguous_mut<'a, T: crate::voxels::VoxelElement>(
    array: &'a mut Voxels,
    name: &str,
) -> Result<&'a mut [T]> {
    let shape = array.shape();
    array.view_mut::<T>()?.into_slice().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{name}: an output buffer of {shape:?} is not contiguous"
        ))
    })
}

impl BlockOp for TupleOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero, on every axis and at every volume — this is the whole claim the
    /// shape makes. Stated rather than defaulted for the reason every reach is:
    /// a silent zero is the one value that produces a complete, well-formed,
    /// wrong volume, and here it happens to be the true one.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.sources
            .iter()
            .map(|&image| SourceInput::voxelwise(image).holding(self.dtype))
            .collect()
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == self.dtype && self.kernel.accepts(dtype)
    }

    fn produces(&self, input: Dtype) -> Dtype {
        self.kernel.produces(input)
    }

    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(Error::InvalidArgument(format!(
            "{}: this op reads {} array(s) and was applied with none of them. A K-ary op runs \
             through `apply_with`, which is what the executor calls once a phase records the \
             images it reads.",
            self.name,
            self.sources.len() + 1
        )))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let inputs = self.gather(input, sources)?;
        let shape = input.shape();
        if out.shape() != shape {
            return Err(Error::InvalidArgument(format!(
                "{}: reach 0, so the output is the extent of the input — {shape:?} — and it was \
                 handed {:?}",
                self.name,
                out.shape()
            )));
        }
        // Output 0, and none of the rest. `apply_side` is handed these same
        // operands and computes outputs `1..K'` from them, so computing them
        // here would be computing them twice — see the type's own documentation
        // for what the split costs and what it buys.
        self.run(&inputs, &mut [out], 0)
    }

    fn side_outputs(&self, volume: [usize; 3]) -> Vec<Output> {
        self.side_names
            .iter()
            .map(|name| Output::new(name.clone(), self.produces(self.dtype), &volume))
            .collect()
    }

    fn side_region(&self, _which: usize, valid: &Region, _volume: [usize; 3]) -> Result<Region> {
        Ok(valid.clone())
    }

    fn apply_side(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        _primary: &Voxels,
        block: &SideBlock<'_>,
    ) -> Result<Vec<ArrayD<f64>>> {
        if self.side_names.is_empty() {
            return Ok(Vec::new());
        }
        // The operands, gathered again and checked again: this call is a total
        // function of them, so an image of the wrong extent or the wrong element
        // type is refused here for the same reason it is refused in
        // `apply_with`, and not because the other call happened to run first.
        let inputs = self.gather(input, sources)?;
        let mut buffers: Vec<Voxels> = (0..self.side_names.len())
            .map(|_| Voxels::zeros(self.produces(self.dtype), input.shape()))
            .collect::<Result<_>>()?;
        {
            // Outputs `1..K'` in **one** tiled pass over the inputs. One pass
            // per output would read every input once per output, which is the
            // 2.5-2.8x this shape exists not to pay; see [`TILE`].
            let mut written: Vec<&mut Voxels> = buffers.iter_mut().collect();
            self.run(&inputs, &mut written, 1)?;
        }
        // The trustworthy sub-box: a side output is written once per output
        // position and the halo is read, not written. Reach 0 makes the two the
        // same here, and slicing is still what the contract asks for rather than
        // something to be right about by accident.
        Ok(buffers
            .iter()
            .map(|array| {
                let widened = array.widened().into_dyn();
                let mut view = widened.view();
                for (axis, (&start, &len)) in block
                    .within
                    .start
                    .iter()
                    .zip(block.within.shape.iter())
                    .enumerate()
                {
                    view.slice_axis_inplace(
                        ndarray::Axis(axis),
                        ndarray::Slice::from(start..start + len),
                    );
                }
                view.to_owned()
            })
            .collect())
    }

    fn cost_per_voxel(&self) -> f64 {
        self.kernel.cost_per_voxel()
    }
}

// ------------------------------------------------------- the first kernel --

/// A **mixing matrix**: `out[o] = sum_c M[o][c] * in[c]`, at every position.
///
/// `K'` rows of `K` coefficients, row-major. The matrix mixes *across* the
/// arrays, which is the whole of what it is for — a diagonal matrix would be `K`
/// independent scalings and would need none of this.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearMap {
    name: &'static str,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
    cost: f64,
}

impl LinearMap {
    /// `rows` outputs by `cols` inputs, row-major.
    pub fn new(name: &'static str, rows: usize, cols: usize, values: Vec<f64>) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::InvalidArgument(format!(
                "{name}: a {rows}x{cols} map reads or writes nothing"
            )));
        }
        if values.len() != rows * cols {
            return Err(Error::InvalidArgument(format!(
                "{name}: a {rows}x{cols} map has {} coefficients and {} were given. A missing \
                 one would be an input silently weighted zero.",
                rows * cols,
                values.len()
            )));
        }
        Ok(Self {
            name,
            rows,
            cols,
            // One multiply-add per coefficient per position.
            cost: (rows * cols) as f64,
            values,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The whole map applied to one position's `K` values, in `f64`.
    ///
    /// The reference the blocked kernel is checked against, and slow on purpose:
    /// it is one expression per output and reads exactly like the definition.
    pub fn at_one_position(&self, values: &[f64]) -> Vec<f64> {
        (0..self.rows)
            .map(|row| {
                (0..self.cols)
                    .map(|col| self.values[row * self.cols + col] * values[col])
                    .sum()
            })
            .collect()
    }
}

/// The inner loop, once per element type.
///
/// **Output-major, and the first input initialises rather than accumulates.**
/// One accumulator is live at a time, so it stays in registers whatever `K'` is;
/// and starting from `M[o][0] * in[0]` instead of from zero removes a whole pass
/// that writes zeroes over a buffer that is about to be overwritten.
///
/// The slice pattern is what makes this vectorise: `dst` and `src` are
/// re-borrowed at the tile's length before the loop, so the bounds check is
/// hoisted out of it and what is left is a load, a broadcast multiply and an
/// add over contiguous memory.
macro_rules! linear_lane {
    ($self:ident, $inputs:ident, $outputs:ident, $from:ident, $len:ident, $float:ty) => {{
        for (which, out) in $outputs.iter_mut().enumerate() {
            let row = $from + which;
            let coefficients = &$self.values[row * $self.cols..(row + 1) * $self.cols];
            let dst = &mut out[..$len];
            let first = coefficients[0] as $float;
            let src = &$inputs[0][..$len];
            for position in 0..$len {
                dst[position] = first * src[position];
            }
            for (col, &coefficient) in coefficients.iter().enumerate().skip(1) {
                let coefficient = coefficient as $float;
                let src = &$inputs[col][..$len];
                for position in 0..$len {
                    dst[position] += coefficient * src[position];
                }
            }
        }
    }};
}

impl TupleKernel for LinearMap {
    fn name(&self) -> &'static str {
        self.name
    }

    fn n_inputs(&self) -> usize {
        self.cols
    }

    fn n_outputs(&self) -> usize {
        self.rows
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn apply_f32(&self, inputs: &[&[f32]], outputs: &mut [&mut [f32]], from: usize, len: usize) {
        linear_lane!(self, inputs, outputs, from, len, f32)
    }

    fn apply_f64(&self, inputs: &[&[f64]], outputs: &mut [&mut [f64]], from: usize, len: usize) {
        linear_lane!(self, inputs, outputs, from, len, f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> LinearMap {
        // Off-diagonal on purpose: a diagonal matrix is K independent scalings
        // and would pass every test here without mixing anything.
        LinearMap::new(
            "mix",
            3,
            3,
            vec![0.0, 1.0, 0.5, 0.25, 0.0, -1.0, 2.0, -0.5, 0.0],
        )
        .expect("a map")
    }

    #[test]
    fn the_tiled_kernel_is_the_definition() {
        let map = map();
        let len = TILE * 2 + 7;
        let held: Vec<Vec<f64>> = (0..3)
            .map(|col| {
                (0..len)
                    .map(|position| (position as f64 * 0.5 + col as f64).sin())
                    .collect()
            })
            .collect();
        let inputs: Vec<&[f64]> = held.iter().map(|values| values.as_slice()).collect();
        let mut written: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; len]).collect();
        {
            let mut outputs: Vec<&mut [f64]> = written
                .iter_mut()
                .map(|values| values.as_mut_slice())
                .collect();
            map.apply_f64(&inputs, &mut outputs, 0, len);
        }
        for position in 0..len {
            let values: Vec<f64> = (0..3).map(|col| held[col][position]).collect();
            let wanted = map.at_one_position(&values);
            for row in 0..3 {
                assert_eq!(written[row][position], wanted[row], "row {row}");
            }
        }
    }

    /// A window of the outputs is the rows it names, value for value.
    ///
    /// This is what lets `apply_with` take output 0 and `apply_side` take the
    /// rest without either computing the other's. A kernel that ignored `from`
    /// would pass every other test in this file: the whole-tuple call would be
    /// right, and only the arrays beside the primary would be one row off.
    #[test]
    fn a_window_of_the_outputs_is_the_rows_it_names() {
        let map = map();
        let len = TILE + 5;
        let held: Vec<Vec<f64>> = (0..3)
            .map(|col| {
                (0..len)
                    .map(|position| (position as f64 * 0.25 + col as f64).cos())
                    .collect()
            })
            .collect();
        let inputs: Vec<&[f64]> = held.iter().map(|values| values.as_slice()).collect();

        let mut whole: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; len]).collect();
        {
            let mut outputs: Vec<&mut [f64]> = whole
                .iter_mut()
                .map(|values| values.as_mut_slice())
                .collect();
            map.apply_f64(&inputs, &mut outputs, 0, len);
        }
        let mut window: Vec<Vec<f64>> = (0..2).map(|_| vec![0.0; len]).collect();
        {
            let mut outputs: Vec<&mut [f64]> = window
                .iter_mut()
                .map(|values| values.as_mut_slice())
                .collect();
            map.apply_f64(&inputs, &mut outputs, 1, len);
        }
        for row in 1..3 {
            assert_eq!(window[row - 1], whole[row], "row {row}");
        }
        // The negative control the assertion above needs: rows 1 and 2 are not
        // each other's and not row 0's, so agreeing with the right one is a
        // statement rather than an accident.
        assert_ne!(window[0], whole[0]);
        assert_ne!(window[0], window[1]);
    }

    #[test]
    fn a_map_whose_size_disagrees_with_its_coefficients_is_refused() {
        assert!(LinearMap::new("short", 2, 3, vec![1.0; 5]).is_err());
        assert!(LinearMap::new("empty", 0, 3, Vec::new()).is_err());
    }

    #[test]
    fn an_arity_that_disagrees_with_the_kernel_is_refused_by_name() {
        let refusal = TupleOp::new("mix", Box::new(map()), vec![7], Vec::new(), Dtype::F32)
            .map(|_| ())
            .expect_err("a refusal")
            .to_string();
        assert!(refusal.contains("reads 3 array(s)"), "{refusal}");
        let refusal = TupleOp::new("mix", Box::new(map()), vec![7, 8], Vec::new(), Dtype::F32)
            .map(|_| ())
            .expect_err("a refusal")
            .to_string();
        assert!(refusal.contains("writes 3 array(s)"), "{refusal}");
    }

    #[test]
    fn one_image_named_twice_is_refused_by_name() {
        let refusal = TupleOp::new(
            "mix",
            Box::new(map()),
            vec![7, 7],
            vec!["a".into(), "b".into()],
            Dtype::F32,
        )
        .map(|_| ())
        .expect_err("a refusal")
        .to_string();
        assert!(refusal.contains("name one image twice"), "{refusal}");
    }
}
