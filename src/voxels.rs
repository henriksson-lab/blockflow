// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A block's data: **rank 3, element type carried as a tag**.
//
// Why a tagged enum and not a generic parameter
// ---------------------------------------------
// The obvious alternative is `BlockOp<T>` / `Environment<T>` with `T` threaded
// through. It was rejected on three pieces of evidence in this tree, not on
// taste:
//
// * **Object safety.** `Chain::Op` is `Box<dyn BlockOp>` and `execute` takes
//   `&dyn Environment`; the conformance suite is written as `for strategy in
//   [Trivial, Enumerating, Greedy]` over trait objects, which is what proves
//   that `Greedy::run` honours `Trivial::decompose`. A generic method or a
//   generic parameter on either trait deletes that arrangement.
// * **One workflow has several element types.** `PhaseDecomposition::dtype` is
//   already `Option<Dtype>` — "what this phase writes, when that is not what it
//   read" — and `Output` carries a dtype per side array. The motivating chain
//   reads `u16`, computes in `f64` and writes `bool`. `Chain<T>` cannot express
//   that at all; a tag can, because the dtype is a *value* that flows down the
//   images the same way a volume does.
// * **Monomorphisation.** The executor, the planner, the cost model and the
//   cache are byte-oriented on purpose (see `dtype.rs`, which says so). A
//   generic parameter would infect all four to buy nothing: none of them reads
//   an element.
//
// What the enum costs is one `match` per op shell, at the seam where the shell
// adapts to a kernel. That is where a `match` belongs — the kernels in
// `src/ops/` stay free functions generic over the element type, and the shell is
// the only thing that knows which one it was handed.
//
// Why rank 3 rather than `ArrayD`
// -------------------------------
// A *image* is a volume. `BlockGeometry`, `Anchor`, `BlockGrid` and `Region`'s
// use in the executor are all rank 3 already, and `ops::view3` existed purely to
// convert a dynamic rank back to the static one at every call. Two dimensions
// are the degenerate case of three, not a separate case, so the dynamic rank
// bought nothing and cost an indirection per index.
//
// A **side output is a different question** and keeps its own rank: one row per
// object, one score per class per position. That is [`SideBuf`], below, which is
// dynamic-rank on purpose.
//
// The measured reason this exists
// -------------------------------
// A read extent measured at 310^3 is 29.8 MB as `bool` and 238 MB as `f64`. At
// 40 threads that is 1.2 GB against 9.5 GB of read buffers — an 8x tax on
// exactly the stages whose justification is memory.

use ndarray::{Array3, ArrayD, ArrayView3, ArrayViewMut3, Axis, IxDyn, Slice};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::Region;

/// A 3-D block of one element type, chosen at run time.
///
/// Construct through [`Voxels::zeros`], [`Voxels::filled`] or `From<Array3<T>>`;
/// read through [`Voxels::view`], which names the type it wanted when the tag
/// says otherwise.
#[derive(Debug, Clone, PartialEq)]
pub enum Voxels {
    Bool(Array3<bool>),
    U8(Array3<u8>),
    U16(Array3<u16>),
    U32(Array3<u32>),
    U64(Array3<u64>),
    I8(Array3<i8>),
    I16(Array3<i16>),
    I32(Array3<i32>),
    I64(Array3<i64>),
    F32(Array3<f32>),
    F64(Array3<f64>),
}

/// `match` over every variant with one body. The variant list appears in this
/// file and nowhere else, so adding an element type is one edit here rather than
/// a hunt through the crate.
macro_rules! over_voxels {
    ($value:expr, |$array:ident| $body:expr) => {
        match $value {
            Voxels::Bool($array) => $body,
            Voxels::U8($array) => $body,
            Voxels::U16($array) => $body,
            Voxels::U32($array) => $body,
            Voxels::U64($array) => $body,
            Voxels::I8($array) => $body,
            Voxels::I16($array) => $body,
            Voxels::I32($array) => $body,
            Voxels::I64($array) => $body,
            Voxels::F32($array) => $body,
            Voxels::F64($array) => $body,
        }
    };
}

/// The same walk, rebuilding the variant it matched.
macro_rules! map_voxels {
    ($value:expr, |$array:ident| $body:expr) => {
        match $value {
            Voxels::Bool($array) => Voxels::Bool($body),
            Voxels::U8($array) => Voxels::U8($body),
            Voxels::U16($array) => Voxels::U16($body),
            Voxels::U32($array) => Voxels::U32($body),
            Voxels::U64($array) => Voxels::U64($body),
            Voxels::I8($array) => Voxels::I8($body),
            Voxels::I16($array) => Voxels::I16($body),
            Voxels::I32($array) => Voxels::I32($body),
            Voxels::I64($array) => Voxels::I64($body),
            Voxels::F32($array) => Voxels::F32($body),
            Voxels::F64($array) => Voxels::F64($body),
        }
    };
}

impl Voxels {
    /// The tag.
    pub fn dtype(&self) -> Dtype {
        match self {
            Voxels::Bool(_) => Dtype::Bool,
            Voxels::U8(_) => Dtype::U8,
            Voxels::U16(_) => Dtype::U16,
            Voxels::U32(_) => Dtype::U32,
            Voxels::U64(_) => Dtype::U64,
            Voxels::I8(_) => Dtype::I8,
            Voxels::I16(_) => Dtype::I16,
            Voxels::I32(_) => Dtype::I32,
            Voxels::I64(_) => Dtype::I64,
            Voxels::F32(_) => Dtype::F32,
            Voxels::F64(_) => Dtype::F64,
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        over_voxels!(self, |array| {
            let dim = array.dim();
            [dim.0, dim.1, dim.2]
        })
    }

    pub fn len(&self) -> usize {
        over_voxels!(self, |array| array.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes this block occupies **decoded, in memory**. The figure the residency
    /// budget is stated in, and the one the 8x above is a ratio of.
    pub fn bytes(&self) -> u64 {
        self.len() as u64 * self.dtype().size_of() as u64
    }

    /// A zero-filled block. `Dtype::F16` has no variant; see [`Voxels::zeros`]'s
    /// error for why.
    pub fn zeros(dtype: Dtype, shape: [usize; 3]) -> Result<Self> {
        Self::filled(dtype, shape, 0.0)
    }

    /// A block of `shape`, every element `value`.
    ///
    /// `value` is an `f64` because that is the crate's one numeric lingua franca
    /// — `constant_maps_to` and `Environment::uniform` speak it, and every
    /// element type here is exactly representable in it except at the extremes
    /// of `u64`/`i64`. Narrowing is Rust's saturating float-to-integer cast, so
    /// an out-of-range constant clamps rather than wrapping or trapping.
    pub fn filled(dtype: Dtype, shape: [usize; 3], value: f64) -> Result<Self> {
        let dim = (shape[0], shape[1], shape[2]);
        Ok(match dtype {
            Dtype::Bool => Voxels::Bool(Array3::from_elem(dim, value != 0.0)),
            Dtype::U8 => Voxels::U8(Array3::from_elem(dim, value as u8)),
            Dtype::U16 => Voxels::U16(Array3::from_elem(dim, value as u16)),
            Dtype::U32 => Voxels::U32(Array3::from_elem(dim, value as u32)),
            Dtype::U64 => Voxels::U64(Array3::from_elem(dim, value as u64)),
            Dtype::I8 => Voxels::I8(Array3::from_elem(dim, value as i8)),
            Dtype::I16 => Voxels::I16(Array3::from_elem(dim, value as i16)),
            Dtype::I32 => Voxels::I32(Array3::from_elem(dim, value as i32)),
            Dtype::I64 => Voxels::I64(Array3::from_elem(dim, value as i64)),
            Dtype::F32 => Voxels::F32(Array3::from_elem(dim, value as f32)),
            Dtype::F64 => Voxels::F64(Array3::from_elem(dim, value)),
            Dtype::F16 => {
                return Err(Error::InvalidArgument(
                    "half-precision has no buffer variant: Rust has no native 16-bit float and \
                     this crate's dependency list is deliberately short, so `Dtype::F16` is a \
                     byte-width tag for storage and not something a block can hold. An op that \
                     is handed one is refused by `accepts` before the run starts."
                        .to_string(),
                ))
            }
        })
    }

    /// A block filled with the value an **unwritten** voxel holds.
    ///
    /// `f64::NAN` for the float types, on the argument the images have always
    /// been NaN-filled with: a voxel nobody wrote must be loud rather than a
    /// convincing zero. **The integer and `bool` types have no such value**, and
    /// pretending otherwise would be worse than admitting it: the maximum is
    /// merely implausible, and for `bool` there is no implausible value at all.
    /// So a coverage hole in a `bool` image is caught by
    /// [`crate::tiling::boxes_tile_exactly`] and by the write accounting, and not
    /// by looking at the data. That is a real loss and it is recorded here rather
    /// than papered over.
    pub fn unwritten(dtype: Dtype, shape: [usize; 3]) -> Result<Self> {
        match dtype {
            Dtype::F32 | Dtype::F64 => Self::filled(dtype, shape, f64::NAN),
            Dtype::Bool => Self::filled(dtype, shape, 1.0),
            _ => Self::filled(dtype, shape, f64::MAX),
        }
    }

    /// A read-only view of the elements, or an error naming both types.
    pub fn view<T: VoxelElement>(&self) -> Result<ArrayView3<'_, T>> {
        T::peek(self).map(|array| array.view()).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "this block holds {} and was read as {}",
                self.dtype().numpy_name(),
                T::DTYPE.numpy_name()
            ))
        })
    }

    /// A writable view of the elements, or an error naming both types.
    pub fn view_mut<T: VoxelElement>(&mut self) -> Result<ArrayViewMut3<'_, T>> {
        let held = self.dtype();
        T::peek_mut(self)
            .map(|array| array.view_mut())
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "this block holds {} and was written as {}",
                    held.numpy_name(),
                    T::DTYPE.numpy_name()
                ))
            })
    }

    /// Is every element the same value? `None` means "no", or "empty".
    ///
    /// Reported as an `f64` because the short circuit it licenses is stated in
    /// `f64` — see [`crate::op::BlockOp::constant_maps_to`]. A NaN-filled image
    /// is *not* uniform under this, which is the existing behaviour and the right
    /// one: `NaN != NaN`, so an unwritten image never short-circuits.
    pub fn uniform(&self) -> Option<f64> {
        over_voxels!(self, |array| {
            let first = *array.iter().next()?;
            array
                .iter()
                .all(|&value| value == first)
                .then(|| first.into_f64())
        })
    }

    /// Every element as an `f64`, whatever the tag says.
    ///
    /// For a shell whose kernel accumulates in `f64` — a mean over a `u8` window
    /// is the ordinary case — and **not** a general-purpose conversion: it
    /// allocates a copy eight bytes a voxel wide, which is exactly what this type
    /// exists to avoid paying by default. A shell that can call its kernel on the
    /// element type directly must do that instead.
    pub fn widened(&self) -> Array3<f64> {
        over_voxels!(self, |array| array.mapv(|value| value.into_f64()))
    }

    /// An owned copy of `region` of this block.
    pub fn slice_region(&self, region: &Region) -> Result<Self> {
        check_rank(region, "block slice")?;
        Ok(map_voxels!(self, |array| {
            let mut view = array.view();
            for (axis, (&start, &len)) in region.start.iter().zip(region.shape.iter()).enumerate() {
                view.slice_axis_inplace(Axis(axis), Slice::from(start..start + len));
            }
            view.to_owned()
        }))
    }

    /// Copy `source` into `region` of this block. Both must hold the same type.
    pub fn assign_region(&mut self, region: &Region, source: &Voxels) -> Result<()> {
        check_rank(region, "block assignment")?;
        self.same_type_as(source, "block assignment")?;
        assign_into(self, region, source)
    }

    /// Copy the whole of `source` over this block.
    pub fn assign(&mut self, source: &Voxels) -> Result<()> {
        self.same_type_as(source, "block copy")?;
        if self.shape() != source.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: source.shape().to_vec(),
            });
        }
        assign_into(self, &Region::whole(&self.shape()), source)
    }

    fn same_type_as(&self, other: &Voxels, what: &str) -> Result<()> {
        if self.dtype() != other.dtype() {
            return Err(Error::InvalidArgument(format!(
                "{what}: this block holds {} and the other holds {}",
                self.dtype().numpy_name(),
                other.dtype().numpy_name()
            )));
        }
        Ok(())
    }
}

fn check_rank(region: &Region, what: &str) -> Result<()> {
    if region.start.len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "{what}: a block is 3-D, got a region of rank {}",
            region.start.len()
        )));
    }
    Ok(())
}

/// The one place a region is written into a block. Both halves are known to
/// agree on the element type by the time this is called.
fn assign_into(target: &mut Voxels, region: &Region, source: &Voxels) -> Result<()> {
    macro_rules! arm {
        ($into:expr, $from:expr) => {{
            let mut view = $into.view_mut();
            for (axis, (&start, &len)) in region.start.iter().zip(region.shape.iter()).enumerate() {
                view.slice_axis_inplace(Axis(axis), Slice::from(start..start + len));
            }
            if view.shape() != $from.shape() {
                return Err(Error::ShapeMismatch {
                    expected: view.shape().to_vec(),
                    got: $from.shape().to_vec(),
                });
            }
            view.assign($from);
        }};
    }
    match (target, source) {
        (Voxels::Bool(into), Voxels::Bool(from)) => arm!(into, from),
        (Voxels::U8(into), Voxels::U8(from)) => arm!(into, from),
        (Voxels::U16(into), Voxels::U16(from)) => arm!(into, from),
        (Voxels::U32(into), Voxels::U32(from)) => arm!(into, from),
        (Voxels::U64(into), Voxels::U64(from)) => arm!(into, from),
        (Voxels::I8(into), Voxels::I8(from)) => arm!(into, from),
        (Voxels::I16(into), Voxels::I16(from)) => arm!(into, from),
        (Voxels::I32(into), Voxels::I32(from)) => arm!(into, from),
        (Voxels::I64(into), Voxels::I64(from)) => arm!(into, from),
        (Voxels::F32(into), Voxels::F32(from)) => arm!(into, from),
        (Voxels::F64(into), Voxels::F64(from)) => arm!(into, from),
        // Unreachable: every caller checks the tags first, and the check names
        // both types. An arm that panicked here would be a worse message.
        (target, source) => return target.same_type_as(source, "block assignment"),
    }
    Ok(())
}

/// One element type a [`Voxels`] can hold.
///
/// The bridge between the tag and a kernel's static type. Deliberately narrow:
/// a kernel states its own bounds (`Ord`, `PartialOrd`, `Copy`, nothing), and
/// this trait says only how to get in and out of the enum.
pub trait VoxelElement: Copy + Send + Sync + PartialEq + 'static {
    const DTYPE: Dtype;
    fn wrap(array: Array3<Self>) -> Voxels;
    fn peek(voxels: &Voxels) -> Option<&Array3<Self>>;
    fn peek_mut(voxels: &mut Voxels) -> Option<&mut Array3<Self>>;
    /// The value as the short circuit's `f64`.
    fn into_f64(self) -> f64;
    /// The nearest value of this type. Saturating for the integers.
    fn from_f64(value: f64) -> Self;
}

macro_rules! numeric_voxel_element {
    ($type:ty, $variant:ident, $dtype:expr) => {
        impl VoxelElement for $type {
            const DTYPE: Dtype = $dtype;

            fn wrap(array: Array3<Self>) -> Voxels {
                Voxels::$variant(array)
            }

            fn peek(voxels: &Voxels) -> Option<&Array3<Self>> {
                match voxels {
                    Voxels::$variant(array) => Some(array),
                    _ => None,
                }
            }

            fn peek_mut(voxels: &mut Voxels) -> Option<&mut Array3<Self>> {
                match voxels {
                    Voxels::$variant(array) => Some(array),
                    _ => None,
                }
            }

            fn into_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value as Self
            }
        }

        impl From<Array3<$type>> for Voxels {
            fn from(array: Array3<$type>) -> Voxels {
                Voxels::$variant(array)
            }
        }
    };
}

numeric_voxel_element!(u8, U8, Dtype::U8);
numeric_voxel_element!(u16, U16, Dtype::U16);
numeric_voxel_element!(u32, U32, Dtype::U32);
numeric_voxel_element!(u64, U64, Dtype::U64);
numeric_voxel_element!(i8, I8, Dtype::I8);
numeric_voxel_element!(i16, I16, Dtype::I16);
numeric_voxel_element!(i32, I32, Dtype::I32);
numeric_voxel_element!(i64, I64, Dtype::I64);
numeric_voxel_element!(f32, F32, Dtype::F32);
numeric_voxel_element!(f64, F64, Dtype::F64);

impl VoxelElement for bool {
    const DTYPE: Dtype = Dtype::Bool;

    fn wrap(array: Array3<Self>) -> Voxels {
        Voxels::Bool(array)
    }

    fn peek(voxels: &Voxels) -> Option<&Array3<Self>> {
        match voxels {
            Voxels::Bool(array) => Some(array),
            _ => None,
        }
    }

    fn peek_mut(voxels: &mut Voxels) -> Option<&mut Array3<Self>> {
        match voxels {
            Voxels::Bool(array) => Some(array),
            _ => None,
        }
    }

    /// `1.0` / `0.0`, which is the same convention `ops::voxelwise::from_set`
    /// states for a mask held as `f64`. One convention, not two.
    fn into_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }

    /// Non-zero is true, matching `ops::voxelwise::is_set`.
    fn from_f64(value: f64) -> Self {
        value != 0.0
    }
}

impl From<Array3<bool>> for Voxels {
    fn from(array: Array3<bool>) -> Voxels {
        Voxels::Bool(array)
    }
}

// ---------------------------------------------------------- side output --

/// A side output's buffer, or a stand-in for one.
///
/// **Dynamic rank, and that is the point.** An image is a volume; the array an op
/// writes beside it often is not — one row per object, one score per class per
/// position — and [`crate::op::Output::shape`] is a `Vec<usize>` for exactly that
/// reason. So this is the one buffer in the crate that keeps `ArrayD`, and it
/// keeps it because the rank is genuinely unknown rather than because nobody got
/// round to pinning it.
///
/// It mirrors `BlockBuf`'s two-variant shape for the same reason `BlockBuf` has
/// one: a simulated run must reach the same code with nothing allocated.
#[derive(Debug, Clone, PartialEq)]
pub enum SideBuf {
    Array(ArrayD<f64>),
    /// No data. Carries only what the accounting needs.
    Accounted {
        elements: usize,
    },
}

impl SideBuf {
    /// A zero-filled buffer of `region`'s shape, at any rank.
    pub fn zeros(region: &Region) -> Self {
        SideBuf::Array(ArrayD::zeros(IxDyn(&region.shape)))
    }

    pub fn elements(&self) -> usize {
        match self {
            SideBuf::Array(array) => array.len(),
            SideBuf::Accounted { elements } => *elements,
        }
    }

    pub fn as_array(&self) -> Option<&ArrayD<f64>> {
        match self {
            SideBuf::Array(array) => Some(array),
            SideBuf::Accounted { .. } => None,
        }
    }

    /// The buffer's data, to be filled in. `None` where there is none, so a
    /// caller writes `if let` rather than special-casing a failure.
    pub fn as_array_mut(&mut self) -> Option<&mut ArrayD<f64>> {
        match self {
            SideBuf::Array(array) => Some(array),
            SideBuf::Accounted { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tag_and_the_width_agree_for_every_variant() {
        for dtype in [
            Dtype::Bool,
            Dtype::U8,
            Dtype::U16,
            Dtype::U32,
            Dtype::U64,
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
            Dtype::I64,
            Dtype::F32,
            Dtype::F64,
        ] {
            let block = Voxels::zeros(dtype, [2, 3, 4]).unwrap();
            assert_eq!(block.dtype(), dtype);
            assert_eq!(block.shape(), [2, 3, 4]);
            assert_eq!(block.len(), 24);
            assert_eq!(block.bytes(), 24 * dtype.size_of() as u64);
            assert_eq!(block.uniform(), Some(0.0), "{dtype:?}");
        }
    }

    /// The 8x, as an arithmetic fact about the representation rather than a
    /// claim about a run.
    #[test]
    fn a_bool_block_is_an_eighth_of_the_f64_block_it_replaces() {
        let extent = [16, 16, 16];
        let narrow = Voxels::zeros(Dtype::Bool, extent).unwrap();
        let wide = Voxels::zeros(Dtype::F64, extent).unwrap();
        assert_eq!(wide.bytes(), narrow.bytes() * 8);
    }

    #[test]
    fn half_precision_is_refused_by_name_rather_than_by_a_panic() {
        let err = Voxels::zeros(Dtype::F16, [1, 1, 1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("half-precision"), "got: {err}");
    }

    #[test]
    fn a_view_of_the_wrong_type_names_both_types() {
        let block = Voxels::zeros(Dtype::U16, [2, 2, 2]).unwrap();
        let err = block.view::<f64>().unwrap_err().to_string();
        assert!(
            err.contains("uint16") && err.contains("float64"),
            "got: {err}"
        );
        assert!(block.view::<u16>().is_ok());
    }

    #[test]
    fn a_region_round_trips_through_slice_and_assign() {
        let mut source = Array3::<u16>::zeros((4, 4, 4));
        for (flat, value) in source.iter_mut().enumerate() {
            *value = flat as u16;
        }
        let source: Voxels = source.into();
        let region = Region::new(&[1, 1, 1], &[2, 2, 2]);
        let cut = source.slice_region(&region).unwrap();
        assert_eq!(cut.shape(), [2, 2, 2]);

        let mut target = Voxels::zeros(Dtype::U16, [4, 4, 4]).unwrap();
        target.assign_region(&region, &cut).unwrap();
        assert_eq!(
            target.view::<u16>().unwrap()[[1, 1, 1]],
            source.view::<u16>().unwrap()[[1, 1, 1]]
        );
        assert_eq!(target.view::<u16>().unwrap()[[0, 0, 0]], 0);
    }

    #[test]
    fn assigning_across_element_types_is_refused_and_says_which() {
        let mut target = Voxels::zeros(Dtype::Bool, [2, 2, 2]).unwrap();
        let source = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        let err = target.assign(&source).unwrap_err().to_string();
        assert!(
            err.contains("bool") && err.contains("float64"),
            "got: {err}"
        );
    }

    /// The sentinel exists for the float types and not for the rest, which is a
    /// stated loss rather than an oversight.
    #[test]
    fn only_the_float_types_have_an_unwritten_sentinel() {
        let floats = Voxels::unwritten(Dtype::F64, [1, 1, 1]).unwrap();
        assert!(floats.view::<f64>().unwrap()[[0, 0, 0]].is_nan());
        let flags = Voxels::unwritten(Dtype::Bool, [1, 1, 1]).unwrap();
        assert!(flags.view::<bool>().unwrap()[[0, 0, 0]]);
    }

    #[test]
    fn a_narrowing_constant_saturates_rather_than_wrapping() {
        let block = Voxels::filled(Dtype::U8, [1, 1, 1], 4000.0).unwrap();
        assert_eq!(block.view::<u8>().unwrap()[[0, 0, 0]], 255);
        let block = Voxels::filled(Dtype::Bool, [1, 1, 1], 7.0).unwrap();
        assert!(block.view::<bool>().unwrap()[[0, 0, 0]]);
    }
}
