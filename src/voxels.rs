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

    /// Every element as **"is it not zero"**, without widening first.
    ///
    /// The mask a run's `bool` image already is, and the mask a numeric image
    /// means. The obvious spelling — `widened().mapv(|value| value != 0.0)` —
    /// is what this replaces, and it allocates an `f64` volume nobody wants in
    /// order to reach a `bool` one: **eight bytes a voxel of pure intermediate**,
    /// converted once on the way in and once on the way out. On a `384^3` mask
    /// that measured about `0.7 s`, roughly a fifth of the stage that asked for
    /// it.
    ///
    /// `!= 0` is also the comparison every element type already has, so nothing
    /// is gained by routing it through `f64` — and for `u64` and `i64` past
    /// `2^53` the detour is not even exact, which is the same trap
    /// [`Voxels::widened_i64`] exists to keep out of identifiers.
    pub fn nonzero(&self) -> Array3<bool> {
        match self {
            Voxels::Bool(array) => array.clone(),
            Voxels::U8(array) => array.mapv(|value| value != 0),
            Voxels::U16(array) => array.mapv(|value| value != 0),
            Voxels::U32(array) => array.mapv(|value| value != 0),
            Voxels::U64(array) => array.mapv(|value| value != 0),
            Voxels::I8(array) => array.mapv(|value| value != 0),
            Voxels::I16(array) => array.mapv(|value| value != 0),
            Voxels::I32(array) => array.mapv(|value| value != 0),
            Voxels::I64(array) => array.mapv(|value| value != 0),
            Voxels::F32(array) => array.mapv(|value| value != 0.0),
            Voxels::F64(array) => array.mapv(|value| value != 0.0),
        }
    }

    /// Every element as an `i64`, **exactly**, or a refusal naming why not.
    ///
    /// The counterpart of [`Voxels::widened`] for a volume of *identifiers*
    /// rather than of measurements: labels, region ids, component numbers. Every
    /// integer type here is an `i64` except `u64` above `i64::MAX`, and that one
    /// is refused with the offending value in the message rather than wrapped —
    /// a wrapped identifier is a negative number no image contains and is
    /// indistinguishable downstream from a real one, which is why this returns a
    /// `Result` where [`Voxels::widened`] does not.
    ///
    /// **This exists because `widened` alone was not enough, and that was found
    /// the hard way.** `f64` has no exact representation past 2^53, so a label
    /// widened through it becomes a label that compares equal to its
    /// neighbours — silently, and only for large values. A consumer's reader
    /// reached an integer path for a `float64` array and returned its IEEE-754
    /// bit patterns as integers; it passed its own test because both sides of
    /// the comparison got the same reinterpretation, and it would have panicked
    /// on the first negative value it met. Both halves of that are refused here.
    ///
    /// The float types are refused by name rather than truncated. Which way to
    /// round is the caller's decision; a caller that means to truncate has
    /// `mapv` and can say so.
    ///
    /// Exactly [`crate::npy::Elements::widened_i64`], one rank down. It
    /// allocates eight bytes a voxel, with the same caution
    /// [`Voxels::widened`] carries.
    pub fn widened_i64(&self, what: &str) -> Result<Array3<i64>> {
        let refuse_float = |name: &str| {
            Error::InvalidArgument(format!(
                "{what}: the image holds {name} and this is the exact integer widening, which has \
                 no answer for a fraction, a `NaN` or an infinity. Which way to round is the \
                 caller's decision and not one to guess here. Read it as itself and say what you \
                 mean with `mapv`, or take `Voxels::widened` if `f64` was what you wanted."
            ))
        };
        Ok(match self {
            Voxels::Bool(array) => array.mapv(i64::from),
            Voxels::U8(array) => array.mapv(i64::from),
            Voxels::U16(array) => array.mapv(i64::from),
            Voxels::U32(array) => array.mapv(i64::from),
            Voxels::I8(array) => array.mapv(i64::from),
            Voxels::I16(array) => array.mapv(i64::from),
            Voxels::I32(array) => array.mapv(i64::from),
            Voxels::I64(array) => array.clone(),
            Voxels::U64(array) => {
                let mut out: Vec<i64> = Vec::with_capacity(array.len());
                for value in array.iter() {
                    out.push(i64::try_from(*value).map_err(|_| {
                        Error::InvalidArgument(format!(
                            "{what}: the image holds uint64 and one element is {value}, which is \
                             past `i64::MAX` and has no `i64`. Wrapping it would turn a large \
                             identifier into a negative one that no image contains, so it is \
                             refused instead. Read it as `u64` if the range is real."
                        ))
                    })?);
                }
                Array3::from_shape_vec(array.dim(), out)
                    .map_err(|error| Error::InvalidArgument(format!("{what}: {error}")))?
            }
            Voxels::F32(_) => return Err(refuse_float("float32")),
            Voxels::F64(_) => return Err(refuse_float("float64")),
        })
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

    /// Copy `from` of `source` into `region` of this block.
    ///
    /// [`Self::assign_region`] with the *source* narrowed too, which is one copy
    /// where composing the two would be two: a caller placing part of one buffer
    /// into part of another would otherwise copy every written voxel once to
    /// extract it and again to place it.
    ///
    /// **`crate::slab` is what wanted it and no longer calls it.** A slab's core
    /// now goes into a [`VoxelsMut`] the slab's own thread holds, because that
    /// placement was the one part of a cut that did not parallelise. This stays
    /// as the owned-destination
    /// form of the same operation — it and [`VoxelsMut::assign_from`] share one
    /// `copy_view`, and `tests/intra_block_slicing.rs` asserts the two write the
    /// same bytes.
    pub fn assign_region_from(
        &mut self,
        region: &Region,
        source: &Voxels,
        from: &Region,
    ) -> Result<()> {
        check_rank(region, "block assignment")?;
        check_rank(from, "block sub-region assignment source")?;
        self.same_type_as(source, "block assignment")?;
        assign_into_from(self, region, source, from)
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

/// **A borrowed, mutable view of a block, with the element-type tag still
/// attached** — and the one thing [`Voxels`] cannot express.
///
/// `Voxels` owns an `Array3` behind a tag, so the only way to hand a thread part
/// of a block was to build a *new* `Voxels`: a copy in, and a copy back. That
/// copy back is the shape of the gap `docs/design/intra-block.md` §5.2 records,
/// and §13.3.1 is the measurement that made it worth closing rather than
/// predicting. §5.2's author declined to build it because §5.3 had measured
/// per-thread cloning as affordable; that was true of the cases measured then
/// and false of a voxelwise chain, where the copy is the whole cost. Threading
/// the placement is **2.9-3.6x** on the pass itself, interleaved.
///
/// **What this is for, and what it is not.** It exists so that
/// [`Self::split_at`] can hand out **disjoint** mutable views of one buffer,
/// which is what lets each slab of `crate::slab` write its own core from its own
/// thread instead of queueing behind one pass at the end. It is deliberately not
/// a second `Voxels`: it has no constructor of its own, no arithmetic, and no
/// `BlockOp` takes one. An op is still handed an owned block, because an op
/// writes the whole buffer it is given and a slab's *answer* is wider than its
/// core — that asymmetry is what the copy in [`Self::assign_from`] resolves, and
/// no borrowed type removes it.
///
/// **There is no `unsafe` here and none is needed.**
/// `ArrayViewMut3::split_at(Axis, index)` consumes a view and returns two that
/// borrow disjoint halves for the same lifetime; the compiler is what says they
/// do not overlap, and each half is `Send` for any `Send` element.
#[derive(Debug)]
pub enum VoxelsMut<'a> {
    Bool(ArrayViewMut3<'a, bool>),
    U8(ArrayViewMut3<'a, u8>),
    U16(ArrayViewMut3<'a, u16>),
    U32(ArrayViewMut3<'a, u32>),
    U64(ArrayViewMut3<'a, u64>),
    I8(ArrayViewMut3<'a, i8>),
    I16(ArrayViewMut3<'a, i16>),
    I32(ArrayViewMut3<'a, i32>),
    I64(ArrayViewMut3<'a, i64>),
    F32(ArrayViewMut3<'a, f32>),
    F64(ArrayViewMut3<'a, f64>),
}

/// One arm per element type, for a body that borrows the whole view rather than
/// rebuilding a [`Voxels`] around it.
///
/// `map_voxels!` cannot serve here: it wraps its body back into a `Voxels`, and
/// every method below either answers something that is not a block or answers
/// *two* views.
macro_rules! map_voxels_mut {
    ($value:expr, |$view:ident| $body:expr) => {
        match $value {
            VoxelsMut::Bool($view) => $body,
            VoxelsMut::U8($view) => $body,
            VoxelsMut::U16($view) => $body,
            VoxelsMut::U32($view) => $body,
            VoxelsMut::U64($view) => $body,
            VoxelsMut::I8($view) => $body,
            VoxelsMut::I16($view) => $body,
            VoxelsMut::I32($view) => $body,
            VoxelsMut::I64($view) => $body,
            VoxelsMut::F32($view) => $body,
            VoxelsMut::F64($view) => $body,
        }
    };
    ($value:expr, |$view:ident| $body:expr, wrap $arm:ident) => {
        match $value {
            VoxelsMut::Bool($view) => $arm!(VoxelsMut::Bool, $body),
            VoxelsMut::U8($view) => $arm!(VoxelsMut::U8, $body),
            VoxelsMut::U16($view) => $arm!(VoxelsMut::U16, $body),
            VoxelsMut::U32($view) => $arm!(VoxelsMut::U32, $body),
            VoxelsMut::U64($view) => $arm!(VoxelsMut::U64, $body),
            VoxelsMut::I8($view) => $arm!(VoxelsMut::I8, $body),
            VoxelsMut::I16($view) => $arm!(VoxelsMut::I16, $body),
            VoxelsMut::I32($view) => $arm!(VoxelsMut::I32, $body),
            VoxelsMut::I64($view) => $arm!(VoxelsMut::I64, $body),
            VoxelsMut::F32($view) => $arm!(VoxelsMut::F32, $body),
            VoxelsMut::F64($view) => $arm!(VoxelsMut::F64, $body),
        }
    };
}

impl<'a> VoxelsMut<'a> {
    /// The whole of `block`, borrowed.
    ///
    /// The only way to make one: there is no owning constructor, because a
    /// `VoxelsMut` that owned anything would be a `Voxels` with a worse name.
    pub fn of(block: &'a mut Voxels) -> Self {
        match block {
            Voxels::Bool(array) => VoxelsMut::Bool(array.view_mut()),
            Voxels::U8(array) => VoxelsMut::U8(array.view_mut()),
            Voxels::U16(array) => VoxelsMut::U16(array.view_mut()),
            Voxels::U32(array) => VoxelsMut::U32(array.view_mut()),
            Voxels::U64(array) => VoxelsMut::U64(array.view_mut()),
            Voxels::I8(array) => VoxelsMut::I8(array.view_mut()),
            Voxels::I16(array) => VoxelsMut::I16(array.view_mut()),
            Voxels::I32(array) => VoxelsMut::I32(array.view_mut()),
            Voxels::I64(array) => VoxelsMut::I64(array.view_mut()),
            Voxels::F32(array) => VoxelsMut::F32(array.view_mut()),
            Voxels::F64(array) => VoxelsMut::F64(array.view_mut()),
        }
    }

    /// The tag, which a borrowed block carries exactly as an owned one does.
    pub fn dtype(&self) -> Dtype {
        match self {
            VoxelsMut::Bool(_) => Dtype::Bool,
            VoxelsMut::U8(_) => Dtype::U8,
            VoxelsMut::U16(_) => Dtype::U16,
            VoxelsMut::U32(_) => Dtype::U32,
            VoxelsMut::U64(_) => Dtype::U64,
            VoxelsMut::I8(_) => Dtype::I8,
            VoxelsMut::I16(_) => Dtype::I16,
            VoxelsMut::I32(_) => Dtype::I32,
            VoxelsMut::I64(_) => Dtype::I64,
            VoxelsMut::F32(_) => Dtype::F32,
            VoxelsMut::F64(_) => Dtype::F64,
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        let shape = map_voxels_mut!(self, |view| view.shape());
        [shape[0], shape[1], shape[2]]
    }

    /// Cut this view in two along `axis`: everything below `index`, and
    /// everything from it.
    ///
    /// **The whole reason this type exists.** The two halves borrow *disjoint*
    /// parts of the same buffer for the same lifetime, so they can go to two
    /// threads and each write its own without synchronisation and without the
    /// compiler being asked to take anything on trust. Repeating it yields as
    /// many disjoint views as there are slabs.
    ///
    /// `index` may be `0` or the axis's whole extent; the empty half is a view
    /// of nothing rather than an error, which is what lets a caller peel views
    /// off in a loop without a special case for the last one.
    pub fn split_at(self, axis: usize, index: usize) -> Result<(Self, Self)> {
        if axis >= 3 {
            return Err(Error::InvalidArgument(format!(
                "a block is 3-D and cannot be split along axis {axis}"
            )));
        }
        let extent = self.shape()[axis];
        if index > extent {
            return Err(Error::InvalidArgument(format!(
                "cannot split an axis of {extent} voxels at {index}"
            )));
        }
        macro_rules! halves {
            ($variant:path, $view:expr) => {{
                let (low, high) = $view;
                ($variant(low), $variant(high))
            }};
        }
        Ok(map_voxels_mut!(
            self,
            |view| view.split_at(Axis(axis), index),
            wrap halves
        ))
    }

    /// Copy `from` of `source` over the whole of this view.
    ///
    /// The counterpart of [`Voxels::assign_region_from`] with the destination
    /// borrowed rather than owned, and it goes through the same `copy_view` — so
    /// a slab placed through a view and a slab placed through an owned block are
    /// the same copy, checked the same way, rather than two implementations that
    /// agree until one of them is edited.
    pub fn assign_from(&mut self, source: &Voxels, from: &Region) -> Result<()> {
        check_rank(from, "borrowed sub-region assignment source")?;
        if self.dtype() != source.dtype() {
            return Err(Error::InvalidArgument(format!(
                "borrowed block assignment: this view holds {} and the source holds {}",
                self.dtype().numpy_name(),
                source.dtype().numpy_name()
            )));
        }
        macro_rules! arm {
            ($view:expr, $outof:expr) => {
                copy_view($view.view_mut(), narrow($outof.view(), from))
            };
        }
        match (self, source) {
            (VoxelsMut::Bool(view), Voxels::Bool(outof)) => arm!(view, outof),
            (VoxelsMut::U8(view), Voxels::U8(outof)) => arm!(view, outof),
            (VoxelsMut::U16(view), Voxels::U16(outof)) => arm!(view, outof),
            (VoxelsMut::U32(view), Voxels::U32(outof)) => arm!(view, outof),
            (VoxelsMut::U64(view), Voxels::U64(outof)) => arm!(view, outof),
            (VoxelsMut::I8(view), Voxels::I8(outof)) => arm!(view, outof),
            (VoxelsMut::I16(view), Voxels::I16(outof)) => arm!(view, outof),
            (VoxelsMut::I32(view), Voxels::I32(outof)) => arm!(view, outof),
            (VoxelsMut::I64(view), Voxels::I64(outof)) => arm!(view, outof),
            (VoxelsMut::F32(view), Voxels::F32(outof)) => arm!(view, outof),
            (VoxelsMut::F64(view), Voxels::F64(outof)) => arm!(view, outof),
            // Unreachable: the tags were compared above. Stated rather than
            // `unreachable!` because a panic in a library is a worse answer than
            // the sentence the comparison already knows how to write.
            (view, source) => Err(Error::InvalidArgument(format!(
                "borrowed block assignment between a {} view and a {} block",
                view.dtype().numpy_name(),
                source.dtype().numpy_name()
            ))),
        }
    }
}

/// Narrow a mutable view to `region` on every axis.
///
/// Split out because two callers narrow the same way — [`Voxels`], which owns
/// its buffer, and [`VoxelsMut`], which borrows one — and a second copy of three
/// lines of axis slicing is a second place for an off-by-one to live.
fn narrow_mut<'v, T>(mut view: ArrayViewMut3<'v, T>, region: &Region) -> ArrayViewMut3<'v, T> {
    for (axis, (&start, &len)) in region.start.iter().zip(region.shape.iter()).enumerate() {
        view.slice_axis_inplace(Axis(axis), Slice::from(start..start + len));
    }
    view
}

/// [`narrow_mut`], for a shared view.
fn narrow<'v, T>(mut view: ArrayView3<'v, T>, region: &Region) -> ArrayView3<'v, T> {
    for (axis, (&start, &len)) in region.start.iter().zip(region.shape.iter()).enumerate() {
        view.slice_axis_inplace(Axis(axis), Slice::from(start..start + len));
    }
    view
}

/// **The one body that copies one narrowed view into another**, whether the
/// destination is an owned block or a borrowed slice of one.
///
/// The shapes are compared here rather than by either caller: two regions that
/// do not agree is the failure this whole family exists to refuse, and refusing
/// it in one place is what stops the two paths from drifting into two answers.
fn copy_view<T: Clone>(mut into: ArrayViewMut3<'_, T>, outof: ArrayView3<'_, T>) -> Result<()> {
    if into.shape() != outof.shape() {
        return Err(Error::ShapeMismatch {
            expected: into.shape().to_vec(),
            got: outof.shape().to_vec(),
        });
    }
    into.assign(&outof);
    Ok(())
}

/// The one place a region is written into a block. Both halves are known to
/// agree on the element type by the time this is called.
fn assign_into_from(
    target: &mut Voxels,
    region: &Region,
    source: &Voxels,
    from: &Region,
) -> Result<()> {
    macro_rules! arm {
        ($into:expr, $outof:expr) => {{
            copy_view(
                narrow_mut($into.view_mut(), region),
                narrow($outof.view(), from),
            )?
        }};
    }
    match (target, source) {
        (Voxels::Bool(into), Voxels::Bool(outof)) => arm!(into, outof),
        (Voxels::U8(into), Voxels::U8(outof)) => arm!(into, outof),
        (Voxels::U16(into), Voxels::U16(outof)) => arm!(into, outof),
        (Voxels::U32(into), Voxels::U32(outof)) => arm!(into, outof),
        (Voxels::U64(into), Voxels::U64(outof)) => arm!(into, outof),
        (Voxels::I8(into), Voxels::I8(outof)) => arm!(into, outof),
        (Voxels::I16(into), Voxels::I16(outof)) => arm!(into, outof),
        (Voxels::I32(into), Voxels::I32(outof)) => arm!(into, outof),
        (Voxels::I64(into), Voxels::I64(outof)) => arm!(into, outof),
        (Voxels::F32(into), Voxels::F32(outof)) => arm!(into, outof),
        (Voxels::F64(into), Voxels::F64(outof)) => arm!(into, outof),
        (target, source) => {
            return Err(Error::InvalidArgument(format!(
                "sub-region assignment between a {} block and a {} one",
                target.dtype().numpy_name(),
                source.dtype().numpy_name()
            )))
        }
    }
    Ok(())
}

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

// ------------------------------------------------- counting a difference --

/// How many elements two rank-3 arrays of one extent differ in, under a stated
/// comparison.
///
/// The `Array`-shaped sibling of [`crate::strategy::differing_voxels`], and it
/// exists because that one takes two [`Voxels`] while the callers that most need
/// it hold an `Array3<bool>` or an `ArrayView3<f64>` read straight off a
/// recording. Wrapping those in a `Voxels` to borrow the assertion would **clone
/// a pipeline volume** to answer a question about it, so the assertion comes to
/// the array instead.
///
/// **The extent check is the whole point and it is not decoration.** The obvious
/// hand-rolled form is `left.iter().zip(right.iter()).filter(..).count()`, and
/// `zip` *truncates*: two volumes of different extents come back differing in a
/// small number of elements rather than refusing, which in a parity suite is the
/// failure reading as a pass. Ten of the thirteen copies `differing_voxels` was
/// written to replace had exactly that shape. A count that cannot fail is a
/// check that cannot fail.
///
/// The element-type half of `differing_voxels`'s refusal is **not** needed here
/// and is not a gap: `T` is one type parameter across both arguments, so two
/// element types cannot be handed to this at all. That check moved from run time
/// to compile time, which is the direction it should move in.
///
/// Views rather than arrays, and **non-contiguous views are fine**: `iter()`
/// walks logical order, so a caller may pass a slice of a larger volume without
/// materialising it. That is the case this function exists for.
pub fn differing_elements_by<T, F>(
    left: ArrayView3<'_, T>,
    right: ArrayView3<'_, T>,
    same: F,
) -> Result<u64>
where
    F: Fn(&T, &T) -> bool,
{
    if left.shape() != right.shape() {
        return Err(Error::ShapeMismatch {
            expected: left.shape().to_vec(),
            got: right.shape().to_vec(),
        });
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .filter(|(one, other)| !same(one, other))
        .count() as u64)
}

/// [`differing_elements_by`] under `PartialEq`: *did these two come out the same
/// value*.
///
/// **Two `NaN`s differ under this and that is correct**, because a `NaN` is not a
/// value that equals anything, itself included. `+0.0` and `-0.0` **agree**,
/// because they are the same value.
///
/// This is what [`crate::strategy::differing_voxels`] does, so a caller moving
/// between the two gets the same answer.
pub fn differing_elements<T: PartialEq>(
    left: ArrayView3<'_, T>,
    right: ArrayView3<'_, T>,
) -> Result<u64> {
    differing_elements_by(left, right, |one, other| one == other)
}

/// [`differing_elements_by`] under the bit pattern: *did these two
/// implementations produce the same `f64`*.
///
/// **Two `NaN`s of identical bits agree under this**, which is what an oracle
/// asserting bit-for-bit reproduction is actually claiming — and the opposite of
/// what [`differing_elements`] answers. `+0.0` and `-0.0` **differ**, which is
/// also the right answer for that claim: a sign-of-zero disagreement is a real
/// difference between two implementations, and value comparison hides it.
///
/// **There are two functions rather than one with a default because neither
/// answer is more correct than the other** — they are answers to two different
/// questions, and the two disagree on precisely the inputs a floating-point
/// parity suite exists to look at. Naming them for the claim rather than for the
/// operator puts that choice at the call site, where the person who knows which
/// question they are asking is.
///
/// Restricted to floating point by [`Bitwise`], deliberately: for an integer the
/// two functions would be the same function, and an API with two names for one
/// behaviour invites a reader to believe there is a difference.
pub fn differing_bits<T: Bitwise>(
    left: ArrayView3<'_, T>,
    right: ArrayView3<'_, T>,
) -> Result<u64> {
    differing_elements_by(left, right, |one, other| one.bits() == other.bits())
}

/// A floating-point type compared by its bit pattern. See [`differing_bits`].
pub trait Bitwise {
    fn bits(&self) -> u64;
}

impl Bitwise for f32 {
    fn bits(&self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Bitwise for f64 {
    fn bits(&self) -> u64 {
        self.to_bits()
    }
}

#[cfg(test)]
mod borrowed_tests {
    use super::*;

    fn ramp(shape: [usize; 3]) -> Voxels {
        let mut array = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        for (index, value) in array.iter_mut().enumerate() {
            *value = index as f64 + 1.0;
        }
        Voxels::F64(array)
    }

    /// **The property the whole type exists for**: the halves are disjoint, they
    /// cover the axis exactly once, and two threads may write them at the same
    /// time.
    ///
    /// Written through `std::thread::scope` rather than in sequence on purpose.
    /// A sequential test would pass on a `split_at` that handed out two views of
    /// the *same* half, because the second write would simply overwrite the
    /// first; only two concurrent writers make the overlap a wrong answer rather
    /// than a race the test cannot lose.
    #[test]
    fn a_split_hands_out_disjoint_halves_that_two_threads_may_write() {
        let mut block = Voxels::zeros(Dtype::F64, [8, 3, 2]).expect("a block");
        let (mut low, mut high) = VoxelsMut::of(&mut block).split_at(0, 3).expect("a split");
        assert_eq!(low.shape(), [3, 3, 2]);
        assert_eq!(high.shape(), [5, 3, 2]);
        let ones = Voxels::F64(Array3::from_elem((3, 3, 2), 1.0));
        let twos = Voxels::F64(Array3::from_elem((5, 3, 2), 2.0));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                low.assign_from(&ones, &Region::whole(&[3, 3, 2]))
                    .expect("the low half")
            });
            scope.spawn(|| {
                high.assign_from(&twos, &Region::whole(&[5, 3, 2]))
                    .expect("the high half")
            });
        });
        let written = block.view::<f64>().expect("f64");
        for i in 0..8 {
            let want = if i < 3 { 1.0 } else { 2.0 };
            for j in 0..3 {
                for k in 0..2 {
                    assert_eq!(written[[i, j, k]], want, "at {i},{j},{k}");
                }
            }
        }
    }

    /// **One structure, not two**: placing a slab's core through a borrowed view
    /// gives the same bytes as placing it through the owned block.
    ///
    /// The two paths share `copy_view`, and this is what says so. Without it the
    /// borrowed path could drift — an off-by-one in the narrowing, a transposed
    /// region — and every slicing test would keep passing, because they compare
    /// a cut answer against an uncut one and both would go through the new path.
    #[test]
    fn the_borrowed_placement_agrees_with_the_owned_one() {
        let block = [9usize, 4, 3];
        let source = ramp([5, 4, 3]);
        let core = Region::new(&[2, 0, 0], &[4, 4, 3]);
        let within = Region::new(&[1, 0, 0], &[4, 4, 3]);

        let mut owned = Voxels::zeros(Dtype::F64, block).expect("a block");
        owned
            .assign_region_from(&core, &source, &within)
            .expect("the owned path");

        let mut borrowed = Voxels::zeros(Dtype::F64, block).expect("a block");
        {
            let (_, rest) = VoxelsMut::of(&mut borrowed)
                .split_at(0, core.start[0])
                .expect("the head");
            let (mut mine, _) = rest.split_at(0, core.shape[0]).expect("the core");
            mine.assign_from(&source, &within)
                .expect("the borrowed path");
        }
        assert_eq!(owned, borrowed, "the two placements must be the same bytes");
        // And they placed something, so the equality is not two empty blocks.
        assert!(owned.view::<f64>().expect("f64").iter().any(|v| *v != 0.0));
    }

    /// The edges of the split, which is what lets a caller peel views off in a
    /// loop with no special case for the last one.
    #[test]
    fn a_split_at_either_end_answers_an_empty_half_and_past_it_is_refused() {
        let mut block = Voxels::zeros(Dtype::U16, [6, 2, 2]).expect("a block");
        let (low, high) = VoxelsMut::of(&mut block)
            .split_at(0, 6)
            .expect("at the end");
        assert_eq!(low.shape(), [6, 2, 2]);
        assert_eq!(
            high.shape(),
            [0, 2, 2],
            "the empty half is a view of nothing"
        );
        let (low, high) = VoxelsMut::of(&mut block)
            .split_at(0, 0)
            .expect("at the start");
        assert_eq!(low.shape(), [0, 2, 2]);
        assert_eq!(high.shape(), [6, 2, 2]);
        let error = VoxelsMut::of(&mut block)
            .split_at(0, 7)
            .expect_err("past the end must be refused");
        assert!(
            format!("{error}").contains("cannot split an axis of 6"),
            "{error}"
        );
        assert!(
            VoxelsMut::of(&mut block).split_at(3, 1).is_err(),
            "a block is 3-D"
        );
    }

    /// The tag is still attached, so a mismatch is a sentence rather than a
    /// transmute.
    #[test]
    fn a_borrowed_block_refuses_a_source_of_another_element_type() {
        let mut block = Voxels::zeros(Dtype::F64, [2, 2, 2]).expect("a block");
        let mut view = VoxelsMut::of(&mut block);
        assert_eq!(view.dtype(), Dtype::F64);
        let wrong = Voxels::zeros(Dtype::U8, [2, 2, 2]).expect("a block");
        let error = view
            .assign_from(&wrong, &Region::whole(&[2, 2, 2]))
            .expect_err("a mismatch must be refused");
        assert!(
            format!("{error}").contains("float64") && format!("{error}").contains("uint8"),
            "the refusal must name both: {error}"
        );
    }
}

#[cfg(test)]
mod tests {

    /// [`Voxels::nonzero`] is what widening then comparing was, on every
    /// variant.
    ///
    /// Asserted against the spelling it replaces rather than against a hand
    /// written expectation, because the point of the method is that it is the
    /// same answer for less work — and a `bool` image, which used to make a
    /// round trip through `f64` to say what it already said, is the arm most
    /// likely to drift.
    #[test]
    fn nonzero_is_the_widened_comparison_without_the_widening() {
        let cases = [
            Voxels::Bool(
                Array3::from_shape_vec((2, 1, 2), vec![true, false, false, true]).unwrap(),
            ),
            Voxels::U8(Array3::from_shape_vec((2, 1, 2), vec![0u8, 1, 255, 0]).unwrap()),
            Voxels::U16(Array3::from_shape_vec((2, 1, 2), vec![0u16, 7, 0, 65535]).unwrap()),
            Voxels::I32(Array3::from_shape_vec((2, 1, 2), vec![0i32, -1, 5, 0]).unwrap()),
            Voxels::F32(Array3::from_shape_vec((2, 1, 2), vec![0.0f32, -0.0, 1.5, -2.0]).unwrap()),
            Voxels::F64(Array3::from_shape_vec((2, 1, 2), vec![0.0f64, -0.0, 1.5, -2.0]).unwrap()),
        ];
        for image in cases {
            assert_eq!(
                image.nonzero(),
                image.widened().mapv(|value| value != 0.0),
                "{:?}",
                image.dtype()
            );
        }
    }
    use super::*;

    /// The exact integer widening is exact, and refuses everything it cannot do
    /// exactly.
    ///
    /// The negative controls are the point. `widened` is infallible and lossy;
    /// this is fallible and is not, and a test that only checked the easy values
    /// would not tell the two apart.
    #[test]
    fn the_exact_integer_widening_refuses_rather_than_rounds() {
        // Exact past 2^53, where `widened`'s `f64` is not. The loss does not
        // show as a wrong number after a round trip — the `as` cast back
        // saturates and hides it — it shows as two *different* labels becoming
        // one, which is the failure that matters for an identifier.
        let large = Voxels::I64(Array3::from_elem((1, 1, 1), i64::MAX));
        assert_eq!(large.widened_i64("labels").unwrap()[[0, 0, 0]], i64::MAX);
        assert_eq!(large.widened()[[0, 0, 0]], (i64::MAX - 1) as f64);

        // Every integer type, and `bool` as zero and one.
        for (image, expected) in [
            (Voxels::Bool(Array3::from_elem((1, 1, 1), true)), 1i64),
            (Voxels::U8(Array3::from_elem((1, 1, 1), u8::MAX)), 255),
            (
                Voxels::U32(Array3::from_elem((1, 1, 1), u32::MAX)),
                4_294_967_295,
            ),
            (Voxels::I8(Array3::from_elem((1, 1, 1), i8::MIN)), -128),
            (
                Voxels::I32(Array3::from_elem((1, 1, 1), i32::MIN)),
                -2_147_483_648,
            ),
            (
                Voxels::U64(Array3::from_elem((1, 1, 1), i64::MAX as u64)),
                i64::MAX,
            ),
        ] {
            assert_eq!(
                image.widened_i64("labels").unwrap()[[0, 0, 0]],
                expected,
                "{:?}",
                image.dtype()
            );
        }

        // The shape is the image's, not a flattening.
        let block = Voxels::U16(Array3::from_shape_fn((2, 3, 4), |(z, y, x)| {
            (z * 12 + y * 4 + x) as u16
        }));
        let wide = block.widened_i64("labels").unwrap();
        assert_eq!(wide.dim(), (2, 3, 4));
        assert_eq!(wide[[1, 2, 3]], 23);

        // `uint64` past `i64::MAX` is refused with the value in the message.
        // Wrapping would have given `-1`, which is a label a caller would
        // believe.
        let past = Voxels::U64(Array3::from_elem((1, 1, 1), u64::MAX));
        let text = past.widened_i64("labels.npy").unwrap_err().to_string();
        assert!(text.contains("labels.npy"), "{text}");
        assert!(text.contains("18446744073709551615"), "{text}");
        assert!(text.contains("i64::MAX"), "{text}");
        assert_eq!(u64::MAX as i64, -1);

        // The float types are refused by name, both of them, and the refusal
        // says what to do instead.
        for (image, name) in [
            (Voxels::F64(Array3::from_elem((1, 1, 1), 1.5f64)), "float64"),
            (Voxels::F32(Array3::from_elem((1, 1, 1), 1.5f32)), "float32"),
        ] {
            let text = image.widened_i64("field.npy").unwrap_err().to_string();
            assert!(text.contains("field.npy") && text.contains(name), "{text}");
            assert!(text.contains("mapv"), "{text}");
        }
    }

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

#[cfg(test)]
mod differing_elements_tests {
    use super::*;
    use ndarray::{s, Array3};

    fn ramp(shape: (usize, usize, usize)) -> Array3<f64> {
        let mut array = Array3::<f64>::zeros(shape);
        for (index, value) in array.iter_mut().enumerate() {
            *value = index as f64;
        }
        array
    }

    /// The defect this whole function exists to prevent: `zip` truncates, so the
    /// hand-rolled form answers a *small* count for volumes of different
    /// extents. Refused by name, in both argument orders.
    #[test]
    fn two_extents_are_refused_rather_than_truncated() {
        let small = ramp((2, 2, 2));
        let large = ramp((4, 2, 2));
        let error = differing_elements(small.view(), large.view())
            .expect_err("two extents must be refused");
        let message = format!("{error}");
        assert!(
            message.contains('2') && message.contains('4'),
            "the refusal must name both extents: {message}"
        );
        assert!(differing_elements(large.view(), small.view()).is_err());

        // What the truncating form would have answered, kept here so the
        // difference between refusing and answering is visible rather than
        // asserted: the first eight elements agree, so a `zip` would report 0
        // differences between a volume of 8 and a volume of 16.
        let truncated = small
            .iter()
            .zip(large.iter())
            .filter(|(one, other)| one != other)
            .count();
        assert_eq!(
            truncated, 0,
            "the truncating form answers zero here, which is the pass that hides the failure"
        );
    }

    #[test]
    fn a_plain_difference_is_counted() {
        let left = ramp((3, 2, 2));
        let mut right = left.clone();
        right[[0, 0, 0]] = -1.0;
        right[[2, 1, 1]] = -1.0;
        assert_eq!(
            differing_elements(left.view(), left.view()).expect("same"),
            0
        );
        assert_eq!(
            differing_elements(left.view(), right.view()).expect("differ"),
            2
        );
        assert_eq!(
            differing_bits(left.view(), right.view()).expect("differ"),
            2
        );
    }

    /// The nuance that makes two functions rather than one: the two comparisons
    /// disagree on exactly the inputs a floating-point parity suite exists to
    /// look at, and they disagree in **opposite directions**.
    #[test]
    fn nan_and_signed_zero_are_where_the_two_comparisons_part() {
        let mut left = Array3::<f64>::zeros((1, 1, 2));
        let mut right = Array3::<f64>::zeros((1, 1, 2));
        left[[0, 0, 0]] = f64::NAN;
        right[[0, 0, 0]] = f64::NAN;
        left[[0, 0, 1]] = 0.0;
        right[[0, 0, 1]] = -0.0;

        // Two NaNs of identical bits: not equal as values, identical as bits.
        assert_eq!(
            left[[0, 0, 0]].to_bits(),
            right[[0, 0, 0]].to_bits(),
            "the fixture must hand both functions the same bits, or this proves nothing"
        );
        // Signed zero: equal as values, different as bits.
        assert!(left[[0, 0, 1]] == right[[0, 0, 1]]);
        assert_ne!(left[[0, 0, 1]].to_bits(), right[[0, 0, 1]].to_bits());

        // So each function counts exactly one of the two, and they are not the
        // same one.
        assert_eq!(
            differing_elements(left.view(), right.view()).expect("values"),
            1,
            "by value: the NaNs differ and the zeroes agree"
        );
        assert_eq!(
            differing_bits(left.view(), right.view()).expect("bits"),
            1,
            "by bits: the NaNs agree and the zeroes differ"
        );
    }

    /// The case the `Array` sibling exists for: a slice of a larger volume,
    /// passed without materialising it. An implementation reaching for
    /// `as_slice()` would fail here.
    #[test]
    fn a_non_contiguous_view_is_compared_in_logical_order() {
        let volume = ramp((4, 4, 4));
        let left = volume.slice(s![1..3, .., ..]);
        let mut other = volume.clone();
        other[[2, 0, 0]] = -1.0;
        let right = other.slice(s![1..3, .., ..]);
        assert!(
            !left.is_standard_layout() || left.len() < volume.len(),
            "the fixture must be a strict sub-view, or this measures the contiguous case"
        );
        assert_eq!(differing_elements(left, right).expect("sliced"), 1);

        // And a sub-view of a *different* extent is still refused.
        assert!(differing_elements(volume.slice(s![1..3, .., ..]), volume.view()).is_err());
    }

    #[test]
    fn it_works_for_the_types_the_callers_actually_hold() {
        let left = Array3::<bool>::from_elem((2, 2, 2), true);
        let mut right = left.clone();
        right[[1, 1, 1]] = false;
        assert_eq!(
            differing_elements(left.view(), right.view()).expect("bool"),
            1
        );

        let a = Array3::<f32>::from_elem((2, 2, 2), 1.0);
        let mut b = a.clone();
        b[[0, 0, 0]] = -0.0;
        b[[0, 0, 1]] = 0.0;
        assert_eq!(
            differing_elements(a.view(), b.view()).expect("f32 values"),
            2,
            "1.0 differs from both zeroes by value"
        );
        assert_eq!(differing_bits(a.view(), b.view()).expect("f32 bits"), 2);
    }
}
