// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The one array file format, read and written here rather than in every
// consumer.
//
// Why it belongs in this crate
// ----------------------------
// `region.rs`'s header used to record that the npy backend stayed in the
// sibling application. That decision was counted: **nineteen private readers in
// one sibling's test directory alone**, and a twentieth written by the worker
// who counted them. Every one of them is the same two hundred lines — a magic
// check, a dict literal, a `descr`, a shape, a memcpy — and every one of them is
// a place the format can be got wrong privately.
//
// The boundary argument was already made next door: [`Dtype::numpy_name`] lives
// in `dtype.rs`, documented as being "for manifests and cross-language
// comparison". A reader takes some bytes and a shape and names nothing about
// what the array is *of*, so by `lib.rs`'s own rule it is general.
//
// Memory order is a decision, not a default
// -----------------------------------------
// **Both orders are read, and neither is transposed.** This is the one place the
// format bites hardest, and it bit a real worker: in the recordings this crate's
// consumers read daily, the coordinate tables are `fortran_order: True` and the
// per-row annotations beside them are C order — the *same* directory, the *same*
// run. A reader that assumes C order turns a `(378, 3)` table into a permutation
// of the same 1134 numbers and **passes every count, every shape and every
// dtype**. There is no assertion downstream that catches it.
//
// So the order is handled rather than assumed: a Fortran-ordered file is
// returned as an array with Fortran strides and the file's own logical shape, so
// `array[[row, column]]` means what the writer meant. A caller that genuinely
// cannot accept one order says so with [`OrderPolicy::Only`] and gets a refusal
// naming the order it found, which is the *other* honest answer. What is not on
// offer is reading a Fortran file as C, because that is the answer that looks
// right.
//
// Streaming, and why the writer never holds a second copy
// ------------------------------------------------------
// A single tile is 1.775 GB as `bool`, and one consumer counts non-zeros without
// widening precisely so as not to pay 14.2 GB for the same volume as `f64`.
// Serialising it through a `Vec<u8>` would double it again, so
// [`write_array_to`] encodes into a fixed 64 KiB buffer and flushes, whatever
// the array's rank, dtype or memory order. [`read_array_from`] is the mirror
// image: it decodes out of a fixed buffer, so the only full-size allocation is
// the array the caller asked for.
//
// [`NpySource`] and [`NpySink`] go further and never hold the volume at all —
// an npy file is a flat, uncompressed, contiguous buffer at a known offset, so a
// rectangular region is a set of contiguous runs and a seek per run. That makes
// this crate's own `RegionSource`/`RegionSink` a better fit for npy than for
// most backends, which is why they are implemented here and not merely
// mentioned.
//
// What is refused, and why refusing is the feature
// ------------------------------------------------
// Every one of these is a file this reader could have half-read into a plausible
// wrong answer:
//
// | refusal | what it prevents |
// |---|---|
// | not the magic | reading a `.npz`, a `.tif` or a truncated download as an array |
// | format version 3.0 or later, or 0.x | a header spelled in a scheme this does not implement |
// | truncated prelude / header / data | a short file yielding a short array |
// | `descr` is a list | a structured dtype flattened into the wrong element count |
// | `descr` names a kind with no variant here | `complex128` read as pairs of `f64` |
// | `descr` says `=` or omits the byte-order character on a multi-byte type | a file whose byte order is whatever the writer's machine was |
// | the element type is not the one asked for | `int64` handed to a caller expecting `float64` |
// | Fortran order under `Only(Order::C)` | the silent transpose above |
// | the byte count contradicts the shape | a shape mis-parse that still fills an array |
// | rank is not 3, for [`Voxels`] | a table read as a volume |
// | `float16` | there is no `Voxels` variant and no Rust primitive to hold it |
//
// Written bytes are numpy's bytes
// -------------------------------
// The writer reproduces numpy's own header spelling, its key order and its
// padding rule exactly — 64-byte alignment of the data, a version chosen by
// header length, `(5,)` for a one-axis shape and `()` for none — so a file
// written here is byte-identical to the one `numpy.save` writes for the same
// array. `tests/npy_files.rs` asserts that against bytes numpy actually
// produced, which is a stronger statement than a round trip through this crate's
// own reader: a round trip only proves the two halves agree with each other.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::{Array3, ArrayBase, ArrayD, ArrayViewD, Data, Dimension, Ix3, IxDyn, ShapeBuilder};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::region::{Region, RegionSink, RegionSource};
use crate::voxels::Voxels;

/// The six bytes every `.npy` file starts with.
const MAGIC: &[u8] = b"\x93NUMPY";

/// numpy pads the header so the data starts at a multiple of this. Files older
/// than numpy 1.9 use 16 instead, which is why the reader takes the *declared*
/// header length and never computes one.
const ALIGNMENT: usize = 64;

/// Bytes encoded or decoded between writes. Fixed, so a volume is never held
/// twice.
const BUFFER: usize = 64 * 1024;

// ------------------------------------------------------------------- order --

/// Which axis varies fastest in the stored bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Order {
    /// Last axis fastest — `numpy`'s default, and what this crate's arrays are.
    C,
    /// First axis fastest — what a transposed or `asfortranarray` array is
    /// saved as, and what a great many recorded coordinate tables are.
    Fortran,
}

impl Order {
    /// The name a refusal uses.
    pub fn name(self) -> &'static str {
        match self {
            Order::C => "C order",
            Order::Fortran => "Fortran order",
        }
    }

    /// The header's spelling of it.
    pub fn fortran_order(self) -> bool {
        matches!(self, Order::Fortran)
    }
}

impl fmt::Display for Order {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// What a caller will accept a file to be stored in.
///
/// A parameter rather than a constant, because which answer is right is the
/// caller's to know: code that indexes the array logically wants
/// [`OrderPolicy::Either`], and code that hands the buffer to something which
/// assumes a layout wants [`OrderPolicy::Only`] and a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderPolicy {
    /// Read either. The array comes back with the file's logical shape and the
    /// strides that make its indexing correct, so nothing is transposed and
    /// nothing is copied into the other order behind the caller's back.
    Either,
    /// Read only this order; the other is refused by name.
    Only(Order),
}

impl OrderPolicy {
    fn check(self, found: Order, what: &str) -> Result<()> {
        match self {
            OrderPolicy::Either => Ok(()),
            OrderPolicy::Only(wanted) if wanted == found => Ok(()),
            OrderPolicy::Only(wanted) => Err(Error::invalid(format!(
                "{what}: the file is stored in {} and this caller accepts only {}. Reading it as \
                 {} would return the same elements in a different arrangement — a plausible wrong \
                 answer rather than an error — so it is refused. Read it with \
                 `OrderPolicy::Either`, which returns an array whose indexing is the writer's.",
                found.name(),
                wanted.name(),
                wanted.name()
            ))),
        }
    }
}

// ------------------------------------------------------------- byte order --

/// What the `descr`'s first character says about the element bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endian {
    Little,
    Big,
    /// One byte wide, so there is nothing to order.
    NotApplicable,
}

// ------------------------------------------------------------------ header --

/// A `.npy` header, parsed.
///
/// Public because a caller often wants the shape and the element type *without*
/// the array — a manifest line, a shape check before a plan, a decision about
/// which of two files to read. That question costs one `read` of the first
/// kilobyte here, against the whole file everywhere else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// The `descr` exactly as the file spells it, kept because a refusal that
    /// quotes it is worth more than one that does not.
    pub descr: String,
    /// The element type it names.
    pub dtype: Dtype,
    /// The byte order it names.
    pub endian: Endian,
    /// The memory order the elements are stored in.
    pub order: Order,
    /// The logical shape. May be empty (a zero-rank array, one element) and may
    /// contain a zero (no elements at all).
    pub shape: Vec<usize>,
    /// The format version, `(1, 0)` or `(2, 0)`.
    pub version: (u8, u8),
    /// Byte offset of the first element.
    pub data_offset: usize,
}

impl Header {
    /// Elements the shape describes. A zero-rank shape is one element, not
    /// none: an empty product is 1, which is also what numpy means by `()`.
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes of data the header declares, or a refusal if that does not fit a
    /// `usize`.
    pub fn data_bytes(&self, what: &str) -> Result<usize> {
        self.elements()
            .checked_mul(self.dtype.size_of())
            .ok_or_else(|| {
                Error::invalid(format!(
                    "{what}: shape {:?} of {} does not fit an address space.",
                    self.shape,
                    self.dtype.numpy_name()
                ))
            })
    }

    /// Total size of a file with this header, in bytes.
    pub fn file_bytes(&self, what: &str) -> Result<usize> {
        Ok(self.data_offset + self.data_bytes(what)?)
    }

    /// Refuse unless the element type is the one the caller is reading as.
    fn check_element<T: NpyElement>(&self, what: &str) -> Result<()> {
        if self.dtype == T::DTYPE {
            return Ok(());
        }
        Err(Error::invalid(format!(
            "{what}: the file holds {} (`{}`) and it is being read as {}. An element type is not \
             converted here — a reader that widened or narrowed silently would turn a wrong \
             `descr` into plausible numbers.",
            self.dtype.numpy_name(),
            self.descr,
            T::DTYPE.numpy_name()
        )))
    }

    /// Parse a header from the front of `bytes`.
    pub fn parse(bytes: &[u8], what: &str) -> Result<Self> {
        let mut cursor = bytes;
        Self::read_from(&mut cursor, what)
    }

    /// Parse a header off the front of a reader, leaving it positioned at the
    /// first element.
    pub fn read_from(reader: &mut impl Read, what: &str) -> Result<Self> {
        let mut prelude = [0u8; 10];
        fill(reader, &mut prelude, what, "the ten-byte prelude")?;
        if &prelude[..6] != MAGIC {
            return Err(Error::invalid(format!(
                "{what}: not a `.npy` file — the first six bytes are {:?} and numpy's magic is \
                 `\\x93NUMPY`.",
                &prelude[..6.min(prelude.len())]
            )));
        }
        let version = (prelude[6], prelude[7]);
        let (declared, data_offset) = match version {
            (1, _) => (u16::from_le_bytes([prelude[8], prelude[9]]) as usize, 10),
            (2, _) => {
                let mut rest = [0u8; 2];
                fill(reader, &mut rest, what, "the four-byte header length")?;
                (
                    u32::from_le_bytes([prelude[8], prelude[9], rest[0], rest[1]]) as usize,
                    12,
                )
            }
            (major, minor) => {
                return Err(Error::invalid(format!(
                    "{what}: `.npy` format version {major}.{minor} is not implemented here, which \
                     reads 1.0 and 2.0. Version 3.0 spells its header in UTF-8, which only a \
                     structured dtype needs and this reader refuses anyway; a 0.x file is not a \
                     format numpy ever wrote."
                )))
            }
        };
        let mut text = vec![0u8; declared];
        fill(
            reader,
            &mut text,
            what,
            &format!("a header of the declared {declared} bytes"),
        )?;
        let text = std::str::from_utf8(&text).map_err(|_| {
            Error::invalid(format!(
                "{what}: the {declared}-byte header is not valid UTF-8, so it is not a header this \
                 reads."
            ))
        })?;
        let (descr, order, shape) = parse_header_text(text, what)?;
        let (dtype, endian) = dtype_of_descr(&descr, what)?;
        Ok(Header {
            descr,
            dtype,
            endian,
            order,
            shape,
            version,
            data_offset: data_offset + declared,
        })
    }

    /// The bytes of a header for this element type, shape and order — numpy's
    /// own spelling, key order and padding.
    pub fn render(dtype: Dtype, shape: &[usize], order: Order, what: &str) -> Result<Vec<u8>> {
        let text = format!(
            "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
            descr_of(dtype, what)?,
            if order.fortran_order() {
                "True"
            } else {
                "False"
            },
            shape_literal(shape)
        );
        // numpy's rule, reproduced rather than approximated: the padding is
        // whatever takes the prelude plus the header plus its closing newline to
        // a multiple of `ALIGNMENT`, and it is never zero — a header that
        // already lands on the boundary gets a whole further block.
        for (major, prelude) in [(1u8, 10usize), (2u8, 12usize)] {
            let padding = ALIGNMENT - ((prelude + text.len() + 1) % ALIGNMENT);
            let declared = text.len() + padding + 1;
            let fits = match major {
                1 => declared <= u16::MAX as usize,
                _ => declared <= u32::MAX as usize,
            };
            if !fits {
                continue;
            }
            let mut bytes = Vec::with_capacity(prelude + declared);
            bytes.extend_from_slice(MAGIC);
            bytes.push(major);
            bytes.push(0);
            match major {
                1 => bytes.extend_from_slice(&(declared as u16).to_le_bytes()),
                _ => bytes.extend_from_slice(&(declared as u32).to_le_bytes()),
            }
            bytes.extend_from_slice(text.as_bytes());
            bytes.resize(prelude + text.len() + padding, b' ');
            bytes.push(b'\n');
            return Ok(bytes);
        }
        Err(Error::invalid(format!(
            "{what}: a header of {} bytes exceeds what format version 2.0 can declare.",
            text.len()
        )))
    }
}

/// numpy's `repr` of a shape tuple, which is not the obvious one for zero and
/// one axes: `()` and `(5,)`.
fn shape_literal(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [only] => format!("({only},)"),
        many => {
            let pieces: Vec<String> = many.iter().map(usize::to_string).collect();
            format!("({})", pieces.join(", "))
        }
    }
}

/// Read exactly `into.len()` bytes, or refuse naming what ran out.
fn fill(reader: &mut impl Read, into: &mut [u8], what: &str, wanted: &str) -> Result<()> {
    let mut filled = 0;
    while filled < into.len() {
        match reader.read(&mut into[filled..]) {
            Ok(0) => {
                return Err(Error::invalid(format!(
                "{what}: truncated — {wanted} needs {} bytes and the file ended after {filled}.",
                into.len()
            )))
            }
            Ok(count) => filled += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(Error::backend(error)),
        }
    }
    Ok(())
}

/// The value of one key of the header's dict literal, as the text just after
/// its colon.
///
/// A scanner rather than a `split`, because the three keys may appear in any
/// order, either quote style is legal, and the whitespace around the colon is
/// the writer's business.
fn after_key<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    for quote in ['\'', '"'] {
        let needle = format!("{quote}{key}{quote}");
        if let Some(at) = text.find(&needle) {
            let rest = text[at + needle.len()..].trim_start();
            if let Some(rest) = rest.strip_prefix(':') {
                return Some(rest.trim_start());
            }
        }
    }
    None
}

/// `descr`, `fortran_order` and `shape`, out of the dict literal.
fn parse_header_text(text: &str, what: &str) -> Result<(String, Order, Vec<usize>)> {
    let trimmed = text.trim_end().trim_end_matches('\n').trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err(Error::invalid(format!(
            "{what}: the header is not a dict literal — it reads {:?}.",
            &trimmed[..trimmed.len().min(60)]
        )));
    }

    let value = after_key(trimmed, "descr").ok_or_else(|| {
        Error::invalid(format!(
            "{what}: the header names no `descr`, so no element type."
        ))
    })?;
    let descr = match value.chars().next() {
        Some(quote @ ('\'' | '"')) => value[quote.len_utf8()..]
            .split(quote)
            .next()
            .ok_or_else(|| Error::invalid(format!("{what}: the `descr` string is not closed.")))?
            .to_string(),
        _ => {
            return Err(Error::invalid(format!(
                "{what}: `descr` is not a plain dtype string but {:?}. A list is a structured \
                 dtype — several named fields per element — which has no single element type and \
                 no variant here, so it is refused rather than flattened.",
                &value[..value.len().min(60)]
            )))
        }
    };

    let value = after_key(trimmed, "fortran_order").ok_or_else(|| {
        Error::invalid(format!(
            "{what}: the header names no `fortran_order`, so it does not say which axis varies \
             fastest. Guessing is the failure this reader exists to avoid."
        ))
    })?;
    let order = if value.starts_with("True") {
        Order::Fortran
    } else if value.starts_with("False") {
        Order::C
    } else {
        return Err(Error::invalid(format!(
            "{what}: `fortran_order` is {:?} and the only values are `True` and `False`.",
            &value[..value.len().min(20)]
        )));
    };

    let value = after_key(trimmed, "shape")
        .ok_or_else(|| Error::invalid(format!("{what}: the header names no `shape`.")))?;
    let inner = value
        .strip_prefix('(')
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| {
            Error::invalid(format!(
                "{what}: `shape` is {:?} and a shape is a parenthesised tuple.",
                &value[..value.len().min(40)]
            ))
        })?;
    let mut shape = Vec::new();
    for piece in inner.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        shape.push(piece.parse::<usize>().map_err(|_| {
            Error::invalid(format!(
                "{what}: `shape` axis {piece:?} is not a non-negative integer."
            ))
        })?);
    }
    Ok((descr, order, shape))
}

/// The element type and byte order a `descr` names.
fn dtype_of_descr(descr: &str, what: &str) -> Result<(Dtype, Endian)> {
    let refuse = |reason: &str| {
        Error::invalid(format!(
            "{what}: `descr` `{descr}` — {reason}. This crate's element types are `bool`, the \
             eight fixed-width integers, `float32` and `float64`, in either byte order."
        ))
    };
    let mut rest = descr;
    let marker = match rest.chars().next() {
        Some(first @ ('<' | '>' | '|' | '=')) => {
            rest = &rest[first.len_utf8()..];
            Some(first)
        }
        Some(_) => None,
        None => return Err(refuse("it is empty")),
    };
    let kind = rest
        .chars()
        .next()
        .ok_or_else(|| refuse("it names no kind"))?;
    let width: usize = rest[kind.len_utf8()..]
        .parse()
        .map_err(|_| refuse("its width is not a number"))?;
    let dtype = match (kind, width) {
        ('b', 1) => Dtype::Bool,
        ('u', 1) => Dtype::U8,
        ('u', 2) => Dtype::U16,
        ('u', 4) => Dtype::U32,
        ('u', 8) => Dtype::U64,
        ('i', 1) => Dtype::I8,
        ('i', 2) => Dtype::I16,
        ('i', 4) => Dtype::I32,
        ('i', 8) => Dtype::I64,
        ('f', 2) => Dtype::F16,
        ('f', 4) => Dtype::F32,
        ('f', 8) => Dtype::F64,
        _ => return Err(refuse("it names a kind and width with no variant here")),
    };
    let endian = if dtype.size_of() == 1 {
        Endian::NotApplicable
    } else {
        match marker {
            Some('<') => Endian::Little,
            Some('>') => Endian::Big,
            Some('=') => {
                return Err(Error::invalid(format!(
                    "{what}: `descr` `{descr}` says `=` — the byte order of whichever machine \
                     wrote it. A file that does not carry its own byte order is read differently \
                     on two machines, so it is refused rather than assumed to be this one's."
                )))
            }
            _ => {
                return Err(Error::invalid(format!(
                    "{what}: `descr` `{descr}` is {width} bytes wide and carries no byte-order \
                     character. Only a one-byte type may leave it out."
                )))
            }
        }
    };
    Ok((dtype, endian))
}

/// The `descr` this crate writes for an element type — numpy's own spelling on
/// a little-endian machine.
pub fn descr_of(dtype: Dtype, what: &str) -> Result<&'static str> {
    Ok(match dtype {
        Dtype::Bool => "|b1",
        Dtype::U8 => "|u1",
        Dtype::I8 => "|i1",
        Dtype::U16 => "<u2",
        Dtype::I16 => "<i2",
        Dtype::U32 => "<u4",
        Dtype::I32 => "<i4",
        Dtype::U64 => "<u8",
        Dtype::I64 => "<i8",
        Dtype::F32 => "<f4",
        Dtype::F64 => "<f8",
        Dtype::F16 => {
            return Err(Error::invalid(format!(
                "{what}: `float16` is not written here. Rust has no native 16-bit float and \
                 `Voxels` has no `F16` variant to hold one, so `Dtype::F16` is a byte-width tag \
                 for storage — see `voxels.rs`, which refuses it for the same reason.",
            )))
        }
    })
}

// ---------------------------------------------------------------- elements --

/// One element type this reads and writes.
///
/// Separate from [`crate::voxels::VoxelElement`], which says how to get in and
/// out of the `Voxels` enum and nothing about bytes. This says only how an
/// element crosses the file boundary, so a caller with a plain `ArrayD<u16>`
/// and no interest in `Voxels` needs neither.
pub trait NpyElement: Copy + Send + Sync + 'static {
    /// The tag whose `descr` this is written under.
    const DTYPE: Dtype;

    /// One element from exactly [`Dtype::size_of`] bytes.
    fn read_element(chunk: &[u8], endian: Endian) -> Self;

    /// One element, little-endian, appended.
    fn append_element(self, out: &mut Vec<u8>);
}

macro_rules! npy_number {
    ($type:ty, $dtype:expr) => {
        impl NpyElement for $type {
            const DTYPE: Dtype = $dtype;

            fn read_element(chunk: &[u8], endian: Endian) -> Self {
                let mut raw = [0u8; std::mem::size_of::<$type>()];
                raw.copy_from_slice(chunk);
                if matches!(endian, Endian::Big) {
                    raw.reverse();
                }
                <$type>::from_le_bytes(raw)
            }

            fn append_element(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}

npy_number!(u8, Dtype::U8);
npy_number!(u16, Dtype::U16);
npy_number!(u32, Dtype::U32);
npy_number!(u64, Dtype::U64);
npy_number!(i8, Dtype::I8);
npy_number!(i16, Dtype::I16);
npy_number!(i32, Dtype::I32);
npy_number!(i64, Dtype::I64);
npy_number!(f32, Dtype::F32);
npy_number!(f64, Dtype::F64);

/// numpy's `|b1` is one byte per element, `0` or `1` on write and *non-zero* on
/// read — the same convention `VoxelElement for bool` states for `f64`.
impl NpyElement for bool {
    const DTYPE: Dtype = Dtype::Bool;

    fn read_element(chunk: &[u8], _endian: Endian) -> Self {
        chunk[0] != 0
    }

    fn append_element(self, out: &mut Vec<u8>) {
        out.push(u8::from(self));
    }
}

// ----------------------------------------------------------------- reading --

/// Build an array of `shape` from elements in `order`'s own traversal.
fn assemble<T>(shape: &[usize], order: Order, values: Vec<T>, what: &str) -> Result<ArrayD<T>> {
    let built = IxDyn(shape).set_f(order.fortran_order());
    ArrayD::from_shape_vec(built, values).map_err(|error| {
        Error::invalid(format!(
            "{what}: shape {shape:?} does not accept the elements read for it ({error})."
        ))
    })
}

/// Decode `count` elements out of a reader through a fixed buffer.
fn decode<T: NpyElement>(
    reader: &mut impl Read,
    count: usize,
    endian: Endian,
    what: &str,
) -> Result<Vec<T>> {
    let width = T::DTYPE.size_of();
    let mut values: Vec<T> = Vec::with_capacity(count);
    let per_pass = (BUFFER / width).max(1);
    let mut raw = vec![0u8; per_pass * width];
    let mut done = 0;
    while done < count {
        let this_pass = per_pass.min(count - done);
        let slice = &mut raw[..this_pass * width];
        fill(
            reader,
            slice,
            what,
            &format!("{count} elements of {}", T::DTYPE.numpy_name()),
        )?;
        values.extend(
            slice
                .chunks_exact(width)
                .map(|chunk| T::read_element(chunk, endian)),
        );
        done += this_pass;
    }
    Ok(values)
}

/// Read a whole array off a reader, holding one copy of it and nothing more.
///
/// `what` names the source in every refusal; `policy` decides what happens to a
/// Fortran-ordered file. Both are the caller's.
pub fn read_array_from<T: NpyElement>(
    reader: &mut impl Read,
    what: &str,
    policy: OrderPolicy,
) -> Result<ArrayD<T>> {
    let header = Header::read_from(reader, what)?;
    policy.check(header.order, what)?;
    header.check_element::<T>(what)?;
    let values = decode::<T>(reader, header.elements(), header.endian, what)?;
    assemble(&header.shape, header.order, values, what)
}

/// Read a whole array out of bytes that are exactly one file.
///
/// Refuses trailing bytes as well as missing ones: a shape that parses to fewer
/// elements than the file holds is the same mistake as one that parses to more,
/// and it is the direction that otherwise succeeds.
pub fn read_array<T: NpyElement>(
    bytes: &[u8],
    what: &str,
    policy: OrderPolicy,
) -> Result<ArrayD<T>> {
    let header = Header::parse(bytes, what)?;
    policy.check(header.order, what)?;
    header.check_element::<T>(what)?;
    let wanted = header.file_bytes(what)?;
    if bytes.len() != wanted {
        return Err(Error::invalid(format!(
            "{what}: the header declares shape {:?} of {}, which is {wanted} bytes with its \
             header, and there are {}.",
            header.shape,
            header.dtype.numpy_name(),
            bytes.len()
        )));
    }
    let mut cursor = &bytes[header.data_offset..];
    let values = decode::<T>(&mut cursor, header.elements(), header.endian, what)?;
    assemble(&header.shape, header.order, values, what)
}

/// The same, from a path. The path names itself in every refusal.
pub fn read_array_file<T: NpyElement>(path: &Path, policy: OrderPolicy) -> Result<ArrayD<T>> {
    let what = path.display().to_string();
    let file = File::open(path).map_err(Error::backend)?;
    let mut reader = std::io::BufReader::with_capacity(BUFFER, file);
    read_array_from::<T>(&mut reader, &what, policy)
}

/// A file's header without its data.
pub fn read_header_file(path: &Path) -> Result<Header> {
    let what = path.display().to_string();
    let file = File::open(path).map_err(Error::backend)?;
    let mut reader = std::io::BufReader::with_capacity(BUFFER, file);
    Header::read_from(&mut reader, &what)
}

/// Every `Voxels` variant, by the tag a header carries.
macro_rules! voxels_by_dtype {
    ($dtype:expr, $what:expr, |$element:ident| $body:expr) => {
        match $dtype {
            Dtype::Bool => {
                type $element = bool;
                $body
            }
            Dtype::U8 => {
                type $element = u8;
                $body
            }
            Dtype::U16 => {
                type $element = u16;
                $body
            }
            Dtype::U32 => {
                type $element = u32;
                $body
            }
            Dtype::U64 => {
                type $element = u64;
                $body
            }
            Dtype::I8 => {
                type $element = i8;
                $body
            }
            Dtype::I16 => {
                type $element = i16;
                $body
            }
            Dtype::I32 => {
                type $element = i32;
                $body
            }
            Dtype::I64 => {
                type $element = i64;
                $body
            }
            Dtype::F32 => {
                type $element = f32;
                $body
            }
            Dtype::F64 => {
                type $element = f64;
                $body
            }
            Dtype::F16 => {
                return Err(Error::invalid(format!(
                    "{}: the file holds `float16`, and `Voxels` has no variant for it — Rust has \
                     no native 16-bit float. Reading it would mean widening, and a reader that \
                     widened would hand back numbers no file contains.",
                    $what
                )))
            }
        }
    };
}

/// Read a rank-3 file as a [`Voxels`], with the element type the file names.
///
/// The rank is checked rather than reshaped: a `(378, 3)` table read as a volume
/// is exactly the mistake this module exists to stop.
pub fn read_voxels_from(reader: &mut impl Read, what: &str, policy: OrderPolicy) -> Result<Voxels> {
    let header = Header::read_from(reader, what)?;
    policy.check(header.order, what)?;
    check_holdable(&header, what)?;
    check_volume(&header, what)?;
    let elements = header.elements();
    let endian = header.endian;
    let shape = header.shape.clone();
    let order = header.order;
    voxels_by_dtype!(header.dtype, what, |Element| {
        let values = decode::<Element>(reader, elements, endian, what)?;
        into_voxels::<Element>(assemble(&shape, order, values, what)?, what)
    })
}

/// The same, out of bytes that are exactly one file.
pub fn read_voxels(bytes: &[u8], what: &str, policy: OrderPolicy) -> Result<Voxels> {
    let header = Header::parse(bytes, what)?;
    policy.check(header.order, what)?;
    check_holdable(&header, what)?;
    check_volume(&header, what)?;
    let wanted = header.file_bytes(what)?;
    if bytes.len() != wanted {
        return Err(Error::invalid(format!(
            "{what}: the header declares shape {:?} of {}, which is {wanted} bytes with its \
             header, and there are {}.",
            header.shape,
            header.dtype.numpy_name(),
            bytes.len()
        )));
    }
    let mut cursor = &bytes[header.data_offset..];
    let elements = header.elements();
    let endian = header.endian;
    let shape = header.shape.clone();
    let order = header.order;
    voxels_by_dtype!(header.dtype, what, |Element| {
        let values = decode::<Element>(&mut cursor, elements, endian, what)?;
        into_voxels::<Element>(assemble(&shape, order, values, what)?, what)
    })
}

/// The same, from a path.
pub fn read_voxels_file(path: &Path, policy: OrderPolicy) -> Result<Voxels> {
    let what = path.display().to_string();
    let file = File::open(path).map_err(Error::backend)?;
    let mut reader = std::io::BufReader::with_capacity(BUFFER, file);
    read_voxels_from(&mut reader, &what, policy)
}

/// Refuse an element type `Voxels` has no variant for, before anything else —
/// so a `float16` file says `float16` whatever its rank is.
fn check_holdable(header: &Header, what: &str) -> Result<()> {
    if header.dtype == Dtype::F16 {
        return Err(Error::invalid(format!(
            "{what}: the file holds `float16`, and `Voxels` has no variant for it — Rust has no              native 16-bit float. Reading it would mean widening, and a reader that widened would              hand back numbers no file contains."
        )));
    }
    Ok(())
}

fn check_volume(header: &Header, what: &str) -> Result<()> {
    if header.shape.len() != 3 {
        return Err(Error::invalid(format!(
            "{what}: the array is {}-dimensional with shape {:?}, and a `Voxels` is rank 3. Read \
             it with `read_array` if the rank is what the file is supposed to be.",
            header.shape.len(),
            header.shape
        )));
    }
    Ok(())
}

fn into_voxels<T>(array: ArrayD<T>, what: &str) -> Result<Voxels>
where
    T: crate::voxels::VoxelElement + NpyElement,
{
    let array: Array3<T> = array.into_dimensionality::<Ix3>().map_err(|error| {
        Error::invalid(format!("{what}: a rank-3 array was not rank 3 ({error})."))
    })?;
    Ok(T::wrap(array))
}

// ----------------------------------------------------------------- writing --

/// Encode an array's elements in `order` through a fixed buffer.
fn encode<T, S, D>(array: &ArrayBase<S, D>, order: Order, out: &mut impl Write) -> Result<()>
where
    T: NpyElement,
    S: Data<Elem = T>,
    D: Dimension,
{
    // `iter()` walks in logical row-major order whatever the strides are, so it
    // is C order by definition; reversing the axes first makes the same walk
    // column-major. Neither materialises anything.
    let view = if order.fortran_order() {
        array.view().into_dyn().reversed_axes()
    } else {
        array.view().into_dyn()
    };
    let mut buffer: Vec<u8> = Vec::with_capacity(BUFFER + 16);
    for value in view.iter() {
        value.append_element(&mut buffer);
        if buffer.len() >= BUFFER {
            out.write_all(&buffer).map_err(Error::backend)?;
            buffer.clear();
        }
    }
    if !buffer.is_empty() {
        out.write_all(&buffer).map_err(Error::backend)?;
    }
    Ok(())
}

/// Write an array as a `.npy`, streamed — nothing here holds a second copy of
/// it, whatever its size.
pub fn write_array_to<T, S, D>(
    array: &ArrayBase<S, D>,
    order: Order,
    what: &str,
    out: &mut impl Write,
) -> Result<()>
where
    T: NpyElement,
    S: Data<Elem = T>,
    D: Dimension,
{
    let header = Header::render(T::DTYPE, array.shape(), order, what)?;
    out.write_all(&header).map_err(Error::backend)?;
    encode(array, order, out)
}

/// The same, into memory — for a small array or a test.
pub fn write_array<T, S, D>(array: &ArrayBase<S, D>, order: Order, what: &str) -> Result<Vec<u8>>
where
    T: NpyElement,
    S: Data<Elem = T>,
    D: Dimension,
{
    let mut bytes = Vec::with_capacity(128 + array.len() * T::DTYPE.size_of());
    write_array_to(array, order, what, &mut bytes)?;
    Ok(bytes)
}

/// The same, to a path, through a buffered writer.
pub fn write_array_file<T, S, D>(path: &Path, array: &ArrayBase<S, D>, order: Order) -> Result<()>
where
    T: NpyElement,
    S: Data<Elem = T>,
    D: Dimension,
{
    let what = path.display().to_string();
    let file = File::create(path).map_err(Error::backend)?;
    let mut out = BufWriter::with_capacity(BUFFER, file);
    write_array_to(array, order, &what, &mut out)?;
    out.flush().map_err(Error::backend)?;
    Ok(())
}

/// Write a [`Voxels`] as a `.npy`, streamed.
pub fn write_voxels_to(
    voxels: &Voxels,
    order: Order,
    what: &str,
    out: &mut impl Write,
) -> Result<()> {
    match voxels {
        Voxels::Bool(array) => write_array_to(array, order, what, out),
        Voxels::U8(array) => write_array_to(array, order, what, out),
        Voxels::U16(array) => write_array_to(array, order, what, out),
        Voxels::U32(array) => write_array_to(array, order, what, out),
        Voxels::U64(array) => write_array_to(array, order, what, out),
        Voxels::I8(array) => write_array_to(array, order, what, out),
        Voxels::I16(array) => write_array_to(array, order, what, out),
        Voxels::I32(array) => write_array_to(array, order, what, out),
        Voxels::I64(array) => write_array_to(array, order, what, out),
        Voxels::F32(array) => write_array_to(array, order, what, out),
        Voxels::F64(array) => write_array_to(array, order, what, out),
    }
}

/// The same, into memory.
pub fn write_voxels(voxels: &Voxels, order: Order, what: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(128 + voxels.bytes() as usize);
    write_voxels_to(voxels, order, what, &mut bytes)?;
    Ok(bytes)
}

/// The same, to a path.
pub fn write_voxels_file(path: &Path, voxels: &Voxels, order: Order) -> Result<()> {
    let what = path.display().to_string();
    let file = File::create(path).map_err(Error::backend)?;
    let mut out = BufWriter::with_capacity(BUFFER, file);
    write_voxels_to(voxels, order, &what, &mut out)?;
    out.flush().map_err(Error::backend)?;
    Ok(())
}

// -------------------------------------------------------------------- runs --

/// Visit the contiguous runs a region occupies in a file of `whole` stored in
/// `order`, in the order those runs appear in the region's own memory layout.
///
/// This is the whole of what makes an npy file a region source: the data is a
/// flat contiguous buffer, so a rectangular box is `region.voxels() /
/// run_length` runs and one seek each. `visit` is handed an offset **in
/// elements** and a run **in elements**.
fn for_each_run(
    order: Order,
    whole: &[usize],
    region: &Region,
    mut visit: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    if region.voxels() == 0 {
        return Ok(());
    }
    let rank = whole.len();
    if rank == 0 {
        // A zero-rank array is one element at offset zero.
        return visit(0, 1);
    }
    let mut strides = vec![0usize; rank];
    let mut running = 1usize;
    if order.fortran_order() {
        for axis in 0..rank {
            strides[axis] = running;
            running *= whole[axis];
        }
    } else {
        for axis in (0..rank).rev() {
            strides[axis] = running;
            running *= whole[axis];
        }
    }
    let contiguous = if order.fortran_order() { 0 } else { rank - 1 };
    // The remaining axes, slowest first — which is the reverse of the stride
    // order on each side.
    let outer: Vec<usize> = if order.fortran_order() {
        (1..rank).rev().collect()
    } else {
        (0..rank - 1).collect()
    };
    let base: usize = (0..rank)
        .map(|axis| region.start[axis] * strides[axis])
        .sum();
    let run = region.shape[contiguous];
    let mut index = vec![0usize; outer.len()];
    loop {
        let offset = base
            + outer
                .iter()
                .zip(index.iter())
                .map(|(&axis, &at)| at * strides[axis])
                .sum::<usize>();
        visit(offset, run)?;
        let mut position = outer.len();
        loop {
            if position == 0 {
                return Ok(());
            }
            position -= 1;
            index[position] += 1;
            if index[position] < region.shape[outer[position]] {
                break;
            }
            index[position] = 0;
        }
    }
}

// ------------------------------------------------------- source and sink --

/// A `.npy` file read by region.
///
/// Holds a file handle and a parsed header and **never the volume**: a region is
/// a seek and a read per contiguous run. That is what makes this crate's
/// `RegionSource` a real fit for the format rather than a wrapper over "read it
/// all and slice" — a 1.775 GB file costs the region and nothing else.
///
/// The element type is static and the file's is not, so [`NpySource::open`]
/// checks them against each other and refuses by name. There is no widening.
pub struct NpySource<T> {
    what: String,
    path: PathBuf,
    header: Header,
    file: Mutex<File>,
    element: PhantomData<fn() -> T>,
}

impl<T: NpyElement> NpySource<T> {
    /// Open `path`, parse its header and check it against `T` and `policy`.
    pub fn open(path: &Path, policy: OrderPolicy) -> Result<Self> {
        let what = path.display().to_string();
        let mut file = File::open(path).map_err(Error::backend)?;
        let header = Header::read_from(&mut file, &what)?;
        policy.check(header.order, &what)?;
        header.check_element::<T>(&what)?;
        let wanted = header.file_bytes(&what)? as u64;
        let actual = file.metadata().map_err(Error::backend)?.len();
        if actual != wanted {
            return Err(Error::invalid(format!(
                "{what}: the header declares shape {:?} of {}, which is {wanted} bytes with its \
                 header, and the file is {actual}.",
                header.shape,
                header.dtype.numpy_name()
            )));
        }
        Ok(Self {
            what,
            path: path.to_path_buf(),
            header,
            file: Mutex::new(file),
            element: PhantomData,
        })
    }

    /// What the file says about itself.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The path it was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl<T: NpyElement> fmt::Debug for NpySource<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NpySource")
            .field("path", &self.path)
            .field("header", &self.header)
            .finish()
    }
}

impl<T: NpyElement> RegionSource<T> for NpySource<T> {
    fn shape(&self) -> &[usize] {
        &self.header.shape
    }

    /// One box, read as the file's own contiguous runs.
    ///
    /// The array has exactly `region.shape` and the **file's strides** — a
    /// Fortran-ordered file yields a Fortran-strided box. Indexing, `iter()`,
    /// `assign` and equality are all logical in `ndarray`, so every consumer in
    /// this crate is unaffected; a caller that wants a contiguous buffer asks
    /// for one with `as_standard_layout`, which is a copy and is therefore the
    /// caller's to ask for.
    fn read_region(&self, region: &Region) -> Result<ArrayD<T>> {
        region.check_within(&self.header.shape, &self.what)?;
        let width = T::DTYPE.size_of();
        let endian = self.header.endian;
        let offset = self.header.data_offset as u64;
        let mut values: Vec<T> = Vec::with_capacity(region.voxels());
        let mut raw: Vec<u8> = Vec::new();
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for_each_run(self.header.order, &self.header.shape, region, |at, run| {
            raw.resize(run * width, 0);
            file.seek(SeekFrom::Start(offset + (at * width) as u64))
                .map_err(Error::backend)?;
            fill(&mut *file, &mut raw, &self.what, "a run of the region")?;
            values.extend(
                raw.chunks_exact(width)
                    .map(|chunk| T::read_element(chunk, endian)),
            );
            Ok(())
        })?;
        drop(file);
        assemble(&region.shape, self.header.order, values, &self.what)
    }

    fn describe(&self) -> String {
        format!(
            "npy {:?} of {} in {} at {}",
            self.header.shape,
            self.header.dtype.numpy_name(),
            self.header.order.name(),
            self.what
        )
    }

    /// Always `None`, and deliberately. A `.npy` file is dense — there is no
    /// absent chunk and no fill value to consult — so the only way to know a
    /// region is zero is to read it, and claiming otherwise would silently zero
    /// real data.
    fn is_known_empty(&self, _region: &Region) -> Option<bool> {
        None
    }
}

/// A `.npy` file written by region.
///
/// The header and the full extent of the data are laid down by
/// [`NpySink::create`] and each region is a seek and a write, so a volume larger
/// than memory is written a block at a time. Writes are order-independent, which
/// is what the trait promises and what the flat layout makes true.
///
/// **An unwritten voxel is the zero byte pattern**, because sizing the file is
/// not the same as filling it. That is the same stated loss `Voxels::unwritten`
/// records for the integer types: a coverage hole is caught by
/// [`crate::tiling::boxes_tile_exactly`] and by the write accounting, not by
/// looking at the data. [`NpySink::create_filled`] pays a full-size write to
/// have a loud value there instead, and which of the two a caller wants is the
/// caller's.
pub struct NpySink<T> {
    what: String,
    path: PathBuf,
    header: Header,
    file: Mutex<File>,
    element: PhantomData<fn() -> T>,
}

impl<T: NpyElement> NpySink<T> {
    /// Create `path` with a header for `shape` and a data area of the right
    /// size, holding zero bytes.
    pub fn create(path: &Path, shape: &[usize], order: Order) -> Result<Self> {
        let sink = Self::lay_out(path, shape, order)?;
        let bytes = sink.header.file_bytes(&sink.what)? as u64;
        {
            let file = sink
                .file
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            file.set_len(bytes).map_err(Error::backend)?;
        }
        Ok(sink)
    }

    /// The same, with every element written as `value` first.
    ///
    /// Costs one full-size pass, which is exactly what [`NpySink::create`]
    /// avoids; it exists so that "an unwritten voxel is loud" is available to a
    /// caller who would rather pay for it than rely on the tiling guard.
    pub fn create_filled(path: &Path, shape: &[usize], order: Order, value: T) -> Result<Self> {
        let sink = Self::lay_out(path, shape, order)?;
        let elements = sink.header.elements();
        let width = T::DTYPE.size_of();
        let per_pass = (BUFFER / width).max(1);
        let mut pattern: Vec<u8> = Vec::with_capacity(per_pass * width);
        for _ in 0..per_pass {
            value.append_element(&mut pattern);
        }
        {
            let mut file = sink
                .file
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            file.seek(SeekFrom::Start(sink.header.data_offset as u64))
                .map_err(Error::backend)?;
            let mut done = 0;
            while done < elements {
                let this_pass = per_pass.min(elements - done);
                file.write_all(&pattern[..this_pass * width])
                    .map_err(Error::backend)?;
                done += this_pass;
            }
        }
        Ok(sink)
    }

    fn lay_out(path: &Path, shape: &[usize], order: Order) -> Result<Self> {
        let what = path.display().to_string();
        let bytes = Header::render(T::DTYPE, shape, order, &what)?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(Error::backend)?;
        file.write_all(&bytes).map_err(Error::backend)?;
        let header = Header::parse(&bytes, &what)?;
        // A header this crate rendered must parse back; if it does not, the two
        // halves have drifted and every file written since is suspect.
        debug_assert_eq!(header.shape, shape);
        Ok(Self {
            what,
            path: path.to_path_buf(),
            header,
            file: Mutex::new(file),
            element: PhantomData,
        })
    }

    /// The header the file carries.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The path it was created at.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl<T: NpyElement> fmt::Debug for NpySink<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NpySink")
            .field("path", &self.path)
            .field("header", &self.header)
            .finish()
    }
}

impl<T: NpyElement> RegionSink<T> for NpySink<T> {
    fn shape(&self) -> &[usize] {
        &self.header.shape
    }

    fn write_region(&self, start: &[usize], data: &ArrayD<T>) -> Result<()> {
        let region = Region::new(start, data.shape());
        region.check_within(&self.header.shape, &self.what)?;
        let width = T::DTYPE.size_of();
        let offset = self.header.data_offset as u64;
        // The same walk `encode` uses, so the run the file wants and the
        // elements the caller handed over are in step by construction.
        let view: ArrayViewD<'_, T> = if self.header.order.fortran_order() {
            data.view().reversed_axes()
        } else {
            data.view()
        };
        let mut source = view.iter();
        let mut raw: Vec<u8> = Vec::new();
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for_each_run(self.header.order, &self.header.shape, &region, |at, run| {
            raw.clear();
            raw.reserve(run * width);
            for _ in 0..run {
                let value = source.next().ok_or_else(|| {
                    Error::invalid(format!(
                        "{}: the region ran out of elements before its runs did.",
                        self.what
                    ))
                })?;
                value.append_element(&mut raw);
            }
            file.seek(SeekFrom::Start(offset + (at * width) as u64))
                .map_err(Error::backend)?;
            file.write_all(&raw).map_err(Error::backend)?;
            Ok(())
        })?;
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        let file = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        file.sync_all().map_err(Error::backend)
    }

    fn describe(&self) -> String {
        format!(
            "npy sink {:?} of {} in {} at {}",
            self.header.shape,
            self.header.dtype.numpy_name(),
            self.header.order.name(),
            self.what
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// The three shape spellings numpy uses, which are not the obvious ones.
    #[test]
    fn a_shape_is_spelled_the_way_numpy_spells_it() {
        assert_eq!(shape_literal(&[]), "()");
        assert_eq!(shape_literal(&[5]), "(5,)");
        assert_eq!(shape_literal(&[2, 3, 4]), "(2, 3, 4)");
    }

    /// Whatever the header, the data starts on a 64-byte boundary and the
    /// padding is never nothing.
    #[test]
    fn every_rendered_header_lands_the_data_on_the_alignment() {
        for rank in 0..6usize {
            let shape: Vec<usize> = (0..rank).map(|axis| axis + 1).collect();
            for order in [Order::C, Order::Fortran] {
                let bytes = Header::render(Dtype::U16, &shape, order, "<memory>").unwrap();
                assert_eq!(bytes.len() % ALIGNMENT, 0, "{shape:?} {order}");
                assert_eq!(*bytes.last().unwrap(), b'\n', "{shape:?} {order}");
                let header = Header::parse(&bytes, "<memory>").unwrap();
                assert_eq!(header.shape, shape);
                assert_eq!(header.order, order);
                assert_eq!(header.data_offset, bytes.len());
            }
        }
    }

    /// A long shape needs a 2.0 header, and the reader takes it.
    #[test]
    fn a_header_too_long_for_two_bytes_becomes_a_version_two_file() {
        let shape: Vec<usize> = (0..25000).map(|_| 1usize).collect();
        let bytes = Header::render(Dtype::Bool, &shape, Order::C, "<memory>").unwrap();
        assert_eq!(bytes[6], 2, "the version was not promoted");
        let header = Header::parse(&bytes, "<memory>").unwrap();
        assert_eq!(header.version, (2, 0));
        assert_eq!(header.shape.len(), 25000);
        assert_eq!(header.elements(), 1);
    }

    /// The dict keys may come in any order and in either quote style.
    #[test]
    fn the_header_is_read_as_a_dict_rather_than_by_position() {
        let text = "{\"shape\": (2, 3), \"fortran_order\": True , \"descr\":\"<i4\"}";
        let (descr, order, shape) = parse_header_text(text, "<memory>").unwrap();
        assert_eq!(descr, "<i4");
        assert_eq!(order, Order::Fortran);
        assert_eq!(shape, vec![2, 3]);
    }

    /// Every element type this crate holds survives a header round trip.
    #[test]
    fn every_element_type_names_itself_in_a_descr_and_reads_back() {
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
            let descr = descr_of(dtype, "<memory>").unwrap();
            let (back, endian) = dtype_of_descr(descr, "<memory>").unwrap();
            assert_eq!(back, dtype, "{descr}");
            if dtype.size_of() == 1 {
                assert_eq!(endian, Endian::NotApplicable, "{descr}");
            } else {
                assert_eq!(endian, Endian::Little, "{descr}");
            }
        }
        assert!(descr_of(Dtype::F16, "<memory>").is_err());
    }

    /// Big-endian is read, not refused — the swap is the whole of it.
    #[test]
    fn the_byte_order_character_is_obeyed_rather_than_assumed() {
        assert_eq!(
            dtype_of_descr(">i4", "<memory>").unwrap(),
            (Dtype::I32, Endian::Big)
        );
        assert_eq!(i32::read_element(&[0x80, 0, 0, 0], Endian::Big), i32::MIN);
        assert_eq!(
            i32::read_element(&[0, 0, 0, 0x80], Endian::Little),
            i32::MIN
        );
        // One byte wide, so the marker cannot matter.
        assert_eq!(
            dtype_of_descr(">u1", "<memory>").unwrap(),
            (Dtype::U8, Endian::NotApplicable)
        );
    }

    /// A Fortran file is read as a Fortran array, not as a transposed C one.
    #[test]
    fn a_fortran_file_round_trips_with_its_own_indexing() {
        let rows = Array2::from_shape_fn((5, 3), |(row, column)| (row * 3 + column) as f64);
        let bytes = write_array(&rows, Order::Fortran, "<memory>").unwrap();
        let back: ArrayD<f64> = read_array(&bytes, "<memory>", OrderPolicy::Either).unwrap();
        assert_eq!(back.shape(), &[5, 3]);
        for row in 0..5 {
            for column in 0..3 {
                assert_eq!(back[[row, column]], rows[[row, column]]);
            }
        }
        // And the bytes really are the other order: the C-order file of the same
        // array differs from byte 128 on.
        let c_order = write_array(&rows, Order::C, "<memory>").unwrap();
        assert_ne!(bytes, c_order);
        assert_eq!(bytes.len(), c_order.len());
    }

    /// The refusal names the order it found and the order it wanted.
    #[test]
    fn a_policy_that_takes_one_order_refuses_the_other_by_name() {
        let rows = Array2::<i16>::zeros((4, 3));
        let bytes = write_array(&rows, Order::Fortran, "<memory>").unwrap();
        let error = read_array::<i16>(&bytes, "table.npy", OrderPolicy::Only(Order::C))
            .expect_err("refused");
        let text = error.to_string();
        assert!(text.contains("table.npy"), "{text}");
        assert!(text.contains("Fortran order"), "{text}");
        assert!(text.contains("C order"), "{text}");
        // The other direction refuses too, so this is a policy and not a bias.
        let c_order = write_array(&rows, Order::C, "<memory>").unwrap();
        assert!(read_array::<i16>(&c_order, "t", OrderPolicy::Only(Order::Fortran)).is_err());
        assert!(read_array::<i16>(&c_order, "t", OrderPolicy::Only(Order::C)).is_ok());
    }

    /// Zero-rank and zero-sized arrays are arrays.
    #[test]
    fn a_shape_may_be_empty_or_contain_a_zero() {
        let scalar = ArrayD::<f64>::from_elem(IxDyn(&[]), 3.5);
        let bytes = write_array(&scalar, Order::C, "<memory>").unwrap();
        assert!(String::from_utf8_lossy(&bytes[10..80]).contains("'shape': (), "));
        let back: ArrayD<f64> = read_array(&bytes, "<memory>", OrderPolicy::Either).unwrap();
        assert_eq!(back, scalar);

        for shape in [vec![0usize, 3], vec![2, 0, 4], vec![0]] {
            let empty = ArrayD::<i16>::zeros(IxDyn(&shape));
            let bytes = write_array(&empty, Order::C, "<memory>").unwrap();
            assert_eq!(bytes.len() % ALIGNMENT, 0, "no data, so only the header");
            let back: ArrayD<i16> = read_array(&bytes, "<memory>", OrderPolicy::Either).unwrap();
            assert_eq!(back.shape(), &shape[..]);
            assert_eq!(back.len(), 0);
        }
    }

    /// Each refusal names what it refused.
    #[test]
    fn every_refusal_says_what_it_found() {
        let good = write_array(&Array1::<u16>::zeros(4), Order::C, "<memory>").unwrap();

        let mut wrong_magic = good.clone();
        wrong_magic[1] = b'X';
        let text = Header::parse(&wrong_magic, "f.npy")
            .unwrap_err()
            .to_string();
        assert!(text.contains("magic"), "{text}");

        let text = Header::parse(&good[..4], "f.npy").unwrap_err().to_string();
        assert!(
            text.contains("truncated") && text.contains("prelude"),
            "{text}"
        );

        let text = Header::parse(&good[..40], "f.npy").unwrap_err().to_string();
        assert!(
            text.contains("truncated") && text.contains("header"),
            "{text}"
        );

        let mut future = good.clone();
        future[6] = 4;
        let text = Header::parse(&future, "f.npy").unwrap_err().to_string();
        assert!(text.contains("version 4.0"), "{text}");

        let text = read_array::<u16>(&good[..good.len() - 2], "f.npy", OrderPolicy::Either)
            .unwrap_err()
            .to_string();
        assert!(text.contains("bytes with its header"), "{text}");

        let text = read_array::<f64>(&good, "f.npy", OrderPolicy::Either)
            .unwrap_err()
            .to_string();
        assert!(
            text.contains("uint16") && text.contains("float64"),
            "{text}"
        );

        let text = dtype_of_descr("=f8", "f.npy").unwrap_err().to_string();
        assert!(text.contains("whichever machine"), "{text}");

        let text = dtype_of_descr("f8", "f.npy").unwrap_err().to_string();
        assert!(text.contains("no byte-order character"), "{text}");

        let text = dtype_of_descr("<c16", "f.npy").unwrap_err().to_string();
        assert!(text.contains("no variant here"), "{text}");
    }

    /// A three-dimensional file is a `Voxels`; anything else says so.
    #[test]
    fn only_a_rank_three_file_is_a_volume() {
        let volume: Voxels =
            Array3::<u16>::from_shape_fn((2, 3, 4), |(z, y, x)| (z * 12 + y * 4 + x) as u16).into();
        let bytes = write_voxels(&volume, Order::C, "<memory>").unwrap();
        assert_eq!(
            read_voxels(&bytes, "<memory>", OrderPolicy::Either).unwrap(),
            volume
        );

        let table = write_array(&Array2::<u16>::zeros((5, 3)), Order::C, "<memory>").unwrap();
        let text = read_voxels(&table, "t.npy", OrderPolicy::Either)
            .unwrap_err()
            .to_string();
        assert!(
            text.contains("2-dimensional") && text.contains("rank 3"),
            "{text}"
        );
    }

    /// The runs of a region are the file's own contiguous stretches, in the
    /// order the region's memory layout wants them.
    #[test]
    fn a_region_is_contiguous_runs_along_the_fastest_axis() {
        let region = Region::new(&[1, 1], &[2, 2]);
        let mut c_runs = Vec::new();
        for_each_run(Order::C, &[4, 5], &region, |at, run| {
            c_runs.push((at, run));
            Ok(())
        })
        .unwrap();
        assert_eq!(c_runs, vec![(6, 2), (11, 2)]);

        let mut f_runs = Vec::new();
        for_each_run(Order::Fortran, &[4, 5], &region, |at, run| {
            f_runs.push((at, run));
            Ok(())
        })
        .unwrap();
        assert_eq!(f_runs, vec![(5, 2), (9, 2)]);

        // An empty region has no runs at all, and a zero-rank one has exactly
        // the single element.
        let mut none = 0;
        for_each_run(Order::C, &[4, 5], &Region::new(&[0, 0], &[0, 5]), |_, _| {
            none += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(none, 0);
        let mut one = Vec::new();
        for_each_run(Order::C, &[], &Region::whole(&[]), |at, run| {
            one.push((at, run));
            Ok(())
        })
        .unwrap();
        assert_eq!(one, vec![(0, 1)]);
    }
}
