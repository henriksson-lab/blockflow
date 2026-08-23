// SPDX-License-Identifier: MIT
//
// The npy reader and writer, against files this crate did not write.
//
// Why a round trip is not enough
// ------------------------------
// `src/npy.rs`'s own tests write a file and read it back. That proves the two
// halves agree with each other and nothing about whether either is right — two
// wrong halves agree perfectly. So this file works from **bytes numpy actually
// produced**, pasted in as byte-string literals with the array that made each
// one named in its doc comment. They are short enough to read: the header is
// ASCII and sits in plain sight, so a reviewer can check the claim without
// running anything.
//
// The set is chosen for the corners the format has rather than for coverage of
// the happy path:
//
// * **Both memory orders**, including a `(5, 3)` table stored Fortran-ordered —
//   the miniature of the trap that motivated the module. Fifteen numbers in a
//   permutation that passes every count.
// * **A `descr` that is not the little-endian spelling** — `>i4`, whose
//   extremes are chosen so that a reader ignoring the `>` gets numbers rather
//   than an error.
// * **Format version 2.0**, whose header length is four bytes rather than two,
//   and **3.0**, which this does not implement and refuses by name.
// * **Shapes with no axes and shapes with a zero in them** — `()`, `(0, 3)`,
//   `(2, 0, 4)` — which numpy spells in three different ways and a naive parser
//   gets wrong in three different ways.
// * **Three dtypes with no variant here** — `float16`, a structured dtype and
//   `complex128` — each of which could be half-read into a plausible array.
// * **The float and integer extremes**, so that "read what numpy wrote" covers
//   the bit patterns and not just the ordinary numbers.
//
// And the other direction: for every case where a little-endian C or Fortran
// file is what numpy produced, this asserts the bytes written **here** are
// byte-identical to it. numpy's header spelling, key order, `(5,)` shape
// literal and 64-byte padding rule are all reproduced rather than approximated,
// so that is a check and not an aspiration.
//
// Recorded files
// --------------
// `a_directory_of_recorded_files_reads_and_re_serialises_exactly` walks a
// directory named by `BLOCKFLOW_NPY_FIXTURES` and, for every file in it, reads
// the array and writes it back, comparing bytes. It is skipped when the
// variable is unset, so the suite is hermetic; the path is a parameter because
// a recording directory is the caller's, not this crate's. `BLOCKFLOW_NPY_MAX`
// caps the file size it will read, because a recording set is measured in
// hundreds of gigabytes and a test is not.

use std::path::{Path, PathBuf};

use blockflow::npy::{
    read_array, read_array_file, read_array_file_as, read_array_mapped, read_elements,
    read_header_file, read_voxels, write_array, write_elements, write_voxels, Endian, Header,
    NpyElement, NpySink, NpySource, Order, OrderPolicy,
};
use blockflow::{Dtype, Region, RegionSink, RegionSource, Voxels};
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};

// ------------------------------------------------ what numpy actually wrote --

/// `np.arange(24, dtype='<u2').reshape(2, 3, 4)`
const C_ORDER_U16_2X3X4: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<u2', 'fortran_order': False, 'shape': (2, 3, 4), }                                                       \x0a\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\x09\x00\x0a\x00\x0b\x00\x0c\x00\x0d\x00\x0e\x00\x0f\x00\x10\x00\x11\x00\x12\x00\x13\x00\x14\x00\x15\x00\x16\x00\x17\x00";

/// `np.asfortranarray(np.arange(15.0).reshape(5, 3))` -- the transpose trap in miniature
const FORTRAN_ORDER_F64_5X3: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f8', 'fortran_order': True, 'shape': (5, 3), }                                                           \x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\"@\x00\x00\x00\x00\x00\x00(@\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00*@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00&@\x00\x00\x00\x00\x00\x00,@";

/// `np.asfortranarray(np.arange(24).reshape(2, 3, 4) % 3 == 0)`
const FORTRAN_ORDER_BOOL_2X3X4: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '|b1', 'fortran_order': True, 'shape': (2, 3, 4), }                                                        \x0a\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x01\x01\x00\x00\x01\x01\x00\x00\x00\x00";

/// `np.array([i32::MIN, -1, 0, 1, i32::MAX], dtype='>i4')` -- a descr that is not the little-endian spelling
const BIG_ENDIAN_I32_5: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '>i4', 'fortran_order': False, 'shape': (5,), }                                                            \x0a\x80\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x01\x7f\xff\xff\xff";

/// `np.arange(6, dtype='<i8').reshape(2, 3)`, forced to a 2.0 header (four-byte length)
const VERSION_2_0_I64_2X3: &[u8] = b"\x93NUMPY\x02\x00t\x00\x00\x00{'descr': '<i8', 'fortran_order': False, 'shape': (2, 3), }                                                        \x0a\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00";

/// the same array, forced to a 3.0 header
const VERSION_3_0_I64_2X3: &[u8] = b"\x93NUMPY\x03\x00t\x00\x00\x00{'descr': '<i8', 'fortran_order': False, 'shape': (2, 3), }                                                        \x0a\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00";

/// `np.array(3.5)` -- shape `()`, one element, no axes
const ZERO_RANK_F64: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f8', 'fortran_order': False, 'shape': (), }                                                              \x0a\x00\x00\x00\x00\x00\x00\x0c@";

/// `np.zeros((0, 3), dtype='<f4')` -- a zero in the shape, no data bytes at all
const EMPTY_F32_0X3: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f4', 'fortran_order': False, 'shape': (0, 3), }                                                          \x0a";

/// `np.zeros((2, 0, 4), dtype='<i2')` -- the zero is not the first axis
const EMPTY_I16_2X0X4: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<i2', 'fortran_order': False, 'shape': (2, 0, 4), }                                                       \x0a";

/// `np.array([1.0, 2.0], dtype='<f2')`
const FLOAT16_2: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f2', 'fortran_order': False, 'shape': (2,), }                                                            \x0a\x00\x3c\x00\x40";

/// a structured dtype -- `descr` is a list rather than a string
const STRUCTURED_2: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': [('a', '<i4'), ('b', '<f8')], 'fortran_order': False, 'shape': (2,), }                                       \x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

/// `np.array([1 + 2j])`
const COMPLEX128_1: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<c16', 'fortran_order': False, 'shape': (1,), }                                                           \x0a\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@";

/// `[0.0, -0.0, nan, inf, -inf, MIN_POSITIVE, EPSILON]` as float64
const F64_EXTREMES_7: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f8', 'fortran_order': False, 'shape': (7,), }                                                            \x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf0\x7f\x00\x00\x00\x00\x00\x00\xf0\xff\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\xb0<";

/// the same seven as float32
const F32_EXTREMES_7: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f4', 'fortran_order': False, 'shape': (7,), }                                                            \x0a\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\xc0\x7f\x00\x00\x80\x7f\x00\x00\x80\xff\x00\x00\x80\x00\x00\x00\x004";

/// `[0, u64::MAX]`
const U64_EXTREMES_2: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<u8', 'fortran_order': False, 'shape': (2,), }                                                            \x0a\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff";

/// `[i64::MIN, i64::MAX]`
const I64_EXTREMES_2: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<i8', 'fortran_order': False, 'shape': (2,), }                                                            \x0a\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff\xff\xff\xff\xff\xff\x7f";

// ---------------------------------------------------- reading numpy's bytes --

/// Every embedded file parses to the header numpy put in it.
#[test]
fn the_header_of_each_recorded_file_is_read_as_numpy_wrote_it() {
    let expected: &[(&[u8], &str, Dtype, Order, &[usize], (u8, u8))] = &[
        (
            C_ORDER_U16_2X3X4,
            "<u2",
            Dtype::U16,
            Order::C,
            &[2, 3, 4],
            (1, 0),
        ),
        (
            FORTRAN_ORDER_F64_5X3,
            "<f8",
            Dtype::F64,
            Order::Fortran,
            &[5, 3],
            (1, 0),
        ),
        (
            FORTRAN_ORDER_BOOL_2X3X4,
            "|b1",
            Dtype::Bool,
            Order::Fortran,
            &[2, 3, 4],
            (1, 0),
        ),
        (BIG_ENDIAN_I32_5, ">i4", Dtype::I32, Order::C, &[5], (1, 0)),
        (
            VERSION_2_0_I64_2X3,
            "<i8",
            Dtype::I64,
            Order::C,
            &[2, 3],
            (2, 0),
        ),
        (ZERO_RANK_F64, "<f8", Dtype::F64, Order::C, &[], (1, 0)),
        (EMPTY_F32_0X3, "<f4", Dtype::F32, Order::C, &[0, 3], (1, 0)),
        (
            EMPTY_I16_2X0X4,
            "<i2",
            Dtype::I16,
            Order::C,
            &[2, 0, 4],
            (1, 0),
        ),
        (FLOAT16_2, "<f2", Dtype::F16, Order::C, &[2], (1, 0)),
    ];
    for (bytes, descr, dtype, order, shape, version) in expected {
        let header = Header::parse(bytes, descr).expect(descr);
        assert_eq!(&header.descr, descr);
        assert_eq!(header.dtype, *dtype, "{descr}");
        assert_eq!(header.order, *order, "{descr}");
        assert_eq!(header.shape, shape.to_vec(), "{descr}");
        assert_eq!(header.version, *version, "{descr}");
        assert_eq!(
            header.file_bytes(descr).expect("a size"),
            bytes.len(),
            "{descr}: the declared shape does not account for the file"
        );
        // numpy's own promise, on files numpy wrote.
        assert_eq!(header.data_offset % 64, 0, "{descr}");
    }
    // `float16` carries a byte order even though this crate cannot hold it,
    // which is why the tag and the refusal are separate questions.
    assert_eq!(
        Header::parse(FLOAT16_2, "f2").unwrap().endian,
        Endian::Little
    );
    assert_eq!(
        Header::parse(BIG_ENDIAN_I32_5, "i4").unwrap().endian,
        Endian::Big
    );
    assert_eq!(
        Header::parse(FORTRAN_ORDER_BOOL_2X3X4, "b1")
            .unwrap()
            .endian,
        Endian::NotApplicable
    );
}

/// The Fortran-ordered table is fifteen numbers in an arrangement, and reading
/// it as C order is the same fifteen numbers in a different one.
///
/// **This is the failure the module exists to prevent**, written down as an
/// assertion rather than as a warning: the count matches, the shape matches, the
/// dtype matches, the sum matches, and the array is wrong. The recorded
/// coordinate tables this crate's consumers read are `(378, 3)` of exactly this
/// kind.
#[test]
fn a_fortran_table_read_as_c_order_is_a_permutation_that_passes_every_count() {
    let table: ArrayD<f64> =
        read_array(FORTRAN_ORDER_F64_5X3, "table", OrderPolicy::Either).expect("read");
    assert_eq!(table.shape(), &[5, 3]);
    for row in 0..5 {
        for column in 0..3 {
            assert_eq!(
                table[[row, column]],
                (row * 3 + column) as f64,
                "row {row} column {column}"
            );
        }
    }

    // And the array that came out of a Fortran file writes back as a *C* file
    // correctly — the strides are the file's, the iteration order is logical,
    // and the two are not the same thing.
    let as_c_order = write_array(&table, Order::C, "<memory>").expect("written");
    let plain = Array2::from_shape_fn((5, 3), |(row, column)| (row * 3 + column) as f64);
    assert_eq!(
        as_c_order,
        write_array(&plain, Order::C, "<memory>").expect("written")
    );

    // The same bytes, decoded as if the header had said C order. Same fifteen
    // numbers; same shape; same total; different array.
    let mut lied = FORTRAN_ORDER_F64_5X3.to_vec();
    let header = String::from_utf8_lossy(&lied[10..128]).to_string();
    // Six characters for six, so the header keeps its declared length.
    let swapped = header.replace("'fortran_order': True, ", "'fortran_order': False,");
    assert_eq!(swapped.len(), header.len());
    lied.splice(10..128, swapped.bytes());
    let wrong: ArrayD<f64> = read_array(&lied, "table", OrderPolicy::Either).expect("read");

    assert_eq!(wrong.shape(), table.shape(), "the shape is no witness");
    assert_eq!(wrong.len(), table.len(), "the count is no witness");
    assert_eq!(
        wrong.sum(),
        table.sum(),
        "the total is no witness either — it is a permutation"
    );
    let mut mine: Vec<u64> = table.iter().map(|value| value.to_bits()).collect();
    let mut theirs: Vec<u64> = wrong.iter().map(|value| value.to_bits()).collect();
    mine.sort_unstable();
    theirs.sort_unstable();
    assert_eq!(mine, theirs, "the multiset is no witness either");
    assert_ne!(wrong, table, "and yet the arrays differ");
}

// ------------------------------------------- the same files, dtype-erased --
//
// `Elements` is the reader for a caller that does not know a file's element type
// in advance and cannot use `read_voxels` because the file is not rank 3. Every
// test below runs it against the same bytes numpy wrote, for the same reason the
// rest of this file does: a round trip through this crate's own writer would
// only prove the two halves agree.
//
// The two capabilities it adds — reading below rank 3, and the exact integer
// widening — are both paths **no recorded file in any consumer's fixture set
// exercises**, because every such file is little-endian and most are volumes.
// So each is tested here with a negative control beside it: a value that comes
// out *differently* if the path is wrong, rather than an assertion that it comes
// out at all. Untested acceptance would be worse than refusal.

/// Every embedded file reads as an `Elements` of the type and shape numpy gave
/// it, at every rank including the two `Voxels` cannot hold.
#[test]
fn an_elements_reads_each_recorded_file_at_its_own_rank_and_type() {
    let expected: &[(&[u8], Dtype, &[usize], usize)] = &[
        (C_ORDER_U16_2X3X4, Dtype::U16, &[2, 3, 4], 24),
        (FORTRAN_ORDER_F64_5X3, Dtype::F64, &[5, 3], 15),
        (FORTRAN_ORDER_BOOL_2X3X4, Dtype::Bool, &[2, 3, 4], 24),
        (BIG_ENDIAN_I32_5, Dtype::I32, &[5], 5),
        (VERSION_2_0_I64_2X3, Dtype::I64, &[2, 3], 6),
        // Rank 0 and a shape with a zero in it: an `Elements` holds both, and
        // `read_voxels` holds neither.
        (ZERO_RANK_F64, Dtype::F64, &[], 1),
        (EMPTY_F32_0X3, Dtype::F32, &[0, 3], 0),
        (EMPTY_I16_2X0X4, Dtype::I16, &[2, 0, 4], 0),
        (F64_EXTREMES_7, Dtype::F64, &[7], 7),
        (U64_EXTREMES_2, Dtype::U64, &[2], 2),
        (I64_EXTREMES_2, Dtype::I64, &[2], 2),
    ];
    for (bytes, dtype, shape, count) in expected {
        let held = read_elements(bytes, "recorded", OrderPolicy::Either)
            .unwrap_or_else(|error| panic!("{:?}: {error}", dtype));
        assert_eq!(held.dtype(), *dtype);
        assert_eq!(held.shape(), *shape, "{dtype:?}");
        assert_eq!(held.ndim(), shape.len(), "{dtype:?}");
        assert_eq!(held.len(), *count, "{dtype:?}");
        assert_eq!(held.is_empty(), *count == 0, "{dtype:?}");
    }

    // The values, not only the tags — a reader that returned an empty array of
    // the right shape would pass everything above.
    let scalar = read_elements(ZERO_RANK_F64, "()", OrderPolicy::Either).expect("read");
    assert_eq!(scalar.get::<f64>("()").expect("f64")[IxDyn(&[])], 3.5);
    let volume = read_elements(C_ORDER_U16_2X3X4, "vol", OrderPolicy::Either).expect("read");
    assert_eq!(
        volume
            .get::<u16>("vol")
            .expect("u16")
            .iter()
            .copied()
            .collect::<Vec<u16>>(),
        (0..24u16).collect::<Vec<u16>>()
    );
}

/// The Fortran table read through `Elements` indexes the writer's way, and the
/// same bytes under `Only(Order::C)` are refused rather than permuted.
///
/// The dtype-erasing reader has to make the same order decision the typed one
/// does; a reader that erased the type and quietly fixed the order would put
/// each row's three components on three different rows.
#[test]
fn an_elements_of_a_fortran_table_indexes_the_way_numpy_wrote_it() {
    let held = read_elements(FORTRAN_ORDER_F64_5X3, "table", OrderPolicy::Either).expect("read");
    let table = held.get::<f64>("table").expect("f64");
    for row in 0..5 {
        for column in 0..3 {
            assert_eq!(
                table[[row, column]],
                (row * 3 + column) as f64,
                "{row},{column}"
            );
        }
    }

    let text = read_elements(
        FORTRAN_ORDER_F64_5X3,
        "table.npy",
        OrderPolicy::Only(Order::C),
    )
    .expect_err("refused")
    .to_string();
    assert!(text.contains("table.npy"), "{text}");
    assert!(
        text.contains("Fortran order") && text.contains("C order"),
        "{text}"
    );
}

/// `Elements` swaps a big-endian file's bytes, and the negative control is that
/// ignoring the `>` would give numbers rather than an error.
///
/// The same claim `a_big_endian_file_is_swapped_rather_than_read_as_this_machine_writes`
/// makes for the typed reader, made again for the erasing one because it is a
/// different code path to the same `read_element` and nothing but a test says
/// they agree. **No fixture set exercises this**: every recorded file any
/// consumer reads is little-endian, so this file's bytes are the only witness.
#[test]
fn an_elements_swaps_a_big_endian_file_and_a_reader_that_did_not_would_not_error() {
    let held = read_elements(BIG_ENDIAN_I32_5, "big-endian", OrderPolicy::Either).expect("read");
    assert_eq!(held.dtype(), Dtype::I32);
    let values: Vec<i32> = held
        .get::<i32>("big-endian")
        .expect("i32")
        .iter()
        .copied()
        .collect();
    assert_eq!(values, vec![i32::MIN, -1, 0, 1, i32::MAX]);
    // The negative control: what the same bytes are if the `>` is ignored. Five
    // plausible integers, no error, and none of them equal to the right answer
    // except the two palindromes.
    let unswapped: Vec<i32> = values.iter().map(|value| value.swap_bytes()).collect();
    assert_eq!(unswapped, vec![128, -1, 0, 16_777_216, -129]);
    assert_ne!(values[0], unswapped[0]);
    assert_ne!(values[4], unswapped[4]);

    // And the swap reaches the exact widening, which is a third path through
    // `read_element`.
    assert_eq!(
        held.widened_i64("big-endian")
            .expect("exact")
            .iter()
            .copied()
            .collect::<Vec<i64>>(),
        vec![i64::from(i32::MIN), -1, 0, 1, i64::from(i32::MAX)]
    );

    // A big-endian file does not write back byte-identically, because this
    // crate writes little-endian — it writes the *same values* under `<i4`.
    let written = write_elements(&held, Order::C, "<memory>").expect("written");
    assert_ne!(written, BIG_ENDIAN_I32_5.to_vec());
    assert_eq!(
        read_elements(&written, "round trip", OrderPolicy::Either).expect("read"),
        held
    );
}

/// The exact integer widening, on the integer extremes numpy wrote.
///
/// `[0, u64::MAX]` and `[i64::MIN, i64::MAX]` are the two files that decide
/// this, and neither is reachable through `f64`: `u64::MAX` has no `i64` at all
/// and `i64::MAX` has no exact `f64`.
#[test]
fn the_exact_widening_is_exact_on_the_recorded_extremes_and_refuses_the_rest() {
    // `int64` at both ends, exactly — and demonstrably not through `f64`.
    let held = read_elements(I64_EXTREMES_2, "i64", OrderPolicy::Either).expect("read");
    let exact: Vec<i64> = held
        .widened_i64("i64")
        .expect("exact")
        .iter()
        .copied()
        .collect();
    assert_eq!(exact, vec![i64::MIN, i64::MAX]);
    // And the `f64` path loses it, which is why the two widenings both exist.
    // The loss does not show as a wrong number after a round trip — the `as`
    // cast back saturates and hides it — it shows as two *different*
    // identifiers becoming one: `i64::MAX` and its neighbour have the same
    // `f64`, so a comparison through `widened` cannot tell them apart.
    let through_f64 = held.widened();
    assert_eq!(through_f64[[1]], (i64::MAX - 1) as f64);
    assert_ne!(exact[1], i64::MAX - 1);

    // `uint64` past `i64::MAX` is refused, with the value in the message.
    let held = read_elements(U64_EXTREMES_2, "u64", OrderPolicy::Either).expect("read");
    assert_eq!(
        held.get::<u64>("u64")
            .expect("u64")
            .iter()
            .copied()
            .collect::<Vec<u64>>(),
        vec![0, u64::MAX]
    );
    let text = held
        .widened_i64("labels.npy")
        .expect_err("refused")
        .to_string();
    assert!(text.contains("labels.npy"), "{text}");
    assert!(text.contains("18446744073709551615"), "{text}");
    // Wrapping would have given `-1`, which is a label a caller would believe.
    assert_eq!(u64::MAX as i64, -1);

    // The float extremes are refused by name rather than truncated. `NaN` and
    // the infinities in this file are exactly the values a truncating reader
    // turns into an arbitrary integer.
    let held = read_elements(F64_EXTREMES_7, "f64", OrderPolicy::Either).expect("read");
    let text = held
        .widened_i64("field.npy")
        .expect_err("refused")
        .to_string();
    assert!(
        text.contains("field.npy") && text.contains("float64"),
        "{text}"
    );
    let held = read_elements(F32_EXTREMES_7, "f32", OrderPolicy::Either).expect("read");
    let text = held
        .widened_i64("field.npy")
        .expect_err("refused")
        .to_string();
    assert!(text.contains("float32"), "{text}");
}

/// A table is not a volume, whichever reader asks.
///
/// The negative control for the rank-free reader: erasing the element type must
/// not erase the rank check as well, because the mistake that check exists to
/// stop — an `(n, 3)` table read as a volume — is exactly the one an erasing
/// reader makes easier to reach.
#[test]
fn an_elements_below_rank_three_still_refuses_to_be_a_volume() {
    for (bytes, rank) in [
        (FORTRAN_ORDER_F64_5X3, "2-dimensional"),
        (BIG_ENDIAN_I32_5, "1-dimensional"),
        (ZERO_RANK_F64, "0-dimensional"),
    ] {
        let held = read_elements(bytes, "t", OrderPolicy::Either).expect("read");
        let text = held.into_voxels("t.npy").expect_err("refused").to_string();
        assert!(text.contains("t.npy"), "{text}");
        assert!(text.contains(rank) && text.contains("rank 3"), "{text}");
    }

    // And a rank-3 file becomes one, so the check is a check and not a wall.
    let held = read_elements(C_ORDER_U16_2X3X4, "vol", OrderPolicy::Either).expect("read");
    let voxels = held.into_voxels("vol.npy").expect("a volume");
    assert_eq!(voxels.shape(), [2, 3, 4]);
    assert_eq!(
        voxels,
        read_voxels(C_ORDER_U16_2X3X4, "vol", OrderPolicy::Either).expect("read")
    );
}

/// The mapping reader converts inside the decode, byte swap included.
///
/// `read_array_mapped` is a fourth path to `read_element`, and the big-endian
/// file is the only witness that it swaps: a mapped read that widened before
/// swapping would give `[128, -1, 0, 16777216, -129]` widened, which is five
/// plausible `i64`s.
#[test]
fn a_mapped_read_swaps_before_it_converts() {
    let wide: ArrayD<i64> = read_array_mapped::<i32, i64>(
        BIG_ENDIAN_I32_5,
        "big-endian",
        OrderPolicy::Either,
        i64::from,
    )
    .expect("read");
    assert_eq!(
        wide.iter().copied().collect::<Vec<i64>>(),
        vec![i64::from(i32::MIN), -1, 0, 1, i64::from(i32::MAX)]
    );

    // The `From`-only convenience form, on a file with no byte order to get
    // wrong, and the widening it gives is the one `From` licenses.
    let temporary = std::env::temp_dir().join("blockflow_npy_mapped_read.npy");
    std::fs::write(&temporary, C_ORDER_U16_2X3X4).expect("a temporary file");
    let wide: ArrayD<f64> =
        read_array_file_as::<u16, f64>(&temporary, OrderPolicy::Either).expect("read");
    assert_eq!(wide.shape(), &[2, 3, 4]);
    assert_eq!(
        wide.iter().copied().collect::<Vec<f64>>(),
        (0..24).map(f64::from).collect::<Vec<f64>>()
    );
    let _ = std::fs::remove_file(&temporary);

    // The negative control: `Src` is checked against the header, so a mapped
    // read is not a way to read one type as another.
    let text =
        read_array_mapped::<i16, i64>(C_ORDER_U16_2X3X4, "vol.npy", OrderPolicy::Either, i64::from)
            .expect_err("refused")
            .to_string();
    assert!(text.contains("uint16") && text.contains("int16"), "{text}");
}

/// The `Elements` writer reproduces numpy's bytes, for the files numpy wrote in
/// this machine's byte order.
///
/// The same claim the typed writer makes, extended to the erasing one: a writer
/// that took its header from the enum rather than from the element type would
/// pass a round trip and fail this.
#[test]
fn writing_an_elements_reproduces_the_bytes_numpy_wrote() {
    for (bytes, order) in [
        (C_ORDER_U16_2X3X4, Order::C),
        (FORTRAN_ORDER_F64_5X3, Order::Fortran),
        (FORTRAN_ORDER_BOOL_2X3X4, Order::Fortran),
        (ZERO_RANK_F64, Order::C),
        (EMPTY_F32_0X3, Order::C),
        (EMPTY_I16_2X0X4, Order::C),
        (F64_EXTREMES_7, Order::C),
        (U64_EXTREMES_2, Order::C),
        (I64_EXTREMES_2, Order::C),
    ] {
        let held = read_elements(bytes, "recorded", OrderPolicy::Either).expect("read");
        assert_eq!(
            write_elements(&held, order, "<memory>").expect("written"),
            bytes.to_vec(),
            "{:?} {order}",
            held.dtype()
        );
    }
}

/// The big-endian file is byte-swapped rather than refused, and the values it
/// carries are the ones that would come out as *numbers* if the `>` were
/// ignored.
#[test]
fn a_big_endian_file_is_swapped_rather_than_read_as_this_machine_writes() {
    let values: ArrayD<i32> =
        read_array(BIG_ENDIAN_I32_5, "big-endian", OrderPolicy::Either).expect("read");
    assert_eq!(
        values.iter().copied().collect::<Vec<i32>>(),
        vec![i32::MIN, -1, 0, 1, i32::MAX]
    );
    // Ignoring the `>` gives `[128, -1, 0, 16777216, -129]` — five plausible
    // integers, no error. `-1` and `0` are palindromes and prove nothing, which
    // is why they are not the whole of the fixture.
    assert_ne!(values[[0]], 128);
    assert_ne!(values[[4]], -129);
}

/// A 2.0 header declares its length in four bytes, and is read.
#[test]
fn a_version_two_header_is_read_and_a_version_three_one_is_refused_by_name() {
    let values: ArrayD<i64> =
        read_array(VERSION_2_0_I64_2X3, "two-oh", OrderPolicy::Either).expect("read");
    assert_eq!(values.shape(), &[2, 3]);
    assert_eq!(
        values.iter().copied().collect::<Vec<i64>>(),
        (0..6).collect::<Vec<i64>>()
    );

    let error = read_array::<i64>(VERSION_3_0_I64_2X3, "three-oh", OrderPolicy::Either)
        .expect_err("refused");
    let text = error.to_string();
    assert!(text.contains("three-oh"), "{text}");
    assert!(text.contains("version 3.0"), "{text}");
    assert!(text.contains("UTF-8"), "{text}");
}

/// The three shapes with no elements or no axes, as numpy spells them.
#[test]
fn a_shape_with_no_axes_or_a_zero_in_it_is_read_as_the_array_it_is() {
    let scalar: ArrayD<f64> = read_array(ZERO_RANK_F64, "()", OrderPolicy::Either).expect("read");
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.len(), 1, "an empty product is one element, not none");
    assert_eq!(scalar[IxDyn(&[])], 3.5);

    let none: ArrayD<f32> = read_array(EMPTY_F32_0X3, "(0,3)", OrderPolicy::Either).expect("read");
    assert_eq!(none.shape(), &[0, 3]);
    assert_eq!(none.len(), 0);

    let none: ArrayD<i16> =
        read_array(EMPTY_I16_2X0X4, "(2,0,4)", OrderPolicy::Either).expect("read");
    assert_eq!(none.shape(), &[2, 0, 4]);
    assert_eq!(none.len(), 0);
    // A rank-3 shape with no elements is still a volume.
    let volume = read_voxels(EMPTY_I16_2X0X4, "(2,0,4)", OrderPolicy::Either).expect("read");
    assert_eq!(volume.shape(), [2, 0, 4]);
    assert!(volume.is_empty());
}

/// The three dtypes with no variant here are each refused by their own name.
#[test]
fn a_dtype_with_no_variant_here_is_refused_by_the_name_the_file_gives_it() {
    let text = read_array::<f32>(FLOAT16_2, "half.npy", OrderPolicy::Either)
        .expect_err("refused")
        .to_string();
    assert!(text.contains("half.npy"), "{text}");
    assert!(text.contains("float16"), "{text}");

    let text = read_voxels(FLOAT16_2, "half.npy", OrderPolicy::Either)
        .expect_err("refused")
        .to_string();
    assert!(
        text.contains("float16") && text.contains("no variant"),
        "{text}"
    );

    let text = read_array::<i32>(STRUCTURED_2, "rows.npy", OrderPolicy::Either)
        .expect_err("refused")
        .to_string();
    assert!(text.contains("rows.npy"), "{text}");
    assert!(text.contains("structured dtype"), "{text}");

    let text = read_array::<f64>(COMPLEX128_1, "wave.npy", OrderPolicy::Either)
        .expect_err("refused")
        .to_string();
    assert!(text.contains("wave.npy"), "{text}");
    assert!(text.contains("<c16"), "{text}");
    assert!(text.contains("no variant here"), "{text}");
}

/// The extremes numpy wrote come back with the bits numpy wrote them with.
#[test]
fn the_extremes_numpy_wrote_come_back_bit_for_bit() {
    let wide: ArrayD<f64> = read_array(F64_EXTREMES_7, "f8", OrderPolicy::Either).expect("read");
    let expected = [
        0.0f64,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        f64::EPSILON,
    ];
    for (index, want) in expected.iter().enumerate() {
        assert_eq!(
            wide[[index]].to_bits(),
            want.to_bits(),
            "float64 element {index}"
        );
    }
    // `-0.0 == 0.0` compares equal, so only the bits catch a writer that lost
    // the sign.
    assert_eq!(wide[[1]], wide[[0]]);
    assert_ne!(wide[[1]].to_bits(), wide[[0]].to_bits());

    let narrow: ArrayD<f32> = read_array(F32_EXTREMES_7, "f4", OrderPolicy::Either).expect("read");
    let expected = [
        0.0f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        f32::EPSILON,
    ];
    for (index, want) in expected.iter().enumerate() {
        assert_eq!(
            narrow[[index]].to_bits(),
            want.to_bits(),
            "float32 element {index}"
        );
    }

    let unsigned: ArrayD<u64> =
        read_array(U64_EXTREMES_2, "u8", OrderPolicy::Either).expect("read");
    assert_eq!(unsigned[[1]], u64::MAX);
    let signed: ArrayD<i64> = read_array(I64_EXTREMES_2, "i8", OrderPolicy::Either).expect("read");
    assert_eq!(signed[[0]], i64::MIN);
    assert_eq!(signed[[1]], i64::MAX);
}

// ---------------------------------------------------- writing numpy's bytes --

/// What is written here is what numpy writes, byte for byte.
///
/// The stronger half of "read what numpy wrote": a reader that agrees with its
/// own writer proves nothing, and a writer whose output is byte-identical to
/// numpy's over rank 0, 1, 2 and 3, both memory orders and four element kinds
/// is not agreeing with itself.
#[test]
fn the_bytes_written_here_are_the_bytes_numpy_writes() {
    let ramp = Array3::from_shape_fn((2, 3, 4), |(z, y, x)| (z * 12 + y * 4 + x) as u16);
    assert_eq!(
        write_array(&ramp, Order::C, "<memory>").expect("written"),
        C_ORDER_U16_2X3X4
    );

    let table = Array2::from_shape_fn((5, 3), |(row, column)| (row * 3 + column) as f64);
    assert_eq!(
        write_array(&table, Order::Fortran, "<memory>").expect("written"),
        FORTRAN_ORDER_F64_5X3,
        "a Fortran-ordered file is written in the file's order, not the array's"
    );

    let flags = Array3::from_shape_fn((2, 3, 4), |(z, y, x)| (z * 12 + y * 4 + x) % 3 == 0);
    assert_eq!(
        write_array(&flags, Order::Fortran, "<memory>").expect("written"),
        FORTRAN_ORDER_BOOL_2X3X4
    );
    // And through the `Voxels` seam, which is the same writer.
    let volume: Voxels = flags.into();
    assert_eq!(
        write_voxels(&volume, Order::Fortran, "<memory>").expect("written"),
        FORTRAN_ORDER_BOOL_2X3X4
    );

    let scalar = ArrayD::<f64>::from_elem(IxDyn(&[]), 3.5);
    assert_eq!(
        write_array(&scalar, Order::C, "<memory>").expect("written"),
        ZERO_RANK_F64,
        "numpy spells a zero-rank shape `()`"
    );

    let none = ArrayD::<f32>::zeros(IxDyn(&[0, 3]));
    assert_eq!(
        write_array(&none, Order::C, "<memory>").expect("written"),
        EMPTY_F32_0X3
    );
    let none = ArrayD::<i16>::zeros(IxDyn(&[2, 0, 4]));
    assert_eq!(
        write_array(&none, Order::C, "<memory>").expect("written"),
        EMPTY_I16_2X0X4
    );

    let extremes = Array1::from_vec(vec![
        0.0f64,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        f64::EPSILON,
    ]);
    assert_eq!(
        write_array(&extremes, Order::C, "<memory>").expect("written"),
        F64_EXTREMES_7,
        "numpy's NaN is the quiet one with no payload, and so is Rust's `f64::NAN`"
    );

    let extremes = Array1::from_vec(vec![0u64, u64::MAX]);
    assert_eq!(
        write_array(&extremes, Order::C, "<memory>").expect("written"),
        U64_EXTREMES_2,
        "numpy spells a one-axis shape `(2,)`"
    );
    let extremes = Array1::from_vec(vec![i64::MIN, i64::MAX]);
    assert_eq!(
        write_array(&extremes, Order::C, "<memory>").expect("written"),
        I64_EXTREMES_2
    );
}

/// A header padded to sixteen bytes rather than sixty-four still reads.
///
/// numpy has aligned the data to 64 since 1.9 and to 16 before that, and a
/// recording set outlives a numpy release. The reader takes the *declared*
/// header length and never computes one, which is what makes both work; this
/// pins that rather than leaving it to be discovered.
#[test]
fn a_file_from_an_older_writer_with_a_sixteen_byte_alignment_still_reads() {
    let text = b"{'descr': '<u2', 'fortran_order': False, 'shape': (2, 3, 4), }";
    let declared = 16 - ((10 + text.len() + 1) % 16) + text.len() + 1;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(declared as u16).to_le_bytes());
    bytes.extend_from_slice(text);
    bytes.resize(10 + declared - 1, b' ');
    bytes.push(b'\n');
    assert_eq!(bytes.len() % 16, 0);
    assert_ne!(bytes.len() % 64, 0, "this is the case 64 would miss");
    bytes.extend_from_slice(&C_ORDER_U16_2X3X4[128..]);

    let values: ArrayD<u16> = read_array(&bytes, "old.npy", OrderPolicy::Either).expect("read");
    let expected: ArrayD<u16> =
        read_array(C_ORDER_U16_2X3X4, "new.npy", OrderPolicy::Either).expect("read");
    assert_eq!(values, expected);
}

// -------------------------------------------------------------- round trips --

/// Every element type, both memory orders, bit-compared, at its extremes.
///
/// The values are chosen so that no single one of them is a witness on its own.
/// `u64::MAX` in particular is *not* evidence by itself — a saturating
/// `f64`-to-`u64` cast returns it unchanged, so a path that widened everything
/// through `f64` would pass on it for the wrong reason. `u64::MAX - 1` and
/// `2^63 + 1` are the witnesses: neither is representable in `f64`, so anything
/// that touched a float loses them. The floats carry `-0.0`, which compares
/// equal to `0.0`, and two different NaNs, which compare equal to nothing at
/// all — both are invisible to `==` and visible to `to_bits`.
#[test]
fn every_element_type_round_trips_its_extremes_through_a_file() {
    fn check<T: NpyElement + PartialEq + std::fmt::Debug>(values: Vec<T>, bits: fn(&T) -> u128) {
        let array = Array1::from_vec(values);
        for order in [Order::C, Order::Fortran] {
            let bytes = write_array(&array, order, "<memory>").expect("written");
            let header = Header::parse(&bytes, "<memory>").expect("a header");
            assert_eq!(header.dtype, T::DTYPE);
            assert_eq!(header.order, order);
            assert_eq!(header.data_offset % 64, 0, "{:?}", T::DTYPE);
            let back: ArrayD<T> =
                read_array(&bytes, "<memory>", OrderPolicy::Either).expect("read");
            assert_eq!(back.shape(), array.shape(), "{:?} {order}", T::DTYPE);
            for (index, (mine, theirs)) in array.iter().zip(back.iter()).enumerate() {
                assert_eq!(
                    bits(mine),
                    bits(theirs),
                    "{:?} {order}: element {index} did not survive",
                    T::DTYPE
                );
            }
        }
    }

    check(vec![false, true, false, true], |v| u128::from(*v));
    check(vec![0u8, 1, u8::MAX, u8::MAX - 1, 128], |v| u128::from(*v));
    check(vec![0i8, -1, i8::MIN, i8::MAX, i8::MIN + 1], |v| {
        u128::from(v.to_le_bytes()[0])
    });
    check(vec![0u16, 1, u16::MAX, u16::MAX - 1], |v| u128::from(*v));
    check(vec![0i16, -1, i16::MIN, i16::MAX, i16::MIN + 1], |v| {
        u128::from(u16::from_le_bytes(v.to_le_bytes()))
    });
    check(vec![0u32, 1, u32::MAX, u32::MAX - 1], |v| u128::from(*v));
    check(vec![0i32, -1, i32::MIN, i32::MAX, i32::MIN + 1], |v| {
        u128::from(u32::from_le_bytes(v.to_le_bytes()))
    });
    check(
        // `u64::MAX` passes a saturating cast unchanged, so it is joined by two
        // values no `f64` can hold.
        vec![0u64, 1, u64::MAX, u64::MAX - 1, (1u64 << 63) + 1],
        |v| u128::from(*v),
    );
    check(
        vec![0i64, -1, i64::MIN, i64::MAX, i64::MIN + 1, i64::MAX - 1],
        |v| u128::from(u64::from_le_bytes(v.to_le_bytes())),
    );
    check(
        vec![
            0.0f32,
            -0.0,
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            f32::MIN,
            f32::MAX,
            f32::from_bits(1),
        ],
        |v| u128::from(v.to_bits()),
    );
    check(
        vec![
            0.0f64,
            -0.0,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_dead_beef),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::MIN,
            f64::MAX,
            f64::from_bits(1),
        ],
        |v| u128::from(v.to_bits()),
    );
}

/// The same, as volumes, through the `Voxels` seam that carries the tag at run
/// time rather than in the type.
#[test]
fn every_voxels_variant_round_trips_through_a_file_in_either_order() {
    let shape = [2usize, 3, 4];
    let count = 24usize;
    let volumes = [
        Voxels::Bool(Array3::from_shape_fn(shape, |(z, y, x)| {
            (z + y + x) % 2 == 0
        })),
        Voxels::U8(Array3::from_shape_fn(shape, |(z, y, x)| {
            [0u8, u8::MAX, 1, u8::MAX - 1][(z + y + x) % 4]
        })),
        Voxels::I8(Array3::from_shape_fn(shape, |(z, y, x)| {
            [0i8, i8::MIN, i8::MAX, i8::MIN + 1][(z + y + x) % 4]
        })),
        Voxels::U16(Array3::from_shape_fn(shape, |(z, y, x)| {
            [0u16, u16::MAX, 1, u16::MAX - 1][(z + y + x) % 4]
        })),
        Voxels::I16(Array3::from_shape_fn(shape, |(z, y, x)| {
            [0i16, i16::MIN, i16::MAX, i16::MIN + 1][(z + y + x) % 4]
        })),
        Voxels::U32(Array3::from_shape_fn(shape, |(z, y, x)| {
            [0u32, u32::MAX, 1, u32::MAX - 1][(z + y + x) % 4]
        })),
        Voxels::I32(Array3::from_shape_fn(shape, |(z, y, x)| {
            [0i32, i32::MIN, i32::MAX, i32::MIN + 1][(z + y + x) % 4]
        })),
        Voxels::U64(Array3::from_shape_fn(shape, |(z, y, x)| {
            [0u64, u64::MAX, u64::MAX - 1, (1u64 << 63) + 1][(z + y + x) % 4]
        })),
        Voxels::I64(Array3::from_shape_fn(shape, |(z, y, x)| {
            [i64::MIN, i64::MAX, i64::MIN + 1, i64::MAX - 1][(z + y + x) % 4]
        })),
        Voxels::F32(Array3::from_shape_fn(shape, |(z, y, x)| {
            [
                -0.0f32,
                f32::NAN,
                f32::NEG_INFINITY,
                f32::from_bits(0x7fc0_1234),
                f32::MIN_POSITIVE,
                f32::EPSILON,
            ][(z + y + x) % 6]
        })),
        Voxels::F64(Array3::from_shape_fn(shape, |(z, y, x)| {
            [
                -0.0f64,
                f64::NAN,
                f64::NEG_INFINITY,
                f64::from_bits(0x7ff8_0000_dead_beef),
                f64::MIN_POSITIVE,
                f64::EPSILON,
            ][(z + y + x) % 6]
        })),
    ];
    assert_eq!(volumes.len(), 11, "one per `Voxels` variant");
    for volume in &volumes {
        assert_eq!(volume.len(), count);
        for order in [Order::C, Order::Fortran] {
            let bytes = write_voxels(volume, order, "<memory>").expect("written");
            let back = read_voxels(&bytes, "<memory>", OrderPolicy::Either).expect("read");
            assert_eq!(back.dtype(), volume.dtype());
            assert_eq!(back.shape(), volume.shape());
            // Bits, not values: NaN is equal to nothing and `-0.0` is equal to
            // `0.0`, so `==` on the volumes would pass either way.
            assert_eq!(
                bits_of(&back),
                bits_of(volume),
                "{:?} in {order}",
                volume.dtype()
            );
        }
    }
}

/// A volume's elements as raw bits, in logical order. The only comparison that
/// is neither blind to `-0.0` nor defeated by NaN.
fn bits_of(volume: &Voxels) -> Vec<u64> {
    match volume {
        Voxels::Bool(a) => a.iter().map(|v| u64::from(*v)).collect(),
        Voxels::U8(a) => a.iter().map(|v| u64::from(*v)).collect(),
        Voxels::I8(a) => a.iter().map(|v| u64::from(v.to_le_bytes()[0])).collect(),
        Voxels::U16(a) => a.iter().map(|v| u64::from(*v)).collect(),
        Voxels::I16(a) => a
            .iter()
            .map(|v| u64::from(u16::from_le_bytes(v.to_le_bytes())))
            .collect(),
        Voxels::U32(a) => a.iter().map(|v| u64::from(*v)).collect(),
        Voxels::I32(a) => a
            .iter()
            .map(|v| u64::from(u32::from_le_bytes(v.to_le_bytes())))
            .collect(),
        Voxels::U64(a) => a.iter().copied().collect(),
        Voxels::I64(a) => a
            .iter()
            .map(|v| u64::from_le_bytes(v.to_le_bytes()))
            .collect(),
        Voxels::F32(a) => a.iter().map(|v| u64::from(v.to_bits())).collect(),
        Voxels::F64(a) => a.iter().map(|v| v.to_bits()).collect(),
    }
}

// ------------------------------------------------------------ by the region --

fn scratch(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "blockflow-npy-{}-{}-{name}.npy",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or_default()
    ));
    path
}

/// A file read by region agrees, box for box, with the whole file read at once.
///
/// This is the npy analogue of `ArrayRegionSource`'s byte-identity property, and
/// it is asserted in both memory orders because the run geometry is different in
/// each: the contiguous stretch is the last axis in one and the first in the
/// other.
#[test]
fn a_region_of_a_file_is_the_same_box_the_whole_file_holds() {
    let whole = Array3::from_shape_fn((5, 6, 7), |(z, y, x)| (z * 42 + y * 7 + x) as i32);
    for order in [Order::C, Order::Fortran] {
        let path = scratch(&format!("regions-{order:?}"));
        blockflow::npy::write_array_file(&path, &whole, order).expect("written");

        let source: NpySource<i32> = NpySource::open(&path, OrderPolicy::Either).expect("opened");
        assert_eq!(source.shape(), &[5, 6, 7]);
        assert_eq!(source.header().order, order);
        assert!(source.describe().contains("int32"), "{}", source.describe());
        assert_eq!(
            source.is_known_empty(&Region::whole(&[5, 6, 7])),
            None,
            "a dense file cannot tell, and says so"
        );

        for start in [[0usize, 0, 0], [1, 2, 3], [3, 4, 5], [4, 5, 6]] {
            for shape in [[1usize, 1, 1], [2, 2, 2], [1, 2, 1]] {
                if (0..3).any(|axis| start[axis] + shape[axis] > [5, 6, 7][axis]) {
                    continue;
                }
                let region = Region::new(&start, &shape);
                let got = source.read_region(&region).expect("a region");
                assert_eq!(got.shape(), &shape[..]);
                for z in 0..shape[0] {
                    for y in 0..shape[1] {
                        for x in 0..shape[2] {
                            assert_eq!(
                                got[[z, y, x]],
                                whole[[start[0] + z, start[1] + y, start[2] + x]],
                                "{order} {start:?} {shape:?} at {z},{y},{x}"
                            );
                        }
                    }
                }
            }
        }
        // The whole volume, as one region.
        let all = source.read_region(&Region::whole(&[5, 6, 7])).expect("all");
        assert_eq!(all, whole.clone().into_dyn());
        // A region outside is refused rather than clamped, as everywhere else.
        assert!(source
            .read_region(&Region::new(&[4, 0, 0], &[2, 6, 7]))
            .is_err());
        assert!(source.read_region(&Region::new(&[0, 0], &[5, 6])).is_err());
        let _ = std::fs::remove_file(&path);
    }
}

/// A volume written a block at a time is the volume, and it is the file numpy
/// would have written for it.
#[test]
fn a_volume_written_by_region_is_the_file_written_all_at_once() {
    let whole = Array3::from_shape_fn((4, 6, 5), |(z, y, x)| (z * 30 + y * 5 + x) as f64);
    for order in [Order::C, Order::Fortran] {
        let path = scratch(&format!("sink-{order:?}"));
        let sink: NpySink<f64> = NpySink::create(&path, &[4, 6, 5], order).expect("created");
        assert_eq!(sink.shape(), &[4, 6, 5]);
        // Four disjoint boxes that tile the volume, written out of order.
        let boxes = [
            ([0usize, 0, 0], [4usize, 2, 5]),
            ([0, 4, 0], [4, 2, 5]),
            ([0, 2, 0], [2, 2, 5]),
            ([2, 2, 0], [2, 2, 5]),
        ];
        for (start, shape) in boxes {
            let mut block = ArrayD::<f64>::zeros(IxDyn(&shape));
            for z in 0..shape[0] {
                for y in 0..shape[1] {
                    for x in 0..shape[2] {
                        block[[z, y, x]] = whole[[start[0] + z, start[1] + y, start[2] + x]];
                    }
                }
            }
            sink.write_region(&start, &block).expect("written");
        }
        sink.finish().expect("durable");
        drop(sink);

        let bytes = std::fs::read(&path).expect("readable");
        let expected = write_array(&whole, order, "<memory>").expect("written");
        assert_eq!(
            bytes, expected,
            "{order}: a volume written by region is not the file written at once"
        );
        let back: ArrayD<f64> = read_array_file(&path, OrderPolicy::Either).expect("read");
        assert_eq!(back, whole.clone().into_dyn());
        let _ = std::fs::remove_file(&path);
    }
}

/// A region never written holds the zero pattern; `create_filled` pays for a
/// loud one instead. Both are stated rather than assumed.
#[test]
fn an_unwritten_region_is_zero_unless_the_caller_paid_for_a_sentinel() {
    let path = scratch("sparse");
    let sink: NpySink<f64> = NpySink::create(&path, &[2, 2, 2], Order::C).expect("created");
    sink.write_region(&[0, 0, 0], &ArrayD::from_elem(IxDyn(&[1, 2, 2]), 7.0))
        .expect("written");
    sink.finish().expect("durable");
    drop(sink);
    let back: ArrayD<f64> = read_array_file(&path, OrderPolicy::Either).expect("read");
    assert_eq!(back[[0, 0, 0]], 7.0);
    assert_eq!(
        back[[1, 0, 0]],
        0.0,
        "an unwritten voxel is a convincing zero"
    );
    let _ = std::fs::remove_file(&path);

    let path = scratch("loud");
    let sink: NpySink<f64> =
        NpySink::create_filled(&path, &[2, 2, 2], Order::C, f64::NAN).expect("created");
    sink.write_region(&[0, 0, 0], &ArrayD::from_elem(IxDyn(&[1, 2, 2]), 7.0))
        .expect("written");
    sink.finish().expect("durable");
    drop(sink);
    let back: ArrayD<f64> = read_array_file(&path, OrderPolicy::Either).expect("read");
    assert_eq!(back[[0, 0, 0]], 7.0);
    assert!(back[[1, 0, 0]].is_nan(), "the caller paid for a loud hole");
    let _ = std::fs::remove_file(&path);
}

/// Opening a file as the wrong element type is refused before a byte is read.
#[test]
fn a_source_of_the_wrong_element_type_is_refused_at_open() {
    let path = scratch("typed");
    blockflow::npy::write_array_file(&path, &Array2::<u16>::zeros((3, 4)), Order::C)
        .expect("written");
    let error = NpySource::<f64>::open(&path, OrderPolicy::Either).expect_err("refused");
    let text = error.to_string();
    assert!(
        text.contains("uint16") && text.contains("float64"),
        "{text}"
    );
    assert!(NpySource::<u16>::open(&path, OrderPolicy::Either).is_ok());

    // And a truncated file is refused at open rather than at the first region.
    let mut bytes = std::fs::read(&path).expect("readable");
    bytes.truncate(bytes.len() - 2);
    std::fs::write(&path, &bytes).expect("written");
    let text = NpySource::<u16>::open(&path, OrderPolicy::Either)
        .expect_err("refused")
        .to_string();
    assert!(text.contains("bytes with its header"), "{text}");
    let _ = std::fs::remove_file(&path);
}

// ------------------------------------------------------------ recorded files --

/// Every file under `BLOCKFLOW_NPY_FIXTURES` reads, and writing it back in the
/// order it was stored in reproduces it byte for byte.
///
/// Skipped when the variable is unset, so the suite stays hermetic; the
/// directory is a parameter because a recording set belongs to whoever recorded
/// it. `BLOCKFLOW_NPY_MAX` caps the bytes a single file may cost, defaulting to
/// 64 MiB — the set this was run against is 232 GB and a test is not.
#[test]
fn a_directory_of_recorded_files_reads_and_re_serialises_exactly() {
    let Ok(root) = std::env::var("BLOCKFLOW_NPY_FIXTURES") else {
        return;
    };
    let cap: u64 = std::env::var("BLOCKFLOW_NPY_MAX")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(64 * 1024 * 1024);

    let mut files = Vec::new();
    collect(Path::new(&root), &mut files);
    assert!(
        !files.is_empty(),
        "{root}: no `.npy` files, so this is not looking where it thinks it is"
    );

    let mut read = 0usize;
    let mut identical = 0usize;
    let mut respelled = 0usize;
    let mut fortran = 0usize;
    let mut skipped = 0usize;
    let mut kinds = std::collections::BTreeSet::new();
    for path in &files {
        let header = read_header_file(path).unwrap_or_else(|error| panic!("{error}"));
        let size = std::fs::metadata(path).expect("a size").len();
        assert_eq!(
            header
                .file_bytes(&path.display().to_string())
                .expect("a size") as u64,
            size,
            "{}: the header does not account for the file",
            path.display()
        );
        kinds.insert((header.dtype, header.order, header.shape.len()));
        if header.order == Order::Fortran {
            fortran += 1;
        }
        if size > cap || header.dtype == Dtype::F16 {
            skipped += 1;
            continue;
        }
        let what = path.display().to_string();
        let bytes = std::fs::read(path).expect("readable");
        let written = reread(&bytes, &what);
        // The data is compared byte for byte; the header is compared field by
        // field. Not every file in a recording set was written by numpy — some
        // writers spell a shape `(64, 64, 64, )`, with a trailing comma numpy
        // does not use — and re-spelling a header is not a difference in the
        // array. Losing a single element byte would be.
        let again = Header::parse(&written, &what).expect("a header this crate rendered");
        assert_eq!(again.dtype, header.dtype, "{what}");
        assert_eq!(again.order, header.order, "{what}");
        assert_eq!(again.shape, header.shape, "{what}");
        assert!(
            written[again.data_offset..] == bytes[header.data_offset..],
            "{what}: re-serialising changed the data"
        );
        if written == bytes {
            identical += 1;
        } else {
            respelled += 1;
        }
        read += 1;
    }
    assert!(read > 0, "{root}: every file was skipped");
    assert!(
        identical > 0,
        "{root}: not one file came back byte-identical, so the header this crate renders matches          nothing in the set"
    );
    assert!(
        fortran > 0,
        "{root}: not one file is Fortran-ordered, so this set does not exercise the case that \
         motivated the module"
    );
    eprintln!(
        "{root}: {} files, {read} read and re-serialised ({identical} byte-identical, {respelled} \
         with a header spelled by another writer), {skipped} skipped, {fortran} Fortran-ordered, \
         {} distinct (dtype, order, rank) combinations: {kinds:?}",
        files.len(),
        kinds.len()
    );
}

/// A file far larger than a block is read a box at a time, and each box is the
/// bytes at the offsets the format puts them at.
///
/// Gated on `BLOCKFLOW_NPY_LARGE`, a path to one recorded file, because the
/// point of it is a file too big to want in a hermetic suite. The check is
/// deliberately **not** against this crate's own whole-file reader — that would
/// compare `for_each_run` with itself. It is against a second, textbook offset
/// computation written here, and against bytes seeked to one voxel at a time.
#[test]
fn a_file_larger_than_a_block_is_read_one_box_at_a_time() {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let Ok(path) = std::env::var("BLOCKFLOW_NPY_LARGE") else {
        return;
    };
    let path = PathBuf::from(path);
    let header = read_header_file(&path).expect("a header");
    assert_eq!(header.dtype, Dtype::U16, "this check is written for uint16");
    assert_eq!(header.shape.len(), 3);
    let shape = header.shape.clone();
    let source: NpySource<u16> = NpySource::open(&path, OrderPolicy::Either).expect("opened");

    let mut file = std::fs::File::open(&path).expect("readable");
    // The textbook flat index, computed here rather than borrowed.
    let flat = |at: [usize; 3]| -> usize {
        match header.order {
            Order::C => (at[0] * shape[1] + at[1]) * shape[2] + at[2],
            Order::Fortran => (at[2] * shape[1] + at[1]) * shape[0] + at[0],
        }
    };

    let boxes = [
        ([0usize, 0, 0], [3usize, 3, 3]),
        ([shape[0] / 2, shape[1] / 3, shape[2] / 5], [4, 5, 6]),
        ([shape[0] - 2, shape[1] - 2, shape[2] - 2], [2, 2, 2]),
    ];
    for (start, extent) in boxes {
        let region = Region::new(&start, &extent);
        let got = source.read_region(&region).expect("a region");
        assert_eq!(got.shape(), &extent[..]);
        for z in 0..extent[0] {
            for y in 0..extent[1] {
                for x in 0..extent[2] {
                    let at = [start[0] + z, start[1] + y, start[2] + x];
                    let mut raw = [0u8; 2];
                    file.seek(SeekFrom::Start((header.data_offset + flat(at) * 2) as u64))
                        .expect("seek");
                    file.read_exact(&mut raw).expect("read");
                    assert_eq!(
                        got[[z, y, x]],
                        u16::from_le_bytes(raw),
                        "{} {start:?} {extent:?} at {z},{y},{x}",
                        path.display()
                    );
                }
            }
        }
    }
    eprintln!(
        "{}: {:?} of {} in {}, {} bytes, read by region",
        path.display(),
        header.shape,
        header.dtype.numpy_name(),
        header.order.name(),
        header.file_bytes("large").expect("a size")
    );
}

/// Read a file and write it back in the order it was stored in.
fn reread(bytes: &[u8], what: &str) -> Vec<u8> {
    let header = Header::parse(bytes, what).unwrap_or_else(|error| panic!("{error}"));
    macro_rules! through {
        ($element:ty) => {{
            let array: ArrayD<$element> = read_array(bytes, what, OrderPolicy::Either)
                .unwrap_or_else(|error| panic!("{error}"));
            write_array(&array, header.order, what).unwrap_or_else(|error| panic!("{error}"))
        }};
    }
    match header.dtype {
        Dtype::Bool => through!(bool),
        Dtype::U8 => through!(u8),
        Dtype::U16 => through!(u16),
        Dtype::U32 => through!(u32),
        Dtype::U64 => through!(u64),
        Dtype::I8 => through!(i8),
        Dtype::I16 => through!(i16),
        Dtype::I32 => through!(i32),
        Dtype::I64 => through!(i64),
        Dtype::F32 => through!(f32),
        Dtype::F64 => through!(f64),
        Dtype::F16 => unreachable!("skipped above"),
    }
}

fn collect(directory: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, into);
        } else if path.extension().is_some_and(|extension| extension == "npy") {
            into.push(path);
        }
    }
}
