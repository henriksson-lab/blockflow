use std::fmt::Debug;

use ndarray::Array3;

use blockflow::voxels::{differing_bits, differing_elements as differing_values, Voxels};
use blockflow::Dtype;

pub trait Bits: Copy + Debug {
    type Repr: Copy + Debug + Eq;

    fn bits(self) -> Self::Repr;
}

macro_rules! bits_self {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Bits for $ty {
                type Repr = $ty;

                fn bits(self) -> Self::Repr {
                    self
                }
            }
        )+
    };
}

bits_self!(bool, u8, u16, u32, u64, i32, i64, usize);

impl Bits for f32 {
    type Repr = u32;

    fn bits(self) -> Self::Repr {
        self.to_bits()
    }
}

impl Bits for f64 {
    type Repr = u64;

    fn bits(self) -> Self::Repr {
        self.to_bits()
    }
}

#[track_caller]
pub fn identical<T: Bits>(got: &Array3<T>, want: &Array3<T>, what: &str) {
    assert_eq!(got.dim(), want.dim(), "{what}: shape");

    let mut first = None;
    let mut differing = 0usize;
    for (index, got) in got.indexed_iter() {
        let want = want[index];
        if got.bits() != want.bits() {
            differing += 1;
            first.get_or_insert((index, *got, want));
        }
    }

    if let Some((index, got, want)) = first {
        panic!(
            "{what}: {differing} voxels differ on the bits; first at {index:?}: got {got:?} \
             ({:?}) and wanted {want:?} ({:?})",
            got.bits(),
            want.bits()
        );
    }
}

#[track_caller]
pub fn differing<T: Bits>(got: &Array3<T>, want: &Array3<T>) -> usize {
    assert_eq!(got.dim(), want.dim(), "arrays have different shapes");
    got.indexed_iter()
        .filter(|(index, got)| got.bits() != want[*index].bits())
        .count()
}

#[track_caller]
pub fn differs<T: Bits>(got: &Array3<T>, want: &Array3<T>, what: &str) {
    let differing = differing(got, want);
    assert!(differing > 0, "{what}: the two agreed everywhere");
}

#[track_caller]
pub fn voxels_differing(left: &Voxels, right: &Voxels) -> usize {
    assert_eq!(
        left.dtype(),
        right.dtype(),
        "two answers of different element types are not comparable"
    );
    assert_eq!(left.shape(), right.shape(), "voxels have different shapes");

    let count = match left.dtype() {
        Dtype::Bool => differing_values(
            left.view::<bool>().expect("bool"),
            right.view::<bool>().expect("bool"),
        ),
        Dtype::U8 => differing_values(
            left.view::<u8>().expect("uint8"),
            right.view::<u8>().expect("uint8"),
        ),
        Dtype::U16 => differing_values(
            left.view::<u16>().expect("uint16"),
            right.view::<u16>().expect("uint16"),
        ),
        Dtype::U32 => differing_values(
            left.view::<u32>().expect("uint32"),
            right.view::<u32>().expect("uint32"),
        ),
        Dtype::U64 => differing_values(
            left.view::<u64>().expect("uint64"),
            right.view::<u64>().expect("uint64"),
        ),
        Dtype::I8 => differing_values(
            left.view::<i8>().expect("int8"),
            right.view::<i8>().expect("int8"),
        ),
        Dtype::I16 => differing_values(
            left.view::<i16>().expect("int16"),
            right.view::<i16>().expect("int16"),
        ),
        Dtype::I32 => differing_values(
            left.view::<i32>().expect("int32"),
            right.view::<i32>().expect("int32"),
        ),
        Dtype::I64 => differing_values(
            left.view::<i64>().expect("int64"),
            right.view::<i64>().expect("int64"),
        ),
        Dtype::F32 => differing_bits(
            left.view::<f32>().expect("float32"),
            right.view::<f32>().expect("float32"),
        ),
        Dtype::F64 => differing_bits(
            left.view::<f64>().expect("float64"),
            right.view::<f64>().expect("float64"),
        ),
        Dtype::F16 => panic!("float16 has no Voxels storage variant"),
    }
    .expect("comparable buffers");
    count as usize
}

#[track_caller]
pub fn voxels_identical(left: &Voxels, right: &Voxels, what: &str) {
    let differing = voxels_differing(left, right);
    assert_eq!(
        differing, 0,
        "{what}: the two buffers differ in {differing} voxels"
    );
}
