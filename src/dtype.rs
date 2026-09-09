// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A tag, not an abstraction. The framework never reads or writes an element; it
// only needs to know **how many bytes one costs**, because every question it
// answers about a decomposition — will a block fit, how many bytes will this
// phase materialise, how much does one cached chunk take — is a byte question.
//
// It is deliberately not a trait and deliberately not generic. A generic
// element parameter would infect `Constraints`, `Decomposition`, `CostModel`
// and the cache key, all of which are type-erased on purpose: the planner
// reasons about a whole workflow whose phases may have different element types,
// and the cache holds chunks from seven arrays at once.
//
// The variant list mirrors what the storage layers this feeds actually carry.
// Callers with their own dtype enum convert at the boundary; that conversion is
// one `match` and belongs on their side, not here.

/// The element type of an array, for byte accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
}

impl Dtype {
    pub const ALL: [Self; 12] = [
        Self::Bool,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F16,
        Self::F32,
        Self::F64,
    ];

    pub const VOXEL_TYPES: [Self; 11] = [
        Self::Bool,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F32,
        Self::F64,
    ];

    /// Every dtype in the canonical order used by storage dispatch tables.
    pub fn all() -> &'static [Self] {
        &Self::ALL
    }

    /// Every dtype that has a concrete [`crate::voxels::Voxels`] variant.
    pub fn voxel_types() -> &'static [Self] {
        &Self::VOXEL_TYPES
    }

    /// Bytes one element occupies **in memory**, decoded.
    ///
    /// Not what it occupies on disk: a compressed `bool` volume measures ~19.7x
    /// smaller than this, which is exactly why the cache has an encoded tier.
    pub fn size_of(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// The NumPy spelling, for manifests and cross-language comparison.
    pub fn numpy_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::F16 => "float16",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    /// The inverse of [`Dtype::numpy_name`].
    ///
    /// Exists because a decomposition now travels: a coordinator hands one to a
    /// worker in another process, and the tag has to survive the trip in the
    /// spelling manifests already use rather than in a second one invented for
    /// the wire.
    pub fn from_numpy_name(name: &str) -> Option<Self> {
        Some(match name {
            "bool" => Self::Bool,
            "uint8" => Self::U8,
            "uint16" => Self::U16,
            "uint32" => Self::U32,
            "uint64" => Self::U64,
            "int8" => Self::I8,
            "int16" => Self::I16,
            "int32" => Self::I32,
            "int64" => Self::I64,
            "float16" => Self::F16,
            "float32" => Self::F32,
            "float64" => Self::F64,
            _ => return None,
        })
    }
}

/// Dispatch from a runtime [`Dtype`] tag to the Rust element type that stores it.
///
/// This macro is for boundary code that receives erased storage and then calls
/// monomorphised implementations. It binds `$element` as a type alias in each
/// arm, and lets the caller spell the `F16` case because this crate treats
/// `Dtype::F16` as a storage tag rather than a `Voxels` element.
#[macro_export]
macro_rules! dtype_dispatch {
    ($dtype:expr, |$element:ident| $body:expr, f16 => $f16:expr) => {{
        match $dtype {
            $crate::Dtype::Bool => {
                type $element = bool;
                $body
            }
            $crate::Dtype::U8 => {
                type $element = u8;
                $body
            }
            $crate::Dtype::U16 => {
                type $element = u16;
                $body
            }
            $crate::Dtype::U32 => {
                type $element = u32;
                $body
            }
            $crate::Dtype::U64 => {
                type $element = u64;
                $body
            }
            $crate::Dtype::I8 => {
                type $element = i8;
                $body
            }
            $crate::Dtype::I16 => {
                type $element = i16;
                $body
            }
            $crate::Dtype::I32 => {
                type $element = i32;
                $body
            }
            $crate::Dtype::I64 => {
                type $element = i64;
                $body
            }
            $crate::Dtype::F16 => $f16,
            $crate::Dtype::F32 => {
                type $element = f32;
                $body
            }
            $crate::Dtype::F64 => {
                type $element = f64;
                $body
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_reads_back_as_the_variant_that_wrote_it() {
        for &dtype in Dtype::all() {
            assert_eq!(Dtype::from_numpy_name(dtype.numpy_name()), Some(dtype));
        }
        // **The absence is deliberate and argued, not pending.** A complex
        // element is refused at the root rather than merely unwritten:
        // `VoxelElement` requires `into_f64` and `from_f64`, `Voxels::filled`
        // takes an `f64`, and `Voxels::uniform` reports one to a short circuit —
        // so a complex block could only project to a real and lie about what it
        // holds. `docs/ops-survey/README.md`'s G3 row carries the argument and
        // the operation that was built without it. This assertion is the pin on
        // that decision; if a complex variant is ever added it is **inverted**
        // here rather than deleted.
        assert_eq!(Dtype::from_numpy_name("complex64"), None);
        assert_eq!(Dtype::from_numpy_name("complex128"), None);
    }

    #[test]
    fn every_variant_has_a_width_and_a_name() {
        for &dtype in Dtype::all() {
            assert!(matches!(dtype.size_of(), 1 | 2 | 4 | 8));
            assert!(!dtype.numpy_name().is_empty());
        }
        assert_eq!(Dtype::F64.size_of(), 8);
        assert_eq!(Dtype::Bool.size_of(), 1);
    }
}
