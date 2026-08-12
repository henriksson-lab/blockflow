// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Why this exists at all
// ----------------------
// The framework used to return `clearmap_rs::ClearMapError`, which is a
// thirty-variant enum covering everything from Elastix invocation to npy header
// parsing. Depending on it would have inverted the dependency this crate was
// extracted to establish, and it would have been dishonest besides: the whole
// framework raises exactly **two** kinds of failure.
//
// * `InvalidArgument` — a plan, a region or a chain is inconsistent. Every
//   guard in `geometry`, `decomposition`, `graph` and `tiling` reports this.
// * `ShapeMismatch` — an op returned an array of the wrong extent, which is the
//   one failure a caller can act on programmatically.
//
// Plus one that only appears at the boundary:
//
// * `Backend` — a `RegionSource`/`RegionSink` implementation outside this crate
//   failed. We deliberately do not model its error; we carry its message. This
//   crate has no way to interpret "zarrs chunk grid has no chunk at the origin"
//   and no business trying.
//
// Conversion is the *caller's* job, and is stated once at the boundary rather
// than at every call site: `clearmap-rs` implements `From<Error> for
// ClearMapError` and `From<ClearMapError> for Error`, and nothing here needs to
// know that either type exists.

use std::fmt;

/// What this crate's operations return.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything the framework can fail with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A plan, region or chain is internally inconsistent. The message names
    /// what and where; it is meant to be read, not matched on.
    InvalidArgument(String),
    /// An array did not have the extent the contract requires — most often an
    /// op returning a different shape than it was handed.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// A source or sink implemented outside this crate failed. The string is
    /// the foreign error's `Display`, preserved verbatim.
    Backend(String),
}

impl Error {
    /// `Error::InvalidArgument` from anything displayable, so guards read as one
    /// line rather than three.
    pub fn invalid(message: impl fmt::Display) -> Self {
        Self::InvalidArgument(message.to_string())
    }

    /// `Error::Backend` from a foreign error.
    pub fn backend(message: impl fmt::Display) -> Self {
        Self::Backend(message.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(formatter, "invalid argument: {message}"),
            Self::ShapeMismatch { expected, got } => {
                write!(
                    formatter,
                    "shape mismatch: expected {expected:?}, got {got:?}"
                )
            }
            Self::Backend(message) => write!(formatter, "backend: {message}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_display_of_each_variant_names_the_problem() {
        assert_eq!(
            Error::invalid("halo 3 exceeds block 2").to_string(),
            "invalid argument: halo 3 exceeds block 2"
        );
        assert_eq!(
            Error::ShapeMismatch {
                expected: vec![4, 4],
                got: vec![4, 3],
            }
            .to_string(),
            "shape mismatch: expected [4, 4], got [4, 3]"
        );
        assert_eq!(
            Error::backend("no such chunk").to_string(),
            "backend: no such chunk"
        );
    }
}
