// SPDX-License-Identifier: MIT
//
//! How a Zarr v3 array is laid out on disk: its chunk grid, optional
//! `sharding_indexed` shards, and the compression inside them.
//!
//! [`crate::zarr_env::Compression`] says how bytes are encoded; this says how
//! they are grouped. The two together are what "write it like level 0" means,
//! which is the default for every array [`crate::intensity_pyramid`] creates.
//!
//! Metadata is produced as a Zarr v3 `zarr.json` document rather than through
//! `zarrs`'s builder, so that what is written is exactly the JSON stated here
//! (and readable by zarr-python, tensorstore and ome-zarr-py), and validated by
//! opening it with `zarrs` afterwards.

use serde_json::{json, Value as Json};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::zarr_env::{compression_of_codecs, Compression};

/// The on-disk grouping of one array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageLayout {
    /// The unit of decoding: the inner chunk of a sharded array, or the chunk
    /// of an unsharded one.
    pub chunk: Vec<usize>,
    /// The shard shape (the array's chunk grid) when sharded. Every entry is a
    /// multiple of the matching `chunk` entry.
    pub shard: Option<Vec<usize>>,
    /// The compression of each `chunk`.
    pub compression: Compression,
    /// Whether the shard index carries a `crc32c` checksum. Ignored unsharded.
    pub index_checksum: bool,
}

impl StorageLayout {
    /// An unsharded layout.
    pub fn chunked(chunk: Vec<usize>, compression: Compression) -> Self {
        Self {
            chunk,
            shard: None,
            compression,
            index_checksum: false,
        }
    }

    /// A sharded layout with a `crc32c` shard index, which is what `zarrs`
    /// and zarr-python write by default.
    pub fn sharded(chunk: Vec<usize>, shard: Vec<usize>, compression: Compression) -> Self {
        Self {
            chunk,
            shard: Some(shard),
            compression,
            index_checksum: true,
        }
    }

    /// Read the layout off an array's `zarr.json` document.
    pub fn of_metadata(metadata: &Json, path: &str) -> Result<Self> {
        let grid = metadata
            .pointer("/chunk_grid/configuration/chunk_shape")
            .and_then(Json::as_array)
            .ok_or_else(|| {
                Error::InvalidArgument(format!("{path}: array has no regular chunk grid"))
            })
            .and_then(|shape| usize_list(shape, path))?;
        let codecs = metadata
            .get("codecs")
            .and_then(Json::as_array)
            .ok_or_else(|| Error::InvalidArgument(format!("{path}: array declares no codecs")))?;
        let compression = compression_of_codecs(codecs, path)?;
        let sharding = codecs
            .iter()
            .find(|codec| codec.get("name").and_then(Json::as_str) == Some("sharding_indexed"));
        Ok(match sharding {
            None => Self::chunked(grid, compression),
            Some(codec) => {
                let inner = codec
                    .pointer("/configuration/chunk_shape")
                    .and_then(Json::as_array)
                    .ok_or_else(|| {
                        Error::InvalidArgument(format!("{path}: sharding has no chunk_shape"))
                    })
                    .and_then(|shape| usize_list(shape, path))?;
                let index_checksum = codec
                    .pointer("/configuration/index_codecs")
                    .and_then(Json::as_array)
                    .is_some_and(|codecs| {
                        codecs
                            .iter()
                            .any(|codec| codec.get("name").and_then(Json::as_str) == Some("crc32c"))
                    });
                Self {
                    chunk: inner,
                    shard: Some(grid),
                    compression,
                    index_checksum,
                }
            }
        })
    }

    /// The array's chunk grid: the shard when sharded, the chunk otherwise.
    /// This is the unit a writer must own exclusively.
    pub fn write_unit(&self) -> &[usize] {
        self.shard.as_deref().unwrap_or(&self.chunk)
    }

    /// This layout fitted to an array of `shape`.
    ///
    /// Chunk dims are capped to the array's extent, and shard dims to the
    /// extent rounded up to a whole number of (capped) chunks, so a small level
    /// is not padded out to a large shard. Each shard dim stays a multiple of
    /// its chunk dim, which `sharding_indexed` requires.
    pub fn fitted_to(&self, shape: &[usize]) -> Result<Self> {
        if self.chunk.len() != shape.len()
            || self
                .shard
                .as_ref()
                .is_some_and(|shard| shard.len() != shape.len())
        {
            return Err(Error::InvalidArgument(format!(
                "layout rank {} does not match array rank {}",
                self.chunk.len(),
                shape.len()
            )));
        }
        let chunk: Vec<usize> = self
            .chunk
            .iter()
            .zip(shape)
            .map(|(&chunk, &extent)| chunk.max(1).min(extent.max(1)))
            .collect();
        let shard = self.shard.as_ref().map(|shard| {
            shard
                .iter()
                .zip(shape)
                .zip(&chunk)
                .map(|((&shard, &extent), &chunk)| {
                    let whole = extent.max(1).div_ceil(chunk) * chunk;
                    let wanted = shard.max(chunk).div_ceil(chunk) * chunk;
                    wanted.min(whole)
                })
                .collect()
        });
        Ok(Self {
            chunk,
            shard,
            compression: self.compression,
            index_checksum: self.index_checksum,
        })
    }

    /// The Zarr v3 `codecs` list for `dtype`.
    pub fn codecs_json(&self, dtype: Dtype) -> Json {
        let mut inner = vec![json!({"name": "bytes", "configuration": {"endian": "little"}})];
        if let Some(codec) = self.compression.codec_json(dtype) {
            inner.push(codec);
        }
        match &self.shard {
            None => Json::Array(inner),
            Some(_) => {
                let mut index =
                    vec![json!({"name": "bytes", "configuration": {"endian": "little"}})];
                if self.index_checksum {
                    index.push(json!({"name": "crc32c"}));
                }
                json!([{
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": self.chunk,
                        "codecs": inner,
                        "index_codecs": index,
                        "index_location": "end"
                    }
                }])
            }
        }
    }

    /// A complete Zarr v3 array `zarr.json` document.
    ///
    /// `fill_value` is a JSON value as Zarr v3 states it (`0`, `"NaN"`, ...);
    /// `dimension_names` is copied verbatim when given.
    pub fn array_metadata(
        &self,
        shape: &[usize],
        dtype: Dtype,
        fill_value: Json,
        dimension_names: Option<Json>,
    ) -> Result<Json> {
        self.check(shape)?;
        let mut metadata = json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": dtype.numpy_name(),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": self.write_unit()}
            },
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": fill_value,
            "codecs": self.codecs_json(dtype),
            "attributes": {}
        });
        if let Some(names) = dimension_names {
            metadata["dimension_names"] = names;
        }
        Ok(metadata)
    }

    fn check(&self, shape: &[usize]) -> Result<()> {
        if self.chunk.len() != shape.len() || self.chunk.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "chunk {:?} is not a positive shape of rank {}",
                self.chunk,
                shape.len()
            )));
        }
        if let Some(shard) = &self.shard {
            if shard.len() != shape.len()
                || shard
                    .iter()
                    .zip(&self.chunk)
                    .any(|(&shard, &chunk)| shard == 0 || shard % chunk != 0)
            {
                return Err(Error::InvalidArgument(format!(
                    "shard {shard:?} must be a positive multiple of chunk {:?} per axis",
                    self.chunk
                )));
            }
        }
        Ok(())
    }

    /// Short human description, e.g. `shard [64, 64, 64] / chunk [32, 32, 32] blosc-zstd5-shuffle`.
    pub fn describe(&self) -> String {
        match &self.shard {
            Some(shard) => format!(
                "shard {shard:?} / chunk {:?} {}",
                self.chunk,
                self.compression.name()
            ),
            None => format!("chunk {:?} {}", self.chunk, self.compression.name()),
        }
    }
}

/// The decoding unit of an opened array: the inner chunk shape when the array
/// is `sharding_indexed`, its chunk shape otherwise.
pub fn read_chunk_shape<S: ?Sized>(array: &zarrs::array::Array<S>) -> Result<Vec<usize>> {
    let metadata = serde_json::to_value(array.metadata()).map_err(Error::backend)?;
    let inner = metadata
        .get("codecs")
        .and_then(Json::as_array)
        .and_then(|codecs| {
            codecs
                .iter()
                .find(|codec| codec.get("name").and_then(Json::as_str) == Some("sharding_indexed"))
        })
        .and_then(|codec| codec.pointer("/configuration/chunk_shape"))
        .and_then(Json::as_array);
    if let Some(inner) = inner {
        return usize_list(inner, "array");
    }
    Ok(array
        .chunk_shape(&vec![0; array.dimensionality()])
        .map_err(Error::backend)?
        .iter()
        .map(|value| value.get() as usize)
        .collect())
}

fn usize_list(values: &[Json], path: &str) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    Error::InvalidArgument(format!("{path}: invalid shape entry {value}"))
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zarr_env::{BloscCname, BloscShuffle};

    /// The array document bf2zarr / zarrs write for a sharded blosc volume.
    fn sharded_blosc() -> Json {
        json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [1030, 2387, 1422],
            "data_type": "uint16",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [160, 640, 640]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0,
            "codecs": [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [32, 128, 128],
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "blosc", "configuration": {
                            "cname": "zstd", "clevel": 5, "shuffle": "shuffle",
                            "typesize": 2, "blocksize": 0}}
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"}
                    ],
                    "index_location": "end"
                }
            }],
            "dimension_names": ["z", "y", "x"]
        })
    }

    #[test]
    fn a_sharded_blosc_document_parses_to_its_inner_compression() {
        let layout = StorageLayout::of_metadata(&sharded_blosc(), "0").unwrap();
        assert_eq!(layout.chunk, vec![32, 128, 128]);
        assert_eq!(layout.shard, Some(vec![160, 640, 640]));
        assert!(layout.index_checksum);
        assert_eq!(
            layout.compression,
            Compression::Blosc {
                cname: BloscCname::Zstd,
                clevel: 5,
                shuffle: BloscShuffle::Byte
            }
        );
        // And it writes back out the same codec chain.
        let written = layout.codecs_json(Dtype::U16);
        assert_eq!(written, sharded_blosc()["codecs"]);
    }

    #[test]
    fn transpose_crc32c_and_numcodecs_shuffle_integers_are_accepted() {
        let codecs = json!([
            {"name": "transpose", "configuration": {"order": [0, 1, 2]}},
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 3, "shuffle": 2}},
            {"name": "crc32c"}
        ]);
        let found = compression_of_codecs(codecs.as_array().unwrap(), "x").unwrap();
        assert_eq!(
            found,
            Compression::Blosc {
                cname: BloscCname::Lz4,
                clevel: 3,
                shuffle: BloscShuffle::Bit
            }
        );
        let bad = json!([{"name": "zfp"}]);
        assert!(compression_of_codecs(bad.as_array().unwrap(), "x").is_err());
    }

    #[test]
    fn fitting_caps_chunks_and_keeps_shards_whole_multiples() {
        let layout = StorageLayout::of_metadata(&sharded_blosc(), "0").unwrap();
        let fitted = layout.fitted_to(&[50, 300, 1000]).unwrap();
        assert_eq!(fitted.chunk, vec![32, 128, 128]);
        assert_eq!(fitted.shard, Some(vec![64, 384, 640]));
        let tiny = layout.fitted_to(&[1, 20, 20]).unwrap();
        assert_eq!(tiny.chunk, vec![1, 20, 20]);
        assert_eq!(tiny.shard, Some(vec![1, 20, 20]));
    }
}
