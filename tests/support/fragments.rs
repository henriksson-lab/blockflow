use std::collections::BTreeMap;

use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;

pub fn sidecars(
    env: &ArrayEnvironment,
    stream: &str,
    phase: usize,
    grid: &BlockGrid,
) -> BTreeMap<[usize; 3], Vec<u8>> {
    grid.cores()
        .into_iter()
        .map(|core| {
            let bytes = env
                .read_sidecar(stream, phase, core.index)
                .expect("the store answers")
                .unwrap_or_else(|| panic!("block {:?} wrote no sidecar on {stream}", core.index));
            (core.index, bytes)
        })
        .collect()
}

pub fn sidecars_present(
    env: &ArrayEnvironment,
    stream: &str,
    phase: usize,
    grid: &BlockGrid,
) -> BTreeMap<[usize; 3], Vec<u8>> {
    grid.cores()
        .into_iter()
        .filter_map(|core| {
            env.read_sidecar(stream, phase, core.index)
                .expect("the store answers")
                .map(|bytes| (core.index, bytes))
        })
        .collect()
}

pub fn decoded_sidecars<T>(
    env: &ArrayEnvironment,
    stream: &str,
    phase: usize,
    grid: &BlockGrid,
    decode: impl Fn(&[u8]) -> blockflow::Result<T>,
) -> BTreeMap<[usize; 3], T> {
    sidecars(env, stream, phase, grid)
        .into_iter()
        .map(|(index, bytes)| {
            let decoded = decode(&bytes).unwrap_or_else(|error| panic!("block {index:?}: {error}"));
            (index, decoded)
        })
        .collect()
}
