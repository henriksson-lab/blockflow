// SPDX-License-Identifier: MIT
// Original work for this crate.

//! `cell-density`: detect cells in one channel of an OME-Zarr image and write
//! them as NGFF object tables attached to it, with optional atlas columns,
//! per-region counts and a density image. See the README for the output.

use cell_density::cli::{run, Args};
use clap::Parser;

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("cell-density: this example requires cargo run --release");
        std::process::exit(1);
    }
    limit_malloc_arenas();
    let args = Args::parse();
    match run(&args)
        .and_then(|report| serde_json::to_string_pretty(&report).map_err(blockflow::Error::backend))
    {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("cell-density: {error}");
            std::process::exit(1);
        }
    }
}

/// Two malloc arenas instead of glibc's eight per core, unless
/// `MALLOC_ARENA_MAX` says otherwise. Forty worker threads each freeing block
/// buffers into an arena of their own left about 3 GB of a whole-tile run's
/// 10 GB peak resident in freed memory; with two the peak was 6.6 GB and the
/// CPU time unchanged. Set before any thread starts.
fn limit_malloc_arenas() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if std::env::var_os("MALLOC_ARENA_MAX").is_none() {
        // SAFETY: `mallopt` only sets an allocator parameter; it is called
        // before any other thread exists.
        unsafe {
            libc::mallopt(libc::M_ARENA_MAX, 2);
        }
    }
}
