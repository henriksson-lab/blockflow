# Repository instructions

- Use release builds for Rust programs, examples, tests, and benchmarks. Pass
  `--release` to `cargo build`, `cargo run`, and `cargo test`, and run binaries
  from `target/release`.
- Use a debug build only when a debugger or another specific diagnostic tool
  requires one, or when the user explicitly requests it.
- Never report performance measurements from a debug build.
- Before building a pipeline, op or example on blockflow, read
  `docs/GUIDE.md`: a short, checked map of the op traits, plan assembly,
  common patterns, publishing and traps. It saves surveying the crate.
