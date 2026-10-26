# Publishing the SAM dependency chain

Blockflow's SAM adapter and training policy use one exact Candle source identity. Optional dependencies must still have resolvable manifests when SAM is disabled. Do not replace these git revisions with paths to sibling checkouts: that breaks every CI job before compilation. `scripts/check_portable_manifests.py` checks all tracked Cargo manifests, including optional, platform-specific and patched dependencies.

The 2026-10-26 fix follows this dependency order:

| Repository | Original working checkout | Revision / branch to push |
| --- | --- | --- |
| `henriksson-lab/gemm` | `/data/henriksson/github/claude/candle/vendor/gemm` | `a9ea398a74911f4a0cc79261044cbdf4ce9efd91`, branch `candle-gemm-tuning` |
| `mahogny/candle` | `/data/henriksson/github/claude/candle` | `b27e8c7c04d16c19d335325d7abd6331254c58a2`, branch `sam2-cpu-perf` |
| `henriksson-lab/sam2-rs` | `/data/henriksson/github/em/found/sam2-rs` | `90b41cc85849580479049d620a73617d240f2990`, branch `main` |
| `henriksson-lab/blockflow` | `/data/henriksson/github/claude/blockflow` | this fix, branch `main` |

First create the lab GEMM fork from <https://github.com/sarah-quinones/gemm>. The old `sarah-ek/gemm` URL redirects there. The original GEMM checkout keeps `origin` for that upstream and `fork` for `git@github.com:henriksson-lab/gemm.git`; publish the patch branch to `fork`. No force-push or upstream branch replacement is necessary: Cargo can fetch a pinned commit from that patch branch.

Push GEMM, then Candle, then SAM2 before pushing Blockflow. Otherwise even a non-SAM CI job cannot resolve its dependency graph. Native CPU tests and dependency resolution are validated locally with temporary per-process Git URL rewrites to the original checkouts. These rewrites are not committed or installed globally. They verify exact git contents without claiming the revisions are already available on GitHub. Remote CI must be rerun after publication.

Candle no longer builds against the untracked `vendor/gemm` directory. Its small custom GEMM changes (packing thresholds and the AVX-512 control used by its linear layer) remain in the separate fork. Reference repositories, weights, caches and build directories are not part of these commits.

Validation for this fix: two manifest-guard tests; 42 internal dependency paths checked; fresh lockfile resolution from an exported checkout without sibling repositories; default and wide-feature Clippy with warnings denied; 1,485 Blockflow library tests; 12 affected Zarr/time-series integration tests; explicit SAM all-target release build; 46 SAM2 CPU library tests; 21 Candle NN CPU tests; and the GEMM packing-threshold regression. All passed. The full hosted OS/model matrix has not been rerun: publication is left to the repository owner.
