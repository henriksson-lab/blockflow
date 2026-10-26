# Publishing the SAM dependency chain

Blockflow's SAM adapter and training policy use one exact Candle source identity. Optional dependencies must still have resolvable manifests when SAM is disabled. Do not replace these git revisions with paths to sibling checkouts: that breaks every CI job before compilation. `scripts/check_portable_manifests.py` checks all tracked Cargo manifests, including optional, platform-specific and patched dependencies.

The 2026-10-26 fix follows this dependency order:

| Repository | Working checkout | Revision / branch to push |
| --- | --- | --- |
| `mahogny/gemm` | `/data/henriksson/github/claude/gemm` | `a9ea398a74911f4a0cc79261044cbdf4ce9efd91`, branch `main` |
| `mahogny/candle` | `/data/henriksson/github/claude/candle` | `0998c8c655ae84330149ddbff509916254b887a1`, branch `integration/stardist-clearmap-ops` |
| `henriksson-lab/sam2-rs` | `/data/henriksson/github/claude/blockflow/sam2-rs` | `edb4cb76cfd85bdc69915893a9addea7ebe3bd62`, branch `main` |
| `henriksson-lab/blockflow` | `/data/henriksson/github/claude/blockflow` | this fix, branch `main` |

The GEMM fork is <https://github.com/mahogny/gemm>, created from <https://github.com/sarah-quinones/gemm>. The old `sarah-ek/gemm` upstream URL redirects there. The patch was transferred unchanged into the standalone GEMM checkout: its `origin` points to `git@github.com:mahogny/gemm.git`. Push its `main` branch normally; the pinned patch commit is unchanged. The old nested checkout is no longer the working location for publishing this fork.

Push GEMM, then Candle, then SAM2 before pushing Blockflow. Otherwise even a non-SAM CI job cannot resolve its dependency graph. Native CPU tests and dependency resolution are validated locally with temporary per-process Git URL rewrites to the original checkouts. These rewrites are not committed or installed globally. They verify exact git contents without claiming the revisions are already available on GitHub. Remote CI must be rerun after publication.

Candle no longer builds against the untracked `vendor/gemm` directory. Its small custom GEMM changes (packing thresholds and the AVX-512 control used by its linear layer) remain in the separate fork. Reference repositories, weights, caches and build directories are not part of these commits.

Validation for this fix: two manifest-guard tests; 42 internal dependency paths checked; fresh lockfile resolution from an exported checkout without sibling repositories; default and wide-feature Clippy with warnings denied; 1,485 Blockflow library tests; 12 affected Zarr/time-series integration tests; explicit SAM all-target release build; 46 SAM2 CPU library tests; 21 Candle NN CPU tests; and the GEMM packing-threshold regression. All passed. The full hosted OS/model matrix has not been rerun: publication is left to the repository owner.

On 2026-10-26 the local SAM2 checkout moved to `/home/mahogny/github/claude/blockflow/sam2-rs` (canonical `/data/henriksson/github/claude/blockflow/sam2-rs`). It remains a separate repository and is ignored by the outer Blockflow checkout. Dependency pins are unchanged by this move.

The next hosted run used Rust 1.99.1. Its additional failures were corrected with explicit rustdoc paths/code formatting, a stable compare-exchange implementation of saturating residency bookkeeping, default target selection for mixed library/binary example packages, release-mode workspace CLI tests, and a dimension-aware atlas test fixture. The exact CI compiler passes docs with warnings denied, default/wide Clippy, the concurrent counter regression, both StarDist example builds, and the three previously failing workspace integration targets.
