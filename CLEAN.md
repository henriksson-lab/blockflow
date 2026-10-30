# Test Suite Cleanup Plan

Goal: reduce the test suite by removing tests that mainly freeze prose, current
research conclusions, report tables, exact diagnostic wording, or trivial API
shape. Keep tests that catch complex regressions: byte identity across
decompositions, independent-oracle agreement, scheduler conservation laws,
race/protocol behavior, unsafe refusals, and model-vs-executor drift.

Current rough scale: `tests/` has 137 Rust test files and about 100k LOC. The
largest file is `tests/measurements.rs` at about 10.8k LOC, so meaningful
cleanup needs both whole-test removal and consolidation inside large files.

## Implemented In The First Pass

- Deleted the remaining source-grep tests in `tests/one_freeing_rule.rs`.
- Deleted redundant `no_two_decompositions_disagree` tests from `image_ops`,
  `normalisation_ops`, `ridge_filter`, `deconvolution`, and the same pattern
  found in `background_removal`.
- Deleted low-risk scenario/report taxonomy tests in `cost_scenarios`.
- Deleted `tests/rank_width_cost.rs` and `tests/pyramid_recipe.rs`.
- Moved benchmark/research/table tests out of default CI with `#[ignore]` in
  `zarr_env`, `zarr_cache`, `barrier_phase`, `barrier_migration`,
  `planner_oracle`, `planner_arena`, `simulate_ranks`, `candidate_window`, and
  `cost_scenarios`.
- Removed the named `measurements.rs` schema/collector-only, repeated builder,
  and constructor smoke tests listed below.
- Moved fitted working-set margin and mask-carrier residency cost tests out of
  default CI.
- Pruned `npy_files.rs` header-version/unsupported-dtype diagnostic tests and
  weakened several exact diagnostic-prose assertions to simple refusal checks.
- Replaced exact diagnostic-prose checks with refusal checks in the barrier
  phase reduction guard and the large unsupported measurement-combination
  bundle.
- Moved additional repeated `measurements.rs` schema/collector and builder-only
  smoke tests out of default CI, while keeping planned-vs-resident numerical
  tests and malformed-input validation active.
- Moved working-set residency cost/economics report tests out of default CI,
  while keeping shape-key, fan-in, and shape-derived residency checks active.
- Second aggressive pass: deleted broad conformance/example suites from
  `tests/` instead of merely ignoring them, so they no longer compile or pin
  APIs during refactoring. This removed the measurement megasuite, NPY and
  row/table conformance suites, many per-op scalar/reference wrappers, broad
  algorithm conformance suites, and fixture/report-only planner artifacts.
- Final aggressive pass: removed the remaining broad algorithm-conformance,
  planner-economics, residency-measurement, scheduler-sweep, and cost-report
  integration suites. The remaining integration suite is mostly framework
  invariants, planner/search conservation, barrier/distributed/cache protocol,
  and executor/simulator agreement. `tests/` files dropped from 146 to 42
  including support files, top-level `tests/*.rs` LOC from about 99k to about
  19.4k, and integration `#[test]` items from 1428 to 256.

## Removal Rule

Remove or move out of default CI when a test's main assertion is one of:

- "the current explanation/report table still says what it said";
- "this README/doc/source phrase still exists";
- exact error-message prose, when the test can assert `is_err()` or an error
  kind instead;
- pairwise decomposition agreement that is implied by all decompositions
  matching the same reference;
- benchmark or cost-economics claims that depend on machine, compression,
  allocator, cache, or fitted constants;
- repeated builder/schema smoke tests where one representative integration test
  gives the same coverage.

Keep when a test checks:

- two independent implementations agree;
- distributed work is neither lost nor duplicated;
- byte identity across block shapes, worker counts, or storage backends;
- dependency, barrier, cache, prefetch, or reissue races;
- malformed input is refused before corrupting execution;
- simulator accounting matches executor accounting.

## Remove First

These are low-value and should be removed outright unless a near-term refactor
specifically needs them.

| Target | Reason | Risk |
|---|---|---|
| `tests/one_freeing_rule.rs::the_freeing_rule_has_exactly_one_home` | Greps source for `Visibility::Internal`. This is architectural policy, not behavior. | Medium |
| `tests/one_freeing_rule.rs::every_shipped_fragment_stream_declares_a_size` | Greps source text for `FragmentOutput::new(...).sized(...)` in a 40-line window. The invariant matters, but source shape is the wrong enforcement. | Medium-high |
| `tests/cost_scenarios.rs::planner_regret_report_names_the_estimated_loss_mechanism` | Commentary taxonomy over scenario names. | Low |
| `tests/cost_scenarios.rs::performance_corpus_covers_the_known_planner_failure_modes` | Corpus bookkeeping and labels, not behavior. | Low |
| `tests/intra_block_slicing.rs::every_shipped_op_refuses_to_be_sliced_today` | Inventory/policy test that shipped ops still refuse slicing and have a reason. | Low-medium |
| `tests/image_ops.rs::no_two_decompositions_disagree` | Redundant if every decomposition already matches the same whole-volume reference. | Low |
| `tests/normalisation_ops.rs::no_two_decompositions_disagree` | Same redundancy pattern. | Low |
| `tests/ridge_filter.rs::no_two_decompositions_disagree` | Same redundancy pattern. | Low |
| `tests/deconvolution.rs::no_two_decompositions_disagree` | Same redundancy pattern. | Low |
| `tests/rank_width_cost.rs` | Tiny one-off cost-ratio test, likely redundant with dtype/width coverage elsewhere. | Low |
| `tests/pyramid_recipe.rs` shape/order recipe tests | Useful as examples, weak as regression tests. Move to example docs if needed. | Low-medium |

## Move Out Of Default CI

These may still be useful as research notebooks, ignored tests, benches, or
scheduled benchmarks. They should not gate every PR.

| Target | Reason |
|---|---|
| `tests/zarr_env.rs::compression_pays_for_bool_and_not_for_float` | Compression/time economics, table printing, machine-sensitive. |
| `tests/barrier_phase.rs::the_four_arms_are_measured_against_each_other` | Aggregate measurement table; focused barrier invariants should stay. |
| `tests/barrier_migration.rs::the_four_arms_are_tabulated_at_every_lattice` | Measurement/tabulation over lattices. |
| `tests/barrier_migration.rs::the_order_check_costs_one_more_pass_over_the_fragment_set` | Cost-accounting claim, not core correctness. |
| Most of `tests/working_set_residency.rs` margin-fitting tests | Fits constants to measured allocator/shape behavior. Keep one structural residency test. |
| `tests/label_materialisation_cost.rs` | Real-volume measurement artifact; keep out of default CI. |
| `tests/zarr_cache.rs::the_prefetch_sweep_has_a_control_at_both_ends` | Cache economics. Keep concurrent correctness tests. |
| `tests/zarr_cache.rs::a_bigger_cache_reads_strictly_fewer_bytes_from_the_store` | Cache economics. Keep answer-preservation and race tests. |
| `tests/planner_oracle.rs::report_planner_choices_against_the_simulator_oracle` | Report generator with regret ceilings. Better as ignored or scheduled planner benchmark. |
| External comparison examples/benchmarks | CellProfiler/OpenCV/skimage/Dask/ImgLib2 comparisons are valuable, but belong in examples, scripts, or scheduled benchmarks. |

## Shrink Or Consolidate

These likely contain useful coverage mixed with repeated narrow checks.

### `tests/measurements.rs`

Biggest cleanup opportunity: about 10.8k LOC.

Prune or consolidate:

- schema/collector-only tests such as
  `centroid_neighbor_rows_have_a_canonical_schema_and_collector`,
  `object_3d_moment_rows_have_ordered_schema_and_collector`, and
  `primitive_colocalization_rows_have_canonical_schema_and_encoder`;
- repeated builder execution tests such as
  `measurement_builder_runs_planned_colocalization`,
  `measurement_builder_runs_planned_costes_colocalization`, and
  `measurement_builder_runs_planned_rank_weighted_colocalization`;
- constructor/name validation bundles such as
  `approximate_distribution_requests_are_named_at_construction`,
  `exact_distribution_requests_are_named_and_validate_row_width`, and
  `projected_convex_requests_are_named_and_validate_hull_width`.

Keep:

- planned-vs-resident numerical tests;
- decomposition-invariance tests for real measurements;
- malformed-data/runtime validation tests where bad rows or bad source facts can
  corrupt execution.

Suggested end state: one or two builder integration tests, one schema smoke test
per public export family if schema is public API, and full numerical/oracle tests
for the complex measurement math.

### `tests/npy_files.rs`

Keep endian, dtype refusal, region IO, and roundtrip coverage. Consider pruning
older-writer/header-version/extreme-value duplication and exact diagnostic
substring checks.

### Exact Diagnostic Prose

Across the suite, replace exact `contains("...")` assertions with structured
error checks where available. If no structured error exists, keep only a minimal
keyword check when the diagnostic itself is important.

Examples to review:

- `tests/barrier_phase.rs` refusal diagnostics around phase reductions;
- repeated diagnostic checks in `tests/measurements.rs`;
- `tests/npy_files.rs` diagnostic wording checks.

## Planner And Simulator Pruning

Remove or move out of CI tests that freeze research conclusions rather than
execution invariants:

- `tests/planner_arena.rs::the_two_judges_record_the_raw_gap_and_simulator_backed_closes_it`;
- `tests/planner_arena.rs::the_phases_overlap_only_under_the_policy_that_fuses`;
- `tests/simulate_ranks.rs::every_ranking_states_the_region_of_rate_space_it_holds_in`;
- `tests/simulate_ranks.rs::the_schedulers_compared`;
- `tests/simulate_ranks.rs::ordering_moves_the_peak_only_at_low_concurrency`;
- `tests/simulate_ranks.rs::block_major_does_not_have_the_smaller_working_set`;
- `tests/candidate_window.rs::an_unbounded_window_is_the_machine_this_crate_has_always_simulated`;
- `tests/candidate_window.rs::a_small_window_moves_a_schedule_that_looks_past_the_first_candidate`;
- `tests/cost_scenarios.rs::simulator_backed_plans_transfer_to_the_other_committed_scenarios`;
- `tests/cost_scenarios.rs::robust_simulator_backed_plans_transfer_under_the_todo4_ceiling`.

Keep planner/simulator tests that enforce conservation laws, independent
oracles, or model/executor agreement:

- `tests/simulator_against_the_executor.rs`;
- DP vs exhaustive tests in `tests/partition_search.rs`;
- `tests/candidate_window.rs::the_window_is_the_first_n_ready_tasks_in_ascending_id_order`;
- `tests/candidate_window.rs::a_window_changes_the_order_and_never_the_work`;
- `tests/simulate_ranks.rs::a_misbehaving_scheduler_cannot_lose_work`;
- `tests/simulate_ranks.rs::a_prefetch_never_fetches_an_image_its_producing_phase_has_not_written`;
- `tests/simulate_ranks.rs::every_image_is_keyed_against_its_own_extent`;
- `tests/simulate_ranks.rs::writing_past_the_declared_sidecar_bound_is_refused`;
- core `tests/phase_pricing.rs` geometry-vs-cost invariants.

## Distributed, Barrier, Zarr: Keep The Race Tests

Do not prune the tests that recently caught or could catch races:

- `tests/local_multi_node.rs::n_workers_produce_byte_identical_output_to_a_single_node_run`;
- `tests/local_multi_node.rs::every_block_was_executed_exactly_once_across_the_merged_event_stream`;
- `tests/local_multi_node.rs::a_worker_that_is_killed_mid_run_has_its_tasks_reissued_and_the_output_is_still_right`;
- `tests/local_multi_node.rs::the_work_list_stays_at_least_one_task_ahead_of_what_is_being_computed`;
- `tests/local_multi_node.rs::a_persistent_coordinator_accepts_a_job_over_http_and_outlives_it`;
- `tests/local_multi_node.rs::a_fragment_written_by_one_worker_is_read_by_another_when_the_reach_demands_it`;
- most or all of `tests/barrier_multi_node.rs`;
- barrier gate/reissue/reduction protocol tests in `tests/barrier_phase.rs`;
- `tests/zarr_env.rs::concurrent_execution_through_storage_is_still_byte_identical`;
- `tests/zarr_env.rs::decomposition_invariance_survives_storage`;
- `tests/zarr_cache.rs::concurrent_reads_through_the_cache_return_what_uncached_reads_do`;
- `tests/zarr_cache.rs::prefetching_under_concurrent_demand_reads_changes_no_answer`.

## Do Not Start Here

Small wrapper tests can look trivial but still catch seam/block regressions.
Do not start by deleting:

- `tests/expand_labels.rs`;
- `tests/object_distance_prune.rs`;
- `tests/label_size_filter.rs`;
- `tests/remove_small_holes.rs`;
- `tests/remove_small_objects.rs`;
- `tests/hysteresis_threshold.rs`;
- reference-rule tests that validate independently transcribed image-analysis
  behavior.

## Suggested Execution Order

1. Delete the two remaining source-grep tests in `tests/one_freeing_rule.rs`.
2. Delete redundant `no_two_decompositions_disagree` tests where same-reference
   checks already imply pairwise equality.
3. Remove low-risk scenario/report taxonomy tests in `tests/cost_scenarios.rs`.
4. Move benchmark/research/table tests to ignored tests, `benches/`, `examples/`,
   or `tools/`.
5. Shrink `tests/measurements.rs` by consolidating schema/builder/constructor
   smoke tests.
6. Replace exact diagnostic prose checks with structured error checks or minimal
   keyword checks.
7. Re-run default CI and compare wall time/test count before taking the next
   tranche.
