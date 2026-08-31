# `costs/` — cost scenarios

Machines to plan for, as files. One is what this crate has **measured**; the
rest are plausible neighbours of it — a slower disk, less memory, two cores
instead of eight — derived from it by stated ratios.

They exist because every planner figure in this crate was taken on one machine.
A search tuned that way is tuned to a host, and nothing inside it can tell.
`tests/cost_scenarios.rs` runs the planner against every file here and records
two things: how well it chooses **on** each machine, and what a plan chosen for
one machine costs **on another**.

## What is in a file

`format`/`version`, a `name` (which is the file's stem, and what a report and a
regression bound refer to), a `note` saying where its numbers came from, and
then three groups:

| group | what it is |
|---|---|
| `terms` | the cost coefficients, keyed exactly as `statistics::Term::key` writes them. **Both judges derive from these** — the planner through `Snapshot::calibrate`, the simulator through `Rates::from_snapshot` — so a scenario cannot tell one half of the crate a different cost from the other. |
| `machine` | worker count, page-cache bytes, IO channels, contention, prefetch depth: the properties of a machine that no *cost* coefficient can express. |
| `storage` | the chunk geometry and the two `Rates` fields no coefficient covers, plus `bytes_per_voxel`, which is what turns the per-voxel terms into per-byte rates. |
| `budget_bytes` | what the planner may spend on in-flight blocks. |

## The provenance rule

**Every number is either measured or is a ratio against something measured, and
the file says which.** `measured.json` cites where each of its figures comes
from — the tile run's per-family compute, `intra-block.md` §7's memory
bandwidth, the contention fitted to 2.41x realised against forty workers — and
carries the crate's own seed, *named as a seed*, where nothing has been
measured. Every other file's note says which transform of the baseline it is.

Typing a plausible-looking absolute for a machine nobody has run on would be a
claim about a machine nobody has measured. A ratio against one that was measured
is a claim the measurement supports, and that is the only kind of claim made
here.

## Changing them

**These files are the record, not a rendering of the generator.** Nothing checks
that they still equal what `regenerate_the_scenario_files` would write, and that
is deliberate: a record which had to agree with today's code would be rewritten
by every change to that code, and a bound recorded against a machine six months
ago would quietly become a bound against this month's idea of it. What is
checked is that they parse, round-trip, and are named consistently.

So there are two ways to change one, and they are different acts:

* **edit the file**, to correct or add a machine. Legitimate. It has to parse
  and round-trip; the suite says so;
* **regenerate the family**: `cargo test --test cost_scenarios -- --ignored
  regenerate_the_scenario_files`, which writes this directory from
  `scenario::measured_baseline` and the transform list in that test. This
  *replaces* the record, so it is a decision rather than a tidy-up. Add a new
  machine shape by adding a `write(...)` there and committing what it produces.

A file that predates a field loads with that field's documented default — a
scenario written before `nodes` existed is a single-computer scenario — so an
old record stays readable rather than needing a migration.

**Keep the old files when the numbers move.** They are a record of what the
planner was held to; a regression bound is only meaningful against a scenario
that has not been quietly re-tuned to pass.

## What the sweep found, 2026-08-30

* The planner chooses the best block edge the ladder offers on **nine of the ten
  machines**. On `forty-cores` its own choice is **1.230x** the best available
  there.
* Disk speed, memory speed, chunk shape and compression all produce **the same
  plan**. The search's answer moves with the worker count and the budget and
  with nothing else here — worth knowing before anyone spends effort calibrating
  a read coefficient.
* On `less-memory` every plan chosen for another machine is **inadmissible**,
  not merely slow.
