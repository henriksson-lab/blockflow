# Percent-Positive Example

This example models a common biology workflow: classify segmented objects as
marker-positive or marker-negative from per-object intensity.

The example deliberately keeps classification and reporting glue inside the
example package. If another workflow needs the same table-level operation, that
is the point where it should be considered for promotion into `blockflow`.

## Build

From the workspace root:

```sh
cargo build -p blockflow-percent-positive --release
```

## Run

Run the Blockflow side on deterministic synthetic fixtures:

```sh
examples/percent-positive/scripts/run_blockflow.sh \
  .tmp/percent-positive/blockflow \
  10
```

Run the Python reference:

```sh
examples/percent-positive/scripts/run_reference.sh \
  .tmp/percent-positive/reference \
  10
```

Run the comparison benchmark:

```sh
examples/percent-positive/scripts/run_benchmark.sh 10
examples/percent-positive/scripts/run_benchmark.sh 50
```

The Blockflow benchmark prepares Zarr fixture arrays first, then measures the
attached label and marker arrays directly. The measurement planner selects a
compute grid; `--chunk` controls fixture storage chunks. The Python reference generates
the same deterministic fixture in memory. Outputs live under
`.tmp/percent-positive/` and are not committed.
