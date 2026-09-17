#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  examples/cellprofiler-human/scripts/run_cellprofiler_reference.sh PIPELINE INPUT_DIR OUTPUT_DIR [CELLPROFILER_BIN]

Runs CellProfiler headlessly to generate the reference CSVs for the
CellProfiler-style benchmark. The pipeline must contain its own export module;
the command line only chooses the input and output directories.

Example:
  examples/cellprofiler-human/scripts/run_cellprofiler_reference.sh \
    .tmp/cellprofiler-human/examples-master/ExampleHuman/ExampleHuman.cppipe \
    .tmp/cellprofiler-human/examples-master/ExampleHuman/images \
    .tmp/cellprofiler-human/reference
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

pipeline="$1"
input_dir="$2"
output_dir="$3"
cellprofiler_bin="${4:-${CELLPROFILER_BIN:-cellprofiler}}"

if [[ ! -f "$pipeline" ]]; then
  echo "CellProfiler pipeline not found: $pipeline" >&2
  exit 1
fi

if [[ ! -d "$input_dir" ]]; then
  echo "CellProfiler input directory not found: $input_dir" >&2
  exit 1
fi

mkdir -p "$output_dir"

{
  printf 'cellprofiler_bin=%q\n' "$cellprofiler_bin"
  printf 'pipeline=%q\n' "$pipeline"
  printf 'input_dir=%q\n' "$input_dir"
  printf 'output_dir=%q\n' "$output_dir"
  printf 'command='
  printf '%q ' "$cellprofiler_bin" -c -r -p "$pipeline" -i "$input_dir" -o "$output_dir"
  printf '\n'
} > "$output_dir/reference-command.txt"

start="${EPOCHREALTIME:-}"
if [[ -z "$start" ]]; then
  start="$(date +%s)"
fi

"$cellprofiler_bin" -c -r -p "$pipeline" -i "$input_dir" -o "$output_dir"

end="${EPOCHREALTIME:-}"
if [[ -z "$end" ]]; then
  end="$(date +%s)"
fi

awk -v start="$start" -v end="$end" -v pipeline="$pipeline" -v input="$input_dir" \
  -v output="$output_dir" 'BEGIN {
    elapsed = end - start;
    printf "{\n";
    printf "\"pipeline\": \"%s\",\n", pipeline;
    printf "\"input_dir\": \"%s\",\n", input;
    printf "\"output_dir\": \"%s\",\n", output;
    printf "\"wall_seconds\": %.9f\n", elapsed;
    printf "}\n";
  }' > "$output_dir/reference-summary.json"
