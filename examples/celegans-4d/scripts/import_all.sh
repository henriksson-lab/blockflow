#!/usr/bin/env bash
# Import the hand-annotated Endrov lineage of every converted recording under a zarr root.
#
#   import_all.sh [ZARR_ROOT] [extra import_endrov_lineage flags...]
#
# For each <rec>.ome.zarr with a DIC/ image and an OST/rmd.ostxml that holds a <nuclineage>,
# writes DIC/{tables,labels,tracking}/endrov (see ../README.md) and appends one line to
# <ZARR_ROOT>/endrov-import-index.tsv: recording, status, counts as JSON (or the error).
# Recordings without a lineage are listed as "no-lineage". Re-running replaces earlier
# outputs (--overwrite). ESTCELL=1 also imports Endrov's model fit (lineage "estcell") as layer
# endrov-estcell -- off by default: its positions do not line up with the DIC nuclei (README).
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-/husky/otherdataset/teresa/4d_elegans/zarr}
shift || true
bin=${IMPORT_BIN:-$here/../../../target/release/import_endrov_lineage}
if [[ ! -x $bin ]]; then
  echo "build first: cargo build --release -p blockflow-celegans-4d" >&2
  exit 1
fi
index=$root/endrov-import-index.tsv
printf 'recording\tlayer\tstatus\tdetail\n' > "$index"
run() { # recording layer args...
  local rec=$1 layer=$2; shift 2
  local out
  if out=$("$bin" --image "$root/$rec/DIC" --layer "$layer" --overwrite "$@" 2>&1); then
    printf '%s\t%s\tok\t%s\n' "$rec" "$layer" "$(tr -d '\n ' <<<"$out")" >> "$index"
  elif grep -q 'no hand-annotated lineage' <<<"$out"; then
    printf '%s\t%s\tno-lineage\t%s\n' "$rec" "$layer" "$(tr '\n\t' '  ' <<<"$out")" >> "$index"
  else
    printf '%s\t%s\tfailed\t%s\n' "$rec" "$layer" "$(tr '\n\t' '  ' <<<"$out")" >> "$index"
  fi
}
for zarr in "$root"/*.ome.zarr; do
  rec=$(basename "$zarr")
  ost=$zarr/OST/rmd.ostxml
  if [[ ! -d $zarr/DIC || ! -f $ost ]]; then
    printf '%s\t-\tskipped\tno DIC/ or OST/rmd.ostxml\n' "$rec" >> "$index"
    continue
  fi
  if ! grep -q '<nuclineage' "$ost"; then
    printf '%s\t-\tno-lineage\t\n' "$rec" >> "$index"
    continue
  fi
  run "$rec" endrov "$@"
  if [[ ${ESTCELL:-0} == 1 ]] && grep -q '<nuclineage id="estcell"' "$ost"; then
    run "$rec" endrov-estcell --lineage estcell "$@"
  fi
done
column -t -s $'\t' "$index" | cut -c1-200
