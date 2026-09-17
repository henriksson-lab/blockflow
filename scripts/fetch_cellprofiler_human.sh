#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-.tmp/cellprofiler-human}"
url="https://github.com/CellProfiler/examples/archive/refs/heads/master.zip"
zip="$out_dir/cellprofiler-examples-master.zip"
expanded="$out_dir/examples-master"

mkdir -p "$out_dir"
curl -L "$url" -o "$zip"
sha256sum "$zip" > "$zip.sha256"

rm -rf "$expanded"
python3 - "$zip" "$expanded" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
expanded = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as archive:
    root = None
    for name in archive.namelist():
        parts = Path(name).parts
        if parts:
            root = parts[0]
            break
    if root is None:
        raise SystemExit(f"{zip_path} is empty")
    archive.extractall(expanded.parent)
    extracted = expanded.parent / root
    if extracted != expanded:
        extracted.rename(expanded)
PY

cat > "$out_dir/README.txt" <<EOF
Downloaded CellProfiler examples archive:
$url

Archive:
$zip

Checksum:
$zip.sha256

Expanded tree:
$expanded

The HT29 human-cell example is under:
$expanded/ExampleHuman/

Keep the expanded images and generated outputs out of git unless a later test
deliberately adds a tiny fixture.
EOF
