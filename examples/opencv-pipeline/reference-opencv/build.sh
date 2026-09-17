#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$script_dir/build"
cmake -S "$script_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$build_dir" --config Release >/dev/null
