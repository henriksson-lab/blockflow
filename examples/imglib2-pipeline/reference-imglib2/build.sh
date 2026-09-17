#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="${MAVEN_REPO_LOCAL:-$script_dir/../../../.tmp/imglib2-pipeline/m2}"
mkdir -p "$repo"
mvn -q -Dmaven.repo.local="$repo" -f "$script_dir/pom.xml" package dependency:build-classpath \
  -Dmdep.outputFile="$script_dir/target/classpath.txt"
