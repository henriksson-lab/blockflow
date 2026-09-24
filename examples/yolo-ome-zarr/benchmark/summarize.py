#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    blockflow = []
    original = []
    for run in range(1, 4):
        text = (args.directory / f"blockflow-{run}.log").read_text()
        match = re.search(r"Ran \d+ block\(s\) in ([0-9.]+)s", text)
        if not match:
            raise SystemExit(f"missing Blockflow timing in run {run}")
        blockflow.append(float(match.group(1)))
        original.append(json.loads((args.directory / f"original-{run}.json").read_text())["seconds"])
    blockflow_median = statistics.median(blockflow)
    original_median = statistics.median(original)
    result = {
        "blockflow_seconds": blockflow,
        "original_seconds": original,
        "blockflow_median_seconds": blockflow_median,
        "original_median_seconds": original_median,
        "blockflow_over_original": blockflow_median / original_median,
        "agreement": json.loads((args.directory / "agreement.json").read_text()),
    }
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
