#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare Blockflow's links against Python `laptrack` on identical detections.

Reads one series' `tracking/<layer>/detections.csv` (and `graph.json` beside
it, for branch edges), re-links the same detections with `laptrack` using the
same squared-distance cost, gates and fixed alternative costs, and reports
where the two edge sets differ.

Frame linking only by default. Pass the same `--gap-closing`, `--max-gap`,
`--splitting`, `--merging` and `--segment-alternative-cost` as the Blockflow
run to compare segment linking too. `--alternative-cost`, `--max-distance` and
`--spacing` must also match. Exact agreement is expected only when no two
assignments tie on cost.

    python compare_laptrack.py DETECTIONS.csv --max-distance 6 \
        --alternative-cost 20 --spacing 0.65 0.65
"""

import argparse
import json
import os
import sys

import pandas as pd
from laptrack import LapTrack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("detections")
    parser.add_argument("--max-distance", type=float, required=True)
    parser.add_argument("--alternative-cost", type=float, required=True)
    parser.add_argument("--spacing", type=float, nargs=2, default=[1.0, 1.0],
                        metavar=("Y", "X"))
    parser.add_argument("--gap-closing", type=float)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--splitting", type=float)
    parser.add_argument("--merging", type=float)
    parser.add_argument("--segment-alternative-cost", type=float)
    args = parser.parse_args()

    df = pd.read_csv(args.detections)
    df = df.sort_values(["t_index", "detection_id"]).reset_index(drop=True)
    df["py"] = df["y"] * args.spacing[0]
    df["px"] = df["x"] * args.spacing[1]

    ours = set()
    first, last = {}, {}
    for track_id, track in df.sort_values("t_index").groupby("track_id"):
        ids = track["detection_id"].tolist()
        first[track_id], last[track_id] = ids[0], ids[-1]
        for a, b in zip(ids, ids[1:]):
            ours.add((a, b))
    graph_path = os.path.join(os.path.dirname(args.detections), "graph.json")
    if os.path.exists(graph_path):
        for child, parents in json.load(open(graph_path)).items():
            for parent in parents:
                ours.add((last[parent], first[int(child)]))
    segment = args.gap_closing or args.splitting or args.merging
    seg_cost = args.segment_alternative_cost

    frames = sorted(df["t_index"].unique())
    lt = LapTrack(
        metric="sqeuclidean",
        cutoff=args.max_distance ** 2,
        gap_closing_cutoff=args.gap_closing ** 2 if args.gap_closing else False,
        gap_closing_max_frame_count=args.max_gap,
        splitting_cutoff=args.splitting ** 2 if args.splitting else False,
        merging_cutoff=args.merging ** 2 if args.merging else False,
        track_start_cost=args.alternative_cost,
        track_end_cost=args.alternative_cost,
        segment_start_cost=seg_cost,
        segment_end_cost=seg_cost,
        no_splitting_cost=seg_cost,
        no_merging_cost=seg_cost,
    )
    # laptrack frames are positions in a dense list; only consecutive acquired
    # indices may link, so each run of consecutive indices is linked alone.
    if segment and len(frames) != frames[-1] - frames[0] + 1:
        sys.exit("segment comparison needs a timeline without missing frames")
    theirs = set()
    run = [frames[0]]
    runs = []
    for t in frames[1:]:
        if t == run[-1] + 1:
            run.append(t)
        else:
            runs.append(run)
            run = [t]
    runs.append(run)
    for run in runs:
        coords, ids = [], []
        for t in run:
            sub = df[df["t_index"] == t]
            coords.append(sub[["py", "px"]].to_numpy())
            ids.append(sub["detection_id"].tolist())
        if len(run) < 2:
            continue
        graph = lt.predict(coords)
        for (fa, ia), (fb, ib) in graph.edges():
            if fa > fb:
                (fa, ia), (fb, ib) = (fb, ib), (fa, ia)
            theirs.add((ids[fa][ia], ids[fb][ib]))

    report = {
        "blockflow_links": len(ours),
        "laptrack_links": len(theirs),
        "agree": len(ours & theirs),
        "only_blockflow": sorted(ours - theirs)[:20],
        "only_laptrack": sorted(theirs - ours)[:20],
        "identical": ours == theirs,
    }
    print(json.dumps(report, indent=2, default=int))
    return 0 if ours == theirs else 1


if __name__ == "__main__":
    sys.exit(main())
