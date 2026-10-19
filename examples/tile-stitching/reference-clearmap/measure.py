#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Run a command under `/usr/bin/time -v` and sample its process tree's memory.

`/usr/bin/time -v` reports the peak RSS of the largest single process only;
the reference implementation runs worker pools, so the summed RSS and PSS of
the whole tree are sampled from /proc as well (PSS counts shared pages once).

Usage: measure.py STATS.json -- COMMAND [ARGS...]

`MEASURE_KILL_GB` (optional) kills the tree when its summed RSS exceeds it,
to keep a run from taking a shared machine down.
"""
import json
import os
import re
import subprocess
import sys
import time


def tree(root):
    children = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(name))
        except OSError:
            pass
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def memory_kb(pid):
    rss = pss = 0
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss = int(line.split()[1])
                elif line.startswith("Pss:"):
                    pss = int(line.split()[1])
    except OSError:
        pass
    return rss, pss


def main():
    stats_path = sys.argv[1]
    if sys.argv[2] != "--":
        sys.exit("usage: measure.py STATS.json -- COMMAND...")
    command = sys.argv[3:]
    time_log = stats_path + ".time"
    started = time.time()
    proc = subprocess.Popen(["/usr/bin/time", "-v", "-o", time_log] + command)
    peak_rss = peak_pss = 0
    kill_kb = float(os.environ.get("MEASURE_KILL_GB", "0")) * 2**20
    killed = False
    while proc.poll() is None:
        rss = pss = 0
        for pid in tree(proc.pid):
            r, p = memory_kb(pid)
            rss += r
            pss += p
        peak_rss, peak_pss = max(peak_rss, rss), max(peak_pss, pss)
        if kill_kb and rss > kill_kb and not killed:
            killed = True
            for pid in tree(proc.pid):
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
        time.sleep(0.5)
    wall = time.time() - started
    stats = {"command": command, "exit": proc.returncode, "killed_for_memory": killed, "wall_s": round(wall, 2),
             "tree_peak_rss_gb": round(peak_rss / 2**20, 3),
             "tree_peak_pss_gb": round(peak_pss / 2**20, 3)}
    try:
        text = open(time_log).read()
        for key, pattern in [("max_rss_gb", r"Maximum resident set size \(kbytes\): (\d+)"),
                             ("user_s", r"User time \(seconds\): ([\d.]+)"),
                             ("system_s", r"System time \(seconds\): ([\d.]+)")]:
            m = re.search(pattern, text)
            if m:
                value = float(m.group(1))
                stats[key] = round(value / 2**20, 3) if key == "max_rss_gb" else value
    except OSError:
        pass
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats), file=sys.stderr)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
