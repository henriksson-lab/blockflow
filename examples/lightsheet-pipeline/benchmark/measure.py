#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Run one command and measure its whole process tree.

    measure.py --out RUN.json [--interval 0.05] [--pss-interval 1.0] -- COMMAND [ARGS...]

Standard library only, Linux only. Measured, for the command and every
process it starts (worker pools, elastix, shells):

- `wall_s`: from fork to the last process of the tree being reaped;
- `user_s`, `sys_s`: `getrusage(RUSAGE_CHILDREN)` before and after. This
  wrapper makes itself a child subreaper (`PR_SET_CHILD_SUBREAPER`), so a
  grandchild orphaned by its parent is reparented here and reaped here, and
  its CPU time is counted too;
- `max_rss_single_gb`: the `ru_maxrss` of `RUSAGE_CHILDREN`, the peak of the
  largest single process of the tree (what `/usr/bin/time` reports);
- `tree_rss_peak_gb`: the peak over samples of the summed `VmRSS` of the
  tree, sampled every `--interval` seconds (default 0.05). Pages shared
  between processes (fork copy-on-write, shared memory, a file mapped by
  several workers) are counted once per process, so for worker pools this is
  an upper bound;
- `tree_anon_peak_gb`: the same over `RssAnon` (no file-backed pages: what a
  memory-mapped input adds to RSS is left out);
- `tree_pss_peak_gb`: the peak of the summed PSS (shared pages split between
  their users; `smaps_rollup`, which walks the page tables and so is sampled
  only every `--pss-interval` seconds, default 1; 0 turns it off). The
  fairest figure for a worker pool; for a single process it equals RSS up to
  the sampling;
- `processes_peak`, `threads_peak`: the largest tree seen;
- `io`: `rchar`/`wchar` (bytes through read/write syscalls, page-cache hits
  included) and `read_bytes`/`write_bytes` (what reached storage; on NFS
  this is what the client sent and fetched), summed over every process at its
  last sample, so a process's last 50 ms are missed; plus `ru_inblock` and
  `ru_oublock` (512-byte blocks) from the rusage;
- the machine while it ran: load averages before, after and sampled once a
  second (`load1_max`, `load1_mean`), and `machine_busy_cpu_s` (all CPUs'
  busy time from `/proc/stat`), from which `other_cpu_s` = busy minus this
  tree's own CPU, the other jobs' CPU during the run (`other_cores_mean`:
  the same per second of wall time).

The command's exit status is the wrapper's. The record is written as JSON to
`--out` and printed on stderr as one line.
"""
import argparse
import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import time

PAGE = os.sysconf("SC_PAGE_SIZE")
CLK = os.sysconf("SC_CLK_TCK")
PR_SET_CHILD_SUBREAPER = 36


def children_of(pid):
    """Direct children of `pid`, over all its threads."""
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/children") as f:
                    out.extend(int(c) for c in f.read().split())
            except OSError:
                pass
    except OSError:
        pass
    return out


def tree(roots):
    seen, stack = [], list(roots)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        stack.extend(children_of(pid))
    return seen


def status_kb(pid):
    """VmRSS, RssAnon (kB) and the thread count of one process."""
    rss = anon = threads = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("RssAnon:"):
                    anon = int(line.split()[1])
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
    except OSError:
        pass
    return rss, anon, threads


def pss_kb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def io_of(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                key, value = line.split(":")
                out[key] = int(value)
    except OSError:
        pass
    return out


def busy_jiffies():
    with open("/proc/stat") as f:
        fields = [int(v) for v in f.readline().split()[1:]]
    # user nice system idle iowait irq softirq steal guest guest_nice
    idle = fields[3] + fields[4]
    total = sum(fields[:8])
    return total - idle


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(v) for v in f.read().split()[:3]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--pss-interval", type=float, default=1.0)
    parser.add_argument("--kill-gb", type=float, default=0.0,
                        help="kill the tree when its summed RSS exceeds this (0: never)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command")

    libc = ctypes.CDLL(None, use_errno=True)
    subreaper = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0

    load_before = loadavg()
    busy0 = busy_jiffies()
    usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    started_wall = time.time()
    proc = subprocess.Popen(command)
    peak = {"rss": 0, "anon": 0, "pss": 0, "processes": 0, "threads": 0}
    last_io = {}
    loads = []
    next_pss = next_load = 0.0
    killed = False
    exit_status = None
    reaped_root = False

    def reap(block):
        """Reap whatever of the tree has exited; returns False once no child is left."""
        nonlocal exit_status, reaped_root
        while True:
            try:
                pid, status = os.waitpid(-1, 0 if block else os.WNOHANG)
            except ChildProcessError:
                return False
            if pid == 0:
                return True
            if pid == proc.pid:
                reaped_root = True
                exit_status = os.waitstatus_to_exitcode(status)
            if block:
                return True

    # Let SIGINT/SIGTERM reach the tree rather than kill the wrapper first.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: [os.kill(p, s) for p in tree(children_of(os.getpid()))
                                         if p != os.getpid()])

    while True:
        now = time.monotonic() - started
        pids = [p for p in tree(children_of(os.getpid())) if p != os.getpid()]
        rss = anon = threads = 0
        for pid in pids:
            r, a, t = status_kb(pid)
            rss += r
            anon += a
            threads += t
            io = io_of(pid)
            if io:
                last_io[pid] = io
        peak["rss"] = max(peak["rss"], rss)
        peak["anon"] = max(peak["anon"], anon)
        peak["processes"] = max(peak["processes"], len(pids))
        peak["threads"] = max(peak["threads"], threads)
        if args.pss_interval > 0 and now >= next_pss:
            peak["pss"] = max(peak["pss"], sum(pss_kb(p) for p in pids))
            next_pss = now + args.pss_interval
        if now >= next_load:
            loads.append(loadavg()[0])
            next_load = now + 1.0
        if args.kill_gb and rss > args.kill_gb * 2**20 and not killed:
            killed = True
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if not reap(False):
            break
        time.sleep(args.interval)
    wall = time.monotonic() - started
    usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    busy1 = busy_jiffies()
    load_after = loadavg()

    user = usage1.ru_utime - usage0.ru_utime
    system = usage1.ru_stime - usage0.ru_stime
    busy = (busy1 - busy0) / CLK
    io_sum = {k: sum(v.get(k, 0) for v in last_io.values())
              for k in ("rchar", "wchar", "read_bytes", "write_bytes")}
    record = {
        "command": command,
        "exit": exit_status if reaped_root else None,
        "killed_for_memory": killed,
        "subreaper": subreaper,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_wall)),
        "wall_s": round(wall, 3),
        "user_s": round(user, 3),
        "sys_s": round(system, 3),
        "cpu_s": round(user + system, 3),
        "max_rss_single_gb": round(usage1.ru_maxrss / 2**20, 4),
        "tree_rss_peak_gb": round(peak["rss"] / 2**20, 4),
        "tree_anon_peak_gb": round(peak["anon"] / 2**20, 4),
        "tree_pss_peak_gb": round(peak["pss"] / 2**20, 4) if args.pss_interval > 0 else None,
        "processes_peak": peak["processes"],
        "threads_peak": peak["threads"],
        "sample_interval_s": args.interval,
        "pss_interval_s": args.pss_interval,
        "io": {**io_sum, "inblock": usage1.ru_inblock - usage0.ru_inblock,
               "oublock": usage1.ru_oublock - usage0.ru_oublock},
        "load_before": load_before,
        "load_after": load_after,
        "load1_max": max(loads) if loads else load_before[0],
        "load1_mean": round(sum(loads) / len(loads), 2) if loads else load_before[0],
        "machine_busy_cpu_s": round(busy, 1),
        "other_cpu_s": round(busy - user - system, 1),
        "other_cores_mean": round((busy - user - system) / wall, 2) if wall > 0 else None,
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: record[k] for k in ("exit", "wall_s", "cpu_s", "tree_rss_peak_gb",
                                             "tree_pss_peak_gb", "max_rss_single_gb",
                                             "load_before")}), file=sys.stderr)
    code = record["exit"]
    sys.exit(code if isinstance(code, int) and code >= 0 else 1)


if __name__ == "__main__":
    main()
