#!/usr/bin/env python3
"""What a halo re-read costs in TIME, cold and warm, against what it costs in BYTES.

A volume of `planes` x `plane_bytes` is tiled along axis 0 into N blocks. Each
block reads its core grown by `halo` planes on both sides, clamped at the ends --
exactly `run_task`'s `fetch` extent. Byte amplification is arithmetic; the
question is whether time follows it.

Cold: posix_fadvise(POSIX_FADV_DONTNEED) over the whole file before each arm.
Warm: the whole file read once first, then the arm.

A halo is by construction the region a neighbouring block just read, so the warm
arm is the one that models a real run after the first block.
"""
import os, sys, time, statistics

PLANES = 512
PLANE_BYTES = 512 * 512 * 8   # 2 MiB, one f64 plane of a 512^2 slice
HALO = 5                      # gaussian_radius(1.0) + 1, vesselize's tubeness
PATH = sys.argv[1] if len(sys.argv) > 1 else "/big/henriksson/temp/halo_probe.bin"

def regions(n, halo):
    out = []
    for i in range(n):
        lo = i * PLANES // n
        hi = (i + 1) * PLANES // n
        elo = max(0, lo - halo)
        ehi = min(PLANES, hi + halo)
        out.append((elo * PLANE_BYTES, (ehi - elo) * PLANE_BYTES))
    return out

def make(path):
    if os.path.exists(path) and os.path.getsize(path) == PLANES * PLANE_BYTES:
        return
    buf = os.urandom(PLANE_BYTES)
    with open(path, "wb") as f:
        for _ in range(PLANES):
            f.write(buf)
        f.flush(); os.fsync(f.fileno())

def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

def read_all(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        while os.read(fd, 1 << 20):
            pass
    finally:
        os.close(fd)

def arm(path, n, cold):
    rs = regions(n, HALO)
    if cold:
        drop(path)
    else:
        read_all(path)
    fd = os.open(path, os.O_RDONLY)
    total = 0
    t0 = time.perf_counter()
    try:
        for off, length in rs:
            os.lseek(fd, off, os.SEEK_SET)
            left = length
            while left:
                b = os.read(fd, min(1 << 20, left))
                if not b:
                    break
                left -= len(b); total += len(b)
    finally:
        os.close(fd)
    return time.perf_counter() - t0, total

def load():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]

def avail():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return line.split()[1] + " kB"
    return "?"

if __name__ == "__main__":
    make(PATH)
    base = PLANES * PLANE_BYTES
    counts = [1, 2, 4, 8, 16, 32, 64, 128]
    reps = 3
    print(f"# volume {base/2**30:.2f} GiB, {PLANES} planes x {PLANE_BYTES/2**20:.0f} MiB, halo {HALO} planes")
    print(f"# load {load()}, MemAvailable {avail()}")
    print(f"{'blocks':>7} {'bytes x':>8} {'cold s':>9} {'cold x':>7} {'warm s':>9} {'warm x':>7} {'warm GB/s':>10}")
    cold0 = warm0 = None
    # interleave: for each rep, walk every block count, both temperatures
    cold = {n: [] for n in counts}
    warm = {n: [] for n in counts}
    byts = {}
    for _ in range(reps):
        for n in counts:
            t, b = arm(PATH, n, cold=True);  cold[n].append(t); byts[n] = b
            t, b = arm(PATH, n, cold=False); warm[n].append(t)
    for n in counts:
        c = min(cold[n]); w = min(warm[n]); b = byts[n]
        if cold0 is None:
            cold0, warm0 = c, w
        print(f"{n:>7} {b/base:>8.3f} {c:>9.4f} {c/cold0:>7.3f} {w:>9.4f} {w/warm0:>7.3f} {b/w/1e9:>10.2f}")
    print(f"# load after {load()}")
