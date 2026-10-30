#!/usr/bin/env python3
"""Generate the conceptual SVG figures used by slides/blockflow.qmd.

Plain SVG, no dependencies: rerun after editing a parameter, or open the
output in Inkscape and edit by hand.

    python3 slides/figures/make_figures.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent

# One palette for every figure.
BLUE = "#4C78A8"     # input / data
ORANGE = "#F58518"   # halo / read extent
GREEN = "#54A24B"    # valid / output
RED = "#E45756"      # failure / gap
TEAL = "#72B7B2"     # cache
YELLOW = "#ECA82C"   # prefetch
GREY = "#BAB0AC"
INK = "#333333"
LIGHT = "#EEEEEE"
FONT = "font-family='Helvetica, Arial, sans-serif'"


def svg(w, h, body):
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
        f"width='{w}' height='{h}' {FONT}>\n"
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' "
        "markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
        f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def rect(x, y, w, h, fill="none", stroke=INK, sw=1, dash=None, rx=0, op=1):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' "
        f"fill-opacity='{op}' stroke='{stroke}' stroke-width='{sw}'{d}/>"
    )


def text(x, y, s, size=14, anchor="middle", fill=INK, weight="normal"):
    return (
        f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' "
        f"fill='{fill}' font-weight='{weight}'>{s}</text>"
    )


def arrow(x1, y1, x2, y2, color=INK, sw=1.5, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
        f"stroke-width='{sw}' marker-end='url(#arrow)'{d}/>"
    )


def write(name, content):
    (OUT / name).write_text(content)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------
# 1. The problem: the image is far larger than memory.
def fig_problem():
    b = []
    cols, rows, s = 12, 7, 24
    x0, y0 = 30, 50
    b.append(text(x0, 30, "Image on disk: TB-scale, chunked (OME-Zarr)", 16, "start", weight="bold"))
    for r in range(rows):
        for c in range(cols):
            fill = BLUE if (r, c) in {(2, 4), (2, 5), (3, 4), (3, 5)} else LIGHT
            b.append(rect(x0 + c * s, y0 + r * s, s, s, fill, "#999", 0.8))
    gx = x0 + cols * s
    b.append(arrow(gx + 15, y0 + 3.5 * s, gx + 85, y0 + 3.5 * s))
    b.append(text(gx + 50, y0 + 3.5 * s - 10, "stream", 13))
    # RAM box
    rx, ry = gx + 95, y0 + 20
    b.append(rect(rx, ry, 130, 128, "white", INK, 1.5, rx=6))
    b.append(text(rx + 65, ry - 8, "Memory", 15, weight="bold"))
    for i, (dx, dy) in enumerate([(15, 15), (15 + 2 * s + 4, 15), (15, 15 + 2 * s + 4)]):
        b.append(rect(rx + dx, ry + dy, 2 * s, 2 * s, BLUE if i < 2 else YELLOW, "#555", 0.8))
    b.append(text(rx + 65, ry + 150, "a few blocks at a time", 12))
    # The knobs
    kx = rx + 175
    b.append(text(kx, 30, "What decides the best order?", 16, "start", weight="bold"))
    knobs = ["memory budget", "threads / CPU cache", "GPUs, and where they are",
             "number of nodes", "read / write speed", "compression ratio + cost",
             "the ops, and their halos", "output precision"]
    for i, k in enumerate(knobs):
        y = y0 + 4 + i * 23
        b.append(rect(kx, y, 200, 19, LIGHT, "none", rx=9))
        b.append(text(kx + 10, y + 14, k, 13, "start"))
    write("problem.svg", svg(kx + 250, 240, b))


# --------------------------------------------------------------------------
# 2. Why neighbours matter: a Gaussian on one output chunk reads around it.
def fig_gaussian():
    import math
    b = []
    n, s = 5, 80            # 5x5 chunks of 80 px
    ox, oy = 40, 60
    r = 30                  # kernel radius (≈ 3 sigma), in px
    b.append(text(ox, 30, "Gaussian blur of one output chunk", 16, "start", weight="bold"))
    cx, cy = ox + 2 * s, oy + 2 * s            # output chunk top-left
    for i in range(n):
        for j in range(n):
            x, y = ox + i * s, oy + j * s
            near = abs(i - 2) <= 1 and abs(j - 2) <= 1 and (i, j) != (2, 2)
            b.append(rect(x, y, s, s, ORANGE if near else LIGHT, "#999", 1,
                          op=0.12 if near else 1))
    # read region = output chunk grown by the kernel radius on every axis
    b.append(rect(cx - r, cy - r, s + 2 * r, s + 2 * r, ORANGE, ORANGE, 2, "6,4", op=0.35))
    b.append(rect(cx, cy, s, s, GREEN, INK, 2, op=0.8))
    b.append(text(cx + s / 2, cy + s - 8, "output", 12, fill="white", weight="bold"))
    # kernel footprints: one at a corner voxel spills into 3 neighbours
    b.append("<defs><radialGradient id='gauss'>"
             f"<stop offset='0' stop-color='#1F3F66' stop-opacity='1'/>"
             f"<stop offset='0.45' stop-color='{BLUE}' stop-opacity='0.7'/>"
             f"<stop offset='1' stop-color='{BLUE}' stop-opacity='0.08'/>"
             "</radialGradient></defs>")
    for (kx, ky) in [(cx + 4, cy + 4), (cx + s - 4, cy + 26)]:
        b.append(f"<rect x='{kx - r}' y='{ky - r}' width='{2 * r}' height='{2 * r}' "
                 f"fill='url(#gauss)' stroke='#1F3F66' stroke-width='1.2' stroke-dasharray='3,2'/>")
        b.append(f"<circle cx='{kx}' cy='{ky}' r='3' fill='white' stroke='{INK}'/>")
    # reach annotation
    ay = cy + s + r + 16
    b.append(arrow(cx + s, ay, cx + s + r, ay, ORANGE, 1.5))
    b.append(text(cx + s + r / 2, ay + 16, "reach", 12, fill="#B35F00"))
    # 1-D kernel profile across the chunk edge
    px, py, pw, ph = ox + n * s + 70, oy + 40, 300, 150
    b.append(text(px, 30 + 40, "Kernel across the chunk edge", 14, "start", weight="bold"))
    mid = px + pw / 2
    sig = pw / 6
    pts = []
    for k in range(101):
        x = px + pw * k / 100
        y = py + ph - ph * 0.9 * math.exp(-((x - mid) / sig) ** 2 / 2)
        pts.append(f"{x:.1f},{y:.1f}")
    edge = mid + 25
    b.append(f"<rect x='{edge}' y='{py}' width='{px + pw - edge}' height='{ph}' "
             f"fill='{ORANGE}' fill-opacity='0.2'/>")
    b.append(f"<rect x='{px}' y='{py}' width='{edge - px}' height='{ph}' "
             f"fill='{GREEN}' fill-opacity='0.15'/>")
    b.append(f"<polyline points='{' '.join(pts)}' fill='none' stroke='{BLUE}' stroke-width='2.5'/>")
    b.append(f"<line x1='{px}' y1='{py + ph}' x2='{px + pw}' y2='{py + ph}' stroke='{INK}'/>")
    b.append(f"<line x1='{edge}' y1='{py - 10}' x2='{edge}' y2='{py + ph + 6}' stroke='{INK}' "
             "stroke-width='1.5' stroke-dasharray='5,3'/>")
    b.append(f"<line x1='{mid}' y1='{py + ph}' x2='{mid}' y2='{py + ph + 6}' stroke='{INK}'/>")
    b.append(text(mid, py + ph + 20, "output voxel", 12))
    b.append(text(edge, py - 14, "chunk edge", 12))
    b.append(text((px + edge) / 2, py + 16, "own chunk", 12, fill="#2E6B27"))
    b.append(text((edge + px + pw) / 2, py + 16, "neighbour", 12, fill="#B35F00"))
    b.append(text(px, py + ph + 52, "±3σ ⇒ reach = 3σ voxels on every axis:", 13, "start"))
    b.append(text(px, py + ph + 72, "read the chunk plus a halo from its 8 (2-D)", 13, "start"))
    b.append(text(px, py + ph + 92, "or 26 (3-D) neighbours", 13, "start"))
    write("gaussian.svg", svg(px + pw + 30, oy + n * s + 30, b))


# --------------------------------------------------------------------------
# 4. Blocks, halo and reach: valid = core ∩ trustworthy.
def halo_panel(ox, oy, halo, reach, title, ok):
    b = []
    n, s = 3, 90
    b.append(text(ox + n * s / 2, oy - 22, title, 15, weight="bold", fill=GREEN if ok else RED))
    for r in range(n):
        for c in range(n):
            b.append(rect(ox + c * s, oy + r * s, s, s, "white", "#999", 1))
    cx, cy = ox + s, oy + s
    # read extent: core grown by halo
    b.append(rect(cx - halo, cy - halo, s + 2 * halo, s + 2 * halo, ORANGE, ORANGE, 2, op=0.18))
    b.append(rect(cx - halo, cy - halo, s + 2 * halo, s + 2 * halo, "none", ORANGE, 2, "6,4"))
    # trustworthy: read extent shrunk by reach
    t = halo - reach
    # valid = core ∩ trustworthy
    v = min(t, 0)
    if ok:
        b.append(rect(cx, cy, s, s, GREEN, INK, 1.5, op=0.75))
    else:
        b.append(rect(cx, cy, s, s, RED, INK, 1.5, op=0.75))  # the gap
        b.append(rect(cx - v, cy - v, s + 2 * v, s + 2 * v, GREEN, "none", op=0.95))
        b.append(rect(cx, cy, s, s, "none", INK, 1.5))
    b.append(rect(cx - t, cy - t, s + 2 * t, s + 2 * t, "none", BLUE, 2, "2,3"))
    b.append(text(ox + n * s / 2, oy + n * s + 22,
                  f"halo {halo} {'≥' if ok else '&lt;'} reach {reach}", 13))
    return b


def fig_halo():
    b = []
    b += halo_panel(40, 50, 26, 18, "halo ≥ reach: blocks tile", True)
    b += halo_panel(400, 50, 12, 26, "halo &lt; reach: gap, tiling check fires", False)
    ly = 370
    items = [(ORANGE, "read extent = core + halo", "6,4", 0.18),
             (BLUE, "trustworthy = read − reach", "2,3", 0),
             (GREEN, "valid = core ∩ trustworthy", None, 0.75),
             (RED, "gap (no block owns it)", None, 0.75)]
    x = 20
    for color, label, dash, op in items:
        b.append(rect(x, ly - 12, 18, 14, color if op else "none", color, 2, dash, op=op or 1))
        b.append(text(x + 24, ly, label, 13, "start"))
        x += 185
    write("halo.svg", svg(760, 390, b))


# --------------------------------------------------------------------------
# 5. The task graph: one task per (block, phase).
BLOCKS, PHASES = 6, 3
CW, CH, GX, GY = 74, 40, 18, 34


def cell_xy(ox, oy, blk, ph):
    return ox + blk * (CW + GX), oy + ph * (CH + GY)


def task_grid_frame(ox, oy):
    b = []
    for blk in range(BLOCKS):
        x, _ = cell_xy(ox, oy, blk, 0)
        b.append(text(x + CW / 2, oy - 12, f"block {blk}", 12))
    for ph in range(PHASES):
        _, y = cell_xy(ox, oy, 0, ph)
        b.append(text(ox - 12, y + CH / 2 + 5, f"phase {ph}", 12, "end"))
    return b


def fig_taskgraph():
    b = []
    ox, oy = 90, 40
    b += task_grid_frame(ox, oy)
    target = (3, 2)
    cone = {target}
    for ph in range(PHASES - 1, 0, -1):
        for (bk, p) in list(cone):
            if p == ph:
                cone |= {(bk + d, ph - 1) for d in (-1, 0, 1) if 0 <= bk + d < BLOCKS}
    for ph in range(PHASES):
        for blk in range(BLOCKS):
            x, y = cell_xy(ox, oy, blk, ph)
            if (blk, ph) == target:
                fill = GREEN
            elif (blk, ph) in cone:
                fill = ORANGE
            else:
                fill = LIGHT
            b.append(rect(x, y, CW, CH, fill, "#777", 1, rx=5, op=0.85 if fill != LIGHT else 1))
            b.append(text(x + CW / 2, y + CH / 2 + 5, f"({blk},{ph})", 12))
    for (bk, p) in sorted(cone):
        if p == 0:
            continue
        x2, y2 = cell_xy(ox, oy, bk, p)
        for d in (-1, 0, 1):
            src = (bk + d, p - 1)
            if src in cone:
                x1, y1 = cell_xy(ox, oy, *src)
                b.append(arrow(x1 + CW / 2, y1 + CH, x2 + CW / 2 + d * 14, y2, INK, 1.2))
    w = ox + BLOCKS * (CW + GX)
    b.append(text(ox, oy + PHASES * (CH + GY) + 6,
                  "task (b, p) needs phase p−1 on b and its neighbours (the halo)", 13, "start"))
    write("taskgraph.svg", svg(w, oy + PHASES * (CH + GY) + 20, b))


# --------------------------------------------------------------------------
# 6. Two walks of the same graph.
def order_fused():
    done, order = set(), []
    def ready(bk, ph):
        return ph == 0 or all((bk + d, ph - 1) in done
                              for d in (-1, 0, 1) if 0 <= bk + d < BLOCKS)
    while len(done) < BLOCKS * PHASES:
        cands = [(bk, ph) for ph in range(PHASES) for bk in range(BLOCKS)
                 if (bk, ph) not in done and ready(bk, ph)]
        nxt = max(cands, key=lambda t: (t[1], -t[0]))   # deepest phase first
        done.add(nxt)
        order.append(nxt)
    return order


def order_phase_major():
    return [(bk, ph) for ph in range(PHASES) for bk in range(BLOCKS)]


def shade(i, n):
    # light -> dark blue along the order
    t = i / (n - 1)
    lo, hi = (0xDD, 0xE8, 0xF5), (0x2C, 0x4F, 0x7C)
    c = [round(a + (z - a) * t) for a, z in zip(lo, hi)]
    return "#%02X%02X%02X" % tuple(c)


def order_panel(ox, oy, order, title, note):
    b = [text(ox - 80, oy - 38, title, 15, "start", weight="bold")]
    b += task_grid_frame(ox, oy)
    n = len(order)
    for i, (bk, ph) in enumerate(order):
        x, y = cell_xy(ox, oy, bk, ph)
        b.append(rect(x, y, CW, CH, shade(i, n), "#777", 1, rx=5))
        b.append(text(x + CW / 2, y + CH / 2 + 5, str(i + 1), 14,
                      fill="white" if i > n / 2 else INK, weight="bold"))
    b.append(text(ox - 80, oy + PHASES * (CH + GY) + 4, note, 13, "start"))
    return b


def fig_orders():
    b = []
    pw = BLOCKS * (CW + GX) + 110
    b += order_panel(90, 70, order_phase_major(), "Phase-major (materialise)",
                     "whole intermediate image is written, then re-read")
    b += order_panel(90 + pw, 70, order_fused(), "Block-major (fuse)",
                     "a wavefront: few intermediates alive at once")
    write("orders.svg", svg(90 + pw + BLOCKS * (CW + GX), PHASES * (CH + GY) + 90, b))


# --------------------------------------------------------------------------
# 8. One byte budget, two classes of lease.
def fig_budget():
    b = []
    x0, y0, W, H = 30, 60, 700, 54
    b.append(text(x0, 35, "One memory budget, in bytes", 16, "start", weight="bold"))
    segs = [(BLUE, 0.42, "compute: blocks in flight", "Reserved"),
            (TEAL, 0.26, "chunk cache", "Opportunistic"),
            (YELLOW, 0.14, "prefetch", "Opportunistic"),
            ("white", 0.18, "free", "")]
    x = x0
    for color, frac, label, cls in segs:
        w = W * frac
        b.append(rect(x, y0, w, H, color, INK, 1, op=0.85))
        b.append(text(x + w / 2, y0 + H / 2 + 5, label, 13,
                      fill="white" if color == BLUE else INK))
        if cls:
            b.append(text(x + w / 2, y0 + H + 20, cls, 12, fill="#666"))
        x += w
    ny = y0 + H + 60
    b.append(text(x0, ny, "Reserved (compute)", 14, "start", weight="bold", fill=BLUE))
    b.append(text(x0, ny + 20, "may wait for room — a blocked worker is slow,", 13, "start"))
    b.append(text(x0, ny + 38, "an unadmitted one is a crash", 13, "start"))
    b.append(text(x0 + 380, ny, "Opportunistic (cache, prefetch)", 14, "start", weight="bold", fill="#3E8A85"))
    b.append(text(x0 + 380, ny + 20, "never waits; is told to release", 13, "start"))
    b.append(text(x0 + 380, ny + 38, "when compute needs the bytes", 13, "start"))
    write("budget.svg", svg(W + 60, ny + 55, b))


# --------------------------------------------------------------------------
# 9. Objects that cross block seams: fragments, then a global merge.
CELLS = [(40, 45, 22), (100, 95, 26), (160, 40, 18), (55, 150, 20),
         (150, 160, 24), (175, 105, 14), (30, 95, 12), (110, 30, 12)]
OBJ_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756",
              "#72B7B2", "#9D755D", "#ECA82C"]
PIECE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2",
                "#72B7B2", "#9D755D", "#ECA82C", "#FF9DA6", "#79706E", "#BAB0AC", "#D67195"]


def frag_panel(ox, oy, mode, uid):
    S = 200
    h = S / 2
    b = [f"<defs>"]
    for i in range(2):
        for j in range(2):
            b.append(f"<clipPath id='c{uid}{i}{j}'><rect x='{ox + i * h}' y='{oy + j * h}' "
                     f"width='{h}' height='{h}'/></clipPath>")
    b.append("</defs>")
    b.append(rect(ox, oy, S, S, "white", INK, 1))
    k = 0
    for o, (cx, cy, r) in enumerate(CELLS):
        crosses = abs(cx - h) < r or abs(cy - h) < r
        if mode == "merged":
            b.append(f"<circle cx='{ox + cx}' cy='{oy + cy}' r='{r}' fill='{OBJ_COLORS[o]}' "
                     f"fill-opacity='0.8' stroke='{INK}'/>")
            continue
        for i in range(2):
            for j in range(2):
                inside_x = (cx + r > i * h) and (cx - r < (i + 1) * h)
                inside_y = (cy + r > j * h) and (cy - r < (j + 1) * h)
                if not (inside_x and inside_y):
                    continue
                if mode == "blocks":
                    fill, op = PIECE_COLORS[k % len(PIECE_COLORS)], 0.8
                else:  # fragments: only seam-touching pieces matter
                    fill, op = (ORANGE, 0.85) if crosses else (GREY, 0.5)
                b.append(f"<circle cx='{ox + cx}' cy='{oy + cy}' r='{r}' fill='{fill}' "
                         f"fill-opacity='{op}' stroke='{INK}' clip-path='url(#c{uid}{i}{j})'/>")
                k += 1
    dash = "6,4" if mode != "merged" else "2,6"
    b.append(f"<line x1='{ox + h}' y1='{oy}' x2='{ox + h}' y2='{oy + S}' stroke='{INK}' "
             f"stroke-width='1.5' stroke-dasharray='{dash}'/>")
    b.append(f"<line x1='{ox}' y1='{oy + h}' x2='{ox + S}' y2='{oy + h}' stroke='{INK}' "
             f"stroke-width='1.5' stroke-dasharray='{dash}'/>")
    return b


def fig_fragments():
    b = []
    oy = 60
    xs = [20, 300, 580]
    titles = ["1. segment each block", "2. fragments at the seams", "3. merge → one id per object"]
    subs = ["volume → fragments", "fragments → fragments", "fragments → volume + table"]
    for idx, (x, mode) in enumerate(zip(xs, ["blocks", "frags", "merged"])):
        b += frag_panel(x, oy, mode, idx)
        b.append(text(x + 100, oy - 30, titles[idx], 15, weight="bold"))
        b.append(text(x + 100, oy - 12, subs[idx], 12, fill="#666"))
        if idx < 2:
            b.append(arrow(x + 212, oy + 100, x + 268, oy + 100, INK, 2))
    b.append(text(400, oy + 232,
                  "same blocks, same task DAG — only the payload is no longer pixels",
                  13))
    write("fragments.svg", svg(800, oy + 245, b))


if __name__ == "__main__":
    fig_problem()
    fig_gaussian()
    fig_halo()
    fig_taskgraph()
    fig_orders()
    fig_budget()
    fig_fragments()
