# SPDX-License-Identifier: MIT
#
# Original work for this crate. Not translated from ClearMap.
#
# Animate a block-ops execution log, so a schedule can be *seen*.
#
# This script ships with the `blockflow` crate — it lives in the crate's own
# `tools/`, and `blockflow::animate` invokes it. It used to live in the
# repository that *consumes* that crate, which pointed the dependency the wrong
# way round: a crate cannot sensibly ask its own consumer for a script.
#
# The input is the JSON written by the crate's `export.rs` (schema
# `clearmap-rs.block_ops.order_log`, version 1 — the name is kept stable across
# the crate extraction, because it is a wire format and consumers key on it);
# that module's header is the contract and this script is one of its consumers.
#
# TWO VIEWS, FOR TWO DIFFERENT QUESTIONS
# --------------------------------------
# `--view 2d` (the default) draws the block grid flat, one square per cell, and
# accepts **two** logs side by side. Produce the pair with:
#
#     cd /home/mahogny/github/claude/blockflow
#     cargo run --release --bin block_progress_export -- OUTDIR [blocks-per-side]
#
# which writes `block_major.json` and `phase_major.json` for the same workflow
# under the two schedules. Why it is worth having: depth-first fusion and
# breadth-first phase-major execution compute exactly the same answer and
# produce completely different pictures. Block-major drives a few blocks all the
# way through the chain while most of the grid is still untouched; phase-major
# sweeps the whole grid once per phase. Two panels that differ are the evidence
# that this animation follows the schedule rather than sweeping generically.
#
# `--view 3d` draws **one** log as what a decomposition actually is: a lattice
# of blocks in three dimensions, with the camera orbiting it. Produce a log
# split on all three axes with:
#
#     cargo run --release --bin block_animate -- --grid 5,5,5 --out run.json
#
# It is for presentation and intuition — the shape of a schedule at a glance.
# For diagnosis, a live view polling the executor beats any recording.
#
# WHAT A SOLID LATTICE HIDES, AND WHAT IS DONE ABOUT IT
# -----------------------------------------------------
# One opaque cube per block shows only the outer shell: on a 6x6x6 lattice, 64
# of the 216 blocks are invisible, and with a schedule that drives a few blocks
# deep before starting the rest, the interesting ones are usually inside.
#
# So a cube's *opacity* carries its state, and the only solid cubes are the ones
# being worked on right now:
#
#   * untouched — no fill, stroke at 0.12: the lattice's extent as a scaffold
#     and nothing more;
#   * touched within the last few steps — opaque, full colour. This is the
#     working front, and a front is a surface, never a shell;
#   * reached but idle — 0.30. Still legible as a colour, see-through;
#   * finished — 0.14. The completed volume becomes a haze the front is plainly
#     visible through, rather than a wall.
#
# *Recency* rather than merely "unfinished" is what makes this hold in the
# middle of a run: a schedule that drives blocks partway and leaves them there
# would otherwise build an opaque mass of half-done blocks, which is the wall
# again by another name.
#
# The alternatives were a cutaway plane (throws away half the data and needs an
# axis chosen for it) and exploding the lattice (helps until the interior fills
# up, and then does not). Fading costs one number per cube and keeps every block
# on screen for the whole run.
#
# WHICH PYTHON
# ------------
# Use the **base** conda env (`python3`, /home/mahogny/miniconda3/bin/python),
# which has manim 0.20.1. Do *not* use `clearmap-gt2` — that is the graph-tool
# environment for ClearMap parity work and has no manim. Picking the wrong one
# is the easy mistake here, hence this paragraph.
#
#     python3 tools/animate_block_progress.py tools/sample_block_progress.json
#
# If manim is missing:  conda install -c conda-forge manim   (or pip install manim)
#
# USAGE
# -----
#     python3 tools/animate_block_progress.py LOG.json [-o NAME] [-q l|m|h]
#     python3 tools/animate_block_progress.py LOG.json --check      # no manim,
#                                                                   # ASCII frames
#     python3 tools/animate_block_progress.py A.json B.json -o both # side by side
#     python3 tools/animate_block_progress.py LOG.json --view 3d    # orbiting
#
# In `--view 2d` the block grid is collapsed along its thinnest axis (or
# `--axes I,J` to choose), so what you see is one square per grid cell. That is
# the right picture for a decomposition split on two axes, and the only one that
# fits two schedules on screen at once.
#
# COST, AND WHAT TO DO ABOUT A BIG LOG
# ------------------------------------
# One rendered frame per *step*, and a step is a batch of events. The batch
# size is chosen automatically so the animation is at most `--max-steps` steps
# (default 200, or 40 in three dimensions) however many events the log holds —
# a 60 000-task log animates in the same wall time as a 36-block one, at coarser
# granularity. `--max-steps` raises or lowers the ceiling; `--seconds-per-step`
# sets the pace.
#
# Three dimensions is the expensive one, and the cost is per cube per *frame*:
# every frame is distinct because the camera is moving, so none of manim's
# caching applies. Measured here, low quality, 15 fps, one 40-core machine:
#
#     lattice     blocks   steps   video   wall clock
#     3 x 3 x 3       27      38   8.2 s      2.6 min
#     5 x 5 x 5      125      50  10.6 s     12.5 min
#
# which is about **0.12 s per block per step** and near enough linear in the
# product. Two consequences worth knowing before starting one:
#
#   * `--max-steps` is the lever, not the lattice. Halving the steps halves the
#     wall clock and costs only granularity, because events are batched to fit.
#   * `BLOCK_CEILING` below refuses a lattice big enough to become an accidental
#     overnight render. It is a round number rather than a characterised curve —
#     it exists to stop an accident, not to predict a runtime. `--allow-large`
#     overrides it if you have the evening.

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

def vheight(mobject) -> float:
    """Visible height. See the note in `construct`: `Mobject.height` measures
    from the origin in manim 0.20 and is wrong for anything that has moved."""
    return float(mobject.get_top()[1] - mobject.get_bottom()[1]) or 1e-6


def vwidth(mobject) -> float:
    """Visible width, for the same reason."""
    return float(mobject.get_right()[0] - mobject.get_left()[0]) or 1e-6


# --------------------------------------------------------------- the log --

SCHEMA = "clearmap-rs.block_ops.order_log"
SUPPORTED_VERSION = 1

# Above this many blocks, `--view 3d` refuses to start. Not a limit of the
# format or of the exporter — a log of any size exports fine and animates fine
# flat — but of what manim's three-dimensional renderer finishes in a sitting:
# at the measured 0.12 s per block per step, 512 blocks is most of an hour at
# the default 40 steps, and twice that is an evening. It is a round number (an
# 8 x 8 x 8 lattice), not a threshold anything changes at.
# `blockflow::animate::VOLUME3D_BLOCK_CEILING` carries the same number and a
# Rust test asserts the two agree; change both or neither.
BLOCK_CEILING = 512

# What a block can be showing. Ordered: a later state never reverts to an
# earlier one within a phase.
IDLE = "idle"
ADMITTED = "admitted"
READ = "read"
APPLIED = "applied"
WRITTEN = "written"

PALETTE = {
    IDLE: "#23262b",
    ADMITTED: "#3b4350",
    READ: "#3f6ea8",
    WRITTEN: "#4e9a63",
    "done": "#9be564",
    "grid": "#5a6068",
    "text": "#d8dde3",
    "muted": "#8b939c",
}
# One colour per chain slot, in chain order. Warm ramp, so "further through the
# chain" reads as "hotter" without needing the legend.
OP_COLOURS = ["#f2c14e", "#f28a30", "#e8563f", "#c1345b", "#8f2d78"]


@dataclass
class BlockState:
    """What one block is showing, and how it got there."""

    phase: int = 0
    slot: int | None = None
    kind: str = IDLE
    ops_done: int = 0

    def colour(self, n_slots: int) -> str:
        """Colour by **how far through the chain** the block is, not by what
        just happened to it.

        This is the choice that makes the picture mean something. Colouring by
        the last transition would paint every block that had just been written
        the same, whether it had finished one op or all of them, and
        block-major and phase-major would then look nearly identical — both
        end up writing every block. Colouring by cumulative progress makes
        block-major a few blocks racing to the end of the ramp while the rest
        of the grid is still dark, and phase-major the whole grid stepping
        through the ramp together.
        """
        if self.ops_done >= n_slots and self.kind == WRITTEN:
            return PALETTE["done"]
        if self.ops_done > 0:
            return OP_COLOURS[(self.ops_done - 1) % len(OP_COLOURS)]
        return PALETTE.get(self.kind, PALETTE[IDLE])


def look_3d(state: BlockState, n_slots: int, active: bool) -> tuple[str, float, float]:
    """`(colour, fill opacity, stroke opacity)` for one cube.

    The colour is the same function of progress the flat view uses, so the two
    views never disagree about what a block is doing. What is added here is the
    opacity, and it is doing the work described in the header: **only what is
    being worked on right now is solid**, so the lattice is never a closed
    shell, whatever fraction of it has been reached.

    `active` — touched within the last few steps — rather than merely "not
    finished" is what makes that hold in the middle of a run. A schedule that
    drives blocks partway and leaves them there would otherwise accumulate an
    opaque mass of half-done blocks, which is exactly the wall this is meant to
    avoid.
    """
    colour = state.colour(n_slots)
    if state.ops_done >= n_slots and state.kind == WRITTEN:
        return colour, 0.14, 0.08
    if state.ops_done == 0 and state.kind == IDLE:
        return colour, 0.0, 0.12
    if active:
        return colour, 0.95, 0.55
    return colour, 0.3, 0.25


@dataclass
class Log:
    """A parsed order log, reduced to what the animation needs."""

    path: Path
    strategy: str
    volume: list[int]
    grid: list[int]
    phases: int
    ops: list[str]
    events: list[dict]
    blocks: list[dict]
    axes: tuple[int, int] = field(default=(1, 2))

    @property
    def rows(self) -> int:
        return self.grid[self.axes[0]]

    @property
    def cols(self) -> int:
        return self.grid[self.axes[1]]

    def cell(self, index: list[int]) -> tuple[int, int]:
        """Block index -> (row, column) in the 2-D projection."""
        return index[self.axes[0]], index[self.axes[1]]


def load(path: str | Path, axes: tuple[int, int] | None = None) -> Log:
    path = Path(path)
    with open(path) as handle:
        document = json.load(handle)

    schema = document.get("schema")
    if schema != SCHEMA:
        raise SystemExit(
            f"{path}: schema is {schema!r}, expected {SCHEMA!r}. This script reads "
            f"the order log written by block_ops::export, not an arbitrary JSON file."
        )
    version = document.get("version")
    if version != SUPPORTED_VERSION:
        raise SystemExit(
            f"{path}: schema version {version}, this script understands "
            f"{SUPPORTED_VERSION}. The exporter's module header says what changed."
        )

    grid = document["grid"]
    if axes is None:
        # Collapse the axis with the fewest blocks: for a volume split along two
        # axes that is the un-split one, and the projection loses nothing.
        thinnest = min(range(3), key=lambda a: (grid[a], a))
        axes = tuple(a for a in range(3) if a != thinnest)  # type: ignore[assignment]

    return Log(
        path=path,
        strategy=document.get("strategy", path.stem),
        volume=document["volume"],
        grid=grid,
        phases=document["phases"],
        ops=[op["name"] for op in document.get("ops", [])],
        events=document["events"],
        blocks=document.get("blocks", []),
        axes=axes,  # type: ignore[arg-type]
    )


def apply_event(states: dict[tuple[int, ...], BlockState], event: dict) -> int | None:
    """Fold one event into the per-block state. Returns the phase it belongs to.

    Mirrors `LatestOpPerChunk` in Rust deliberately: the same fold, so a frame
    of this animation is what a live poll of the running executor would have
    shown at that moment.

    An event type this does not know is ignored rather than rejected. That is
    what lets the exporter add event types — cache and prefetch events, say —
    without a schema version bump breaking every drawing of every old log: the
    fields this reads are the ones the schema promises, and the rest is somebody
    else's business.
    """
    kind = event["type"]
    phase = event.get("phase")
    index = event.get("index")
    if index is None:
        return phase
    key = tuple(index)
    state = states.setdefault(key, BlockState())

    if kind == "task_admitted":
        state.phase, state.kind = phase, ADMITTED
    elif kind == "block_read":
        state.phase, state.kind = phase, READ
    elif kind == "op_applied":
        state.phase, state.kind, state.slot = phase, APPLIED, event["slot"]
        state.ops_done = max(state.ops_done, event["slot"] + 1)
    elif kind == "block_short_circuited":
        # The block holds what computing would have produced, so it counts as
        # having reached the last skipped slot. Same rule as the Rust listener.
        slots = event.get("slots") or []
        if slots:
            state.slot = slots[-1]
            state.ops_done = max(state.ops_done, slots[-1] + 1)
        state.phase, state.kind = phase, APPLIED
    elif kind == "block_written":
        state.phase, state.kind = phase, WRITTEN
    return phase


def step_size(n_events: int, max_steps: int) -> int:
    """Events per rendered step, so a huge log animates in bounded time."""
    return max(1, -(-n_events // max(1, max_steps)))


def summarise(log: Log) -> str:
    counts: dict[str, int] = {}
    for event in log.events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    parts = ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    return (
        f"{log.path.name}: strategy {log.strategy}, volume {log.volume}, "
        f"grid {log.grid}, {log.phases} phases, {len(log.events)} events\n  {parts}"
    )


# ------------------------------------------------------- ASCII check mode --

ASCII_STATE = {IDLE: ".", ADMITTED: ":", READ: "r"}


def ascii_frames(log: Log, max_steps: int, out=sys.stdout) -> None:
    """Render to text. Needs no manim, and is how the data path is checked."""
    states: dict[tuple[int, ...], BlockState] = {}
    n_slots = len(log.ops) or 1
    every = step_size(len(log.events), max_steps)
    print(summarise(log), file=out)
    print(
        f"  projecting axes {log.axes} -> {log.rows} x {log.cols} grid, "
        f"{every} events per step",
        file=out,
    )
    phase = 0
    for start in range(0, len(log.events), every):
        for event in log.events[start : start + every]:
            at = apply_event(states, event)
            if at is not None:
                phase = max(phase, at)
        print(f"\n-- events {start}..{start + every} | phase {phase} --", file=out)
        for row in range(log.rows):
            line = []
            for col in range(log.cols):
                found = [
                    state
                    for index, state in states.items()
                    if log.cell(list(index)) == (row, col)
                ]
                if not found:
                    line.append(ASCII_STATE[IDLE])
                    continue
                state = max(found, key=lambda s: (s.phase, s.ops_done))
                # Same rule as the colours: how far through the chain, not
                # what just happened. `#` is "every op applied and written".
                if state.ops_done >= n_slots and state.kind == WRITTEN:
                    line.append("#")
                elif state.ops_done > 0:
                    line.append(str(state.ops_done))
                else:
                    line.append(ASCII_STATE.get(state.kind, "?"))
            print("   " + " ".join(line), file=out)


# ------------------------------------------------------------- the scene --


def build_scene(logs: list[Log], max_steps: int, seconds_per_step: float):
    """Construct the manim Scene class. Imported lazily so `--check` works
    without manim installed."""
    from manim import (  # noqa: PLC0415
        DOWN,
        LEFT,
        RIGHT,
        UP,
        Rectangle,
        Scene,
        Square,
        Text,
        VGroup,
    )

    class BlockProgress(Scene):
        def construct(self) -> None:
            self.camera.background_color = "#15171a"
            panels = []
            for log in logs:
                panels.append(self.build_panel(log))

            board = VGroup(*[panel["group"] for panel in panels]).arrange(
                RIGHT, buff=0.9
            )
            n_slots = max(len(log.ops) for log in logs) or 1

            title = Text(
                "block-ops execution: "
                + " vs ".join(log.strategy for log in logs),
                font_size=26,
                color=PALETTE["text"],
            )
            legend = self.build_legend(logs[0])
            layout = VGroup(title, board, legend).arrange(DOWN, buff=0.35)
            # `Mobject.height` / `.width` are not reliable here — manim 0.20
            # reports a `Text`'s extent measured from the origin once it has
            # been moved, so a `Text` sitting 2 units below centre claims to be
            # 2 units tall. Everything below measures the visible box instead.
            layout.scale(min(7.2 / vheight(layout), 13.6 / vwidth(layout)))
            self.add(layout)

            # Where each caption lives, fixed once the layout is settled. The
            # caption is rebuilt as text changes, and a rebuilt `Text` knows
            # nothing about the scaling the layout applied, so its target size
            # and position are recorded here rather than re-derived.
            for panel in panels:
                panel["caption_box"] = (
                    panel["caption"].get_center(),
                    vheight(panel["caption"]),
                )

            # Fold the logs forward in lockstep, one batch of events per frame.
            every = max(step_size(len(log.events), max_steps) for log in logs)
            longest = max(len(log.events) for log in logs)
            states: list[dict[tuple[int, ...], BlockState]] = [{} for _ in logs]
            for start in range(0, longest, every):
                for panel, log, state in zip(panels, logs, states):
                    phase = panel["phase"]
                    for event in log.events[start : start + every]:
                        at = apply_event(state, event)
                        if at is not None:
                            phase = max(phase, at)
                    panel["phase"] = phase
                    for index, block in state.items():
                        square = panel["squares"].get(log.cell(list(index)))
                        if square is not None:
                            square.set_fill(block.colour(n_slots), opacity=1.0)
                    # Rendering a `Text` is by far the most expensive thing per
                    # frame, so the caption is rebuilt only when its content
                    # actually changes — progress is quantised to 5 %.
                    seen = min(start + every, len(log.events))
                    percent = 5 * round(100 * seen / max(len(log.events), 1) / 5)
                    wanted = f"phase {phase} of {log.phases}    {percent:>3d} %"
                    if wanted != panel["caption_text"]:
                        panel["caption_text"] = wanted
                        centre, height = panel["caption_box"]
                        replacement = Text(
                            wanted, font_size=16, color=PALETTE["muted"]
                        )
                        replacement.scale(height / vheight(replacement))
                        replacement.move_to(centre)
                        panel["caption"].become(replacement)
                    # The phase boundary, made visible: the frame around the
                    # grid takes the current phase's colour.
                    panel["frame"].set_stroke(
                        OP_COLOURS[phase % len(OP_COLOURS)], width=3
                    )
                self.wait(seconds_per_step)
            self.wait(1.0)

        # -- panel construction --

        def build_panel(self, log: Log) -> dict:
            cell = min(4.2 / max(log.rows, 1), 4.2 / max(log.cols, 1), 0.55)
            squares: dict[tuple[int, int], Square] = {}
            rows = []
            for row in range(log.rows):
                row_group = []
                for col in range(log.cols):
                    square = Square(side_length=cell)
                    square.set_fill(PALETTE[IDLE], opacity=1.0)
                    square.set_stroke(PALETTE["grid"], width=1)
                    squares[(row, col)] = square
                    row_group.append(square)
                rows.append(VGroup(*row_group).arrange(RIGHT, buff=0.04))
            grid = VGroup(*rows).arrange(DOWN, buff=0.04)

            frame = Rectangle(
                width=grid.width + 0.25,
                height=grid.height + 0.25,
                stroke_color=PALETTE["grid"],
                stroke_width=3,
            ).move_to(grid)

            heading = Text(log.strategy, font_size=22, color=PALETTE["text"])
            # A placeholder of the final shape, so `arrange` reserves the right
            # space and the caption box recorded after layout is the real one.
            caption = Text(
                f"phase 0 of {log.phases}      0 %",
                font_size=16,
                color=PALETTE["muted"],
            )
            group = VGroup(heading, VGroup(grid, frame), caption).arrange(
                DOWN, buff=0.22
            )
            return {
                "group": group,
                "squares": squares,
                "frame": frame,
                "caption": caption,
                "caption_text": None,
                "phase": 0,
            }

        def build_legend(self, log: Log):
            from manim import RIGHT, Square, Text, VGroup  # noqa: PLC0415

            entries = []
            for name, colour in [
                ("idle", PALETTE[IDLE]),
                ("admitted", PALETTE[ADMITTED]),
                ("read", PALETTE[READ]),
            ]:
                entries.append((name, colour))
            for slot, op in enumerate(log.ops):
                entries.append((f"{slot}:{op}", OP_COLOURS[slot % len(OP_COLOURS)]))
            entries.append(("written", PALETTE[WRITTEN]))
            entries.append(("done", PALETTE["done"]))

            items = []
            for name, colour in entries:
                swatch = Square(side_length=0.18)
                swatch.set_fill(colour, opacity=1.0)
                swatch.set_stroke(PALETTE["grid"], width=1)
                label = Text(name, font_size=14, color=PALETTE["muted"])
                items.append(VGroup(swatch, label).arrange(RIGHT, buff=0.12))
            return VGroup(*items).arrange(RIGHT, buff=0.4)

    return BlockProgress


# ------------------------------------------------------ the 3-D scene --

# How many steps a block stays solid after it was last touched. Small on
# purpose: the bright set is the working front, and a front that is a few steps
# thick reads as motion while a front fifty steps thick reads as a wall.
RECENT_STEPS = 2


def block_indices(log: Log) -> list[tuple[int, int, int]]:
    """Every block the run touched, from the log's own block table.

    Taken from the table rather than from `grid`, because `grid` is the maximum
    index plus one and a run that skipped a block would otherwise be drawn with
    a cube that never does anything. Falls back to the full lattice for a log
    with no table.
    """
    listed = [tuple(entry["index"]) for entry in log.blocks]
    if listed:
        return listed
    ranges = [range(n) for n in log.grid]
    return [
        (i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]
    ]


def build_scene_3d(
    log: Log, max_steps: int, seconds_per_step: float, rotation_rate: float
):
    """Construct the manim `ThreeDScene`. Imported lazily, like the flat one."""
    import numpy as np  # noqa: PLC0415
    from manim import (  # noqa: PLC0415
        DEGREES,
        DOWN,
        UL,
        Cube,
        Text,
        ThreeDScene,
        VGroup,
    )

    indices = block_indices(log)
    grid = log.grid
    # The lattice spans about 4.6 units whatever its shape, so a 3-cube and a
    # 12-cube lattice are framed the same and the camera settings below need no
    # per-log tuning. 0.82 of the pitch leaves a visible seam between cubes.
    pitch = 4.6 / max(max(grid), 1)
    side = pitch * 0.82
    n_slots = len(log.ops) or 1

    def position(index):
        """Block index -> a point in manim's frame.

        Volume axis 2 goes right, axis 1 goes down the screen and axis 0 goes
        into the depth, which puts a slice of the lattice on screen the same way
        round as the flat view draws it. Centred on the origin so the camera
        orbits the middle of the lattice rather than its corner.
        """
        offsets = [(index[a] - (grid[a] - 1) / 2) * pitch for a in range(3)]
        return np.array([offsets[2], -offsets[1], -offsets[0]])

    class BlockFlight(ThreeDScene):
        def construct(self) -> None:
            self.camera.background_color = "#15171a"

            cubes: dict[tuple[int, ...], Cube] = {}
            lattice = VGroup()
            for index in indices:
                cube = Cube(side_length=side)
                cube.set_fill(PALETTE[IDLE], opacity=0.0)
                cube.set_stroke(PALETTE["grid"], width=1, opacity=0.12)
                cube.move_to(position(index))
                cubes[tuple(index)] = cube
                lattice.add(cube)
            self.add(lattice)

            self.set_camera_orientation(phi=68 * DEGREES, theta=-52 * DEGREES, zoom=0.95)

            # Labels belong to the viewer, not to the lattice, so they are fixed
            # in the frame and do not tumble with the camera.
            title = Text(
                f"{log.strategy}    {grid[0]}x{grid[1]}x{grid[2]} blocks",
                font_size=24,
                color=PALETTE["text"],
            )
            caption = Text(
                f"phase 0 of {log.phases}      0 %",
                font_size=18,
                color=PALETTE["muted"],
            )
            heading = VGroup(title, caption).arrange(DOWN, buff=0.15).to_corner(UL)
            legend = self.build_legend(log)
            legend.scale(0.7).to_edge(DOWN, buff=0.25)
            self.add_fixed_in_frame_mobjects(heading, legend)

            caption_centre = caption.get_center()
            caption_height = vheight(caption)
            caption_text = None

            self.begin_ambient_camera_rotation(rate=rotation_rate)

            states: dict[tuple[int, ...], BlockState] = {}
            shown: dict[tuple[int, ...], tuple[str, float, float]] = {}
            touched: dict[tuple[int, ...], int] = {}
            every = step_size(len(log.events), max_steps)
            phase = 0
            for step, start in enumerate(range(0, len(log.events), every)):
                for event in log.events[start : start + every]:
                    at = apply_event(states, event)
                    if at is not None:
                        phase = max(phase, at)
                    index = event.get("index")
                    if index is not None:
                        touched[tuple(index)] = step
                for index, state in states.items():
                    cube = cubes.get(index)
                    if cube is None:
                        continue
                    # Restyling a cube costs six faces; most cubes are unchanged
                    # in most steps, so compare first and touch only the few
                    # that moved.
                    active = step - touched.get(index, -RECENT_STEPS - 1) <= RECENT_STEPS
                    look = look_3d(state, n_slots, active)
                    if shown.get(index) == look:
                        continue
                    shown[index] = look
                    colour, fill_opacity, stroke_opacity = look
                    cube.set_fill(colour, opacity=fill_opacity)
                    cube.set_stroke(
                        PALETTE["grid"], width=1, opacity=stroke_opacity
                    )
                seen = min(start + every, len(log.events))
                percent = 5 * round(100 * seen / max(len(log.events), 1) / 5)
                wanted = f"phase {phase} of {log.phases}    {percent:>3d} %"
                if wanted != caption_text:
                    caption_text = wanted
                    replacement = Text(wanted, font_size=18, color=PALETTE["muted"])
                    replacement.scale(caption_height / vheight(replacement))
                    replacement.move_to(caption_centre)
                    # Replaced rather than `become`d, which is what the flat
                    # scene does: a fixed-in-frame mobject is registered with
                    # the camera by its *glyphs*, and mutating it in place
                    # leaves the camera holding the glyphs it was given. The
                    # symptom is a caption frozen at its first value.
                    self.remove_fixed_in_frame_mobjects(caption)
                    self.remove(caption)
                    caption = replacement
                    self.add_fixed_in_frame_mobjects(caption)
                self.wait(seconds_per_step)

            self.stop_ambient_camera_rotation()
            self.wait(0.6)

        def build_legend(self, log: Log):
            from manim import RIGHT, Square, Text, VGroup  # noqa: PLC0415

            entries = [("idle", PALETTE[IDLE]), ("in flight", PALETTE[READ])]
            for slot, op in enumerate(log.ops):
                entries.append((f"{slot}:{op}", OP_COLOURS[slot % len(OP_COLOURS)]))
            entries.append(("done", PALETTE["done"]))
            items = []
            for name, colour in entries:
                swatch = Square(side_length=0.18)
                swatch.set_fill(colour, opacity=1.0)
                swatch.set_stroke(PALETTE["grid"], width=1)
                label = Text(name, font_size=14, color=PALETTE["muted"])
                items.append(VGroup(swatch, label).arrange(RIGHT, buff=0.12))
            return VGroup(*items).arrange(RIGHT, buff=0.4)

    return BlockFlight


MISSING_MANIM = (
    "manim is not installed in this interpreter ({executable}).\n"
    "Use the base conda env (python3 / /home/mahogny/miniconda3/bin/python), "
    "which has manim 0.20.1, or install with:\n"
    "    conda install -c conda-forge manim\n"
    "Meanwhile `--check` renders the same data as text and needs nothing, and "
    "exporting a log needs no Python at all."
)


def have_manim() -> bool:
    try:
        import manim  # noqa: F401,PLC0415
    except Exception:
        return False
    return True


def render(
    logs: list[Log],
    out_name: str,
    quality: str,
    max_steps: int,
    spf: float,
    view: str = "2d",
    fps: int | None = None,
    media_dir: str | None = None,
    rotation_rate: float = 0.25,
):
    try:
        from manim import tempconfig  # noqa: PLC0415
    except ImportError:
        raise SystemExit(MISSING_MANIM.format(executable=sys.executable))

    if view == "3d":
        scene_class = build_scene_3d(logs[0], max_steps, spf, rotation_rate)
    else:
        scene_class = build_scene(logs, max_steps, spf)
    quality_name = {
        "l": "low_quality",
        "m": "medium_quality",
        "h": "high_quality",
    }[quality]
    settings = {
        "quality": quality_name,
        "output_file": out_name,
        "media_dir": media_dir or os.environ.get("BLOCK_PROGRESS_MEDIA", "media"),
        "verbosity": "WARNING",
    }
    if fps:
        settings["frame_rate"] = fps
    with tempconfig(settings):
        scene = scene_class()
        scene.render()
        # The writer's path is the whole path; `config.output_file` is only the
        # name it was asked for, and a caller downstream of this script needs
        # something it can open.
        return scene.renderer.file_writer.movie_file_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate a block_ops order log (schema "
        f"{SCHEMA} v{SUPPORTED_VERSION})."
    )
    parser.add_argument("logs", nargs="*", help="order log JSON files (1 or 2)")
    parser.add_argument("-o", "--output", default="block_progress", help="output name")
    parser.add_argument("-q", "--quality", default="l", choices=["l", "m", "h"])
    parser.add_argument(
        "--view",
        default="2d",
        choices=["2d", "3d"],
        help="2d: the grid flat, one or two logs side by side (default). "
        "3d: one log as an orbiting lattice of cubes.",
    )
    parser.add_argument(
        "--axes",
        default=None,
        help="2d only: two axes to project onto, e.g. 1,2. Default: drop the thinnest.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="ceiling on rendered steps; events are batched to fit "
        "(default 200 flat, 40 in three dimensions)",
    )
    parser.add_argument(
        "--seconds-per-step", type=float, default=None, help="pace (default 0.12 s)"
    )
    parser.add_argument("--fps", type=int, default=None, help="frames per second")
    parser.add_argument(
        "--media-dir", default=None, help="where manim writes (default ./media)"
    )
    parser.add_argument(
        "--rotation-rate",
        type=float,
        default=0.25,
        help="3d only: camera orbit, radians per second (default 0.25)",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help=f"3d only: render more than {BLOCK_CEILING} blocks anyway",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print ASCII frames instead of rendering; needs no manim",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="exit 0 if this interpreter can render, non-zero otherwise; "
        "draws nothing. What blockflow::animate::renderer_available asks.",
    )
    args = parser.parse_args()

    if args.probe:
        if have_manim():
            print("manim is available")
            return
        raise SystemExit(MISSING_MANIM.format(executable=sys.executable))
    if not args.logs:
        parser.error("a log to draw is required")

    if args.max_steps is None:
        args.max_steps = 40 if args.view == "3d" else 200
    if args.seconds_per_step is None:
        args.seconds_per_step = 0.15 if args.view == "3d" else 0.12

    axes = None
    if args.axes:
        first, second = (int(part) for part in args.axes.split(","))
        axes = (first, second)

    logs = [load(path, axes) for path in args.logs]
    if len(logs) > 2:
        raise SystemExit("at most two logs: more than two panels stops being legible")
    if args.view == "3d":
        if len(logs) > 1:
            raise SystemExit(
                "--view 3d draws one log. Two schedules side by side is what the "
                "flat view is for; that comparison is why it exists."
            )
        blocks = len(block_indices(logs[0]))
        if blocks > BLOCK_CEILING and not args.allow_large:
            raise SystemExit(
                f"{blocks} blocks is past the practical ceiling of {BLOCK_CEILING} "
                "for --view 3d: the cost is per cube per frame and this would run "
                "for hours. Lower --max-steps, use a smaller lattice, draw it flat "
                "with --view 2d, or pass --allow-large if you meant it."
            )

    if args.check:
        for log in logs:
            ascii_frames(log, min(args.max_steps, 12))
        return

    for log in logs:
        print(summarise(log))
    path = render(
        logs,
        args.output,
        args.quality,
        args.max_steps,
        args.seconds_per_step,
        view=args.view,
        fps=args.fps,
        media_dir=args.media_dir,
        rotation_rate=args.rotation_rate,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
