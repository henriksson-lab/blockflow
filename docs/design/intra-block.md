# Intra-block threading

*This note is the empirical half of a question the project owner asked: the unit
of parallelism here is the block, a sibling implementation threads the inside of
one instead, and would that pay. It is measurement and a design that others can
build from. **Nothing in `src/` was changed to take these figures** — the
threading is a harness in `clearmap-ng/tests/intra_block.rs`, which slices a
shipped op from outside the crate through its public `ArrayView3` /
`ArrayViewMut3` signature. That this was possible at all is the first finding.*

*Read §0 before any number below.*

---

## 0. What every figure here is

**One socket, twenty cores, forty hardware threads, 27.5 MiB of L3** — an Intel
Xeon Gold 6138 at 2.00 GHz — and **six other workers were running throughout**.
The load average is printed beside every table in the harness and ran between
**58 and 84** for the whole session, so the machine was between 1.5x and 2.1x
oversubscribed while every thread-scaling number below was taken.

Consequences, and they are not decoration:

* **Every speedup here is a lower bound.** At sixteen threads the arms reported
  `cpu/wall` between 6.5 and 9.9, not 16. On a quiet machine the same CPU-seconds
  would have bought more.
* **Ratios are claimed; absolutes are not.** Two runs of the identical
  configuration put the 32-thread arm at `3.81x` and at `5.41x` and reversed its
  ordering against the 16-thread arm. The plateau is about `5x`; *where* in
  16-32 it sits is not a fact this machine can produce.
* **CPU-seconds are the trustworthy column.** They are immune to the scheduler in
  a way wall time is not, and the strongest result in this note (§3) is one that
  only appears there.

The op is `clearmap_ng::vesselize::tubeness_into` at ClearMap's own
`TubifyOptions::default()` — a truncated separable Gaussian at `sigma = 1.0`, a
second-difference Hessian, a closed-form eigenvalue solve, and a fold. Its reach
is `gaussian_radius(1.0) + 1 = 5` planes. It was chosen because it is the arm
that dominates this pipeline's binarisation and because the sibling's version of
it (`hessian_eigenvalue_core`, `par_chunks_mut` over x-planes) is the existence
proof the question was asked about.

---

## 1. The correctness control, and it is not a tolerance

`the_slab_decomposition_is_bit_identical` asserts that cutting a block into
slabs, each grown by the op's reach, produces **the same f64 bits** as computing
the uncut block — at 1, 2, 3, 4 and 8 slabs; under both thread assignments of
§4; and nested two levels deep as blocks-of-slabs.

It is bit-identity rather than a tolerance because the property is exactly
derivable. A separable Gaussian of radius `r` followed by a one-voxel stencil
means a voxel at least `r + 1` from the edge of the grown slab sums exactly the
taps it would have summed over the whole block, in the same order, from the same
values. No sum is reassociated; only *which thread runs the loop* changes. The
slab halo is `Tubify::reach`, which is `r + 1`, so every written voxel has that
margin.

**Every timing in this note is void without it.** A slab that computed a cheaper
wrong answer would show a speedup, and that is the shape of measurement this
project has been caught by repeatedly. The control also asserts the reference
answered something non-zero, so bit-identity cannot hold vacuously.

---

## 2. Does it pay, and where does it stop

One block, `160^3` f64 (32.8 MB — note that this **already exceeds L3**), one
slab per thread. `computed/vol` is the redundant arithmetic the halo creates and
is arithmetic, not measurement.

| threads | wall s | speedup | cpu s | cpu/wall | computed/vol |
|--:|--:|--:|--:|--:|--:|
| 1 | 3.6943 | 1.000 | 3.40 | 0.92 | 1.000 |
| 2 | 2.2034 | 1.677 | 3.62 | 1.64 | 1.062 |
| 4 | 1.1572 | 3.192 | 3.92 | 3.39 | 1.188 |
| 8 | 0.9396 | 3.932 | 4.77 | 5.08 | 1.438 |
| 16 | 0.7344 | **5.030** | 6.44 | 8.77 | 1.938 |
| 32 | 0.9700 | 3.809 | 9.85 | 10.15 | 2.938 |

*(load 58.8 rising to 68.3 across the run)*

**Yes, and it saturates at about 5x.** A second run at load 62-66 gave `5.05x`
at sixteen and `5.41x` at thirty-two, so the turnover point is not stable but the
plateau is. The mechanism is not in doubt: at thirty-two slabs of a 160-plane
block the ten planes of halo per slab are **triple** the arithmetic.

That figure is a lower bound twice over — the machine was oversubscribed, and
the per-thread work here includes `tubeness_into`'s own `to_owned` of the slab,
which a framework-level implementation would not repeat.

---

## 3. The result that matters most: threads cost the halo and nothing else

Divide each row's CPU-seconds by its arithmetic amplification:

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
|--:|--:|--:|--:|--:|--:|--:|
| cpu / amplification | 3.40 | 3.41 | 3.30 | 3.32 | 3.32 | 3.35 |

**Flat to within 3% across a thirty-two-fold change in thread count and a
three-fold change in redundant work.** The same quantity computed from §4's
table (amplification 2.938 and 4.906, four to sixteen threads) gives 3.27-3.31,
and from a `256^3` block (134 MB, 4.9x L3) it gives 12.26-13.24 against a
one-thread 13.24 — flat again, at a five-fold larger working set.

So on this op, on this machine, **the CPU cost of a voxel of arithmetic does not
depend on how many threads are inside the block.** There is no memory-system
term. Threads cost exactly the redundant arithmetic the halo creates and nothing
else.

This is the same conclusion the session log kept outside this repository reached
from the other side — `clearmap-rs/forme2.md` §28, which is not part of this
crate and is named here only so the provenance is followable — *"It
is redundant work, not contention. If workers contended for memory the
all-workers time would rise as blocks were added; it does not"* — and it
**refutes the mechanism** offered elsewhere in the record for why 21 blocks on a
pool of 40 might run 1.7x faster per voxel *"because each running block has the
memory system more to itself"*. That passage says of itself that the magnitudes
are not claimed; this measurement says the effect is not there to be claimed, at
block sizes from 1.2x to 4.9x L3.

---

## 4. Cache separation, measured, and it does not matter here

The owner flagged this as not obvious: *"There will be little fight over L3 if
chunks are very separate."* So: the same thirty-two leaves, the same halo, the
same total voxels, assigned two ways. **Separated** gives thread `t` a
contiguous run of leaves. **Interleaved** gives it leaves `t, t+T, t+2T, ...`, so
the threads walk the volume together. The arithmetic is identical by
construction.

| arm | threads | wall s | cpu s |
|---|--:|--:|--:|
| 4t / 32 leaves separated | 4 | 3.7519 | 9.73 |
| 4t / 32 leaves interleaved | 4 | 3.6541 | 9.76 |
| 8t / 32 leaves separated | 8 | 2.0058 | 9.76 |
| 8t / 32 leaves interleaved | 8 | 2.3309 | 9.74 |
| 16t / 32 leaves separated | 16 | 1.2854 | 9.75 |
| 16t / 32 leaves interleaved | 16 | 1.4995 | 9.63 |
| 8t / 64 leaves separated | 8 | 4.4491 | 16.06 |
| 8t / 64 leaves interleaved | 8 | 4.1029 | 16.27 |

*(load 69.4 rising to 82.9 across the run — which is most of what the wall column
is showing)*

**Wall time disagrees in both directions** — separated wins two rows by up to
1.17x, interleaved wins two by up to 1.08x — and **CPU-seconds are flat to
1.5%** in every pair. If interleaving cost cache, the stall cycles would appear
as CPU time, because a stalled thread is a running thread. They do not.

**The honest reading is that this experiment could not make the effect appear on
this op.** That is not the same as proving no op has it. What it does establish
is that the effect is not large enough to be a design input for a stencil filter
at these sizes, and that anyone who wants to claim it must show it in
CPU-seconds, not in wall time on a shared box.

---

## 5. Zero-copy — the owner's stated precondition, which did not survive

> *"we likely need zerocopy strategies for it to ever pay off (so some unsafe
> might be needed, or slicing strategies)"*

Three separate claims are folded together there. They come apart differently.

### 5.1 `unsafe` is not needed for the shape this feature has

`ArrayViewMut3::split_at(Axis, index)` consumes a mutable view and returns two
that borrow **disjoint** halves for the original lifetime. Repeating it yields as
many disjoint mutable views as there are slabs, each `Send`, each handed to its
own thread. `split_cores` in the harness is nineteen lines and contains no raw
pointer, no `Send`/`Sync` impl of its own and no `unsafe`. The read side is a
shared `ArrayView3`, which is `Sync` for any `Sync` element.

That is the whole of the zero-copy answer for an **out-of-place** op — one that
reads a buffer and writes a different one, which is every `BlockOp` in this crate
by signature (`fn apply(&self, input: &Voxels, out: &mut Voxels, ...)`).

`unsafe` is needed for exactly one shape: an op that **reads and writes the same
buffer**, where the write set of one thread interleaves with the read set of
another. `clearmap-rs`'s `gaussian_filter_3d` is that case and is the model for
how to state it — its `SharedVolume` carries a raw pointer with `Send`/`Sync`
impls, and the invariant is written at the construction site rather than at the
type: *the caller hands each task a disjoint index set; every element is read by
exactly one task and written by exactly one task, and a task reads only the line
it writes.* If this framework ever grows an in-place op, that is the invariant
the `unsafe` would rest on, and the construction site is where it must be argued.
**No shipped op here is in that position today.**

### 5.2 What the framework's own types cannot express, and this is the real gap

`Voxels` is an owned `Array3<T>` behind a dtype tag, and there is **no borrowed
variant** — no `VoxelsView`, no `VoxelsMut`. `Voxels::view_mut::<T>()` hands out
an `ArrayViewMut3<T>` only after the caller has committed to an element type,
which is precisely what the tag exists to avoid.

So the executor cannot today say *"here is a mutable view of part of this
block"*. Handing a thread a sub-block means constructing a new `Voxels`, which
means a copy in and a copy back. **The zero-copy requirement is a type-level
gap, not a soundness one**, and closing it is one new enum:

```
enum VoxelsMut<'a> { Bool(ArrayViewMut3<'a, bool>), U8(..), .. }
```

with `split_at(Axis, usize) -> (VoxelsMut<'a>, VoxelsMut<'a>)` written once
through the `map_voxels!` macro that already exists in `voxels.rs`, and a
`BlockOp` method that takes it. That is the minimum, and it is mechanical.

### 5.3 But the precondition itself is **refuted for this op**

The claim is testable, so it was tested. `intra_cloning` gives every thread its
own copy of the **whole block** — what a framework that could not slice would
have to do — with the clone inside the thread and inside the timed region.

| threads | sliced (wall s) | cloned (wall s) | cloned / sliced |
|--:|--:|--:|--:|
| 1 | 3.4322 | 3.2104 | 0.94 |
| 4 | 1.1590 | 1.2789 | 1.10 |
| 16 | 0.6792 | 0.6491 | 0.96 |
| 32 | 0.6343 | 0.6719 | 1.06 |

**No consistent sign, and a worst case of 10%.** Cloning the block per thread is
free here, and the arithmetic says why: the block is 32.8 MB, page-cache copy
bandwidth on this machine measured at 3.1-4.3 GB/s (§7), so one clone is about
8 ms against 3.4 s of serial compute — a ratio of four hundred to one. Even
thirty-two of them is 0.26 s of memcpy spread over thirty-two threads.

**The rule this generalises to**, and it is the thing a planner would need:

> Per-thread cloning costs `T x block_bytes / bandwidth` and is negligible when
> that is small against the serial compute it is buying down. Equivalently:
> cloning is affordable when the op's arithmetic intensity exceeds about
> `T x bandwidth / (block_bytes / serial_seconds)`. For `tubeness` at `160^3`
> that is 400:1 and the clone disappears. For a voxelwise map or a threshold —
> a few nanoseconds per voxel against a copy of the same voxel — it would be the
> whole cost.

So zero-copy is **not** a precondition for the feature. It is a precondition for
the feature applying to *cheap* ops, and the same arithmetic that decides whether
to thread at all decides whether the copy matters. A first implementation may
copy.

---

## 6. Fetch and compute do not overlap, and that bounds what threading can recover

The owner asked whether *"blocks can only start processing once all chunks that
are required are present"*. In `strategy.rs::run_task` they can not:

```
let mut buf = env.read(task.phase, fetch)?;     // blocking, whole fetch extent
let read_ns = started.elapsed().as_nanos() as u64;
...
let next = env.apply(slots[slot], &buf, &sources, place)?;
```

`env.read` returns a complete buffer before any op sees a voxel. The prefetcher
(`src/prefetch.rs`) overlaps reads **across** blocks — it is a scheduler over a
plan that is already enumerated, and it explicitly *"cannot block compute"* — but
within one task the read is a serial prefix.

**This is not a footnote; it is measurable, and it is why §7's ordering comes out
the way it does.** A fixed sixteen-thread budget, one volume, and leaf geometry
held *identical* across the row so that every configuration does exactly the same
arithmetic (the CPU column proves it):

| arm | wall s | cpu s | fetch amplification |
|---|--:|--:|--:|
| 1 block x 16 threads | 0.7575 | 6.07 | 1.000x |
| 2 blocks x 8 threads | 0.7044 | 6.13 | 1.062x |
| 4 blocks x 4 threads | 0.6778 | 6.05 | 1.188x |
| 8 blocks x 2 threads | 0.6480 | 6.03 | 1.438x |
| 16 blocks x 1 thread | **0.6372** | 6.04 | 1.938x |

*(load 64.4, `160^3`)*

**More blocks is monotonically faster, even though more blocks read nearly twice
the volume.** The arithmetic is identical, the CPU-seconds are identical to 1.7%,
and the ordering is entirely the serial prefix: one block's read happens on one
thread while fifteen wait, and sixteen blocks' reads happen on sixteen.

That is the sharpest single fact in this note. **At equal thread budget, block
parallelism beats intra-block threading whenever there are enough blocks to
spend the budget on** — by 1.19x here and by 1.50x at `256^3`. Intra-block
threading is not a better way to use threads; it is a way to use threads that
would otherwise be parked.

---

## 7. The regime map

This is the answer to *"it makes sense as a planner decision"*. Fix the block
count at what a budget or a halo makes affordable, then sweep threads per block.
Each row is against **its own one-thread column**, which is what `blockflow` does
today.

**`160^3` block, load 76.5 rising to 84.0:**

| blocks | 1t | 2t | 4t | 8t | 16t |
|--:|--:|--:|--:|--:|--:|
| 1 | 1.000 *(4.2699 s)* | 1.935 | 3.245 | 4.153 | **5.385** |
| 2 | 1.000 *(1.8749 s)* | 1.514 | 2.175 | 2.745 | **3.065** |
| 4 | 1.000 *(1.1318 s)* | 1.475 | **1.848** | 1.829 | 1.522 |

**`256^3` block (134 MB), load 76.1:**

| blocks | 1t | 2t | 4t | 8t | 16t |
|--:|--:|--:|--:|--:|--:|
| 1 | 1.000 *(13.7208 s)* | 1.877 | 3.229 | 4.097 | **4.571** |
| 2 | 1.000 *(7.1365 s)* | 1.410 | 2.284 | 3.093 | **3.199** |
| 4 | 1.000 *(3.9667 s)* | 1.684 | **1.985** | 1.920 | 1.661 |

Read across and down:

* **One block: 4.6-5.4x, and it is the whole of the case for the feature.** A
  stage that plans to one block has thirty-nine parked workers and no other way
  to use them. This is the regime the outside session log already names — *"this
  only bites when block count < core count"*, in `clearmap-rs/forme.md` under the
  heading *"And none of the 19x was parallelism"*. That file is not part of this
  crate and is named here only so the provenance is followable; it is cited by
  heading rather than by line because it is a live document that has been
  rewritten several times under this one, and a line number would rot silently
  where a heading does not. Two stages have now been measured in the regime
  independently (`vesselize` here; the cells `maxima` chain reported alongside).
* **Four blocks: 1.85-1.99x, peaking at four threads each and falling after.**
  The halo is being paid twice, once by the block cut and once by the slab cut,
  and it overtakes the threads.
* **The best absolute is never in the one-block row.** At `160^3`, `4b x 4t` is
  0.6125 s against `1b x 16t` at 0.7929; at `256^3`, 1.9984 against 3.0020. So
  **when the planner is free to choose the block count, it should**; intra-block
  threading is the fallback for when it is not.
* **Larger blocks did not help the feature.** The `256^3` block has a *cheaper*
  slab halo (1.586x at sixteen slabs against 1.938x at `160^3`) and yet a
  *smaller* one-block speedup (4.57x against 5.39x). "Large blocks plus
  intra-block threading" is not the winning corner.

So the planner-space rule is:

> Give a phase intra-block threads **`floor(workers / n_blocks)`**, capped where
> the slab halo overtakes them — measured here at about four slabs per block once
> the block cut is already paying a halo, and about sixteen when it is not. Give
> it **one** whenever `n_blocks >= workers`, because in that regime it is a loss.

---

## 8. The halo, priced in time rather than in voxels

The coordinator's addition: *"Halo cost should also not be overstated given io
cache."* A halo is by construction the region a neighbouring block just read, so
it is the part of the volume most likely to be resident, and every figure in this
project has priced it as a first read.

A 1 GiB file, 512 planes of 2 MiB, tiled along the slowest axis into `N` blocks
each grown by five planes — exactly `run_task`'s `fetch` extent — read cold
(`posix_fadvise(POSIX_FADV_DONTNEED)` over the whole file first) and warm (whole
file read once first). Load 76.3 rising to 82.2.

| blocks | bytes x | cold s | cold x | warm s | warm x |
|--:|--:|--:|--:|--:|--:|
| 1 | 1.000 | 12.13 | 1.000 | 0.289 | 1.000 |
| 4 | 1.059 | 7.82 | 0.645 | 0.280 | 0.968 |
| 16 | 1.293 | 11.30 | 0.932 | 0.323 | 1.118 |
| 32 | 1.605 | 12.07 | 0.995 | 0.501 | 1.734 |
| 64 | 2.230 | 14.60 | 1.204 | 0.773 | 2.676 |
| 128 | 3.477 | 15.95 | 1.315 | 0.861 | 2.978 |

**Cold: a 3.48x increase in bytes costs 1.32x in time**, non-monotonically (four
blocks came out *faster* than one). Cold reads on this storage are bound by
latency and readahead, not by bytes, so halo amplification is largely free there.

**Warm: 3.48x bytes costs 2.98x in time.** A page-cache read is a memcpy and
tracks bytes almost exactly — but only above about sixteen blocks; below that,
warm time is flat despite 1.29x the bytes.

**And the fact that dwarfs both: cold is 42x warm at one block** (12.13 s against
0.289 s). Whether the data is resident matters an order of magnitude more than
how much halo is re-read.

The honest scope: the warm arm has the *whole file* resident, which is an upper
bound on warmth; a real run has only the neighbour's halo resident. The truth is
between the two rows, and the cold row is the one that says halo bytes are cheap.
**Both directions of the coordinator's framing are therefore live, and the
measurement picks one: halo bytes overstate halo time, so small blocks are more
attractive than byte amplification suggests, which *weakens* the case for
intra-block threading rather than strengthening it.** §6 and §7 agree
independently.

---

## 9. Where CPU accounting belongs, and what it must not do

The owner wants total CPU *"so we can tell if it is efficient, and how many
threads to give out"*. Wall time cannot answer it; §3 and §4 are results that
exist only in the CPU column.

**At the phase level — this is the cheap, correct place.** `execute` already
walks phases. Sampling `/proc/self/stat`'s `utime + stime` at each phase boundary
is process-wide, includes threads that have already exited, and costs two integer
parses of a page that is already resident. Emitted beside the phase's existing
wall time it gives `cpu/wall`, which against `Hints::concurrency` is the parallel
efficiency the owner is asking for, directly. It is **off the block path
entirely**, so it cannot perturb what it measures.

**At the task level it needs a per-thread clock, and process-wide counters will
not do.** Under concurrency, `utime + stime` around one task includes every other
task running at the same time. The instrument is
`clock_gettime(CLOCK_THREAD_CPUTIME_ID)`, which is a vDSO call of about 20 ns and
attributes exactly; `statistics::Recorder` already accumulates per-slot
`(voxels, nanos)` per block and is the place it belongs. It would need `libc`,
which this crate does not have — that is a real cost and it is the reason to
start at the phase level.

**What it must not do**, and the failure mode is on record: a prior instrument
built a `String` shape key **while the block's buffers were live**, so the
measurement moved by an amount varying with the chain's name lengths. Two integer
adds per buffer was the acceptable form. The rule generalises to: *nothing may
allocate or format inside the region being timed.* The harness for this note
obeys it — its `cpu_ns()` allocates a `String` and is therefore called between
arms and never while a block is live, which is written above the function.

**And one trap specific to this instrument.** Summing
`/proc/self/task/*/schedstat` gives nanoseconds instead of 10 ms ticks and is the
tempting choice. It is wrong here: `std::thread::scope` joins its threads before
the arm ends, a joined thread is gone from `task/`, and its CPU goes with it. The
instrument would free the resource it was measuring — a shape this project has
been caught by already.

---

## 10. What this does not fix, and what it would cost

* **It does not transfer to every op, and reach cannot tell you which.** Slicing
  is exact iff the op is a **stencil** — the output at `v` is a function only of
  the input within `reach` of `v`. That is not the same as having a bounded
  reach, and §12.2 corrects this note's first attempt at saying why. **A new
  declaration is required** and it is now built: `BlockOp::slicing`.
* **It does not shorten the read.** §6: the block's fetch is a serial prefix and
  intra-block threading leaves it exactly where it was. Whatever fraction of a
  task is IO is a fraction this feature cannot touch, and Amdahl applies to it at
  the block level.
* **It compounds the halo when the block cut already pays one.** §7's four-block
  row turns over at four threads. `Chain::reach` makes sequential reaches **add**,
  so a chain of `k` ops needs a slab halo of the chain's *total* reach — the
  amplification tables in the harness are for a single op and understate a chain.
* **It must be switchable off, and the owner said so.** The natural form is a
  per-phase `threads: usize` in the decomposition with `1` meaning today's
  behaviour exactly, so that the no-feature configuration stays the one
  everything else is measured against. At `threads == 1` the slab cut is the
  whole block and §1's control already proves the result is bit-identical.
* **It fights the executor's own pool.** `worker_pool` builds one shared rayon
  pool per thread count and `Hints::concurrency` sizes it. Threads spawned inside
  a task are not that pool's, so `n_blocks x threads_per_block` can exceed the
  machine without anything noticing. Whatever is built must spend one budget, not
  two — which is another reason it belongs in the planner rather than in an op.

## 11. What would make this note obsolete

* **A measurement on a quiet machine.** Every speedup here is a lower bound taken
  at load 58-84 on forty threads. The one-block row could be materially better
  than 5.4x and this note would not know.
* **An op with a memory-system term.** §3 and §4 found none, on a stencil filter
  with a closed-form eigenvalue solve per voxel — an op with high arithmetic
  intensity. A bandwidth-bound op (a threshold, a voxelwise map, a copy) is the
  case where cache separation and per-thread cloning could both matter, and
  neither was measured here. §5.3 gives the arithmetic for predicting it; it has
  not been checked against a measurement.
* **Fetch and compute overlapping.** §6's ordering — more blocks always faster at
  equal arithmetic — rests entirely on the block read being serial. If `env.read`
  ever streamed into a compute that had already started, the one-block row would
  stop paying that prefix and §7's rule would need re-deriving.
* **A `VoxelsMut<'a>`.** §5.2's gap is mechanical to close and closing it would
  turn §5.3's "a first implementation may copy" from a concession into a choice.
  *(The output half of it was built; see §13.3.1. The input half — a shared
  borrowed variant and a `BlockOp` that accepts one — is still open, and is what
  keeps a memory-bound chain at 1.2-1.4x rather than nearer four.)*

---

*Harnesses: `clearmap-ng/tests/intra_block.rs` (`cargo test --release --test
intra_block -- --ignored --nocapture`; `INTRA_BLOCK_EDGE`, `INTRA_BLOCK_ROUNDS`
and `INTRA_BLOCK_BUDGET` set the geometry). The two non-ignored tests in it are
the bit-identity control and the halo arithmetic, and they run in the default
suite. §8's storage measurement is `blockflow/tools/halo_warmth.py`, which takes
the file to probe as its one argument and needs a filesystem it may drop the page
cache on.*


---

## 12. What was built, and what §10 got wrong

*§1-11 are the measurement and are unedited. This section is written by the work
that built the two pieces the measurement asked for, and it corrects §10 by
number rather than quietly agreeing with it.*

### 12.1 The two pieces

**`BlockOp::slicing() -> Slicing`**, in `src/op.rs`, with `Combine::slicing` and
`Chain::slicing` beside it. `Slicing::Stencil` is a claim about the kernel;
`Slicing::Whole(&'static str)` refuses **and says why**, because the interesting
cases all look sliceable from outside. `Chain::slicing` folds it with the first
refusal carried out, so a caller learns which part refused.

It has a default, `Slicing::UNDECLARED`, and the argument for that is written
beside the method rather than left implicit, because `FragmentOp::reach` had a
silent zero removed this session and the two look like the same question:

> A forgotten `reach` is a **correctness** failure with no diagnostic — the plan
> believes the halo is zero, allocates none, and every block computes its edges
> from data it never read. A forgotten `slicing` is today's behaviour exactly:
> one task per block on one thread. **Zero costs correctness; `Whole` costs
> performance**, and a default is affordable on the second where it is not on the
> first.

That argument has **one condition, and it is load-bearing**: it holds only while
the declaration is the sole source of truth. The moment anything infers
sliceability from another signal the default stops protecting anyone. Nothing in
this crate may derive a slicing; it may only read one.

**`slab::apply_sliced`**, in `src/slab.rs` — `Chain::apply_placed` with the block
cut into `threads` slabs run concurrently, plus `SlabCut`, which is the cut's
arithmetic on its own and has no threads in it. No `unsafe`. Slabs are disjoint
owned buffers joined by one serial pass, which §5.3 says is affordable and which
is why the `VoxelsMut` gap of §5.2 **was not closed** — it remains an
improvement, not a precondition. One helper was added for it,
`Voxels::assign_region_from`, so placing a slab's core is one copy rather than
two.

It refuses four things by name, each of which would otherwise be a complete,
well-formed, wrong volume: an undeclared chain; a chain reading a stored image
through a source leaf, because narrowing that buffer needs the reach of the *arm*
the leaf sits in and `reach_spec` folds that away; an output lattice that is not
the input lattice; and a reach that spans a whole axis, where every slab would do
the whole block's arithmetic.

### 12.2 §10's first bullet was wrong about which trait

§10 said a declaration was needed because `ops::label`, `ops::components` and
`ops::fill` have bounded reach and produce block-local identifiers a fragment
join reconciles. **The fact is true and the argument was not**: every one of
those is a `FragmentOp`, a different trait on a different phase kind, and none of
them can reach `BlockOp::slicing` at all. `impl BlockOp for` and `impl FragmentOp
for` partition the ops directory, and the whole labelling family is on the other
side of that line.

The real reason, and it is better because it is inside `BlockOp`'s own domain:
**a reach says what an op reads; it does not say the answer is a function of what
was read.** An op may read strictly within its reach and still carry state along
the buffer — `ops::sliding` maintains a window with `joining`/`leaving` sets
rather than re-summing it, `ops::local` runs an `f64` accumulator — and then the
answer depends on where the scan began. A cut moves where the scan begins. Where
that state is integer the answer survives; where it is floating point it moves in
the last place, on an interior stripe. No signal available to the framework
separates those two cases.

A second class comes apart the other way: `ops::resample` and `ops::lattice` have
ordinary bounded reaches and no index correspondence between a slab's core in the
two buffers. That one is caught by comparing shapes in `apply_sliced` rather than
by the declaration, deliberately — two guards over two different failures.

The fragment-join seam is kept on record because the reasoning transfers exactly,
and because slab parallelism for fragment phases would need **its own**
declaration rather than this one widened.

### 12.3 The acceptance bar, and the mutant that found a hole in it

`tests/intra_block_slicing.rs`, in the **default** suite because it is a
correctness property. Bit-identity against the uncut block at 2, 3, 4, 5 and 8
threads, over a single op, a sequence whose reaches add, and a fan-in with a
declared combine. Not a tolerance: §1's argument makes the property exactly
derivable, and a tolerance would pass the one failure this mechanism can produce
and the one nobody would notice.

Four controls sit beside it, and two of them earned their place:

* **A position-dependent op.** Without it every test would pass on a
  `slab_placement` that shifted wrongly or not at all, and every anchored op in
  the crate would compute the block corner's answer for every slab. The test also
  asserts the op really is position-dependent, or it could not detect a wrong
  anchor.
* **An op that declares `Stencil` and is not one** — it folds across its buffer.
  Cutting it must change the answer. Without this, green would mean the
  assertions ran, not that they could fire.
* **Every shipped op refuses today**, which is the record of the default doing
  its job and the place a newly declared op will show up.
* **The uncut path at one thread**, which is a different branch and proves
  nothing about the cut, so every other assertion is at two or more.

**Two mutants were run against it and the second found a hole in the test rather
than in the code.** Dropping the slab anchor shift failed the two
position-sensitive tests and nothing else — correct. Dropping the halo to zero
failed three tests and **left `a_stencil_survives_the_cut_bit_for_bit` green**.
The fixture's fold was `tanh` of a running accumulator; `tanh` saturates to
exactly `1.0` past an argument of about 20 and the fixture's values reach 140, so
every accumulator pinned at `1.0` and the op returned the same answer whatever it
read. **A test written to be sensitive was measuring nothing.** The fold is now a
halving accumulator that cannot saturate, both mutants fail the tests they should,
and the restore was verified green with the mutant string absent rather than
assumed.

### 12.4 What is specified here and built elsewhere

Neither of these is in this note's own files, and both are precise enough to hand
over:

* **The planner rule** — `floor(workers / n_blocks)` threads per block, capped
  where the slab halo overtakes them (~4 slabs once a block cut already pays a
  halo, ~16 when it does not), and **1 whenever `n_blocks >= workers`**, because
  §6 measured that regime as a loss. It belongs in `strategy.rs` and
  `decomposition.rs`. `apply_sliced` takes the thread count as an argument and
  has no opinion about it, which is what makes it a planner parameter rather than
  a global switch — and at `1` it takes the uncut path outright, so the
  no-feature configuration stays exactly the one every measurement here is
  against.
* **CPU accounting at the phase boundary** — §9, with its two traps: nothing may
  allocate or format inside a timed region, and summing
  `/proc/self/task/*/schedstat` loses joined threads' CPU, so the instrument
  would free what it is measuring.

### 12.5 What §11 still wants, unchanged

The shipped ops are **not** declared. `src/ops/**` was held by another worker
while this was built, so the per-op declarations are the next edit and none of
them is guessed here — *and this paragraph is now out of date twice over: four
`BlockOp`s were declared shortly after it was written (§12.3 records them), and
three `Combine`s and `VoxelwiseMapOp` were declared when the feature was wired.
§13.5 is the current list.* The bar it states is unchanged: an op is `Stencil` only when someone has put it in
`tests/intra_block_slicing.rs` and watched bit-identity hold, which is the bar
this note asked for and the one it now has.

---

## 13. What was wired, and the two claims of §12 that did not survive

*§1-11 are the measurement and are unedited; §12 is the primitive. This section
is written by the work that connected the two, and it corrects §12 by number.*

### 13.1 The thing that was missing, stated plainly

`slab::apply_sliced`, `SlabPolicy` and `Constraints::slab_policy` were all
built, all tested, and **nothing called any of them**. `apply_sliced` appeared
outside its own module only in doc comments; `slabs_for` appeared nowhere
outside `decomposition.rs`; the executor's `Decomposition` had no slab field and
no slab anything. Setting the policy could not move a voxel or a nanosecond.
§12.4 said as much — *"specified here and built elsewhere"* — and the elsewhere
had not happened.

The header of this note still says *"Nothing in `src/` was changed to take these
figures"*, and that sentence is now false. It is left standing because it is
true of §1-11, which is what it is attached to.

### 13.2 What connects it

* **`Hints::slab_policy`.** The advisory half of the plan, which is where a
  quantity belongs whose worst possible misuse is being slower. `Strategy::plan`
  copies `Constraints::slab_policy` into it — the one method holding both — so a
  caller states the policy once, at the constraint, and the executor reads it
  where it reads every other performance decision.
* **`strategy::execute_phases`** evaluates `floor(workers / n_blocks)` per phase,
  from `Hints::concurrency` and the phase's own block count, and hands it to
  `run_task`. `n_blocks` is the plan's, not the ready heap's, so the answer does
  not depend on the order the heap emptied in.
* **`Environment::apply_sliced`**, defaulted to `apply`. `sources` was
  deliberately *not* defaulted when it was added, on the argument that an
  environment silently ignoring an operand returns a well-formed wrong volume.
  A slab count is the opposite case and the difference is a property rather than
  a taste: an environment that ignores it returns **the same bits**, because
  bit-identity is this feature's acceptance bar. The three environments that
  hold real arrays now share one body, `env::apply_chain_to_block`, because
  three copies of an application is three places for a planner's decision to be
  dropped from.
* **`slab::apply_at_most`**, beside `apply_sliced` and sharing its `plan_cut`.
  The planner *offers* threads to a chain it has not looked at; `apply_sliced`'s
  refusals are errors because its caller *asked*. An offer that failed the run
  on every undeclared op would fail every plan this crate has, so an offer that
  cannot be taken is declined and the block runs uncut. Only the decision to cut
  is swallowed; everything the ops do propagates.
* **`Stats::slabs_run` and `Stats::blocks_sliced`.** Without them a green suite
  says the assertions ran, not that anything was cut.
* **`WorkerOptions::threads`**, and `execute_task_with_reduction`'s `slabs`, for
  the distributed side. §13.7.
* **`VoxelsMut<'a>`**, so a slab writes its own core rather than queueing behind
  one pass at the end. §13.3.1.

**Every one of these was mutated**, each restored and re-verified green with the
mutant string absent rather than assumed. An executor that never cuts fails only
the positive wiring test, which is correct. A slab halo of zero fails every
bit-identity case, including the combine, composite and `bool` ones. An
undeclared `DifferenceCombine` fails the combine case; an undeclared
`VoxelwiseMapOp` fails **only** the composite diamond, which is exactly what a
composite is for; an undeclared `VoxelwiseMaskOp` fails the shells case. A
`Strategy::plan` that drops the constraint-to-hints line fails the one assertion
that line has. An entry point that takes `slabs` and ignores it fails the entry
point's own test. A `WorkerOptions` default of anything but one fails the
assertion that stands between this feature and every existing deployment. A
fragment phase that reports a cut fails the reduction-safety test. And a
`split_at` handing out two views of the same half fails the concurrent-write
test, which is written through `thread::scope` for exactly that reason — a
sequential version would pass on an overlap, because the second write would
simply land on top of the first. A `LocalOptions` default of anything but one
fails the same assertion for the runner that starts the workers.

**Twelve mutants across the two passes, and every one of them failed exactly the
test it should and nothing else.** That is the weak claim and the right one:
none of them found a hole, in the code or in a test. The value is the *and
nothing else* — a mutant that failed half the file would say the tests overlap
rather than that each is about what it names. §12.3's own mutant, which found a
saturating fixture, is the outcome worth having and did not repeat here.

### 13.3 Does it pay once wired

*These are the figures **as first wired**, with the slab cores placed by one
serial pass. That pass was later threaded and §13.3.1 has the arms afterwards;
the rows below are kept because they are what the conclusion in this section was
drawn from, and because the agreement they show is the point.*

`128^3` and `112^3` `f64`, one block, four workers, `SlabPolicy::CAP` slabs, arms
**interleaved** round by round and the ratio taken as the median — because on
this box two runs of one configuration have differed by 1.5x while an
interleaved ratio moved by 2%. Load 35-39 on forty threads throughout.

| chain | radius | computed/written | wall ratio | cores busy, uncut | cores busy, cut |
|---|--:|--:|--:|--:|--:|
| median filter, `128^3` | 1 | 1.047 | **3.62x** | 1.00 | 3.76 |
| median filter, `128^3` | 3 | 1.141 | **3.39x** | 1.00 | 3.38-3.80 |
| median filter, `112^3` | 2 | 1.107 | **3.51x** | 1.00 | 3.78 |
| background diamond, `112^3` | 2 | 1.214 | **3.08x** | 1.00 | 3.71 |

**Yes.** And the interesting column is not the ratio, it is the product:
`3.076 x 1.214 = 3.74` against **3.71 cores busy**, and `3.509 x 1.107 = 3.885`
against 3.78. **§3's result reproduces independently** — the threads cost the
redundant arithmetic the halo creates and nothing else, here on two different
ops and through the whole executor rather than a harness.

That is a much better efficiency than §2's 8.77 cores for 5.03x, and the reason
is not a better implementation: it is `SlabPolicy::CAP`. Four slabs of a 112-plane
block at reach 4 is `1.21x` the arithmetic where sixteen slabs of a 160-plane
block at reach 5 was `1.94x`. The cap is buying the efficiency and paying for it
in ceiling — **3.1-3.6x where §2 reached 5.0x** — and that trade is the cap's
whole content.

### 13.3.1 Where it lost, and what the borrowed output did about it

`VoxelwiseMapOp` was declared a stencil (§13.5 says why it had to be), so a
one-block phase whose whole chain is voxelwise is now offered a cut. **It took
one and was slower for it.** `192^3` `f64`, one block, four workers, interleaved:

| chain | cost/voxel | computed/written | wall ratio | cores busy, uncut | cores busy, cut |
|---|--:|--:|--:|--:|--:|
| one map | 1.00 | 1.000 | **0.94x** | 1.00 | 1.05-1.36 |
| three maps in sequence | 3.00 | 1.000 | **0.93x** | 1.00 | 1.11-1.41 |

**`mean_cores_busy` is what identified it.** The amplification is exactly
`1.000` — a reach-0 cut creates *no* redundant arithmetic at all — so the halo
cannot be the explanation and wall time alone would leave the cause open. The CPU
column closes it: four slabs kept **1.1-1.4** cores busy against a filter's 3.7.
The threads were waiting, not working, and what they were waiting on was memory.

**The obvious cost guard would have got it wrong, so none was added.** Amdahl
with this crate's own measured copy constant says cut when
`cost_per_voxel x (1 - 1/slabs) > IDENTITY_COST`; that declines the single map
(0.75 against 0.95, correct) and *admits* the three-map sequence (2.25 against
0.95, measured at 0.93x). It is wrong because `cost_per_voxel` is a compute
figure and this regime is not compute-bound. A threshold fitted to two points
against a curve nobody has is tuning.

#### What it actually was, measured rather than modelled

The candidate was the **join**: `run_cut` ended with one serial pass placing
every slab's core into the output. It is the only part of a cut that does not
parallelise, so it is the Amdahl term. Modelling it from a bandwidth figure gave
19 ms for a 56.6 MB block. **Measured on its own it was 60-120 ms** — off by
three to six times, which is the whole reason this project measures.

It is also strongly non-linear in block size, which the model would never have
given: **5.8 ms at `160^3`, 90 ms at `224^3`** — 2.7x the bytes for 15x the time,
across a 27.5 MiB L3. So what threading it buys *rises* with the block, which is
the opposite of how the halo behaves and is the first term in this note that
does.

#### `VoxelsMut<'a>`, which is half of §5.2

§5.2 asked for two things: the borrowed enum with `split_at`, **and a `BlockOp`
method that takes one**. This is the first, and the first is the half that
removes the *serial* copy; the second would remove the per-slab input copy and is
a signature change across every op in the crate. Calling it "§5.2 closed" would
be claiming the larger half.

§5.2's author declined to build any of it, on the ground that §5.3 had measured
per-thread cloning as affordable. **That was true of the cases measured then and
false of the one above**, which is the evidence that reopens it rather than an
argument that overrides it.

`crate::voxels::VoxelsMut` is the borrowed, tag-carrying half of `Voxels`, and
`split_at(axis, index)` hands out **disjoint** mutable views of one buffer for
one lifetime. `run_cut` now peels one view per slab core before any thread starts
and each slab places its own answer. **No `unsafe`**: the compiler is what says
the halves do not overlap, and `ArrayViewMut3` is `Send` for a `Send` element.
The serial pass is gone entirely.

Measured against itself, interleaved, with the two arms asserted to write the
same volume:

| block | serial join | threaded join | |
|---|--:|--:|--:|
| `160^3` (32.8 MB) | 0.0058 s | 0.0020 s | **2.9x** |
| `192^3` (56.6 MB) | 0.1188 s | 0.0347 s | **3.4x** |
| `224^3` (89.9 MB) | 0.0902 s | 0.0259 s | **3.5x** |

**The ratios are the claim and the absolutes are not**, and this table shows why:
the `192^3` row is *larger* than the `224^3` one, which is impossible by size and
is the load moving between two runs. Each row's two numbers were taken
interleaved against each other and are comparable; two rows were not. The size
trend quoted above is the `160^3` and `224^3` rows, which were taken minutes
apart at load 40-77.

#### And the arms afterwards

`224^3` (89.9 MB), one block, four workers, four slabs, seven interleaved
rounds, load 40-77:

| chain | cost/voxel | computed/written | wall ratio |
|---|--:|--:|--:|
| one map | 1.00 | 1.000 | **1.18x** |
| three maps in sequence | 3.00 | 1.000 | **1.37x** |
| median filter | 104 | 1.027 | **3.07x** |
| background diamond | 210 | 1.054 | **3.01x** |

**Every arm is now at or above one**, where two of them were below it. At
`160^3` the one-map arm is `1.02x` — break-even rather than a win, because the
join it removed is only 5.8 ms there. So the gain scales with the block, exactly
as the join's own cost does.

**What is *not* fixed, and a borrowed output cannot fix it.** Two copies remain
and both are per-slab: the input slice in, and the answer buffer out. They are
parallel, so they are not Amdahl, but a memory-bound map does not run four times
faster on four threads and that is why the voxelwise arms gain 1.2-1.4x rather
than 4x. Removing the input copy needs a *shared* borrowed variant **and** a
`BlockOp` that accepts one — a signature change across every op in the crate.
Removing the answer buffer is not a types problem at all: an op writes the whole
buffer it is handed and a slab's extent is wider than its core, so the two cannot
be the same allocation while `apply` means what it means.

**One more fact the harness turned up.** `run_task` applies a phase's slots one
at a time and offers the cut to each, so a fused phase of `k` slots pays `k` cuts
and `k` joins rather than one — the three-map row reports twelve slabs, not four.
That used to make it the *worse* of the two voxelwise rows (0.93x against 0.94x)
and now makes it the better one (1.37x against 1.18x), because what is repeated
`k` times is now the cheap join instead of the expensive one.

### 13.4 The negative control, which is the half that had to be proved

`a_well_cut_plan_is_untouched_by_the_policy` runs a six-block plan on four
workers under both policies and asserts the same answer, the same
`decomposition_fingerprint`, the same reads, the same read voxels, the same
`ops_applied`, the same `estimated_work`, `same_work_as`, and
**`blocks_sliced == 0`**. §6 measured that regime as a loss and the rule answers
one slab in it; this is the assertion that the rule is the one the executor
actually applies.

`an_undeclared_chain_declines_the_planners_offer_and_runs_uncut` is the same
statement for the other axis: the offer meets an undeclared chain, declines, and
costs exactly what switching the policy off costs.

Both were run under a mutant that never cuts. The mutant fails
`the_executor_cuts_a_block_when_the_lattice_leaves_workers_parked` and leaves
these two green, which is correct and is what says they are about the regime
they name.

### 13.5 The declarations, and what each unblocked

**Four `BlockOp`s declared `Stencil`; zero `Combine`s did — and a `Parallel` node
is only as sliceable as its narrowest part.** Both arms this framework's consumer
spends its time in are fan-ins, so every one of them refused. *The declaration
sits on the op; the refusal sits on the chain around it.*

Declared here, each with a bit-identity case beside it in
`tests/intra_block_slicing.rs`:

* **`DifferenceCombine`** — `a - b` per voxel, and the operand order is
  load-bearing, so a cut that mirrored them would show here and nowhere else.
* **`ArithmeticCombine`** — reaches the tests twice, through the arithmetic
  kernel (`Subtract`) and the selection one (`Maximum`), because `pair`
  dispatches on the element type and the two are different code.
* **`LogicCombine`** — its arms end in a threshold in the fixture, because two
  arms non-zero everywhere give an `And` of one everywhere and an `Xor` of zero
  everywhere: an answer that does not depend on what was read, which is the trap
  §12.3 already paid for once.
* **`VoxelwiseMapOp`** — and this one is why the other three would otherwise have
  been assertions about a chain nothing could cut. `ops::background::
  remove_background` is an *identity map* against a grey opening under a
  difference; with the rank filters and the sink declared and the map not, the
  node still refused. Its argument is one the crate already depends on and does
  not widen: `MapFn` states purity as a precondition, and **a block boundary is
  already a cut** — this op reaches zero, so every block grid already partitions
  the volume into pieces it is applied to separately, and the conformance suite
  asserts the answer does not move across those grids.

* **`VoxelwiseMaskOp`** — the same argument as the map's in full, and declared
  later for a reason that was the *bar* rather than the argument: it writes
  `Bool` where the map writes `f64`, and this file's two bar helpers read `f64`,
  so declaring it would have meant a case with no bit-identity behind it or a
  relaxed bar. The helpers were generalised instead — per element type, refusing
  the rest **by name** rather than widening everything to `f64`, because a bar
  that is bit-identity cannot rest on a lossy comparison. The vacuity guard got
  stronger on the way: "the reference answered zero everywhere" would pass a
  `bool` answer of `true` everywhere, and it is now "the reference answered one
  value everywhere".

The composite is held too:
`the_background_removal_diamond_survives_the_cut_bit_for_bit` runs the shipped
diamond uncut and cut at every thread count, and asserts its reach is the
opening's *two* passes, so a cut that used one filter's reach would read short at
every seam.

Each fixture goes through `assert_the_fixture_can_see_its_halo` — the three-way
probe of §12.3, now extracted so a new case cannot forget it. A halo-to-zero
mutant fails every one of the new bit-identity tests, which is what says the
fixtures can see what they are testing.

### 13.6 The claims that did not survive

* **§12.4 said `apply_sliced` "takes the thread count as an argument and has no
  opinion about it, which is what makes it a planner parameter".** True of the
  primitive and *insufficient* for the planner. A count with no opinion still has
  four refusals, and a planner that has not looked at the chain meets them
  constantly. Wiring the primitive directly would have failed every plan holding
  an undeclared op — which is nearly every plan. The planner needs an entry point
  whose refusals are **fallbacks**, and that is a second function
  (`apply_at_most`) rather than a second argument.
* **§10's last bullet said the feature "fights the executor's own pool" and that
  `n_blocks x threads_per_block` could exceed the machine "without anything
  noticing".** It cannot in `execute_phases`, and the reason is the plan's shape
  rather than a clamp: a wave is at most `min(concurrency, ready)` tasks, a phase
  of `n` blocks offers at most `n` ready tasks and `floor(concurrency / n)` slabs
  each, and two one-block phases cannot both be ready because a one-block phase's
  task depends on the whole of the phase below it. The argument is written at the
  site in `strategy.rs`, and it is why there is no clamp there. **It is true of
  the distributed runner**, where `workers x threads` is what is asked of a box
  and nothing checks it against the box — exactly as `workers` alone was never
  checked. §13.7.
* **The serial join was assumed to be small, including by the work that wrote
  it.** A bandwidth model put it at 19 ms for a 56.6 MB block; it measured at
  three to six times that and is steeply non-linear in block size. It was the
  whole of the voxelwise loss at large blocks and none of it at small ones, and
  neither of those is what a model would have said. §13.3.1.

### 13.7 The distributed worker, which was the last place it had not reached

`strategy::execute_task_with_reduction` passed one slab and could not do
otherwise: it runs one block and was told nothing about the pool it runs in. It
now takes `slabs`, and `distributed::worker` supplies it from a new
`WorkerOptions::threads` — default **1**, which is what every recorded
distributed run was taken at, reachable as `--threads` on the worker binary and
as `LocalOptions::threads` on the local runner.

**The number the worker passes is `slabs_for(threads, 1)`, and the `1` is the
finding.** The planner's rule is `floor(workers / n_blocks)`, and `n_blocks` is
asking *how many blocks are in flight on this machine*. In `execute_phases` a
wave of tasks makes that the plan's block count; in a worker it does not — the
worker's main loop computes **one task at a time** however many blocks the plan
has, because `ahead` deepens the *claim* pipeline and not the compute. So a
worker on a node with cores to spare leaves all but one parked, and unlike the
single-node executor there is no block-level parallelism inside the process to
spend them on instead. That is §7's one-block row, which is the whole case for
the feature, and it is the regime with no alternative rather than the regime
where slabs merely win.

**What the signature change costs.** One call site in this crate, a compile error
naming the line for anybody outside it, and `1` as the answer that keeps their
behaviour identical. It is a parameter rather than a fourth entry point because
this function is called for a real run and never as a convenience, which is this
crate's own test for when exhaustiveness is worth paying — somebody has to decide
how many threads a node spends on one block, and that is exactly the decision
that should not be inheritable. The two wrappers above it keep their arity.

**A worker option rather than something carried in the job**, on the precedent
already in `WorkflowSpec::cache_bytes`: how many cores a node has and how many
worker processes share them are facts about the node, not the job. It is also why
no slab *policy* has to cross the wire — `threads: 1` is the off switch and it is
the default. `workers x threads` is what the local runner asks of a machine and
nothing checks it against the machine's cores, exactly as `workers` alone was
never checked.

**And it cannot disturb a hoisted reduction.** A barrier phase's reduction is
derived from the fragment set rather than transported, so every worker must fold
byte-identical bytes with no election; anything that could make two workers
disagree is a distributed wrong answer with no diagnostic. Slabs never reach a
fragment phase, because `run_task` dispatches one before the offer exists.
`a_fragment_phase_is_never_offered_a_cut` asserts that with the offer live in the
same run — the plan is deliberately one block, so the policy really is asking for
four slabs while the fragment phase takes none of them — and a mutant that has a
fragment phase report a cut fails it.

### 13.8 What is left

* **The input copy.** §13.3.1: a shared borrowed variant plus a `BlockOp` that
  accepts one, which is a signature change across every op in the crate. It is
  the remaining half of what keeps a memory-bound chain at 1.2-1.4x instead of
  nearer four.
* **One cut per phase rather than per slot.** A fused phase of `k` slots pays `k`
  cuts. Cutting once and running the slot sequence inside each slab would pay one
  join and would carry the phase's total reach instead of each slot's; it is a
  change to `run_task`'s buffer handling and to nothing here.
* **The accounting environment reports `slabs_run: 0`** rather than one per
  application, because it holds no data and runs no slab. A caller comparing a
  simulated run's slab count against a real one is comparing two different
  questions.

*Harness for §13.3: `the_cut_pays_on_a_one_block_plan` in
`tests/intra_block_slicing.rs`, `#[ignore]`d because it is a measurement.
`INTRA_SLAB_EDGE`, `INTRA_SLAB_RADIUS`, `INTRA_SLAB_WORKERS`, `INTRA_SLAB_ROUNDS`
and `INTRA_SLAB_CHAIN` set it. It asserts the two arms agree and that one was
cut and the other was not, so a timing cannot be reported for two identical
runs.*
