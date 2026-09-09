# Distributed Locality And Process Fixtures

This note records the history behind the local multi-node process tests. The
tests keep the executable invariants; this document keeps the measurements and
fixture choices that explain why those invariants are checked the way they are.

## Why These Tests Use Processes

The local multi-node tests intentionally start real worker processes over real
sockets and shared files. A thread-based fake would share one address space, one
cache, one allocator, one memory budget, and one set of file handles. That would
exercise message shapes, but it would miss the failures this layer is meant to
catch: a worker reading an intermediate before another worker flushes it, two
processes writing the same file, an event stream that only merges correctly
because both ends were one object, or a work list that stays ahead only because
the network is a function call.

The executable claims are:

1. multiple workers produce byte-identical output to a single-node run;
2. every block executes exactly once under the normal coverage criteria;
3. a worker death without a lease stops the job and names what was lost;
4. a worker death with an explicit lease reissues the task;
5. the work list stays at least one task ahead from both worker and coordinator
   observations.

Each claim has a premise as well as an assertion. Several processes must really
write fragments, a worker must really die while the job is running, and the
coordinator must really withhold or hand out work. The tests assert those
premises before using the run as evidence.

## Why Cohort Gating Was Refused

Holding all work until the whole worker cohort has joined was tried as a way to
make the multi-process premises deterministic. It did make the premises hold,
but it changed the scheduler contract: one slow worker made the whole job slow.
On a loaded machine, this was common rather than rare. In the recorded run,
thirty-nine of forty cohort-gated runs took longer than ten seconds, compared
with none without the gate, and failures increased rather than decreased.

The accepted fixture shape is therefore pull-based handout plus explicit premise
checks, not a coordinator gate.

## Worker Death Fixtures

The worker-death tests originally killed a process after the coordinator
reported two tasks complete. That was a sampling race. The runner polled the
same HTTP server that workers were posting events to, while a sixteen-block
probe job finished in milliseconds after hundreds of milliseconds of process
startup. Under load, the runner could move from "two done" to "all done" between
samples, kill a worker that had already exited, and make the test read a fixture
failure as a design failure.

The current fixture makes the death the worker's own observation via
`LocalOptions::abort_worker_after`. The block count remains raised so survivors
still have work to finish after the death, and every death test first asserts
that a worker actually died.

The no-lease and lease cases are a pair. The lease case alone cannot distinguish
"expiry is off" from "expiry is broken"; the no-lease case alone cannot
distinguish "no lease" from "no claims". Running the same death against both
contracts pins the default and the opt-in behavior against each other.

## Spread Fixtures

Tests that claim several worker processes did work cannot rely on the handout to
spread a short job. The coordinator promises correct output for any assignment,
not an even spread.

Measured over forty runs at load 50, the original sixteen-block, three-worker
fragment job usually spread well, with examples such as `[12, 12, 8]`,
`[9, 13, 10]`, and `[11, 13, 8]`. The failure mode was startup skew: one worker
could finish all work before its peers joined, producing rows like `[32, 0, 0]`.

The fixture now uses a longer job where late joiners still have work available.
The test still asserts the premise, because sizing makes the condition likely
but only the assertion makes it checked.
