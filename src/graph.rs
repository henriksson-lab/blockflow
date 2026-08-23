// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `docs/design/BLOCK_OPS.md` §"Emit a dependency graph, do not inline the
// loops". The unit of work is **(block, phase)**: apply one phase to one block.
// Such a task depends on the previous phase's tasks whose *valid* outputs cover
// its read extent — itself plus its neighbours.
//
// Why emit it rather than write nested loops
// ------------------------------------------
// It subsumes the loop-nest question. `for block { for phase }` is fusion;
// `for phase { for block }` is phase-major materialisation. Both are walks of
// this one graph under different priorities, so "fuse or materialise" stops
// being a structural choice baked into the executor and becomes a scheduling
// **priority** — variable, measurable, and choosable per run without touching
// any code that moves data.
//
// It is also the only thing that makes multi-node possible later without a
// rewrite. Tasks are independent given their inputs; distribution needs a
// scheduler and a placement rule, and placement is a graph partitioning problem
// that is only *available* if the dependencies are explicit.
//
// Granularity, deliberately
// -------------------------
// Nodes are `(block, phase)` — thousands, never per-voxel or per-chunk. Ten
// thousand blocks over six phases is sixty thousand tasks: trivial to hold,
// trivial to schedule, and small enough that the priority policy can be
// recomputed rather than cached.
//
// Dependencies are computed by **region intersection**, not by block index.
// That is what keeps per-phase block sizes open: two phases with different
// grids have no index correspondence at all, but their regions still intersect
// perfectly well.

use std::collections::BTreeSet;

use crate::region::Region;

use super::decomposition::Decomposition;
use super::geometry::{regions_intersect, BlockGeometry, BlockGrid};

/// One unit of work: apply phase `phase` to block `block`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    /// Index into `TaskGraph::tasks`.
    pub id: usize,
    pub phase: usize,
    /// Flat block index **within its phase's grid**.
    pub block: usize,
    pub index: [usize; 3],
    pub geometry: BlockGeometry,
    /// Task ids in the previous phase whose valid output this task reads.
    pub deps: Vec<usize>,
    /// The same thing for every image this task reads through a source leaf:
    /// one entry per image in `PhaseDecomposition::source_images`.
    ///
    /// **Kept apart from `deps` rather than merged into it**, because the two
    /// are checked against different regions of different images.
    /// `dependencies_cover_reads` asks whether the union of a set of valid
    /// regions covers what is fetched, and that question is only well posed one
    /// image at a time — merged, two producers of two images would each cover
    /// the fetch and the sum would be twice what was asked for.
    ///
    /// **Explicit rather than inferred from the phase order.** A source image is
    /// written by a phase that has already run, so one can argue that the
    /// transitive dependency is there anyway — but only while every image
    /// between the two is on the same lattice, and a phase that resamples
    /// breaks that argument without breaking any test. The edge that matters is
    /// cheap to state, so it is stated.
    pub source_deps: Vec<SourceDep>,
}

/// One image a task reads through a source leaf, and the tasks that wrote the
/// part of it the task reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDep {
    pub image: usize,
    /// Task ids in phase `image - 1`. Empty for image 0, which no phase writes
    /// — the same fact that makes image 0 the original source node.
    pub deps: Vec<usize>,
}

impl Task {
    /// How many completions this task waits on.
    ///
    /// A producer that appears in both lists is counted twice and decremented
    /// twice, which is why nothing here deduplicates: `TaskGraph::dependents`
    /// pushes one entry per occurrence, so the two stay balanced without either
    /// side having to know about the other.
    pub fn n_dependencies(&self) -> usize {
        self.deps.len()
            + self
                .source_deps
                .iter()
                .map(|source| source.deps.len())
                .sum::<usize>()
    }
}

/// The whole schedule, as data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskGraph {
    pub tasks: Vec<Task>,
    /// `phase_ranges[p]` is the half-open span of `tasks` belonging to phase p.
    pub phase_ranges: Vec<(usize, usize)>,
    /// `barriers[p]` is [`PhaseDecomposition::barrier`]: no task of phase `p`
    /// may start until **every** task of every earlier phase has finished.
    ///
    /// # The edges are not the whole ordering, and this is the exception
    ///
    /// Everywhere else in this file a dependency is a region intersection and a
    /// scheduler that runs a task once its `deps` are done is correct. A barrier
    /// phase is the one case where that is **not** enough: its blocks fetch only
    /// their own cores, so their `deps` are the handful of tasks covering those
    /// cores and can complete long before the rest of the phase below has. A
    /// scheduler that ignores this field will start such a block early and the
    /// op will answer from an incomplete fragment set — a plausible wrong answer
    /// whose wrongness depends on the schedule.
    ///
    /// # Why it is a phase-level fact and not `blocks x blocks` edges
    ///
    /// It was the edges first, and they were correct and free for both
    /// schedulers — the indegree machinery both already have enforced it with no
    /// new code. What they cost is a **product where every other edge here is a
    /// sum**. `a_barrier_at_a_large_block_count_is_priced` in
    /// `tests/barrier_phase.rs` builds both forms side by side and prints what
    /// each holds and how long each takes, so the figure is measured on the
    /// machine that asks rather than quoted here; `docs/design/barriers.md` §8.2
    /// has the run it was decided on. [`producers_of`] below rejected
    /// `O(blocks^2)` at the scale this crate targets, about a different feature
    /// and before this one existed, and that argument applies here unchanged.
    ///
    /// And a barrier exists **for** fine cuts: `docs/design/barriers.md` §2.2 is
    /// that the toll it removes is worst precisely where cutting finely is most
    /// wanted. Paying a quadratic to express one bit per phase, at the block
    /// counts the feature is for, is the wrong side of that trade.
    ///
    /// "Every task of `p` waits for every task of `p-1`" is a statement about two
    /// *phases*, and this is it, stated once. What it costs is that a scheduler
    /// has to act on it; see [`Self::is_barrier`] for who does.
    ///
    /// [`PhaseDecomposition::barrier`]: crate::decomposition::PhaseDecomposition::barrier
    pub barriers: Vec<bool>,
}

impl TaskGraph {
    pub fn build(decomposition: &Decomposition) -> Self {
        let mut tasks: Vec<Task> = Vec::with_capacity(decomposition.n_tasks());
        let mut phase_ranges = Vec::with_capacity(decomposition.n_phases());

        for (phase_index, phase) in decomposition.phases.iter().enumerate() {
            let start = tasks.len();
            let previous = phase_ranges.last().copied();
            for (block, geometry) in phase.blocks.iter().enumerate() {
                let deps = match previous {
                    None => Vec::new(),
                    // `source`, not `read`: what this task depends on is what it
                    // *fetches*, and the two differ exactly when the phase reads
                    // across grids. Using the read extent there would look up
                    // the previous phase's lattice with coordinates from a
                    // different space — the seam where a shape change used to
                    // become two decompositions with no edge between them at
                    // all.
                    Some((from, previous_end)) => producers_of(
                        &tasks,
                        from,
                        previous_end,
                        &decomposition.phases[phase_index - 1].grid,
                        &geometry.source,
                    ),
                };
                // Every image this phase reads besides the one it is handed.
                // Read at the *same* region — a source leaf has reach 0 — which
                // is exactly why `check_source_images` requires the two images
                // to be on one lattice: without that, `geometry.source` would be
                // the wrong integers here.
                let source_deps = decomposition.phases[phase_index]
                    .source_images
                    .iter()
                    .map(|&image| SourceDep {
                        image,
                        deps: match image {
                            // Image 0 is written by no phase. It is the original
                            // source node, and the reason "an image with no
                            // producing phase" is a case of a rule rather than a
                            // special case.
                            0 => Vec::new(),
                            // A supplied input is the same case, arrived at from
                            // the other end: it was handed to the run, so it is
                            // ready before the first task and depends on
                            // nothing. Written out rather than left to fall
                            // through the `get` below — which would answer
                            // `None` for it, correctly and by accident.
                            _ if crate::assemble::is_supplied_image(image) => Vec::new(),
                            // `get`, not an index: a forward reference has no
                            // entry here yet, and it is `check_source_images`
                            // that must report it — by name, with the two phase
                            // numbers — rather than this line panicking on a
                            // plan somebody handed us.
                            _ => match phase_ranges.get(image - 1) {
                                None => Vec::new(),
                                Some(&(from, end)) => producers_of(
                                    &tasks,
                                    from,
                                    end,
                                    &decomposition.phases[image - 1].grid,
                                    &geometry.source,
                                ),
                            },
                        },
                    })
                    .collect();
                tasks.push(Task {
                    id: start + block,
                    phase: phase_index,
                    block,
                    index: geometry.index,
                    geometry: geometry.clone(),
                    deps,
                    source_deps,
                });
            }
            phase_ranges.push((start, tasks.len()));
        }

        Self {
            tasks,
            phase_ranges,
            barriers: decomposition
                .phases
                .iter()
                .map(|phase| phase.barrier)
                .collect(),
        }
    }

    /// Whether phase `phase` may only start once every earlier phase is
    /// complete. See [`Self::barriers`].
    ///
    /// # Every scheduler over this graph must consult it
    ///
    /// Two do, and they do the same two things, so a third has a shape to copy:
    ///
    /// * `strategy::execute_phases` holds such a phase's tasks out of the ready
    ///   heap and releases them together when every earlier phase's remaining
    ///   count reaches zero. That is also where `FragmentOp::reduce` runs, which
    ///   is a second reason the gap has to exist.
    /// * `distributed::coordinator::Job::ready` excludes them from the ready set
    ///   on the same condition, counted from its own per-phase completion
    ///   tallies.
    ///
    /// **Neither can deadlock**, on the argument `strategy.rs` already wrote for
    /// an iterative phase: a task of phase `p` waits only on earlier phases, so
    /// holding phase `p`'s tasks back blocks nothing that phase `p`'s tasks
    /// need, phase 0's tasks are ready from the start, and the property carries
    /// forward.
    ///
    /// **And it is the only enforcement**, which is worth saying plainly because
    /// it briefly was not: while the barrier was `blocks x blocks` edges the
    /// indegree enforced it too, and a property enforced in one place that reads
    /// as though it is enforced in two is worse than one honestly enforced once.
    /// It is enforced once, here, by whoever schedules.
    pub fn is_barrier(&self, phase: usize) -> bool {
        self.barriers.get(phase).copied().unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub fn n_phases(&self) -> usize {
        self.phase_ranges.len()
    }

    pub fn tasks_in_phase(&self, phase: usize) -> &[Task] {
        let (from, to) = self.phase_ranges[phase];
        &self.tasks[from..to]
    }

    /// How many tasks depend on each task.
    pub fn dependents(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.tasks.len()];
        for task in &self.tasks {
            for &dep in &task.deps {
                out[dep].push(task.id);
            }
            // One entry per occurrence, deliberately: a task that is both a
            // producer of the previous image and a producer of an image read by
            // a source leaf appears twice here and is counted twice by
            // `Task::n_dependencies`, so the indegree still reaches zero
            // exactly once.
            for source in &task.source_deps {
                for &dep in &source.deps {
                    out[dep].push(task.id);
                }
            }
        }
        out
    }

    /// The union of a task's dependencies' valid regions must cover what it
    /// fetches, or it is reading something nobody produced.
    ///
    /// Checked by area rather than by set difference: the previous phase's
    /// valid regions tile that phase's volume (that is `Decomposition::check`),
    /// so covering the fetched region is equivalent to the intersected areas
    /// summing to it. The two facts together are what make this cheap.
    ///
    /// The quantity is `source`, for the reason given in `build`. A phase that
    /// changes shape is exactly where this check has to keep working, since it
    /// is the check that says the two halves of such a plan are one plan.
    ///
    /// # A barrier phase is checked exactly like every other, and that is the point
    ///
    /// There is no special case below for [`TaskGraph::barriers`] and the next
    /// reader should not add one. A barrier changes *when a phase may start*, not
    /// *what a block reads*: such a phase fetches its own core, its `deps` are
    /// the tasks covering that core, and the sum below comes out exact for the
    /// ordinary reason. That is what makes relieving the halo safe — the fetch
    /// shrinks and the guard on the fetch keeps its full force, which it would
    /// not if the deps had been widened to the whole phase to make the guard
    /// pass.
    pub fn dependencies_cover_reads(&self, decomposition: &Decomposition) -> Result<(), String> {
        for task in &self.tasks {
            // Every image a source leaf reads, on the same argument and against
            // the same region: a source leaf has reach 0, so what it reads is
            // what the task fetches. Image 0 is skipped because no phase writes
            // it — it is there before the run, which is the whole reason a
            // image with no producer is a case of the rule rather than an
            // exception to it.
            for source in &task.source_deps {
                // ... and a supplied input for the same reason arrived at from
                // the other end: it was handed to the run, so no phase produces
                // it and there is nothing whose valid regions could cover the
                // fetch. That it is *there* at all is the environment's check,
                // and its extent is `check_source_images`'.
                if source.image == 0 || crate::assemble::is_supplied_image(source.image) {
                    continue;
                }
                let covered: usize = source
                    .deps
                    .iter()
                    .map(|&dep| {
                        intersection_voxels(&self.tasks[dep].geometry.valid, &task.geometry.source)
                    })
                    .sum();
                let wanted = task.geometry.source.voxels();
                if covered != wanted {
                    return Err(format!(
                        "task (phase {}, block {:?}) reads {wanted} voxels of image {} through a \
                         source leaf, and the {} task(s) of phase {} it depends on produce \
                         {covered} of them",
                        task.phase,
                        task.index,
                        source.image,
                        source.deps.len(),
                        source.image - 1,
                    ));
                }
            }
            if task.phase == 0 {
                continue;
            }
            let mut covered = 0usize;
            for &dep in &task.deps {
                covered +=
                    intersection_voxels(&self.tasks[dep].geometry.valid, &task.geometry.source);
            }
            let wanted = task.geometry.source.voxels();
            if covered != wanted {
                return Err(format!(
                    "task (phase {}, block {:?}) reads {wanted} voxels but its {} \
                     dependencies only produce {covered} of them; phase {} halo {:?} \
                     reach {:?}",
                    task.phase,
                    task.index,
                    task.deps.len(),
                    task.phase - 1,
                    decomposition.phases[task.phase - 1].halo,
                    decomposition.phases[task.phase - 1].reach,
                ));
            }
        }
        Ok(())
    }

    /// Distinct blocks touched, per phase.
    pub fn blocks_per_phase(&self) -> Vec<usize> {
        self.phase_ranges
            .iter()
            .map(|&(from, to)| {
                self.tasks[from..to]
                    .iter()
                    .map(|task| task.block)
                    .collect::<BTreeSet<_>>()
                    .len()
            })
            .collect()
    }
}

/// The tasks in `tasks[from..end]` — one whole phase, laid out on `grid` — whose
/// valid output intersects `wanted`.
///
/// Candidates come from the *grid*, not from a scan: a phase's cores form a
/// regular lattice, so the blocks that can possibly overlap a region are an
/// index range per axis. Scanning every task of the phase was O(blocks^2) per
/// edge, which is 45 M comparisons at 6700 blocks and would have made the DAG
/// cost more to build than the work it schedules.
///
/// Shared by the two kinds of edge — the phase before, and an image a source leaf
/// reads — because they are the same question asked of a different phase, and a
/// second copy of this arithmetic is a second place for the clamping to be
/// wrong.
fn producers_of(
    tasks: &[Task],
    from: usize,
    end: usize,
    grid: &BlockGrid,
    wanted: &Region,
) -> Vec<usize> {
    let counts = grid.blocks_per_axis();
    let edge = grid.block();
    let mut ranges = [(0usize, 0usize); 3];
    for axis in 0..3 {
        let lo = wanted.start[axis] / edge[axis];
        let hi = if wanted.shape[axis] == 0 {
            lo
        } else {
            (wanted.start[axis] + wanted.shape[axis] - 1) / edge[axis]
        };
        ranges[axis] = (lo.min(counts[axis] - 1), hi.min(counts[axis] - 1));
    }
    let mut found = Vec::new();
    for i in ranges[0].0..=ranges[0].1 {
        for j in ranges[1].0..=ranges[1].1 {
            for k in ranges[2].0..=ranges[2].1 {
                let flat = (i * counts[1] + j) * counts[2] + k;
                let candidate = from + flat;
                debug_assert!(candidate < end);
                let producer: &Task = &tasks[candidate];
                if producer.geometry.valid.voxels() > 0
                    && regions_intersect(&producer.geometry.valid, wanted)
                {
                    found.push(candidate);
                }
            }
        }
    }
    found
}

fn intersection_voxels(left: &Region, right: &Region) -> usize {
    (0..left.start.len())
        .map(|axis| {
            let lo = left.start[axis].max(right.start[axis]);
            let hi =
                (left.start[axis] + left.shape[axis]).min(right.start[axis] + right.shape[axis]);
            hi.saturating_sub(lo)
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decomposition::PhaseDecomposition;
    use crate::dtype::Dtype;
    use crate::geometry::BlockGrid;

    fn two_phase(reach: [usize; 3], halo: [usize; 3]) -> Decomposition {
        let grid = BlockGrid::new([64, 4, 4], [16, 4, 4]).unwrap();
        Decomposition {
            volume: [64, 4, 4],
            dtype: Dtype::F64,
            phases: vec![
                PhaseDecomposition::derive(
                    vec![0],
                    vec!["first".to_string()],
                    reach,
                    halo,
                    grid.clone(),
                ),
                PhaseDecomposition::derive(vec![1], vec!["second".to_string()], reach, halo, grid),
            ],
            chain_reach: [reach[0] * 2, reach[1] * 2, reach[2] * 2],
        }
    }

    #[test]
    fn a_zero_reach_phase_depends_only_on_its_own_block() {
        let graph = TaskGraph::build(&two_phase([0, 0, 0], [0, 0, 0]));
        assert_eq!(graph.len(), 8);
        for task in graph.tasks_in_phase(1) {
            assert_eq!(task.deps.len(), 1);
            assert_eq!(graph.tasks[task.deps[0]].block, task.block);
        }
    }

    #[test]
    fn a_halo_makes_a_task_depend_on_its_neighbours() {
        let graph = TaskGraph::build(&two_phase([4, 0, 0], [4, 0, 0]));
        let phase_one = graph.tasks_in_phase(1);
        // first and last blocks have one neighbour, interior blocks have two
        assert_eq!(phase_one[0].deps.len(), 2);
        assert_eq!(phase_one[1].deps.len(), 3);
        assert_eq!(phase_one[3].deps.len(), 2);
    }

    #[test]
    fn the_graph_is_thousands_of_nodes_not_billions() {
        // 64^3 blocks over a 2048^3 volume with 6 phases: the design's own
        // sanity bound, asserted so a future change to the unit of work is
        // caught here rather than in a heap profile.
        let grid = BlockGrid::new([2048, 2048, 2048], [256, 256, 256]).unwrap();
        assert_eq!(grid.n_blocks(), 512);
        let phases = (0..6)
            .map(|slot| {
                PhaseDecomposition::derive(
                    vec![slot],
                    vec![format!("op{slot}")],
                    [0, 0, 0],
                    [0, 0, 0],
                    grid.clone(),
                )
            })
            .collect();
        let decomposition = Decomposition {
            volume: [2048, 2048, 2048],
            dtype: Dtype::U16,
            phases,
            chain_reach: [0, 0, 0],
        };
        let graph = TaskGraph::build(&decomposition);
        assert_eq!(graph.len(), 3072);
    }

    /// An image read by a source leaf is an edge in the graph, to the phase that
    /// wrote it — not an assumption that the phase order made it available.
    #[test]
    fn a_source_image_is_an_edge_to_the_phase_that_wrote_it() {
        let mut plan = two_phase([4, 0, 0], [4, 0, 0]);
        // Phase 1 reads image 1 as its input and image 0 as a second arm.
        plan.phases[1] = plan.phases[1].clone().with_source_images([0]);
        let graph = TaskGraph::build(&plan);
        for task in graph.tasks_in_phase(0) {
            assert!(task.source_deps.is_empty());
            assert_eq!(task.n_dependencies(), 0);
        }
        for task in graph.tasks_in_phase(1) {
            assert_eq!(task.source_deps.len(), 1);
            // Image 0 is written by nobody, so it waits on nothing extra.
            assert_eq!(task.source_deps[0].image, 0);
            assert!(task.source_deps[0].deps.is_empty());
            assert_eq!(task.n_dependencies(), task.deps.len());
        }

        // A three-phase plan whose last phase reads image 1: now there is a
        // producing phase, and the edge points at the tasks that covered the
        // fetch.
        let mut plan = two_phase([4, 0, 0], [4, 0, 0]);
        let grid = BlockGrid::new([64, 4, 4], [16, 4, 4]).unwrap();
        plan.phases.push(
            PhaseDecomposition::derive(
                vec![2],
                vec!["third".to_string()],
                [0, 0, 0],
                [0, 0, 0],
                grid,
            )
            .with_source_images([1]),
        );
        let graph = TaskGraph::build(&plan);
        for task in graph.tasks_in_phase(2) {
            let source = &task.source_deps[0];
            assert_eq!(source.image, 1);
            // Phase 0 wrote image 1, and a zero-reach block reads its own core.
            assert_eq!(source.deps.len(), 1);
            assert_eq!(graph.tasks[source.deps[0]].phase, 0);
            assert_eq!(graph.tasks[source.deps[0]].block, task.block);
            assert_eq!(task.n_dependencies(), task.deps.len() + 1);
            // and both edges are recorded, so the producer releases it once for
            // each
            assert_eq!(
                graph.dependents()[source.deps[0]]
                    .iter()
                    .filter(|&&id| id == task.id)
                    .count(),
                1
            );
        }
        graph.dependencies_cover_reads(&plan).unwrap();
    }

    #[test]
    fn dependencies_cover_reads_when_the_halo_is_sufficient() {
        let decomposition = two_phase([4, 0, 0], [4, 0, 0]);
        decomposition.check().unwrap();
        TaskGraph::build(&decomposition)
            .dependencies_cover_reads(&decomposition)
            .unwrap();
    }
}
