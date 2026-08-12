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
use super::geometry::{regions_intersect, BlockGeometry};

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
}

/// The whole schedule, as data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskGraph {
    pub tasks: Vec<Task>,
    /// `phase_ranges[p]` is the half-open span of `tasks` belonging to phase p.
    pub phase_ranges: Vec<(usize, usize)>,
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
                    Some((from, previous_phase)) => {
                        // Candidates come from the *grid*, not from a scan: the
                        // previous phase's cores form a regular lattice, so the
                        // blocks that can possibly overlap a read extent are an
                        // index range per axis. Scanning every previous task was
                        // O(blocks^2) per boundary, which is 45 M comparisons at
                        // 6700 blocks and would have made the DAG cost more to
                        // build than the work it schedules.
                        // `source`, not `read`: what this task depends on is
                        // what it *fetches*, and the two differ exactly when the
                        // phase reads across grids. Using the read extent there
                        // would look up the previous phase's lattice with
                        // coordinates from a different space — the seam where a
                        // shape change used to become two decompositions with no
                        // edge between them at all.
                        let grid = &decomposition.phases[phase_index - 1].grid;
                        let counts = grid.blocks_per_axis();
                        let edge = grid.block();
                        let source = &geometry.source;
                        let mut ranges = [(0usize, 0usize); 3];
                        for axis in 0..3 {
                            let lo = source.start[axis] / edge[axis];
                            let hi = if source.shape[axis] == 0 {
                                lo
                            } else {
                                (source.start[axis] + source.shape[axis] - 1) / edge[axis]
                            };
                            ranges[axis] = (lo.min(counts[axis] - 1), hi.min(counts[axis] - 1));
                        }
                        let mut deps = Vec::new();
                        for i in ranges[0].0..=ranges[0].1 {
                            for j in ranges[1].0..=ranges[1].1 {
                                for k in ranges[2].0..=ranges[2].1 {
                                    let flat = (i * counts[1] + j) * counts[2] + k;
                                    let candidate = from + flat;
                                    debug_assert!(candidate < previous_phase);
                                    let producer: &Task = &tasks[candidate];
                                    if producer.geometry.valid.voxels() > 0
                                        && regions_intersect(&producer.geometry.valid, source)
                                    {
                                        deps.push(candidate);
                                    }
                                }
                            }
                        }
                        deps
                    }
                };
                tasks.push(Task {
                    id: start + block,
                    phase: phase_index,
                    block,
                    index: geometry.index,
                    geometry: geometry.clone(),
                    deps,
                });
            }
            phase_ranges.push((start, tasks.len()));
        }

        Self {
            tasks,
            phase_ranges,
        }
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
    pub fn dependencies_cover_reads(&self, decomposition: &Decomposition) -> Result<(), String> {
        for task in &self.tasks {
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

    #[test]
    fn dependencies_cover_reads_when_the_halo_is_sufficient() {
        let decomposition = two_phase([4, 0, 0], [4, 0, 0]);
        decomposition.check().unwrap();
        TaskGraph::build(&decomposition)
            .dependencies_cover_reads(&decomposition)
            .unwrap();
    }
}
