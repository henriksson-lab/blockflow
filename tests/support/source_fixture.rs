use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::Dtype;

#[derive(Debug, Clone)]
pub struct SourcePhase {
    slots: Vec<usize>,
    reach: [usize; 3],
    halo: [usize; 3],
}

impl SourcePhase {
    pub fn new(slots: impl Into<Vec<usize>>, reach: [usize; 3], halo: [usize; 3]) -> Self {
        Self {
            slots: slots.into(),
            reach,
            halo,
        }
    }

    pub fn with_equal_halo(slots: impl Into<Vec<usize>>, reach: [usize; 3]) -> Self {
        Self::new(slots, reach, reach)
    }
}

pub fn declared_plan(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: &BlockGrid,
    chain_reach: [usize; 3],
    phases: impl IntoIterator<Item = SourcePhase>,
) -> Decomposition {
    let slots = chain.slots();
    let phases = phases
        .into_iter()
        .map(|phase| {
            let names = phase
                .slots
                .iter()
                .map(|&slot| slots[slot].display_name())
                .collect();
            PhaseDecomposition::derive(phase.slots, names, phase.reach, phase.halo, grid.clone())
        })
        .collect();
    let mut plan = Decomposition {
        volume,
        dtype,
        phases,
        chain_reach,
    };
    plan.declare_dtypes(chain).unwrap();
    plan.declare_source_images(chain).unwrap();
    plan
}

pub fn declared_one_phase_per_slot(
    chain: &Chain,
    volume: [usize; 3],
    dtype: Dtype,
    grid: &BlockGrid,
    chain_reach: [usize; 3],
    reaches: &[[usize; 3]],
) -> Decomposition {
    let slots = chain.slots();
    assert_eq!(
        slots.len(),
        reaches.len(),
        "one reach must be supplied for each chain slot"
    );
    declared_plan(
        chain,
        volume,
        dtype,
        grid,
        chain_reach,
        reaches
            .iter()
            .enumerate()
            .map(|(slot, &reach)| SourcePhase::with_equal_halo(vec![slot], reach)),
    )
}
