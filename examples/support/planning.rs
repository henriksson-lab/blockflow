//! Planner entry point shared by the example binaries.

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::Constraints;
use blockflow::op::Chain;
use blockflow::strategy::Enumerating;
use blockflow::Result;

/// Let the planner choose the partition and block grid for a pixel stage.
/// Later assembled stages retain the builder's configured grid unless their
/// own planning step explicitly changes it.
#[allow(dead_code)]
pub fn pixels(builder: &mut PlanBuilder, chain: Chain) -> Result<()> {
    let constraints = constraints(builder.grid().volume());
    builder.partition(chain, &Enumerating::default(), &constraints)?;
    Ok(())
}

pub fn constraints(shape: [usize; 3]) -> Constraints {
    Constraints {
        split_axes: if shape[0] == 1 {
            vec![1, 2]
        } else {
            vec![0, 1, 2]
        },
        ..Constraints::default()
    }
}
