// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Kept peak rows to a seed label image.
//!
//! `classical_centers::CenterSeedsOp` does this for rows whose first column is
//! a `U64` identity; a response-peak table has only its `response`. Here the
//! label is the row's linear voxel index plus one, which is unique and does not
//! depend on the block grid. Labels are sparse, which nothing downstream minds.

use blockflow::fragment::{BlockOutput, BlockView, FragmentInput, FragmentOp};
use blockflow::ops::merge_rows;
use blockflow::table::Schema;
use blockflow::{Dtype, Error, Result};

pub struct PeakSeedsOp {
    name: &'static str,
    rows: String,
    rows_phase: usize,
    schema: Schema,
}

impl PeakSeedsOp {
    pub fn new(name: &'static str, rows: impl Into<String>, rows_phase: usize, schema: Schema) -> Self {
        Self {
            name,
            rows: rows.into(),
            rows_phase,
            schema,
        }
    }
}

impl FragmentOp for PeakSeedsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.rows, self.rows_phase)]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let bytes = at.own(&self.rows).unwrap_or(&[]);
        let volume = at.volume();
        let rows = merge_rows(volume, self.schema.clone(), std::iter::once((at.index, bytes)))?;
        let mut out = at.output_buffer(0.0)?;
        if let Some(array) = out.as_array_mut() {
            let mut labels = array.view_mut::<u32>()?;
            for row in rows {
                let linear = (row.at[0] * volume[1] + row.at[1]) * volume[2] + row.at[2] + 1;
                let label = u32::try_from(linear)
                    .map_err(|_| Error::invalid("peak seed label does not fit u32"))?;
                let local = [0, 1, 2].map(|axis| row.at[axis] - at.read.start[axis]);
                labels[local] = label;
            }
        }
        Ok(BlockOutput::nothing().with_pixels(out))
    }
}
