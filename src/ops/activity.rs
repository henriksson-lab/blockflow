// SPDX-License-Identifier: MIT
//! Conservative, per-run block occupancy metadata for masks and labels.
//! Missing metadata is unknown. Recompute in the DAG for each source generation;
//! these summaries are not a persistent cache and cannot outlive their source.
use crate::fragment::{
    BlockOutput, BlockPreparation, BlockView, Coverage, FragmentOp, FragmentOutput,
    PreparationView, SeamFold, SidecarSize, SourceBlocks,
};
use crate::op::SourceInput;
use crate::reach::Reach;
use crate::sidecar::Lifecycle;
use crate::{Dtype, Error, ImageId, Result};
const MAGIC: &[u8] = b"BFACT001";
pub fn encode(active: bool) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.push(u8::from(active));
    bytes
}
pub fn decode(bytes: &[u8]) -> Result<bool> {
    if bytes.len() != 9 || &bytes[..8] != MAGIC || bytes[8] > 1 {
        return Err(Error::invalid("invalid mask activity fragment"));
    }
    Ok(bytes[8] != 0)
}
/// One exact occupancy flag per core. Any nonzero (including non-finite)
/// mask value is conservatively active. Zero alone licenses skipping.
pub struct ActivityOp {
    image: ImageId,
    dtype: Dtype,
    stream: String,
    lifecycle: Lifecycle,
}
impl ActivityOp {
    pub fn new(
        image: ImageId,
        dtype: Dtype,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
    ) -> Self {
        Self {
            image,
            dtype,
            stream: stream.into(),
            lifecycle,
        }
    }
}
impl FragmentOp for ActivityOp {
    fn name(&self) -> &'static str {
        "mask activity"
    }
    fn reach(&self, _axis: usize, _len: usize) -> usize {
        0
    }
    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::new(self.image, Reach::symmetric([0; 3])).holding(self.dtype)]
    }
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }
    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(&self.stream, self.lifecycle, Coverage::EveryBlock)
                .sized(SidecarSize::fixed(9)),
        ]
    }
    fn prepares(&self) -> bool {
        true
    }
    fn prepare(&self, at: &PreparationView<'_>) -> Result<BlockPreparation> {
        Ok(match at.known_uniform(self.image.index(), at.core) {
            Some(value) => BlockPreparation::Complete {
                fragments: vec![(self.stream.clone(), encode(value != 0.0))],
                constant: None,
            },
            None => BlockPreparation::Run,
        })
    }
    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid("activity requires source pixels"))
    }
    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let buf = sources.get(self.image.index())?;
        // A simulation cannot assert that a real mask is empty.
        let active = match buf {
            crate::env::BlockBuf::Array(a) => {
                let a = a.widened();
                let o: [usize; 3] = std::array::from_fn(|i| at.core.start[i] - at.read.start[i]);
                let n = &at.core.shape;
                a.slice(ndarray::s![
                    o[0]..o[0] + n[0],
                    o[1]..o[1] + n[1],
                    o[2]..o[2] + n[2]
                ])
                .iter()
                .any(|v| *v != 0.0)
            }
            _ => true,
        };
        Ok(BlockOutput::fragment(&self.stream, encode(active)))
    }
}
