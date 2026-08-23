// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// JSON, by hand, because the crate has `serde_json` and not `serde` — and
// because what travels here is small, closed and worth reading in the source
// rather than inferring from derives.
//
// The one rule this file exists to keep: **nothing in the protocol names an
// element type.** A message carries block indices, phases, regions and counts.
// A `Decomposition` carries a `Dtype` tag because byte accounting needs one,
// and that is a *width*, not an element type — no message ever carries a voxel.
// The pending dtype work rewrites `apply` and the environment; if it ever
// forced a change here, that would be a sign the protocol had acquired a
// dependency it should not have.

use serde_json::{json, Value};

use crate::decomposition::{Decomposition, PhaseDecomposition};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::reach::Reach;
use crate::region::Region;

// ------------------------------------------------------------- reading --

pub fn get<'a>(object: &'a Value, name: &str) -> Result<&'a Value> {
    object
        .get(name)
        .ok_or_else(|| Error::invalid(format!("message has no {name:?} field")))
}

pub fn text(object: &Value, name: &str) -> Result<String> {
    get(object, name)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::invalid(format!("{name:?} is not a string")))
}

pub fn text_or(object: &Value, name: &str, fallback: &str) -> String {
    object
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or(fallback)
        .to_string()
}

pub fn number(object: &Value, name: &str) -> Result<u64> {
    get(object, name)?
        .as_u64()
        .ok_or_else(|| Error::invalid(format!("{name:?} is not a whole number")))
}

pub fn number_or(object: &Value, name: &str, fallback: u64) -> u64 {
    object.get(name).and_then(Value::as_u64).unwrap_or(fallback)
}

pub fn count(object: &Value, name: &str) -> Result<usize> {
    Ok(number(object, name)? as usize)
}

pub fn real_or(object: &Value, name: &str, fallback: f64) -> f64 {
    object.get(name).and_then(Value::as_f64).unwrap_or(fallback)
}

pub fn flag(object: &Value, name: &str, fallback: bool) -> bool {
    object
        .get(name)
        .and_then(Value::as_bool)
        .unwrap_or(fallback)
}

pub fn triple(object: &Value, name: &str) -> Result<[usize; 3]> {
    let array = get(object, name)?
        .as_array()
        .ok_or_else(|| Error::invalid(format!("{name:?} is not an array")))?;
    if array.len() != 3 {
        return Err(Error::invalid(format!(
            "{name:?} has {} entries, and this crate is three-dimensional",
            array.len()
        )));
    }
    let mut out = [0usize; 3];
    for (axis, entry) in array.iter().enumerate() {
        out[axis] = entry
            .as_u64()
            .ok_or_else(|| Error::invalid(format!("{name:?}[{axis}] is not a whole number")))?
            as usize;
    }
    Ok(out)
}

pub fn triple_or(object: &Value, name: &str, fallback: [usize; 3]) -> [usize; 3] {
    triple(object, name).unwrap_or(fallback)
}

pub fn counts(object: &Value, name: &str) -> Result<Vec<usize>> {
    get(object, name)?
        .as_array()
        .ok_or_else(|| Error::invalid(format!("{name:?} is not an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| Error::invalid(format!("{name:?} holds a non-number")))
        })
        .collect()
}

pub fn strings(object: &Value, name: &str) -> Result<Vec<String>> {
    get(object, name)?
        .as_array()
        .ok_or_else(|| Error::invalid(format!("{name:?} is not an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::invalid(format!("{name:?} holds a non-string")))
        })
        .collect()
}

pub fn array<'a>(object: &'a Value, name: &str) -> Result<&'a Vec<Value>> {
    get(object, name)?
        .as_array()
        .ok_or_else(|| Error::invalid(format!("{name:?} is not an array")))
}

// ------------------------------------------------------------ optional --
// A duration that may not exist, and where **not existing is the answer** —
// not a stand-in for a number nobody supplied.
//
// The one such field is the claim lease, whose default is that there is no
// lease at all (see `coordinator`, and the module header's "nodes do not
// die"). Encoding "no expiry" as a very large number of milliseconds would put
// a value on the wire that arithmetic can overflow and that a reader has to
// recognise by magnitude; `null` is the same statement with nothing to get
// wrong. Absent reads as `None` for the same reason: a message that never
// mentions a lease is a message asking for none, which is the default anyway,
// so the two spellings agree rather than diverging.

/// Milliseconds, or nothing. `null`, absent and a non-number all read as
/// `None`; only an actual number is a duration.
pub fn millis_or_none(object: &Value, name: &str) -> Option<std::time::Duration> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .map(std::time::Duration::from_millis)
}

/// The other half, so the two spellings cannot drift apart.
pub fn millis_json(duration: Option<std::time::Duration>) -> Value {
    match duration {
        Some(duration) => json!(duration.as_millis() as u64),
        None => Value::Null,
    }
}

// ------------------------------------------------------------- regions --

pub fn region_json(region: &Region) -> Value {
    json!({"start": region.start, "shape": region.shape})
}

pub fn region_at(object: &Value, name: &str) -> Result<Region> {
    let inner = get(object, name)?;
    let start = triple(inner, "start")?;
    let shape = triple(inner, "shape")?;
    Ok(Region::new(&start, &shape))
}

// ------------------------------------------------------- decomposition --
//
// A decomposition is **binding**, so it is the one thing a worker must be given
// rather than allowed to derive: "workers receive certainty". It is also
// entirely integers, which is why it can travel at all.
//
// What crosses the wire is the *generators* — slots, names, reach, halo, and
// the grid — and not the per-block geometry they imply. Two reasons, and the
// second is the important one:
//
// * size. Five thousand blocks of three regions each is megabytes of JSON to
//   say what four triples already say.
// * **the derivation is the contract.** `PhaseDecomposition::derive` is what
//   turns halo and reach into read extents and valid regions, and it is where
//   the halo-as-hint inversion lives. Shipping its *output* would let a worker
//   run geometry the coordinator's own rules did not produce. Shipping its
//   input means both sides run the same function.
//
// The fingerprint is carried alongside and checked after rebuilding, so
// "both sides run the same function" is verified rather than assumed — a
// version skew between coordinator and worker shows up as a refusal to start,
// not as a subtly different seam.

pub fn decomposition_json(decomposition: &Decomposition) -> Result<Value> {
    // A per-block fetch region is the one thing here that is *not* a generator:
    // it is an arbitrary mapping, and a receiver cannot re-derive it by running
    // the same function. Shipping the regions themselves would break the rule
    // this module is built on, so a plan that uses them is refused rather than
    // silently flattened — which the fingerprint would catch on the far end
    // anyway, as a mystery instead of a sentence.
    for (index, phase) in decomposition.phases.iter().enumerate() {
        if phase.reads_across_grids() {
            return Err(Error::invalid(format!(
                "phase {index} reads across grids, and a per-block fetch region is not a \
                 generator the far end can rebuild by running the same derivation. Sending \
                 the regions would put geometry on the wire that neither end derived; \
                 distributing such a plan needs the mapping to be expressible as a rule, \
                 which is what the op-mandated lattice work is for."
            )));
        }
    }
    let phases: Vec<Value> = decomposition
        .phases
        .iter()
        .map(|phase| {
            let mut entry = json!({
                "slots": phase.slots,
                "names": phase.names,
                "reach": phase.reach.to_json(),
                "halo": phase.halo.to_json(),
                "block": phase.grid.block(),
            });
            // Written only when the phase changes it, so a plan over one volume
            // and one element type is the document it was before a phase could
            // own either.
            if phase.volume() != decomposition.volume {
                entry["volume"] = json!(phase.volume());
            }
            if let Some(dtype) = phase.dtype {
                entry["dtype"] = json!(dtype.numpy_name());
            }
            // Likewise for the images a phase reads besides its own input: a
            // generator like the rest — the far end re-derives the geometry, and
            // *which images* is a fact only the plan carries — and absent for
            // every plan that reads one image, which is the document this was
            // before source leaves existed.
            if !phase.source_images.is_empty() {
                entry["source_levels"] = json!(phase.source_images);
            }
            // **Not a generator: a barrier cannot be re-derived at all.** The
            // far end rebuilds the geometry by running the same derivation over
            // the same numbers, and every other field here is an input to that.
            // This one is not derived from anything — it comes from the op's
            // `FragmentOp::barrier`, and the far end does not hold the ops when
            // it builds the plan. Dropped, it would produce a plan that runs,
            // answers from however complete a fragment set the schedule left,
            // and disagrees with the sender's fingerprint. Written only when it
            // is set, so a plan with no barrier is the document it was before
            // barriers existed.
            if phase.barrier {
                entry["barrier"] = json!(true);
            }
            entry
        })
        .collect();
    Ok(json!({
        "volume": decomposition.volume,
        "dtype": decomposition.dtype.numpy_name(),
        "chain_reach": decomposition.chain_reach,
        "fingerprint": decomposition.fingerprint().to_string(),
        "phases": phases,
    }))
}

pub fn decomposition_from_json(value: &Value) -> Result<Decomposition> {
    let volume = triple(value, "volume")?;
    let name = text(value, "dtype")?;
    let dtype = Dtype::from_numpy_name(&name).ok_or_else(|| {
        Error::invalid(format!("{name:?} is not an element type this build knows"))
    })?;
    let mut phases = Vec::new();
    for phase in array(value, "phases")? {
        let block = triple(phase, "block")?;
        // Absent means "the same as the image below", which is what a phase
        // that does not change shape or element type says by saying nothing.
        let phase_volume = match phase.get("volume") {
            Some(_) => triple(phase, "volume")?,
            None => volume,
        };
        // A reach is a triple on the wire when a triple is all it says, and an
        // object when it says more. `Reach::from_json` reads both, so a
        // document written before the richer forms existed rebuilds unchanged.
        let mut rebuilt = PhaseDecomposition::derive(
            counts(phase, "slots")?,
            strings(phase, "names")?,
            Reach::from_json(get(phase, "reach")?)?,
            Reach::from_json(get(phase, "halo")?)?,
            BlockGrid::new(phase_volume, block)?,
        );
        if phase.get("dtype").is_some() {
            let name = text(phase, "dtype")?;
            rebuilt = rebuilt.with_dtype(Dtype::from_numpy_name(&name).ok_or_else(|| {
                Error::invalid(format!("{name:?} is not an element type this build knows"))
            })?);
        }
        if phase.get("source_levels").is_some() {
            rebuilt = rebuilt.with_source_images(counts(phase, "source_levels")?);
        }
        if let Some(value) = phase.get("barrier") {
            rebuilt = rebuilt.with_barrier(value.as_bool().ok_or_else(|| {
                Error::invalid(format!("phase barrier is {value}, which is not a boolean"))
            })?);
        }
        phases.push(rebuilt);
    }
    let decomposition = Decomposition {
        volume,
        dtype,
        phases,
        chain_reach: triple(value, "chain_reach")?,
    };
    if let Ok(claimed) = text(value, "fingerprint") {
        let rebuilt = decomposition.fingerprint().to_string();
        if claimed != rebuilt {
            return Err(Error::invalid(format!(
                "the decomposition rebuilt from the wire fingerprints {rebuilt}, but the \
                 sender said {claimed}. The two ends derive block geometry from the same \
                 four triples, so a disagreement means the two ends are different builds. \
                 Refusing to start: a worker running geometry the coordinator did not \
                 choose would seam differently and the output would be wrong rather than \
                 slow."
            )));
        }
    }
    Ok(decomposition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::Chain;
    use crate::probes::IdentityOp;
    use crate::strategy::{Enumerating, Strategy, Workflow};

    fn a_decomposition() -> Decomposition {
        let chain = Chain::sequence(vec![
            Chain::op(IdentityOp::new("a", [3, 0, 0])),
            Chain::op(IdentityOp::new("b", [0, 0, 2]).with_order([2, 1, 0])),
        ]);
        let workflow = Workflow::new(chain, [64, 16, 16], Dtype::U16);
        let constraints = crate::decomposition::Constraints {
            block_candidates: vec![16, 32],
            split_axes: vec![0],
            ..Default::default()
        };
        Enumerating::default()
            .decompose(&workflow, &constraints)
            .unwrap()
    }

    #[test]
    fn a_decomposition_survives_the_wire_exactly() {
        let original = a_decomposition();
        let rebuilt = decomposition_from_json(&decomposition_json(&original).unwrap()).unwrap();
        assert_eq!(rebuilt, original);
        assert_eq!(rebuilt.fingerprint(), original.fingerprint());
    }

    #[test]
    fn a_fingerprint_that_does_not_match_is_refused_rather_than_run() {
        // The version-skew case, forged: the geometry says one thing and the
        // sender claims another. A worker must not quietly run either.
        let mut document = decomposition_json(&a_decomposition()).unwrap();
        document["fingerprint"] = json!("12345");
        let error = decomposition_from_json(&document).unwrap_err();
        assert!(error.to_string().contains("different builds"), "{error}");
    }

    /// The images a phase reads besides its own input are part of the plan, so
    /// they cross the wire — and a plan that reads one image says nothing, so
    /// the document is the one it was before source leaves existed.
    #[test]
    fn the_images_a_phase_also_reads_survive_the_wire_and_are_absent_without_them() {
        let plain = a_decomposition();
        assert!(decomposition_json(&plain).unwrap()["phases"][0]
            .get("source_levels")
            .is_none());

        let mut original = plain.clone();
        let last = original.phases.len() - 1;
        original.phases[last] = original.phases[last].clone().with_source_images([0]);
        let document = decomposition_json(&original).unwrap();
        assert_eq!(document["phases"][last]["source_levels"], json!([0]));

        let rebuilt = decomposition_from_json(&document).unwrap();
        assert_eq!(rebuilt, original);
        assert_eq!(rebuilt.fingerprint(), original.fingerprint());
        // and it really is a different plan from the one that reads one image
        assert_ne!(original.fingerprint(), plain.fingerprint());
    }

    #[test]
    fn regions_survive_the_wire() {
        let region = Region::new(&[1, 2, 3], &[4, 5, 6]);
        let wrapped = json!({"read": region_json(&region)});
        assert_eq!(region_at(&wrapped, "read").unwrap(), region);
    }

    #[test]
    fn a_missing_field_says_which_one() {
        let error = triple(&json!({}), "volume").unwrap_err();
        assert!(error.to_string().contains("\"volume\""), "{error}");
        let error = triple(&json!({"volume": [1, 2]}), "volume").unwrap_err();
        assert!(error.to_string().contains("three-dimensional"), "{error}");
    }
}
