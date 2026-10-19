// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Artery/vein classification of the reduced graph's edges.
//!
//! Every array is per edge; `&`, `|` and `!` are elementwise:
//!
//! ```text
//! 1  artery = drop_small(artery_binary, arteries_min_noise_edges)
//! 2  restr  = restrictive_vein_mask(...)                    ; artery &= !restr
//! 3  perm   = vein_mask(restr, artery, permissive_vein_radius)
//! 4  artery = trace(artery, max_artery_trace_iter, artery_condition(perm))
//! 5  vein   = vein_mask(restr, artery, final_vein_radius)
//! 6  vein   = trace(vein, max_vein_trace_iter, vein_condition(dilate(artery, 1)))
//! 7  artery = drop_small(artery, min_artery_component_edges)
//! 8  vein   = drop_small(vein, min_vein_component_edges)
//! ```
//!
//! with
//!
//! ```text
//! restrictive_vein_mask = radius >= restrictive_vein_radius
//!                       & artery_intensity in [lo, hi]          if present
//!                       & vein_intensity >= vein_intensity_min  if both present
//! vein_mask(restr, artery, R) = (restr | radius >= R
//!                                | vein_intensity >= vein_intensity_min   if present
//!                                | vein_binary                            if present) & !artery
//! artery_condition(perm)[e] = distance_to_surface >= distance_to_surface_min  if present
//!                           & !perm & radius >= artery_trace_radius
//!                           & artery_intensity >= artery_intensity_min        if present
//! vein_condition(dil)[e]    = !dil & radius >= vein_trace_radius
//!                           & vein_intensity >= vein_intensity_min            if present
//!                           & artery_intensity in [lo, hi]                    if present
//! ```
//!
//! The things a plausible transcription gets wrong: the artery trace tests
//! against the *permissive* vein mask, fixed before tracing; the vein trace
//! is blocked by the arteries **dilated by one edge**, so an edge sharing a
//! vertex with an artery cannot become a vein; the small-component cleanup
//! runs on the line graph induced on the labelled edges, and a component of
//! exactly the minimum size is kept; every threshold is `>=`.
//!
//! The trace is not a flood fill: each round labels the frontier edges that
//! satisfy the condition, and an edge is tested once. Both conditions are pure
//! functions of the edge, so retesting would change nothing. The iteration cap
//! binds on real data, so the cap is part of the answer.
//!
//! Radii and thresholds are in µm by default ([`ClassifyParams::radius`]
//! names the edge column).

use std::collections::VecDeque;

use blockflow::error::{Error, Result};

use super::vessel::VesselGraph;

#[derive(Debug, Clone, PartialEq)]
pub struct ClassifyParams {
    /// The edge column read as the vessel radius.
    pub radius: String,
    /// Artery-channel intensities a vein may have.
    pub vein_intensity_range: (f64, f64),
    pub restrictive_vein_radius: f64,
    pub permissive_vein_radius: f64,
    pub final_vein_radius: f64,
    pub arteries_min_noise_edges: usize,
    pub artery_intensity_min: f64,
    pub artery_trace_radius: f64,
    pub vein_trace_radius: f64,
    pub vein_intensity_min: f64,
    pub distance_to_surface_min: f64,
    pub max_artery_trace_iter: usize,
    pub max_vein_trace_iter: usize,
    pub min_artery_component_edges: usize,
    pub min_vein_component_edges: usize,
}

impl Default for ClassifyParams {
    fn default() -> Self {
        Self {
            radius: "radius_units".to_string(),
            vein_intensity_range: (0.0, 2500.0),
            restrictive_vein_radius: 15.0,
            permissive_vein_radius: 16.25,
            final_vein_radius: 12.5,
            arteries_min_noise_edges: 2,
            artery_intensity_min: 200.0,
            artery_trace_radius: 6.4,
            vein_trace_radius: 8.0,
            vein_intensity_min: 150.0,
            distance_to_surface_min: 15.0,
            max_artery_trace_iter: 5,
            max_vein_trace_iter: 5,
            min_artery_component_edges: 30,
            min_vein_component_edges: 30,
        }
    }
}

/// The classifier's per-edge inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct VesselSignals {
    pub radii: Vec<f64>,
    pub artery_binary: Vec<bool>,
    pub artery_intensity: Option<Vec<f64>>,
    pub vein_binary: Option<Vec<bool>>,
    pub vein_intensity: Option<Vec<f64>>,
    pub distance_to_surface: Option<Vec<f64>>,
}

impl VesselSignals {
    /// The edge columns of a graph: `params.radius`, `artery_binary`
    /// (required), and `artery_raw`, `vein_binary`, `vein_raw`,
    /// `distance_to_surface` where present.
    pub fn from_graph(graph: &VesselGraph, params: &ClassifyParams) -> Result<Self> {
        let scalar = |name: &str| -> Option<Vec<f64>> {
            graph.edge(name).map(|column| column.values().to_vec())
        };
        let flags = |name: &str| -> Result<Option<Vec<bool>>> {
            graph.edge(name).map(|column| column.as_flags()).transpose()
        };
        let radii = scalar(&params.radius).ok_or_else(|| {
            Error::invalid(format!(
                "the graph has no edge column `{}` to classify by; edge columns are {:?}.",
                params.radius,
                graph.edge_names()
            ))
        })?;
        let artery_binary = flags("artery_binary")?.ok_or_else(|| {
            Error::invalid(
                "classification needs an `artery_binary` edge column: the arteries are traced \
                 from the edges an artery-channel mask covers.",
            )
        })?;
        let signals = Self {
            radii,
            artery_binary,
            artery_intensity: scalar("artery_raw"),
            vein_binary: flags("vein_binary")?,
            vein_intensity: scalar("vein_raw"),
            distance_to_surface: scalar("distance_to_surface"),
        };
        signals.check(graph.n_edges())?;
        Ok(signals)
    }

    pub fn len(&self) -> usize {
        self.radii.len()
    }

    pub fn is_empty(&self) -> bool {
        self.radii.is_empty()
    }

    fn check(&self, n_edges: usize) -> Result<()> {
        let lengths = [
            ("radii", Some(self.radii.len())),
            ("artery_binary", Some(self.artery_binary.len())),
            (
                "artery_intensity",
                self.artery_intensity.as_ref().map(Vec::len),
            ),
            ("vein_binary", self.vein_binary.as_ref().map(Vec::len)),
            ("vein_intensity", self.vein_intensity.as_ref().map(Vec::len)),
            (
                "distance_to_surface",
                self.distance_to_surface.as_ref().map(Vec::len),
            ),
        ];
        for (name, len) in lengths {
            if let Some(len) = len {
                if len != n_edges {
                    return Err(Error::invalid(format!(
                        "the signal `{name}` has {len} values and the graph has {n_edges} edges."
                    )));
                }
            }
        }
        Ok(())
    }
}

/// The line graph: one vertex per edge, adjacent when two edges share an
/// endpoint. A line vertex is numbered by its edge id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineGraph {
    offsets: Vec<u32>,
    neighbours: Vec<u32>,
}

impl LineGraph {
    pub fn of(edges: &[[u32; 2]], n_vertices: usize) -> Result<Self> {
        if u32::try_from(edges.len()).is_err() {
            return Err(Error::invalid(format!(
                "{} edges do not fit a u32 line-graph vertex id.",
                edges.len()
            )));
        }
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n_vertices];
        for (edge, ends) in edges.iter().enumerate() {
            for end in ends {
                let end = *end as usize;
                if end >= n_vertices {
                    return Err(Error::invalid(format!(
                        "edge {edge} names vertex {end} of a {n_vertices}-vertex graph."
                    )));
                }
                incident[end].push(edge as u32);
            }
        }
        let mut offsets = vec![0u32];
        let mut neighbours = Vec::new();
        let mut mine = Vec::new();
        for (edge, ends) in edges.iter().enumerate() {
            mine.clear();
            for end in ends {
                mine.extend_from_slice(&incident[*end as usize]);
            }
            mine.sort_unstable();
            mine.dedup();
            neighbours.extend(mine.iter().copied().filter(|other| *other != edge as u32));
            offsets.push(neighbours.len() as u32);
        }
        Ok(Self {
            offsets,
            neighbours,
        })
    }

    pub fn n_edges(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn neighbours(&self, edge: usize) -> &[u32] {
        &self.neighbours[self.offsets[edge] as usize..self.offsets[edge + 1] as usize]
    }

    /// `label` grown by `steps` line-graph neighbours.
    pub fn dilate(&self, label: &[bool], steps: usize) -> Result<Vec<bool>> {
        self.check_len(label)?;
        let mut out = label.to_vec();
        for _ in 0..steps {
            let previous = out.clone();
            for edge in 0..self.n_edges() {
                if previous[edge] {
                    for other in self.neighbours(edge) {
                        out[*other as usize] = true;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Components of the line graph induced on the labelled edges: a
    /// component id per labelled edge, and the size of each component.
    pub fn components(&self, label: &[bool]) -> Result<(Vec<Option<u32>>, Vec<usize>)> {
        self.check_len(label)?;
        let mut component = vec![None; self.n_edges()];
        let mut counts = Vec::new();
        let mut queue = VecDeque::new();
        for start in 0..self.n_edges() {
            if !label[start] || component[start].is_some() {
                continue;
            }
            let id = counts.len() as u32;
            let mut count = 0usize;
            component[start] = Some(id);
            queue.push_back(start);
            while let Some(edge) = queue.pop_front() {
                count += 1;
                for other in self.neighbours(edge) {
                    let other = *other as usize;
                    if label[other] && component[other].is_none() {
                        component[other] = Some(id);
                        queue.push_back(other);
                    }
                }
            }
            counts.push(count);
        }
        Ok((component, counts))
    }

    fn check_len(&self, label: &[bool]) -> Result<()> {
        if label.len() != self.n_edges() {
            return Err(Error::invalid(format!(
                "a label of {} entries on a line graph of {} edges.",
                label.len(),
                self.n_edges()
            )));
        }
        Ok(())
    }
}

/// The classification and every intermediate mask, for inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifyTrace {
    pub pre_filter: Vec<bool>,
    pub restrictive: Vec<bool>,
    pub artery_minus_restrictive: Vec<bool>,
    pub permissive: Vec<bool>,
    pub artery_traced: Vec<bool>,
    pub vein_mask_final: Vec<bool>,
    pub vein_traced: Vec<bool>,
    pub artery: Vec<bool>,
    pub vein: Vec<bool>,
}

impl ClassifyTrace {
    pub fn arteries(&self) -> usize {
        self.artery.iter().filter(|value| **value).count()
    }

    pub fn veins(&self) -> usize {
        self.vein.iter().filter(|value| **value).count()
    }
}

/// Run the eight steps.
pub fn classify_edges(
    line: &LineGraph,
    signals: &VesselSignals,
    params: &ClassifyParams,
) -> Result<ClassifyTrace> {
    signals.check(line.n_edges())?;
    let pre_filter = drop_small_components(
        line,
        &signals.artery_binary,
        params.arteries_min_noise_edges,
    )?;
    let restrictive = restrictive_vein_mask(signals, params);
    let artery_minus_restrictive: Vec<bool> = pre_filter
        .iter()
        .zip(&restrictive)
        .map(|(artery, vein)| *artery && !*vein)
        .collect();
    let permissive = vein_mask(
        signals,
        params,
        &restrictive,
        &artery_minus_restrictive,
        params.permissive_vein_radius,
    );
    let artery_traced = trace(
        line,
        &artery_minus_restrictive,
        params.max_artery_trace_iter,
        |edge| {
            if let Some(distance) = &signals.distance_to_surface {
                if distance[edge] < params.distance_to_surface_min {
                    return false;
                }
            }
            if permissive[edge] || signals.radii[edge] < params.artery_trace_radius {
                return false;
            }
            match &signals.artery_intensity {
                Some(intensity) => intensity[edge] >= params.artery_intensity_min,
                None => true,
            }
        },
    )?;
    let vein_mask_final = vein_mask(
        signals,
        params,
        &restrictive,
        &artery_traced,
        params.final_vein_radius,
    );
    let blocked = line.dilate(&artery_traced, 1)?;
    let (lo, hi) = params.vein_intensity_range;
    let vein_traced = trace(line, &vein_mask_final, params.max_vein_trace_iter, |edge| {
        if blocked[edge] || signals.radii[edge] < params.vein_trace_radius {
            return false;
        }
        if let Some(vein) = &signals.vein_intensity {
            if vein[edge] < params.vein_intensity_min {
                return false;
            }
        }
        if let Some(artery) = &signals.artery_intensity {
            if !(lo..=hi).contains(&artery[edge]) {
                return false;
            }
        }
        true
    })?;
    let artery = drop_small_components(line, &artery_traced, params.min_artery_component_edges)?;
    let vein = drop_small_components(line, &vein_traced, params.min_vein_component_edges)?;
    Ok(ClassifyTrace {
        pre_filter,
        restrictive,
        artery_minus_restrictive,
        permissive,
        artery_traced,
        vein_mask_final,
        vein_traced,
        artery,
        vein,
    })
}

fn restrictive_vein_mask(signals: &VesselSignals, params: &ClassifyParams) -> Vec<bool> {
    let (lo, hi) = params.vein_intensity_range;
    (0..signals.len())
        .map(|edge| {
            let large = signals.radii[edge] >= params.restrictive_vein_radius;
            match (&signals.artery_intensity, &signals.vein_intensity) {
                (None, _) => large,
                (Some(artery), None) => (lo..=hi).contains(&artery[edge]) && large,
                (Some(artery), Some(vein)) => {
                    (lo..=hi).contains(&artery[edge])
                        && vein[edge] >= params.vein_intensity_min
                        && large
                }
            }
        })
        .collect()
}

fn vein_mask(
    signals: &VesselSignals,
    params: &ClassifyParams,
    restrictive: &[bool],
    artery: &[bool],
    radius: f64,
) -> Vec<bool> {
    (0..signals.len())
        .map(|edge| {
            let mut seed = restrictive[edge] || signals.radii[edge] >= radius;
            if let Some(vein) = &signals.vein_intensity {
                seed |= vein[edge] >= params.vein_intensity_min;
            }
            if let Some(vein) = &signals.vein_binary {
                seed |= vein[edge];
            }
            seed && !artery[edge]
        })
        .collect()
}

fn trace(
    line: &LineGraph,
    seed: &[bool],
    max_iterations: usize,
    condition: impl Fn(usize) -> bool,
) -> Result<Vec<bool>> {
    line.check_len(seed)?;
    let mut label = seed.to_vec();
    let mut unchecked: Vec<bool> = seed.iter().map(|value| !*value).collect();
    for _ in 0..max_iterations {
        let dilated = line.dilate(&label, 1)?;
        let frontier: Vec<usize> = (0..line.n_edges())
            .filter(|edge| !label[*edge] && dilated[*edge] && unchecked[*edge])
            .collect();
        if frontier.is_empty() {
            break;
        }
        for edge in frontier {
            if condition(edge) {
                label[edge] = true;
            }
            unchecked[edge] = false;
        }
    }
    Ok(label)
}

/// Unlabel every labelled component of fewer than `min_edges` edges.
pub fn drop_small_components(
    line: &LineGraph,
    label: &[bool],
    min_edges: usize,
) -> Result<Vec<bool>> {
    let (component, counts) = line.components(label)?;
    Ok(label
        .iter()
        .zip(&component)
        .map(|(set, id)| *set && id.is_some_and(|id| counts[id as usize] >= min_edges))
        .collect())
}

/// Classify a graph's edges and store the result as `artery` and `vein` edge
/// flag columns.
pub fn classify(graph: &mut VesselGraph, params: &ClassifyParams) -> Result<ClassifyTrace> {
    let signals = VesselSignals::from_graph(graph, params)?;
    let line = LineGraph::of(graph.edges(), graph.n_vertices())?;
    let trace = classify_edges(&line, &signals, params)?;
    graph.set_edge("artery", super::columns::Column::flags(&trace.artery))?;
    graph.set_edge("vein", super::columns::Column::flags(&trace.vein))?;
    Ok(trace)
}
