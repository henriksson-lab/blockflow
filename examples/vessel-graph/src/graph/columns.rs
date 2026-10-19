// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Per-vertex columns carried from one graph to the next.
//!
//! A column measured on the raw graph (a radius, an intensity) reaches each
//! later level by a different rule, and only two of them aggregate:
//!
//! | level | rule |
//! |---|---|
//! | cleaned vertex | a surviving voxel keeps its value; a clique centre takes a fold ([`Aggregate`]) over the clique's members, each column of a wide property folded on its own |
//! | reduced vertex | a **subset**: the vertex keeps the value it had in the cleaned graph, nothing is folded over the chain it ends |
//! | polyline point | a gather along the edge geometry, one value per point in path order |
//! | reduced edge | a fold over that same polyline, endpoints included ([`EdgeFold`]) |
//!
//! The four radius columns are carried independently, so after a clique fold
//! the scalar radius is in general **not** the norm of the axial one: the
//! maximum of the norms is not the norm of the per-axis maxima. Rebuilding one
//! from the other would be wrong at every clique whose members disagree.

use blockflow::error::{Error, Result};

use super::clean::{CleanGraph, CleanVertex};
use super::raw::RawGraph;
use super::reduce::ReducedGraph;

/// How a group of rows folds into one. `total_cmp`, so a NaN or a signed zero
/// gives the same answer in every order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregate {
    Max,
    Min,
}

impl Aggregate {
    pub fn fold(self, values: impl Iterator<Item = f64>) -> Option<f64> {
        values.reduce(|current, value| {
            let better = match self {
                Self::Max => value.total_cmp(&current).is_gt(),
                Self::Min => value.total_cmp(&current).is_lt(),
            };
            if better {
                value
            } else {
                current
            }
        })
    }
}

/// How a scalar vertex column folds onto a reduced edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeFold {
    Reduce(Aggregate),
    /// A 0/1 column becomes 1 when at least `fraction` of the polyline's
    /// points are 1. The default for the binary channel masks is two thirds.
    Vote {
        fraction: f64,
    },
}

impl EdgeFold {
    pub const BINARY: Self = Self::Vote {
        fraction: 2.0 / 3.0,
    };
}

/// A dense per-row property of `width` values per row.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    width: usize,
    values: Vec<f64>,
}

impl Column {
    pub fn new(width: usize, values: Vec<f64>) -> Result<Self> {
        if width == 0 {
            return Err(Error::invalid(
                "a property column of width 0 has no values.",
            ));
        }
        if !values.len().is_multiple_of(width) {
            return Err(Error::invalid(format!(
                "{} values do not divide into rows of {width}.",
                values.len()
            )));
        }
        Ok(Self { width, values })
    }

    pub fn scalar(values: Vec<f64>) -> Self {
        Self { width: 1, values }
    }

    pub fn rows3(rows: &[[f64; 3]]) -> Self {
        Self {
            width: 3,
            values: rows.iter().flatten().copied().collect(),
        }
    }

    pub fn flags(values: &[bool]) -> Self {
        Self::scalar(values.iter().map(|set| f64::from(u8::from(*set))).collect())
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn len(&self) -> usize {
        self.values.len() / self.width
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn row(&self, index: usize) -> &[f64] {
        &self.values[index * self.width..(index + 1) * self.width]
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn to_rows3(&self) -> Result<Vec<[f64; 3]>> {
        if self.width != 3 {
            return Err(Error::invalid(format!(
                "a column of width {} is not a list of 3-vectors.",
                self.width
            )));
        }
        Ok(self.values.as_chunks::<3>().0.to_vec())
    }

    /// The rows at `indices`, in that order.
    pub fn gather(&self, indices: &[u32]) -> Result<Self> {
        let mut values = Vec::with_capacity(indices.len() * self.width);
        for index in indices {
            let index = *index as usize;
            if index >= self.len() {
                return Err(Error::invalid(format!(
                    "a gather names row {index} of a column with {} of them.",
                    self.len()
                )));
            }
            values.extend_from_slice(self.row(index));
        }
        Ok(Self {
            width: self.width,
            values,
        })
    }

    pub fn as_flags(&self) -> Result<Vec<bool>> {
        if self.width != 1 {
            return Err(Error::invalid(format!(
                "a column of width {} is not a flag per row.",
                self.width
            )));
        }
        self.values
            .iter()
            .map(|value| match *value {
                0.0 => Ok(false),
                1.0 => Ok(true),
                other => Err(Error::invalid(format!(
                    "{other} is neither 0 nor 1, so this column is not a set of flags."
                ))),
            })
            .collect()
    }
}

/// A raw-vertex column carried onto the cleaned vertices: survivors keep their
/// row, a centre folds its clique's members column by column.
pub fn through_clean(
    raw: &RawGraph,
    cleaned: &CleanGraph,
    column: &Column,
    fold: Aggregate,
) -> Result<Column> {
    if column.len() != raw.n_vertices() {
        return Err(Error::invalid(format!(
            "a property column of {} rows was handed a raw graph of {} vertices.",
            column.len(),
            raw.n_vertices()
        )));
    }
    let anchors: Vec<u32> = cleaned
        .cliques()
        .iter()
        .map(|clique| clique.anchor())
        .collect();
    let width = column.width();
    let mut values = Vec::with_capacity(cleaned.n_vertices() * width);
    for (index, origin) in cleaned.origins().iter().enumerate() {
        match origin {
            CleanVertex::Voxel(at) => {
                let id = raw.id_of(*at).ok_or_else(|| {
                    Error::invalid(format!(
                        "cleaned vertex {index} is the voxel {at:?}, which is not a vertex of the \
                         raw graph it was handed."
                    ))
                })?;
                values.extend_from_slice(column.row(id as usize));
            }
            CleanVertex::Centre { anchor } => {
                let clique = anchors.binary_search(anchor).map_err(|_| {
                    Error::invalid(format!(
                        "cleaned vertex {index} is the centre of a clique anchored at {anchor}, \
                         and the graph has no such clique."
                    ))
                })?;
                let members = cleaned.cliques()[clique].members();
                for axis in 0..width {
                    let folded = fold
                        .fold(members.iter().map(|m| column.row(*m as usize)[axis]))
                        .ok_or_else(|| Error::invalid("a clique with no member"))?;
                    values.push(folded);
                }
            }
        }
    }
    Column::new(width, values)
}

/// A cleaned-vertex column onto the reduced vertices: a subset.
pub fn through_reduce(reduced: &ReducedGraph, column: &Column) -> Result<Column> {
    check_source(reduced, column)?;
    column.gather(reduced.sources())
}

/// A cleaned-vertex column gathered along every polyline point.
pub fn along_geometry(reduced: &ReducedGraph, column: &Column) -> Result<Column> {
    check_source(reduced, column)?;
    column.gather(reduced.require_geometry()?.points())
}

/// A scalar cleaned-vertex column folded onto every reduced edge over its
/// whole polyline.
pub fn onto_edges(reduced: &ReducedGraph, column: &Column, fold: EdgeFold) -> Result<Column> {
    check_source(reduced, column)?;
    if column.width() != 1 {
        return Err(Error::invalid(format!(
            "a column of width {} was handed to the vertex-to-edge fold, which folds scalars; \
             carry a wide column along the geometry instead.",
            column.width()
        )));
    }
    if let EdgeFold::Vote { .. } = fold {
        column.as_flags()?;
    }
    let geometry = reduced.require_geometry()?;
    let mut values = Vec::with_capacity(geometry.n_edges());
    for edge in 0..geometry.n_edges() {
        let path = geometry.path(edge);
        let along = path.iter().map(|point| column.values()[*point as usize]);
        values.push(match fold {
            EdgeFold::Reduce(aggregate) => aggregate
                .fold(along)
                .ok_or_else(|| Error::invalid(format!("edge {edge} has an empty polyline")))?,
            EdgeFold::Vote { fraction } => {
                let mean = along.sum::<f64>() / path.len() as f64;
                f64::from(u8::from(mean >= fraction))
            }
        });
    }
    Ok(Column::scalar(values))
}

fn check_source(reduced: &ReducedGraph, column: &Column) -> Result<()> {
    if column.len() != reduced.source_vertices() {
        return Err(Error::invalid(format!(
            "a property column of {} rows was handed a reduction of a {}-vertex graph.",
            column.len(),
            reduced.source_vertices()
        )));
    }
    Ok(())
}

/// The Euclidean norm of a row.
pub fn norm(row: &[f64]) -> f64 {
    row.iter().map(|value| value * value).sum::<f64>().sqrt()
}
