// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The small dense algebra both placements share: a grounded-Laplacian solve
//! and connected components over an edge list.
//!
//! Both placement steps fit positions to pairwise displacements by least
//! squares, `min sum_ij (x_j - x_i - d_ij)^2`, with one node grounded. With the
//! node grounded and the component connected, the incidence matrix has full
//! column rank, so the normal equations `M^T M x = M^T d` (a grounded graph
//! Laplacian, positive definite) have the same solution as the pseudo-inverse.
//! The systems are one unknown per tile (rigid) or per cluster of tiles in one
//! slice (wobbly), so a dense elimination is enough.

use blockflow::error::{Error, Result};

/// Gaussian elimination with partial pivoting on a dense square system.
pub fn solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    for column in 0..n {
        let mut pivot = column;
        for row in column + 1..n {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][column].abs() <= f64::EPSILON {
            return Err(Error::invalid(format!(
                "the placement's normal equations are singular at column {column} of {n}: the \
                 graph handed to the solver is not connected through its grounded node"
            )));
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let pivot_row = matrix[column].clone();
        let pivot_rhs = rhs[column];
        let diagonal = pivot_row[column];
        for row in column + 1..n {
            let factor = matrix[row][column] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for (target, source) in matrix[row].iter_mut().zip(&pivot_row).skip(column) {
                *target -= factor * source;
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for column in row + 1..n {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    Ok(solution)
}

/// Least squares over one connected graph of `nodes` nodes: `edges[k] = (pre,
/// post)` asks for `x[post] - x[pre] = displacement[k]` on every dimension.
/// Node 0 is grounded at zero; the caller translates afterwards.
pub fn fit_positions(
    nodes: usize,
    edges: &[(usize, usize)],
    displacements: &[Vec<f64>],
    ndim: usize,
) -> Result<Vec<Vec<f64>>> {
    let unknowns = nodes.saturating_sub(1);
    let mut fitted = vec![vec![0.0; ndim]; nodes];
    if unknowns == 0 {
        return Ok(fitted);
    }
    for dim in 0..ndim {
        let mut normal = vec![vec![0.0; unknowns]; unknowns];
        let mut rhs = vec![0.0; unknowns];
        for (&(pre, post), displacement) in edges.iter().zip(displacements) {
            // One row of the incidence matrix: -1 at `pre`, +1 at `post`, and
            // nothing for the grounded node.
            let mut entries = Vec::with_capacity(2);
            if pre > 0 {
                entries.push((pre - 1, -1.0));
            }
            if post > 0 {
                entries.push((post - 1, 1.0));
            }
            for &(row, row_value) in &entries {
                rhs[row] += row_value * displacement[dim];
                for &(column, column_value) in &entries {
                    normal[row][column] += row_value * column_value;
                }
            }
        }
        for (node, value) in solve(normal, rhs)?.into_iter().enumerate() {
            fitted[node + 1][dim] = value;
        }
    }
    Ok(fitted)
}

/// Connected components of `edges` over `n` nodes, labelled in increasing order
/// of their lowest member.
pub fn connected_components(edges: &[(usize, usize)], n: usize) -> Vec<usize> {
    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    let mut parent = (0..n).collect::<Vec<_>>();
    for &(a, b) in edges {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let mut label_of: Vec<Option<usize>> = vec![None; n];
    let mut labels = vec![0usize; n];
    let mut next = 0;
    for (node, slot) in labels.iter_mut().enumerate() {
        let root = find(&mut parent, node);
        *slot = *label_of[root].get_or_insert_with(|| {
            next += 1;
            next - 1
        });
    }
    labels
}

/// Half to even, the rule every rounding of a fitted position here uses.
pub fn round_even(value: f64) -> isize {
    value.round_ties_even() as isize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_is_fitted_exactly() {
        let fitted = fit_positions(3, &[(0, 1), (1, 2)], &[vec![4.0, -1.0], vec![2.0, 3.0]], 2)
            .expect("a fit");
        assert_eq!(
            fitted,
            vec![vec![0.0, 0.0], vec![4.0, -1.0], vec![6.0, 2.0]]
        );
    }

    #[test]
    fn components_are_labelled_by_their_lowest_member() {
        assert_eq!(
            connected_components(&[(3, 1), (2, 4)], 5),
            vec![0, 1, 2, 1, 2]
        );
    }

    #[test]
    fn a_disconnected_system_is_refused() {
        let error = solve(vec![vec![0.0]], vec![1.0]).expect_err("singular");
        assert!(error.to_string().contains("singular"), "{error}");
    }
}
