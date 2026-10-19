// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Binary smoothing by configuration: every voxel's 3x3x3 neighbourhood is a
//! 27-bit index into a table that says whether the voxel is set afterwards.
//!
//! The pass itself is `blockflow::ops::configuration::ConfigurationPassOp`;
//! what lives here is the **rule** — the table — derived from its definition.
//! A set voxel is cleared when it is nearly isolated or when its neighbourhood
//! matches one of a handful of thin-protrusion templates under their
//! symmetries; a clear voxel is set when it is nearly enclosed, bridges an
//! opposite pair of faces, or when the complement of its neighbourhood matches
//! a deletion template. That removes spurs and fills pinholes without changing
//! the mask's topology at the scale of one voxel.
//!
//! Every clause is a *template* — these neighbours set, these clear, the rest
//! free — so the 2^27-entry table is written by `assign_matching` one template
//! at a time rather than by evaluating a predicate 134 million times. The most
//! expensive clause ("at least twenty of twenty-six set") is 313,912 templates
//! and still a fraction of a second.

use std::sync::{Arc, OnceLock};

use blockflow::op::Chain;
use blockflow::ops::configuration::{
    ConfigurationPassOp, ConfigurationTable, ConfigurationTemplate,
};
use blockflow::{Error, Result};

/// Voxels in a 3x3x3 neighbourhood.
const POSITIONS: usize = 27;

/// `x + 3y + 9z` with `x` the fastest term: `ops::configuration`'s bit numbering,
/// which a test pins at all 27 positions.
type Position = usize;

const CENTRE: Position = 13;

fn at(x: usize, y: usize, z: usize) -> Position {
    x + 3 * y + 9 * z
}

fn coordinates(position: Position) -> [usize; 3] {
    [position % 3, (position / 3) % 3, position / 9]
}

#[cfg(test)]
fn offset_of(position: Position) -> [isize; 3] {
    let [x, y, z] = coordinates(position);
    [x as isize - 1, y as isize - 1, z as isize - 1]
}

/// One face of the cube: every position whose `axis` coordinate is `value`.
fn plane(axis: usize, value: usize) -> Vec<Position> {
    (0..POSITIONS)
        .filter(|position| coordinates(*position)[axis] == value)
        .collect()
}

/// The six positions sharing a face with the centre.
fn face_neighbours() -> Vec<Position> {
    (0..POSITIONS)
        .filter(|position| {
            coordinates(*position)
                .iter()
                .filter(|value| **value != 1)
                .count()
                == 1
        })
        .collect()
}

/// The three pairs of face neighbours opposite one another through the centre.
fn opposite_pairs() -> [[Position; 2]; 3] {
    [
        [at(0, 1, 1), at(2, 1, 1)],
        [at(1, 0, 1), at(1, 2, 1)],
        [at(1, 1, 0), at(1, 1, 2)],
    ]
}

/// A rigid motion of the cube, as **where each position reads from**: `moved[p]`
/// is the position whose value ends up at `p`. That is the direction a template
/// needs, since a pattern matched against a rotated neighbourhood is the
/// pattern's positions mapped through this.
type Symmetry = [Position; POSITIONS];

fn identity() -> Symmetry {
    std::array::from_fn(|position| position)
}

/// A quarter turn about `axis`, right-handed in the two axes it moves.
fn quarter_turn(axis: usize) -> Symmetry {
    let (first, second) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    std::array::from_fn(|position| {
        let mut source = coordinates(position);
        source[first] = 2 - coordinates(position)[second];
        source[second] = coordinates(position)[first];
        at(source[0], source[1], source[2])
    })
}

/// The point reflection through the centre. Not a rotation, and that matters:
/// the templates are chiral, so a family built from rotations alone would be a
/// different rule that looked right.
fn inversion() -> Symmetry {
    std::array::from_fn(|position| {
        let [x, y, z] = coordinates(position);
        at(2 - x, 2 - y, 2 - z)
    })
}

fn compose(outer: &Symmetry, inner: &Symmetry) -> Symmetry {
    std::array::from_fn(|position| outer[inner[position]])
}

fn turns(axis: usize, quarter: usize) -> Symmetry {
    let step = quarter_turn(axis);
    let mut symmetry = identity();
    for _ in 0..quarter {
        symmetry = compose(&step, &symmetry);
    }
    symmetry
}

fn with_inversions(family: Vec<Symmetry>) -> Vec<Symmetry> {
    let mut all = family.clone();
    for symmetry in &family {
        all.push(compose(symmetry, &inversion()));
    }
    all
}

/// The six images of a face direction.
fn face_symmetries() -> Vec<Symmetry> {
    with_inversions(vec![identity(), turns(0, 1), turns(1, 1)])
}

/// The twelve images of an edge direction.
fn edge_symmetries() -> Vec<Symmetry> {
    with_inversions(vec![
        identity(),
        turns(2, 1),
        turns(2, 2),
        turns(2, 3),
        turns(1, 3),
        turns(1, 1),
    ])
}

/// The eight images of a corner direction.
fn corner_symmetries() -> Vec<Symmetry> {
    with_inversions(vec![identity(), turns(2, 1), turns(2, 2), turns(2, 3)])
}

/// The twenty-four images of a corner-on-a-face.
fn corner_face_symmetries() -> Vec<Symmetry> {
    let about_axis: Vec<Symmetry> = (0..4).map(|quarter| turns(2, quarter)).collect();
    let mut family = about_axis.clone();
    for onto in [turns(0, 1), turns(1, 1)] {
        for symmetry in &about_axis {
            family.push(compose(symmetry, &onto));
        }
    }
    with_inversions(family)
}

/// One clause of the rule: these positions set, these clear, the rest free.
#[derive(Debug, Clone, Default)]
struct Pattern {
    set: Vec<Position>,
    clear: Vec<Position>,
}

impl Pattern {
    fn set(mut self, positions: impl IntoIterator<Item = Position>) -> Self {
        self.set.extend(positions);
        self
    }

    fn clear(mut self, positions: impl IntoIterator<Item = Position>) -> Self {
        self.clear.extend(positions);
        self
    }

    /// The same clause, seen through a rigid motion.
    fn under(&self, symmetry: &Symmetry) -> ConfigurationTemplate {
        let mask = |positions: &[Position]| -> u32 {
            positions
                .iter()
                .fold(0, |all, position| all | (1u32 << symmetry[*position]))
        };
        ConfigurationTemplate::from_masks(mask(&self.set), mask(&self.clear))
            .expect("a pattern names each position once and a symmetry is a bijection")
    }
}

/// The centre and its `+z` neighbour set, everything at or below the centre's
/// own plane clear except the `+y` row, which the three `U` templates differ on.
fn u_prefix() -> Pattern {
    Pattern::default()
        .set([CENTRE, at(1, 1, 2)])
        .clear(plane(2, 0))
        .clear([
            at(0, 0, 1),
            at(1, 0, 1),
            at(2, 0, 1),
            at(0, 1, 1),
            at(2, 1, 1),
        ])
}

/// The deletion clauses, each with the family of motions it is applied under.
/// A clause fires if any image of it matches, so the order cannot matter.
fn deletion_patterns() -> Vec<(Pattern, Vec<Symmetry>)> {
    // A face direction: the centre continues only along `+z`.
    let u0 = Pattern::default()
        .set([CENTRE, at(1, 1, 2)])
        .clear(plane(2, 0))
        .clear(plane(2, 1).into_iter().filter(|p| *p != CENTRE));
    let u1 = u_prefix()
        .set([at(0, 2, 1), at(0, 2, 2)])
        .clear([at(1, 2, 1), at(2, 2, 1)]);
    let u2 = u_prefix()
        .set([at(1, 2, 1), at(1, 2, 2)])
        .clear([at(0, 2, 1), at(2, 2, 1)]);
    // An edge direction: the centre continues along `+y+z`, with the two faces
    // it turns away from clear and its own row clear beside it.
    let r2 = Pattern::default()
        .set([CENTRE, at(1, 2, 2)])
        .clear(plane(2, 0))
        .clear(plane(1, 0))
        .clear([at(0, 1, 1), at(2, 1, 1)]);
    // A corner direction: the centre continues along `-x+y+z`, with the three
    // faces opposite that corner clear.
    let s3 = Pattern::default()
        .set([CENTRE, at(0, 2, 2)])
        .clear(plane(2, 0))
        .clear(plane(1, 0))
        .clear(plane(0, 2));
    vec![
        (u0, face_symmetries()),
        (u1, corner_face_symmetries()),
        (u2, corner_face_symmetries()),
        (r2, edge_symmetries()),
        (s3, corner_symmetries()),
    ]
}

fn surroundings() -> Vec<Position> {
    (0..POSITIONS)
        .filter(|position| *position != CENTRE)
        .collect()
}

/// Every subset of `bits` of size at most `most`, each exactly once.
fn subsets_up_to(bits: &[u32], most: usize, visit: &mut impl FnMut(u32)) {
    fn walk(bits: &[u32], left: usize, from: usize, so_far: u32, visit: &mut impl FnMut(u32)) {
        visit(so_far);
        if left == 0 {
            return;
        }
        for index in from..bits.len() {
            walk(bits, left - 1, index + 1, so_far | bits[index], visit);
        }
    }
    walk(bits, most, 0, 0, visit);
}

/// A set voxel with at most this many set voxels in its neighbourhood, itself
/// included, is removed.
const REMOVED_AT_MOST: usize = 2;

/// A clear voxel with all but this many of its 26 neighbours set is filled:
/// "all but the six faces".
const FILLED_ABOVE_GAP: usize = 6;

/// Face neighbours a clear voxel needs, beside an opposite pair, to be filled as
/// a gap in a surface.
const BRIDGED_FACES: usize = 3;

/// The topology-preserving table. Its two halves are disjoint by construction —
/// every clause of the first requires the centre set and every clause of the
/// second requires it clear — so the writes are independent.
fn topological_table() -> ConfigurationTable {
    let mut table = ConfigurationTable::identity();
    let centre_bit = 1u32 << CENTRE;
    let around: Vec<u32> = surroundings()
        .into_iter()
        .map(|position| 1u32 << position)
        .collect();
    let all_around: u32 = around.iter().fold(0, |all, bit| all | bit);

    // A set voxel is cleared when it is nearly alone ...
    subsets_up_to(&around, REMOVED_AT_MOST - 1, &mut |kept| {
        let template = ConfigurationTemplate::from_masks(centre_bit | kept, all_around & !kept)
            .expect("the centre is not one of the surroundings");
        table.assign_matching(&template, false);
    });
    // ... or its neighbourhood matches a deletion template.
    for (pattern, family) in deletion_patterns() {
        for symmetry in &family {
            table.assign_matching(&pattern.under(symmetry), false);
        }
    }

    // A clear voxel is set when it is nearly enclosed ...
    subsets_up_to(&around, FILLED_ABOVE_GAP, &mut |missing| {
        let template =
            ConfigurationTemplate::from_masks(all_around & !missing, centre_bit | missing)
                .expect("the centre is not one of the surroundings");
        table.assign_matching(&template, true);
    });
    // ... or it bridges an opposite pair of faces with enough faces set ...
    let faces: Vec<u32> = face_neighbours()
        .into_iter()
        .map(|position| 1u32 << position)
        .collect();
    let all_faces: u32 = faces.iter().fold(0, |all, bit| all | bit);
    let pairs = opposite_pairs();
    for pattern in 0..(1u32 << faces.len()) {
        let present: u32 = faces
            .iter()
            .enumerate()
            .filter(|(bit, _)| pattern >> bit & 1 == 1)
            .fold(0, |all, (_, mask)| all | mask);
        let opposed = pairs
            .iter()
            .any(|pair| pair.iter().all(|position| present >> *position & 1 == 1));
        if !opposed || present.count_ones() < BRIDGED_FACES as u32 {
            continue;
        }
        let template =
            ConfigurationTemplate::from_masks(present, centre_bit | (all_faces & !present))
                .expect("the centre is not a face neighbour");
        table.assign_matching(&template, true);
    }
    // ... or the complement of its neighbourhood matches a deletion template.
    for (pattern, family) in deletion_patterns() {
        for symmetry in &family {
            table.assign_matching(&pattern.under(symmetry).inverted(), true);
        }
    }
    table
}

/// The majority of the seven-voxel face neighbourhood: the cheap alternative,
/// with none of the topology guarantees.
fn majority_table() -> ConfigurationTable {
    let mut positions = vec![CENTRE];
    positions.extend(face_neighbours());
    let mut table = ConfigurationTable::zeros();
    for pattern in 0..(1u32 << positions.len()) {
        if pattern.count_ones() * 2 <= positions.len() as u32 {
            continue;
        }
        let mut set = 0u32;
        let mut clear = 0u32;
        for (bit, position) in positions.iter().enumerate() {
            if pattern >> bit & 1 == 1 {
                set |= 1 << position;
            } else {
                clear |= 1 << position;
            }
        }
        table.assign_matching(
            &ConfigurationTemplate::from_masks(set, clear).expect("each position appears once"),
            true,
        );
    }
    table
}

/// Which table a pass applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// Spur removal and pinhole filling that keeps the topology.
    Topological,
    /// Face-neighbourhood majority.
    Majority,
}

impl Rule {
    /// The table, built once per process.
    pub fn table(self) -> Arc<ConfigurationTable> {
        static TOPOLOGICAL: OnceLock<Arc<ConfigurationTable>> = OnceLock::new();
        static MAJORITY: OnceLock<Arc<ConfigurationTable>> = OnceLock::new();
        match self {
            Rule::Topological => TOPOLOGICAL
                .get_or_init(|| Arc::new(topological_table()))
                .clone(),
            Rule::Majority => MAJORITY.get_or_init(|| Arc::new(majority_table())).clone(),
        }
    }
}

/// A stated number of passes of one rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmoothParams {
    pub rule: Rule,
    /// Each pass reads the whole result of the previous one, so the op's reach
    /// is this count.
    pub passes: usize,
}

impl Default for SmoothParams {
    fn default() -> Self {
        Self {
            rule: Rule::Topological,
            passes: 6,
        }
    }
}

impl SmoothParams {
    pub fn validate(&self) -> Result<()> {
        if self.passes == 0 {
            return Err(Error::invalid(
                "smooth: zero passes is no smoothing; leave the step out instead",
            ));
        }
        Ok(())
    }

    /// The pass op, over a `Bool` stream.
    pub fn node(&self) -> Chain {
        Chain::op(ConfigurationPassOp::new(
            "smooth",
            self.rule.table(),
            self.passes,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blockflow::ops::configuration::configuration_bit;

    #[test]
    fn the_position_numbering_is_the_frameworks_bit_numbering() {
        for position in 0..POSITIONS {
            assert_eq!(
                configuration_bit(offset_of(position)).unwrap() as usize,
                position
            );
        }
        assert_eq!(offset_of(CENTRE), [0, 0, 0]);
    }

    #[test]
    fn every_symmetry_is_a_bijection_that_fixes_the_centre() {
        let families = [
            face_symmetries(),
            edge_symmetries(),
            corner_symmetries(),
            corner_face_symmetries(),
        ];
        for family in &families {
            for symmetry in family {
                let mut seen = [false; POSITIONS];
                for position in 0..POSITIONS {
                    assert!(!seen[symmetry[position]], "a symmetry is not a bijection");
                    seen[symmetry[position]] = true;
                }
                assert_eq!(symmetry[CENTRE], CENTRE);
            }
        }
        assert_eq!(
            families.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![6, 12, 8, 24]
        );
        for axis in 0..3 {
            assert_eq!(turns(axis, 4), identity());
            assert_ne!(turns(axis, 1), identity());
        }
    }

    #[test]
    fn each_family_covers_the_directions_it_is_named_for() {
        let images = |family: Vec<Symmetry>, from: Position| -> usize {
            let mut seen: Vec<Position> = family.iter().map(|symmetry| symmetry[from]).collect();
            seen.sort_unstable();
            seen.dedup();
            seen.len()
        };
        assert_eq!(images(face_symmetries(), at(1, 1, 2)), 6);
        assert_eq!(images(edge_symmetries(), at(1, 2, 2)), 12);
        assert_eq!(images(corner_symmetries(), at(0, 2, 2)), 8);
        assert_eq!(images(corner_face_symmetries(), at(1, 1, 2)), 6);
    }

    #[test]
    fn subsets_are_enumerated_once_each() {
        let bits = [1u32, 2, 4, 8];
        let binomial =
            |n: usize, k: usize| (0..k).fold(1, |total, index| total * (n - index) / (index + 1));
        for most in 0..=4usize {
            let mut seen = Vec::new();
            subsets_up_to(&bits, most, &mut |subset| seen.push(subset));
            let mut unique = seen.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(seen.len(), unique.len());
            assert_eq!(
                seen.len(),
                (0..=most).map(|k| binomial(4, k)).sum::<usize>()
            );
        }
    }

    #[test]
    fn the_two_halves_of_the_rule_cannot_overwrite_each_other() {
        for (pattern, family) in deletion_patterns() {
            for symmetry in &family {
                let template = pattern.under(symmetry);
                assert_eq!(template.set_mask() >> CENTRE & 1, 1);
                assert_eq!(template.inverted().clear_mask() >> CENTRE & 1, 1);
            }
        }
    }

    #[test]
    fn the_tables_move_voxels_and_differ_from_each_other() {
        let topological = Rule::Topological.table();
        let majority = Rule::Majority.table();
        assert!(topological.count_moving() > 1_000_000);
        assert!(majority.count_moving() > 1_000_000);
        assert_ne!(topological.count_set(), majority.count_set());
        for table in [&topological, &majority] {
            assert!(!table.get(0));
            assert!(table.get((1 << 27) - 1));
        }
    }
}
