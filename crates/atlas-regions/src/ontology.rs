// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The region hierarchy of an atlas, and the collapse of a region id onto an
//! ancestor at a given level.

use std::collections::BTreeMap;
use std::path::Path;

use blockflow::error::{Error, Result};

/// One node of the region hierarchy: its own level, and its parent.
///
/// Only what the collapse walks. Names and acronyms, when known, are kept by
/// the [`Ontology`] beside the nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionNode {
    /// Depth in the hierarchy (`st_level` in the Allen tree). `None` for a node
    /// with no level, which stops a collapse there.
    pub level: Option<i64>,
    /// The parent region (`parent_structure_id`). `None` for the root.
    pub parent: Option<i64>,
}

/// A region's human-readable labels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RegionNames {
    pub name: Option<String>,
    pub acronym: Option<String>,
}

/// The region hierarchy of an atlas.
///
/// Built from `(id, level, parent)` values with [`Ontology::new`], or read from
/// an Allen-style JSON-lines structure file with [`Ontology::from_jsonl`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ontology {
    nodes: BTreeMap<i64, RegionNode>,
    names: BTreeMap<i64, RegionNames>,
    /// The Allen `st_level` of each region that has one, whichever level the
    /// nodes carry.
    structure_levels: BTreeMap<i64, i64>,
    /// `color_hex_triplet` as RGB.
    colors: BTreeMap<i64, [u8; 3]>,
}

/// Which number a nested structure file's regions carry as their level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OntologyLevel {
    /// The depth in the tree, the root record at 0: what the reference's
    /// annotation class calls a label's `level` and collapses on.
    #[default]
    TreeDepth,
    /// The Allen `st_level` field, as [`Ontology::from_jsonl`] reads it.
    StructureLevel,
}

impl Ontology {
    /// `(id, level, parent)` per region.
    ///
    /// A repeated id is refused rather than last-one-wins: a duplicate means
    /// the table is not a tree, and a collapse through it would silently take
    /// whichever record came last.
    pub fn new(records: impl IntoIterator<Item = (i64, Option<i64>, Option<i64>)>) -> Result<Self> {
        let mut nodes = BTreeMap::new();
        for (id, level, parent) in records {
            if nodes.insert(id, RegionNode { level, parent }).is_some() {
                return Err(Error::invalid(format!(
                    "region id {id} appears twice in the ontology; a region tree holds each id \
                     once"
                )));
            }
        }
        Ok(Self {
            nodes,
            ..Self::default()
        })
    }

    /// Reads an Allen-style structure file: one JSON object per line with
    /// `id`, `st_level`, `parent_structure_id` and optionally `name` and
    /// `acronym`. Other fields are ignored.
    ///
    /// Integers may be written as JSON floats (`2.0`), as the commonly shipped
    /// `ABA_annotation_last.jsonl` does; a number with a fractional part is
    /// refused rather than truncated.
    pub fn from_jsonl(text: &str) -> Result<Self> {
        let mut records = Vec::new();
        let mut names = Vec::new();
        let mut extras = Vec::new();
        for (line_number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let at = line_number + 1;
            let record: serde_json::Value = serde_json::from_str(line).map_err(|error| {
                Error::invalid(format!("line {at}: not a JSON record: {error}"))
            })?;
            let id = integral(&record, "id", at)?
                .ok_or_else(|| Error::invalid(format!("line {at}: the record has no `id`")))?;
            let structure_level = integral(&record, "st_level", at)?;
            records.push((
                id,
                structure_level,
                integral(&record, "parent_structure_id", at)?,
            ));
            names.push((id, record_names(&record)));
            extras.push((id, structure_level, record_color(&record, at)?));
        }
        let mut ontology = Self::new(records)?;
        ontology.names = names.into_iter().collect();
        ontology.add_extras(extras);
        Ok(ontology)
    }

    /// Reads the nested Allen structure graph the reference loads by default
    /// (`ABA_annotation.json`, `ABA_annotation_last.json`): an object whose
    /// `msg` array holds the root record, each record with its `children`.
    ///
    /// The parent is taken from the nesting (the older file leaves
    /// `parent_structure_id` null); a `parent_structure_id` that contradicts
    /// it is refused. `level` picks the tree depth (the reference's level) or
    /// `st_level` as each node's level; the other stays available through
    /// [`Self::depth`] and [`Self::structure_level`].
    pub fn from_ontology_json(text: &str, level: OntologyLevel) -> Result<Self> {
        let document: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| Error::invalid(format!("not a JSON structure graph: {error}")))?;
        let roots = document
            .get("msg")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                Error::invalid("the structure graph has no `msg` array holding the root record")
            })?;
        let mut records = Vec::new();
        let mut names = Vec::new();
        let mut extras = Vec::new();
        // An explicit stack: the tree is a dozen deep, but a hostile file
        // should not be able to overflow the call stack.
        let mut stack: Vec<(&serde_json::Value, Option<i64>, i64)> =
            roots.iter().rev().map(|root| (root, None, 0)).collect();
        while let Some((record, parent, depth)) = stack.pop() {
            let at = records.len() + 1;
            let id = integral(record, "id", at)?
                .ok_or_else(|| Error::invalid(format!("record {at}: the record has no `id`")))?;
            let declared = integral(record, "parent_structure_id", at)?;
            if declared.is_some() && declared != parent {
                return Err(Error::invalid(format!(
                    "region {id} declares parent {declared:?} but is nested under {parent:?}"
                )));
            }
            let structure_level = integral(record, "st_level", at)?;
            let node_level = match level {
                OntologyLevel::TreeDepth => Some(depth),
                OntologyLevel::StructureLevel => structure_level,
            };
            records.push((id, node_level, parent));
            names.push((id, record_names(record)));
            extras.push((id, structure_level, record_color(record, at)?));
            if let Some(children) = record.get("children").and_then(serde_json::Value::as_array) {
                for child in children.iter().rev() {
                    stack.push((child, Some(id), depth + 1));
                }
            }
        }
        let mut ontology = Self::new(records)?;
        ontology.names = names.into_iter().collect();
        ontology.add_extras(extras);
        Ok(ontology)
    }

    /// [`Ontology::from_ontology_json`] on a file.
    pub fn from_ontology_json_file(path: impl AsRef<Path>, level: OntologyLevel) -> Result<Self> {
        Self::from_ontology_json(&read_text(path.as_ref())?, level)
    }

    /// Reads either structure file: JSON lines when the first non-blank line
    /// is a whole record, the nested graph otherwise. `level` applies to the
    /// nested graph only; the JSON-lines file always carries `st_level`.
    pub fn from_file(path: impl AsRef<Path>, level: OntologyLevel) -> Result<Self> {
        let text = read_text(path.as_ref())?;
        let first = text.lines().map(str::trim).find(|line| !line.is_empty());
        let one_record_per_line = first
            .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .is_some_and(|value| value.get("msg").is_none());
        if one_record_per_line {
            Self::from_jsonl(&text)
        } else {
            Self::from_ontology_json(&text, level)
        }
    }

    /// Reads the region list a prepared atlas keeps in its annotation's
    /// `image-label.properties`: per entry `label-value`, `parent`, `level`
    /// (tree depth), `st_level`, `name`, `acronym` and `color_hex_triplet`,
    /// any of them but the value optional. `level` picks which number the
    /// nodes carry, as for [`Self::from_ontology_json`].
    pub fn from_image_label_properties(
        properties: &[serde_json::Value],
        level: OntologyLevel,
    ) -> Result<Self> {
        let mut records = Vec::new();
        let mut names = Vec::new();
        let mut extras = Vec::new();
        for (index, entry) in properties.iter().enumerate() {
            let at = index + 1;
            let id = integral(entry, "label-value", at)?
                .ok_or_else(|| Error::invalid(format!("property {at} has no `label-value`")))?;
            let structure_level = integral(entry, "st_level", at)?;
            let node_level = match level {
                OntologyLevel::TreeDepth => integral(entry, "level", at)?,
                OntologyLevel::StructureLevel => structure_level,
            };
            records.push((id, node_level, integral(entry, "parent", at)?));
            names.push((id, record_names(entry)));
            extras.push((id, structure_level, record_color(entry, at)?));
        }
        let mut ontology = Self::new(records)?;
        ontology.names = names.into_iter().collect();
        ontology.add_extras(extras);
        Ok(ontology)
    }

    fn add_extras(&mut self, extras: Vec<(i64, Option<i64>, Option<[u8; 3]>)>) {
        for (id, structure_level, color) in extras {
            if let Some(structure_level) = structure_level {
                self.structure_levels.insert(id, structure_level);
            }
            if let Some(color) = color {
                self.colors.insert(id, color);
            }
        }
    }

    /// [`Ontology::from_jsonl`] on a file.
    pub fn from_jsonl_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_jsonl(&read_text(path.as_ref())?)
    }

    /// Attaches an RGB colour to a region already in the tree.
    pub fn set_color(&mut self, id: i64, color: [u8; 3]) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(Error::invalid(format!(
                "region id {id} is not in the ontology"
            )));
        }
        self.colors.insert(id, color);
        Ok(())
    }

    /// Attaches a name and acronym to a region already in the tree.
    pub fn set_names(&mut self, id: i64, names: RegionNames) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(Error::invalid(format!(
                "region id {id} is not in the ontology"
            )));
        }
        self.names.insert(id, names);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn contains(&self, id: i64) -> bool {
        self.nodes.contains_key(&id)
    }

    pub fn get(&self, id: i64) -> Option<RegionNode> {
        self.nodes.get(&id).copied()
    }

    /// Every region id, ascending.
    pub fn ids(&self) -> impl Iterator<Item = i64> + '_ {
        self.nodes.keys().copied()
    }

    pub fn name(&self, id: i64) -> Option<&str> {
        self.names.get(&id).and_then(|names| names.name.as_deref())
    }

    pub fn acronym(&self, id: i64) -> Option<&str> {
        self.names
            .get(&id)
            .and_then(|names| names.acronym.as_deref())
    }

    /// The region's RGB colour (`color_hex_triplet`), when the file gave one.
    pub fn color(&self, id: i64) -> Option<[u8; 3]> {
        self.colors.get(&id).copied()
    }

    /// The parent region; `None` for a root or an unknown id.
    pub fn parent(&self, id: i64) -> Option<i64> {
        self.nodes.get(&id).and_then(|node| node.parent)
    }

    /// The level the nodes carry (see [`OntologyLevel`]).
    pub fn level(&self, id: i64) -> Option<i64> {
        self.nodes.get(&id).and_then(|node| node.level)
    }

    /// The Allen `st_level`, when the file gave one.
    pub fn structure_level(&self, id: i64) -> Option<i64> {
        self.structure_levels.get(&id).copied()
    }

    /// The number of ancestors: 0 for a root. The reference's label level
    /// when the tree starts at its `universe` record, as the shipped files do.
    pub fn depth(&self, id: i64) -> Result<usize> {
        Ok(self.ancestors(id)?.len() - 1)
    }

    /// The direct children of `id`, ascending.
    pub fn children(&self, id: i64) -> Vec<i64> {
        self.nodes
            .iter()
            .filter(|(child, node)| node.parent == Some(id) && **child != id)
            .map(|(child, _)| *child)
            .collect()
    }

    /// Every region without a parent, ascending.
    pub fn roots(&self) -> Vec<i64> {
        self.nodes
            .iter()
            .filter(|(id, node)| node.parent.is_none_or(|parent| parent == **id))
            .map(|(id, _)| *id)
            .collect()
    }

    /// The ancestor of `id` at or above `level`: walk to the parent while the
    /// node's level is greater than `level`.
    ///
    /// A node with no level or no parent is its own answer, so every walk ends
    /// at the root. An id the tree does not hold is refused: the annotation
    /// volume and the tree are then not the same atlas, and mapping the id to
    /// anything would be a wrong region rather than a missing one.
    pub fn collapse(&self, id: i64, level: i64) -> Result<i64> {
        let mut current = id;
        let mut steps = 0usize;
        loop {
            let node = self.node(current, id)?;
            match node.level {
                Some(node_level) if node_level > level => {}
                _ => return Ok(current),
            }
            let Some(parent) = node.parent.filter(|parent| *parent != current) else {
                return Ok(current);
            };
            current = parent;
            steps += 1;
            if steps > self.nodes.len() {
                return Err(cycle(id));
            }
        }
    }

    /// `id` followed by its ancestors up to the root.
    pub fn ancestors(&self, id: i64) -> Result<Vec<i64>> {
        let mut path = vec![id];
        let mut current = id;
        while let Some(parent) = self.node(current, id)?.parent.filter(|p| *p != current) {
            path.push(parent);
            current = parent;
            if path.len() > self.nodes.len() + 1 {
                return Err(cycle(id));
            }
        }
        Ok(path)
    }

    /// Whether `id` is `ancestor` or lies below it.
    pub fn is_within(&self, id: i64, ancestor: i64) -> Result<bool> {
        Ok(self.ancestors(id)?.contains(&ancestor))
    }

    fn node(&self, current: i64, from: i64) -> Result<RegionNode> {
        self.nodes.get(&current).copied().ok_or_else(|| {
            Error::invalid(format!(
                "region id {current} is not in the ontology (reached from {from}); the \
                 annotation volume and the region tree have to describe the same atlas"
            ))
        })
    }
}

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|error| {
        Error::invalid(format!(
            "cannot read the region tree {}: {error}",
            path.display()
        ))
    })
}

fn record_names(record: &serde_json::Value) -> RegionNames {
    let text_field = |field: &str| {
        record
            .get(field)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    };
    RegionNames {
        name: text_field("name"),
        acronym: text_field("acronym"),
    }
}

/// `color_hex_triplet` (`"BFDAE3"`, optionally with `#`) as RGB.
fn record_color(record: &serde_json::Value, at: usize) -> Result<Option<[u8; 3]>> {
    let Some(hex) = record
        .get("color_hex_triplet")
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    parse_hex_color(hex)
        .map(Some)
        .ok_or_else(|| Error::invalid(format!("record {at}: colour {hex:?} is not RRGGBB")))
}

pub(crate) fn parse_hex_color(hex: &str) -> Option<[u8; 3]> {
    let hex = hex.trim().trim_start_matches('#');
    if hex.len() != 6 || !hex.is_ascii() {
        return None;
    }
    let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    Some([channel(0)?, channel(2)?, channel(4)?])
}

fn cycle(id: i64) -> Error {
    Error::invalid(format!(
        "walking up from region {id} visited more parents than the ontology has records, so \
         the hierarchy contains a cycle"
    ))
}

/// A whole-number JSON field, written either as an integer or as a float.
fn integral(record: &serde_json::Value, field: &str, line: usize) -> Result<Option<i64>> {
    let value = match record.get(field) {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(value) => value,
    };
    if let Some(integer) = value.as_i64() {
        return Ok(Some(integer));
    }
    let number = value.as_f64().ok_or_else(|| {
        Error::invalid(format!("line {line}: `{field}` is {value}, not a number"))
    })?;
    if number.fract() != 0.0 || !number.is_finite() || number.abs() > 2f64.powi(53) {
        return Err(Error::invalid(format!(
            "line {line}: `{field}` is {number}, which is not a whole number"
        )));
    }
    Ok(Some(number as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `10` and `11` are leaves at level 3 under `2` (level 1); `12` is at
    /// level 2; `0` is the root with no level.
    fn tree() -> Ontology {
        Ontology::new([
            (0, None, None),
            (2, Some(1), Some(0)),
            (10, Some(3), Some(2)),
            (11, Some(3), Some(2)),
            (12, Some(2), Some(0)),
        ])
        .expect("an ontology")
    }

    #[test]
    fn the_collapse_walks_to_the_first_ancestor_at_or_below_the_level() {
        let tree = tree();
        assert_eq!(tree.collapse(10, 3).unwrap(), 10);
        assert_eq!(tree.collapse(10, 2).unwrap(), 2);
        assert_eq!(tree.collapse(11, 2).unwrap(), 2);
        assert_eq!(tree.collapse(10, 1).unwrap(), 2);
        assert_eq!(tree.collapse(10, 0).unwrap(), 0);
        assert_eq!(tree.collapse(12, 2).unwrap(), 12);
        assert_eq!(tree.collapse(0, -5).unwrap(), 0);
    }

    #[test]
    fn an_unknown_region_is_refused() {
        let error = tree().collapse(77, 2).expect_err("no such region");
        assert!(error.to_string().contains("not in the ontology"), "{error}");
    }

    #[test]
    fn a_repeated_region_id_is_refused() {
        let error = Ontology::new([(1, Some(1), None), (1, Some(2), None)]).expect_err("duplicate");
        assert!(error.to_string().contains("twice"), "{error}");
    }

    #[test]
    fn a_cyclic_hierarchy_is_refused() {
        let cyclic = Ontology::new([(1, Some(5), Some(2)), (2, Some(5), Some(1))]).unwrap();
        assert!(cyclic
            .collapse(1, 0)
            .unwrap_err()
            .to_string()
            .contains("cycle"));
        assert!(cyclic
            .ancestors(1)
            .unwrap_err()
            .to_string()
            .contains("cycle"));
    }

    #[test]
    fn ancestors_run_to_the_root() {
        let tree = tree();
        assert_eq!(tree.ancestors(10).unwrap(), vec![10, 2, 0]);
        assert!(tree.is_within(11, 2).unwrap());
        assert!(!tree.is_within(12, 2).unwrap());
        assert!(tree.is_within(0, 0).unwrap());
    }

    #[test]
    fn the_structure_file_accepts_integers_written_as_floats() {
        let text = r#"
{"id":0,"acronym":"universe","name":"universe","st_level":null,"parent_structure_id":null}
{"id":997.0,"acronym":"root","name":"root","st_level":0.0,"parent_structure_id":0}
{"id":8,"acronym":"grey","name":"Basic cell groups and regions","st_level":1,"parent_structure_id":997.0,"color_hex_triplet":"BFDAE3"}
"#;
        let tree = Ontology::from_jsonl(text).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(
            tree.get(8),
            Some(RegionNode {
                level: Some(1),
                parent: Some(997)
            })
        );
        assert_eq!(tree.get(0).unwrap().level, None);
        assert_eq!(tree.acronym(8), Some("grey"));
        assert_eq!(tree.name(997), Some("root"));
        assert_eq!(tree.collapse(8, 0).unwrap(), 997);
        assert_eq!(tree.color(8), Some([0xBF, 0xDA, 0xE3]));
        assert_eq!(tree.structure_level(8), Some(1));
        assert_eq!(tree.depth(8).unwrap(), 2);
        assert_eq!(tree.children(997), vec![8]);
        assert_eq!(tree.roots(), vec![0]);
    }

    const NESTED: &str = r#"{"success": true, "msg": [
      {"id": 0, "acronym": "universe", "name": "universe", "color_hex_triplet": "000000",
       "st_level": null, "parent_structure_id": null, "children": [
        {"id": 997, "acronym": "root", "name": "root", "color_hex_triplet": "FFFFFF",
         "st_level": 0, "parent_structure_id": null, "children": [
          {"id": 8, "acronym": "grey", "name": "Basic cell groups and regions",
           "color_hex_triplet": "BFDAE3", "st_level": 1, "parent_structure_id": 997,
           "children": [
             {"id": 567, "acronym": "CH", "name": "Cerebrum", "color_hex_triplet": "B0F0FF",
              "st_level": 2, "parent_structure_id": 8, "children": []}]},
          {"id": 1009, "acronym": "fiber tracts", "name": "fiber tracts",
           "color_hex_triplet": "CCCCCC", "st_level": 1, "children": []}]}]}]}"#;

    #[test]
    fn the_nested_structure_graph_takes_parents_from_the_nesting() {
        let tree = Ontology::from_ontology_json(NESTED, OntologyLevel::TreeDepth).unwrap();
        assert_eq!(tree.len(), 5);
        assert_eq!(tree.parent(997), Some(0));
        assert_eq!(tree.parent(1009), Some(997));
        assert_eq!(tree.level(567), Some(3));
        assert_eq!(tree.structure_level(567), Some(2));
        assert_eq!(tree.depth(567).unwrap(), 3);
        assert_eq!(tree.children(997), vec![8, 1009]);
        assert_eq!(tree.acronym(567), Some("CH"));
        assert_eq!(tree.color(1009), Some([0xCC; 3]));
        // The reference collapses on depth: level 2 of Cerebrum is `grey`.
        assert_eq!(tree.collapse(567, 2).unwrap(), 8);
        let by_st = Ontology::from_ontology_json(NESTED, OntologyLevel::StructureLevel).unwrap();
        assert_eq!(by_st.level(567), Some(2));
        assert_eq!(by_st.collapse(567, 1).unwrap(), 8);
    }

    #[test]
    fn a_nesting_that_contradicts_the_declared_parent_is_refused() {
        let bad = NESTED.replace(
            r#""parent_structure_id": 8"#,
            r#""parent_structure_id": 997"#,
        );
        let error = Ontology::from_ontology_json(&bad, OntologyLevel::TreeDepth).unwrap_err();
        assert!(error.to_string().contains("nested under"), "{error}");
    }

    #[test]
    fn a_fractional_level_is_refused() {
        let error = Ontology::from_jsonl(r#"{"id":3,"st_level":2.5}"#).unwrap_err();
        assert!(error.to_string().contains("not a whole number"), "{error}");
        let error = Ontology::from_jsonl(r#"{"st_level":2}"#).unwrap_err();
        assert!(error.to_string().contains("no `id`"), "{error}");
    }
}
