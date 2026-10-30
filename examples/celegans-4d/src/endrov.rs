// SPDX-License-Identifier: MIT

//! **Endrov `nuclineage` objects**: parsing and Endrov's own interpolation.
//!
//! An Endrov recording (`*.ost`) keeps its annotations in `rmd.ostxml`. A
//! lineage is a `<nuclineage id=..>` element holding one `<nuc name=..>` per
//! nucleus (one cell between divisions):
//!
//! ```xml
//! <nuc name="ABplp" [start=".." end=".."]>
//!   <pos f="1620" x="37.99" y="27.62" z="18.0" r="1.53" />   <!-- keyframe, µm, f = frame id -->
//!   <child name="ABplpa" />
//!   <exp name="exp"><v f="1680" l="0.13" /></exp>             <!-- expression level per frame -->
//! </nuc>
//! ```
//!
//! Positions are hand-placed **keyframes**; everything between them is
//! interpolated. [`Nucleus::interpolate`] reproduces Endrov's rule (see the
//! function for the source reference), so a consumer can tell a keyframe from
//! an interpolated position — the distinction the held-out-keyframe
//! benchmark is built on.
//!
//! Nothing here reads pixels or knows about OME-Zarr; the mapping from
//! Endrov world coordinates to an image lives with the caller.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use blockflow::{Error, Result};

/// One hand-placed position: frame id (Endrov frame time, seconds) and a
/// sphere in Endrov world coordinates (µm).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyframe {
    pub frame: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: f64,
}

/// One expression series, as stored: `(frame, level)`, sorted by frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Expression {
    pub levels: Vec<(f64, f64)>,
}

/// One nucleus — a cell between divisions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Nucleus {
    pub name: String,
    /// Keyframes, sorted by frame, frames unique.
    pub keyframes: Vec<Keyframe>,
    pub children: Vec<String>,
    /// Derived from the children lists of the lineage.
    pub parent: Option<String>,
    /// `start=` / `end=` attributes: Endrov's override of the lifetime.
    pub override_start: Option<f64>,
    pub override_end: Option<f64>,
    /// `<exp name=..>` series by name.
    pub expression: BTreeMap<String, Expression>,
}

/// One `<nuclineage>` object.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Lineage {
    pub id: String,
    pub nuclei: BTreeMap<String, Nucleus>,
}

/// Read `rmd.ostxml` from a file, a `.ost` directory, or a `.ost.zip`
/// archive (the first `*/rmd.ostxml` entry).
pub fn read_ostxml(path: &Path) -> Result<String> {
    if path.is_dir() {
        return read_ostxml(&path.join("rmd.ostxml"));
    }
    let is_zip = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("zip"));
    if !is_zip {
        return fs::read_to_string(path)
            .map_err(|e| Error::backend(format!("read {}: {e}", path.display())));
    }
    let file = fs::File::open(path)
        .map_err(|e| Error::backend(format!("open {}: {e}", path.display())))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| Error::backend(format!("{} is not a zip archive: {e}", path.display())))?;
    let name = archive
        .file_names()
        .filter(|n| n.rsplit('/').next() == Some("rmd.ostxml"))
        .min_by_key(|n| n.matches('/').count())
        .map(str::to_owned)
        .ok_or_else(|| Error::invalid(format!("{} holds no rmd.ostxml", path.display())))?;
    let mut text = String::new();
    archive
        .by_name(&name)
        .map_err(Error::backend)?
        .read_to_string(&mut text)
        .map_err(Error::backend)?;
    Ok(text)
}

fn number(node: roxmltree::Node<'_, '_>, attribute: &str) -> Result<f64> {
    let text = node.attribute(attribute).ok_or_else(|| {
        Error::invalid(format!(
            "<{}> at byte {} has no {attribute}= attribute",
            node.tag_name().name(),
            node.range().start
        ))
    })?;
    text.trim().parse::<f64>().map_err(|_| {
        Error::invalid(format!(
            "<{}> at byte {}: {attribute}={text:?} is not a number",
            node.tag_name().name(),
            node.range().start
        ))
    })
}

fn optional_number(node: roxmltree::Node<'_, '_>, attribute: &str) -> Result<Option<f64>> {
    match node.attribute(attribute) {
        None => Ok(None),
        Some(_) => number(node, attribute).map(Some),
    }
}

/// Every `<nuclineage>` in an `rmd.ostxml` document, in document order.
///
/// Endrov writes one `<pos>` per frame; a repeated frame keeps the last one,
/// as loading into Endrov's frame-keyed map would. Children named but absent
/// from the lineage are kept in `children` (Endrov tolerates them) and get no
/// parent link.
pub fn parse_lineages(text: &str) -> Result<Vec<Lineage>> {
    let document = roxmltree::Document::parse(text)
        .map_err(|e| Error::invalid(format!("rmd.ostxml is not well-formed XML: {e}")))?;
    let mut lineages = Vec::new();
    for element in document
        .descendants()
        .filter(|n| n.has_tag_name("nuclineage"))
    {
        let id = element.attribute("id").unwrap_or("").to_owned();
        let mut nuclei = BTreeMap::new();
        let data = element
            .children()
            .find(|n| n.has_tag_name("data"))
            .unwrap_or(element);
        for nuc in data.children().filter(|n| n.has_tag_name("nuc")) {
            let name = nuc
                .attribute("name")
                .ok_or_else(|| Error::invalid(format!("a <nuc> in lineage {id:?} has no name")))?
                .to_owned();
            let mut nucleus = Nucleus {
                name: name.clone(),
                override_start: optional_number(nuc, "start")?,
                override_end: optional_number(nuc, "end")?,
                ..Nucleus::default()
            };
            for child in nuc.children().filter(roxmltree::Node::is_element) {
                match child.tag_name().name() {
                    "pos" => {
                        let key = Keyframe {
                            frame: number(child, "f")?,
                            x: number(child, "x")?,
                            y: number(child, "y")?,
                            z: number(child, "z")?,
                            r: number(child, "r")?,
                        };
                        nucleus.keyframes.push(key);
                    }
                    "child" => {
                        if let Some(child_name) = child.attribute("name") {
                            nucleus.children.push(child_name.to_owned());
                        }
                    }
                    "exp" => {
                        let series_name = child.attribute("name").unwrap_or("exp").to_owned();
                        let mut levels = Vec::new();
                        for v in child.children().filter(|n| n.has_tag_name("v")) {
                            levels.push((number(v, "f")?, number(v, "l")?));
                        }
                        levels.sort_by(|a, b| a.0.total_cmp(&b.0));
                        nucleus
                            .expression
                            .insert(series_name, Expression { levels });
                    }
                    _ => {}
                }
            }
            // Stable sort, then keep the last of equal frames.
            nucleus
                .keyframes
                .sort_by(|a, b| a.frame.total_cmp(&b.frame));
            let mut unique: Vec<Keyframe> = Vec::with_capacity(nucleus.keyframes.len());
            for key in nucleus.keyframes.drain(..) {
                match unique.last_mut() {
                    Some(last) if last.frame == key.frame => *last = key,
                    _ => unique.push(key),
                }
            }
            nucleus.keyframes = unique;
            nuclei.insert(name, nucleus);
        }
        let links: Vec<(String, String)> = nuclei
            .values()
            .flat_map(|n| n.children.iter().map(|c| (n.name.clone(), c.clone())))
            .collect();
        for (parent, child) in links {
            if let Some(nucleus) = nuclei.get_mut(&child) {
                nucleus.parent = Some(parent);
            }
        }
        lineages.push(Lineage { id, nuclei });
    }
    Ok(lineages)
}

/// Where an interpolated position comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A hand-placed keyframe at exactly this frame.
    Keyframe,
    /// Linear between the keyframes before and after.
    Interpolated,
    /// After the last keyframe, the last position held — Endrov keeps a
    /// nucleus there until its first child appears or its `end=`.
    Held,
}

impl Source {
    pub fn code(self) -> u8 {
        match self {
            Self::Keyframe => 0,
            Self::Interpolated => 1,
            Self::Held => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Keyframe => "keyframe",
            Self::Interpolated => "interpolated",
            Self::Held => "held",
        }
    }
}

/// How long a nucleus with no child (and no `end=`) lasts after its last
/// keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafEnd {
    /// End at the last keyframe. Annotators stop clicking when they stop
    /// annotating, not when the cell ends, so this is the default.
    LastKeyframe,
    /// Endrov's own rule: hold the last position for ever (to the end of the
    /// recording).
    Hold,
}

/// One position of a nucleus at one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub frame: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: f64,
    pub source: Source,
    /// The keyframe at or before `frame`.
    pub key_before: f64,
    /// The keyframe at or after `frame`; `None` while held.
    pub key_after: Option<f64>,
}

impl Nucleus {
    /// `getFirstFrame`: `start=` if given, else the first keyframe.
    pub fn first_frame(&self) -> Option<f64> {
        self.override_start
            .or_else(|| self.keyframes.first().map(|k| k.frame))
    }

    /// Exactly one keyframe and no relatives: an event or landmark marker
    /// (`gast`, `venc`, `:anterior`, ...), not a cell.
    pub fn is_marker(&self) -> bool {
        self.keyframes.len() == 1 && self.children.is_empty() && self.parent.is_none()
    }

    /// Endrov's auto-generated names for unidentified nuclei are `:` + a
    /// number.
    pub fn is_named(&self) -> bool {
        !self.name.starts_with(':')
    }

    /// Position at `frame`, or `None` where the nucleus does not exist.
    ///
    /// `child_first` is the earliest first frame over the children (see
    /// [`Lineage::child_first_frame`]). This is `Lineage.Particle.interpolatePos`
    /// (`ev/endrov/typeLineage/Lineage.java:1033-1107` in github.com/mahogny/Endrov,
    /// `NucLineage.java` before the 2011 rename) with one deliberate
    /// difference: before the first keyframe Endrov returns the first
    /// position flagged invisible (`isVisible() == frameBefore != null`);
    /// here it is absent.
    ///
    /// - outside `[start=, end=]` (both inclusive): absent;
    /// - at or after the first frame of any child: absent — parent and
    ///   children never share a frame;
    /// - between keyframes: linear in x, y, z and r;
    /// - after the last keyframe: held, for a parent until its child appears,
    ///   for a leaf per `leaf_end` (`end=`, when given, always wins).
    pub fn interpolate(
        &self,
        frame: f64,
        child_first: Option<f64>,
        leaf_end: LeafEnd,
    ) -> Option<Sample> {
        let first = self.keyframes.first()?;
        let last = self.keyframes.last()?;
        if self.override_end.is_some_and(|end| frame > end)
            || self.override_start.is_some_and(|start| frame < start)
            || child_first.is_some_and(|c| frame >= c)
            || frame < first.frame
        {
            return None;
        }
        if frame >= last.frame {
            let leaf = child_first.is_none() && self.override_end.is_none();
            if frame > last.frame && leaf && leaf_end == LeafEnd::LastKeyframe {
                return None;
            }
            return Some(Sample {
                frame,
                x: last.x,
                y: last.y,
                z: last.z,
                r: last.r,
                source: if frame == last.frame {
                    Source::Keyframe
                } else {
                    Source::Held
                },
                key_before: last.frame,
                key_after: (frame == last.frame).then_some(last.frame),
            });
        }
        // first.frame <= frame < last.frame: a bracketing pair exists.
        let after = self.keyframes.partition_point(|k| k.frame < frame);
        let b = &self.keyframes[after];
        if b.frame == frame {
            return Some(Sample {
                frame,
                x: b.x,
                y: b.y,
                z: b.z,
                r: b.r,
                source: Source::Keyframe,
                key_before: frame,
                key_after: Some(frame),
            });
        }
        let a = &self.keyframes[after - 1];
        let s = (frame - a.frame) / (b.frame - a.frame);
        let lerp = |p: f64, q: f64| p * (1.0 - s) + q * s;
        Some(Sample {
            frame,
            x: lerp(a.x, b.x),
            y: lerp(a.y, b.y),
            z: lerp(a.z, b.z),
            r: lerp(a.r, b.r),
            source: Source::Interpolated,
            key_before: a.frame,
            key_after: Some(b.frame),
        })
    }
}

impl Expression {
    /// `LineageExp.interpolateLevel`
    /// (`ev/endrov/typeLineage/LineageExp.java:48-65`): linear between
    /// stored frames, the first or last level held outside them.
    pub fn level(&self, frame: f64) -> Option<f64> {
        let (first, last) = (self.levels.first()?, self.levels.last()?);
        if frame <= first.0 {
            return Some(first.1);
        }
        if frame >= last.0 {
            return Some(last.1);
        }
        let after = self.levels.partition_point(|v| v.0 < frame);
        let (b, a) = (self.levels[after], self.levels[after - 1]);
        if b.0 == frame {
            return Some(b.1);
        }
        let s = (frame - a.0) / (b.0 - a.0);
        Some(a.1 * (1.0 - s) + b.1 * s)
    }

    /// Whether a level is stored at exactly `frame`.
    pub fn measured_at(&self, frame: f64) -> bool {
        self.levels
            .binary_search_by(|v| v.0.total_cmp(&frame))
            .is_ok()
    }
}

impl Lineage {
    /// The earliest [`Nucleus::first_frame`] over the children of `name`
    /// present in the lineage: the frame at which `name` stops existing.
    pub fn child_first_frame(&self, name: &str) -> Option<f64> {
        self.nuclei
            .get(name)?
            .children
            .iter()
            .filter_map(|c| self.nuclei.get(c)?.first_frame())
            .min_by(f64::total_cmp)
    }

    /// Sample nucleus `name` at `frame` (see [`Nucleus::interpolate`]).
    pub fn sample(&self, name: &str, frame: f64, leaf_end: LeafEnd) -> Option<Sample> {
        self.nuclei
            .get(name)?
            .interpolate(frame, self.child_first_frame(name), leaf_end)
    }

    /// Every expression series name used in the lineage.
    pub fn expression_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .nuclei
            .values()
            .flat_map(|n| n.expression.keys().cloned())
            .collect();
        names.sort();
        names.dedup();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<ost version="3.3"><imageset id="im"><_ostchild>
      <nuclineage id="1"><data>
        <nuc name="P0">
          <pos f="0" x="0" y="0" z="0" r="1" />
          <pos f="100" x="10" y="20" z="30" r="3" />
          <child name="AB" /><child name="P1" />
          <exp name="exp"><v f="0" l="1" /><v f="100" l="3" /></exp>
        </nuc>
        <nuc name="AB"><pos f="160" x="1" y="1" z="1" r="1" /><pos f="200" x="2" y="2" z="2" r="1" /></nuc>
        <nuc name="P1" end="300"><pos f="150" x="5" y="5" z="5" r="1" /></nuc>
        <nuc name="gast"><pos f="50" x="0" y="0" z="0" r="1" /></nuc>
      </data></nuclineage></_ostchild></imageset></ost>"#;

    #[test]
    fn endrov_interpolation_rules() {
        let lineage = parse_lineages(XML).unwrap().remove(0);
        assert_eq!(lineage.nuclei["AB"].parent.as_deref(), Some("P0"));
        assert!(lineage.nuclei["gast"].is_marker());
        // Linear between keyframes.
        let s = lineage.sample("P0", 50.0, LeafEnd::LastKeyframe).unwrap();
        assert_eq!(
            (s.x, s.y, s.z, s.r, s.source),
            (5.0, 10.0, 15.0, 2.0, Source::Interpolated)
        );
        // A parent is held after its last keyframe until the earliest child (P1 at 150).
        let s = lineage.sample("P0", 120.0, LeafEnd::LastKeyframe).unwrap();
        assert_eq!((s.x, s.source, s.key_after), (10.0, Source::Held, None));
        assert!(lineage.sample("P0", 150.0, LeafEnd::LastKeyframe).is_none());
        // Before the first keyframe: absent.
        assert!(lineage.sample("AB", 155.0, LeafEnd::Hold).is_none());
        // A leaf ends at its last keyframe unless held or given end=.
        assert!(lineage.sample("AB", 201.0, LeafEnd::LastKeyframe).is_none());
        assert!(lineage.sample("AB", 9999.0, LeafEnd::Hold).is_some());
        assert_eq!(
            lineage
                .sample("P1", 300.0, LeafEnd::LastKeyframe)
                .unwrap()
                .source,
            Source::Held
        );
        assert!(lineage.sample("P1", 301.0, LeafEnd::Hold).is_none());
        // Expression: linear inside, held outside.
        let e = &lineage.nuclei["P0"].expression["exp"];
        assert_eq!(
            (e.level(-5.0), e.level(25.0), e.level(500.0)),
            (Some(1.0), Some(1.5), Some(3.0))
        );
    }
}
