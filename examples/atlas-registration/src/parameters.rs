// SPDX-License-Identifier: MIT
// Original work for this crate.

// elastix's parameter-file text format, as a map.
//
// A `TransformParameters.N.txt` is a flat list of `(Key value value ...)`
// entries, one per line, with `//` line comments. Values are bare tokens or
// double-quoted strings. That is the whole grammar; elastix's own reader
// (`itk::ParameterFileParser`) accepts entries spanning several lines and this
// one does too, because the B-spline files elastix writes do exactly that once
// `NumberOfParameters` is large.
//
// **A key that is absent and a key that is present with no values are different
// facts** and are kept apart: `(WriteResultImage)` is a legal entry with zero
// values, and reporting it as absent would let a transform file's silence be
// read as a default.
//
// This module knows nothing about transforms. It is the lexical layer, and the
// separation is what lets [`crate::transform`] refuse an unknown `Transform`
// keyword *by name* rather than by failing to find a field it expected.

use std::collections::BTreeMap;

use blockflow::error::{Error, Result};

/// One elastix parameter file, parsed.
///
/// Ordered by key, because the order entries appear in a transform parameter
/// file is not meaningful and a `BTreeMap` makes two files that say the same
/// thing compare equal.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParameterMap {
    entries: BTreeMap<String, Vec<String>>,
}

impl ParameterMap {
    /// Parse the text of one parameter file.
    ///
    /// Refuses a duplicate key rather than taking the last: elastix's own parser
    /// keeps every occurrence and `ReadParameter` indexes into them, so a file
    /// with two `(Transform ...)` lines means something this reader cannot
    /// express, and guessing which one wins is a silent wrong answer.
    pub fn parse(text: &str) -> Result<Self> {
        let mut entries: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let rest = strip_comments(text);
        let mut cursor = 0usize;
        while let Some(open) = rest[cursor..].find('(') {
            let open = cursor + open;
            let close = rest[open..].find(')').ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "elastix parameter file: the entry starting at byte {open} is never closed by \
                     `)`. elastix's own parser refuses this too."
                ))
            })? + open;
            let body = &rest[open + 1..close];
            cursor = close + 1;
            let mut tokens = tokenize(body)?;
            if tokens.is_empty() {
                return Err(Error::InvalidArgument(
                    "elastix parameter file: an entry `()` has no key".to_string(),
                ));
            }
            let key = tokens.remove(0);
            if entries.insert(key.clone(), tokens).is_some() {
                return Err(Error::InvalidArgument(format!(
                    "elastix parameter file: the key `{key}` appears twice. elastix keeps both and \
                     indexes into them; this reader will not guess which one a transform meant."
                )));
            }
        }
        Ok(Self { entries })
    }

    /// Every key present, in order. For diagnostics and for the tests that
    /// assert a file was read whole.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// The raw values of a key, or `None` if the key is absent.
    ///
    /// `Some(&[])` is a key present with no values, which is not the same fact.
    pub fn values(&self, key: &str) -> Option<&[String]> {
        self.entries.get(key).map(Vec::as_slice)
    }

    /// A required key's values.
    pub fn required(&self, key: &str) -> Result<&[String]> {
        self.values(key).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "elastix parameter file: `({key} ...)` is missing, and this reader will not \
                 substitute a default for it."
            ))
        })
    }

    /// A required single string, with the surrounding quotes removed.
    pub fn string(&self, key: &str) -> Result<&str> {
        let values = self.required(key)?;
        match values {
            [one] => Ok(unquote(one)),
            other => Err(Error::InvalidArgument(format!(
                "elastix parameter file: `({key} ...)` has {} values and one string was expected",
                other.len()
            ))),
        }
    }

    /// An optional string, absent-or-present.
    pub fn optional_string(&self, key: &str) -> Result<Option<&str>> {
        if self.values(key).is_none() {
            return Ok(None);
        }
        self.string(key).map(Some)
    }

    /// elastix's `"true"` / `"false"`, with the value used when the key is
    /// absent supplied by the caller **and named at the call site**, so a
    /// default is a decision somebody wrote down rather than a `false` that fell
    /// out of `unwrap_or_default`.
    pub fn flag(&self, key: &str, absent: bool) -> Result<bool> {
        match self.optional_string(key)? {
            None => Ok(absent),
            Some("true") => Ok(true),
            Some("false") => Ok(false),
            Some(other) => Err(Error::InvalidArgument(format!(
                "elastix parameter file: `({key} \"{other}\")` is neither \"true\" nor \"false\""
            ))),
        }
    }

    /// A required list of floats.
    pub fn floats(&self, key: &str) -> Result<Vec<f64>> {
        self.required(key)?
            .iter()
            .map(|token| {
                token.parse::<f64>().map_err(|error| {
                    Error::InvalidArgument(format!(
                        "elastix parameter file: `({key} ...)` holds `{token}`, which is not a \
                         number: {error}"
                    ))
                })
            })
            .collect()
    }

    /// A required list of floats of an exact length.
    pub fn floats_n(&self, key: &str, count: usize) -> Result<Vec<f64>> {
        let values = self.floats(key)?;
        if values.len() != count {
            return Err(Error::InvalidArgument(format!(
                "elastix parameter file: `({key} ...)` has {} values and {count} were expected",
                values.len()
            )));
        }
        Ok(values)
    }

    /// A required list of floats of length three, as an array.
    pub fn triple(&self, key: &str) -> Result<[f64; 3]> {
        let values = self.floats_n(key, 3)?;
        Ok([values[0], values[1], values[2]])
    }

    /// A required list of non-negative integers.
    pub fn usizes(&self, key: &str) -> Result<Vec<usize>> {
        self.required(key)?
            .iter()
            .map(|token| {
                token.parse::<usize>().map_err(|error| {
                    Error::InvalidArgument(format!(
                        "elastix parameter file: `({key} ...)` holds `{token}`, which is not a \
                         non-negative integer: {error}"
                    ))
                })
            })
            .collect()
    }

    /// A required list of integers.
    pub fn isizes(&self, key: &str) -> Result<Vec<isize>> {
        self.required(key)?
            .iter()
            .map(|token| {
                token.parse::<isize>().map_err(|error| {
                    Error::InvalidArgument(format!(
                        "elastix parameter file: `({key} ...)` holds `{token}`, which is not an \
                         integer: {error}"
                    ))
                })
            })
            .collect()
    }

    /// A required single non-negative integer.
    pub fn usize(&self, key: &str) -> Result<usize> {
        let values = self.usizes(key)?;
        match values.as_slice() {
            [one] => Ok(*one),
            other => Err(Error::InvalidArgument(format!(
                "elastix parameter file: `({key} ...)` has {} values and one integer was expected",
                other.len()
            ))),
        }
    }

    /// Insert an entry. Used by the tests that build a file programmatically and
    /// by nothing else; the reading path never mutates.
    pub fn set(&mut self, key: impl Into<String>, values: Vec<String>) {
        self.entries.insert(key.into(), values);
    }
}

/// Strip `//` line comments, which elastix's parser does before anything else.
///
/// A `//` inside a quoted string is **not** a comment, and parameter files in
/// the wild have quoted values on lines that also carry comments, so the quote
/// state has to be tracked rather than assumed absent.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let bytes = line.as_bytes();
        let mut in_quotes = false;
        let mut end = bytes.len();
        let mut at = 0;
        while at < bytes.len() {
            match bytes[at] {
                b'"' => in_quotes = !in_quotes,
                b'/' if !in_quotes && at + 1 < bytes.len() && bytes[at + 1] == b'/' => {
                    end = at;
                    break;
                }
                _ => {}
            }
            at += 1;
        }
        out.push_str(&line[..end]);
        out.push('\n');
    }
    out
}

/// Split an entry body into tokens, keeping quoted strings whole.
fn tokenize(body: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut have = false;
    for character in body.chars() {
        match character {
            '"' => {
                in_quotes = !in_quotes;
                have = true;
                current.push('"');
            }
            c if c.is_whitespace() && !in_quotes => {
                if have {
                    tokens.push(std::mem::take(&mut current));
                    have = false;
                }
            }
            c => {
                have = true;
                current.push(c);
            }
        }
    }
    if in_quotes {
        return Err(Error::InvalidArgument(format!(
            "elastix parameter file: the entry `({body})` has an unterminated quoted string"
        )));
    }
    if have {
        tokens.push(current);
    }
    Ok(tokens)
}

/// Remove one layer of surrounding double quotes, if present.
fn unquote(token: &str) -> &str {
    token
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AFFINE: &str = r#"
// a comment
(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters 1.1 0.2 -0.3 0.05 0.9 0.15 -0.25 0.1 1.05 3.5 -2.25 7.125)
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(Spacing 2.0 3.0 5.0) // trailing comment
(UseDirectionCosines "true")
"#;

    #[test]
    fn parses_the_shape_elastix_writes() {
        let map = ParameterMap::parse(AFFINE).unwrap();
        assert_eq!(map.string("Transform").unwrap(), "AffineTransform");
        assert_eq!(map.usize("NumberOfParameters").unwrap(), 12);
        assert_eq!(map.floats("TransformParameters").unwrap().len(), 12);
        assert_eq!(map.triple("Spacing").unwrap(), [2.0, 3.0, 5.0]);
        assert!(map.flag("UseDirectionCosines", false).unwrap());
        assert!(!map.flag("ComputeZYX", false).unwrap());
    }

    #[test]
    fn an_entry_spanning_lines_is_one_entry() {
        let map = ParameterMap::parse("(TransformParameters 1 2\n 3 4\n 5 6)").unwrap();
        assert_eq!(map.floats("TransformParameters").unwrap().len(), 6);
    }

    #[test]
    fn absent_and_empty_are_different() {
        let map = ParameterMap::parse("(WriteResultImage)").unwrap();
        assert_eq!(map.values("WriteResultImage"), Some(&[][..]));
        assert_eq!(map.values("Nothing"), None);
    }

    #[test]
    fn a_duplicate_key_is_refused() {
        let error = ParameterMap::parse("(Transform \"A\")\n(Transform \"B\")").unwrap_err();
        assert!(format!("{error}").contains("appears twice"), "{error}");
    }

    #[test]
    fn an_unclosed_entry_is_refused() {
        assert!(ParameterMap::parse("(Transform \"A\"").is_err());
    }

    #[test]
    fn a_missing_key_names_itself() {
        let map = ParameterMap::parse("(Transform \"A\")").unwrap();
        let error = map.required("GridSize").unwrap_err();
        assert!(format!("{error}").contains("GridSize"), "{error}");
    }
}
