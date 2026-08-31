// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **A random forest, laid out flat, for classifying every voxel of a volume.**
//!
//! Step 3 and 4 of `docs/design/pixel-classification.md`. That document sets out
//! why this is written here rather than taken from a crate, and the reason is
//! worth repeating because it decides the layout below: the workload is
//! **inference-dominated by orders of magnitude**. Training sees a few thousand
//! brush-stroke voxels and its cost is irrelevant; prediction sees every voxel of
//! a volume this crate exists to handle at 10^8 to 10^9 of them, through every
//! tree. `linfa-trees` fits single trees only, and `smartcore`'s forest keeps its
//! fitted trees private with no access to the node structure — so neither can be
//! converted into a layout we control, and the layout is the whole game.
//!
//! `smartcore` earns a place as a **dev-dependency oracle** instead: train both
//! on the same labels and assert the predictions agree. That is the witness shape
//! `ops::scikitimage_watershed` already uses against skimage.
//!
//! # The layout, and the invariant that makes a walk safe
//!
//! One `Vec<Node>` for the whole forest, with a root index per tree. A node is
//! 24 bytes and carries a feature column, a threshold, and two child indices; a
//! leaf is marked by [`LEAF`](crate::forest::LEAF) in its feature field and its
//! `left` becomes an offset into the forest's flat vote array.
//!
//! **Every child index is strictly greater than its parent's.** That is checked
//! once, in [`Forest::new`](crate::forest::Forest::new), and it is what makes the
//! walk provably
//! terminate without a depth counter in the hot loop — a walk strictly increases
//! its index at every step and the array is finite. A forest built depth-first,
//! which is how [`Forest::train`](crate::forest::Forest::train) builds one and how
//! every tree library emits
//! one, satisfies it for free. The alternative — trusting the input and bounding
//! the loop by a maximum depth — costs a comparison per node on the hottest path
//! in the crate and turns a malformed forest into a silently truncated answer
//! instead of a refusal.
//!
//! # What the predictor is, in this crate's terms
//!
//! A [`Combine`](crate::op::Combine), not a [`BlockOp`](crate::op::BlockOp): it
//! reads 91 images and writes one, which is precisely a fan-in. See
//! [`crate::ops::classify`]. Two properties follow and both matter to the
//! planner:
//!
//! * **reach zero** — a voxel's class depends on that voxel's features and
//!   nothing around it, so the predictor adds nothing to any halo and is
//!   decomposition-invariant by construction;
//! * **not a fold** — it needs every channel at a voxel at once to walk a tree,
//!   so it cannot declare a
//!   [`fold_carrier`](crate::op::Combine::fold_carrier) and its fan-in holds one
//!   buffer per arm. That is the open residency question step 2 recorded.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// The value in [`Node::feature`] that marks a leaf.
///
/// `u32::MAX` rather than a separate tag byte, because a node is read a thousand
/// times per voxel and a branch on a field already loaded is free where a wider
/// struct is not.
pub const LEAF: u32 = u32::MAX;

/// One node: a split, or a leaf.
///
/// 24 bytes — `f64` threshold, three `u32` — and the threshold is `f64` rather
/// than `f32` deliberately. Packing to 16 would fit more of a large forest in
/// cache, and `docs/design/pixel-classification.md` lists the precision question
/// as unmeasured; what it would cost is that a threshold fitted in `f64` and
/// stored in `f32` sends some voxels down the other branch, so the packed forest
/// is *not* the forest that was trained. That is a change to make with a
/// measurement of both sides, not a default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    /// The feature column this node splits on, or [`LEAF`].
    pub feature: u32,
    /// Taken when `features[feature] <= threshold`. For a leaf, the offset into
    /// [`Forest::votes`] where this leaf's class distribution begins.
    pub left: u32,
    /// Taken when `features[feature] > threshold`. Unread for a leaf.
    pub right: u32,
    /// Unread for a leaf.
    pub threshold: f64,
}

impl Node {
    pub fn split(feature: u32, threshold: f64, left: u32, right: u32) -> Self {
        Self {
            feature,
            left,
            right,
            threshold,
        }
    }

    /// A leaf whose class distribution starts at `votes` in [`Forest::votes`].
    pub fn leaf(votes: u32) -> Self {
        Self {
            feature: LEAF,
            left: votes,
            right: 0,
            threshold: 0.0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.feature == LEAF
    }
}

/// A trained forest, and the channel list it was trained against.
#[derive(Debug, Clone, PartialEq)]
pub struct Forest {
    nodes: Vec<Node>,
    roots: Vec<u32>,
    votes: Vec<f32>,
    classes: usize,
    channels: Vec<String>,
}

impl Forest {
    /// Build one, checking everything a walk will assume.
    ///
    /// The checks are not defensive programming against a caller who cannot be
    /// trusted; they are what buys the hot loop its unconditional walk. Each one
    /// is a thing that would otherwise be an infinite loop, an out-of-bounds
    /// index or a wrong answer at 10^9 voxels:
    ///
    /// * **every child index exceeds its parent's**, which makes the walk
    ///   terminate;
    /// * every child index is in range;
    /// * every split's feature column is within `channels`;
    /// * every leaf's vote offset admits a whole class distribution;
    /// * every root is a node;
    /// * the channel names are distinct, since they are what a stack is matched
    ///   against.
    pub fn new(
        nodes: Vec<Node>,
        roots: Vec<u32>,
        votes: Vec<f32>,
        classes: usize,
        channels: Vec<String>,
    ) -> Result<Self> {
        if classes < 2 {
            return Err(Error::InvalidArgument(format!(
                "a classifier needs at least two classes; got {classes}. One class is a \
                 constant image and `ops::voxelwise` writes those for nothing."
            )));
        }
        if roots.is_empty() {
            return Err(Error::InvalidArgument(
                "a forest with no trees predicts nothing; it would answer every voxel from \
                 an empty vote tally, which has no argmax"
                    .to_string(),
            ));
        }
        if channels.is_empty() {
            return Err(Error::InvalidArgument(
                "a forest must name the channels it was trained on; the names are what lets \
                 it be refused against a different feature stack rather than silently \
                 reading the wrong column"
                    .to_string(),
            ));
        }
        let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
        for (index, name) in channels.iter().enumerate() {
            if let Some(first) = seen.insert(name.as_str(), index) {
                return Err(Error::InvalidArgument(format!(
                    "the channel name {name:?} appears at both {first} and {index}. A split \
                     names a column by index and the names are what make an index mean \
                     something, so two columns may not share one."
                )));
            }
        }
        for (index, node) in nodes.iter().enumerate() {
            if node.is_leaf() {
                let start = node.left as usize;
                if start + classes > votes.len() {
                    return Err(Error::InvalidArgument(format!(
                        "node {index} is a leaf whose votes begin at {start}, but {classes} \
                         classes do not fit in the {} votes stored",
                        votes.len()
                    )));
                }
                continue;
            }
            if node.feature as usize >= channels.len() {
                return Err(Error::InvalidArgument(format!(
                    "node {index} splits on feature {} but the forest names only {} channels",
                    node.feature,
                    channels.len()
                )));
            }
            if !node.threshold.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "node {index} has a threshold of {}, which no comparison against a \
                     feature can be meaningful against",
                    node.threshold
                )));
            }
            for (side, child) in [("left", node.left), ("right", node.right)] {
                if child as usize >= nodes.len() {
                    return Err(Error::InvalidArgument(format!(
                        "node {index}'s {side} child is {child}, past the {} nodes stored",
                        nodes.len()
                    )));
                }
                if child as usize <= index {
                    return Err(Error::InvalidArgument(format!(
                        "node {index}'s {side} child is {child}, which does not exceed its \
                         parent. Every child must come after its parent: that is what makes \
                         a walk terminate without counting depth, and a forest that breaks \
                         it could loop forever on one voxel."
                    )));
                }
            }
        }
        for (tree, &root) in roots.iter().enumerate() {
            if root as usize >= nodes.len() {
                return Err(Error::InvalidArgument(format!(
                    "tree {tree}'s root is node {root}, past the {} nodes stored",
                    nodes.len()
                )));
            }
        }
        Ok(Self {
            nodes,
            roots,
            votes,
            classes,
            channels,
        })
    }

    pub fn trees(&self) -> usize {
        self.roots.len()
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn channels(&self) -> &[String] {
        &self.channels
    }

    /// Where each tree begins, in tree order.
    ///
    /// Exposed because a tree's **root split** is the one taken over its whole
    /// bag, and is therefore the place a property of the fit — which column
    /// `mtry` let it consider — is visible undiluted by the deep nodes where a
    /// handful of rows makes any column look good.
    pub fn roots_used(&self) -> &[u32] {
        &self.roots
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn votes(&self) -> &[f32] {
        &self.votes
    }

    /// The deepest path in any tree, in nodes visited.
    ///
    /// Not used by the walk — see the module header for why it is not — but it is
    /// what `cost_per_voxel` is proportional to, so the predictor asks for it
    /// once when it is built rather than declaring a constant that a differently
    /// shaped forest would make wrong.
    pub fn depth(&self) -> usize {
        let mut depth = vec![0usize; self.nodes.len()];
        // One reverse pass suffices *because* children exceed parents: by the
        // time index `i` is read, every node it can reach has been settled.
        for index in (0..self.nodes.len()).rev() {
            let node = self.nodes[index];
            depth[index] = if node.is_leaf() {
                1
            } else {
                1 + depth[node.left as usize].max(depth[node.right as usize])
            };
        }
        self.roots
            .iter()
            .map(|&root| depth[root as usize])
            .max()
            .unwrap_or(0)
    }

    /// The mean root-to-leaf path length, weighted by nothing — the arithmetic
    /// mean over leaves.
    ///
    /// This, and not [`Self::depth`], is what a voxel actually pays: a balanced
    /// tree's average path is close to its depth and a badly unbalanced one's is
    /// far below it, and a cost model built on the maximum would misprice the
    /// second by the ratio.
    pub fn mean_path(&self) -> f64 {
        let mut total = 0.0;
        for &root in &self.roots {
            let (sum, leaves) = self.path_sum(root as usize, 1);
            if leaves > 0 {
                total += sum as f64 / leaves as f64;
            }
        }
        total / self.roots.len() as f64
    }

    fn path_sum(&self, index: usize, depth: usize) -> (usize, usize) {
        let node = self.nodes[index];
        if node.is_leaf() {
            return (depth, 1);
        }
        let (left_sum, left_leaves) = self.path_sum(node.left as usize, depth + 1);
        let (right_sum, right_leaves) = self.path_sum(node.right as usize, depth + 1);
        (left_sum + right_sum, left_leaves + right_leaves)
    }

    /// **The hot loop.** One tree, one voxel: the leaf's vote offset.
    ///
    /// No bounds check is skipped and none needs to be — `new` established that
    /// every index in play is in range — and no depth counter is carried,
    /// because every step strictly increases `index` over a finite array.
    #[inline]
    fn walk(&self, root: u32, features: &[f64]) -> u32 {
        let mut index = root as usize;
        loop {
            let node = self.nodes[index];
            if node.feature == LEAF {
                return node.left;
            }
            index = if features[node.feature as usize] <= node.threshold {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }

    /// Accumulate every tree's leaf distribution for one voxel into `into`,
    /// which must hold [`Self::classes`] values and is **added to** rather than
    /// overwritten.
    ///
    /// Adding rather than assigning is what lets a caller reuse one buffer over
    /// a block without a clear per voxel — the predictor does exactly that — and
    /// it is stated here because a caller who assumed otherwise would get a
    /// running total that only ever grew.
    pub fn accumulate(&self, features: &[f64], into: &mut [f32]) {
        for &root in &self.roots {
            let at = self.walk(root, features) as usize;
            for class in 0..self.classes {
                into[class] += self.votes[at + class];
            }
        }
    }

    /// The class with the most votes, ties going to the **lowest** class index.
    ///
    /// A rule rather than whatever the comparison happened to do: a forest with
    /// an even number of trees ties often on a two-class problem, and a tie
    /// broken by iteration order is a tie broken by the order the trees were
    /// bagged in, which is a seed.
    pub fn predict(&self, features: &[f64], scratch: &mut [f32]) -> u32 {
        scratch.fill(0.0);
        self.accumulate(features, scratch);
        let mut best = 0usize;
        for class in 1..self.classes {
            if scratch[class] > scratch[best] {
                best = class;
            }
        }
        best as u32
    }

    /// The share of the vote `class` received, in `[0, 1]`.
    pub fn probability(&self, features: &[f64], class: usize, scratch: &mut [f32]) -> f64 {
        scratch.fill(0.0);
        self.accumulate(features, scratch);
        let total: f32 = scratch.iter().sum();
        if total > 0.0 {
            scratch[class] as f64 / total as f64
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------- the trainer --

/// The labelled voxels a forest is fitted to: one row per voxel, one column per
/// channel.
///
/// **Sparse by construction, and that is the workload.** Labkit and ilastik are
/// built on brush strokes — a few thousand labelled voxels out of 10^9 — and
/// retraining in seconds from them. Nothing here is shaped for a dense label
/// volume, and nothing needs to be: the rows are gathered at the labelled voxels
/// and the unlabelled ones are never materialised.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    features: Vec<f64>,
    labels: Vec<u32>,
    channels: Vec<String>,
    classes: usize,
}

impl Samples {
    /// `features` is row-major: `rows x channels.len()`.
    pub fn new(features: Vec<f64>, labels: Vec<u32>, channels: Vec<String>) -> Result<Self> {
        if channels.is_empty() {
            return Err(Error::InvalidArgument(
                "samples must name their channels; the names travel into the forest and are \
                 what lets it be refused against a different feature stack"
                    .to_string(),
            ));
        }
        if labels.is_empty() {
            return Err(Error::InvalidArgument(
                "no labelled voxels; there is nothing to fit".to_string(),
            ));
        }
        if features.len() != labels.len() * channels.len() {
            return Err(Error::InvalidArgument(format!(
                "{} feature values do not make {} rows of {} channels",
                features.len(),
                labels.len(),
                channels.len()
            )));
        }
        if let Some(bad) = features.iter().position(|value| !value.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "the feature value at {bad} is {}. A threshold fitted against it would be \
                 met by no voxel or by every voxel, depending on which side of the \
                 comparison the non-finite value landed.",
                features[bad]
            )));
        }
        let classes = labels.iter().copied().max().unwrap_or(0) as usize + 1;
        if classes < 2 {
            return Err(Error::InvalidArgument(
                "every labelled voxel has the same class, so there is nothing to \
                 discriminate. A forest fitted to one class answers it everywhere."
                    .to_string(),
            ));
        }
        Ok(Self {
            features,
            labels,
            channels,
            classes,
        })
    }

    pub fn rows(&self) -> usize {
        self.labels.len()
    }

    pub fn columns(&self) -> usize {
        self.channels.len()
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn channels(&self) -> &[String] {
        &self.channels
    }

    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// One row, as the predictor would see it.
    pub fn row(&self, row: usize) -> &[f64] {
        let columns = self.channels.len();
        &self.features[row * columns..(row + 1) * columns]
    }

    fn at(&self, row: usize, column: usize) -> f64 {
        self.features[row * self.channels.len() + column]
    }
}

/// How a forest is fitted. Every field has a default that matches what the two
/// reference tools do, and the defaults are what the workflow wrappers use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingSpec {
    /// How many trees. Labkit's WEKA default is 200; this is 100, which is
    /// scikit-learn's and ilastik's, and the cost of prediction is linear in it.
    pub trees: usize,
    /// Columns considered at each split. `None` is `round(sqrt(columns))`, which
    /// is the standard choice for classification and is what makes the trees
    /// decorrelated enough for bagging to be worth anything.
    pub mtry: Option<usize>,
    /// A hard stop, so a pathological split sequence cannot build a tree as deep
    /// as the sample count. Prediction cost is linear in the path length, so
    /// this is a cost ceiling as much as a regulariser.
    pub max_depth: usize,
    /// A node with fewer rows than this becomes a leaf.
    pub min_samples_leaf: usize,
    /// The bagging and `mtry` draws. **The whole fit is a function of this**, so
    /// two runs with the same seed give the same forest, byte for byte.
    pub seed: u64,
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self {
            trees: 100,
            mtry: None,
            max_depth: 20,
            min_samples_leaf: 1,
            seed: 0x5eed,
        }
    }
}

/// `splitmix64`, and not `rand`.
///
/// The crate does not depend on `rand` and this needs three things from a
/// generator: that it be deterministic given a seed, that it be reproducible
/// across platforms, and that it not correlate consecutive draws enough to bias
/// which columns `mtry` offers. `splitmix64` is eight lines, is the standard
/// seeder for the xoshiro family, and passes BigCrush; a linear congruential
/// generator would have failed the third — its low bits cycle with a short
/// period, and `next_below` reads exactly those.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, bound)`, by Lemire's multiply-shift — which is unbiased
    /// only with the rejection step, and has it. A plain modulo would favour the
    /// low `u64::MAX % bound` values, which for `mtry` over 91 columns is a
    /// standing preference for the first few features.
    fn next_below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let draw = self.next();
            let wide = (draw as u128) * (bound as u128);
            if (wide as u64) >= threshold {
                return (wide >> 64) as usize;
            }
        }
    }
}

/// One node under construction, before the tree is flattened.
struct Building {
    rows: Vec<usize>,
    depth: usize,
}

impl Forest {
    /// **Fit a forest.** Bagging, `mtry` column subsampling, and Gini.
    ///
    /// Nothing here is novel and nothing here needs to be — the design document
    /// is explicit that training is the cheap end of this workload and does not
    /// have to be world-class to match a tool that retrains per dataset. What it
    /// has to be is *deterministic*, so that a fit can be reproduced and
    /// compared against an oracle, and that is what `spec.seed` buys.
    ///
    /// The trees are built depth-first with each subtree appended after its
    /// parent, which is what establishes the invariant [`Forest::new`] checks:
    /// every child index exceeds its parent's, so a walk terminates.
    pub fn train(samples: &Samples, spec: &TrainingSpec) -> Result<Forest> {
        if spec.trees == 0 {
            return Err(Error::InvalidArgument(
                "a forest of no trees has no vote to take".to_string(),
            ));
        }
        if spec.max_depth == 0 {
            return Err(Error::InvalidArgument(
                "a maximum depth of zero admits no tree, not even a single leaf".to_string(),
            ));
        }
        let columns = samples.columns();
        let mtry = spec
            .mtry
            .unwrap_or_else(|| ((columns as f64).sqrt().round() as usize).max(1))
            .clamp(1, columns);
        let min_leaf = spec.min_samples_leaf.max(1);

        let mut rng = SplitMix(spec.seed);
        let mut nodes: Vec<Node> = Vec::new();
        let mut roots: Vec<u32> = Vec::new();
        let mut votes: Vec<f32> = Vec::new();

        for _ in 0..spec.trees {
            // Bagging: `rows` draws with replacement, which is the bootstrap.
            let bag: Vec<usize> = (0..samples.rows())
                .map(|_| rng.next_below(samples.rows()))
                .collect();
            roots.push(nodes.len() as u32);
            grow(
                samples,
                Building {
                    rows: bag,
                    depth: 1,
                },
                spec.max_depth,
                min_leaf,
                mtry,
                &mut rng,
                &mut nodes,
                &mut votes,
            );
        }

        Forest::new(
            nodes,
            roots,
            votes,
            samples.classes(),
            samples.channels().to_vec(),
        )
    }
}

/// Append one subtree, returning nothing: the caller knows its root is where
/// `nodes` stood when it called.
#[allow(clippy::too_many_arguments)]
fn grow(
    samples: &Samples,
    node: Building,
    max_depth: usize,
    min_leaf: usize,
    mtry: usize,
    rng: &mut SplitMix,
    nodes: &mut Vec<Node>,
    votes: &mut Vec<f32>,
) {
    let classes = samples.classes();
    let mut tally = vec![0.0f32; classes];
    for &row in &node.rows {
        tally[samples.labels[row] as usize] += 1.0;
    }
    let pure = tally.iter().filter(|&&count| count > 0.0).count() <= 1;

    let leaf = |nodes: &mut Vec<Node>, votes: &mut Vec<f32>, tally: &[f32]| {
        let at = votes.len() as u32;
        // Normalised, so that a tree grown on a large bag does not outvote one
        // grown on a small one. Every tree gets one vote, distributed.
        let total: f32 = tally.iter().sum();
        for &count in tally {
            votes.push(if total > 0.0 { count / total } else { 0.0 });
        }
        nodes.push(Node::leaf(at));
    };

    if pure || node.depth >= max_depth || node.rows.len() <= min_leaf {
        leaf(nodes, votes, &tally);
        return;
    }

    let Some((column, threshold)) = best_split(samples, &node.rows, mtry, min_leaf, rng) else {
        leaf(nodes, votes, &tally);
        return;
    };

    let (mut left, mut right) = (Vec::new(), Vec::new());
    for &row in &node.rows {
        if samples.at(row, column) <= threshold {
            left.push(row);
        } else {
            right.push(row);
        }
    }

    // The parent is written now, with placeholder children, and patched once
    // both subtrees are appended. That ordering is the invariant: `here` is
    // fixed before anything below it exists, so every child index exceeds it.
    let here = nodes.len();
    nodes.push(Node::split(column as u32, threshold, 0, 0));
    let left_at = nodes.len() as u32;
    grow(
        samples,
        Building {
            rows: left,
            depth: node.depth + 1,
        },
        max_depth,
        min_leaf,
        mtry,
        rng,
        nodes,
        votes,
    );
    let right_at = nodes.len() as u32;
    grow(
        samples,
        Building {
            rows: right,
            depth: node.depth + 1,
        },
        max_depth,
        min_leaf,
        mtry,
        rng,
        nodes,
        votes,
    );
    nodes[here].left = left_at;
    nodes[here].right = right_at;
}

/// The best `(column, threshold)` over `mtry` randomly drawn columns, by Gini
/// decrease, or `None` if no column admits a split that leaves both sides at
/// least `min_leaf` rows.
///
/// **Thresholds are midpoints between consecutive distinct values**, and the
/// comparison is `<=`. That pairing is not arbitrary: a threshold equal to an
/// observed value would put every row holding it on the left, so the split a
/// caller reading the tree would expect from `x <= v` and the split that
/// separates the two groups differ by exactly the ties. A midpoint has no ties
/// by construction.
fn best_split(
    samples: &Samples,
    rows: &[usize],
    mtry: usize,
    min_leaf: usize,
    rng: &mut SplitMix,
) -> Option<(usize, f64)> {
    let classes = samples.classes();
    let columns = samples.columns();
    let mut best: Option<(f64, usize, f64)> = None;

    // Drawn with replacement, which is `mtry` *tries* rather than `mtry`
    // distinct columns. The difference is small at `sqrt(91) = 10` out of 91 and
    // it removes a rejection loop from the inner search; scikit-learn draws
    // without replacement and the distinction is not one any published
    // comparison of forests turns on.
    let mut order: Vec<usize> = Vec::with_capacity(rows.len());
    for _ in 0..mtry {
        let column = rng.next_below(columns);
        order.clear();
        order.extend_from_slice(rows);
        order.sort_by(|&left, &right| {
            samples
                .at(left, column)
                .total_cmp(&samples.at(right, column))
        });

        let mut left_tally = vec![0.0f64; classes];
        let mut right_tally = vec![0.0f64; classes];
        for &row in &order {
            right_tally[samples.labels[row] as usize] += 1.0;
        }
        let total = order.len() as f64;

        for at in 0..order.len().saturating_sub(1) {
            let row = order[at];
            let label = samples.labels[row] as usize;
            left_tally[label] += 1.0;
            right_tally[label] -= 1.0;

            let here = samples.at(row, column);
            let next = samples.at(order[at + 1], column);
            if here == next {
                continue;
            }
            let (left_count, right_count) = ((at + 1) as f64, total - (at + 1) as f64);
            if (at + 1) < min_leaf || (order.len() - at - 1) < min_leaf {
                continue;
            }
            // The weighted Gini impurity of the two sides. Minimising it is
            // maximising the decrease, since the parent's impurity is fixed
            // across every candidate at this node.
            let impurity = left_count * gini(&left_tally, left_count)
                + right_count * gini(&right_tally, right_count);
            if best.is_none_or(|(current, _, _)| impurity < current) {
                best = Some((impurity, column, here + (next - here) / 2.0));
            }
        }
    }
    best.map(|(_, column, threshold)| (column, threshold))
}

fn gini(tally: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    1.0 - tally
        .iter()
        .map(|&count| {
            let share = count / total;
            share * share
        })
        .sum::<f64>()
}
