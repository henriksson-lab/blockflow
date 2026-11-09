//! Bounded, source-grid vessel tracing. This resident graph operation consumes a
//! caller-cropped ROI; it never reads a dataset or performs annotation mutations.
//! XYZ voxel-centre coordinates, X-fast arrays. Physical edge costs use spacing.
use serde::{Deserialize, Serialize};
#[path = "vessel_trace_constraints.rs"]
mod constraints;
pub use constraints::{
    candidate_branches, fit_paths_with_required_points, trace_with_required_points,
    ConstrainedTraceResult, PointMatch, RequiredPoint,
};
use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TraceConfig {
    pub shape: [usize; 3],
    pub spacing: [f64; 3],
    /// Alternatives: any one source may initiate a trace.
    pub source_group: Vec<[f64; 3]>,
    /// One trace per target group, ending at its cheapest reachable member.
    pub target_groups: Vec<Vec<[f64; 3]>>,
    pub max_voxels: usize,
    pub max_expansions: usize,
    pub image_weight: f64,
    pub clearance_weight: f64,
    pub refinement_iterations: usize,
    pub max_move: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceResult {
    pub paths: Vec<Vec<[f64; 3]>>,
    pub expanded: usize,
    pub endpoint_status: String,
}
#[derive(Clone, Copy)]
struct Entry(f64, usize);
impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0 && self.1 == o.1
    }
}
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then_with(|| o.1.cmp(&self.1))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
fn xyz(i: usize, s: [usize; 3]) -> [f64; 3] {
    [
        (i % s[0]) as f64,
        ((i / s[0]) % s[1]) as f64,
        (i / (s[0] * s[1])) as f64,
    ]
}
fn index(p: [f64; 3], s: [usize; 3]) -> Option<usize> {
    let mut q = [0; 3];
    for a in 0..3 {
        if !p[a].is_finite() || p[a] < 0. || p[a] > (s[a] - 1) as f64 {
            return None;
        }
        q[a] = p[a].round() as usize;
    }
    Some(q[0] + s[0] * (q[1] + s[1] * q[2]))
}
/// Exact piecewise-constant voxel membership along a segment: includes grid-face
/// intervals, so smoothing cannot jump a thin forbidden septum.
fn segment_ok(a: [f64; 3], b: [f64; 3], s: [usize; 3], allowed: &[u8]) -> bool {
    if index(a, s).is_none() || index(b, s).is_none() {
        return false;
    }
    let mut t = vec![0., 1.];
    for k in 0..3 {
        let d = b[k] - a[k];
        if d.abs() < 1e-12 {
            continue;
        }
        let lo = a[k].min(b[k]);
        let hi = a[k].max(b[k]);
        for v in (lo - 0.5).floor() as i64..=(hi + 0.5).ceil() as i64 {
            let u = (v as f64 + 0.5 - a[k]) / d;
            if u > 0. && u < 1. {
                t.push(u);
            }
        }
    }
    t.sort_by(f64::total_cmp);
    t.windows(2).all(|w| {
        let u = (w[0] + w[1]) * 0.5;
        let p = std::array::from_fn(|k| a[k] + u * (b[k] - a[k]));
        index(p, s).is_some_and(|i| allowed[i] != 0)
    })
}
fn edt_line(x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let sites: Vec<_> = (0..n).filter(|&i| x[i].is_finite()).collect();
    if sites.is_empty() {
        return vec![f64::INFINITY; n];
    }
    let mut v = vec![0; n];
    let mut z = vec![0.; n + 1];
    let mut k = 0;
    v[0] = sites[0];
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let a = h * h;
    for &q in sites.iter().skip(1) {
        let mut s;
        loop {
            let p = v[k];
            s = ((x[q] + a * (q * q) as f64) - (x[p] + a * (p * p) as f64))
                / (2. * a * (q as f64 - p as f64));
            if s > z[k] || k == 0 {
                break;
            }
            k -= 1;
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    let mut r = vec![0.; n];
    k = 0;
    for q in 0..n {
        while z[k + 1] < (q as f64) {
            k += 1;
        }
        let d = q as f64 - v[k] as f64;
        r[q] = a * d * d + x[v[k]];
    }
    r
}
fn edt(m: &[u8], shape: [usize; 3], spacing: [f64; 3]) -> Vec<f64> {
    let mut a: Vec<f64> = m
        .iter()
        .map(|&v| if v == 0 { 0. } else { f64::INFINITY })
        .collect();
    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        for y in 0..shape[v] {
            for x in 0..shape[u] {
                let mut p = [0; 3];
                p[u] = x;
                p[v] = y;
                let ids: Vec<_> = (0..shape[axis])
                    .map(|z| {
                        p[axis] = z;
                        p[0] + shape[0] * (p[1] + shape[1] * p[2])
                    })
                    .collect();
                let line: Vec<_> = ids.iter().map(|&i| a[i]).collect();
                let b = edt_line(&line, spacing[axis]);
                for (&i, d) in ids.iter().zip(b) {
                    a[i] = d;
                }
            }
        }
    }
    a.into_iter().map(f64::sqrt).collect()
}

/// Conservative local support for an intensity gate: at least two orthogonal
/// axes must have observed gate-failing samples on BOTH sides within the search
/// radius. An ROI boundary is never a wall. This is local wall evidence, not a
/// tissue classifier; half-cut/open lumens can correctly remain unsupported.
/// Six linear scans, O(N), and no full-image context outside the supplied ROI.
pub fn wall_supported_mask(
    gate: &[u8],
    shape: [usize; 3],
    spacing: [f64; 3],
    search_voxels: usize,
) -> Result<Vec<u8>, String> {
    let n = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("ROI overflow")?;
    if shape.contains(&0)
        || n != gate.len()
        || n > 4_000_000
        || search_voxels == 0
        || search_voxels > 64
        || spacing.iter().any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("invalid wall-support ROI/search calibration".into());
    }
    let pitch = spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let mut count = vec![0u8; n];
    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        let limit = (search_voxels as f64 * pitch / spacing[axis]).floor() as usize;
        if limit == 0 {
            continue;
        }
        for y in 0..shape[v] {
            for x in 0..shape[u] {
                let mut ids = Vec::with_capacity(shape[axis]);
                for z in 0..shape[axis] {
                    let mut p = [0; 3];
                    p[u] = x;
                    p[v] = y;
                    p[axis] = z;
                    ids.push(p[0] + shape[0] * (p[1] + shape[1] * p[2]));
                }
                let mut left = vec![false; ids.len()];
                let mut last = None;
                for (k, &i) in ids.iter().enumerate() {
                    if gate[i] == 0 {
                        last = Some(k);
                    } else {
                        left[k] = last.is_some_and(|j| k - j <= limit);
                    }
                }
                last = None;
                for (k, &i) in ids.iter().enumerate().rev() {
                    if gate[i] == 0 {
                        last = Some(k);
                    } else if left[k] && last.is_some_and(|j| j - k <= limit) {
                        count[i] += 1;
                    }
                }
            }
        }
    }
    Ok(gate
        .iter()
        .zip(count)
        .map(|(&m, c)| u8::from(m != 0 && c >= 2))
        .collect())
}

/// Evidence is finite in `[0,1]` (larger means more lumen-like); allowed=0 is a
/// hard barrier. No-path is an error, never a straight-line fallback. Source and
/// target fractional points are connected only when their entire snapped link
/// is allowed. Endpoints remain fixed during bounded physical refinement.
pub fn trace(evidence: &[f32], allowed: &[u8], c: &TraceConfig) -> Result<TraceResult, String> {
    let n = c
        .shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("ROI size overflow")?;
    if c.shape.contains(&0)
        || n > c.max_voxels
        || n > 4_000_000
        || evidence.len() != n
        || allowed.len() != n
    {
        return Err("invalid or oversized ROI".into());
    }
    if c.spacing.iter().any(|x| !x.is_finite() || *x <= 0.)
        || !c.image_weight.is_finite()
        || c.image_weight < 0.
        || !c.clearance_weight.is_finite()
        || c.clearance_weight < 0.
        || !c.max_move.is_finite()
        || c.max_move < 0.
        || c.refinement_iterations > 200
        || c.max_expansions > 8_000_000
        || c.max_expansions == 0
    {
        return Err("invalid tracing limits/calibration".into());
    }
    if c.source_group.is_empty()
        || c.source_group.len() > 64
        || c.target_groups.is_empty()
        || c.target_groups.len() > 32
        || evidence
            .iter()
            .any(|x| !x.is_finite() || *x < 0. || *x > 1.)
    {
        return Err("invalid evidence or endpoint groups".into());
    }
    let endpoint = |p: [f64; 3]| -> Result<usize, String> {
        let i = index(p, c.shape).ok_or("endpoint outside ROI")?;
        if allowed[i] == 0 || !segment_ok(p, xyz(i, c.shape), c.shape, allowed) {
            return Err("endpoint in forbidden background".into());
        }
        Ok(i)
    };
    let sources = c
        .source_group
        .iter()
        .map(|&p| endpoint(p))
        .collect::<Result<Vec<_>, _>>()?;
    let targets = c
        .target_groups
        .iter()
        .map(|g| {
            if g.is_empty() || g.len() > 64 {
                return Err("invalid target group".into());
            }
            g.iter()
                .map(|&p| endpoint(p))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    // Exact observed-background EDT, deliberately no zero padding at ROI/acquisition caps.
    let clearance = edt(allowed, c.shape, c.spacing);
    let pitch = c.spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let penalty = |i: usize| {
        1. + c.image_weight * (1. - evidence[i] as f64)
            + c.clearance_weight * pitch / (pitch + clearance[i])
    };
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut root = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();
    for (j, &i) in sources.iter().enumerate() {
        dist[i] = 0.;
        root[i] = j;
        heap.push(Entry(0., i));
    }
    let mut reached = vec![None; targets.len()];
    let mut target_lookup = std::collections::HashMap::<usize, Vec<(usize, usize)>>::new();
    for (g, group) in targets.iter().enumerate() {
        for (j, &i) in group.iter().enumerate() {
            target_lookup.entry(i).or_default().push((g, j));
        }
    }
    let mut expanded = 0;
    while let Some(Entry(d, i)) = heap.pop() {
        if d != dist[i] {
            continue;
        }
        expanded += 1;
        if expanded > c.max_expansions {
            return Err("trace work budget exhausted".into());
        }
        if let Some(groups) = target_lookup.get(&i) {
            for &(g, j) in groups {
                if reached[g].is_none() {
                    reached[g] = Some((i, j));
                }
            }
        }
        if reached.iter().all(Option::is_some) {
            break;
        }
        let p = xyz(i, c.shape);
        for a in 0..3 {
            for sign in [-1., 1.] {
                let mut q = p;
                q[a] += sign;
                let Some(j) = index(q, c.shape) else {
                    continue;
                };
                if allowed[j] == 0 {
                    continue;
                }
                let cost = c.spacing[a] * (0.5 * (penalty(i) + penalty(j)));
                let v = d + cost;
                if v < dist[j] {
                    dist[j] = v;
                    prev[j] = i;
                    root[j] = root[i];
                    heap.push(Entry(v, j));
                }
            }
        }
    }
    let mut paths = Vec::new();
    for (g, end) in reached.into_iter().enumerate() {
        let (mut i, j) = end.ok_or("no lumen-constrained path to a target group")?;
        let source = root[i];
        let mut path = vec![c.target_groups[g][j]];
        loop {
            if path.len() > 32_768 {
                return Err("trace path sample budget exceeded; shorten the ROI".into());
            }
            let p = xyz(i, c.shape);
            if path.last() != Some(&p) {
                path.push(p);
            }
            if prev[i] == usize::MAX {
                break;
            }
            i = prev[i];
        }
        let src = c.source_group[source];
        if path.last() != Some(&src) {
            path.push(src);
        }
        path.reverse();
        paths.push(path);
    }
    // Assemble the common Dijkstra tree BEFORE fitting. Degree-two chains
    // share exact junction nodes; the shared trunk is stored/refined only once.
    let mut adjacency =
        std::collections::BTreeMap::<[u64; 3], std::collections::BTreeSet<[u64; 3]>>::new();
    let mut endpoints = std::collections::BTreeSet::new();
    for path in &paths {
        endpoints.insert(path.first().unwrap().map(f64::to_bits));
        endpoints.insert(path.last().unwrap().map(f64::to_bits));
        for w in path.windows(2) {
            let a = w[0].map(f64::to_bits);
            let b = w[1].map(f64::to_bits);
            if a != b {
                adjacency.entry(a).or_default().insert(b);
                adjacency.entry(b).or_default().insert(a);
            }
        }
    }
    let terminals = adjacency
        .iter()
        .filter(|(p, n)| n.len() != 2 || endpoints.contains(*p))
        .map(|(&p, _)| p)
        .collect::<std::collections::BTreeSet<_>>();
    let mut visited = std::collections::BTreeSet::new();
    let mut chains = Vec::new();
    for &start in &terminals {
        for &next in &adjacency[&start] {
            let key = if start < next {
                (start, next)
            } else {
                (next, start)
            };
            if !visited.insert(key) {
                continue;
            }
            let mut chain = vec![start.map(f64::from_bits)];
            let mut previous = start;
            let mut at = next;
            loop {
                chain.push(at.map(f64::from_bits));
                if terminals.contains(&at) {
                    break;
                }
                let following = *adjacency[&at].iter().find(|&&p| p != previous).unwrap();
                let key = if at < following {
                    (at, following)
                } else {
                    (following, at)
                };
                visited.insert(key);
                previous = at;
                at = following;
            }
            // Orient single-chain endpoint editing in the requested direction.
            if paths.len() == 1 && chain.first() != paths[0].first() {
                chain.reverse();
            }
            chains.push(chain);
        }
    }
    paths = chains;
    for path in &mut paths {
        refine_path(path, &penalty, allowed, c, &[]);
    }
    Ok(TraceResult {
        paths,
        expanded,
        endpoint_status:
            "user-selected; anatomical termination and radius unknown; ROI exits censored".into(),
    })
}

fn refine_path(
    path: &mut [[f64; 3]],
    penalty: &impl Fn(usize) -> f64,
    allowed: &[u8],
    c: &TraceConfig,
    anchors: &[[f64; 3]],
) {
    if path.len() < 3 {
        return;
    }
    let pitch = c.spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let initial = path.to_vec();
    // Generalized from recall_relax_mask_graph: bounded coordinate descent
    // of image/medial attraction + physical bending + initial-path anchor.
    // Fixed endpoints and exact segment membership are hard constraints.
    let sample = |p: [f64; 3]| -> f64 {
        let lo = p.map(|v| v.floor() as usize);
        let mut value = 0.;
        for z in 0..=1 {
            for y in 0..=1 {
                for x in 0..=1 {
                    let off = [x, y, z];
                    let q: [usize; 3] =
                        std::array::from_fn(|a| (lo[a] + off[a]).min(c.shape[a] - 1));
                    let weight = (0..3)
                        .map(|a| {
                            if off[a] == 0 {
                                1. - (p[a] - lo[a] as f64)
                            } else {
                                p[a] - lo[a] as f64
                            }
                        })
                        .product::<f64>();
                    value += weight * penalty(q[0] + c.shape[0] * (q[1] + c.shape[1] * q[2]));
                }
            }
        }
        value
    };
    let energy = |p: &[[f64; 3]], k: usize| -> f64 {
        let mut e = 0.15
            * (0..3)
                .map(|a| ((p[k][a] - initial[k][a]) * c.spacing[a] / pitch).powi(2))
                .sum::<f64>()
            + sample(p[k]);
        for j in k.saturating_sub(1).max(1)..=(k + 1).min(p.len() - 2) {
            e += 0.2
                * (0..3)
                    .map(|a| {
                        ((p[j - 1][a] - 2. * p[j][a] + p[j + 1][a]) * c.spacing[a] / pitch).powi(2)
                    })
                    .sum::<f64>();
        }
        e
    };
    for iteration in 0..c.refinement_iterations {
        for k in 1..path.len() - 1 {
            let old = path[k];
            if anchors.contains(&old) {
                continue;
            }
            let mut winner = old;
            let mut best = energy(path, k);
            let step = if iteration < c.refinement_iterations * 2 / 3 {
                0.2 * pitch
            } else {
                0.05 * pitch
            };
            let mut trials = vec![std::array::from_fn(|a| {
                old[a] * 0.5 + (path[k - 1][a] + path[k + 1][a]) * 0.25
            })];
            for a in 0..3 {
                for sign in [-1., 1.] {
                    let mut p = old;
                    p[a] += sign * step / c.spacing[a];
                    trials.push(p);
                }
            }
            for candidate in trials {
                let movement = (0..3)
                    .map(|a| ((candidate[a] - initial[k][a]) * c.spacing[a]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if movement > c.max_move
                    || index(candidate, c.shape).is_none()
                    || !segment_ok(path[k - 1], candidate, c.shape, allowed)
                    || !segment_ok(candidate, path[k + 1], c.shape, allowed)
                {
                    continue;
                }
                path[k] = candidate;
                let e = energy(path, k);
                if e < best {
                    best = e;
                    winner = candidate;
                }
            }
            path[k] = winner;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> TraceConfig {
        TraceConfig {
            shape: [17, 17, 3],
            spacing: [1., 1., 4.],
            source_group: vec![[2., 2., 1.]],
            target_groups: vec![vec![[14., 14., 1.]]],
            max_voxels: 10000,
            max_expansions: 10000,
            image_weight: 8.,
            clearance_weight: 2.,
            refinement_iterations: 20,
            max_move: 1.,
        }
    }
    #[test]
    fn curved_tube_and_groups_roundtrip() {
        let mut c = cfg();
        let n = c.shape.iter().product();
        let mut m = vec![0; n];
        for y in 2..=14 {
            m[2 + 17 * (y + 17)] = 1;
        }
        for x in 2..=14 {
            m[x + 17 * (14 + 17)] = 1;
        }
        let p = m.iter().map(|&x| x as f32).collect::<Vec<_>>();
        c.target_groups.push(vec![[2., 8., 1.]]);
        let r = trace(&p, &m, &c).unwrap();
        assert_eq!(r.paths.len(), 2);
        assert!(r.paths.iter().flatten().any(|v| v[1] > 13. && v[0] < 3.));
        for path in &r.paths {
            for w in path.windows(2) {
                assert!(segment_ok(w[0], w[1], c.shape, &m));
            }
        }
        let s = serde_json::to_string(&c).unwrap();
        let copy: TraceConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(copy.spacing, [1., 1., 4.]);
    }
    #[test]
    fn physical_route_choice_and_subvoxel_endpoints() {
        let mut c = cfg();
        c.spacing = [1., 1., 10.];
        c.refinement_iterations = 0;
        c.clearance_weight = 0.;
        c.source_group = vec![[2.1, 2., 1.]];
        c.target_groups = vec![vec![[13.9, 2., 1.]]];
        let mut m = vec![0; 17 * 17 * 3];
        for y in 2..=8 {
            for x in [2, 14] {
                m[x + 17 * (y + 17)] = 1;
            }
        }
        for x in 2..=14 {
            m[x + 17 * (8 + 17)] = 1;
            m[x + 17 * (2 + 17 * 2)] = 1;
        }
        let p = vec![1.; m.len()];
        let r = trace(&p, &m, &c).unwrap();
        assert_eq!(r.paths[0].first(), Some(&c.source_group[0]));
        assert_eq!(r.paths[0].last(), Some(&c.target_groups[0][0]));
        assert!(r.paths[0].iter().any(|p| p[1] == 8.));
        assert!(r.paths[0].iter().all(|p| p[2] == 1.));
        c.spacing = [1.; 3];
        let r = trace(&p, &m, &c).unwrap();
        assert!(r.paths[0].iter().any(|p| p[2] == 2.));
    }
    #[test]
    fn y_groups_have_one_shared_trunk_and_junction() {
        let mut c = cfg();
        c.refinement_iterations = 20;
        c.source_group = vec![[8., 2., 1.]];
        c.target_groups = vec![vec![[2., 12., 1.]], vec![[14., 12., 1.]]];
        let mut m = vec![0; 17 * 17 * 3];
        for y in 2..=12 {
            m[8 + 17 * (y + 17)] = 1;
        }
        for x in 2..=14 {
            m[x + 17 * (12 + 17)] = 1;
        }
        let r = trace(&vec![1.; m.len()], &m, &c).unwrap();
        assert_eq!(r.paths.len(), 3);
        let junction = [8., 12., 1.];
        assert_eq!(
            r.paths
                .iter()
                .filter(|p| p.first() == Some(&junction) || p.last() == Some(&junction))
                .count(),
            3
        );
        assert_eq!(
            r.paths.iter().filter(|p| p.contains(&[8., 5., 1.])).count(),
            1
        );
    }
    #[test]
    fn dark_exterior_is_not_enclosed_and_roi_caps_are_not_walls() {
        let shape = [13, 13, 3];
        let mut gate = vec![1; 13 * 13 * 3];
        for z in 0..3 {
            for y in 3..=9 {
                for x in 3..=9 {
                    if x == 3 || x == 9 || y == 3 || y == 9 {
                        gate[x + 13 * (y + 13 * z)] = 0;
                    }
                }
            }
        }
        let supported = wall_supported_mask(&gate, shape, [1., 1., 4.], 8).unwrap();
        assert_eq!(supported[6 + 13 * (6 + 13)], 1);
        assert_eq!(supported[1 + 13 * (1 + 13)], 0);
        assert_eq!(supported[6 + 13 * 6], 1);
        assert!(wall_supported_mask(&vec![1; gate.len()], shape, [1.; 3], 8)
            .unwrap()
            .iter()
            .all(|&v| v == 0));
    }
    #[test]
    fn forbidden_septum_no_fallback() {
        let c = cfg();
        let n = c.shape.iter().product();
        let mut m = vec![1; n];
        for z in 0..3 {
            for y in 0..17 {
                m[8 + 17 * (y + 17 * z)] = 0;
            }
        }
        assert!(trace(&vec![1.; n], &m, &c)
            .unwrap_err()
            .contains("no lumen"));
        assert!(!segment_ok([7., 5., 1.], [9., 5., 1.], c.shape, &m));
    }
    #[test]
    fn bounded_work_and_background_endpoint() {
        let mut c = cfg();
        let n = c.shape.iter().product();
        c.max_expansions = 1;
        assert!(trace(&vec![1.; n], &vec![1; n], &c).is_err());
        c.max_expansions = 10000;
        assert!(trace(&vec![1.; n], &vec![0; n], &c).is_err());
    }
}
