//! Required points with latent branch assignment. Bounded image-supported
//! candidate routing, fixed original endpoints/junctions, and no added spurs.
use super::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};
type P = [f64; 3];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RequiredPoint {
    /// Source-grid voxel-centre coordinates; no branch identifier is needed.
    pub position: P,
    /// Physical distance, in the same units as TraceConfig::spacing. Zero is exact.
    pub radius: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PointMatch {
    pub required_index: usize,
    pub branch_index: usize,
    pub position: P,
    pub distance: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstrainedTraceResult {
    pub paths: Vec<Vec<P>>,
    pub matches: Vec<PointMatch>,
    pub changed_branches: Vec<usize>,
    pub expanded: usize,
    /// A valid result can be returned when the bounded search found a solution
    /// but could not enumerate every alternative. No global optimality is claimed.
    pub search_truncated: bool,
}
fn distance(a: P, b: P, spacing: P) -> f64 {
    (0..3)
        .map(|k| ((a[k] - b[k]) * spacing[k]).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn nearest(path: &[P], p: P, spacing: P) -> (P, f64) {
    let mut best = (path[0], f64::INFINITY);
    for w in path.windows(2) {
        let d: P = std::array::from_fn(|k| (w[1][k] - w[0][k]) * spacing[k]);
        let q: P = std::array::from_fn(|k| (p[k] - w[0][k]) * spacing[k]);
        let den = d.iter().map(|v| v * v).sum::<f64>();
        let t = if den > 0. {
            (0..3).map(|k| q[k] * d[k]).sum::<f64>() / den
        } else {
            0.
        }
        .clamp(0., 1.);
        let q = std::array::from_fn(|k| w[0][k] + t * (w[1][k] - w[0][k]));
        let dd = distance(p, q, spacing);
        if dd < best.1 {
            best = (q, dd);
        }
    }
    best
}
/// Broad-phase search only. Final branch assignment uses supported route cost,
/// rather than binding a point to the geometrically nearest branch.
pub fn candidate_branches(
    paths: &[Vec<P>],
    points: &[RequiredPoint],
    spacing: P,
    search_radius: f64,
) -> Result<Vec<usize>, String> {
    if !search_radius.is_finite()
        || search_radius < 0.
        || spacing.iter().any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("invalid branch-search calibration".into());
    }
    if points.iter().any(|p| {
        p.position.iter().any(|v| !v.is_finite()) || !p.radius.is_finite() || p.radius < 0.
    }) {
        return Err("invalid required point".into());
    }
    let mut out = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        if path.len() < 2 || path.iter().flatten().any(|v| !v.is_finite()) {
            return Err("invalid existing branch".into());
        }
        if points
            .iter()
            .any(|p| nearest(path, p.position, spacing).1 <= search_radius + p.radius)
        {
            out.push(i);
        }
    }
    Ok(out)
}
fn cells(path: &[P], shape: [usize; 3]) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for w in path.windows(2) {
        let d: P = std::array::from_fn(|k| w[1][k] - w[0][k]);
        let (mut low, mut high) = (0f64, 1f64);
        for k in 0..3 {
            if d[k].abs() < 1e-12 {
                if w[0][k] < 0. || w[0][k] > (shape[k] - 1) as f64 {
                    high = -1.;
                }
            } else {
                let a = -w[0][k] / d[k];
                let b = ((shape[k] - 1) as f64 - w[0][k]) / d[k];
                low = low.max(a.min(b));
                high = high.min(a.max(b));
            }
        }
        if low > high {
            continue;
        }
        let a: P = std::array::from_fn(|k| w[0][k] + low * d[k]);
        let b: P = std::array::from_fn(|k| w[0][k] + high * d[k]);
        let mut ts = vec![0., 1.];
        for k in 0..3 {
            let dd = b[k] - a[k];
            if dd.abs() < 1e-12 {
                continue;
            }
            for v in (a[k].min(b[k]) - 0.5).floor() as i64..=(a[k].max(b[k]) + 0.5).ceil() as i64 {
                let t = (v as f64 + 0.5 - a[k]) / dd;
                if t > 0. && t < 1. {
                    ts.push(t);
                }
            }
        }
        ts.sort_by(f64::total_cmp);
        for t in ts.windows(2).map(|v| (v[0] + v[1]) * 0.5).chain([0., 1.]) {
            let p = std::array::from_fn(|k| {
                (a[k] + t * (b[k] - a[k])).clamp(0., (shape[k] - 1) as f64)
            });
            if let Some(i) = index(p, shape) {
                out.insert(i);
            }
        }
    }
    out
}
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.is_empty() {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    for k in 0..items.len() {
        let mut tail = items.to_vec();
        let head = tail.remove(k);
        for mut p in permutations(&tail) {
            p.insert(0, head);
            out.push(p);
        }
    }
    out
}
fn score(path: &[P], penalty: &[f64], c: &TraceConfig) -> f64 {
    path.windows(2)
        .map(|w| {
            distance(w[0], w[1], c.spacing)
                * 0.5
                * (penalty[index(w[0], c.shape).unwrap()] + penalty[index(w[1], c.shape).unwrap()])
        })
        .sum()
}
#[derive(Clone)]
struct Route {
    points: Vec<P>,
    anchors: Vec<P>,
    cells: BTreeSet<usize>,
    cost: f64,
}
struct Budget {
    left: usize,
    used: usize,
    states: usize,
    truncated: bool,
}
// A shortest supported link to any of the bounded physical-ball representatives.
// Previously used cells are forbidden by the caller, preventing an out-and-back spur.
fn link(
    start: P,
    goals: &[P],
    allowed: &[u8],
    penalty: &[f64],
    c: &TraceConfig,
    budget: &mut Budget,
) -> Result<Option<Vec<P>>, String> {
    let Some(si) = index(start, c.shape) else {
        return Ok(None);
    };
    if allowed[si] == 0 || !segment_ok(start, xyz(si, c.shape), c.shape, allowed) {
        return Ok(None);
    }
    let mut ends = BTreeMap::<usize, Vec<P>>::new();
    for &p in goals {
        if let Some(i) = index(p, c.shape) {
            if allowed[i] != 0 && segment_ok(xyz(i, c.shape), p, c.shape, allowed) {
                ends.entry(i).or_default().push(p);
            }
        }
    }
    if ends.is_empty() {
        return Ok(None);
    }
    let n = allowed.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut previous = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();
    dist[si] = distance(start, xyz(si, c.shape), c.spacing) * penalty[si];
    heap.push(Entry(dist[si], si));
    let mut best: Option<(f64, usize, P)> = None;
    while let Some(Entry(d, i)) = heap.pop() {
        if d != dist[i] {
            continue;
        }
        if best.as_ref().is_some_and(|b| d > b.0) {
            break;
        }
        if budget.left == 0 {
            budget.truncated = true;
            return Err(
                "required-point route budget exhausted; use a smaller ROI or fewer points".into(),
            );
        }
        budget.left -= 1;
        budget.used += 1;
        if let Some(points) = ends.get(&i) {
            for &p in points {
                let cost = d + distance(xyz(i, c.shape), p, c.spacing) * penalty[i];
                if best.as_ref().is_none_or(|b| cost < b.0) {
                    best = Some((cost, i, p));
                }
            }
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
                let dd = d + c.spacing[a] * 0.5 * (penalty[i] + penalty[j]);
                if dd < dist[j] {
                    dist[j] = dd;
                    previous[j] = i;
                    heap.push(Entry(dd, j));
                }
            }
        }
    }
    let Some((_, mut at, end)) = best else {
        return Ok(None);
    };
    let mut path = vec![end];
    loop {
        let p = xyz(at, c.shape);
        if path.last() != Some(&p) {
            path.push(p);
        }
        if at == si {
            break;
        }
        at = previous[at];
        if at == usize::MAX || path.len() > 32768 {
            return Ok(None);
        }
    }
    if path.last() != Some(&start) {
        path.push(start);
    }
    path.reverse();
    Ok(Some(path))
}
fn representatives(point: &RequiredPoint, allowed: &[u8], c: &TraceConfig) -> Vec<P> {
    let mut candidates = Vec::<(f64, P)>::new();
    let mut lo = [0; 3];
    let mut hi = [0; 3];
    for k in 0..3 {
        lo[k] = (point.position[k] - point.radius / c.spacing[k])
            .floor()
            .max(0.) as usize;
        hi[k] = (point.position[k] + point.radius / c.spacing[k])
            .ceil()
            .min((c.shape[k] - 1) as f64) as usize;
    }
    for z in lo[2]..=hi[2] {
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                let p = [x as f64, y as f64, z as f64];
                let i = x + c.shape[0] * (y + c.shape[1] * z);
                let d = distance(p, point.position, c.spacing);
                if allowed[i] != 0 && d <= point.radius + 1e-10 {
                    candidates.push((d, p));
                }
            }
        }
    }
    if let Some(i) = index(point.position, c.shape) {
        if allowed[i] != 0 {
            candidates.push((0., point.position));
        }
    }
    candidates.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.map(f64::to_bits).cmp(&b.1.map(f64::to_bits)))
    });
    candidates.dedup_by(|a, b| a.1 == b.1);
    candidates.truncate(64);
    candidates.into_iter().map(|v| v.1).collect()
}
fn anchor(path: &mut Vec<P>, point: P) -> P {
    let mut best = (f64::INFINITY, 0, point);
    for (i, w) in path.windows(2).enumerate() {
        let (q, d) = nearest(w, point, [1.; 3]);
        if d < best.0 {
            best = (d, i, q);
        }
    }
    // The caller supplies an already projected physical-metric point. It lies
    // on the segment; this second lookup only identifies its insertion index.
    if path[best.1] == best.2 {
        return best.2;
    }
    if path[best.1 + 1] == best.2 {
        return best.2;
    }
    path.insert(best.1 + 1, best.2);
    best.2
}
fn route(
    branch: usize,
    subset: u8,
    paths: &[Vec<P>],
    fixed: &[Vec<P>],
    points: &[RequiredPoint],
    penalty: &[f64],
    allowed: &[u8],
    c: &TraceConfig,
    budget: &mut Budget,
) -> Result<Option<Route>, String> {
    let original = &paths[branch];
    let first = original[0];
    let last = *original.last().unwrap();
    let si = index(first, c.shape).unwrap();
    let ti = index(last, c.shape).unwrap();
    let mut domain = allowed.to_vec();
    for (other, path) in paths.iter().enumerate() {
        if other != branch {
            for i in cells(path, c.shape) {
                domain[i] = 0;
            }
        }
    }
    for path in fixed {
        for i in cells(path, c.shape) {
            domain[i] = 0;
        }
    }
    // Shared fixed endpoints remain usable; other branches are never moved.
    domain[si] = allowed[si];
    domain[ti] = allowed[ti];
    let ids: Vec<_> = (0..points.len())
        .filter(|&i| subset & (1 << i) != 0)
        .collect();
    let mut choices = Vec::new();
    if original
        .windows(2)
        .all(|w| segment_ok(w[0], w[1], c.shape, &domain))
        && ids
            .iter()
            .all(|&i| nearest(original, points[i].position, c.spacing).1 <= points[i].radius + 1e-9)
    {
        choices.push(original.clone());
    }
    for order in permutations(&ids) {
        for reverse in [false, true] {
            if budget.left == 0 {
                budget.truncated = true;
                break;
            }
            let (start, end) = if reverse {
                (last, first)
            } else {
                (first, last)
            };
            let mut path = vec![start];
            let mut occupied = HashSet::new();
            let mut success = true;
            for step in 0..=order.len() {
                if step < order.len()
                    && path.len() > 1
                    && nearest(&path, points[order[step]].position, c.spacing).1
                        <= points[order[step]].radius + 1e-9
                {
                    continue;
                }
                let mut mask = domain.clone();
                for &i in &occupied {
                    mask[i] = 0;
                }
                let current = *path.last().unwrap();
                let ci = index(current, c.shape).unwrap();
                mask[ci] = domain[ci];
                let endi = index(end, c.shape).unwrap();
                if step == order.len() {
                    mask[endi] = domain[endi];
                } else if endi != ci {
                    mask[endi] = 0;
                }
                let goals = if step == order.len() {
                    vec![end]
                } else {
                    representatives(&points[order[step]], &mask, c)
                };
                let next = match link(current, &goals, &mask, penalty, c, budget) {
                    Ok(Some(p)) => p,
                    Ok(None) => {
                        success = false;
                        break;
                    }
                    Err(_) => {
                        success = false;
                        break;
                    }
                };
                occupied.extend(cells(&next, c.shape));
                path.extend(next.into_iter().skip(1));
                if path.len() > 32768 {
                    success = false;
                    break;
                }
            }
            if success {
                if reverse {
                    path.reverse();
                }
                if ids.iter().all(|&i| {
                    nearest(&path, points[i].position, c.spacing).1 <= points[i].radius + 1e-9
                }) {
                    choices.push(path);
                }
            }
        }
    }
    let mut best = None;
    for mut path in choices {
        let mut anchors = Vec::new();
        for &i in &ids {
            let (p, d) = nearest(&path, points[i].position, c.spacing);
            if d > points[i].radius + 1e-9 {
                continue;
            }
            anchors.push(anchor(&mut path, p));
        }
        let cost = score(&path, penalty, c);
        if best.as_ref().is_none_or(|r: &Route| cost < r.cost) {
            best = Some(Route {
                cells: cells(&path, c.shape),
                points: path,
                anchors,
                cost,
            });
        }
    }
    Ok(best)
}
fn compatible(a: &Route, b: &Route, shape: [usize; 3]) -> bool {
    let shared: BTreeSet<_> = [a.points[0], *a.points.last().unwrap()]
        .into_iter()
        .filter(|p| *p == b.points[0] || *p == *b.points.last().unwrap())
        .filter_map(|p| index(p, shape))
        .collect();
    a.cells.intersection(&b.cells).all(|i| shared.contains(i))
}

/// Fit only branches assigned to required points. All other branch polylines
/// are returned exactly unchanged. Input endpoints/junctions are fixed.
/// Candidate orders and assignments are bounded; failure means no supported
/// construction was found, not a proof of anatomical impossibility.
pub fn fit_paths_with_required_points(
    evidence: &[f32],
    allowed: &[u8],
    c: &TraceConfig,
    paths: &[Vec<P>],
    points: &[RequiredPoint],
    fixed_paths: &[Vec<P>],
) -> Result<ConstrainedTraceResult, String> {
    let n = c
        .shape
        .iter()
        .try_fold(1usize, |v, &x| v.checked_mul(x))
        .ok_or("ROI overflow")?;
    if c.shape.contains(&0)
        || n == 0
        || n > 4_000_000
        || n > c.max_voxels
        || n != allowed.len()
        || n != evidence.len()
        || c.spacing.iter().any(|v| !v.is_finite() || *v <= 0.)
        || evidence
            .iter()
            .any(|v| !v.is_finite() || *v < 0. || *v > 1.)
        || !c.image_weight.is_finite()
        || c.image_weight < 0.
        || !c.clearance_weight.is_finite()
        || c.clearance_weight < 0.
        || !c.max_move.is_finite()
        || c.max_move < 0.
        || c.refinement_iterations > 200
        || c.max_expansions == 0
        || c.max_expansions > 8_000_000
    {
        return Err("invalid bounded tracing fields/calibration".into());
    }
    if paths.is_empty()
        || paths.len() > 64
        || points.len() > 4
        || paths.iter().any(|p| {
            p.len() < 2 || p.len() > 32768 || p.iter().any(|&p| index(p, c.shape).is_none())
        })
        || paths.iter().map(Vec::len).sum::<usize>() > 200_000
        || fixed_paths
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
    {
        return Err("required-point fit supports at most4 points and64 bounded branches".into());
    }
    let pitch = c.spacing.iter().copied().fold(f64::INFINITY, f64::min);
    if points.iter().any(|p| {
        index(p.position, c.shape).is_none()
            || !p.radius.is_finite()
            || p.radius < 0.
            || p.radius > 128. * pitch
    }) {
        return Err("invalid required point or physical tolerance".into());
    }
    if points.is_empty() {
        return Ok(ConstrainedTraceResult {
            paths: paths.to_vec(),
            matches: vec![],
            changed_branches: vec![],
            expanded: 0,
            search_truncated: false,
        });
    }
    let clearance = edt(allowed, c.shape, c.spacing);
    let penalty: Vec<_> = (0..n)
        .map(|i| {
            1. + c.image_weight * (1. - evidence[i] as f64)
                + c.clearance_weight * pitch / (pitch + clearance[i])
        })
        .collect();
    let mut budget = Budget {
        left: c.max_expansions,
        used: 0,
        states: 0,
        truncated: false,
    };
    let mut cache = BTreeMap::<(usize, u8), Option<Route>>::new();
    let mut best: Option<(f64, Vec<(usize, u8)>)> = None;
    struct Search<'a> {
        evidence: &'a [u8],
        cfg: &'a TraceConfig,
        paths: &'a [Vec<P>],
        fixed: &'a [Vec<P>],
        points: &'a [RequiredPoint],
        penalty: &'a [f64],
    }
    fn visit(
        s: &Search<'_>,
        remaining: u8,
        used: &mut BTreeSet<usize>,
        chosen: &mut Vec<(usize, u8)>,
        cost: f64,
        cache: &mut BTreeMap<(usize, u8), Option<Route>>,
        budget: &mut Budget,
        best: &mut Option<(f64, Vec<(usize, u8)>)>,
    ) -> Result<(), String> {
        if remaining == 0 {
            if best.as_ref().is_none_or(|b| cost < b.0) {
                *best = Some((cost, chosen.clone()));
            }
            return Ok(());
        }
        if budget.left == 0 || budget.states >= 16384 {
            budget.truncated = true;
            return Ok(());
        }
        budget.states += 1;
        let first = 1u8 << remaining.trailing_zeros();
        let mut subset = remaining;
        while subset > 0 {
            if subset & first != 0 {
                for branch in 0..s.paths.len() {
                    if used.contains(&branch) {
                        continue;
                    }
                    let key = (branch, subset);
                    if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(key) {
                        let r = route(
                            branch, subset, s.paths, s.fixed, s.points, s.penalty, s.evidence,
                            s.cfg, budget,
                        )?;
                        entry.insert(r);
                    }
                    let Some(r) = cache.get(&key).unwrap() else {
                        continue;
                    };
                    if !chosen
                        .iter()
                        .all(|k| compatible(r, cache[k].as_ref().unwrap(), s.cfg.shape))
                    {
                        continue;
                    }
                    let extra = r.cost - score(&s.paths[branch], s.penalty, s.cfg);
                    used.insert(branch);
                    chosen.push(key);
                    visit(
                        s,
                        remaining & !subset,
                        used,
                        chosen,
                        cost + extra,
                        cache,
                        budget,
                        best,
                    )?;
                    chosen.pop();
                    used.remove(&branch);
                }
            }
            subset = (subset - 1) & remaining;
        }
        Ok(())
    }
    let search = Search {
        evidence: allowed,
        cfg: c,
        paths,
        fixed: fixed_paths,
        points,
        penalty: &penalty,
    };
    visit(
        &search,
        (1 << points.len()) - 1,
        &mut BTreeSet::new(),
        &mut Vec::new(),
        0.,
        &mut cache,
        &mut budget,
        &mut best,
    )?;
    let Some((_, chosen)) = best else {
        return Err(if budget.truncated{"Required-point search budget exhausted; reduce ROI/search radius or number of points"}else{"No supported interior route found for the required points; check points, gate or endpoints (a dangling spur is not added)"}.into());
    };
    let mut output = paths.to_vec();
    for &(branch, subset) in &chosen {
        let r = cache[&(branch, subset)].as_ref().unwrap();
        let mut domain = vec![0u8; n];
        for &i in &r.cells {
            domain[i] = allowed[i];
        }
        let mut path = r.points.clone();
        refine_path(&mut path, &|i| penalty[i], &domain, c, &r.anchors);
        if !path
            .windows(2)
            .all(|w| segment_ok(w[0], w[1], c.shape, allowed))
        {
            return Err("Required-point refinement left the allowed region".into());
        }
        output[branch] = path;
    }
    let mut matches = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let (b, (q, d)) = output
            .iter()
            .enumerate()
            .map(|(b, path)| (b, nearest(path, p.position, c.spacing)))
            .min_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
            .unwrap();
        if d > p.radius + pitch * 1e-7 {
            return Err("Required point was not satisfied after refinement".into());
        }
        matches.push(PointMatch {
            required_index: i,
            branch_index: b,
            position: q,
            distance: d,
        });
    }
    let changed_branches = (0..paths.len())
        .filter(|&i| paths[i] != output[i])
        .collect();
    Ok(ConstrainedTraceResult {
        paths: output,
        matches,
        changed_branches,
        expanded: budget.used,
        search_truncated: budget.truncated,
    })
}
pub fn trace_with_required_points(
    evidence: &[f32],
    allowed: &[u8],
    c: &TraceConfig,
    points: &[RequiredPoint],
) -> Result<ConstrainedTraceResult, String> {
    let base = trace(evidence, allowed, c)?;
    if points.is_empty() {
        return Ok(ConstrainedTraceResult {
            paths: base.paths,
            matches: vec![],
            changed_branches: vec![],
            expanded: base.expanded,
            search_truncated: false,
        });
    }
    let mut fit = c.clone();
    fit.max_expansions = c.max_expansions.saturating_sub(base.expanded);
    if fit.max_expansions == 0 {
        return Err("Required-point route budget exhausted".into());
    }
    let mut r = fit_paths_with_required_points(evidence, allowed, &fit, &base.paths, points, &[])?;
    r.expanded += base.expanded;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(shape: [usize; 3]) -> TraceConfig {
        TraceConfig {
            shape,
            spacing: [1.; 3],
            source_group: vec![[1., 3., 0.]],
            target_groups: vec![vec![[13., 3., 0.]]],
            max_voxels: 10000,
            max_expansions: 100000,
            image_weight: 4.,
            clearance_weight: 2.,
            refinement_iterations: 20,
            max_move: 0.75,
        }
    }
    fn set(m: &mut [u8], shape: [usize; 3], x: usize, y: usize) {
        m[x + shape[0] * y] = 1;
    }
    #[test]
    fn required_detour_is_interior_without_a_new_spur() {
        let c = config([15, 9, 1]);
        let mut m = vec![0; 135];
        for x in 1..=13 {
            set(&mut m, c.shape, x, 3);
            set(&mut m, c.shape, x, 7);
        }
        for y in 3..=7 {
            set(&mut m, c.shape, 1, y);
            set(&mut m, c.shape, 13, y);
        }
        let p = RequiredPoint {
            position: [7.2, 7., 0.],
            radius: 0.,
        };
        let r = trace_with_required_points(&vec![1.; 135], &m, &c, &[p]).unwrap();
        assert_eq!(r.paths.len(), 1);
        assert!(r.matches[0].distance < 1e-9);
        assert_eq!(r.paths[0].first(), Some(&[1., 3., 0.]));
        assert_eq!(r.paths[0].last(), Some(&[13., 3., 0.]));
        assert!(r.paths[0].contains(&[7.2, 7., 0.]));
    }
    #[test]
    fn dead_end_hint_is_not_satisfied_by_adding_a_leaf() {
        let c = config([15, 9, 1]);
        let mut m = vec![0; 135];
        for x in 1..=13 {
            set(&mut m, c.shape, x, 3);
        }
        for y in 3..=7 {
            set(&mut m, c.shape, 7, y);
        }
        assert!(trace_with_required_points(
            &vec![1.; 135],
            &m,
            &c,
            &[RequiredPoint {
                position: [7., 7., 0.],
                radius: 0.
            }]
        )
        .is_err());
    }
    #[test]
    fn y_tree_assigns_point_without_branch_id() {
        let mut c = config([15, 12, 1]);
        c.source_group = vec![[7., 1., 0.]];
        c.target_groups = vec![vec![[1., 10., 0.]], vec![[13., 10., 0.]]];
        let mut m = vec![0; 180];
        for y in 1..=5 {
            set(&mut m, c.shape, 7, y);
        }
        for x in 1..=13 {
            set(&mut m, c.shape, x, 5);
        }
        for y in 5..=10 {
            set(&mut m, c.shape, 1, y);
            set(&mut m, c.shape, 13, y);
        }
        let r = trace_with_required_points(
            &vec![1.; 180],
            &m,
            &c,
            &[RequiredPoint {
                position: [1., 8., 0.],
                radius: 0.,
            }],
        )
        .unwrap();
        assert_eq!(r.paths.len(), 3);
        assert!(r.matches[0].distance < 1e-9);
        assert!(r.paths[r.matches[0].branch_index]
            .iter()
            .any(|p| p[0] == 1.));
    }
    #[test]
    fn unordered_points_and_unchanged_other_branch() {
        let c = config([15, 9, 1]);
        let m = vec![1; 135];
        let paths = vec![
            vec![[1., 3., 0.], [13., 3., 0.]],
            vec![[1., 0., 0.], [13., 0., 0.]],
        ];
        let r = fit_paths_with_required_points(
            &vec![1.; 135],
            &m,
            &c,
            &paths,
            &[
                RequiredPoint {
                    position: [10.2, 5., 0.],
                    radius: 0.,
                },
                RequiredPoint {
                    position: [4.2, 5., 0.],
                    radius: 0.,
                },
            ],
            &[],
        )
        .unwrap();
        assert!(r.matches.iter().all(|p| p.distance < 1e-8));
        assert_eq!(r.paths[1], paths[1]);
        assert_eq!(r.paths.len(), 2);
    }
    #[test]
    fn tolerance_is_physical_and_failure_is_explicit() {
        let mut c = config([15, 9, 2]);
        c.spacing = [1., 1., 4.];
        let mut m = vec![0; 270];
        for x in 1..=13 {
            m[x + 15 * 3] = 1;
        }
        assert!(trace_with_required_points(
            &vec![1.; 270],
            &m,
            &c,
            &[RequiredPoint {
                position: [7., 3., 1.],
                radius: 1.
            }]
        )
        .is_err());
        let r = trace_with_required_points(
            &vec![1.; 270],
            &m,
            &c,
            &[RequiredPoint {
                position: [7., 4., 0.],
                radius: 1.,
            }],
        )
        .unwrap();
        assert!(r.matches[0].distance <= 1. + 1e-9);
    }
    #[test]
    fn empty_constraints_preserve_legacy_output_exactly() {
        let c = config([15, 9, 1]);
        let allowed = vec![1; 135];
        let evidence = vec![1.; 135];
        let old = trace(&evidence, &allowed, &c).unwrap();
        let new = trace_with_required_points(&evidence, &allowed, &c, &[]).unwrap();
        assert_eq!(old.paths, new.paths);
        assert_eq!(old.expanded, new.expanded);
    }
    #[test]
    fn assignment_uses_supported_course_not_nearest_branch() {
        let c = config([15, 9, 1]);
        let mut allowed = vec![0; 135];
        // Nearer branch has no connecting route to the required point.
        for x in 1..=13 {
            set(&mut allowed, c.shape, x, 2);
            set(&mut allowed, c.shape, x, 8);
        }
        for y in 4..=8 {
            set(&mut allowed, c.shape, 1, y);
            set(&mut allowed, c.shape, 13, y);
        }
        for x in 1..=13 {
            set(&mut allowed, c.shape, x, 4);
        }
        let paths = vec![
            vec![[1., 2., 0.], [13., 2., 0.]],
            vec![[1., 8., 0.], [13., 8., 0.]],
        ];
        let pin = RequiredPoint {
            position: [7., 4., 0.],
            radius: 0.,
        };
        let r = fit_paths_with_required_points(&vec![1.; 135], &allowed, &c, &paths, &[pin], &[])
            .unwrap();
        assert_eq!(r.matches[0].branch_index, 1);
        assert_eq!(r.paths[0], paths[0]);
        assert_eq!(r.paths.len(), 2);
    }
}
