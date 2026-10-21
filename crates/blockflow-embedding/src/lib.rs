//! Numerical embeddings for Blockflow analysis artifacts.
//!
//! The crate owns preprocessing and algorithm provenance so applications do
//! not silently compute different coordinates from the same object table.

use hnsw_rs::prelude::{DistL2, Hnsw};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::panic::{catch_unwind, AssertUnwindSafe};
use umap_rs::{GraphParams, ManifoldParams, OptimizationParams, Umap, UmapConfig};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DenseFeatures {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}

impl DenseFeatures {
    pub fn new(rows: usize, columns: usize, values: Vec<f32>) -> Result<Self, String> {
        if rows < 2 || columns == 0 || values.len() != rows.saturating_mul(columns) {
            return Err("feature matrix shape is invalid".into());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("feature matrix contains a non-finite value".into());
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Preprocessing {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    pub constant_columns: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResult {
    pub rows: usize,
    pub dimensions: usize,
    pub coordinates: Vec<f32>,
    pub preprocessing: Preprocessing,
    pub explained_variance_ratio: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UmapOptions {
    pub dimensions: usize,
    pub n_neighbors: usize,
    pub min_dist: f32,
    pub spread: f32,
    pub epochs: Option<usize>,
    pub pca_dimensions: usize,
    pub initialization_seed: u64,
}

impl Default for UmapOptions {
    fn default() -> Self {
        Self {
            dimensions: 2,
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            epochs: None,
            pca_dimensions: 50,
            initialization_seed: 42,
        }
    }
}

/// Most components a PCA keeps: what a collection manifest may store.
pub const MAX_PCA_DIMENSIONS: usize = 64;

/// The first `dimensions` principal components (2 to [`MAX_PCA_DIMENSIONS`]):
/// all can be stored, and any two or three displayed.
pub fn pca(features: &DenseFeatures, dimensions: usize) -> Result<EmbeddingResult, String> {
    if !(2..=MAX_PCA_DIMENSIONS).contains(&dimensions)
        || dimensions > features.columns
        || dimensions >= features.rows
    {
        return Err(format!(
            "PCA dimensions must be 2 to {MAX_PCA_DIMENSIONS}, at most the feature count ({}) \
             and below the row count ({}); {dimensions} were asked for",
            features.columns, features.rows
        ));
    }
    let (standardized, preprocessing) = standardize(features);
    let decomposition =
        pca_standardized(&standardized, features.rows, features.columns, dimensions)?;
    Ok(EmbeddingResult {
        rows: features.rows,
        dimensions,
        coordinates: decomposition.coordinates,
        preprocessing,
        explained_variance_ratio: decomposition.explained_variance_ratio,
    })
}

pub fn umap(features: &DenseFeatures, options: &UmapOptions) -> Result<EmbeddingResult, String> {
    validate_dimensions(features, options.dimensions)?;
    if options.n_neighbors < 2 || options.n_neighbors >= features.rows {
        return Err("UMAP neighbours must be between 2 and row_count - 1".into());
    }
    if !options.min_dist.is_finite()
        || !options.spread.is_finite()
        || options.min_dist < 0.0
        || options.spread <= 0.0
        || options.min_dist > options.spread
    {
        return Err("UMAP min_dist and spread are invalid".into());
    }

    let (standardized, preprocessing) = standardize(features);
    let pca_dimensions = options
        .pca_dimensions
        .max(options.dimensions)
        .min(features.columns)
        .min(features.rows.saturating_sub(1));
    let pca_projection = pca_standardized(
        &standardized,
        features.rows,
        features.columns,
        pca_dimensions,
    )?;
    let working = pca_projection.coordinates;
    let (knn_indices, knn_distances) =
        approximate_neighbors(&working, features.rows, pca_dimensions, options.n_neighbors)?;
    let init = pca_initialization(
        &working,
        features.rows,
        pca_dimensions,
        options.dimensions,
        options.initialization_seed,
    );
    let data = Array2::from_shape_vec((features.rows, pca_dimensions), working)
        .map_err(|error| error.to_string())?;
    let indices = Array2::from_shape_vec((features.rows, options.n_neighbors), knn_indices)
        .map_err(|error| error.to_string())?;
    let distances = Array2::from_shape_vec((features.rows, options.n_neighbors), knn_distances)
        .map_err(|error| error.to_string())?;
    let init = Array2::from_shape_vec((features.rows, options.dimensions), init)
        .map_err(|error| error.to_string())?;
    let config = UmapConfig {
        n_components: options.dimensions,
        manifold: ManifoldParams {
            min_dist: options.min_dist,
            spread: options.spread,
            ..Default::default()
        },
        graph: GraphParams {
            n_neighbors: options.n_neighbors,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: options.epochs,
            ..Default::default()
        },
    };
    // umap-rs currently panics for malformed graph inputs. Everything is
    // validated above, but contain a dependency panic at the job boundary.
    let fitted = catch_unwind(AssertUnwindSafe(|| {
        Umap::new(config).fit(data.view(), indices.view(), distances.view(), init.view())
    }))
    .map_err(|_| "UMAP implementation rejected the validated input".to_owned())?;
    let embedding = fitted.into_embedding();
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err("UMAP produced a non-finite coordinate".into());
    }
    Ok(EmbeddingResult {
        rows: features.rows,
        dimensions: options.dimensions,
        coordinates: embedding.into_raw_vec_and_offset().0,
        preprocessing,
        explained_variance_ratio: pca_projection.explained_variance_ratio,
    })
}

fn validate_dimensions(features: &DenseFeatures, dimensions: usize) -> Result<(), String> {
    if !(2..=3).contains(&dimensions)
        || dimensions > features.columns
        || dimensions >= features.rows
    {
        return Err("UMAP dimensions must be 2 or 3 and fit the feature matrix".into());
    }
    Ok(())
}

fn standardize(features: &DenseFeatures) -> (Vec<f32>, Preprocessing) {
    let mut means = vec![0.0_f64; features.columns];
    let mut m2 = vec![0.0_f64; features.columns];
    for (row_index, row) in features.values.chunks_exact(features.columns).enumerate() {
        let count = (row_index + 1) as f64;
        for (column, &value) in row.iter().enumerate() {
            let value = f64::from(value);
            let delta = value - means[column];
            means[column] += delta / count;
            m2[column] += delta * (value - means[column]);
        }
    }
    let mut scales = m2
        .iter()
        .map(|sum| {
            (sum / (features.rows.saturating_sub(1) as f64))
                .max(0.0)
                .sqrt()
        })
        .collect::<Vec<_>>();
    let constant_columns = scales
        .iter()
        .enumerate()
        .filter_map(|(index, scale)| (*scale <= f64::EPSILON).then_some(index))
        .collect::<Vec<_>>();
    for scale in &mut scales {
        if *scale <= f64::EPSILON {
            *scale = 1.0;
        }
    }
    let standardized = features
        .values
        .chunks_exact(features.columns)
        .flat_map(|row| {
            row.iter().enumerate().map(|(column, &value)| {
                ((f64::from(value) - means[column]) / scales[column]) as f32
            })
        })
        .collect();
    (
        standardized,
        Preprocessing {
            means,
            scales,
            constant_columns,
        },
    )
}

struct PcaDecomposition {
    coordinates: Vec<f32>,
    explained_variance_ratio: Vec<f64>,
}

fn pca_standardized(
    values: &[f32],
    rows: usize,
    columns: usize,
    dimensions: usize,
) -> Result<PcaDecomposition, String> {
    if dimensions == 0 || dimensions > columns || dimensions >= rows {
        return Err("PCA component count does not fit the feature matrix".into());
    }
    let mut covariance = vec![0.0_f64; columns * columns];
    for row in values.chunks_exact(columns) {
        for left in 0..columns {
            for right in left..columns {
                covariance[left * columns + right] += f64::from(row[left]) * f64::from(row[right]);
            }
        }
    }
    let denominator = rows.saturating_sub(1) as f64;
    for left in 0..columns {
        for right in left..columns {
            let value = covariance[left * columns + right] / denominator;
            covariance[left * columns + right] = value;
            covariance[right * columns + left] = value;
        }
    }
    let (eigenvalues, eigenvectors) = jacobi_eigen(covariance, columns);
    let mut order = (0..columns).collect::<Vec<_>>();
    order.sort_by(|left, right| eigenvalues[*right].total_cmp(&eigenvalues[*left]));
    let total_variance = eigenvalues.iter().copied().map(|v| v.max(0.0)).sum::<f64>();
    let mut coordinates = vec![0.0_f32; rows * dimensions];
    let mut ratios = Vec::with_capacity(dimensions);
    for (component, &source) in order.iter().take(dimensions).enumerate() {
        let sign = (0..columns)
            .max_by(|left, right| {
                eigenvectors[*left * columns + source]
                    .abs()
                    .total_cmp(&eigenvectors[*right * columns + source].abs())
            })
            .map(|index| {
                if eigenvectors[index * columns + source] < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            })
            .unwrap_or(1.0);
        for row in 0..rows {
            let mut score = 0.0_f64;
            for column in 0..columns {
                score += f64::from(values[row * columns + column])
                    * eigenvectors[column * columns + source]
                    * sign;
            }
            coordinates[row * dimensions + component] = score as f32;
        }
        ratios.push(if total_variance > 0.0 {
            eigenvalues[source].max(0.0) / total_variance
        } else {
            0.0
        });
    }
    Ok(PcaDecomposition {
        coordinates,
        explained_variance_ratio: ratios,
    })
}

fn jacobi_eigen(mut matrix: Vec<f64>, size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut vectors = vec![0.0_f64; size * size];
    for index in 0..size {
        vectors[index * size + index] = 1.0;
    }
    let tolerance = 1e-12;
    for _ in 0..(size * size * 50).max(1) {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0_f64;
        for row in 0..size {
            for column in row + 1..size {
                let value = matrix[row * size + column].abs();
                if value > largest {
                    largest = value;
                    p = row;
                    q = column;
                }
            }
        }
        if largest <= tolerance {
            break;
        }
        let app = matrix[p * size + p];
        let aqq = matrix[q * size + q];
        let apq = matrix[p * size + q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (sine, cosine) = angle.sin_cos();
        for index in 0..size {
            if index != p && index != q {
                let aip = matrix[index * size + p];
                let aiq = matrix[index * size + q];
                let next_p = cosine * aip - sine * aiq;
                let next_q = sine * aip + cosine * aiq;
                matrix[index * size + p] = next_p;
                matrix[p * size + index] = next_p;
                matrix[index * size + q] = next_q;
                matrix[q * size + index] = next_q;
            }
        }
        matrix[p * size + p] =
            cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q * size + q] =
            sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p * size + q] = 0.0;
        matrix[q * size + p] = 0.0;
        for row in 0..size {
            let vip = vectors[row * size + p];
            let viq = vectors[row * size + q];
            vectors[row * size + p] = cosine * vip - sine * viq;
            vectors[row * size + q] = sine * vip + cosine * viq;
        }
    }
    let values = (0..size)
        .map(|index| matrix[index * size + index])
        .collect();
    (values, vectors)
}

fn approximate_neighbors(
    values: &[f32],
    rows: usize,
    columns: usize,
    neighbors: usize,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let vectors = values
        .chunks_exact(columns)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let max_connections = (neighbors * 2).clamp(16, 64);
    let layers = 16.min((rows as f64).ln().ceil() as usize).max(1);
    let construction_search = (neighbors * 8).max(100);
    let hnsw = Hnsw::<f32, DistL2>::new(
        max_connections,
        rows,
        layers,
        construction_search,
        DistL2 {},
    );
    let insert = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| (vector.as_slice(), index))
        .collect::<Vec<_>>();
    hnsw.parallel_insert_slice(&insert);
    let search = hnsw.parallel_search(&vectors, neighbors + 1, (neighbors * 4).max(64));
    let mut indices = Vec::with_capacity(rows * neighbors);
    let mut distances = Vec::with_capacity(rows * neighbors);
    for (row, found) in search.into_iter().enumerate() {
        for neighbor in found
            .into_iter()
            .filter(|neighbor| neighbor.d_id != row)
            .take(neighbors)
        {
            indices.push(u32::try_from(neighbor.d_id).map_err(|_| "too many UMAP rows")?);
            distances.push(neighbor.distance);
        }
        while indices.len() < (row + 1) * neighbors {
            indices.push(u32::MAX);
            distances.push(f32::INFINITY);
        }
    }
    Ok((indices, distances))
}

fn pca_initialization(
    values: &[f32],
    rows: usize,
    columns: usize,
    dimensions: usize,
    seed: u64,
) -> Vec<f32> {
    let mut output = vec![0.0; rows * dimensions];
    let mut state = seed.max(1);
    for component in 0..dimensions {
        let largest = (0..rows)
            .map(|row| values[row * columns + component].abs())
            .fold(0.0_f32, f32::max)
            .max(f32::EPSILON);
        for row in 0..rows {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 40) as f32 / (1_u32 << 24) as f32 - 0.5) * 1e-4;
            output[row * dimensions + component] =
                values[row * columns + component] * (10.0 / largest) + noise;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_finds_the_dominant_axis_and_is_deterministic() {
        let features = DenseFeatures::new(
            5,
            3,
            vec![
                -2.0, -4.0, 1.0, -1.0, -2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 4.0, 1.0,
            ],
        )
        .unwrap();
        let first = pca(&features, 2).unwrap();
        let second = pca(&features, 2).unwrap();
        assert_eq!(first, second);
        assert!(first.explained_variance_ratio[0] > 0.99);
        assert!(first.coordinates[0] < first.coordinates[8]);
        assert_eq!(first.preprocessing.constant_columns, vec![2]);
    }

    #[test]
    fn rejects_non_finite_input_and_invalid_umap_neighbours() {
        assert!(DenseFeatures::new(2, 1, vec![0.0, f32::NAN]).is_err());
        let features = DenseFeatures::new(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let options = UmapOptions {
            n_neighbors: 3,
            ..Default::default()
        };
        assert!(umap(&features, &options).is_err());
    }

    #[test]
    fn umap_produces_finite_coordinates_with_hnsw_neighbours() {
        let mut values = Vec::new();
        for row in 0..24 {
            let cluster = if row < 12 { 0.0 } else { 10.0 };
            values.extend([cluster + (row % 4) as f32, cluster + (row / 4) as f32]);
        }
        let features = DenseFeatures::new(24, 2, values).unwrap();
        let result = umap(
            &features,
            &UmapOptions {
                n_neighbors: 4,
                epochs: Some(5),
                pca_dimensions: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((result.rows, result.dimensions), (24, 2));
        assert_eq!(result.coordinates.len(), 48);
        assert!(result.coordinates.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn pca_keeps_ten_components_in_decreasing_variance() {
        // 40 rows of 12 features with decreasing spread per feature
        let mut seed = 7_u64;
        let mut values = Vec::new();
        for _ in 0..40 {
            for column in 0..12 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                values.push(unit * (12 - column) as f32);
            }
        }
        let features = DenseFeatures::new(40, 12, values).unwrap();
        let result = pca(&features, 10).unwrap();
        assert_eq!((result.rows, result.dimensions), (40, 10));
        assert_eq!(result.coordinates.len(), 400);
        assert!(result.coordinates.iter().all(|value| value.is_finite()));
        let ratios = &result.explained_variance_ratio;
        assert_eq!(ratios.len(), 10);
        assert!(ratios.windows(2).all(|pair| pair[0] >= pair[1]));
        // more components than features, or than rows allow, are refused
        assert!(pca(&features, 13).is_err());
        assert!(pca(&DenseFeatures::new(5, 12, vec![0.0; 60]).unwrap(), 5).is_err());
    }
}
