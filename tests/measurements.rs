use std::sync::Arc;

use blockflow::assemble::ImageId;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::{fold_fragments, fragment_phase, FragmentOp, PhaseWork};
use blockflow::ops::{
    assert_boundary_measurement_decomposition_invariant, assert_boundary_measurement_invariant,
    assert_custom_boundary_builder_invariant, assert_custom_region_builder_invariant,
    assert_object_measurement_invariant, assert_region_measurement_decomposition_invariant,
    assert_region_measurement_invariant, auto_distribution_measurement_schema,
    auto_distribution_measurement_schema_set, auto_distribution_measurements,
    boundary_distance_relationship_measurement_schema, centroid_neighbor_measurement_schema,
    centroid_relationship_measurement_schema, collect_auto_distribution_measurements,
    collect_auto_distribution_rows_with_set, collect_boundary_distance_relationship_measurements,
    collect_boundary_distance_relationship_rows_with_contract, collect_boundary_rows,
    collect_centroid_neighbor_measurements, collect_centroid_neighbor_rows_with_contract,
    collect_centroid_relationship_measurements, collect_centroid_relationship_rows_with_contract,
    collect_class_a_shapes, collect_class_a_values, collect_colocalization_measurements,
    collect_colocalization_rows_with_contract, collect_component_measurements,
    collect_component_rows, collect_contact_rows, collect_costes_colocalization_measurements,
    collect_costes_colocalization_rows_with_contract, collect_custom_boundary_rows,
    collect_custom_object_rows, collect_custom_region_rows, collect_distribution_measurements,
    collect_enclosing_sphere_measurements, collect_enclosing_sphere_rows_with_contract,
    collect_exact_distribution_measurements, collect_exact_distribution_measurements_set,
    collect_exact_distribution_rows_with_set, collect_exact_label_radius_measurements,
    collect_exact_label_radius_rows, collect_expansion_relationship_measurements,
    collect_expansion_relationship_rows_with_contract, collect_glcm_texture_measurements,
    collect_glcm_texture_rows_with_contract, collect_granularity_measurements,
    collect_granularity_rows_with_set, collect_object_3d_moment_measurements,
    collect_object_3d_moment_measurements_set, collect_object_3d_moment_rows_with_set,
    collect_object_convex_hull_measurements, collect_object_convex_hull_rows_with_contract,
    collect_object_geometry_measurements, collect_object_geometry_rows_with_contract,
    collect_object_hu_moment_measurements, collect_object_hu_moment_rows_with_contract,
    collect_object_projected_convex_measurements, collect_object_projected_convex_measurements_set,
    collect_object_projected_convex_rows_with_contract,
    collect_object_voxel_face_convex_hull_measurements,
    collect_object_voxel_face_convex_hull_rows_with_contract,
    collect_object_weighted_hu_moment_measurements,
    collect_object_weighted_hu_moment_rows_with_contract, collect_object_zernike3d_measurements,
    collect_object_zernike3d_measurements_set, collect_object_zernike3d_rows_with_contract,
    collect_object_zernike_moment_measurements, collect_object_zernike_moment_measurements_set,
    collect_object_zernike_moment_rows_with_contract,
    collect_rank_weighted_colocalization_measurements,
    collect_rank_weighted_colocalization_rows_with_contract, collect_shapes,
    collect_shared_boundary_radius_measurements, collect_shared_boundary_radius_rows,
    collect_topology_measurements, collect_topology_rows, collect_touching_neighbor_measurements,
    collect_touching_neighbor_rows_with_contract, colocalization_measurement_schema,
    colocalization_measurements, component_measurement_schema, contact_fraction_of_boundary,
    costes_colocalization_measurement_schema, costes_colocalization_measurements,
    decode_custom_measurement_rows, enclosing_sphere_measurement_schema,
    encode_auto_distribution_measurements, encode_auto_distribution_measurements_set,
    encode_boundary_distance_relationship_measurements, encode_centroid_neighbor_measurements,
    encode_centroid_relationship_measurements, encode_colocalization_measurements,
    encode_component_measurements, encode_costes_colocalization_measurements,
    encode_enclosing_sphere_measurements, encode_exact_distribution_measurements,
    encode_exact_distribution_measurements_set, encode_exact_label_radius_measurements,
    encode_expansion_relationship_measurements, encode_granularity_measurements,
    encode_object_3d_moment_measurements, encode_object_3d_moment_measurements_set,
    encode_object_convex_hull_measurements, encode_object_geometry_measurements,
    encode_object_hu_moment_measurements, encode_object_projected_convex_measurements,
    encode_object_projected_convex_measurements_set,
    encode_object_voxel_face_convex_hull_measurements,
    encode_object_weighted_hu_moment_measurements, encode_object_zernike3d_measurements_set,
    encode_object_zernike_moment_measurements, encode_object_zernike_moment_measurements_set,
    encode_rank_weighted_colocalization_measurements, encode_shared_boundary_radius_measurements,
    encode_topology_measurements, encode_touching_neighbor_measurements,
    equivalent_sphere_diameter, equivalent_sphere_diameter_for_voxels, equivalent_sphere_radius,
    equivalent_sphere_radius_for_voxels, exact_distribution_measurement_schema,
    exact_distribution_measurement_schema_set, exact_distribution_measurements,
    exact_label_radius_measurements, expansion_relationship_measurement_schema,
    expansion_until_adjacent_relationships_from_boundary_distances, fuse_glcm_texture_measurements,
    glcm_texture_measurements, granularity_measurement_schema, granularity_spectrum_measurements,
    object_3d_moment_measurement_schema, object_3d_moment_measurement_schema_set,
    object_3d_moment_measurements, object_3d_moment_measurements_set,
    object_boundary_distance_relationships, object_centroid_relationships,
    object_centroid_relationships_from_shapes, object_component_measurements,
    object_convex_hull_measurement_schema, object_convex_hull_measurements,
    object_enclosing_sphere_measurements, object_expansion_until_adjacent_relationships,
    object_geometry_measurement_schema, object_geometry_measurements,
    object_hu_moment_measurement_schema, object_hu_moments_measurements,
    object_projected_convex_measurement_schema, object_projected_convex_measurement_schema_set,
    object_projected_convex_measurements, object_projected_convex_measurements_set,
    object_topology_measurements, object_topology_measurements_with,
    object_voxel_face_convex_hull_measurement_schema, object_voxel_face_convex_hull_measurements,
    object_weighted_hu_moment_measurement_schema, object_weighted_hu_moments_measurements,
    object_zernike3d_measurement_schema_set, object_zernike3d_measurements,
    object_zernike3d_measurements_set, object_zernike_moment_measurement_schema,
    object_zernike_moment_measurement_schema_set, object_zernike_moments_measurements,
    object_zernike_moments_measurements_set, orientation_yx, radius_measurement_schema,
    rank_weighted_colocalization_measurement_schema, rank_weighted_colocalization_measurements,
    run_boundary_measure, run_object_measure, run_region_measure, shape_boundary_measurements,
    shared_boundary_distance_field, shared_boundary_radius_measurements,
    summarize_centroid_neighbors, summarize_centroid_neighbors_within,
    summarize_touching_neighbors, topology_measurement_schema,
    touching_neighbor_measurement_schema, ApproxDistributionSet, ApproxMode, ApproxTolerance,
    AutoDistributionOp, AutoDistributionSet, BoundaryFeature, BoundaryMeasure, BoundaryMeasureOp,
    BoundaryMeasurements, ColocalizationContract, ColocalizationFeature,
    ColocalizationMeasurements, ColocalizationPairsOp, ColocalizationSumsOp, Connectivity,
    ContactFeature, ContactMeasurements, CostesColocalizationFeature,
    CostesColocalizationMeasurements, DistanceParams, DistributionFeature,
    DistributionMeasurements, DistributionPercentile, DistributionSet, ElementShape,
    EnclosingSphereOp, ExactDistributionMeasurements, ExactDistributionOp, ExactDistributionSet,
    ExactDistributionTallyOp, ExactLabelRadiusMeasurements, ExactLabelRadiusOp, FeatureScalar,
    FixedPoint, FoldLaw, FusedGlcmTextureMeasurements, GlcmOffset, GlcmQuantization,
    GlcmTextureContract, GlcmTextureFeature, GlcmTextureMeasurements, GlcmTextureOp,
    GranularityFeature, GranularityOp, GranularityRadius, GranularitySet, HuMomentIndex,
    IntensityFeature, IntensityImage, IntensityMeasurements, IntensitySet, LabelImage,
    MeasureSource, MeasureValues, MeasurementFrame, MeasurementFrameId, MeasurementKey,
    MeasurementSourceFact, MeasurementSourceFacts, Measurements, MergeColocalizationSumsOp,
    MergeCostesColocalizationOp, MergeExactDistributionOp, MergeGlcmTextureOp,
    MergeObjectMoment3dOp, MergeObjectProjectedConvexOp, MergeObjectZernikeMomentsOp,
    MergeRankWeightedColocalizationOp, Moment3d, Moment3dKey, MultiGlcmTextureOp, NeighborSummary,
    ObjectBoundaryDistanceFeature, ObjectBoundaryDistanceMeasurements, ObjectComponentFeature,
    ObjectComponentMeasurements, ObjectComponentOp, ObjectConvexHullFeature,
    ObjectConvexHullMeasurements, ObjectConvexHullOp, ObjectConvexHullView,
    ObjectEnclosingSphereFeature, ObjectEnclosingSphereMeasurements, ObjectExpansionFeature,
    ObjectExpansionMeasurements, ObjectGeometryFeature, ObjectGeometryMeasurements,
    ObjectGeometryOp, ObjectHuMomentFeature, ObjectHuMomentsMeasurements, ObjectHuMomentsOp,
    ObjectInputs, ObjectMeasure, ObjectMeasureMergeOp, ObjectMoment3dFeature,
    ObjectMoment3dMeasurements, ObjectMoment3dOp, ObjectMoment3dSet, ObjectNeighborFeature,
    ObjectNeighborMeasurements, ObjectProjectedConvexContract, ObjectProjectedConvexFeature,
    ObjectProjectedConvexMeasurements, ObjectProjectedConvexOp, ObjectProjectionContract,
    ObjectRelationshipFeature, ObjectRelationshipMeasurements, ObjectTopologyConvention,
    ObjectTopologyFeature, ObjectTopologyMeasurements, ObjectTopologyOp, ObjectView,
    ObjectVoxelFaceConvexHullFeature, ObjectVoxelFaceConvexHullMeasurements,
    ObjectWeightedHuMomentFeature, ObjectWeightedHuMomentsMeasurements, ObjectWeightedHuMomentsOp,
    ObjectZernike3dContract, ObjectZernike3dDescriptor, ObjectZernike3dFeature,
    ObjectZernike3dMeasurements, ObjectZernike3dSet, ObjectZernikeMoment,
    ObjectZernikeMomentContract, ObjectZernikeMomentFeature, ObjectZernikeMomentSet,
    ObjectZernikeMomentsMeasurements, ObjectZernikeMomentsOp, PhysicalSpacing, ProjectedConvexSet,
    ProjectionAxis, RadiusFeature, RankWeightedColocalizationFeature,
    RankWeightedColocalizationMeasurements, RegionMeasure, RegionMeasureOp, RegionShape,
    ShapeBoundaryFeature, ShapeFeature, ShapeMeasurements, ShapeSet,
    SharedBoundaryRadiusMeasurements, SharedBoundaryRadiusOp, TouchingNeighborFeature, VoxelCount,
    WithinDistanceThreshold, Zernike3dKey, ZernikeMomentKey,
};
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::simulate::{Machine, PlanOrder, Rates, Run};
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::{encoded_schema, Column, RowBuilder, Schema, Table, Value};
use blockflow::{
    Anchor, BlockGrid, BlockOp, Chain, Decomposition, Dtype, PhaseDecomposition, Reach, Result,
    Voxels,
};
use ndarray::Array3;

const VOLUME: [usize; 3] = [6, 4, 3];

struct CoordinateValues;

impl BlockOp for CoordinateValues {
    fn name(&self) -> &'static str {
        "values"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(&self, _input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let mut out = out.view_mut::<f64>()?;
        for (index, slot) in out.indexed_iter_mut() {
            let z = at.offset[0] + index.0;
            let y = at.offset[1] + index.1;
            let x = at.offset[2] + index.2;
            *slot = z as f64 * 10.0 + y as f64 + x as f64 / 10.0;
        }
        Ok(())
    }
}

struct ObjectPointCountMeasure;

impl ObjectMeasure for ObjectPointCountMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.point_count").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label"), Column::u64("points")]).unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::coordinates()
    }

    fn cost_per_object(&self) -> f64 {
        11.0
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        out.push(
            object.bbox_min(),
            &[
                Value::U64(object.label()),
                Value::U64(object.points().len() as u64),
            ],
        )
    }
}

struct ObjectBoundaryPointCountMeasure;

impl ObjectMeasure for ObjectBoundaryPointCountMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_point_count").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label"), Column::u64("boundary_points")]).unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::boundary_points()
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        let boundary_points = object.boundary_points().unwrap();
        if object.label() == 9 {
            assert_eq!(object.points().len(), 27);
            assert_eq!(boundary_points.len(), 26);
        }
        out.push(
            object.bbox_min(),
            &[
                Value::U64(object.label()),
                Value::U64(boundary_points.len() as u64),
            ],
        )
    }
}

struct ObjectProjectedHullMeasure;

impl ObjectMeasure for ObjectProjectedHullMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.projected_hull").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![
            Column::u64("label"),
            Column::u64("hull_vertices"),
            Column::f64("max_hull_distance"),
        ])
        .unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::projected_hull(ProjectionAxis::Z)
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        let hull = object.projected_hull_vertices().unwrap();
        let mut max_squared: f64 = 0.0;
        for (index, a) in hull.iter().enumerate() {
            for b in &hull[index + 1..] {
                max_squared =
                    max_squared.max((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
            }
        }
        if object.label() == 2 {
            assert_eq!(hull.len(), 5);
            assert!((max_squared.sqrt() - 8.0f64.sqrt()).abs() < 1.0e-12);
        }
        out.push(
            object.bbox_min(),
            &[
                Value::U64(object.label()),
                Value::U64(hull.len() as u64),
                Value::F64(max_squared.sqrt()),
            ],
        )
    }
}

struct ObjectConvexHullPrereqMeasure {
    spacing: PhysicalSpacing,
}

impl ObjectMeasure for ObjectConvexHullPrereqMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.convex_hull_prereq").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![
            Column::u64("label"),
            Column::u64("hull_vertices"),
            Column::u64("hull_faces"),
            Column::f64("surface_area"),
            Column::f64("volume"),
            Column::f64("max_feret_diameter"),
        ])
        .unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::convex_hull(self.spacing)
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        let hull: ObjectConvexHullView = object.convex_hull().unwrap();
        out.push(
            object.bbox_min(),
            &[
                Value::U64(object.label()),
                Value::U64(hull.vertices() as u64),
                Value::U64(hull.faces() as u64),
                Value::F64(hull.surface_area()),
                Value::F64(hull.volume()),
                Value::F64(hull.max_feret_diameter()),
            ],
        )
    }
}

struct EmptyObjectMeasure;

impl ObjectMeasure for EmptyObjectMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.empty").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::none()
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        out.push(object.bbox_min(), &[Value::U64(object.label())])
    }
}

struct WrongValueObjectMeasure;

impl ObjectMeasure for WrongValueObjectMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.wrong_value").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::coordinates()
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        out.push(object.bbox_min(), &[Value::F64(object.label() as f64)])
    }
}

struct BadSchemaObjectMeasure;

impl ObjectMeasure for BadSchemaObjectMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.object_bad_schema").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("object_id"), Column::u64("points")]).unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::coordinates()
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        out.push(
            object.bbox_min(),
            &[
                Value::U64(object.label()),
                Value::U64(object.points().len() as u64),
            ],
        )
    }
}

struct CollidingStreamObjectMeasure;

impl ObjectMeasure for CollidingStreamObjectMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("0.points").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label"), Column::u64("points")]).unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::coordinates()
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        out.push(
            object.bbox_min(),
            &[
                Value::U64(object.label()),
                Value::U64(object.points().len() as u64),
            ],
        )
    }
}

struct BadCostObjectMeasure;

impl ObjectMeasure for BadCostObjectMeasure {
    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.object_bad_cost").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label"), Column::u64("points")]).unwrap()
    }

    fn wants(&self) -> ObjectInputs {
        ObjectInputs::coordinates()
    }

    fn cost_per_object(&self) -> f64 {
        f64::NAN
    }

    fn apply(&self, object: ObjectView<'_>, out: &mut RowBuilder) -> Result<()> {
        out.push(
            object.bbox_min(),
            &[
                Value::U64(object.label()),
                Value::U64(object.points().len() as u64),
            ],
        )
    }
}

#[derive(Clone, Default)]
struct RegionSum {
    count: u64,
    sum: f64,
}

#[derive(Clone)]
struct RegionSumMeasure;

impl RegionMeasure for RegionSumMeasure {
    type Partial = RegionSum;

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_sum").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![
            Column::u64("label"),
            Column::u64("count"),
            Column::f64("sum"),
        ])
        .unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        vec![MeasureSource::intensity("channel0").unwrap()]
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::approximate(1.0e-12).unwrap()
    }

    fn cost_per_voxel(&self) -> f64 {
        7.5
    }

    fn zero(&self) -> Self::Partial {
        RegionSum::default()
    }

    fn add(
        &self,
        partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        values: MeasureValues<'_>,
    ) -> Result<()> {
        partial.count += 1;
        partial.sum += values.get(0).unwrap();
        Ok(())
    }

    fn merge(&self, into: &mut Self::Partial, from: Self::Partial) -> Result<()> {
        into.count += from.count;
        into.sum += from.sum;
        Ok(())
    }

    fn finish(&self, label: u64, partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push(
            [0, 0, 0],
            &[
                Value::U64(label),
                Value::U64(partial.count),
                Value::F64(partial.sum),
            ],
        )
    }
}

#[derive(Clone, Default)]
struct DriftRegionSum {
    count: f64,
}

struct DriftRegionMeasure;

impl RegionMeasure for DriftRegionMeasure {
    type Partial = DriftRegionSum;

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_drift").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label"), Column::f64("count")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        Vec::new()
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::approximate(0.25).unwrap()
    }

    fn zero(&self) -> Self::Partial {
        DriftRegionSum::default()
    }

    fn add(
        &self,
        partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        partial.count += 1.0;
        Ok(())
    }

    fn merge(&self, into: &mut Self::Partial, from: Self::Partial) -> Result<()> {
        into.count += from.count + 1.0;
        Ok(())
    }

    fn finish(&self, label: u64, partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label), Value::F64(partial.count)])
    }
}

#[derive(Clone, Default)]
struct RegionCount {
    count: u64,
}

struct RegionCountMeasure;

impl RegionMeasure for RegionCountMeasure {
    type Partial = RegionCount;

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_count").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label"), Column::u64("count")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        Vec::new()
    }

    fn zero(&self) -> Self::Partial {
        RegionCount::default()
    }

    fn add(
        &self,
        partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        partial.count += 1;
        Ok(())
    }

    fn merge(&self, into: &mut Self::Partial, from: Self::Partial) -> Result<()> {
        into.count += from.count;
        Ok(())
    }

    fn finish(&self, label: u64, partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label), Value::U64(partial.count)])
    }
}

struct OrderedRegionMeasure;

impl RegionMeasure for OrderedRegionMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.ordered").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        Vec::new()
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::OrderedOnly
    }

    fn zero(&self) -> Self::Partial {}

    fn add(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

struct WrongRowRegionMeasure;

impl RegionMeasure for WrongRowRegionMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_wrong_row").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        Vec::new()
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn zero(&self) -> Self::Partial {}

    fn add(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::F64(label as f64)])
    }
}

struct BadSchemaRegionMeasure;

impl RegionMeasure for BadSchemaRegionMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_bad_schema").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::f64("label")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        Vec::new()
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn zero(&self) -> Self::Partial {}

    fn add(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

struct BadCostRegionMeasure;

impl RegionMeasure for BadCostRegionMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_bad_cost").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        Vec::new()
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn cost_per_voxel(&self) -> f64 {
        -1.0
    }

    fn zero(&self) -> Self::Partial {}

    fn add(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

struct DuplicateSourceRegionMeasure;

impl RegionMeasure for DuplicateSourceRegionMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_duplicate_sources").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        vec![
            MeasureSource::intensity("channel0").unwrap(),
            MeasureSource::intensity("channel0").unwrap(),
        ]
    }

    fn zero(&self) -> Self::Partial {}

    fn add(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

struct TwoSourceRegionMeasure;

impl RegionMeasure for TwoSourceRegionMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.region_two_sources").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn sources(&self) -> Vec<MeasureSource> {
        vec![
            MeasureSource::intensity("channel0").unwrap(),
            MeasureSource::intensity("channel1").unwrap(),
        ]
    }

    fn zero(&self) -> Self::Partial {}

    fn add(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _values: MeasureValues<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

#[derive(Clone, Default)]
struct BoundaryContactCount {
    contacts: u64,
    background: u64,
}

#[derive(Clone)]
struct BoundaryContactCountMeasure {
    connectivity: Connectivity,
}

impl BoundaryMeasure for BoundaryContactCountMeasure {
    type Partial = BoundaryContactCount;

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_contacts").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![
            Column::u64("label"),
            Column::u64("contacts"),
            Column::u64("background"),
        ])
        .unwrap()
    }

    fn label_reach(&self) -> Reach {
        Reach::symmetric([1, 1, 1])
    }

    fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    fn cost_per_voxel(&self) -> f64 {
        3.25
    }

    fn zero(&self) -> Self::Partial {
        BoundaryContactCount::default()
    }

    fn add_contact(
        &self,
        partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        neighbour: u64,
        _face_or_offset: [isize; 3],
    ) -> Result<()> {
        partial.contacts += 1;
        if neighbour == 0 {
            partial.background += 1;
        }
        Ok(())
    }

    fn merge(&self, into: &mut Self::Partial, from: Self::Partial) -> Result<()> {
        into.contacts += from.contacts;
        into.background += from.background;
        Ok(())
    }

    fn finish(&self, label: u64, partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push(
            [0, 0, 0],
            &[
                Value::U64(label),
                Value::U64(partial.contacts),
                Value::U64(partial.background),
            ],
        )
    }
}

#[derive(Clone, Default)]
struct DriftBoundaryContacts {
    contacts: f64,
}

struct DriftBoundaryMeasure;

impl BoundaryMeasure for DriftBoundaryMeasure {
    type Partial = DriftBoundaryContacts;

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_drift").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label"), Column::f64("contacts")]).unwrap()
    }

    fn label_reach(&self) -> Reach {
        Reach::symmetric([1, 1, 1])
    }

    fn connectivity(&self) -> Connectivity {
        Connectivity::Faces
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::approximate(0.25).unwrap()
    }

    fn zero(&self) -> Self::Partial {
        DriftBoundaryContacts::default()
    }

    fn add_contact(
        &self,
        partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _neighbour: u64,
        _face_or_offset: [isize; 3],
    ) -> Result<()> {
        partial.contacts += 1.0;
        Ok(())
    }

    fn merge(&self, into: &mut Self::Partial, from: Self::Partial) -> Result<()> {
        into.contacts += from.contacts + 1.0;
        Ok(())
    }

    fn finish(&self, label: u64, partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push(
            [0, 0, 0],
            &[Value::U64(label), Value::F64(partial.contacts)],
        )
    }
}

struct BadCostBoundaryMeasure;

impl BoundaryMeasure for BadCostBoundaryMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_bad_cost").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn label_reach(&self) -> Reach {
        Reach::symmetric([1, 1, 1])
    }

    fn connectivity(&self) -> Connectivity {
        Connectivity::Faces
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn cost_per_voxel(&self) -> f64 {
        f64::NEG_INFINITY
    }

    fn zero(&self) -> Self::Partial {}

    fn add_contact(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _neighbour: u64,
        _face_or_offset: [isize; 3],
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

struct WideBoundaryMeasure;

impl BoundaryMeasure for WideBoundaryMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_wide").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn label_reach(&self) -> Reach {
        Reach::symmetric([2, 1, 1])
    }

    fn connectivity(&self) -> Connectivity {
        Connectivity::Faces
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn zero(&self) -> Self::Partial {}

    fn add_contact(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _neighbour: u64,
        _face_or_offset: [isize; 3],
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

struct OrderedBoundaryMeasure;

impl BoundaryMeasure for OrderedBoundaryMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_ordered").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn label_reach(&self) -> Reach {
        Reach::symmetric([1, 1, 1])
    }

    fn connectivity(&self) -> Connectivity {
        Connectivity::Faces
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::OrderedOnly
    }

    fn zero(&self) -> Self::Partial {}

    fn add_contact(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _neighbour: u64,
        _face_or_offset: [isize; 3],
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

struct WrongRowBoundaryMeasure;

impl BoundaryMeasure for WrongRowBoundaryMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_wrong_row").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label")]).unwrap()
    }

    fn label_reach(&self) -> Reach {
        Reach::symmetric([1, 1, 1])
    }

    fn connectivity(&self) -> Connectivity {
        Connectivity::Faces
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn zero(&self) -> Self::Partial {}

    fn add_contact(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _neighbour: u64,
        _face_or_offset: [isize; 3],
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::F64(label as f64)])
    }
}

struct BadSchemaBoundaryMeasure;

impl BoundaryMeasure for BadSchemaBoundaryMeasure {
    type Partial = ();

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new("test.boundary_bad_schema").unwrap()
    }

    fn schema(&self) -> Schema {
        Schema::new(vec![Column::u64("label_a")]).unwrap()
    }

    fn label_reach(&self) -> Reach {
        Reach::symmetric([1, 1, 1])
    }

    fn connectivity(&self) -> Connectivity {
        Connectivity::Faces
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn zero(&self) -> Self::Partial {}

    fn add_contact(
        &self,
        _partial: &mut Self::Partial,
        _at: [usize; 3],
        _label: u64,
        _neighbour: u64,
        _face_or_offset: [isize; 3],
    ) -> Result<()> {
        Ok(())
    }

    fn merge(&self, _into: &mut Self::Partial, _from: Self::Partial) -> Result<()> {
        Ok(())
    }

    fn finish(&self, label: u64, _partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        out.push([0, 0, 0], &[Value::U64(label)])
    }
}

fn labels() -> Voxels {
    let array = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        if y == 0 && x == 0 {
            1.0
        } else if (1..5).contains(&z) && y >= 1 && x >= 1 {
            2.0
        } else {
            0.0
        }
    });
    array.into()
}

fn coordinate_values() -> Voxels {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        z as f64 * 10.0 + y as f64 + x as f64 / 10.0
    })
    .into()
}

fn integer_coordinate_values(scale: f64) -> Voxels {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        scale * (z as f64 * 10.0 + y as f64 + x as f64)
    })
    .into()
}

fn convex_hull_labels() -> Voxels {
    let mut array = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    array[[0, 0, 0]] = 2.0;
    array[[1, 0, 0]] = 2.0;
    array[[0, 1, 0]] = 2.0;
    array[[0, 0, 1]] = 2.0;
    for z in 2..=3 {
        for y in 1..=2 {
            for x in 1..=2 {
                array[[z, y, x]] = 5.0;
            }
        }
    }
    array.into()
}

fn contact_labels() -> Voxels {
    let mut array = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    array[[1, 1, 0]] = 1.0;
    array[[1, 1, 1]] = 2.0;
    array[[1, 2, 1]] = 2.0;
    array.into()
}

fn base(block: [usize; 3]) -> Decomposition {
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["values".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new(VOLUME, block).unwrap(),
        )],
        chain_reach: [0, 0, 0],
    }
}

fn workflow() -> Workflow {
    Workflow::new(Chain::op(CoordinateValues), VOLUME, Dtype::F64)
}

#[test]
fn measurement_builder_compiles_to_tabulation_phases() {
    let plan = Measurements::for_labels(0usize)
        .shape(ShapeSet::standard())
        .intensity(IntensityImage::<0>::new(1usize), IntensitySet::standard())
        .stream("objects")
        .lifecycle(Lifecycle::Persistent)
        .build(base([3, 2, 3]))
        .unwrap();

    assert_eq!(plan.decomposition.n_phases(), 3);
    assert_eq!(plan.rows_phase, Some(2));
    assert_eq!(plan.stream, "objects");
    assert_eq!(plan.fold_law(), FoldLaw::ExactAssociative);
    assert!(plan.shape.wants_any());
    assert!(plan.intensity.wants_any());
}

#[test]
fn measurement_builder_runs_and_collects_derived_rows() {
    let plan = Measurements::for_labels(0usize)
        .shape(ShapeSet::standard())
        .intensity(IntensityImage::<0>::new(1usize), IntensitySet::standard())
        .stream("objects")
        .build(base([3, 2, 3]))
        .unwrap();
    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();

    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());

    execute_phases(
        "measurements",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let class_a_rows = plan.class_a_rows().unwrap();
    assert_eq!(class_a_rows.stream(), plan.stream.as_str());
    assert_eq!(Some(class_a_rows.phase()), plan.rows_phase);
    let values = collect_class_a_values(&env, &class_a_rows, VOLUME, plan.fixed).unwrap();
    let shapes = collect_class_a_shapes(&env, &class_a_rows, VOLUME, plan.fixed).unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(shapes.len(), 2);

    let first = IntensityMeasurements::from_values(&values[0]);
    assert_eq!(first.label, 1);
    assert_eq!(first.count, VOLUME[0] as u64);
    assert_eq!(first.finite_count, first.count);
    assert!(first.mean.is_some());
    assert!(first.min.unwrap() <= first.max.unwrap());
    assert_eq!(IntensityFeature::ALL.len(), 10);
    assert_eq!(IntensityFeature::Mean.column_name(), "mean");
    assert_eq!(
        IntensityFeature::WeightedCentroidX.column_name(),
        "weighted_centroid_x"
    );
    assert_eq!(
        first.feature(IntensityFeature::Count),
        Some(first.count as f64)
    );
    assert_eq!(
        first.feature(IntensityFeature::FiniteCount),
        Some(first.finite_count as f64)
    );
    assert_eq!(first.feature(IntensityFeature::Nonfinite), Some(0.0));
    assert_eq!(first.feature(IntensityFeature::Sum), Some(first.sum));
    assert_eq!(first.feature(IntensityFeature::Mean), first.mean);
    assert_eq!(first.feature(IntensityFeature::Min), first.min);
    assert_eq!(first.feature(IntensityFeature::Max), first.max);
    assert_eq!(
        first.feature(IntensityFeature::WeightedCentroidX),
        first.weighted_centroid.map(|centroid| centroid[2])
    );

    let shape = ShapeMeasurements::from_shape(&shapes[0]);
    assert_eq!(shape.label, 1);
    assert_eq!(shape.count, VOLUME[0] as u64);
    assert_eq!(shape.bbox_min, [0, 0, 0]);
    assert_eq!(shape.bbox_max, [6, 1, 1]);
    assert_eq!(shape.bbox_extent, [6, 1, 1]);
    assert_eq!(shape.bbox_volume, 6);
    assert_eq!(shape.bbox_fill_fraction, Some(1.0));
    assert_eq!(
        shape.equivalent_sphere_diameter,
        equivalent_sphere_diameter(shape.count)
    );
    assert_eq!(
        shape.equivalent_sphere_radius,
        equivalent_sphere_radius(shape.count)
    );
}

#[test]
fn equivalent_sphere_helpers_accept_nominal_voxel_counts() {
    let voxels = VoxelCount::new(8);
    assert_eq!(voxels.get(), 8);
    assert_eq!(
        equivalent_sphere_radius_for_voxels(voxels),
        equivalent_sphere_radius(8)
    );
    assert_eq!(
        equivalent_sphere_diameter_for_voxels(voxels),
        equivalent_sphere_diameter(8)
    );
    assert_eq!(equivalent_sphere_radius_for_voxels(VoxelCount::new(0)), 0.0);
    assert_eq!(
        equivalent_sphere_diameter_for_voxels(VoxelCount::new(0)),
        0.0
    );
}

#[test]
fn shape_only_measurement_runs_without_value_image() {
    let plan = Measurements::for_labels(0usize)
        .shape(ShapeSet::standard())
        .build(base([3, 2, 3]))
        .unwrap();
    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());

    execute_phases(
        "measurements",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let shapes = collect_shapes(
        &env,
        &plan.stream,
        plan.rows_phase.unwrap(),
        VOLUME,
        plan.fixed,
    )
    .unwrap();
    assert_eq!(shapes.len(), 2);
    let first = ShapeMeasurements::from_shape(&shapes[0]);
    assert_eq!(first.label, 1);
    assert_eq!(first.bbox_extent, [6, 1, 1]);
    let second = ShapeMeasurements::from_shape(&shapes[1]);
    assert_eq!(second.label, 2);
    assert_eq!(second.bbox_min, [1, 1, 1]);
    assert_eq!(second.bbox_max, [5, 4, 3]);
    assert_eq!(second.bbox_extent, [4, 3, 2]);
    assert_eq!(second.bbox_fill_fraction, Some(1.0));
}

#[test]
fn measurement_builder_fuses_multiple_basic_intensity_channels() {
    let input = labels();
    let supplied = vec![
        integer_coordinate_values(1.0),
        integer_coordinate_values(2.0),
    ];
    let facts = MeasurementSourceFacts::from_inputs(&input, &supplied).unwrap();
    let plan = Measurements::for_labels(0usize)
        .shape(ShapeSet::standard())
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .intensity(
            IntensityImage::<1>::new(ImageId::supplied(1)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .stream("objects")
        .build_checked(base([3, 2, 3]), facts)
        .unwrap();

    assert_eq!(plan.rows_phase, Some(2));
    assert_eq!(plan.class_a_intensity_rows_phase(0), Some(2));
    assert_eq!(plan.class_a_intensity_rows_phase(1), Some(3));
    assert_eq!(plan.class_a_intensity_stream(0), Some("objects"));
    assert_eq!(
        plan.class_a_intensity_stream(1),
        Some("objects.intensity.1")
    );

    let env =
        ArrayEnvironment::with_inputs(input, supplied, &plan.decomposition, [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "multi-intensity measurements",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let first_rows = plan.class_a_intensity_rows(0).unwrap();
    let second_rows = plan.class_a_intensity_rows(1).unwrap();
    let first = collect_class_a_values(&env, &first_rows, VOLUME, plan.fixed).unwrap();
    let second = collect_class_a_values(&env, &second_rows, VOLUME, plan.fixed).unwrap();
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.label, b.label);
        assert_eq!(a.count, b.count);
        assert_eq!(b.sum, 2.0 * a.sum);
        assert_eq!(b.min, 2.0 * a.min);
        assert_eq!(b.max, 2.0 * a.max);
    }
}

#[test]
fn centroid_relationships_are_derived_from_planned_shape_rows() {
    let plan = Measurements::for_labels(0usize)
        .shape(ShapeSet::basic())
        .build(base([3, 2, 3]))
        .unwrap();
    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());

    execute_phases(
        "measurements",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let shapes = collect_shapes(
        &env,
        &plan.stream,
        plan.rows_phase.unwrap(),
        VOLUME,
        plan.fixed,
    )
    .unwrap();
    let got = object_centroid_relationships_from_shapes(&shapes, PhysicalSpacing::unit()).unwrap();
    let expected =
        object_centroid_relationships(labels().view::<f64>().unwrap(), PhysicalSpacing::unit())
            .unwrap();
    assert_eq!(got, expected);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].labels, [1, 2]);
    assert_eq!(got[0].centroid_delta, [0.0, 2.0, 1.5]);
    assert_eq!(got[0].centroid_distance, 2.5);
}

#[test]
fn measurement_builder_runs_planned_centroid_relationships() {
    let spacing = PhysicalSpacing::unit();
    let expected = object_centroid_relationships(labels().view::<f64>().unwrap(), spacing).unwrap();
    for block in [[6, 4, 3], [3, 2, 2]] {
        let plan = Measurements::for_labels(0usize)
            .centroid_relationships(spacing)
            .stream("objects")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.rows_phase, Some(2));
        assert_eq!(plan.centroid_relationship_rows_phase(), Some(3));
        assert_eq!(plan.centroid_relationship_spacing(), Some(spacing));
        let relationship_rows = plan.centroid_relationship_rows_with_contract().unwrap();
        assert_eq!(relationship_rows.contract(), spacing);
        assert_eq!(
            relationship_rows.rows().stream(),
            plan.centroid_relationship_stream().unwrap()
        );
        assert_eq!(
            relationship_rows.rows().phase(),
            plan.centroid_relationship_rows_phase().unwrap()
        );
        let env =
            ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());

        execute_phases(
            "measurements",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got =
            collect_centroid_relationship_rows_with_contract(&env, &relationship_rows, VOLUME)
                .unwrap();
        assert_eq!(got, expected, "block {block:?}");
    }
}

#[test]
fn centroid_neighbor_rows_have_a_canonical_schema_and_collector() {
    let rows = vec![
        ObjectNeighborMeasurements {
            label: 1,
            within_distance_neighbors: 1,
            closest_label: Some(2),
            closest_distance: Some(4.0),
            second_closest_label: Some(3),
            second_closest_distance: Some(10.0),
            angle_between_closest: Some(0.0),
        },
        ObjectNeighborMeasurements {
            label: 2,
            within_distance_neighbors: 1,
            closest_label: Some(1),
            closest_distance: Some(4.0),
            second_closest_label: None,
            second_closest_distance: None,
            angle_between_closest: None,
        },
    ];
    let encoded = encode_centroid_neighbor_measurements(&rows).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, centroid_neighbor_measurement_schema());
    assert_eq!(schema.columns()[0].name(), "label");
    assert_eq!(schema.columns()[1].name(), "within_distance_neighbors");
    assert_eq!(schema.columns()[6].name(), "angle_between_closest");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("neighbor.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("neighbor.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_centroid_neighbor_measurements(&env, "neighbor.rows", 0, VOLUME).unwrap();
    assert_eq!(got, rows);

    assert!(
        encode_centroid_neighbor_measurements(&[ObjectNeighborMeasurements {
            label: 0,
            within_distance_neighbors: 0,
            closest_label: None,
            closest_distance: None,
            second_closest_label: None,
            second_closest_distance: None,
            angle_between_closest: None,
        }])
        .is_err()
    );
}

#[test]
fn measurement_builder_runs_planned_centroid_neighbor_summaries() {
    let spacing = PhysicalSpacing::unit();
    let threshold = WithinDistanceThreshold::new(3.0).unwrap();
    let label_array = labels().view::<f64>().unwrap().to_owned();
    let relationships = object_centroid_relationships(label_array.view(), spacing).unwrap();
    let expected = summarize_centroid_neighbors_within(&relationships, threshold).unwrap();

    for block in [[6, 4, 3], [3, 2, 2]] {
        let plan = Measurements::for_labels(0usize)
            .centroid_neighbor_summary_within(spacing, threshold)
            .stream("neighbors")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.centroid_relationship_spacing(), Some(spacing));
        assert_eq!(plan.centroid_neighbor_threshold(), Some(threshold));
        assert_eq!(plan.centroid_neighbor_rows_phase(), Some(4));
        let rows = plan.centroid_neighbor_rows_with_contract().unwrap();
        assert_eq!(rows.stream(), plan.centroid_neighbor_stream().unwrap());
        assert_eq!(rows.phase(), plan.centroid_neighbor_rows_phase().unwrap());
        assert_eq!(rows.contract(), threshold);

        let env =
            ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());
        execute_phases(
            "builder centroid neighbor summaries",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got = collect_centroid_neighbor_rows_with_contract(&env, &rows, VOLUME).unwrap();
        assert_eq!(got, expected, "block {block:?}");
    }

    let mismatch = match Measurements::for_labels(0usize)
        .centroid_relationships(PhysicalSpacing::new([2.0, 1.0, 1.0]).unwrap())
        .centroid_neighbor_summary(spacing, 3.0)
        .unwrap()
        .build(base([3, 2, 2]))
    {
        Ok(_) => panic!("mismatched centroid neighbor spacing unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(mismatch.contains("same spacing"));
}

#[test]
fn measurement_builder_runs_planned_boundary_distance_relationships() {
    let spacing = PhysicalSpacing::unit();
    let expected =
        object_boundary_distance_relationships(labels().view::<f64>().unwrap(), spacing).unwrap();
    for block in [[6, 4, 3], [3, 2, 2]] {
        let plan = Measurements::for_labels(0usize)
            .boundary_distance_relationships(spacing)
            .stream("objects")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.rows_phase, None);
        assert_eq!(plan.boundary_distance_relationship_rows_phase(), Some(2));
        assert_eq!(plan.boundary_distance_relationship_spacing(), Some(spacing));
        let relationship_rows = plan
            .boundary_distance_relationship_rows_with_contract()
            .unwrap();
        assert_eq!(relationship_rows.contract(), spacing);
        assert_eq!(
            relationship_rows.rows().stream(),
            plan.boundary_distance_relationship_stream().unwrap()
        );
        assert_eq!(
            relationship_rows.rows().phase(),
            plan.boundary_distance_relationship_rows_phase().unwrap()
        );
        let env =
            ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());

        execute_phases(
            "measure boundary-distance relationships",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got = collect_boundary_distance_relationship_rows_with_contract(
            &env,
            &relationship_rows,
            VOLUME,
        )
        .unwrap();
        assert_eq!(got, expected, "block {block:?}");
    }
}

#[test]
fn measurement_builder_runs_planned_expansion_until_adjacent_relationships() {
    let spacing = PhysicalSpacing::unit();
    let expected =
        object_expansion_until_adjacent_relationships(labels().view::<f64>().unwrap(), spacing)
            .unwrap();
    for block in [[6, 4, 3], [3, 2, 2]] {
        let plan = Measurements::for_labels(0usize)
            .expansion_until_adjacent_relationships(spacing)
            .stream("objects")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.rows_phase, None);
        assert_eq!(
            plan.expansion_until_adjacent_relationship_rows_phase(),
            Some(2)
        );
        assert_eq!(
            plan.expansion_until_adjacent_relationship_spacing(),
            Some(spacing)
        );
        let relationship_rows = plan
            .expansion_until_adjacent_relationship_rows_with_contract()
            .unwrap();
        assert_eq!(relationship_rows.contract(), spacing);
        assert_eq!(
            relationship_rows.rows().stream(),
            plan.expansion_until_adjacent_relationship_stream().unwrap()
        );
        assert_eq!(
            relationship_rows.rows().phase(),
            plan.expansion_until_adjacent_relationship_rows_phase()
                .unwrap()
        );
        let env =
            ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());

        execute_phases(
            "measure expansion-until-adjacent relationships",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got =
            collect_expansion_relationship_rows_with_contract(&env, &relationship_rows, VOLUME)
                .unwrap();
        assert_eq!(got, expected, "block {block:?}");
    }
}

#[test]
fn centroid_relationship_rows_have_a_canonical_schema_and_collector() {
    let relationships = vec![
        ObjectRelationshipMeasurements {
            labels: [1, 2],
            centroid_distance: 5.0,
            centroid_delta: [3.0, 4.0, 0.0],
        },
        ObjectRelationshipMeasurements {
            labels: [1, 3],
            centroid_distance: 13.0,
            centroid_delta: [3.0, 4.0, 12.0],
        },
    ];
    let encoded = encode_centroid_relationship_measurements(&relationships).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, centroid_relationship_measurement_schema());
    assert_eq!(ObjectRelationshipFeature::ALL.len(), 4);
    assert_eq!(
        ObjectRelationshipFeature::CentroidDelta2.column_name(),
        "centroid_delta_2"
    );
    assert_eq!(
        relationships[1].feature(ObjectRelationshipFeature::CentroidDistance),
        13.0
    );
    assert_eq!(
        relationships[1].feature(ObjectRelationshipFeature::CentroidDelta2),
        12.0
    );

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("relationship.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("relationship.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got =
        collect_centroid_relationship_measurements(&env, "relationship.rows", 0, VOLUME).unwrap();
    assert_eq!(got, relationships);
    assert!(
        encode_centroid_relationship_measurements(&[ObjectRelationshipMeasurements {
            labels: [2, 1],
            centroid_distance: 1.0,
            centroid_delta: [1.0, 0.0, 0.0],
        }])
        .is_err()
    );
    let nan_delta = encode_centroid_relationship_measurements(&[ObjectRelationshipMeasurements {
        labels: [1, 2],
        centroid_distance: 1.0,
        centroid_delta: [f64::NAN, 0.0, 0.0],
    }])
    .unwrap_err()
    .to_string();
    assert!(nan_delta.contains("non-finite physical delta"));
}

#[test]
fn boundary_distance_relationship_rows_have_a_canonical_schema_and_collector() {
    let relationships = vec![
        ObjectBoundaryDistanceMeasurements {
            labels: [1, 2],
            boundary_distance: 5.0,
            boundary_delta: [3.0, 4.0, 0.0],
            nearest_points: [[1, 2, 3], [4, 6, 3]],
        },
        ObjectBoundaryDistanceMeasurements {
            labels: [1, 3],
            boundary_distance: 13.0,
            boundary_delta: [3.0, 4.0, 12.0],
            nearest_points: [[0, 0, 0], [1, 2, 6]],
        },
    ];
    let encoded = encode_boundary_distance_relationship_measurements(&relationships).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, boundary_distance_relationship_measurement_schema());
    assert_eq!(ObjectBoundaryDistanceFeature::ALL.len(), 4);
    assert_eq!(
        ObjectBoundaryDistanceFeature::BoundaryDelta1.column_name(),
        "boundary_delta_1"
    );
    assert_eq!(
        relationships[0].feature(ObjectBoundaryDistanceFeature::BoundaryDistance),
        5.0
    );
    assert_eq!(
        relationships[0].feature(ObjectBoundaryDistanceFeature::BoundaryDelta1),
        4.0
    );

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("boundary-distance.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("boundary-distance.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_boundary_distance_relationship_measurements(
        &env,
        "boundary-distance.rows",
        0,
        VOLUME,
    )
    .unwrap();
    assert_eq!(got, relationships);
    assert!(encode_boundary_distance_relationship_measurements(&[
        ObjectBoundaryDistanceMeasurements {
            labels: [2, 1],
            boundary_distance: 1.0,
            boundary_delta: [1.0, 0.0, 0.0],
            nearest_points: [[0, 0, 0], [1, 0, 0]],
        }
    ])
    .is_err());
    let nan_delta =
        encode_boundary_distance_relationship_measurements(&[ObjectBoundaryDistanceMeasurements {
            labels: [1, 2],
            boundary_distance: 1.0,
            boundary_delta: [0.0, f64::NAN, 0.0],
            nearest_points: [[0, 0, 0], [1, 0, 0]],
        }])
        .unwrap_err()
        .to_string();
    assert!(nan_delta.contains("non-finite physical delta"));
}

#[test]
fn shape_measurements_report_yx_orientation_angle() {
    let x_shape = RegionShape {
        label: 1,
        at: [0, 0, 2],
        count: 4,
        position: [0, 0, 6],
        central: [0, 0, 0, 0, 0, 6],
        bbox_min: [0, 0, 0],
        bbox_max: [1, 1, 4],
    };
    let x_measurements = ShapeMeasurements::from_shape(&x_shape);
    assert_eq!(orientation_yx(&x_shape), Some(0.0));
    assert_eq!(x_measurements.orientation_yx, Some(0.0));

    let y_shape = RegionShape {
        label: 1,
        at: [0, 2, 0],
        count: 4,
        position: [0, 6, 0],
        central: [0, 0, 0, 6, 0, 0],
        bbox_min: [0, 0, 0],
        bbox_max: [1, 4, 1],
    };
    let y_measurements = ShapeMeasurements::from_shape(&y_shape);
    assert!((y_measurements.orientation_yx.unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
}

#[test]
fn shape_measurements_expose_named_scalar_features() {
    let shape = RegionShape {
        label: 7,
        at: [0, 0, 2],
        count: 4,
        position: [0, 0, 6],
        central: [0, 0, 0, 0, 0, 6],
        bbox_min: [0, 0, 0],
        bbox_max: [1, 1, 4],
    };
    let measurements = ShapeMeasurements::from_shape(&shape);
    assert_eq!(ShapeFeature::ALL.len(), 25);
    assert_eq!(ShapeFeature::Count.column_name(), "count");
    assert_eq!(ShapeFeature::OrientationYx.column_name(), "orientation_yx");
    assert_eq!(measurements.feature(ShapeFeature::Count), Some(4.0));
    assert_eq!(measurements.feature(ShapeFeature::CentroidX), Some(1.5));
    assert_eq!(measurements.feature(ShapeFeature::BboxExtentX), Some(4.0));
    assert_eq!(measurements.feature(ShapeFeature::BboxVolume), Some(4.0));
    assert_eq!(
        measurements.feature(ShapeFeature::BboxFillFraction),
        Some(1.0)
    );
    assert_eq!(
        measurements.feature(ShapeFeature::EquivalentSphereDiameter),
        Some(measurements.equivalent_sphere_diameter)
    );
    assert_eq!(
        measurements.feature(ShapeFeature::PrincipalAxisLength0),
        measurements.principal_axis_lengths.map(|length| length[0])
    );
    assert_eq!(measurements.feature(ShapeFeature::OrientationYx), Some(0.0));

    let point = ShapeMeasurements::from_shape(&RegionShape {
        label: 8,
        at: [0, 0, 0],
        count: 1,
        position: [0, 0, 0],
        central: [0; 6],
        bbox_min: [0, 0, 0],
        bbox_max: [1, 1, 1],
    });
    assert_eq!(point.feature(ShapeFeature::PrincipalAxisLength0), Some(0.0));
    assert_eq!(point.feature(ShapeFeature::OrientationYx), None);
    assert_eq!(point.feature(ShapeFeature::Eccentricity), None);
}

#[test]
fn integer_feature_projection_has_an_exact_precision_boundary() {
    let largest = FeatureScalar::MAX_EXACT_INTEGER;
    assert_eq!(
        FeatureScalar::exact_u64(largest).unwrap().get(),
        largest as f64
    );
    assert_eq!(FeatureScalar::exact_u64(largest + 1), None);

    let shape = ShapeMeasurements {
        label: 1,
        count: largest + 1,
        centroid: None,
        bbox_min: [0, 0, 0],
        bbox_max: [largest + 1, 1, 1],
        bbox_extent: [largest + 1, 1, 1],
        bbox_volume: largest + 1,
        bbox_fill_fraction: None,
        equivalent_sphere_radius: 0.0,
        equivalent_sphere_diameter: 0.0,
        principal_axis_lengths: None,
        orientation: None,
        orientation_yx: None,
        eccentricity: None,
    };
    assert_eq!(shape.feature(ShapeFeature::Count), None);
    assert_eq!(shape.feature(ShapeFeature::BboxExtentZ), None);
    assert_eq!(shape.feature(ShapeFeature::BboxVolume), None);

    let intensity = IntensityMeasurements {
        label: 1,
        count: largest + 1,
        finite_count: largest,
        nonfinite: largest + 1,
        sum: 0.0,
        mean: None,
        min: None,
        max: None,
        weighted_centroid: None,
    };
    assert_eq!(intensity.feature(IntensityFeature::Count), None);
    assert_eq!(
        intensity.feature(IntensityFeature::FiniteCount),
        Some(largest as f64)
    );
    assert_eq!(intensity.feature(IntensityFeature::Nonfinite), None);

    let wide_usize = usize::try_from(largest + 1).unwrap();
    let geometry = ObjectGeometryMeasurements {
        label: 1,
        count: largest + 1,
        bbox_min: [wide_usize, 0, 0],
        bbox_max: [wide_usize, 1, 1],
        physical_bbox_extent: [1.0, 1.0, 1.0],
        max_voxel_feret_diameter: 1.0,
    };
    assert_eq!(geometry.checked_feature(ObjectGeometryFeature::Count), None);
    assert_eq!(
        geometry.checked_feature(ObjectGeometryFeature::BboxMinZ),
        None
    );
    assert!(geometry.feature(ObjectGeometryFeature::Count).is_finite());

    let topology = ObjectTopologyMeasurements {
        label: 1,
        convention: ObjectTopologyConvention::default(),
        components: wide_usize,
        tunnels: 1,
        cavities: 1,
        euler_number: -(largest as i64) - 1,
    };
    assert_eq!(
        topology.checked_feature(ObjectTopologyFeature::Components),
        None
    );
    assert_eq!(
        topology.checked_feature(ObjectTopologyFeature::EulerNumber),
        None
    );

    let components = ObjectComponentMeasurements {
        label: 1,
        count: largest + 1,
        connectivity: Connectivity::Faces,
        components: wide_usize,
    };
    assert_eq!(
        components.checked_feature(ObjectComponentFeature::Count),
        None
    );
    assert_eq!(
        components.checked_feature(ObjectComponentFeature::Components),
        None
    );

    let sphere = ObjectEnclosingSphereMeasurements {
        label: 1,
        count: largest + 1,
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
        support_points: wide_usize,
    };
    assert_eq!(
        sphere.checked_feature(ObjectEnclosingSphereFeature::Count),
        None
    );
    assert_eq!(
        sphere.checked_feature(ObjectEnclosingSphereFeature::SupportPoints),
        None
    );

    let hull = ObjectConvexHullMeasurements {
        label: 1,
        count: largest + 1,
        hull_vertices: wide_usize,
        hull_faces: wide_usize,
        surface_area: 1.0,
        volume: 1.0,
        max_hull_feret_diameter: 1.0,
    };
    assert_eq!(hull.checked_feature(ObjectConvexHullFeature::Count), None);
    assert_eq!(
        hull.checked_feature(ObjectConvexHullFeature::HullVertices),
        None
    );

    let boundary = BoundaryMeasurements {
        label: 1,
        at: [0, 0, 0],
        count: largest + 1,
        boundary_voxels: largest + 1,
        boundary_faces: [largest, 1, 1],
    };
    assert_eq!(
        boundary.checked_feature(BoundaryFeature::Count, PhysicalSpacing::unit()),
        None
    );
    assert_eq!(
        boundary.checked_feature(BoundaryFeature::BoundaryFaces, PhysicalSpacing::unit()),
        None
    );

    let contact = ContactMeasurements {
        labels: [1, 2],
        at: [0, 0, 0],
        faces: largest + 1,
        faces_by_axis: [largest + 1, 0, 0],
    };
    assert_eq!(
        contact.checked_feature(ContactFeature::Faces, PhysicalSpacing::unit()),
        None
    );

    let neighbors = NeighborSummary {
        label: 1,
        touching_neighbors: largest + 1,
        contact_faces: largest + 1,
        contact_faces_by_axis: [largest + 1, 0, 0],
    };
    assert_eq!(
        neighbors.checked_feature(
            TouchingNeighborFeature::TouchingNeighbors,
            PhysicalSpacing::unit()
        ),
        None
    );
}

fn expected_boundaries() -> Vec<BoundaryMeasurements> {
    let labels = labels();
    let labels = labels.view::<f64>().unwrap();
    let mut counts = std::collections::BTreeMap::<u64, (u64, [u64; 3], u64, [u64; 3])>::new();
    for ((z, y, x), &raw) in labels.indexed_iter() {
        let label = raw as u64;
        if label == 0 {
            continue;
        }
        let entry = counts.entry(label).or_insert((0, [0; 3], 0, [0; 3]));
        entry.0 += 1;
        entry.1[0] += z as u64;
        entry.1[1] += y as u64;
        entry.1[2] += x as u64;
        let mut boundary_faces = [0; 3];
        for axis in 0..3 {
            for step in [-1isize, 1] {
                let mut neighbour = [z as isize, y as isize, x as isize];
                neighbour[axis] += step;
                if neighbour[axis] < 0 || neighbour[axis] >= VOLUME[axis] as isize {
                    continue;
                }
                let other = labels[[
                    neighbour[0] as usize,
                    neighbour[1] as usize,
                    neighbour[2] as usize,
                ]] as u64;
                if other != label {
                    boundary_faces[axis] += 1;
                }
            }
        }
        if boundary_faces.iter().any(|faces| *faces > 0) {
            entry.2 += 1;
        }
        for axis in 0..3 {
            entry.3[axis] += boundary_faces[axis];
        }
    }
    counts
        .into_iter()
        .map(
            |(label, (count, position, boundary_voxels, boundary_faces))| {
                let mut at = [0usize; 3];
                for axis in 0..3 {
                    at[axis] = ((2 * position[axis] as u128 + count as u128) / (2 * count as u128))
                        as usize;
                }
                BoundaryMeasurements {
                    label,
                    at,
                    count,
                    boundary_voxels,
                    boundary_faces,
                }
            },
        )
        .collect()
}

fn expected_contacts() -> Vec<ContactMeasurements> {
    vec![ContactMeasurements {
        labels: [1, 2],
        at: [1, 1, 0],
        faces: 1,
        faces_by_axis: [0, 0, 1],
    }]
}

#[test]
fn boundary_measurement_uses_a_label_halo_and_is_decomposition_invariant() {
    let expected = expected_boundaries();
    assert_eq!(expected.len(), 2);

    for block in [[6, 4, 3], [3, 2, 2]] {
        let plan = Measurements::for_labels(0usize)
            .shape(ShapeSet::boundary())
            .stream("objects")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.rows_phase, None);
        let boundary_rows = plan.boundary_rows().unwrap();
        assert_eq!(boundary_rows.stream(), plan.boundary_stream().unwrap());
        assert_eq!(boundary_rows.phase(), plan.boundary_rows_phase().unwrap());
        let env =
            ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());

        execute_phases(
            "measurements",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got = collect_boundary_rows(&env, &boundary_rows, VOLUME).unwrap();
        assert_eq!(got, expected, "block {block:?}");
    }

    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    assert_eq!(
        expected[0].surface_area(spacing),
        expected[0].boundary_faces[0] as f64 * 3.0 * 5.0
            + expected[0].boundary_faces[1] as f64 * 2.0 * 5.0
            + expected[0].boundary_faces[2] as f64 * 2.0 * 3.0
    );
    assert_eq!(BoundaryFeature::ALL.len(), 7);
    assert_eq!(
        BoundaryFeature::BoundaryFacesZ.column_name(),
        "boundary_faces_z"
    );
    assert_eq!(
        expected[0].feature(BoundaryFeature::Count, spacing),
        expected[0].count as f64
    );
    assert_eq!(
        expected[0].feature(BoundaryFeature::BoundaryVoxels, spacing),
        expected[0].boundary_voxels as f64
    );
    assert_eq!(
        expected[0].feature(BoundaryFeature::BoundaryFaces, spacing),
        expected[0].boundary_faces.iter().sum::<u64>() as f64
    );
    assert_eq!(
        expected[0].feature(BoundaryFeature::SurfaceArea, spacing),
        expected[0].surface_area(spacing)
    );
    assert_eq!(
        expected[0].surface_area(PhysicalSpacing::unit()),
        expected[0].boundary_faces.iter().sum::<u64>() as f64
    );
    assert!(PhysicalSpacing::new([1.0, 0.0, 1.0]).is_err());
    assert!(PhysicalSpacing::new([1.0, f64::NAN, 1.0]).is_err());
}

#[test]
fn contact_measurement_uses_canonical_faces_and_is_decomposition_invariant() {
    let expected = expected_contacts();

    for block in [[6, 4, 3], [2, 2, 1]] {
        let plan = Measurements::for_labels(0usize)
            .shape(ShapeSet::contacts())
            .stream("objects")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.boundary_stream(), None);
        assert_eq!(plan.boundary_rows_phase(), None);
        let contact_rows = plan.contact_rows().unwrap();
        assert_eq!(contact_rows.stream(), plan.contact_stream().unwrap());
        assert_eq!(contact_rows.phase(), plan.contact_rows_phase().unwrap());
        let env = ArrayEnvironment::new(contact_labels(), plan.decomposition.n_phases(), [2, 2, 2])
            .unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());

        execute_phases(
            "measurements",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got = collect_contact_rows(&env, &contact_rows, VOLUME).unwrap();
        assert_eq!(got, expected, "block {block:?}");
    }

    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    assert_eq!(expected[0].contact_area(spacing), 6.0);
    assert_eq!(ContactFeature::ALL.len(), 5);
    assert_eq!(ContactFeature::FacesX.column_name(), "faces_x");
    assert_eq!(
        expected[0].feature(ContactFeature::Faces, spacing),
        expected[0].faces as f64
    );
    assert_eq!(
        expected[0].feature(ContactFeature::FacesX, spacing),
        expected[0].faces_by_axis[2] as f64
    );
    assert_eq!(
        expected[0].feature(ContactFeature::ContactArea, spacing),
        expected[0].contact_area(spacing)
    );

    let summaries = summarize_touching_neighbors(&expected).unwrap();
    assert_eq!(summaries.len(), 2);
    assert_eq!(summaries[0].label, 1);
    assert_eq!(summaries[0].touching_neighbors, 1);
    assert_eq!(summaries[0].contact_faces, 1);
    assert_eq!(summaries[0].contact_area(spacing), 6.0);
    assert_eq!(TouchingNeighborFeature::ALL.len(), 6);
    assert_eq!(
        TouchingNeighborFeature::ContactArea.column_name(),
        "contact_area"
    );
    assert_eq!(
        summaries[0].feature(TouchingNeighborFeature::TouchingNeighbors, spacing),
        1.0
    );
    assert_eq!(
        summaries[0].feature(TouchingNeighborFeature::ContactFaces, spacing),
        1.0
    );
    assert_eq!(
        summaries[0].feature(TouchingNeighborFeature::ContactArea, spacing),
        summaries[0].contact_area(spacing)
    );
    assert_eq!(summaries[1].label, 2);
    assert_eq!(summaries[1].touching_neighbors, 1);

    let boundary = BoundaryMeasurements {
        label: 1,
        at: [1, 1, 0],
        count: 1,
        boundary_voxels: 1,
        boundary_faces: [0, 0, 1],
    };
    assert_eq!(
        contact_fraction_of_boundary(&expected[0], &boundary, PhysicalSpacing::unit()).unwrap(),
        Some(1.0)
    );
    assert_eq!(
        contact_fraction_of_boundary(
            &expected[0],
            &BoundaryMeasurements {
                label: 3,
                ..boundary
            },
            PhysicalSpacing::unit()
        )
        .unwrap(),
        None
    );
    assert!(summarize_touching_neighbors(&[ContactMeasurements {
        labels: [2, 1],
        ..expected[0]
    }])
    .is_err());
}

#[test]
fn touching_neighbor_rows_have_a_canonical_schema_and_collector() {
    let rows = summarize_touching_neighbors(&expected_contacts()).unwrap();
    let encoded = encode_touching_neighbor_measurements(&rows).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, touching_neighbor_measurement_schema());
    assert_eq!(schema.columns()[0].name(), "label");
    assert_eq!(schema.columns()[1].name(), "touching_neighbors");
    assert_eq!(schema.columns()[5].name(), "contact_faces_2");

    let env = ArrayEnvironment::new(contact_labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("touching.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("touching.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_touching_neighbor_measurements(&env, "touching.rows", 0, VOLUME).unwrap();
    assert_eq!(got, rows);

    assert!(encode_touching_neighbor_measurements(&[NeighborSummary {
        label: 0,
        touching_neighbors: 0,
        contact_faces: 0,
        contact_faces_by_axis: [0, 0, 0],
    }])
    .is_err());
    assert!(encode_touching_neighbor_measurements(&[NeighborSummary {
        label: 1,
        touching_neighbors: 2,
        contact_faces: 1,
        contact_faces_by_axis: [0, 0, 1],
    }])
    .is_err());
}

#[test]
fn measurement_builder_runs_planned_touching_neighbor_summaries() {
    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    let expected = summarize_touching_neighbors(&expected_contacts()).unwrap();

    for block in [[6, 4, 3], [2, 2, 1]] {
        let plan = Measurements::for_labels(0usize)
            .touching_neighbor_summary(spacing)
            .stream("objects")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.boundary_stream(), None);
        assert_eq!(plan.boundary_rows_phase(), None);
        assert!(plan.contact_rows().is_some());
        assert_eq!(plan.touching_neighbor_spacing(), Some(spacing));
        assert_eq!(plan.touching_neighbor_rows_phase(), Some(3));
        let rows = plan.touching_neighbor_rows_with_contract().unwrap();
        assert_eq!(rows.stream(), plan.touching_neighbor_stream().unwrap());
        assert_eq!(rows.phase(), plan.touching_neighbor_rows_phase().unwrap());
        assert_eq!(rows.contract(), spacing);

        let env = ArrayEnvironment::new(contact_labels(), plan.decomposition.n_phases(), [2, 2, 2])
            .unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());
        execute_phases(
            "builder touching-neighbor summaries",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got = collect_touching_neighbor_rows_with_contract(&env, &rows, VOLUME).unwrap();
        assert_eq!(got, expected, "block {block:?}");
        assert_eq!(got[0].contact_area(spacing), 6.0);
    }
}

#[test]
fn shape_boundary_metrics_are_derived_after_merge() {
    let shape = ShapeMeasurements {
        label: 4,
        count: 8,
        centroid: None,
        bbox_min: [0; 3],
        bbox_max: [2; 3],
        bbox_extent: [2; 3],
        bbox_volume: 8,
        bbox_fill_fraction: Some(1.0),
        equivalent_sphere_radius: equivalent_sphere_radius(8),
        equivalent_sphere_diameter: equivalent_sphere_diameter(8),
        principal_axis_lengths: None,
        orientation: None,
        orientation_yx: None,
        eccentricity: None,
    };
    let boundary = BoundaryMeasurements {
        label: 4,
        at: [0; 3],
        count: 8,
        boundary_voxels: 8,
        boundary_faces: [8, 8, 8],
    };
    let metrics = shape_boundary_measurements(&shape, &boundary, PhysicalSpacing::unit())
        .unwrap()
        .unwrap();
    assert_eq!(metrics.label, 4);
    assert_eq!(metrics.physical_volume, 8.0);
    assert_eq!(metrics.surface_area, 24.0);
    assert_eq!(metrics.surface_to_volume_ratio, 3.0);
    assert_eq!(ShapeBoundaryFeature::ALL.len(), 7);
    assert_eq!(
        ShapeBoundaryFeature::SurfaceToVolumeRatio.column_name(),
        "surface_to_volume_ratio"
    );
    assert_eq!(
        metrics.feature(ShapeBoundaryFeature::PhysicalVolume),
        metrics.physical_volume
    );
    assert_eq!(
        metrics.feature(ShapeBoundaryFeature::SurfaceArea),
        metrics.surface_area
    );
    assert_eq!(
        metrics.feature(ShapeBoundaryFeature::SurfaceToVolumeRatio),
        metrics.surface_to_volume_ratio
    );
    assert_eq!(
        metrics.feature(ShapeBoundaryFeature::IsoperimetricQuotient),
        metrics.isoperimetric_quotient
    );
    assert_eq!(metrics.form_factor, metrics.isoperimetric_quotient);
    assert!((metrics.compactness * metrics.form_factor - 1.0).abs() < 1.0e-12);
    assert!(
        (metrics.isoperimetric_quotient - metrics.sphericity.powi(3)).abs() < 1.0e-12,
        "{metrics:?}"
    );

    assert_eq!(
        shape_boundary_measurements(
            &shape,
            &BoundaryMeasurements {
                label: 5,
                ..boundary
            },
            PhysicalSpacing::unit()
        )
        .unwrap(),
        None
    );
    assert!(shape_boundary_measurements(
        &shape,
        &BoundaryMeasurements {
            count: 7,
            ..boundary
        },
        PhysicalSpacing::unit(),
    )
    .is_err());
    assert!(shape_boundary_measurements(
        &shape,
        &BoundaryMeasurements {
            boundary_faces: [0; 3],
            ..boundary
        },
        PhysicalSpacing::unit(),
    )
    .is_err());
}

#[test]
fn approximate_distribution_requests_are_named_at_construction() {
    let explicit = ApproxDistributionSet::new(16, -1.0, 7.0).unwrap();
    let compatible = DistributionSet::new(16, -1.0, 7.0).unwrap();
    assert_eq!(explicit, compatible);
    assert_eq!(explicit.mode(), ApproxMode::<0>);

    let intensity = IntensitySet::approx_distribution_set(explicit).unwrap();
    assert_eq!(intensity.distribution_set(), Some(explicit));
    assert_eq!(
        IntensitySet::approx_distribution(explicit.bins(), explicit.min(), explicit.max())
            .unwrap()
            .distribution_set(),
        Some(explicit)
    );
    assert_eq!(
        IntensitySet::standard()
            .with_approx_distribution_set(ApproxDistributionSet::new(8, 0.0, 8.0).unwrap())
            .unwrap()
            .distribution_set()
            .unwrap()
            .bins(),
        8
    );
    assert!(ApproxDistributionSet::new(0, 0.0, 1.0).is_err());
    assert!(IntensitySet::approx_distribution(0, 0.0, 1.0).is_err());
}

#[test]
fn distribution_percentiles_are_validated_selectors() {
    assert!(DistributionPercentile::new(f64::NAN).is_err());
    assert!(DistributionPercentile::new(-0.1).is_err());
    assert!(DistributionPercentile::new(1.1).is_err());

    let percentile = DistributionPercentile::new(0.75).unwrap();
    assert_eq!(percentile.get(), 0.75);
    assert_eq!(percentile.column_name(), "percentile_0.750000");
    assert_eq!(DistributionPercentile::MIN.get(), 0.0);
    assert_eq!(DistributionPercentile::MEDIAN.get(), 0.5);
    assert_eq!(DistributionPercentile::MAX.get(), 1.0);
    assert_eq!(
        DistributionFeature::Percentile(percentile).column_name(),
        "percentile_0.750000"
    );
    assert_eq!(
        DistributionFeature::percentile(0.75).unwrap(),
        DistributionFeature::Percentile(percentile)
    );
}

#[test]
fn raw_selector_helpers_lower_through_validated_keys() {
    let distribution = DistributionMeasurements {
        label: 1,
        at: [0, 0, 0],
        count: 4,
        nonfinite: 0,
        underflow: 0,
        overflow: 0,
        set: DistributionSet::new(4, 0.0, 4.0).unwrap(),
        bins: vec![1, 1, 1, 1],
    };
    let percentile = DistributionPercentile::new(0.5).unwrap();
    assert_eq!(
        distribution.percentile(0.5).unwrap(),
        distribution.percentile_key(percentile).unwrap()
    );
    assert_eq!(
        distribution
            .feature(DistributionFeature::Percentile(percentile))
            .unwrap(),
        distribution.percentile_key(percentile).unwrap()
    );
    assert!(distribution.percentile(f64::NAN).is_err());

    let exact = ExactDistributionMeasurements {
        label: 2,
        at: [0, 0, 0],
        count: 4,
        nonfinite: 0,
        values: vec![1.0, 2.0, 4.0, 8.0],
    };
    assert_eq!(
        exact.percentile(0.5).unwrap(),
        exact.percentile_key(percentile).unwrap()
    );
    assert_eq!(
        exact
            .feature(DistributionFeature::Percentile(percentile))
            .unwrap(),
        exact.percentile_key(percentile).unwrap()
    );
    assert!(exact.percentile(f64::NAN).is_err());

    let granularity = blockflow::ops::GranularityMeasurements {
        label: 3,
        count: 8,
        finite_count: 8,
        nonfinite: 0,
        original_sum: 100.0,
        radii: vec![1, 2, 4],
        opened_sums: vec![80.0, 50.0, 50.0],
    };
    let radius = GranularityRadius::new(2).unwrap();
    assert_eq!(
        granularity.survival_fraction(2),
        granularity.survival_fraction_key(radius)
    );
    assert_eq!(
        granularity.loss_fraction(2),
        granularity.loss_fraction_key(radius)
    );
    assert_eq!(
        granularity.differential_loss_fraction(2),
        granularity.differential_loss_fraction_key(radius)
    );
    assert_eq!(granularity.survival_fraction(0), None);

    let hu = ObjectHuMomentsMeasurements {
        label: 4,
        projected_count: 3,
        projection_axis: ProjectionAxis::Z,
        centroid: [1.0, 2.0],
        hu: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    };
    let hu_index = HuMomentIndex::new(6).unwrap();
    assert_eq!(hu.hu(6), hu.hu_key(hu_index));
    assert_eq!(
        hu.feature(ObjectHuMomentFeature::Hu(hu_index)),
        hu.hu_key(hu_index)
    );
    assert_eq!(hu.hu(7), None);

    let weighted_hu = ObjectWeightedHuMomentsMeasurements {
        label: 5,
        projected_count: 3,
        finite_weight_count: 3,
        nonfinite_weight_count: 0,
        projection_axis: ProjectionAxis::Z,
        weight_sum: 6.0,
        centroid: Some([1.0, 2.0]),
        hu: Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    };
    assert_eq!(weighted_hu.hu(6), weighted_hu.hu_key(hu_index));
    assert_eq!(
        weighted_hu.feature(ObjectWeightedHuMomentFeature::Hu(hu_index)),
        weighted_hu.hu_key(hu_index)
    );
    assert_eq!(weighted_hu.hu(7), None);
}

#[test]
fn exact_distribution_requests_are_named_and_validate_row_width() {
    assert!(ExactDistributionSet::new(0).is_err());
    let too_wide = ExactDistributionSet::new(usize::MAX)
        .unwrap_err()
        .to_string();
    assert!(too_wide.contains("row width overflow"));
    let set = ExactDistributionSet::new(32).unwrap();
    assert_eq!(set.max_values(), 32);

    assert!(ExactDistributionOp::new(
        "exact distribution",
        0usize,
        ImageId::supplied(0),
        0,
        "exact.rows",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ExactDistributionOp::new_set(
        "exact distribution",
        0usize,
        ImageId::supplied(0),
        set,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ExactDistributionTallyOp::new_set(
        "exact distribution partials",
        0usize,
        ImageId::supplied(0),
        set,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(MergeExactDistributionOp::new(
        "merge exact distribution",
        "",
        0,
        [1, 1, 1],
        1,
        "exact.rows",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(MergeExactDistributionOp::new(
        "merge exact distribution",
        "exact.partials",
        0,
        [1, 1, 1],
        0,
        "exact.rows",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(MergeExactDistributionOp::new_set(
        "merge exact distribution",
        "exact.partials",
        0,
        [1, 0, 1],
        set,
        "exact.rows",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(MergeExactDistributionOp::new_set(
        "merge exact distribution",
        "exact.partials",
        0,
        [1, 1, 1],
        set,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());

    let op = ExactDistributionOp::new_set(
        "exact distribution",
        0usize,
        ImageId::supplied(0),
        set,
        "exact.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(op.max_values(), 32);

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    let error = collect_exact_distribution_measurements(&env, "exact.rows", 0, VOLUME, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("max_values must be greater than zero"));
}

#[test]
fn object_moment_order_requests_are_named_and_validate_schema_width() {
    let moment0 = ObjectMoment3dSet::new(0).unwrap();
    let moment6 = ObjectMoment3dSet::new(6).unwrap();
    assert_eq!(moment0.max_order(), 0);
    assert_eq!(moment6.max_order(), 6);
    assert!(ObjectMoment3dSet::new(7).is_err());

    let zernike0 = ObjectZernikeMomentSet::new(0).unwrap();
    let zernike12 = ObjectZernikeMomentSet::new(12).unwrap();
    assert_eq!(zernike0.max_order(), 0);
    assert_eq!(zernike12.max_order(), 12);
    assert!(ObjectZernikeMomentSet::new(13).is_err());

    let moment_op = ObjectMoment3dOp::new_set(
        "measure 3D moments",
        0usize,
        PhysicalSpacing::unit(),
        moment6,
        "moment.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(moment_op.max_order(), 6);
    MergeObjectMoment3dOp::new_set(
        "merge 3D moments",
        "moment.points",
        0,
        [1, 1, 1],
        PhysicalSpacing::unit(),
        moment0,
        "moment.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();

    let zernike_op = ObjectZernikeMomentsOp::new_set(
        "measure Zernike moments",
        0usize,
        ProjectionAxis::Z,
        zernike12,
        "zernike.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(zernike_op.max_order(), 12);
    MergeObjectZernikeMomentsOp::new_set(
        "merge Zernike moments",
        "zernike.points",
        0,
        [1, 1, 1],
        ProjectionAxis::Z,
        zernike0,
        "zernike.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    let error = collect_object_3d_moment_measurements(&env, "moment3d.rows", 0, VOLUME, 7)
        .unwrap_err()
        .to_string();
    assert!(error.contains("max_order must be <= 6"));

    let error = collect_object_zernike_moment_measurements(&env, "zernike.rows", 0, VOLUME, 13)
        .unwrap_err()
        .to_string();
    assert!(error.contains("max_order must be <= 12"));
}

#[test]
fn projected_convex_requests_are_named_and_validate_hull_width() {
    assert!(ProjectedConvexSet::new(0).is_err());
    let too_wide = ProjectedConvexSet::new(usize::MAX).unwrap_err().to_string();
    assert!(too_wide.contains("row width overflow"));
    let set = ProjectedConvexSet::new(16).unwrap();
    assert_eq!(set.max_hull_vertices(), 16);

    assert!(ObjectProjectedConvexOp::new(
        "measure projected convex geometry",
        0usize,
        PhysicalSpacing::unit(),
        ProjectionAxis::Z,
        0,
        "projected-convex.rows",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(MergeObjectProjectedConvexOp::new(
        "merge projected convex geometry",
        "projected-convex.points",
        0,
        [1, 1, 1],
        PhysicalSpacing::unit(),
        ProjectionAxis::Z,
        0,
        "projected-convex.rows",
        Lifecycle::DeleteOnExit,
    )
    .is_err());

    let op = ObjectProjectedConvexOp::new_set(
        "measure projected convex geometry",
        0usize,
        PhysicalSpacing::unit(),
        ProjectionAxis::Z,
        set,
        "projected-convex.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(op.max_hull_vertices(), 16);

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    let error =
        collect_object_projected_convex_measurements(&env, "projected-convex.rows", 0, VOLUME, 0)
            .unwrap_err()
            .to_string();
    assert!(error.contains("max_hull_vertices must be greater than zero"));
}

#[test]
fn standalone_object_ops_validate_output_streams_at_construction() {
    let moment_set = ObjectMoment3dSet::new(1).unwrap();
    let projected_set = ProjectedConvexSet::new(4).unwrap();
    let zernike_set = ObjectZernikeMomentSet::new(2).unwrap();
    assert!(ObjectGeometryOp::new(
        "measure geometry",
        0usize,
        PhysicalSpacing::unit(),
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectMoment3dOp::new_set(
        "measure 3D moments",
        0usize,
        PhysicalSpacing::unit(),
        moment_set,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(EnclosingSphereOp::new(
        "measure enclosing sphere",
        0usize,
        PhysicalSpacing::unit(),
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectConvexHullOp::new(
        "measure convex hull",
        0usize,
        PhysicalSpacing::unit(),
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectProjectedConvexOp::new_set(
        "measure projected convex",
        0usize,
        PhysicalSpacing::unit(),
        ProjectionAxis::Z,
        projected_set,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectHuMomentsOp::new(
        "measure Hu moments",
        0usize,
        ProjectionAxis::Z,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectZernikeMomentsOp::new_set(
        "measure Zernike moments",
        0usize,
        ProjectionAxis::Z,
        zernike_set,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectWeightedHuMomentsOp::new(
        "measure weighted Hu moments",
        0usize,
        ImageId::supplied(0),
        ProjectionAxis::Z,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
}

#[test]
fn distribution_measurement_uses_fixed_bins_and_is_decomposition_invariant() {
    let set = DistributionSet::new(60, 0.0, 60.0).unwrap();
    let mut expected_bins = vec![0; 60];
    for value in [0usize, 10, 20, 30, 40, 50] {
        expected_bins[value] += 1;
    }

    for block in [[6, 4, 3], [3, 2, 2]] {
        let plan = Measurements::for_labels(0usize)
            .intensity(
                IntensityImage::<0>::new(1usize),
                IntensitySet::approx_distribution(set.bins(), set.min(), set.max()).unwrap(),
            )
            .stream("objects")
            .build(base(block))
            .unwrap();
        assert_eq!(plan.rows_phase, None);
        assert_eq!(plan.distribution_rows_phase(), Some(2));
        let distribution_rows = plan.distribution_rows().unwrap();
        assert_eq!(distribution_rows.phase(), 2);
        assert_eq!(
            distribution_rows.stream(),
            plan.distribution_stream().unwrap()
        );
        let distribution_phase = plan.distribution_rows_phase().unwrap();
        let distribution_stream = plan.distribution_stream().unwrap().to_string();
        let env =
            ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
        let mut work = vec![PhaseWork::Pixels];
        work.extend(plan.phase_work());

        execute_phases(
            "measurements",
            &workflow(),
            &plan.decomposition,
            &Hints::default(),
            &env,
            &[],
            &work,
        )
        .unwrap();

        let got = collect_distribution_measurements(
            &env,
            &distribution_stream,
            distribution_phase,
            VOLUME,
            set,
        )
        .unwrap();
        assert_eq!(got.len(), 2);
        let first = &got[0];
        assert_eq!(first.label, 1);
        assert_eq!(first.count, 6);
        assert_eq!(first.nonfinite, 0);
        assert_eq!(first.underflow, 0);
        assert_eq!(first.overflow, 0);
        assert_eq!(first.bins, expected_bins, "block {block:?}");
        assert_eq!(first.in_range_count(), 6);
        assert_eq!(first.median(), Some(30.5));
        assert_eq!(first.quartiles(), Some([10.5, 30.5, 40.5]));
        assert_eq!(
            first
                .percentile_key(DistributionPercentile::new(0.25).unwrap())
                .unwrap(),
            Some(10.5)
        );
        assert!(first.std_dev().unwrap() > 0.0);
        assert_eq!(first.mad(), Some(10.0));
        assert_eq!(DistributionFeature::SUMMARIES.len(), 7);
        assert_eq!(
            DistributionFeature::percentile(0.5).unwrap().column_name(),
            "percentile_0.500000"
        );
        assert!(DistributionFeature::percentile(f64::NAN).is_err());
        assert_eq!(
            first.feature(DistributionFeature::InRangeCount).unwrap(),
            Some(6.0)
        );
        assert_eq!(
            first
                .feature(DistributionFeature::percentile(0.25).unwrap())
                .unwrap(),
            Some(10.5)
        );
        assert_eq!(
            first.feature(DistributionFeature::Median).unwrap(),
            first.median()
        );
        assert_eq!(
            first.feature(DistributionFeature::Quartile3).unwrap(),
            Some(40.5)
        );
        assert_eq!(
            first.feature(DistributionFeature::Mad).unwrap(),
            first.mad()
        );
    }
}

#[test]
fn auto_distribution_discovers_global_labelled_range() {
    let mut labels = Array3::<f64>::zeros((1, 2, 4));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 0, 2]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[0, 1, 1]] = 5.0;

    let mut values = Array3::<f64>::zeros((1, 2, 4));
    values[[0, 0, 0]] = -1.0;
    values[[0, 0, 1]] = 0.0;
    values[[0, 0, 2]] = 3.0;
    values[[0, 1, 0]] = 1.0;
    values[[0, 1, 1]] = f64::NAN;
    values[[0, 1, 2]] = 100.0;

    let rows = auto_distribution_measurements(labels.view(), values.view(), 4).unwrap();
    assert_eq!(rows.len(), 2);

    let first = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(first.set.min(), -1.0);
    assert_eq!(first.set.max(), 3.0);
    assert_eq!(first.count, 3);
    assert_eq!(first.nonfinite, 0);
    assert_eq!(first.underflow, 0);
    assert_eq!(first.overflow, 0);
    assert_eq!(first.bins, vec![1, 1, 0, 1]);
    assert_eq!(first.median(), Some(0.5));

    let second = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(second.count, 2);
    assert_eq!(second.nonfinite, 1);
    assert_eq!(second.bins, vec![0, 0, 1, 0]);
}

#[test]
fn auto_distribution_validates_inputs_and_constant_ranges() {
    let mut labels = Array3::<f64>::zeros((1, 1, 2));
    labels[[0, 0, 0]] = 3.0;
    labels[[0, 0, 1]] = 3.0;
    let values = Array3::<f64>::from_elem((1, 1, 2), 5.0);

    let rows = auto_distribution_measurements(labels.view(), values.view(), 2).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].set.min(), 4.5);
    assert_eq!(rows[0].set.max(), 5.5);
    assert_eq!(rows[0].bins, vec![0, 2]);

    assert!(auto_distribution_measurements(labels.view(), values.view(), 0).is_err());
    let wrong_shape = Array3::<f64>::zeros((1, 1, 3));
    assert!(auto_distribution_measurements(labels.view(), wrong_shape.view(), 2).is_err());

    let mut invalid_labels = labels.clone();
    invalid_labels[[0, 0, 0]] = 1.5;
    assert!(auto_distribution_measurements(invalid_labels.view(), values.view(), 2).is_err());

    let mut nonfinite_values = Array3::<f64>::from_elem((1, 1, 2), f64::NAN);
    let rows = auto_distribution_measurements(labels.view(), nonfinite_values.view(), 2).unwrap();
    assert!(rows.is_empty());
    nonfinite_values[[0, 0, 0]] = f64::INFINITY;
    let rows = auto_distribution_measurements(labels.view(), nonfinite_values.view(), 2).unwrap();
    assert!(rows.is_empty());
}

#[test]
fn auto_distribution_rows_carry_range_schema_and_collector() {
    let rows = vec![
        DistributionMeasurements {
            label: 2,
            at: [0, 1, 2],
            count: 4,
            nonfinite: 0,
            underflow: 0,
            overflow: 0,
            set: DistributionSet::new(3, -1.0, 2.0).unwrap(),
            bins: vec![1, 2, 1],
        },
        DistributionMeasurements {
            label: 5,
            at: [1, 0, 0],
            count: 2,
            nonfinite: 1,
            underflow: 0,
            overflow: 0,
            set: DistributionSet::new(3, -1.0, 2.0).unwrap(),
            bins: vec![0, 1, 0],
        },
    ];
    let set = AutoDistributionSet::new(3).unwrap();
    let encoded = encode_auto_distribution_measurements_set(&rows, set).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, auto_distribution_measurement_schema(3).unwrap());
    assert_eq!(schema, auto_distribution_measurement_schema_set(set));
    assert_eq!(schema.columns()[5].name(), "range_min");
    assert_eq!(schema.columns()[6].name(), "range_max");
    assert_eq!(schema.columns()[9].name(), "bin_2");
    assert!(auto_distribution_measurement_schema(0).is_err());
    assert!(encode_auto_distribution_measurements(&rows, 0).is_err());

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("auto-distribution.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("auto-distribution.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_auto_distribution_measurements(&env, "auto-distribution.rows", 0, VOLUME, 3)
        .unwrap();
    assert_eq!(got, rows);

    let mut wrong_width = rows[0].clone();
    wrong_width.bins.pop();
    assert!(encode_auto_distribution_measurements(&[wrong_width], 3).is_err());
}

fn planned_auto_distribution(block: [usize; 3]) -> Vec<DistributionMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let set = AutoDistributionSet::new(8).unwrap();
    let op = AutoDistributionOp::new_set(
        "measure auto distribution",
        0usize,
        ImageId::supplied(0),
        set,
        "auto-distribution.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let values = coordinate_values();
    let env =
        ArrayEnvironment::with_inputs(labels(), vec![values], &decomposition, [2, 2, 2]).unwrap();
    execute_phases(
        "planned auto distribution",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_auto_distribution_measurements(
        &env,
        "auto-distribution.planned.rows",
        1,
        VOLUME,
        set.bins(),
    )
    .unwrap()
}

#[test]
fn planned_auto_distribution_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_auto_distribution(VOLUME);
    let split = planned_auto_distribution([2, 2, 2]);
    assert_eq!(coarse, split);

    let values = coordinate_values().view::<f64>().unwrap().to_owned();
    let reference =
        auto_distribution_measurements(labels().view::<f64>().unwrap(), values.view(), 8).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_auto_distribution() {
    let values = coordinate_values();
    let set = AutoDistributionSet::new(8).unwrap();
    let expected = auto_distribution_measurements(
        labels().view::<f64>().unwrap(),
        values.view::<f64>().unwrap(),
        set.bins(),
    )
    .unwrap();
    let plan = Measurements::for_labels(0usize)
        .auto_distribution_set(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            set,
        )
        .unwrap()
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.auto_distribution_rows_phase(0), Some(1));
    assert_eq!(plan.auto_distribution_set(0), Some(set));
    let rows = plan.auto_distribution_rows_with_set(0).unwrap();
    assert_eq!(rows.stream(), plan.auto_distribution_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.auto_distribution_rows_phase(0).unwrap());
    assert_eq!(rows.contract(), set);

    let env = ArrayEnvironment::with_inputs(labels(), vec![values], &plan.decomposition, [2, 2, 2])
        .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder auto distribution",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_auto_distribution_rows_with_set(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn auto_distribution_plan_has_simulator_visible_full_volume_cost() {
    let auto_set = AutoDistributionSet::new(8).unwrap();
    let auto = Measurements::for_labels(0usize)
        .auto_distribution_set(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            auto_set,
        )
        .unwrap()
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    let fixed_set = ApproxDistributionSet::new(8, 0.0, 60.0).unwrap();
    let fixed = Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensitySet::approx_distribution_set(fixed_set).unwrap(),
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();

    assert_eq!(
        auto.decomposition.n_phases(),
        2,
        "automatic distribution is currently one full-volume row phase after the source phase"
    );
    assert_eq!(
        fixed.decomposition.n_phases(),
        3,
        "fixed-range distribution already uses compact tally plus merge phases"
    );

    let simulate_plan = |decomposition: &Decomposition, mut work: Vec<PhaseWork<'_>>| {
        work.insert(0, PhaseWork::Pixels);
        Run::new(decomposition, &work)
            .machine(Machine {
                workers: 4,
                ..Machine::default()
            })
            .rates(Rates::default())
            .go(&mut PlanOrder)
            .unwrap()
    };
    let auto_outcome = simulate_plan(&auto.decomposition, auto.phase_work());
    let fixed_outcome = simulate_plan(&fixed.decomposition, fixed.phase_work());

    assert!(
        auto_outcome.tasks_run < fixed_outcome.tasks_run,
        "the current automatic-range path is one simulator-visible full-volume phase, not the \
         compact two-phase histogram path"
    );
    assert!(
        auto_outcome.fetched_bytes > 0 && fixed_outcome.fetched_bytes > 0,
        "the simulator must see source reads for both distribution plans before it can justify \
         replacing automatic range discovery"
    );
}

#[test]
fn exact_distribution_stores_sorted_values_for_order_statistics() {
    let mut labels = Array3::<f64>::zeros((1, 2, 4));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 0, 2]] = 2.0;
    labels[[0, 0, 3]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[0, 1, 1]] = 5.0;

    let mut values = Array3::<f64>::zeros((1, 2, 4));
    values[[0, 0, 0]] = 7.0;
    values[[0, 0, 1]] = 1.0;
    values[[0, 0, 2]] = 9.0;
    values[[0, 0, 3]] = f64::NAN;
    values[[0, 1, 0]] = f64::INFINITY;
    values[[0, 1, 1]] = f64::NEG_INFINITY;

    let rows = exact_distribution_measurements(labels.view(), values.view()).unwrap();
    assert_eq!(rows.len(), 2);

    let exact = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(exact.count, 4);
    assert_eq!(exact.nonfinite, 1);
    assert_eq!(exact.finite_count().unwrap(), 3);
    assert_eq!(exact.values, vec![1.0, 7.0, 9.0]);
    assert_eq!(exact.percentile(0.0).unwrap(), Some(1.0));
    assert_eq!(exact.percentile(0.5).unwrap(), Some(7.0));
    assert_eq!(exact.percentile(1.0).unwrap(), Some(9.0));
    assert_eq!(
        exact
            .percentile_key(DistributionPercentile::new(0.75).unwrap())
            .unwrap(),
        Some(9.0)
    );
    assert_eq!(exact.median(), Some(7.0));
    assert_eq!(exact.quartiles(), Some([7.0, 7.0, 9.0]));
    assert_eq!(exact.mean(), Some(17.0 / 3.0));
    assert!((exact.std_dev().unwrap() - (104.0f64 / 9.0).sqrt()).abs() < 1.0e-12);
    assert_eq!(exact.mad(), Some(2.0));
    assert_eq!(
        exact.feature(DistributionFeature::InRangeCount).unwrap(),
        Some(3.0)
    );
    assert_eq!(
        exact
            .feature(DistributionFeature::percentile(0.75).unwrap())
            .unwrap(),
        Some(9.0)
    );
    assert_eq!(
        exact.feature(DistributionFeature::Quartile1).unwrap(),
        Some(7.0)
    );
    assert_eq!(
        exact.feature(DistributionFeature::Mean).unwrap(),
        exact.mean()
    );
    assert_eq!(
        exact.feature(DistributionFeature::StdDev).unwrap(),
        exact.std_dev()
    );
    assert!(exact.percentile(f64::NAN).is_err());

    let nonfinite = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(nonfinite.count, 2);
    assert_eq!(nonfinite.nonfinite, 2);
    assert_eq!(nonfinite.values, Vec::<f64>::new());
    assert_eq!(nonfinite.median(), None);
    assert_eq!(nonfinite.quartiles(), None);
    assert_eq!(nonfinite.mad(), None);
    assert_eq!(
        nonfinite.feature(DistributionFeature::Median).unwrap(),
        None
    );
}

#[test]
fn exact_distribution_validates_inputs() {
    let mut labels = Array3::<f64>::zeros((1, 1, 2));
    labels[[0, 0, 0]] = 2.0;
    let values = Array3::<f64>::zeros((1, 1, 2));
    let wrong_shape = Array3::<f64>::zeros((1, 1, 3));
    assert!(exact_distribution_measurements(labels.view(), wrong_shape.view()).is_err());

    labels[[0, 0, 1]] = 1.5;
    assert!(exact_distribution_measurements(labels.view(), values.view()).is_err());
}

#[test]
fn exact_distribution_rows_have_bounded_schema_and_collector() {
    let rows = vec![
        ExactDistributionMeasurements {
            label: 2,
            at: [0, 1, 2],
            count: 4,
            nonfinite: 1,
            values: vec![1.0, 7.0, 9.0],
        },
        ExactDistributionMeasurements {
            label: 5,
            at: [1, 0, 0],
            count: 2,
            nonfinite: 2,
            values: Vec::new(),
        },
    ];
    let set = ExactDistributionSet::new(3).unwrap();
    let encoded = encode_exact_distribution_measurements_set(&rows, set).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, exact_distribution_measurement_schema(3).unwrap());
    assert_eq!(schema, exact_distribution_measurement_schema_set(set));
    assert!(exact_distribution_measurement_schema(0).is_err());
    assert_eq!(schema.columns()[3].name(), "value_count");
    assert_eq!(schema.columns()[6].name(), "value_2");
    assert!(encode_exact_distribution_measurements(&rows, 0).is_err());

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("exact-distribution.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("exact-distribution.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got =
        collect_exact_distribution_measurements(&env, "exact-distribution.rows", 0, VOLUME, 3)
            .unwrap();
    assert_eq!(got, rows);

    assert!(encode_exact_distribution_measurements(&rows, 2).is_err());

    let mut unsorted = rows[0].clone();
    unsorted.values = vec![7.0, 1.0, 9.0];
    assert!(encode_exact_distribution_measurements(&[unsorted], 3).is_err());

    let mut wrong_count = rows[0].clone();
    wrong_count.count = 5;
    assert!(encode_exact_distribution_measurements(&[wrong_count], 3).is_err());
}

fn planned_exact_distribution(block: [usize; 3]) -> Vec<ExactDistributionMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let set = ExactDistributionSet::new(32).unwrap();
    let tally = ExactDistributionTallyOp::new_set(
        "measure exact distribution partials",
        0usize,
        ImageId::supplied(0),
        set,
        "exact-distribution.planned.partials",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&tally, grid.clone()).unwrap());
    let merge = MergeExactDistributionOp::new_set(
        "merge exact distribution",
        "exact-distribution.planned.partials",
        1,
        grid.blocks_per_axis(),
        set,
        "exact-distribution.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    decomposition
        .phases
        .push(fragment_phase(&merge, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let values = coordinate_values();
    let env =
        ArrayEnvironment::with_inputs(labels(), vec![values], &decomposition, [2, 2, 2]).unwrap();
    execute_phases(
        "planned exact distribution",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&tally),
            PhaseWork::Fragments(&merge),
        ],
    )
    .unwrap();
    collect_exact_distribution_measurements_set(
        &env,
        "exact-distribution.planned.rows",
        2,
        VOLUME,
        ExactDistributionSet::new(32).unwrap(),
    )
    .unwrap()
}

#[test]
fn planned_exact_distribution_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_exact_distribution(VOLUME);
    let split = planned_exact_distribution([2, 2, 2]);
    assert_eq!(coarse, split);

    let values = coordinate_values().view::<f64>().unwrap().to_owned();
    let reference =
        exact_distribution_measurements(labels().view::<f64>().unwrap(), values.view()).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_exact_distribution() {
    let values = coordinate_values();
    let expected = exact_distribution_measurements(
        labels().view::<f64>().unwrap(),
        values.view::<f64>().unwrap(),
    )
    .unwrap();
    let plan = Measurements::for_labels(0usize)
        .exact_distribution(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            32,
        )
        .unwrap()
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.exact_distribution_rows_phase(0), Some(2));
    assert_eq!(plan.exact_distribution_max_values(0), Some(32));
    assert_eq!(
        plan.exact_distribution_set(0),
        Some(ExactDistributionSet::new(32).unwrap())
    );
    let rows = plan.exact_distribution_rows_with_set(0).unwrap();
    assert_eq!(rows.stream(), plan.exact_distribution_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.exact_distribution_rows_phase(0).unwrap());
    assert_eq!(rows.contract(), plan.exact_distribution_set(0).unwrap());
    let stream = rows.stream().to_string();
    let phase = rows.phase();

    let env = ArrayEnvironment::with_inputs(labels(), vec![values], &plan.decomposition, [2, 2, 2])
        .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder exact distribution",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let rows = plan.exact_distribution_rows_with_set(0).unwrap();
    assert_eq!(rows.stream(), stream);
    assert_eq!(rows.phase(), phase);
    let got = collect_exact_distribution_rows_with_set(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn granularity_spectrum_reports_opened_intensity_survival() {
    let mut labels = Array3::<f64>::zeros((5, 5, 5));
    let mut values = Array3::<f64>::zeros((5, 5, 5));
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                labels[[z, y, x]] = 2.0;
                values[[z, y, x]] = 2.0;
            }
        }
    }
    labels[[0, 0, 0]] = 5.0;
    values[[0, 0, 0]] = 10.0;
    labels[[4, 4, 4]] = 7.0;
    values[[4, 4, 4]] = f64::NAN;

    let set = GranularitySet::new(vec![1, 1])
        .unwrap()
        .with_shape(ElementShape::Box);
    let granularity =
        granularity_spectrum_measurements(labels.view(), values.view(), &set).unwrap();
    assert_eq!(granularity.len(), 3);

    let cube = granularity.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(cube.count, 27);
    assert_eq!(cube.finite_count, 27);
    assert_eq!(cube.nonfinite, 0);
    assert_eq!(cube.original_sum, 54.0);
    assert_eq!(cube.radii, vec![1]);
    assert_eq!(cube.opened_sums, vec![54.0]);
    assert_eq!(cube.survival_fractions(), Some(vec![1.0]));
    assert_eq!(cube.loss_fractions(), Some(vec![0.0]));
    assert_eq!(cube.differential_loss_fractions(), Some(vec![0.0]));
    assert_eq!(cube.total_loss_fraction(), Some(0.0));
    assert_eq!(cube.mean_survival_fraction(), Some(1.0));
    assert_eq!(cube.peak_loss_radius(), Some(1));

    let speck = granularity.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(speck.opened_sums, vec![0.0]);
    assert_eq!(speck.survival_fractions(), Some(vec![0.0]));
    assert_eq!(speck.loss_fractions(), Some(vec![1.0]));
    assert_eq!(speck.differential_loss_fractions(), Some(vec![1.0]));
    assert_eq!(speck.total_loss_fraction(), Some(1.0));
    assert_eq!(speck.mean_survival_fraction(), Some(0.0));
    assert_eq!(speck.peak_loss_radius(), Some(1));

    let nonfinite = granularity.iter().find(|row| row.label == 7).unwrap();
    assert_eq!(nonfinite.finite_count, 0);
    assert_eq!(nonfinite.nonfinite, 1);
    assert_eq!(nonfinite.survival_fractions(), None);
    assert_eq!(nonfinite.differential_loss_fractions(), None);
}

#[test]
fn granularity_spectrum_validates_radius_set_and_shapes() {
    assert!(GranularitySet::new(Vec::new()).is_err());
    assert!(GranularitySet::new(vec![0]).is_err());
    assert!(GranularityRadius::new(0).is_err());
    assert!(GranularityFeature::survival_fraction(0).is_err());
    assert!(GranularityFeature::loss_fraction(0).is_err());
    assert!(GranularityFeature::differential_loss_fraction(0).is_err());

    let labels = Array3::<f64>::zeros((1, 1, 1));
    let values = Array3::<f64>::zeros((1, 1, 2));
    let set = GranularitySet::new(vec![1]).unwrap();
    assert!(GranularitySet::from_radius_keys(Vec::new()).is_err());
    let keyed = GranularitySet::from_radius_keys(vec![
        GranularityRadius::new(3).unwrap(),
        GranularityRadius::new(1).unwrap(),
        GranularityRadius::new(3).unwrap(),
    ])
    .unwrap();
    assert_eq!(keyed.radii(), &[1, 3]);
    assert_eq!(
        keyed.radius_keys(),
        vec![
            GranularityRadius::new(1).unwrap(),
            GranularityRadius::new(3).unwrap()
        ]
    );
    assert_eq!(GranularityRadius::new(2).unwrap().get(), 2);
    assert_eq!(
        GranularityFeature::survival_fraction(2).unwrap(),
        GranularityFeature::SurvivalFraction(GranularityRadius::new(2).unwrap())
    );
    assert!(granularity_spectrum_measurements(labels.view(), values.view(), &set).is_err());
}

#[test]
fn granularity_postprocessing_summarizes_pattern_spectrum() {
    let row = blockflow::ops::GranularityMeasurements {
        label: 3,
        count: 8,
        finite_count: 8,
        nonfinite: 0,
        original_sum: 100.0,
        radii: vec![1, 2, 4],
        opened_sums: vec![80.0, 50.0, 50.0],
    };
    assert_eq!(row.survival_fractions(), Some(vec![0.8, 0.5, 0.5]));
    let losses = row.loss_fractions().unwrap();
    assert!((losses[0] - 0.2).abs() < 1.0e-12);
    assert_eq!(&losses[1..], &[0.5, 0.5]);
    assert_eq!(row.differential_loss_fractions(), Some(vec![0.2, 0.3, 0.0]));
    assert_eq!(row.total_loss_fraction(), Some(0.5));
    assert_eq!(row.initial_loss_fraction(), Some(0.2));
    assert_eq!(row.mean_survival_fraction(), Some(0.6));
    assert_eq!(row.survival_auc(), Some(3.8));
    assert_eq!(row.loss_auc(), Some(3.2));
    assert_eq!(row.peak_loss_radius(), Some(2));
    assert_eq!(GranularityFeature::SUMMARIES.len(), 6);
    assert_eq!(
        GranularityFeature::survival_fraction(2)
            .unwrap()
            .column_name(),
        "survival_fraction_r2"
    );
    assert_eq!(
        row.feature(GranularityFeature::survival_fraction(2).unwrap()),
        Some(0.5)
    );
    let radius_two = GranularityRadius::new(2).unwrap();
    assert_eq!(row.survival_fraction_key(radius_two), Some(0.5));
    assert_eq!(row.loss_fraction_key(radius_two), Some(0.5));
    assert_eq!(row.differential_loss_fraction_key(radius_two), Some(0.3));
    assert_eq!(row.survival_fraction(2), Some(0.5));
    assert_eq!(row.loss_fraction(2), Some(0.5));
    assert_eq!(row.differential_loss_fraction(2), Some(0.3));
    assert_eq!(row.survival_fraction(0), None);
    assert_eq!(
        row.feature(GranularityFeature::loss_fraction(2).unwrap()),
        Some(0.5)
    );
    assert_eq!(
        row.feature(GranularityFeature::differential_loss_fraction(2).unwrap()),
        Some(0.3)
    );
    assert_eq!(GranularityFeature::LossAuc.column_name(), "loss_auc");
    assert_eq!(row.feature(GranularityFeature::LossAuc), Some(3.2));
    assert_eq!(row.feature(GranularityFeature::PeakLossRadius), Some(2.0));
    assert_eq!(
        row.feature(GranularityFeature::survival_fraction(3).unwrap()),
        None
    );

    let invalid = blockflow::ops::GranularityMeasurements {
        radii: vec![1],
        opened_sums: vec![80.0, 50.0],
        ..row
    };
    assert_eq!(invalid.initial_loss_fraction(), None);
    assert_eq!(invalid.survival_auc(), None);
    assert_eq!(invalid.peak_loss_radius(), None);
    assert_eq!(invalid.feature(GranularityFeature::TotalLossFraction), None);
}

#[test]
fn granularity_rows_have_a_canonical_schema_and_collector() {
    let set = GranularitySet::new(vec![2, 1])
        .unwrap()
        .with_shape(ElementShape::Box);
    let rows = vec![
        blockflow::ops::GranularityMeasurements {
            label: 3,
            count: 8,
            finite_count: 8,
            nonfinite: 0,
            original_sum: 100.0,
            radii: vec![1, 2],
            opened_sums: vec![80.0, 50.0],
        },
        blockflow::ops::GranularityMeasurements {
            label: 4,
            count: 2,
            finite_count: 1,
            nonfinite: 1,
            original_sum: 7.0,
            radii: vec![1, 2],
            opened_sums: vec![3.0, 0.0],
        },
    ];
    let encoded = encode_granularity_measurements(&rows, &set).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, granularity_measurement_schema(&set));
    assert_eq!(
        schema.columns().last().expect("opened sum column").name(),
        "opened_sum_r2"
    );

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("granularity.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("granularity.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_granularity_measurements(&env, "granularity.rows", 0, VOLUME, &set).unwrap();
    assert_eq!(got, rows);

    let invalid = blockflow::ops::GranularityMeasurements {
        label: 3,
        count: 8,
        finite_count: 8,
        nonfinite: 0,
        original_sum: 100.0,
        radii: vec![1],
        opened_sums: vec![80.0],
    };
    assert!(encode_granularity_measurements(&[invalid], &set).is_err());
}

fn granularity_fixture() -> (Array3<f64>, Array3<f64>, GranularitySet) {
    let mut labels = Array3::<f64>::zeros((5, 5, 5));
    let mut values = Array3::<f64>::zeros((5, 5, 5));
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                labels[[z, y, x]] = 2.0;
                values[[z, y, x]] = 2.0;
            }
        }
    }
    labels[[0, 0, 0]] = 5.0;
    values[[0, 0, 0]] = 10.0;
    labels[[4, 4, 4]] = 7.0;
    values[[4, 4, 4]] = f64::NAN;
    let set = GranularitySet::new(vec![1])
        .unwrap()
        .with_shape(ElementShape::Box);
    (labels, values, set)
}

fn planned_granularity(block: [usize; 3]) -> Vec<blockflow::ops::GranularityMeasurements> {
    let (labels, values, set) = granularity_fixture();
    let mut decomposition = Decomposition {
        volume: [5, 5, 5],
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["values".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new([5, 5, 5], block).unwrap(),
        )],
        chain_reach: [0, 0, 0],
    };
    let grid = decomposition.phases[0].grid.clone();
    let op = GranularityOp::new(
        "measure granularity",
        0usize,
        ImageId::supplied(0),
        set.clone(),
        "granularity.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::with_inputs(
        labels.into(),
        vec![values.into()],
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    execute_phases(
        "planned granularity",
        &Workflow::new(Chain::op(CoordinateValues), [5, 5, 5], Dtype::F64),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_granularity_measurements(&env, "granularity.planned.rows", 1, [5, 5, 5], &set).unwrap()
}

#[test]
fn planned_granularity_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_granularity([5, 5, 5]);
    let split = planned_granularity([2, 2, 2]);
    assert_eq!(coarse, split);

    let (labels, values, set) = granularity_fixture();
    let reference = granularity_spectrum_measurements(labels.view(), values.view(), &set).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_granularity() {
    let labels = labels().view::<f64>().unwrap().to_owned();
    let (values, _) = colocalization_channels();
    let set = GranularitySet::new(vec![1])
        .unwrap()
        .with_shape(ElementShape::Box);
    let expected = granularity_spectrum_measurements(labels.view(), values.view(), &set).unwrap();

    let plan = Measurements::for_labels(0usize)
        .granularity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            set.clone(),
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.granularity_rows_phase(0), Some(1));
    assert_eq!(plan.granularity_set(0), Some(&set));
    let granularity_rows = plan.granularity_rows_with_set(0).unwrap();
    assert_eq!(
        granularity_rows.stream(),
        plan.granularity_stream(0).unwrap()
    );
    assert_eq!(
        granularity_rows.phase(),
        plan.granularity_rows_phase(0).unwrap()
    );
    assert_eq!(granularity_rows.contract_ref(), &set);
    let env = ArrayEnvironment::with_inputs(
        labels.into(),
        vec![values.into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder granularity",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_granularity_rows_with_set(&env, &granularity_rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn shared_boundary_radius_names_the_shared_boundary_approximation() {
    let mut labels = Array3::<f64>::zeros((5, 5, 5));
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                labels[[z, y, x]] = 7.0;
            }
        }
    }

    let params = DistanceParams::default();
    let field = shared_boundary_distance_field(labels.view(), &params).unwrap();
    assert_eq!(field[[1, 1, 1]], 0.0);
    assert_eq!(field[[2, 2, 2]], 1.0);

    let radii = shared_boundary_radius_measurements(labels.view(), &params).unwrap();
    assert_eq!(radii.len(), 1);
    assert_eq!(radii[0].label, 7);
    assert_eq!(radii[0].count, 27);
    assert_eq!(radii[0].finite_count, 27);
    assert_eq!(radii[0].max, Some(1.0));
    assert_eq!(radii[0].median, Some(0.0));
    assert_eq!(radii[0].mean, Some(1.0 / 27.0));

    labels[[0, 0, 0]] = 1.5;
    assert!(shared_boundary_radius_measurements(labels.view(), &params).is_err());
}

#[test]
fn exact_label_radius_uses_per_label_distance_fields() {
    let mut labels = Array3::<f64>::zeros((5, 5, 5));
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                labels[[z, y, x]] = 7.0;
            }
        }
    }

    let params = DistanceParams::default();
    let exact = exact_label_radius_measurements(labels.view(), &params).unwrap();
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].label, 7);
    assert_eq!(exact[0].count, 27);
    assert_eq!(exact[0].finite_count, 27);
    assert_eq!(exact[0].max, Some(2.0));
    assert_eq!(exact[0].median, Some(1.0));
    assert_eq!(exact[0].mean, Some(28.0 / 27.0));

    let shared = shared_boundary_radius_measurements(labels.view(), &params).unwrap();
    assert_eq!(shared[0].max, Some(1.0));

    labels[[0, 0, 0]] = 1.5;
    assert!(exact_label_radius_measurements(labels.view(), &params).is_err());
}

#[test]
fn radius_measurement_rows_have_a_canonical_schema_and_collectors() {
    let shared = vec![SharedBoundaryRadiusMeasurements {
        label: 7,
        count: 27,
        finite_count: 27,
        mean: Some(1.0 / 27.0),
        max: Some(1.0),
        median: Some(0.0),
    }];
    let encoded_shared = encode_shared_boundary_radius_measurements(&shared).unwrap();
    assert_eq!(
        encoded_schema(&encoded_shared).unwrap(),
        radius_measurement_schema()
    );

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("shared-radius.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("shared-radius.rows", 0, [0, 0, 0], &encoded_shared)
        .unwrap();
    let got_shared =
        collect_shared_boundary_radius_measurements(&env, "shared-radius.rows", 0, VOLUME).unwrap();
    assert_eq!(got_shared, shared);
    assert_eq!(RadiusFeature::ALL.len(), 5);
    assert_eq!(RadiusFeature::Median.column_name(), "median");
    assert_eq!(got_shared[0].feature(RadiusFeature::Count), Some(27.0));
    assert_eq!(got_shared[0].feature(RadiusFeature::Mean), Some(1.0 / 27.0));

    let exact = vec![ExactLabelRadiusMeasurements {
        label: 7,
        count: 27,
        finite_count: 27,
        mean: Some(28.0 / 27.0),
        max: Some(2.0),
        median: Some(1.0),
    }];
    let encoded_exact = encode_exact_label_radius_measurements(&exact).unwrap();
    assert_eq!(
        encoded_schema(&encoded_exact).unwrap(),
        radius_measurement_schema()
    );
    env.declare_sidecar("exact-radius.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("exact-radius.rows", 0, [0, 0, 0], &encoded_exact)
        .unwrap();
    let got_exact =
        collect_exact_label_radius_measurements(&env, "exact-radius.rows", 0, VOLUME).unwrap();
    assert_eq!(got_exact, exact);
    assert_eq!(got_exact[0].feature(RadiusFeature::Max), Some(2.0));
    assert_eq!(got_exact[0].feature(RadiusFeature::Median), Some(1.0));

    assert!(
        encode_shared_boundary_radius_measurements(&[SharedBoundaryRadiusMeasurements {
            label: 0,
            count: 1,
            finite_count: 1,
            mean: Some(1.0),
            max: Some(1.0),
            median: Some(1.0),
        }])
        .is_err()
    );
    assert!(
        encode_exact_label_radius_measurements(&[ExactLabelRadiusMeasurements {
            label: 7,
            count: 1,
            finite_count: 0,
            mean: Some(1.0),
            max: None,
            median: None,
        }])
        .is_err()
    );
}

fn planned_shared_boundary_radius(block: [usize; 3]) -> Vec<SharedBoundaryRadiusMeasurements> {
    let params = DistanceParams::default();
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = SharedBoundaryRadiusOp::new(
        "measure shared-boundary radius",
        0usize,
        params,
        "shared-radius.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned shared-boundary radius",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_shared_boundary_radius_measurements(&env, "shared-radius.planned.rows", 1, VOLUME)
        .unwrap()
}

fn planned_exact_label_radius(block: [usize; 3]) -> Vec<ExactLabelRadiusMeasurements> {
    let params = DistanceParams::default();
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = ExactLabelRadiusOp::new(
        "measure exact-label radius",
        0usize,
        params,
        "exact-radius.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned exact-label radius",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_exact_label_radius_measurements(&env, "exact-radius.planned.rows", 1, VOLUME).unwrap()
}

#[test]
fn planned_one_phase_ops_validate_output_streams_at_construction() {
    let params = DistanceParams::default();
    let granularity = GranularitySet::new(vec![1]).unwrap();
    assert!(SharedBoundaryRadiusOp::new(
        "measure shared-boundary radius",
        0usize,
        params,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ExactLabelRadiusOp::new(
        "measure exact-label radius",
        0usize,
        params,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(GranularityOp::new(
        "measure granularity",
        0usize,
        ImageId::supplied(0),
        granularity,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectTopologyOp::new(
        "measure topology",
        0usize,
        ObjectTopologyConvention::default(),
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
    assert!(ObjectComponentOp::new(
        "measure components",
        0usize,
        Connectivity::Faces,
        "",
        Lifecycle::DeleteOnExit,
    )
    .is_err());
}

#[test]
fn planned_shared_boundary_radius_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_shared_boundary_radius(VOLUME);
    let split = planned_shared_boundary_radius([2, 2, 2]);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference =
        shared_boundary_radius_measurements(labels.view(), &DistanceParams::default()).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn planned_exact_label_radius_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_exact_label_radius(VOLUME);
    let split = planned_exact_label_radius([2, 2, 2]);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference =
        exact_label_radius_measurements(labels.view(), &DistanceParams::default()).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_shared_boundary_radius() {
    let params = DistanceParams::default();
    let expected =
        shared_boundary_radius_measurements(labels().view::<f64>().unwrap(), &params).unwrap();
    let plan = Measurements::for_labels(0usize)
        .shared_boundary_radius(params)
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.shared_boundary_radius_rows_phase(), Some(1));
    assert_eq!(plan.shared_boundary_radius_params(), Some(params));
    let radius_rows = plan.shared_boundary_radius_rows().unwrap();
    assert_eq!(
        radius_rows.stream(),
        plan.shared_boundary_radius_stream().unwrap()
    );
    assert_eq!(
        radius_rows.phase(),
        plan.shared_boundary_radius_rows_phase().unwrap()
    );

    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder shared-boundary radius",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_shared_boundary_radius_rows(&env, &radius_rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn measurement_builder_runs_planned_exact_label_radius() {
    let params = DistanceParams::default();
    let expected =
        exact_label_radius_measurements(labels().view::<f64>().unwrap(), &params).unwrap();
    let plan = Measurements::for_labels(0usize)
        .exact_label_radius(params)
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.exact_label_radius_rows_phase(), Some(1));
    assert_eq!(plan.exact_label_radius_params(), Some(params));
    let radius_rows = plan.exact_label_radius_rows().unwrap();
    assert_eq!(
        radius_rows.stream(),
        plan.exact_label_radius_stream().unwrap()
    );
    assert_eq!(
        radius_rows.phase(),
        plan.exact_label_radius_rows_phase().unwrap()
    );

    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder exact-label radius",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_exact_label_radius_rows(&env, &radius_rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn object_geometry_reports_voxel_centre_feret_with_physical_spacing() {
    let mut labels = Array3::<f64>::zeros((4, 4, 4));
    labels[[0, 0, 0]] = 2.0;
    labels[[2, 1, 3]] = 2.0;
    labels[[3, 3, 3]] = 5.0;

    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    let objects = object_geometry_measurements(labels.view(), spacing).unwrap();
    assert_eq!(objects.len(), 2);

    let first = &objects[0];
    assert_eq!(first.label, 2);
    assert_eq!(first.count, 2);
    assert_eq!(first.bbox_min, [0, 0, 0]);
    assert_eq!(first.bbox_max, [3, 2, 4]);
    assert_eq!(first.physical_bbox_extent, [6.0, 6.0, 20.0]);
    let expected = (4.0f64 * 4.0 + 3.0 * 3.0 + 15.0 * 15.0).sqrt();
    assert_eq!(first.max_voxel_feret_diameter, expected);
    assert_eq!(ObjectGeometryFeature::ALL.len(), 11);
    assert_eq!(
        ObjectGeometryFeature::MaxVoxelFeretDiameter.column_name(),
        "max_voxel_feret_diameter"
    );
    assert_eq!(first.feature(ObjectGeometryFeature::Count), 2.0);
    assert_eq!(first.feature(ObjectGeometryFeature::BboxMaxX), 4.0);
    assert_eq!(
        first.feature(ObjectGeometryFeature::PhysicalBboxExtentX),
        20.0
    );
    assert_eq!(
        first.feature(ObjectGeometryFeature::MaxVoxelFeretDiameter),
        expected
    );

    let second = &objects[1];
    assert_eq!(second.label, 5);
    assert_eq!(second.count, 1);
    assert_eq!(second.max_voxel_feret_diameter, 0.0);

    labels[[0, 0, 1]] = -1.0;
    assert!(object_geometry_measurements(labels.view(), spacing).is_err());
}

#[test]
fn object_3d_moments_report_physical_central_moments() {
    let mut labels = Array3::<f64>::zeros((2, 2, 4));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 0, 3]] = 2.0;
    labels[[1, 1, 1]] = 5.0;

    let spacing = PhysicalSpacing::new([1.0, 1.0, 2.0]).unwrap();
    let set = ObjectMoment3dSet::new(3).unwrap();
    let rows = object_3d_moment_measurements_set(labels.view(), spacing, set).unwrap();
    assert_eq!(
        object_3d_moment_measurements(labels.view(), spacing, set.max_order()).unwrap(),
        rows
    );
    assert_eq!(rows.len(), 2);

    let line = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(line.count, 3);
    assert_eq!(line.max_order, 3);
    assert_eq!(line.centroid, [0.0, 0.0, 8.0 / 3.0]);
    assert_eq!(line.moments.len(), 20);
    assert_eq!(line.central([0, 0, 0]), Some(3.0));
    assert!(line.central([0, 0, 1]).unwrap().abs() < 1.0e-12);
    assert!((line.central([0, 0, 2]).unwrap() - 56.0 / 3.0).abs() < 1.0e-12);
    assert!((line.central([0, 0, 3]).unwrap() - 160.0 / 9.0).abs() < 1.0e-12);
    assert!(line.central([1, 0, 0]).unwrap().abs() < 1.0e-12);
    assert!(line.central([4, 0, 0]).is_none());
    assert_eq!(line.normalized([0, 0, 0]), Some(1.0));
    let expected_normalized = (56.0 / 3.0) / 3.0f64.powf(5.0 / 3.0);
    assert!((line.normalized([0, 0, 2]).unwrap() - expected_normalized).abs() < 1.0e-12);
    assert_eq!(ObjectMoment3dFeature::Count.column_name(), "count");
    let second_x_order = Moment3dKey::new([0, 0, 2]).unwrap();
    assert_eq!(second_x_order.get(), [0, 0, 2]);
    assert_eq!(line.central_key(second_x_order), line.central([0, 0, 2]));
    assert_eq!(
        line.normalized_key(second_x_order),
        line.normalized([0, 0, 2])
    );
    assert!(Moment3dKey::new([7, 0, 0]).is_err());
    assert!(Moment3dKey::new([usize::MAX, 1, 0]).is_err());
    assert!(ObjectMoment3dFeature::central([7, 0, 0]).is_err());
    assert!(line.central([7, 0, 0]).is_none());
    assert!(line.normalized([7, 0, 0]).is_none());
    assert_eq!(
        ObjectMoment3dFeature::central([0, 0, 2])
            .unwrap()
            .column_name(),
        "central_0_0_2"
    );
    assert_eq!(line.feature(ObjectMoment3dFeature::Count), Some(3.0));
    assert_eq!(
        line.feature(ObjectMoment3dFeature::CentroidX),
        Some(8.0 / 3.0)
    );
    assert_eq!(
        line.feature(ObjectMoment3dFeature::central([0, 0, 2]).unwrap()),
        line.central([0, 0, 2])
    );
    assert_eq!(
        line.feature(ObjectMoment3dFeature::normalized([0, 0, 2]).unwrap()),
        line.normalized([0, 0, 2])
    );
    assert_eq!(
        line.feature(ObjectMoment3dFeature::central([4, 0, 0]).unwrap()),
        None
    );

    let singleton = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(singleton.central([0, 0, 0]), Some(1.0));
    assert_eq!(singleton.central([0, 0, 2]), Some(0.0));
}

#[test]
fn object_3d_moments_validate_labels_and_order() {
    let mut labels = Array3::<f64>::zeros((1, 1, 1));
    labels[[0, 0, 0]] = 2.0;
    assert!(object_3d_moment_measurements(labels.view(), PhysicalSpacing::unit(), 7).is_err());

    labels[[0, 0, 0]] = 1.5;
    assert!(object_3d_moment_measurements(labels.view(), PhysicalSpacing::unit(), 3).is_err());
}

#[test]
fn object_3d_moment_rows_have_ordered_schema_and_collector() {
    let rows = vec![ObjectMoment3dMeasurements {
        label: 2,
        count: 3,
        centroid: [1.0, 2.0, 3.0],
        max_order: 1,
        moments: vec![
            Moment3d {
                order: [0, 0, 0],
                central: 3.0,
                normalized: 1.0,
            },
            Moment3d {
                order: [0, 0, 1],
                central: 0.5,
                normalized: 0.25,
            },
            Moment3d {
                order: [0, 1, 0],
                central: 1.5,
                normalized: 0.75,
            },
            Moment3d {
                order: [1, 0, 0],
                central: 2.5,
                normalized: 1.25,
            },
        ],
    }];

    let set = ObjectMoment3dSet::new(1).unwrap();
    let encoded = encode_object_3d_moment_measurements_set(&rows, set).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, object_3d_moment_measurement_schema(1).unwrap());
    assert_eq!(
        schema,
        object_3d_moment_measurement_schema_set(set).unwrap()
    );
    assert_eq!(
        encoded,
        encode_object_3d_moment_measurements(&rows, 1).unwrap()
    );
    assert_eq!(schema.columns()[6].name(), "central_0_0_0");
    assert_eq!(schema.columns()[7].name(), "normalized_0_0_0");
    assert_eq!(schema.columns()[12].name(), "central_1_0_0");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("moment3d.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("moment3d.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_object_3d_moment_measurements(&env, "moment3d.rows", 0, VOLUME, 1).unwrap();
    assert_eq!(got, rows);
    let got =
        collect_object_3d_moment_measurements_set(&env, "moment3d.rows", 0, VOLUME, set).unwrap();
    assert_eq!(got, rows);

    assert!(encode_object_3d_moment_measurements(
        &[ObjectMoment3dMeasurements {
            label: 0,
            count: 1,
            centroid: [0.0, 0.0, 0.0],
            max_order: 0,
            moments: vec![Moment3d {
                order: [0, 0, 0],
                central: 1.0,
                normalized: 1.0,
            }],
        }],
        0,
    )
    .is_err());
    assert!(encode_object_3d_moment_measurements(
        &[ObjectMoment3dMeasurements {
            label: 3,
            count: 1,
            centroid: [0.0, 0.0, 0.0],
            max_order: 1,
            moments: vec![Moment3d {
                order: [0, 0, 1],
                central: 1.0,
                normalized: 1.0,
            }],
        }],
        1,
    )
    .is_err());
    assert!(object_3d_moment_measurement_schema(7).is_err());
}

fn planned_object_3d_moments(
    block: [usize; 3],
    spacing: PhysicalSpacing,
    max_order: usize,
) -> Vec<ObjectMoment3dMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let set = ObjectMoment3dSet::new(max_order).unwrap();
    let op = ObjectMoment3dOp::new_set(
        "measure 3D object moments",
        0usize,
        spacing,
        set,
        "moment3d.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned 3D object moments",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_object_3d_moment_measurements_set(&env, "moment3d.planned.rows", 1, VOLUME, set)
        .unwrap()
}

#[test]
fn planned_object_3d_moments_are_decomposition_invariant_and_match_resident_reference() {
    let spacing = PhysicalSpacing::new([1.0, 2.0, 3.0]).unwrap();
    let coarse = planned_object_3d_moments(VOLUME, spacing, 2);
    let split = planned_object_3d_moments([2, 2, 2], spacing, 2);
    assert_eq!(coarse, split);

    let reference =
        object_3d_moment_measurements(labels().view::<f64>().unwrap(), spacing, 2).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_object_3d_moments() {
    let spacing = PhysicalSpacing::new([1.0, 2.0, 3.0]).unwrap();
    let expected =
        object_3d_moment_measurements(labels().view::<f64>().unwrap(), spacing, 2).unwrap();
    let plan = Measurements::for_labels(0usize)
        .object_3d_moments(spacing, 2)
        .unwrap()
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_3d_moment_rows_phase(), Some(2));
    assert_eq!(plan.object_3d_moment_max_order(), Some(2));
    assert_eq!(
        plan.object_3d_moment_set(),
        Some(ObjectMoment3dSet::new(2).unwrap())
    );
    let rows = plan.object_3d_moment_rows_with_set().unwrap();
    assert_eq!(rows.stream(), plan.object_3d_moment_stream().unwrap());
    assert_eq!(rows.phase(), plan.object_3d_moment_rows_phase().unwrap());
    assert_eq!(rows.contract(), ObjectMoment3dSet::new(2).unwrap());
    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder 3D object moments",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_object_3d_moment_rows_with_set(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn object_projected_convex_reports_footprint_hull_and_enclosing_circle() {
    let mut labels = Array3::<f64>::zeros((2, 3, 3));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 2.0;
    labels[[0, 2, 2]] = 5.0;
    labels[[1, 2, 2]] = 5.0;

    let rows = object_projected_convex_measurements(
        labels.view(),
        PhysicalSpacing::unit(),
        ProjectionAxis::Z,
    )
    .unwrap();
    assert_eq!(
        object_projected_convex_measurements_set(
            labels.view(),
            PhysicalSpacing::unit(),
            ProjectionAxis::Z,
            ProjectedConvexSet::new(5).unwrap()
        )
        .unwrap(),
        rows
    );
    assert!(object_projected_convex_measurements_set(
        labels.view(),
        PhysicalSpacing::unit(),
        ProjectionAxis::Z,
        ProjectedConvexSet::new(4).unwrap()
    )
    .is_err());
    assert_eq!(rows.len(), 2);

    let l_shape = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(l_shape.projected_count, 3);
    assert_eq!(l_shape.projection_axis, ProjectionAxis::Z);
    assert_eq!(l_shape.projected_area, 3.0);
    assert_eq!(l_shape.convex_hull_area, 3.5);
    assert_eq!(l_shape.solidity, Some(6.0 / 7.0));
    assert_eq!(l_shape.hull_vertices.len(), 5);
    assert!((l_shape.max_projected_feret_diameter - 8.0f64.sqrt()).abs() < 1.0e-12);
    assert_eq!(l_shape.min_enclosing_circle_center, [0.5, 0.5]);
    assert!((l_shape.min_enclosing_circle_radius - 0.5f64.sqrt()).abs() < 1.0e-12);
    assert_eq!(l_shape.min_enclosing_circle_support_points, 2);
    assert_eq!(ObjectProjectedConvexFeature::ALL.len(), 10);
    assert_eq!(
        ObjectProjectedConvexFeature::MaxProjectedFeretDiameter.column_name(),
        "max_projected_feret_diameter"
    );
    assert_eq!(
        l_shape.feature(ObjectProjectedConvexFeature::ProjectedCount),
        Some(3.0)
    );
    assert_eq!(
        l_shape.feature(ObjectProjectedConvexFeature::Solidity),
        l_shape.solidity
    );
    assert_eq!(
        l_shape.feature(ObjectProjectedConvexFeature::HullVertexCount),
        Some(l_shape.hull_vertices.len() as f64)
    );
    assert_eq!(
        l_shape.feature(ObjectProjectedConvexFeature::MinEnclosingCircleRadius),
        Some(l_shape.min_enclosing_circle_radius)
    );

    let collapsed = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(collapsed.projected_count, 1);
    assert_eq!(collapsed.projected_area, 1.0);
    assert_eq!(collapsed.convex_hull_area, 1.0);
    assert_eq!(collapsed.solidity, Some(1.0));
    assert!((collapsed.max_projected_feret_diameter - 2.0f64.sqrt()).abs() < 1.0e-12);
    assert_eq!(collapsed.min_enclosing_circle_radius, 0.0);

    let scaled = object_projected_convex_measurements(
        labels.view(),
        PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap(),
        ProjectionAxis::Z,
    )
    .unwrap();
    let scaled_l = scaled.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(scaled_l.projected_area, 45.0);
    assert_eq!(scaled_l.convex_hull_area, 52.5);
    assert_eq!(scaled_l.solidity, Some(6.0 / 7.0));
    assert!((scaled_l.max_projected_feret_diameter - 136.0f64.sqrt()).abs() < 1.0e-12);

    labels[[1, 0, 0]] = 1.5;
    assert!(object_projected_convex_measurements(
        labels.view(),
        PhysicalSpacing::unit(),
        ProjectionAxis::Y
    )
    .is_err());
}

#[test]
fn object_projected_convex_rows_have_bounded_schema_and_collector() {
    let rows = vec![
        ObjectProjectedConvexMeasurements {
            label: 2,
            projected_count: 3,
            projection_axis: ProjectionAxis::Z,
            projected_area: 3.0,
            convex_hull_area: 3.5,
            solidity: Some(6.0 / 7.0),
            hull_vertices: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            max_projected_feret_diameter: 2.0f64.sqrt(),
            min_enclosing_circle_center: [0.5, 0.5],
            min_enclosing_circle_radius: 0.5f64.sqrt(),
            min_enclosing_circle_support_points: 2,
        },
        ObjectProjectedConvexMeasurements {
            label: 5,
            projected_count: 1,
            projection_axis: ProjectionAxis::Y,
            projected_area: 1.0,
            convex_hull_area: 1.0,
            solidity: Some(1.0),
            hull_vertices: vec![[2.0, 2.0]],
            max_projected_feret_diameter: 0.0,
            min_enclosing_circle_center: [2.0, 2.0],
            min_enclosing_circle_radius: 0.0,
            min_enclosing_circle_support_points: 1,
        },
    ];
    let set = ProjectedConvexSet::new(3).unwrap();
    let encoded = encode_object_projected_convex_measurements_set(&rows, set).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(
        schema,
        object_projected_convex_measurement_schema(3).unwrap()
    );
    assert_eq!(schema, object_projected_convex_measurement_schema_set(set));
    assert!(object_projected_convex_measurement_schema(0).is_err());
    assert_eq!(schema.columns()[6].name(), "hull_vertex_count");
    assert_eq!(schema.columns()[12].name(), "hull_vertex_0_0");
    assert_eq!(schema.columns()[17].name(), "hull_vertex_2_1");
    assert!(encode_object_projected_convex_measurements(&rows, 0).is_err());

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("projected-convex.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("projected-convex.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got =
        collect_object_projected_convex_measurements(&env, "projected-convex.rows", 0, VOLUME, 3)
            .unwrap();
    assert_eq!(got, rows);

    assert!(encode_object_projected_convex_measurements(&rows, 2).is_err());

    let mut no_vertices = rows[0].clone();
    no_vertices.hull_vertices.clear();
    assert!(encode_object_projected_convex_measurements(&[no_vertices], 3).is_err());

    let mut bad_circle = rows[1].clone();
    bad_circle.min_enclosing_circle_support_points = 2;
    assert!(encode_object_projected_convex_measurements(&[bad_circle], 3).is_err());
}

fn planned_object_projected_convex(
    block: [usize; 3],
    spacing: PhysicalSpacing,
    projection_axis: ProjectionAxis,
    max_hull_vertices: usize,
) -> Vec<ObjectProjectedConvexMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let set = ProjectedConvexSet::new(max_hull_vertices).unwrap();
    let op = ObjectProjectedConvexOp::new_set(
        "measure projected convex geometry",
        0usize,
        spacing,
        projection_axis,
        set,
        "projected-convex.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned projected convex geometry",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_object_projected_convex_measurements_set(
        &env,
        "projected-convex.planned.rows",
        1,
        VOLUME,
        set,
    )
    .unwrap()
}

#[test]
fn planned_object_projected_convex_is_decomposition_invariant_and_matches_resident_reference() {
    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    let coarse = planned_object_projected_convex(VOLUME, spacing, ProjectionAxis::Z, 16);
    let split = planned_object_projected_convex([2, 2, 2], spacing, ProjectionAxis::Z, 16);
    assert_eq!(coarse, split);

    let reference = object_projected_convex_measurements(
        labels().view::<f64>().unwrap(),
        spacing,
        ProjectionAxis::Z,
    )
    .unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_object_projected_convex() {
    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    let expected = object_projected_convex_measurements(
        labels().view::<f64>().unwrap(),
        spacing,
        ProjectionAxis::Z,
    )
    .unwrap();
    let plan = Measurements::for_labels(0usize)
        .object_projected_convex(spacing, ProjectionAxis::Z, 16)
        .unwrap()
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_projected_convex_rows_phase(), Some(2));
    assert_eq!(plan.object_projected_convex_max_hull_vertices(), Some(16));
    assert_eq!(
        plan.object_projected_convex_set(),
        Some(ProjectedConvexSet::new(16).unwrap())
    );
    assert_eq!(
        plan.object_projected_convex_contract(),
        Some(ObjectProjectedConvexContract::new(
            spacing,
            ProjectionAxis::Z,
            ProjectedConvexSet::new(16).unwrap()
        ))
    );
    assert_eq!(plan.object_projected_convex_spacing(), Some(spacing));
    assert_eq!(
        plan.object_projected_convex_projection_axis(),
        Some(ProjectionAxis::Z)
    );
    let rows = plan.object_projected_convex_rows_with_contract().unwrap();
    assert_eq!(
        rows.stream(),
        plan.object_projected_convex_stream().unwrap()
    );
    assert_eq!(
        rows.phase(),
        plan.object_projected_convex_rows_phase().unwrap()
    );
    assert_eq!(
        rows.contract(),
        ObjectProjectedConvexContract::new(
            spacing,
            ProjectionAxis::Z,
            ProjectedConvexSet::new(16).unwrap()
        )
    );

    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder projected convex geometry",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_object_projected_convex_rows_with_contract(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn object_enclosing_sphere_is_exact_for_voxel_centres() {
    let mut labels = Array3::<f64>::zeros((5, 5, 5));
    labels[[4, 0, 0]] = 2.0;
    labels[[4, 0, 4]] = 2.0;
    labels[[0, 0, 0]] = 7.0;
    labels[[2, 0, 0]] = 7.0;
    labels[[0, 2, 0]] = 7.0;
    labels[[0, 0, 2]] = 7.0;
    labels[[4, 4, 4]] = 11.0;
    labels[[4, 2, 2]] = 11.0;
    labels[[2, 4, 2]] = 11.0;
    labels[[2, 2, 4]] = 11.0;

    let spheres =
        object_enclosing_sphere_measurements(labels.view(), PhysicalSpacing::unit()).unwrap();
    assert_eq!(spheres.len(), 3);

    let line = spheres.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(line.count, 2);
    assert_eq!(line.center, [4.0, 0.0, 2.0]);
    assert_eq!(line.radius, 2.0);
    assert_eq!(line.support_points, 2);
    assert_eq!(ObjectEnclosingSphereFeature::ALL.len(), 6);
    assert_eq!(
        ObjectEnclosingSphereFeature::SupportPoints.column_name(),
        "support_points"
    );
    assert_eq!(line.feature(ObjectEnclosingSphereFeature::Count), 2.0);
    assert_eq!(line.feature(ObjectEnclosingSphereFeature::CenterX), 2.0);
    assert_eq!(
        line.feature(ObjectEnclosingSphereFeature::Radius),
        line.radius
    );
    assert_eq!(
        line.feature(ObjectEnclosingSphereFeature::SupportPoints),
        2.0
    );

    let tetrahedron = spheres.iter().find(|row| row.label == 7).unwrap();
    assert_eq!(tetrahedron.count, 4);
    for coordinate in tetrahedron.center {
        assert!((coordinate - 2.0 / 3.0).abs() < 1.0e-12);
    }
    assert!((tetrahedron.radius - (8.0f64 / 3.0).sqrt()).abs() < 1.0e-12);
    assert_eq!(tetrahedron.support_points, 3);

    let regular = spheres.iter().find(|row| row.label == 11).unwrap();
    assert_eq!(regular.count, 4);
    assert_eq!(regular.center, [3.0, 3.0, 3.0]);
    assert!((regular.radius - 3.0f64.sqrt()).abs() < 1.0e-12);
    assert_eq!(regular.support_points, 4);

    labels[[4, 4, 3]] = -1.0;
    assert!(object_enclosing_sphere_measurements(labels.view(), PhysicalSpacing::unit()).is_err());
}

#[test]
fn object_geometry_and_sphere_rows_have_canonical_schemas_and_collectors() {
    let geometry = vec![ObjectGeometryMeasurements {
        label: 2,
        count: 2,
        bbox_min: [0, 0, 0],
        bbox_max: [3, 2, 4],
        physical_bbox_extent: [6.0, 6.0, 20.0],
        max_voxel_feret_diameter: (4.0f64 * 4.0 + 3.0 * 3.0 + 15.0 * 15.0).sqrt(),
    }];
    let encoded_geometry = encode_object_geometry_measurements(&geometry).unwrap();
    assert_eq!(
        encoded_schema(&encoded_geometry).unwrap(),
        object_geometry_measurement_schema()
    );

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("object-geometry.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("object-geometry.rows", 0, [0, 0, 0], &encoded_geometry)
        .unwrap();
    let got_geometry =
        collect_object_geometry_measurements(&env, "object-geometry.rows", 0, VOLUME).unwrap();
    assert_eq!(got_geometry, geometry);

    let spheres = vec![ObjectEnclosingSphereMeasurements {
        label: 2,
        count: 2,
        center: [1.0, 2.0, 3.0],
        radius: 4.0,
        support_points: 2,
    }];
    let encoded_spheres = encode_enclosing_sphere_measurements(&spheres).unwrap();
    assert_eq!(
        encoded_schema(&encoded_spheres).unwrap(),
        enclosing_sphere_measurement_schema()
    );
    env.declare_sidecar("enclosing-sphere.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("enclosing-sphere.rows", 0, [0, 0, 0], &encoded_spheres)
        .unwrap();
    let got_spheres =
        collect_enclosing_sphere_measurements(&env, "enclosing-sphere.rows", 0, VOLUME).unwrap();
    assert_eq!(got_spheres, spheres);

    assert!(
        encode_object_geometry_measurements(&[ObjectGeometryMeasurements {
            label: 0,
            count: 1,
            bbox_min: [0, 0, 0],
            bbox_max: [1, 1, 1],
            physical_bbox_extent: [1.0, 1.0, 1.0],
            max_voxel_feret_diameter: 0.0,
        }])
        .is_err()
    );
    assert!(
        encode_object_geometry_measurements(&[ObjectGeometryMeasurements {
            label: 3,
            count: 1,
            bbox_min: [0, 0, 0],
            bbox_max: [0, 1, 1],
            physical_bbox_extent: [1.0, 1.0, 1.0],
            max_voxel_feret_diameter: 0.0,
        }])
        .is_err()
    );
    assert!(
        encode_enclosing_sphere_measurements(&[ObjectEnclosingSphereMeasurements {
            label: 3,
            count: 1,
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
            support_points: 0,
        }])
        .is_err()
    );
    assert!(
        encode_enclosing_sphere_measurements(&[ObjectEnclosingSphereMeasurements {
            label: 3,
            count: 1,
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
            support_points: 2,
        }])
        .is_err()
    );
}

fn planned_object_geometry(
    block: [usize; 3],
    spacing: PhysicalSpacing,
) -> Vec<ObjectGeometryMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = ObjectGeometryOp::new(
        "measure object geometry",
        0usize,
        spacing,
        "object-geometry.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned object geometry",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_object_geometry_measurements(&env, "object-geometry.planned.rows", 1, VOLUME).unwrap()
}

fn planned_enclosing_sphere(
    block: [usize; 3],
    spacing: PhysicalSpacing,
) -> Vec<ObjectEnclosingSphereMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = EnclosingSphereOp::new(
        "measure enclosing sphere",
        0usize,
        spacing,
        "enclosing-sphere.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned enclosing sphere",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_enclosing_sphere_measurements(&env, "enclosing-sphere.planned.rows", 1, VOLUME).unwrap()
}

#[test]
fn planned_object_geometry_is_decomposition_invariant_and_matches_resident_reference() {
    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    let coarse = planned_object_geometry(VOLUME, spacing);
    let split = planned_object_geometry([2, 2, 2], spacing);
    assert_eq!(coarse, split);

    let reference = object_geometry_measurements(labels().view::<f64>().unwrap(), spacing).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn planned_enclosing_sphere_is_decomposition_invariant_and_matches_resident_reference() {
    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    let coarse = planned_enclosing_sphere(VOLUME, spacing);
    let split = planned_enclosing_sphere([2, 2, 2], spacing);
    assert_eq!(coarse, split);

    let reference =
        object_enclosing_sphere_measurements(labels().view::<f64>().unwrap(), spacing).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_object_geometry_and_enclosing_sphere() {
    let spacing = PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap();
    let expected_geometry =
        object_geometry_measurements(labels().view::<f64>().unwrap(), spacing).unwrap();
    let expected_spheres =
        object_enclosing_sphere_measurements(labels().view::<f64>().unwrap(), spacing).unwrap();
    let plan = Measurements::for_labels(0usize)
        .object_geometry(spacing)
        .enclosing_sphere(spacing)
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_geometry_rows_phase(), Some(2));
    assert_eq!(plan.enclosing_sphere_rows_phase(), Some(4));
    assert_eq!(plan.object_geometry_spacing(), Some(spacing));
    assert_eq!(plan.enclosing_sphere_spacing(), Some(spacing));
    let geometry_rows = plan.object_geometry_rows_with_contract().unwrap();
    assert_eq!(
        geometry_rows.stream(),
        plan.object_geometry_stream().unwrap()
    );
    assert_eq!(
        geometry_rows.phase(),
        plan.object_geometry_rows_phase().unwrap()
    );
    assert_eq!(geometry_rows.contract(), spacing);
    let sphere_rows = plan.enclosing_sphere_rows_with_contract().unwrap();
    assert_eq!(
        sphere_rows.stream(),
        plan.enclosing_sphere_stream().unwrap()
    );
    assert_eq!(
        sphere_rows.phase(),
        plan.enclosing_sphere_rows_phase().unwrap()
    );
    assert_eq!(sphere_rows.contract(), spacing);

    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder object geometry and enclosing sphere",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got_geometry =
        collect_object_geometry_rows_with_contract(&env, &geometry_rows, VOLUME).unwrap();
    assert_eq!(got_geometry, expected_geometry);
    let got_spheres =
        collect_enclosing_sphere_rows_with_contract(&env, &sphere_rows, VOLUME).unwrap();
    assert_eq!(got_spheres, expected_spheres);
}

#[test]
fn object_convex_hull_reports_point_set_area_and_volume() {
    let mut labels = Array3::<f64>::zeros((3, 3, 3));
    labels[[0, 0, 0]] = 2.0;
    labels[[1, 0, 0]] = 2.0;
    labels[[0, 1, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;

    for z in 1..=2 {
        for y in 1..=2 {
            for x in 1..=2 {
                labels[[z, y, x]] = 5.0;
            }
        }
    }

    let rows = object_convex_hull_measurements(labels.view(), PhysicalSpacing::unit()).unwrap();
    assert_eq!(rows.len(), 2);

    let tetrahedron = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(tetrahedron.count, 4);
    assert_eq!(tetrahedron.hull_vertices, 4);
    assert_eq!(tetrahedron.hull_faces, 4);
    assert!((tetrahedron.surface_area - (1.5 + 3.0f64.sqrt() / 2.0)).abs() < 1.0e-12);
    assert!((tetrahedron.volume - 1.0 / 6.0).abs() < 1.0e-12);
    assert!((tetrahedron.max_hull_feret_diameter - 2.0f64.sqrt()).abs() < 1.0e-12);

    let cube = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(cube.count, 8);
    assert_eq!(cube.hull_vertices, 8);
    assert_eq!(cube.hull_faces, 6);
    assert!((cube.surface_area - 6.0).abs() < 1.0e-12);
    assert!((cube.volume - 1.0).abs() < 1.0e-12);
    assert!((cube.max_hull_feret_diameter - 3.0f64.sqrt()).abs() < 1.0e-12);
    assert_eq!(ObjectConvexHullFeature::ALL.len(), 6);
    assert_eq!(
        ObjectConvexHullFeature::MaxHullFeretDiameter.column_name(),
        "max_hull_feret_diameter"
    );
    assert_eq!(cube.feature(ObjectConvexHullFeature::Count), 8.0);
    assert_eq!(cube.feature(ObjectConvexHullFeature::HullVertices), 8.0);
    assert_eq!(
        cube.feature(ObjectConvexHullFeature::SurfaceArea),
        cube.surface_area
    );
    assert_eq!(cube.feature(ObjectConvexHullFeature::Volume), cube.volume);

    let scaled = object_convex_hull_measurements(
        labels.view(),
        PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap(),
    )
    .unwrap();
    let scaled_cube = scaled.iter().find(|row| row.label == 5).unwrap();
    assert!((scaled_cube.surface_area - 62.0).abs() < 1.0e-12);
    assert!((scaled_cube.volume - 30.0).abs() < 1.0e-12);
    assert!((scaled_cube.max_hull_feret_diameter - 38.0f64.sqrt()).abs() < 1.0e-12);

    labels[[2, 2, 2]] = 1.5;
    assert!(object_convex_hull_measurements(labels.view(), PhysicalSpacing::unit()).is_err());
}

#[test]
fn object_convex_hull_rows_have_canonical_schema_and_collector() {
    let rows = vec![
        ObjectConvexHullMeasurements {
            label: 2,
            count: 4,
            hull_vertices: 4,
            hull_faces: 4,
            surface_area: 2.5,
            volume: 1.0 / 6.0,
            max_hull_feret_diameter: 2.0f64.sqrt(),
        },
        ObjectConvexHullMeasurements {
            label: 5,
            count: 2,
            hull_vertices: 2,
            hull_faces: 0,
            surface_area: 0.0,
            volume: 0.0,
            max_hull_feret_diameter: 1.0,
        },
    ];
    let encoded = encode_object_convex_hull_measurements(&rows).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, object_convex_hull_measurement_schema());
    assert_eq!(schema.columns()[2].name(), "hull_vertices");
    assert_eq!(schema.columns()[3].name(), "hull_faces");
    assert_eq!(schema.columns()[6].name(), "max_hull_feret_diameter");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("convex-hull.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("convex-hull.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_object_convex_hull_measurements(&env, "convex-hull.rows", 0, VOLUME).unwrap();
    assert_eq!(got, rows);

    let mut zero_label = rows[0];
    zero_label.label = 0;
    assert!(encode_object_convex_hull_measurements(&[zero_label]).is_err());

    let mut too_many_vertices = rows[0];
    too_many_vertices.hull_vertices = 5;
    assert!(encode_object_convex_hull_measurements(&[too_many_vertices]).is_err());

    let mut no_faces = rows[0];
    no_faces.hull_faces = 0;
    assert!(encode_object_convex_hull_measurements(&[no_faces]).is_err());
}

#[test]
fn object_voxel_face_convex_hull_reports_continuous_area_volume_and_feret() {
    let mut labels = Array3::<f64>::zeros((3, 3, 3));
    labels[[1, 1, 1]] = 7.0;

    let rows =
        object_voxel_face_convex_hull_measurements(labels.view(), PhysicalSpacing::unit()).unwrap();
    assert_eq!(rows.len(), 1);
    let voxel = &rows[0];
    assert_eq!(voxel.label, 7);
    assert_eq!(voxel.count, 1);
    assert_eq!(voxel.hull_vertices, 8);
    assert_eq!(voxel.hull_faces, 6);
    assert!((voxel.surface_area - 6.0).abs() < 1.0e-12);
    assert!((voxel.volume - 1.0).abs() < 1.0e-12);
    assert!((voxel.max_feret_diameter - 3.0f64.sqrt()).abs() < 1.0e-12);
    assert_eq!(ObjectVoxelFaceConvexHullFeature::ALL.len(), 6);
    assert_eq!(
        ObjectVoxelFaceConvexHullFeature::MaxFeretDiameter.column_name(),
        "max_feret_diameter"
    );
    assert_eq!(
        voxel.feature(ObjectVoxelFaceConvexHullFeature::Count),
        voxel.count as f64
    );
    assert_eq!(
        voxel.feature(ObjectVoxelFaceConvexHullFeature::SurfaceArea),
        voxel.surface_area
    );

    let centre_rows = object_convex_hull_measurements(labels.view(), PhysicalSpacing::unit())
        .expect("single centre point convex hull is valid");
    let centre = &centre_rows[0];
    assert_eq!(centre.count, 1);
    assert_eq!(centre.hull_vertices, 1);
    assert_eq!(centre.hull_faces, 0);
    assert_eq!(centre.surface_area, 0.0);
    assert_eq!(centre.volume, 0.0);
    assert_eq!(centre.max_hull_feret_diameter, 0.0);

    let scaled = object_voxel_face_convex_hull_measurements(
        labels.view(),
        PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap(),
    )
    .unwrap();
    let scaled_voxel = &scaled[0];
    assert!((scaled_voxel.surface_area - 62.0).abs() < 1.0e-12);
    assert!((scaled_voxel.volume - 30.0).abs() < 1.0e-12);
    assert!((scaled_voxel.max_feret_diameter - 38.0f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn object_voxel_face_convex_hull_rows_have_canonical_schema_and_collector() {
    let rows = vec![ObjectVoxelFaceConvexHullMeasurements {
        label: 7,
        count: 1,
        hull_vertices: 8,
        hull_faces: 6,
        surface_area: 6.0,
        volume: 1.0,
        max_feret_diameter: 3.0f64.sqrt(),
    }];
    let encoded = encode_object_voxel_face_convex_hull_measurements(&rows).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, object_voxel_face_convex_hull_measurement_schema());
    assert_eq!(schema.columns()[2].name(), "hull_vertices");
    assert_eq!(schema.columns()[3].name(), "hull_faces");
    assert_eq!(schema.columns()[6].name(), "max_feret_diameter");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("voxel-face-convex-hull.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("voxel-face-convex-hull.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_object_voxel_face_convex_hull_measurements(
        &env,
        "voxel-face-convex-hull.rows",
        0,
        VOLUME,
    )
    .unwrap();
    assert_eq!(got, rows);

    let mut zero_label = rows[0];
    zero_label.label = 0;
    assert!(encode_object_voxel_face_convex_hull_measurements(&[zero_label]).is_err());

    let mut no_vertices = rows[0];
    no_vertices.hull_vertices = 0;
    assert!(encode_object_voxel_face_convex_hull_measurements(&[no_vertices]).is_err());

    let mut no_faces = rows[0];
    no_faces.hull_faces = 0;
    assert!(encode_object_voxel_face_convex_hull_measurements(&[no_faces]).is_err());
}

fn planned_object_convex_hull(
    block: [usize; 3],
    spacing: PhysicalSpacing,
) -> Vec<ObjectConvexHullMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = ObjectConvexHullOp::new(
        "measure planned convex hull",
        0usize,
        spacing,
        "convex-hull.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env =
        ArrayEnvironment::new(convex_hull_labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned convex hull",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_object_convex_hull_measurements(&env, "convex-hull.planned.rows", 1, VOLUME).unwrap()
}

#[test]
fn planned_object_convex_hull_is_decomposition_invariant_and_matches_resident_reference() {
    let spacing = PhysicalSpacing::new([1.0, 2.0, 3.0]).unwrap();
    let coarse = planned_object_convex_hull(VOLUME, spacing);
    let split = planned_object_convex_hull([2, 2, 2], spacing);
    assert_eq!(coarse, split);

    let labels = convex_hull_labels().view::<f64>().unwrap().to_owned();
    let reference = object_convex_hull_measurements(labels.view(), spacing).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_object_convex_hull() {
    let spacing = PhysicalSpacing::new([1.0, 2.0, 3.0]).unwrap();
    let labels_array = convex_hull_labels().view::<f64>().unwrap().to_owned();
    let expected = object_convex_hull_measurements(labels_array.view(), spacing).unwrap();

    let plan = Measurements::for_labels(0usize)
        .object_convex_hull(spacing)
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_convex_hull_rows_phase(), Some(2));
    assert_eq!(plan.object_convex_hull_spacing(), Some(spacing));
    let rows = plan.object_convex_hull_rows_with_contract().unwrap();
    assert_eq!(rows.stream(), plan.object_convex_hull_stream().unwrap());
    assert_eq!(rows.phase(), plan.object_convex_hull_rows_phase().unwrap());
    assert_eq!(rows.contract(), spacing);

    let env = ArrayEnvironment::new(
        convex_hull_labels(),
        plan.decomposition.n_phases(),
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder convex hull",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_object_convex_hull_rows_with_contract(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn measurement_builder_runs_planned_object_voxel_face_convex_hull() {
    let spacing = PhysicalSpacing::new([1.0, 2.0, 3.0]).unwrap();
    let labels_array = convex_hull_labels().view::<f64>().unwrap().to_owned();
    let expected =
        object_voxel_face_convex_hull_measurements(labels_array.view(), spacing).unwrap();

    let plan = Measurements::for_labels(0usize)
        .object_voxel_face_convex_hull(spacing)
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_voxel_face_convex_hull_rows_phase(), Some(2));
    assert_eq!(plan.object_voxel_face_convex_hull_spacing(), Some(spacing));
    let rows = plan
        .object_voxel_face_convex_hull_rows_with_contract()
        .unwrap();
    assert_eq!(
        rows.stream(),
        plan.object_voxel_face_convex_hull_stream().unwrap()
    );
    assert_eq!(
        rows.phase(),
        plan.object_voxel_face_convex_hull_rows_phase().unwrap()
    );
    assert_eq!(rows.contract(), spacing);

    let env = ArrayEnvironment::new(
        convex_hull_labels(),
        plan.decomposition.n_phases(),
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder voxel-face convex hull",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got =
        collect_object_voxel_face_convex_hull_rows_with_contract(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn object_hu_moments_use_projected_binary_points() {
    let mut labels = Array3::<f64>::zeros((3, 3, 3));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[1, 1, 0]] = 5.0;
    labels[[2, 1, 0]] = 5.0;

    let rows = object_hu_moments_measurements(labels.view(), ProjectionAxis::Z).unwrap();
    assert_eq!(rows.len(), 2);

    let line = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(line.projected_count, 2);
    assert_eq!(line.projection_axis, ProjectionAxis::Z);
    assert_eq!(line.centroid, [0.0, 0.5]);
    assert_eq!(line.hu[0], 0.125);
    assert_eq!(line.hu[1], 0.015625);
    assert_eq!(&line.hu[2..], &[0.0; 5]);
    assert_eq!(
        ObjectHuMomentFeature::ProjectedCount.column_name(),
        "projected_count"
    );
    assert_eq!(HuMomentIndex::new(6).unwrap().get(), 6);
    assert!(HuMomentIndex::new(7).is_err());
    assert!(ObjectHuMomentFeature::hu(7).is_err());
    assert_eq!(ObjectHuMomentFeature::hu(6).unwrap().column_name(), "hu_6");
    let hu0 = HuMomentIndex::new(0).unwrap();
    assert_eq!(line.hu_key(hu0), Some(line.hu[0]));
    assert_eq!(line.hu(0), Some(line.hu[0]));
    assert_eq!(line.hu(7), None);
    assert_eq!(
        line.feature(ObjectHuMomentFeature::ProjectedCount),
        Some(2.0)
    );
    assert_eq!(
        line.feature(ObjectHuMomentFeature::Centroid1),
        Some(line.centroid[1])
    );
    assert_eq!(
        line.feature(ObjectHuMomentFeature::hu(0).unwrap()),
        Some(line.hu[0])
    );

    let collapsed = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(collapsed.projected_count, 1);
    assert_eq!(collapsed.centroid, [1.0, 0.0]);
    assert_eq!(collapsed.hu, [0.0; 7]);
}

#[test]
fn object_hu_moments_validate_labels_and_projection_axis() {
    let mut labels = Array3::<f64>::zeros((2, 2, 2));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;

    let x_projection = object_hu_moments_measurements(labels.view(), ProjectionAxis::X).unwrap();
    assert_eq!(x_projection[0].projected_count, 1);
    assert_eq!(x_projection[0].centroid, [0.0, 0.0]);

    labels[[1, 1, 1]] = 1.5;
    assert!(object_hu_moments_measurements(labels.view(), ProjectionAxis::Y).is_err());
}

#[test]
fn object_hu_moment_rows_have_canonical_schema_and_collector() {
    let rows = vec![ObjectHuMomentsMeasurements {
        label: 2,
        projected_count: 3,
        projection_axis: ProjectionAxis::Z,
        centroid: [1.0, 2.0],
        hu: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    }];
    let encoded = encode_object_hu_moment_measurements(&rows).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, object_hu_moment_measurement_schema());
    assert_eq!(schema.columns()[2].name(), "projection_axis");
    assert_eq!(schema.columns()[5].name(), "hu_0");
    assert_eq!(schema.columns()[11].name(), "hu_6");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("hu.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("hu.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_object_hu_moment_measurements(&env, "hu.rows", 0, VOLUME).unwrap();
    assert_eq!(got, rows);

    assert!(
        encode_object_hu_moment_measurements(&[ObjectHuMomentsMeasurements {
            label: 0,
            projected_count: 1,
            projection_axis: ProjectionAxis::Z,
            centroid: [0.0, 0.0],
            hu: [0.0; 7],
        }])
        .is_err()
    );
    assert!(
        encode_object_hu_moment_measurements(&[ObjectHuMomentsMeasurements {
            label: 3,
            projected_count: 0,
            projection_axis: ProjectionAxis::Y,
            centroid: [0.0, 0.0],
            hu: [0.0; 7],
        }])
        .is_err()
    );
    let mut nonfinite = rows[0];
    nonfinite.hu[3] = f64::NAN;
    assert!(encode_object_hu_moment_measurements(&[nonfinite]).is_err());
}

#[test]
fn object_zernike_moments_use_projected_binary_points() {
    let mut labels = Array3::<f64>::zeros((3, 3, 3));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[1, 1, 0]] = 5.0;
    labels[[2, 1, 0]] = 5.0;

    let set = ObjectZernikeMomentSet::new(2).unwrap();
    let rows =
        object_zernike_moments_measurements_set(labels.view(), ProjectionAxis::Z, set).unwrap();
    assert_eq!(
        object_zernike_moments_measurements(labels.view(), ProjectionAxis::Z, set.max_order())
            .unwrap(),
        rows
    );
    assert_eq!(rows.len(), 2);

    let line = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(line.projected_count, 2);
    assert_eq!(line.projection_axis, ProjectionAxis::Z);
    assert_eq!(line.max_order, 2);
    assert_eq!(line.centroid, [0.0, 0.5]);
    assert_eq!(line.radius, 0.5);
    assert_eq!(
        line.moments
            .iter()
            .map(|moment| (moment.order, moment.repetition))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 1), (2, 0), (2, 2)]
    );
    assert!((line.moments[0].real - 1.0 / std::f64::consts::PI).abs() < 1.0e-12);
    assert!(line.moments[1].magnitude < 1.0e-12);
    assert!((line.moments[2].real - 3.0 / std::f64::consts::PI).abs() < 1.0e-12);
    assert!((line.moments[3].real - 3.0 / std::f64::consts::PI).abs() < 1.0e-12);
    let key = ZernikeMomentKey::new(2, 2).unwrap();
    assert_eq!(key.order(), 2);
    assert_eq!(key.repetition(), 2);
    assert!(ZernikeMomentKey::new(2, 1).is_err());
    assert!(ZernikeMomentKey::new(1, 2).is_err());
    assert!(ZernikeMomentKey::new(usize::MAX, 0).is_err());
    assert!(ZernikeMomentKey::new(13, 1).is_err());
    assert!(ObjectZernikeMomentFeature::real(2, 1).is_err());
    assert_eq!(
        ObjectZernikeMomentFeature::magnitude(2, 2)
            .unwrap()
            .column_name(),
        "zernike_2_2_magnitude"
    );
    let radial_key = ZernikeMomentKey::new(2, 0).unwrap();
    assert_eq!(line.moment_key(radial_key), Some(line.moments[2]));
    assert_eq!(line.moment(2, 0), Some(line.moments[2]));
    assert_eq!(line.moment(2, 1), None);
    assert_eq!(
        line.feature(ObjectZernikeMomentFeature::ProjectedCount),
        Some(2.0)
    );
    assert_eq!(
        line.feature(ObjectZernikeMomentFeature::Centroid1),
        Some(0.5)
    );
    assert_eq!(
        line.feature(ObjectZernikeMomentFeature::real(2, 0).unwrap()),
        Some(line.moments[2].real)
    );
    assert_eq!(
        line.feature(ObjectZernikeMomentFeature::magnitude(2, 2).unwrap()),
        Some(line.moments[3].magnitude)
    );
    assert_eq!(
        line.feature(ObjectZernikeMomentFeature::imag(4, 0).unwrap()),
        None
    );

    let collapsed = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(collapsed.projected_count, 1);
    assert_eq!(collapsed.radius, 0.0);
    assert!(collapsed.moments.iter().all(|moment| moment.imag == 0.0));
}

#[test]
fn object_zernike_moment_rows_have_bounded_schema_and_collector() {
    let rows = vec![ObjectZernikeMomentsMeasurements {
        label: 2,
        projected_count: 3,
        projection_axis: ProjectionAxis::Z,
        max_order: 1,
        centroid: [1.0, 2.0],
        radius: 2.5,
        moments: vec![
            ObjectZernikeMoment {
                order: 0,
                repetition: 0,
                real: 0.1,
                imag: 0.0,
                magnitude: 0.1,
            },
            ObjectZernikeMoment {
                order: 1,
                repetition: 1,
                real: 0.2,
                imag: -0.3,
                magnitude: (0.2f64 * 0.2 + 0.3 * 0.3).sqrt(),
            },
        ],
    }];
    let set = ObjectZernikeMomentSet::new(1).unwrap();
    let encoded = encode_object_zernike_moment_measurements_set(&rows, set).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, object_zernike_moment_measurement_schema(1).unwrap());
    assert_eq!(
        schema,
        object_zernike_moment_measurement_schema_set(set).unwrap()
    );
    assert_eq!(
        encoded,
        encode_object_zernike_moment_measurements(&rows, 1).unwrap()
    );
    assert_eq!(schema.columns()[3].name(), "max_order");
    assert_eq!(schema.columns()[7].name(), "zernike_0_0_real");
    assert_eq!(schema.columns()[12].name(), "zernike_1_1_magnitude");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("zernike.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("zernike.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got =
        collect_object_zernike_moment_measurements(&env, "zernike.rows", 0, VOLUME, 1).unwrap();
    assert_eq!(got, rows);
    let got = collect_object_zernike_moment_measurements_set(&env, "zernike.rows", 0, VOLUME, set)
        .unwrap();
    assert_eq!(got, rows);

    assert!(object_zernike_moment_measurement_schema(13).is_err());
    let mut wrong_order = rows[0].clone();
    wrong_order.max_order = 2;
    assert!(encode_object_zernike_moment_measurements(&[wrong_order], 1).is_err());
    let mut wrong_layout = rows[0].clone();
    wrong_layout.moments[1].repetition = 0;
    assert!(encode_object_zernike_moment_measurements(&[wrong_layout], 1).is_err());
    let mut nonfinite = rows[0].clone();
    nonfinite.moments[0].real = f64::NAN;
    assert!(encode_object_zernike_moment_measurements(&[nonfinite], 1).is_err());
}

#[test]
fn object_zernike3d_descriptors_use_normalized_physical_points() {
    let mut labels = Array3::<f64>::zeros((3, 3, 3));
    labels[[0, 0, 0]] = 2.0;
    labels[[1, 1, 1]] = 5.0;
    labels[[1, 1, 2]] = 5.0;

    let set = ObjectZernike3dSet::new(2).unwrap();
    let rows =
        object_zernike3d_measurements_set(labels.view(), PhysicalSpacing::unit(), set).unwrap();
    assert_eq!(
        object_zernike3d_measurements(labels.view(), PhysicalSpacing::unit(), set.max_order())
            .unwrap(),
        rows
    );
    assert_eq!(rows.len(), 2);

    let point = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(point.count, 1);
    assert_eq!(point.max_order, 2);
    assert_eq!(point.centroid, [0.0, 0.0, 0.0]);
    assert_eq!(point.radius, 0.0);
    assert_eq!(
        point.descriptors,
        vec![
            ObjectZernike3dDescriptor {
                order: 0,
                value: 1.0,
            },
            ObjectZernike3dDescriptor {
                order: 1,
                value: 0.0,
            },
            ObjectZernike3dDescriptor {
                order: 2,
                value: 0.0,
            },
        ]
    );

    let line = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(line.count, 2);
    assert_eq!(line.centroid, [1.0, 1.0, 1.5]);
    assert_eq!(line.radius, 0.5);
    assert_eq!(line.descriptor(0).unwrap().value, 1.0);
    assert_eq!(line.descriptor(1).unwrap().value, 2.0);
    assert_eq!(line.descriptor(2).unwrap().value, 3.0);
    assert_eq!(Zernike3dKey::new(2).unwrap().order(), 2);
    assert!(Zernike3dKey::new(7).is_err());
    assert_eq!(
        ObjectZernike3dFeature::descriptor(2).unwrap().column_name(),
        "zernike3d_2_descriptor"
    );
    assert_eq!(line.feature(ObjectZernike3dFeature::Count), Some(2.0));
    assert_eq!(
        line.feature(ObjectZernike3dFeature::Descriptor(
            Zernike3dKey::new(2).unwrap()
        )),
        Some(line.descriptors[2].value)
    );

    let scaled = object_zernike3d_measurements_set(
        labels.view(),
        PhysicalSpacing::new([2.0, 3.0, 5.0]).unwrap(),
        set,
    )
    .unwrap();
    let scaled_point = scaled.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(scaled_point.centroid, [0.0, 0.0, 0.0]);
    assert_eq!(scaled_point.descriptors, point.descriptors);
}

#[test]
fn object_zernike3d_rows_have_bounded_schema_and_collector() {
    let rows = vec![ObjectZernike3dMeasurements {
        label: 2,
        count: 3,
        max_order: 2,
        centroid: [1.0, 2.0, 3.0],
        radius: 4.0,
        descriptors: vec![
            ObjectZernike3dDescriptor {
                order: 0,
                value: 1.0,
            },
            ObjectZernike3dDescriptor {
                order: 1,
                value: 0.5,
            },
            ObjectZernike3dDescriptor {
                order: 2,
                value: 0.25,
            },
        ],
    }];
    let set = ObjectZernike3dSet::new(2).unwrap();
    let encoded = encode_object_zernike3d_measurements_set(&rows, set).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(
        schema,
        object_zernike3d_measurement_schema_set(set).unwrap()
    );
    assert_eq!(schema.columns()[2].name(), "max_order");
    assert_eq!(schema.columns()[7].name(), "zernike3d_0_descriptor");
    assert_eq!(schema.columns()[9].name(), "zernike3d_2_descriptor");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("zernike3d.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("zernike3d.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_object_zernike3d_measurements(&env, "zernike3d.rows", 0, VOLUME, 2).unwrap();
    assert_eq!(got, rows);
    let got =
        collect_object_zernike3d_measurements_set(&env, "zernike3d.rows", 0, VOLUME, set).unwrap();
    assert_eq!(got, rows);

    let mut wrong_order = rows[0].clone();
    wrong_order.max_order = 1;
    assert!(encode_object_zernike3d_measurements_set(&[wrong_order], set).is_err());
    let mut wrong_layout = rows[0].clone();
    wrong_layout.descriptors[1].order = 2;
    assert!(encode_object_zernike3d_measurements_set(&[wrong_layout], set).is_err());
    let mut nonfinite = rows[0].clone();
    nonfinite.descriptors[0].value = f64::NAN;
    assert!(encode_object_zernike3d_measurements_set(&[nonfinite], set).is_err());
}

#[test]
fn object_weighted_hu_moments_accumulate_projected_weights() {
    let mut labels = Array3::<f64>::zeros((2, 2, 2));
    labels[[0, 0, 0]] = 2.0;
    labels[[1, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[0, 1, 1]] = 5.0;

    let mut weights = Array3::<f64>::zeros((2, 2, 2));
    weights[[0, 0, 0]] = 1.0;
    weights[[1, 0, 0]] = 1.0;
    weights[[0, 0, 1]] = 2.0;
    weights[[0, 1, 0]] = 0.0;
    weights[[0, 1, 1]] = f64::NAN;

    let rows =
        object_weighted_hu_moments_measurements(labels.view(), weights.view(), ProjectionAxis::Z)
            .unwrap();
    assert_eq!(rows.len(), 2);

    let weighted = rows.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(weighted.projected_count, 2);
    assert_eq!(weighted.finite_weight_count, 3);
    assert_eq!(weighted.nonfinite_weight_count, 0);
    assert_eq!(weighted.weight_sum, 4.0);
    assert_eq!(weighted.centroid, Some([0.0, 0.5]));
    let hu = weighted.hu.unwrap();
    assert_eq!(hu[0], 1.0 / 16.0);
    assert_eq!(hu[1], 1.0 / 256.0);
    assert_eq!(&hu[2..], &[0.0; 5]);

    let zero = rows.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(zero.projected_count, 0);
    assert_eq!(zero.finite_weight_count, 1);
    assert_eq!(zero.nonfinite_weight_count, 1);
    assert_eq!(zero.weight_sum, 0.0);
    assert_eq!(zero.centroid, None);
    assert_eq!(zero.hu, None);
}

#[test]
fn object_weighted_hu_moments_validate_inputs() {
    let mut labels = Array3::<f64>::zeros((1, 1, 2));
    labels[[0, 0, 0]] = 2.0;
    let mut weights = Array3::<f64>::ones((1, 1, 2));
    let wrong_shape = Array3::<f64>::ones((1, 1, 3));
    assert!(object_weighted_hu_moments_measurements(
        labels.view(),
        wrong_shape.view(),
        ProjectionAxis::Z
    )
    .is_err());

    labels[[0, 0, 1]] = 1.5;
    assert!(object_weighted_hu_moments_measurements(
        labels.view(),
        weights.view(),
        ProjectionAxis::Z
    )
    .is_err());

    labels[[0, 0, 1]] = 0.0;
    weights[[0, 0, 0]] = -1.0;
    assert!(object_weighted_hu_moments_measurements(
        labels.view(),
        weights.view(),
        ProjectionAxis::Z
    )
    .is_err());
}

#[test]
fn object_weighted_hu_moment_rows_have_canonical_schema_and_collector() {
    let rows = vec![
        ObjectWeightedHuMomentsMeasurements {
            label: 2,
            projected_count: 2,
            finite_weight_count: 3,
            nonfinite_weight_count: 0,
            projection_axis: ProjectionAxis::Z,
            weight_sum: 4.0,
            centroid: Some([0.0, 0.5]),
            hu: Some([1.0 / 16.0, 1.0 / 256.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        ObjectWeightedHuMomentsMeasurements {
            label: 5,
            projected_count: 0,
            finite_weight_count: 1,
            nonfinite_weight_count: 1,
            projection_axis: ProjectionAxis::Y,
            weight_sum: 0.0,
            centroid: None,
            hu: None,
        },
    ];
    let encoded = encode_object_weighted_hu_moment_measurements(&rows).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, object_weighted_hu_moment_measurement_schema());
    assert_eq!(schema.columns()[6].name(), "has_centroid");
    assert_eq!(schema.columns()[9].name(), "has_hu");
    assert_eq!(schema.columns()[16].name(), "hu_6");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("weighted-hu.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("weighted-hu.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_object_weighted_hu_moment_measurements(&env, "weighted-hu.rows", 0, VOLUME)
        .unwrap();
    assert_eq!(got, rows);
    assert_eq!(
        ObjectWeightedHuMomentFeature::FiniteWeightCount.column_name(),
        "finite_weight_count"
    );
    assert!(ObjectWeightedHuMomentFeature::hu(7).is_err());
    assert_eq!(
        ObjectWeightedHuMomentFeature::hu(6).unwrap().column_name(),
        "hu_6"
    );
    let hu0 = HuMomentIndex::new(0).unwrap();
    assert_eq!(got[0].hu_key(hu0), got[0].hu.map(|hu| hu[0]));
    assert_eq!(got[0].hu(0), got[0].hu.map(|hu| hu[0]));
    assert_eq!(got[0].hu(7), None);
    assert_eq!(
        got[0].feature(ObjectWeightedHuMomentFeature::ProjectedCount),
        Some(2.0)
    );
    assert_eq!(
        got[0].feature(ObjectWeightedHuMomentFeature::WeightSum),
        Some(4.0)
    );
    assert_eq!(
        got[0].feature(ObjectWeightedHuMomentFeature::Centroid1),
        Some(0.5)
    );
    assert_eq!(
        got[0].feature(ObjectWeightedHuMomentFeature::hu(0).unwrap()),
        got[0].hu.map(|hu| hu[0])
    );
    assert_eq!(
        got[1].feature(ObjectWeightedHuMomentFeature::Centroid0),
        None
    );
    assert_eq!(
        got[1].feature(ObjectWeightedHuMomentFeature::hu(0).unwrap()),
        None
    );
    assert_eq!(got[1].hu_key(hu0), None);

    let mut missing_hu = rows[0];
    missing_hu.hu = None;
    assert!(encode_object_weighted_hu_moment_measurements(&[missing_hu]).is_err());

    let mut zero_with_centroid = rows[1];
    zero_with_centroid.centroid = Some([0.0, 0.0]);
    zero_with_centroid.hu = Some([0.0; 7]);
    assert!(encode_object_weighted_hu_moment_measurements(&[zero_with_centroid]).is_err());

    let mut too_many_projected = rows[0];
    too_many_projected.projected_count = 4;
    assert!(encode_object_weighted_hu_moment_measurements(&[too_many_projected]).is_err());
}

fn planned_object_hu(
    block: [usize; 3],
    projection_axis: ProjectionAxis,
) -> Vec<ObjectHuMomentsMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = ObjectHuMomentsOp::new(
        "measure planned Hu moments",
        0usize,
        projection_axis,
        "hu.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned Hu moments",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_object_hu_moment_measurements(&env, "hu.planned.rows", 1, VOLUME).unwrap()
}

fn planned_object_zernike(
    block: [usize; 3],
    projection_axis: ProjectionAxis,
    max_order: usize,
) -> Vec<ObjectZernikeMomentsMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let set = ObjectZernikeMomentSet::new(max_order).unwrap();
    let op = ObjectZernikeMomentsOp::new_set(
        "measure planned Zernike moments",
        0usize,
        projection_axis,
        set,
        "zernike.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned Zernike moments",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_object_zernike_moment_measurements_set(&env, "zernike.planned.rows", 1, VOLUME, set)
        .unwrap()
}

fn planned_builder_object_zernike3d(
    block: [usize; 3],
    spacing: PhysicalSpacing,
    set: ObjectZernike3dSet,
) -> Vec<ObjectZernike3dMeasurements> {
    let plan = Measurements::for_labels(0usize)
        .object_zernike3d_set(spacing, set)
        .unwrap()
        .stream("zernike3d-planned")
        .build(base(block))
        .unwrap();
    let rows = plan.object_zernike3d_rows_with_contract().unwrap();
    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "planned builder 3D Zernike descriptors",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();
    collect_object_zernike3d_rows_with_contract(&env, &rows, VOLUME).unwrap()
}

fn planned_object_weighted_hu(
    block: [usize; 3],
    projection_axis: ProjectionAxis,
) -> Vec<ObjectWeightedHuMomentsMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = ObjectWeightedHuMomentsOp::new(
        "measure planned weighted Hu moments",
        0usize,
        ImageId::supplied(0),
        projection_axis,
        "weighted-hu.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![coordinate_values()],
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    execute_phases(
        "planned weighted Hu moments",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_object_weighted_hu_moment_measurements(&env, "weighted-hu.planned.rows", 1, VOLUME)
        .unwrap()
}

#[test]
fn planned_object_hu_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_object_hu(VOLUME, ProjectionAxis::Z);
    let split = planned_object_hu([2, 2, 2], ProjectionAxis::Z);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference = object_hu_moments_measurements(labels.view(), ProjectionAxis::Z).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn planned_object_zernike_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_object_zernike(VOLUME, ProjectionAxis::Z, 3);
    let split = planned_object_zernike([2, 2, 2], ProjectionAxis::Z, 3);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference =
        object_zernike_moments_measurements(labels.view(), ProjectionAxis::Z, 3).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn planned_object_zernike3d_is_decomposition_invariant_and_matches_resident_reference() {
    let spacing = PhysicalSpacing::new([1.0, 2.0, 3.0]).unwrap();
    let set = ObjectZernike3dSet::new(3).unwrap();
    let coarse = planned_builder_object_zernike3d(VOLUME, spacing, set);
    let split = planned_builder_object_zernike3d([2, 2, 2], spacing, set);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference = object_zernike3d_measurements_set(labels.view(), spacing, set).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn planned_object_weighted_hu_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_object_weighted_hu(VOLUME, ProjectionAxis::Z);
    let split = planned_object_weighted_hu([2, 2, 2], ProjectionAxis::Z);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let weights = coordinate_values().view::<f64>().unwrap().to_owned();
    let reference =
        object_weighted_hu_moments_measurements(labels.view(), weights.view(), ProjectionAxis::Z)
            .unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_object_hu_and_weighted_hu() {
    let labels_array = labels().view::<f64>().unwrap().to_owned();
    let weights = coordinate_values();
    let weights_array = weights.view::<f64>().unwrap().to_owned();
    let expected_hu =
        object_hu_moments_measurements(labels_array.view(), ProjectionAxis::Z).unwrap();
    let expected_weighted = object_weighted_hu_moments_measurements(
        labels_array.view(),
        weights_array.view(),
        ProjectionAxis::Z,
    )
    .unwrap();

    let plan = Measurements::for_labels(0usize)
        .object_hu_moments(ProjectionAxis::Z)
        .object_weighted_hu_moments(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            ProjectionAxis::Z,
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_hu_moment_rows_phase(), Some(2));
    assert_eq!(plan.object_weighted_hu_moment_rows_phase(), Some(4));
    assert_eq!(
        plan.object_hu_moment_contract(),
        Some(ObjectProjectionContract::new(ProjectionAxis::Z))
    );
    assert_eq!(
        plan.object_weighted_hu_moment_contract(),
        Some(ObjectProjectionContract::new(ProjectionAxis::Z))
    );
    assert_eq!(
        plan.object_hu_moment_projection_axis(),
        Some(ProjectionAxis::Z)
    );
    assert_eq!(
        plan.object_weighted_hu_moment_projection_axis(),
        Some(ProjectionAxis::Z)
    );
    let hu_rows = plan.object_hu_moment_rows_with_contract().unwrap();
    assert_eq!(hu_rows.stream(), plan.object_hu_moment_stream().unwrap());
    assert_eq!(hu_rows.phase(), plan.object_hu_moment_rows_phase().unwrap());
    assert_eq!(
        hu_rows.contract(),
        ObjectProjectionContract::new(ProjectionAxis::Z)
    );
    let weighted_rows = plan.object_weighted_hu_moment_rows_with_contract().unwrap();
    assert_eq!(
        weighted_rows.stream(),
        plan.object_weighted_hu_moment_stream().unwrap()
    );
    assert_eq!(
        weighted_rows.phase(),
        plan.object_weighted_hu_moment_rows_phase().unwrap()
    );
    assert_eq!(
        weighted_rows.contract(),
        ObjectProjectionContract::new(ProjectionAxis::Z)
    );

    let env =
        ArrayEnvironment::with_inputs(labels(), vec![weights], &plan.decomposition, [2, 2, 2])
            .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder Hu moments",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got_hu = collect_object_hu_moment_rows_with_contract(&env, &hu_rows, VOLUME).unwrap();
    let got_weighted =
        collect_object_weighted_hu_moment_rows_with_contract(&env, &weighted_rows, VOLUME).unwrap();
    assert_eq!(got_hu, expected_hu);
    assert_eq!(got_weighted, expected_weighted);
}

#[test]
fn measurement_builder_runs_planned_object_zernike() {
    let labels_array = labels().view::<f64>().unwrap().to_owned();
    let expected =
        object_zernike_moments_measurements(labels_array.view(), ProjectionAxis::Z, 3).unwrap();

    let plan = Measurements::for_labels(0usize)
        .object_zernike_moments(ProjectionAxis::Z, 3)
        .unwrap()
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_zernike_moment_rows_phase(), Some(2));
    assert_eq!(plan.object_zernike_moment_max_order(), Some(3));
    assert_eq!(
        plan.object_zernike_moment_set(),
        Some(ObjectZernikeMomentSet::new(3).unwrap())
    );
    assert_eq!(
        plan.object_zernike_moment_contract(),
        Some(ObjectZernikeMomentContract::new(
            ProjectionAxis::Z,
            ObjectZernikeMomentSet::new(3).unwrap()
        ))
    );
    assert_eq!(
        plan.object_zernike_moment_projection_axis(),
        Some(ProjectionAxis::Z)
    );
    let rows = plan.object_zernike_moment_rows_with_contract().unwrap();
    assert_eq!(rows.stream(), plan.object_zernike_moment_stream().unwrap());
    assert_eq!(
        rows.phase(),
        plan.object_zernike_moment_rows_phase().unwrap()
    );
    assert_eq!(
        rows.contract(),
        ObjectZernikeMomentContract::new(
            ProjectionAxis::Z,
            ObjectZernikeMomentSet::new(3).unwrap()
        )
    );
    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder Zernike moments",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_object_zernike_moment_rows_with_contract(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn measurement_builder_runs_planned_object_zernike3d() {
    let spacing = PhysicalSpacing::new([1.0, 2.0, 3.0]).unwrap();
    let labels_array = labels().view::<f64>().unwrap().to_owned();
    let set = ObjectZernike3dSet::new(3).unwrap();
    let expected = object_zernike3d_measurements_set(labels_array.view(), spacing, set).unwrap();

    let plan = Measurements::for_labels(0usize)
        .object_zernike3d_set(spacing, set)
        .unwrap()
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.object_zernike3d_rows_phase(), Some(2));
    assert_eq!(plan.object_zernike3d_set(), Some(set));
    assert_eq!(
        plan.object_zernike3d_contract(),
        Some(ObjectZernike3dContract::new(spacing, set))
    );
    assert_eq!(plan.object_zernike3d_spacing(), Some(spacing));
    let rows = plan.object_zernike3d_rows_with_contract().unwrap();
    assert_eq!(rows.stream(), plan.object_zernike3d_stream().unwrap());
    assert_eq!(rows.phase(), plan.object_zernike3d_rows_phase().unwrap());
    assert_eq!(rows.contract(), ObjectZernike3dContract::new(spacing, set));

    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder 3D Zernike descriptors",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_object_zernike3d_rows_with_contract(&env, &rows, VOLUME).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn object_measure_extensions_run_through_framework_materialized_objects() {
    let mut labels = Array3::<f64>::zeros((2, 2, 2));
    labels[[0, 0, 0]] = 3.0;
    labels[[0, 0, 1]] = 3.0;
    labels[[1, 1, 1]] = 4.0;

    let measure = ObjectPointCountMeasure;
    assert_eq!(measure.key().as_str(), "test.point_count");
    assert_object_measurement_invariant(labels.view(), &measure).unwrap();

    let first = run_object_measure(labels.view(), &measure).unwrap();
    let second = run_object_measure(labels.view(), &measure).unwrap();
    assert_eq!(first, second);
    let schema = encoded_schema(&first).unwrap();
    assert_eq!(schema.columns()[0].name(), "label");
    assert_eq!(schema.columns()[1].name(), "points");

    let cube = Array3::<f64>::from_elem((3, 3, 3), 9.0);
    let boundary_measure = ObjectBoundaryPointCountMeasure;
    assert_object_measurement_invariant(cube.view(), &boundary_measure).unwrap();
    let boundary_rows = run_object_measure(cube.view(), &boundary_measure).unwrap();
    let boundary_schema = encoded_schema(&boundary_rows).unwrap();
    assert_eq!(boundary_schema.columns()[1].name(), "boundary_points");

    let mut l_shape = Array3::<f64>::zeros((1, 2, 2));
    l_shape[[0, 0, 0]] = 2.0;
    l_shape[[0, 0, 1]] = 2.0;
    l_shape[[0, 1, 0]] = 2.0;
    let hull_measure = ObjectProjectedHullMeasure;
    assert_object_measurement_invariant(l_shape.view(), &hull_measure).unwrap();
    let hull_rows = run_object_measure(l_shape.view(), &hull_measure).unwrap();
    let hull_schema = encoded_schema(&hull_rows).unwrap();
    assert_eq!(hull_schema.columns()[1].name(), "hull_vertices");

    labels[[1, 0, 0]] = 2.5;
    assert!(run_object_measure(labels.view(), &measure).is_err());
}

#[test]
fn object_measure_extensions_validate_inputs_and_rows() {
    let mut labels = Array3::<f64>::zeros((1, 1, 1));
    labels[[0, 0, 0]] = 1.0;

    assert!(MeasurementKey::new("").is_err());
    assert!(MeasurementKey::new("bad key").is_err());
    assert!(MeasurementKey::new(".bad").is_err());
    assert!(MeasurementKey::new("bad.").is_err());
    assert!(MeasurementKey::new("bad..key").is_err());
    assert!(MeasurementKey::new("bad/key").is_err());
    assert!(MeasurementKey::new("good_key-1.part2").is_ok());
    assert!(run_object_measure(labels.view(), &EmptyObjectMeasure).is_err());
    assert!(run_object_measure(labels.view(), &WrongValueObjectMeasure).is_err());
    let error = run_object_measure(labels.view(), &BadSchemaObjectMeasure)
        .unwrap_err()
        .to_string();
    assert!(error.contains("schema must start with a u64 \"label\" column"));
}

#[test]
fn measurement_builder_runs_planned_custom_object_measure() {
    let mut labels = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 3.0;
    labels[[0, 0, 1]] = 3.0;
    labels[[5, 3, 2]] = 4.0;
    let measure = Arc::new(ObjectPointCountMeasure);
    let plan = Measurements::for_labels(0usize)
        .custom_object(measure.clone())
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    let rows = plan.custom_object_rows(0).unwrap();
    assert_eq!(rows.stream(), plan.custom_object_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.custom_object_rows_phase(0).unwrap());
    assert_eq!(rows.phase(), 2);

    let env = ArrayEnvironment::new(
        labels.clone().into(),
        plan.decomposition.n_phases(),
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder custom object measure",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_custom_object_rows(&env, &rows, VOLUME, measure.schema(), |row| {
        Ok((row.u64(0)?, row.u64(1)?))
    })
    .unwrap();
    assert_eq!(got, vec![(3, 2), (4, 1)]);

    let resident = run_object_measure(labels.view(), measure.as_ref()).unwrap();
    let resident = decode_custom_measurement_rows(VOLUME, measure.schema(), &resident, |row| {
        Ok((row.u64(0)?, row.u64(1)?))
    })
    .unwrap();
    assert_eq!(got, resident);
}

#[test]
fn custom_object_measure_can_request_3d_convex_hull_prerequisite() {
    let spacing = PhysicalSpacing::unit();
    let labels = convex_hull_labels().view::<f64>().unwrap().to_owned();
    let measure = Arc::new(ObjectConvexHullPrereqMeasure { spacing });
    let expected = object_convex_hull_measurements(labels.view(), spacing)
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row.label,
                row.hull_vertices as u64,
                row.hull_faces as u64,
                row.surface_area,
                row.volume,
                row.max_hull_feret_diameter,
            )
        })
        .collect::<Vec<_>>();

    let resident = run_object_measure(labels.view(), measure.as_ref()).unwrap();
    let resident = decode_custom_measurement_rows(VOLUME, measure.schema(), &resident, |row| {
        Ok((
            row.u64(0)?,
            row.u64(1)?,
            row.u64(2)?,
            row.f64(3)?,
            row.f64(4)?,
            row.f64(5)?,
        ))
    })
    .unwrap();
    assert_eq!(resident, expected);

    let plan = Measurements::for_labels(0usize)
        .custom_object(measure.clone())
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    let rows = plan.custom_object_rows(0).unwrap();
    assert_eq!(rows.stream(), plan.custom_object_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.custom_object_rows_phase(0).unwrap());

    let env = ArrayEnvironment::new(
        labels.clone().into(),
        plan.decomposition.n_phases(),
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder custom object convex-hull prerequisite",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let planned = collect_custom_object_rows(&env, &rows, VOLUME, measure.schema(), |row| {
        Ok((
            row.u64(0)?,
            row.u64(1)?,
            row.u64(2)?,
            row.f64(3)?,
            row.f64(4)?,
            row.f64(5)?,
        ))
    })
    .unwrap();
    assert_eq!(planned, expected);
}

#[test]
fn measurement_builder_validates_custom_object_measures() {
    let duplicate = Measurements::for_labels(0usize)
        .custom_object(Arc::new(ObjectPointCountMeasure))
        .custom_object(Arc::new(ObjectPointCountMeasure))
        .build(base([2, 2, 2]));
    let error = match duplicate {
        Ok(_) => panic!("duplicate custom object key unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("requested more than once"));

    let empty = Measurements::for_labels(0usize)
        .custom_object(Arc::new(EmptyObjectMeasure))
        .build(base([2, 2, 2]));
    let error = match empty {
        Ok(_) => panic!("empty custom object measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("requested no object inputs"));

    let bad_schema = Measurements::for_labels(0usize)
        .custom_object(Arc::new(BadSchemaObjectMeasure))
        .build(base([2, 2, 2]));
    let error = match bad_schema {
        Ok(_) => panic!("bad-schema custom object measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("schema must start with a u64 \"label\" column"));

    let colliding_stream = Measurements::for_labels(0usize)
        .custom_object(Arc::new(CollidingStreamObjectMeasure))
        .build(base([2, 2, 2]));
    let error = match colliding_stream {
        Ok(_) => panic!("stream-colliding custom object measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("generates duplicate stream"));

    let bad_cost = Measurements::for_labels(0usize)
        .custom_object(Arc::new(BadCostObjectMeasure))
        .build(base([2, 2, 2]));
    let error = match bad_cost {
        Ok(_) => panic!("bad-cost custom object measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("invalid planner cost"));

    let empty_stream = ObjectMeasureMergeOp::new(
        "merge custom object measure",
        "custom.object.points",
        0,
        [1, 1, 1],
        Arc::new(ObjectPointCountMeasure),
        "",
        Lifecycle::DeleteOnExit,
    );
    let error = match empty_stream {
        Ok(_) => panic!("empty custom object stream unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("output stream must not be empty"));

    let bad_cost_stream = ObjectMeasureMergeOp::new(
        "merge custom object measure",
        "custom.object.points",
        0,
        [1, 1, 1],
        Arc::new(BadCostObjectMeasure),
        "custom.object.rows",
        Lifecycle::DeleteOnExit,
    );
    let error = match bad_cost_stream {
        Ok(_) => panic!("bad-cost custom object op unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("invalid planner cost"));

    let op = ObjectMeasureMergeOp::new(
        "merge custom object measure",
        "custom.object.points",
        0,
        [1, 1, 1],
        Arc::new(ObjectPointCountMeasure),
        "custom.object.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(op.cost_per_voxel(), 11.0);
    assert_eq!(op.fold_law(), FoldLaw::ExactAssociative);

    let plan = Measurements::for_labels(0usize)
        .custom_object(Arc::new(ObjectPointCountMeasure))
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(
        plan.custom_object_fold_law(0),
        Some(FoldLaw::ExactAssociative)
    );
    assert_eq!(plan.custom_object_fold_law(1), None);
}

#[test]
fn region_measure_extensions_run_label_local_accumulators() {
    let mut labels = Array3::<f64>::zeros((2, 2, 3));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[1, 0, 0]] = 2.0;

    let mut values = Array3::<f64>::zeros((2, 2, 3));
    values[[0, 0, 0]] = 1.0;
    values[[0, 0, 1]] = 3.0;
    values[[0, 1, 0]] = 10.0;
    values[[1, 0, 0]] = 7.0;

    let measure = RegionSumMeasure;
    assert_region_measurement_invariant(labels.view(), &[values.view()], &measure).unwrap();
    assert_region_measurement_decomposition_invariant(
        labels.view(),
        &[values.view()],
        &measure,
        &[[1, 1, 1], [1, 2, 2], [2, 1, 3]],
    )
    .unwrap();
    assert_region_measurement_invariant(labels.view(), &[], &RegionCountMeasure).unwrap();
    let encoded = run_region_measure(labels.view(), &[values.view()], &measure).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema.columns()[0].name(), "label");
    assert_eq!(schema.columns()[1].name(), "count");
    assert_eq!(schema.columns()[2].name(), "sum");

    labels[[0, 1, 1]] = 1.5;
    assert!(run_region_measure(labels.view(), &[values.view()], &measure).is_err());
}

#[test]
fn planned_region_measure_op_matches_resident_extension() {
    let mut labels = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[1, 0, 0]] = 2.0;

    let mut values = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    values[[0, 0, 0]] = 1.0;
    values[[0, 0, 1]] = 3.0;
    values[[0, 1, 0]] = 10.0;
    values[[1, 0, 0]] = 7.0;

    let mut decomposition = base([2, 2, 2]);
    let grid = decomposition.phases[0].grid.clone();
    let op = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        vec![ImageId::supplied(0)],
        RegionSumMeasure,
        "region.custom.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding_labels(Dtype::F64)
    .holding_source(0, Dtype::F64)
    .unwrap();
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::with_inputs(
        labels.clone().into(),
        vec![values.clone().into()],
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    execute_phases(
        "planned custom region measure",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();

    let measure = RegionSumMeasure;
    let mut table = Table::new(VOLUME, measure.schema()).unwrap();
    fold_fragments(&env, "region.custom.rows", &mut |key, bytes| {
        if key.phase == 1 {
            table.write(key.block, bytes)?;
        }
        Ok(())
    })
    .unwrap();
    table.seal().unwrap();
    let got = table
        .scan(&Region::whole(&VOLUME))
        .unwrap()
        .map(|row| Ok((row.u64(0)?, row.u64(1)?, row.f64(2)?)))
        .collect::<Result<Vec<_>>>()
        .unwrap();

    let resident = run_region_measure(labels.view(), &[values.view()], &measure).unwrap();
    let resident = decode_custom_measurement_rows(VOLUME, measure.schema(), &resident, |row| {
        Ok((row.u64(0)?, row.u64(1)?, row.f64(2)?))
    })
    .unwrap();
    assert_eq!(got, resident);
}

fn custom_region_fixture() -> (Array3<f64>, Array3<f64>) {
    let mut labels = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[1, 0, 0]] = 2.0;

    let mut values = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    values[[0, 0, 0]] = 1.0;
    values[[0, 0, 1]] = 3.0;
    values[[0, 1, 0]] = 10.0;
    values[[1, 0, 0]] = 7.0;

    (labels, values)
}

fn custom_region_row_tuples(encoded: &[u8], measure: &RegionSumMeasure) -> Vec<(u64, u64, f64)> {
    decode_custom_measurement_rows(VOLUME, measure.schema(), encoded, |row| {
        Ok((row.u64(0)?, row.u64(1)?, row.f64(2)?))
    })
    .unwrap()
}

fn collect_custom_region_builder_rows(block: [usize; 3]) -> Vec<(u64, u64, f64)> {
    let (labels, values) = custom_region_fixture();
    let plan = Measurements::for_labels(0usize)
        .custom_region(
            RegionSumMeasure,
            vec![IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64)],
        )
        .stream("custom")
        .build(base(block))
        .unwrap();
    let rows = plan.custom_region_rows(0).unwrap();
    assert_eq!(rows.stream(), plan.custom_region_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.custom_region_rows_phase(0).unwrap());

    let env = ArrayEnvironment::with_inputs(
        labels.into(),
        vec![values.into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder custom region measure",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let measure = RegionSumMeasure;
    collect_custom_region_rows(&env, &rows, VOLUME, measure.schema(), |row| {
        Ok((row.u64(0)?, row.u64(1)?, row.f64(2)?))
    })
    .unwrap()
}

fn resident_custom_region_rows() -> Vec<(u64, u64, f64)> {
    let (labels, values) = custom_region_fixture();
    let measure = RegionSumMeasure;
    let resident = run_region_measure(labels.view(), &[values.view()], &measure).unwrap();
    custom_region_row_tuples(&resident, &measure)
}

#[test]
fn planned_custom_region_builder_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = collect_custom_region_builder_rows(VOLUME);
    let split_xy = collect_custom_region_builder_rows([2, 2, 3]);
    let split_xyz = collect_custom_region_builder_rows([2, 2, 2]);

    assert_eq!(coarse, split_xy);
    assert_eq!(split_xy, split_xyz);
    assert_eq!(split_xyz, resident_custom_region_rows());
}

#[test]
fn public_custom_region_builder_harness_checks_planned_execution() {
    let (labels, values) = custom_region_fixture();
    assert_custom_region_builder_invariant(
        labels.view(),
        &[values.view()],
        RegionSumMeasure,
        &[VOLUME, [2, 2, 3], [2, 2, 2]],
    )
    .unwrap();
}

#[test]
fn measurement_builder_runs_planned_custom_region_measure() {
    let mut labels = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 0]] = 5.0;
    labels[[1, 0, 0]] = 2.0;

    let mut values = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    values[[0, 0, 0]] = 1.0;
    values[[0, 0, 1]] = 3.0;
    values[[0, 1, 0]] = 10.0;
    values[[1, 0, 0]] = 7.0;

    let plan = Measurements::for_labels(0usize)
        .custom_region(
            RegionSumMeasure,
            vec![IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64)],
        )
        .stream("custom")
        .build(base([2, 2, 2]))
        .unwrap();
    let rows = plan.custom_region_rows(0).unwrap();
    assert_eq!(rows.stream(), plan.custom_region_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.custom_region_rows_phase(0).unwrap());
    assert_eq!(rows.phase(), 1);

    let env = ArrayEnvironment::with_inputs(
        labels.clone().into(),
        vec![values.clone().into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder custom region measure",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let measure = RegionSumMeasure;
    let got = collect_custom_region_rows(&env, &rows, VOLUME, measure.schema(), |row| {
        Ok((row.u64(0)?, row.u64(1)?, row.f64(2)?))
    })
    .unwrap();

    let resident = run_region_measure(labels.view(), &[values.view()], &measure).unwrap();
    let resident = custom_region_row_tuples(&resident, &measure);
    assert_eq!(got, resident);
}

#[test]
fn custom_region_plan_has_simulator_visible_full_volume_extension_cost() {
    let plan = Measurements::for_labels(0usize)
        .custom_region(
            RegionSumMeasure,
            vec![IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64)],
        )
        .stream("custom")
        .build(base([2, 2, 2]))
        .unwrap();
    let rows = plan.custom_region_rows(0).unwrap();
    assert_eq!(rows.phase(), 1);
    assert_eq!(
        plan.decomposition.n_phases(),
        2,
        "custom region extensions currently use one full-volume row phase after the source phase"
    );

    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    let outcome = Run::new(&plan.decomposition, &work)
        .machine(Machine {
            workers: 4,
            ..Machine::default()
        })
        .rates(Rates::default())
        .go(&mut PlanOrder)
        .unwrap();
    assert!(
        outcome.tasks_run > 0 && outcome.fetched_bytes > 0,
        "the simulator must see custom-region full-volume label/source reads before a compact \
         partial/merge replacement can be justified"
    );
}

#[test]
fn region_measure_extensions_validate_sources_fold_law_and_rows() {
    assert!(MeasureSource::intensity("").is_err());
    assert!(MeasureSource::intensity("bad source").is_err());
    assert!(MeasureSource::intensity("bad/source").is_err());
    assert!(MeasureSource::intensity("bad..source").is_err());
    assert!(MeasureSource::intensity("good.source-0").is_ok());
    assert!(ApproxTolerance::new(f64::NAN).is_err());
    assert!(ApproxTolerance::new(-1.0).is_err());
    assert_eq!(ApproxTolerance::new(0.25).unwrap().get(), 0.25);
    assert!(FoldLaw::approximate(f64::INFINITY).is_err());

    let labels = Array3::<f64>::from_elem((1, 1, 1), 1.0);
    let values = Array3::<f64>::from_elem((1, 1, 1), 2.0);
    let wrong_shape = Array3::<f64>::zeros((1, 1, 2));
    let measure = RegionSumMeasure;

    assert!(run_region_measure(labels.view(), &[], &measure).is_err());
    assert!(run_region_measure(labels.view(), &[wrong_shape.view()], &measure).is_err());
    assert!(run_region_measure(labels.view(), &[values.view()], &OrderedRegionMeasure).is_err());
    let error = run_region_measure(
        labels.view(),
        &[values.view(), values.view()],
        &DuplicateSourceRegionMeasure,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("declares source \"channel0\" more than once"));
    assert!(run_region_measure(labels.view(), &[], &WrongRowRegionMeasure).is_err());
    let error = run_region_measure(labels.view(), &[], &BadSchemaRegionMeasure)
        .unwrap_err()
        .to_string();
    assert!(error.contains("schema must start with a u64 \"label\" column"));

    let drift_labels = Array3::<f64>::from_elem((2, 1, 1), 1.0);
    let error = assert_region_measurement_decomposition_invariant(
        drift_labels.view(),
        &[],
        &DriftRegionMeasure,
        &[[1, 1, 1]],
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("within tolerance 0.25"));

    let empty_stream = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        vec![ImageId::supplied(0)],
        RegionSumMeasure,
        "",
        Lifecycle::DeleteOnExit,
    );
    let error = match empty_stream {
        Ok(_) => panic!("empty region measure stream unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("output stream must not be empty"));

    let bad_schema = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        Vec::new(),
        BadSchemaRegionMeasure,
        "custom.region.rows",
        Lifecycle::DeleteOnExit,
    );
    let error = match bad_schema {
        Ok(_) => panic!("bad-schema custom region measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("schema must start with a u64 \"label\" column"));

    let bad_cost = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        Vec::new(),
        BadCostRegionMeasure,
        "custom.region.rows",
        Lifecycle::DeleteOnExit,
    );
    let error = match bad_cost {
        Ok(_) => panic!("bad-cost custom region measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("invalid planner cost"));

    let duplicate_sources = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        vec![ImageId::supplied(0), ImageId::supplied(1)],
        DuplicateSourceRegionMeasure,
        "custom.region.rows",
        Lifecycle::DeleteOnExit,
    );
    let error = match duplicate_sources {
        Ok(_) => panic!("duplicate-source custom region measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("declares source \"channel0\" more than once"));

    let same_source_op = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        vec![ImageId::from(0usize)],
        RegionSumMeasure,
        "custom.region.rows",
        Lifecycle::DeleteOnExit,
    );
    let error = match same_source_op {
        Ok(_) => panic!("same-source custom region op unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("custom region source image"));

    let op = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        vec![ImageId::supplied(1)],
        RegionSumMeasure,
        "custom.region.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(op.cost_per_voxel(), 7.5);
    assert_eq!(op.fold_law(), FoldLaw::approximate(1.0e-12).unwrap());

    let exact_op = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        Vec::new(),
        RegionCountMeasure,
        "custom.region.count.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(exact_op.fold_law(), FoldLaw::ExactAssociative);

    let f16_source = RegionMeasureOp::new(
        "planned custom region measure",
        0usize,
        vec![ImageId::supplied(1)],
        RegionSumMeasure,
        "custom.region.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding_source(0, Dtype::F16);
    let error = match f16_source {
        Ok(_) => panic!("f16 custom region source unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("intensity dtype float16"));

    let duplicate = Measurements::for_labels(0usize)
        .custom_region(RegionCountMeasure, Vec::<IntensityImage<0>>::new())
        .custom_region(RegionCountMeasure, Vec::<IntensityImage<0>>::new())
        .build(base([2, 2, 2]));
    let error = match duplicate {
        Ok(_) => panic!("duplicate custom region key unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("requested more than once"));

    let same_source = Measurements::for_labels(0usize)
        .custom_region(RegionSumMeasure, vec![IntensityImage::<0>::new(0usize)])
        .build(base([2, 2, 2]));
    let error = match same_source {
        Ok(_) => panic!("same-source custom region unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("custom region source image"));

    let missing_source = Measurements::for_labels(0usize)
        .custom_region(RegionSumMeasure, vec![IntensityImage::<0>::new(99usize)])
        .build(base([2, 2, 2]));
    let error = match missing_source {
        Ok(_) => panic!("missing custom region source unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("custom region source image 99"));
    assert!(error.contains("is not image 0"));

    let duplicate_source_image = Measurements::for_labels(0usize)
        .custom_region(
            TwoSourceRegionMeasure,
            vec![
                IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
                IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            ],
        )
        .build(base([2, 2, 2]));
    let error = match duplicate_source_image {
        Ok(_) => panic!("duplicate custom region source image unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("supplied more than once"));

    let plan = Measurements::for_labels(0usize)
        .custom_region(
            RegionSumMeasure,
            vec![IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64)],
        )
        .custom_region(RegionCountMeasure, Vec::<IntensityImage<0>>::new())
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(
        plan.custom_region_fold_law(0),
        Some(FoldLaw::approximate(1.0e-12).unwrap())
    );
    assert_eq!(
        plan.custom_region_fold_law(1),
        Some(FoldLaw::ExactAssociative)
    );
    assert_eq!(plan.custom_region_fold_law(2), None);
    assert_eq!(plan.fold_law(), FoldLaw::approximate(1.0e-12).unwrap());
}

#[test]
fn boundary_measure_extensions_run_framework_owned_neighbor_iteration() {
    let mut labels = Array3::<f64>::zeros((2, 2, 3));
    labels[[0, 0, 0]] = 1.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 1]] = 3.0;
    labels[[1, 0, 0]] = 1.0;

    let faces = BoundaryContactCountMeasure {
        connectivity: Connectivity::Faces,
    };
    let wide = BoundaryContactCountMeasure {
        connectivity: Connectivity::FacesEdgesAndCorners,
    };

    assert_boundary_measurement_invariant(labels.view(), &faces).unwrap();
    assert_boundary_measurement_decomposition_invariant(
        labels.view(),
        &faces,
        &[[1, 1, 1], [1, 2, 2], [2, 1, 3]],
    )
    .unwrap();
    let encoded_faces = run_boundary_measure(labels.view(), &faces).unwrap();
    let encoded_wide = run_boundary_measure(labels.view(), &wide).unwrap();
    assert_ne!(encoded_faces, encoded_wide);

    let schema = encoded_schema(&encoded_faces).unwrap();
    assert_eq!(schema.columns()[0].name(), "label");
    assert_eq!(schema.columns()[1].name(), "contacts");
    assert_eq!(schema.columns()[2].name(), "background");

    labels[[0, 1, 2]] = 1.5;
    assert!(run_boundary_measure(labels.view(), &faces).is_err());
}

#[test]
fn planned_boundary_measure_op_matches_resident_extension() {
    let mut labels = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 1.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 1]] = 3.0;
    labels[[1, 0, 0]] = 1.0;

    let measure = BoundaryContactCountMeasure {
        connectivity: Connectivity::Faces,
    };
    let mut decomposition = base([2, 2, 2]);
    let grid = decomposition.phases[0].grid.clone();
    let op = BoundaryMeasureOp::new(
        "planned custom boundary measure",
        0usize,
        measure,
        "boundary.custom.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding_labels(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env =
        ArrayEnvironment::new(labels.clone().into(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned custom boundary measure",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();

    let measure = BoundaryContactCountMeasure {
        connectivity: Connectivity::Faces,
    };
    let mut table = Table::new(VOLUME, measure.schema()).unwrap();
    fold_fragments(&env, "boundary.custom.rows", &mut |key, bytes| {
        if key.phase == 1 {
            table.write(key.block, bytes)?;
        }
        Ok(())
    })
    .unwrap();
    table.seal().unwrap();
    let got = table
        .scan(&Region::whole(&VOLUME))
        .unwrap()
        .map(|row| Ok((row.u64(0)?, row.u64(1)?, row.u64(2)?)))
        .collect::<Result<Vec<_>>>()
        .unwrap();

    let resident = run_boundary_measure(labels.view(), &measure).unwrap();
    let resident = custom_boundary_row_tuples(&resident, &measure);
    assert_eq!(got, resident);
}

#[test]
fn measurement_builder_runs_planned_custom_boundary_measure() {
    let mut labels = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 1.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 1]] = 3.0;
    labels[[1, 0, 0]] = 1.0;

    let plan = Measurements::for_labels(0usize)
        .custom_boundary(BoundaryContactCountMeasure {
            connectivity: Connectivity::Faces,
        })
        .stream("custom")
        .build(base([2, 2, 2]))
        .unwrap();
    let rows = plan.custom_boundary_rows(0).unwrap();
    assert_eq!(rows.stream(), plan.custom_boundary_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.custom_boundary_rows_phase(0).unwrap());
    assert_eq!(rows.phase(), 2);

    let env = ArrayEnvironment::new(
        labels.clone().into(),
        plan.decomposition.n_phases(),
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder custom boundary measure",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let measure = BoundaryContactCountMeasure {
        connectivity: Connectivity::Faces,
    };
    let got = collect_custom_boundary_rows(&env, &rows, VOLUME, measure.schema(), |row| {
        Ok((row.u64(0)?, row.u64(1)?, row.u64(2)?))
    })
    .unwrap();

    let resident = run_boundary_measure(labels.view(), &measure).unwrap();
    let resident = custom_boundary_row_tuples(&resident, &measure);
    assert_eq!(got, resident);
}

fn custom_boundary_fixture() -> Array3<f64> {
    let mut labels = Array3::<f64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    labels[[0, 0, 0]] = 1.0;
    labels[[0, 0, 1]] = 2.0;
    labels[[0, 1, 1]] = 3.0;
    labels[[1, 0, 0]] = 1.0;
    labels
}

fn face_boundary_contacts_crossing_block_seams(labels: &Array3<f64>, block: [usize; 3]) -> usize {
    let shape = labels.shape();
    let offsets = [[1isize, 0isize, 0isize], [0, 1, 0], [0, 0, 1]];
    let mut crossing = 0;
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let label = labels[[z, y, x]];
                if label == 0.0 {
                    continue;
                }
                let block_a = [z / block[0], y / block[1], x / block[2]];
                for [dz, dy, dx] in offsets {
                    let nz = z as isize + dz;
                    let ny = y as isize + dy;
                    let nx = x as isize + dx;
                    if nz < 0
                        || ny < 0
                        || nx < 0
                        || nz >= shape[0] as isize
                        || ny >= shape[1] as isize
                        || nx >= shape[2] as isize
                    {
                        continue;
                    }
                    let neighbour_at = [nz as usize, ny as usize, nx as usize];
                    let neighbour = labels[neighbour_at];
                    if neighbour == label {
                        continue;
                    }
                    let block_b = [
                        neighbour_at[0] / block[0],
                        neighbour_at[1] / block[1],
                        neighbour_at[2] / block[2],
                    ];
                    if block_a != block_b {
                        crossing += 1;
                    }
                }
            }
        }
    }
    crossing
}

fn collect_custom_boundary_builder_rows(block: [usize; 3]) -> Vec<(u64, u64, u64)> {
    let labels = custom_boundary_fixture();
    let plan = Measurements::for_labels(0usize)
        .custom_boundary(BoundaryContactCountMeasure {
            connectivity: Connectivity::Faces,
        })
        .stream("custom")
        .build(base(block))
        .unwrap();
    let rows = plan.custom_boundary_rows(0).unwrap();
    assert_eq!(rows.stream(), plan.custom_boundary_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.custom_boundary_rows_phase(0).unwrap());

    let env =
        ArrayEnvironment::new(labels.into(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder custom boundary measure",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let measure = BoundaryContactCountMeasure {
        connectivity: Connectivity::Faces,
    };
    collect_custom_boundary_rows(&env, &rows, VOLUME, measure.schema(), |row| {
        Ok((row.u64(0)?, row.u64(1)?, row.u64(2)?))
    })
    .unwrap()
}

fn custom_boundary_row_tuples(
    encoded: &[u8],
    measure: &BoundaryContactCountMeasure,
) -> Vec<(u64, u64, u64)> {
    decode_custom_measurement_rows(VOLUME, measure.schema(), encoded, |row| {
        Ok((row.u64(0)?, row.u64(1)?, row.u64(2)?))
    })
    .unwrap()
}

fn resident_custom_boundary_rows() -> Vec<(u64, u64, u64)> {
    let labels = custom_boundary_fixture();
    let measure = BoundaryContactCountMeasure {
        connectivity: Connectivity::Faces,
    };
    let resident = run_boundary_measure(labels.view(), &measure).unwrap();
    custom_boundary_row_tuples(&resident, &measure)
}

#[test]
fn planned_custom_boundary_builder_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = collect_custom_boundary_builder_rows(VOLUME);
    let split_xy = collect_custom_boundary_builder_rows([2, 2, 3]);
    let split_xyz = collect_custom_boundary_builder_rows([2, 2, 2]);
    let seam_everywhere = collect_custom_boundary_builder_rows([1, 1, 1]);
    let fixture = custom_boundary_fixture();

    assert_eq!(
        face_boundary_contacts_crossing_block_seams(&fixture, VOLUME),
        0,
        "the whole-volume boundary extension run should be the no-seam reference"
    );
    assert!(
        face_boundary_contacts_crossing_block_seams(&fixture, [1, 1, 1]) > 0,
        "the one-voxel boundary extension run must exercise cross-block contacts"
    );
    assert_eq!(coarse, split_xy);
    assert_eq!(split_xy, split_xyz);
    assert_eq!(split_xyz, seam_everywhere);
    assert_eq!(seam_everywhere, resident_custom_boundary_rows());
}

#[test]
fn public_custom_boundary_builder_harness_checks_planned_execution() {
    let labels = custom_boundary_fixture();
    assert_custom_boundary_builder_invariant(
        labels.view(),
        BoundaryContactCountMeasure {
            connectivity: Connectivity::Faces,
        },
        &[VOLUME, [2, 2, 3], [2, 2, 2], [1, 1, 1]],
    )
    .unwrap();
}

#[test]
fn boundary_measure_extensions_validate_reach_fold_law_and_rows() {
    let mut labels = Array3::<f64>::zeros((2, 2, 2));
    labels[[0, 0, 0]] = 1.0;
    assert!(run_boundary_measure(labels.view(), &WideBoundaryMeasure).is_err());
    assert!(run_boundary_measure(labels.view(), &OrderedBoundaryMeasure).is_err());
    assert!(run_boundary_measure(labels.view(), &WrongRowBoundaryMeasure).is_err());
    let error = run_boundary_measure(labels.view(), &BadSchemaBoundaryMeasure)
        .unwrap_err()
        .to_string();
    assert!(error.contains("schema must start with a u64 \"label\" column"));

    let mut drift_labels = Array3::<f64>::zeros((2, 1, 2));
    drift_labels[[0, 0, 0]] = 1.0;
    drift_labels[[0, 0, 1]] = 2.0;
    drift_labels[[1, 0, 0]] = 1.0;
    let error = assert_boundary_measurement_decomposition_invariant(
        drift_labels.view(),
        &DriftBoundaryMeasure,
        &[[1, 1, 1]],
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("within tolerance 0.25"));

    let wide_builder = Measurements::for_labels(0usize)
        .custom_boundary(WideBoundaryMeasure)
        .build(base([2, 2, 2]));
    let error = match wide_builder {
        Ok(_) => panic!("wide-reach custom boundary measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("requested reach"));

    let empty_stream = BoundaryMeasureOp::new(
        "planned custom boundary measure",
        0usize,
        BoundaryContactCountMeasure {
            connectivity: Connectivity::Faces,
        },
        "",
        Lifecycle::DeleteOnExit,
    );
    let error = match empty_stream {
        Ok(_) => panic!("empty boundary measure stream unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("output stream must not be empty"));

    let bad_schema = BoundaryMeasureOp::new(
        "planned custom boundary measure",
        0usize,
        BadSchemaBoundaryMeasure,
        "custom.boundary.rows",
        Lifecycle::DeleteOnExit,
    );
    let error = match bad_schema {
        Ok(_) => panic!("bad-schema custom boundary measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("schema must start with a u64 \"label\" column"));

    let bad_cost = BoundaryMeasureOp::new(
        "planned custom boundary measure",
        0usize,
        BadCostBoundaryMeasure,
        "custom.boundary.rows",
        Lifecycle::DeleteOnExit,
    );
    let error = match bad_cost {
        Ok(_) => panic!("bad-cost custom boundary measure unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("invalid planner cost"));

    let op = BoundaryMeasureOp::new(
        "planned custom boundary measure",
        0usize,
        BoundaryContactCountMeasure {
            connectivity: Connectivity::Faces,
        },
        "custom.boundary.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    assert_eq!(op.cost_per_voxel(), 3.25);
    assert_eq!(op.fold_law(), FoldLaw::ExactAssociative);

    let duplicate = Measurements::for_labels(0usize)
        .custom_boundary(BoundaryContactCountMeasure {
            connectivity: Connectivity::Faces,
        })
        .custom_boundary(BoundaryContactCountMeasure {
            connectivity: Connectivity::FacesEdgesAndCorners,
        })
        .build(base([2, 2, 2]));
    let error = match duplicate {
        Ok(_) => panic!("duplicate custom boundary key unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("requested more than once"));

    let plan = Measurements::for_labels(0usize)
        .custom_boundary(BoundaryContactCountMeasure {
            connectivity: Connectivity::Faces,
        })
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(
        plan.custom_boundary_fold_law(0),
        Some(FoldLaw::ExactAssociative)
    );
    assert_eq!(plan.custom_boundary_fold_law(1), None);
    assert_eq!(plan.fold_law(), FoldLaw::ExactAssociative);
}

#[test]
fn object_topology_reports_components_cavities_and_euler_number() {
    let mut labels = Array3::<f64>::zeros((5, 5, 5));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 0, 3]] = 2.0;
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                if [z, y, x] != [2, 2, 2] {
                    labels[[z, y, x]] = 7.0;
                }
            }
        }
    }

    let topology = object_topology_measurements(labels.view()).unwrap();
    assert_eq!(topology.len(), 2);

    let disconnected = topology.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(
        disconnected.convention,
        ObjectTopologyConvention::Foreground26Background6
    );
    assert_eq!(disconnected.components, 2);
    assert_eq!(disconnected.tunnels, 0);
    assert_eq!(disconnected.cavities, 0);
    assert_eq!(disconnected.euler_number, 2);
    assert_eq!(ObjectTopologyFeature::ALL.len(), 4);
    assert_eq!(
        ObjectTopologyFeature::EulerNumber.column_name(),
        "euler_number"
    );
    assert_eq!(
        disconnected.feature(ObjectTopologyFeature::Components),
        disconnected.components as f64
    );
    assert_eq!(
        disconnected.feature(ObjectTopologyFeature::EulerNumber),
        disconnected.euler_number as f64
    );

    let shell = topology.iter().find(|row| row.label == 7).unwrap();
    assert_eq!(shell.components, 1);
    assert_eq!(shell.tunnels, 0);
    assert_eq!(shell.cavities, 1);
    assert_eq!(shell.euler_number, 2);

    labels[[4, 4, 4]] = 1.5;
    assert!(object_topology_measurements(labels.view()).is_err());
}

#[test]
fn object_topology_supports_complementary_connectivity_conventions() {
    let mut labels = Array3::<f64>::zeros((2, 2, 2));
    labels[[0, 0, 0]] = 2.0;
    labels[[1, 1, 1]] = 2.0;

    let wide = object_topology_measurements_with(
        labels.view(),
        ObjectTopologyConvention::Foreground26Background6,
    )
    .unwrap();
    let narrow = object_topology_measurements_with(
        labels.view(),
        ObjectTopologyConvention::Foreground6Background26,
    )
    .unwrap();

    assert_eq!(
        wide[0].convention,
        ObjectTopologyConvention::Foreground26Background6
    );
    assert_eq!(wide[0].components, 1);
    assert_eq!(
        narrow[0].convention,
        ObjectTopologyConvention::Foreground6Background26
    );
    assert_eq!(narrow[0].components, 2);
    assert_ne!(wide[0].components, narrow[0].components);
}

#[test]
fn object_component_measurements_honor_foreground_connectivity() {
    let mut labels = Array3::<f64>::zeros((3, 3, 3));
    labels[[0, 0, 0]] = 2.0;
    labels[[0, 1, 1]] = 2.0;
    labels[[1, 1, 1]] = 2.0;
    labels[[0, 2, 0]] = 5.0;
    labels[[1, 1, 1]] = 5.0;

    let faces = object_component_measurements(labels.view(), Connectivity::Faces).unwrap();
    let edges = object_component_measurements(labels.view(), Connectivity::FacesAndEdges).unwrap();
    let corners =
        object_component_measurements(labels.view(), Connectivity::FacesEdgesAndCorners).unwrap();

    let face_label_2 = faces.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(face_label_2.count, 2);
    assert_eq!(face_label_2.connectivity, Connectivity::Faces);
    assert_eq!(face_label_2.components, 2);
    assert_eq!(ObjectComponentFeature::ALL.len(), 2);
    assert_eq!(ObjectComponentFeature::Count.column_name(), "count");
    assert_eq!(
        face_label_2.feature(ObjectComponentFeature::Count),
        face_label_2.count as f64
    );
    assert_eq!(
        face_label_2.feature(ObjectComponentFeature::Components),
        face_label_2.components as f64
    );
    assert_eq!(
        edges.iter().find(|row| row.label == 2).unwrap().components,
        1
    );

    assert_eq!(
        faces.iter().find(|row| row.label == 5).unwrap().components,
        2
    );
    assert_eq!(
        edges.iter().find(|row| row.label == 5).unwrap().components,
        2
    );
    assert_eq!(
        corners
            .iter()
            .find(|row| row.label == 5)
            .unwrap()
            .components,
        1
    );

    labels[[2, 2, 2]] = -1.0;
    assert!(object_component_measurements(labels.view(), Connectivity::Faces).is_err());
}

#[test]
fn topology_and_component_rows_have_canonical_schemas_and_collectors() {
    let topology = vec![
        ObjectTopologyMeasurements {
            label: 2,
            convention: ObjectTopologyConvention::Foreground26Background6,
            components: 2,
            tunnels: 0,
            cavities: 0,
            euler_number: 2,
        },
        ObjectTopologyMeasurements {
            label: 7,
            convention: ObjectTopologyConvention::Foreground26Background6,
            components: 1,
            tunnels: 2,
            cavities: 0,
            euler_number: -1,
        },
    ];
    let encoded_topology = encode_topology_measurements(&topology).unwrap();
    assert_eq!(
        encoded_schema(&encoded_topology).unwrap(),
        topology_measurement_schema()
    );
    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("topology.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("topology.rows", 0, [0, 0, 0], &encoded_topology)
        .unwrap();
    let got_topology = collect_topology_measurements(&env, "topology.rows", 0, VOLUME).unwrap();
    assert_eq!(got_topology, topology);

    let components = vec![
        ObjectComponentMeasurements {
            label: 2,
            count: 3,
            connectivity: Connectivity::Faces,
            components: 2,
        },
        ObjectComponentMeasurements {
            label: 5,
            count: 2,
            connectivity: Connectivity::FacesEdgesAndCorners,
            components: 1,
        },
    ];
    let encoded_components = encode_component_measurements(&components).unwrap();
    let component_schema = encoded_schema(&encoded_components).unwrap();
    assert_eq!(component_schema, component_measurement_schema());
    assert_eq!(component_schema.columns()[2].name(), "connectivity");
    env.declare_sidecar("component.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("component.rows", 0, [0, 0, 0], &encoded_components)
        .unwrap();
    let got_components = collect_component_measurements(&env, "component.rows", 0, VOLUME).unwrap();
    assert_eq!(got_components, components);

    assert!(encode_topology_measurements(&[ObjectTopologyMeasurements {
        label: 2,
        convention: ObjectTopologyConvention::Foreground26Background6,
        components: 1,
        tunnels: 0,
        cavities: 0,
        euler_number: 2,
    }])
    .is_err());
    assert!(
        encode_component_measurements(&[ObjectComponentMeasurements {
            label: 5,
            count: 1,
            connectivity: Connectivity::FacesAndEdges,
            components: 2,
        }])
        .is_err()
    );
}

fn planned_topology(block: [usize; 3]) -> Vec<ObjectTopologyMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = ObjectTopologyOp::new(
        "measure topology",
        0usize,
        ObjectTopologyConvention::default(),
        "topology.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned topology",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_topology_measurements(&env, "topology.planned.rows", 1, VOLUME).unwrap()
}

fn planned_components(
    block: [usize; 3],
    connectivity: Connectivity,
) -> Vec<ObjectComponentMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let op = ObjectComponentOp::new(
        "measure components",
        0usize,
        connectivity,
        "components.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&op, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let env = ArrayEnvironment::new(labels(), decomposition.n_phases(), [2, 2, 2]).unwrap();
    execute_phases(
        "planned components",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Pixels, PhaseWork::Fragments(&op)],
    )
    .unwrap();
    collect_component_measurements(&env, "components.planned.rows", 1, VOLUME).unwrap()
}

#[test]
fn planned_topology_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_topology(VOLUME);
    let split = planned_topology([2, 2, 2]);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference = object_topology_measurements(labels.view()).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn planned_components_are_decomposition_invariant_and_match_resident_reference() {
    let coarse = planned_components(VOLUME, Connectivity::FacesAndEdges);
    let split = planned_components([2, 2, 2], Connectivity::FacesAndEdges);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference =
        object_component_measurements(labels.view(), Connectivity::FacesAndEdges).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_topology_and_components() {
    let label_array = labels().view::<f64>().unwrap().to_owned();
    let convention = ObjectTopologyConvention::Foreground6Background26;
    let expected_topology =
        object_topology_measurements_with(label_array.view(), convention).unwrap();
    let expected_components =
        object_component_measurements(label_array.view(), Connectivity::FacesAndEdges).unwrap();

    let plan = Measurements::for_labels(0usize)
        .shape(
            ShapeSet::topology_with_convention(convention)
                .with_components(Connectivity::FacesAndEdges),
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    assert_eq!(plan.topology_rows_phase(), Some(1));
    assert_eq!(plan.component_rows_phase(), Some(2));
    let topology_rows = plan.topology_rows().unwrap();
    assert_eq!(topology_rows.stream(), plan.topology_stream().unwrap());
    assert_eq!(topology_rows.phase(), plan.topology_rows_phase().unwrap());
    let component_rows = plan.component_rows().unwrap();
    assert_eq!(component_rows.stream(), plan.component_stream().unwrap());
    assert_eq!(component_rows.phase(), plan.component_rows_phase().unwrap());

    let env = ArrayEnvironment::new(labels(), plan.decomposition.n_phases(), [2, 2, 2]).unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder topology",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got_topology = collect_topology_rows(&env, &topology_rows, VOLUME).unwrap();
    let got_components = collect_component_rows(&env, &component_rows, VOLUME).unwrap();
    assert_eq!(got_topology, expected_topology);
    assert_eq!(got_components, expected_components);
}

#[test]
fn centroid_relationships_report_nearest_neighbor_summaries() {
    let mut labels = Array3::<f64>::zeros((1, 1, 6));
    labels[[0, 0, 0]] = 1.0;
    labels[[0, 0, 2]] = 2.0;
    labels[[0, 0, 5]] = 3.0;

    let relationships = object_centroid_relationships(
        labels.view(),
        PhysicalSpacing::new([1.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    assert_eq!(relationships.len(), 3);
    assert_eq!(relationships[0].labels, [1, 2]);
    assert_eq!(relationships[0].centroid_delta, [0.0, 0.0, 4.0]);
    assert_eq!(relationships[0].centroid_distance, 4.0);
    assert_eq!(relationships[1].labels, [1, 3]);
    assert_eq!(relationships[1].centroid_distance, 10.0);
    assert_eq!(relationships[2].labels, [2, 3]);
    assert_eq!(relationships[2].centroid_distance, 6.0);

    let threshold = WithinDistanceThreshold::new(6.0).unwrap();
    assert_eq!(threshold.get(), 6.0);
    let summaries = summarize_centroid_neighbors_within(&relationships, threshold).unwrap();
    assert_eq!(
        summaries,
        summarize_centroid_neighbors(&relationships, 6.0).unwrap()
    );
    assert_eq!(summaries.len(), 3);

    let first = summaries.iter().find(|row| row.label == 1).unwrap();
    assert_eq!(first.within_distance_neighbors, 1);
    assert_eq!(first.closest_label, Some(2));
    assert_eq!(first.closest_distance, Some(4.0));
    assert_eq!(first.second_closest_label, Some(3));
    assert_eq!(first.second_closest_distance, Some(10.0));
    assert_eq!(first.angle_between_closest, Some(0.0));
    assert_eq!(ObjectNeighborFeature::ALL.len(), 6);
    assert_eq!(
        ObjectNeighborFeature::SecondClosestDistance.column_name(),
        "second_closest_distance"
    );
    assert_eq!(
        first.feature(ObjectNeighborFeature::WithinDistanceNeighbors),
        Some(1.0)
    );
    assert_eq!(
        first.feature(ObjectNeighborFeature::ClosestLabel),
        Some(2.0)
    );
    assert_eq!(
        first.feature(ObjectNeighborFeature::ClosestDistance),
        first.closest_distance
    );
    assert_eq!(
        first.feature(ObjectNeighborFeature::AngleBetweenClosest),
        first.angle_between_closest
    );

    let second = summaries.iter().find(|row| row.label == 2).unwrap();
    assert_eq!(second.within_distance_neighbors, 2);
    assert_eq!(second.closest_label, Some(1));
    assert_eq!(second.second_closest_label, Some(3));
    assert_eq!(second.angle_between_closest, Some(std::f64::consts::PI));

    labels[[0, 0, 1]] = 1.5;
    assert!(object_centroid_relationships(labels.view(), PhysicalSpacing::unit()).is_err());

    labels[[0, 0, 1]] = 18_446_744_073_709_551_616.0;
    assert!(object_centroid_relationships(labels.view(), PhysicalSpacing::unit()).is_err());
}

#[test]
fn boundary_distance_relationships_report_nearest_boundary_centres() {
    let mut labels = Array3::<f64>::zeros((3, 1, 7));
    labels[[0, 0, 0]] = 1.0;
    labels[[0, 0, 1]] = 1.0;
    labels[[0, 0, 4]] = 2.0;
    labels[[2, 0, 4]] = 3.0;

    let relationships = object_boundary_distance_relationships(
        labels.view(),
        PhysicalSpacing::new([3.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    assert_eq!(relationships.len(), 3);

    let first = relationships
        .iter()
        .find(|row| row.labels == [1, 2])
        .unwrap();
    assert_eq!(first.nearest_points, [[0, 0, 1], [0, 0, 4]]);
    assert_eq!(first.boundary_delta, [0.0, 0.0, 6.0]);
    assert_eq!(first.boundary_distance, 6.0);

    let second = relationships
        .iter()
        .find(|row| row.labels == [2, 3])
        .unwrap();
    assert_eq!(second.nearest_points, [[0, 0, 4], [2, 0, 4]]);
    assert_eq!(second.boundary_delta, [6.0, 0.0, 0.0]);
    assert_eq!(second.boundary_distance, 6.0);

    labels[[1, 0, 0]] = 1.5;
    assert!(
        object_boundary_distance_relationships(labels.view(), PhysicalSpacing::unit()).is_err()
    );
}

#[test]
fn expansion_until_adjacent_relationships_derive_equal_front_distance() {
    let mut labels = Array3::<f64>::zeros((3, 1, 7));
    labels[[0, 0, 0]] = 1.0;
    labels[[0, 0, 1]] = 1.0;
    labels[[0, 0, 4]] = 2.0;
    labels[[2, 0, 4]] = 3.0;

    let relationships = object_expansion_until_adjacent_relationships(
        labels.view(),
        PhysicalSpacing::new([3.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    assert_eq!(relationships.len(), 3);

    let first = relationships
        .iter()
        .find(|row| row.labels == [1, 2])
        .unwrap();
    assert_eq!(first.nearest_points, [[0, 0, 1], [0, 0, 4]]);
    assert_eq!(first.boundary_distance, 6.0);
    assert_eq!(first.expansion_distance, 3.0);

    let boundary_rows = object_boundary_distance_relationships(
        labels.view(),
        PhysicalSpacing::new([3.0, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let derived =
        expansion_until_adjacent_relationships_from_boundary_distances(&boundary_rows).unwrap();
    assert_eq!(derived, relationships);
}

#[test]
fn expansion_relationship_rows_have_a_canonical_schema_and_collector() {
    let relationships = vec![
        ObjectExpansionMeasurements {
            labels: [1, 2],
            expansion_distance: 2.5,
            boundary_distance: 5.0,
            nearest_points: [[0, 0, 1], [0, 0, 4]],
        },
        ObjectExpansionMeasurements {
            labels: [1, 3],
            expansion_distance: 6.5,
            boundary_distance: 13.0,
            nearest_points: [[0, 0, 0], [2, 0, 4]],
        },
    ];
    let encoded = encode_expansion_relationship_measurements(&relationships).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, expansion_relationship_measurement_schema());
    assert_eq!(schema.columns()[2].name(), "expansion_distance");
    assert_eq!(schema.columns()[3].name(), "boundary_distance");
    assert_eq!(schema.columns()[9].name(), "nearest_b_2");
    assert_eq!(ObjectExpansionFeature::ALL.len(), 2);
    assert_eq!(
        ObjectExpansionFeature::ExpansionDistance.column_name(),
        "expansion_distance"
    );
    assert_eq!(
        relationships[0].feature(ObjectExpansionFeature::ExpansionDistance),
        2.5
    );
    assert_eq!(
        relationships[0].feature(ObjectExpansionFeature::BoundaryDistance),
        5.0
    );

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("expansion.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("expansion.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got =
        collect_expansion_relationship_measurements(&env, "expansion.rows", 0, VOLUME).unwrap();
    assert_eq!(got, relationships);

    assert!(
        encode_expansion_relationship_measurements(&[ObjectExpansionMeasurements {
            labels: [2, 1],
            expansion_distance: 0.5,
            boundary_distance: 1.0,
            nearest_points: [[0, 0, 0], [0, 0, 1]],
        }])
        .is_err()
    );
    assert!(
        encode_expansion_relationship_measurements(&[ObjectExpansionMeasurements {
            labels: [1, 2],
            expansion_distance: 0.75,
            boundary_distance: 1.0,
            nearest_points: [[0, 0, 0], [0, 0, 1]],
        }])
        .is_err()
    );
}

#[test]
fn centroid_relationship_summary_validates_rows() {
    assert!(WithinDistanceThreshold::new(f64::NAN).is_err());
    assert!(WithinDistanceThreshold::new(-1.0).is_err());
    assert!(summarize_centroid_neighbors(&[], f64::NAN).is_err());
    assert!(summarize_centroid_neighbors(
        &[ObjectRelationshipMeasurements {
            labels: [2, 1],
            centroid_distance: 1.0,
            centroid_delta: [1.0, 0.0, 0.0],
        }],
        1.0,
    )
    .is_err());
    assert!(summarize_centroid_neighbors(
        &[ObjectRelationshipMeasurements {
            labels: [1, 2],
            centroid_distance: 2.0,
            centroid_delta: [1.0, 0.0, 0.0],
        }],
        1.0,
    )
    .is_err());
}

#[test]
fn glcm_texture_reports_directed_in_object_cooccurrence() {
    let mut labels = Array3::<f64>::zeros((1, 2, 4));
    for x in 0..4 {
        labels[[0, 0, x]] = 3.0;
    }
    labels[[0, 1, 0]] = 5.0;
    labels[[0, 1, 1]] = 5.0;
    labels[[0, 1, 2]] = 5.0;

    let mut values = Array3::<f64>::zeros((1, 2, 4));
    values[[0, 0, 0]] = 0.0;
    values[[0, 0, 1]] = 1.0;
    values[[0, 0, 2]] = 1.0;
    values[[0, 0, 3]] = 2.0;
    values[[0, 1, 0]] = 0.0;
    values[[0, 1, 1]] = f64::NAN;
    values[[0, 1, 2]] = 3.0;

    let measurements = glcm_texture_measurements(
        labels.view(),
        values.view(),
        GlcmQuantization::new(3, 0.0, 2.0).unwrap(),
        GlcmOffset::new([0, 0, 1]).unwrap(),
    )
    .unwrap();
    assert_eq!(measurements.len(), 2);

    let first = &measurements[0];
    assert_eq!(first.label, 3);
    assert_eq!(first.levels, 3);
    assert_eq!(first.offset, [0, 0, 1]);
    assert_eq!(first.pairs, 3);
    assert_eq!(first.excluded, 0);
    assert_eq!(first.matrix, vec![0, 1, 0, 0, 1, 1, 0, 0, 0]);
    assert_eq!(GlcmTextureFeature::ALL.len(), 25);
    assert_eq!(GlcmTextureFeature::Contrast.column_name(), "contrast");
    assert_eq!(
        GlcmTextureFeature::MaximalCorrelationCoefficient.column_name(),
        "maximal_correlation_coefficient"
    );
    assert_eq!(
        first.feature(GlcmTextureFeature::Contrast),
        first.contrast()
    );
    assert_eq!(
        first.feature(GlcmTextureFeature::InformationMeasureCorrelation2),
        first.information_measure_correlation_2()
    );
    assert_eq!(
        first.feature(GlcmTextureFeature::MaximalCorrelationCoefficient),
        first.maximal_correlation_coefficient()
    );
    assert_eq!(first.contrast(), Some(2.0 / 3.0));
    assert_eq!(first.dissimilarity(), Some(2.0 / 3.0));
    assert_eq!(first.homogeneity(), Some(2.0 / 3.0));
    assert_eq!(first.inverse_difference_moment(), Some(2.0 / 3.0));
    assert!((first.inverse_difference_normalized().unwrap() - 5.0 / 6.0).abs() < 1.0e-12);
    assert!((first.inverse_difference_moment_normalized().unwrap() - 14.0 / 15.0).abs() < 1.0e-12);
    assert_eq!(first.angular_second_moment(), Some(1.0 / 3.0));
    assert_eq!(first.energy(), Some((1.0f64 / 3.0).sqrt()));
    assert_eq!(first.max_probability(), Some(1.0 / 3.0));
    assert!(first.entropy().unwrap() > 1.58);
    assert_eq!(first.variance(), Some(2.0 / 9.0));
    assert_eq!(first.autocorrelation(), Some(1.0));
    assert!(first.cluster_shade().unwrap().abs() < 1.0e-12);
    assert!((first.cluster_tendency().unwrap() - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((first.cluster_prominence().unwrap() - 2.0 / 3.0).abs() < 1.0e-12);
    assert_eq!(first.sum_average(), Some(2.0));
    assert_eq!(first.sum_variance(), Some(2.0 / 3.0));
    assert!((first.sum_entropy().unwrap() - first.entropy().unwrap()).abs() < 1.0e-12);
    assert_eq!(first.difference_average(), Some(2.0 / 3.0));
    assert_eq!(first.difference_variance(), Some(2.0 / 9.0));
    assert!((first.difference_entropy().unwrap() - 0.9182958340544896).abs() < 1.0e-12);
    assert!((first.correlation().unwrap() - 0.5).abs() < 1.0e-12);
    let marginal_entropy =
        -(1.0f64 / 3.0) * (1.0f64 / 3.0).log2() - (2.0f64 / 3.0) * (2.0f64 / 3.0).log2();
    let joint_entropy = 3.0f64.log2();
    let marginal_cross_entropy = -((1.0f64 / 3.0) * (2.0f64 / 9.0).log2()
        + (1.0f64 / 3.0) * (4.0f64 / 9.0).log2()
        + (1.0f64 / 3.0) * (2.0f64 / 9.0).log2());
    let expected_imc1 = (joint_entropy - marginal_cross_entropy) / marginal_entropy;
    assert!((first.information_measure_correlation_1().unwrap() - expected_imc1).abs() < 1.0e-12);
    let expected_imc2 = (1.0 - 2.0f64.powf(-2.0 * (2.0 * marginal_entropy - joint_entropy))).sqrt();
    assert!((first.information_measure_correlation_2().unwrap() - expected_imc2).abs() < 1.0e-12);
    assert!((first.maximal_correlation_coefficient().unwrap() - 0.5).abs() < 1.0e-12);

    let second = &measurements[1];
    assert_eq!(second.label, 5);
    assert_eq!(second.pairs, 0);
    assert_eq!(second.excluded, 2);
    assert_eq!(second.contrast(), None);
    assert_eq!(second.max_probability(), None);
    assert_eq!(second.inverse_difference_moment(), None);
    assert_eq!(second.inverse_difference_normalized(), None);
    assert_eq!(second.inverse_difference_moment_normalized(), None);
    assert_eq!(second.difference_average(), None);
    assert_eq!(second.cluster_tendency(), None);
    assert_eq!(second.sum_entropy(), None);
    assert_eq!(second.difference_entropy(), None);
    assert_eq!(second.information_measure_correlation_1(), None);
    assert_eq!(second.information_measure_correlation_2(), None);
    assert_eq!(second.maximal_correlation_coefficient(), None);
}

#[test]
fn glcm_texture_maximal_correlation_coefficient_handles_independent_and_perfect_cases() {
    let independent = GlcmTextureMeasurements {
        label: 1,
        levels: 2,
        offset: [0, 0, 1],
        pairs: 4,
        excluded: 0,
        matrix: vec![1, 1, 1, 1],
    };
    assert!(independent.maximal_correlation_coefficient().unwrap().abs() < 1.0e-12);

    let perfect = GlcmTextureMeasurements {
        label: 1,
        levels: 2,
        offset: [0, 0, 1],
        pairs: 4,
        excluded: 0,
        matrix: vec![2, 0, 0, 2],
    };
    assert!((perfect.maximal_correlation_coefficient().unwrap() - 1.0).abs() < 1.0e-12);
}

#[test]
fn glcm_texture_features_reject_malformed_matrix_shape() {
    let malformed = GlcmTextureMeasurements {
        label: 1,
        levels: 3,
        offset: [0, 0, 1],
        pairs: 4,
        excluded: 0,
        matrix: vec![1, 1, 1, 1],
    };

    assert_eq!(malformed.contrast(), None);
    assert_eq!(malformed.angular_second_moment(), None);
    assert_eq!(malformed.max_probability(), None);
    assert_eq!(malformed.entropy(), None);
    assert_eq!(malformed.sum_average(), None);
    assert_eq!(malformed.difference_average(), None);
    assert_eq!(malformed.correlation(), None);
    assert_eq!(malformed.information_measure_correlation_1(), None);
    assert_eq!(malformed.information_measure_correlation_2(), None);
    assert_eq!(malformed.maximal_correlation_coefficient(), None);
    assert_eq!(malformed.feature(GlcmTextureFeature::Contrast), None);
}

#[test]
fn glcm_texture_rows_fuse_across_offsets() {
    let forward = GlcmTextureMeasurements {
        label: 3,
        levels: 3,
        offset: [0, 0, 1],
        pairs: 3,
        excluded: 1,
        matrix: vec![0, 1, 0, 0, 1, 1, 0, 0, 0],
    };
    let backward = GlcmTextureMeasurements {
        label: 3,
        levels: 3,
        offset: [0, 0, -1],
        pairs: 3,
        excluded: 2,
        matrix: vec![0, 0, 0, 1, 1, 0, 0, 1, 0],
    };
    let other_label = GlcmTextureMeasurements {
        label: 5,
        levels: 3,
        offset: [0, 1, 0],
        pairs: 0,
        excluded: 1,
        matrix: vec![0; 9],
    };

    let fused =
        fuse_glcm_texture_measurements(&[forward.clone(), backward.clone(), other_label]).unwrap();
    assert_eq!(fused.len(), 2);

    let row = fused.iter().find(|row| row.label == 3).unwrap();
    assert_eq!(row.levels, 3);
    assert_eq!(row.offsets, vec![[0, 0, 1], [0, 0, -1]]);
    assert_eq!(row.pairs, 6);
    assert_eq!(row.excluded, 3);
    assert_eq!(row.matrix, vec![0, 1, 0, 1, 2, 1, 0, 1, 0]);
    assert_eq!(
        row.feature(GlcmTextureFeature::DifferenceAverage),
        row.difference_average()
    );
    assert_eq!(row.contrast(), Some(2.0 / 3.0));
    assert!((row.inverse_difference_moment().unwrap() - 2.0 / 3.0).abs() < 1.0e-12);
    assert!(row.inverse_difference_normalized().unwrap() > row.homogeneity().unwrap());
    assert_eq!(row.angular_second_moment(), Some(2.0 / 9.0));
    assert_eq!(row.energy(), Some((2.0f64 / 9.0).sqrt()));
    assert!((row.cluster_tendency().unwrap() - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((row.difference_average().unwrap() - 2.0 / 3.0).abs() < 1.0e-12);
    assert!(row.information_measure_correlation_2().unwrap() > 0.0);
    assert!((row.maximal_correlation_coefficient().unwrap() - 0.5).abs() < 1.0e-12);

    let empty = fused.iter().find(|row| row.label == 5).unwrap();
    assert_eq!(empty.pairs, 0);
    assert_eq!(empty.contrast(), None);
    assert_eq!(empty.difference_average(), None);
    assert_eq!(empty.cluster_tendency(), None);
    assert_eq!(empty.information_measure_correlation_2(), None);

    assert!(fuse_glcm_texture_measurements(&[forward.clone(), forward]).is_err());
    let malformed = GlcmTextureMeasurements {
        pairs: 4,
        ..backward
    };
    assert!(fuse_glcm_texture_measurements(&[malformed]).is_err());
    let invalid = FusedGlcmTextureMeasurements {
        label: 0,
        levels: 3,
        offsets: Vec::new(),
        pairs: 0,
        excluded: 0,
        matrix: vec![0; 9],
    };
    assert_eq!(invalid.contrast(), None);
}

#[test]
fn glcm_texture_validates_request_shape_and_labels() {
    assert!(GlcmQuantization::new(0, 0.0, 1.0).is_err());
    assert!(GlcmQuantization::new(1, 0.0, 1.0).is_err());
    assert!(GlcmQuantization::new(usize::MAX, 0.0, 1.0).is_err());
    assert!(GlcmQuantization::new(2, 1.0, 1.0).is_err());
    assert!(GlcmOffset::new([0, 0, 0]).is_err());

    let labels = Array3::<f64>::zeros((1, 1, 2));
    let values = Array3::<f64>::zeros((1, 2, 2));
    assert!(glcm_texture_measurements(
        labels.view(),
        values.view(),
        GlcmQuantization::new(2, 0.0, 1.0).unwrap(),
        GlcmOffset::new([0, 0, 1]).unwrap(),
    )
    .is_err());

    let mut invalid_labels = Array3::<f64>::zeros((1, 1, 2));
    invalid_labels[[0, 0, 0]] = 1.5;
    let values = Array3::<f64>::zeros((1, 1, 2));
    assert!(glcm_texture_measurements(
        invalid_labels.view(),
        values.view(),
        GlcmQuantization::new(2, 0.0, 1.0).unwrap(),
        GlcmOffset::new([0, 0, 1]).unwrap(),
    )
    .is_err());
}

fn planned_glcm_texture(block: [usize; 3]) -> Vec<GlcmTextureMeasurements> {
    let quantization = GlcmQuantization::new(4, -2.0, 6.0).unwrap();
    let offset = GlcmOffset::new([0, 0, 1]).unwrap();
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let partials = GlcmTextureOp::new(
        "GLCM texture partials",
        0usize,
        ImageId::supplied(0),
        quantization,
        offset,
        "glcm.partials",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64);
    let merge = MergeGlcmTextureOp::new(
        "GLCM texture merge",
        "glcm.partials",
        1,
        grid.blocks_per_axis(),
        quantization,
        offset,
        "glcm.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    decomposition
        .phases
        .push(fragment_phase(&partials, grid.clone()).unwrap());
    decomposition
        .phases
        .push(fragment_phase(&merge, grid).unwrap());
    decomposition.check().unwrap();

    let (values, _) = colocalization_channels();
    let env =
        ArrayEnvironment::with_inputs(labels(), vec![values.into()], &decomposition, [2, 2, 2])
            .unwrap();
    execute_phases(
        "planned GLCM texture",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&partials),
            PhaseWork::Fragments(&merge),
        ],
    )
    .unwrap();
    collect_glcm_texture_measurements(&env, "glcm.rows", 2, VOLUME, quantization, offset).unwrap()
}

#[test]
fn planned_glcm_ops_validate_streams_and_lattice_at_construction() {
    let quantization = GlcmQuantization::new(4, -2.0, 6.0).unwrap();
    let offset = GlcmOffset::new([0, 0, 1]).unwrap();

    let error = match GlcmTextureOp::new(
        "GLCM texture partials",
        0usize,
        ImageId::supplied(0),
        quantization,
        offset,
        "",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty GLCM partial stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));

    let error = match MergeGlcmTextureOp::new(
        "GLCM texture merge",
        "",
        1,
        [1, 1, 1],
        quantization,
        offset,
        "glcm.rows",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty GLCM merge partial stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));

    let error = match MergeGlcmTextureOp::new(
        "GLCM texture merge",
        "glcm.partials",
        1,
        [1, 0, 1],
        quantization,
        offset,
        "glcm.rows",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("zero-block GLCM merge lattice was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("zero length on axis 1"));

    let error = match MergeGlcmTextureOp::new(
        "GLCM texture merge",
        "glcm.partials",
        1,
        [1, 1, 1],
        quantization,
        offset,
        "",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty GLCM row stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("output stream must not be empty"));
}

#[test]
fn planned_glcm_texture_uses_halo_and_is_decomposition_invariant() {
    let coarse = planned_glcm_texture([6, 4, 3]);
    let split = planned_glcm_texture([2, 2, 2]);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let (values, _) = colocalization_channels();
    let reference = glcm_texture_measurements(
        labels.view(),
        values.view(),
        GlcmQuantization::new(4, -2.0, 6.0).unwrap(),
        GlcmOffset::new([0, 0, 1]).unwrap(),
    )
    .unwrap();
    assert_eq!(split, reference);
    assert!(split.iter().any(|row| row.pairs > 0));
}

#[test]
fn measurement_builder_runs_planned_glcm_texture() {
    let quantization = GlcmQuantization::new(4, -2.0, 6.0).unwrap();
    let offset = GlcmOffset::new([0, 0, 1]).unwrap();
    let plan = Measurements::for_labels(0usize)
        .glcm_texture(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            quantization,
            offset,
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    let texture_rows = plan.texture_rows_with_contract(0).unwrap();
    assert_eq!(texture_rows.stream(), plan.texture_stream(0).unwrap());
    assert_eq!(texture_rows.phase(), plan.texture_rows_phase(0).unwrap());
    let contract = GlcmTextureContract::new(quantization, offset);
    assert_eq!(contract.quantization(), quantization);
    assert_eq!(contract.offset(), offset);
    assert_eq!(contract.parts(), (quantization, offset));
    assert_eq!(plan.glcm_texture_contract(0), Some(contract));
    assert_eq!(plan.texture_contract(0), Some((quantization, offset)));
    assert_eq!(texture_rows.contract(), contract);

    let (values, _) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![values.clone().into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder GLCM texture",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_glcm_texture_rows_with_contract(&env, &texture_rows, VOLUME).unwrap();
    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference =
        glcm_texture_measurements(labels.view(), values.view(), quantization, offset).unwrap();
    assert_eq!(got, reference);
}

#[test]
fn planned_multi_offset_glcm_texture_shares_one_source_scan() {
    let quantization = GlcmQuantization::new(4, -2.0, 6.0).unwrap();
    let offsets = vec![
        GlcmOffset::new([0, 0, 1]).unwrap(),
        GlcmOffset::new([0, 0, -1]).unwrap(),
    ];
    let mut decomposition = base([2, 2, 2]);
    let grid = decomposition.phases[0].grid.clone();
    let partial_streams = vec!["glcm.forward".to_string(), "glcm.backward".to_string()];
    let empty_stream = MultiGlcmTextureOp::new(
        "multi-offset GLCM texture",
        0usize,
        ImageId::supplied(0),
        quantization,
        offsets.clone(),
        vec!["glcm.forward".to_string(), String::new()],
        Lifecycle::DeleteOnExit,
    );
    let error = match empty_stream {
        Ok(_) => panic!("empty multi-offset GLCM stream unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));
    let partials = MultiGlcmTextureOp::new(
        "multi-offset GLCM texture",
        0usize,
        ImageId::supplied(0),
        quantization,
        offsets.clone(),
        partial_streams.clone(),
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64);
    decomposition
        .phases
        .push(fragment_phase(&partials, grid.clone()).unwrap());
    let merges = offsets
        .iter()
        .zip(partial_streams.iter())
        .enumerate()
        .map(|(index, (&offset, stream))| {
            MergeGlcmTextureOp::new(
                "multi-offset GLCM merge",
                stream,
                1,
                grid.blocks_per_axis(),
                quantization,
                offset,
                format!("glcm.rows.{index}"),
                Lifecycle::DeleteOnExit,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for merge in &merges {
        decomposition
            .phases
            .push(fragment_phase(merge, grid.clone()).unwrap());
    }
    decomposition.check().unwrap();
    assert_eq!(decomposition.n_phases(), 4);

    let (values, _) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![values.clone().into()],
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels, PhaseWork::Fragments(&partials)];
    work.extend(
        merges
            .iter()
            .map(|merge| PhaseWork::Fragments(merge as &dyn blockflow::fragment::FragmentOp)),
    );
    execute_phases(
        "planned multi-offset GLCM texture",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let mut planned_rows = Vec::new();
    for (index, &offset) in offsets.iter().enumerate() {
        planned_rows.extend(
            collect_glcm_texture_measurements(
                &env,
                &format!("glcm.rows.{index}"),
                index + 2,
                VOLUME,
                quantization,
                offset,
            )
            .unwrap(),
        );
    }
    let planned_fused = fuse_glcm_texture_measurements(&planned_rows).unwrap();
    let labels = labels().view::<f64>().unwrap().to_owned();
    let mut resident_rows = Vec::new();
    for &offset in &offsets {
        resident_rows.extend(
            glcm_texture_measurements(labels.view(), values.view(), quantization, offset).unwrap(),
        );
    }
    let resident_fused = fuse_glcm_texture_measurements(&resident_rows).unwrap();
    assert_eq!(planned_fused, resident_fused);
}

#[test]
fn measurement_builder_groups_compatible_glcm_texture_offsets() {
    let quantization = GlcmQuantization::new(4, -2.0, 6.0).unwrap();
    let forward = GlcmOffset::new([0, 0, 1]).unwrap();
    let backward = GlcmOffset::new([0, 0, -1]).unwrap();
    let image = IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64);
    let plan = Measurements::for_labels(0usize)
        .glcm_texture(image, quantization, forward)
        .glcm_texture(image, quantization, backward)
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(
        plan.decomposition.n_phases(),
        4,
        "base pixels + one shared GLCM scan + two offset merges"
    );

    let (values, _) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![values.clone().into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder grouped GLCM texture",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let mut planned_rows = Vec::new();
    for index in 0..2 {
        let contract = plan.glcm_texture_contract(index).unwrap();
        let (quantization, offset) = contract.parts();
        assert_eq!(plan.texture_contract(index), Some((quantization, offset)));
        planned_rows.extend(
            collect_glcm_texture_measurements(
                &env,
                plan.texture_stream(index).unwrap(),
                plan.texture_rows_phase(index).unwrap(),
                VOLUME,
                quantization,
                offset,
            )
            .unwrap(),
        );
    }
    let planned_fused = fuse_glcm_texture_measurements(&planned_rows).unwrap();

    let labels = labels().view::<f64>().unwrap().to_owned();
    let mut resident_rows = Vec::new();
    for offset in [forward, backward] {
        resident_rows.extend(
            glcm_texture_measurements(labels.view(), values.view(), quantization, offset).unwrap(),
        );
    }
    let resident_fused = fuse_glcm_texture_measurements(&resident_rows).unwrap();
    assert_eq!(planned_fused, resident_fused);
}

#[test]
fn colocalization_reports_labelled_channel_pair_reductions() {
    let mut labels = Array3::<f64>::zeros((1, 2, 3));
    labels[[0, 0, 0]] = 7.0;
    labels[[0, 0, 1]] = 7.0;
    labels[[0, 0, 2]] = 7.0;
    labels[[0, 1, 0]] = 7.0;
    labels[[0, 1, 1]] = 9.0;
    labels[[0, 1, 2]] = 9.0;

    let mut a = Array3::<f64>::zeros((1, 2, 3));
    a[[0, 0, 0]] = 1.0;
    a[[0, 0, 1]] = 2.0;
    a[[0, 0, 2]] = 3.0;
    a[[0, 1, 0]] = f64::NAN;
    a[[0, 1, 1]] = 5.0;
    a[[0, 1, 2]] = 0.0;

    let mut b = Array3::<f64>::zeros((1, 2, 3));
    b[[0, 0, 0]] = 2.0;
    b[[0, 0, 1]] = 4.0;
    b[[0, 0, 2]] = 6.0;
    b[[0, 1, 0]] = 8.0;
    b[[0, 1, 1]] = 0.0;
    b[[0, 1, 2]] = 2.0;

    let measurements = colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(measurements.len(), 2);

    let first = measurements.iter().find(|row| row.label == 7).unwrap();
    assert_eq!(first.count, 4);
    assert_eq!(first.finite_count, 3);
    assert_eq!(first.sum_a, 6.0);
    assert_eq!(first.sum_b, 12.0);
    assert_eq!(first.sum_a2, 14.0);
    assert_eq!(first.sum_b2, 56.0);
    assert_eq!(first.sum_ab, 28.0);
    assert!((first.pearson().unwrap() - 1.0).abs() < 1.0e-12);
    assert!((first.slope_b_on_a().unwrap() - 2.0).abs() < 1.0e-12);
    assert!((first.overlap_coefficient().unwrap() - 1.0).abs() < 1.0e-12);
    assert_eq!(first.manders_m1(), Some(1.0));
    assert_eq!(first.manders_m2(), Some(1.0));
    assert_eq!(ColocalizationFeature::ALL.len(), 5);
    assert_eq!(
        ColocalizationFeature::OverlapCoefficient.column_name(),
        "overlap_coefficient"
    );
    for feature in ColocalizationFeature::ALL {
        let selected = first.feature(feature);
        let direct = match feature {
            ColocalizationFeature::Pearson => first.pearson(),
            ColocalizationFeature::SlopeBOnA => first.slope_b_on_a(),
            ColocalizationFeature::OverlapCoefficient => first.overlap_coefficient(),
            ColocalizationFeature::MandersM1 => first.manders_m1(),
            ColocalizationFeature::MandersM2 => first.manders_m2(),
        };
        assert_eq!(selected, direct);
    }

    let second = measurements.iter().find(|row| row.label == 9).unwrap();
    assert_eq!(second.count, 2);
    assert_eq!(second.finite_count, 2);
    assert_eq!(second.positive_a, 5.0);
    assert_eq!(second.positive_b, 2.0);
    assert_eq!(second.positive_a_where_b, 0.0);
    assert_eq!(second.positive_b_where_a, 0.0);
    assert_eq!(second.manders_m1(), Some(0.0));
    assert_eq!(second.manders_m2(), Some(0.0));
}

#[test]
fn costes_colocalization_reports_thresholded_manders() {
    let mut labels = Array3::<f64>::zeros((1, 2, 4));
    for x in 0..5 {
        labels[[0, x / 4, x % 4]] = 7.0;
    }
    labels[[0, 1, 1]] = 9.0;

    let mut a = Array3::<f64>::zeros((1, 2, 4));
    let mut b = Array3::<f64>::zeros((1, 2, 4));
    for (x, (av, bv)) in [(1.0, 3.0), (2.0, 2.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)]
        .into_iter()
        .enumerate()
    {
        a[[0, x / 4, x % 4]] = av;
        b[[0, x / 4, x % 4]] = bv;
    }
    a[[0, 1, 1]] = 4.0;
    b[[0, 1, 1]] = f64::NAN;

    let rows = costes_colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(rows.len(), 2);

    let costes = rows.iter().find(|row| row.label == 7).unwrap();
    assert_eq!(costes.finite_count, 5);
    assert_eq!(costes.threshold_a, Some(3.0));
    assert!((costes.threshold_b.unwrap() - 5.8).abs() < 1.0e-12);
    assert_eq!(costes.below_threshold_pearson, Some(-1.0));
    assert_eq!(costes.manders_m1, Some(1.0));
    assert_eq!(costes.manders_m2, Some(0.75));
    assert_eq!(CostesColocalizationFeature::ALL.len(), 5);
    assert_eq!(
        CostesColocalizationFeature::BelowThresholdPearson.column_name(),
        "below_threshold_pearson"
    );
    for feature in CostesColocalizationFeature::ALL {
        let selected = costes.feature(feature);
        let direct = match feature {
            CostesColocalizationFeature::ThresholdA => costes.threshold_a,
            CostesColocalizationFeature::ThresholdB => costes.threshold_b,
            CostesColocalizationFeature::BelowThresholdPearson => costes.below_threshold_pearson,
            CostesColocalizationFeature::MandersM1 => costes.manders_m1,
            CostesColocalizationFeature::MandersM2 => costes.manders_m2,
        };
        assert_eq!(selected, direct);
    }

    let degenerate = rows.iter().find(|row| row.label == 9).unwrap();
    assert_eq!(degenerate.finite_count, 0);
    assert_eq!(degenerate.threshold_a, None);
    assert_eq!(degenerate.manders_m1, None);
    for feature in CostesColocalizationFeature::ALL {
        assert_eq!(degenerate.feature(feature), None);
    }
}

#[test]
fn rank_weighted_colocalization_reports_average_tied_ranks() {
    let mut labels = Array3::<f64>::zeros((1, 3, 3));
    for x in 0..3 {
        labels[[0, 0, x]] = 7.0;
    }
    labels[[0, 1, 0]] = 9.0;
    labels[[0, 1, 1]] = 9.0;
    labels[[0, 2, 0]] = 11.0;

    let mut a = Array3::<f64>::zeros((1, 3, 3));
    let mut b = Array3::<f64>::zeros((1, 3, 3));
    for (x, (av, bv)) in [(1.0, 10.0), (2.0, 30.0), (3.0, 20.0)]
        .into_iter()
        .enumerate()
    {
        a[[0, 0, x]] = av;
        b[[0, 0, x]] = bv;
    }
    a[[0, 1, 0]] = 2.0;
    b[[0, 1, 0]] = 5.0;
    a[[0, 1, 1]] = 2.0;
    b[[0, 1, 1]] = 1.0;
    a[[0, 2, 0]] = f64::NAN;
    b[[0, 2, 0]] = 1.0;

    let rows =
        rank_weighted_colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(rows.len(), 3);

    let ranked = rows.iter().find(|row| row.label == 7).unwrap();
    assert_eq!(ranked.finite_count, 3);
    assert!((ranked.a_weighted_by_b_rank.unwrap() - 13.0 / 18.0).abs() < 1.0e-12);
    assert!((ranked.b_weighted_by_a_rank.unwrap() - 13.0 / 18.0).abs() < 1.0e-12);
    assert_eq!(RankWeightedColocalizationFeature::ALL.len(), 2);
    assert_eq!(
        RankWeightedColocalizationFeature::AWeightedByBRank.column_name(),
        "a_weighted_by_b_rank"
    );
    for feature in RankWeightedColocalizationFeature::ALL {
        let selected = ranked.feature(feature);
        let direct = match feature {
            RankWeightedColocalizationFeature::AWeightedByBRank => ranked.a_weighted_by_b_rank,
            RankWeightedColocalizationFeature::BWeightedByARank => ranked.b_weighted_by_a_rank,
        };
        assert_eq!(selected, direct);
    }

    let tied = rows.iter().find(|row| row.label == 9).unwrap();
    assert_eq!(tied.finite_count, 2);
    assert_eq!(tied.a_weighted_by_b_rank, Some(0.75));
    assert_eq!(tied.b_weighted_by_a_rank, Some(0.75));

    let degenerate = rows.iter().find(|row| row.label == 11).unwrap();
    assert_eq!(degenerate.finite_count, 0);
    assert_eq!(degenerate.a_weighted_by_b_rank, None);
    assert_eq!(degenerate.b_weighted_by_a_rank, None);
    for feature in RankWeightedColocalizationFeature::ALL {
        assert_eq!(degenerate.feature(feature), None);
    }
}

#[test]
fn primitive_colocalization_rows_have_canonical_schema_and_encoder() {
    let rows = vec![
        ColocalizationMeasurements {
            label: 7,
            count: 3,
            finite_count: 2,
            sum_a: 3.0,
            sum_b: 5.0,
            sum_a2: 5.0,
            sum_b2: 13.0,
            sum_ab: 8.0,
            positive_a: 3.0,
            positive_b: 5.0,
            positive_a_where_b: 3.0,
            positive_b_where_a: 5.0,
        },
        ColocalizationMeasurements {
            label: 9,
            count: 1,
            finite_count: 0,
            sum_a: 0.0,
            sum_b: 0.0,
            sum_a2: 0.0,
            sum_b2: 0.0,
            sum_ab: 0.0,
            positive_a: 0.0,
            positive_b: 0.0,
            positive_a_where_b: 0.0,
            positive_b_where_a: 0.0,
        },
    ];
    let encoded = encode_colocalization_measurements(&rows).unwrap();
    let schema = encoded_schema(&encoded).unwrap();
    assert_eq!(schema, colocalization_measurement_schema());
    assert_eq!(schema.columns()[0].name(), "label");
    assert_eq!(schema.columns()[11].name(), "positive_b_where_a");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("coloc.primitive.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("coloc.primitive.rows", 0, [0, 0, 0], &encoded)
        .unwrap();
    let got = collect_colocalization_measurements(&env, "coloc.primitive.rows", 0, VOLUME).unwrap();
    assert_eq!(got, rows);

    let mut invalid = rows[0];
    invalid.label = 0;
    assert!(encode_colocalization_measurements(&[invalid]).is_err());

    invalid = rows[0];
    invalid.finite_count = invalid.count + 1;
    assert!(encode_colocalization_measurements(&[invalid]).is_err());

    invalid = rows[0];
    invalid.sum_a = f64::NAN;
    assert!(encode_colocalization_measurements(&[invalid]).is_err());

    invalid = rows[0];
    invalid.positive_a_where_b = invalid.positive_a + 1.0;
    assert!(encode_colocalization_measurements(&[invalid]).is_err());

    invalid = rows[1];
    invalid.sum_a = 1.0;
    assert!(encode_colocalization_measurements(&[invalid]).is_err());
}

#[test]
fn advanced_colocalization_rows_have_canonical_schemas_and_collectors() {
    let costes_rows = vec![
        CostesColocalizationMeasurements {
            label: 7,
            finite_count: 5,
            threshold_a: Some(3.0),
            threshold_b: Some(5.8),
            below_threshold_pearson: Some(-1.0),
            manders_m1: Some(1.0),
            manders_m2: Some(0.75),
        },
        CostesColocalizationMeasurements {
            label: 9,
            finite_count: 0,
            threshold_a: None,
            threshold_b: None,
            below_threshold_pearson: None,
            manders_m1: None,
            manders_m2: None,
        },
    ];
    let encoded_costes = encode_costes_colocalization_measurements(&costes_rows).unwrap();
    let costes_schema = encoded_schema(&encoded_costes).unwrap();
    assert_eq!(costes_schema, costes_colocalization_measurement_schema());
    assert_eq!(costes_schema.columns().len(), 12);
    assert_eq!(costes_schema.columns()[2].name(), "has_threshold_a");
    assert_eq!(costes_schema.columns()[11].name(), "manders_m2");

    let env = ArrayEnvironment::new(labels(), 1, [2, 2, 2]).unwrap();
    env.declare_sidecar("costes.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("costes.rows", 0, [0, 0, 0], &encoded_costes)
        .unwrap();
    let got_costes =
        collect_costes_colocalization_measurements(&env, "costes.rows", 0, VOLUME).unwrap();
    assert_eq!(got_costes, costes_rows);

    let rank_rows = vec![
        RankWeightedColocalizationMeasurements {
            label: 7,
            finite_count: 3,
            a_weighted_by_b_rank: Some(13.0 / 18.0),
            b_weighted_by_a_rank: Some(13.0 / 18.0),
        },
        RankWeightedColocalizationMeasurements {
            label: 11,
            finite_count: 0,
            a_weighted_by_b_rank: None,
            b_weighted_by_a_rank: None,
        },
    ];
    let encoded_rank = encode_rank_weighted_colocalization_measurements(&rank_rows).unwrap();
    let rank_schema = encoded_schema(&encoded_rank).unwrap();
    assert_eq!(
        rank_schema,
        rank_weighted_colocalization_measurement_schema()
    );
    assert_eq!(rank_schema.columns().len(), 6);
    assert_eq!(rank_schema.columns()[2].name(), "has_a_weighted_by_b_rank");
    assert_eq!(rank_schema.columns()[5].name(), "b_weighted_by_a_rank");

    env.declare_sidecar("rank-weighted.rows", Lifecycle::Persistent)
        .unwrap();
    env.write_sidecar("rank-weighted.rows", 0, [0, 0, 0], &encoded_rank)
        .unwrap();
    let got_rank =
        collect_rank_weighted_colocalization_measurements(&env, "rank-weighted.rows", 0, VOLUME)
            .unwrap();
    assert_eq!(got_rank, rank_rows);

    let mut incomplete_costes = costes_rows[0];
    incomplete_costes.threshold_b = None;
    assert!(encode_costes_colocalization_measurements(&[incomplete_costes]).is_err());

    let mut manders_without_threshold = costes_rows[1];
    manders_without_threshold.manders_m1 = Some(1.0);
    assert!(encode_costes_colocalization_measurements(&[manders_without_threshold]).is_err());

    let mut rank_without_samples = rank_rows[1];
    rank_without_samples.a_weighted_by_b_rank = Some(1.0);
    assert!(encode_rank_weighted_colocalization_measurements(&[rank_without_samples]).is_err());
}

#[test]
fn colocalization_validates_source_shapes_and_labels() {
    let labels = Array3::<f64>::zeros((1, 1, 2));
    let a = Array3::<f64>::zeros((1, 1, 2));
    let b = Array3::<f64>::zeros((1, 2, 2));
    assert!(colocalization_measurements(labels.view(), a.view(), b.view()).is_err());
    assert!(costes_colocalization_measurements(labels.view(), a.view(), b.view()).is_err());
    assert!(rank_weighted_colocalization_measurements(labels.view(), a.view(), b.view()).is_err());

    let mut invalid_labels = Array3::<f64>::zeros((1, 1, 2));
    invalid_labels[[0, 0, 0]] = -1.0;
    let a = Array3::<f64>::zeros((1, 1, 2));
    let b = Array3::<f64>::zeros((1, 1, 2));
    assert!(colocalization_measurements(invalid_labels.view(), a.view(), b.view()).is_err());
    assert!(costes_colocalization_measurements(invalid_labels.view(), a.view(), b.view()).is_err());
    assert!(
        rank_weighted_colocalization_measurements(invalid_labels.view(), a.view(), b.view())
            .is_err()
    );
}

#[test]
fn planned_colocalization_ops_validate_streams_and_lattice_at_construction() {
    let fixed = FixedPoint::bits(24).unwrap();

    let error = match ColocalizationSumsOp::new(
        "colocalization partials",
        0usize,
        ImageId::supplied(0),
        ImageId::supplied(1),
        fixed,
        "",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty colocalization partial stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));

    let error = match ColocalizationPairsOp::new(
        "colocalization pairs",
        0usize,
        ImageId::supplied(0),
        ImageId::supplied(1),
        "",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty colocalization pair stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));

    let error = match MergeColocalizationSumsOp::new(
        "colocalization merge",
        "",
        1,
        [1, 1, 1],
        fixed,
        "coloc.rows",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty colocalization merge partial stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));

    let error = match MergeColocalizationSumsOp::new(
        "colocalization merge",
        "coloc.partials",
        1,
        [1, 0, 1],
        fixed,
        "coloc.rows",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("zero-block colocalization merge lattice was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("zero length on axis 1"));

    let error = match MergeCostesColocalizationOp::new(
        "costes merge",
        "",
        1,
        [1, 1, 1],
        "costes.rows",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty Costes partial stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));

    let error = match MergeCostesColocalizationOp::new(
        "costes merge",
        "costes.partials",
        1,
        [1, 1, 1],
        "",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty Costes output stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("output stream must not be empty"));

    let error = match MergeRankWeightedColocalizationOp::new(
        "rank-weighted merge",
        "",
        1,
        [1, 1, 1],
        "rank.rows",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty rank-weighted partial stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("partial stream must not be empty"));

    let error = match MergeRankWeightedColocalizationOp::new(
        "rank-weighted merge",
        "rank.partials",
        1,
        [1, 1, 1],
        "",
        Lifecycle::DeleteOnExit,
    ) {
        Ok(_) => panic!("empty rank-weighted output stream was accepted"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("output stream must not be empty"));
}

fn colocalization_channels() -> (Array3<f64>, Array3<f64>) {
    let a = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        z as f64 + y as f64 * 0.5 + x as f64 * 0.25 - 2.0
    });
    let b = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        z as f64 * 0.75 - y as f64 * 0.5 + x as f64 * 0.125 + 1.0
    });
    (a, b)
}

fn planned_colocalization(block: [usize; 3]) -> Vec<blockflow::ops::ColocalizationMeasurements> {
    let fixed = FixedPoint::bits(24).unwrap();
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let partials = ColocalizationSumsOp::new(
        "colocalization partials",
        0usize,
        ImageId::supplied(0),
        ImageId::supplied(1),
        fixed,
        "coloc.partials",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64, Dtype::F64);
    let merge = MergeColocalizationSumsOp::new(
        "colocalization merge",
        "coloc.partials",
        1,
        grid.blocks_per_axis(),
        fixed,
        "coloc.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    decomposition
        .phases
        .push(fragment_phase(&partials, grid.clone()).unwrap());
    decomposition
        .phases
        .push(fragment_phase(&merge, grid).unwrap());
    decomposition.check().unwrap();

    let (a, b) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![a.into(), b.into()],
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    execute_phases(
        "planned colocalization",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&partials),
            PhaseWork::Fragments(&merge),
        ],
    )
    .unwrap();
    collect_colocalization_measurements(&env, "coloc.rows", 2, VOLUME).unwrap()
}

fn planned_costes_colocalization(block: [usize; 3]) -> Vec<CostesColocalizationMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let pairs = ColocalizationPairsOp::new(
        "measure Costes colocalization pairs",
        0usize,
        ImageId::supplied(0),
        ImageId::supplied(1),
        "costes.planned.pairs",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64, Dtype::F64);
    let merge = MergeCostesColocalizationOp::new(
        "merge Costes colocalization",
        "costes.planned.pairs",
        1,
        grid.blocks_per_axis(),
        "costes.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    decomposition
        .phases
        .push(fragment_phase(&pairs, grid.clone()).unwrap());
    decomposition
        .phases
        .push(fragment_phase(&merge, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let (a, b) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![a.into(), b.into()],
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    execute_phases(
        "planned Costes colocalization",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&pairs),
            PhaseWork::Fragments(&merge),
        ],
    )
    .unwrap();
    collect_costes_colocalization_measurements(&env, "costes.planned.rows", 2, VOLUME).unwrap()
}

fn planned_rank_weighted_colocalization(
    block: [usize; 3],
) -> Vec<RankWeightedColocalizationMeasurements> {
    let mut decomposition = base(block);
    let grid = decomposition.phases[0].grid.clone();
    let pairs = ColocalizationPairsOp::new(
        "measure rank-weighted colocalization pairs",
        0usize,
        ImageId::supplied(0),
        ImageId::supplied(1),
        "rank-weighted.planned.pairs",
        Lifecycle::DeleteOnExit,
    )
    .unwrap()
    .holding(Dtype::F64, Dtype::F64, Dtype::F64);
    let merge = MergeRankWeightedColocalizationOp::new(
        "merge rank-weighted colocalization",
        "rank-weighted.planned.pairs",
        1,
        grid.blocks_per_axis(),
        "rank-weighted.planned.rows",
        Lifecycle::DeleteOnExit,
    )
    .unwrap();
    decomposition
        .phases
        .push(fragment_phase(&pairs, grid.clone()).unwrap());
    decomposition
        .phases
        .push(fragment_phase(&merge, grid.clone()).unwrap());
    decomposition.check().unwrap();

    let (a, b) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![a.into(), b.into()],
        &decomposition,
        [2, 2, 2],
    )
    .unwrap();
    execute_phases(
        "planned rank-weighted colocalization",
        &workflow(),
        &decomposition,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&pairs),
            PhaseWork::Fragments(&merge),
        ],
    )
    .unwrap();
    collect_rank_weighted_colocalization_measurements(&env, "rank-weighted.planned.rows", 2, VOLUME)
        .unwrap()
}

#[test]
fn planned_colocalization_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_colocalization([6, 4, 3]);
    let split = planned_colocalization([2, 2, 2]);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let (a, b) = colocalization_channels();
    let reference = colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(split.len(), reference.len());

    for (planned, resident) in split.iter().zip(reference.iter()) {
        assert_eq!(planned.label, resident.label);
        assert_eq!(planned.count, resident.count);
        assert_eq!(planned.finite_count, resident.finite_count);
        for (name, got, want) in [
            ("sum_a", planned.sum_a, resident.sum_a),
            ("sum_b", planned.sum_b, resident.sum_b),
            ("sum_a2", planned.sum_a2, resident.sum_a2),
            ("sum_b2", planned.sum_b2, resident.sum_b2),
            ("sum_ab", planned.sum_ab, resident.sum_ab),
            ("positive_a", planned.positive_a, resident.positive_a),
            ("positive_b", planned.positive_b, resident.positive_b),
            (
                "positive_a_where_b",
                planned.positive_a_where_b,
                resident.positive_a_where_b,
            ),
            (
                "positive_b_where_a",
                planned.positive_b_where_a,
                resident.positive_b_where_a,
            ),
        ] {
            assert!(
                (got - want).abs()
                    <= resident.count as f64 * FixedPoint::bits(24).unwrap().resolution(),
                "{name}: planned {got} resident {want}"
            );
        }
        assert!((planned.pearson().unwrap() - resident.pearson().unwrap()).abs() < 1.0e-6);
    }
}

#[test]
fn planned_costes_colocalization_is_decomposition_invariant_and_matches_resident_reference() {
    let coarse = planned_costes_colocalization(VOLUME);
    let split = planned_costes_colocalization([2, 2, 2]);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let (a, b) = colocalization_channels();
    let reference = costes_colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn planned_rank_weighted_colocalization_is_decomposition_invariant_and_matches_resident_reference()
{
    let coarse = planned_rank_weighted_colocalization(VOLUME);
    let split = planned_rank_weighted_colocalization([2, 2, 2]);
    assert_eq!(coarse, split);

    let labels = labels().view::<f64>().unwrap().to_owned();
    let (a, b) = colocalization_channels();
    let reference =
        rank_weighted_colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(split, reference);
}

#[test]
fn measurement_builder_runs_planned_colocalization() {
    let plan = Measurements::for_labels(0usize)
        .colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensityImage::<1>::new(ImageId::supplied(1)).holding(Dtype::F64),
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    let rows = plan.colocalization_rows_with_contract(0).unwrap();
    assert_eq!(rows.stream(), plan.colocalization_stream(0).unwrap());
    assert_eq!(rows.phase(), plan.colocalization_rows_phase(0).unwrap());
    let contract: ColocalizationContract = plan.colocalization_contract(0).unwrap();
    assert_eq!(rows.contract(), contract);
    assert_eq!(contract.labels(), ImageId::from(0usize));
    assert_eq!(contract.channel_a(), ImageId::supplied(0));
    assert_eq!(contract.channel_a_dtype(), Some(Dtype::F64));
    assert_eq!(contract.channel_b(), ImageId::supplied(1));
    assert_eq!(contract.channel_b_dtype(), Some(Dtype::F64));

    let (a, b) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![a.clone().into(), b.clone().into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder colocalization",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_colocalization_rows_with_contract(&env, &rows, VOLUME).unwrap();
    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference = colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(got, reference);
}

#[test]
fn measurement_builder_runs_planned_costes_colocalization() {
    let plan = Measurements::for_labels(0usize)
        .costes_colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensityImage::<1>::new(ImageId::supplied(1)).holding(Dtype::F64),
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    let rows = plan.costes_colocalization_rows_with_contract(0).unwrap();
    assert_eq!(rows.stream(), plan.costes_colocalization_stream(0).unwrap());
    assert_eq!(
        rows.phase(),
        plan.costes_colocalization_rows_phase(0).unwrap()
    );
    assert_eq!(rows.phase(), 2);
    let contract = plan.costes_colocalization_contract(0).unwrap();
    assert_eq!(rows.contract(), contract);
    assert_eq!(contract.labels(), ImageId::from(0usize));
    assert_eq!(contract.channel_a(), ImageId::supplied(0));
    assert_eq!(contract.channel_a_dtype(), Some(Dtype::F64));
    assert_eq!(contract.channel_b(), ImageId::supplied(1));
    assert_eq!(contract.channel_b_dtype(), Some(Dtype::F64));

    let (a, b) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![a.clone().into(), b.clone().into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder Costes colocalization",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_costes_colocalization_rows_with_contract(&env, &rows, VOLUME).unwrap();
    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference = costes_colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(got, reference);
}

#[test]
fn measurement_builder_runs_planned_rank_weighted_colocalization() {
    let plan = Measurements::for_labels(0usize)
        .rank_weighted_colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensityImage::<1>::new(ImageId::supplied(1)).holding(Dtype::F64),
        )
        .stream("objects")
        .build(base([2, 2, 2]))
        .unwrap();
    assert_eq!(plan.rows_phase, None);
    let rows = plan
        .rank_weighted_colocalization_rows_with_contract(0)
        .unwrap();
    assert_eq!(
        rows.stream(),
        plan.rank_weighted_colocalization_stream(0).unwrap()
    );
    assert_eq!(
        rows.phase(),
        plan.rank_weighted_colocalization_rows_phase(0).unwrap()
    );
    assert_eq!(rows.phase(), 2);
    let contract = plan.rank_weighted_colocalization_contract(0).unwrap();
    assert_eq!(rows.contract(), contract);
    assert_eq!(contract.labels(), ImageId::from(0usize));
    assert_eq!(contract.channel_a(), ImageId::supplied(0));
    assert_eq!(contract.channel_a_dtype(), Some(Dtype::F64));
    assert_eq!(contract.channel_b(), ImageId::supplied(1));
    assert_eq!(contract.channel_b_dtype(), Some(Dtype::F64));

    let (a, b) = colocalization_channels();
    let env = ArrayEnvironment::with_inputs(
        labels(),
        vec![a.clone().into(), b.clone().into()],
        &plan.decomposition,
        [2, 2, 2],
    )
    .unwrap();
    let mut work = vec![PhaseWork::Pixels];
    work.extend(plan.phase_work());
    execute_phases(
        "builder rank-weighted colocalization",
        &workflow(),
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )
    .unwrap();

    let got = collect_rank_weighted_colocalization_rows_with_contract(&env, &rows, VOLUME).unwrap();
    let labels = labels().view::<f64>().unwrap().to_owned();
    let reference =
        rank_weighted_colocalization_measurements(labels.view(), a.view(), b.view()).unwrap();
    assert_eq!(got, reference);
}

#[test]
fn unsupported_measurement_combinations_are_refused() {
    let missing_label_source = match Measurements::for_labels(99usize)
        .shape(ShapeSet::basic())
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("missing label source unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(missing_label_source.contains("label image 99 is not image 0"));

    let missing_value_source = match Measurements::for_labels(0usize)
        .intensity(IntensityImage::<0>::new(99usize), IntensitySet::standard())
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("missing value source unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(missing_value_source.contains("intensity image 99 is not image 0"));

    let missing_channel_source = match Measurements::for_labels(0usize)
        .colocalization(
            IntensityImage::<0>::new(1usize),
            IntensityImage::<1>::new(99usize),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("missing colocalization source unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(missing_channel_source.contains("colocalization intensity image 99 is not image 0"));

    let duplicate_basic = match Measurements::for_labels(0usize)
        .shape(ShapeSet::standard())
        .intensity(IntensityImage::<0>::new(1usize), IntensitySet::standard())
        .intensity(IntensityImage::<1>::new(1usize), IntensitySet::standard())
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("duplicate-channel measurement unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(duplicate_basic.contains("image 1 more than once"));

    let f16_labels = match Measurements::for_labels(0usize)
        .labels(LabelImage::new(0usize).holding(Dtype::F16))
        .shape(ShapeSet::basic())
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("f16 label dtype unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(f16_labels.contains("label dtype float16"));

    let f16_values = match Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(1usize).holding(Dtype::F16),
            IntensitySet::standard(),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("f16 intensity dtype unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(f16_values.contains("intensity dtype float16"));

    let empty_stream = match Measurements::for_labels(0usize)
        .stream("")
        .shape(ShapeSet::basic())
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("empty measurement stream unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(empty_stream.contains("output stream root must not be empty"));

    let same_source_distribution = match Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(0usize),
            IntensitySet::distribution(8, 0.0, 8.0).unwrap(),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source distribution unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_distribution.contains("both image 0"));

    let same_source_exact_distribution = match Measurements::for_labels(0usize)
        .exact_distribution(IntensityImage::<0>::new(0usize), 32)
        .unwrap()
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source exact distribution unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_exact_distribution.contains("exact-distribution intensity image"));

    let same_source_texture = match Measurements::for_labels(0usize)
        .glcm_texture(
            IntensityImage::<0>::new(0usize),
            GlcmQuantization::new(2, 0.0, 1.0).unwrap(),
            GlcmOffset::new([0, 0, 1]).unwrap(),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source texture unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_texture.contains("texture intensity image"));

    let same_source_granularity = match Measurements::for_labels(0usize)
        .granularity(
            IntensityImage::<0>::new(0usize),
            GranularitySet::new(vec![1]).unwrap(),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source granularity unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_granularity.contains("granularity intensity image"));

    let same_source_weighted_hu = match Measurements::for_labels(0usize)
        .object_weighted_hu_moments(IntensityImage::<0>::new(0usize), ProjectionAxis::Z)
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source weighted Hu unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_weighted_hu.contains("weighted-Hu intensity image"));

    let same_source_colocalization = match Measurements::for_labels(0usize)
        .colocalization(
            IntensityImage::<0>::new(0usize),
            IntensityImage::<1>::new(1usize),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source colocalization unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_colocalization.contains("colocalization intensity image"));

    let same_source_costes = match Measurements::for_labels(0usize)
        .costes_colocalization(
            IntensityImage::<0>::new(0usize),
            IntensityImage::<1>::new(1usize),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source Costes colocalization unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_costes.contains("colocalization intensity image"));

    let same_source_rank_weighted = match Measurements::for_labels(0usize)
        .rank_weighted_colocalization(
            IntensityImage::<0>::new(0usize),
            IntensityImage::<1>::new(1usize),
        )
        .build(base([3, 2, 3]))
    {
        Ok(_) => panic!("same-source rank-weighted colocalization unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(same_source_rank_weighted.contains("colocalization intensity image"));
}

#[test]
fn fallible_measurement_source_fact_registration_rejects_duplicates() {
    let first = MeasurementSourceFact::new(Dtype::U8, VOLUME).unwrap();
    let second = MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap();
    let mut facts = MeasurementSourceFacts::new();

    facts.try_insert_source(0usize, first).unwrap();
    let error = facts
        .try_insert_source(0usize, second)
        .unwrap_err()
        .to_string();

    assert!(error.contains("source 0 is already registered"));
    assert_eq!(facts.get(0usize).unwrap().dtype(), Dtype::U8);

    let overwrite = facts.with_source(0usize, second);
    assert_eq!(overwrite.get(0usize).unwrap().dtype(), Dtype::F64);
}

#[test]
fn checked_measurement_build_validates_source_facts_before_planning() {
    let plan = Measurements::for_labels(0usize)
        .shape(ShapeSet::standard())
        .intensity(IntensityImage::<0>::new(1usize), IntensitySet::standard())
        .build_checked(base([3, 2, 3]), MeasurementSourceFacts::new())
        .unwrap();
    assert_eq!(plan.rows_phase, Some(2));

    let missing_supplied = match Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .build_checked(base([3, 2, 3]), MeasurementSourceFacts::new())
    {
        Ok(_) => panic!("missing supplied source facts unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(missing_supplied.contains("has no registered dtype and extent"));

    let supplied_without_hints = Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensitySet::standard(),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::from_inputs(&labels(), &[coordinate_values()]).unwrap(),
        )
        .unwrap();
    assert_eq!(
        supplied_without_hints
            .decomposition
            .dtype_at(ImageId::supplied(0).index()),
        Dtype::F64
    );

    let shared_spacing = PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap();
    let shared_frame_facts = MeasurementSourceFacts::from_inputs(&labels(), &[coordinate_values()])
        .unwrap()
        .with_spacing(shared_spacing);
    assert_eq!(
        shared_frame_facts.get(0usize).unwrap().spacing(),
        Some(shared_spacing)
    );
    assert_eq!(
        shared_frame_facts
            .get(ImageId::supplied(0))
            .unwrap()
            .spacing(),
        Some(shared_spacing)
    );
    let checked_shared_frame = Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensitySet::standard(),
        )
        .build_checked(base([3, 2, 3]), shared_frame_facts)
        .unwrap();
    assert_eq!(checked_shared_frame.rows_phase, Some(2));

    let checked_physical_frame = Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensitySet::standard(),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new()
                .with_source(
                    0usize,
                    MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
                )
                .with_source(
                    ImageId::supplied(0),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
                )
                .with_source_spacing(0usize, PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap())
                .unwrap()
                .with_source_spacing(
                    ImageId::supplied(0),
                    PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    assert_eq!(checked_physical_frame.rows_phase, Some(2));

    let missing_frame_source = MeasurementSourceFacts::new()
        .with_source_spacing(99usize, PhysicalSpacing::unit())
        .unwrap_err()
        .to_string();
    assert!(missing_frame_source.contains("source 99 has no registered dtype and extent"));

    let zero_frame_id = MeasurementFrameId::new(0).unwrap_err().to_string();
    assert!(zero_frame_id.contains("coordinate frame id is zero"));

    let frame_id = MeasurementFrameId::new(7).unwrap();
    let checked_named_frame = Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensitySet::standard(),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new()
                .with_source(
                    0usize,
                    MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
                )
                .with_source(
                    ImageId::supplied(0),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
                )
                .with_source_physical_frame(
                    0usize,
                    MeasurementFrame::from_spacing_and_id(
                        PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap(),
                        frame_id,
                    ),
                )
                .unwrap()
                .with_source_physical_frame(
                    ImageId::supplied(0),
                    MeasurementFrame::from_spacing_and_id(
                        PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap(),
                        frame_id,
                    ),
                )
                .unwrap(),
        )
        .unwrap();
    assert_eq!(checked_named_frame.rows_phase, Some(2));

    let physical_frame_id_mismatch = match Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensitySet::standard(),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new()
                .with_source(
                    0usize,
                    MeasurementSourceFact::new(Dtype::F64, VOLUME)
                        .unwrap()
                        .with_physical_frame(MeasurementFrame::from_spacing_and_id(
                            PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap(),
                            MeasurementFrameId::new(7).unwrap(),
                        )),
                )
                .with_source(
                    ImageId::supplied(0),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME)
                        .unwrap()
                        .with_physical_frame(MeasurementFrame::from_spacing_and_id(
                            PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap(),
                            MeasurementFrameId::new(8).unwrap(),
                        )),
                ),
        ) {
        Ok(_) => panic!("mismatched physical source frame id unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(
        physical_frame_id_mismatch.contains("coordinate frame"),
        "{physical_frame_id_mismatch}"
    );
    assert!(
        physical_frame_id_mismatch.contains("label image"),
        "{physical_frame_id_mismatch}"
    );

    let physical_frame_mismatch = match Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensitySet::standard(),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new()
                .with_source(
                    0usize,
                    MeasurementSourceFact::new(Dtype::F64, VOLUME)
                        .unwrap()
                        .with_spacing(PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap()),
                )
                .with_source(
                    ImageId::supplied(0),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME)
                        .unwrap()
                        .with_spacing(PhysicalSpacing::new([0.5, 1.0, 3.0]).unwrap()),
                ),
        ) {
        Ok(_) => panic!("mismatched physical source frame unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(physical_frame_mismatch.contains("physical spacing"));
    assert!(physical_frame_mismatch.contains("label image"));

    let channel_pair_frame_mismatch = match Measurements::for_labels(0usize)
        .colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensityImage::<1>::new(ImageId::supplied(1)),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new()
                .with_source(
                    0usize,
                    MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
                )
                .with_source(
                    ImageId::supplied(0),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME)
                        .unwrap()
                        .with_spacing(PhysicalSpacing::new([0.5, 1.0, 2.0]).unwrap()),
                )
                .with_source(
                    ImageId::supplied(1),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME)
                        .unwrap()
                        .with_spacing(PhysicalSpacing::new([0.5, 1.0, 3.0]).unwrap()),
                ),
        ) {
        Ok(_) => panic!("mismatched physical channel pair unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(channel_pair_frame_mismatch.contains("physical spacing"));
    assert!(channel_pair_frame_mismatch.contains("channel A"));

    let bad_input_shape = match Measurements::for_labels(0usize)
        .shape(ShapeSet::basic())
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::from_inputs(&Array3::<f64>::zeros((2, 2, 2)).into(), &[])
                .unwrap(),
        ) {
        Ok(_) => panic!("mismatched input source extent unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(bad_input_shape.contains("source 0 has extent [2, 2, 2]"));
    assert!(bad_input_shape.contains("base plan expects"));

    let missing_base_image = match Measurements::for_labels(99usize)
        .shape(ShapeSet::basic())
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new().with_source(
                99usize,
                MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
            ),
        ) {
        Ok(_) => panic!("source outside the base plan unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(missing_base_image.contains("is not image 0 or an output"));

    let unrequested_bad_phase_fact = match Measurements::for_labels(0usize)
        .shape(ShapeSet::basic())
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new().with_source(
                99usize,
                MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
            ),
        ) {
        Ok(_) => panic!("unrequested source outside the base plan unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(unrequested_bad_phase_fact.contains("source registry fact"));
    assert!(unrequested_bad_phase_fact.contains("is not image 0 or an output"));

    let dtype_mismatch = match Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new().with_source(
                ImageId::supplied(0),
                MeasurementSourceFact::new(Dtype::U8, VOLUME).unwrap(),
            ),
        ) {
        Ok(_) => panic!("mismatched supplied source dtype unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(dtype_mismatch.contains("was declared as F64"));
    assert!(dtype_mismatch.contains("source registry says U8"));

    let zero_extent_source = MeasurementSourceFact::new(Dtype::F64, [0, 2, 2])
        .unwrap_err()
        .to_string();
    assert!(zero_extent_source.contains("zero length on axis 0"));

    let extent_mismatch = match Measurements::for_labels(0usize)
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new().with_source(
                ImageId::supplied(0),
                MeasurementSourceFact::new(Dtype::F64, [2, 2, 2]).unwrap(),
            ),
        ) {
        Ok(_) => panic!("mismatched supplied source extent unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(extent_mismatch.contains("has extent [2, 2, 2]"));
    assert!(extent_mismatch.contains("base plan expects"));

    let bad_registered_label = match Measurements::for_labels(0usize)
        .shape(ShapeSet::basic())
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new().with_source(
                0usize,
                MeasurementSourceFact::new(Dtype::F16, VOLUME).unwrap(),
            ),
        ) {
        Ok(_) => panic!("registered f16 label source unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(bad_registered_label.contains("source 0 has dtype F16"));
    assert!(bad_registered_label.contains("base plan expects F64"));

    let checked_colocalization = Measurements::for_labels(0usize)
        .colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensityImage::<1>::new(ImageId::supplied(1)),
        )
        .costes_colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensityImage::<1>::new(ImageId::supplied(1)),
        )
        .rank_weighted_colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensityImage::<1>::new(ImageId::supplied(1)),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new()
                .with_source(
                    ImageId::supplied(0),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
                )
                .with_source(
                    ImageId::supplied(1),
                    MeasurementSourceFact::new(Dtype::F32, VOLUME).unwrap(),
                ),
        )
        .unwrap();
    assert!(checked_colocalization.colocalization_rows(0).is_some());
    assert!(checked_colocalization
        .costes_colocalization_rows(0)
        .is_some());
    assert!(checked_colocalization
        .rank_weighted_colocalization_rows(0)
        .is_some());

    let bad_colocalization_extent = match Measurements::for_labels(0usize)
        .rank_weighted_colocalization(
            IntensityImage::<0>::new(ImageId::supplied(0)),
            IntensityImage::<1>::new(ImageId::supplied(1)),
        )
        .build_checked(
            base([3, 2, 3]),
            MeasurementSourceFacts::new()
                .with_source(
                    ImageId::supplied(0),
                    MeasurementSourceFact::new(Dtype::F64, VOLUME).unwrap(),
                )
                .with_source(
                    ImageId::supplied(1),
                    MeasurementSourceFact::new(Dtype::F64, [2, 2, 2]).unwrap(),
                ),
        ) {
        Ok(_) => panic!("mismatched colocalization source extent unexpectedly compiled"),
        Err(error) => error.to_string(),
    };
    assert!(bad_colocalization_extent.contains("colocalization intensity image"));
    assert!(bad_colocalization_extent.contains("has extent [2, 2, 2]"));
}
