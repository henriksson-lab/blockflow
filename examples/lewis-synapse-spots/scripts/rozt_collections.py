"""ROZT analysis.build-collection job payloads over every published lewis-synapse
table: one UMAP and one PCA collection of the per-synapse features.
Usage: python3 -I rozt_collections.py OME_ZARR_DIR OUT_DIR [DATASET_PREFIX] [OUTPUT_DIR_IN_ROOT]
Writes OUT_DIR/job-umap.json and job-pca.json; POST each to ROZT's /v1/jobs.
DATASET_PREFIX is how ROZT names the images (default `lewis-omezarr-`); the
collections go to the `teresa` output root under OUTPUT_DIR_IN_ROOT."""
import sys, os, glob, json

SIZE = ['area_um2', 'eq_radius_um', 'inradius_um']
LAYOUT = [
    'eccentricity', 'solidity', 'spot_density_um2', 'spot_area_fraction',
    'spot_area_median_um2', 'spot_area_iqr_um2', 'spot_major_median_um', 'spot_minor_median_um',
    'spot_ecc_median', 'pixel_cv', 'spot_int_cv', 'spot_int_gini', 'spot_int_log_p90_p10',
    'punctate_fraction', 'spot_radial_index', 'spot_radial_index_w',
    'radial_bin1', 'radial_bin2', 'radial_bin3', 'radial_bin4', 'central_enrichment',
    'radial_spot_density_bin1', 'radial_spot_density_bin2', 'radial_spot_density_bin3',
    'radial_spot_density_bin4', 'polarity', 'nn_mean_um', 'clark_evans', 'nn_cv',
    'ripley_h_0p5um', 'ripley_h_1um', 'ripley_h_1p5um', 'ripley_h_0p1R', 'ripley_h_0p2R',
]
FEATURES = SIZE + LAYOUT
PCA_COMPONENTS = 10
MEASUREMENTS = FEATURES + ['n_spots', 'touches_frame', 'background', 'area_px', 'raw_mean']


def main(zdir, out, prefix='lewis-omezarr-', where='lewis/analysis'):
    names = sorted(os.path.basename(p)[:-len('.ome.zarr')] for p in glob.glob(f'{zdir}/*.ome.zarr')
                   if os.path.isdir(f'{p}/tables/lewis-synapse'))
    sources = [dict(datasetId=prefix + n, tableId='lewis-synapse', identityDomain='labels/lewis-synapse',
                    identityColumn='label_id', xColumn='centroid_x', yColumn='centroid_y', zColumn='centroid_z',
                    radiusColumn='radius_px', fallbackRadius=100.0, locatorSpace='voxel') for n in names]
    computations = {
        'umap': dict(kind='umap', featureColumns=FEATURES, nNeighbors=15, minDist=0.3, spread=1.0,
                     pcaDimensions=20, initializationSeed=42),
        'pca': dict(kind='pca', featureColumns=FEATURES),
    }
    os.makedirs(out, exist_ok=True)
    for kind, computation in computations.items():
        cid = f'lewis-synapse-{kind}'
        # PCA stores PCA_COMPONENTS components; any pair can be chosen as display axes later.
        axes = ([f'pca_{i}' for i in range(1, PCA_COMPONENTS + 1)] if kind == 'pca'
                else [f'{kind}_x', f'{kind}_y'])
        job = dict(
            definitionId='analysis.build-collection', inputDataset=sources[0]['datasetId'],
            payload=dict(collectionId=cid, displayName=f'Lewis LFA-1 synapses · {kind.upper()}', sources=sources,
                         measurementColumns=MEASUREMENTS,
                         embedding=dict(id=kind, displayName=kind.upper(), xColumn=axes[0], yColumn=axes[1],
                                        **(dict(zColumn=axes[2], additionalColumns=axes[3:]) if len(axes) > 2 else {}),
                                        computation=computation),
                         displayRecipes=[], datasetDisplayRecipes=[]),
            output=dict(kind='separateDataset', configuredRootId='teresa',
                        relativeName=f'{where}/{cid}.rozt-analysis', conflictPolicy='replace'))
        with open(f'{out}/job-{kind}.json', 'w') as f:
            json.dump(job, f)
    print(f'{len(sources)} sources, {len(FEATURES)} embedding features')


if __name__ == '__main__':
    main(*sys.argv[1:5])
