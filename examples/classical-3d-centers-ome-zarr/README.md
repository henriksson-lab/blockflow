# Classical 3D centres over OME-Zarr

This normal Blockflow example selects one OME-Zarr channel, detects bright 3D
regional maxima with an anisotropic Gaussian, Difference of Gaussians, or
Laplacian of Gaussian response, selects candidate and foreground-like peaks by
Otsu thresholds, and suppresses nearby peaks in physical units. It writes a
spatially indexed point annotation under `tables/<layer>` for newvolim.

Always run it in release mode:

```sh
cargo run --release -p blockflow-classical-3d-centers-ome-zarr -- \
  --zarr /path/to/image.ome.zarr --channel 0 --time 0 \
  --method dog --sigma-um 2 --min-distance-um 4
```

The table records `z,y,x`, the response, raw and smoothed intensities, and the
voxel scale. `table.csv` is emitted for the current viewer compatibility path.

Use `--watershed` to add an instance segmentation seeded by the detected
centres. The example then writes:

- `tables/<layer>-centers`: the original point annotation and scores;
- `labels/<layer>`: the foreground-constrained watershed label pyramid;
- `tables/<layer>`: measurements linked to the instance labels.

The watershed foreground threshold uses a distributed fixed-bin Otsu
histogram. Its temporary storage is proportional to the number of blocks and
histogram bins, rather than the number of voxels. Voxels below the foreground
threshold never enter the watershed queue. The exact watershed currently uses
whole-volume dense arrays, so validate memory on a representative crop before
using `--watershed` on a large volume.

```sh
cargo run --release -p blockflow-classical-3d-centers-ome-zarr -- \
  --zarr /path/to/image.ome.zarr --channel 0 --time 0 \
  --method dog --sigma-um 2 --min-distance-um 4 \
  --watershed --layer classical-3d-watershed-v1
```

For point-only parameter checks, `--crop-start Z Y X --crop-shape Z Y X`
limits processing while keeping point coordinates in the level-0 global
coordinate system. Crop watershed publication is refused because its label
metadata would require an explicit crop translation.

## PBMC validation run

The release binary was run on
`/husky/otherdataset/teresa/single/clustered-pbmcs.ome.zarr` with DoG,
`--sigma-um 2`, `--min-distance-um 4`, 256 histogram bins, blocks of
`32 x 128 x 128`, and two workers.

| Input | Centres | Watershed objects | Wall time | Peak RSS |
|---|---:|---:|---:|---:|
| Crowded `60 x 512 x 512` crop | 41 | 34 | 13.57 s | 735 MiB |
| Full `60 x 1592 x 3333` volume, centres only | 80 | — | 4 min 26.99 s | 591 MiB |

The full table contains 40 centres inside the crowded crop box and 40 outside
it. This agrees with the manual sanity ranges of about 40--50 cells in the
dense centre and 37--40 cells in the sparse outer parts. These approximate
counts do not replace object-level manual review. The published full result is
`tables/classical3d-dog-v1`.
