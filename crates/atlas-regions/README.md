# atlas-regions

Atlas region lookups for registered light-sheet data: an annotation volume and
its region tree, applied to points and masks; atlas orientation and
preparation. Shared by the cell-density, vessel-graph and atlas-registration
examples. The library needs no blockflow features; the `zarr` feature adds
OME-Zarr reading and writing of atlases and the `atlas-prepare` CLI.

Originally developed as part of a ClearMap reimplementation.

## Pipeline

1. Read the region tree into an `Ontology` (from values, or from an Allen-style
   JSON-lines structure file).
2. Build the distance-to-surface field once per atlas with
   `distance_to_surface(annotation, &DistanceParams)`: the mask
   `annotation > 0` through `blockflow::ops::distance`.
3. Map points (cells, graph vertices, polyline points) into atlas voxel
   coordinates (registration is not this crate's job) and call
   `Annotator::columns` to get region id, hemisphere and distance per point,
   optionally collapsed onto the ancestor at a hierarchy level.
4. Fold a per-point column onto segments with `polyline_minima`, for example
   the smallest distance to the surface along each vessel edge.

## Orientation

`Orientation` is a signed axis permutation in `[z, y, x]` order: output axis
`a` is input axis `source[a]`, reversed when `flip[a]` (transpose, then flip).
The reference writes orientations as signed 1-based tuples in its own
`(x, y, z)` array order, which is the transpose of ours, so
`Orientation::from_xyz(o)` sets, for our axis `a`,
`source[a] = 3 - |o[2 - a]|` and `flip[a] = o[2 - a] < 0`. Examples:
`(1, -2, 3)` flips `y`; `(-1, 2, 3)` (`"right"`) flips `x`; `(2, -1, 3)` swaps
`y` and `x` and flips the new `y`.

Points are voxel coordinates with voxel centres at integers, so a flipped axis
maps `i` to `n - 1 - i`; `shape` is always the **unoriented** volume's shape,
in both directions, as in the reference's `orient_points`.

## Atlas preparation

`prepare_atlas` follows the reference's `prepare_annotation_files`: orient,
then crop in the oriented frame (`Crop`, Python slices; `Crop::parse_xyz`
takes the reference's slicing order). An optional `HemisphereSelection` then
narrows the crop to the bounding box of one hemisphere label plus a margin. In
the shipped 25 µm Allen atlas the hemispheres split our `z` (left-right) axis
at 228: `255` below, `0` above (so `0` is a hemisphere there, not background).
`PreparedWindow` records orientation, source shape and crop, and maps points
between source and prepared voxels (`to_prepared`, `to_source`).

### `atlas-prepare` and the atlas layout

```sh
cargo build --release -p atlas-regions --features zarr --bin atlas-prepare
target/release/atlas-prepare OUT.ome.zarr \
  --annotation A.ome.zarr --reference R.ome.zarr \
  --hemispheres H.ome.zarr --distance-to-surface D.ome.zarr \
  --ontology ABA_annotation_last.json \
  [--orientation 1,-2,3] [--slicing-xyz :,:,0:256 | --crop z0:z1,y0:y1,x0:x1] \
  [--hemisphere 255 --hemisphere-margin 4] [--voxel-size-um 25] [--replace]
```

The ontology may be the nested structure graph (the reference's default,
`ABA_annotation_last.json`) or the JSON-lines copy. The output is one
OME-Zarr 0.5 image:

```text
OUT.ome.zarr/
  zarr.json              reference multiscale; attributes.atlas = {name, voxel_size_um,
                         window (PreparedWindow::to_json), distance_to_surface_unit: "voxel", ...}
  0 .. 4                 reference levels (2x block means), 25 µm at level 0
  labels/annotation      uint64 region ids; image-label.properties per region:
                         label-value, name, acronym, parent, level (tree depth,
                         universe = 0, the reference's level), st_level,
                         color_hex_triplet; image-label.colors (RGBA)
  labels/hemispheres     uint64 hemisphere labels (0 and 255 for the Allen atlas)
  distance_to_surface/   nested multiscale float32 image, in voxels
```

The distance to the surface is a continuous field, so it is a nested image
rather than a label; its values are in voxels as the reference ships them
(multiply by `voxel_size_um`, 25, for micrometres). `open_prepared_atlas`
reads everything back, with the region tree rebuilt from the properties.

Source volumes are converted from the reference's TIFFs with
`img2omezarr convert` (decompress `.tif.lzma` with `xz -dc --format=lzma`).
The annotation and distance TIFFs carry no axis metadata, and img2omezarr
then reads their pages as time points; rewrite them first as OME-TIFF with
`ZYX` axes (for example with `tifffile.imwrite(..., ome=True,
metadata={'axes': 'ZYX'})`).

## API

```rust
// region tree
Ontology::new(records: impl IntoIterator<Item = (i64, Option<i64>, Option<i64>)>) -> Result<Ontology> // (id, level, parent)
Ontology::from_jsonl(text: &str) -> Result<Ontology>
Ontology::from_jsonl_file(path: impl AsRef<Path>) -> Result<Ontology>
ontology.collapse(id: i64, level: i64) -> Result<i64>   // ancestor at or above `level`
ontology.ancestors(id) -> Result<Vec<i64>>              // id, parent, ..., root
ontology.is_within(id, ancestor) -> Result<bool>
ontology.get(id) -> Option<RegionNode>                  // { level, parent }
ontology.name(id) / ontology.acronym(id) -> Option<&str>
ontology.ids(), len(), contains(id), set_names(id, RegionNames), set_color(id, [u8; 3])
Ontology::from_ontology_json(text: &str, OntologyLevel) -> Result<Ontology>   // nested graph
Ontology::from_ontology_json_file(path, OntologyLevel) -> Result<Ontology>
Ontology::from_file(path, OntologyLevel) -> Result<Ontology>                  // either format
Ontology::from_image_label_properties(&[Value], OntologyLevel) -> Result<Ontology>
OntologyLevel::{TreeDepth (default), StructureLevel}
ontology.parent(id) / level(id) / structure_level(id) -> Option<i64>
ontology.depth(id) -> Result<usize>; children(id) -> Vec<i64>; roots() -> Vec<i64>
ontology.color(id) -> Option<[u8; 3]>

// point lookups
AnnotateParams { level: Option<i64>, annotation_invalid: i64, hemisphere_invalid: i64,
                 rounding: Rounding, distance_out_of_bounds: OutOfBounds }  // Default: None, 0, -1, Nearest, Clamp
Annotator::new(params).with_ontology(Arc<Ontology>)
annotator.columns(AtlasVolumes, &[[f64; 3]]) -> Result<AtlasColumns>
annotator.regions(&Voxels, &[[f64; 3]]) -> Result<Vec<i64>>
annotator.hemispheres(&Voxels, &[[f64; 3]]) -> Result<Vec<i64>>
annotator.distances(&Voxels, &[[f64; 3]]) -> Result<Vec<f64>>
AtlasVolumes::new(&annotation).with_hemispheres(&v).with_distance_to_surface(&v)
AtlasColumns { annotation: Vec<i64>, hemisphere: Vec<i64>, distance_to_surface: Vec<f64> }

// primitives
voxel_index(point, shape, Rounding, OutOfBounds) -> Option<[usize; 3]>
label_at(&Voxels, [usize; 3]) -> Result<i64>
field_at(&Voxels, [usize; 3]) -> Result<f64>
polyline_minima(values: &[f64], ranges: &[[usize; 2]]) -> Result<Vec<f64>>

// surface
mask_from_annotation(&Voxels) -> Array3<bool>                       // annotation > 0
distance_to_surface(&Voxels, &DistanceParams) -> Result<Voxels>     // F64 field

// orientation
Orientation::IDENTITY; Orientation::new(source: [usize; 3], flip: [bool; 3]) -> Result<_>
Orientation::from_xyz([i32; 3]) -> Result<_>; to_xyz() -> [i32; 3]; parse(&str) -> Result<_>
o.inverse(), o.then(&next), o.source(), o.flip(), o.is_identity()
o.orient_shape([usize; 3]) / unorient_shape(oriented_shape)
o.orient_resolution([f64; 3]) / unorient_resolution
o.orient_point(p, shape) / orient_points(&[_], shape)       // shape: unoriented
o.unorient_point(p, shape) / unorient_points(&[_], shape)   // shape: unoriented
o.orient_voxels(&Voxels) -> Voxels / unorient_voxels; orient_array(&Array3<T>)

// preparation
AxisRange { start: Option<i64>, stop: Option<i64> }; AxisRange::parse("a:b"); resolve(len)
Crop([AxisRange; 3]); Crop::ALL; parse_zyx(&str); parse_xyz(&str); from_xyz([_; 3])
HemisphereSelection { value: i64, margin: usize }
PrepareParams { orientation, crop, hemisphere: Option<HemisphereSelection> }
AtlasStack { annotation: Voxels, reference, hemispheres, distance_to_surface: Option<Voxels> }
prepare_atlas(&AtlasStack, &PrepareParams) -> Result<(AtlasStack, PreparedWindow)>
PreparedWindow { orientation, source_shape, start, shape }
  .to_prepared(p) / .to_source(p) / .to_json() / ::from_json(&Value)

// atlas_zarr (feature `zarr`)
read_volume(&Path) -> Result<Voxels>
read_atlas_sources(&AtlasSources) -> Result<AtlasStack>
write_prepared_atlas(root, &AtlasStack, &PreparedWindow, &Ontology, &WriteAtlasParams)
  -> Result<WrittenAtlas>
open_prepared_atlas(root, OntologyLevel) -> Result<PreparedAtlas>
  // PreparedAtlas { root, stack, window, voxel_size_um, ontology }
region_properties(&Ontology) -> Result<Vec<Value>>; region_colors(&Ontology) -> Vec<Value>
```

## Input

- Points in **atlas voxel coordinates**, voxel centres at integers (use
  `Rounding::Floor` if voxel `i` spans `[i, i + 1)`). Row `i` of every output
  is point `i`.
- Integer annotation and hemisphere volumes as `blockflow::voxels::Voxels`; a
  float annotation is refused rather than truncated. The distance field may be
  any non-64-bit-integer type. The volumes need not share a shape: each lookup
  is bounded by the volume it reads.
- The structure file may write integers as JSON floats (`"st_level": 2.0`), as
  the commonly shipped `ABA_annotation_last.jsonl` does.
- Levels differ between the two structure files: the JSON-lines reader uses
  `st_level`, while the reference's annotation class (and
  `OntologyLevel::TreeDepth`, the default for the nested file) uses the depth
  in the tree with `universe` at 0. `st_level` is kept beside the depth either
  way.

## Run

```sh
cargo test --release -p atlas-regions
# with the shipped Allen tree:
ATLAS_REGIONS_TREE=/path/to/ABA_annotation_last.jsonl \
  cargo test --release -p atlas-regions -- --include-ignored
# OME-Zarr writer, and the prepared Allen atlas (ATLAS_REGIONS_PREPARED):
cargo test --release -p atlas-regions --features zarr -- --include-ignored
```

## Output

Plain Rust values (`Vec<i64>`, `Vec<f64>`, `Voxels`); writing them into an
object table or OME-Zarr is up to the example that uses them.

## Validation

- Unit tests on keyed synthetic volumes (each voxel's value names its
  coordinates), so a wrong axis or rounding shows up as a wrong number:
  rounding, bounds, sentinels, the collapse (and that it skips out-of-atlas
  rows), refusals of unknown ids, duplicates, cycles, float atlases and
  64-bit fields.
- The distance field is checked on a slab with a known profile and with an
  anisotropic pitch; the transform itself is tested in blockflow.
- An end-to-end synthetic test (`tests/atlas.rs`) runs tree, field, lookup and
  polyline fold together.
- An ignored test loads the shipped 1328-record Allen tree and checks that
  every region collapses at every level and that known olfactory and cortical
  regions merge where expected.

- Orientation is checked against the reference's `orientation.py`
  (`orient`, `orient_points`, `orient_shape`, `orient_resolution`, both
  directions) for twelve orientations on a 4 x 3 x 2 array:
  `tests/orientation_reference.rs`, fixture from
  `scripts/orientation_fixtures.py`. All agree exactly.
- Preparation against the reference's `prepare_annotation_files` body on the
  2017 Allen atlas with orientation `(1, -2, 3)` and slicing `z 0:256` (its
  CellMap example): annotation, hemispheres and reference identical; distance
  to surface within 3.8e-6 voxels (float32 storage). `atlas-prepare` takes
  31 s and 4.4 GB peak for the full atlas.

## Limitations

- Everything is in memory: lookups read whole `Voxels`, and
  `distance_to_surface` is resident (about 8 bytes a voxel; 616 MB for the
  25 µm mouse atlas). For larger volumes, append
  `blockflow::ops::distance::append_to` to a plan on the mask instead.
- Lookups are single-threaded; they are cheap next to the volume reads.
- Nearest-voxel sampling only, no interpolation of the distance field.
- No transform: points must already be in the atlas frame.
- Preparation is resident: every volume, oriented, in memory (fine for 25 µm
  atlases, about 2 GB with all four).
- A hemisphere selection crops; it does not clear the other hemisphere's
  voxels inside the margin.
