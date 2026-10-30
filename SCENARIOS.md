# Biology Image-Analysis Scenarios

Purpose: choose realistic examples for this framework. These scenarios are
based on common ImageJ/Fiji and CellProfiler workflows. They are not all
implementation commitments; missing operations are called out so later work can
prioritize by example value.

## Research Sources

- CellProfiler examples: human cells, fly cells, tumor counting, comet assay,
  percent-positive scoring, yeast colonies/patches, tissue neighbors, wound
  healing, illumination correction, colocalization, speckle counting, object
  tracking and worm workflows.
  https://cellprofiler.org/examples
- CellProfiler `IdentifyPrimaryObjects`: object identification, thresholding,
  declumping by intensity/shape/LoG, watershed, object filtering and direct
  count/location measurements.
  https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.4/modules/objectprocessing.html
- CellProfiler measurement modules: size/shape, intensity, texture,
  colocalization, skeleton measurements.
  https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureObjectSizeShape.html
  https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-3.1.9/modules/measurement.html
  https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageIntensity.html
- ImageJ particle analysis: threshold, optional watershed, analyze particles,
  count/area/average size and ROI output.
  https://imagej.net/imaging/particle-analysis
- ImageJ colocalization overview and related imaging topics.
  https://imagej.net/imaging/colocalization-analysis
- Fiji TrackMate: spot/cell detection, linking over time, track analysis such
  as velocity/displacement and per-spot intensity/radius/orientation.
  https://imagej.net/plugins/trackmate/
- ImgLib2 core/library docs and paper: generic n-dimensional image processing,
  type/container/view separation, virtual images, sections/views, cells cached
  to disk, ImageJ2/Fiji integration and benchmark claims.
  https://github.com/imglib/imglib2
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3496339/
- ImgLib2 examples and documentation: Gaussian convolution, Fourier
  convolution/FFT, interpolation, sparse data, out-of-bounds views, morphology
  and LoopBuilder pixel loops.
  https://imglib2.net/imglib2/examples
  https://imglib2.net/imglib2/morphological-operations
  https://imglib2.net/imglib2/loopbuilder
  https://imglib2.net/imglib2/benchmarks

## Scenario Set

### 1. Nuclei/Cell Counting And Morphology

Biology: count nuclei or cells in fluorescence microscopy; measure object size,
shape and location.

Typical workflow:

1. Correct illumination or subtract background.
2. Smooth or enhance objects.
3. Threshold foreground.
4. Split touching objects by distance watershed or maxima-seeded watershed.
5. Remove border objects and size outliers.
6. Measure count, centroid, area/volume, perimeter/surface, axis lengths,
   eccentricity, solidity and shape descriptors.

Blockflow example shape:

- `background`, `smooth`, `threshold`, `distance`, `regional maxima`,
  `watershed`, `cleanup`, `measure`.
- Export object rows and image-level count summaries.

Likely coverage:

- Strong: thresholding, morphology, distance transform, watershed, cleanup,
  labels, many measurements.
- Check/gaps: a high-level "IdentifyPrimaryObjects-like" recipe builder would
  make this example much easier to read.

### 2. Per-Cell Intensity And Percent-Positive Scoring

Biology: quantify marker expression per cell and report percent positive.

Typical workflow:

1. Segment nuclei/cells.
2. Measure intensity per object in one or more channels.
3. Apply fixed, global or control-derived threshold to per-object intensity.
4. Report positive count, negative count, percent positive and per-object table.

Blockflow example shape:

- Label image plus intensity channel(s).
- `Measurements::for_labels(...).intensity(...)`.
- Derived table step for classification/scoring.

Likely coverage:

- Strong: per-object intensity, multi-channel basic intensity fusion,
  thresholding, row/table outputs.
- Check/gaps: table-level classification helpers and percent-positive summary
  examples.

### 3. Cytoplasm-Nucleus Translocation

Biology: quantify whether a marker moves from cytoplasm to nucleus after
treatment.

Typical workflow:

1. Segment nuclei as primary objects.
2. Segment whole cells from nuclei plus cytoplasmic/cell stain.
3. Define cytoplasm as cell minus nucleus.
4. Measure nuclear and cytoplasmic marker intensity.
5. Compute ratio or difference per cell.

Blockflow example shape:

- Nuclei labels, cell labels, tertiary cytoplasm labels.
- Per-region intensity measurements.
- Join rows by parent cell/nucleus relation.

Likely coverage:

- Partial: segmentation, label operations, intensity measurement.
- Gaps: explicit secondary/tertiary object construction recipes, parent-child
  relation joins and per-object ratio helpers.

### 4. Foci/Speckle Counting Inside Nuclei

Biology: count DNA damage foci, RNA spots, protein puncta or other subnuclear
spots per nucleus.

Typical workflow:

1. Segment nuclei.
2. Enhance small bright spots with LoG/difference-of-Gaussians/top-hat.
3. Detect regional maxima or thresholded connected components.
4. Relate foci to containing nuclei.
5. Report foci count, foci intensity and foci-per-nucleus aggregates.

Blockflow example shape:

- Nucleus label image.
- Spot-enhancement filter, spot detection, label/point output.
- Object relationship or containment aggregation.

Likely coverage:

- Strong: LoG/edges/features may exist through kernels and interest/response
  peaks; regional maxima and point/table outputs exist.
- Gaps: polished "foci inside parent object" relation helper, containment join,
  per-parent aggregate summaries.

### 5. Wound Healing / Scratch Assay

Biology: quantify closure of a wound/scratch area in a cell monolayer.

Typical workflow:

1. Correct illumination/background.
2. Segment tissue/cell-covered region, often without individual cells.
3. Clean mask by morphology and hole filling.
4. Measure open wound area, covered area, width and closure over time.

Blockflow example shape:

- Binary mask pipeline: background, threshold, morphology, fill holes.
- Global area and possibly per-row/column width profile.

Likely coverage:

- Strong: thresholding, morphology, fill holes, area via tabulation.
- Gaps: line/profile measurements for wound width and time-series summary.

### 6. Colocalization Assay

Biology: quantify spatial coincidence between two fluorescent channels.

Typical workflow:

1. Optional registration/chromatic shift correction.
2. Optional segmentation or mask restriction.
3. Measure Pearson, overlap, Manders, Costes thresholded Manders and rank
   weighted colocalization over whole image or per object.

Blockflow example shape:

- Two intensity channels plus optional label mask.
- Colocalization measurements and object-restricted colocalization rows.

Likely coverage:

- Strong: primitive colocalization, Costes, rank-weighted colocalization and
  measurement rows are present.
- Gaps: chromatic shift calibration and registration recipe.

### 7. Texture Phenotyping

Biology: distinguish cell types, tissue phenotypes, organelle patterns or
colonies by texture.

Typical workflow:

1. Segment objects or define tissue regions.
2. Quantize intensities.
3. Compute GLCM/Haralick texture at one or more offsets/scales.
4. Combine texture with shape and intensity for classification.

Blockflow example shape:

- Label image plus intensity channel.
- `glcm_texture` measurement with multi-offset grouping.
- Feature table for downstream classifier.

Likely coverage:

- Strong: GLCM texture measurements and grouped offsets exist.
- Gaps: end-to-end classifier training/evaluation examples.

### 8. Tissue Neighbor / Contact Analysis

Biology: quantify tissue architecture: how many neighbors each cell has, shared
boundary length/area and nearest-cell relationships.

Typical workflow:

1. Segment adjacent cells in tissue.
2. Find object boundaries and contact pairs.
3. Count touching neighbors and contact area.
4. Optionally compute centroid-distance or boundary-distance neighbors.

Blockflow example shape:

- Label image.
- Boundary/contact measurement, touching-neighbor summaries, centroid/boundary
  relationship rows.

Likely coverage:

- Strong: boundary/contact rows, touching neighbor summaries, centroid and
  boundary-distance relationships exist.
- Gaps: polished tissue-neighbor example and visualization output.

### 9. Colony / Plaque / Tumor Nodule Counting

Biology: count and size yeast colonies, plaques, tumor nodules or other large
round-ish objects.

Typical workflow:

1. Illumination correction/background subtraction.
2. Global or adaptive threshold.
3. Cleanup, hole filling and optional watershed.
4. Measure count, area, circularity, intensity and classify by size/shape.

Blockflow example shape:

- Global threshold or local threshold, morphology cleanup, connected components,
  shape measurements.

Likely coverage:

- Strong: thresholding, cleanup, connected components, measurements.
- Gaps: grid/plate alignment recipes for 96-well or colony-array layouts.

### 10. Comet Assay

Biology: measure DNA damage from comet head/tail shape and intensity.

Typical workflow:

1. Correct background/illumination.
2. Segment each comet.
3. Split head and tail, often by intensity/shape along a major axis.
4. Measure tail length, tail intensity, head intensity and percent DNA in tail.

Blockflow example shape:

- Segmentation plus orientation/principal-axis measurement.
- Region split along axis and intensity summaries.

Likely coverage:

- Partial: shape, principal axes, intensity and background are available.
- Gaps: comet-specific head/tail splitting and profile/axis integration helper.

### 11. Cell / Organelle / Particle Tracking

Biology: follow cells, nuclei, vesicles, bacteria or organelles through time and
measure velocity, displacement, division or intensity changes.

Typical workflow:

1. Detect spots or segment cell-like objects in each frame.
2. Measure per-object position/intensity/shape.
3. Link objects between frames into tracks.
4. Compute track duration, displacement, velocity, diffusion and lineage events.

Blockflow example shape:

- Frame-wise segmentation or spot detection.
- Row table of detections.
- Tracking/linking stage over rows.

Likely coverage:

- Partial: detection, labels, points and measurements exist.
- Major gaps: time axis abstractions, track linking algorithms, track tables,
  lineage events and track-level measurements.

### 12. Neurite / Vasculature / Branching Skeleton Analysis

Biology: measure neurite outgrowth, blood vessels, roots or other branching
networks.

Typical workflow:

1. Segment elongated structures.
2. Skeletonize mask.
3. Relate skeleton to seed objects such as soma/nuclei.
4. Measure skeleton length, trunks, branches, endpoints and branch points.

Blockflow example shape:

- Threshold/morphology, skeletonization, graph/branch measurement.

Likely coverage:

- Partial: skeletonization exists.
- Gaps: skeleton graph extraction, branch/end-point classification, per-seed
  skeleton assignment and branch metrics.

### 13. 3-D Object Analysis

Biology: quantify nuclei, spheroids, organoids, embryos, tissue volumes or
segmented organelles in z-stacks.

Typical workflow:

1. Segment volume using threshold/watershed/random walker/level sets.
2. Cleanup in 3-D.
3. Measure volume, surface area, Feret diameter, convex hull, moments,
   topology and per-object intensity.

Blockflow example shape:

- 3-D label image plus intensity image.
- Object geometry, voxel-face convex hull, topology, components, intensity.

Likely coverage:

- Strong: many 3-D object measurements, topology, convex hull variants,
  random walker and level-set references.
- Gaps: marching cubes surface meshes and polished 3-D visualization/export.

### 14. Illumination Correction / Background Removal

Biology: normalize uneven illumination before segmentation or intensity
measurement.

Typical workflow:

1. Estimate smooth background field from blank/reference images or from image
   morphology/smoothing.
2. Divide or subtract background.
3. Apply downstream segmentation/measurement.

Blockflow example shape:

- Large-scale morphology/background chain, smoothing, voxelwise arithmetic.

Likely coverage:

- Strong enough for examples using morphology/smoothing chains.
- Gaps: CellProfiler-like correction-field calculation/apply recipe, batch
  aggregate correction across many images.

### 15. Registration / Alignment Before Measurement

Biology: align channels, time points, fields, template grids or repeated
samples before measurement.

Typical workflow:

1. Estimate transform or load known transform.
2. Apply affine/projective/remap/polar transform.
3. Measure aligned channels or combine tiles.

Blockflow example shape:

- Transform ops plus downstream measurement.

Likely coverage:

- Partial: affine/projective/remap/rotate/polar/log-polar transforms exist.
- Gaps: transform estimation, feature matching, chromatic-shift calibration,
  stitching/mosaicking.

## Suggested Example Priority

1. Nuclei/cell counting and morphology.
2. Per-cell intensity and percent-positive scoring.
3. Colocalization per image and per object.
4. Foci/speckles inside nuclei.
5. Wound healing area assay.
6. Tissue neighbor/contact analysis.
7. 3-D object measurement.
8. Texture phenotyping.
9. Random-walker/level-set segmentation examples.
10. Tracking and skeleton analysis, after missing row/time/graph pieces are
    implemented.

## Missing-Ops Summary

High-value gaps for later implementation:

- Secondary/tertiary object construction recipes: cells from nuclei,
  cytoplasm = cell minus nucleus, parent-child relations.
- Parent/child and containment joins for foci-per-nucleus and cell compartment
  workflows.
- Time-lapse tracking: frame axis, linking, track tables and track-level
  measurements.
- Skeleton graph metrics: branch points, endpoints, trunks, branch lengths and
  seed-object assignment.
- Plate/grid workflows: well/grid alignment and per-well summaries.
- Comet-assay profile helpers: major-axis head/tail split and axis-integrated
  intensity.
- Registration estimation: feature matching, chromatic shift correction,
  stitching/mosaicking.
- Mesh/surface export for 3-D: marching cubes and mesh measurements.
- High-level recipe builders that package common ImageJ/CellProfiler pipelines
  without hiding the underlying ops.

## ImgLib2 Benchmark Notes

Assumption: "imglib" here means current ImgLib2, not the older ImgLib1. ImgLib2
is worth treating as a benchmark target for library design and performance, not
only as a catalog of biology workflows.

### What ImgLib2 Is Good At

- Generic n-dimensional processing. Algorithms are intended to work across 2-D,
  3-D, time, channel and higher-dimensional data without rewriting loops.
- Generic pixel types. Numeric and non-numeric pixel types are represented
  through type interfaces rather than fixed image structs.
- Storage abstraction. Algorithms can run over array-backed, planar, cell/tiled,
  disk-cached, on-demand and virtual data sources through common access APIs.
- Cheap views. Crops, hyperslices, extensions and other views can be represented
  without copying data.
- Out-of-bounds policies. Convolution/stencil code can choose mirror, constant,
  periodic or other extension behavior through views.
- Real-coordinate access. Interpolated views and `realtransform` support
  nearest/linear/Lanczos interpolation, affine transforms, homographies,
  deformation fields and other real transforms.
- Pixelwise loops. `LoopBuilder` expresses one-to-six-image pointwise kernels,
  supports multithreading, chunk-local reductions and specialized iteration
  strategies for image containers.
- Neighborhood abstractions. Examples include hyperspheres and local
  neighborhoods for filters, local minima/maxima and texture-like operations.
- Built-in algorithm families. Common examples include Gaussian convolution,
  Fourier convolution/FFT, interpolation, sparse-data interpolation and
  morphology.
- Morphology. Dilation, erosion, opening and closing are available for arbitrary
  dimensionality, with structuring element support and decomposed shapes.
- Sparse data. Real-point sample lists, KD-trees, nearest-neighbor search,
  k-nearest search, radius search and sparse interpolation are supported.

### What To Benchmark Later

Use benchmarks that separate "algorithm surface" from "storage/access model".
ImgLib2's strongest claim is genericity without too much performance loss, so
benchmarking only one dense array loop would understate what it is designed to
do.

1. Dense pointwise arithmetic:
   - Blockflow voxelwise/LoopBuilder equivalent over `u8`, `f32`, `f64`.
   - Single input, two inputs, and three-input expression.
   - Warmed runs matter because ImgLib2/Java relies on JIT optimization.
2. Stencil/convolution:
   - Gaussian blur or separable convolution.
   - Compare border policies: mirror/clamp/constant where both frameworks can
     express them.
   - Include 2-D and 3-D.
3. Morphology:
   - Erode/dilate/open/close with a ball/box structuring element.
   - Include decomposed-vs-direct structuring elements if Blockflow exposes an
     equivalent.
4. Views/crops/hyperslices:
   - Apply a pointwise or convolution op to a cropped view/slice without
     copying.
   - This tests Blockflow's block planning against ImgLib2's view abstraction.
5. Tiled/cell storage:
   - ImgLib2 `CellImg` or disk-cached cells vs Blockflow chunks/cache/prefetch.
   - Use volumes larger than cache and record repeated-read behavior.
6. Real transforms:
   - Affine/projective/remap-like transform with linear interpolation.
   - Include transform-only and transform-followed-by-measurement.
7. Sparse point data:
   - Nearest-neighbor or inverse-distance rendering from sparse samples.
   - Blockflow may need explicit point/KD-tree support before this is fair.
8. Connected image-analysis workflow:
   - Threshold -> morphology -> connected components/labels -> measurements.
   - This is closer to biology use than microbenchmarks.

### Benchmark Caveats

- Java warmup is real. Record cold and warmed timings separately, matching the
  ImgLib2 benchmark style where repeated iterations expose JIT optimization.
- Compare equivalent storage layouts. Dense array vs dense array is different
  from tiled/cached vs tiled/cached.
- Compare equivalent border behavior and dimensionality. ImgLib2 makes
  out-of-bounds policy explicit; Blockflow examples should do the same.
- Include memory footprint and IO/fetch counts for large volumes. Runtime alone
  will hide why a plan wins.
- Separate single-thread and multithreaded cases. ImgLib2 `LoopBuilder` can
  split chunks across threads; Blockflow should report worker count and
  scheduling policy.
- For biology examples, compare end-to-end rows as well as pixels. Object
  tables and measurement semantics matter more than only image throughput.

### Possible Blockflow Gaps Exposed By ImgLib2

- General virtual/view API for cheap crops, hyperslices and out-of-bounds
  extension as first-class user-facing concepts.
- Generic sparse sample interpolation and KD-tree nearest-neighbor operations.
- A polished pointwise expression builder comparable to `LoopBuilder`.
- More explicit border-policy choices for all stencil and transform examples.
- Benchmark harness that can compare dense, chunked and cached layouts under
  the same operation.
