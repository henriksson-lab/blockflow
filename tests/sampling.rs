use blockflow::{dtype::Dtype, error::Result, sampling::*, voxels::Voxels};
use ndarray::Array3;
struct Virtual {
    layout: SourceLayout,
    reads: usize,
    masked: Option<[usize; 3]>,
}
impl SampleSource for Virtual {
    fn layout(&self) -> &SourceLayout {
        &self.layout
    }
    fn read_chunk(&mut self, start: [usize; 3], shape: [usize; 3]) -> Result<SampleChunk> {
        self.reads += 1;
        // Nonrepeating coordinate field, no allocation proportional to image.
        let f = |(z, y, x): (usize, usize, usize)| {
            1000. + 3. * (start[0] + z) as f64 + 2. * (start[1] + y) as f64 + (start[2] + x) as f64
        };
        let values = match self.layout.dtype {
            Dtype::U16 => Voxels::from(Array3::from_shape_fn(shape, |p| f(p) as u16)),
            Dtype::I16 => Voxels::from(Array3::from_shape_fn(shape, |p| f(p) as i16 - 2000)),
            Dtype::F32 => Voxels::from(Array3::from_shape_fn(shape, |p| f(p) as f32)),
            _ => Voxels::from(Array3::from_shape_fn(shape, f)),
        };
        let valid = self.masked.map(|bad| {
            (0..shape[0])
                .flat_map(|z| {
                    (0..shape[1]).flat_map(move |y| {
                        (0..shape[2])
                            .map(move |x| [z + start[0], y + start[1], x + start[2]] != bad)
                    })
                })
                .collect()
        });
        Ok(SampleChunk { values, valid })
    }
}
fn source(dtype: Dtype) -> Virtual {
    Virtual {
        layout: SourceLayout {
            shape_zyx: [11, 13, 17],
            chunk_zyx: [3, 4, 5],
            dtype,
            origin_xyz: [-11., 17., 31.],
            spacing_xyz: [0.7, 1.3, 2.1],
            additional_workspace_bytes: 128,
        },
        reads: 0,
        masked: None,
    }
}
fn budget() -> SamplingBudget {
    SamplingBudget {
        max_working_bytes: 1_000_000,
        max_chunk_reads: 1000,
        max_decoded_bytes: 1_000_000,
    }
}
#[test]
fn typed_seams_edges_and_gradients_use_calibrated_pixel_centres() {
    for dtype in [Dtype::U16, Dtype::I16, Dtype::F32, Dtype::F64] {
        let mut s = source(dtype);
        let q = [
            [4.8, 3.7, 2.9],
            [0., 0., 0.],
            [16., 12., 10.],
            [10.3, 8.2, 6.1],
        ];
        let p: Vec<_> = q
            .iter()
            .map(|q| {
                std::array::from_fn(|i| s.layout.origin_xyz[i] + s.layout.spacing_xyz[i] * q[i])
            })
            .collect();
        let mut out = vec![Sample::default(); p.len()];
        let stats = sample_points(&mut s, &p, &mut out, budget()).unwrap();
        for (q, v) in q.iter().zip(out) {
            assert!(v.valid);
            assert!(
                (v.value
                    - (1000. + q[0] + 2. * q[1] + 3. * q[2]
                        - if dtype == Dtype::I16 { 2000. } else { 0. }))
                .abs()
                    < 1e-8
            );
            for i in 0..3 {
                assert!(
                    (v.gradient_xyz[i] - [1., 2., 3.][i] / s.layout.spacing_xyz[i]).abs() < 1e-8
                );
            }
        }
        assert_eq!(
            stats.peak_planned_working_bytes,
            sampling_working_bytes(&s.layout, p.len()).unwrap()
        );
        assert_eq!(stats.chunk_reads, s.reads);
        assert!(stats.peak_planned_working_bytes <= budget().max_working_bytes);
    }
}
#[test]
fn invalidity_is_distinct_from_preflight_memory_and_io_errors() {
    let mut s = source(Dtype::F64);
    s.masked = Some([3, 4, 5]);
    let q = [[5.2, 4.1, 3.3], [100., 2., 2.], [f64::NAN, 2., 2.]];
    let p: Vec<_> = q
        .iter()
        .map(|q| std::array::from_fn(|i| s.layout.origin_xyz[i] + s.layout.spacing_xyz[i] * q[i]))
        .collect();
    let mut out = vec![Sample::default(); p.len()];
    let stats = sample_points(&mut s, &p, &mut out, budget()).unwrap();
    assert!(out.iter().all(|v| !v.valid && v.value == 0.));
    assert_eq!(stats.outside_points, 2);
    assert_eq!(stats.invalid_points, 3);
    for tiny in [
        SamplingBudget {
            max_working_bytes: 1,
            ..budget()
        },
        SamplingBudget {
            max_chunk_reads: 0,
            ..budget()
        },
        SamplingBudget {
            max_decoded_bytes: 0,
            ..budget()
        },
    ] {
        s.reads = 0;
        assert!(sample_points(&mut s, &p, &mut out, tiny).is_err());
        assert_eq!(s.reads, 0);
    }
}
#[test]
fn virtual_volume_size_does_not_increase_planned_working_memory() {
    let mut s = source(Dtype::F64);
    let p = [[-8.5, 20., 35.]];
    let mut out = [Sample::default()];
    let small = sample_points(&mut s, &p, &mut out, budget()).unwrap();
    let value = out[0];
    s.layout.shape_zyx = [1_000_000; 3];
    let large = sample_points(&mut s, &p, &mut out, budget()).unwrap();
    assert_eq!(
        small.peak_planned_working_bytes,
        large.peak_planned_working_bytes
    );
    assert_eq!(small.decoded_bytes, large.decoded_bytes);
    assert_eq!(value, out[0]);
}
#[test]
fn singleton_section_has_zero_normal_gradient() {
    let mut s = source(Dtype::F32);
    s.layout.shape_zyx[0] = 1;
    s.layout.chunk_zyx[0] = 1;
    let p = [[-8.5, 20., 31.], [-8.5, 20., 32.]];
    let mut out = [Sample::default(); 2];
    sample_points(&mut s, &p, &mut out, budget()).unwrap();
    assert!(out[0].valid);
    assert_eq!(out[0].gradient_xyz[2], 0.);
    assert!(!out[1].valid);
}

#[test]
fn a_masked_derivative_neighbor_does_not_erase_an_observed_pixel_centre() {
    let mut s = source(Dtype::F64);
    s.masked = Some([3, 4, 6]);
    let q = [5., 4., 3.];
    let p = [std::array::from_fn(|i| {
        s.layout.origin_xyz[i] + q[i] * s.layout.spacing_xyz[i]
    })];
    let mut output = [Sample::default()];
    let stats = sample_points(&mut s, &p, &mut output, budget()).unwrap();
    assert!(output[0].valid);
    assert!(!output[0].gradient_valid);
    assert!((output[0].value - 1022.).abs() < 1e-9);
    assert_eq!(stats.invalid_points, 0);
    assert_eq!(stats.invalid_gradients, 1);
}

#[test]
fn empty_batch_needs_no_source_workspace_or_io() {
    let mut s = source(Dtype::F64);
    let stats = sample_points(
        &mut s,
        &[],
        &mut [],
        SamplingBudget {
            max_working_bytes: 0,
            max_chunk_reads: 0,
            max_decoded_bytes: 0,
        },
    )
    .unwrap();
    assert_eq!(stats.peak_planned_working_bytes, 0);
    assert_eq!(s.reads, 0);
}

#[test]
fn physical_centre_roundoff_does_not_bridge_a_masked_boundary() {
    let mut s = source(Dtype::F64);
    s.masked = Some([3, 3, 5]);
    let p = [[
        s.layout.origin_xyz[0] + 5. * s.layout.spacing_xyz[0],
        s.layout.origin_xyz[1] + 4. * s.layout.spacing_xyz[1],
        s.layout.origin_xyz[2] + 3. * s.layout.spacing_xyz[2],
    ]];
    let mut out = [Sample::default()];
    sample_points(&mut s, &p, &mut out, budget()).unwrap();
    assert!(out[0].valid);
    assert!((out[0].value - 1022.).abs() < 1e-9);
}

#[test]
fn public_memory_preflight_matches_sampler_admission_without_allocating() {
    let mut s = source(Dtype::U16);
    assert_eq!(sampling_working_bytes(&s.layout, 0).unwrap(), 0);
    assert!(sampling_working_bytes(&s.layout, usize::MAX).is_err());
    let p = [[-10.3, 18.3, 33.1]];
    let mut output = [Sample::default()];
    let planned = sampling_working_bytes(&s.layout, p.len()).unwrap();
    assert!(sample_points(
        &mut s,
        &p,
        &mut output,
        SamplingBudget {
            max_working_bytes: planned - 1,
            ..budget()
        }
    )
    .is_err());
    assert_eq!(s.reads, 0);
    let stats = sample_points(
        &mut s,
        &p,
        &mut output,
        SamplingBudget {
            max_working_bytes: planned,
            ..budget()
        },
    )
    .unwrap();
    assert_eq!(stats.peak_planned_working_bytes, planned);
    assert!(output[0].valid);
}
