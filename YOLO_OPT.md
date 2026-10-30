# YOLO GPU optimization findings

## Target

Match the original FP32 PyTorch YOLOv11 throughput on the Quadro RTX 5000 while
preserving the current detection agreement. The matched workload is 32
invocations of 640 x 640 RGB images at batch size one. The reference median is
0.838 seconds and 38.18 images/s.

The target is achieved. The LibTorch CUDA path with four Blockflow workers has
a 0.758 second median, 42.22 images/s, and is 1.11× faster than the original.
All 158 detections match.

## Where the time goes

Synchronized stage timings after warmup give these representative per-image
costs:

| Stage | CubeCL baseline | Original PyTorch |
|---|---:|---:|
| RGB construction | 4.5 ms | included below |
| Resize | 4.0 ms | included below |
| HWC u8 to CHW f32 | 3.8 ms | included below |
| Decode and CPU preparation | 12.3 ms total | 9.5 ms |
| Host to GPU tensor creation | 5.8 ms | 1.6 ms |
| Model forward | 27-29 ms | 9.7 ms |
| NMS and result download | 1.5-1.8 ms | 6.5 ms |

The model split is approximately 12 ms in the backbone, 6 ms in the FPN, and
9.4 ms in the detection head. About 7.6 ms of the head is its convolution
branches. Anchor generation and decode account for about 1.6 ms together.

Eight Blockflow workers improve the complete phase from about 2.13 seconds to
1.46 seconds by overlapping reads and CPU preparation. Sixteen workers regress
to 1.78 seconds. Parallelism hides some adapter work but does not remove the
model bottleneck.

## CUDA evidence

Nsight Systems attributes 71.4% of Burn/CubeCL GPU kernel time to
`direct_conv2d_kernel`. Across the captured steady-state forwards, CubeCL
convolution takes about 15 ms per image. PyTorch uses cuDNN Winograd and GEMM
kernels and spends roughly 2 ms per image in convolution kernels.

CubeCL launches roughly 760 kernels per forward, compared with about 450 for
the PyTorch path. The extra launches add several milliseconds of host overhead.
Exhaustive CubeCL autotuning still selected the same class of kernels and gave
2.129 seconds end to end, so changing the autotune level does not address the
gap.

The Rust NMS copies the complete 2.82 MB prediction tensor to the CPU for every
image. It is fast on this sparse fixture, but the trace records 98.8 MB of
device-to-host traffic. PyTorch filters and suppresses on the GPU and downloads
only retained detections.

## Optimization order

1. **Use a cuDNN-class convolution backend. Completed.** The `yolo-libtorch`
   feature runs the same Burn graph through PyTorch/LibTorch 2.9 and cuDNN. Its
   model forward is about 7.5 ms per image, compared with 27-29 ms for CubeCL.
2. **Keep inference fusion enabled.** Calling `YOLO::fuse()` after loading
   weights removes BatchNorm execution, preserves all 158 matched detections,
   and reduces model time modestly.
3. **Cache fixed-shape anchors and strides.** `make_anchors` currently reads the
   stride tensor back to the CPU and uploads newly allocated anchor tensors on
   every forward. Cache them for each feature-map shape.
4. **Reduce kernel launches.** Fuse Conv, bias and SiLU where the backend
   permits it, and remove avoidable copies, slices, and scalar kernels. This is
   secondary to convolution throughput.
5. **Upload bytes and convert on the GPU.** The adapter currently converts RGB
   bytes into a 4.9 MB CHW f32 allocation before upload. Upload 1.23 MB of u8
   data, then cast, reorder and normalize on the GPU, preferably with pinned
   reusable staging buffers.
6. **Avoid redundant CPU image work.** Read u8 Blockflow buffers without
   widening through f64, and skip the resize copy when a block is already 640 x
   640. These steps mainly improve latency and reduce the workers needed to
   saturate the GPU.
7. **Move candidate filtering and NMS to the GPU.** Download only retained
   detections. This matters more for models with many classes or dense scenes
   than for the present fixture.

## Acceptance check

After each model/backend change, run the release benchmark three times at batch
size one and compare all detections. The optimization is complete when the
Blockflow median is within 5% of the original median with no unmatched
detections. Also record the best worker count separately so scheduling gains do
not hide a slower single-invocation model.
