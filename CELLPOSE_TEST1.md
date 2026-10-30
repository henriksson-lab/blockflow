  ### Timing

  - Start: 2026-09-22 11:07:20 +02:00
  - Publication completed: approximately 16:40:27 +02:00
  - Operational start-to-finish: 5 h 33 min 7 s
  - Timed processing commands: 5 h 30 min 2 s
  - Segmentation itself took approximately 4 h 10 min

  The extra time came from discovering and fixing the pyramid resampler. It was reading level 0 directly for every scale, reaching 167 GiB RSS. Pyramid generation now proceeds sequentially between adjacent levels; the resumed finalization used only 1.44 GiB peak RSS and took 24 seconds.

  ### Quality check

  About 12% of detected objects have maximum DAPI intensity ≤4. These may be background detections caused by Cellpose’s per-block percentile normalization. The annotation should be visually reviewed in newvolim before downstream quantification.

  ### Follow-ups to remember

  - Compare speed against original Cellpose with identical model, GPU, image region, and timed scope.
  - Test --workers 2 to overlap Zarr I/O and label writing with inference.
  - Profile network, dynamics, reads, and writes separately.
  - GPU was saturated during kernels but active only about 73% overall.
  - Check cooling: GPU reached 88°C, though the driver reported no thermal violation.

  Worked for 5h 45m 51s · done 4:42 PM

