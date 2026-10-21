# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Adaptations that make the reference's checkout run in parallel here.
Neither touches an algorithm; importing this module applies both, and forked
workers inherit them.

1. **Worker sinks reopened in `w+`.** `ClearMap.IO.MMP.VirtualSource.as_real`
   reopens a memory-mapped file in the mode it was created with. A block
   processing sink created by `io.initialize` is in mode `w+`, so every worker
   that touches its block re-creates the file (`open_memmap(mode='w+')`
   truncates it), and a worker writing into its mapping at that moment dies
   of SIGBUS: `vasculature.binarize` with a file sink and more than two
   processes fails with `BrokenProcessPool` (with two it mostly survives).
   Here a worker reopens such a file `r+` (it exists: the parent created it).
2. **Rank filter masks.** The environment's build of the rank filters' Cython
   takes `const char` masks and structuring elements, which the Python side
   hands over as uint8 / bool: a type view, as
   `vessel-graph/reference-clearmap/binarize_roles.py` does.
"""
import numpy as np

import ClearMap.IO.MMP as _mmp
import ClearMap.ImageProcessing.Filter.Rank.Rank as _rank

_as_real = _mmp.VirtualSource.as_real


def _as_real_existing(self):
    if getattr(self, "_mode", None) == "w+":
        self._mode = "r+"
    return _as_real(self)


_mmp.VirtualSource.as_real = _as_real_existing

_initialize_selem = _rank._initialize_selem
_rank._initialize_selem = lambda selem, ndim: np.asarray(_initialize_selem(selem, ndim), dtype=np.int8)
_apply_code = _rank._apply_code


def _apply_code_int8_mask(*a, **kw):
    mask = kw.get("mask")
    if mask is not None and mask.dtype != np.int8:
        kw["mask"] = np.ascontiguousarray(mask).view(np.int8)
    return _apply_code(*a, **kw)


_rank._apply_code = _apply_code_int8_mask
