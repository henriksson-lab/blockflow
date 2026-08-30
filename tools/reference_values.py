#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate the recorded reference values the ground-truth tests pin.

Every number in this crate's tests that came from outside the crate was printed
by this script, so the recordings are reproducible rather than magic. Run a
subcommand and paste its output into the test file it names.

    python3 tools/reference_values.py --versions
    python3 tools/reference_values.py gaussian     # tests/gaussian_kernel.rs
    python3 tools/reference_values.py resample     # tests/resample_ground_truth.rs

`--versions` prints the versions the recordings in the tree were taken under; a
test that names a version names one of those.
"""

import argparse
import sys

import numpy as np
import scipy
import scipy.ndimage as ndi


def versions() -> None:
    print(f"numpy   {np.__version__}")
    print(f"scipy   {scipy.__version__}")
    try:
        import skimage

        print(f"skimage {skimage.__version__}")
    except ImportError:
        print("skimage not installed")


def rust_f64(value: float) -> str:
    """A float64 literal Rust parses back to the same bits.

    `repr` of a Python float is the shortest string that round-trips through
    IEEE-754 double, and Rust's `f64` parser is correctly rounded, so the bits
    survive the trip.
    """
    text = repr(float(value))
    if "e" in text or "E" in text or "." in text:
        return text
    return text + ".0"


MASK64 = (1 << 64) - 1


def lcg_volume(shape):
    """The fixture volume, built by the same recurrence the Rust side uses.

    A 64-bit LCG with Knuth's MMIX constants, seeded at 1, taking bits 40..63
    of each state and scaling into `[0, 1)`. The volume is therefore a function
    of its shape and of nothing else, and neither side has to ship an array.
    """
    state = 1
    count = int(np.prod(shape))
    out = np.empty(count, dtype=np.float64)
    for index in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
        out[index] = (state >> 40) / 16777216.0
    return out.reshape(shape)


# --------------------------------------------------------------- gaussian --


def gaussian() -> None:
    # Only (sigma, truncate) pairs where scipy's `int(truncate * sigma + 0.5)`
    # and this crate's `ceil(truncate * sigma)` give the same radius. Where they
    # differ the two are filtering with different kernel widths, and comparing
    # them would measure the truncation convention rather than the weights.
    pairs = [
        (0.5, 4.0),
        (1.0, 4.0),
        (1.0, 3.0),
        (1.25, 4.0),
        (1.5, 3.0),
        (2.0, 4.0),
        (2.5, 3.0),
    ]
    print("// tests/gaussian_kernel.rs :: SCIPY_KERNELS")
    print(f"// scipy {scipy.__version__}, _gaussian_kernel1d(sigma, order=0, radius)")
    for sigma, truncate in pairs:
        radius = int(truncate * sigma + 0.5)
        assert radius == int(np.ceil(truncate * sigma)), (sigma, truncate)
        weights = ndi._filters._gaussian_kernel1d(sigma, 0, radius)
        body = ", ".join(rust_f64(w) for w in weights)
        print(f"    ({rust_f64(sigma)}, {rust_f64(truncate)}, &[{body}]),")

    shape = (6, 5, 4)
    volume = lcg_volume(shape)
    sigma = (1.5, 0.5, 1.0)
    print()
    print("// tests/gaussian_kernel.rs :: the fixture, so a drifted generator is caught")
    print(f"//   lcg_volume({shape})[0, 0, 0] = {rust_f64(volume[0, 0, 0])}")
    print(f"//   lcg_volume({shape})[5, 4, 3] = {rust_f64(volume[-1, -1, -1])}")
    print(f"//   sum                          = {rust_f64(volume.sum())}")
    print()
    print("// tests/gaussian_kernel.rs :: SCIPY_REFLECT / SCIPY_NEAREST")
    print(
        f"// scipy {scipy.__version__}, gaussian_filter(lcg_volume({shape}), "
        f"sigma={sigma}, truncate=4.0, mode=<mode>), C order"
    )
    for mode in ("reflect", "nearest"):
        field = ndi.gaussian_filter(volume, sigma=sigma, truncate=4.0, mode=mode)
        print(f"// mode='{mode}'")
        flat = field.ravel(order="C")
        for start in range(0, flat.size, 4):
            row = ", ".join(rust_f64(v) for v in flat[start : start + 4])
            print(f"    {row},")


# --------------------------------------------------------------- resample --


def resample() -> None:
    """`tests/resample_ground_truth.rs`.

    Compared through `Resample::to_extent`, not `Resample::new`, and the reason
    is that the two libraries turn a requested *factor* into an output extent
    differently: this crate takes `floor(in * up / down)` and keeps the exact
    rational `up/down` as the scale, while `scipy.ndimage.zoom` takes
    `round(in * zoom)` and then rescales to `out / in`. Stating both extents
    removes that question and leaves exactly the sample map, which is what the
    comparison is about. Checked here: at these extents the crate's ratio
    `Ratio::new(out, in)` and SciPy's effective zoom `out / in` are the same
    number.
    """
    shape = (7, 5, 4)
    target = (5, 8, 3)
    volume = lcg_volume(shape)
    zoom = tuple(t / s for t, s in zip(target, shape))
    field = ndi.zoom(volume, zoom=zoom, order=1, mode="nearest", grid_mode=True)
    assert field.shape == target, (field.shape, target)

    print("// tests/resample_ground_truth.rs :: the fixture")
    print(f"//   lcg_volume({shape})[0, 0, 0] = {rust_f64(volume[0, 0, 0])}")
    print(f"//   lcg_volume({shape})[6, 4, 3] = {rust_f64(volume[-1, -1, -1])}")
    print(f"//   sum                          = {rust_f64(volume.sum())}")
    print()
    print("// tests/resample_ground_truth.rs :: SCIPY_ZOOM")
    print(
        f"// scipy {scipy.__version__}, zoom(lcg_volume({shape}), zoom={zoom}, order=1,"
    )
    print(f"//   mode='nearest', grid_mode=True) -> shape {target}, C order")
    flat = field.ravel(order="C")
    for start in range(0, flat.size, 4):
        row = ", ".join(rust_f64(v) for v in flat[start : start + 4])
        print(f"    {row},")


COMMANDS = {"gaussian": gaussian, "resample": resample}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    parser.add_argument("--versions", action="store_true")
    args = parser.parse_args()
    if args.versions or args.command is None:
        versions()
        return 0
    COMMANDS[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
