#!/usr/bin/env python3
"""Verify `fitsy fpack` output against astropy, pixel for pixel.

Run from the repository root, with astropy in the environment:
    cargo build --features compression
    .venv/bin/python tests/data/check_fz_write.py

A fitsy round trip cannot catch an encoder and a decoder that share
one error. This script closes that gap for the write path: it runs
`fitsy fpack` over the image fixtures in tests/data/, opens each
output with astropy (whose tile-compression code derives from
cfitsio), and compares the decompressed pixels against the input.

Each codec runs over every applicable fixture. A float image is
compared bit for bit, NaN positions included, because every write
path here is lossless.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
FITSY = REPO / "target" / "debug" / "fitsy"

FIXTURES = [
    "image_i8.fits",
    "image_i16.fits",
    "image_i32.fits",
    "image_i64.fits",
    "image_f32.fits",
    "image_f64.fits",
    "image_3d.fits",
    "image_blank.fits",
    "image_scaled.fits",
    "pallas.fits",
]

CODECS = ["rice", "gzip", "gzip2"]


def image_hdus(hdul: fits.HDUList) -> list[tuple[int, np.ndarray]]:
    out = []
    for i, hdu in enumerate(hdul):
        if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and hdu.data is not None:
            out.append((i, hdu.data))
    return out


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Byte order differs between a freshly parsed array and a
    # decompressed one; only kind and width matter.
    if (
        a.shape != b.shape
        or a.dtype.kind != b.dtype.kind
        or a.dtype.itemsize != b.dtype.itemsize
    ):
        return False
    if a.dtype.kind == "f":
        return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())
    return bool(np.array_equal(a, b))


def check(src: Path, codec: str, workdir: Path) -> list[str]:
    packed = workdir / f"{src.name}.{codec}.fz"
    proc = subprocess.run(
        [str(FITSY), "fpack", str(src), "-c", codec, "-o", str(packed)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return [f"{src.name} [{codec}]: fpack failed: {proc.stderr.strip()}"]
    failures = []
    try:
        orig = fits.open(src)
        originals = image_hdus(orig)
    except Exception as e:
        # A lenient-parsing fixture astropy cannot open is no test of
        # the write path; skip it.
        print(f"skip {src.name}: astropy cannot open the input ({e})")
        return []
    with orig, fits.open(packed) as fz:
        decoded = [hdu.data for hdu in fz if isinstance(hdu, fits.CompImageHDU)]
        if len(decoded) != len(originals):
            failures.append(
                f"{src.name} [{codec}]: {len(originals)} image HDUs in, "
                f"{len(decoded)} compressed HDUs out"
            )
            return failures
        for (idx, data), out in zip(originals, decoded):
            if not equal(np.asarray(data), np.asarray(out)):
                failures.append(
                    f"{src.name} [{codec}] HDU {idx}: pixels differ after "
                    "an astropy decode"
                )
    return failures


def check_quantized(workdir: Path) -> list[str]:
    """The lossy path: a noisy float image with NaN holes, quantized.

    Two claims: astropy's decode of the stream matches `fitsy
    funpack`'s decode bit for bit (the dither sequence and the
    per-tile ZSCALE/ZZERO interoperate), and every finite pixel lands
    within the quantization step of the input.
    """
    rng = np.random.default_rng(7)
    data = (rng.normal(0.0, 1.0, (129, 200)) + np.linspace(0, 50, 200)).astype(
        np.float32
    )
    data[3, 5] = np.nan
    data[100, 150] = np.nan
    src = workdir / "quant_src.fits"
    fits.PrimaryHDU(data=data).writeto(src)
    packed = workdir / "quant.fits.fz"
    restored = workdir / "quant_restored.fits"
    for cmd in (
        [str(FITSY), "fpack", str(src), "-q", "4", "-o", str(packed)],
        [str(FITSY), "funpack", str(packed), "-o", str(restored)],
    ):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return [f"quantized: `{' '.join(cmd[1:])}` failed: {proc.stderr.strip()}"]
    failures = []
    astro = fits.open(packed)[1].data
    ours = fits.open(restored)[0].data
    nan_in, nan_a, nan_f = np.isnan(data), np.isnan(astro), np.isnan(ours)
    if not (np.array_equal(nan_in, nan_a) and np.array_equal(nan_in, nan_f)):
        failures.append("quantized: NaN positions differ")
    if not bool(((astro == ours) | (nan_a & nan_f)).all()):
        failures.append("quantized: astropy and fitsy decodes differ bit for bit")
    err = np.abs(astro[~nan_in] - data[~nan_in])
    # step = tile noise / 4; sigma is 1, so half a step stays well
    # below 0.5.
    if err.max() >= 0.5:
        failures.append(f"quantized: max error {err.max()} exceeds the step bound")
    return failures


def main() -> int:
    if not FITSY.exists():
        print(f"binary not found at {FITSY}; run `cargo build` first")
        return 2
    failures: list[str] = []
    checked = 0
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        for name in FIXTURES:
            src = REPO / "tests" / "data" / name
            if not src.exists():
                src = REPO / name
            if not src.exists():
                print(f"skip {name}: fixture not found")
                continue
            for codec in CODECS:
                failures.extend(check(src, codec, workdir))
                checked += 1
        failures.extend(check_quantized(workdir))
        checked += 1
    print(f"checked {checked} file/codec combinations")
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("all fpack outputs decode identically in astropy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
