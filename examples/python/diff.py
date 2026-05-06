"""Compare two FITS files with `fitsy.diff`.

Run from the repo root::

    python examples/python/diff.py
"""

import os
import tempfile

import fitsy
import numpy as np

with tempfile.TemporaryDirectory() as td:
    a = os.path.join(td, "a.fits")
    b = os.path.join(td, "b.fits")

    img = np.arange(64, dtype="f4").reshape(8, 8)
    fitsy.write(a, [fitsy.image(img, header={"OBJECT": "before"})])

    img2 = img.copy()
    img2[0, 0] = 99.0
    fitsy.write(b, [fitsy.image(img2, header={"OBJECT": "after"})])

    d = fitsy.diff(a, b, rtol=0.0, max_diffs=10)
    print("identical?", d.identical)
    print("hdu_counts:", d.hdu_counts)
    print(d)  # multi-line human-readable summary

    # Ignore non-physical keyword churn (e.g. CHECKSUM) for round-trip
    # comparisons of independently-written files.
    d2 = fitsy.diff(a, b, ignore_keywords=["OBJECT"])
    print("after ignoring OBJECT:", d2.identical)
