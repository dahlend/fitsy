"""Write a new FITS file: image + binary table.

Run from the repo root:

    python examples/python/writing_files.py
"""

import os
import tempfile

import fitsy
import numpy as np

img = np.random.default_rng(0).normal(size=(64, 64)).astype("f4")
tbl = {
    "RA": np.array([10.0, 11.0, 12.0]),
    "DEC": np.array([-5.0, -5.5, -6.0]),
    "NAME": ["a", "bb", "ccc"],
}

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "out.fits")
    fitsy.write(
        path,
        [
            fitsy.image(img, header={"OBJECT": "noise"}),
            fitsy.bintable(tbl, extname="CATALOG"),
        ],
    )

    # Round-trip: read it back and check.
    with fitsy.open(path) as f:
        print("HDU count:", len(f))
        print("primary axes:", f[0].axes)
        print("table columns:", f[1].column_names)
