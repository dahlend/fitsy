"""Read binary and ASCII tables: columns, rows, and structured arrays.

Run from the repo root::

    python examples/python/tables.py
"""

import os
import tempfile

import fitsy
import numpy as np

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "catalog.fits")
    fitsy.write(
        path,
        [
            fitsy.image(np.zeros((1, 1), dtype="i2")),  # required primary
            fitsy.bintable(
                {
                    "RA": np.array([10.0, 20.0, 30.0]),
                    "DEC": np.array([-5.0, 0.0, 5.0]),
                    "NAME": ["alpha", "beta", "gamma"],
                },
                extname="CATALOG",
            ),
        ],
    )

    with fitsy.open(path) as f:
        tbl = f["CATALOG"]
        print("columns:", tbl.column_names)
        print("nrows  :", len(tbl))

        # Whole-table structured numpy array (zero-copy where possible).
        arr = tbl.data
        print("structured dtype:", arr.dtype)

        # Single row by integer index.
        row = tbl[0]
        print("row 0 :", row["RA"], row["DEC"], row["NAME"])

        # Slice of rows -> list of row dicts.
        first_two = tbl[:2]
        print("first two RAs:", [r["RA"] for r in first_two])

        # Pull a column by name -> 1-D numpy array.
        ras = tbl["RA"]
        print("RA column dtype:", ras.dtype)
