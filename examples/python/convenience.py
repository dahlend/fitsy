"""Module-level convenience functions: `getdata`, `getval`, `setval`,
`delval`, `info`, `append`.

These mirror ``astropy.io.fits`` and are the right tool when you only
need a single read or write and don't want to manage a `FitsFile`.

Run from the repo root::

    python examples/python/convenience.py
"""

import os
import tempfile

import fitsy
import numpy as np

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "scratch.fits")
    fitsy.write(
        path,
        [
            fitsy.image(
                np.arange(16, dtype="i2").reshape(4, 4), header={"OBJECT": "demo"}
            )
        ],
    )

    # One-shot reads.
    arr = fitsy.getdata(path)
    print("shape:", arr.shape)

    arr2, hdr = fitsy.getdata(path, header=True)
    print("OBJECT:", hdr["OBJECT"])

    obj = fitsy.getval(path, "OBJECT")
    print("getval :", obj)

    # One-shot writes (open + edit + atomic rewrite).
    fitsy.setval(path, "OBSERVER", value="Edwin Hubble", comment="discoverer")
    fitsy.setval(path, "OBJECT", value="NGC 2403")
    fitsy.delval(path, "OBSERVER")

    # Stream a new HDU onto the end without rewriting the existing file.
    # `fitsy.append` mirrors astropy's signature: it takes a raw
    # numpy array (and optional header dict), not a builder.
    fitsy.append(
        path,
        np.zeros((2, 2), dtype="f4"),
        header={"EXTNAME": "MASK"},
    )

    # `info` returns the human-readable HDU summary as a string.
    print(fitsy.info(path))
