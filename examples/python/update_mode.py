"""Edit a FITS file in place: header, pixel patches, and structural changes.

Run from the repo root::

    python examples/python/update_mode.py
"""

import os
import tempfile

import fitsy
import numpy as np

with tempfile.TemporaryDirectory() as td:
    work = os.path.join(td, "edited.fits")

    # ---- 0. Stage a fresh file --------------------------------------------
    # Synthesised so the example is self-contained.
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 1000, size=(256, 256), dtype=np.int32)
    fitsy.write(
        work,
        [fitsy.image(pixels, header={"OBJECT": "demo"})],
    )

    # ---- 1. Header edits ---------------------------------------------------
    # On clean __exit__ (or `f.flush()` / `f.close()`), changes are
    # persisted to `work` via a sibling temp file + atomic rename.
    with fitsy.open(work, mode="update") as f:
        f[0].header["OBSERVER"] = "Edwin Hubble"
        f[0].header["HISTORY"] = "added by fitsy update_mode example"

    # ---- 2. In-place pixel patches ----------------------------------------
    # `section[a:b, c:d] = arr` writes only the touched bytes via the
    # writable file via positional ``pwrite`` (O(patch), not O(file))
    # and is durable on flush.
    with fitsy.open(work, mode="update") as f:
        hdu = f[0]
        patch = np.full((50, 50), 1000, dtype=hdu.data.dtype)
        hdu.section[100:150, 200:250] = patch
        f.flush()  # explicit; __exit__ would also do this

    # ---- 3. Structural mutation -------------------------------------------
    # Append a derived extension. Structural edits flip the file's
    # dirty bit and trigger a full rewrite on flush (still atomic).
    with fitsy.open(work, mode="update") as f:
        thumb = f[0].data[::8, ::8].astype("f4")
        f.append(fitsy.image(thumb, header={"EXTNAME": "THUMB"}))

    # ---- 4. Save-as (no source mutation) ----------------------------------
    # ``writeto`` writes the in-memory state of the file to a new
    # path. The source is never touched. This works from a readonly
    # handle too -- handy when you just want to copy or convert a
    # file, or to snapshot the result of an update session.
    saved_as = os.path.join(td, "snapshot.fits")
    with fitsy.open(work) as f:
        f.writeto(saved_as)

    # ---- 5. Confirm round-trip --------------------------------------------
    with fitsy.open(work) as f:
        print("HDU count:", len(f))
        print("OBSERVER  :", f[0].header["OBSERVER"])
        print("THUMB axes:", f["THUMB"].axes)
        print("patch px  :", f[0].data[125, 225])
    with fitsy.open(saved_as) as f:
        print("snapshot HDU count:", len(f))
