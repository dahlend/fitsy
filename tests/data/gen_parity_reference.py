"""Snapshot astropy's reading of every fixture into a reference file.

``tests/python/test_reference_parity.py`` compares fitsy against the
output of this script instead of against a live astropy, so the
cross-check runs everywhere rather than only where astropy happens to
be installed. This is the same arrangement the Rust suite already uses
for ``wcs_standard.csv`` and friends (see ``gen_wcs_test_data.py``).

Run it when the fixture corpus changes, or to re-audit against a newer
astropy::

    pip install astropy numpy
    python tests/data/gen_parity_reference.py

It rewrites ``tests/data/parity_reference.json``. Review the diff:
a change there is astropy changing its mind or a fixture changing, and
either way it is worth a look before committing.

Only *sampled* values are stored, not whole arrays -- enough to catch a
regression, small enough to read a diff of. The sampling is
deterministic, so a rerun with the same inputs produces the same file.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import astropy
    from astropy.io import fits as afits
    from astropy.wcs import WCS as AWCS
    from astropy.wcs import _wcs
except ImportError:  # pragma: no cover - developer tooling
    sys.exit("this generator needs astropy: pip install astropy")

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "parity_reference.json"

# Cap on stored samples per array/column. Large enough that a real
# decode bug shows up, small enough that the reference file stays
# reviewable.
MAX_SAMPLES = 64

# Header keywords whose value is structural / format-dependent and may
# legitimately differ between libraries (e.g. BITPIX after scaling, or
# a recomputed checksum).
STRUCTURAL_KEYS = {
    "SIMPLE", "BITPIX", "NAXIS", "EXTEND", "PCOUNT", "GCOUNT", "XTENSION",
    "BSCALE", "BZERO", "BLANK", "DATASUM", "CHECKSUM", "ZIMAGE", "ZBITPIX",
    "ZNAXIS", "ZTILE1", "ZTILE2", "ZCMPTYPE", "ZNAME1", "ZVAL1", "ZNAME2",
    "ZVAL2", "ZQUANTIZ", "ZDITHER0", "ZHECKSUM", "ZDATASUM", "ZBLANK",
    "ZSCALE", "ZZERO", "TFIELDS", "EXTNAME", "EXTVER", "INHERIT", "GROUPS",
    "THEAP",
}  # fmt: skip
STRUCTURAL_PREFIXES = (
    "NAXIS", "TFORM", "TTYPE", "TUNIT", "TDISP", "TNULL", "TSCAL", "TZERO",
    "TDIM", "TBCOL", "TLMIN", "TLMAX", "ZNAXIS", "ZTILE", "ZNAME", "ZVAL",
    "ZFORM", "ZCTYP",
)  # fmt: skip


def is_structural(key: str) -> bool:
    k = key.upper().strip()
    if not k or k in {"COMMENT", "HISTORY", ""}:
        return True
    if k in STRUCTURAL_KEYS:
        return True
    return any(k.startswith(p) and k[len(p) :].isdigit() for p in STRUCTURAL_PREFIXES)


def sample_indices(n: int) -> list[int]:
    """Up to `MAX_SAMPLES` evenly spread indices into `0..n`.

    Endpoints included: off-by-one decode bugs live at the edges.
    """
    if n <= 0:
        return []
    if n <= MAX_SAMPLES:
        return list(range(n))
    return sorted(
        {int(round(i * (n - 1) / (MAX_SAMPLES - 1))) for i in range(MAX_SAMPLES)}
    )


def jsonable(v):
    """Coerce a numpy/FITS value into something `json` can hold."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        # json writes NaN/Infinity as bare tokens; Python reads them
        # back, and Python is the only consumer here.
        return f
    if isinstance(v, (complex, np.complexfloating)):
        return {"__complex__": [float(v.real), float(v.imag)]}
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("latin-1")
    if isinstance(v, (str, np.str_)):
        return str(v)
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.ravel().tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return str(v)


def snapshot_header(hdr) -> dict:
    out = {}
    for k in hdr.keys():
        if not k or is_structural(k):
            continue
        try:
            out[k] = jsonable(hdr[k])
        except Exception:
            continue
    return out


def snapshot_image(hdu) -> dict | None:
    try:
        data = hdu.data
    except Exception:
        return None
    if data is None:
        return None
    flat = np.asarray(data, dtype=np.float64).ravel()
    idx = sample_indices(flat.size)
    finite = flat[np.isfinite(flat)]
    return {
        "shape": list(np.asarray(data).shape),
        "index": idx,
        "value": [float(flat[i]) for i in idx],
        # Whole-array aggregates catch a regression that misses every
        # sampled index, e.g. a handful of corrupted tiles.
        "n_finite": int(finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "sum": float(finite.sum()) if finite.size else None,
    }


def snapshot_wcs(hdu) -> dict | None:
    """astropy's sky positions on a fixed 5x5 interior grid."""
    try:
        awcs = AWCS(hdu.header)
    except Exception:
        return None
    if not awcs.has_celestial:
        return None
    shape = np.asarray(hdu.data).shape if hdu.data is not None else None
    if shape is None or len(shape) < 2:
        return None
    # fitsy reports axes fastest-first; numpy shape is slowest-first.
    nx, ny = shape[-1], shape[-2]
    xs = np.linspace(max(1, nx * 0.1), max(2, nx * 0.9), 5)
    ys = np.linspace(max(1, ny * 0.1), max(2, ny * 0.9), 5)
    pix = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    try:
        lon_axis, lat_axis = awcs.wcs.lng, awcs.wcs.lat
        crpix = awcs.wcs.crpix  # 1-based
        full = np.tile(crpix - 1.0, (pix.shape[0], 1))  # 0-based
        full[:, lon_axis] = pix[:, 0]
        full[:, lat_axis] = pix[:, 1]
        # `all_pix2world`, not `wcs_pix2world`: the latter is the
        # linear WCS only and skips SIP / lookup-table distortion.
        world = awcs.all_pix2world(full, 0)
    except Exception:
        return None
    sky = np.column_stack([world[:, lon_axis], world[:, lat_axis]])
    if not np.isfinite(sky).any():
        return None
    return {
        "ctype": [str(c) for c in awcs.wcs.ctype],
        "pix": pix.tolist(),
        "sky": [[float(a), float(b)] for a, b in sky],
    }


def snapshot_table(hdu) -> dict | None:
    if not hasattr(hdu, "columns"):
        return None
    try:
        data = hdu.data
    except Exception:
        return None
    if data is None:
        return None
    n_rows = len(data)
    rows = sample_indices(n_rows)
    columns = {}
    for col in hdu.columns:
        name = col.name
        try:
            acol = data[name]
        except Exception:
            continue
        if hasattr(acol, "filled"):
            acol = (
                acol.filled(np.nan)
                if np.issubdtype(acol.dtype, np.floating)
                else acol.filled(0)
            )
        acol = np.asarray(acol)
        try:
            cells = [jsonable(acol[r]) for r in rows]
        except Exception:
            continue
        columns[name] = {"row": rows, "value": cells}
    if not columns:
        return None
    return {"n_rows": n_rows, "columns": columns}


def main() -> None:
    paths = sorted(
        p
        for p in DATA_DIR.iterdir()
        if p.suffix == ".fits" or p.name.endswith((".fits.fz", ".fits.gz"))
    )
    if not paths:
        sys.exit(f"no fixtures found under {DATA_DIR}")

    files: dict[str, dict] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in paths:
            try:
                hdus = afits.open(path)
            except Exception as e:
                print(f"  !! astropy cannot open {path.name}: {e}", file=sys.stderr)
                continue
            entry: dict = {"n_hdus": len(hdus), "hdus": {}}
            for i, hdu in enumerate(hdus):
                per_hdu = {"header": snapshot_header(hdu.header)}
                for key, fn in (
                    ("image", snapshot_image),
                    ("wcs", snapshot_wcs),
                    ("table", snapshot_table),
                ):
                    try:
                        got = fn(hdu)
                    except Exception:
                        got = None
                    if got is not None:
                        per_hdu[key] = got
                entry["hdus"][str(i)] = per_hdu
            hdus.close()
            files[path.name] = entry
            print(f"  {path.name}: {len(entry['hdus'])} HDU(s)")

    payload = {
        "generated_with": {
            "astropy": astropy.__version__,
            "wcslib": _wcs.WCSLIB_VERSION,
            "numpy": np.__version__,
        },
        "max_samples": MAX_SAMPLES,
        "files": files,
    }
    # Plain JSON, sorted and one value per line: the point of
    # checking this in is that a regeneration produces a diff someone
    # can read. A compressed blob would not.
    text = json.dumps(payload, indent=1, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
