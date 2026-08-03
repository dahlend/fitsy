"""Cross-check fitsy against a frozen astropy/wcslib reading of every
fixture in ``tests/data/``.

Compares, per HDU:

- header keyword values (numeric/bool/string), structural keys excluded
- image data arrays (shape, sampled pixels, whole-array aggregates)
- WCS pix->world on a fixed interior grid, where both detect a WCS
- table column data (BinTable and AsciiTable), sampled by row

The reference side comes from ``tests/data/parity_reference.json``,
written by ``tests/data/gen_parity_reference.py``, rather than from a
live astropy. That is deliberate. As a live comparison this suite was
the largest thing in the Python tests and ran nowhere except a
developer's machine that happened to have astropy installed -- CI
installs only pytest and numpy, so all of it silently skipped. Frozen,
it runs everywhere, and re-auditing against a newer astropy is a
regeneration plus a diff. The Rust suite has used the same arrangement
for ``wcs_standard.csv`` since v0.2.0.

What is lost by freezing: this no longer notices astropy *changing*.
That is what regenerating is for, and a change there deserves a human
look anyway rather than a red build.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import fitsy
import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "tests/data"
REFERENCE_PATH = DATA_DIR / "parity_reference.json"

if not REFERENCE_PATH.exists():  # pragma: no cover - misconfigured checkout
    raise AssertionError(
        f"{REFERENCE_PATH} is missing; regenerate it with "
        "`python tests/data/gen_parity_reference.py`"
    )

REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

# Sorted so parametrized ids are stable across runs.
ALL_FILES = sorted(REFERENCE["files"])


def reference_for(name: str) -> dict:
    return REFERENCE["files"][name]


def open_fitsy(name: str):
    path = DATA_DIR / name
    assert path.exists(), (
        f"{name} is in the reference but not in tests/data/; "
        "regenerate the reference after changing the corpus"
    )
    return fitsy.open(str(path), lenient=True)


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            af, bf = float(a), float(b)
            if math.isnan(af) and math.isnan(bf):
                return True
            return math.isclose(af, bf, rel_tol=1e-10, abs_tol=1e-10)
        return int(a) == int(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


# --------------------------------------------------------------------
# headers
# --------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_FILES)
def test_header_parity(name):
    """Common header keys agree (structural keys excluded)."""
    ref = reference_for(name)
    mismatches: list[str] = []
    with open_fitsy(name) as ffile:
        for i_str, per_hdu in sorted(ref["hdus"].items(), key=lambda kv: int(kv[0])):
            i = int(i_str)
            if i >= len(ffile):
                mismatches.append(f"HDU {i}: absent from fitsy ({len(ffile)} HDUs)")
                continue
            try:
                fhdr = ffile[i].header
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                mismatches.append(f"HDU {i}: fitsy header fetch failed: {e}")
                continue
            for k, expected in per_hdu["header"].items():
                # A key astropy exposes and fitsy does not is a real
                # difference, not something to step over: dropping a
                # card is exactly the regression this suite exists to
                # catch.
                if k not in fhdr:
                    mismatches.append(
                        f"HDU {i} {k!r}: absent from fitsy, reference={expected!r}"
                    )
                    continue
                try:
                    got = fhdr[k]
                except Exception as e:  # noqa: BLE001 - reported, not swallowed
                    mismatches.append(f"HDU {i} {k!r}: fitsy lookup raised {e!r}")
                    continue
                if not values_equal(got, expected):
                    mismatches.append(
                        f"HDU {i} {k!r}: fitsy={got!r} reference={expected!r}"
                    )
    if mismatches:
        shown = "\n  ".join(mismatches[:20])
        extra = (
            f"\n  ... and {len(mismatches) - 20} more" if len(mismatches) > 20 else ""
        )
        pytest.fail(f"{name}: {len(mismatches)} header mismatches:\n  {shown}{extra}")


# --------------------------------------------------------------------
# image data
# --------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_FILES)
def test_image_data_parity(name):
    """Image HDU pixels agree where the reference has an array."""
    ref = reference_for(name)
    compared = 0
    problems: list[str] = []
    with open_fitsy(name) as ffile:
        for i_str, per_hdu in sorted(ref["hdus"].items(), key=lambda kv: int(kv[0])):
            expected = per_hdu.get("image")
            if expected is None or int(i_str) >= len(ffile):
                continue
            i = int(i_str)
            try:
                fhdu = ffile[i]
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                problems.append(f"HDU {i}: fitsy could not build the HDU: {e}")
                continue
            # Not every HDU astropy hands back an array for is an
            # image to fitsy -- a tile-compressed HDU is a BINTABLE
            # here. Those are covered by the Rust compression suite.
            if not isinstance(fhdu, fitsy.ImageHdu):
                continue
            # Past this point the reference *has* pixels, so fitsy
            # failing to produce them is a failure, not a reason to
            # stop comparing. Swallowing these turned a regression
            # that broke every read into a green skip.
            try:
                fdata = fhdu.data
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                problems.append(f"HDU {i}: fitsy .data raised {e!r}")
                continue
            if fdata is None:
                problems.append(
                    f"HDU {i}: fitsy .data is None, reference has "
                    f"shape {expected['shape']}"
                )
                continue
            assert list(fdata.shape) == expected["shape"], (
                f"{name} HDU {i} shape: fitsy={list(fdata.shape)} "
                f"reference={expected['shape']}"
            )
            flat = np.asarray(fdata, dtype=np.float64).ravel()
            idx = np.array(expected["index"], dtype=np.intp)
            got = flat[idx]
            want = np.array(expected["value"], dtype=np.float64)
            mask = np.isfinite(got) & np.isfinite(want)
            if mask.any():
                # Generous: BSCALE/BZERO promotion and float32 storage
                # both cost precision the comparison should not police.
                np.testing.assert_allclose(
                    got[mask],
                    want[mask],
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"{name} HDU {i} sampled pixels",
                )
            # Aggregates over the whole array, so a regression that
            # misses every sampled index still shows up.
            finite = flat[np.isfinite(flat)]
            assert finite.size == expected["n_finite"], (
                f"{name} HDU {i}: {finite.size} finite pixels, "
                f"reference has {expected['n_finite']}"
            )
            if finite.size:
                for stat, want_stat in (
                    ("min", expected["min"]),
                    ("max", expected["max"]),
                    ("sum", expected["sum"]),
                ):
                    got_stat = float(getattr(finite, stat)())
                    scale = max(abs(want_stat), 1.0)
                    assert abs(got_stat - want_stat) <= 1e-6 * scale, (
                        f"{name} HDU {i} {stat}: fitsy={got_stat!r} "
                        f"reference={want_stat!r}"
                    )
            compared += 1
    if problems:
        pytest.fail(
            f"{name}: {len(problems)} image issues:\n  " + "\n  ".join(problems)
        )
    if compared == 0:
        pytest.skip("no comparable image HDUs")


# --------------------------------------------------------------------
# WCS
# --------------------------------------------------------------------


def great_circle_arcsec(a, b):
    ra1, dec1 = np.deg2rad(a[:, 0]), np.deg2rad(a[:, 1])
    ra2, dec2 = np.deg2rad(b[:, 0]), np.deg2rad(b[:, 1])
    sep = 2.0 * np.arcsin(
        np.sqrt(
            np.clip(
                np.sin((dec1 - dec2) / 2) ** 2
                + np.cos(dec1) * np.cos(dec2) * np.sin((ra1 - ra2) / 2) ** 2,
                0.0,
                1.0,
            )
        )
    )
    return np.rad2deg(sep) * 3600.0


@pytest.mark.parametrize("name", ALL_FILES)
def test_wcs_parity(name):
    """Celestial WCS pix->world agrees with the reference."""
    if name == "dss_plate.fits":
        pytest.xfail(
            "DSS plate solution is fitsy-specific; astropy uses placeholder TAN"
        )
    ref = reference_for(name)
    compared = 0
    problems: list[str] = []
    worst_seen = 0.0
    with open_fitsy(name) as ffile:
        for i_str, per_hdu in sorted(ref["hdus"].items(), key=lambda kv: int(kv[0])):
            expected = per_hdu.get("wcs")
            if expected is None or int(i_str) >= len(ffile):
                continue
            i = int(i_str)
            try:
                fhdu = ffile[i]
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                problems.append(f"HDU {i}: fitsy could not build the HDU: {e}")
                continue
            if not isinstance(fhdu, fitsy.ImageHdu):
                continue
            # The reference only holds a `wcs` entry where astropy
            # found a celestial WCS, so fitsy finding none -- or
            # raising on the way -- is a real divergence. This used to
            # fall through to `continue`, which meant a parser that
            # stopped recognizing celestial axes skipped the suite
            # green.
            try:
                fwcs = fhdu.wcs()
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                problems.append(f"HDU {i}: fitsy .wcs() raised {e!r}")
                continue
            if fwcs is None:
                problems.append(
                    f"HDU {i}: fitsy found no WCS; reference has "
                    f"CTYPE {expected['ctype']}"
                )
                continue
            if not fwcs.is_celestial:
                problems.append(
                    f"HDU {i}: fitsy WCS is not celestial; reference has "
                    f"CTYPE {expected['ctype']}"
                )
                continue
            # astropy silently rewrites CTYPE='RA---TPV' to 'RA---TAN'
            # and discards the PV distortion terms, so the reference
            # holds the linear-only sky position. fitsy *does* apply
            # TPV, so the two legitimately diverge by the distortion
            # magnitude. `tests/wcs_integration.rs::tpv_matches_reference`
            # covers TPV properly.
            f_ctype = [str(fhdu.header.get(f"CTYPE{n}", "")) for n in (1, 2)]
            if any("TPV" in c for c in f_ctype) and not any(
                "TPV" in c for c in expected["ctype"]
            ):
                continue
            pix = np.array(expected["pix"], dtype=np.float64)
            want = np.array(expected["sky"], dtype=np.float64)
            try:
                got = fwcs.pixel_to_world(pix, origin=0)
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                problems.append(f"HDU {i}: pixel_to_world failed: {e}")
                continue
            sep = great_circle_arcsec(np.asarray(got, dtype=np.float64), want)
            finite = sep[np.isfinite(sep)]
            if finite.size == 0:
                continue
            worst = float(finite.max())
            worst_seen = max(worst_seen, worst)
            if worst > 1.0:
                problems.append(
                    f'HDU {i}: max separation {worst:.4g} arcsec exceeds 1"'
                )
            compared += 1
    if compared == 0:
        pytest.skip("no comparable celestial WCS HDUs")
    if problems:
        pytest.fail(
            f'{name}: {len(problems)} WCS issues (worst={worst_seen:.4g}"):\n  '
            + "\n  ".join(problems[:20])
        )


# --------------------------------------------------------------------
# tables
# --------------------------------------------------------------------


def cell_matches(got, want) -> bool:
    """One table cell against its reference value."""
    if isinstance(want, dict) and "__complex__" in want:
        re_, im_ = want["__complex__"]
        return complex(got) == pytest.approx(complex(re_, im_), rel=1e-9, abs=1e-9)
    if isinstance(want, list):
        arr = np.asarray(got).ravel()
        if arr.size != len(want):
            return False
        if all(isinstance(w, str) for w in want):
            return [str(x).strip() for x in arr] == [w.strip() for w in want]
        return np.allclose(
            np.asarray(arr, dtype=float),
            np.asarray(want, dtype=float),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
    if isinstance(want, str):
        return str(got).strip() == want.strip()
    if isinstance(want, bool):
        return bool(got) == want
    if want is None:
        return got is None
    gf = np.asarray(got, dtype=float).ravel()
    if gf.size != 1:
        return False
    return values_equal(float(gf[0]), float(want)) or math.isclose(
        float(gf[0]), float(want), rel_tol=1e-6, abs_tol=1e-6
    )


@pytest.mark.parametrize("name", ALL_FILES)
def test_table_parity(name):
    """BinTable and AsciiTable column data agrees with the reference."""
    ref = reference_for(name)
    compared = 0
    problems: list[str] = []
    with open_fitsy(name) as ffile:
        for i_str, per_hdu in sorted(ref["hdus"].items(), key=lambda kv: int(kv[0])):
            expected = per_hdu.get("table")
            if expected is None or int(i_str) >= len(ffile):
                continue
            i = int(i_str)
            try:
                fhdu = ffile[i]
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                problems.append(f"HDU {i}: fitsy could not build the HDU: {e}")
                continue
            # A tile-compressed image is a BINTABLE to astropy too, so
            # the reference can hold a `table` entry for an HDU fitsy
            # unwraps into an image. Not a divergence.
            if not isinstance(fhdu, (fitsy.BinTable, fitsy.AsciiTable)):
                continue
            want_rows = expected["n_rows"]
            if fhdu.n_rows != want_rows:
                problems.append(
                    f"HDU {i}: n_rows fitsy={fhdu.n_rows} reference={want_rows}"
                )
                continue
            for col_name, col_ref in expected["columns"].items():
                if col_name not in fhdu.column_names:
                    problems.append(
                        f"HDU {i}: column {col_name!r} missing from fitsy "
                        f"(has {sorted(fhdu.column_names)})"
                    )
                    continue
                try:
                    fcol = fhdu.column(col_name)
                except Exception as e:  # noqa: BLE001 - reported, not swallowed
                    problems.append(f"HDU {i} col {col_name!r}: fitsy read failed: {e}")
                    continue
                # Nullable integer columns come back masked where TNULL
                # marks a hole; the reference stored astropy's fill, so
                # masked cells are excused rather than compared.
                mask = None
                if hasattr(fcol, "mask") and hasattr(fcol, "filled"):
                    mask = np.asarray(fcol.mask, dtype=bool)
                    fcol = np.asarray(fcol.filled(0))
                for row, want in zip(col_ref["row"], col_ref["value"]):
                    if mask is not None and row < mask.size and np.any(mask[row]):
                        continue
                    got = fcol[row]
                    if not cell_matches(got, want):
                        problems.append(
                            f"HDU {i} col {col_name!r} row {row}: "
                            f"fitsy={got!r} reference={want!r}"
                        )
            compared += 1
    if compared == 0:
        pytest.skip("no comparable table HDUs")
    if problems:
        shown = "\n  ".join(problems[:20])
        extra = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        pytest.fail(f"{name}: {len(problems)} table mismatches:\n  {shown}{extra}")


# --------------------------------------------------------------------
# the reference itself
# --------------------------------------------------------------------


def test_reference_covers_the_corpus():
    """Every fixture must be in the reference.

    Without this, adding a fixture and forgetting to regenerate would
    silently leave it uncompared -- the exact failure mode that made
    the live-astropy version of this suite worthless in CI.
    """
    on_disk = {
        p.name
        for p in DATA_DIR.iterdir()
        if p.suffix == ".fits" or p.name.endswith((".fits.fz", ".fits.gz"))
    }
    missing = sorted(on_disk - set(ALL_FILES))
    assert not missing, (
        f"{len(missing)} fixture(s) absent from the reference: {missing}\n"
        "Regenerate with `python tests/data/gen_parity_reference.py`."
    )
