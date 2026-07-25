"""Temporary cross-check: fitsy vs astropy on every FITS file in data/.

Skipped when astropy isn't installed. Compares, per HDU:
- Header keyword values (numeric/bool/string), excluding structural keys.
- Image data arrays (shape + values).
- WCS pix->world on a small sample grid (when both libraries detect a WCS).
- Table HDU column data (BinTable and AsciiTable).

This file is a one-off audit; delete once the audit is done.
"""

from __future__ import annotations

import math
from pathlib import Path

import fitsy
import numpy as np
import pytest

astropy = pytest.importorskip("astropy")
from astropy.io import fits as afits  # noqa: E402
from astropy.wcs import WCS as AWCS  # noqa: E402
from astropy.wcs import FITSFixedWarning  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "tests/data"

ALL_FITS = sorted(
    p
    for p in DATA_DIR.iterdir()
    if p.suffix in {".fits"} or p.name.endswith((".fits.fz", ".fits.gz"))
)

# Header keywords whose value is structural / format-dependent and may
# legitimately differ between libraries (e.g., BITPIX after scaling,
# checksum recomputation, etc.).
STRUCTURAL_KEYS = {
    "SIMPLE",
    "BITPIX",
    "NAXIS",
    "EXTEND",
    "PCOUNT",
    "GCOUNT",
    "XTENSION",
    "BSCALE",
    "BZERO",
    "BLANK",
    "DATASUM",
    "CHECKSUM",
    "ZIMAGE",
    "ZBITPIX",
    "ZNAXIS",
    "ZTILE1",
    "ZTILE2",
    "ZCMPTYPE",
    "ZNAME1",
    "ZVAL1",
    "ZNAME2",
    "ZVAL2",
    "ZQUANTIZ",
    "ZDITHER0",
    "ZHECKSUM",
    "ZDATASUM",
    "ZBLANK",
    "ZSCALE",
    "ZZERO",
    "TFIELDS",
    "EXTNAME",
    "EXTVER",
    "INHERIT",
    "GROUPS",
    "THEAP",
}
# NAXISn, TFORMn, TTYPEn etc. handled via prefix below.
STRUCTURAL_PREFIXES = (
    "NAXIS",
    "TFORM",
    "TTYPE",
    "TUNIT",
    "TDISP",
    "TNULL",
    "TSCAL",
    "TZERO",
    "TDIM",
    "TBCOL",
    "TLMIN",
    "TLMAX",
    "ZNAXIS",
    "ZTILE",
    "ZNAME",
    "ZVAL",
    "ZFORM",
    "ZCTYP",
)


def _is_structural(key: str) -> bool:
    k = key.upper().strip()
    if not k or k in {"COMMENT", "HISTORY", ""}:
        return True
    if k in STRUCTURAL_KEYS:
        return True
    return any(k.startswith(p) and k[len(p) :].isdigit() for p in STRUCTURAL_PREFIXES)


def _values_equal(a, b) -> bool:
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


@pytest.mark.parametrize("path", ALL_FITS, ids=lambda p: p.name)
def test_header_parity(path):
    """Common header keys agree (structural keys excluded)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ahdus = afits.open(path)
        except Exception as e:
            raise AssertionError(f"astropy failed to open {path.name}: {e}") from e
        try:
            ffile = fitsy.open(str(path), lenient=True)
        except Exception as e:
            raise AssertionError(f"fitsy failed to open {path.name}: {e}") from e

        n = min(len(ahdus), len(ffile))
        mismatches: list[str] = []
        for i in range(n):
            try:
                fhdr = ffile[i].header
            except Exception as e:
                mismatches.append(f"HDU {i}: fitsy header fetch failed: {e}")
                continue
            ahdr = ahdus[i].header
            common = set(fhdr.keys()) & set(ahdr.keys())
            for k in common:
                if _is_structural(k):
                    continue
                try:
                    fv = fhdr[k]
                except Exception:
                    continue
                av = ahdr[k]
                if not _values_equal(fv, av):
                    mismatches.append(f"HDU {i} {k!r}: fitsy={fv!r} astropy={av!r}")
        ahdus.close()
        if mismatches:
            shown = "\n  ".join(mismatches[:20])
            extra = (
                f"\n  ... and {len(mismatches) - 20} more"
                if len(mismatches) > 20
                else ""
            )
            pytest.fail(
                f"{path.name}: {len(mismatches)} header mismatches:\n  {shown}{extra}"
            )


@pytest.mark.parametrize("path", ALL_FITS, ids=lambda p: p.name)
def test_image_data_parity(path):
    """Image HDU data arrays agree where both libraries return one."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ahdus = afits.open(path)
        except Exception as e:
            raise AssertionError(f"astropy failed to open {path.name}: {e}") from e
        try:
            ffile = fitsy.open(str(path), lenient=True)
        except Exception as e:
            raise AssertionError(f"fitsy failed to open {path.name}: {e}") from e

        n = min(len(ahdus), len(ffile))
        compared = 0
        for i in range(n):
            ahdu = ahdus[i]
            try:
                fhdu = ffile[i]
            except Exception:
                continue
            if not isinstance(fhdu, fitsy.ImageHdu):
                continue
            try:
                fdata = fhdu.data
            except Exception:
                continue
            if fdata is None:
                continue
            try:
                adata = ahdu.data
            except Exception:
                continue
            if adata is None:
                continue
            assert (
                fdata.shape == adata.shape
            ), f"HDU {i} shape mismatch: fitsy={fdata.shape} astropy={adata.shape}"
            # Compare with generous tolerance (BSCALE/BZERO float promotion).
            af = np.asarray(adata, dtype=np.float64)
            ff = np.asarray(fdata, dtype=np.float64)
            mask = np.isfinite(af) & np.isfinite(ff)
            if mask.any():
                np.testing.assert_allclose(
                    ff[mask],
                    af[mask],
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"{path.name} HDU {i} pixel mismatch",
                )
            compared += 1
        ahdus.close()
        if compared == 0:
            pytest.skip("no comparable image HDUs")


@pytest.mark.parametrize("path", ALL_FITS, ids=lambda p: p.name)
def test_wcs_parity(path):
    """Celestial WCS pix->world agrees across the libraries."""
    # DSS plate-solution files carry both a non-standard plate model
    # (AMDX*/AMDY* coefficients) and a placeholder RA---TAN WCS with
    # dummy CRVAL values. fitsy uses the accurate plate solution;
    # astropy.wcs only knows the placeholder. Comparing the two is
    # not meaningful, so skip files where the AMD plate model is
    # present.
    if path.name == "dss_plate.fits":
        pytest.xfail(
            "DSS plate solution is fitsy-specific; astropy uses placeholder TAN"
        )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        warnings.simplefilter("ignore")
        try:
            ahdus = afits.open(path)
        except Exception as e:
            raise AssertionError(f"astropy failed to open {path.name}: {e}") from e
        try:
            ffile = fitsy.open(str(path), lenient=True)
        except Exception as e:
            raise AssertionError(f"fitsy failed to open {path.name}: {e}") from e

        n = min(len(ahdus), len(ffile))
        compared = 0
        max_arcsec_seen = 0.0
        problems: list[str] = []
        for i in range(n):
            try:
                fhdu = ffile[i]
            except Exception:
                continue
            if not isinstance(fhdu, fitsy.ImageHdu):
                continue
            try:
                fwcs = fhdu.wcs()
            except Exception:
                fwcs = None
            if fwcs is None or not fwcs.is_celestial:
                continue
            try:
                awcs = AWCS(ahdus[i].header)
            except Exception:
                continue
            if not awcs.has_celestial:
                continue
            # astropy 7.x silently rewrites CTYPE='RA---TPV' to 'RA---TAN'
            # and discards PV distortion terms, so all_pix2world returns
            # the linear-only sky position. Skip the comparison when that
            # happens -- fitsy DOES apply TPV, so the answers diverge by
            # the distortion magnitude.
            try:
                a_ctype = list(awcs.wcs.ctype)
                f_ctype = [
                    str(fhdu.header.get("CTYPE1", "")),
                    str(fhdu.header.get("CTYPE2", "")),
                ]
                if any("TPV" in c for c in f_ctype) and not any(
                    "TPV" in c for c in a_ctype
                ):
                    continue
            except Exception:
                pass
            axes = fhdu.axes
            if len(axes) < 2:
                continue
            nx, ny = axes[0], axes[1]
            # 5x5 grid in image interior.
            xs = np.linspace(max(1, nx * 0.1), max(2, nx * 0.9), 5)
            ys = np.linspace(max(1, ny * 0.1), max(2, ny * 0.9), 5)
            pix = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
            try:
                fsky = fwcs.pixel_to_celestial_many(pix, origin=0)
            except Exception as e:
                problems.append(f"HDU {i}: fitsy pixel_to_celestial_many failed: {e}")
                continue
            try:
                # Use all_pix2world so SIP / distortion lookup tables are
                # applied (wcs_pix2world is the *linear* WCS only).
                # We project the celestial axis pair through a full
                # NAXIS-D pixel vector centered on CRPIX so non-celestial
                # axes don't poison the result.
                lon_axis, lat_axis = awcs.wcs.lng, awcs.wcs.lat
                crpix = awcs.wcs.crpix  # 1-based
                full = np.tile(crpix - 1.0, (pix.shape[0], 1))  # 0-based
                full[:, lon_axis] = pix[:, 0]
                full[:, lat_axis] = pix[:, 1]
                world = awcs.all_pix2world(full, 0)
                asky = np.column_stack([world[:, lon_axis], world[:, lat_axis]])
            except Exception as e:
                problems.append(f"HDU {i}: astropy all_pix2world failed: {e}")
                continue

            # Compare on the sphere: great-circle separation in arcsec.
            ra1 = np.deg2rad(fsky[:, 0])
            dec1 = np.deg2rad(fsky[:, 1])
            ra2 = np.deg2rad(asky[:, 0])
            dec2 = np.deg2rad(asky[:, 1])
            dra = ra1 - ra2
            sep = 2.0 * np.arcsin(
                np.sqrt(
                    np.clip(
                        np.sin((dec1 - dec2) / 2) ** 2
                        + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2) ** 2,
                        0.0,
                        1.0,
                    )
                )
            )
            sep_arcsec = np.rad2deg(sep) * 3600.0
            worst = float(np.nanmax(sep_arcsec))
            max_arcsec_seen = max(max_arcsec_seen, worst)
            if worst > 1.0:  # 1 arcsec tolerance
                problems.append(
                    f"HDU {i}: max separation {worst:.4g} arcsec exceeds tolerance"
                )
            compared += 1

        ahdus.close()
        if compared == 0:
            pytest.skip("no comparable celestial WCS HDUs")
        if problems:
            pytest.fail(
                f"{path.name}: {len(problems)} WCS issues"
                f' (worst={max_arcsec_seen:.4g}"):\n  ' + "\n  ".join(problems[:20])
            )


@pytest.mark.parametrize("path", ALL_FITS, ids=lambda p: p.name)
def test_table_parity(path):
    """BinTable and AsciiTable column data agrees between fitsy and astropy."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ahdus = afits.open(path)
        except Exception as e:
            raise AssertionError(f"astropy failed to open {path.name}: {e}") from e
        try:
            ffile = fitsy.open(str(path), lenient=True)
        except Exception as e:
            raise AssertionError(f"fitsy failed to open {path.name}: {e}") from e

        n = min(len(ahdus), len(ffile))
        compared = 0
        problems: list[str] = []

        for i in range(n):
            try:
                fhdu = ffile[i]
            except Exception:
                continue
            if not isinstance(fhdu, (fitsy.BinTable, fitsy.AsciiTable)):
                continue
            ahdu = ahdus[i]
            if not hasattr(ahdu, "columns") or ahdu.data is None:
                continue

            # Row count
            if fhdu.n_rows != len(ahdu.data):
                problems.append(
                    f"HDU {i}: n_rows fitsy={fhdu.n_rows} astropy={len(ahdu.data)}"
                )
                continue

            for col_name in fhdu.column_names:
                try:
                    fcol = fhdu.column(col_name)
                except Exception as e:
                    problems.append(f"HDU {i} col {col_name!r}: fitsy read failed: {e}")
                    continue
                try:
                    acol = ahdu.data[col_name]
                except Exception as e:
                    problems.append(
                        f"HDU {i} col {col_name!r}: astropy read failed: {e}"
                    )
                    continue

                # Flatten astropy masked arrays to plain ndarray
                if hasattr(acol, "filled"):
                    acol_data = np.asarray(
                        acol.filled(np.nan)
                        if np.issubdtype(acol.dtype, np.floating)
                        else acol.filled(0)
                    )
                else:
                    acol_data = np.asarray(acol)

                # Same flattening for fitsy: MaskedArrays come back from
                # nullable integer columns with TNULL holes. Astropy may
                # return the raw sentinel (no mask), so we treat masked
                # cells as equal to whatever the unmasked counterpart is.
                if hasattr(fcol, "filled") and hasattr(fcol, "mask"):
                    fmask = np.asarray(fcol.mask, dtype=bool)
                    if fmask.any():
                        # Substitute astropy's value at masked positions so
                        # downstream comparisons treat sentinels as a match.
                        fdata = np.array(fcol.filled(0))
                        if (
                            isinstance(acol_data, np.ndarray)
                            and acol_data.shape == fdata.shape
                        ):
                            fdata[fmask] = acol_data[fmask]
                        fcol = fdata
                    else:
                        fcol = np.asarray(fcol.filled(0))

                # Lists (strings, bools, variable-length, generic) -- element-wise
                if isinstance(fcol, list):
                    if len(fcol) != len(acol_data):
                        problems.append(
                            f"HDU {i} col {col_name!r}: "
                            f"len fitsy={len(fcol)} astropy={len(acol_data)}"
                        )
                        continue
                    # Cells may themselves be arrays (vector columns). Determine
                    # whether we're doing string or numeric comparison.
                    first_cell = fcol[0] if fcol else None
                    is_numeric_cell = isinstance(
                        first_cell, np.ndarray
                    ) and np.issubdtype(first_cell.dtype, np.number)
                    for r, (fv, av) in enumerate(zip(fcol, acol_data)):
                        if is_numeric_cell:
                            fvf = np.asarray(fv, dtype=float).ravel()
                            avf = np.asarray(av, dtype=float).ravel()
                            valid = np.isfinite(fvf) & np.isfinite(avf)
                            if valid.any() and not np.allclose(
                                fvf[valid], avf[valid], rtol=1e-5, atol=1e-5
                            ):
                                problems.append(
                                    f"HDU {i} col {col_name!r} row {r}:"
                                    " numeric vector mismatch"
                                )
                                break
                        else:
                            fvs = (
                                bool(fv)
                                if isinstance(fv, (bool, np.bool_))
                                else str(fv).strip()
                            )
                            avs = (
                                bool(av)
                                if isinstance(av, (bool, np.bool_))
                                else str(av).strip()
                            )
                            if fvs != avs:
                                problems.append(
                                    f"HDU {i} col {col_name!r} row {r}:"
                                    f" fitsy={fv!r} astropy={av!r}"
                                )
                                break
                    compared += 1
                    continue

                fcol_arr = np.asarray(fcol)
                if fcol_arr.shape != acol_data.shape:
                    problems.append(
                        f"HDU {i} col {col_name!r}: "
                        f"shape fitsy={fcol_arr.shape} astropy={acol_data.shape}"
                    )
                    continue

                if fcol_arr.dtype.kind in ("U", "S", "O"):
                    for r, (fv, av) in enumerate(zip(fcol_arr.flat, acol_data.flat)):
                        if str(fv).strip() != str(av).strip():
                            problems.append(
                                f"HDU {i} col {col_name!r} row {r}:"
                                f" fitsy={fv!r} astropy={av!r}"
                            )
                            break
                elif fcol_arr.dtype.kind == "b":
                    if not np.array_equal(fcol_arr, acol_data):
                        problems.append(
                            f"HDU {i} col {col_name!r}: bool column mismatch"
                        )
                else:
                    valid = np.isfinite(fcol_arr.astype(float)) & np.isfinite(
                        acol_data.astype(float)
                    )
                    if valid.any():
                        try:
                            np.testing.assert_allclose(
                                fcol_arr[valid],
                                acol_data[valid],
                                rtol=1e-5,
                                atol=1e-5,
                                err_msg=f"{path.name} HDU {i} col {col_name!r}",
                            )
                        except AssertionError as exc:
                            problems.append(
                                f"HDU {i} col {col_name!r}: {str(exc)[:300]}"
                            )
                compared += 1

        ahdus.close()
        if compared == 0:
            pytest.skip("no comparable table HDUs")
        if problems:
            shown = "\n  ".join(problems[:20])
            extra = (
                f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
            )
            pytest.fail(f"{path.name}: table mismatches:\n  {shown}{extra}")
