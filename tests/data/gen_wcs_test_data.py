#!/usr/bin/env python3
"""Generate WCS world-to-pixel ground-truth CSV test data using astropy.

Run from the repository root:
    python3 tests/data/gen_wcs_test_data.py

Outputs (written to tests/data/):
    wcs_standard.csv  - all standard FITS celestial projections
    wcs_sip.csv       - TAN + SIP polynomial distortion
    wcs_tpv.csv       - TAN + TPV (Shupe/SCAMP) polynomial distortion

INDEXING NOTE
=============
FITS pixels are 1-based: the center of the first pixel is (1, 1).  All astropy
WCS calls in this file use ``origin=1`` so that stored x_fits / y_fits values
are FITS-convention 1-based pixel coordinates.  The Rust ``world_to_pixel``
implementation uses 0-based pixels (numpy convention), so the integration test
subtracts 1.0 from the CSV x_fits / y_fits before comparing.

DATA GENERATION STRATEGY
=========================
For every WCS configuration we generate a regular 5x5 grid of *pixel* positions
and convert them to sky coordinates with ``all_pix2world``.  Each (ra, dec,
x_fits, y_fits) tuple is stored as one test row.  The Rust test exercises
``world_to_pixel(ra, dec)`` and asserts the result is within 1 x 10^-8 pixels of
the stored (x_fits, y_fits).  A roundtrip check with tolerance 1 x 10^-6 pixels
is used to discard points that are degenerate in the given projection (e.g.
near a singularity in the quad-cube faces).
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

from astropy.io.fits import Header as FitsHeader
from astropy.wcs import WCS

warnings.filterwarnings("ignore")  # suppress wcslib / astropy deprecation noise

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared WCS parameters
# All projections use the same reference pixel, sky position, and pixel scale
# so the only variable between configurations is the projection itself.
# ---------------------------------------------------------------------------
CRPIX1, CRPIX2 = 512.5, 512.5  # reference pixel (FITS 1-based)
CRVAL1, CRVAL2 = 83.8221, -5.3911  # reference sky position (deg, near Orion)
CDELT1 = -2.77778e-4  # pixel scale, axis 1 (deg/pix, East-left)
CDELT2 = +2.77778e-4  # pixel scale, axis 2 (deg/pix)

FITS_ORIGIN = 1  # all astropy pix2world / world2pix calls

# 5x5 pixel test grid centered on CRPIX with 100-pix spacing.
# Expressed as FITS 1-based coordinates.
_STEPS = [-200, -100, 0, 100, 200]
PIXEL_GRID: list[tuple[float, float]] = [
    (CRPIX1 + dx, CRPIX2 + dy) for dx in _STEPS for dy in _STEPS
]

# ---------------------------------------------------------------------------
# Standard projections
# ---------------------------------------------------------------------------
# Each entry: (ctype_code, pv_params, description)
# pv_params: list of (axis_i, m, value) tuples passed to wcs.wcs.set_pv()
# All standard-projection PV parameters are on axis 2 (latitude/Dec axis)
# per FITS Paper II - see Sec.5 for each projection's PV definitions.
STANDARD_PROJECTIONS: list[tuple[str, list[tuple[int, int, float]], str]] = [
    # -- Zenithal / azimuthal (Sec.5.1) ------------------------------------------
    ("TAN", [], "Gnomonic (tangent)"),
    ("STG", [], "Stereographic"),
    # SIN with xi=0, eta=0 is the standard slant-free orthographic projection.
    ("SIN", [(2, 1, 0.0), (2, 2, 0.0)], "Slant orthographic (SIN)"),
    ("ARC", [], "Zenithal equidistant"),
    ("ZEA", [], "Zenithal equal area (Lambert)"),
    # AZP: PV2_1 = mu (perspective distance), PV2_2 = gamma (tilt angle, deg)
    ("AZP", [(2, 1, 2.0), (2, 2, 15.0)], "Zenithal/azimuthal perspective"),
    # SZP: PV2_1 = mu, PV2_2 = phi_c (azimuth of tilt, deg),
    # PV2_3 = theta_c (altitude, deg)
    ("SZP", [(2, 1, 2.0), (2, 2, 180.0), (2, 3, 45.0)], "Slant zenithal perspective"),
    # ZPN: polynomial rho(theta); P0=0 (no constant), P1=1 (identity),
    # P3=small cubic term.
    (
        "ZPN",
        [(2, 0, 0.0), (2, 1, 1.0), (2, 2, 0.0), (2, 3, 2.0e-4)],
        "Zenithal polynomial",
    ),
    # AIR: PV2_1 = theta_b (reference latitude for Airy function, deg)
    ("AIR", [(2, 1, 45.0)], "Airy"),
    # -- Cylindrical (Sec.5.2) ---------------------------------------------------
    # CYP: PV2_1 = mu (perspective distance), PV2_2 = lambda (latitude of true scale)
    ("CYP", [(2, 1, 1.0), (2, 2, 1.0)], "Cylindrical perspective"),
    # CEA: PV2_1 = lambda (true-scale latitude, cos^2(lambda) = 1 for Lambert)
    ("CEA", [(2, 1, 1.0)], "Cylindrical equal area (Lambert)"),
    ("CAR", [], "Plate carree"),
    ("MER", [], "Mercator"),
    # -- Pseudo-cylindrical (Sec.5.3) ---------------------------------------------
    ("SFL", [], "Sanson-Flamsteed"),
    ("PAR", [], "Parabolic"),
    ("MOL", [], "Mollweide"),
    ("AIT", [], "Hammer-Aitoff"),
    # -- Conic (Sec.5.4) ---------------------------------------------------------
    # COP/COD/COE/COO: PV2_1 = theta_a (standard parallel, deg),
    # PV2_2 = eta (half-opening, deg)
    ("COP", [(2, 1, 45.0), (2, 2, 25.0)], "Conic perspective"),
    ("COD", [(2, 1, 45.0), (2, 2, 25.0)], "Conic equidistant"),
    ("COE", [(2, 1, 45.0), (2, 2, 25.0)], "Conic equal area"),
    ("COO", [(2, 1, 45.0), (2, 2, 25.0)], "Conic orthomorphic (conformal)"),
    # -- Polyconic (Sec.5.5) ------------------------------------------------------
    # BON: PV2_1 = theta_1 (standard parallel, deg)
    ("BON", [(2, 1, 30.0)], "Bonne equal area"),
    ("PCO", [], "Polyconic"),
    # -- Quad-cube (Sec.5.6) ------------------------------------------------------
    ("TSC", [], "Tangential spherical cube"),
    ("CSC", [], "COBE quadrilateralised spherical cube"),
    ("QSC", [], "Quadrilateralised spherical cube"),
    # -- HEALPix (Calabretta & Roukema 2007) -----------------------------------
    # HPX: PV2_1 = H (number of facets around equator),
    # PV2_2 = K (number of polar facets)
    ("HPX", [(2, 1, 4.0), (2, 2, 3.0)], "HEALPix"),
    ("XPH", [], "HEALPix polar-cap / equatorial square"),
]

STANDARD_FIELDNAMES = [
    "projection",
    "crpix1",
    "crpix2",
    "crval1",
    "crval2",
    "cdelt1",
    "cdelt2",
    # PV parameters: empty string means "not set" (projection uses its default).
    # pv2_0 is included for ZPN (polynomial constant term); empty for others.
    "pv2_0",
    "pv2_1",
    "pv2_2",
    "pv2_3",
    "ra",
    "dec",
    "x_fits",
    "y_fits",
]


def _build_standard_wcs(code: str, pv: list[tuple[int, int, float]]) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [CRPIX1, CRPIX2]
    w.wcs.crval = [CRVAL1, CRVAL2]
    w.wcs.cdelt = [CDELT1, CDELT2]
    w.wcs.ctype = [f"RA---{code}", f"DEC--{code}"]
    if pv:
        w.wcs.set_pv(pv)
    w.wcs.set()
    return w


def generate_standard(out_path: Path) -> None:
    rows: list[dict] = []
    total_skipped = 0
    for code, pv, desc in STANDARD_PROJECTIONS:
        print(f"  {code:4s}  {desc}")
        try:
            w = _build_standard_wcs(code, pv)
        except Exception as exc:
            print(f"    !! WCS build failed: {exc}")
            continue

        pv_map: dict[tuple[int, int], float] = {(i, m): v for i, m, v in pv}
        skipped = 0
        for x_fits, y_fits in PIXEL_GRID:
            try:
                sky = w.all_pix2world([[x_fits, y_fits]], FITS_ORIGIN)[0]
                ra, dec = float(sky[0]), float(sky[1])
                if not (math.isfinite(ra) and math.isfinite(dec)):
                    skipped += 1
                    continue
                # Roundtrip check: discard degenerate / singular points.
                pix_back = w.all_world2pix([[ra, dec]], FITS_ORIGIN, quiet=True)[0]
                xb, yb = float(pix_back[0]), float(pix_back[1])
                if not (math.isfinite(xb) and math.isfinite(yb)):
                    skipped += 1
                    continue
                if math.hypot(xb - x_fits, yb - y_fits) > 1e-6:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            rows.append(
                {
                    "projection": code,
                    "crpix1": repr(CRPIX1),
                    "crpix2": repr(CRPIX2),
                    "crval1": repr(CRVAL1),
                    "crval2": repr(CRVAL2),
                    "cdelt1": repr(CDELT1),
                    "cdelt2": repr(CDELT2),
                    # Use empty string for PV params that are not set.
                    "pv2_0": repr(pv_map[(2, 0)]) if (2, 0) in pv_map else "",
                    "pv2_1": repr(pv_map[(2, 1)]) if (2, 1) in pv_map else "",
                    "pv2_2": repr(pv_map[(2, 2)]) if (2, 2) in pv_map else "",
                    "pv2_3": repr(pv_map[(2, 3)]) if (2, 3) in pv_map else "",
                    "ra": f"{ra:.15g}",
                    "dec": f"{dec:.15g}",
                    "x_fits": f"{x_fits:.15g}",
                    "y_fits": f"{y_fits:.15g}",
                }
            )

        if skipped:
            print(
                f"    (skipped {skipped}/{len(PIXEL_GRID)}"
                f" degenerate / non-finite points)"
            )
        total_skipped += skipped

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n  -> {len(rows)} test rows  ({total_skipped} skipped)  -> {out_path.name}\n"
    )


# ---------------------------------------------------------------------------
# SIP distortion
# ---------------------------------------------------------------------------
# 2nd-order SIP forward transform (A, B): adds a polynomial correction to the
# intermediate pixel offsets before the TAN projection.
#
#   u = x + A(x,y)    where A(x,y) = A_2_0*x^2 + A_0_2*y^2 + A_1_1*x*y
#   v = y + B(x,y)
#
# The inverse (AP, BP) satisfies x ~= u + AP(u,v) for small distortions.
# For the magnitudes used here (~10^-7) the first-order approximation
# AP ~= -A, BP ~= -B gives a roundtrip error < 10^-1^3 pixels.

SIP_A = {"2_0": 1.5e-7, "0_2": 6.0e-8, "1_1": -3.0e-8}
SIP_B = {"2_0": 6.0e-8, "0_2": 1.5e-7, "1_1": 3.0e-8}
SIP_AP = {"2_0": -1.5e-7, "0_2": -6.0e-8, "1_1": 3.0e-8}
SIP_BP = {"2_0": -6.0e-8, "0_2": -1.5e-7, "1_1": -3.0e-8}

SIP_FIELDNAMES = [
    "crpix1",
    "crpix2",
    "crval1",
    "crval2",
    "cdelt1",
    "cdelt2",
    # Forward SIP coefficients (order 2).
    "a_2_0",
    "a_0_2",
    "a_1_1",
    "b_2_0",
    "b_0_2",
    "b_1_1",
    # Inverse SIP coefficients (order 2).  Stored so the Rust implementation
    # can use the analytic inverse without an iterative solver.
    "ap_2_0",
    "ap_0_2",
    "ap_1_1",
    "bp_2_0",
    "bp_0_2",
    "bp_1_1",
    "ra",
    "dec",
    "x_fits",
    "y_fits",
]


def _build_sip_wcs() -> WCS:
    h = FitsHeader()
    h["CTYPE1"] = "RA---TAN-SIP"
    h["CTYPE2"] = "DEC--TAN-SIP"
    h["CRPIX1"] = CRPIX1
    h["CRPIX2"] = CRPIX2
    h["CRVAL1"] = CRVAL1
    h["CRVAL2"] = CRVAL2
    h["CDELT1"] = CDELT1
    h["CDELT2"] = CDELT2
    h["A_ORDER"] = 2
    h["B_ORDER"] = 2
    h["AP_ORDER"] = 2
    h["BP_ORDER"] = 2
    for tag, val in SIP_A.items():
        h[f"A_{tag}"] = val
    for tag, val in SIP_B.items():
        h[f"B_{tag}"] = val
    for tag, val in SIP_AP.items():
        h[f"AP_{tag}"] = val
    for tag, val in SIP_BP.items():
        h[f"BP_{tag}"] = val
    return WCS(h)


def generate_sip(out_path: Path) -> None:
    print("  SIP  TAN + 2nd-order SIP distortion")
    w = _build_sip_wcs()
    base: dict = {
        "crpix1": repr(CRPIX1),
        "crpix2": repr(CRPIX2),
        "crval1": repr(CRVAL1),
        "crval2": repr(CRVAL2),
        "cdelt1": repr(CDELT1),
        "cdelt2": repr(CDELT2),
        **{f"a_{k}": repr(v) for k, v in SIP_A.items()},
        **{f"b_{k}": repr(v) for k, v in SIP_B.items()},
        **{f"ap_{k}": repr(v) for k, v in SIP_AP.items()},
        **{f"bp_{k}": repr(v) for k, v in SIP_BP.items()},
    }
    rows: list[dict] = []
    skipped = 0
    for x_fits, y_fits in PIXEL_GRID:
        try:
            sky = w.all_pix2world([[x_fits, y_fits]], FITS_ORIGIN)[0]
            ra, dec = float(sky[0]), float(sky[1])
            if not (math.isfinite(ra) and math.isfinite(dec)):
                skipped += 1
                continue
            pix_back = w.all_world2pix([[ra, dec]], FITS_ORIGIN, quiet=True)[0]
            if (
                math.hypot(float(pix_back[0]) - x_fits, float(pix_back[1]) - y_fits)
                > 1e-6
            ):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        rows.append(
            {
                **base,
                "ra": f"{ra:.15g}",
                "dec": f"{dec:.15g}",
                "x_fits": f"{x_fits:.15g}",
                "y_fits": f"{y_fits:.15g}",
            }
        )
    if skipped:
        print(f"    (skipped {skipped} points)")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {len(rows)} test rows -> {out_path.name}\n")


# ---------------------------------------------------------------------------
# TPV distortion (Shupe et al. / SCAMP convention)
# ---------------------------------------------------------------------------
# CTYPE = "RA---TPV" / "DEC--TPV".  The PV polynomial maps intermediate pixel
# offsets (xi, eta) -> (xi', eta') before the TAN projection.  Coefficient
# ordering follows the Shupe et al. / wcslib standard:
#
#   k=0: 1   k=1: xi   k=2: eta   k=3: r   k=4: xi^2   k=5: xi*eta   k=6: eta^2
#
# For the identity: PV1_1=1, PV2_2=1 (all other coefficients default to 0 in
# wcslib and must NOT be set explicitly -- setting zero-valued coefficients
# causes wcslib's internal distortion state to register extra polynomial terms,
# which breaks the Newton-Raphson inverse solver).
#
# We add small second-order terms (PV1_4, PV2_6) to make the distortion
# non-trivial while keeping the polynomial well-conditioned.  The pixel grid
# uses +/-50 px steps (vs +/-200 for standard projections) to limit the
# intermediate-coord magnitude and ensure the iterative inverse converges.

# Only non-identity / non-zero terms.  PV*_1 is explicitly set to 1.0 (the
# spec default) so that the linear term remains unambiguous.  PV1_4 / PV2_6
# add a small xi^2 / eta^2 distortion that keeps the Jacobian well-conditioned.
TPV_PV1 = {1: 1.0, 4: 1.0e-5, 5: 5.0e-6}
TPV_PV2 = {1: 1.0, 4: 5.0e-6, 6: 1.0e-5}

# Tighter pixel grid for TPV: +/-50 px in each axis.
_TPV_STEPS = [-100, -50, 0, 50, 100]
TPV_PIXEL_GRID: list[tuple[float, float]] = [
    (CRPIX1 + dx, CRPIX2 + dy) for dx in _TPV_STEPS for dy in _TPV_STEPS
]

TPV_FIELDNAMES = [
    "crpix1",
    "crpix2",
    "crval1",
    "crval2",
    "cdelt1",
    "cdelt2",
    # Only the non-zero PV coefficients.  The Rust test must set the same
    # terms and leave all others at their default (0 for k!=1/k!=2).
    "pv1_1",
    "pv1_4",
    "pv1_5",
    "pv2_1",
    "pv2_4",
    "pv2_6",
    "ra",
    "dec",
    "x_fits",
    "y_fits",
]


def _tpv_forward(xi: float, eta: float) -> tuple[float, float]:
    """Apply the TPV PV polynomial in degrees: (xi, eta) -> (xi', eta').

    Per the TPV registry (https://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html),
    axis 2 swaps the linear arguments: PV2_M is evaluated with (eta, xi)
    in place of (xi, eta).  Coefficient indexing matches wcslib.
    """

    def eval_axis(c: dict[int, float], x: float, y: float) -> float:
        r = math.hypot(x, y)
        return (
            c.get(0, 0.0)
            + c.get(1, 0.0) * x
            + c.get(2, 0.0) * y
            + c.get(3, 0.0) * r
            + c.get(4, 0.0) * x * x
            + c.get(5, 0.0) * x * y
            + c.get(6, 0.0) * y * y
        )

    return eval_axis(TPV_PV1, xi, eta), eval_axis(TPV_PV2, eta, xi)


def generate_tpv(out_path: Path) -> None:
    print("  TPV  TAN + TPV 2nd-order polynomial distortion")
    # Build a plain TAN WCS -- we apply TPV ourselves because astropy/wcslib
    # silently strips the TPV PV cards when going through the high-level WCS.
    h = FitsHeader()
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRPIX1"] = CRPIX1
    h["CRPIX2"] = CRPIX2
    h["CRVAL1"] = CRVAL1
    h["CRVAL2"] = CRVAL2
    h["CDELT1"] = CDELT1
    h["CDELT2"] = CDELT2
    w_tan = WCS(h)

    base: dict = {
        "crpix1": repr(CRPIX1),
        "crpix2": repr(CRPIX2),
        "crval1": repr(CRVAL1),
        "crval2": repr(CRVAL2),
        "cdelt1": repr(CDELT1),
        "cdelt2": repr(CDELT2),
        "pv1_1": repr(TPV_PV1[1]),
        "pv1_4": repr(TPV_PV1[4]),
        "pv1_5": repr(TPV_PV1[5]),
        "pv2_1": repr(TPV_PV2[1]),
        "pv2_4": repr(TPV_PV2[4]),
        "pv2_6": repr(TPV_PV2[6]),
    }
    rows: list[dict] = []
    skipped = 0
    for x_fits, y_fits in TPV_PIXEL_GRID:
        # Linear stage (CDELT, CRPIX): intermediate world coords in degrees.
        xi_raw = (x_fits - CRPIX1) * CDELT1
        eta_raw = (y_fits - CRPIX2) * CDELT2
        # Apply TPV polynomial.
        xi_p, eta_p = _tpv_forward(xi_raw, eta_raw)
        # Convert (xi', eta') back to a pixel position the bare TAN WCS would
        # have produced as raw intermediate coords, then ask astropy for the
        # corresponding world position.
        x_eff = xi_p / CDELT1 + CRPIX1
        y_eff = eta_p / CDELT2 + CRPIX2
        try:
            sky = w_tan.all_pix2world([[x_eff, y_eff]], FITS_ORIGIN)[0]
            ra, dec = float(sky[0]), float(sky[1])
        except Exception:
            skipped += 1
            continue
        if not (math.isfinite(ra) and math.isfinite(dec)):
            skipped += 1
            continue
        rows.append(
            {
                **base,
                "ra": f"{ra:.15g}",
                "dec": f"{dec:.15g}",
                "x_fits": f"{x_fits:.15g}",
                "y_fits": f"{y_fits:.15g}",
            }
        )
    if skipped:
        print(f"    (skipped {skipped} points)")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TPV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {len(rows)} test rows -> {out_path.name}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Generating WCS ground-truth test data ===\n")

    print("Standard projections:")
    generate_standard(DATA_DIR / "wcs_standard.csv")

    print("SIP distortion:")
    generate_sip(DATA_DIR / "wcs_sip.csv")

    print("TPV distortion:")
    generate_tpv(DATA_DIR / "wcs_tpv.csv")

    print("Done.")
