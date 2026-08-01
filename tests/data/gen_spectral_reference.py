"""Generate wcslib reference values for the Paper III spectral algorithms.

Covers the air-wavelength dispersion relation (Sec.4) and the grism
coordinate function (Sec.5.1), which have no closed-form check that is
independent of the implementation under test.

The three ``AWAV-GRA`` headers are the real KPNO spectrographs printed
in Paper III Figs. 3-5.

Run with the repo venv:
    .venv/bin/python tests/data/gen_spectral_reference.py
"""

from pathlib import Path

import numpy as np
from astropy.wcs import WCS

# A flat line format, not JSON: no Rust test in this repo pulls in a
# JSON parser and the fixture does not need one.
#   CASE <name>
#   KEYF|KEYI|KEYS <keyword> <value>
#   DATA <pixel> <world>
OUT = Path(__file__).parent / "spectral_reference.txt"

# (name, header, pixel samples). Pixels are 0-based, as fitsy's API is.
CASES = [
    # --- air <-> vacuum dispersion, Paper III Sec.4 ---
    (
        "awav_linear",
        {
            "CTYPE1": "AWAV",
            "CRVAL1": 5.0e-7,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e-10,
            "CUNIT1": "m",
        },
    ),
    (
        "awav_f2a",
        {
            "CTYPE1": "AWAV-F2A",
            "CRVAL1": 5.0e-7,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e-10,
            "CUNIT1": "m",
        },
    ),
    (
        "wave_a2w",
        {
            "CTYPE1": "WAVE-A2W",
            "CRVAL1": 5.0e-7,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e-10,
            "CUNIT1": "m",
        },
    ),
    (
        "freq_a2f",
        {
            "CTYPE1": "FREQ-A2F",
            "CRVAL1": 6.0e14,
            "CRPIX1": 1.0,
            "CDELT1": -1.0e11,
            "CUNIT1": "Hz",
        },
    ),
    (
        "awav_w2a",
        {
            "CTYPE1": "AWAV-W2A",
            "CRVAL1": 5.0e-7,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e-10,
            "CUNIT1": "m",
        },
    ),
    (
        "velo_a2v",
        {
            "CTYPE1": "VELO-A2V",
            "CRVAL1": 0.0,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e3,
            "CUNIT1": "m/s",
            "RESTWAV": 5.0e-7,
        },
    ),
    (
        "awav_v2a",
        {
            "CTYPE1": "AWAV-V2A",
            "CRVAL1": 5.0e-7,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e-10,
            "CUNIT1": "m",
            "RESTWAV": 5.0e-7,
        },
    ),
    # --- grism, Paper III Sec.5.1 ---
    # Pure grating in vacuum.
    (
        "wave_gri_grating",
        {
            "CTYPE1": "WAVE-GRI",
            "CRVAL1": 5.0e-7,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e-10,
            "CUNIT1": "m",
            "PV1_0": 3.16e5,
            "PV1_1": 1.0,
            "PV1_2": 13.9,
        },
    ),
    # Grism proper: n_r and n'_r active.
    (
        "wave_gri_grism",
        {
            "CTYPE1": "WAVE-GRI",
            "CRVAL1": 7.2452e-7,
            "CRPIX1": 1.0,
            "CDELT1": 2.956e-10,
            "CUNIT1": "m",
            "PV1_0": 4.50e5,
            "PV1_1": 1.0,
            "PV1_2": 27.0,
            "PV1_3": 1.765,
            "PV1_4": -1.077e6,
        },
    ),
    # Non-zero epsilon and theta, the small geometric corrections.
    (
        "wave_gri_eps_theta",
        {
            "CTYPE1": "WAVE-GRI",
            "CRVAL1": 5.0e-7,
            "CRPIX1": 1.0,
            "CDELT1": 1.0e-10,
            "CUNIT1": "m",
            "PV1_0": 3.16e5,
            "PV1_1": 1.0,
            "PV1_2": 13.9,
            "PV1_5": 2.5,
            "PV1_6": 1.75,
        },
    ),
    # Spectral type != wavelength, so the S->P->lambda chain runs.
    (
        "freq_gri",
        {
            "CTYPE1": "FREQ-GRI",
            "CRVAL1": 6.0e14,
            "CRPIX1": 1.0,
            "CDELT1": -1.0e11,
            "CUNIT1": "Hz",
            "PV1_0": 3.16e5,
            "PV1_1": 1.0,
            "PV1_2": 13.9,
        },
    ),
    # --- the three real KPNO headers from Paper III Figs. 3-5 ---
    (
        "kpno_coude",
        {
            "CTYPE1": "AWAV-GRA",
            "CUNIT1": "Angstrom",
            "CRPIX1": 1801.7,
            "CRVAL1": 5225.2,
            "CDELT1": -0.4334,
            "PV1_0": 3.16e5,
            "PV1_1": 1.0,
            "PV1_2": 13.9,
        },
    ),
    (
        "kpno_hydra",
        {
            "CTYPE1": "AWAV-GRA",
            "CUNIT1": "Angstrom",
            "CRPIX1": 944.8,
            "CRVAL1": 5136.8,
            "CDELT1": -0.1287,
            "PV1_0": 3.16e5,
            "PV1_1": 11.0,
            "PV1_2": 64.8,
        },
    ),
    (
        "kpno_mars",
        {
            "CTYPE1": "AWAV-GRA",
            "CUNIT1": "Angstrom",
            "CRPIX1": 719.8,
            "CRVAL1": 7245.2,
            "CDELT1": 2.956,
            "PV1_0": 4.50e5,
            "PV1_1": 1.0,
            "PV1_2": 27.0,
            "PV1_3": 1.765,
            "PV1_4": -1.077e6,
        },
    ),
]

PIXELS = [0.0, 1.0, 200.0, 500.0, 1000.0, 1500.0, 2047.0]


def main() -> None:
    import astropy.units as u

    out = []
    for name, hdr in CASES:
        h = {"NAXIS": 1, "NAXIS1": 2048, **hdr}
        w = WCS(h)
        world = w.wcs_pix2world(np.array(PIXELS).reshape(-1, 1), 0).ravel()
        assert np.all(np.isfinite(world)), f"{name}: wcslib produced non-finite values"
        # Confirm wcslib itself round-trips, so a mismatch later is ours.
        back = w.wcs_world2pix(world.reshape(-1, 1), 0).ravel()
        assert np.allclose(back, PIXELS, atol=1e-6), f"{name}: wcslib round trip failed"

        # wcslib normalises the whole description to SI, so it answers in
        # 'm' for a header that declared 'Angstrom'. fitsy answers in the
        # declared CUNIT, so undo that here rather than in the test.
        declared = hdr.get("CUNIT1", "")
        actual = str(w.wcs.cunit[0]).strip()
        if declared and actual and declared != actual:
            factor = (1 * u.Unit(actual)).to_value(u.Unit(declared))
            world = world * factor
            print(f"  {name:22} [{actual} -> {declared}, x{factor:g}]")

        # fitsy uses the same Cox (2000) refractivity as wcslib, so
        # every code -- including the ones that cross between air and
        # vacuum -- is held to the same tight tolerance.
        tol = 1e-9

        lines = [f"CASE {name}", f"TOL {tol!r}", "KEYI NAXIS 1", "KEYI NAXIS1 2048"]
        for key, value in hdr.items():
            if isinstance(value, str):
                lines.append(f"KEYS {key} {value}")
            else:
                lines.append(f"KEYF {key} {value!r}")
        for p, wv in zip(PIXELS, world):
            lines.append(f"DATA {p!r} {float(wv)!r}")
        out.append("\n".join(lines))
        print(f"  {name:22} {world[0]:.10g} .. {world[-1]:.10g}")

    header = (
        "# wcslib reference values for Paper III spectral algorithms.\n"
        "# Generated by tests/data/gen_spectral_reference.py -- do not edit.\n"
        "# Pixels are 0-based; world values are in the header's own CUNIT.\n"
    )
    OUT.write_text(header + "\n\n".join(out) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    print("Generating wcslib spectral reference values:")
    main()
