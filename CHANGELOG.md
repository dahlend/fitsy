# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.0]

### Added

- `Header` can now be built from Python via `Header()`, `Header(mapping)`, and `Header.fromstring`/`frombytes`.
- `fitsy.image`, `fitsy.compressed_image`, and `fitsy.append` now accept a `Header`, not just a `dict`.
- `lenient` parsing now tolerates most malformed header content:
  bad values, non-standard keywords, a lower-case `end` terminator, and
  broken `CONTINUE` chains.
- Regression tests decoding astropy-written RICE, HCOMPRESS, and dithered
  fixtures, plus WCS reference points in previously-untested regions.

### Changed

- Header parsing is now lenient by default.
- Non-ASCII bytes in header comments are now sanitized rather than
  rejected by default.

### Fixed

- RICE decode/encode was off by one pixel versus cfitsio/fpack.
- Subtractive-dither seed was off by one, corrupting dithered float tiles.
- HCOMPRESS now uses 64-bit coefficients for 32-bit and float tiles.
- SIN slant projection had a sign error in the `PV2_1` term.
- HPX polar-zone formulas were incorrect in both directions.
- XPH inverse projection was missing a factor of one half.
- Quad-cube (TSC/CSC/QSC) equatorial face layout was mirrored.
- Celestial rotation flipped the pole longitude sign at `delta_p = +/-90`.
- DSS RA was mirrored; AMDX/AMDY terms 14-20 are no longer misread as
  polynomial terms.
- TNX/ZPX ordinary-polynomial surfaces were wrongly normalized.
- HPX now applies its documented `H=4`, `K=3` defaults when PV cards are absent.
- Scaled integer table columns now honor `TNULL` (undefined -> NaN).
- Long-string writer no longer splits a `''` escape across `CONTINUE` cards.
- Random Groups files can now be re-serialized.
- The writer accepts integral-real `PCOUNT`/`GCOUNT` (e.g. `0.`) that the
  lenient reader already accepts.
- `Header::parse` tolerates trailing bytes after the header block.
- `Wcs::crval` on the celestial axis pair was zeroed out after parsing a
  header instead of holding the true reference value.


## [v0.1.4]

### Fixed

- Polynomial distortions (SIP, TPV, TNX) were failing to converge for large
  values (outside of frame). This is now fixed.


## [v0.1.3]

### Added

- Added header validation tests.
- Added best-effort parsing of time and observer positions from Headers.

## [v0.1.2]

### Added

- Added support for constructing ImageData and ImageBuilder from a dmatrix or faer.

## [v0.1.1]

Initial Release!