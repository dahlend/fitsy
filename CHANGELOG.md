# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `atol` on `fitsy.diff` / `DiffOptions::absolute_tolerance`.
- `Wcs.pixel_shape` and `Wcs.footprint()`.
- `Clone` on `Wcs`, `WcsFit`, the HDU views, and the builders.

### Changed

- `Wcs::to_header` round-trips: it now writes the projection
  parameters, pole conventions, TPV, TNX/ZPX, DSS, spectral and `-TAB`
  cards it used to drop.
- `diff` compares image pixels and table cells numerically, in physical
  units, honouring `rtol`/`atol`.
- Batch WCS transforms yield `NaN` outside the projection's domain
  instead of failing the whole call.
- The image, table, and WCS entry points accept any array-like, and
  native byte-order arrays skip the Python round trip on write.
- Scaled `BITPIX` 8/16 images decode to `float32`, matching astropy.
- `fitsy.getdata`, `setval` and `delval` follow `astropy.io.fits` for
  empty HDUs, omitted values and missing keywords.
- Type stubs ship inside the package beside a `py.typed` marker.
- `funpack` recomputes `CHECKSUM`/`DATASUM` per HDU, as cfitsio does;
  `-C` opts out.
- `bintable` writes `list[list[str]]` as an `nA` column with `TDIMn`.
- Character columns reject non-ASCII instead of writing it through.
- Reading `hdu.data` in update mode no longer forces a rewrite.
- `RandomGroups.group()` arrays are frozen, like the table accessors.
- Unscaled image reads decode into numpy's buffer: ~3x faster, half the
  peak memory. Adds `bytemuck`, behind the `python` feature.
- `FitsFile::wcs` and `fitsy info` / `checksum` read headers only,
  instead of every HDU's data section.
- `fitsy header` prints reals as the shortest round-tripping decimal.
- SIP evaluation no longer allocates, and its Newton inverse uses the
  analytic Jacobian.
- Breaking (Rust): `Projection` implementors must supply `pv2`.
- Breaking (Rust): new `BinValue::StrArray` for `A` columns with `TDIMn`.
- Breaking (Rust): `CelestialRotation::new` takes `phi0`;
  `LinearTransform::from_crota` takes the rotated axis pair.

### Fixed

- `CROTAi` was read only from `CROTA2` and only for `NAXIS = 2`, so
  every cube came back with its rotation dropped (Sec.8.2).
- `PVi_1`/`PVi_2`/`PVi_3`/`PVi_4` on the longitude axis were ignored,
  so a relocated fiducial point changed nothing (Sec.8.2).
- `yzLN`/`yzLT` and generic `xLON`/`xLAT` CTYPE pairs parsed as
  non-celestial, returning unprojected values (Sec.8.4). A fitted
  `CelestialFrame::Other` WCS also lost its celestial block on a
  `to_header` round trip.
- A non-finite linear matrix, and `CDELTi = 0` with `CROTA`, built a
  WCS that answered every query with `Ok(NaN)` (Sec.8.2).
- Reading an empty (`NAXIS = 0`) image HDU failed instead of returning
  an empty array.
- `Wcs::world_to_pixel` on a DSS plate solution returned 0 for every
  non-celestial axis.
- `FitsAppender::open` truncated trailing bytes that `FitsFile::open`
  tolerates, even when nothing was appended; `finish` trims now.
- The `-`/`_` keyword-lookup fallback ran both ways, so `first("CD1_1")`
  could answer with an unrelated `CD1-1` card (Sec.4.1.2.1).
- Unsigned `I`/`J` table columns read the lower half of their range
  2^16 / 2^32 too high.
- Tile-compressed headers stripped the leading `Z` off every keyword,
  so `ZP` became `P`.
- `ZTILE` and the container's `CHECKSUM`/`DATASUM` leaked into the
  decompressed image header.
- An overflowing variable-length-array descriptor panicked.
- `DATASUM` was written unpadded, under the 8-character minimum.
- An absurd `NAXISn` product overflowed when padded to a block.
- `hdu.data[...] = v` in update mode was dropped on `flush`,
  tile-compressed images included.
- `BinTable.data` / `AsciiTable.data` were writeable but rebuilt per
  access, so edits vanished; now frozen.
- `A` columns with a `TDIMn` returned one concatenated string.
- A `TDIMn` product smaller than the `TFORMn` repeat was ignored.
- Native-pole selection took the wrong root of Paper II eq. (9),
  mirroring the sky at `LONPOLE = 180`.
- `TDB` was reduced as `TT`, and a `23:59:60` stamp landed one second
  late.
- A spectral axis giving only `RESTWAV` panicked.
- `FitsFile.wcs()` never resolved `-TAB` axes, which also extrapolated
  past the Paper III Sec.6.1.2 limit.
- `diff` reported two files as identical when they differed only in an
  HDU whose `XTENSION` fitsy does not recognize.
- Images whose header layout cards disagreed with the HDU failed to
  read.
- The Python test suite no longer requires astropy, so CI runs all of
  it rather than skipping half.


## [v0.2.0]

### Added

- `Header` can now be built from Python via `Header()`,
  `Header(mapping)`, and `Header.fromstring`/`frombytes`.
- `fitsy.image`, `fitsy.compressed_image`, and `fitsy.append` now
  accept a `Header`, not just a `dict`.
- `lenient` parsing now tolerates most malformed header content:
  bad values, non-standard keywords, a lower-case `end` terminator, and
  broken `CONTINUE` chains.
- Regression tests decoding astropy-written RICE, HCOMPRESS, and
  dithered fixtures, plus WCS reference points in previously-untested
  regions.

### Changed

- Header parsing is now lenient by default.
- Non-ASCII bytes in header comments are now sanitized rather than
  rejected by default.

### Fixed

- RICE decode/encode was off by one pixel versus cfitsio/fpack.
- Subtractive-dither seed was off by one, corrupting dithered float
  tiles.
- HCOMPRESS now uses 64-bit coefficients for 32-bit and float tiles.
- SIN slant projection had a sign error in the `PV2_1` term.
- HPX polar-zone formulas were incorrect in both directions.
- XPH inverse projection was missing a factor of one half.
- Quad-cube (TSC/CSC/QSC) equatorial face layout was mirrored.
- Celestial rotation flipped the pole longitude sign at
  `delta_p = +/-90`.
- DSS RA was mirrored; AMDX/AMDY terms 14-20 are no longer misread as
  polynomial terms.
- TNX/ZPX ordinary-polynomial surfaces were wrongly normalized.
- HPX now applies its documented `H=4`, `K=3` defaults when PV cards
  are absent.
- Scaled integer table columns now honor `TNULL` (undefined -> NaN).
- Long-string writer no longer splits a `''` escape across
  `CONTINUE` cards.
- Random Groups files can now be re-serialized.
- The writer accepts integral-real `PCOUNT`/`GCOUNT` (e.g. `0.`) that
  the lenient reader already accepts.
- `Header::parse` tolerates trailing bytes after the header block.
- `Wcs::crval` on the celestial axis pair was zeroed out after parsing a
  header instead of holding the true reference value.


## [v0.1.4]

### Fixed

- Polynomial distortions (SIP, TPV, TNX) were failing to converge for
  large values (outside of frame). This is now fixed.


## [v0.1.3]

### Added

- Added header validation tests.
- Added best-effort parsing of time and observer positions from Headers.

## [v0.1.2]

### Added

- Added support for constructing ImageData and ImageBuilder from a
  dmatrix or faer.

## [v0.1.1]

Initial Release!