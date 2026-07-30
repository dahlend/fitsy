# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ZCMPTYPE = 'NOCOMPRESS'` (Sec.10 Table 10): tiles hold the pixel
  bytes verbatim. Verified against an astropy fixture; Table 10 is done.
- Binary-table WCS (Sec.8.2 Table 22), both the pixel-list and
  BINTABLE-vector forms, via `TableWcs`; wcslib-validated to 1e-11 deg.
- `Wcs` carries `SSYSSRCa`/`ZSOURCEa`/`VELANGLa` and per-axis `CNAMEia`,
  and `to_header` now writes the whole spectral-frame group.
- Random-groups parameter scaling (Sec.6.1.2): `group_parameters`, and
  `group_parameter_by_name`, which sums repeated `PTYPEn` slots.
- Air-wavelength axes: `AWAV` and the `A2F`/`A2W`/`A2V`/`F2A`/`W2A`/
  `V2A` codes, on the Cox (2000) index, matching wcslib and astropy.
- Grism axes: `-GRI`/`-GRA` (Paper III Sec.5.1), the disperser carried
  in `PVk_0..6`; validated against wcslib and the Paper III headers.
- Time axes (Sec.9.5.3): `Wcs::time` resolves the scale per Sec.9.2.1,
  plus `TREFPOS`/`TREFDIR`/`PLEPHEM` and `CZPHS`/`CPERI`/`CRDER`/`CSYER`.
- Multi-dimensional `-TAB` (Paper III Sec.6.1.1): axes sharing one
  coordinate array interpolate together M-linearly; wcslib-validated.
- `fitsy::units`: a Sec.4.3 unit parser returning scale and dimensions,
  a lenient mode for informal spellings, and `mag`/`dB`/`Np` as levels.
- `atol` on `fitsy.diff` / `DiffOptions::absolute_tolerance`.
- `Wcs.pixel_shape` and `Wcs.footprint()`.
- `Clone` on `Wcs`, `WcsFit`, the HDU views, and the builders.

### Changed

- Breaking: `Wcs` keeps only keywords its description uses; the
  round-trip contract is `from_header(to_header(w)) == w`, not bytes.
- `Wcs::from_header` rejects an alternate code outside `' '` and
  `'A'`-`'Z'` (Sec.8.2); it previously accepted any ASCII character.
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
- Breaking (Rust): `header::units::si_factor` is gone (see
  `fitsy::units`); `real_in_unit` refuses a dimension mismatch.
- Breaking (Rust): `TabAxis` becomes `TabGroup`, owning every axis that
  shares a coordinate array; `Wcs` gains `celestial_pair`.
- Breaking (Rust): `wcs::units` is gone, folded into `units::factor_to`.
- `rust.missing_docs = "deny"`: every public item carries a doc comment.
- Breaking (Rust): the parallel `naxis`/`ctype`/`cunit`/`crval` fields
  become accessors on `Axis`/`LinearTransform`; desync is impossible.
- Breaking (Rust): registry-tracking enums (`ProjectionKind`,
  `SpectralKind`, `BinValue`, ...) are `#[non_exhaustive]`.
- Breaking (Rust): `Grism` is exhaustive again -- Paper III Table 7
  fixes the disperser at exactly seven parameters.
- Breaking (Rust): `Projection` implementors must supply `pv2`.
- Breaking (Rust): new `BinValue::StrArray` for `A` columns with `TDIMn`.
- Breaking (Rust): `CelestialRotation::new` takes `phi0`;
  `LinearTransform::from_crota` takes the rotated axis pair.

### Fixed

- ASCII table real fields follow Sec.7.2.5 in full: the implicit
  decimal point (`F10.3` over `12345` is 12.345) and `1.234+05` forms.
- A blank ASCII table numeric field reads as zero (Sec.7.2.5), not
  `None`.
- `TNULLn` applies to any ASCII table column, not just `I` (Sec.7.2.4),
  and `AsciiTableBuilder` refuses `NaN` in a float column without it.
- An orphaned `CONTINUE` is commentary (Sec.4.2.1.2), not an error; an
  unmatched trailing `&` is the literal last character.
- The Sec.4.3.1 numeric multiplier may abut its units: `10**(46)erg/s`
  and `10-3m` now parse; `m1.5` is still refused.
- `CUNIT` fell back to a factor of 1.0 when unrecognized (`'km s-1'`
  read as m/s); units are now parsed and checked dimensionally.
- Python `Header.value_in_si` could never convert: its blank internal
  target made every annotated lookup a mismatch. It now converts.
- `RESTFRQa`/`RESTWAVa` ignored the alternate version code on both
  read and write (Table 22 footnote 4).
- `PVi_m` was collected only to `m = 19`, dropping ZPN's `PV2_20`
  (Sec.8.2 allows `m` up to 99).
- A linear velocity axis without `RESTFRQ`/`RESTWAV` failed the whole
  parse; Paper III needs a rest quantity only for the `*2V`/`V2*` codes.
- An unrecognized spectral algorithm code fell through to the linear
  pipeline; codes outside Sec.8.4 Table 26 are now rejected.
- A spectral code disagreeing with its type (`ZOPT-F2V`, Paper III
  Sec.3.3.1) was accepted and silently reinterpreted; now rejected.
- `TIMESYS` values carrying a realization (`TT(TAI)`, Sec.9.2.1)
  matched nothing; the Sec.9.3 units `cy`/`ta`/`Ba` parse too.
- A descending `-TAB` index vector with a legal repeated value
  (Sec.6.1.1) was rejected; `PVi_2` (`EXTLEVEL`) is parsed and written.
- `Wcs::from_header` panicked on a CTYPE whose fourth byte fell inside
  a multi-byte character.
- A negative `PCOUNT`/`GCOUNT` was cast straight to `u64`; Sec.7.1.3
  makes both non-negative, and the error now says so.
- ISO-8601 dates were range-checked only against 31, so `2024-02-31`
  parsed and rolled forward.
- A lone `MJDREFI`/`MJDREFF` was ignored; Sec.9.2.2 defaults each half
  to zero.
- Signed five-digit years (Sec.9.1.1, `-04713-11-24T12:00:00`) failed
  to parse.
- `TIMEOFFS` was never applied to `TSTART`/`TSTOP`-derived times
  (Sec.9.4.1).
- `AZP`, `SZP` and `COP` projected points past their own fold, placing
  far-side sources inside the image; each forward errors, like wcslib.
- `HPX` put the polar facet one facet too far at exactly `phi = +180`
  (`x` was 223.08 instead of 136.92 at `H = 4`, `theta = 88`).
- Native longitude was normalised to `[-180, 180)` where Paper II
  defines `(-180, 180]`, mirroring the antipodal meridian.
- The three projection-domain fixes lift wcslib agreement over a
  280-point sweep of all 28 projections from 174/224 to 275/280.
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