# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.0]

This release centers on world coordinates. It restructures the WCS API
and removes the celestial-only transforms. The parser now reads
binary-table, time and grism axes. Transforms accept a batch of points
and run faster. The release fixes many projection, pole, spectral and
time defects. It also fixes defects in the table, compression and
update paths.

### Added

- Binary-table WCS via `TableWcs`, both the pixel-list and
  BINTABLE-vector forms.
- Multi-dimensional `-TAB`: axes sharing one coordinate array
  interpolate together.
- Time axes: `Wcs::time`, plus `TREFPOS`/`TREFDIR`/`PLEPHEM` and
  `CZPHS`/`CPERI`/`CRDER`/`CSYER`.
- `Wcs` carries `SSYSSRCa`/`ZSOURCEa`/`VELANGLa` and `CNAMEia`, and
  `to_header` writes the spectral-frame group.
- Air-wavelength axes: `AWAV` and the `A2F`/`A2W`/`A2V`/`F2A`/`W2A`/
  `V2A` codes, on the Cox (2000) index.
- Grism axes `-GRI`/`-GRA`, the disperser carried in `PVk_0..6`.
- `Wcs.pixel_shape`, and `Wcs.footprint()` returning the corner pixels
  in world coordinates.
- `Wcs::axis_kind`, `Wcs::axis_kinds`, `AxisKind` and `Wcs::is_tabular`:
  what each axis carries, by meaning rather than position.
- Batch transforms `Wcs::pixel_to_world_many` / `world_to_pixel_many`,
  taking the points flat, `NAXIS` per point.
- Python `Wcs.pixel_to_world` / `world_to_pixel` take one point or
  many: a length-`naxis` sequence, or an `(N, naxis)` array.
- `fitsy::units`: a unit parser, a lenient mode for informal
  spellings, and `mag`/`dB`/`Np` as levels.
- `atol` on `fitsy.diff` / `DiffOptions::absolute_tolerance`.
- `ZCMPTYPE = 'NOCOMPRESS'`, completing Table 10.
- Random-groups parameter scaling: `group_parameters` and
  `group_parameter_by_name`.
- `Clone` on `Wcs`, `WcsFit`, the HDU views, and the builders.

### Changed

- Breaking: `wcs::sip`, `wcs::tpv`, `wcs::tnx` and `wcs::dss` moved
  under `wcs::distortion`, which `wcs` re-exports from.
- Breaking: `wcs::projections` merged into `wcs::projection`.
- Breaking: `Projection` is an enum over Table 13; the trait,
  `ProjectionKind` and `build()` are gone, folded into `from_code`.
- Breaking: distortion and plate-model parameters are private
  and set at construction; `Dss` gains `new`.
- Breaking: `CelestialRotation::new` takes `phi0`;
  `LinearTransform::from_crota` takes the rotated axis pair.
- Breaking: `CelestialRotation::theta_p` becomes a method, so
  the cached sine and cosine cannot fall out of step with it.
- Breaking: `TabAxis` becomes `TabGroup`, owning every axis that
  shares a coordinate array; `Wcs` gains `celestial_pair`.
- Breaking: a Python `wcs()` accessor raises when a `-TAB` axis has
  no file to resolve against; `fitsy.Wcs(header)` stays header-only.
- Breaking: `Grism` is exhaustive again -- Paper III Table 7
  fixes the disperser at exactly seven parameters.
- Breaking: `Wcs` keeps only keywords its description uses; the
  round-trip contract is `from_header(to_header(w)) == w`, not bytes.
- Breaking: the parallel `naxis`/`ctype`/`cunit`/`crval` fields
  become accessors on `Axis`/`LinearTransform`; desync is impossible.
- Breaking: `Wcs::linear` is private, read via `linear()` and
  replaced via `set_linear`, which rejects a rank mismatch.
- Breaking: `#[non_exhaustive]` is dropped across the API -- the
  standard is complete; it stays on `Hdu` and `FitsError`.
- Breaking: `header::units::si_factor` is gone (see `fitsy::units`);
  `real_in_unit` refuses a dimension mismatch.
- Breaking: `wcs::units` is gone, folded into `units::factor_to`.
- Breaking: new `BinValue::StrArray` for `A` columns with `TDIMn`.
- `Wcs::from_header` rejects an alternate code outside `' '` and
  `'A'`-`'Z'`.
- `ImageHdu.wcs()` (Python) and `FitsFile::wcs_inherited` resolve
  `-TAB` lookup axes, as `FitsFile::wcs` does.
- `Wcs::to_header` round-trips: it writes the projection parameters,
  pole conventions, TPV, TNX/ZPX, DSS, spectral and `-TAB` cards.
- Batch WCS transforms yield `NaN` outside the projection's domain
  instead of failing the whole call.
- WCS transforms are substantially faster: a batch builds its buffers
  once, a two-axis celestial WCS is specialized end to end (TNX and ZPX
  included), SIP evaluates by Horner's scheme in one pass, the
  projection and distortion inverses use bracketed Newton with analytic
  derivatives rather than fixed-width bisection, and the Python
  bindings read a C-contiguous batch in place.
- Reads do less work: `FitsFile::wcs` and `fitsy info` / `checksum`
  read headers only, instead of every HDU's data section, and `flush`
  and `writeto` share one slot snapshot and one temp file.
- Scaled `BITPIX` 8/16 images decode to `float32`, matching astropy.
- Unscaled image reads decode into numpy's buffer, which is faster and
  halves peak memory. Adds `bytemuck`, behind the `python` feature.
- `bintable` writes `list[list[str]]` as an `nA` column with `TDIMn`.
- Character columns reject non-ASCII instead of writing it through.
- `RandomGroups.group()` arrays are frozen, like the table accessors.
- `diff` compares image pixels and table cells numerically, in physical
  units, honoring `rtol`/`atol`.
- `funpack` recomputes `CHECKSUM`/`DATASUM` per HDU, as cfitsio does;
  `-C` opts out.
- `fitsy header` prints reals as the shortest round-tripping decimal.
- The image, table, and WCS entry points accept any array-like, and
  native byte-order arrays skip the Python round trip on write.
- `fitsy.getdata`, `setval` and `delval` follow `astropy.io.fits` for
  empty HDUs, omitted values and missing keywords.
- Reading `hdu.data` in update mode no longer forces a rewrite.
- Type stubs ship inside the package beside a `py.typed` marker.
- `rust.missing_docs = "deny"`: every public item carries a doc comment.
- The Python test suite no longer requires astropy, so CI runs all of
  it rather than skipping half.

### Removed

- Breaking: `Wcs::pixel_to_celestial`, `celestial_to_pixel` and their
  `_many` forms, in Rust and Python. Use `pixel_to_world` /
  `world_to_pixel`, locating the pair with `axis_kinds`.

### Fixed

- The celestial rotation read the latitude from `asin`, ill-conditioned
  at the zenithal pole.
- Native-pole selection took the wrong root of Paper II eq. (9),
  mirroring the sky at `LONPOLE = 180`.
- `yzLN`/`yzLT` and generic `xLON`/`xLAT` CTYPE pairs parsed as
  non-celestial, returning unprojected values.
- `PVi_1`/`PVi_2`/`PVi_3`/`PVi_4` on the longitude axis were ignored,
  so a relocated fiducial point changed nothing.
- Native longitude was normalized to `[-180, 180)` where Paper II
  defines `(-180, 180]`, mirroring the antipodal meridian.
- `to_header` dropped `RADESYS` for a tabular celestial pair
  (`RA---TAB`/`DEC--TAB`): the card was gated on the `CelestialBlock`,
  which a tabular pair does not carry. Without `EQUINOX` the frame
  re-parsed as the ICRS default.
- A fitted `CelestialFrame::Other` WCS lost its celestial block on a
  `to_header` round trip.
- `Air::theta_b` left its cached branch constants stale when assigned,
  so the projection disagreed with its own `PV2_1`.
- `AIR` folded silently below a `PV2_1` of about -76.5 degrees, where
  `R` turns over.
- `ZEA`, `SZP` and `AIR` each canceled at the pole their field sits
  on; all three now use the half-angle form.
- `AZP`, `SZP` and `COP` projected points past their own fold, placing
  far-side sources inside the image; each forward errors, like wcslib.
- The slant `SIN` accepted the whole sphere, where only the
  orthographic case was checked. Paper II eq. (30) projects along
  `(PV2_1, PV2_2, 1)` and is two-to-one, so a point on the far
  hemisphere round-tripped to its reflection.
- `HPX` put the polar facet one facet too far at exactly `phi = +180`.
- `PVi_m` was collected only to `m = 19`, dropping ZPN's `PV2_20`
  (Sec.8.2 allows `m` up to 99).
- `Dss::world_to_pixel` returned `Ok` with the last iterate once its
  Newton loop hit the step limit; non-convergence is now an error.
- `Wcs::world_to_pixel` on a DSS plate solution returned 0 for every
  non-celestial axis.
- `CROTAi` was read only from `CROTA2` and only for `NAXIS = 2`, so
  every cube came back with its rotation dropped.
- A non-finite linear matrix, and `CDELTi = 0` with `CROTA`, built a
  WCS that answered every query with `Ok(NaN)`.
- `RESTFRQa`/`RESTWAVa` ignored the alternate version code on both
  read and write.
- A linear velocity axis without `RESTFRQ`/`RESTWAV` failed the whole
  parse; Paper III needs a rest quantity only for the `*2V`/`V2*` codes.
- An unrecognized spectral algorithm code fell through to the linear
  pipeline; codes outside Sec.8.4 Table 26 are now rejected.
- A spectral code disagreeing with its type (`ZOPT-F2V`) was accepted
  and silently reinterpreted.
- A spectral axis giving only `RESTWAV` panicked.
- `TIMESYS` values carrying a realization (`TT(TAI)`) matched nothing;
  the units `cy`/`ta`/`Ba` parse too.
- A lone `MJDREFI`/`MJDREFF` was ignored; Sec.9.2.2 defaults each half
  to zero.
- `TIMEOFFS` was never applied to `TSTART`/`TSTOP`-derived times.
- `TDB` was reduced as `TT`, and a `23:59:60` stamp landed one second
  late.
- A descending `-TAB` index vector with a legal repeated value was
  rejected; `PVi_2` (`EXTLEVEL`) is parsed and written.
- `FitsFile.wcs()` never resolved `-TAB` axes, which also extrapolated
  past the Paper III Sec.6.1.2 limit.
- `Wcs::from_header` panicked on a CTYPE whose fourth byte fell inside
  a multi-byte character.
- The numeric multiplier may abut its units: `10**(46)erg/s` and
  `10-3m` now parse; `m1.5` is still refused.
- `CUNIT` fell back to a factor of 1.0 when unrecognized; units are now
  parsed and checked dimensionally.
- Python `Header.value_in_si` could never convert: its blank internal
  target made every annotated lookup a mismatch.
- An orphaned `CONTINUE` is commentary, not an error; an unmatched
  trailing `&` is the literal last character.
- ISO-8601 dates were range-checked only against 31, so `2024-02-31`
  parsed and rolled forward.
- Signed five-digit years (`-04713-11-24T12:00:00`) failed to parse.
- The `-`/`_` keyword-lookup fallback ran both ways, so `first("CD1_1")`
  could answer with an unrelated `CD1-1` card.
- `DATASUM` was written unpadded, under the 8-character minimum.
- ASCII table real fields follow Sec.7.2.5 in full: the implicit
  decimal point (`F10.3` over `12345` is 12.345) and `1.234+05` forms.
- A blank ASCII table numeric field reads as zero (Sec.7.2.5), not
  `None`.
- `TNULLn` applies to any ASCII table column, not just `I` (Sec.7.2.4),
  and `AsciiTableBuilder` refuses `NaN` in a float column without it.
- Unsigned `I`/`J` table columns read the lower half of their range
  2^16 / 2^32 too high.
- An overflowing variable-length-array descriptor panicked.
- `BinTable.data` / `AsciiTable.data` were writeable but rebuilt per
  access, so edits vanished; now frozen.
- `A` columns with a `TDIMn` returned one concatenated string.
- A `TDIMn` product smaller than the `TFORMn` repeat was ignored.
- Tile-compressed headers stripped the leading `Z` off every keyword,
  so `ZP` became `P`.
- `ZTILE` and the container's `CHECKSUM`/`DATASUM` leaked into the
  decompressed image header.
- A negative `PCOUNT`/`GCOUNT` was cast straight to `u64`; Sec.7.1.3
  makes both non-negative.
- Reading an empty (`NAXIS = 0`) image HDU failed instead of returning
  an empty array.
- An absurd `NAXISn` product overflowed when padded to a block.
- `hdu.data[...] = v` in update mode was dropped on `flush`,
  tile-compressed images included.
- Images whose header layout cards disagreed with the HDU failed to
  read.
- `FitsAppender::open` truncated trailing bytes that `FitsFile::open`
  tolerates, even when nothing was appended; `finish` trims now.
- `diff` reported two files as identical when they differed only in an
  HDU whose `XTENSION` fitsy does not recognize.


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