# fitsy examples

Runnable examples for the Rust crate and for the Python bindings.

The user guide ([docs/guide/](../docs/guide/)) pulls several of these
files in with `literalinclude`, so those snippets and the runnable code
do not drift apart. Every Python script below is included that way. On
the Rust side, [docs/guide/wcs.rst](../docs/guide/wcs.rst) includes
`wcs.rs` and `fit_wcs.rs`, and
[docs/guide/tables.rst](../docs/guide/tables.rst) includes
`ascii_table.rs`. The other Rust examples stand on their own.

## Layout

- `*.rs` -- Rust examples. Run one with `cargo run --example NAME`.
- `python/*.py` -- the matching Python scripts.
- `data/` -- sample FITS files that the examples read.

## Rust examples

Run these from the repository root:

```sh
cargo run --example read_image
cargo run --example read_table
cargo run --example write_image
cargo run --example write_table
cargo run --example wcs
cargo run --example fit_wcs
cargo run --example ascii_table
cargo run --example compress_image
cargo run --example nalgebra_interop --features nalgebra
cargo run --example faer_interop --features faer
```

- `read_image.rs` -- opens an image, inspects the header, and decodes
  the pixels.
- `read_table.rs` -- walks the columns of a binary table and decodes
  each cell. It writes its own input first, so it needs no sample file.
- `write_image.rs` -- builds and writes a two-dimensional image.
- `write_table.rs` -- builds and writes a binary table of several
  columns.
- `wcs.rs` -- transforms pixel coordinates to sky coordinates, in
  single and batch form, and reports the local pixel scale.
- `fit_wcs.rs` -- fits a celestial WCS from pixel and sky pairs.
- `ascii_table.rs` -- writes and reads an ASCII `TABLE` extension,
  including the `TNULL` sentinel for an undefined numeric cell.
- `compress_image.rs` -- writes a tile-compressed image (`.fz`), reads
  and edits its header without decoding a tile, decompresses it, and
  unpacks the whole file.
- `nalgebra_interop.rs` -- pixels and the WCS as `nalgebra` matrices,
  and back. Needs `--features nalgebra`.
- `faer_interop.rs` -- the same with `faer`. Needs `--features faer`.

## Python examples

Install the package first, with either `pip install fitsy` or
`maturin develop` in this repository. Then run these
from the repository root:

```sh
python examples/python/quickstart.py
python examples/python/reading_images.py
python examples/python/tables.py
python examples/python/writing_files.py
python examples/python/wcs.py
python examples/python/fit_wcs.py
python examples/python/convenience.py
python examples/python/update_mode.py
python examples/python/diff.py
python examples/python/ascii_tables.py
```

- `quickstart.py` -- opens a file, reads its pixels, and uses its WCS.
- `reading_images.py` -- image pixels, the `NAXIS` and numpy axis
  orders, and dtype handling.
- `tables.py` -- a binary table: columns, rows and structured arrays.
- `writing_files.py` -- writes a new file holding an image and a
  binary table.
- `wcs.py` -- pixel and sky transforms on the bundled NGC 2403 image.
- `fit_wcs.py` -- fits a celestial WCS from pixel and sky pairs.
- `convenience.py` -- the module-level functions `getdata`, `getval`,
  `setval`, `delval`, `info` and `append`.
- `update_mode.py` -- edits a file in place: header, pixel patches and
  structural changes.
- `diff.py` -- compares two files with `fitsy.diff`.
- `ascii_tables.py` -- writes and reads an ASCII `TABLE` extension,
  including the `TNULL` sentinel for an undefined numeric cell.

## Sample data

`data/ngc2403.fits.gz` is a 1948 photographic-plate scan of NGC 2403.
It holds one image HDU of 1448 by 2172 pixels with `BITPIX = 16`, and a
`TAN` WCS with SIP distortion. The `read_image` and `wcs` examples read
it, as do the `quickstart.py`, `reading_images.py` and `wcs.py`
scripts.
