# fitsy examples

Runnable examples for both the Rust crate and the Python bindings.
Every snippet in the user guide ([docs/guide/](../docs/guide/)) is a
`literalinclude` of one of these files, so the docs and the runnable
code never drift apart.

## Layout

| Path                  | Contents                                     |
|-----------------------|----------------------------------------------|
| `*.rs`                | Rust examples (`cargo run --example NAME`)   |
| `python/*.py`         | Parallel Python scripts                      |
| `data/`               | Sample FITS files used by the examples       |

## Rust examples

Run from the repo root:

```sh
cargo run --example read_image
cargo run --example read_table
cargo run --example write_image
cargo run --example write_table
cargo run --example wcs
cargo run --example fit_wcs
```

| Example         | Description                                                |
|-----------------|------------------------------------------------------------|
| `read_image.rs` | Open an image, inspect header, decode pixels               |
| `read_table.rs` | Iterate columns of a binary table and decode cells         |
| `write_image.rs`| Build and write a 2D image FITS file                       |
| `write_table.rs`| Build and write a multi-column binary table                |
| `wcs.rs`        | Pixel <-> sky transforms, batch, pixel scale, header parse |
| `fit_wcs.rs`    | Fit a celestial WCS from pixel/sky correspondences         |

## Python examples

Install the package first (either `pip install fitsy` or
`maturin develop --features python` in this repo), then run from the
repo root:

```sh
python examples/python/quickstart.py
python examples/python/reading_images.py
python examples/python/wcs.py
python examples/python/fit_wcs.py
python examples/python/writing_files.py
```

## Sample data

`data/ngc2403.fits.gz` is a 1948 photographic-plate scan of NGC 2403
(1448 x 2172 pixels, 16-bit, TAN+SIP WCS) used by the WCS, image, and
quickstart examples.
