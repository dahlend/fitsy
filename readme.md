<div align="center">
<img src="docs/_static/logo.svg" alt="Fitsy" height="48"/>
</div>

----

Read and write astronomical FITS files with WCS coordinates.

A pure-Rust implementation of FITS file I/O and WCS coordinate
transforms. Reads and writes images of all `BITPIX` types, binary
and ASCII tables, and random-groups HDUs; reads `.fz` compressed
images; parses the standard WCS suite plus the SIP, TPV, TNX, and
DSS conventions.

Available as a Python package and a Rust crate.

## Python

```bash
pip install fitsy
```

```python
import fitsy
import numpy as np

# Read
with fitsy.open("image.fits") as f:
    hdu = f[0]                             # ImageHdu
    data = hdu.data                        # full array in RAM, native byte order
    tile = hdu.section[0:256, 0:256]       # decode only this slice (large files)
    wcs = hdu.wcs()
    ra, dec = wcs.pixel_to_celestial(512.0, 512.0)

# Write
img = fitsy.image(np.zeros((512, 512), dtype=np.float32),
                  header={"OBJECT": "test"})
fitsy.write("out.fits", [img])
```

Supports:
- Images of all `BITPIX` types
- Binary and ASCII tables (fixed-width columns)
- Tile-compressed image read (RICE_1, GZIP_1/2, HCOMPRESS_1, PLIO_1)
- Random-groups HDUs
- WCS celestial projections from Paper II + SIP, TPV, TNX, DSS conventions

### Build the wheel from source

```bash
maturin build --release
# or for local development:
maturin develop --features python
```

## Rust

```toml
[dependencies]
fitsy = { version = "0.1", features = ["compression"] }
```

```rust
use fitsy::{FitsFile, Hdu, ImageBuilder, write};

// Read
let file = FitsFile::open("image.fits")?;
let Hdu::Image(img) = file.hdu(0)? else {
    return Err("not an image".into());
};
let data = img.read_physical()?;       // BZERO/BSCALE applied, f64 output
let wcs = file.wcs(0, ' ')?.unwrap();
let (ra, dec) = wcs.pixel_to_celestial(512.0, 512.0)?;

// Write
let pixels = vec![0.0_f32; 512 * 512];
let hdu = ImageBuilder::new(vec![512u64, 512], pixels)?
    .primary(true)
    .card("OBJECT", "test", None)
    .build()?;
write("out.fits", &[hdu], /* overwrite = */ false)?;
```

Optional features: `compression` (default), `nalgebra`, `faer`, `python`.


## License

Apache 2.0 or MIT, at your option.
