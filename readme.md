<div align="center">
<img src="docs/_static/logo.svg" alt="Fitsy" height="48"/>
</div>

----

Read and write astronomical FITS files with WCS coordinates.

- Reads and writes fits files of all image types, and tables (bin and ascii).
- Query subsets of images without loading the whole thing.
- Reads `.fz` compressed images.
- Parses all WCS Projections, including spectral (if we are missing one let me know!)
- SIP, TPV, TNX, and DSS distortion support.
- Support for fitting WCS as well.
- Hierarchy/History support.

Available as a Python package, a Rust crate with minimal dependencies, and a
command-line tool.

Full documentation: <https://dahlend.github.io/fitsy/>

I have tried my best to make this fully compliant with modern fits requirements, if
something is missing please let me know.

## Memory Mapping

The fits standard defines data using big-endian values, all modern computers are little
endian. What this means is that when you load data from a fits file, the moment you do
anything with it (even plot it) your computer has to flip the endian-ness of the data.
This means putting it into memory. As a result of this, memory mapping is pretty much
useless in practice.

Because of this, Fitsy has optimized loading subsections of data from images into memory
instead of memory mapping, this includes editing in place.

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
    ra, dec = wcs.pixel_to_world([512.0, 512.0])

# Write
img = fitsy.image(np.zeros((512, 512), dtype=np.float32),
                  header={"OBJECT": "test"})
fitsy.write("out.fits", [img])
```

### Build the wheel from source

```bash
maturin build --release
# or for local development:
maturin develop
```

## Rust

```toml
[dependencies]
fitsy = { version = "0.3", features = ["compression"] }
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
let world = wcs.pixel_to_world(&[512.0, 512.0])?;

// Write
let pixels = vec![0.0_f32; 512 * 512];
let hdu = ImageBuilder::new(vec![512u64, 512], pixels)?
    .primary(true)
    .card("OBJECT", "test", None)
    .build()?;
write("out.fits", &[hdu], /* overwrite = */ false)?;
```

Optional features: `compression` (default), `nalgebra`, `faer`, `python`.

## Command line

The crate also installs a `fitsy` binary:

```bash
cargo install fitsy
```

```
fitsy info <file>                        # one line per HDU, plus WCS details
fitsy header <file> [--hdu N] [filter]   # dump parsed header cards
fitsy checksum <file>                    # verify CHECKSUM / DATASUM
fitsy stats <file> [--hdu N]             # pixel statistics for image HDUs
fitsy funpack <input> [-o out]           # write a tile-decompressed copy
```

## License

Apache 2.0 or MIT, at your option.
