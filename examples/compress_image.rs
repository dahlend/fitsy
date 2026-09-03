//! Write a tile-compressed image (`.fz`), read it, and unpack it.
//!
//! A compressed image is a `BINTABLE` carrying `ZIMAGE = T`, one row
//! per tile. `write_hdu_compressed` builds that table from an ordinary
//! image header and its pixels.
//!
//! Reading takes two forms, and the cheap one comes first. The header
//! of a compressed image is recovered from its `Z` keywords, so
//! `FitsFile::image_header` answers without decoding a tile. Only
//! `FitsFile::image` decompresses, and it returns the same `ImageHdu`
//! an uncompressed HDU gives.
//!
//! The file this writes is the same thing `fitsy fpack` produces, and
//! `funpack` reads it.
//!
//! Run from the repository root:
//!
//!     cargo run --example compress_image

use fitsy::{Codec, FitsFile, FitsWriter, Hdu, Header, ImageBuilder, TileOpts, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("fitsy_example.fits.fz");

    // A 256x256 frame with structure, so compression has something to
    // work with. Random noise would barely compress at all.
    let (nx, ny) = (256_u64, 256_u64);
    let mut pixels: Vec<i16> = Vec::with_capacity((nx * ny) as usize);
    for y in 0..ny {
        for x in 0..nx {
            let r = ((x as f64 - 128.0).powi(2) + (y as f64 - 128.0).powi(2)).sqrt();
            pixels.push((1000.0 * (-r / 60.0).exp()) as i16 + (x % 7) as i16);
        }
    }

    // Build the image exactly as an uncompressed one, header and all.
    // Compression carries these cards through to the table.
    let src = ImageBuilder::new(vec![nx, ny], pixels.clone())?
        .primary(true)
        .card("OBJECT", Value::from("synthetic source"), None)
        .card("BUNIT", Value::from("counts"), None)
        .build()?;

    // RICE_1 on integers, one row of tiles at a time. `TileOpts::new()`
    // would give GZIP_1 with the default tile shape.
    let opts = TileOpts::new()
        .codec(Codec::Rice1 { blocksize: 32 })
        .tile(vec![nx, 1]);

    // A compressed image is an extension, so the file needs a primary
    // HDU ahead of it. `write_hdu_compressed` writes that stub itself.
    let mut out = std::fs::File::create(&path)?;
    let mut w = FitsWriter::new(&mut out);
    w.write_hdu_compressed(&src, &opts)?;
    w.finish()?;

    let on_disk = std::fs::metadata(&path)?.len();
    println!(
        "wrote {} ({on_disk} bytes for {} bytes of pixels)",
        path.display(),
        pixels.len() * 2
    );

    let f = FitsFile::open(&path)?;

    // Look at the file without decoding anything. Both calls read the
    // header bytes that `open` already loaded, so this costs the same
    // for a 4-byte image and a 4-gigabyte one.
    for i in 0..f.len() {
        println!("HDU {i}: {}", f.kind(i)?);
    }
    let head = f.image_header(1)?;
    println!("  BITPIX={} axes={:?}", head.bitpix()?, head.axes()?);
    if let Some(Value::String(object)) = head.first("OBJECT") {
        println!("  OBJECT={object}");
    }

    // Editing a card needs no decoding either. Every card the
    // convention does not own is stored in the table header as it
    // stands, so rewrite that header with the tiles it already has.
    let Hdu::CompressedImage(cz) = f.hdu(1)? else {
        return Err("HDU 1 is not a compressed image".into());
    };
    println!(
        "tiles of {:?}, was a primary array: {}",
        cz.tile_shape(),
        cz.was_primary()
    );
    let mut edited = cz.as_bintable().header().clone();
    edited.set("OBJECT", Value::from("renamed source"), None)?;
    let mut buf: Vec<u8> = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu_parts(&Header::empty_primary(), &[])?;
    w.write_hdu_parts(&edited, cz.as_bintable().data_bytes())?;
    w.finish()?;
    if let Some(Value::String(object)) = FitsFile::from_bytes(buf)?.image_header(1)?.first("OBJECT")
    {
        println!("edited OBJECT={object}, and no tile was re-encoded");
    }

    // Now decompress. `image` returns the same `ImageHdu` type a plain
    // image HDU gives, so the pixels come out through the ordinary
    // accessor with no byte-order step.
    let img = f.image(1)?;
    let got = img.read_raw::<i16>()?;
    println!("pixels identical: {}", got.as_slice() == pixels.as_slice());

    // Unpack the whole file. This drops the stub and puts the image
    // back in the primary slot, which is what `fitsy funpack` runs.
    let unpacked = std::env::temp_dir().join("fitsy_example.fits");
    let n = f.write_decompressed(&unpacked, true, true)?;
    println!("unpacked {n} HDU to {}", unpacked.display());
    println!("  HDU 0 is now {}", FitsFile::open(&unpacked)?.kind(0)?);

    Ok(())
}
