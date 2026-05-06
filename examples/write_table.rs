//! Build and write a binary table HDU.
//!
//! Run from the repo root:
//!
//!     cargo run --example write_table

use fitsy::{BinFieldKind, BinTableBuilder, FitsWriter, ImageBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("fitsy_example_write_table.fits");

    // Empty primary HDU.
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())?
        .primary(true)
        .build()?;

    // Three columns: ID (J), RA/DEC (D), NAME (8A).
    let mut bt = BinTableBuilder::new();
    bt.add_column("ID", BinFieldKind::I32, 1, None, None)?;
    bt.add_column("RA", BinFieldKind::F64, 1, Some("deg"), None)?;
    bt.add_column("DEC", BinFieldKind::F64, 1, Some("deg"), None)?;
    bt.add_column("NAME", BinFieldKind::Char, 8, None, None)?;
    bt.extname("OBJECTS");

    // Pack rows column-by-column within each row, big-endian.
    let rows_in: &[(i32, f64, f64, &str)] = &[
        (1, 10.684, 41.269, "M31"),
        (2, 23.462, 30.660, "M33"),
        (3, 114.209, 65.592, "NGC2403"),
    ];
    let mut row_bytes = Vec::new();
    for (id, ra, dec, name) in rows_in {
        row_bytes.extend_from_slice(&id.to_be_bytes());
        row_bytes.extend_from_slice(&ra.to_be_bytes());
        row_bytes.extend_from_slice(&dec.to_be_bytes());
        // Char column: NUL-pad (or space-pad) to declared width.
        let mut buf = [b' '; 8];
        let n = name.len().min(8);
        buf[..n].copy_from_slice(&name.as_bytes()[..n]);
        row_bytes.extend_from_slice(&buf);
    }
    let (h, data) = bt.build(rows_in.len(), row_bytes)?;

    let mut out = std::fs::File::create(&path)?;
    let mut w = FitsWriter::new(&mut out);
    w.write_hdu(&primary.0, &primary.1)?;
    w.write_hdu(&h, &data)?;
    w.finish()?;

    println!("wrote {} ({} rows)", path.display(), rows_in.len());
    Ok(())
}
