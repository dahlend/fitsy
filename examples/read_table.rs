//! Read a binary table: column metadata and per-cell decoding.
//!
//! Run from the repo root:
//!
//!     cargo run --example read_table
//!
//! For self-containment the example first writes a small two-column
//! BINTABLE to a temp file, then reads it back. See
//! `examples/write_table.rs` for the writer side in isolation.

use fitsy::{BinFieldKind, BinTableBuilder, BinValue, FitsFile, FitsWriter, Hdu, ImageBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("fitsy_example_table.fits");

    // Build + write a small catalog table.
    write_demo_table(&path)?;

    // Read it back.
    let f = FitsFile::open(&path)?;
    let Hdu::BinTable(t) = f.hdu(1)? else {
        return Err("HDU 1 is not a binary table".into());
    };

    println!("rows: {}", t.n_rows());
    println!("columns:");
    for col in t.columns() {
        println!(
            "  {:>3}  {:<8}  {:?} x {}  unit={:?}",
            col.index, col.name, col.format.kind, col.format.repeat, col.unit
        );
    }

    // Decode cells of one column by name.
    let id_col = t.column_by_name("ID").ok_or("missing ID column")?.clone();
    let mag_col = t.column_by_name("MAG").ok_or("missing MAG column")?.clone();
    println!("\nrow  ID    MAG");
    for row in 0..t.n_rows() {
        let id = match t.cell_value(row, &id_col)? {
            BinValue::Int(v) => v[0].unwrap_or(0),
            other => return Err(format!("unexpected ID kind: {other:?}").into()),
        };
        let mag = match t.cell_value(row, &mag_col)? {
            BinValue::F32(v) => f64::from(v[0]),
            other => return Err(format!("unexpected MAG kind: {other:?}").into()),
        };
        println!("{row:>3}  {id:>4}  {mag:>5.2}");
    }

    Ok(())
}

fn write_demo_table(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    // Empty primary HDU (every FITS file needs one).
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())?
        .primary(true)
        .build()?;

    // Two scalar columns: ID (J = i32), MAG (E = f32).
    let mut bt = BinTableBuilder::new();
    bt.add_column("ID", BinFieldKind::I32, 1, None, None)?;
    bt.add_column("MAG", BinFieldKind::F32, 1, Some("mag"), None)?;
    bt.extname("CATALOG");

    // Pack 5 rows in big-endian, column-by-column within each row.
    let mut rows = Vec::new();
    for (i, mag) in (1_i32..=5).zip([18.4_f32, 17.9, 19.1, 16.2, 20.5]) {
        rows.extend_from_slice(&i.to_be_bytes());
        rows.extend_from_slice(&mag.to_be_bytes());
    }
    let (h, data) = bt.build(5, rows)?;

    let mut out = std::fs::File::create(path)?;
    let mut w = FitsWriter::new(&mut out);
    w.write_hdu(&primary.0, &primary.1)?;
    w.write_hdu(&h, &data)?;
    w.finish()?;
    Ok(())
}
