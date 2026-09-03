//! Build, write and read back an ASCII table extension.
//!
//! This shows `AsciiTableBuilder` declaring fixed-width columns, the
//! `TNULL` sentinel that marks an undefined integer cell, and the
//! typed cells that come back on the read side.
//!
//! Run from the repository root:
//!
//!     cargo run --example ascii_table

use fitsy::hdu::builder::AsciiColumnData;
use fitsy::{AsciiCell, AsciiFormat, AsciiTableBuilder, FitsFile, FitsWriter, Hdu, ImageBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("fitsy_example_ascii_table.fits");

    // A file needs a primary HDU. An ASCII table is an extension, so
    // write an empty image first.
    let primary = ImageBuilder::new(Vec::<u64>::new(), Vec::<f32>::new())?
        .primary(true)
        .build()?;

    // Each column declares its own fixed field width through its
    // `TFORMn` code. `A8` holds eight characters, `I6` holds a
    // six-column integer, and `F9.3` holds a fixed-point real with
    // three decimal places.
    let mut b = AsciiTableBuilder::new();
    b.add_column(
        "NAME",
        AsciiFormat::A(8),
        AsciiColumnData::Str(vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
        ]),
    )?;
    b.add_column(
        "COUNT",
        AsciiFormat::I(6),
        // The middle cell is undefined, so this column needs a TNULL.
        AsciiColumnData::Int(vec![Some(12), None, Some(37)]),
    )?;
    // TNULL is the only way an ASCII table marks a value undefined:
    // Standard Sec.7.2.5 gives a blank numeric field the value zero.
    b.tnull("---")?;
    b.add_column(
        "FLUX",
        AsciiFormat::F(9, 3),
        AsciiColumnData::Float(vec![1.5, 2.25, 3.125]),
    )?;
    b.unit("Jy")?;
    b.extname("CATALOG");
    let table = b.build()?;

    let mut out = std::fs::File::create(&path)?;
    let mut w = FitsWriter::new(&mut out);
    w.write_hdu(&primary)?;
    w.write_hdu(&table)?;
    w.finish()?;

    // Read it back.
    let f = FitsFile::open(&path)?;
    let Hdu::AsciiTable(tbl) = f.hdu_by_name("CATALOG", None)? else {
        return Err("CATALOG is not an ASCII table".into());
    };

    println!(
        "rows: {}  row width: {} bytes",
        tbl.n_rows(),
        tbl.row_size()
    );
    for col in tbl.columns() {
        println!(
            "  col {} {:8} TFORM={:?} TBCOL={} unit={:?}",
            col.index, col.name, col.format, col.start, col.unit
        );
    }

    // `cell_value` returns `Ok(None)` for a cell that matches TNULL.
    let count = tbl.column_by_name("COUNT").ok_or("no COUNT column")?;
    for row in 0..tbl.n_rows() {
        match tbl.cell_value(row, count)? {
            Some(AsciiCell::Int(v)) => println!("  COUNT[{row}] = {v}"),
            Some(other) => println!("  COUNT[{row}] = {other:?}"),
            None => println!("  COUNT[{row}] = undefined (matched TNULL)"),
        }
    }

    std::fs::remove_file(&path)?;
    Ok(())
}
