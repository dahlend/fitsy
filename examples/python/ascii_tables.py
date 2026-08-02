"""Write and read an ASCII table: fixed-width columns and TNULL.

Run from the repository root:

    python examples/python/ascii_tables.py
"""

import os
import tempfile

import fitsy

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "catalog.fits")

    # `ascii_table` picks a TFORM code per column from the value kind.
    # `formats` overrides that choice. A numeric column that holds an
    # undefined cell needs a matching `tnulls` entry, because Standard
    # Sec.7.2.5 gives a blank numeric field the value zero, so TNULL is
    # the only marker of an undefined value. The sentinel must fit the
    # field width, which is why COUNT declares `I6` rather than taking
    # the narrower automatic width.
    fitsy.write(
        path,
        [
            fitsy.ascii_table(
                {
                    "NAME": ["alpha", "beta", "gamma"],
                    "COUNT": [12, None, 37],
                    "FLUX": [1.5, 2.25, 3.125],
                },
                formats={"COUNT": "I6", "FLUX": "F9.3"},
                tnulls={"COUNT": "---"},
                units={"FLUX": "Jy"},
                extname="CATALOG",
            ),
        ],
    )

    with fitsy.open(path) as f:
        tbl = f["CATALOG"]
        print("columns:", tbl.column_names)
        print("nrows  :", tbl.n_rows)

        # An ASCII integer column reads back as float64, and a cell
        # that matched TNULL is nan. A string cell keeps the padding
        # that its fixed-width field carries.
        counts = tbl.column("COUNT")
        print("COUNT dtype:", counts.dtype, " values:", counts)

        for i in range(tbl.n_rows):
            row = tbl.row(i)
            name, count, flux = row["NAME"], row["COUNT"], row["FLUX"]
            print(f"  {name!r:9} count={count!r:7} flux={flux}")

        print("FLUX column:", tbl.column("FLUX"))
