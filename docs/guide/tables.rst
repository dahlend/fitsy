Tables
======

fitsy reads both binary tables (``XTENSION = 'BINTABLE'``) and
ASCII tables (``XTENSION = 'TABLE'``) into the
:class:`fitsy.BinTable` and :class:`fitsy.AsciiTable` types.

Both expose the same access patterns:

- ``len(tbl)`` -- number of rows.
- ``tbl.column_names`` -- list of column names in declared order.
- ``tbl.data`` -- structured :class:`numpy.ndarray` (one record per row).
- ``tbl["COLNAME"]`` -- a 1-D :class:`numpy.ndarray` for that column.
- ``tbl[i]`` -- a row dict ``{colname: value}``.
- ``tbl[a:b]`` -- a list of row dicts.

Column edits do not currently round-trip through ``writeto`` --
table data is re-emitted from the bytes captured at load time.
Use :func:`fitsy.write` with a fresh :class:`fitsy.BinTableBuilder`
to author a new table.

An ASCII integer column reads back as ``float64``, and a cell that
matched ``TNULL`` is ``nan``. The Rust API keeps the two apart:
``AsciiTableHdu::cell_value`` returns ``None`` for a ``TNULL`` match
and an ``AsciiCell::Int`` otherwise.

Examples
--------

Reading a binary table:

.. literalinclude:: ../../examples/python/tables.py
   :language: python

Writing and reading an ASCII table, including the ``TNULL`` sentinel
that marks an undefined numeric cell:

.. literalinclude:: ../../examples/python/ascii_tables.py
   :language: python

The same in Rust:

.. literalinclude:: ../../examples/ascii_table.rs
   :language: rust
