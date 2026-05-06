Comparing files
===============

:func:`fitsy.diff` compares two FITS files HDU-by-HDU and returns a
:class:`fitsy.FitsDiff` object that is falsy when the files are
identical and stringifies into a human-readable report.

Tunables:

- ``rtol`` / ``atol`` -- floating-point tolerance for numeric data.
- ``max_diffs`` -- cap on per-HDU pixel difference reports.
- ``ignore_keywords`` -- header keywords to skip (e.g. ``"CHECKSUM"``,
  ``"DATASUM"``, ``"DATE"``).

Example
-------

.. literalinclude:: ../../examples/python/diff.py
   :language: python
