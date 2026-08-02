Comparing files
===============

:func:`fitsy.diff` compares two FITS files HDU-by-HDU and returns a
:class:`fitsy.FitsDiff` object that is falsy when the files are
identical and stringifies into a human-readable report.

Tunables:

- ``rtol`` / ``atol`` -- relative and absolute tolerance for every
  floating-point comparison: header card values, image pixels, and
  table cells. Combined as ``|a - b| <= atol + rtol * |b|`` (the
  ``numpy.isclose`` form). Both default to ``0.0``, i.e. exact
  equality. ``rtol`` alone cannot reconcile values straddling zero,
  which is what ``atol`` is for.
- ``max_diffs`` -- cap on per-HDU difference reports.
- ``ignore_keywords`` -- header keywords to skip (e.g. ``"CHECKSUM"``,
  ``"DATASUM"``, ``"DATE"``).

What gets compared
------------------

- Image pixels, numerically and in physical units --
  ``BZERO``/``BSCALE`` applied, ``BLANK`` mapped to NaN. Reported
  indices are pixel numbers and reported values are decoded pixels,
  so two files that store the same physical image with different
  ``BSCALE`` compare equal on data.
- Tile-compressed images, on their decompressed pixels, so
  re-compressing a file with different tile bytes is not a data
  difference.
- Table cells, per column, in decoded (post-``TSCAL``/``TZERO``)
  values. Differences are reported as ``COLUMN[row]``.
- Random-groups HDUs, and HDUs whose ``XTENSION`` fitsy does not
  recognize, are the gap: they have no decoded form and report a
  single byte-level "differs" verdict.

Byte-identical data whose scaling cards also match short-circuits
before any decoding -- the common case for files that match.
Otherwise both data sections are decoded, which costs rather more
than reading the files: image pixels decode to ``float64``, so
comparing two ``BITPIX = 16`` images holds four times the on-disk
size in memory per side.

Example
-------

.. literalinclude:: ../../examples/python/diff.py
   :language: python
