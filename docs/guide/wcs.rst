World coordinates
=================

Python
------

Every :class:`fitsy.ImageHdu` exposes a :meth:`~fitsy.ImageHdu.wcs`
helper that returns a :class:`fitsy.Wcs` instance (or ``None`` if the
header carries no WCS). The same parser is reachable from
:meth:`fitsy.FitsFile.wcs`, which additionally resolves ``-TAB``
look-up axes.

.. literalinclude:: ../../examples/python/wcs.py
   :language: python

Rust
----

``FitsFile::wcs`` resolves ``-TAB`` axes automatically;
``ImageHdu::wcs`` is a lighter-weight alternative when you already
have the HDU in hand and know there are no tabular axes.

.. literalinclude:: ../../examples/wcs.rs
   :language: rust

Pixel coordinate convention
---------------------------

.. important::

   Both the Python and Rust APIs default to **0-based pixel
   coordinates** (numpy / C convention): the center of the first pixel
   is ``0.0``, and the center of the last pixel along ``NAXISn`` is
   ``float(NAXISn) - 1``. A numpy index ``[row, col]`` maps directly
   to ``pixel_to_celestial(col, row)``.

   Pass ``origin=1`` (Python) to use the FITS 1-based convention
   (matching ``CRPIX`` in the header). The Rust API is always 0-based;
   subtract 1 from FITS-native coordinates before calling.

   .. code-block:: python

      # The pixel at numpy index [row, col] = [128, 256]
      ra, dec = wcs.pixel_to_celestial(256.0, 128.0)

Batch transforms
----------------

Use :meth:`~fitsy.Wcs.pixel_to_celestial_many` and
:meth:`~fitsy.Wcs.celestial_to_pixel_many` for ``(N, 2)`` numpy inputs.
The Rust equivalents take ``&[(f64, f64)]`` slices.

Fitting a WCS
-------------

:func:`fitsy.fit_wcs` (Python) and ``fitsy::wcs::fit_celestial_wcs``
(Rust) solve for a celestial WCS given pixel <-> sky correspondences.

.. literalinclude:: ../../examples/python/fit_wcs.py
   :language: python

.. literalinclude:: ../../examples/fit_wcs.rs
   :language: rust

