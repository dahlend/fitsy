World coordinates
=================

Python
------

Every :class:`fitsy.ImageHdu` exposes a :meth:`~fitsy.ImageHdu.wcs`
helper that returns a :class:`fitsy.Wcs` instance (or ``None`` if the
header carries no WCS). The same parser is reachable from
:meth:`fitsy.FitsFile.wcs`, which additionally resolves ``-TAB``
look-up axes -- their coordinate array lives in a separate BINTABLE,
which a header alone cannot reach, so ``ImageHdu.wcs()`` leaves such
an axis unresolved and using it raises.

A ``-TAB`` axis is defined over its table plus half a sample step at
each end (FITS Paper III Sec.6.1.2, covering the outer halves of the
boundary pixels). A pixel beyond that range has no coordinate. A
single-point call raises there, and a batch method returns ``nan``.

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

Most projections cover only part of the plane -- SIN's unit circle,
ZPN below ``PV2_0``, AZP beyond the horizon -- so a wide field
routinely mixes valid and invalid pixels. A batch method puts
``nan`` in those slots and returns everything else. Mask the result
with ``numpy.isfinite`` to drop them:

.. code-block:: python

   sky = wcs.pixel_to_celestial_many(pixels)
   good = numpy.isfinite(sky).all(axis=1)

They raise only when the *whole* WCS cannot transform: no celestial
axis pair, or unresolved ``-TAB`` axes. The single-point
:meth:`~fitsy.Wcs.pixel_to_celestial` and
:meth:`~fitsy.Wcs.celestial_to_pixel` still raise for an
out-of-domain coordinate, which is where to go for a diagnostic
message explaining *why* a point failed.

Image extent and footprint
--------------------------

A WCS parsed from an image header also records that image's size as
:attr:`~fitsy.Wcs.pixel_shape` (FITS axis order, ``NAXIS1`` first),
which :meth:`~fitsy.Wcs.footprint` uses to return the sky positions of
the four corner pixels.

.. code-block:: python

   with fitsy.open("image.fits") as f:
       wcs = f[0].wcs()
       print(wcs.pixel_shape)   # e.g. (1448, 2172)
       print(wcs.footprint())   # (4, 2) array of (ra, dec)

``pixel_shape`` is a snapshot of the ``NAXISn`` cards, not part of the
coordinate description. It is ``None`` for a WCS from
:func:`fitsy.fit_wcs`, since no image exists; no transform consults
it; and nothing revalidates it, so after cropping or rebinning it
still describes the original image.

Fitting a WCS
-------------

:func:`fitsy.fit_wcs` (Python) and ``fitsy::wcs::fit_celestial_wcs``
(Rust) solve for a celestial WCS given pixel <-> sky correspondences.

Use :meth:`~fitsy.Wcs.to_header` to turn the result -- or any parsed
WCS -- back into a :class:`fitsy.Header` you can merge into an HDU.
It writes everything the reader understands, so parsing the output
reproduces the original transform: the linear pipeline, ``LONPOLE`` /
``LATPOLE`` and the projection's ``PVi_m`` parameters, SIP, TPV,
TNX/ZPX, DSS plate solutions, spectral rest quantities, and ``-TAB``
pointer cards.

Two things a bare header cannot carry: ``NAXISn`` (emitted as zero
placeholders, since a WCS has no image attached) and the BINTABLE a
``-TAB`` axis points at, which must be written as its own extension.

.. literalinclude:: ../../examples/python/fit_wcs.py
   :language: python

.. literalinclude:: ../../examples/fit_wcs.rs
   :language: rust

