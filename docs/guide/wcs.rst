World coordinates
=================

A World Coordinate System turns a pixel position into a world
coordinate, and back. The common case is a celestial image: pixel to
right ascension and declination.

Getting a WCS
-------------

:meth:`fitsy.ImageHdu.wcs` and :meth:`fitsy.FitsFile.wcs` return the
same :class:`fitsy.Wcs`, so ``f[i].wcs()`` and ``f.wcs(i)`` are
equivalent. Both return ``None`` when the header carries no WCS.

.. code-block:: python

   with fitsy.open("image.fits") as f:
       wcs = f[0].wcs()
       ra, dec = wcs.pixel_to_world([256.0, 128.0])
       px, py = wcs.world_to_pixel([ra, dec])

A single point goes in as a length-``naxis`` sequence and comes back
as a list. ``print(wcs)`` summarizes the keywords the description
uses.

fitsy applies the distortion a header declares, on every transform,
with no extra call: SIP (``RA---TAN-SIP``), TPV, TNX/ZPX, and DSS
plate solutions. The bundled NGC 2403 scan carries ``TAN`` with SIP,
so the example below exercises that path.

.. literalinclude:: ../../examples/python/wcs.py
   :language: python

Rust
----

``FitsFile::wcs`` takes the HDU index and the alternate-description
letter. ``Wcs::from_header`` parses a header you already hold.

.. literalinclude:: ../../examples/wcs.rs
   :language: rust

Pixel coordinate convention
---------------------------

.. important::

   Both the Python and Rust APIs default to **0-based pixel
   coordinates** (numpy / C convention): the center of the first pixel
   is ``0.0``, and the center of the last pixel along ``NAXISn`` is
   ``float(NAXISn) - 1``. A numpy index ``[row, col]`` maps directly
   to ``pixel_to_world([col, row])``.

   Pass ``origin=1`` (Python) to use the FITS 1-based convention
   (matching ``CRPIX`` in the header). The Rust API is always 0-based;
   subtract 1 from FITS-native coordinates before calling.

   .. code-block:: python

      # The pixel at numpy index [row, col] = [128, 256]
      ra, dec = wcs.pixel_to_world([256.0, 128.0])

Batch transforms
----------------

:meth:`~fitsy.Wcs.pixel_to_world` and
:meth:`~fitsy.Wcs.world_to_pixel` take one point or many. A
length-``naxis`` sequence is one point and returns a list. An
``(N, naxis)`` array is ``N`` points and returns an ``(N, naxis)``
array. One path serves every axis kind, celestial or not:

.. code-block:: python

   sky = wcs.pixel_to_world(np.array([[31.0, 23.0], [10.0, 40.0]]))

The column count must be exactly ``naxis``. An ``(naxis, N)`` array is
the transpose of a batch, not a batch of ``N`` points. Such an array
raises a ``ValueError`` instead of pairing the wrong values together.
Pass ``pixels.T`` when your points run down the columns.

In Rust the batch entry points are ``Wcs::pixel_to_world_many`` and
``Wcs::world_to_pixel_many``, which take the points flat --
``NAXIS`` values per point, end to end -- and return the same
layout. They see no shape. They reject only a length that is not a
whole multiple of ``NAXIS``.

Prefer a batch call over a loop. In Python the gain is about fortyfold,
because one call crosses the language boundary once and converts its
arguments once. In Rust it is two to seven times, from building the
working buffers once for the whole call rather than per point; the
cheaper the projection, the more that saving is worth.

Most projections cover only part of the plane -- SIN's unit circle,
ZPN below ``PV2_0``, AZP beyond the horizon -- so a wide field
routinely mixes valid and invalid pixels. A batch method puts
``nan`` in those slots and returns everything else. Mask the result
with ``numpy.isfinite`` to drop them:

.. code-block:: python

   sky = wcs.pixel_to_world(pixels)
   good = numpy.isfinite(sky).all(axis=1)

A batch call raises only when the *whole* WCS cannot transform: a
malformed point count, or an unresolved lookup table (see
`Lookup-table axes`_). Passing a single point instead raises for an
out-of-domain coordinate, which is where to go for a diagnostic
message explaining *why* that point failed.

Celestial images
----------------

:attr:`~fitsy.Wcs.is_celestial` reports whether the description
declares a longitude and latitude pair, and
:meth:`~fitsy.Wcs.celestial_axes` gives their zero-based indices.
A plain sky image puts them at ``(0, 1)``:

.. code-block:: python

   wcs.is_celestial          # True
   wcs.celestial_axes()      # (0, 1)

:meth:`~fitsy.Wcs.pixel_scale_at` measures the local scale in
arcseconds per pixel, one value per celestial axis:

.. code-block:: python

   sx, sy = wcs.pixel_scale_at(724.0, 1086.0)   # (0.9725, 0.9731)

fitsy measures this by finite difference on the sphere, so the result
carries the projection distortion and any local skew at that pixel. It
is a great-circle distance, always positive, not the signed ``CDELT``
value. An image with flipped ``RA`` still reports a positive scale.
The scale of a wide field varies across the image, so measure it where
you need it. The call raises when the WCS declares no celestial pair.

Image extent and footprint
--------------------------

A WCS parsed from an image header also records that image's size as
:attr:`~fitsy.Wcs.pixel_shape` (FITS axis order, ``NAXIS1`` first),
which :meth:`~fitsy.Wcs.footprint` uses to return the world positions of
the corner pixels.

.. code-block:: python

   with fitsy.open("image.fits") as f:
       wcs = f[0].wcs()
       print(wcs.pixel_shape)   # e.g. (1448, 2172)
       print(wcs.footprint())   # (4, 2) array of (ra, dec)

The shape is ``(2**k, naxis)``, where ``k`` is the number of axes
``pixel_shape`` covers. That is ``naxis`` for a normal image. A
two-axis image gives the familiar four corners; a three-axis cube gives
eight, covering the spectral or time axis as well. Corners come back in
Gray-code order, so consecutive corners differ on one axis alone, and a
two-axis image walks counter-clockwise from the origin and closes the
ring.

These are corners, not an axis-aligned bounding box. A rotated image has
corners outside the box its own minimum and maximum describe, and an
``RA`` axis crossing zero makes such a box meaningless -- a one-degree
field straddling the wrap reports ``RA`` from ``0.18`` to ``359.81``.
Use the corner polygon, or take minima and maxima yourself on axes where
you know neither hazard applies.

Corners go through the batch transform, so they follow its ``nan``
rule. A corner outside the projection's domain comes back as a row of
``nan`` rather than raising. A wide-field ``SIN`` or ``AZP`` image can
put every corner outside that domain. Test the result with
``numpy.isfinite``. Pass one corner to
:meth:`~fitsy.Wcs.pixel_to_world` to read the reason it failed.

``WCSAXES`` may exceed ``NAXIS``. A coordinate axis past the end of
``pixel_shape`` then has no length to take a corner from. That axis
holds its reference pixel for every corner. The corner count follows
the image, and every corner still carries a full ``naxis``-value world
vector.

``pixel_shape`` is a snapshot of the ``NAXISn`` cards, not part of the
coordinate description. It is ``None`` for a WCS from
:func:`fitsy.fit_wcs`, since no image exists; no transform consults
it; and nothing revalidates it, so after cropping or rebinning it
still describes the original image.

Alternate descriptions
----------------------

A header may carry up to 26 more descriptions of the same pixels,
each tagged with a letter from ``A`` to ``Z`` (Standard Sec.8.2).
A survey image often uses one to publish a second astrometric
solution. Pass the letter to select it; ``' '`` (the default) selects
the primary description:

.. code-block:: python

   wcs = f[0].wcs("A")       # or f.wcs(0, "A")

The result is ``None`` when the header carries no description for
that letter, which is how to test for one.

Finding an axis by meaning
--------------------------

:meth:`~fitsy.Wcs.pixel_to_world` returns one value per axis, in axis
order. :meth:`~fitsy.Wcs.axis_kinds` says what each of those values is,
so a caller locates an axis by what it carries rather than by where it
sits:

.. code-block:: python

   with fitsy.open("cube.fits") as f:
       wcs = f[0].wcs()
       kinds = wcs.axis_kinds()      # ['longitude', 'latitude', 'spectral']
       world = wcs.pixel_to_world([31.0, 23.0, 4.0])
       freq = world[kinds.index("spectral")]

Each entry is one of ``'longitude'``, ``'latitude'``, ``'spectral'``,
``'time'``, ``'phase'``, ``'stokes'`` or ``'linear'``. Reading by
meaning matters because axis order is not fixed: FITS permits
``CTYPE1 = 'DEC--TAN'`` with ``CTYPE2 = 'RA---TAN'``, and real archives
hold such headers. On one, ``axis_kinds()`` reports
``['latitude', 'longitude']``, while indexing position 0 for right
ascension returns declination.

The kind names the type half of ``CTYPEia``, the part before the
algorithm code. ``RA---TAN`` and ``RA---TAB`` are both ``'longitude'``.

Fitting a WCS
-------------

:func:`fitsy.fit_wcs` (Python) and ``fitsy::wcs::fit_celestial_wcs``
(Rust) solve for a celestial WCS given pixel <-> sky correspondences.

Use :meth:`~fitsy.Wcs.to_header` to turn the result -- or any parsed
WCS -- back into a :class:`fitsy.Header` you can merge into an HDU.
It writes everything the reader understands, so parsing the output
reproduces the original transform: the linear pipeline, ``LONPOLE`` /
``LATPOLE`` and the projection's ``PVi_m`` parameters, SIP, TPV,
TNX/ZPX, DSS plate solutions, and spectral rest quantities.

One thing a bare header cannot carry is ``NAXISn``, emitted as zero
placeholders, since a WCS has no image attached.

.. literalinclude:: ../../examples/python/fit_wcs.py
   :language: python

.. literalinclude:: ../../examples/fit_wcs.rs
   :language: rust

Lookup-table axes
-----------------

A ``-TAB`` axis reads its coordinates from an array in a separate
BINTABLE rather than from a formula (FITS Paper III Sec.6). Nothing
above changes for one, except where the table comes from.

Both :meth:`fitsy.ImageHdu.wcs` and :meth:`fitsy.FitsFile.wcs` load
the table from the file the HDU was opened from, so a ``-TAB`` axis
transforms like any other. An HDU or ``FitsFile`` built in memory has
no file to search, and raises at the ``wcs()`` call.
``fitsy.Wcs(header)`` parses a header alone, without the table; its
transforms raise on the unresolved axis. In Rust, ``FitsFile::wcs``
and ``FitsFile::wcs_inherited`` load the table, ``Wcs::from_header``
does not, and ``Wcs::resolve_tab`` loads it after the fact.

:meth:`~fitsy.Wcs.is_tabular` reports whether a given axis takes this
path.

A ``-TAB`` axis is defined over its table plus half a sample step at
each end (Paper III Sec.6.1.2, covering the outer halves of the
boundary pixels). A pixel beyond that range has no coordinate. A
single-point call raises there, and a batch method returns ``nan``.

:meth:`~fitsy.Wcs.to_header` writes the ``PSi_m`` / ``PVi_m`` pointer
cards, but not the table they name. Write that BINTABLE as its own
extension.
