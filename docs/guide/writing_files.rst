Writing FITS files
==================

Three builder functions convert numpy / Python data into HDU specs:

* :func:`fitsy.image` -- image HDUs from a numpy array
* :func:`fitsy.bintable` -- BINTABLE from a column dict
* :func:`fitsy.ascii_table` -- ASCII TABLE from a column dict

Hand the resulting list to :func:`fitsy.write`.

.. literalinclude:: ../../examples/python/writing_files.py
   :language: python

By default :func:`fitsy.write` refuses to clobber an existing file.
Pass ``overwrite=True`` to replace it.

Headers
-------

The ``header`` argument to :func:`fitsy.image` accepts a plain
``dict``. Values may be scalars or ``(value, comment)`` tuples.

Pixel scaling on write (``BSCALE`` / ``BZERO``)
-----------------------------------------------

:func:`fitsy.image` writes the supplied numpy array verbatim: the
buffer's dtype determines ``BITPIX`` and the pixel bytes are emitted
without further transformation. ``fitsy`` does **not** invert
``BSCALE`` / ``BZERO`` from physical units back to a raw integer
representation. This matches the behavior of ``astropy.io.fits``.

Two consequences:

* If the input header carries ``BSCALE`` and ``BZERO``, the values
  in the array are interpreted on read as
  ``physical = BZERO + BSCALE * raw``. Writing them back without
  changing the keywords means the new file's "raw" pixels are your
  current array, not the original raw integers.
* To round-trip a scaled integer image (e.g. one that was opened
  with ``hdu.data`` returning floats), drop the ``BSCALE`` /
  ``BZERO`` cards from the new header and write the data as the
  intended dtype, or apply the inverse transform yourself before
  building the HDU.

Unsigned integer images (``uint16``, ``uint32``, ``uint64``) are
the one exception: both :func:`fitsy.image` and the Rust
``ImageBuilder::from_u16`` / ``from_u32`` / ``from_u64``
constructors offset-encode pixels into the matching signed
``BITPIX`` and emit ``BSCALE = 1`` with the standard ``BZERO``
offset automatically (FITS Standard Sec.4.4.2.5). Round-tripping
``uint*`` arrays through fitsy is lossless.

ASCII tables
------------

For text-formatted ``TABLE`` extensions, use
:func:`fitsy.ascii_table`. ``formats`` overrides the per-column
``TFORM``; ``tnulls`` supplies a string sentinel for ``None`` cells
in integer columns.
