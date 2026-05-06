Reading images
==============

The :class:`fitsy.ImageHdu` class wraps an image extension. Pixel
data is returned as :class:`numpy.ndarray` whose shape is the
*reverse* of :attr:`~fitsy.ImageHdu.axes` (numpy is row-major while
FITS lists axes fastest-first).

.. literalinclude:: ../../examples/python/reading_images.py
   :language: python

``data`` is always returned in physical units with an appropriate
numpy dtype, matching the behavior of ``astropy.io.fits``.

.. _reading-large-images:

Working with large images
-------------------------

``hdu.data`` materializes the entire array in memory. fitsy reads
file bytes lazily: :func:`fitsy.open` parses every HDU's header
up front (small, bounded by the FITS 2880-byte block size) but
defers reading each HDU's pixel/table bytes until that HDU is
actually accessed. Opening a 50-GB multi-extension mosaic costs
only the headers; touching ``f[3].data`` reads HDU 3's data section
and nothing else.

The numpy array returned by ``hdu.data`` is always a freshly
allocated, native-byte-order array. FITS stores numeric data
big-endian, so on every modern little-endian host the read path
must byteswap to deliver a native dtype -- and byteswapping
inherently allocates a fresh buffer. fitsy embraces this:
one explicit copy on read, then every downstream operation runs
at full native speed.

For images that do not fit in RAM, or when only a small region is
needed, use :attr:`~fitsy.ImageHdu.section`:

.. code-block:: python

   with fitsy.open("big.fits") as f:
       hdu = f[0]
       tile = hdu.section[100:200, 100:200]   # reads and decodes only this slice

The section reader uses positional ``pread`` to fetch only the
requested rows -- the full image is never materialised. (One
exception: when the header carries non-trivial ``BSCALE`` /
``BZERO`` / ``BLANK`` scaling, the partial-region read falls back
to materialising the full array so the scaling pass can run.
Touching ``hdu.data`` first will preload the array, after which
section reads slice the in-memory copy for free.)

When the file is opened with ``mode='update'``, assigning to a
section writes only the affected pixel bytes back to disk via
positional ``pwrite`` -- no full-image rewrite is performed:

.. code-block:: python

   with fitsy.open("big.fits", mode="update") as f:
       f[0].section[100:200, 100:200] = patch

In-place writes are not crash-safe: a process death mid-patch can
leave the file with some rows updated and others not. This matches
astropy's mmap-backed update path. Callers that need atomicity
should snapshot the file before patching (or use a temp-file +
rename rewrite via ``writeto``).

In-place writes require contiguous slicing (``start:stop`` with
step 1, plus non-negative integer indices) on an HDU with
identity scaling (``BSCALE=1``, ``BZERO=0``, no ``BLANK``).
Anything else (fancy indexing, negative steps, ``Ellipsis``,
boolean masks, or scaled HDUs) raises a ``ValueError`` so that
the caller can either narrow the key or explicitly opt into a
full-file rewrite via ``hdu.data[...] = value``.

Limitations of lazy reads
-------------------------

The lazy-read and in-place-write paths above apply to **plain image
HDUs** (BITPIX-encoded pixel arrays). Other HDU kinds currently
materialise their data section in full on first access:

* **Tile-compressed images** (BINTABLE with ``ZIMAGE = T``,
  unwrapped via the Rust ``FitsFile::image`` accessor) are decompressed
  tile-by-tile into one contiguous numpy array on the first
  ``.data`` access. Section reads and in-place writes are not
  supported for compressed HDUs -- the underlying tiles must be
  re-encoded as a unit, so ``hdu.section[...] = patch`` falls
  through to a full-image rewrite on the next ``flush()``.
* **Random-groups HDUs** hold the entire data section in a single
  numpy structured array; there is no streaming accessor.
* **ASCII tables** and **BINTABLE** HDUs read all rows eagerly
  into the per-HDU data cache. Per-row or per-column streaming
  is not yet implemented; for very wide or very long tables
  consider opening the file, processing one HDU at a time, and
  dropping references to release memory.

If you need partial reads from these formats today, the workaround
is to open the file, materialise the HDU you need, immediately
extract the subset you care about, and let the rest go out of
scope. The Rust ``FitsFile::verify_checksums`` helper streams
each HDU's data section in 1-MiB chunks without populating the
cache, so it is safe to call on arbitrarily large files.
