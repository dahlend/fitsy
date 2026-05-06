fitsy
=====

``fitsy`` is a FITS reader and writer written in Rust, with a Python interface.

The Python API is intentionally narrow: open a file, walk its HDUs,
read pixels and tables as :class:`numpy.ndarray`, and write new files
from numpy arrays. Compute positions with WCS.

.. toctree::
   :maxdepth: 1
   :caption: User guide

   guide/install
   guide/quickstart
   guide/reading_images
   guide/writing_files
   guide/update_mode
   guide/tables
   guide/wcs
   guide/diff
   guide/verifying
   guide/convenience

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/reading
   api/headers
   api/writing
   api/wcs
   api/errors
