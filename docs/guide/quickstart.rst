Quickstart
==========

Every code sample below is a runnable script in the ``examples/``
directory at the repo root. The image used throughout the guide is a
1948 photographic-plate scan of NGC 2403 (1448 x 2172 pixels, TAN+SIP
WCS), bundled at ``examples/data/ngc2403.fits.gz``.

Run any example with:

.. code-block:: shell

   python examples/python/quickstart.py     # Python
   cargo run --example wcs                  # Rust

.. literalinclude:: ../../examples/python/quickstart.py
   :language: python

Writing files and reading tables are covered in their own sections;
see :doc:`writing_files` and :doc:`reading_images`.
