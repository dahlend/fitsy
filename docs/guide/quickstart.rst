Quickstart
==========

Most code in this guide is included from a runnable script under
``examples/`` at the repository root, so the page and the script
cannot drift apart. A few short snippets are written inline.

The image used throughout the guide is a 1948 photographic-plate scan
of NGC 2403. It holds 1448 by 2172 pixels and a ``TAN`` WCS with SIP
distortion, and it is bundled at ``examples/data/ngc2403.fits.gz``.

Run any example with:

.. code-block:: shell

   python examples/python/quickstart.py     # Python
   cargo run --example wcs                  # Rust

.. literalinclude:: ../../examples/python/quickstart.py
   :language: python

Writing files and reading tables are covered in their own sections;
see :doc:`writing_files` and :doc:`reading_images`.
