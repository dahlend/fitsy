Installation
============

``fitsy`` is built from a Rust crate using `maturin
<https://www.maturin.rs/>`_.

From PyPI
---------

.. code-block:: console

   $ pip install fitsy

Wheels are published for CPython 3.10+ on Linux (x86_64, aarch64),
macOS (x86_64, arm64), and Windows (x86_64). Installing from a
wheel does not require a Rust toolchain.

From source
-----------

Requires a recent stable Rust toolchain (``rustup install stable``)
and Python 3.10+.

.. code-block:: console

   $ pip install maturin
   $ git clone https://github.com/ddahlen/fitsy
   $ cd fitsy
   $ maturin develop --release --features python

To build a wheel instead of installing in-place:

.. code-block:: console

   $ maturin build --release --features python
   $ pip install target/wheels/fitsy-*.whl

Documentation extras
--------------------

To build these docs locally:

.. code-block:: console

   $ pip install -e '.[docs]'
   $ sphinx-build -b html docs docs/_build/html
