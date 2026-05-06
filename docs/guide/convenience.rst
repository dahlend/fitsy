Convenience functions
=====================

For one-shot reads and small edits, fitsy mirrors the
``astropy.io.fits`` module-level helpers so existing astropy code
needs no rework:

- :func:`fitsy.getdata` -- pixel array (and optionally header) from a single HDU.
- :func:`fitsy.getheader` -- header of a single HDU.
- :func:`fitsy.getval` / :func:`fitsy.setval` / :func:`fitsy.delval` --
  read, write, or delete a single header card without keeping a handle open.
- :func:`fitsy.info` -- list of ``(index, name, ver, kind, axes)`` tuples.
- :func:`fitsy.append` -- stream a new image HDU onto the end of an
  existing file without rewriting it.

For repeated access to the same file, opening once with
:func:`fitsy.open` is faster -- the helpers each do their own open/close.

Example
-------

.. literalinclude:: ../../examples/python/convenience.py
   :language: python
