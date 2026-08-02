Convenience functions
=====================

For a one-shot read or a small edit, fitsy offers module-level
helpers. Each one opens the file, does its work, and closes it
again:

- :func:`fitsy.getdata` -- pixel array from one HDU, and its header
  as well when ``header=True``.
- :func:`fitsy.getheader` -- header of a single HDU.
- :func:`fitsy.getval` / :func:`fitsy.setval` / :func:`fitsy.delval` --
  read, write, or delete a single header card without keeping a handle open.
- :func:`fitsy.info` -- list of ``(index, name, ver, kind, dims)``
  tuples. ``dims`` is the axis list for an image and the row count for
  a table.
- :func:`fitsy.append` -- stream a new image HDU onto the end of an
  existing file without rewriting it.

For repeated access to one file, open it once with
:func:`fitsy.open`. That is faster, because each helper above does its
own open and close.

Example
-------

.. literalinclude:: ../../examples/python/convenience.py
   :language: python
