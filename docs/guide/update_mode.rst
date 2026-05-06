Updating files in place
=======================

When you need to modify an existing FITS file rather than read or
write a fresh one, open it with ``mode="update"``. The handle then
exposes mutable headers, in-place pixel patches, and structural
mutations (append/delete/insert HDUs).

Two persistence paths, with different durability guarantees:

* **Header edits, structural mutations, and ``set_data``** are
  staged in memory and committed on a clean ``__exit__``,
  :meth:`~fitsy.FitsFile.flush`, or :meth:`~fitsy.FitsFile.close`
  via a sibling temp file + atomic ``rename``. A crash mid-flush
  leaves either the old bytes or the new ones -- never a
  half-written file.
* **In-place pixel patches** via ``hdu.section[...] = arr`` write
  only the touched bytes via positional ``pwrite`` and are
  therefore ``O(patch)`` rather than ``O(file)``. They are
  persisted as soon as the assignment returns and require no
  flush, but they are **not** crash-atomic: a process death
  mid-patch can leave some rows updated and others not (matches
  astropy's mmap-backed update path). Snapshot the file first if
  you need atomicity.

Save-as
-------

``writeto(path)`` writes the in-memory state of the handle to a
*different* path. This works from a readonly handle and never
touches the source file -- handy for snapshots or format conversion.
Writing back over the source path requires update mode and is an
alias for ``flush()``.

Example
-------

.. literalinclude:: ../../examples/python/update_mode.py
   :language: python
