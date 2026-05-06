Verifying file integrity
========================

FITS files may carry two integrity keywords (Pence & Seaman 1995,
later folded into the FITS Standard):

* ``DATASUM`` -- ASCII-decimal 1's-complement sum of the padded data
  bytes for the HDU.
* ``CHECKSUM`` -- 16-character ASCII-armoured 1's-complement sum of
  the entire HDU (header + data) computed *with* the ``CHECKSUM``
  card present in the header. A correctly stamped HDU sums to all
  ones.

``fitsy`` does not verify these keywords automatically on
:func:`fitsy.open`. To check them explicitly, use the Rust
``FitsFile::verify_checksums`` method:

.. code-block:: rust

   use fitsy::FitsFile;

   let f = FitsFile::open("image.fits")?;
   for (i, report) in f.verify_checksums()?.iter().enumerate() {
       println!(
           "HDU {i}: checksum_ok = {:?}, datasum_ok = {:?}",
           report.checksum_ok, report.datasum_ok,
       );
   }

Each ``ChecksumReport`` carries ``checksum_ok`` and ``datasum_ok``
as ``Option<bool>``. ``None`` means the keyword was absent (no
verdict possible); ``Some(true)`` / ``Some(false)`` indicates a
successful or failed check.

From Python the same check is exposed as
:py:meth:`fitsy.FitsFile.verify_checksums`, which streams the data
section of each HDU from disk (no full materialisation) and
returns a list of ``dict``\ s, one per HDU:

.. code-block:: python

   import fitsy
   with fitsy.open("image.fits") as f:
       for report in f.verify_checksums():
           # {'hdu': 0, 'checksum_ok': True, 'datasum_ok': True}
           print(report)

``checksum_ok`` and ``datasum_ok`` are ``True`` / ``False`` /
``None`` (mirroring the Rust ``Option<bool>``).

Stamping new files
------------------

To embed ``CHECKSUM`` / ``DATASUM`` cards in files you write, opt
in at write time. Both Python entry points are supported:

.. code-block:: python

   import fitsy

   # One-shot writes:
   fitsy.write("out.fits", [fitsy.image(arr)], checksums=True)

   # In-place edits via the FitsFile handle:
   with fitsy.open("in.fits", mode="update") as f:
       f[0].header["OBSERVER"] = "you"
       f.add_checksums()       # stamp on the next flush / writeto
       f.flush()

The flag is sticky for the lifetime of the ``FitsFile``: once
:meth:`~fitsy.FitsFile.add_checksums` is called, every subsequent
write through that handle stamps fresh values. Computation runs
against the final byte layout of each HDU, so the values are
guaranteed self-consistent regardless of any intervening header or
pixel edits.

The Rust equivalents are
``FitsWriter::with_checksums()`` (chained on the writer
constructor) and ``fitsy::checksum::stamp_checksum`` (low-level,
operates on already-serialized header + data bytes).
