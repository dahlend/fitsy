Reading FITS files
==================

.. autofunction:: fitsy.open

.. autoclass:: fitsy.FitsFile
   :members:
   :special-members: __len__, __getitem__, __enter__, __exit__

.. autoclass:: fitsy.ImageHdu
   :members:

.. autoclass:: fitsy.BinTable
   :members:
   :special-members: __getitem__

.. autoclass:: fitsy.AsciiTable
   :members:
   :special-members: __getitem__

.. autoclass:: fitsy.RandomGroups
   :members:

Convenience functions
---------------------

.. autofunction:: fitsy.getdata
.. autofunction:: fitsy.getheader
.. autofunction:: fitsy.getval
.. autofunction:: fitsy.info

Comparing files
---------------

.. autofunction:: fitsy.diff

.. autoclass:: fitsy.FitsDiff
   :members:
   :special-members: __bool__, __str__
