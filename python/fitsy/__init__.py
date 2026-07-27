"""Re-export the compiled extension into the top-level ``fitsy`` namespace.

This shim exists so the package is a real directory on disk, which is
what lets the hand-written type stubs (``__init__.pyi``) and the PEP
561 ``py.typed`` marker ship *inside* the package where type checkers
look for them. Everything callable lives in the Rust extension module
``fitsy.fitsy``; see ``src/python/``.
"""

from .fitsy import *  # noqa: F401,F403
from .fitsy import __all__, __doc__  # noqa: F401
