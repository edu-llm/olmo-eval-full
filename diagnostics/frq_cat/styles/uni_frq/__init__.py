"""The ``uni_frq`` CAT style: unidimensional (2PL) free-response adaptive testing.

Registers a single style over TutorEval's graduated unidimensional item bank. The
registry discovers styles by scanning ``styles/`` and importing each subpackage, so
adding this style is adding a directory; there is no shared list to edit, which keeps
parallel style branches conflict-free. This module holds the only registration.
"""

from __future__ import annotations

from ...registry import register
from .style import UniFrqStyle

register("uni_frq")(UniFrqStyle)

__all__ = ["UniFrqStyle"]
