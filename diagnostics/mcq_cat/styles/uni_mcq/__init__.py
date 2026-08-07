"""The ``uni_mcq`` CAT style: unidimensional MCQ adaptive testing.

Runs a Fisher-information CAT over a calibrated unidimensional item bank vendored from
ATLAS (Route C) or Open LLM Leaderboard v2 (Route B), and reports an ability score plus
a p-IRT predicted accuracy.

This module contains the **only** registration for the style. The registry discovers
styles by scanning ``styles/`` and importing each subpackage, so adding a style means
adding a directory -- there is no shared list to edit, which is what keeps parallel
style branches conflict-free.
"""

from __future__ import annotations

from ...registry import register
from .style import UniMcqStyle

register("uni_mcq")(UniMcqStyle)

__all__ = ["UniMcqStyle"]
