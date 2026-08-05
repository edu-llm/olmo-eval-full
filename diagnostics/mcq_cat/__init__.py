"""MCQ CAT diagnostics: frozen scaffolding for adaptive multiple-choice testing.

This package holds the stable contract (:mod:`diagnostics.mcq_cat.base`), the
auto-discovering style registry (:mod:`diagnostics.mcq_cat.registry`), the CLI
runner (:mod:`diagnostics.mcq_cat.runner`), and shared utilities under
:mod:`diagnostics.mcq_cat.common`.

Individual CAT styles live under ``styles/<name>/`` and self-register. Adding a
style means adding a new directory only; nothing in this scaffolding is edited.
See ``Plan/mcq_cat_diagnostics/README.md`` for the design.
"""
