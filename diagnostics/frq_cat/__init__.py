"""FRQ CAT diagnostics: frozen scaffolding for adaptive free-response testing.

This package holds the stable contract (:mod:`diagnostics.frq_cat.base`), the
auto-discovering style registry (:mod:`diagnostics.frq_cat.registry`), the CLI
runner (:mod:`diagnostics.frq_cat.runner`), and shared utilities under
:mod:`diagnostics.frq_cat.common`.

It mirrors :mod:`diagnostics.mcq_cat`, but grading is two-stage: the checkpoint
under test generates a free-response answer (respgen) and a frozen LLM-as-a-judge
grades it per rubric criterion. Individual CAT styles live under ``styles/<name>/``
and self-register. Adding a style means adding a new directory only; nothing in
this scaffolding is edited. See ``Plan/frq_cat_diagnostics/README.md`` for the
design and the calibration graduation-package layout each style ships.
"""
