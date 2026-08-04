"""Namespace for MCQ CAT styles.

Each style lives in its own subpackage ``styles/<name>/`` and self-registers via
``@register(...)`` from :mod:`diagnostics.mcq_cat.registry`. This package starts
empty on the scaffolding branch; styles are added later on their own branches.
The registry discovers styles by scanning this package, so no file here is edited
when a style is added.
"""
