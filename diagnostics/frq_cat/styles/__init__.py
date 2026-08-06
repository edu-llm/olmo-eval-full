"""Namespace for FRQ CAT styles.

Each style lives in its own subpackage ``styles/<name>/`` and self-registers via
``@register(...)`` from :mod:`diagnostics.frq_cat.registry`. This package starts
empty on the scaffolding branch; styles are added later on their own branches
(``flow/uni-frq``, ``flow/mirt-frq``), each shipping a calibration graduation
package (fitted bank + scenarios + judge config + provenance) under its directory.
The registry discovers styles by scanning this package, so no file here is edited
when a style is added.
"""
