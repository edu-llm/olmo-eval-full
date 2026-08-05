"""Shared, stable utilities for MCQ CAT diagnostics.

These modules hold the roughly 80 percent every CAT style reuses: S3 IO,
benchmark preparation, IRT parameter loading, checkpoint inference / scoring, and
the generic CAT engine. They are written once so styles neither reimplement nor
edit them.
"""
