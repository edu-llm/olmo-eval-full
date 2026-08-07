"""Shared, stable utilities for MCQ CAT diagnostics.

These modules hold the roughly 80 percent every CAT style reuses: S3 IO,
benchmark preparation, IRT parameter loading, checkpoint inference / scoring, and
the generic CAT engine. They are written once so styles neither reimplement nor
edit them.

Grading comes in two schemes, one per benchmark modality: :mod:`.inference` scores
multiple-choice items by continuation log-likelihood, :mod:`.generative` samples a
completion and extracts an answer from it, and :mod:`.grading` holds the single table
that maps a dataset's modality to one of them along with the guard against the wrong
pairing.
"""
