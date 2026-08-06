"""Shared, stable utilities for FRQ CAT diagnostics.

The ~80% every FRQ CAT style reuses: S3 IO, scenario/criterion bank loading, IRT
parameter loading, response generation from the served checkpoint (respgen), the
frozen LLM-as-a-judge client, and the generic CAT engine. Written once and frozen
so styles neither reimplement nor edit them.
"""
