"""Runtime context for external evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider
    from olmo_eval.inference.registry import ProviderLookup


@dataclass(frozen=True)
class ExternalEvalContext:
    """Providers available to an external evaluation.

    The primary provider remains available as ``provider`` for compatibility
    with the existing single-provider API. Named auxiliary providers are
    resolved lazily through ``inference_pool``.
    """

    provider: InferenceProvider
    inference_pool: ProviderLookup | None = None

    def get_provider(self, name: str) -> InferenceProvider:
        """Resolve a named auxiliary provider."""
        if self.inference_pool is None:
            raise RuntimeError("No auxiliary inference providers configured.")
        return self.inference_pool.get(name)
