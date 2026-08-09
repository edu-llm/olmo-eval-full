"""Provider registry - name-based lookup with replica support."""

from __future__ import annotations

import inspect
import itertools
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from olmo_eval.common.logging import get_logger

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider
    from olmo_eval.inference.providers.config import ProviderConfig

logger = get_logger(__name__)


@runtime_checkable
class ProviderLookup(Protocol):
    """Protocol for provider lookup by name."""

    def get(self, name: str) -> InferenceProvider: ...

    @property
    def names(self) -> list[str]: ...


class ReplicaSet:
    """Holds resolved configs for a provider with round-robin access."""

    def __init__(self, name: str, configs: list[ProviderConfig]):
        if not configs:
            raise ValueError(f"ReplicaSet {name!r} requires at least one config")
        self._name = name
        self._configs = configs
        self._providers: list[InferenceProvider | None] = [None] * len(configs)
        self._counter = itertools.count()

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._configs[0].model

    @property
    def num_replicas(self) -> int:
        return len(self._configs)

    def next(self) -> InferenceProvider:
        """Get next provider in round-robin order (lazy creation)."""
        idx = next(self._counter) % len(self._configs)
        if self._providers[idx] is None:
            self._providers[idx] = self._configs[idx].create_provider()
        provider = self._providers[idx]
        assert provider is not None
        return provider

    def get_config(self, idx: int = 0) -> ProviderConfig:
        return self._configs[idx]

    async def aclose(self) -> None:
        """Release every provider instance that this replica set created.

        Providers are constructed lazily, so untouched replicas require no
        cleanup.  Some providers expose an async client cleanup method as well
        as a synchronous server cleanup method; invoke both when present.
        """

        errors: list[Exception] = []
        for provider in self._providers:
            if provider is None:
                continue
            for method_name in ("aclose", "close"):
                method = getattr(provider, method_name, None)
                if not callable(method):
                    continue
                try:
                    result = method()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    errors.append(exc)
        self._providers = [None] * len(self._configs)
        if errors:
            raise ExceptionGroup(
                f"failed to close provider replica set {self._name!r}",
                errors,
            )


class ProviderRegistry:
    """Registry of providers by name with replica support."""

    def __init__(self) -> None:
        self._replica_sets: dict[str, ReplicaSet] = {}

    @classmethod
    def from_resolved_configs(cls, configs: dict[str, list[ProviderConfig]]) -> ProviderRegistry:
        """Create registry from resolved ProviderConfigs."""
        registry = cls()
        for name, config_list in configs.items():
            registry._replica_sets[name] = ReplicaSet(name, config_list)
        return registry

    @classmethod
    def from_serialized(cls, data: dict[str, list[dict[str, Any]]]) -> ProviderRegistry | None:
        """Create registry from serialized config dicts."""
        if not data:
            return None

        from olmo_eval.inference.providers.config import ProviderConfig

        configs: dict[str, list[ProviderConfig]] = {}
        for name, config_dicts in data.items():
            configs[name] = [ProviderConfig.from_dict(d) for d in config_dicts]

        return cls.from_resolved_configs(configs)

    def get(self, name: str) -> InferenceProvider:
        """Get provider by name (round-robin for multi-replica)."""
        if name not in self._replica_sets:
            raise KeyError(f"Unknown provider {name!r}. Available: {self.names}")
        return self._replica_sets[name].next()

    def get_replica_set(self, name: str) -> ReplicaSet:
        if name not in self._replica_sets:
            raise KeyError(f"Unknown provider {name!r}. Available: {self.names}")
        return self._replica_sets[name]

    @property
    def names(self) -> list[str]:
        return sorted(self._replica_sets.keys())

    @property
    def models(self) -> list[str]:
        return sorted(rs.model for rs in self._replica_sets.values())

    def to_serialized(self) -> dict[str, list[dict[str, Any]]]:
        return {
            name: [rs.get_config(i).to_dict() for i in range(rs.num_replicas)]
            for name, rs in self._replica_sets.items()
        }

    async def aclose(self) -> None:
        """Release all provider clients while attempting every replica set."""

        errors: list[Exception] = []
        for replica_set in self._replica_sets.values():
            try:
                await replica_set.aclose()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("failed to close one or more provider registries", errors)
