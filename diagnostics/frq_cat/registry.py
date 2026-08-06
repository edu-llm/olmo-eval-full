"""Registry and auto-discovery for FRQ CAT styles.

Styles self-register with :func:`register`. Discovery scans the ``styles`` package
with :func:`pkgutil.iter_modules` and imports each subpackage so its registration
runs. There is no central list of styles and no shared dispatch to edit: adding a
style means adding a new directory under ``styles/`` whose ``__init__`` calls
``@register(...)``. That keeps this file frozen while styles are added on their own
branches.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable

from .base import FrqCatStyle

log = logging.getLogger("frq_cat.registry")

_REGISTRY: dict[str, type[FrqCatStyle]] = {}
_discovered = False


def register(name: str) -> Callable[[type[FrqCatStyle]], type[FrqCatStyle]]:
    """Class decorator that registers a CAT style under ``name``.

    The decorator also stamps ``cls.name`` so the style and the registry agree on
    the identifier used by ``--cat-style``.
    """

    def decorator(cls: type[FrqCatStyle]) -> type[FrqCatStyle]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"CAT style {name!r} is already registered by {existing.__qualname__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def discover_styles(*, force: bool = False) -> None:
    """Import every subpackage under ``styles/`` so styles self-register.

    Idempotent after the first successful run unless ``force`` is set. Handles an
    empty ``styles/`` directory gracefully (nothing to import).
    """
    global _discovered
    if _discovered and not force:
        return

    from . import styles

    for module_info in pkgutil.iter_modules(styles.__path__, styles.__name__ + "."):
        importlib.import_module(module_info.name)

    _discovered = True


def get_cat(name: str) -> FrqCatStyle:
    """Resolve a registered style by name and return an instance.

    Runs discovery first, so any registered style is resolvable without this file
    being modified when styles are added.
    """
    discover_styles()
    try:
        cls = _REGISTRY[name]
    except KeyError:
        available = ", ".join(available_styles()) or "<none>"
        raise KeyError(f"Unknown CAT style {name!r}. Available styles: {available}") from None
    return cls()


def available_styles() -> list[str]:
    """Return the sorted names of all discovered, registered styles."""
    discover_styles()
    return sorted(_REGISTRY)
