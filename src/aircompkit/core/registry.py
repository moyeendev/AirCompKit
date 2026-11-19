"""Plug-in registry (FEATURES C2).

Every swappable component registers under a *kind* and a *name*::

    @register("channel", "rayleigh")
    class Rayleigh: ...

    get("channel", "rayleigh")       # -> Rayleigh
    available("channel")             # -> ["rayleigh", ...]

Configs refer to components by name, so users can add their own without touching the core.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

KINDS: tuple[str, ...] = (
    "devices",
    "channel",
    "power_control",
    "aircomp",
    "receiver",
    "attack",
    "defense",
    "aggregator",
    "dataset",
    "partitioner",
    "model",
    "metric",
    "reproduction",
)

_REGISTRY: dict[str, dict[str, Any]] = {k: {} for k in KINDS}


def _check_kind(kind: str) -> None:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown component kind {kind!r}; expected one of {', '.join(KINDS)}")


def register(kind: str, name: str, *, overwrite: bool = False) -> Callable[[T], T]:
    """Class/function decorator that registers a component under ``kind``/``name``."""
    _check_kind(kind)
    key = name.lower()

    def deco(obj: T) -> T:
        if key in _REGISTRY[kind] and not overwrite and _REGISTRY[kind][key] is not obj:
            raise ValueError(f"{kind} {name!r} is already registered")
        _REGISTRY[kind][key] = obj
        return obj

    return deco


def get(kind: str, name: str) -> Any:
    """Return the component registered under ``kind``/``name`` (case-insensitive)."""
    _check_kind(kind)
    try:
        return _REGISTRY[kind][name.lower()]
    except KeyError:
        opts = ", ".join(sorted(_REGISTRY[kind])) or "none registered"
        raise KeyError(f"no {kind} named {name!r} (available: {opts})") from None


def available(kind: str | None = None) -> list[str] | dict[str, list[str]]:
    """Names registered for one kind, or a ``{kind: names}`` map for all kinds."""
    if kind is None:
        return {k: sorted(v) for k, v in _REGISTRY.items()}
    _check_kind(kind)
    return sorted(_REGISTRY[kind])


def build(kind: str, spec: Any) -> Any:
    """Turn a config spec into a component instance.

    ``spec`` may be an instance already, a name (``"rayleigh"``) or a dict
    ``{"name": "rician", "k_factor": 5}``.
    """
    if isinstance(spec, str):
        return get(kind, spec)()
    if isinstance(spec, dict):
        params = dict(spec)
        try:
            name = params.pop("name")
        except KeyError:
            raise KeyError(f"{kind} spec needs a 'name' key: {spec!r}") from None
        return get(kind, name)(**params)
    return spec


def spec_of(kind: str, obj: Any) -> dict[str, Any]:
    """Inverse of ``build`` for registered dataclass components."""
    _check_kind(kind)
    for name, cls in _REGISTRY[kind].items():
        if type(obj) is cls:
            break
    else:
        raise KeyError(f"{type(obj).__name__} is not a registered {kind}")
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return {"name": name}
    return {"name": name, **_plain(dataclasses.asdict(obj))}


def _plain(x: Any) -> Any:
    """Convert numpy scalars/arrays inside a params dict to built-in types (for JSON/YAML)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def unregister(kind: str, name: str) -> None:
    """Remove a component (mainly for tests)."""
    _check_kind(kind)
    _REGISTRY[kind].pop(name.lower(), None)
