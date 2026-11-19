"""Paper reproductions (FEATURES R1-R3). Each module registers a ``reproduction`` entry:
a callable ``run(trials=None, quick=False, out=None) -> dict``; ``CHECKS[name](result)``
returns the tolerance checks that the reproduction CI enforces (``--check``).

    aircompkit reproduce               # list
    aircompkit reproduce yang2020 --quick --check --out results/yang2020
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aircompkit.reproduce import yang2020, zhu2020

CHECKS: dict[str, Callable[[dict[str, Any]], list[tuple[str, bool, str]]]] = {
    "yang2020": yang2020.check, "zhu2020": zhu2020.check}


def report(name: str, res: dict[str, Any]) -> bool:
    """Print the checks of reproduction ``name`` on ``res``; ``True`` when all pass."""
    checks = CHECKS[name](res)
    print(f"\nchecks ({name}):")
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return all(ok for _, ok, _ in checks)
