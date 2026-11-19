"""Results container (FEATURES C6): arrays plus the exact config and versions that made them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Result:
    config: dict[str, Any]
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def summary(self, z: float = 1.96) -> dict[str, tuple[float, float]]:
        """``{name: (mean, CI half-width)}`` for every per-trial 1-D array."""
        from aircompkit.metrics import mean_ci

        return {k: mean_ci(v, z) for k, v in self.arrays.items() if v.ndim == 1}

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Write ``<path>.npz`` (arrays) and ``<path>.json`` (config + meta)."""
        p = Path(path)
        stem = p.with_suffix("") if p.suffix in {".npz", ".json"} else p
        stem.parent.mkdir(parents=True, exist_ok=True)
        npz, js = stem.with_suffix(".npz"), stem.with_suffix(".json")
        np.savez_compressed(npz, **self.arrays)  # type: ignore[arg-type]
        js.write_text(json.dumps({"config": self.config, "meta": self.meta}, indent=2))
        return npz, js

    @classmethod
    def load(cls, path: str | Path) -> Result:
        p = Path(path)
        stem = p.with_suffix("") if p.suffix in {".npz", ".json"} else p
        with np.load(stem.with_suffix(".npz")) as z:
            arrays = {k: z[k] for k in z.files}
        info = json.loads(stem.with_suffix(".json").read_text())
        return cls(config=info["config"], arrays=arrays, meta=info.get("meta", {}))
