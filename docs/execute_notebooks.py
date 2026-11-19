"""Execute the tutorial notebooks (needs ``pip install -e ".[docs]"``).

    python docs/execute_notebooks.py           # run all; fail on the first error
    python docs/execute_notebooks.py --write   # run and store outputs in the notebooks
    python docs/execute_notebooks.py 03        # only notebooks whose name contains "03"

Notebooks run with their own folder as working directory and the non-interactive Agg-backed
inline backend; per-cell timing metadata is dropped so stored outputs diff cleanly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent / "tutorials"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="substrings of notebook names to run")
    ap.add_argument("--write", action="store_true", help="save the outputs into the notebooks")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    args = ap.parse_args(argv)
    books = sorted(p for p in HERE.glob("*.ipynb")
                   if not args.only or any(s in p.name for s in args.only))
    for path in books:
        nb = nbformat.read(path, as_version=4)
        t = time.perf_counter()
        NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
        print(f"{path.name}: ok ({time.perf_counter() - t:.0f} s)", flush=True)
        if args.write:
            for cell in nb.cells:
                cell.metadata.pop("execution", None)
            nbformat.write(nb, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
