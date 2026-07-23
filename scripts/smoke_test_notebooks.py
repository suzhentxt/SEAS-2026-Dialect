from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

EXPECTED = (
    "01_eda_preprocessing.ipynb",
    "02_lm_dialect_probing.ipynb",
    "03_text_normalization.ipynb",
)
os.environ.setdefault("MPLBACKEND", "Agg")


def display(value=None, *args, **kwargs):
    if value is not None:
        print(value)


def main() -> int:
    failures = 0
    for name in EXPECTED:
        path = NOTEBOOK_DIR / name
        with path.open(encoding="utf-8") as handle:
            nb = json.load(handle)
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        n_code = len(code_cells)
        n_md = sum(1 for c in nb["cells"] if c.get("cell_type") == "markdown")
        namespace = {"__name__": "__notebook__", "display": display}
        previous = Path.cwd()
        os.chdir(NOTEBOOK_DIR)
        for i, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source, filename=f"{name}:cell-{i}")
                exec(compile(source, f"{name}:cell-{i}", "exec"), namespace)
            except Exception as exc:
                failures += 1
                print(f"FAIL {name} code cell {i}: {exc}")
                break
        os.chdir(previous)
        print(f"{name}: {n_code} code cells, {n_md} markdown cells — execution OK")
    if failures:
        print(f"\n{failures} code cell(s) failed to parse")
        return 1
    print("\nSmoke test 3/3 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
