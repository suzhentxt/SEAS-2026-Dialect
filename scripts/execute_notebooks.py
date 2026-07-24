from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
EXPECTED_NOTEBOOKS = (
    "01_eda_preprocessing.ipynb",
    "02_lm_dialect_probing.ipynb",
    "03_text_normalization.ipynb",
)


def configure_runtime_dirs() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vialect-seas-jupyter"
    runtime_dirs = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root / "cache",
        "JUPYTER_CONFIG_DIR": cache_root / "config",
        "JUPYTER_DATA_DIR": cache_root / "data",
        "JUPYTER_RUNTIME_DIR": cache_root / "runtime",
        "IPYTHONDIR": cache_root / "ipython",
    }
    os.environ.setdefault("MPLBACKEND", "Agg")
    for variable, directory in runtime_dirs.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(directory))


def execute_notebook(path: Path, output_dir: Path, timeout: int) -> Path:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    output_path = output_dir / path.name
    try:
        client.execute()
    except CellExecutionError:
        nbformat.write(notebook, output_path)
        print(f"FAIL {path.name}; partial output: {output_path}", file=sys.stderr)
        raise
    nbformat.write(notebook, output_path)
    print(f"PASS {path.name} -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute all student notebooks in fresh Jupyter kernels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "vialect-seas-executed-notebooks",
    )
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_runtime_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_NOTEBOOKS:
        execute_notebook(NOTEBOOK_DIR / name, args.output_dir, args.timeout)
    print(f"Notebook execution {len(EXPECTED_NOTEBOOKS)}/{len(EXPECTED_NOTEBOOKS)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
