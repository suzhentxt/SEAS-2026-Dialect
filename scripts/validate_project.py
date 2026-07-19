from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from vialect_seas.data import assert_balanced_split, load_jsonl  # noqa: E402


EXPECTED_NOTEBOOKS = (
    "01_eda_preprocessing.ipynb",
    "02_lm_dialect_probing.ipynb",
    "03_text_normalization.ipynb",
)
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def validate_data() -> None:
    train = load_jsonl(PROJECT_ROOT / "data" / "train_240.jsonl")
    test = load_jsonl(PROJECT_ROOT / "data" / "test_300.jsonl")
    assert_balanced_split(train, sources_per_task=20)
    assert_balanced_split(test, sources_per_task=25)

    overlap = set(train["sample_id"]) & set(test["sample_id"])
    assert not overlap, f"Train/test leakage: {sorted(overlap)[:5]}"
    assert train["dialect_text"].str.len().gt(0).all()
    assert train["standard_text"].str.len().gt(0).all()
    assert test["dialect_text"].str.len().gt(0).all()
    assert test["standard_text"].str.len().gt(0).all()

    nli = pd.concat([train, test]).query("task == 'NLI'")
    assert (nli["standard_text"] == nli["hypothesis"]).all(), "NLI target is not hypothesis"


def validate_notebooks() -> None:
    for name in EXPECTED_NOTEBOOKS:
        path = PROJECT_ROOT / "notebooks" / name
        assert path.exists(), f"Missing notebook: {name}"
        with path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)
        assert notebook.get("nbformat") == 4
        cells = notebook.get("cells", [])
        assert len(cells) >= 12, f"Notebook {name} is too small"
        assert any(cell.get("cell_type") == "markdown" for cell in cells)
        assert any(cell.get("cell_type") == "code" for cell in cells)
        content = "\n".join("".join(cell.get("source", [])) for cell in cells)
        assert "Mục tiêu học tập" in content
        assert "Insight" in content
        code_content = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "code"
        )
        assert code_content.count('"""Your code here"""') >= 3, (
            f"Notebook {name} needs at least three student code cells"
        )
        assert code_content.count("HINT") >= 3, f"Notebook {name} is missing HINTs"
        assert "SELF-CHECK" in code_content, f"Notebook {name} is missing self-checks"


def validate_python() -> None:
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def validate_no_tokens() -> None:
    suffixes = {".py", ".md", ".ipynb", ".json", ".jsonl", ".csv", ".txt"}
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert not TOKEN_PATTERN.search(content), f"Possible Hugging Face token in {path}"


def main() -> None:
    validate_data()
    validate_notebooks()
    validate_python()
    validate_no_tokens()
    print("SEAS project validation passed")


if __name__ == "__main__":
    main()
