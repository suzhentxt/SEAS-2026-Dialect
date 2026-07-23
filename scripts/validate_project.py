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

from vialect_seas.data import (  # noqa: E402
    assert_balanced_split,
    clean_boundary_noise,
    load_jsonl,
    split_train_dev_by_source,
)


EXPECTED_NOTEBOOKS = (
    "01_eda_preprocessing.ipynb",
    "02_lm_dialect_probing.ipynb",
    "03_text_normalization.ipynb",
)
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")
EXPECTED_DOCS = (
    "docs/PROJECT_BRIEF.md",
    "docs/TEACHING_PLAN.md",
    "docs/PROJECT_RUBRIC.md",
    "docs/COLAB_RUNBOOK.md",
    "docs/MODEL_AND_DATA_PROVENANCE.md",
    "MODEL_CARD.md",
    "DATA_CARD.md",
    "LICENSE",
)


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
    internal_train, dev = split_train_dev_by_source(train, dev_sources_per_task=4, seed=2026)
    assert len(internal_train) == 192 and len(dev) == 48
    assert not (set(internal_train["sample_id"]) & set(dev["sample_id"]))

    combined = pd.concat([train, test], ignore_index=True)
    assert not combined["standard_text"].str.contains("thông inh", case=False).any()
    for column in ("dialect_text", "standard_text", "source_text", "hypothesis"):
        present = combined[column].notna()
        assert (
            combined.loc[present, column]
            .map(clean_boundary_noise)
            .eq(combined.loc[present, column])
            .all()
        ), (
            f"{column} is not normalized"
        )


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

    normalization_payload = json.loads(
        (PROJECT_ROOT / "notebooks" / "03_text_normalization.ipynb").read_text(
            encoding="utf-8"
        )
    )
    normalization = "\n".join(
        "".join(cell.get("source", []))
        for cell in normalization_payload["cells"]
    )
    required_protocol = (
        "EXPERIMENT_START_MODEL_ID",
        "split_train_dev_by_source",
        "eval_dataset=dev_tok",
        'metric_for_best_model="cer"',
        "load_best_model_at_end=True",
        "RUN_FINAL_TEST",
        "RUN_DOWNSTREAM_RECOVERY",
    )
    for marker in required_protocol:
        assert marker in normalization, f"Normalization protocol missing {marker}"
    assert "Delta PPL so với chuẩn vàng" not in normalization
    assert "tốt hơn chuẩn" not in normalization

    probing_payload = json.loads(
        (PROJECT_ROOT / "notebooks" / "02_lm_dialect_probing.ipynb").read_text(
            encoding="utf-8"
        )
    )
    probing = "\n".join(
        "".join(cell.get("source", []))
        for cell in probing_payload["cells"]
    )
    assert "gold_candidate_score" in probing
    assert '"confidence"' not in probing


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


def validate_repository_contract() -> None:
    for relative in EXPECTED_DOCS:
        assert (PROJECT_ROOT / relative).exists(), f"Missing repository file: {relative}"
    assert (PROJECT_ROOT / "scripts" / "prepare_student_data.py").exists()
    assert (PROJECT_ROOT / "scripts" / "rehearse_t4.py").exists()
    assert (PROJECT_ROOT / ".github" / "workflows" / "validate.yml").exists()

    models = pd.read_csv(
        PROJECT_ROOT / "data" / "model_results" / "task_performance_matrix_precise.csv"
    )
    assert models["Model"].nunique() == 10, "Expected direct-prompting results for 10 models"
    comparison_path = (
        PROJECT_ROOT / "data" / "model_results" / "normalization_model_comparison.csv"
    )
    comparison_columns = pd.read_csv(comparison_path).columns.tolist()
    assert comparison_columns == [
        "model_id", "split", "CER", "WER", "chrF", "exact_match"
    ]


def main() -> None:
    validate_data()
    validate_notebooks()
    validate_python()
    validate_no_tokens()
    validate_repository_contract()
    print("SEAS project validation passed")


if __name__ == "__main__":
    main()
