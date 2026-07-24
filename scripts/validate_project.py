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
from vialect_seas.metrics import (  # noqa: E402
    evaluate_predictions,
    paired_cluster_bootstrap,
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
        cell_ids = [cell.get("id") for cell in cells]
        assert all(cell_ids), f"Notebook {name} has cells without an id"
        assert len(cell_ids) == len(set(cell_ids)), (
            f"Notebook {name} has duplicate cell ids"
        )
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
    normalization_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in normalization_payload["cells"]
        if cell.get("cell_type") == "code"
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
    assert "paired_cluster_bootstrap" in probing
    assert "MODEL_REVISIONS" in probing
    assert "identity_metric_summary" in normalization
    assert "experiment_config.json" in normalization
    assert "model_revision" in normalization
    assert 'userdata.get(secret_name)' in normalization_code
    assert '"HF_TOKEN", "PRIVATE_NORMALIZER_REVISION"' in normalization_code
    assert "cer_improvement_ci" in normalization_code
    assert "CER_baseline - CER_LoRA" in normalization_code


def validate_python() -> None:
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def validate_metric_contract() -> None:
    decomposed = "a\u0301   b"
    scored = evaluate_predictions(
        pd.DataFrame(
            {
                "standard_text": ["á b"],
                "prediction": [decomposed],
            }
        )
    )
    assert scored.loc[0, "cer"] == 0
    assert scored.loc[0, "wer"] == 0
    assert scored.loc[0, "exact_match"] == 1

    paired = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "model_id": "m",
                "target_dialect": "d",
                "variant": variant,
                "correct": correct,
            }
            for sample_id, variant, correct in (
                ("a", "standard", 1),
                ("a", "dialect", 0),
                ("b", "standard", 1),
                ("b", "dialect", 1),
            )
        ]
    )
    interval = paired_cluster_bootstrap(
        paired,
        value_column="correct",
        group_by=["model_id", "target_dialect"],
        n_resamples=100,
    )
    assert len(interval) == 1
    assert interval.loc[0, "mean"] == 0.5
    assert interval.loc[0, "n_sources"] == 2


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
    assert (PROJECT_ROOT / "scripts" / "execute_notebooks.py").exists()
    assert (PROJECT_ROOT / ".github" / "workflows" / "validate.yml").exists()
    ci_requirements = PROJECT_ROOT / "requirements-ci.txt"
    assert ci_requirements.exists()
    required_ci_packages = {
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "sacrebleu",
        "scikit-learn",
        "nbclient",
        "nbformat",
        "ipykernel",
    }
    installed_ci_packages = {
        line.split("==", 1)[0]
        for line in ci_requirements.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert installed_ci_packages == required_ci_packages
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "validate.yml"
    ).read_text(encoding="utf-8")
    assert "pip install -r requirements-ci.txt" in workflow
    assert "python scripts/execute_notebooks.py" in workflow

    probing_source = (
        PROJECT_ROOT / "src" / "vialect_seas" / "probing.py"
    ).read_text(encoding="utf-8")
    assert "return_offsets_mapping=True" in probing_source
    assert "completion_ids = runner.tokenizer" not in probing_source
    assert probing_source.count("refs/heads/main") == 0

    normalization_source = (
        PROJECT_ROOT / "src" / "vialect_seas" / "normalization.py"
    ).read_text(encoding="utf-8")
    assert "PRIVATE_NORMALIZER_REVISION" in normalization_source
    assert "config_sha256" in normalization_source

    models = pd.read_csv(
        PROJECT_ROOT / "data" / "model_results" / "task_performance_matrix_precise.csv"
    )
    assert models["Model"].nunique() == 10, "Expected direct-prompting results for 10 models"
    comparison_path = (
        PROJECT_ROOT / "data" / "model_results" / "normalization_model_comparison.csv"
    )
    comparison_columns = pd.read_csv(comparison_path).columns.tolist()
    assert comparison_columns == [
        "model_id", "model_revision", "config_sha256", "split",
        "CER", "WER", "chrF", "exact_match"
    ]


def main() -> None:
    validate_data()
    validate_notebooks()
    validate_python()
    validate_metric_contract()
    validate_no_tokens()
    validate_repository_contract()
    print("SEAS project validation passed")


if __name__ == "__main__":
    main()
