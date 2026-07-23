from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vialect_seas.data import (  # noqa: E402
    DIALECTS,
    TASKS,
    assert_balanced_split,
    clean_boundary_noise,
    write_jsonl,
)


# Corrections are narrow and auditable. Do not add broad spell-check rules here.
TEXT_CORRECTIONS = {
    "MCQA_0056_2": {
        "thông inh": "thông minh",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the SEAS student splits")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT.parent / "data" / "vialectbench_finalized_6.json",
        help="Path to the finalized VialectBench JSON array",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data" / "selection_metadata.json",
        help="Metadata containing the frozen train/test source IDs",
    )
    return parser.parse_args()


def corrected_text(sample_id: str, value: object) -> str:
    text = clean_boundary_noise(value) or ""
    for old, new in TEXT_CORRECTIONS.get(sample_id, {}).items():
        text = text.replace(old, new)
    return text


def prepare_row(row: dict, split: str) -> dict:
    sample_id = str(row["sample_id"])
    task = str(row["task"])
    original = corrected_text(sample_id, row.get("original_text"))
    raw_hypothesis = row.get("hypothesis")
    hypothesis = (
        corrected_text(sample_id, raw_hypothesis)
        if raw_hypothesis is not None
        else None
    )
    if task == "NLI" and not hypothesis:
        raise ValueError(f"NLI sample {sample_id} has no hypothesis")
    standard = hypothesis if task == "NLI" else original
    dialect = corrected_text(sample_id, row.get("dialect_text"))
    return {
        "sample_id": sample_id,
        "task": task,
        "target_dialect": row["target_dialect"],
        "dialect_text": dialect,
        "standard_text": standard,
        "source_text": original,
        "hypothesis": hypothesis,
        "label": row.get("label"),
        "domain": row.get("domain"),
        "split": split,
    }


def build_split(source_rows: list[dict], ids_by_task: dict, split: str) -> list[dict]:
    lookup = {
        (str(row["task"]), str(row["sample_id"]), str(row["target_dialect"])): row
        for row in source_rows
        if row.get("task") in TASKS and row.get("target_dialect") in DIALECTS
    }
    prepared: list[dict] = []
    for task in TASKS:
        for sample_id in ids_by_task[task]:
            for dialect in DIALECTS:
                key = (task, sample_id, dialect)
                if key not in lookup:
                    raise KeyError(f"Frozen selection is missing source row {key}")
                prepared.append(prepare_row(lookup[key], split))
    return prepared


def write_split(rows: list[dict], stem: str) -> None:
    write_jsonl(rows, PROJECT_ROOT / "data" / f"{stem}.jsonl")
    pd.DataFrame(rows).to_csv(
        PROJECT_ROOT / "data" / f"{stem}.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\r\n",
    )


def main() -> None:
    args = parse_args()
    source_bytes = args.source.read_bytes()
    source_rows = json.loads(source_bytes)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))

    train_rows = build_split(source_rows, selection["train_source_ids"], "train")
    test_rows = build_split(source_rows, selection["test_source_ids"], "test")
    train = pd.DataFrame(train_rows)
    test = pd.DataFrame(test_rows)
    assert_balanced_split(train, sources_per_task=20)
    assert_balanced_split(test, sources_per_task=25)
    overlap = set(train["sample_id"]) & set(test["sample_id"])
    if overlap:
        raise ValueError(f"Frozen split leakage: {sorted(overlap)[:5]}")

    write_split(train_rows, "train_240")
    write_split(test_rows, "test_300")

    selection["source"] = f"data/{args.source.name}"
    selection["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    selection["text_corrections"] = TEXT_CORRECTIONS
    args.selection.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(train_rows)} train rows and {len(test_rows)} test rows")


if __name__ == "__main__":
    main()
