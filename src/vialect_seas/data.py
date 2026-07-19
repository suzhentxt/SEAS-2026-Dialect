from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DIALECTS = ("PNB", "PNT2", "PNT3")
TASKS = ("MCQA", "NLI", "QA", "SENT")
BOUNDARY_NOISE_CHARS = ' \t\r\n/"\\'


def clean_boundary_noise(value: object) -> str | None:
    """Use the same boundary-cleaning rule as the VialectBench import pipeline."""
    if value is None:
        return None
    return str(value).replace("\r\n", "\n").strip(BOUNDARY_NOISE_CHARS)


def standard_text_for(row: dict) -> str:
    """Return the normalization reference for a finalized VialectBench row."""
    if row.get("task") == "NLI":
        value = row.get("hypothesis")
        if not value:
            raise ValueError(f"NLI sample {row.get('sample_id')} has no hypothesis")
        return clean_boundary_noise(value) or ""
    return clean_boundary_noise(row.get("original_text")) or ""


def load_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return pd.DataFrame(rows)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def assert_balanced_split(frame: pd.DataFrame, sources_per_task: int) -> None:
    required = {"sample_id", "task", "target_dialect", "dialect_text", "standard_text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if set(frame["task"]) != set(TASKS):
        raise ValueError(f"Unexpected tasks: {sorted(set(frame['task']))}")
    if set(frame["target_dialect"]) != set(DIALECTS):
        raise ValueError(f"Unexpected dialects: {sorted(set(frame['target_dialect']))}")

    expected_rows = sources_per_task * len(TASKS) * len(DIALECTS)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")

    counts = frame.groupby(["task", "target_dialect"]).size()
    if not (counts == sources_per_task).all():
        raise ValueError(f"Unbalanced task/dialect cells:\n{counts}")

    source_counts = frame.groupby("task")["sample_id"].nunique()
    if not (source_counts == sources_per_task).all():
        raise ValueError(f"Unexpected unique source counts:\n{source_counts}")

    nli = frame[frame["task"] == "NLI"]
    if "hypothesis" in nli.columns and not (nli["standard_text"] == nli["hypothesis"]).all():
        raise ValueError("NLI standard_text must equal hypothesis")
