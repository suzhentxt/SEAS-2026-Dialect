from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DIALECTS = ("PNB", "PNT2", "PNT3")
TASKS = ("MCQA", "NLI", "QA", "SENT")
BOUNDARY_NOISE_CHARS = ' \t\r\n/"\\'


def clean_boundary_noise(value: object) -> str | None:
    """Use the same boundary-cleaning rule as the VialectBench import pipeline."""
    if value is None:
        return None
    text = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(lines).strip(BOUNDARY_NOISE_CHARS)


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


def split_train_dev_by_source(
    frame: pd.DataFrame,
    dev_sources_per_task: int = 4,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a paired frame by source ID, preserving all dialect rows together."""
    rng = np.random.default_rng(seed)
    dev_ids: set[str] = set()
    sources = frame[["task", "sample_id"]].drop_duplicates()
    for _, group in sources.groupby("task", sort=True):
        ids = np.array(sorted(group["sample_id"].astype(str)))
        if len(ids) <= dev_sources_per_task:
            raise ValueError("Each task needs more sources than the requested dev size")
        rng.shuffle(ids)
        dev_ids.update(ids[:dev_sources_per_task])

    dev_mask = frame["sample_id"].astype(str).isin(dev_ids)
    train = frame.loc[~dev_mask].reset_index(drop=True)
    dev = frame.loc[dev_mask].reset_index(drop=True)
    overlap = set(train["sample_id"]) & set(dev["sample_id"])
    if overlap:
        raise ValueError(f"Train/dev source leakage: {sorted(overlap)[:5]}")
    return train, dev


def identity_rate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exact-copy rates by task and dialect for data-quality auditing."""
    audited = frame.assign(identity_pair=frame["dialect_text"] == frame["standard_text"])
    return (
        audited.groupby(["task", "target_dialect"], observed=True)["identity_pair"]
        .agg(identity_pairs="sum", rows="size", identity_rate="mean")
        .reset_index()
    )
