from __future__ import annotations

from collections.abc import Sequence
import unicodedata

import numpy as np
import pandas as pd


def normalize_metric_text(value: object) -> str:
    """Normalize Unicode and whitespace before computing text metrics."""
    text = unicodedata.normalize("NFC", str(value))
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())


def _levenshtein(reference: Sequence, prediction: Sequence) -> int:
    previous = list(range(len(prediction) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for pred_index, pred_item in enumerate(prediction, start=1):
            substitution = previous[pred_index - 1] + (ref_item != pred_item)
            insertion = current[pred_index - 1] + 1
            deletion = previous[pred_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    reference = normalize_metric_text(reference)
    prediction = normalize_metric_text(prediction)
    if not reference:
        return float(bool(prediction))
    return _levenshtein(reference, prediction) / len(reference)


def word_error_rate(reference: str, prediction: str) -> float:
    reference_tokens = normalize_metric_text(reference).split()
    prediction_tokens = normalize_metric_text(prediction).split()
    if not reference_tokens:
        return float(bool(prediction_tokens))
    return _levenshtein(reference_tokens, prediction_tokens) / len(reference_tokens)


def exact_match(reference: str, prediction: str) -> float:
    return float(
        normalize_metric_text(reference).lower()
        == normalize_metric_text(prediction).lower()
    )


def character_f_score(reference: str, prediction: str) -> float:
    """Return sentence-level chrF on a 0-100 scale."""
    from sacrebleu.metrics import CHRF

    normalized_reference = normalize_metric_text(reference)
    normalized_prediction = normalize_metric_text(prediction)
    return float(
        CHRF().sentence_score(normalized_prediction, [normalized_reference]).score
    )


def evaluate_predictions(
    frame: pd.DataFrame,
    reference_column: str = "standard_text",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    scored = frame.copy()
    references = scored[reference_column].map(normalize_metric_text)
    predictions = scored[prediction_column].map(normalize_metric_text)
    scored["cer"] = [
        character_error_rate(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    scored["wer"] = [
        word_error_rate(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    scored["exact_match"] = [
        exact_match(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    scored["chrf"] = [
        character_f_score(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    scored["reference_words"] = references.str.split().str.len()
    scored["prediction_words"] = predictions.str.split().str.len()
    scored["length_ratio"] = np.where(
        scored["reference_words"] > 0,
        scored["prediction_words"] / scored["reference_words"],
        np.nan,
    )
    return scored


def metric_summary(scored: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    metrics = ["cer", "wer", "chrf", "exact_match", "length_ratio"]
    if not by:
        return scored[metrics].mean().to_frame().T
    return scored.groupby(by, observed=True)[metrics].mean().reset_index()


def identity_metric_summary(
    scored: pd.DataFrame,
    dialect_column: str = "dialect_text",
    reference_column: str = "standard_text",
) -> pd.DataFrame:
    """Report metrics separately for exact-copy and non-identity input pairs."""
    audited = scored.copy()
    audited["identity_pair"] = (
        audited[dialect_column].map(normalize_metric_text)
        == audited[reference_column].map(normalize_metric_text)
    )
    summary = metric_summary(audited, by=["identity_pair"])
    counts = (
        audited.groupby("identity_pair", observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    return summary.merge(counts, on="identity_pair", how="left")


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    group_by: list[str],
    cluster_column: str = "sample_id",
    variant_column: str = "variant",
    baseline_variant: str = "standard",
    comparison_variant: str = "dialect",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> pd.DataFrame:
    """Bootstrap a paired baseline-minus-comparison mean by source cluster."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    rows = []
    grouper = group_by[0] if len(group_by) == 1 else group_by
    for group_key, group in frame.groupby(grouper, observed=True, sort=True):
        pivot = group.pivot_table(
            index=cluster_column,
            columns=variant_column,
            values=value_column,
            aggfunc="mean",
        ).dropna(subset=[baseline_variant, comparison_variant])
        differences = (
            pivot[baseline_variant] - pivot[comparison_variant]
        ).to_numpy(dtype=float)
        if differences.size == 0:
            continue
        samples = rng.choice(
            differences,
            size=(n_resamples, differences.size),
            replace=True,
        ).mean(axis=1)
        alpha = (1 - confidence_level) / 2
        keys = (group_key,) if len(group_by) == 1 else tuple(group_key)
        row = dict(zip(group_by, keys))
        row.update(
            {
                "mean": float(differences.mean()),
                "ci_low": float(np.quantile(samples, alpha)),
                "ci_high": float(np.quantile(samples, 1 - alpha)),
                "n_sources": int(differences.size),
                "n_resamples": int(n_resamples),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
