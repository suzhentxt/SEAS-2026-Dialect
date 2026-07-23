from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


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
    reference = str(reference)
    prediction = str(prediction)
    if not reference:
        return float(bool(prediction))
    return _levenshtein(reference, prediction) / len(reference)


def word_error_rate(reference: str, prediction: str) -> float:
    reference_tokens = str(reference).split()
    prediction_tokens = str(prediction).split()
    if not reference_tokens:
        return float(bool(prediction_tokens))
    return _levenshtein(reference_tokens, prediction_tokens) / len(reference_tokens)


def exact_match(reference: str, prediction: str) -> float:
    normalize = lambda text: " ".join(str(text).lower().split())
    return float(normalize(reference) == normalize(prediction))


def character_f_score(reference: str, prediction: str) -> float:
    """Return sentence-level chrF on a 0-100 scale."""
    from sacrebleu.metrics import CHRF

    return float(CHRF().sentence_score(str(prediction), [str(reference)]).score)


def evaluate_predictions(
    frame: pd.DataFrame,
    reference_column: str = "standard_text",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    scored = frame.copy()
    scored["cer"] = [
        character_error_rate(ref, pred)
        for ref, pred in zip(scored[reference_column], scored[prediction_column])
    ]
    scored["wer"] = [
        word_error_rate(ref, pred)
        for ref, pred in zip(scored[reference_column], scored[prediction_column])
    ]
    scored["exact_match"] = [
        exact_match(ref, pred)
        for ref, pred in zip(scored[reference_column], scored[prediction_column])
    ]
    scored["chrf"] = [
        character_f_score(ref, pred)
        for ref, pred in zip(scored[reference_column], scored[prediction_column])
    ]
    scored["reference_words"] = scored[reference_column].str.split().str.len()
    scored["prediction_words"] = scored[prediction_column].str.split().str.len()
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
