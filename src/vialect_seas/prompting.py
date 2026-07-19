"""Build zero-shot task prompts from the student paired-normalization rows.

This is the student-project adaptation of the research codebase's
``src/prompting.py``. The research code expects cases with a ``variants``
dict (premise/hypothesis/question/context/options/text); the student data
instead stores paired text in ``standard_text`` / ``dialect_text`` plus a
``source_text`` context and an UPPERCASE ``task`` label. This module bridges
that gap so the probing logic from ``probe_models.py`` can run on the
student data.
"""
from __future__ import annotations

import json
import re
from typing import Any


# UPPERCASE student task -> lowercase probe task.
TASK_TO_PROBE = {"SENT": "sentiment", "NLI": "nli", "MCQA": "mcqa", "QA": "qa"}

# Fixed label candidates for classification tasks (matches probe_models.py).
LABEL_CANDIDATES = {
    "mcqa": ["A", "B", "C", "D"],
    "nli": ["entailment", "neutral", "contradiction"],
    "sentiment": [
        "Anger", "Disgust", "Enjoyment", "Fear", "Sadness", "Surprise", "Other",
    ],
}

# Tasks with fixed label candidates (logprob scoring). QA is generative.
CLASSIFICATION_TASKS = frozenset(LABEL_CANDIDATES)


def probe_task(student_task: str) -> str:
    return TASK_TO_PROBE.get(student_task, student_task.lower())


def is_classification(student_task: str) -> bool:
    return probe_task(student_task) in LABEL_CANDIDATES


def normalize_label(task: str, label: str) -> str:
    """Normalize a gold/predicted label to the canonical candidate form."""
    ptask = probe_task(task)
    text = str(label).strip()
    if ptask == "mcqa":
        match = re.search(r"\b([ABCD])\b", text.upper())
        return match.group(1) if match else text.upper()
    if ptask == "nli":
        lowered = text.lower()
        for cand in LABEL_CANDIDATES["nli"]:
            if cand in lowered:
                return cand
        return text.lower()
    if ptask == "sentiment":
        lowered = text.lower()
        for cand in LABEL_CANDIDATES["sentiment"]:
            if cand.lower() in lowered:
                return cand
        return text
    return text


def candidate_completion(student_task: str, label: str) -> str:
    """Format a label candidate as the JSON completion the model should emit."""
    ptask = probe_task(student_task)
    if ptask in {"sentiment", "nli"}:
        return json.dumps({"label": label}, ensure_ascii=False, separators=(",", ":"))
    if ptask == "mcqa":
        return json.dumps({"answer": label}, ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"Task {student_task} has no fixed label candidates.")


def build_task_prompt(row: dict, variant: str = "dialect") -> str:
    """Build a zero-shot task prompt from a student data row.

    ``variant`` is ``"dialect"`` (uses ``dialect_text``) or ``"standard"``
    (uses ``standard_text``). The non-paired context (premise for NLI) comes
    from ``source_text``.
    """
    ptask = probe_task(row["task"])
    text_key = "dialect_text" if variant == "dialect" else "standard_text"
    text = str(row.get(text_key, "")).strip()
    source = str(row.get("source_text", "")).strip()
    instruction = "Chỉ trả lời JSON hợp lệ, không giải thích.\n"

    if ptask == "sentiment":
        return (
            "Bạn là hệ thống phân loại cảm xúc tiếng Việt.\n"
            "Chọn đúng một nhãn trong danh sách: Anger, Disgust, Enjoyment, "
            "Fear, Sadness, Surprise, Other.\n"
            f"{instruction}"
            'Định dạng: {"label":"<nhãn>"}\n'
            f"Câu: {text}\n"
            "JSON:"
        )

    if ptask == "nli":
        # source_text is the premise; the variant text is the hypothesis.
        return (
            "Xác định quan hệ NLI giữa tiền đề và giả thuyết.\n"
            "Chọn đúng một nhãn: entailment, neutral, contradiction.\n"
            f"{instruction}"
            'Định dạng: {"label":"<nhãn>"}\n'
            f"Tiền đề: {source}\n"
            f"Giả thuyết: {text}\n"
            "JSON:"
        )

    if ptask == "mcqa":
        # student_text already contains the question + A/B/C/D options.
        return (
            "Đọc ngữ cảnh và chọn một đáp án đúng nhất trong bốn lựa chọn A, B, C, D.\n"
            f"{instruction}"
            'Định dạng: {"answer":"A"}\n'
            f"{text}\n"
            "JSON:"
        )

    if ptask == "qa":
        return (
            "Trả lời câu hỏi dựa trên ngữ cảnh. Câu trả lời là một cụm ngắn.\n"
            f"{instruction}"
            'Định dạng: {"answer":"<câu trả lời>"}\n'
            f"Ngữ cảnh: {source}\n"
            f"Câu hỏi: {text}\n"
            "JSON:"
        )

    raise ValueError(f"Unsupported task: {row['task']}")


def parse_prediction(student_task: str, output: str) -> str:
    """Extract a predicted label from a model's raw output."""
    ptask = probe_task(student_task)
    text = (output or "").strip()
    json_matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for json_text in reversed(json_matches):
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            continue
        if ptask in {"sentiment", "nli"} and data.get("label"):
            return normalize_label(student_task, str(data["label"]))
        if ptask == "mcqa" and (data.get("answer") or data.get("label")):
            return normalize_label(student_task, str(data.get("answer") or data.get("label")))
        if ptask == "qa" and data.get("answer"):
            return str(data["answer"]).strip()
    return normalize_label(student_task, text) if ptask != "qa" else text


def gold_label(row: dict) -> str | None:
    """Canonical gold label for a student row (None for generative QA)."""
    ptask = probe_task(row["task"])
    if ptask == "qa":
        return str(row.get("label", "")).strip() or None
    raw = row.get("label")
    return normalize_label(row["task"], str(raw)) if raw is not None else None
