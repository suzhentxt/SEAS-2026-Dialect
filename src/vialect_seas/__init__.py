"""Utilities for the SEAS 2026 VialectBench student project."""

from .data import DIALECTS, TASKS, load_jsonl
from .metrics import character_error_rate, word_error_rate

__all__ = [
    "DIALECTS",
    "TASKS",
    "load_jsonl",
    "character_error_rate",
    "word_error_rate",
]
