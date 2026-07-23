from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd


DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B",
    "bigscience/bloom-560m",
    "VietAI/gpt-neo-1.3B-vietnamese-news",
)


def load_causal_lm(model_id: str, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def mean_token_nll(text: str, tokenizer, model, device: str, max_length: int = 256) -> tuple[float, int]:
    import torch

    encoded = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    with torch.inference_mode():
        output = model(**encoded, labels=labels)
    token_count = max(int(encoded["attention_mask"].sum().item()) - 1, 1)
    return float(output.loss.item()), token_count


def score_texts(
    texts: Iterable[str],
    tokenizer,
    model,
    device: str,
    max_length: int = 256,
) -> pd.DataFrame:
    rows = []
    for text in texts:
        nll, token_count = mean_token_nll(text, tokenizer, model, device, max_length)
        rows.append({"text": text, "nll": nll, "ppl": math.exp(nll), "tokens": token_count})
    return pd.DataFrame(rows)


def score_pairs(frame: pd.DataFrame, model_id: str, max_length: int = 256) -> pd.DataFrame:
    import gc
    import torch

    tokenizer, model, device = load_causal_lm(model_id)
    standard = score_texts(frame["standard_text"], tokenizer, model, device, max_length)
    dialect = score_texts(frame["dialect_text"], tokenizer, model, device, max_length)

    result = frame.reset_index(drop=True).copy()
    result["model_id"] = model_id
    result["standard_nll"] = standard["nll"]
    result["dialect_nll"] = dialect["nll"]
    result["delta_nll"] = result["dialect_nll"] - result["standard_nll"]
    result["standard_ppl"] = standard["ppl"]
    result["dialect_ppl"] = dialect["ppl"]

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Zero-shot task probing (adapted from src/probe_models.py).
#
# These functions score fixed label candidates by their sequence log-probability
# and read a predicted label + candidate-normalized proxy score off the
# softmax. These scores are not calibrated model probabilities.
# ---------------------------------------------------------------------------


class TextGeneratorRunner:
    """Lightweight container for a loaded causal LM used for generation."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


def load_text_generator(model_id: str) -> TextGeneratorRunner:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    device = next(model.parameters()).device
    return TextGeneratorRunner(model=model, tokenizer=tokenizer, device=device)


def format_chat_prompt(runner: TextGeneratorRunner, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(runner.tokenizer, "chat_template", None):
        return runner.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def generate(runner: TextGeneratorRunner, prompt: str, max_new_tokens: int = 64) -> str:
    import torch

    chat_prompt = format_chat_prompt(runner, prompt)
    inputs = runner.tokenizer(chat_prompt, return_tensors="pt").to(runner.device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = runner.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=runner.tokenizer.pad_token_id,
            eos_token_id=runner.tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, input_length:]
    return runner.tokenizer.decode(generated_ids, skip_special_tokens=True)


def score_completion(
    runner: TextGeneratorRunner, prompt: str, completion: str
) -> dict:
    """Sum the log-probability of ``completion`` tokens conditioned on ``prompt``."""
    import torch

    chat_prompt = format_chat_prompt(runner, prompt)
    prompt_ids = runner.tokenizer(
        chat_prompt, add_special_tokens=True, return_tensors="pt"
    )["input_ids"][0]
    completion_ids = runner.tokenizer(
        completion, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    if completion_ids.numel() == 0:
        return {"sequence_logprob": float("-inf"), "avg_token_logprob": float("-inf"), "num_tokens": 0}

    input_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(runner.device)
    with torch.inference_mode():
        logits = runner.model(input_ids=input_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    start = prompt_ids.numel() - 1
    end = start + completion_ids.numel()
    completion_log_probs = token_log_probs[0, start:end]
    sequence_logprob = float(completion_log_probs.sum().item())
    avg_token_logprob = sequence_logprob / int(completion_ids.numel())
    return {
        "sequence_logprob": sequence_logprob,
        "avg_token_logprob": avg_token_logprob,
        "num_tokens": int(completion_ids.numel()),
    }


def softmax_scores(scores: dict) -> dict:
    import math

    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {
        label: math.exp(score - max_score) if math.isfinite(score) else 0.0
        for label, score in scores.items()
    }
    total = sum(exp_scores.values())
    if total <= 0:
        uniform = 1.0 / len(scores)
        return {label: uniform for label in scores}
    return {label: value / total for label, value in exp_scores.items()}


def score_label_distribution(runner: TextGeneratorRunner, row: dict, variant: str = "dialect"):
    """Score every label candidate for a classification row.

    Returns None for non-classification tasks (QA). Otherwise returns a dict
    with the softmax distribution, predicted label, proxy confidence, and the
    gold candidate's normalized score.
    """
    from .prompting import (
        LABEL_CANDIDATES,
        build_task_prompt,
        candidate_completion,
        gold_label,
        probe_task,
    )

    ptask = probe_task(row["task"])
    candidates = LABEL_CANDIDATES.get(ptask)
    if not candidates:
        return None

    prompt = build_task_prompt(row, variant=variant)
    label_logprobs = {}
    for label in candidates:
        scored = score_completion(runner, prompt, candidate_completion(row["task"], label))
        label_logprobs[label] = scored["avg_token_logprob"]

    label_probs = softmax_scores(label_logprobs)
    predicted = max(label_probs, key=label_probs.get)
    proxy_confidence = label_probs[predicted]
    gold = gold_label(row)
    return {
        "variant": variant,
        "label_probs": label_probs,
        "proxy_confidence": proxy_confidence,
        "prediction": predicted,
        "gold": gold,
        "correct": (predicted == gold) if gold else None,
        "gold_candidate_score": label_probs.get(gold) if gold else None,
    }


def probe_classification_rows(
    frame: pd.DataFrame, runner: TextGeneratorRunner, variants=("standard", "dialect")
) -> pd.DataFrame:
    """Run zero-shot label scoring over a DataFrame of student rows.

    For each row and each variant, computes the predicted label, candidate
    scores, and correctness. Non-classification rows (QA) are skipped.
    """
    from .prompting import is_classification

    records = []
    for _, row in frame.iterrows():
        if not is_classification(row["task"]):
            continue
        for variant in variants:
            dist = score_label_distribution(runner, row, variant=variant)
            if dist is None:
                continue
            records.append({
                "sample_id": row.get("sample_id"),
                "task": row["task"],
                "target_dialect": row.get("target_dialect"),
                "model_id": getattr(runner.model, "name_or_path", ""),
                "variant": variant,
                "prediction": dist["prediction"],
                "gold": dist["gold"],
                "correct": dist["correct"],
                "proxy_confidence": dist["proxy_confidence"],
                "gold_candidate_score": dist["gold_candidate_score"],
            })
    return pd.DataFrame(records)
