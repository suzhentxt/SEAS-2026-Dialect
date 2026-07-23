from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Iterable
from pathlib import Path


PRIVATE_NORMALIZER_ID = "tarudesu/mbart-large-50"
BASE_MODEL_ID = "facebook/mbart-large-50"
BASE_MODEL_REVISION = "4ef55a20b36c6903b832e38f0604ab4bdf22c7d6"
PRIVATE_NORMALIZER_REVISION = os.environ.get("PRIVATE_NORMALIZER_REVISION")
# Controlled experiment: baseline and LoRA both start from this checkpoint.
EXPERIMENT_START_MODEL_ID = PRIVATE_NORMALIZER_ID
EXPERIMENT_START_MODEL_REVISION = PRIVATE_NORMALIZER_REVISION


def get_hf_token(required: bool = False) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata

            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    if required and not token:
        raise RuntimeError("Set HF_TOKEN in the environment or Colab Secrets")
    return token


def load_seq2seq_model(
    model_id: str,
    private: bool = False,
    revision: str | None = None,
):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

    token = get_hf_token(required=private)
    kwargs = {"token": token} if token else {}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    config = AutoConfig.from_pretrained(model_id, **kwargs)
    if getattr(config, "early_stopping", None) is None:
        config.early_stopping = False
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config, **kwargs)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = "vi_VN"
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = "vi_VN"
    lang_ids = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_ids, dict) and "vi_VN" in lang_ids:
        model.generation_config.forced_bos_token_id = lang_ids["vi_VN"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def model_requires_token(model_id: str) -> bool:
    return model_id == PRIVATE_NORMALIZER_ID


def load_experiment_start_model(revision: str | None = None):
    """Load the single checkpoint shared by baseline evaluation and LoRA."""
    resolved_revision = (
        revision
        or EXPERIMENT_START_MODEL_REVISION
        or os.environ.get("PRIVATE_NORMALIZER_REVISION")
    )
    if not resolved_revision or len(resolved_revision) != 40:
        raise RuntimeError(
            "Set PRIVATE_NORMALIZER_REVISION to the private checkpoint's full "
            "40-character commit SHA before loading the experiment model."
        )
    tokenizer, model, device = load_seq2seq_model(
        EXPERIMENT_START_MODEL_ID,
        private=model_requires_token(EXPERIMENT_START_MODEL_ID),
        revision=resolved_revision,
    )
    model._vialect_revision = resolved_revision
    return tokenizer, model, device


def resolved_model_revision(model) -> str:
    """Read the immutable Hugging Face commit resolved during model loading."""
    current = model
    for _ in range(5):
        requested_revision = getattr(current, "_vialect_revision", None)
        if requested_revision:
            return str(requested_revision)
        config = getattr(current, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
        next_model = getattr(current, "base_model", None)
        if next_model is None or next_model is current:
            break
        current = next_model
    raise RuntimeError("Could not resolve the loaded model's Hugging Face commit hash")


def save_experiment_config(
    config: dict,
    path: str | Path,
    model_revision: str,
    extra: dict | None = None,
) -> dict:
    """Save a reproducible config manifest and a SHA-256 digest."""
    if len(model_revision) != 40:
        raise ValueError("model_revision must be a full 40-character commit SHA")
    package_names = (
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "sacrebleu",
    )
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    payload = {
        "model_id": EXPERIMENT_START_MODEL_ID,
        "model_revision": model_revision,
        "experiment_config": config,
        "packages": packages,
        "extra": extra or {},
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_experiment_config(path: str | Path) -> dict:
    """Load and verify an experiment config manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.pop("config_sha256")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ValueError("experiment_config.json hash mismatch")
    payload["config_sha256"] = expected
    return payload


def generate_normalizations(
    texts: str | Iterable[str],
    tokenizer,
    model,
    device: str,
    max_length: int = 192,
    batch_size: int = 8,
) -> str | list[str]:
    import torch

    single = isinstance(texts, str)
    values = [texts] if single else list(texts)
    predictions = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        kwargs = {
            "max_length": max_length,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "early_stopping": False,
        }
        lang_ids = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_ids, dict) and "vi_VN" in lang_ids:
            kwargs["forced_bos_token_id"] = lang_ids["vi_VN"]
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **kwargs)
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return predictions[0] if single else predictions


def attach_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    return get_peft_model(model, config)


def make_preprocess_function(tokenizer, max_length: int = 192):
    def preprocess(examples):
        model_inputs = tokenizer(
            examples["dialect_text"],
            max_length=max_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["standard_text"],
            max_length=max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess
