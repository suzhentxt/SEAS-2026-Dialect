from __future__ import annotations

import os
from collections.abc import Iterable


PRIVATE_NORMALIZER_ID = "tarudesu/mbart-large-50"
BASE_MODEL_ID = "facebook/mbart-large-50"


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


def load_seq2seq_model(model_id: str, private: bool = False):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

    token = get_hf_token(required=private)
    kwargs = {"token": token} if token else {}
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


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
