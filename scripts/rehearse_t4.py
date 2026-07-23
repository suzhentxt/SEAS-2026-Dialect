from __future__ import annotations

import importlib.metadata
import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vialect_seas.data import load_jsonl, split_train_dev_by_source  # noqa: E402
from vialect_seas.metrics import evaluate_predictions, metric_summary  # noqa: E402
from vialect_seas.normalization import (  # noqa: E402
    attach_lora,
    generate_normalizations,
    load_experiment_start_model,
    make_preprocess_function,
    resolved_model_revision,
)


def main() -> None:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("This rehearsal requires a CUDA GPU (target: Colab T4)")

    train_pool = load_jsonl(PROJECT_ROOT / "data" / "train_240.jsonl")
    train, dev = split_train_dev_by_source(train_pool, dev_sources_per_task=4, seed=2026)
    rehearsal_train = train.head(24).reset_index(drop=True)
    rehearsal_dev = dev.head(8).reset_index(drop=True)
    adapter_path = PROJECT_ROOT / "outputs" / "rehearsal_adapter"
    report_path = PROJECT_ROOT / "outputs" / "t4_rehearsal.json"

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    tokenizer, baseline_model, device = load_experiment_start_model()
    model_revision = resolved_model_revision(baseline_model)
    baseline_started = time.perf_counter()
    baseline_predictions = generate_normalizations(
        rehearsal_dev["dialect_text"], tokenizer, baseline_model, device, batch_size=4
    )
    baseline_seconds = time.perf_counter() - baseline_started
    baseline_scored = evaluate_predictions(
        rehearsal_dev.assign(prediction=baseline_predictions)
    )

    model = attach_lora(baseline_model, rank=8, alpha=16, dropout=0.05)
    preprocess = make_preprocess_function(tokenizer, max_length=192)
    train_ds = Dataset.from_pandas(
        rehearsal_train[["dialect_text", "standard_text"]], preserve_index=False
    )
    train_tok = train_ds.map(
        preprocess, batched=True, remove_columns=train_ds.column_names
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(PROJECT_ROOT / "outputs" / "rehearsal_run"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        fp16=True,
        seed=2026,
        data_seed=2026,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - train_started

    before_reload = generate_normalizations(
        rehearsal_dev["dialect_text"], tokenizer, model, device, batch_size=4
    )
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    del trainer, model, baseline_model
    torch.cuda.empty_cache()
    reload_tokenizer, reload_base, reload_device = load_experiment_start_model()
    reloaded = PeftModel.from_pretrained(reload_base, adapter_path).to(reload_device)
    reloaded.eval()
    after_reload = generate_normalizations(
        rehearsal_dev["dialect_text"], reload_tokenizer, reloaded, reload_device, batch_size=4
    )
    if before_reload != after_reload:
        raise AssertionError("Reloaded adapter predictions differ from pre-save predictions")

    lora_scored = evaluate_predictions(rehearsal_dev.assign(prediction=after_reload))
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "model_revision": model_revision,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "datasets_version": importlib.metadata.version("datasets"),
        "peft_version": importlib.metadata.version("peft"),
        "accelerate_version": importlib.metadata.version("accelerate"),
        "baseline_rows": len(rehearsal_dev),
        "train_rows": len(rehearsal_train),
        "baseline_seconds": baseline_seconds,
        "one_epoch_seconds": train_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "baseline_metrics": metric_summary(baseline_scored).iloc[0].to_dict(),
        "lora_metrics": metric_summary(lora_scored).iloc[0].to_dict(),
        "reload_deterministic": True,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
