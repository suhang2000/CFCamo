#!/usr/bin/env python3
"""SFT trainer for CFCamo — TRL SFTTrainer + Qwen3-VL-4B + custom multimodal collator.

Design:
  - Use TRL SFTTrainer (the HF cookbook recommended path), but take a custom
    data_collator route instead of relying on TRL's built-in
    DataCollatorForVisionLanguageModeling (0.29 not fully tested against Qwen3-VL).
  - The collator reuses cfcamo.data.cfcamo_sft_collate_fn.

Usage (1-epoch cold-start SFT, single GPU):
  python scripts/train/sft_train.py \\
    --train-jsonl data/cfcod/sft/sft_balanced.jsonl \\
    --base-model Qwen/Qwen3-VL-4B-Instruct \\
    --output-dir checkpoints/cfcamo-sft-4b \\
    --epochs 1 --batch-size 2 --grad-accum 8 --lr 2e-5

  Quick smoke test: add --debug-n 5 --batch-size 1 --grad-accum 1.

Pitfalls:
  - SFTConfig(max_length=None) must be set, else truncation cuts image tokens.
  - Qwen3-VL needs AutoModelForImageTextToText (transformers >= 4.57).
  - remove_unused_columns=False is required, else SFTTrainer drops the
    image/problem columns the collator needs.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from functools import partial

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))  # fallback if cfcamo is not pip-installed
from cfcamo.data import CFCamoSFTDataset, cfcamo_sft_collate_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-jsonl", required=True, type=pathlib.Path,
                   help="SFT corpus jsonl (see README for the data layout)")
    p.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct",
                   help="HF model id (Qwen/Qwen3-VL-4B-Instruct or -8B-Instruct)")
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2,
                   help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="gradient accumulation steps; effective bs = bs * accum * n_gpu")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-pixels", type=int, default=768 * 768,
                   help="Qwen3-VL processor max_pixels (caps image token count)")
    p.add_argument("--no-bf16", action="store_true",
                   help="Disable bf16 (default: bf16 on)")
    p.add_argument("--no-flash-attn", action="store_true",
                   help="Use eager attention (slower, but no flash_attn dep)")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--debug-n", type=int, default=0,
                   help="Take only first N samples (smoke test); 0 = full dataset")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--save-only-model", action="store_true", default=True,
                   help="save model weights only (no optimizer/scheduler state); "
                        "shrinks a 4B checkpoint from ~40GB to ~8GB")
    p.add_argument("--save-with-optimizer", dest="save_only_model", action="store_false",
                   help="override --save-only-model so the ckpt includes the optimizer (for resuming training)")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Override num_train_epochs; -1 = use epochs")
    p.add_argument("--report-to", default="none",
                   help="wandb / tensorboard / none")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[args] {vars(args)}")

    # 1. Processor (image_processor + tokenizer)
    print(f"[load] processor: {args.base_model}  max_pixels={args.max_pixels}")
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        max_pixels=args.max_pixels,
    )
    print(f"  processor: {type(processor).__name__}")
    print(f"  tokenizer: {type(processor.tokenizer).__name__}")
    print(f"  image_processor: {type(processor.image_processor).__name__}")

    # 2. Model
    bf16 = not args.no_bf16
    model_dtype = torch.bfloat16 if bf16 else torch.float32
    attn_impl = "eager" if args.no_flash_attn else "flash_attention_2"
    print(f"[load] model: {args.base_model}  dtype={model_dtype}  attn={attn_impl}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=model_dtype,
        attn_implementation=attn_impl,
        device_map=None,  # single-GPU / FSDP handled by the Trainer
    )
    print(f"  model: {type(model).__name__}  params={sum(p.numel() for p in model.parameters()):,}")

    # 3. Dataset
    ds = CFCamoSFTDataset(args.train_jsonl)
    if args.debug_n > 0:
        ds.rows = ds.rows[: args.debug_n]
        print(f"[debug] truncated to {len(ds)} samples")
    print(f"[data] {len(ds)} samples")

    # 4. SFTConfig
    cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=bf16,
        max_length=None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=6,  # 22 vCPUs available; 6 workers speed up PIL decoding
        dataloader_pin_memory=True,
    )

    # 5. Trainer
    collator = partial(cfcamo_sft_collate_fn, processor=processor)
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        data_collator=collator,
        processing_class=processor,
    )

    print(f"[train] starting epochs={args.epochs} max_steps={args.max_steps} "
          f"effective_bs={args.batch_size * args.grad_accum}")
    trainer.train()
    # Save the final model to the output_dir root (intermediate checkpoints are
    # in checkpoint-* subdirs). This matches the RL config model_path and the
    # README merge command, both of which point at output_dir directly.
    print(f"[train] done. saving final model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
