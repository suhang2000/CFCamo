"""PEFT-based LoRA merge: SFT_base + lora_adapter → merged HF model.

Use case: when EasyR1's model_merger.py needs FSDP shards but those have been
deleted, this script reconstructs the merged model from lora_adapter alone.

Math: W_merged = W_base + LoRA_A @ LoRA_B (mathematically identical to FSDP merge).

Usage:
  python scripts/eval/merge_lora.py \
    --base checkpoints/cfcamo-sft-4b \
    --lora checkpoints/rl_lora/global_step_252/actor/lora_adapter \
    --out checkpoints/cfcamo-rl-lora
"""
from __future__ import annotations

import argparse
import pathlib
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SFT base model path")
    ap.add_argument("--lora", required=True, help="lora_adapter dir (with adapter_model.safetensors)")
    ap.add_argument("--out", required=True, help="output merged HF dir")
    ap.add_argument("--copy-tokenizer", action="store_true", default=True,
                    help="also copy tokenizer files from base (default True)")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_path = pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[peft-merge] loading base from {args.base}")
    base = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"[peft-merge] loading LoRA adapter from {args.lora}")
    lora_model = PeftModel.from_pretrained(base, args.lora)
    print("[peft-merge] merging LoRA into base...")
    merged = lora_model.merge_and_unload()
    print(f"[peft-merge] saving merged model to {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)
    print("[peft-merge] saving tokenizer/processor")
    proc = AutoProcessor.from_pretrained(args.base)
    proc.save_pretrained(args.out)
    print(f"[peft-merge] done. Files at {args.out}:")
    for p in sorted(out_path.iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
