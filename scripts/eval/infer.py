#!/usr/bin/env python3
"""Single-image inference for a CFCamo model (quick try-it script).

Runs the detect-or-abstain prompt on one image and prints the predicted box(es)
in 0-1000 coordinates, or reports abstention when no camouflaged object is
found. Optionally saves an overlay. For the rigorous paired CF-COD benchmark,
use scripts/eval/eval_cfcod.py instead.

Usage:
  python scripts/eval/infer.py \\
    --model checkpoints/cfcamo-rl-lora \\
    --image path/to/image.jpg \\
    [--save-overlay out.png]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from PIL import Image, ImageDraw

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))  # fallback if cfcamo is not pip-installed
from cfcamo.data import CFCAMO_SYSTEM_PROMPT, CFCAMO_USER_PROMPT  # noqa: E402
from cfcamo.parser import parse_response  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or hub id")
    ap.add_argument("--image", required=True, type=pathlib.Path)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--save-overlay", type=pathlib.Path, default=None,
                    help="optional path to save the input with predicted boxes drawn")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto",
    )
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [
        {"role": "system", "content": CFCAMO_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": CFCAMO_USER_PROMPT},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
        )
    text = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )[0]

    print("=== raw response ===")
    print(text)

    p = parse_response(text)
    print("\n=== parsed ===")
    if p.kind == "refuse":
        print("Prediction: ABSTAIN (no camouflaged object)")
    elif p.kind == "detect" and p.bboxes:
        print(f"Prediction: DETECT, {len(p.bboxes)} box(es) in 0-1000 coords:")
        for b in p.bboxes:
            print("  ", b)
        if args.save_overlay:
            W, H = image.size
            draw = ImageDraw.Draw(image)
            for x1, y1, x2, y2 in p.bboxes:
                draw.rectangle(
                    [x1 * W / 1000.0, y1 * H / 1000.0, x2 * W / 1000.0, y2 * H / 1000.0],
                    outline="red", width=3,
                )
            args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
            image.save(args.save_overlay)
            print(f"[saved overlay] {args.save_overlay}")
    else:
        print("Prediction: invalid / no commit")


if __name__ == "__main__":
    main()
