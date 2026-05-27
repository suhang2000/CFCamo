"""CFCamo SFT data adapter: jsonl row -> Qwen3-VL chat messages + label masking.

build_chat_messages, find_assistant_start, and mask_assistant_labels are pure
Python (tokenizer-decoupled); the processor integration (apply_chat_template +
image load + label masking) lives in CFCamoSFTDataset.

Paired design: the detect and abstain examples share an identical chat structure
(same system + same user image/prompt) and differ only in the assistant
response, so the model must decide from the image, not from a prompt shortcut.
"""
from __future__ import annotations

import json
import pathlib
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

# CFCAMO prompt — must match the inference-time prompt for apples-to-apples eval.
# Canonical CFCamo prompts. SFT, RL, inference, and CF-COD evaluation all use
# these. Output schema: <think>...</think> then a 0-1000 bbox (single or nested
# array) or <no_camouflage/>. The SFT/RL jsonl carries a per-row 'system' /
# 'problem' override; these are the defaults.
CFCAMO_SYSTEM_PROMPT = (
    "You are a camouflaged object detector. Output in this exact format:\n\n"
    "<think>your reasoning here</think>\n"
    "followed by ONE of:\n"
    "  - <bbox>[x1,y1,x2,y2]</bbox>  for a single camouflaged object\n"
    "  - <bbox>[[x1,y1,x2,y2],[x3,y3,x4,y4]]</bbox>  for multiple objects\n"
    "  - <no_camouflage/>  if no camouflaged object is present\n\n"
    "Coordinates are normalized to [0, 1000] where 1000 = full image dimension."
)

CFCAMO_USER_PROMPT = (
    "Identify and locate any camouflaged object in the image.\n\n"
    "In <think></think>, briefly consider scene textures, visual anomalies, "
    "and if any object blends in. Then output ONE of:\n"
    "- <bbox>[x1,y1,x2,y2]</bbox> for one object, or [[x1,y1,x2,y2],...] for multiple\n"
    "- <no_camouflage/> if no camouflaged object"
)

IGNORE_INDEX = -100


def build_chat_messages(row: dict) -> list[dict]:
    """sft jsonl row → OpenAI-style messages [system, user(image+text), assistant].

    Accepts both jsonl key layouts (image_path/solution[/system] or image/response):
      - image:    row["image_path"] OR row["image"]
      - response: row["solution"]   OR row["response"]
      - system:   row["system"]     OR CFCAMO_SYSTEM_PROMPT (default)
      - problem:  row["problem"]    OR CFCAMO_USER_PROMPT
    """
    image = row.get("image_path") or row.get("image")
    if image is None:
        raise KeyError("row missing 'image_path' / 'image' field")
    response = row.get("solution") or row.get("response")
    if response is None:
        raise KeyError("row missing 'solution' / 'response' field")
    system = row.get("system", CFCAMO_SYSTEM_PROMPT)
    problem = row.get("problem", CFCAMO_USER_PROMPT)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": problem},
            ],
        },
        {"role": "assistant", "content": response},
    ]


def mask_assistant_labels(input_ids: Sequence[int], assistant_start: int) -> list[int]:
    """Build SFT labels: -100 before assistant_start, input_ids afterward.

    SFT computes cross-entropy only on the assistant response; the system + user
    prompt and the image placeholder tokens are masked out.
    """
    n = len(input_ids)
    if assistant_start < 0 or assistant_start > n:
        raise ValueError(
            f"assistant_start={assistant_start} out of range [0, {n}]"
        )
    return [IGNORE_INDEX] * assistant_start + list(input_ids[assistant_start:])


def find_assistant_start(input_ids: Sequence[int], marker_token: int) -> int:
    """Index of the first assistant-response token.

    The rendered Qwen3-VL assistant header ends with a fixed token sequence
    (e.g. "<|im_start|>assistant\\n"); pass the last token of that sequence as the
    marker. We return (last occurrence of marker) + 1 -- using the last (not
    first) occurrence is robust to multi-turn chats.
    """
    last_idx = -1
    for i, tok in enumerate(input_ids):
        if tok == marker_token:
            last_idx = i
    if last_idx < 0:
        raise ValueError(
            f"marker_token={marker_token} not found in input_ids "
            f"(len={len(input_ids)}). Chat template render likely broken."
        )
    return last_idx + 1


def load_sft_jsonl(path: pathlib.Path | str) -> list[dict]:
    """Read an SFT jsonl file, skipping blank lines."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sft jsonl not found: {p}")
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


class CFCamoSFTDataset(Dataset):
    """torch Dataset over an SFT jsonl. Lazy: __getitem__ does not open images;
    PIL load is deferred to the collate_fn so the dataset stays picklable for
    multiprocessing dataloaders.
    """

    def __init__(self, jsonl_path: pathlib.Path | str):
        self.rows: list[dict] = load_sft_jsonl(jsonl_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def cfcamo_sft_collate_fn(batch: list[dict], processor) -> dict:
    """Collate N rows into a Qwen3-VL SFT batch dict.

    Returns input_ids / attention_mask / labels (n, L) and the Qwen3-VL image
    tensors. Labels mask everything before the assistant response and all
    padding (-100).

    Masking (HF cookbook): per sample, render full = chat_template(messages) and
    prompt_only = chat_template(messages[:-1], add_generation_prompt=True). Both
    full and prompt_only are encoded through the multimodal processor with the
    same image, because Qwen-VL expands one image marker into many visual tokens.
    The prompt attention length is then used to set labels[:plen] = -100, and
    labels[attention_mask == 0] = -100.
    """
    if len(batch) == 0:
        raise ValueError("cfcamo_sft_collate_fn: empty batch")

    full_texts: list[str] = []
    prompt_texts: list[str] = []
    images: list = []

    for row in batch:
        msgs = build_chat_messages(row)
        full = processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        prompt = processor.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        full_texts.append(full)
        prompt_texts.append(prompt)
        # accept both {image_path} and {image} jsonl key layouts
        img_path = row.get("image_path") or row.get("image")
        if img_path is None:
            raise KeyError("row missing 'image_path' / 'image' field")
        images.append(Image.open(img_path).convert("RGB"))

    prompt_inputs = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    if "attention_mask" not in prompt_inputs:
        raise KeyError("processor output missing attention_mask for prompt-only inputs")
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

    inputs = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = inputs["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        plen = min(plen, labels.shape[1])
        labels[i, :plen] = IGNORE_INDEX
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = IGNORE_INDEX

    inputs["labels"] = labels
    return inputs
