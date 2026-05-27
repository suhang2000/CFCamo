"""EasyR1/verl batch reward adapter for CFCamo (CPR).

Drop this file in as the EasyR1 ``reward_function`` module:
  REWARD_NAME = "cfcamo"
  REWARD_TYPE = "batch"
  def compute_score(reward_inputs: list[dict]) -> list[dict]: ...

Data convention (one row per sample, ``ground_truth`` is a JSON string):
  {"kind": "orig"|"cf", "gt_mask_path": ..., "image_path": ..., "pair_id": ...}
  - orig rows need image_path (for the SAM mask) and gt_mask_path.
  - pair_id links an orig row with its counterfactual row -> enables the
    coupling bonus (rollouts paired by in-batch order).

Reward: see cfcamo.reward (CPR = detect reward on x_o + abstain reward on x_c
+ paired coupling bonus).

Env knobs:
  CFCAMO_ETA              coupling weight eta (default 1.0; paper value).
  CFCAMO_REWARD_VARIANT   "" (default, SAM mask IoU) or "bbox_iou"
                          (Section 5.4 ablation: box-vs-box IoU, no SAM).
  CFCAMO_SAM_PATH         SAM2 checkpoint path (see cfcamo.sam_reward_helper).
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
from PIL import Image

from cfcamo.parser import parse_response
from cfcamo.reward import (
    compute_abstain_reward,
    compute_detect_reward_bbox_iou,
    compute_detect_reward_with_mask,
)

REWARD_NAME = "cfcamo"
REWARD_TYPE = "batch"

_VALID_KINDS = {"orig", "cf"}

# Coupling weight (eta). Paper uses 1.0. Read once at import (not per-sample).
_ETA = float(os.environ.get("CFCAMO_ETA", "1.0"))

# Quality term for the detect reward:
#   ""        -> SAM-refined mask IoU (paper main path)
#   bbox_iou  -> box-vs-box IoU, no SAM (~30% faster; Section 5.4 ablation)
_REWARD_VARIANT = os.environ.get("CFCAMO_REWARD_VARIANT", "").lower()
if _REWARD_VARIANT not in ("", "bbox_iou"):
    raise ValueError(f"CFCAMO_REWARD_VARIANT must be ''/'bbox_iou', got {_REWARD_VARIANT!r}")


def encode_ground_truth(
    kind: str,
    gt_mask_path: str | None,
    image_path: str | None = None,
    pair_id: str | None = None,
) -> str:
    """Build the EasyR1 ``ground_truth`` JSON string (used in dataset prep)."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    out = {"kind": kind, "gt_mask_path": gt_mask_path}
    if image_path is not None:
        out["image_path"] = image_path
    if pair_id is not None:
        out["pair_id"] = pair_id
    return json.dumps(out)


def load_gt_mask_binary(path: str) -> np.ndarray:
    """COD GT mask (grayscale 0~255) -> bool array (threshold 128)."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img)
    return arr >= 128


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """CPR batch reward: per-sample detect/abstain reward + paired coupling.

    Returns one dict per input with ``overall`` (used for the advantage) plus
    diagnostic metrics (format_valid / is_refuse / is_detect / is_orig /
    coupling / mask_iou).
    """
    # 1a. Decode + parse all inputs (no SAM yet).
    parsed_list: list[dict] = []
    for inp in reward_inputs:
        response = inp["response"]
        try:
            gt = json.loads(inp["ground_truth"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"bad ground_truth: {inp.get('ground_truth')!r}") from exc
        kind = gt.get("kind")
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
        parsed_list.append({
            "kind": kind,
            "pair_id": gt.get("pair_id"),
            "response": response,
            "parse": parse_response(response),
            "gt": gt,
            "base": 0.0,
            "extra": {},
        })

    # 1b. Collect orig+detect samples -> one batched SAM forward -> distribute
    #     the predicted masks. Skipped entirely for the bbox_iou variant (no SAM).
    pred_masks: dict[int, np.ndarray] = {}
    if _REWARD_VARIANT != "bbox_iou":
        from cfcamo.sam_reward_helper import sam_predict_masks_batch
        batch_requests: list[tuple[str, list]] = []
        batch_idx_map: list[int] = []
        for i, p in enumerate(parsed_list):
            if p["kind"] != "orig":
                continue
            parse = p["parse"]
            if parse.kind != "detect" or not parse.bboxes:
                continue
            if parse.bbox_out_of_range or parse.bbox_inverted:
                continue
            image_path = p["gt"].get("image_path")
            if not image_path:
                raise ValueError("orig sample missing image_path (SAM mask reward needs it)")
            batch_requests.append((image_path, parse.bboxes))
            batch_idx_map.append(i)
        if batch_requests:
            batch_masks = sam_predict_masks_batch(batch_requests)
            for batch_pos, m in enumerate(batch_masks):
                pred_masks[batch_idx_map[batch_pos]] = m

    # 1c. Per-sample base reward.
    for i, p in enumerate(parsed_list):
        parse = p["parse"]
        if p["kind"] == "orig":
            mask_path = p["gt"].get("gt_mask_path")
            if not mask_path:
                raise ValueError("orig sample missing gt_mask_path")
            gt_mask = load_gt_mask_binary(mask_path)
            if _REWARD_VARIANT == "bbox_iou":
                base, extra = compute_detect_reward_bbox_iou(parse, gt_mask)
            else:
                base, extra = compute_detect_reward_with_mask(parse, gt_mask, pred_masks.get(i))
            p["extra"] = extra
            p["base"] = float(base)
        else:  # cf
            base, extra = compute_abstain_reward(p["response"])
            p["extra"] = extra
            p["base"] = float(base)

    # 2. Coupling: pair orig/cf rollouts by pair_id (in-batch order). Reward the
    #    ideal joint behavior (orig detect AND cf abstain) with +eta on both.
    coupling_bonus: list[float] = [0.0] * len(parsed_list)
    pair_groups: dict[str, dict[str, list[int]]] = {}
    for i, p in enumerate(parsed_list):
        if p["pair_id"] is None:
            continue
        g = pair_groups.setdefault(p["pair_id"], {"orig": [], "cf": []})
        g[p["kind"]].append(i)
    for g in pair_groups.values():
        # Paper-main data is 1:1 (equal orig/cf rollouts per pair_id). Unequal
        # counts are paired by in-batch order; any remainder gets no coupling.
        if len(g["orig"]) != len(g["cf"]):
            warnings.warn(
                f"pair_id has {len(g['orig'])} orig vs {len(g['cf'])} cf "
                "rollouts; coupling pairs by order and drops the remainder.",
                stacklevel=2,
            )
        for orig_i, cf_i in zip(g["orig"], g["cf"]):
            if parsed_list[orig_i]["parse"].kind == "detect" and \
               parsed_list[cf_i]["parse"].kind == "refuse":
                coupling_bonus[orig_i] = _ETA
                coupling_bonus[cf_i] = _ETA

    # 3. Final per-sample output: base + coupling.
    out: list[dict[str, float]] = []
    for i, p in enumerate(parsed_list):
        out.append({
            "overall": p["base"] + coupling_bonus[i],
            "format_valid": 1.0 if p["parse"].format_valid else 0.0,
            "is_refuse": 1.0 if p["parse"].kind == "refuse" else 0.0,
            "is_detect": 1.0 if p["parse"].kind == "detect" else 0.0,
            "is_orig": 1.0 if p["kind"] == "orig" else 0.0,
            "coupling": coupling_bonus[i],
            "mask_iou": float(p["extra"].get("mask_iou", 0.0)),
        })
    return out
