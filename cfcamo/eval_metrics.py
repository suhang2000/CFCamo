"""CFCamo eval metrics — paired binary metrics + optional bbox-vs-mask IoU.

  - compute_metrics(jsonl_path): binary metrics only.
  - compute_metrics(jsonl_path, mask_loader=fn): also orig_iou_avg + pair_iou_avg,
    where mask_loader(id) -> np.ndarray (binary mask) or None (skip IoU).

Metrics:
  - orig_iou_avg: mean IoU over orig samples with kind=detect.
  - pair_iou_avg: mean IoU over pair-success samples (orig detect AND cf refuse).
  pair_success_rate is schema-level; the IoU averages add a precision dimension.

Notes:
  - a None from mask_loader skips that sample's IoU (e.g. a test set without masks);
  - orig refuse has no box, so no IoU;
  - the IoU averages return 0.0 (not nan) when no sample qualifies.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Optional

import numpy as np

from cfcamo.parser import parse_response
from cfcamo.reward import bbox_to_mask_iou

MaskLoader = Callable[[str], Optional[np.ndarray]]


def compute_metrics(jsonl_path: pathlib.Path,
                    mask_loader: MaskLoader | None = None) -> dict[str, Any]:
    """Compute paired eval metrics from results jsonl.

    Each row in jsonl: {id, orig_response, cf_response, ...}.

    Args:
        jsonl_path: path to *_results.jsonl
        mask_loader: callable(id) → np.ndarray binary mask or None.
                     If provided, additionally compute orig_iou_avg + pair_iou_avg.

    Returns:
        dict with:
          - n: pair count
          - format_valid_rate, orig_detect_rate, orig_refuse_rate,
            cf_refuse_rate, cf_false_detect_rate, pair_success_rate (schema-level)
          - when mask_loader is provided:
            - orig_iou_avg: mean IoU over orig samples with kind=detect
            - pair_iou_avg: mean IoU over pair-success samples
            - orig_iou_count: number of orig-detect samples that had a mask
            - pair_iou_count: number of pair-success samples that had a mask
    """
    rows = []
    for ln in open(jsonl_path):
        s = ln.strip()
        if s:
            rows.append(json.loads(s))
    n = len(rows)
    if n == 0:
        return {"n": 0}

    n_orig_detect = n_orig_refuse = 0
    n_cf_detect = n_cf_refuse = 0
    n_orig_format_valid = n_cf_format_valid = 0
    n_pair_success = 0
    sum_orig_iou = sum_pair_iou = 0.0
    n_orig_iou = n_pair_iou = 0

    for r in rows:
        po = parse_response(r["orig_response"])
        pc = parse_response(r["cf_response"])
        if po.kind == "detect": n_orig_detect += 1
        elif po.kind == "refuse": n_orig_refuse += 1
        if pc.kind == "detect": n_cf_detect += 1
        elif pc.kind == "refuse": n_cf_refuse += 1
        if po.format_valid: n_orig_format_valid += 1
        if pc.format_valid: n_cf_format_valid += 1
        is_pair_success = (po.kind == "detect" and pc.kind == "refuse")
        if is_pair_success:
            n_pair_success += 1

        if mask_loader is not None and po.kind == "detect" and po.bboxes:
            mask = mask_loader(r["id"])
            if mask is not None:
                iou = bbox_to_mask_iou(po.bboxes, mask)
                sum_orig_iou += iou
                n_orig_iou += 1
                if is_pair_success:
                    sum_pair_iou += iou
                    n_pair_iou += 1

    out: dict[str, Any] = {
        "n": n,
        "format_valid_rate": (n_orig_format_valid + n_cf_format_valid) / (2 * n),
        "orig_detect_rate": n_orig_detect / n,
        "orig_refuse_rate": n_orig_refuse / n,
        "cf_refuse_rate": n_cf_refuse / n,
        "cf_false_detect_rate": n_cf_detect / n,
        "pair_success_rate": n_pair_success / n,
    }
    if mask_loader is not None:
        out["orig_iou_avg"] = sum_orig_iou / max(1, n_orig_iou)
        out["pair_iou_avg"] = sum_pair_iou / max(1, n_pair_iou)
        out["orig_iou_count"] = n_orig_iou
        out["pair_iou_count"] = n_pair_iou
    return out
