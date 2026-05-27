"""Counterfactual Paired Reward (CPR) for CFCamo RL.

A paired sample is (x_o original, x_c target-absent counterfactual). The reward
couples target-present detection on x_o with target-absent abstention on x_c:

  detect reward  r_det(x_o):     correct detect  -> +1 + IoU(pred, GT)  (+0.1 if schema-valid)
                                 wrong abstain    -> -1
                                 no commit        ->  0
  abstain reward r_abs(x_c):     correct abstain  -> +2                 (+0.1 if schema-valid)
                                 wrong detect     -> -1
                                 no commit        ->  0
  coupling bonus r_cpl:          +eta to BOTH members iff (x_o detect AND x_c abstain)
  total = r_det/r_abs + r_cpl

Design notes:
  - bbox coords are Qwen-native 0~1000 canonical (NOT 0~1).
  - The +2 abstain magnitude matches the max detect reward (1 + IoU(=1) + 0.1 = 2.1),
    so always-detect and always-abstain strategies have equal expected reward; only
    the paired coupling bonus breaks the tie toward the correct joint behavior.
  - IoU(pred, GT): the main setting refines the predicted box into a mask with a
    frozen SAM2 and scores mask IoU; the ``bbox_iou`` variant (Section 5.4 ablation)
    scores box-vs-box IoU directly, skipping SAM for ~30% faster training.

The batched EasyR1 entry point lives in ``cfcamo.easyr1_reward.compute_score``.
"""
from __future__ import annotations

import numpy as np

from cfcamo.parser import parse_response

_COORD_MAX = 1000.0

# Coupling weight (eta). The paper uses 1.0; override per-run via the
# CFCAMO_ETA env var read in cfcamo.easyr1_reward.
DEFAULT_ETA = 1.0


def _mask_to_canonical_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Binary mask (H, W) -> bbox in 0~1000 canonical coords, or None if empty.

    Used only by the bbox_iou reward variant (Section 5.4 ablation).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    H, W = mask.shape
    x1 = float(xs.min()) * _COORD_MAX / max(W, 1)
    y1 = float(ys.min()) * _COORD_MAX / max(H, 1)
    x2 = float(xs.max()) * _COORD_MAX / max(W, 1)
    y2 = float(ys.max()) * _COORD_MAX / max(H, 1)
    return int(x1), int(y1), int(x2), int(y2)


def bbox_iou_canonical(
    pred_bbox: tuple[int, int, int, int] | None,
    gt_bbox: tuple[int, int, int, int] | None,
) -> float:
    """Axis-aligned bbox IoU on canonical 0-1000 coords.

    Used by the bbox_iou reward variant (Section 5.4 ablation): replaces the
    SAM-based mask IoU with direct bbox IoU -- no GPU/SAM forward.
    None inputs -> 0.0 (caller pre-validates degenerate boxes).
    """
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    px1, py1, px2, py2 = [float(v) for v in pred_bbox]
    gx1, gy1, gx2, gy2 = [float(v) for v in gt_bbox]
    ix1 = max(px1, gx1); iy1 = max(py1, gy1)
    ix2 = min(px2, gx2); iy2 = min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_to_mask_iou(bbox_or_list, mask: np.ndarray) -> float:
    """IoU of bbox(es) in 0~1000 canonical coords with a binary mask.

    Accepts a single (x1,y1,x2,y2) tuple or a list of such tuples (multi-target
    COD). Boxes are clipped to [0, 1000]; inverted boxes are skipped; an empty
    mask or zero union returns 0. Used by the COD bbox-IoU evaluation metric.
    """
    if (
        isinstance(bbox_or_list, tuple)
        and len(bbox_or_list) == 4
        and all(isinstance(c, (int, np.integer)) for c in bbox_or_list)
    ):
        bboxes = [bbox_or_list]
    else:
        bboxes = list(bbox_or_list)

    if not bboxes or mask.size == 0:
        return 0.0

    H, W = mask.shape
    pred_mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in bboxes:
        x1 = max(0, min(int(_COORD_MAX), int(x1)))
        y1 = max(0, min(int(_COORD_MAX), int(y1)))
        x2 = max(0, min(int(_COORD_MAX), int(x2)))
        y2 = max(0, min(int(_COORD_MAX), int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        px1 = int(x1 * W / _COORD_MAX)
        py1 = int(y1 * H / _COORD_MAX)
        px2 = int(x2 * W / _COORD_MAX)
        py2 = int(y2 * H / _COORD_MAX)
        pred_mask[py1:py2, px1:px2] = True

    inter = int(np.logical_and(pred_mask, mask).sum())
    union = int(np.logical_or(pred_mask, mask).sum())
    if union == 0:
        return 0.0
    return inter / union


def compute_detect_reward_with_mask(
    parse,
    gt_mask: np.ndarray,
    pred_mask_uint8: np.ndarray | None,
) -> tuple[float, dict]:
    """Detect reward on x_o (target present); SAM-refined mask IoU.

    Caller passes the precomputed SAM mask (batched SAM forward upstream).

      refuse:                 -1
      no commit:               0
      detect, malformed bbox: +1
      detect, valid bbox:     +1 + mask_iou (+0.1 if schema-valid)
    """
    fmt = 1.0 if parse.format_valid else 0.0
    metrics = {
        "mask_iou": 0.0, "is_refuse": 0.0, "is_invalid": 0.0,
        "is_detect": 0.0, "bbox_invalid": 0.0, "fmt": fmt,
    }

    if parse.kind == "refuse":
        metrics["is_refuse"] = 1.0
        return -1.0, metrics

    if parse.kind != "detect" or not parse.bboxes:
        metrics["is_invalid"] = 1.0
        return 0.0, metrics

    metrics["is_detect"] = 1.0

    if parse.bbox_out_of_range or parse.bbox_inverted:
        metrics["bbox_invalid"] = 1.0
        return 1.0, metrics

    if pred_mask_uint8 is None:
        metrics["bbox_invalid"] = 1.0
        return 1.0, metrics

    from cfcamo.sam_reward_helper import mask_iou as _mask_iou
    iou = _mask_iou(pred_mask_uint8, gt_mask)
    metrics["mask_iou"] = iou
    base = 1.0 + iou
    if fmt > 0.5:
        base += 0.1
    return base, metrics


def compute_detect_reward_bbox_iou(
    parse,
    gt_mask: np.ndarray,
) -> tuple[float, dict]:
    """Detect reward on x_o, bbox_iou variant (Section 5.4 ablation, no SAM).

    Same reward structure as ``compute_detect_reward_with_mask`` but the quality
    term is box-vs-box IoU (GT box extracted from the mask) instead of SAM mask IoU.
    """
    fmt = 1.0 if parse.format_valid else 0.0
    metrics = {
        "mask_iou": 0.0, "is_refuse": 0.0, "is_invalid": 0.0,
        "is_detect": 0.0, "bbox_invalid": 0.0, "fmt": fmt,
    }
    if parse.kind == "refuse":
        metrics["is_refuse"] = 1.0
        return -1.0, metrics
    if parse.kind != "detect" or not parse.bboxes:
        metrics["is_invalid"] = 1.0
        return 0.0, metrics
    metrics["is_detect"] = 1.0
    if parse.bbox_out_of_range or parse.bbox_inverted:
        metrics["bbox_invalid"] = 1.0
        return 1.0, metrics
    gt_bbox = _mask_to_canonical_bbox(gt_mask)
    iou = max(bbox_iou_canonical(tuple(pb), gt_bbox) for pb in parse.bboxes) if gt_bbox else 0.0
    metrics["mask_iou"] = iou  # keep key name for metric-reporting consistency
    base = 1.0 + iou
    if fmt > 0.5:
        base += 0.1
    return base, metrics


def compute_abstain_reward(text: str) -> tuple[float, dict]:
    """Abstain reward on x_c (target absent); expect <no_camouflage/>.

      abstain:        +2 (+0.1 if schema-valid)
      detect (false): -1
      no commit:       0

    +2 matches the max detect reward (1 + IoU(=1) + 0.1), so always-detect and
    always-abstain have equal expected reward; the coupling bonus breaks the tie.
    """
    p = parse_response(text)
    fmt = 1.0 if p.format_valid else 0.0
    metrics = {
        "is_refuse": 0.0, "is_detect": 0.0, "is_invalid": 0.0, "fmt": fmt,
    }
    if p.kind == "refuse":
        metrics["is_refuse"] = 1.0
        base = 2.0
        if fmt > 0.5:
            base += 0.1
        return base, metrics
    if p.kind == "detect":
        metrics["is_detect"] = 1.0
        return -1.0, metrics
    metrics["is_invalid"] = 1.0
    return 0.0, metrics
