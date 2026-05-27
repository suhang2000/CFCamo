"""SAM forward helper for the mask-level RL reward.

A lazy singleton SAM instance per process, pinned to the local-rank GPU, so each
RL actor runs its own SAM. SAMWrapper caches the image embedding by image_id, so
the multiple box queries for one prompt's rollouts share a single set_image pass.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

_SAM = None
# SAM2 checkpoint path. Override with the CFCAMO_SAM_PATH env var, or place
# the .pt under ./checkpoints/ (see README for the download link).
_SAM_PATH_DEFAULT = "checkpoints/sam2.1_hiera_large.pt"


def get_sam(device: Optional[str] = None, model_path: Optional[str] = None):
    """Lazy SAM init, one instance per process.

    The RL launcher creates each reward actor with
    runtime_env = build_reward_runtime_env(gpu_id) which sets:
      - CUDA_VISIBLE_DEVICES=<gpu_id>
      - CFCAMO_SAM_DEVICE=cuda:0
      - RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0  (so Ray doesn't strip the above)
    So in normal use this fn just reads CFCAMO_SAM_DEVICE.

    Device selection:
      1. Explicit `device` arg (highest priority, used in tests)
      2. env CFCAMO_SAM_DEVICE (set by launcher, "cuda:0")
      3. f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() (fallback)
      4. "cpu" if no cuda (last resort, warn)
    """
    global _SAM
    if _SAM is not None:
        return _SAM

    if device is None:
        device = os.environ.get("CFCAMO_SAM_DEVICE")
    if device is None:
        try:
            import torch
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device = f"cuda:{local_rank}"
            else:
                device = "cpu"
                print(
                    "[sam_reward_helper] WARNING: torch.cuda not available + "
                    "CFCAMO_SAM_DEVICE not set. Falling back to CPU (~50× slower). "
                    "Likely launcher missing runtime_env / RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0.",
                    flush=True,
                )
        except ImportError:
            device = "cpu"

    if model_path is None:
        model_path = os.environ.get("CFCAMO_SAM_PATH", _SAM_PATH_DEFAULT)

    from cfcamo.sam_wrapper import SAMWrapper
    print(f"[sam_reward_helper] init SAM on device={device}", flush=True)
    _SAM = SAMWrapper(model_path=model_path, device=device)
    return _SAM


def sam_predict_mask_from_bboxes(
    image_path: str,
    bboxes: list[tuple[int, int, int, int]],
    coord_max: int = 1000,
) -> np.ndarray:
    """Per-sample bbox→mask SAM (kept for tests + eval scripts not yet batched).

    Multi-bbox: one SAM call per box, masks combined by element-wise max (union).
    set_image caches by image_path, reusing the embedding across a box's queries.
    Out-of-range / inverted boxes are filtered by the caller, not here.

    For batch RL training, prefer sam_predict_masks_batch() (5-15× faster).
    """
    sam = get_sam()
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    merged: Optional[np.ndarray] = None
    for x1, y1, x2, y2 in bboxes:
        bb_pix = [
            float(x1) * W / coord_max,
            float(y1) * H / coord_max,
            float(x2) * W / coord_max,
            float(y2) * H / coord_max,
        ]
        mask_bool, _score, _logits = sam.predict(
            image=img, points=None, labels=None,
            bbox=bb_pix, image_id=image_path,
        )
        m = mask_bool.astype(np.uint8) * 255
        merged = m if merged is None else np.maximum(merged, m)

    if merged is None:
        return np.zeros((H, W), dtype=np.uint8)
    return merged


def sam_predict_masks_batch(
    requests: list[tuple[str, list[tuple[int, int, int, int]]]],
    coord_max: int = 1000,
) -> list[np.ndarray]:
    """Batched bbox→mask SAM via SAMWrapper.predict_batch().

    Args:
        requests: list of (image_path, bboxes_list). Multi-bbox per request handled
                  via per-bbox SAM call + union.
        coord_max: bbox coord canonical max (1000 for Qwen3-VL).

    Returns:
        list of uint8 mask (H_i, W_i) in same order as requests, one per request.
        Multi-bbox in one request → union mask.

    Performance: vs per-sample sam_predict_mask_from_bboxes, this calls
    predictor.set_image_batch() once for all unique images (vs N calls), 5-15×
    faster on typical RL batch (16 prompt × 8 rollout = 128 samples on 16 unique
    images).

    Algorithm:
      1. Flatten requests → list of (image_path, single_bbox)
      2. SAMWrapper.predict_batch() — set_image_batch on unique images +
         _predict_from_batch_embedding per bbox
      3. Re-aggregate per-request via union (max).
    """
    if not requests:
        return []

    sam = get_sam()

    img_cache: dict[str, "Image.Image"] = {}
    def _img(p):
        if p not in img_cache:
            img_cache[p] = Image.open(p).convert("RGB")
        return img_cache[p]

    # Pre-fetch image sizes for bbox → pixel coord
    sizes: dict[str, tuple[int, int]] = {}
    for path, _ in requests:
        if path not in sizes:
            im = _img(path)
            sizes[path] = (im.size[0], im.size[1])  # (W, H)

    # Flatten to single-bbox calls + remember mapping back to requests
    flat_images: list = []
    flat_bboxes: list[list[float]] = []
    flat_image_ids: list[str] = []
    bbox_to_req_idx: list[int] = []  # which request does this flat bbox belong to

    for req_idx, (path, bboxes) in enumerate(requests):
        if not bboxes:
            continue
        W, H = sizes[path]
        for x1, y1, x2, y2 in bboxes:
            bb_pix = [
                float(x1) * W / coord_max,
                float(y1) * H / coord_max,
                float(x2) * W / coord_max,
                float(y2) * H / coord_max,
            ]
            flat_images.append(np.array(_img(path)))
            flat_bboxes.append(bb_pix)
            flat_image_ids.append(path)
            bbox_to_req_idx.append(req_idx)

    if not flat_images:
        return [np.zeros((sizes[p][1], sizes[p][0]), dtype=np.uint8)
                for p, _ in requests]

    # Batched SAM forward
    flat_results = sam.predict_batch(flat_images, flat_bboxes, image_ids=flat_image_ids)

    # Re-aggregate per request via union (max)
    out_masks: list[Optional[np.ndarray]] = [None] * len(requests)
    for flat_idx, (mask_bool, _score, _logits) in enumerate(flat_results):
        req_idx = bbox_to_req_idx[flat_idx]
        m = mask_bool.astype(np.uint8) * 255
        out_masks[req_idx] = m if out_masks[req_idx] is None else np.maximum(out_masks[req_idx], m)

    # Fill empty requests (no bboxes) with zero mask of correct size
    for i, (path, bboxes) in enumerate(requests):
        if out_masks[i] is None:
            W, H = sizes[path]
            out_masks[i] = np.zeros((H, W), dtype=np.uint8)

    # Release cached features + CUDA cache to keep per-process GPU memory
    # bounded across RL steps.
    _release_sam_cached_features(sam)

    return out_masks


def _release_sam_cached_features(sam) -> None:
    """Release SAM2 predictor cached image features + PyTorch CUDA cache.

    set_image_batch() embeddings would otherwise accumulate across RL steps.
    Catches all exceptions and never raises (the reward fn must not fail).
    """
    try:
        pred = sam.predictor
        for attr in ("_features", "_orig_hw"):
            if hasattr(pred, attr):
                setattr(pred, attr, None)
        for attr in ("_is_image_set", "_is_batch"):
            if hasattr(pred, attr):
                setattr(pred, attr, False)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def mask_iou(pred_mask_uint8: np.ndarray, gt_mask_bool: np.ndarray) -> float:
    """IoU of pred (uint8 0/255) vs GT mask (bool). Resize pred to gt shape if mismatch."""
    if pred_mask_uint8.shape != gt_mask_bool.shape:
        pred_mask_uint8 = np.array(
            Image.fromarray(pred_mask_uint8).resize(
                (gt_mask_bool.shape[1], gt_mask_bool.shape[0]), Image.NEAREST
            ),
            dtype=np.uint8,
        )
    pred_bool = pred_mask_uint8 > 127
    inter = int(np.logical_and(pred_bool, gt_mask_bool).sum())
    union = int(np.logical_or(pred_bool, gt_mask_bool).sum())
    if union == 0:
        return 0.0
    return inter / union
