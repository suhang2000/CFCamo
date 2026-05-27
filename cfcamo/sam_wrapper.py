"""SAM2 wrapper used by the CPR reward to turn a predicted box into a mask.

Features:
  - predict(): box / point prompts -> (mask, score, logits). Also accepts an
    optional mask_input (low-res logits) for iterative refinement.
  - set_image() embedding cache: avoids re-encoding the same image across the
    multiple box queries of one RL rollout group.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional, Tuple
from PIL import Image as PILImage
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAMWrapper:
    """Wraps the SAM2 predictor, with optional mask_input (prior logits) refinement.

    Example:
        sam = SAMWrapper(model_path="...", device="cuda:0")
        mask0, score0, logits0 = sam.predict(image, points, labels, bbox)
        mask1, score1, logits1 = sam.predict(
            image, correction_pts, correction_labels, mask_input=logits0,
        )
    """

    SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    def __init__(self, model_path: str, device: Optional[str] = None):
        """Args:
            model_path: SAM2 checkpoint path (e.g. sam2.1_hiera_large.pt)
            device: device string; auto-detected when None
        """
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        sam_model = build_sam2(self.SAM2_CFG, model_path)
        sam_model = sam_model.to(self.device)
        self.predictor = SAM2ImagePredictor(sam_model)
        self._current_image_id: Optional[str | int] = None  # logical image identity

    def predict(
        self,
        image: PILImage.Image,
        points: Optional[List[Tuple[int, int]]],
        labels: Optional[List[int]],
        bbox: Optional[List[int]] = None,
        mask_input: Optional[np.ndarray] = None,
        image_id: Optional[str | int] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run SAM2 segmentation prediction.

        Args:
            image:       input PIL image (RGB)
            points:      foreground/background points, [(x, y), ...]
            labels:      point labels, 1=foreground, 0=background
            bbox:        optional bounding box, [x1, y1, x2, y2]
            mask_input:  optional low-res logits (1, 256, 256) float32 from a
                         previous step, used as a geometric prior for refinement
            image_id:    optional logical image id; consecutive calls with the
                         same id reuse the cached image encoding (skip set_image)

        Returns:
            mask:    predicted binary mask, (H, W) bool
            score:   confidence score, float
            logits:  low-res logits (1, 256, 256) float32 for the next iteration
        """
        img_np = np.array(image.convert("RGB"))

        if image_id is None or image_id != self._current_image_id:
            self.predictor.set_image(img_np)
            self._current_image_id = image_id

        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_bbox = np.array(bbox).reshape(1, 4) if bbox is not None else None

        sam_mask_input = None
        if mask_input is not None:
            arr = np.asarray(mask_input, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]  # (H,W) → (1,H,W)
            assert arr.ndim == 3 and arr.shape[0] == 1, (
                f"mask_input must be (1,H,W), got {arr.shape}"
            )
            sam_mask_input = arr

        mask_pred, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_bbox,
            mask_input=sam_mask_input,   # optional geometric prior
            multimask_output=False,
        )

        # mask_pred: (1, H, W)   scores: (1,)   logits: (1, H', W')
        return (
            mask_pred[0].astype(bool),   # (H, W) bool
            float(scores[0]),
            logits[0:1],                 # (1, H', W') float32, keep batch dim
        )

    def predict_batch(
        self,
        images: list[np.ndarray],
        bboxes: list[list[int]],
        image_ids: list[str] | None = None,
    ) -> list[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Batched SAM2 segmentation.

        Uses SAM2ImagePredictor.set_image_batch() to encode the unique images
        once (a repeated image_id is encoded a single time and reused across its
        boxes). Returns the same format as predict(): [(mask, score, logits), ...].
        """
        if not images and not bboxes:
            return []
        if len(images) != len(bboxes):
            raise ValueError("images and bboxes must have the same length.")
        if image_ids is not None and len(image_ids) != len(images):
            raise ValueError("image_ids and images must have the same length.")

        rgb_images = [self._to_rgb_array(image) for image in images]
        group_keys = image_ids or [str(index) for index in range(len(rgb_images))]

        unique_images: list[np.ndarray] = []
        image_index_map: dict[str, int] = {}
        for key, image in zip(group_keys, rgb_images, strict=True):
            if key in image_index_map:
                continue
            image_index_map[key] = len(unique_images)
            unique_images.append(image)

        if not hasattr(self.predictor, "set_image_batch"):
            results = []
            for image, bbox, image_id in zip(rgb_images, bboxes, image_ids or [None] * len(rgb_images), strict=True):
                results.append(
                    self.predict(
                        image=PILImage.fromarray(image),
                        points=None,
                        labels=None,
                        bbox=bbox,
                        mask_input=None,
                        image_id=image_id,
                    )
                )
            return results

        self.predictor.set_image_batch(unique_images)
        self._current_image_id = None

        results: list[Tuple[np.ndarray, float, np.ndarray]] = []
        for key, bbox in zip(group_keys, bboxes, strict=True):
            image_idx = image_index_map[key]
            results.append(self._predict_from_batch_embedding(bbox=bbox, img_idx=image_idx))
        return results

    @staticmethod
    def _to_rgb_array(image: PILImage.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, PILImage.Image):
            return np.array(image.convert("RGB"))

        arr = np.asarray(image)
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        if arr.shape[2] == 4:
            return arr[..., :3]
        if arr.shape[2] != 3:
            raise ValueError(f"Unsupported channel count: {arr.shape}")
        return arr

    def _predict_from_batch_embedding(
        self,
        *,
        bbox: list[int],
        img_idx: int,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        box = np.array(bbox, dtype=np.float32).reshape(1, 4)

        if hasattr(self.predictor, "_prep_prompts") and hasattr(self.predictor, "_predict"):
            mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                None,
                None,
                box,
                None,
                True,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self.predictor._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output=False,
                return_logits=False,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            return (
                masks_np[0].astype(bool),
                float(iou_predictions_np[0]),
                low_res_masks_np[0:1],
            )

        all_masks, all_scores, all_logits = self.predictor.predict_batch(
            point_coords_batch=[None] * (img_idx + 1),
            point_labels_batch=[None] * (img_idx + 1),
            box_batch=[None] * img_idx + [box],
            mask_input_batch=[None] * (img_idx + 1),
            multimask_output=False,
            return_logits=False,
        )
        return (
            all_masks[img_idx][0].astype(bool),
            float(all_scores[img_idx][0]),
            all_logits[img_idx][0:1],
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """IoU of two binary masks."""
        pred = pred_mask.astype(bool)
        gt = (gt_mask > 128).astype(bool) if gt_mask.max() > 1 else gt_mask.astype(bool)
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def logits_to_mask(logits: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Low-res logits -> binary mask (for visualization only, not mask_input)."""
        return (logits[0] > threshold).astype(bool)
