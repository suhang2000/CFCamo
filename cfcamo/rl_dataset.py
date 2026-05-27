"""CFCamo RL dataset builder: pair-aware flatten keeps (orig, cf) adjacent in a batch.

The coupling reward is computed within a batch (eta * 1[orig detect AND cf
abstain]), and the EasyR1 batch reward fn only sees the current batch, so an orig
and its cf (same pair_id) must land in the same batch. A standard shuffle would
split pairs, so we shuffle at the pair level and flatten with (orig, cf) kept
adjacent; any dataloader batch holding whole pair blocks then has complete pairs.
Requires data.shuffle: false in the YAML.
"""
from __future__ import annotations

import json
import random
from typing import Iterable

import sys
import pathlib
_THIS = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))  # make the repo root importable
from cfcamo.data import CFCAMO_USER_PROMPT  # noqa: E402


def encode_ground_truth_with_pair_id(kind: str, gt_mask_path: str | None,
                                      pair_id: str,
                                      image_path: str | None = None) -> str:
    """JSON-encode ground_truth field for EasyR1 (includes pair_id, vs old encode_ground_truth).

    EasyR1 reward fn reads pair_id from the ground_truth field for cross-sample coupling.

    image_path: required on orig samples (SAM mask reward needs the original image);
    cf sample optional (cf reward does not call SAM).

    Schema:
      {"kind": "orig"|"cf", "gt_mask_path": str|None, "pair_id": str,
       "image_path": str|None}
    """
    if kind not in {"orig", "cf"}:
        raise ValueError(f"kind must be 'orig' or 'cf', got {kind!r}")
    out = {
        "kind": kind,
        "gt_mask_path": gt_mask_path,
        "pair_id": pair_id,
    }
    if image_path is not None:
        out["image_path"] = image_path
    return json.dumps(out)


def _pair_to_samples(row: dict, orig_repeat: int = 1) -> list[dict]:
    """1 cf_manifest row → (orig_repeat orig + 1 cf) RL samples.

    Field schema:
      id:      "{pair_id}__{kind}[__{rep_idx}]"  (unique sample id)
      pair_id: cf_manifest row id                (used for cross-sample pairing)
      kind:    "orig" | "cf"
      problem: CFCAMO_USER_PROMPT
      answer:  JSON encoded ground_truth (includes pair_id)
      images:  [path]                  (orig uses the image field, cf uses the cf field)
      source:  dataset source (CAMO/COD10K/...)

    When orig_repeat>1, the orig sample is duplicated N times (used for the dataset 2:1 ablation).
    rep_idx is only added to orig (cf has 1 copy; no suffix, to stay compatible with old ids).
    """
    pair_id = row["id"]
    out = []
    for rep_i in range(orig_repeat):
        suffix = f"__{rep_i}" if orig_repeat > 1 else ""
        out.append({
            "id":      f"{pair_id}__orig{suffix}",
            "pair_id": pair_id,
            "kind":    "orig",
            "problem": CFCAMO_USER_PROMPT,
            # image_path is used by the SAM mask reward
            "answer":  encode_ground_truth_with_pair_id(
                "orig", row["mask"], pair_id, image_path=row["image"],
            ),
            "images":  [row["image"]],
            "source":  row.get("source", "unknown"),
        })
    out.append({
        "id":      f"{pair_id}__cf",
        "pair_id": pair_id,
        "kind":    "cf",
        "problem": CFCAMO_USER_PROMPT,
        "answer":  encode_ground_truth_with_pair_id("cf", None, pair_id),
        "images":  [row["cf"]],
        "source":  row.get("source", "unknown"),
    })
    return out


def pair_aware_flatten(pairs: Iterable[dict], seed: int = 42,
                       require_has_cf: bool = True,
                       orig_repeat: int = 1) -> list[dict]:
    """cf_manifest pair list → flat sample list, with the same pair_id adjacent.

    Args:
        pairs: list of cf_manifest rows (each has id, image, mask, cf, has_cf, source)
        seed: shuffle the pair-level order (leaves the intra-pair layout intact)
        require_has_cf: skip pairs with has_cf=False (no cf image)
        orig_repeat: number of orig sample copies (default 1; T6 ablation uses 2 → 2:1 orig:cf).
                     When >1, layout is [o,o,...,cf, o,o,...,cf,...] block_size=orig_repeat+1.
                     Coupling can still zip(orig_indices, cf_indices) over the min length.

    Returns:
        list of samples, len = (orig_repeat + 1) * valid pair count.

    Invariant: intra-pair layout is fixed; a batch holding a whole number of pair
    blocks therefore holds complete pairs.
    """
    if not isinstance(orig_repeat, int) or orig_repeat < 1:
        raise ValueError(f"orig_repeat must be int >= 1, got {orig_repeat!r}")

    valid_pairs = []
    for row in pairs:
        if require_has_cf and not row.get("has_cf"):
            continue
        valid_pairs.append(row)

    rng = random.Random(seed)
    rng.shuffle(valid_pairs)  # only shuffle the pair-level order

    flat: list[dict] = []
    for row in valid_pairs:
        flat.extend(_pair_to_samples(row, orig_repeat=orig_repeat))
    return flat
