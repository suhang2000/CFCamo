"""EasyR1 reward_function entry point for CFCamo (CPR).

EasyR1 loads this file as a standalone module via importlib. We make the
``cfcamo`` package importable (normally via ``pip install -e .``; the sys.path
insert is a fallback for a non-installed checkout) and re-export the batch
reward.

In the RL config:
  worker.reward.reward_function: configs/easyr1_reward_wrapper.py:compute_score

Reward behavior is controlled by env vars read in cfcamo.easyr1_reward
(CFCAMO_ETA, CFCAMO_REWARD_VARIANT, CFCAMO_SAM_PATH).
"""
import pathlib
import sys

# Fallback: make the repo root importable if cfcamo is not pip-installed.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cfcamo.easyr1_reward import (  # noqa: E402
    REWARD_NAME,
    REWARD_TYPE,
    compute_score,
    encode_ground_truth,
    load_gt_mask_binary,
)

__all__ = [
    "REWARD_NAME",
    "REWARD_TYPE",
    "compute_score",
    "encode_ground_truth",
    "load_gt_mask_binary",
]
