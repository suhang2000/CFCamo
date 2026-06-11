# CFCamo: A Counterfactual Detect-or-Abstain Framework for Camouflaged Object Detection

Code, benchmark, and weights for **CFCamo**.

> Vision-language reinforcement learning localizes camouflaged targets well,
> but a complementary decision is rarely tested: when an image contains *no*
> camouflaged target, should the agent still predict a box? Because standard COD
> training and evaluation data are positive-only, agents can acquire an
> **over-detect bias** that standard evaluation does not expose. To make this
> measurable we build **CF-COD**, a paired benchmark that removes the
> camouflaged target from each test image while keeping a plausible background,
> and scores whether a model detects on the original and abstains on the
> target-absent counterfactual via **Pair Accuracy (PA)**. **CFCamo** then trains
> a Qwen3-VL-4B agent with **Counterfactual Sequence Policy Optimization (CSPO)**
> and a **Counterfactual Paired Reward (CPR)** that couples original-image
> detection with counterfactual abstention.

- 📄 Paper: [arXiv:2606.11231](https://arxiv.org/abs/2606.11231)
- 🤗 Benchmark (CF-COD) + training splits: [cfcamo/CF-COD](https://huggingface.co/datasets/cfcamo/CF-COD)
- 🤗 Weights:
  [cfcamo-sft-4b](https://huggingface.co/cfcamo/cfcamo-sft-4b) (cold-start) ·
  [cfcamo-rl-lora](https://huggingface.co/cfcamo/cfcamo-rl-lora) (LoRA adapter) ·
  [cfcamo-rl-full](https://huggingface.co/cfcamo/cfcamo-rl-full) (paper main, Full FT)

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core (training + metrics)
pip install -e ".[eval]"    # adds vLLM for large-batch evaluation
```

Separately required:
- **SAM2** — install from <https://github.com/facebookresearch/sam2> and point
  `CFCAMO_SAM_PATH` to the `.pt` checkpoint (used only for the mask-IoU reward
  term; the `bbox_iou` reward variant needs no SAM).
- **EasyR1** — the RL stage runs on the external
  [EasyR1](https://github.com/hiyouga/EasyR1) framework (see *Training* below).

## Data & weights

Download from the Hugging Face Hub (links above) and arrange as below.

```
data/cfcod/
├── test/cf/{CAMO-test,CHAMELEON,COD10K-test,NC4K}/*.png   # from our HF dataset
├── train/cf/*.png                                          # from our HF dataset (COD10K-train)
├── test/cf_manifest_test.jsonl                             # CF-COD paired test
├── train/cf_manifest.jsonl                                 # RL paired-source manifest (4040)
├── sft/sft_balanced.jsonl                                  # SFT cold-start (1000 rows)
├── CAMO-test/{Imgs,GT}/...                                 # from upstream COD release
├── CHAMELEON/{Imgs,GT}/...
├── COD10K-test/{Imgs,GT}/...
├── COD10K-train/{Imgs,GT}/...                              # for SFT/RL training
└── NC4K/{Imgs,GT}/...
checkpoints/
├── cfcamo-sft-4b/                                          # SFT cold-start (init for RL)
├── cfcamo-rl-lora/                                         # RL LoRA adapter (paper §5)
└── cfcamo-rl-full/                                         # CFCamo-4B (Full FT), paper main
```

The CF (target-removed) images are produced by an off-the-shelf inpainter
(ObjectClear) from the original CAMO / COD10K / CHAMELEON / NC4K images. The
upstream COD images and masks are **not redistributed here** — fetch them from
their original sources (see the consolidated pointer at
<https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod>)
and respect each dataset's terms.

## Quick start (single image)

```bash
python scripts/eval/infer.py \
  --model checkpoints/cfcamo-rl-full \
  --image path/to/image.jpg \
  --save-overlay out.png
```

Prints the predicted box(es) in `[0,1000]` coordinates, or reports abstention
(`<no_camouflage/>`) when no camouflaged object is found.

## Training

**Stage 1 — SFT cold-start** (single GPU):

```bash
python scripts/train/sft_train.py \
  --train-jsonl data/cfcod/sft/sft_balanced.jsonl \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --output-dir checkpoints/cfcamo-sft-4b \
  --epochs 1 --batch-size 2 --grad-accum 8 --lr 2e-5
```

**Stage 2 — CSPO/CPR RL** (via EasyR1). Point EasyR1 at a config; the reward is
wired through `configs/easyr1_reward_wrapper.py:compute_score`:

```bash
# LoRA (single large GPU) -- paper main checkpoint at step 252 (epsilon=0.5)
python -m verl.trainer.main config=configs/rl_lora.yaml
# Full fine-tuning (multi-GPU) -- checkpoint at step 126 (epsilon=0.5)
python -m verl.trainer.main config=configs/rl_full.yaml
```

Merge a LoRA adapter into a standalone HF model:

```bash
python scripts/eval/merge_lora.py \
  --base checkpoints/cfcamo-sft-4b \
  --lora checkpoints/rl_lora/global_step_252/actor/lora_adapter \
  --out checkpoints/cfcamo-rl-lora
```

## Evaluation (CF-COD, paper Table II)

```bash
python scripts/eval/eval_cfcod.py \
  --pair-source first-n --pair-n 2352 \
  --cf-manifest data/cfcod/test/cf_manifest_test.jsonl \
  --data-root data/cfcod \
  --models "CFCamo=checkpoints/cfcamo-rl-full,Base=Qwen/Qwen3-VL-4B-Instruct" \
  --out-dir results/cfcod_eval
```

Reports FormatValid / Orig→Detect / CF→Abstain / **Pair Accuracy** / OrigIoU.

**Section 5.4 ablation (reward without SAM mask).** The default CPR refines the
predicted box into a mask with SAM2 and scores mask IoU; set
`CFCAMO_REWARD_VARIANT=bbox_iou` for the box-vs-box IoU variant (no SAM, ~30%
faster training).

## Reward (CPR)

`cfcamo/reward.py` + `cfcamo/easyr1_reward.py`. For a paired sample
`(x_o, x_c)`:

| term | x_o (detect) | x_c (abstain) |
|---|---|---|
| correct | `+1 + IoU(pred, GT)` (+0.1 if schema-valid) | `+2` (+0.1 if schema-valid) |
| wrong | `-1` (abstains) | `-1` (detects) |
| no commit | `0` | `0` |

Plus a coupling bonus `+eta` (default `1.0`) added to **both** members iff
`x_o` detects **and** `x_c` abstains.

## Repository layout

```
cfcamo/            # library: data, parser, reward (CPR), eval metrics, SAM wrapper
configs/           # prompt template + RL configs (LoRA / Full FT) + reward wrapper
scripts/train/     # sft_train.py
scripts/eval/      # eval_cfcod.py, infer.py, merge_lora.py
```

## Citation

```bibtex
@article{li2026cfcamo,
  title   = {{CFCamo}: A Counterfactual Detect-or-Abstain Framework for Camouflaged Object Detection},
  author  = {Li, Suhang and Yoshie, Osamu and Ieiri, Yuya},
  journal = {arXiv preprint arXiv:2606.11231},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
