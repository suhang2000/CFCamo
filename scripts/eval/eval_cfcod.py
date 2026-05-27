#!/usr/bin/env python3
"""N-model paired CF-COD evaluation (Pair Accuracy + IoU).

Evaluates one or more models on the CF-COD paired benchmark: each test image and
its target-absent counterfactual are scored for Det(y_o) and Abs(y_c); the
paper metrics (FormatValid / Orig->Detect / CF->Abstain / Pair Accuracy /
OrigIoU) come from cfcamo.eval_metrics.

Pair source:
  --pair-source first-n  take the first --pair-n pairs from --cf-manifest
  --pair-source probe    take an ID list from --probe-results

Models:
  --models "name1=path1,name2=path2,..."   (each gets its own vLLM init, run
  sequentially to avoid vLLM multi-model deadlocks).

Usage:
  python scripts/eval/eval_cfcod.py \\
    --pair-source first-n --pair-n 2352 \\
    --cf-manifest data/cfcod/test/cf_manifest_test.jsonl \\
    --models "CFCamo-LoRA=checkpoints/cfcamo-rl-lora,Base=Qwen/Qwen3-VL-4B-Instruct" \\
    --out-dir results/cfcod_eval
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import re
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))  # fallback if cfcamo is not pip-installed
from cfcamo.data import CFCAMO_SYSTEM_PROMPT, CFCAMO_USER_PROMPT  # noqa: E402
from cfcamo.eval_metrics import compute_metrics  # noqa: E402

DETECTION_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process should enclosed within <think> </think> tags, and the bounding box, points and points labels should be enclosed within <bbox></bbox>, <points></points>, and <labels></labels>, respectively. i.e., "
    "<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, <points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>"
    "Where 1 indicates a foreground (object) point, and 0 indicates a background point."
)
DETECTION_USER_PROMPT = (
    "Identify and locate the camouflaged object in the image.\n"
    "Output the result using the exact format:\n"
    "<think> reasoning process here </think> <bbox>[x1,y1,x2,y2]</bbox>, "
    "<points>[[x3,y3],[x4,y4],...]</points>, <labels>[1,0,...]</labels>"
)

# Force-detect prompt (no abstain option): forces a box for the standard COD
# benchmark (Table I, S-measure etc.), where every image contains a target.
CFCAMO_SYSTEM_PROMPT_DETECT = (
    "You are a camouflaged object detector. There IS a camouflaged object in this image. "
    "Locate it precisely.\n\n"
    "Output in this exact format:\n"
    "<think>your reasoning here</think>\n"
    "followed by ONE of:\n"
    "  - <bbox>[x1,y1,x2,y2]</bbox>  for a single camouflaged object\n"
    "  - <bbox>[[x1,y1,x2,y2],[x3,y3,x4,y4]]</bbox>  for multiple objects\n\n"
    "Coordinates are normalized to [0, 1000] where 1000 = full image dimension."
)
CFCAMO_USER_PROMPT_DETECT = (
    "Identify and locate the camouflaged object in the image.\n\n"
    "In <think></think>, briefly consider scene textures and visual anomalies, then output:\n"
    "- <bbox>[x1,y1,x2,y2]</bbox> for one object, or [[x1,y1,x2,y2],...] for multiple"
)

PROMPT_CONFIGS = {
    # paper-final detect-or-abstain prompt (CF-COD paired eval, Table II)
    "cfcamo": (CFCAMO_SYSTEM_PROMPT, CFCAMO_USER_PROMPT),
    # force-detect (no abstain) for the standard COD benchmark (Table I)
    "detect": (CFCAMO_SYSTEM_PROMPT_DETECT, CFCAMO_USER_PROMPT_DETECT),
    # Seg-R1 baseline native prompt (points + labels schema)
    "detection": (DETECTION_SYSTEM_PROMPT, DETECTION_USER_PROMPT),
}

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{sys_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{usr_prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def safe_filename(name: str) -> str:
    """Model name → filesystem-safe stem (e.g. 'Seg-R1' → 'seg_r1')."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def parse_models_arg(arg: str) -> list[tuple[str, str]]:
    """'name1=path1,name2=path2' → [(name1, path1), ...]. Comma in path is not supported."""
    out = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--models entry must be 'name=path', got: {part!r}")
        name, path = part.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def load_pairs(args: argparse.Namespace) -> list[dict]:
    """Build the pair list from probe_results (50 anchors) or cf_manifest first-n / all-pairs.

    Manifest paths (image/mask/cf) are joined against args.data_root, so the
    released manifest can ship with portable relative paths.
    """
    if args.pair_source == "probe":
        target_ids = [json.loads(l)["id"] for l in open(args.probe_results)]
    elif args.all_pairs:
        target_ids = []
        for ln in open(args.cf_manifest):
            r = json.loads(ln)
            if r.get("has_cf"):
                target_ids.append(r["id"])
    else:  # first-n
        target_ids = []
        for ln in open(args.cf_manifest):
            r = json.loads(ln)
            if r.get("has_cf"):
                target_ids.append(r["id"])
            if len(target_ids) >= args.pair_n:
                break
    cf_paths = {json.loads(l)["id"]: json.loads(l) for l in open(args.cf_manifest)}
    pairs = []
    for tid in target_ids:
        m = cf_paths.get(tid)
        if not m:
            print(f"  [skip] {tid}: not in cf_manifest")
            continue
        pairs.append({
            "id": tid,
            "source": m.get("source", "unknown"),
            "image": str(args.data_root / m["image"]),
            "cf": str(args.data_root / m["cf"]),
        })
    return pairs


def _run_one_prompt_pass(llm, sampling, name: str, pairs: list[dict],
                          out_path: pathlib.Path, batch: int,
                          prompt_mode: str, resume: bool) -> None:
    """Inner loop for a single prompt pass — assumes vLLM is already loaded.

    Uses PrefetchIterator to overlap CPU image load with GPU vLLM forward
    (eliminates ~1.6s GPU-idle gap per batch; see cfcamo/eval_prefetch.py).
    """
    from cfcamo.eval_prefetch import PrefetchIterator

    done_ids = set()
    if resume and out_path.is_file():
        for ln in open(out_path):
            try:
                done_ids.add(json.loads(ln)["id"])
            except Exception:
                pass
    pairs_todo = [p for p in pairs if p["id"] not in done_ids]
    if not pairs_todo:
        print(f"  [skip {prompt_mode}] {out_path} already complete ({len(done_ids)} pairs)")
        return
    if done_ids:
        print(f"  [resume {prompt_mode}] skip {len(done_ids)}, todo {len(pairs_todo)}")

    sys_p, usr_p = PROMPT_CONFIGS[prompt_mode]
    full_prompt = PROMPT_TEMPLATE.format(sys_prompt=sys_p, usr_prompt=usr_p)
    pair_per_batch = max(1, batch // 2)
    chunks = [pairs_todo[i: i + pair_per_batch]
              for i in range(0, len(pairs_todo), pair_per_batch)]
    fout = open(out_path, "a" if done_ids else "w", buffering=1)
    t1 = time.time()
    n_done = 0
    with PrefetchIterator(chunks, full_prompt) as prefetch:
        for prompts, meta in prefetch:
            if not prompts:
                continue
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            per_pair: dict[str, dict] = {}
            for out, (sid, source, kind) in zip(outputs, meta):
                per_pair.setdefault(sid, {"id": sid, "source": source})
                per_pair[sid][kind + "_response"] = out.outputs[0].text
            for sid, pr in per_pair.items():
                if "orig_response" in pr and "cf_response" in pr:
                    fout.write(json.dumps(pr, ensure_ascii=False) + "\n")
                    n_done += 1
            elapsed = time.time() - t1
            rate = n_done / max(0.01, elapsed)
            eta = (len(pairs_todo) - n_done) / max(0.01, rate)
            print(f"  [{prompt_mode} {n_done:4d}/{len(pairs_todo)} todo] {rate:.2f} pair/s "
                  f"elapsed={elapsed:.1f}s eta={eta/60:.1f}min", flush=True)
    fout.close()
    print(f"  [done {prompt_mode}] {n_done} pairs in {time.time()-t1:.1f}s -> {out_path}")


def run_model_eval(name: str, model_path: str, pairs: list[dict],
                   out_dir: pathlib.Path, batch: int = 16,
                   max_tokens: int = 512, gpu_mem: float = 0.85,
                   resume: bool = True, max_model_len: int = 12288,
                   max_pixels: int = 768 * 768,
                   prompt_modes: list[str] | None = None) -> None:
    """Load the vLLM model once + run N prompt passes (saves cold start). Frees GPU via del when done.

    max_model_len=12288: large images (high-resolution NC4K) can have ~8000+ image tokens, leave headroom.
    max_pixels=768*768: limits the vLLM image processor, caps image token count at ~750.
    prompt_modes: ["cfcamo"] / ["detection"] / ["cfcamo", "detection"] (vLLM reused)
    """
    if prompt_modes is None:
        prompt_modes = ["cfcamo"]

    # All-skip short-circuit
    all_done = True
    for mode in prompt_modes:
        suffix = f"_{mode}" if mode != "cfcamo" else ""
        out_path = out_dir / f"{safe_filename(name)}{suffix}_results.jsonl"
        if not out_path.is_file():
            all_done = False
            break
        done_ids = set()
        for ln in open(out_path):
            try:
                done_ids.add(json.loads(ln)["id"])
            except Exception:
                pass
        if len(done_ids) < len(pairs):
            all_done = False
            break
    if all_done and resume:
        print(f"\n[skip-all] {name}: all {prompt_modes} prompt(s) complete")
        return

    from vllm import LLM, SamplingParams

    print(f"\n========== {name} : {model_path} (modes: {prompt_modes}) ==========")
    t0 = time.time()
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"min_pixels": 28 * 28 * 4, "max_pixels": max_pixels},
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    print(f"  [load] {time.time()-t0:.1f}s")

    for mode in prompt_modes:
        suffix = f"_{mode}" if mode != "cfcamo" else ""
        out_path = out_dir / f"{safe_filename(name)}{suffix}_results.jsonl"
        _run_one_prompt_pass(llm, sampling, name, pairs, out_path, batch, mode, resume)

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    time.sleep(3)


def render_table(rows: list[tuple[str, dict]]) -> str:
    has_iou = any("orig_iou_avg" in m for _, m in rows)
    cols = [
        ("n", "n"),
        ("format_valid_rate", "FormatValid"),
        ("orig_detect_rate", "Orig→Detect (↑)"),
        ("orig_refuse_rate", "Orig→Refuse (↓)"),
        ("cf_refuse_rate", "CF→Refuse (↑)"),
        ("cf_false_detect_rate", "CF→FalseDet (↓)"),
        ("pair_success_rate", "PairSuccess (↑)"),
    ]
    if has_iou:
        cols.append(("orig_iou_avg", "OrigIoU (↑)"))
        cols.append(("pair_iou_avg", "PairIoU (↑)"))
    header = "| Model | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for name, m in rows:
        cells = [name]
        for k, _ in cols:
            v = m.get(k, "—")
            if isinstance(v, float):
                cells.append(f"{v*100:.1f}%" if k != "n" else f"{int(v)}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair-source", choices=["probe", "first-n"], default="first-n")
    p.add_argument("--pair-n", type=int, default=500)
    p.add_argument("--probe-results", type=pathlib.Path, default=None,
                   help="ID list for --pair-source probe (optional)")
    p.add_argument("--cf-manifest", required=True, type=pathlib.Path)
    p.add_argument("--data-root", default=pathlib.Path("data/cfcod"),
                   type=pathlib.Path,
                   help="Root joined with manifest image/mask/cf relative paths "
                        "(default: data/cfcod).")
    p.add_argument("--models", required=True, type=str,
                   help="'name1=path1,name2=path2,...'")
    p.add_argument("--out-dir", required=True, type=pathlib.Path)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--no-resume", action="store_true",
                   help="Force re-eval even if output jsonl exists")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip eval, only aggregate existing jsonls into table")
    p.add_argument("--per-dataset", action="store_true",
                   help="Split aggregation by source field (CAMO/COD10K), used for paper Table 2")
    p.add_argument("--all-pairs", action="store_true",
                   help="Skip --pair-n, use all pairs with has_cf=True (paper Table 2: run full CAMO 250 + COD10K test split)")
    p.add_argument("--prompt-mode",
                   choices=["cfcamo", "detect", "detection"],
                   default="cfcamo",
                   help="cfcamo (PAPER-FINAL, default; detect-or-abstain, CF-COD "
                        "paired eval) / detect (force-detect, no abstain, for the "
                        "standard COD S-measure benchmark) / detection (Seg-R1 "
                        "baseline native points+labels prompt)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models_arg(args.models)
    print(f"[models] {len(models)} model(s):")
    for name, path in models:
        print(f"  - {name}: {path}")

    pairs = load_pairs(args)
    print(f"[pairs] {len(pairs)} pairs from {args.pair_source}")

    # 1. Eval each model (output filenames get the prompt mode as a suffix)
    prompt_modes = [args.prompt_mode]

    if not args.aggregate_only:
        for name, path in models:
            run_model_eval(name, path, pairs, args.out_dir,
                           batch=args.batch, max_tokens=args.max_tokens,
                           gpu_mem=args.gpu_mem, resume=not args.no_resume,
                           prompt_modes=prompt_modes)

    # 2. Build mask_loader from cf_manifest
    cf_paths = {json.loads(l)["id"]: json.loads(l) for l in open(args.cf_manifest)}

    def mask_loader(sample_id: str):
        m = cf_paths.get(sample_id)
        if not m or "mask" not in m:
            return None
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(args.data_root / m["mask"])
            if img.mode != "L":
                img = img.convert("L")
            return np.array(img) >= 128
        except Exception:
            return None

    # 3. Aggregate (per prompt_mode)
    print("\n========== Metrics ==========\n")

    for cur_mode in prompt_modes:
        suffix = f"_{cur_mode}" if cur_mode != "cfcamo" else ""
        print(f"\n----- Aggregate prompt={cur_mode} -----")

        if args.per_dataset:
            all_sources = sorted(set(p["source"] for p in pairs))
            print(f"[per-dataset] sources: {all_sources} (prompt={cur_mode})")
            for src in all_sources:
                print(f"\n--- Source: {src} (prompt={cur_mode}) ---")
                rows: list[tuple[str, dict]] = []
                for name, path in models:
                    base_jsonl = args.out_dir / f"{safe_filename(name)}{suffix}_results.jsonl"
                    if not base_jsonl.is_file():
                        print(f"  [skip] {name}: {base_jsonl} missing")
                        continue
                    src_jsonl = args.out_dir / f"{safe_filename(name)}{suffix}_{src.lower()}_results.jsonl"
                    with open(base_jsonl) as fin, open(src_jsonl, "w") as fout:
                        for ln in fin:
                            r = json.loads(ln)
                            if r.get("source") == src:
                                fout.write(ln)
                    rows.append((name, compute_metrics(src_jsonl, mask_loader)))
                table = render_table(rows)
                print(table)
                summary_path = args.out_dir / f"summary{suffix}_{src.lower()}.md"
                summary_path.write_text(
                    f"# Per-dataset paired eval — Source: {src}, prompt: {cur_mode}\n\n"
                    f"T=0 greedy. IoU vs GT mask.\n\n"
                    f"{table}\n\n"
                    f"Sources:\n" +
                    "\n".join(f"- {name}: `{path}`" for name, path in models)
                )
                print(f"  [summary] -> {summary_path}")
        else:
            rows: list[tuple[str, dict]] = []
            for name, path in models:
                out_path = args.out_dir / f"{safe_filename(name)}{suffix}_results.jsonl"
                if out_path.is_file():
                    rows.append((name, compute_metrics(out_path, mask_loader)))
                else:
                    print(f"  [skip aggregate] {name}: {out_path} missing")
            table = render_table(rows)
            print(table)
            summary_path = args.out_dir / f"summary{suffix}.md"
            summary_path.write_text(
                f"# N-model paired eval ({args.pair_source}, {len(pairs)} pair, prompt={cur_mode})\n\n"
                f"T=0 greedy. IoU vs GT mask (where available).\n\n"
                f"{table}\n\n"
                f"Sources:\n" +
                "\n".join(f"- {name}: `{path}`" for name, path in models)
            )
            print(f"\n[summary] -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
