"""Background-thread image prefetch for vLLM eval pipeline.

Used by scripts/eval/eval_cfcod.py to overlap CPU image load with GPU
vLLM forward. Eliminates the ~1.6s GPU-idle gap per batch (16 PIL.open +
decode sequentially) under the original sync loop.

Why threading not multiprocessing/DataLoader:
  - vLLM has already loaded the model to GPU; multiprocessing fork would
    inherit CUDA context (Linux default `fork` start method) and risk
    silent deadlock or duplicate VRAM allocation.
  - PIL.Image.open releases the GIL during disk IO + decode → real concurrency.
  - A single prefetch thread is sufficient: vLLM forward takes 5-10s/batch,
    chunk load takes ~1.6s, so 1 background worker fully hides the gap.

Typical usage:
    chunks = [pairs[i:i+B] for i in range(0, len(pairs), B)]
    with PrefetchIterator(chunks, full_prompt) as prefetch:
        for prompts, meta in prefetch:
            if not prompts:
                continue
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            ...
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image


def load_chunk_images(chunk: list[dict], full_prompt: str) -> tuple[list[dict], list[tuple]]:
    """Load PIL images for a chunk of (orig, cf) pairs and build vLLM prompts.

    Args:
        chunk: List of pair dicts with keys "id", "source", "image" (orig path),
            "cf" (cf path). Same shape as cf_manifest_test.jsonl entries.
        full_prompt: The vLLM prompt template string (already formatted with
            system + user prompts).

    Returns:
        (prompts, meta) where
          prompts = [{"prompt": str, "multi_modal_data": {"image": PIL.Image}}, ...]
          meta    = [(id, source, kind), ...]   kind ∈ {"orig", "cf"}
        Both lists have the same length (2 × len(chunk) if all images load OK).
        Broken images are skipped with stderr log; their pair partner still loads.
    """
    prompts: list[dict] = []
    meta: list[tuple] = []
    for r in chunk:
        for kind, p in [("orig", r["image"]), ("cf", r["cf"])]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as exc:
                print(f"  [skip] {r['id']} {kind}: {exc}", flush=True)
                continue
            prompts.append({"prompt": full_prompt, "multi_modal_data": {"image": img}})
            meta.append((r["id"], r["source"], kind))
    return prompts, meta


class PrefetchIterator:
    """Yields (prompts, meta) for each chunk; prefetches next in background.

    Context manager (recommended) — guarantees ThreadPoolExecutor shutdown:

        with PrefetchIterator(chunks, full_prompt) as it:
            for prompts, meta in it:
                ...  # GPU forward; meanwhile next chunk's load_chunk_images
                     # is running in the worker thread

    Properties:
      - Order-preserving: chunk N+1 is yielded only after chunk N is consumed.
      - Single worker thread (max_workers=1) — sufficient to hide ~1.6s load
        behind ~5-10s vLLM forward. More workers add CPU contention without
        further GPU-idle reduction.
      - Exceptions raised by load_chunk_images surface on the consumer's
        next() call (Future.result() re-raises).
    """

    def __init__(self, chunks: list[list[dict]], full_prompt: str):
        self._chunks = list(chunks)  # snapshot
        self._full_prompt = full_prompt
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._idx = 0
        self._next_future: Any = None
        # Kick off first prefetch immediately
        if self._chunks:
            self._next_future = self._executor.submit(
                load_chunk_images, self._chunks[0], full_prompt
            )

    def __iter__(self):
        return self

    def __next__(self) -> tuple[list[dict], list[tuple]]:
        if self._idx >= len(self._chunks):
            raise StopIteration
        # Wait for current chunk to finish loading
        current = self._next_future.result()  # re-raises exceptions
        self._idx += 1
        # Submit next chunk's load (if any) so it runs while consumer processes
        # the chunk we're about to return
        if self._idx < len(self._chunks):
            self._next_future = self._executor.submit(
                load_chunk_images, self._chunks[self._idx], self._full_prompt
            )
        else:
            self._next_future = None
        return current

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # wait=False: don't block on the in-flight load if consumer aborted
        self._executor.shutdown(wait=False, cancel_futures=True)
        return False  # don't suppress exceptions
