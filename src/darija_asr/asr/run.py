"""Resumable transcription of a run's working set with one ASR model.

Every `flush_every` clips a parquet part is written (and pushed to the hub), so
a Colab disconnect loses at most one part. Rerunning skips finished ids.
Use --shard/--num-shards to split one run across several Colab sessions.
"""
from __future__ import annotations

import hashlib
import time

import pandas as pd
from tqdm import tqdm

from .. import audio as audio_utils
from ..store import Run, done_ids, read_workset, write_part
from . import load_backend


def in_shard(clip_id: str, shard: int, num_shards: int) -> bool:
    return int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % num_shards == shard


def transcribe(cfg: dict, model: str, shard: int = 0, num_shards: int = 1, limit: int | None = None) -> None:
    run = Run(cfg)
    out_dir = run.asr_dir(model)
    # With hub_repo set, start from the shared state: clips added by `prepare --add`
    # elsewhere, and this model's parts written by other machines/sessions.
    run.pull("workset", quiet=True)
    run.pull(f"asr/{model}", quiet=True)
    meta = read_workset(run, with_audio=False)
    finished = done_ids(out_dir)
    todo = meta[~meta["id"].isin(finished)]
    todo = todo[[in_shard(i, shard, num_shards) for i in todo["id"]]]
    if limit:
        todo = todo.head(limit)
    todo_ids = set(todo["id"])
    print(f"[{model}] {len(finished)} done, {len(todo)} to go (shard {shard}/{num_shards})")
    if not todo_ids:
        return

    model_cfg = cfg["asr"]["models"][model]
    backend = load_backend(model, model_cfg)
    flush_every = int(cfg["asr"].get("flush_every", 200))
    rows: list[dict] = []
    audio_seconds = compute_seconds = 0.0

    unpushed: list = []  # parts whose upload failed go up with the next one

    def flush():
        nonlocal rows
        if rows:
            unpushed.append(write_part(pd.DataFrame(rows), out_dir))
            if run.push_files(unpushed, f"{model}: +{len(rows)} (shard {shard}/{num_shards})"):
                unpushed.clear()
            rows = []

    pbar = tqdm(total=len(todo_ids), desc=model)
    # One working-set shard in memory at a time keeps RAM flat on Colab.
    for shard_file in sorted(run.workset_dir.glob("shard-*.parquet")):
        part = pd.read_parquet(shard_file)
        part = part[part["id"].isin(todo_ids)]
        # Similar lengths per batch means less padding.
        records = part.sort_values("duration", ascending=False).to_dict("records")
        for start in range(0, len(records), backend.batch_size):
            batch = records[start:start + backend.batch_size]
            arrays = [audio_utils.decode(r["audio"]) for r in batch]
            t0 = time.time()
            try:
                texts, error = backend.transcribe(arrays), None
            except Exception as exc:  # noqa: BLE001 - record and continue; rerun retries it
                texts, error = [None] * len(batch), repr(exc)[:500]
            elapsed = time.time() - t0
            audio_seconds += sum(r["duration"] for r in batch)
            compute_seconds += elapsed
            for r, text in zip(batch, texts):
                rows.append({"id": r["id"], "text": text, "error": error, "seconds": elapsed / len(batch)})
            pbar.update(len(batch))
            pbar.set_postfix(rtfx=f"{audio_seconds / max(compute_seconds, 1e-6):.1f}")
            if len(rows) >= flush_every:
                flush()
    flush()
    pbar.close()


def smoke(cfg: dict, model: str, n: int = 5) -> None:
    """Transcribe a few clips and print them next to the reference, without saving."""
    run = Run(cfg)
    ws = pd.read_parquet(sorted(run.workset_dir.glob("shard-*.parquet"))[0]).head(n)
    backend = load_backend(model, cfg["asr"]["models"][model])
    t0 = time.time()
    texts = backend.transcribe([audio_utils.decode(a) for a in ws["audio"]])
    elapsed = time.time() - t0
    for rec, text in zip(ws.itertuples(), texts):
        print(f"\n--- {rec.id} ({rec.duration:.1f}s)\nREF: {rec.reference}\nHYP: {text}")
    print(f"\nRTFx {ws['duration'].sum() / elapsed:.1f} on {n} clips")
