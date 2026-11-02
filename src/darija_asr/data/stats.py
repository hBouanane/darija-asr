"""Look at a split before choosing selection settings.

Metadata stats cover every sampled file (cheap). Durations need the audio, so
they are measured on a few random row groups (~100 MB download each).
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import audio as audio_utils
from .select import duration_range, list_split_files, read_metadata

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _quantiles(s: pd.Series) -> str:
    return "  ".join(f"p{int(q * 100)}={v:.2f}" for q, v in s.quantile(QUANTILES).items())


def stats(cfg: dict, sample_groups: int = 3) -> None:
    from huggingface_hub import HfFileSystem

    src, sel = cfg["source"], cfg["select"]
    files = list_split_files(src["dataset"], src["config"], src["split"])
    rng = random.Random(sel["seed"])
    rng.shuffle(files)
    if sel.get("files"):
        files = files[: sel["files"]]
    fs = HfFileSystem()
    meta = read_metadata(fs, files)

    per_channel = meta["channel"].value_counts()
    print(f"\n{src['config']}/{src['split']}: {len(meta)} clips in {len(files)} files, {len(per_channel)} channels")
    print(f"clips per channel: {_quantiles(per_channel.astype(float))}")
    print("largest channels: " + ", ".join(f"{c} {n}" for c, n in per_channel.head(5).items()))
    if "pesq_hyp" in meta:
        print(f"pesq: {_quantiles(meta['pesq_hyp'])}")
        for t in (2.5, 3.0, 3.5):
            print(f"  pesq >= {t}: {(meta['pesq_hyp'] >= t).mean():.0%}")

    lo, hi = duration_range(sel)
    if "duration" in meta:
        durations = meta["duration"]
        source = "duration column"
    else:
        keys = list(meta.groupby(["_file", "_rg"]).groups)
        picked = rng.sample(keys, min(sample_groups, len(keys)))
        durations = []
        for path, rg in picked:
            with fs.open(path) as f:
                table = pq.ParquetFile(f).read_row_group(int(rg), columns=["audio"])
            durations += [audio_utils.duration(a["bytes"]) for a in table.column("audio").to_pylist()]
        durations = pd.Series(durations)
        source = f"audio of {len(picked)} sampled row groups, {len(durations)} clips"
    print(f"duration (s) from {source}: {_quantiles(durations)}")
    print(f"  in current range {lo}-{hi}s: {durations.between(lo, hi).mean():.0%}")
    hist, edges = np.histogram(durations.clip(upper=30), bins=[0, 1, 2, 3, 5, 8, 12, 15, 20, 25, 30])
    for n, a, b in zip(hist, edges[:-1], edges[1:]):
        print(f"  {a:>4.0f}-{b:<4.0f}s {'#' * int(50 * n / max(hist.max(), 1)):<50} {n}")
