"""Build a run's working set from MoulSot-Full without downloading the whole split.

MoulSot parquet files are split into row groups of ~800 clips (~100 MB), and
in the extra-* splits a file holds only one to a few YouTube channels. Reading one
clip costs its whole row group, so the unit of sampling is the row group:
  1. read only the small metadata columns (id, channel, pesq, ...) of the files,
  2. visit row groups round-robin across channels, in a seeded random order,
  3. download each one's audio, measure every clip's duration from its audio,
     and keep the clips that pass the filters, up to the per-channel cap,
until the target number of clips (or hours) is reached.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .. import audio as audio_utils
from ..store import Run, read_ids_file

META_COLUMNS = ["id", "channel", "pesq_hyp", "duration", "text"]
DEFAULT_DURATION = (3.0, 15.0)


def list_split_files(dataset: str, config: str, split: str) -> list[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    paths = fs.ls(f"datasets/{dataset}/{config}", detail=False)
    files = sorted(p for p in paths if Path(p).name.startswith(f"{split}-") and p.endswith(".parquet"))
    if not files:
        raise SystemExit(f"No parquet files for {dataset}/{config} split={split}")
    return files


def read_metadata(fs, files: list[str]) -> pd.DataFrame:
    """Metadata columns of every row plus its file and row group (no audio)."""
    frames = []
    for path in tqdm(files, desc="metadata"):
        with fs.open(path) as f:
            pf = pq.ParquetFile(f)
            cols = [c for c in META_COLUMNS if c in pf.schema_arrow.names]
            df = pf.read(columns=cols).to_pandas()
            sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        df["_file"] = path
        df["_rg"] = np.repeat(np.arange(len(sizes)), sizes)
        df["_local"] = np.concatenate([np.arange(s) for s in sizes])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def duration_range(sel: dict) -> tuple[float, float]:
    lo, hi = sel.get("duration") or DEFAULT_DURATION
    return float(lo), float(hi)


def eligible(meta: pd.DataFrame, sel: dict, has_reference: bool, exclude: set[str]) -> pd.DataFrame:
    """Filters that need no audio. Duration is checked again on the audio itself."""
    df = meta[~meta["id"].isin(exclude)]
    if sel.get("pesq_min") is not None and "pesq_hyp" in df:
        df = df[df["pesq_hyp"] >= sel["pesq_min"]]
    if "duration" in df:
        lo, hi = duration_range(sel)
        df = df[df["duration"].between(lo, hi)]
    if has_reference:
        df = df[df["text"].fillna("").str.strip() != ""]
    return df


def visit_order(rows: pd.DataFrame, seed: int) -> list[tuple[str, int]]:
    """Row groups ordered round-robin over channels: every channel's first group
    comes before any channel's second, so an early stop still covers many channels."""
    rng = random.Random(seed)
    groups = rows.groupby(["_file", "_rg"])["channel"].agg(lambda s: s.mode().iat[0])
    by_channel: dict[str, list] = defaultdict(list)
    for key, channel in groups.items():
        by_channel[channel].append(key)
    channels = sorted(by_channel)
    rng.shuffle(channels)
    for ch in channels:
        rng.shuffle(by_channel[ch])
    order = []
    for depth in range(max(len(v) for v in by_channel.values())):
        order += [by_channel[ch][depth] for ch in channels if depth < len(by_channel[ch])]
    return order


def prepare(cfg: dict, force: bool = False) -> None:
    from huggingface_hub import HfFileSystem

    run = Run(cfg)
    if any(run.workset_dir.glob("shard-*.parquet")) and not force:
        raise SystemExit(f"{run.workset_dir} already has shards; pass --force to rebuild it.")
    for old in run.workset_dir.glob("shard-*.parquet"):
        old.unlink()

    src, sel = cfg["source"], cfg["select"]
    has_reference = src.get("has_reference", True)
    exclude: set[str] = set()
    for p in sel.get("exclude_ids") or []:
        exclude |= read_ids_file(p)
    lo, hi = duration_range(sel)
    max_clips = sel.get("n")
    max_seconds = sel["hours"] * 3600 if sel.get("hours") else None
    if not max_clips and not max_seconds:
        raise SystemExit("select needs `n` (clips) or `hours`")
    cap = sel.get("max_per_channel")

    files = list_split_files(src["dataset"], src["config"], src["split"])
    random.Random(sel["seed"]).shuffle(files)
    if sel.get("files"):
        files = files[: sel["files"]]
    print(f"{src['dataset']}/{src['config']}/{src['split']}: {len(files)} files; "
          f"duration {lo}-{hi}s, pesq >= {sel.get('pesq_min')}, max {cap} per channel")

    fs = HfFileSystem()
    rows = eligible(read_metadata(fs, files), sel, has_reference, exclude)
    print(f"{len(rows)} clips pass the metadata filters, from {rows['channel'].nunique()} channels")

    run.workset_dir.mkdir(parents=True, exist_ok=True)
    per_channel: dict[str, int] = defaultdict(int)
    buffer: list[dict] = []
    kept = shard = 0
    seconds = 0.0
    rng = np.random.default_rng(sel["seed"])
    pbar = tqdm(total=max_clips or int(max_seconds), desc="clips" if max_clips else "seconds")

    def done() -> bool:
        return (max_clips and kept >= max_clips) or (max_seconds and seconds >= max_seconds)

    groups = {key: g for key, g in rows.groupby(["_file", "_rg"])}
    for path, rg in visit_order(rows, sel["seed"]):
        if done():
            break
        group = groups[(path, rg)]
        if cap:
            # Skip the download when every channel in this group is already full.
            group = group[group["channel"].map(lambda c: per_channel[c] < cap)]
        if group.empty:
            continue
        with fs.open(path) as f:
            table = pq.ParquetFile(f).read_row_group(int(rg), columns=["id", "audio"])
        audio_by_id = dict(zip(table.column("id").to_pylist(),
                               (a["bytes"] for a in table.column("audio").to_pylist())))
        for rec in group.iloc[rng.permutation(len(group))].to_dict("records"):
            if done():
                break
            if cap and per_channel[rec["channel"]] >= cap:
                continue
            audio_bytes = audio_by_id[rec["id"]]
            dur = audio_utils.duration(audio_bytes)
            if not lo <= dur <= hi:
                continue
            buffer.append({"id": rec["id"], "channel": rec["channel"], "duration": dur,
                           "pesq_hyp": rec.get("pesq_hyp"),
                           "reference": rec.get("text") if has_reference else None,
                           "audio": audio_bytes})
            per_channel[rec["channel"]] += 1
            kept += 1
            seconds += dur
            pbar.update(1 if max_clips else int(dur))
            if len(buffer) >= sel["shard_size"]:
                pd.DataFrame(buffer).to_parquet(run.workset_dir / f"shard-{shard:05d}.parquet", index=False)
                shard += 1
                buffer = []
    if buffer:
        pd.DataFrame(buffer).to_parquet(run.workset_dir / f"shard-{shard:05d}.parquet", index=False)
    pbar.close()

    n_channels = sum(1 for v in per_channel.values() if v)
    print(f"Working set: {kept} clips, {seconds / 3600:.2f} h, {n_channels} channels -> {run.workset_dir}")
    if not done():
        print("Warning: ran out of clips before reaching the target. Raise `files` / `max_per_channel`, "
              "widen `duration`, or lower `pesq_min`.")
    run.push("workset", f"workset for {run.name}")
