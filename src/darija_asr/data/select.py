"""Build a run's working set from MoulSot-Full without downloading the whole split.

The source parquet files are grouped by YouTube channel, so reading the first N
rows would give one channel. Instead:
  1. read only the small metadata columns of the sampled files (HTTP range reads),
  2. filter and sample rows across files and channels,
  3. read the audio column only for the row groups that hold selected rows.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .. import audio as audio_utils
from ..store import Run, read_ids_file

META_COLUMNS = ["id", "channel", "pesq_hyp", "duration", "text"]


def list_split_files(dataset: str, config: str, split: str) -> list[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    paths = fs.ls(f"datasets/{dataset}/{config}", detail=False)
    files = sorted(p for p in paths if Path(p).name.startswith(f"{split}-") and p.endswith(".parquet"))
    if not files:
        raise SystemExit(f"No parquet files for {dataset}/{config} split={split}")
    return files


def _read_metadata(fs, files: list[str]) -> pd.DataFrame:
    frames = []
    for path in tqdm(files, desc="metadata"):
        with fs.open(path) as f:
            pf = pq.ParquetFile(f)
            cols = [c for c in META_COLUMNS if c in pf.schema_arrow.names]
            df = pf.read(columns=cols).to_pandas()
        df["_file"] = path
        df["_row"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _sample(meta: pd.DataFrame, sel: dict, has_reference: bool, exclude: set[str]) -> pd.DataFrame:
    df = meta[~meta["id"].isin(exclude)]
    if sel.get("min_pesq") is not None and "pesq_hyp" in df:
        df = df[df["pesq_hyp"] >= sel["min_pesq"]]
    if "duration" in df:
        df = df[df["duration"].between(sel["min_duration"], sel["max_duration"])]
    if has_reference:
        df = df[df["text"].fillna("").str.strip() != ""]
    df = df.sample(frac=1.0, random_state=sel["seed"])
    if sel.get("max_per_channel"):
        df = df.groupby("channel", sort=False).head(sel["max_per_channel"])
    # Without a duration column some picks fail the duration filter after the
    # audio is read, so take a margin.
    margin = 1.0 if "duration" in meta else 1.25
    return df.head(int(sel["n"] * margin))


def prepare(cfg: dict, force: bool = False) -> None:
    from huggingface_hub import HfFileSystem

    run = Run(cfg)
    if run.workset_dir.exists() and any(run.workset_dir.glob("shard-*.parquet")) and not force:
        raise SystemExit(f"{run.workset_dir} already has shards; pass --force to rebuild it.")
    for old in run.workset_dir.glob("shard-*.parquet"):
        old.unlink()

    src, sel = cfg["source"], cfg["select"]
    has_reference = src.get("has_reference", True)
    exclude: set[str] = set()
    for p in sel.get("exclude_ids") or []:
        exclude |= read_ids_file(p)

    files = list_split_files(src["dataset"], src["config"], src["split"])
    rng = random.Random(sel["seed"])
    rng.shuffle(files)
    if sel.get("files"):
        files = files[: sel["files"]]
    print(f"Sampling from {len(files)} parquet files of {src['dataset']}/{src['config']}/{src['split']}")

    fs = HfFileSystem()
    picks = _sample(_read_metadata(fs, files), sel, has_reference, exclude)
    print(f"Picked {len(picks)} candidate clips from {picks['channel'].nunique()} channels")

    run.workset_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[dict] = []
    written = shard = 0
    target = sel["n"]
    pbar = tqdm(total=target, desc="audio")
    for path, group in picks.groupby("_file", sort=False):
        if written + len(buffer) >= target:
            break
        records = {r["id"]: r for r in group.to_dict("records")}
        with fs.open(path) as f:
            pf = pq.ParquetFile(f)
            starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
            row_idx = np.sort(group["_row"].values)
            rg_of_row = np.searchsorted(starts, row_idx, side="right") - 1
            for rg in np.unique(rg_of_row):
                local = (row_idx[rg_of_row == rg] - starts[rg]).tolist()
                table = pf.read_row_group(int(rg), columns=["id", "audio"]).take(local)
                for rid, aud in zip(table.column("id").to_pylist(), table.column("audio").to_pylist()):
                    record = records[rid]
                    audio_bytes = aud["bytes"]
                    dur = record.get("duration")
                    if dur is None or pd.isna(dur):
                        dur = audio_utils.duration(audio_bytes)
                    if not (sel["min_duration"] <= dur <= sel["max_duration"]):
                        continue
                    buffer.append({
                        "id": rid,
                        "channel": record["channel"],
                        "duration": float(dur),
                        "pesq_hyp": record.get("pesq_hyp"),
                        "reference": record.get("text") if has_reference else None,
                        "audio": audio_bytes,
                    })
                    pbar.update(1)
                    if written + len(buffer) >= target:
                        break
                    if len(buffer) >= sel["shard_size"]:
                        pd.DataFrame(buffer).to_parquet(run.workset_dir / f"shard-{shard:05d}.parquet", index=False)
                        written += len(buffer)
                        shard += 1
                        buffer = []
                if written + len(buffer) >= target:
                    break
    if buffer:
        pd.DataFrame(buffer).to_parquet(run.workset_dir / f"shard-{shard:05d}.parquet", index=False)
        written += len(buffer)
    pbar.close()
    print(f"Working set: {written} clips in {run.workset_dir}")
    run.push("workset", f"workset for {run.name}")
