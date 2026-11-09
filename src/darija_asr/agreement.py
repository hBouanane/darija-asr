"""Compare transcripts of the same clip and route it: accept / llm / drop.

Distances are symmetric normalized Levenshtein (edits / longer length) on the
scoring-normalized text: character level ("cer") and word level ("wer").
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd
from rapidfuzz.distance import Levenshtein

from .normalize import latin_ratio, normalize
from .store import Run, latest_per_id, read_parts, read_workset


def pair_distance(a: str, b: str) -> tuple[float, float]:
    cer = Levenshtein.normalized_distance(a, b)
    wer = Levenshtein.normalized_distance(a.split(), b.split())
    return cer, wer


def route(texts: dict[str, str], accept_cer: float, drop_cer: float) -> tuple[str, dict]:
    """texts: source -> normalized text (empty strings = missing)."""
    present = {k: v for k, v in texts.items() if v}
    stats: dict = {"n_sources": len(present)}
    if len(present) < 2:
        return "drop", stats
    cers = []
    for a, b in combinations(sorted(present), 2):
        cer, wer = pair_distance(present[a], present[b])
        stats[f"cer_{a}_{b}"] = cer
        stats[f"wer_{a}_{b}"] = wer
        cers.append(cer)
    stats["max_cer"], stats["min_cer"] = max(cers), min(cers)
    stats["mean_cer"] = sum(cers) / len(cers)
    if stats["max_cer"] <= accept_cer and len(present) == len(texts):
        return "accept", stats
    if stats["min_cer"] > drop_cer:
        return "drop", stats
    return "llm", stats


def load_sources(run: Run, sources: list[str]) -> pd.DataFrame:
    """One row per working-set clip, one raw-text column per source."""
    df = read_workset(run, with_audio=False)[["id", "channel", "duration", "reference"]]
    for src in sources:
        if src == "reference":
            continue
        parts = latest_per_id(read_parts(run.asr_dir(src)))
        if parts.empty:
            raise SystemExit(f"No transcripts for {src!r} in {run.asr_dir(src)}; run `darija transcribe` first.")
        df = df.merge(parts[["id", "text"]].rename(columns={"text": src}), on="id", how="left")
    return df


def compute_agreement(cfg: dict) -> pd.DataFrame:
    run = Run(cfg)
    run.pull_for_judging()  # transcripts from every machine, without the audio
    acfg = cfg["agreement"]
    sources = acfg["sources"]
    df = load_sources(run, sources)
    rows = []
    for rec in df.to_dict("records"):
        normed = {s: normalize(rec.get(s)) for s in sources}
        decision, stats = route(normed, acfg["accept_cer"], acfg["drop_cer"])
        rows.append({"id": rec["id"], "route": decision, **stats,
                     **{f"latin_{s}": latin_ratio(rec.get(s)) for s in sources}})
    out = df[["id", "channel", "duration"]].merge(pd.DataFrame(rows), on="id")
    out.to_parquet(run.agreement_path, index=False)
    run.push("agreement.parquet")

    counts = out["route"].value_counts()
    print(f"{len(out)} clips -> " + ", ".join(f"{k}: {v} ({v / len(out):.0%})" for k, v in counts.items()))
    pair_cols = [c for c in out.columns if c.startswith("cer_")]
    if pair_cols:
        print("Median pairwise CER:\n" + out[pair_cols].median().round(3).to_string())
    return out
