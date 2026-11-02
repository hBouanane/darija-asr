"""Score every system against the human-checked gold set.

Systems found automatically in the gold run: `reference` (Gemini), every ASR
model that transcribed it, and `merged` (the final labels). Add any other
model's predictions with --hyp name=path.parquet (columns: id, text), e.g. a
fine-tuned Qwen3-ASR checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.gold import load_gold
from ..store import Run, latest_per_id, read_parts, read_workset
from .metrics import score


def collect_systems(run: Run, extra: dict[str, str]) -> dict[str, dict[str, str]]:
    systems: dict[str, dict[str, str]] = {}
    ws = read_workset(run, with_audio=False)
    if ws["reference"].notna().any():
        systems["reference"] = ws.set_index("id")["reference"].dropna().to_dict()
    asr_root = run.root / "asr"
    for model_dir in sorted(asr_root.iterdir()) if asr_root.exists() else []:
        df = latest_per_id(read_parts(model_dir))
        if not df.empty:
            systems[model_dir.name] = df[df["error"].isna()].set_index("id")["text"].to_dict()
    if run.labels_path.exists():
        systems["merged"] = pd.read_parquet(run.labels_path).set_index("id")["text"].to_dict()
    for name, path in extra.items():
        systems[name] = pd.read_parquet(path).set_index("id")["text"].to_dict()
    return systems


def evaluate(cfg: dict, gold_csv: str | None = None, extra: dict[str, str] | None = None) -> pd.DataFrame:
    run = Run(cfg)
    gold = load_gold(gold_csv or run.gold_dir / "gold.csv")
    systems = collect_systems(run, extra or {})
    refs = gold["gold_text"].tolist()

    rows, by_channel = [], []
    for name, hyps_by_id in systems.items():
        hyps = [hyps_by_id.get(i) for i in gold["id"]]
        coverage = sum(h is not None for h in hyps) / len(hyps)
        norm, raw = score(refs, hyps, True), score(refs, hyps, False)
        rows.append({"system": name, "wer": norm["wer"], "cer": norm["cer"],
                     "raw_wer": raw["wer"], "raw_cer": raw["cer"], "coverage": coverage, "n": norm["n"]})
        for channel, g in gold.assign(hyp=hyps).groupby("channel"):
            s = score(g["gold_text"].tolist(), g["hyp"].tolist())
            by_channel.append({"system": name, "channel": channel, **s})

    table = pd.DataFrame(rows).sort_values("wer")
    run.eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(by_channel).to_csv(run.eval_dir / "by_channel.csv", index=False, encoding="utf-8-sig")
    report = (f"# Gold-set evaluation ({len(gold)} checked clips)\n\n"
              "wer/cer: after scoring normalization (spelling variants ignored). "
              "raw_*: exact text, shows spelling consistency.\n"
              "Missing hypotheses (errors, rejections) count as empty; see coverage.\n\n"
              + table.to_markdown(index=False, floatfmt=".4f") + "\n")
    (run.eval_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Per-channel scores: {Path(run.eval_dir / 'by_channel.csv')}")
    return table
