"""Export the gold run for human correction.

Writes gold/audio/<n>.wav and gold/gold.csv (UTF-8 with BOM so Excel/Sheets
show Arabic correctly). Correct `gold_text` while listening, and put `x` in
`checked` for each row you verified; eval only uses checked rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import soundfile as sf

from .. import audio as audio_utils
from ..store import Run, read_workset

GOLD_COLUMNS = ["n", "id", "channel", "duration", "audio_file", "gold_text", "checked", "comment"]


def export_gold(cfg: dict, prefill: str = "reference", ids_file: str = "resources/gold_ids.txt") -> None:
    run = Run(cfg)
    ws = read_workset(run)
    prefills = {}
    if prefill == "merged" and run.labels_path.exists():
        prefills = pd.read_parquet(run.labels_path).set_index("id")["text"].to_dict()

    audio_dir = run.gold_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, rec in enumerate(ws.sort_values(["channel", "id"]).itertuples(index=False)):
        name = f"{n:04d}.wav"
        sf.write(audio_dir / name, audio_utils.decode(rec.audio), audio_utils.SAMPLE_RATE)
        text = {"reference": rec.reference, "merged": prefills.get(rec.id), "empty": ""}[prefill]
        rows.append({"n": n, "id": rec.id, "channel": rec.channel, "duration": round(rec.duration, 2),
                     "audio_file": f"audio/{name}", "gold_text": text or "", "checked": "", "comment": ""})

    csv_path = run.gold_dir / "gold.csv"
    if csv_path.exists():
        raise SystemExit(f"{csv_path} exists; refusing to overwrite your corrections.")
    pd.DataFrame(rows, columns=GOLD_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8-sig")
    Path(ids_file).write_text("\n".join(r["id"] for r in rows) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} clips to {run.gold_dir}; ids written to {ids_file}")
    run.push("gold", "gold export")


def load_gold(path: str | Path) -> pd.DataFrame:
    """Checked gold rows only: id, channel, gold_text."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    checked = df["checked"].str.strip().str.lower().isin({"x", "1", "y", "yes", "true", "ok"})
    df = df[checked & (df["gold_text"].str.strip() != "")]
    if df.empty:
        raise SystemExit(f"No checked rows in {path}. Mark verified rows with `x` in the `checked` column.")
    return df[["id", "channel", "gold_text"]].reset_index(drop=True)
