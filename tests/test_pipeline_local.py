"""End-to-end over the CPU stages with fake transcripts (no models, no API)."""
import io

import numpy as np
import pandas as pd
import soundfile as sf

from darija_asr.agreement import compute_agreement
from darija_asr.eval.run import evaluate
from darija_asr.merge.run import _judged_ids, build_labels
from darija_asr.store import Run, done_ids, write_part


def _wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


def test_cpu_stages(tmp_path):
    cfg = {
        "run_name": "t", "workdir": str(tmp_path), "hub_repo": None,
        "agreement": {"sources": ["reference", "cohere", "voxtral"], "prefer": ["reference", "cohere"],
                      "accept_cer": 0.05, "drop_cer": 0.6},
        "merge": {"mode": "disagreements"},
    }
    run = Run(cfg)
    run.workset_dir.mkdir(parents=True)
    pd.DataFrame({
        "id": ["a", "b", "c"], "channel": ["ch1", "ch1", "ch2"], "duration": [1.0, 1.0, 1.0],
        "pesq_hyp": [3.0] * 3, "reference": ["واش نتا مزيان", "كنمشي للدار", "hello world"],
        "audio": [_wav_bytes()] * 3,
    }).to_parquet(run.workset_dir / "shard-00000.parquet")

    write_part(pd.DataFrame({"id": ["a", "b", "c"], "text": ["واش نتا مزيان", "كنمشي للدار", "واحد جوج"],
                             "error": [None] * 3, "seconds": [0.1] * 3}), run.asr_dir("cohere"))
    # voxtral: b failed first, then succeeded on retry
    write_part(pd.DataFrame({"id": ["a", "b", "c"], "text": ["واش نتا مزيان", None, "تلاتة ربعة"],
                             "error": [None, "oom", None], "seconds": [0.1] * 3}), run.asr_dir("voxtral"))
    write_part(pd.DataFrame({"id": ["b"], "text": ["كنمشي لدار"], "error": [None], "seconds": [0.1]}),
               run.asr_dir("voxtral"))
    assert done_ids(run.asr_dir("voxtral")) == {"a", "b"} | {"c"}

    agreement = compute_agreement(cfg).set_index("id")
    assert agreement.at["a", "route"] == "accept"
    assert agreement.at["b", "route"] == "llm"
    assert agreement.at["c", "route"] == "drop"

    write_part(pd.DataFrame([{"id": "b", "text": "كنمشي للدار", "reject": False, "confidence": "medium",
                              "notes": "", "error": None}]), run.merge_dir)
    labels = build_labels(cfg).set_index("id")
    assert set(labels.index) == {"a", "b"}
    assert labels.at["b", "method"] == "llm"
    assert "max_cer" in labels and "cer_cohere_reference" in labels

    # mode=all: every clip with a candidate goes to the judge; no accept shortcut.
    cfg["merge"]["mode"] = "all"
    assert _judged_ids(cfg, compute_agreement(cfg)) == {"a", "b", "c"}
    assert set(build_labels(cfg)["id"]) == {"b"}
    cfg["merge"]["mode"] = "disagreements"
    build_labels(cfg)

    run.gold_dir.mkdir(parents=True)
    pd.DataFrame({"n": [0, 1], "id": ["a", "b"], "channel": ["ch1", "ch1"], "duration": [1, 1],
                  "audio_file": ["", ""], "gold_text": ["واش نتا مزيان", "كنمشي للدار"],
                  "checked": ["x", "x"], "comment": ["", ""]}).to_csv(run.gold_dir / "gold.csv", index=False,
                                                                     encoding="utf-8-sig")
    table = evaluate(cfg).set_index("system")
    assert table.at["merged", "wer"] == 0
    assert table.at["voxtral", "wer"] > 0
