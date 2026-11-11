"""Multi-machine pieces: workset index, waiting for all sources, OpenAI-style judge parsing."""
import json

import pandas as pd

from darija_asr.agreement import compute_agreement
from darija_asr.config import load_config
from darija_asr.merge.judge_openai import parse_response
from darija_asr.merge.run import _pending
from darija_asr.store import Run, read_workset, write_part


def _run(tmp_path, sources):
    cfg = {"run_name": "t", "workdir": str(tmp_path), "hub_repo": None,
           "agreement": {"sources": sources, "prefer": sources, "accept_cer": 0.05, "drop_cer": 0.6},
           "merge": {"mode": "all", "require_all_sources": True}}
    run = Run(cfg)
    run.workset_dir.mkdir(parents=True)
    meta = pd.DataFrame({"id": ["a", "b"], "channel": ["c1", "c1"], "duration": [1.0, 2.0],
                         "pesq_hyp": [3.0, 3.0], "reference": [None, None]})
    # Only the small index is present, as on a machine that only judges.
    meta.to_parquet(run.workset_dir / "index-20260101-000000-00000.parquet")
    return cfg, run


def test_read_workset_uses_index_without_audio(tmp_path):
    _, run = _run(tmp_path, ["cohere"])
    assert list(read_workset(run, with_audio=False)["id"]) == ["a", "b"]


def test_merge_waits_for_every_source(tmp_path):
    cfg, run = _run(tmp_path, ["cohere", "voxtral"])
    write_part(pd.DataFrame({"id": ["a", "b"], "text": ["واحد", "جوج"], "error": [None, None]}), run.asr_dir("cohere"))
    write_part(pd.DataFrame({"id": ["a"], "text": ["واحد"], "error": [None]}), run.asr_dir("voxtral"))
    compute_agreement(cfg)
    assert list(_pending(cfg, run, None)["id"]) == ["a"]  # b still waits for voxtral
    cfg["merge"]["require_all_sources"] = False
    assert sorted(_pending(cfg, run, None)["id"]) == ["a", "b"]


def test_openai_response_parsing():
    verdict = {"text": "واش نتا مزيان", "reject": False, "confidence": "high", "notes": ""}
    body = {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": "```json\n" + json.dumps(verdict, ensure_ascii=False) + "\n```"}}]}
    row = parse_response(body)
    assert row["error"] is None and row["text"] == "واش نتا مزيان" and row["output_tokens"] == 5
    body["choices"][0]["finish_reason"] = "length"
    assert parse_response(body)["error"] == "max_tokens"


def test_hub_repo_from_env(monkeypatch):
    monkeypatch.setenv("DARIJA_HUB_REPO", "me/work")
    assert load_config("configs/extra.yaml")["hub_repo"] == "me/work"


def test_gold_ui_page(tmp_path):
    from darija_asr.data.gold import GOLD_COLUMNS
    from darija_asr.data.gold_ui import build_page, gold_records

    cfg, run = _run(tmp_path, ["cohere", "voxtral"])
    run.gold_dir.mkdir(parents=True)
    pd.DataFrame([{"n": 0, "id": "a", "channel": "c1", "duration": 1.0, "audio_file": "audio/0000.wav",
                   "gold_text": "واحد", "checked": "", "comment": ""}], columns=GOLD_COLUMNS
                 ).to_csv(run.gold_dir / "gold.csv", index=False, encoding="utf-8-sig")
    write_part(pd.DataFrame({"id": ["a"], "text": ["واحد"], "error": [None]}), run.asr_dir("cohere"))
    write_part(pd.DataFrame({"id": ["a"], "text": ["وحد </script>"], "error": [None]}), run.asr_dir("voxtral"))
    write_part(pd.DataFrame([{"id": "a", "text": "واحد", "reject": False, "confidence": "high", "notes": "",
                              "error": None}]), run.merge_dir)
    recs = gold_records(cfg)
    assert recs[0]["candidates"] == {"cohere": "واحد", "voxtral": "وحد </script>"}
    assert recs[0]["judge"]["confidence"] == "high"
    page = build_page(recs, "gold")
    assert page.count("</script>") == 2  # the candidate's "</script>" is escaped inside the JSON


def test_hub_auth_error_stops_retries(tmp_path, monkeypatch):
    import darija_asr.store as store

    class Resp:
        status_code = 401

    class Unauthorized(Exception):
        response = Resp()

    calls = []

    class Api:
        def create_repo(self, *a, **k):
            calls.append(1)
            raise Unauthorized()

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: Api())
    monkeypatch.setattr(store, "_auth_failed", False)
    monkeypatch.setattr(store.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("no retry wait")))
    cfg, run = _run(tmp_path, ["cohere"])
    cfg["hub_repo"] = "me/work"
    f = run.workset_dir / "index-20260101-000000-00000.parquet"
    assert run.push_files([f]) is False and run.push_files([f]) is False
    assert calls == [1]  # second call skipped without touching the hub
