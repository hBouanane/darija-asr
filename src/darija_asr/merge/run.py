"""Run the LLM judge over a run's clips and build the final labels."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from ..agreement import load_sources
from ..store import Run, done_ids, latest_per_id, read_parts, write_part
from .judge import Judge, custom_id, error_row, parse_message
from .prompt import build_system_prompt, build_user_message

BATCH_CHUNK = 20000  # requests per Message Batch (API limits: 100k requests / 256 MB)


def _judge(cfg: dict) -> Judge:
    mcfg = cfg["merge"]
    return Judge(mcfg, build_system_prompt(mcfg.get("style_guide"), mcfg.get("lexicon")))


def _judged_ids(cfg: dict, agreement: pd.DataFrame) -> set[str]:
    """Clips the judge should see, according to merge.mode."""
    mode = cfg["merge"]["mode"]
    if mode == "all":
        # Every clip with at least one candidate, whatever its route.
        return set(agreement.loc[agreement["n_sources"] >= 1, "id"])
    if mode == "disagreements":
        return set(agreement.loc[agreement["route"] == "llm", "id"])
    raise SystemExit(f"merge.mode must be `all` or `disagreements`, got {mode!r}")


def _pending(cfg: dict, run: Run, limit: int | None) -> pd.DataFrame:
    """Clips that need a judgement and don't have one yet, with their candidate texts."""
    if not run.agreement_path.exists():
        raise SystemExit("Run `darija agree` first.")
    agreement = pd.read_parquet(run.agreement_path, columns=["id", "route", "n_sources"])
    ids = _judged_ids(cfg, agreement)
    ids -= done_ids(run.merge_dir)
    ids -= _ids_in_open_batches(run)
    sources = load_sources(run, cfg["agreement"]["sources"])
    todo = sources[sources["id"].isin(ids)]
    return todo.head(limit) if limit else todo


def _messages(cfg: dict, todo: pd.DataFrame) -> list[tuple[str, str]]:
    srcs = cfg["agreement"]["sources"]
    reveal = cfg["merge"].get("reveal_sources", False)
    return [(rec["id"], build_user_message(rec["id"], {s: rec.get(s) for s in srcs}, reveal))
            for rec in todo.to_dict("records")]


def merge_sync(cfg: dict, limit: int | None = None) -> None:
    run = Run(cfg)
    todo = _pending(cfg, run, limit)
    print(f"{len(todo)} clips to judge with {cfg['merge']['model']}")
    if todo.empty:
        return
    judge = _judge(cfg)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(cfg["merge"].get("workers", 8))) as pool:
        futures = {pool.submit(judge.call, msg): cid for cid, msg in _messages(cfg, todo)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judge"):
            rows.append({"id": futures[fut], **fut.result()})
            if len(rows) >= 200:
                write_part(pd.DataFrame(rows), run.merge_dir)
                rows = []
    if rows:
        write_part(pd.DataFrame(rows), run.merge_dir)
    run.push("merge")
    _summary(run)


# -- Message Batches (50% cheaper, asynchronous) --------------------------------

def _batches_file(run: Run):
    return run.merge_dir / "batches.json"


def _load_batches(run: Run) -> list[dict]:
    path = _batches_file(run)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _save_batches(run: Run, batches: list[dict]) -> None:
    run.merge_dir.mkdir(parents=True, exist_ok=True)
    _batches_file(run).write_text(json.dumps(batches, indent=1), encoding="utf-8")


def _ids_in_open_batches(run: Run) -> set[str]:
    return {cid for b in _load_batches(run) if not b["collected"] for cid in b["ids"].values()}


def batch_submit(cfg: dict, limit: int | None = None) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    run = Run(cfg)
    todo = _pending(cfg, run, limit)
    print(f"{len(todo)} clips to submit")
    if todo.empty:
        return
    judge = _judge(cfg)
    batches = _load_batches(run)
    messages = _messages(cfg, todo)
    for start in range(0, len(messages), BATCH_CHUNK):
        chunk = messages[start:start + BATCH_CHUNK]
        id_map = {custom_id(cid): cid for cid, _ in chunk}
        batch = judge.client.messages.batches.create(requests=[
            Request(custom_id=custom_id(cid), params=MessageCreateParamsNonStreaming(**judge.params(msg)))
            for cid, msg in chunk
        ])
        batches.append({"batch_id": batch.id, "ids": id_map, "collected": False})
        _save_batches(run, batches)
        print(f"Submitted batch {batch.id} with {len(chunk)} requests")
    run.push("merge")


def batch_collect(cfg: dict) -> None:
    run = Run(cfg)
    batches = _load_batches(run)
    judge = _judge(cfg)
    for b in batches:
        if b["collected"]:
            continue
        status = judge.client.messages.batches.retrieve(b["batch_id"])
        if status.processing_status != "ended":
            counts = status.request_counts
            print(f"{b['batch_id']}: {status.processing_status} "
                  f"(processing {counts.processing}, succeeded {counts.succeeded}, errored {counts.errored})")
            continue
        rows = []
        # Results arrive in any order: key by custom_id.
        for result in judge.client.messages.batches.results(b["batch_id"]):
            clip_id = b["ids"][result.custom_id]
            kind = result.result.type
            if kind == "succeeded":
                rows.append({"id": clip_id, **parse_message(result.result.message)})
            elif kind == "errored":
                rows.append({"id": clip_id, **error_row(f"batch_errored: {result.result.error.type}")})
            else:  # canceled / expired: resubmitted by the next batch-submit
                rows.append({"id": clip_id, **error_row(f"batch_{kind}")})
        if rows:
            write_part(pd.DataFrame(rows), run.merge_dir)
        b["collected"] = True
        _save_batches(run, batches)
        print(f"{b['batch_id']}: collected {len(rows)} results")
    run.push("merge")
    _summary(run)


def _summary(run: Run) -> None:
    df = latest_per_id(read_parts(run.merge_dir))
    if df.empty:
        return
    ok = df[df["error"].isna()]
    print(f"Judged {len(ok)} clips ({len(df) - len(ok)} errors, rerun to retry); "
          f"rejected {int(ok['reject'].sum())}; confidence: {ok['confidence'].value_counts().to_dict()}")
    if "input_tokens" in df:
        print(f"Tokens: input {int(df['input_tokens'].sum())}, cache read {int(df['cache_read_tokens'].sum())}, "
              f"output {int(df['output_tokens'].sum())}")


# -- Final labels ----------------------------------------------------------------

def build_labels(cfg: dict) -> pd.DataFrame:
    """id -> training text. LLM verdicts win. Only in `disagreements` mode do
    accepted clips fall back to the preferred source's text."""
    run = Run(cfg)
    agreement = pd.read_parquet(run.agreement_path)
    sources = load_sources(run, cfg["agreement"]["sources"]).set_index("id")
    merged = latest_per_id(read_parts(run.merge_dir))
    merged = merged[merged["error"].isna()].set_index("id") if not merged.empty else merged

    use_accept = cfg["merge"]["mode"] == "disagreements"
    prefer = cfg["agreement"]["prefer"]
    rows = []
    for rec in agreement.to_dict("records"):
        cid = rec["id"]
        if not merged.empty and cid in merged.index:
            m = merged.loc[cid]
            if m["reject"] or not str(m["text"]).strip():
                continue
            rows.append({"id": cid, "text": m["text"].strip(), "method": "llm", "confidence": m["confidence"]})
        elif use_accept and rec["route"] == "accept":
            text = next((sources.at[cid, s] for s in prefer
                         if isinstance(sources.at[cid, s], str) and sources.at[cid, s].strip()), None)
            if text:
                rows.append({"id": cid, "text": text.strip(), "method": "agree", "confidence": "high"})
    # Agreement scores travel with each label, for filtering training data later.
    score_cols = ["id", "channel", "duration", "route", "n_sources"] + [
        c for c in agreement.columns if c.startswith(("cer_", "wer_")) or c in ("max_cer", "min_cer", "mean_cer")]
    labels = pd.DataFrame(rows).merge(agreement[score_cols], on="id")
    labels.to_parquet(run.labels_path, index=False)
    run.push("labels.parquet")

    print(f"{len(labels)} labelled clips ({labels['duration'].sum() / 3600:.1f} h) of {len(agreement)}; "
          f"by method: {labels['method'].value_counts().to_dict()}")
    if not merged.empty:
        rejected = int(merged["reject"].sum())
        print(f"Rejected by the judge: {rejected}")
    sent = len(_judged_ids(cfg, agreement))
    missing = sent - len(merged)
    if missing > 0:
        print(f"Note: {missing} clips sent to the judge have no verdict yet (pending or errored); rerun merge.")
    return labels
