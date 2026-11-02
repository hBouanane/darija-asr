"""On-disk layout of a run, resumable parquet parts, and HF Hub sync.

work/<run_name>/
    workset/shard-*.parquet      id, channel, duration, pesq_hyp, reference, audio (bytes)
    asr/<model>/part-*.parquet   id, text, error, seconds
    agreement.parquet            pairwise CER/WER + route per clip
    merge/part-*.parquet         LLM judge outputs
    merge/batches.json           submitted Message Batch ids (batch mode)
    labels.parquet               final id -> text for training
    gold/                        exported audio + gold.csv for human correction
    eval/                        reports
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import pandas as pd


class Run:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg["run_name"]
        self.root = Path(cfg["workdir"]) / self.name

    @property
    def workset_dir(self) -> Path:
        return self.root / "workset"

    def asr_dir(self, model: str) -> Path:
        return self.root / "asr" / model

    @property
    def agreement_path(self) -> Path:
        return self.root / "agreement.parquet"

    @property
    def merge_dir(self) -> Path:
        return self.root / "merge"

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.parquet"

    @property
    def gold_dir(self) -> Path:
        return self.root / "gold"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    # -- hub sync ---------------------------------------------------------
    def push(self, subpath: str = "", message: str | None = None) -> None:
        """Upload root/subpath to the hub repo under <run_name>/subpath. No-op without hub_repo."""
        repo = self.cfg.get("hub_repo")
        if not repo:
            return
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        local = self.root / subpath
        if not local.exists():
            return
        path_in_repo = f"{self.name}/{subpath}".rstrip("/")
        if local.is_file():
            api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                            repo_id=repo, repo_type="dataset",
                            commit_message=message or f"update {path_in_repo}")
        else:
            api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo,
                              repo_id=repo, repo_type="dataset",
                              commit_message=message or f"update {path_in_repo}")

    def pull(self, subpath: str = "") -> None:
        """Download <run_name>/subpath from the hub repo into the local work dir."""
        repo = self.cfg.get("hub_repo")
        if not repo:
            raise SystemExit("hub_repo is not set in the config")
        from huggingface_hub import snapshot_download

        pattern = f"{self.name}/{subpath}".rstrip("/")
        snapshot_download(repo_id=repo, repo_type="dataset",
                          local_dir=str(Path(self.cfg["workdir"])),
                          allow_patterns=[pattern, f"{pattern}/*", f"{pattern}/**"])


def write_part(df: pd.DataFrame, directory: Path, prefix: str = "part") -> Path:
    """Write an append-only part file; names never collide across sessions."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:6]}.parquet"
    df.to_parquet(path, index=False)
    return path


def read_parts(directory: Path, pattern: str = "part-*.parquet", columns=None) -> pd.DataFrame:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    if not files:
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def latest_per_id(df: pd.DataFrame) -> pd.DataFrame:
    """Parts are append-only; when an id was retried keep its last successful row."""
    if df.empty:
        return df
    if "error" in df.columns:
        df = df.assign(_ok=df["error"].isna()).sort_values("_ok", kind="stable")
        df = df.drop(columns="_ok")
    return df.drop_duplicates("id", keep="last").reset_index(drop=True)


def done_ids(directory: Path) -> set[str]:
    """Ids that already have a successful row, so a rerun skips them."""
    df = read_parts(directory)
    if df.empty:
        return set()
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return set(df["id"])


def read_workset(run: Run, with_audio: bool = True) -> pd.DataFrame:
    cols = None if with_audio else ["id", "channel", "duration", "pesq_hyp", "reference"]
    df = read_parts(run.workset_dir, "shard-*.parquet", columns=cols)
    if df.empty:
        raise SystemExit(f"No working set in {run.workset_dir}. Run `darija prepare` (or `darija pull`) first.")
    return df


def read_ids_file(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
