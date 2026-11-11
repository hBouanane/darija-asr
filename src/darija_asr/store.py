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


_warned_no_hub = False


def _warn_no_hub() -> None:
    global _warned_no_hub
    if not _warned_no_hub:
        _warned_no_hub = True
        print("[hub] WARNING: no hub_repo (set DARIJA_HUB_REPO): results stay on this machine only, "
              "and other machines (Kaggle, Colab) will not see them.", flush=True)


_auth_failed = False


def _auth_error(exc: Exception) -> bool:
    """401/403 from the hub: a bad or read-only token. Retrying cannot fix it, so
    stop uploading for the rest of this process (files stay on disk for `darija push`)."""
    global _auth_failed
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status not in (401, 403):
        return False
    if not _auth_failed:
        _auth_failed = True
        print(f"[hub] ERROR {status}: Hugging Face rejected the token. Set HF_TOKEN to a token with WRITE "
              "access (huggingface.co/settings/tokens). Uploads are skipped for the rest of this command; "
              "the files stay on disk, upload them later with `darija push -c <config> --path <folder>`.",
              flush=True)
    return True


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
    # Several machines can work on one run at once (e.g. Voxtral on Kaggle while
    # Cohere and MoulSot run on Colab). That is safe because every writer only
    # adds new, uniquely named files (part-<time>-<uuid>, shard-<time>-...) and
    # never rewrites a shared one; the hub orders the commits.
    def push(self, subpath: str = "", message: str | None = None, retries: int = 5) -> bool:
        """Upload root/subpath (file or folder) to the hub repo under <run_name>/subpath.

        No-op without hub_repo. Never raises: a failed upload is retried, then
        reported, and the next push uploads the same (append-only) files again.
        """
        repo = self.cfg.get("hub_repo")
        local = self.root / subpath
        if not repo:
            _warn_no_hub()
            return False
        if not local.exists() or _auth_failed:
            return False
        from huggingface_hub import HfApi

        api = HfApi()
        path_in_repo = f"{self.name}/{subpath}".rstrip("/").replace("\\", "/")
        for attempt in range(retries):
            try:
                api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
                if local.is_file():
                    api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                    repo_id=repo, repo_type="dataset",
                                    commit_message=message or f"update {path_in_repo}")
                else:
                    # Resumable: retries failed files, skips ones already uploaded, and
                    # batches commits (the hub allows ~1000 requests per 5 minutes).
                    # The repo mirrors workdir/<run>/..., so upload workdir filtered to this path.
                    api.upload_large_folder(repo_id=repo, repo_type="dataset", private=True,
                                            folder_path=str(Path(self.cfg["workdir"])),
                                            allow_patterns=[f"{path_in_repo}/**"], num_workers=2)
                return True
            except Exception as exc:  # noqa: BLE001 - network/hub errors must not stop a long run
                if _auth_error(exc):
                    return False
                wait = 10 * 2 ** attempt
                print(f"[hub] push of {path_in_repo} failed ({exc!r:.200}); retry in {wait}s")
                time.sleep(wait)
        print(f"[hub] giving up on {path_in_repo} for now; it will go up with the next push")
        return False

    def push_file(self, path: Path, message: str | None = None) -> bool:
        """Upload one file that lives under the run folder (e.g. a fresh part)."""
        return self.push_files([path], message)

    def push_files(self, paths: list[Path], message: str | None = None, retries: int = 5) -> bool:
        """Upload files under the run folder in one commit (adds only, never deletes)."""
        repo = self.cfg.get("hub_repo")
        if not repo:
            _warn_no_hub()
            return False
        if not paths or _auth_failed:
            return False
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        for attempt in range(retries):
            ops = [CommitOperationAdd(path_in_repo=f"{self.name}/{p.relative_to(self.root).as_posix()}",
                                      path_or_fileobj=str(p)) for p in paths]
            try:
                api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
                api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                                  commit_message=message or f"add {len(ops)} file(s) to {self.name}")
                return True
            except Exception as exc:  # noqa: BLE001 - network/hub errors must not stop a long run
                if _auth_error(exc):
                    return False
                wait = 10 * 2 ** attempt
                print(f"[hub] upload of {len(ops)} file(s) failed ({exc!r:.200}); retry in {wait}s")
                time.sleep(wait)
        print("[hub] giving up for now; `darija push` uploads everything later")
        return False

    def pull(self, subpath: str = "", quiet: bool = False, patterns: list[str] | None = None) -> None:
        """Download <run_name>/subpath (or <run_name>/<pattern>s) from the hub repo.

        Only adds/updates files; local files that are not on the hub stay.
        """
        repo = self.cfg.get("hub_repo")
        if not repo:
            if quiet:
                return
            raise SystemExit("hub_repo is not set in the config")
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        if patterns:
            allow = [f"{self.name}/{p}" for p in patterns]
        else:
            pattern = f"{self.name}/{subpath}".rstrip("/")
            allow = [pattern, f"{pattern}/*", f"{pattern}/**"]
        try:
            snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir=str(Path(self.cfg["workdir"])), allow_patterns=allow)
        except RepositoryNotFoundError:
            if not quiet:
                raise
            print(f"[hub] {repo} does not exist yet; nothing to pull")


    def pull_for_judging(self) -> None:
        """Everything agree/merge/labels need, without the (large) workset audio."""
        self.pull(patterns=["workset/index-*.parquet", "asr/**", "merge/**"], quiet=True)


def write_part(df: pd.DataFrame, directory: Path, prefix: str = "part") -> Path:
    """Write an append-only part file; names never collide across sessions or machines."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}.parquet"
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


WORKSET_META = ["id", "channel", "duration", "pesq_hyp", "reference"]


def read_workset(run: Run, with_audio: bool = True) -> pd.DataFrame:
    """The run's clips. Without audio it reads the small index files when they exist,
    so a machine that only judges never has to download the audio."""
    if not with_audio and any(run.workset_dir.glob("index-*.parquet")):
        df = read_parts(run.workset_dir, "index-*.parquet", columns=WORKSET_META)
    else:
        df = read_parts(run.workset_dir, "shard-*.parquet", columns=None if with_audio else WORKSET_META)
    if df.empty:
        raise SystemExit(f"No working set in {run.workset_dir}. Run `darija prepare` (or `darija pull`) first.")
    return df


def read_ids_file(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
