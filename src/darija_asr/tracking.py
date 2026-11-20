"""Optional Weights & Biases tracking for long stages (transcribe, merge, labels).

On when a project is set: `wandb.project` in the config, or the WANDB_PROJECT
environment variable (log in once with `wandb login` or WANDB_API_KEY). Off
otherwise, and never fatal: if wandb is missing or fails, the stage runs untracked.
"""
from __future__ import annotations

import os

import pandas as pd


class Tracker:
    def __init__(self, cfg: dict, job: str, config: dict | None = None):
        self.wb = None
        wcfg = cfg.get("wandb") or {}
        project = wcfg.get("project") or os.environ.get("WANDB_PROJECT")
        if not project:
            return
        try:
            import wandb

            self.wb = wandb.init(project=project, entity=wcfg.get("entity"), job_type=job,
                                 name=f"{cfg['run_name']}-{job}", group=cfg["run_name"],
                                 config=config or {}, reinit=True)
        except Exception as exc:  # noqa: BLE001 - tracking must never stop the pipeline
            print(f"[wandb] disabled ({exc!r:.150})")
            self.wb = None

    def log(self, metrics: dict) -> None:
        if self.wb:
            self.wb.log(metrics)

    def table(self, name: str, df: pd.DataFrame, max_rows: int = 500) -> None:
        if self.wb and not df.empty:
            import wandb

            self.wb.log({name: wandb.Table(dataframe=df.head(max_rows).astype(str))})

    def summary(self, metrics: dict) -> None:
        if self.wb:
            self.wb.summary.update(metrics)

    def finish(self) -> None:
        if self.wb:
            self.wb.finish()
            self.wb = None
