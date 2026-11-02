"""`darija <command> --config configs/<run>.yaml ...` — run from the repo root."""
from __future__ import annotations

import argparse

from .config import apply_overrides, load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="darija", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", "-c", required=True)
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="override a config value, e.g. --set select.n=30 (repeatable)")
        return sp

    sp = cmd("stats", "show clip/channel/pesq/duration distributions of the source split")
    sp.add_argument("--sample-groups", type=int, default=3,
                    help="row groups (~100 MB each) to download for durations when the split has none")

    sp = cmd("prepare", "sample clips from MoulSot-Full into the run's working set")
    sp.add_argument("--force", action="store_true", help="rebuild an existing working set")

    sp = cmd("smoke", "transcribe a few clips with one model and print them (nothing saved)")
    sp.add_argument("--model", required=True)
    sp.add_argument("-n", type=int, default=5)

    sp = cmd("transcribe", "resumable transcription of the working set with one model")
    sp.add_argument("--model", required=True)
    sp.add_argument("--shard", type=int, default=0)
    sp.add_argument("--num-shards", type=int, default=1)
    sp.add_argument("--limit", type=int)

    cmd("agree", "pairwise agreement between sources and routing (accept / llm / drop)")

    sp = cmd("merge", "LLM judge on routed clips")
    sp.add_argument("--mode", choices=["sync", "batch-submit", "batch-collect"], default="sync")
    sp.add_argument("--limit", type=int)

    cmd("labels", "build labels.parquet (final text per clip)")

    sp = cmd("gold-export", "export the gold run as wav + gold.csv for hand correction")
    sp.add_argument("--prefill", choices=["reference", "merged", "empty"], default="reference")

    sp = cmd("eval", "score all systems against the checked gold set")
    sp.add_argument("--gold", help="path to gold.csv (default: <run>/gold/gold.csv)")
    sp.add_argument("--hyp", action="append", default=[], metavar="NAME=PATH",
                    help="extra predictions parquet with id,text columns (repeatable)")

    sp = cmd("push", "upload the run folder (or part of it) to hub_repo")
    sp.add_argument("--path", default="")
    sp = cmd("pull", "download the run folder (or part of it) from hub_repo")
    sp.add_argument("--path", default="")

    sp = sub.add_parser("build-train", help="join labels + audio from runs into a training dataset")
    sp.add_argument("--config", "-c", action="append", required=True, help="repeatable, one per run")
    sp.add_argument("--out", required=True, help="local dir, or hub repo id with --push")
    sp.add_argument("--push", action="store_true")
    sp.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    sp.add_argument("--valid-fraction", type=float, default=0.02)

    args = p.parse_args(argv)

    if args.cmd == "build-train":
        from .train.data import build_train

        build_train([load_config(c) for c in args.config], args.out, args.min_confidence,
                    args.valid_fraction, push=args.push)
        return

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.cmd == "stats":
        from .data.stats import stats

        stats(cfg, args.sample_groups)
    elif args.cmd == "prepare":
        from .data.select import prepare

        prepare(cfg, force=args.force)
    elif args.cmd == "smoke":
        from .asr.run import smoke

        smoke(cfg, args.model, args.n)
    elif args.cmd == "transcribe":
        from .asr.run import transcribe

        transcribe(cfg, args.model, args.shard, args.num_shards, args.limit)
    elif args.cmd == "agree":
        from .agreement import compute_agreement

        compute_agreement(cfg)
    elif args.cmd == "merge":
        from .merge import run as merge_run

        {"sync": lambda: merge_run.merge_sync(cfg, args.limit),
         "batch-submit": lambda: merge_run.batch_submit(cfg, args.limit),
         "batch-collect": lambda: merge_run.batch_collect(cfg)}[args.mode]()
    elif args.cmd == "labels":
        from .merge.run import build_labels

        build_labels(cfg)
    elif args.cmd == "gold-export":
        from .data.gold import export_gold

        export_gold(cfg, args.prefill)
    elif args.cmd == "eval":
        from .eval.run import evaluate

        extra = dict(h.split("=", 1) for h in args.hyp)
        evaluate(cfg, args.gold, extra)
    elif args.cmd in ("push", "pull"):
        from .store import Run

        run = Run(cfg)
        run.push(args.path) if args.cmd == "push" else run.pull(args.path)


if __name__ == "__main__":
    main()
