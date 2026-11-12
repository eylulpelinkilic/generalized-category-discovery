"""Command line. Run from Experiment2_SCD/:

    python -m e2 --config configs/<name>.yaml [--set key=value ...] <command> [options]
"""
from __future__ import annotations

import argparse
import json
import sys

from e2.config import load_config


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_vocab(cfg, args):
    from e2 import vocab_stage

    _print({"vocabulary": str(vocab_stage.run(cfg))})


def cmd_cub_embed(cfg, args):
    from e2 import embed

    _print(embed.cub_embeddings(cfg, args.clip or cfg["fixed"]["clip"]))


def cmd_fixed(cfg, args):
    from e2 import fixed

    _print({"fixed_cluster": str(fixed.run(cfg, args.clip or cfg["fixed"]["clip"]))})


def cmd_refine_gcd(cfg, args):
    from e2 import refine_gcd

    _print(refine_gcd.run(cfg, args.clip or cfg["refine_on_gcd"]["clip"], args.cluster_seed, args.mode))


def cmd_repro(cfg, args):
    from e2 import repro

    _print(getattr(repro, args.stage.replace("-", "_"))(cfg, args))


def cmd_report(cfg, args):
    from e2 import report

    _print({"report": str(report.write(cfg))})


def cmd_smoke(cfg, args):
    from e2 import smoke

    _print(smoke.run(cfg))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m e2", description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], help="override, e.g. paths.outputs=/kaggle/working/e2")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("vocab", help="rebuild the WordNet vocabulary and verify it against the official embeddings"
                   ).set_defaults(fn=cmd_vocab)
    q = sub.add_parser("cub-embed", help="CUB: recompute CLIP image embeddings for Experiment 1's rows and validate")
    q.add_argument("--clip")
    q.set_defaults(fn=cmd_cub_embed)
    q = sub.add_parser("fixed", help="protocol B: mean vs vote naming on Experiment 1's fixed clusters")
    q.add_argument("--clip")
    q.set_defaults(fn=cmd_fixed)
    q = sub.add_parser("refine-gcd", help="protocol R: SCD refinement started from Experiment 1's GCD clusters")
    q.add_argument("--clip")
    q.add_argument("--cluster-seed", type=int, default=0)
    q.add_argument("--mode", choices=["unsupervised", "partially_supervised"], default="unsupervised")
    q.set_defaults(fn=cmd_refine_gcd)
    q = sub.add_parser("repro", help="protocol A (ImageNet-100): features, clustering, naming, evaluation")
    q.add_argument("stage", choices=["features", "cluster", "name", "evaluate", "all"])
    q.add_argument("--setting", choices=["unsupervised", "partially_supervised", "both"], default="both")
    q.add_argument("--seed", type=int, default=None, help="clustering seed (default: all configured)")
    q.add_argument("--variants", default=None, help="comma list (default: all configured)")
    q.set_defaults(fn=cmd_repro)
    sub.add_parser("report", help="write reports/ from the completed stages").set_defaults(fn=cmd_report)
    sub.add_parser("smoke", help="every protocol-A stage on synthetic data (software check only)"
                   ).set_defaults(fn=cmd_smoke)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    try:
        args.fn(cfg, args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")
