"""Command line: one subcommand per stage. Run from Experiment1_GCD_CLIP_Baseline/:

    python -m e1 <command> --config configs/<name>.yaml [--set key=value ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from e1.config import Layout, load_config, section_hash
from e1.provenance import write_json


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_prepare_data(cfg, args):
    from e1 import manifests, sources

    layout = Layout(cfg)
    ds = cfg["dataset"]
    if ds["name"] == "smoke":
        meta = manifests.build_smoke(layout.data, layout.base / "images")
    else:
        root = Path(ds["root"]) if ds.get("root") else None
        if ds["name"] == "cub" and args.download:
            root = Path(sources.fetch_cub(args.download)["path"])
        if root is None:
            raise SystemExit("set --set dataset.root=<path> (or --download <dir> for CUB)")
        build = manifests.build_imagenet100 if ds["name"] == "imagenet100" else manifests.build_cub
        meta = build(root, layout.data, heldout_seed=ds["heldout_seed"], verify_images=ds["verify_images"],
                     workers=ds["workers"])
        if args.upstream_check:
            from e1 import upstream_check
            check = (upstream_check.check_imagenet if ds["name"] == "imagenet100" else upstream_check.check_cub)(
                root, manifests.Manifest(layout.data))
            write_json(layout.data / "upstream_partition_check.json", check)
            if not check["passed"]:
                raise SystemExit(f"partition differs from the upstream implementation: {check}")
            meta["upstream_partition_check"] = check
    _print({k: meta[k] for k in ("counts", "manifest_sha256")} | {"checks_files": meta["checks"]["files"]})


def cmd_fetch_dino(cfg, args):
    from e1 import sources

    _print(sources.fetch(sources.DINO_VITB16["url"], Path(args.dest) / sources.DINO_VITB16["file"],
                         sources.DINO_VITB16["bytes"]))


def cmd_benchmark(cfg, args):
    from e1.gcd_train import benchmark

    layout = Layout(cfg)
    results = [benchmark(cfg, layout, args.train_seed, args.iters, amp) for amp in args.amp_modes]
    write_json(layout.gcd_run(args.train_seed).parent / f"benchmark_{int(time.time())}.json", results)
    _print(results)


def cmd_train(cfg, args):
    from e1.gcd_train import Trainer

    budget = args.time_budget_hours * 3600 if args.time_budget_hours else None
    _print(Trainer(cfg, Layout(cfg), args.train_seed).train(time_budget_s=budget, max_epochs=args.max_epochs))


def cmd_extract_gcd(cfg, args):
    from e1.gcd_features import extract

    _print(extract(cfg, Layout(cfg), args.train_seed))


def cmd_estimate_k(cfg, args):
    from e1.clustering import estimate_k

    _print({k: v for k, v in estimate_k(cfg, Layout(cfg), args.train_seed).items() if k != "evaluations"})


def cmd_cluster(cfg, args):
    from e1 import clustering

    layout = Layout(cfg)
    out = clustering.run(cfg, layout, args.train_seed, args.k_mode, args.cluster_seed)
    _print({"dir": str(out), **clustering.evaluate_clustering(layout, out)})


def cmd_verify_clip(cfg, args):
    from e1.clip_model import verify_against_openai
    from e1.gcd_train import select_device
    from e1.manifests import Manifest
    from e1.vocab import PILOT_TEMPLATE

    layout = Layout(cfg)
    m = Manifest(layout.data)
    ids = m.ids("unlabelled")[:: max(1, len(m.ids("unlabelled")) // args.n_images)][: args.n_images]
    texts = [PILOT_TEMPLATE.format(n) for n in m.classes["canonical_name"].tolist()]
    res = verify_against_openai(cfg["clip"]["models"][args.clip], m.paths(ids), texts,
                                device=str(select_device(cfg.get("device", "auto"))))
    write_json(layout.clip(args.clip) / "verification.json", res)
    _print(res)
    if not res["passed"]:
        raise SystemExit("CLIP verification failed")


def cmd_extract_clip(cfg, args):
    from e1.clip_extract import extract_images

    _print(extract_images(cfg, Layout(cfg), args.clip))


def cmd_ground(cfg, args):
    from e1 import ground

    _print({"dir": str(ground.run(cfg, Layout(cfg), args.train_seed, args.k_mode, args.cluster_seed, args.clip))})


def cmd_export_diagnostics(cfg, args):
    from e1.diagnostics_export import export

    out = export(cfg, args.train_seed, args.k_mode, args.cluster_seed, args.clip, population=args.population,
                 n=args.n, seed=args.seed, image_root=args.image_root, image_cache=args.image_cache)
    _print(json.loads((out / "export_summary.json").read_text()))


def cmd_compare_diagnostics(cfg, args):
    from e1.diagnostics_compare import compare

    layout = Layout(cfg)
    d = layout.grounding(args.train_seed, args.k_mode, args.cluster_seed, args.clip) / "diagnostics" / "clustering_examples"
    _print({"comparison": str(compare(d))})


def cmd_report(cfg, args):
    from e1.report import write_report

    _print({"report": str(write_report(cfg, Layout(cfg)))})


def cmd_smoke(cfg, args):
    """Every stage on synthetic data (software check)."""
    from e1 import clustering, ground
    from e1.clip_extract import extract_images
    from e1.gcd_features import extract
    from e1.gcd_train import Trainer
    from e1.report import write_report

    layout = Layout(cfg)
    if not (layout.data / "manifest.csv").exists():
        cmd_prepare_data(cfg, args)
    s, c, clip = cfg["seeds"]["train"][0], cfg["seeds"]["cluster"][0], cfg["clip"]["primary"]
    _print(Trainer(cfg, layout, s).train())
    extract(cfg, layout, s)
    for k_mode in ("known", "estimated"):
        out = clustering.run(cfg, layout, s, k_mode, c)
        clustering.evaluate_clustering(layout, out)
    extract_images(cfg, layout, clip)
    ground.run(cfg, layout, s, "known", c, clip)
    _print({"report": str(write_report(cfg, layout)), "NOTE": "synthetic smoke run: software check only"})


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m e1", description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], help="override, e.g. dataset.root=/data")
    sub = p.add_subparsers(dest="cmd", required=True)
    seed = lambda q: q.add_argument("--train-seed", type=int, default=0)  # noqa: E731

    q = sub.add_parser("prepare-data")
    q.add_argument("--download", help="CUB only: directory to download and extract the official archive into")
    q.add_argument("--upstream-check", action="store_true", help="replay the upstream dataset functions")
    q.set_defaults(fn=cmd_prepare_data)
    q = sub.add_parser("fetch-dino")
    q.add_argument("--dest", required=True)
    q.set_defaults(fn=cmd_fetch_dino)
    q = sub.add_parser("benchmark")
    seed(q)
    q.add_argument("--iters", type=int, default=20)
    q.add_argument("--amp-modes", type=lambda s: [x == "amp" for x in s.split(",")], default=[False, True],
                   help="comma list of fp32,amp")
    q.set_defaults(fn=cmd_benchmark)
    q = sub.add_parser("train")
    seed(q)
    q.add_argument("--time-budget-hours", type=float)
    q.add_argument("--max-epochs", type=int)
    q.set_defaults(fn=cmd_train)
    for name, fn in (("extract-gcd", cmd_extract_gcd), ("estimate-k", cmd_estimate_k)):
        q = sub.add_parser(name)
        seed(q)
        q.set_defaults(fn=fn)
    for name, fn in (("cluster", cmd_cluster), ("ground", cmd_ground)):
        q = sub.add_parser(name)
        seed(q)
        q.add_argument("--k-mode", choices=["known", "estimated"], default="known")
        q.add_argument("--cluster-seed", type=int, default=0)
        if name == "ground":
            q.add_argument("--clip", default=None)
        q.set_defaults(fn=fn)
    for name, fn in (("verify-clip", cmd_verify_clip), ("extract-clip", cmd_extract_clip)):
        q = sub.add_parser(name)
        q.add_argument("--clip", default=None)
        q.add_argument("--n-images", type=int, default=32)
        q.set_defaults(fn=fn)
    q = sub.add_parser("export-diagnostics", help="correctly/incorrectly clustered examples with similarity context")
    seed(q)
    q.add_argument("--k-mode", choices=["known", "estimated"], default="known")
    q.add_argument("--cluster-seed", type=int, default=0)
    q.add_argument("--clip", default=None)
    q.add_argument("--population", choices=["dev", "calib", "discovery"], default="dev",
                   help="dev (default) keeps exploratory inspection off the evaluation population")
    q.add_argument("--n", type=int, default=50, help="images per group")
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--image-root", help="dataset image root; CUB images are fetched from the official archive if omitted")
    q.add_argument("--image-cache", help="where fetched CUB images are kept (default ~/.cache/e1_cub_images)")
    q.set_defaults(fn=cmd_export_diagnostics)
    q = sub.add_parser("compare-diagnostics", help="compare similarity rows of incorrectly vs correctly clustered examples")
    seed(q)
    q.add_argument("--k-mode", choices=["known", "estimated"], default="known")
    q.add_argument("--cluster-seed", type=int, default=0)
    q.add_argument("--clip", default=None)
    q.set_defaults(fn=cmd_compare_diagnostics)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    q = sub.add_parser("smoke")
    q.add_argument("--download", default=None)
    q.add_argument("--upstream-check", action="store_true")
    q.set_defaults(fn=cmd_smoke)

    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if cfg["dataset"].get("image_root"):
        import os
        os.environ["E1_IMAGE_ROOT"] = str(cfg["dataset"]["image_root"])
    if getattr(args, "clip", "unset") is None:
        args.clip = cfg["clip"]["primary"]
    try:
        args.fn(cfg, args)
    except (RuntimeError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")
