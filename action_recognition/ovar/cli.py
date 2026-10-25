"""Command line: `python -m ovar <command> --help`."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ovar.config import PROJECT_ROOT, load_experiment, load_paths, project_path

PINNED_REPOS = {
    "Open-VCLIP": ("https://github.com/wengzejia1/Open-VCLIP.git", "047a817e94b88c02c56fde2f1dd0fbf03f3be638"),
    "XOV-Action": ("https://github.com/KunyuLin/XOV-Action.git", "f90e6f7ab2f06beb19d09450f59811bcdfbef131"),
    "tc-clip": ("https://github.com/naver-ai/tc-clip.git", "8728fe4ca2b27f00e7235ed8f81f59cb933c8ed9"),
}


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_build_manifests(args):
    from ovar.benchmark import build_all

    _print(build_all())


def cmd_validate(args):
    import pandas as pd

    from ovar.benchmark import load_benchmark
    from ovar.manifest import load_manifest, load_vocab, validate

    bench, paths = load_benchmark(), load_paths(args.paths)
    xov = project_path(bench["xov_repo"])
    out_root = project_path(paths.get("runs_dir", "runs")) / "validation"
    failed = False
    for name in args.datasets:
        spec = bench["datasets"][name]
        split = args.split or spec["split"]
        official = {k: xov / v for k, v in spec["lists"].items()} if split == spec["split"] else None
        report = validate(load_manifest(name, split), load_vocab(name),
                          expected=spec["expected"] if split == spec["split"] else None,
                          official_lists=official, paths=paths, check_files=args.check_files,
                          check_decode=args.check_decode, decode_limit=args.decode_limit)
        out = out_root / f"{name}_{split}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report["failures"], columns=["video_id", "video_path", "check", "detail"]).to_csv(
            out / "failures.csv", index=False)
        summary = {k: v for k, v in report.items() if k != "failures"}
        summary["n_failures_logged"] = len(report["failures"])
        (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"== {name} ({split}): {'OK' if not report['errors'] else 'ERRORS'}  -> {out}")
        _print(summary)
        failed |= bool(report["errors"])
    sys.exit(1 if failed else 0)


def _clip_for_text(args):
    from ovar.backbones.clip_frozen import FrozenCLIP

    return FrozenCLIP({"arch": args.arch, "pretrained": "openai"}, "cpu",
                      project_path(load_paths().get("cache_dir", "cache")))


def cmd_audit(args):
    from ovar.audit import run_audit, run_lexical_audit

    audit = run_lexical_audit() if args.method == "lexical" else run_audit(_clip_for_text(args))
    out = PROJECT_ROOT / "data" / "audit" / f"category_audit_{args.method}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    flagged = audit[audit["review_flag"] != ""]
    print(f"wrote {out}; {len(flagged)} of {len(audit)} classes flagged for human review")
    print(flagged[["dataset", "action_name", "official_status", "k400_match_1", "review_flag"]].to_string())


def cmd_infer(args):
    from ovar.inference import run
    from ovar.metrics import evaluate_run

    cfg = load_experiment(args.experiment, args.set, args.paths)
    run_dir = run(cfg, per_class=args.per_class, limit=args.limit, temporal=args.temporal,
                  variant_name=args.variant, allow_download=args.allow_download)
    summary = evaluate_run(run_dir, n_boot=args.n_boot)
    print(f"run directory: {run_dir}")
    _print(summary)


def cmd_evaluate(args):
    from ovar.metrics import evaluate_run

    _print(evaluate_run(args.run_dir, n_boot=args.n_boot))


def cmd_diagnose(args):
    from ovar.diagnostics import load_diagnostic, run_suite, summarise

    if args.summarise_only:
        diag = load_diagnostic(args.config)
        out = summarise(project_path(load_paths(args.paths).get("runs_dir", "runs")) / "diagnostics" /
                        diag["name"], n_boot=args.n_boot)
    else:
        out = run_suite(args.config, args.set, n_boot=args.n_boot)
    print(f"diagnostics written to {out}")
    print((out / "summary.csv").read_text())


def cmd_report(args):
    from ovar.report import results_report

    runs = args.runs or project_path(load_paths(args.paths).get("runs_dir", "runs"))
    print(f"wrote {results_report(runs, args.out)}")


def cmd_error_analysis(args):
    from ovar.report import error_analysis

    print(f"wrote {error_analysis(args.run_dir, args.diagnostics, args.out)}")


def cmd_export_native(args):
    from ovar.benchmark import load_benchmark
    from ovar.native import export_openvclip, export_tcclip

    bench = load_benchmark()
    datasets = {n: bench["datasets"][n]["split"] for n in args.datasets}
    fn = export_openvclip if args.backbone == "openvclip" else export_tcclip
    for path in fn(datasets, args.out, per_class=args.per_class, seed=args.seed):
        print(f"wrote {path}")


def cmd_compare_native(args):
    from ovar.native import compare_openvclip

    _print(compare_openvclip(args.official, args.run_dir, args.dataset, trust_pickle=args.trust_pickle))


def cmd_fetch_clip(args):
    from huggingface_hub import hf_hub_download

    from ovar.backbones.clip_frozen import APPROX_DOWNLOAD_MB

    repo = {"ViT-B-32": "timm/vit_base_patch32_clip_224.openai",
            "ViT-B-16": "timm/vit_base_patch16_clip_224.openai"}[args.arch]
    need_mb = APPROX_DOWNLOAD_MB[args.arch]
    free_mb = shutil.disk_usage(Path.home()).free / 2**20
    print(f"{args.arch}: about {need_mb} MB from huggingface.co/{repo}; free disk: {free_mb:.0f} MB")
    if free_mb < need_mb * 1.2:
        sys.exit("not enough free disk space; free some space first")
    if not args.yes:
        sys.exit("re-run with --yes to download")
    print(hf_hub_download(repo, "open_clip_model.safetensors"))


def cmd_fetch_repos(args):
    root = PROJECT_ROOT / "third_party"
    root.mkdir(exist_ok=True)
    for name, (url, commit) in PINNED_REPOS.items():
        target = root / name
        if target.exists():
            print(f"{name}: exists, skipping")
            continue
        target.mkdir()
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", url],
                    ["git", "fetch", "-q", "--depth", "1", "origin", commit],
                    ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=target, check=True)
        print(f"{name}: {commit}")


def cmd_smoke_test(args):
    from ovar import fixtures
    from ovar.backbones import BackboneUnavailable
    from ovar.diagnostics import run_suite
    from ovar.inference import run
    from ovar.manifest import load_manifest, load_vocab, validate
    from ovar.metrics import evaluate_run
    from ovar.report import error_analysis

    paths = load_paths(args.paths)
    smoke_root = project_path(paths.get("runs_dir", "runs")) / "smoke"
    paths = {**paths, "runs_dir": str(smoke_root)}
    entry = fixtures.build(smoke_root / "fixtures")
    report = validate(load_manifest(entry["name"], "", path=entry["manifest"]),
                      load_vocab(entry["name"], path=entry["vocab"]),
                      paths={"dataset_roots": {entry["name"]: entry["root"]}}, check_decode=True)
    print(f"fixture validation errors (1 corrupt file expected): {report['errors']}")
    assert report["files"]["decode_failed"] == 1, report

    model = {"name": "clip_vitb32", "backbone": "clip_frozen", "arch": "ViT-B-32", "pretrained": "openai"}
    if args.fake:
        model = {"name": "fake_for_tests", "backbone": "fake_for_tests"}
    cfg = fixtures.experiment(entry, model, paths)
    try:
        run_dir = run(cfg)
    except BackboneUnavailable as exc:
        sys.exit(f"{exc}\nRe-run with --fake to test the pipeline without CLIP weights.")
    summary = evaluate_run(run_dir, n_boot=200)
    import pandas as pd

    pred = pd.read_csv(run_dir / entry["name"] / "predictions.csv", keep_default_na=False)
    assert len(pred) == 8 and (pred["status"] == "failed").sum() == 1, "denominator or failure logging broken"
    assert summary["per_dataset"][entry["name"]]["overall"]["n"] == 8
    pure = pred[pred["video_id"].isin(["red_a", "red_b", "green_a", "green_short", "blue_a", "blue_tiny"])]
    if not args.fake:
        assert pure["correct"].all(), f"CLIP mislabels solid colours (channel order?):\n{pure}"

    diag_file = smoke_root / "fixture_diag.yaml"
    diag_file.write_text("name: smoke_fixture\nexperiment: unused\ntemporal: [shuffled, single_frame]\n"
                         "degradations:\n  - {name: dark_0.25, kind: brightness, factor: 0.25}\n"
                         "  - {name: blur_2, kind: blur, sigma: 2.0}\n"
                         "  - {name: down_0.25, kind: downsample, factor: 0.25}\n"
                         f"actor_boxes: {entry['boxes']}\n"
                         "spatial:\n  - {name: actor_crop, context: 1.0}\n"
                         "  - {name: actor_context, context: 2.0}\n")
    diag_dir = run_suite(diag_file, experiment_cfg=cfg, n_boot=200)
    diag = pd.read_csv(diag_dir / "summary.csv", keep_default_na=False)
    shuffled = diag[diag["variant"] == "temporal_shuffled"].iloc[0]
    assert float(shuffled["prediction_change_rate"]) == 0.0, "mean pooling must be order invariant"
    crop = pd.read_csv(diag_dir / "actor_crop" / entry["name"] / "predictions.csv",
                       keep_default_na=False).set_index("video_id")
    assert crop.loc["blue_tiny", "status"] == "failed" and len(crop) == 8, "missing boxes must be logged"
    single = pd.read_csv(diag_dir / "temporal_single_frame" / entry["name"] / "predictions.csv",
                         keep_default_na=False).set_index("video_id")
    if not args.fake:
        assert single.loc["red_then_blue", "pred_name"] == "blue", "single-frame variant not applied"
    out = error_analysis(run_dir, [diag_dir], smoke_root / "SMOKE_TEST_REPORT.md")
    print(f"SMOKE TEST PASSED (synthetic fixtures; backbone={model['name']}). Report: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ovar")
    parser.add_argument("--paths", help="paths YAML (default configs/paths.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    all_ds = ["ucf101", "hmdb51", "arid", "necdrone"]

    sub.add_parser("build-manifests", help="manifests + vocabularies from XOV-Action files").set_defaults(
        fn=cmd_build_manifests)

    p = sub.add_parser("validate", help="validate manifests (and optionally files/decoding)")
    p.add_argument("--datasets", nargs="+", default=all_ds)
    p.add_argument("--split", help="manifest split (default: the evaluation split)")
    p.add_argument("--check-files", action="store_true")
    p.add_argument("--check-decode", action="store_true")
    p.add_argument("--decode-limit", type=int)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("audit-categories", help="screen familiar/unfamiliar status against Kinetics400")
    p.add_argument("--method", choices=["lexical", "clip"], default="lexical")
    p.add_argument("--arch", default="ViT-B-32", help="CLIP text encoder for --method clip")
    p.set_defaults(fn=cmd_audit)

    for name, fn, help_text in (("infer", cmd_infer, "run an experiment, then evaluate it"),):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("experiment", help="configs/experiments/<name>.yaml or a path")
        p.add_argument("--per-class", type=int, help="deterministic small subset: k videos per class")
        p.add_argument("--limit", type=int)
        p.add_argument("--temporal", default="ordered", choices=["ordered", "shuffled", "single_frame", "reversed"])
        p.add_argument("--variant", help="suffix for the run directory")
        p.add_argument("--set", action="append", default=[], help="override, e.g. device=cpu")
        p.add_argument("--allow-download", action="store_true")
        p.add_argument("--n-boot", type=int, default=1000)
        p.set_defaults(fn=fn)

    p = sub.add_parser("evaluate", help="(re)compute metrics for a run directory")
    p.add_argument("run_dir")
    p.add_argument("--n-boot", type=int, default=1000)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("diagnose", help="run a diagnostic suite (configs/diagnostics)")
    p.add_argument("config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--summarise-only", action="store_true", help="recompute comparisons from saved runs")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("report", help="write reports/results.md from real (non-synthetic) runs")
    p.add_argument("--runs")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("error-analysis", help="error analysis for a run (+ diagnostics)")
    p.add_argument("run_dir")
    p.add_argument("--diagnostics", nargs="*", default=[])
    p.add_argument("--out")
    p.set_defaults(fn=cmd_error_analysis)

    p = sub.add_parser("export-native", help="manifests in official Open-VCLIP / TC-CLIP format")
    p.add_argument("--backbone", choices=["openvclip", "tcclip"], required=True)
    p.add_argument("--datasets", nargs="+", default=all_ds)
    p.add_argument("--out", required=True)
    p.add_argument("--per-class", type=int, help="same seeded subset as `infer --per-class`")
    p.add_argument("--seed", type=int, default=0, help="must equal the experiment's seed")
    p.set_defaults(fn=cmd_export_native)

    p = sub.add_parser("compare-native", help="official Open-VCLIP results vs adapter run")
    p.add_argument("--official", required=True, help="pickle written by TEST.SAVE_RESULTS_PATH")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--trust-pickle", action="store_true")
    p.set_defaults(fn=cmd_compare_native)

    p = sub.add_parser("fetch-clip", help="download OpenAI CLIP weights (explicit, large)")
    p.add_argument("--arch", choices=["ViT-B-32", "ViT-B-16"], required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_fetch_clip)

    sub.add_parser("fetch-repos", help="clone the official repositories at pinned commits").set_defaults(
        fn=cmd_fetch_repos)

    p = sub.add_parser("smoke-test", help="end-to-end check on synthetic fixtures")
    p.add_argument("--fake", action="store_true", help="use the test backbone instead of CLIP weights")
    p.set_defaults(fn=cmd_smoke_test)

    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except RuntimeError as exc:  # includes BackboneUnavailable: actionable, no traceback needed
        sys.exit(f"error: {exc}")
