"""Generate the resumable Kaggle notebooks (one per dataset):

    python kaggle/make_notebooks.py        # writes kaggle/e1_cub.ipynb and kaggle/e1_imagenet100.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = {
    "cub": """
# Experiment 1 on Kaggle: CUB-200-2011 (GCD discovery + frozen CLIP grounding)

**Inputs.** None needed: the official CUB archive is downloaded from data.caltech.edu (MD5-checked)
and the DINO checkpoint from dl.fbaipublicfiles.com. GPU T4, Internet on.

**Resuming.** Every epoch writes a full checkpoint under `/kaggle/working/e1_runs`. If a session
ends before 200 epochs, open this notebook again, *Add Input → Notebook output* of the previous
version, and run again: section 2 copies the previous `e1_runs` back and training continues from
the last completed epoch (same configuration and manifest, or it refuses). CUB needs about one
session; the benchmark in section 5 gives the measured estimate.
""",
    "imagenet100": """
# Experiment 1 on Kaggle: ImageNet-100 (GCD discovery + frozen CLIP grounding)

**Inputs.** Join the competition **ImageNet Object Localization Challenge**
(`imagenet-object-localization-challenge`, accept its rules once on kaggle.com) and attach it with
*Add Input → Competitions*. It holds the ILSVRC-2012 CLS-LOC images used by GCD. GPU T4, Internet on.

**Resuming.** Full ImageNet-100 training (200 epochs, ~127k images, two views) does not fit in one
12-hour session: the first session measures throughput and prints the estimated number of
sessions. Each session trains until its time budget, saving a full checkpoint every epoch under
`/kaggle/working/e1_runs`. For the next session, open this notebook again, *Add Input → Notebook
output* of the previous version (keep the competition input), and run again; training continues
from the last completed epoch and refuses to resume a different configuration or manifest.
""",
}

SETUP = r'''
import json, os, shutil, subprocess, sys, time
from pathlib import Path

DATASET = "{dataset}"
REPO, REF = "eylulpelinkilic/generalized-category-discovery", "main"   # set REF to a commit to pin
TRAIN_SEED = 0              # 0 is the primary seed; run 1 and 2 as separate notebook copies
TRAIN_HOURS = {train_hours}         # training budget per session (Kaggle stops at 12 h)
CLUSTER_SEEDS = [0, 1, 2]
RUN_BENCHMARK = True        # once per dataset; skipped if a benchmark file exists
RUN_ESTIMATED_K = True
RUN_VITB16 = True           # CLIP ViT-B/16 robustness comparison after the primary ViT-B/32
WORK = Path("/kaggle/working")
CODE = WORK / "generalized-category-discovery"
EXP = CODE / "Experiment1_GCD_CLIP_Baseline"
RUNS = WORK / "e1_runs"
STATUS, T0 = {{}}, time.time()


def run(cmd, cwd=None, allow_fail=False):
    print(f"$ {{cmd}}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd or (EXP if EXP.exists() else WORK), text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line.rstrip()])[-12:]
    code = p.wait()
    if code and not allow_fail:
        raise RuntimeError(f"exit {{code}}: {{cmd}}\n" + "\n".join(tail))
    return code


def step(name, fn, needs=()):
    missing = [n for n in needs if STATUS.get(n) != "ok"]
    if missing:
        STATUS[name] = f"skipped: needs {{missing}}"
        return
    try:
        fn()
        STATUS[name] = "ok"
    except Exception as exc:
        STATUS[name] = f"failed: {{exc}}"
        print(f"!! {{name}} failed: {{exc}}")


if not CODE.exists():
    run(f"git clone -q https://github.com/{{REPO}}.git {{CODE}}", cwd=WORK)
run(f"git fetch -q origin {{REF}} && git checkout -q FETCH_HEAD", cwd=CODE)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
print("code commit", COMMIT)
run("pip install -q open_clip_torch==3.3.0 ftfy regex")
run("pip install -q git+https://github.com/openai/CLIP.git", allow_fail=True)   # reference for verify-clip
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "Settings -> Accelerator: GPU T4"
'''

RESTORE = r'''
def restore_previous():
    """Copy e1_runs/<dataset> from an attached previous notebook output (if any)."""
    for depth in range(5):
        for cand in Path("/kaggle/input").glob("*/" * depth + f"e1_runs/{DATASET}"):
            target = RUNS / DATASET
            if not target.exists():
                print("restoring previous outputs from", cand)
                shutil.copytree(cand, target)
            return str(cand)
    return None


PREVIOUS = restore_previous()
print("previous outputs:", PREVIOUS)
'''

DATA = {
    "cub": r'''
def locate_cub():
    for depth in range(5):
        for cand in Path("/kaggle/input").glob("*/" * depth + "CUB_200_2011/images.txt"):
            return cand.parent
    sys.path.insert(0, str(EXP))
    from e1.sources import fetch_cub
    return Path(fetch_cub("/tmp/cub")["path"])


IMAGE_ROOT = locate_cub()
SETS = f"--set paths.runs_root={RUNS} --set dataset.root={IMAGE_ROOT.parent} --set dataset.image_root={IMAGE_ROOT}"
''',
    "imagenet100": r'''
def locate_imagenet():
    for depth in range(6):
        for cand in Path("/kaggle/input").glob("*/" * depth + "CLS-LOC"):
            if (cand / "train").is_dir():
                return cand
    raise RuntimeError("ImageNet CLS-LOC not found: attach the competition imagenet-object-localization-challenge")


IMAGE_ROOT = locate_imagenet()      # .../ILSVRC/Data/CLS-LOC; the builder finds LOC_*.txt/csv from /kaggle/input
SETS = f"--set paths.runs_root={RUNS} --set dataset.root=/kaggle/input --set dataset.image_root={IMAGE_ROOT}"
''',
}

PIPELINE = r'''
CFG = f"--config configs/{DATASET}.yaml"
WEIGHTS = Path("/tmp/weights")
DINO = WEIGHTS / "dino_vitbase16_pretrain.pth"
E1 = f"python -m e1 {CFG} {SETS} --set gcd_train.dino_checkpoint={DINO}"


def prepare():
    if (RUNS / DATASET / "data" / "manifest.csv").exists():
        print("manifest exists (immutable); checking it loads")
        run(f"{E1} report")
        return
    run(f"{E1} prepare-data --upstream-check")


def weights():
    run(f"{E1} fetch-dino --dest {WEIGHTS}")


def bench():
    if list((RUNS / DATASET / "gcd").glob("benchmark_*.json")):
        print("benchmark exists")
        return
    run(f"{E1} benchmark --train-seed {TRAIN_SEED} --iters 20 --amp-modes fp32,amp")


def train():
    left = TRAIN_HOURS - (time.time() - T0) / 3600
    run(f"{E1} train --train-seed {TRAIN_SEED} --time-budget-hours {max(0.2, left):.2f}")


def finished():
    return any((RUNS / DATASET / "gcd").glob(f"seed{TRAIN_SEED}_*/checkpoints/final.pt"))


def downstream():
    run(f"{E1} extract-gcd --train-seed {TRAIN_SEED}")
    for c in CLUSTER_SEEDS:
        run(f"{E1} cluster --train-seed {TRAIN_SEED} --k-mode known --cluster-seed {c}")
    if RUN_ESTIMATED_K:
        run(f"{E1} cluster --train-seed {TRAIN_SEED} --k-mode estimated --cluster-seed 0")
    clips = ["vitb32"] + (["vitb16"] if RUN_VITB16 else [])
    for clip in clips:
        run(f"{E1} verify-clip --clip {clip} --n-images 32")
        run(f"{E1} extract-clip --clip {clip}")
        run(f"{E1} ground --train-seed {TRAIN_SEED} --k-mode known --cluster-seed 0 --clip {clip}")


step("prepare_data", prepare)
step("weights", weights)
if RUN_BENCHMARK:
    step("benchmark", bench, needs=["prepare_data", "weights"])
step("train", train, needs=["prepare_data", "weights"])
if finished():
    step("downstream", downstream, needs=["prepare_data"])
else:
    STATUS["downstream"] = "waiting: training not finished (attach this version's output and run again)"
step("report", lambda: run(f"{E1} report"))
'''

PACKAGE = r'''
(RUNS / DATASET / "NOTEBOOK_STATUS.json").write_text(json.dumps({"commit": COMMIT, "status": STATUS,
                                                               "train_seed": TRAIN_SEED}, indent=2))
# small download: everything except checkpoints, CLIP shards and image matrices (those stay in this
# version's output for the next session and for the mean-shift experiment)
zip_path = WORK / f"e1_results_{DATASET}.zip"
if zip_path.exists():
    zip_path.unlink()
run(f"cd {RUNS} && zip -qr {zip_path} {DATASET} -x '*/checkpoints/*' '*/image_shards/*' '*/R_*.npy' "
    f"'*/features/*.npz' '*/images/*'", cwd=RUNS)
report = RUNS / DATASET / "report" / "BASELINE_REPORT.md"
if report.exists():
    from IPython.display import Markdown, display
    display(Markdown(report.read_text()))
print("=" * 60)
for k, v in STATUS.items():
    print(f"{k:14s} {v}")
print("=" * 60)
'''


def notebook(dataset: str, train_hours: float) -> dict:
    cells = []
    md = lambda t: cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)})  # noqa: E731
    code = lambda t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],  # noqa: E731
                                   "source": t.strip().splitlines(True)})
    md(INTRO[dataset])
    md("## 1. Code and environment")
    code(SETUP.format(dataset=dataset, train_hours=train_hours))
    md("## 2. Restore the previous session (if its output is attached)")
    code(RESTORE)
    md("## 3. Data location")
    code(DATA[dataset])
    md("## 4–7. Manifest, weights, benchmark, training (time-budgeted), and every downstream stage once "
       "training has reached its final epoch")
    code(PIPELINE)
    md("## 8. Status, report and results archive")
    code(PACKAGE)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    for ds, hours in (("cub", 10.5), ("imagenet100", 10.5)):
        path = HERE / f"e1_{ds}.ipynb"
        path.write_text(json.dumps(notebook(ds, hours), indent=1) + "\n")
        print("wrote", path)
