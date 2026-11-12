"""Generate the Kaggle notebooks of Experiment 2:

    python kaggle/make_notebooks.py     # writes kaggle/e2_imagenet100.ipynb and kaggle/e2_cub.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = {
    "imagenet100": """
# Experiment 2 on Kaggle: SCD reproduction on ImageNet-100 (protocol A)

**What runs.** Rebuild + verify the 67,893-name WordNet vocabulary against the authors' text embeddings; CLIP
ViT-B/16 (official package, fp16), DINO ViT-B/16 and the authors' ImageNet-100 GCD ViT-B/16 features with CLIP's
preprocessing; k-means (unsupervised) and CSS-k-means (partially supervised); every naming variant; evaluation.
No textual enhancement (CC12M) here.

**Inputs.** GPU T4, Internet on. Attach the competition **ImageNet Object Localization Challenge**
(`imagenet-object-localization-challenge`). Optional: the output of an Experiment 1 ImageNet-100 notebook
(its `e1_runs/imagenet100/data` manifest is reused; otherwise it is built here with the upstream partition check).

**Resuming.** Every stage writes to `/kaggle/working/e2_outputs` and skips work already done. If a session ends,
open this notebook again, *Add Input → Notebook output* of the previous version, and run again.
""",
    "cub": """
# Experiment 2 on Kaggle: CUB protocols B (fixed clusters) and R (refinement from GCD clusters)

**Inputs.** GPU or CPU; Internet on (the official CUB archive is streamed from data.caltech.edu and MD5-checked).
Attach the output of the Experiment 1 CUB notebook (it contains `e1_runs/cub`). These stages also run locally
(`README.md`); this notebook is for machines without the Experiment 1 outputs.
""",
}

SETUP = r'''
import json, os, shutil, subprocess, sys, time
from pathlib import Path

DATASET = "{dataset}"
REPO, REF = "eylulpelinkilic/generalized-category-discovery", "main"   # set REF to a commit to pin
WORK = Path("/kaggle/working")
CODE = WORK / "generalized-category-discovery"
EXP1 = CODE / "Experiment1_GCD_CLIP_Baseline"
EXP2 = CODE / "Experiment2_SCD"
E1_RUNS = WORK / "e1_runs"
E2_OUT = WORK / "e2_outputs"
STATUS, T0 = {{}}, time.time()


def run(cmd, cwd=None, allow_fail=False):
    print(f"$ {{cmd}}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd or EXP2, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
run("pip install -q open_clip_torch==3.3.0 ftfy regex nltk ortools", cwd=WORK)
run("pip install -q git+https://github.com/openai/CLIP.git", cwd=WORK)
import torch
print(torch.__version__, torch.cuda.is_available())
'''

RESTORE = r'''
def restore(name, target):
    for depth in range(5):
        for cand in Path("/kaggle/input").glob("*/" * depth + name):
            if not target.exists():
                print("restoring", cand, "->", target)
                shutil.copytree(cand, target)
            return str(cand)
    return None


PREV_E1 = restore(f"e1_runs/{DATASET}", E1_RUNS / DATASET)
PREV_E2 = restore("e2_outputs", E2_OUT)
print("previous Experiment 1 outputs:", PREV_E1, "| previous Experiment 2 outputs:", PREV_E2)
'''

DATA = {
    "imagenet100": r'''
def locate_imagenet():
    for depth in range(6):
        for cand in Path("/kaggle/input").glob("*/" * depth + "CLS-LOC"):
            if (cand / "train").is_dir():
                return cand
    raise RuntimeError("attach the competition imagenet-object-localization-challenge")


IMAGE_ROOT = locate_imagenet()
os.environ["E1_IMAGE_ROOT"] = str(IMAGE_ROOT)
E1 = (f"python -m e1 --config configs/imagenet100.yaml --set paths.runs_root={E1_RUNS} "
      f"--set dataset.root=/kaggle/input --set dataset.image_root={IMAGE_ROOT}")
E2 = f"python -m e2 --config configs/imagenet100.yaml --set paths.outputs={E2_OUT} --set e1.runs_root={E1_RUNS}"
''',
    "cub": r'''
assert (E1_RUNS / "cub" / "grounding").exists(), "attach the Experiment 1 CUB notebook output (e1_runs/cub)"
E2 = (f"python -m e2 --config configs/cub.yaml --set paths.outputs={E2_OUT} --set e1.runs_root={E1_RUNS} "
      f"--set dataset.archive_cache={WORK / 'cub_images'}")
''',
}

PIPELINE = {
    "imagenet100": r'''
CSS_SEEDS = [0]          # CSS-k-means is the slowest stage (min-cost flow, 10 inits x 10 iterations); add 1, 2 later


def manifest():
    if (E1_RUNS / DATASET / "data" / "manifest.csv").exists():
        print("Experiment 1 manifest present")
        return
    run(f"{E1} prepare-data --upstream-check", cwd=EXP1)


step("manifest", manifest)
step("vocabulary", lambda: run(f"{E2} vocab"))
step("features", lambda: run(f"{E2} repro features"), needs=["manifest"])
step("cluster_unsupervised", lambda: run(f"{E2} repro cluster --setting unsupervised"), needs=["features"])
for s in CSS_SEEDS:
    step(f"cluster_partial_s{s}", lambda s=s: run(f"{E2} repro cluster --setting partially_supervised --seed {s}"),
         needs=["features"])
step("name", lambda: run(f"{E2} repro name"), needs=["vocabulary", "features"])
step("evaluate", lambda: run(f"{E2} repro evaluate"), needs=["name"])
step("report", lambda: run(f"{E2} report"))
''',
    "cub": r'''
step("vocabulary", lambda: run(f"{E2} vocab"))
step("embeddings", lambda: run(f"{E2} cub-embed --clip vitb32"))
step("protocol_B", lambda: run(f"{E2} fixed --clip vitb32"), needs=["embeddings"])
for c in (0, 1, 2):
    for mode in ("unsupervised", "partially_supervised"):
        step(f"protocol_R_c{c}_{mode}", lambda c=c, mode=mode: run(f"{E2} refine-gcd --cluster-seed {c} --mode {mode}"),
             needs=["embeddings"])
step("report", lambda: run(f"{E2} report"))
''',
}

PACKAGE = r'''
(E2_OUT / f"NOTEBOOK_STATUS_{DATASET}.json").write_text(json.dumps({"commit": COMMIT, "status": STATUS}, indent=2))
zip_path = WORK / f"e2_results_{DATASET}.zip"
if zip_path.exists():
    zip_path.unlink()
# small download: results, traces, errors and galleries; features, assets, caches and top-k files stay in this
# version's output for the next session
run(f"cd {E2_OUT} && zip -qr {zip_path} . -x 'cache/*' 'assets/*' '*/features/*' '*/topk/*' '*/inputs/*/embeddings.npz'",
    cwd=WORK)
if (EXP2 / "reports").exists():
    shutil.copytree(EXP2 / "reports", WORK / "e2_reports", dirs_exist_ok=True)
print("=" * 60)
for k, v in STATUS.items():
    print(f"{k:24s} {v}")
print("=" * 60)
'''


def notebook(dataset: str) -> dict:
    cells = []
    md = lambda t: cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)})  # noqa: E731
    code = lambda t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],  # noqa: E731
                                   "source": t.strip().splitlines(True)})
    md(INTRO[dataset])
    md("## 1. Code and environment")
    code(SETUP.format(dataset=dataset))
    md("## 2. Restore previous outputs (if attached)")
    code(RESTORE)
    md("## 3. Data and commands")
    code(DATA[dataset])
    md("## 4. Stages (each skips completed work)")
    code(PIPELINE[dataset])
    md("## 5. Status and results archive")
    code(PACKAGE)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    for ds in ("imagenet100", "cub"):
        path = HERE / f"e2_{ds}.ipynb"
        path.write_text(json.dumps(notebook(ds), indent=1) + "\n")
        print("wrote", path)
