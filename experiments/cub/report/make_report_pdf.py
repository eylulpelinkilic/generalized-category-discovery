"""Build the Experiment 1 CUB results report (HTML -> PDF). All numbers come from report_cub.json."""
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = json.load(open(sys.argv[1]))
OUT_HTML, COMPARISON_PNG = Path(sys.argv[2]), Path(sys.argv[3])
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1f", "#5a5a5a", "#e4e4e4"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.edgecolor": "#bdbdbd", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def ci(v, d=1):
    return f"{100 * v[0]:.{d}f} <span class='ci'>[{100 * v[1][0]:.{d}f}, {100 * v[1][1]:.{d}f}]</span>"


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


g32, g16 = R["grounding"]["vitb32"], R["grounding"]["vitb16"]
sized32 = pd.DataFrame(g32["sized"])
sizes = [20, 50, 100, 150]

# ---- figure: AUROC by vocabulary size
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ends = []
for m, col, lab in (("max_mean_sim", BLUE, "max similarity"), ("msp_dev", ORANGE, "softmax (dev T)"),
                    ("mcm_aggregate", AQUA, "MCM"), ("vote_fraction", YELLOW, "vote fraction")):
    t = sized32[(sized32.method == m) & sized32["size"].isin(sizes)].sort_values("size")
    ax.plot(t["size"], t["auroc_label_available_mean"], color=col, lw=2, marker="o", ms=4.5,
            markeredgecolor="white", markeredgewidth=0.8)
    ends.append([t["auroc_label_available_mean"].iloc[-1], lab])
ends.sort(reverse=True)                      # spread end labels so they never overlap
for i in range(1, len(ends)):
    ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.022)
for y, lab in ends:
    ax.text(153, y, lab, va="center", fontsize=8, color=INK)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.text(20, 0.505, "chance", fontsize=7.5, color=MUTED, va="bottom")
ax.set_xticks(sizes)
ax.set_xlabel("candidate names in the vocabulary (of 200)")
ax.set_ylabel("AUROC: label present vs absent")
ax.set_xlim(14, 185)
ax.set_ylim(0.45, 0.85)
style(ax)
F_AUROC = fig_b64(fig)

# ---- figure: end-to-end accuracy with and without rejection (CI from the cluster bootstrap)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
boot = g32["bootstrap"]["metrics"]
for m, col, lab in (("max_mean_sim", BLUE, "with rejection (max similarity, calibrated)"), ("forced_top1", ORANGE, "always name (forced)")):
    xs = [20, 50, 100, 150, 200]
    pts = [boot[f"{m}|size={s}"]["end_to_end@tau_e2e"] for s in xs]
    y = [p[0] for p in pts]
    lo = [p[1][0] for p in pts]
    hi = [p[1][1] for p in pts]
    ax.fill_between(xs, lo, hi, color=col, alpha=0.15, linewidth=0)
    ax.plot(xs, y, color=col, lw=2, marker="o", ms=4.5, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(lab, (xs[0], y[0]), xytext=(4, 9 if m == "max_mean_sim" else -12), textcoords="offset points",
                fontsize=8, color=INK)
ax.set_xticks([20, 50, 100, 150, 200])
ax.set_xlabel("candidate names in the vocabulary (of 200)")
ax.set_ylabel("end-to-end accuracy (images)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylim(0, 1)
style(ax)
F_E2E = fig_b64(fig)

# ---- figure: training loss
fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.plot(range(1, len(R["training"]["losses"]) + 1), R["training"]["losses"], color=BLUE, lw=1.4)
ax.set_xlabel("epoch")
ax.set_ylabel("training loss")
style(ax)
F_LOSS = fig_b64(fig)
F_COMP = base64.b64encode(COMPARISON_PNG.read_bytes()).decode()

# ---- tables
cl = {(c["k_mode"], c["seed"]): c for c in R["clustering"]}
rows_disc = "".join(
    f"<tr><td>{'known' if k == 'known' else 'estimated'} K = {c['K']}</td><td>{c['seed']}</td><td>{pct(c['v2_all'])}</td>"
    f"<td>{pct(c['v2_old'])}</td><td>{pct(c['v2_new'])}</td><td>{c['nmi']:.3f}</td><td>{c['ari']:.3f}</td>"
    f"<td>{c['singleton_clusters']}</td></tr>"
    for (k, s), c in sorted(cl.items(), key=lambda kv: (kv[0][0] != "known", kv[0][1])))

def comp_row(name, g, m):
    c = g["complete"][m]
    return (f"<tr><td>{name}</td><td>{pct(c['image_weighted'])}</td><td>{pct(c['class_macro'])}</td>"
            f"<td>{pct(c['old'])}</td><td>{pct(c['new'])}</td></tr>")

rows_complete = (comp_row("ViT-B/32, forced top-1 (80 prompts)", g32, "forced_top1") + comp_row("ViT-B/32, MCM prompt", g32, "mcm_aggregate")
                 + comp_row("ViT-B/16, forced top-1 (80 prompts)", g16, "forced_top1") + comp_row("ViT-B/16, MCM prompt", g16, "mcm_aggregate"))

def size_rows(g):
    s = pd.DataFrame(g["sized"])
    out = []
    for size in (200, 150, 100, 50, 20):
        r = s[(s["size"] == size) & (s.method == "max_mean_sim")].iloc[0]
        f = s[(s["size"] == size) & (s.method == "forced_top1")].iloc[0]
        b = g["bootstrap"]["metrics"][f"max_mean_sim|size={size}"]
        au = "—" if np.isnan(r["auroc_label_available_mean"]) else f"{r['auroc_label_available_mean']:.2f}"
        out.append(f"<tr><td>{size}{' (complete)' if size == 200 else ''}</td><td>{au}</td><td>{pct(f['forced_accuracy_available_mean'])}</td>"
                   f"<td>{pct(r['coverage@tau_e2e_mean'])}</td><td>{pct(r['accuracy_among_accepted@tau_e2e_mean'])}</td>"
                   f"<td>{pct(r['false_acceptance_absent@tau_e2e_mean'])}</td><td>{ci(b['end_to_end@tau_e2e'])}</td>"
                   f"<td>{pct(f['end_to_end_accuracy@tau_e2e_mean'])}</td>"
                   f"<td>{pct(r['end_to_end_accuracy_old@tau_e2e_mean'])} / {pct(r['end_to_end_accuracy_new@tau_e2e_mean'])}</td></tr>")
    return "".join(out)

METHOD_NAMES = {"max_mean_sim": "max similarity", "margin": "top-1 − top-2 margin", "msp_pilot": "softmax, pilot T = 0.01",
                "msp_dev": "softmax, dev T* = 0.02", "mcm_aggregate": "MCM (T = 1)", "mcm_image_mean": "MCM, per-image mean",
                "vote_fraction": "vote fraction", "neg_softmax_entropy": "softmax entropy", "neg_vote_entropy": "vote entropy"}
pdiff = g32["bootstrap"]["paired_differences"]
au_rows = []
for m, lab in METHOD_NAMES.items():
    cells = []
    for size in (150, 100, 50, 20):
        v = sized32[(sized32.method == m) & (sized32["size"] == size)]["auroc_label_available_mean"].iloc[0]
        cells.append(f"<td>{v:.2f}</td>")
    if m == "max_mean_sim":
        d = "reference"
    else:
        dd = pdiff[f"{m}-max_mean_sim|size=100|auroc"]
        d = f"{100 * dd[0]:+.1f} <span class='ci'>[{100 * dd[1][0]:+.1f}, {100 * dd[1][1]:+.1f}]</span>"
    au_rows.append(f"<tr><td>{lab}</td>{''.join(cells)}<td>{d}</td></tr>")
au_rows = "".join(au_rows)

pair = pd.DataFrame(g32["paired"])
def prow(variant, method):
    r = pair[(pair.variant == variant) & (pair.method == method)].iloc[0]
    rep = "—" if np.isnan(r["named_replacement"]) else pct(r["named_replacement"], 0)
    return f"<td>{pct(r['named_target'], 0)}</td><td>{rep}</td><td>{pct(r['accepted@tau_e2e'], 0)}</td>"
rows_pair = "".join(f"<tr><td>{lab}</td>{prow(v, 'max_mean_sim')}</tr>" for v, lab in
                    (("paired_present", "true name present"), ("paired_random", "true name replaced by a random class"),
                     ("paired_hard", "true name replaced by the most similar class name")))

comp = pd.DataFrame(R["diag"]["comparison"]).set_index("feature")
FEAT = [("sq_dist_to_gcd_centre", "distance to the GCD cluster centre", 3),
        ("profile_corr_percentile_in_cluster", "typicality of the image's row within its cluster (percentile)", 2),
        ("profile_corr_with_cluster_mean", "correlation of the image's row with its cluster's mean row", 2),
        ("norm_entropy_T0.02", "flatness of the row (normalised entropy)", 2),
        ("top1_minus_top2", "gap between the two highest similarities", 4),
        ("cluster_mean_pairwise_profile_corr", "cluster: agreement among member rows", 2),
        ("cluster_mean_top1_minus_top2", "cluster: peakedness of the mean row", 4)]
def frow(f, lab, d):
    r = comp.loc[f]
    a, ac = r["auroc_incorrect_higher_selected"], json.loads(r["auroc_ci95_selected"])
    b, bc = r["auroc_incorrect_higher_all_in_clusters"], json.loads(r["auroc_ci95_all_in_clusters_cluster_bootstrap"])
    return (f"<tr><td>{lab}</td><td>{r['median_incorrect']:.{d}f}</td><td>{r['median_correct']:.{d}f}</td>"
            f"<td>{a:.2f} <span class='ci'>[{ac[0]:.2f}, {ac[1]:.2f}]</span></td><td>{b:.2f} <span class='ci'>[{bc[0]:.2f}, {bc[1]:.2f}]</span></td></tr>")
rows_feat = "".join(frow(*x) for x in FEAT)

v32 = R["clip"]["vitb32"]
c0 = cl[("known", 0)]
dsum = R["diag"]["summary"]
clu = g32["clusters"]
b = R["benchmark"]
fp32 = next(x for x in b if not x["amp"])
amp = next(x for x in b if x["amp"])
counts = R["counts"]
max_au_100 = sized32[(sized32.method == "max_mean_sim") & (sized32["size"] == 100)]["auroc_label_available_mean"].iloc[0]
hard = pair[(pair.variant == "paired_hard") & (pair.method == "max_mean_sim")].iloc[0]
rand = pair[(pair.variant == "paired_random") & (pair.method == "max_mean_sim")].iloc[0]

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Experiment 1 results: CUB-200-2011</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
:root {{ --ink:#1f1f1f; --muted:#5a5a5a; --rule:#d9d9d9; --accent:#2a78d6; --soft:#f4f6f9; }}
body {{ font-family: Helvetica, Arial, sans-serif; color: var(--ink); font-size: 9.6pt; line-height: 1.42; background: #fff; }}
h1 {{ font-size: 18pt; margin: 0 0 2mm 0; letter-spacing: -0.2px; }}
h2 {{ font-size: 12.5pt; margin: 6mm 0 2mm 0; padding-top: 1mm; border-top: 1px solid var(--rule); }}
h3 {{ font-size: 10.5pt; margin: 4mm 0 1.5mm 0; }}
p {{ margin: 0 0 2mm 0; }}
.sub {{ color: var(--muted); font-size: 9pt; margin-bottom: 4mm; }}
.box {{ background: var(--soft); border-left: 3px solid var(--accent); padding: 3mm 4mm; margin: 3mm 0; }}
.box ul {{ margin: 0; padding-left: 4.5mm; }} .box li {{ margin: 0 0 1.4mm 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 2mm 0 1mm 0; font-size: 8.6pt; page-break-inside: avoid; }}
th {{ text-align: left; font-weight: 600; border-bottom: 1px solid #9e9e9e; padding: 1.3mm 1.6mm; vertical-align: bottom; }}
td {{ border-bottom: 1px solid #ececec; padding: 1.1mm 1.6mm; vertical-align: top; }}
td:not(:first-child), th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
td:not(:first-child) {{ white-space: nowrap; }}
.ci {{ color: var(--muted); font-size: 7.8pt; }}
.cap {{ color: var(--muted); font-size: 8.2pt; margin: 0 0 3mm 0; }}
.fig {{ page-break-inside: avoid; margin: 2mm 0; }}
.fig img {{ width: 100%; }}
.two {{ display: flex; gap: 5mm; }} .two > div {{ flex: 1; }}
.flow {{ display: flex; gap: 1.5mm; align-items: stretch; margin: 2mm 0 3mm 0; font-size: 8pt; }}
.flow div {{ flex: 1; border: 1px solid var(--rule); border-radius: 2mm; padding: 2mm; background: #fff; }}
.flow b {{ display: block; font-size: 8.4pt; margin-bottom: 0.8mm; }}
.small {{ font-size: 8.3pt; color: var(--muted); }}
.pb {{ page-break-before: always; }}
code {{ font-family: Menlo, monospace; font-size: 8pt; }}
</style></head><body>

<h1>Experiment 1: GCD discovery + frozen CLIP grounding</h1>
<div class="sub">Results on CUB-200-2011 · training seed 0 · report of 10 November 2026 · code commit <code>{R['training']['commit'][:7]}</code></div>

<div class="box"><b>Summary</b><ul>
<li><b>The discovery stage reproduces GCD.</b> On the unlabelled training images (known K = 200) the v2 accuracy is
{pct(c0['v2_all'])} / {pct(c0['v2_old'])} / {pct(c0['v2_new'])} (All / Old / New), against 51.3 / 56.6 / 48.7 in the GCD paper; three clustering seeds give {pct(min(c['v2_all'] for c in R['clustering'] if c['k_mode']=='known'))}–{pct(max(c['v2_all'] for c in R['clustering'] if c['k_mode']=='known'))} All.</li>
<li><b>CLIP names {pct(g32['complete']['forced_top1']['image_weighted'])}% of discovered images correctly</b> with the complete 200-name vocabulary
(95% CI {pct(g32['bootstrap']['metrics']['forced_top1|size=200']['forced_accuracy_available'][1][0])}–{pct(g32['bootstrap']['metrics']['forced_top1|size=200']['forced_accuracy_available'][1][1])}; Old {pct(g32['complete']['forced_top1']['old'])}, New {pct(g32['complete']['forced_top1']['new'])}).
Because each cluster gets one name, the best achievable with these clusters is {pct(clu['ceiling'])}%.</li>
<li><b>Rejection ("unmapped") helps only when many true names are missing.</b> With the complete vocabulary it costs accuracy; with 20 of 200 names it raises end-to-end accuracy from {pct(sized32[(sized32.method=='forced_top1')&(sized32['size']==20)]['end_to_end_accuracy@tau_e2e_mean'].iloc[0])}% to {pct(sized32[(sized32.method=='max_mean_sim')&(sized32['size']==20)]['end_to_end_accuracy@tau_e2e_mean'].iloc[0])}%, mostly by rejecting.</li>
<li><b>The simplest score separates present from missing names best</b> (max similarity, AUROC {max_au_100:.2f} at 100 names), but separation is weak at large vocabularies.
When a class's name is swapped for a similar-looking class, {pct(hard['named_replacement'], 0)}% of its images take the substitute's name.</li>
<li><b>Clustering errors are image-level.</b> In the development pool, misclustered images are farther from their GCD centre and less typical of their cluster's similarity pattern; clusters that contain errors look like the others.</li>
</ul></div>

<h2>1. What was run</h2>
<div class="flow">
<div><b>1 · Data</b>CUB-200-2011, SSB split: 100 known, 100 novel classes; half of the known-class training images are labelled.</div>
<div><b>2 · Discovery</b>Upstream GCD: DINO ViT-B/16, contrastive training 200 epochs (final epoch), semi-supervised k-means.</div>
<div><b>3 · Representation</b>Frozen OpenAI CLIP ViT-B/32 (and B/16) image and text embeddings.</div>
<div><b>4 · Grounding</b>Each cluster gets the name with the highest mean similarity, or is rejected ("unmapped") by a calibrated score.</div>
<div><b>5 · Evaluation</b>Image-level, on the unlabelled images; vocabularies of 200, 150, 100, 50 and 20 names.</div>
</div>
<p>All settings were fixed before the run (<code>PROTOCOL.md</code>). Labels are kept in separate tables by permission: labelled anchors
({counts['labelled']} images) for training; held-out official test images of the known classes for choosing prompts and the
softmax temperature (development pool, {counts['heldout_dev']}) and for setting rejection thresholds (calibration pool, {counts['heldout_calib']});
the {counts['unlabelled']} unlabelled training images ({counts['unlabelled_known']} known-class, {counts['unlabelled_novel']} novel-class) are used only for the final evaluation.
Evaluation is transductive: discovery saw these images without labels.</p>
<p>Incomplete vocabularies are random draws of 150, 100, 50 or 20 class names (5 draws each), shared by all clusters.
An image whose true name is in the vocabulary is correct only if its cluster is accepted with that name; an image whose name is missing is
correct only if its cluster is rejected (<b>end-to-end accuracy</b>). Thresholds are calibrated once across all vocabulary sizes.
Intervals are 95% bootstrap intervals that resample discovered clusters together with their images.</p>

<h2>2. Checks before the results</h2>
<table><tr><th>Check</th><th>Result</th></tr>
<tr><td>Labelled/unlabelled split equals the upstream GCD code's</td><td>{'passed' if R['upstream_check']['passed'] else 'FAILED'}: {R['upstream_check']['upstream_labelled']:,} labelled, {R['upstream_check']['upstream_unlabelled']:,} unlabelled</td></tr>
<tr><td>All images present and decodable</td><td>{R['files']['image_check']}; {R['files']['missing']} missing, {R['files']['corrupt']} corrupt</td></tr>
<tr><td>Training completed</td><td>{R['training']['epochs']} epochs in {R['training']['hours']} h on a {R['training']['env']['gpu']}, fp32 as upstream</td></tr>
<tr><td>CLIP matches OpenAI's official package (ViT-B/32)</td><td>min. cosine {v32['image_min_cosine_quickgelu_vs_reference']:.6f} (images), {v32['text_min_cosine_quickgelu_vs_reference']:.6f} (texts); tokens identical</td></tr>
<tr><td>The pilot's CLIP setup (GELU activation)</td><td>agrees with the official model on only {pct(v32['zero_shot_agreement_gelu_vs_reference'], 0)}% of zero-shot labels</td></tr>
<tr><td>Software tests</td><td>26 tests pass (partitions, k-means repairs, metrics, resumption, label isolation)</td></tr></table>
<div class="two"><div><div class="fig"><img src="data:image/png;base64,{F_LOSS}"></div>
<p class="cap">Training loss per epoch (seed 0). It drops in the first ~10 epochs and is flat afterwards.</p></div>
<div><p class="small">Throughput on a T4 (batch 128 images, 256 views): {fp32['images_per_sec']} images/s in fp32, {amp['images_per_sec']} with mixed precision.
Peak GPU memory {fp32['peak_gpu_gb']} GB. The earlier pilot's CLIP numbers should not be compared with these: its activation function differed from the trained model's.</p></div></div>

<h2>3. Discovery (GCD) on the unlabelled images</h2>
<table><tr><th>Number of clusters</th><th>Clustering seed</th><th>All</th><th>Old</th><th>New</th><th>NMI</th><th>ARI</th><th>Singletons</th></tr>
{rows_disc}
<tr><td><i>GCD paper (arXiv v2, Table 3)</i></td><td></td><td><i>51.3</i></td><td><i>56.6</i></td><td><i>48.7</i></td><td></td><td></td><td></td></tr></table>
<p class="cap">v2 accuracy: one Hungarian assignment of clusters to classes over all unlabelled images, with Old/New measured afterwards.
K was estimated at {R['estimate_k']['K_estimate']} from the labelled images only (true 200). The Hungarian assignment is used for these numbers only, never for naming.</p>
<p>Cluster quality limits naming: the median cluster purity is {clu['purity_median']:.2f}, and {clu['mixed']} of 200 clusters have no majority class;
they hold {clu['images_in_mixed']:,} of the {clu['images']:,} unlabelled images ({pct(clu['images_in_mixed']/clu['images'], 0)}%).</p>

<h2>4. Naming clusters with the complete vocabulary</h2>
<table><tr><th>CLIP model and prompts</th><th>Image-weighted</th><th>Class-macro</th><th>Old</th><th>New</th></tr>{rows_complete}
<tr><td><i>Ceiling: every cluster named after its majority class</i></td><td><i>{pct(clu['ceiling'])}</i></td><td></td><td></td><td></td></tr></table>
<p class="cap">Accuracy (%) of the cluster's name for each of its images. The development pool chose the 80-template prompt ensemble
over the single prompt for both models (T* = {g32['selection']['T_dev']}).</p>
<p>CLIP reaches about three quarters of what the clusters allow. Known-class images are named better than novel-class ones,
partly because known-class clusters are purer. ViT-B/16 is not better than ViT-B/32 here.</p>

<h2>5. Naming with rejection when true names are missing (CLIP ViT-B/32)</h2>
<div class="two"><div><div class="fig"><img src="data:image/png;base64,{F_AUROC}"></div>
<p class="cap">How well each score separates images whose true name is present from images whose name is missing (0.5 = chance).</p></div>
<div><div class="fig"><img src="data:image/png;base64,{F_E2E}"></div>
<p class="cap">End-to-end accuracy with and without rejection; shaded bands are 95% cluster-bootstrap intervals.</p></div></div>
<table><tr><th>Vocabulary</th><th>AUROC</th><th>Named correctly (name present)</th><th>Accepted</th><th>Correct among accepted</th><th>Missing-name images accepted</th><th>End-to-end, with rejection</th><th>End-to-end, forced</th><th>End-to-end Old / New</th></tr>
{size_rows(g32)}</table>
<p class="cap">Max-similarity score with its calibrated threshold (τ maximising calibration end-to-end accuracy), mean over 5 vocabulary draws per size.
"Named correctly" ignores rejection. Percentages except AUROC.</p>

<h3>All rejection scores</h3>
<table><tr><th>Score</th><th>AUROC, 150</th><th>100</th><th>50</th><th>20</th><th>Difference vs max similarity at 100 (AUROC points)</th></tr>{au_rows}</table>
<p class="cap">Paired differences use the same bootstrap resamples. Softmax-based scores, including MCM, change scale with the vocabulary size;
because thresholds are calibrated across all sizes at once, MCM ends up rejecting nearly every cluster (Section 8).</p>

<h3>Similar substitutes</h3>
<table><tr><th>10-name vocabulary built around one target class</th><th>Images given the target's name</th><th>Given the substitute's name</th><th>Accepted</th></tr>{rows_pair}</table>
<p class="cap">Averaged over all 200 target classes; images of the target class only. The "most similar" substitute is chosen by CLIP text similarity,
which favours CLIP; it shows the failure, not its typical rate. Example: Belted Kingfisher clusters named "Ringed Kingfisher".</p>

<h2>6. Correctly vs incorrectly clustered images (development pool)</h2>
<p>To look at errors without touching the evaluation images, {dsum['eligible']['incorrect'] + dsum['eligible']['correct']:,} held-out images of the known classes were assigned to the GCD
clusters and a cluster-to-class map was rebuilt on them (accuracy {pct(dsum['hungarian']['accuracy'])}%). 50 incorrectly and 50 correctly clustered images
(seed 42) were exported with their full similarity rows and their clusters' matrices. Every check passed (exported files are byte-identical
to the originals; the similarities reproduce the run's saved cluster scores within 2×10⁻⁷).</p>
<table><tr><th>Label-free measurement</th><th>Incorrect (median)</th><th>Correct (median)</th><th>AUROC, selected 100</th><th>AUROC, all {R['diag']['meta']['n_all_in_exported_clusters'][0] + R['diag']['meta']['n_all_in_exported_clusters'][1]} in these clusters</th></tr>{rows_feat}</table>
<p class="cap">AUROC above 0.5: higher for incorrectly clustered images; below 0.5: lower. Right column: every development image in the 63 exported clusters,
resampling clusters. Known classes only; exploratory (19 measurements compared, no multiplicity correction).</p>
<div class="fig"><img src="data:image/png;base64,{F_COMP}"></div>
<p class="cap">The 100 selected images; black bars are medians. Panels using the true label (top row and "rank_true_in_cluster_mean") describe the errors but could not detect them.</p>
<p>Misclustered images look like boundary cases: far from the GCD centre, less like their cluster-mates, and with flatter similarity rows.
The clusters containing them do not differ in spread or peakedness. These effects are moderate (AUROC 0.63–0.73) and suggest measuring
density and its gradient per image rather than per cluster in the planned mean-shift experiment.</p>

<h2>7. Limitations</h2>
<ul>
<li>One training seed and one dataset. Seeds 1 and 2 and ImageNet-100 have not run; no claim about variability across training runs yet.</li>
<li>CUB's fine-grained names are hard for CLIP: rows are nearly flat (top-two gaps below 0.01), so results may differ on ImageNet-100.</li>
<li>The development pool contains no novel classes, so Section 6 covers known-class errors only.</li>
<li>The GCD paper's estimated K for CUB (231) was not verified verbatim and is not compared.</li>
</ul>

<h2>8. Open decisions and next runs</h2>
<ul>
<li><b>Per-size calibration.</b> The vocabulary size is known when naming, so calibrating thresholds per size is legitimate and would fix the
softmax/MCM scale problem. Proposed as a secondary analysis before ImageNet-100.</li>
<li><b>Mixed precision.</b> fp32 training scales to roughly 175–210 GPU-hours per ImageNet-100 seed on a T4 (estimate from the CUB benchmark);
mixed precision to 62–75. Proposal: rerun CUB seed 0 with mixed precision and adopt it if results match within seed variation.</li>
<li><b>Seeds 1 and 2 on CUB</b> (about 10 hours each) to measure training-seed variability.</li>
</ul>

<h2>Appendix: provenance</h2>
<p class="small">Run <code>cub/gcd/seed0_3a2d202f8444</code> · clustering <code>known_c0_fcf46ef1931c</code> · grounding <code>vitb32_831109500c95</code> ·
final checkpoint sha256 <code>{R['training']['ckpt'][:16]}…</code> · code commit <code>{R['training']['commit']}</code> ·
{R['training']['env']['gpu']}, torch {R['training']['env']['torch']}. Result files: <code>experiments/cub/</code>; detailed report <code>report/BASELINE_REPORT.md</code>;
diagnostics <code>grounding/seed0_known_c0/vitb32_831109500c95/diagnostics/clustering_examples/</code>.
GCD paper reference: Vaze et al., CVPR 2022, arXiv:2201.02609v2, Table 3.</p>
</body></html>"""
OUT_HTML.write_text(html)
print("wrote", OUT_HTML)
