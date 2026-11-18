"""Protocol A stages `repro cluster`, `repro name`, `repro evaluate` end to end on a synthetic ImageNet-like set
(software check, not a result). Uses real ImageNet wnids, the verified 67,893-name vocabulary and the authors'
text weights from outputs/ (skipped when the vocabulary stage has not been run on this machine)."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e2  # noqa: E402,F401
from e2.config import EXP2, Layout, load_config  # noqa: E402

VOCAB_DIRS = sorted((EXP2 / "outputs" / "vocabulary").glob("wordnet30_official_*/alignment.json"))
# known: tench, goldfish, dalmatian (duplicate column); novel: Cardigan (duplicate column), hammerhead, electric ray
WNIDS = ["n01440764", "n01443537", "n02110341", "n02113186", "n01494475", "n01496331"]


def write_e1_manifest(data: Path, n_per_class=30, n_lab=10):
    data.mkdir(parents=True)
    rows, lab, ev = [], [], []
    for c, w in enumerate(WNIDS):
        for i in range(n_per_class):
            sid = f"imagenet:train:{w}_{i}"
            member = "labelled" if (c < 3 and i < n_lab) else "unlabelled"
            rows.append({"sample_id": sid, "rel_path": f"train/{w}/{w}_{i}.JPEG", "source_split": "train",
                         "gcd_membership": member, "dataset_version": "synthetic"})
            (lab if member == "labelled" else ev).append({"sample_id": sid, "class_id": c})
    pd.DataFrame(rows).to_csv(data / "manifest.csv", index=False)
    pd.DataFrame(lab).to_csv(data / "labels_labelled.csv", index=False)
    pd.DataFrame(ev).to_csv(data / "labels_eval.csv", index=False)
    pd.DataFrame(columns=["sample_id", "class_id"]).to_csv(data / "labels_heldout.csv", index=False)
    pd.DataFrame([{"class_id": c, "class_key": w, "source_index": c, "canonical_name": w, "aliases": "",
                   "gcd_status": "known" if c < 3 else "novel", "name_source": "synthetic"}
                  for c, w in enumerate(WNIDS)]).to_csv(data / "classes.csv", index=False)
    (data / "manifest_meta.json").write_text(json.dumps({"image_root": str(data), "manifest_sha256": "synthetic"}))


@unittest.skipUnless(VOCAB_DIRS and json.loads(VOCAB_DIRS[-1].read_text())["passed"],
                     "run `python -m e2 --config configs/cub.yaml vocab` first")
class TestReproStages(unittest.TestCase):
    def test_cluster_name_evaluate(self):
        from e2 import repro, vocab_stage
        from e1.manifests import Manifest

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = load_config("imagenet100", [
                f"paths.outputs={tmp / 'out'}", f"e1.runs_root={tmp / 'e1'}", "repro.n_cluster=6",
                "repro.kmeans.seeds=[0]", "repro.kmeans.n_init=2", "repro.css_kmeans.seeds=[0]",
                "repro.css_kmeans.n_init=2", "repro.css_kmeans.size_min=2", "repro.css_kmeans.size_max=60"])
            (tmp / "out").mkdir()
            (tmp / "out" / "vocabulary").symlink_to(VOCAB_DIRS[-1].parent.parent)
            write_e1_manifest(tmp / "e1" / "imagenet100" / "data")
            wn, vocab, W = vocab_stage.load(cfg)
            m = Manifest(tmp / "e1" / "imagenet100" / "data")
            ids = m.ids("labelled") + m.ids("unlabelled")
            y = np.array([int(s.split(":")[2].split("_")[0] == w) and WNIDS.index(w)
                          for s in ids for w in [s.split(":")[2].split("_")[0]]])
            targets = repro.class_targets(wn, m)
            rng = np.random.RandomState(0)
            clip = np.stack([W[vocab.first_index[targets[c]]] for c in y]) + 0.045 * rng.randn(len(y), W.shape[1])
            proto = rng.randn(6, 32)
            for model, F in (("clip", clip), ("dino", proto[y] + 0.3 * rng.randn(len(y), 32)),
                             ("gcd", proto[y] + 0.3 * rng.randn(len(y), 32))):
                d = Layout(cfg).features(model, "synthetic")
                d.mkdir(parents=True)
                F = (F / np.linalg.norm(F, axis=1, keepdims=True)).astype(np.float32)
                np.savez(d / "features.npz", sample_ids=np.array(ids).astype(str), features=F)
            args = types.SimpleNamespace(setting="both", seed=None, variants=None)
            clusters = repro.cluster(cfg, args)
            self.assertEqual(len(clusters), 2)
            runs = repro.name(cfg, args)
            self.assertEqual(len(runs), 6 + 7)                        # known_filter_fixed: partially supervised only
            out = repro.evaluate(cfg, args)
            summary = json.loads(Path(out["summary"]).read_text())
            self.assertEqual(len(summary["runs"]), 13)
            self.assertIn("ub", summary["zero_shot"])
            official = [r for r in summary["runs"] if r["variant"] == "official"]
            self.assertTrue(all(r["stop_reason"] in {"converged", "cycle", "max_iter"} for r in official))
            for r in summary["runs"]:
                run_dir = next(p for p in Path(r["seed_dir"]).iterdir() if p.name == r["variant"])
                for f in ("metrics.json", "trace.json", "image_predictions.csv.gz", "incorrect_decisions.csv.gz",
                          "error_patterns.json", "final_clusters.csv"):
                    self.assertTrue((run_dir / f).exists(), f"{run_dir.name}/{f}")
            # the stages are resumable: a second `name` skips every completed run
            self.assertTrue(all(v == "exists" for v in repro.name(cfg, args).values()))


if __name__ == "__main__":
    unittest.main()
