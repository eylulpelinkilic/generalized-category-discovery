"""End-to-end properties on a tiny synthetic dataset (software checks; random weights)."""
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e1 import clustering, ground
from e1.clip_extract import extract_images, load_images
from e1.config import Layout, load_config
from e1.gcd_features import extract, load_features
from e1.gcd_train import Trainer
from e1.manifests import Manifest, build_smoke
from e1.matrices import similarity
from e1.provenance import write_atomic


def make_cfg(root):
    return load_config("smoke", [f"paths.runs_root={root}", "gcd_train.epochs=2", "episodes.sizes=[3,2]",
                                 "episodes.paired_distractors=1", "grounding.n_bootstrap=20",
                                 "grounding.representations=[all,centre1]"])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cfg = make_cfg(cls.tmp)
        cls.layout = Layout(cls.cfg)
        build_smoke(cls.layout.data, cls.layout.base / "images", n_classes=4, train_per_class=8, heldout_per_class=4)
        Trainer(cls.cfg, cls.layout, 0).train(max_epochs=1)          # interrupted after one epoch ...
        cls.resumed = Trainer(cls.cfg, cls.layout, 0).train()          # ... and resumed
        extract(cls.cfg, cls.layout, 0)
        cls.cdir = clustering.run(cls.cfg, cls.layout, 0, "known", 0)
        clustering.evaluate_clustering(cls.layout, cls.cdir)
        extract_images(cls.cfg, cls.layout, "vitb32")
        cls.gdir = ground.run(cls.cfg, cls.layout, 0, "known", 0, "vitb32")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_resume_is_identical_to_an_uninterrupted_run(self):
        self.assertTrue(self.resumed["resumed_from_checkpoint"])
        with tempfile.TemporaryDirectory() as other:
            cfg = make_cfg(other)
            lay = Layout(cfg)
            shutil.copytree(self.layout.data, lay.data)
            Trainer(cfg, lay, 0).train()
            a = torch.load(self.layout.gcd_run(0) / "checkpoints" / "final.pt", weights_only=False)
            b = torch.load(lay.gcd_run(0) / "checkpoints" / "final.pt", weights_only=False)
            self.assertEqual(a["epoch_completed"], b["epoch_completed"])
            self.assertEqual(a["train_ids_hash"], b["train_ids_hash"])
            for k in a["model"]:
                self.assertTrue(torch.equal(a["model"][k], b["model"][k]), k)

    def test_resume_refuses_a_different_manifest(self):
        ckpt = self.layout.gcd_run(0) / "checkpoints" / "last.pt"
        state = torch.load(ckpt, weights_only=False)
        state["manifest_sha256"] = "other"
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d)
            lay = Layout(cfg)
            shutil.copytree(self.layout.data, lay.data)
            (lay.gcd_run(0) / "checkpoints").mkdir(parents=True)
            torch.save(state, lay.gcd_run(0) / "checkpoints" / "last.pt")
            with self.assertRaises(RuntimeError):
                Trainer(cfg, lay, 0).train()

    def test_clip_cache_reuse_order_and_fresh_agreement(self):
        again = extract_images(self.cfg, self.layout, "vitb32")
        self.assertEqual(again["shards_written"], 0)
        m = Manifest(self.layout.data)
        ids = m.ids("unlabelled")
        cached = load_images(self.layout.clip("vitb32"), ids)
        np.testing.assert_allclose(load_images(self.layout.clip("vitb32"), ids[::-1]), cached[::-1])
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d)
            lay = Layout(cfg)
            shutil.copytree(self.layout.data, lay.data)
            extract_images(cfg, lay, "vitb32")
            np.testing.assert_allclose(load_images(lay.clip("vitb32"), ids), cached, atol=1e-5)

    def test_sample_ids_align_across_gcd_clip_and_matrices(self):
        rows = pd.read_csv(self.gdir / "rows.csv")
        a = pd.read_csv(self.cdir / "assignments.csv")
        disc = rows[rows["population"] == "discovery"]
        self.assertEqual(set(disc["sample_id"]), set(a.loc[~a["is_anchor"], "sample_id"]))
        merged = disc.merge(a, on="sample_id", suffixes=("", "_gcd"))
        self.assertTrue((merged["cluster_id"] == merged["cluster_id_gcd"]).all())
        ids_u, _ = load_features(self.layout.gcd_run(0), "unlabelled")
        self.assertEqual(sorted(ids_u), sorted(a.loc[~a["is_anchor"], "sample_id"]))
        R = np.load(self.gdir / "R_single.npy")
        emb = load_images(self.layout.clip("vitb32"), rows["sample_id"].tolist())
        T = np.load(self.gdir / "text_embeddings_single.npz")["per_class"]
        np.testing.assert_allclose(R, similarity(emb, T), atol=1e-6)

    def test_anchors_are_excluded_from_discovery_aggregates(self):
        rows = pd.read_csv(self.gdir / "rows.csv")
        R = np.load(self.gdir / "R_single.npy")
        agg = np.load(self.gdir / "discovery_aggregates_single.npz")
        for k in range(len(agg["A"])):
            disc = rows[(rows["population"] == "discovery") & (rows["cluster_id"] == k)]["row"].to_numpy()
            if len(disc):
                np.testing.assert_allclose(agg["A"][k], R[disc].mean(0), atol=1e-6)
                self.assertEqual(agg["n_members"][k], len(disc))

    def test_diagnostic_export_saved_and_recomputed_similarities(self):
        import json

        from e1.diagnostics_export import export

        kw = dict(population="dev", n=3, seed=42, image_root=str(self.layout.base / "images"), allow_synthetic=True)
        out = export(self.cfg, 0, "known", 0, "vitb32", **kw)
        s = json.loads((out / "export_summary.json").read_text())
        self.assertTrue(s["validation"]["passed"], s["validation"])
        self.assertTrue(s["similarities"]["source"].startswith("saved"))
        ex = pd.read_csv(out / "examples.csv")
        self.assertEqual(len(ex), s["selected"]["incorrect"] + s["selected"]["correct"])
        self.assertEqual(s["selected"]["shortfall"]["incorrect"], max(0, 3 - s["eligible"]["incorrect"]))
        for d in ex["cluster_context_dir"]:
            self.assertTrue((out / d / "similarity_matrix.npz").exists() and (out / d / "heatmap.png").exists())
        strategy = json.loads((self.gdir / "selection.json").read_text())["prompt_strategy"]
        saved = self.gdir / f"R_{strategy}.npy"
        moved = saved.with_suffix(".bak")
        saved.rename(moved)
        try:
            out2 = export(self.cfg, 0, "known", 0, "vitb32", **kw)
            s2 = json.loads((out2 / "export_summary.json").read_text())
            self.assertTrue(s2["similarities"]["source"].startswith("recomputed"))
            self.assertTrue(s2["similarities"]["validation"]["passed"], s2["similarities"]["validation"])
            pd.testing.assert_frame_equal(ex.drop(columns=["clip_image_pred_score"]),
                                          pd.read_csv(out2 / "examples.csv").drop(columns=["clip_image_pred_score"]))
        finally:
            moved.rename(saved)

    def test_permuting_evaluation_labels_leaves_predictions_unchanged(self):
        before = pd.read_csv(self.gdir / "cluster_predictions.csv.gz")
        path = self.layout.data / "labels_eval.csv"
        original = path.read_text()
        t = pd.read_csv(path)
        t["class_id"] = np.random.RandomState(0).permutation(t["class_id"].to_numpy())
        try:
            write_atomic(path, t.to_csv(index=False))
            ground.run(self.cfg, self.layout, 0, "known", 0, "vitb32")
            after = pd.read_csv(self.gdir / "cluster_predictions.csv.gz")
        finally:
            write_atomic(path, original)
        pd.testing.assert_frame_equal(before, after)


if __name__ == "__main__":
    unittest.main()
