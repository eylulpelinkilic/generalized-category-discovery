"""End-to-end pipeline on synthetic fixtures with the deterministic test backbone (no weights)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ovar import fixtures, inference
from ovar.backbones.fake import FakeBackbone
from ovar.cache import file_identity
from ovar.config import stable_hash
from ovar.diagnostics import run_suite
from ovar.frames import degrade, temporal_positions
from ovar.metrics import evaluate_run
from ovar.report import error_analysis, results_report


class TestCacheKeys(unittest.TestCase):
    def test_every_component_changes_the_key(self):
        base = {"model": {"sha": "x"}, "sampling": {"n": 16}, "spatial": {"p": "clip"}, "pixel": None,
                "video": {"path": "a.mp4", "size": 1, "mtime_ns": 2}}
        key = stable_hash(base)
        self.assertEqual(key, stable_hash(json.loads(json.dumps(base))))
        for part, change in (("model", {"sha": "y"}), ("sampling", {"n": 8}), ("spatial", {"p": "vifi"}),
                             ("pixel", {"kind": "blur"}), ("video", {"path": "a.mp4", "size": 1, "mtime_ns": 3})):
            self.assertNotEqual(key, stable_hash({**base, part: change}), part)

    def test_file_identity_tracks_modification(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"a")
        before = file_identity(f.name)
        os.utime(f.name, ns=(before["mtime_ns"] + 10**9, before["mtime_ns"] + 10**9))
        self.assertNotEqual(before, file_identity(f.name))
        os.unlink(f.name)


class TestTransforms(unittest.TestCase):
    def test_temporal_positions(self):
        self.assertEqual(temporal_positions("single_frame", 16, "k").tolist(), [8] * 16)
        shuffled = temporal_positions("shuffled", 16, "k")
        self.assertEqual(sorted(shuffled.tolist()), list(range(16)))
        np.testing.assert_array_equal(shuffled, temporal_positions("shuffled", 16, "k"))
        self.assertFalse(np.array_equal(shuffled, temporal_positions("shuffled", 16, "other")))

    def test_degradations_preserve_shape(self):
        frames = np.random.default_rng(0).integers(0, 256, (3, 30, 40, 3), dtype=np.uint8)
        for spec in ({"kind": "brightness", "factor": 0.5}, {"kind": "downsample", "factor": 0.25},
                     {"kind": "blur", "sigma": 1.5}):
            out = degrade(frames, spec)
            self.assertEqual(out.shape, frames.shape)
            self.assertEqual(out.dtype, np.uint8)
        self.assertLessEqual(degrade(frames, {"kind": "brightness", "factor": 0.5}).max(), 128)


class TestFixturePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.entry = fixtures.build(root / "fixtures")
        paths = {"cache_dir": str(root / "cache"), "runs_dir": str(root / "runs"), "_source": "test"}
        cls.cfg = fixtures.experiment(cls.entry, {"name": "fake", "backbone": "fake_for_tests"}, paths)
        cls.backbone = FakeBackbone()
        cls.run_dir = inference.run(cls.cfg, backbone=cls.backbone)
        cls.summary = evaluate_run(cls.run_dir, n_boot=50)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_failures_stay_in_denominator(self):
        pred = pd.read_csv(self.run_dir / "fixture_colors" / "predictions.csv", keep_default_na=False)
        self.assertEqual(len(pred), 8)
        broken = pred.set_index("video_id").loc["broken"]
        self.assertEqual((broken["status"], broken["correct"]), ("failed", 0))
        self.assertEqual(self.summary["per_dataset"]["fixture_colors"]["overall"]["n"], 8)
        scores = np.load(self.run_dir / "fixture_colors" / "scores.npy")
        self.assertEqual(scores.shape, (8, 3))
        self.assertTrue(np.isnan(scores[pred["video_id"] == "broken"]).all())

    def test_run_is_marked_synthetic_and_excluded_from_results(self):
        self.assertTrue(json.loads((self.run_dir / "run.json").read_text())["synthetic"])
        out = results_report(self.run_dir.parent, Path(self.tmp.name) / "results.md")
        self.assertIn("No experiment on real benchmark videos", out.read_text())

    def test_second_run_uses_cache_with_identical_scores(self):
        again = inference.run(self.cfg, backbone=self.backbone, run_dir=Path(self.tmp.name) / "again")
        pred = pd.read_csv(again / "fixture_colors" / "predictions.csv", keep_default_na=False)
        self.assertEqual(int(pred["cached"].sum()), 7)
        np.testing.assert_array_equal(np.load(again / "fixture_colors" / "scores.npy"),
                                      np.load(self.run_dir / "fixture_colors" / "scores.npy"))

    def test_derived_temporal_variant_equals_reencoding(self):
        derived = inference.run(self.cfg, backbone=self.backbone, temporal="single_frame",
                                run_dir=Path(self.tmp.name) / "derived")
        self.backbone.frame_independent = False  # force decoding with reordered indices
        try:
            recoded = inference.run(self.cfg, backbone=self.backbone, temporal="single_frame",
                                    run_dir=Path(self.tmp.name) / "recoded")
        finally:
            self.backbone.frame_independent = True
        np.testing.assert_allclose(np.load(derived / "fixture_colors" / "scores.npy"),
                                   np.load(recoded / "fixture_colors" / "scores.npy"), atol=1e-5)

    def test_diagnostics_and_error_analysis(self):
        diag = Path(self.tmp.name) / "diag.yaml"
        diag.write_text("name: t\nexperiment: unused\ntemporal: [shuffled, single_frame]\n"
                        "degradations:\n  - {name: dark, kind: brightness, factor: 0.25}\n"
                        "spatial:\n  - {name: actor_crop, context: 1.0}\n")
        out = run_suite(diag, experiment_cfg=self.cfg, n_boot=20, backbone=self.backbone)
        summary = pd.read_csv(out / "summary.csv", keep_default_na=False).set_index("variant")
        self.assertEqual(float(summary.loc["temporal_shuffled", "prediction_change_rate"]), 0.0)
        self.assertIn("invariant by construction", summary.loc["temporal_shuffled", "note"])
        self.assertTrue(summary.loc["actor_crop", "status"].startswith("not run"))
        report = error_analysis(self.run_dir, [out], Path(self.tmp.name) / "ea.md").read_text()
        self.assertIn("SYNTHETIC FIXTURE RUN", report)
        self.assertIn("## Hypotheses", report)


if __name__ == "__main__":
    unittest.main()
