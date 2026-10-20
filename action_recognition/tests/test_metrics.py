import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ovar.backbones.base import template_average, video_embedding, video_scores
from ovar.metrics import cluster_bootstrap, dataset_metrics, evaluate_run

CLASSES = pd.DataFrame({"action_id": [0, 1, 2], "official_name": ["a", "b", "c"],
                        "category_status": ["familiar", "familiar", "unfamiliar"]})


def predictions(rows):
    df = pd.DataFrame(rows, columns=["action_id", "pred_id", "status", "category_status", "source_group_id"])
    df["correct"] = ((df["status"] == "ok") & (df["action_id"] == df["pred_id"])).astype(int)
    df["top5_ids"] = df["pred_id"].astype(str)
    return df


class TestAccuracy(unittest.TestCase):
    def test_hand_computed_with_failure_in_denominator(self):
        pred = predictions([
            (0, 0, "ok", "familiar", "g1"), (0, 2, "ok", "familiar", "g1"),
            (1, 1, "ok", "familiar", "g2"), (1, -1, "failed", "familiar", "g3"),
            (2, 2, "ok", "unfamiliar", "g4"), (2, 0, "ok", "unfamiliar", "g5"),
        ])
        metrics, per_class, confusion, _ = dataset_metrics(pred, CLASSES, n_boot=50)
        acc = metrics["accuracy"]
        self.assertEqual((acc["overall"]["n"], acc["overall"]["n_correct"]), (6, 3))
        self.assertAlmostEqual(acc["familiar"]["acc"], 2 / 4)
        self.assertAlmostEqual(acc["unfamiliar"]["acc"], 1 / 2)
        self.assertEqual(metrics["n_failed"], 1)
        self.assertEqual(confusion.loc["b", "FAILED"], 1)
        self.assertEqual(confusion.loc["a", "c"], 1)
        self.assertEqual(per_class.set_index("action_name").loc["a", "most_confused_with"], "c")
        self.assertAlmostEqual(metrics["mean_class_accuracy"], (0.5 + 0.5 + 0.5) / 3)

    def test_secondary_unfamiliar_metric_excludes_reviewed_classes(self):
        from ovar.metrics import near_familiar_classes

        self.assertIn("JumpRope", near_familiar_classes("ucf101"))
        classes = pd.DataFrame({"action_id": [0, 1], "official_name": ["JumpRope", "YoYo"],
                                "category_status": ["unfamiliar", "unfamiliar"]})
        pred = predictions([(0, 0, "ok", "unfamiliar", "g1"), (1, 0, "ok", "unfamiliar", "g2")])
        pred["dataset_name"], pred["action_name"] = "ucf101", ["JumpRope", "YoYo"]
        metrics = dataset_metrics(pred, classes, n_boot=10)[0]
        self.assertEqual(metrics["accuracy"]["unfamiliar"]["acc"], 0.5)
        secondary = metrics["secondary_unfamiliar_excluding_near_familiar"]
        self.assertEqual((secondary["n"], secondary["acc"]), (1, 0.0))

    def test_familiar_video_predicted_as_unfamiliar_class_is_wrong(self):
        # Full-vocabulary prediction: no restriction of candidates per subgroup.
        pred = predictions([(0, 2, "ok", "familiar", "g1"), (2, 2, "ok", "unfamiliar", "g2")])
        acc = dataset_metrics(pred, CLASSES, n_boot=10)[0]["accuracy"]
        self.assertEqual(acc["familiar"]["acc"], 0.0)

    def test_scores_rank_like_openvclip_logits(self):
        rng = np.random.default_rng(1)
        frames, text = rng.standard_normal((3, 8, 16)), rng.standard_normal((5, 16))
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        ours = video_scores(frames, text)
        f = frames / np.linalg.norm(frames, axis=-1, keepdims=True)
        official = (100 * (f @ text.T).mean(1)).sum(0)  # per-view mean of frame logits, summed over views
        self.assertEqual(np.argsort(ours).tolist(), np.argsort(official).tolist())
        scaled = frames * rng.uniform(0.1, 10, (3, 8, 1))
        np.testing.assert_allclose(video_scores(scaled, text), ours, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(video_embedding(frames))), 1.0, places=6)

    def test_template_average_is_unit_norm(self):
        import torch

        out = template_average([torch.randn(4, 8), torch.randn(4, 8)])
        np.testing.assert_allclose(out.norm(dim=-1).numpy(), 1.0, atol=1e-6)


class TestBootstrap(unittest.TestCase):
    def test_deterministic_and_groupwise(self):
        correct = np.array([1, 1, 0, 0, 1, 0], float)
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        masks = {"all": np.ones(6, bool)}
        s1 = cluster_bootstrap(correct, groups, masks, 200, seed=3)["all"]
        s2 = cluster_bootstrap(correct, groups, masks, 200, seed=3)["all"]
        np.testing.assert_array_equal(s1, s2)
        # Group means are 1, 0, 0.5: every resample is a mean of those values.
        allowed = {round(x, 6) for x in (0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1)}
        self.assertTrue({round(x, 6) for x in s1} <= allowed)

    def test_single_group_gives_degenerate_interval(self):
        s = cluster_bootstrap(np.array([1, 0, 1.]), np.array(["g"] * 3), {"all": np.ones(3, bool)}, 50, 0)
        self.assertTrue(np.allclose(s["all"], 2 / 3))


class TestCrossDomain(unittest.TestCase):
    def test_unweighted_average_differs_from_pooled(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            datasets = {"big": [(0, 0, "ok", "familiar", f"g{i}") for i in range(9)] +
                               [(0, 1, "ok", "familiar", "g9")],
                        "small": [(2, 2, "ok", "unfamiliar", "h0"), (2, 0, "ok", "unfamiliar", "h1")]}
            for name, rows in datasets.items():
                (run / name).mkdir()
                df = predictions(rows)
                df["source_group_id"] = name + ":" + df["source_group_id"]
                df.to_csv(run / name / "predictions.csv", index=False)
                CLASSES.to_csv(run / name / "classes.csv", index=False)
            (run / "run.json").write_text(json.dumps({"run_name": "t", "synthetic": True,
                                                      "datasets": {k: {} for k in datasets}}))
            summary = evaluate_run(run, n_boot=100)
            self.assertAlmostEqual(summary["domain_average_unweighted"]["overall"]["acc"], (0.9 + 0.5) / 2)
            self.assertAlmostEqual(summary["pooled_over_videos"]["overall"]["acc"], 10 / 12)


if __name__ == "__main__":
    unittest.main()
