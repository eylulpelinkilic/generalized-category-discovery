"""Software checks for Experiment 2 (synthetic inputs; never research results)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e2  # noqa: E402,F401  (sets up paths to Experiment 1 and SCD)
from e1 import grounding as G  # noqa: E402
from e2 import voting as V  # noqa: E402
from e2.config import load_config  # noqa: E402


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestConfig(unittest.TestCase):
    def test_base_inheritance_is_local(self):
        cfg = load_config("cub")
        self.assertEqual(cfg["paths"]["outputs"], "outputs")
        self.assertEqual(cfg["naming"]["topk_vote"], 3)
        self.assertEqual(cfg["fixed"]["clip"], "vitb32")


class TestMeanVersusVote(unittest.TestCase):
    """Small deterministic case: the cluster mean and per-image plurality voting pick different labels."""

    def setUp(self):
        # one cluster, 5 members, 3 classes
        self.R = np.array([[0.30, 0.29, 0.00], [0.30, 0.29, 0.00], [0.30, 0.29, 0.00],
                           [0.00, 0.28, 0.40], [0.00, 0.28, 0.40]], dtype=np.float32)
        self.grp = np.zeros(5, dtype=int)
        self.A = self.R.mean(0, keepdims=True)

    def test_disagreement_and_own_confidence(self):
        cols = np.array([0, 1, 2])
        sc = G.score_clusters(self.A, self.R, self.grp, self.A, self.R, cols, 1.0)
        mean_label = int(sc["forced_top1"][0][0])
        e1_vote_fraction = float(sc["vote_fraction"][1][0])
        vr = V.score_votes(self.A, self.R, self.grp, cols, k=1)
        self.assertEqual(mean_label, 1)                  # mean similarity: class 1
        self.assertEqual(int(vr["pred"][0]), 0)          # plurality: class 0 (3 of 5 votes)
        self.assertEqual(e1_vote_fraction, 0.0)          # Experiment 1's score: support for the *mean* label
        self.assertAlmostEqual(vr["scores"]["win_fraction"][0], 0.6)
        self.assertAlmostEqual(vr["scores"]["margin"][0], (3 - 2) / 5)
        p = np.array([3, 0, 2]) / 5
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        self.assertAlmostEqual(vr["scores"]["neg_entropy"][0], 1 - h / np.log(3))
        self.assertAlmostEqual(vr["scores"]["mean_sim"][0], self.A[0, 0])

    def test_votes_recomputed_within_episode_vocabulary(self):
        vr = V.score_votes(self.A, self.R, self.grp, np.array([1, 2]), k=1)   # class 0 withheld
        self.assertEqual(int(np.array([1, 2])[vr["pred"][0]]), 1)            # its 3 voters now vote for class 1
        self.assertAlmostEqual(vr["scores"]["win_fraction"][0], 0.6)


class TestVoteEdgeCases(unittest.TestCase):
    def test_tie_goes_to_higher_mean_then_lowest_column(self):
        votes = np.array([[2.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
        A = np.array([[0.1, 0.2, 0.0], [0.3, 0.3, 0.0]])
        r = V.vote_predict(votes, A, np.array([4.0, 2.0]))
        self.assertEqual(r["pred"].tolist(), [1, 0])

    def test_empty_single_member_and_single_concept(self):
        R = np.array([[0.2, 0.1]], dtype=np.float32)
        A = np.array([[0.2, 0.1], [np.nan, np.nan]], dtype=np.float32)
        vr = V.score_votes(A, R, np.array([0]), np.array([0, 1]), k=1)
        self.assertEqual(vr["pred"].tolist(), [0, -1])
        self.assertTrue(np.isnan(vr["scores"]["win_fraction"][1]))
        self.assertEqual([vr["scores"][s][0] for s in ("win_fraction", "margin", "neg_entropy")], [1.0, 1.0, 1.0])
        one = V.score_votes(A, R, np.array([0]), np.array([1]), k=1)
        self.assertEqual(one["scores"]["margin"][0], 1.0)
        self.assertEqual(one["scores"]["neg_entropy"][0], 1.0)

    def test_topk_votes_count_each_image_k_times(self):
        R = np.array([[0.3, 0.2, 0.1], [0.1, 0.3, 0.2]], dtype=np.float32)
        votes = V.vote_matrix(R, np.array([0, 0]), 1, k=2)
        self.assertEqual(votes.tolist(), [[1.0, 2.0, 1.0]])

    def test_image_ties_to_lowest_column(self):
        self.assertEqual(V.image_votes(np.array([[0.2, 0.2, 0.1]]), 1).tolist(), [[0]])
        self.assertEqual(V.image_votes(np.array([[0.1, 0.2, 0.2]]), 2).tolist(), [[1, 2]])


class TestDistinct(unittest.TestCase):
    def test_infeasible_with_fewer_names_than_clusters(self):
        R = unit(np.eye(3))
        with self.assertRaises(ValueError):
            V.distinct_names(R, np.array([0, 1, 2]), 3, np.array([0, 1]), 1, 2, 1)

    def test_competition_resolved(self):
        # clusters 0 and 1 both prefer class 0; cluster 1 has more support, cluster 0's second choice is class 1
        R = np.array([[0.5, 0.4, 0.0]] * 3 + [[0.4, 0.5, 0.0]] * 2 + [[0.5, 0.0, 0.1]] * 6, dtype=np.float32)
        grp = np.array([0] * 5 + [1] * 6)
        pred, info = V.distinct_names(R, grp, 2, np.array([0, 1, 2]), 1, 3, 2)
        self.assertEqual(pred.tolist(), [1, 0])
        plural = V.score_votes(np.zeros((2, 3)), R, grp, np.array([0, 1, 2]), 1)["pred"]
        self.assertEqual(plural.tolist(), [0, 0])            # independent decisions can share a name


class TestFixedPopulation(unittest.TestCase):
    def test_predict_population_columns_map_to_class_ids(self):
        from e2.fixed import predict_population

        rng = np.random.RandomState(0)
        C = unit(np.eye(4) + 0.01 * rng.randn(4, 4))
        emb = unit(np.concatenate([C[[1]] + 0.05 * rng.randn(5, 4), C[[3]] + 0.05 * rng.randn(5, 4)]))
        rows = pd.DataFrame({"row": np.arange(10), "sample_id": [f"s{i}" for i in range(10)],
                             "population": "discovery", "cluster_id": [0] * 5 + [1] * 5, "sq_dist_to_centre": 0.0})
        R = {s: emb @ C.T for s in ("single", "ensemble80", "mcm")}
        inp = {"rows": rows, "emb": emb, "K": 2}
        eps = pd.DataFrame([{"episode_id": "full", "kind": "shared", "size": 4, "class_ids": "0 1 2 3"},
                            {"episode_id": "drop3", "kind": "shared", "size": 3, "class_ids": "0 1 2"}])
        df = predict_population(inp, R, "discovery", "ensemble80", 1.0, eps, [1, 3],
                                {"k": 3, "num_common_vote": 10, "num_common_linear": 2})
        get = lambda ep, m: df[(df.episode_id == ep) & (df.method == m)].sort_values("cluster_id")["pred_class_id"].tolist()  # noqa: E731
        self.assertEqual(get("full", "plurality_forced"), [1, 3])
        self.assertEqual(get("full", "forced_top1"), [1, 3])
        self.assertEqual(get("full", "distinct_top3_forced"), [1, 3])
        self.assertNotIn(3, get("drop3", "plurality_forced"))           # withheld class cannot be predicted
        self.assertEqual(len(df[(df.episode_id == "drop3") & df.method.str.startswith("distinct")]), 0)


class TestSmoke(unittest.TestCase):
    NLTK = Path(__file__).resolve().parents[1] / "outputs" / "cache" / "nltk_data" / "corpora"

    @unittest.skipUnless(NLTK.exists(), "WordNet not downloaded")
    def test_protocol_a_smoke(self):
        import tempfile

        from e2 import smoke

        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config("imagenet100", [f"paths.outputs={tmp}"])
            cfg["vocabulary"]["nltk_dir"] = str(Path(__file__).resolve().parents[1] / "outputs" / "cache" / "nltk_data")
            out = smoke.run(cfg)
        self.assertIn(out["unsupervised/official"]["stop_reason"], {"converged", "cycle", "max_iter"})
        self.assertEqual(out["unsupervised/voting_once"]["stop_reason"], "single_pass")


if __name__ == "__main__":
    unittest.main()
