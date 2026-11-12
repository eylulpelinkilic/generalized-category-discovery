"""Software checks of the SCD re-implementation (synthetic inputs only; no research results)."""
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd.assignment import linear_assignment, max_weight_assignment  # noqa: E402
from scd.clustering import constrained_assignment, constrained_ss_kmeans  # noqa: E402
from scd.metrics import cluster_acc_v2, name_iou, semantic_accuracy  # noqa: E402
from scd.naming import (InsufficientCandidates, NamingConfig, candidate_pool, cluster_counters,  # noqa: E402
                        image_topk, initial_cluster_names, name_clusters, refine)
from scd.vocabulary import Vocabulary, normalise  # noqa: E402

NLTK_DIR = Path(__file__).resolve().parents[2] / "Experiment2_SCD" / "outputs" / "cache" / "nltk_data"


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestAssignment(unittest.TestCase):
    def test_munkres_matches_scipy_optimum(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            w = rng.randint(0, 6, size=(7, 7))
            a, b = max_weight_assignment(w, "munkres"), max_weight_assignment(w, "scipy")
            self.assertEqual(w[a[:, 0], a[:, 1]].sum(), w[b[:, 0], b[:, 1]].sum())
            self.assertTrue(np.array_equal(a[:, 0], np.arange(7)))          # rows sorted, like the official helper

    def test_rectangular(self):
        ind = linear_assignment(np.array([[1, 2, 3], [3, 1, 2]]))
        self.assertEqual(ind.tolist(), [[0, 0], [1, 1]])


class TestVotes(unittest.TestCase):
    def test_counter_ties_follow_first_appearance(self):
        topk = np.array([[5, 2], [2, 5], [7, 7]])
        c = cluster_counters(topk, np.array([0, 0, 0]), [0], k=2)[0]
        self.assertEqual(c.most_common(3), [(5, 2), (2, 2), (7, 2)])         # all tied: data order
        self.assertEqual(Counter([5, 2, 2, 5, 7, 7]).most_common(1), [(5, 2)])

    def test_pool_orders(self):
        counters = {0: Counter({9: 3, 1: 2}), 1: Counter({1: 4, 4: 1})}
        self.assertEqual(candidate_pool(counters, [0, 1], 2, "first_appearance"), [9, 1, 4])
        # official: list(set(voted)) with voted built cluster by cluster; CPython set order depends on the
        # insertion sequence (a set literal {9, 1, 4} can iterate differently), so compare the same construction
        self.assertEqual(candidate_pool(counters, [0, 1], 2, "official_set"), list(set([9, 1, 1, 4])))

    def test_known_names_are_excluded(self):
        topk = np.array([[1, 2], [1, 3]])
        c = cluster_counters(topk, np.array([0, 0]), [0], k=2, exclude=[1])[0]
        self.assertNotIn(1, c)


class TestMeanVersusVote(unittest.TestCase):
    """Deterministic case where the cluster-mean prediction and per-image plurality voting disagree."""

    def test_disagreement(self):
        # 3 classes; 5 images. Two images are close to class 2, three images lean weakly to class 0 but sit near 1.
        R = np.array([[0.30, 0.29, 0.00], [0.30, 0.29, 0.00], [0.30, 0.29, 0.00],
                      [0.00, 0.28, 0.40], [0.00, 0.28, 0.40]])
        mean_pred = int(np.argmax(R.mean(0)))
        votes = np.bincount(R.argmax(1), minlength=3)
        self.assertEqual(mean_pred, 1)                  # mean: class 1 (0.286 vs 0.18 / 0.16)
        self.assertEqual(int(np.argmax(votes)), 0)      # plurality: class 0 (3 votes); class 1 gets none


class TestNaming(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(["x", "y", "z", "w"], source="test")

    def test_hungarian_resolves_competition(self):
        # cluster 0 votes x:5 y:4 ; cluster 1 votes x:6 z:1 -> distinct names y / x (10 > 6)
        topk = np.array([[0, 1]] * 4 + [[0, 3]] + [[0, 2]] + [[0, 3]] * 5)
        topk = topk[:, :1].repeat(2, 1)
        rows = [[0]] * 5 + [[1]] * 4 + [[0]] * 6 + [[2]]
        topk = np.array(rows)
        labels = np.array([0] * 9 + [1] * 7)
        cfg = NamingConfig(topk_vote=1, num_common_vote=2, num_common_linear=2)
        step = name_clusters(topk, labels, [0, 1], cfg, num_slots=2)
        got = {sl["cluster"]: sl["v"] for sl in step["slots"]}
        self.assertEqual(got, {0: 1, 1: 0})
        plur = name_clusters(topk, labels, [0, 1], NamingConfig(topk_vote=1, num_common_vote=2,
                                                                num_common_linear=2, assignment="plurality"), 2)
        self.assertEqual({sl["cluster"]: sl["v"] for sl in plur["slots"]}, {0: 0, 1: 0})

    def test_insufficient_pool(self):
        topk = np.array([[0], [0], [0]])
        labels = np.array([0, 1, 2])
        cfg = NamingConfig(topk_vote=1, num_common_vote=1, num_common_linear=1, insufficient_pool="error")
        with self.assertRaises(InsufficientCandidates):
            name_clusters(topk, labels, [0, 1, 2], cfg, num_slots=3)
        step = name_clusters(topk, labels, [0, 1, 2], NamingConfig(topk_vote=1, num_common_vote=1,
                                                                   num_common_linear=1), num_slots=3)
        self.assertEqual(step["repairs"][0]["repair"], "slots_left_unnamed")
        self.assertEqual(sum(sl["v"] is not None for sl in step["slots"]), 1)

    def test_pool_deepening(self):
        topk = np.array([[0, 1], [0, 1], [0, 2]])
        labels = np.array([0, 1, 2])
        step = name_clusters(topk, labels, [0, 1, 2], NamingConfig(topk_vote=2, num_common_vote=1,
                                                                   num_common_linear=1), num_slots=3)
        self.assertEqual(step["repairs"][0]["repair"], "pool_deepened")

    def test_refine_converges_and_traces(self):
        rng = np.random.RandomState(0)
        T = unit(np.eye(4) + 0.05 * rng.randn(4, 4))
        X = unit(np.repeat(T[:3], 20, 0) + 0.1 * rng.randn(60, 4))
        topk, _, _ = image_topk(X, T, 2)
        init = np.repeat([0, 1, 2], 20)
        init[[0, 21, 42]] = [1, 2, 0]                                 # some initial clustering errors
        res = refine(X, T, self.vocab, topk, init, NamingConfig(topk_vote=2), n_cluster=3)
        self.assertEqual(res["stop_reason"], "converged")
        self.assertEqual(sorted(res["names"]), ["x", "y", "z"])
        self.assertTrue(all("labels_after" in it for it in res["iterations"]))
        self.assertEqual(sorted(initial_cluster_names(res).values()), ["x", "y", "z"])

    def test_iteration_limit(self):
        rng = np.random.RandomState(1)
        T = unit(rng.randn(4, 4))
        X = unit(rng.randn(30, 4))
        topk, _, _ = image_topk(X, T, 2)
        res = refine(X, T, self.vocab, topk, np.arange(30) % 3, NamingConfig(topk_vote=2, max_iter=1), n_cluster=3)
        self.assertIn(res["stop_reason"], {"max_iter", "converged"})
        self.assertEqual(res["n_iterations"], 1)

    def test_known_filter_official_defect(self):
        """From iteration 2 the official filter holds name positions, so a known name can be voted again."""
        vocab = Vocabulary(["k", "a", "b", "c"], source="test")        # 'k' is the known name
        T = unit(np.eye(4))
        X = unit(np.array([[1, .2, 0, 0]] * 6 + [[0, 1, 0, 0]] * 6 + [[.9, 0, 1, 0]] * 6, dtype=float))
        topk, _, _ = image_topk(X, T, 2)
        init = np.repeat([0, 1, 2], 6)                                # cluster 0 holds the labelled anchors
        out = {}
        for mode in ("official", "fixed"):
            res = refine(X, T, vocab, topk, init, NamingConfig(topk_vote=2, num_common_vote=2, num_common_linear=2,
                                                              known_filter=mode),
                         n_cluster=3, cluster_ids=[0, 1, 2], known_names=["k"], labelled_clusters=[0])
            voted = [n for it in res["iterations"] for n in it["voted_names"]]
            out[mode] = (res, voted)
        self.assertNotIn("k", out["fixed"][1])
        self.assertEqual(out["official"][0]["iterations"][0]["exclude_size"], 1)
        if out["official"][0]["n_iterations"] > 1:
            # iteration 2 excludes positions in the selected list, not the vocabulary index of 'k'
            self.assertEqual(out["official"][0]["iterations"][1]["exclude_size"], 1)

    def test_known_name_must_be_in_vocabulary(self):
        with self.assertRaises(ValueError):
            refine(np.zeros((1, 2)), np.zeros((4, 2)), self.vocab, np.zeros((1, 1), int), np.zeros(1, int),
                   NamingConfig(topk_vote=1, num_common_vote=1, num_common_linear=1), n_cluster=2,
                   known_names=["withheld"], labelled_clusters=[0])


class TestTopK(unittest.TestCase):
    def test_ties_to_lowest_index_and_columns(self):
        T = unit(np.array([[1, 0], [1, 0], [0, 1], [1, 0]]))
        X = unit(np.array([[1, 0]]))
        idx, val, info = image_topk(X, T, 2)
        self.assertEqual(idx.tolist(), [[0, 1]])
        idx, _, _ = image_topk(X, T, 2, columns=[3, 1, 2])
        self.assertEqual(idx.tolist(), [[1, 3]])
        self.assertEqual(info["n_rows_with_tie_at_k"], 1)

    def test_fp16_emulation_creates_ties(self):
        T = unit(np.array([[1.0, 0.0], [1.0, 1e-4], [0.0, 1.0]]))
        X = unit(np.array([[1.0, 0.0]]))
        self.assertEqual(image_topk(X, T, 1, logit_dtype="float16")[0].tolist(), [[0]])


class TestClustering(unittest.TestCase):
    def test_constrained_sizes_and_anchors(self):
        rng = np.random.RandomState(0)
        u = unit(np.concatenate([rng.randn(40, 3) + [4, 0, 0], rng.randn(10, 3) + [0, 4, 0], rng.randn(10, 3) + [0, 0, 4]]))
        l = unit(np.array([[4, 0, 0], [4.2, 0, 0], [0, 4, 0]], dtype=float))
        out = constrained_ss_kmeans(u, l, np.array([7, 7, 3]), k=3, size_min=15, size_max=30, n_init=2, max_iter=5)
        sizes = np.bincount(out["labels_unlabelled"], minlength=3)
        self.assertTrue((sizes >= 15).all() and (sizes <= 30).all())
        self.assertEqual(out["labels_labelled"].tolist(), [1, 1, 0])   # rank among sorted labelled classes
        again = constrained_ss_kmeans(u, l, np.array([7, 7, 3]), k=3, size_min=15, size_max=30, n_init=2, max_iter=5)
        self.assertTrue(np.array_equal(out["labels_unlabelled"], again["labels_unlabelled"]))

    def test_assignment_infeasible(self):
        with self.assertRaises(ValueError):
            constrained_assignment(np.zeros((5, 2)), np.zeros((3, 2)), size_min=2, size_max=5)


class TestMetrics(unittest.TestCase):
    def test_sacc_and_iou(self):
        s = semantic_accuracy(["a", "b", "b", "c"], ["a", "a", "b", "b"])
        self.assertAlmostEqual(s["image"], 0.5)
        self.assertAlmostEqual(s["class"], 0.5)
        self.assertAlmostEqual(name_iou(["a", "b"], ["b", "c"]), 1 / 3)

    def test_cluster_acc_permutation(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([2, 2, 0, 0, 1, 0])
        r = cluster_acc_v2(y, p, np.array([True, True, True, True, False, False]))
        self.assertAlmostEqual(r["all"], 5 / 6)
        self.assertAlmostEqual(r["old"], 1.0)
        self.assertAlmostEqual(r["new"], 0.5)

    @unittest.skipUnless((NLTK_DIR / "corpora").exists(), "WordNet not downloaded")
    def test_soft_sacc_scale(self):
        from scd.metrics import SoftScorer
        from scd.vocabulary import LCH_MAX, load_wordnet, wordnet_dicts

        wn = load_wordnet(NLTK_DIR)
        w2s, _, n2w = wordnet_dicts(wn)
        sc = SoftScorer(w2s, n2w)
        r = sc.soft_sacc(["dog", "cat"], ["dog", "dog"])
        self.assertAlmostEqual(sc.score("dog", "dog"), LCH_MAX)
        self.assertAlmostEqual(r["official"], (1 + sc.score("cat", "dog") / LCH_MAX) / 2)


class TestVocabulary(unittest.TestCase):
    def test_duplicates_after_normalisation(self):
        v = Vocabulary(["Cardigan", "x-ray", "cardigan", "x_ray", "dog"], source="test")
        self.assertEqual(v.names, ["cardigan", "x_ray", "cardigan", "x_ray", "dog"])
        self.assertEqual(v.first_index["cardigan"], 0)
        self.assertEqual(v.duplicate_columns, [2, 3])
        self.assertEqual(normalise("A-b"), "a_b")

    def test_templates_match_official(self):
        from scd.text import imagenet_templates

        self.assertEqual(len(imagenet_templates()), 80)


if __name__ == "__main__":
    unittest.main()
