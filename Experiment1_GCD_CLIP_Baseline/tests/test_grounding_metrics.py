"""Matrices, grounding scores and metrics (synthetic arrays)."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from e1 import grounding as G
from e1 import metrics as M
from e1.matrices import aggregate, select_members, similarity


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class TestMatrices(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.v, self.t = unit(rng.randn(20, 16)), unit(rng.randn(5, 16))
        self.R = similarity(self.v, self.t)
        self.members = [np.array([0, 3, 7]), np.array([5]), np.array([], dtype=int)]

    def test_cosine_range_and_normalisation_check(self):
        self.assertTrue((self.R <= 1).all() and (self.R >= -1).all())
        with self.assertRaises(ValueError):
            similarity(self.v * 2, self.t)

    def test_A_and_B_match_direct_computation(self):
        agg = aggregate(self.R, self.v, self.members)
        rows = self.members[0]
        np.testing.assert_allclose(agg["A"][0], (self.v[rows] @ self.t.T).mean(0), rtol=1e-5, atol=1e-6)
        m = self.v[rows].mean(0)
        np.testing.assert_allclose(agg["B"][0], (m / np.linalg.norm(m)) @ self.t.T, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(agg["A"][0], agg["norm"][0] * agg["B"][0], rtol=1e-5, atol=1e-6)
        self.assertEqual(np.argmax(agg["A"][0]), np.argmax(agg["B"][0]))
        np.testing.assert_allclose(agg["B"][1], self.R[5], rtol=1e-5, atol=1e-6)     # singleton: |m| = 1
        self.assertTrue(np.isnan(agg["A"][2]).all() and agg["n"][2] == 0)            # empty group

    def test_member_selection_is_label_free_and_excludes_other_populations(self):
        rows = pd.DataFrame({"row": [0, 1, 2, 3, 4], "sample_id": list("abcde"), "cluster_id": [0, 0, 0, 1, 1],
                             "sq_dist_to_centre": [0.3, 0.1, 0.2, 0.5, 0.4]})
        self.assertEqual(select_members(rows, "cluster_id", [0, 1], "centre2")[0].tolist(), [1, 2])
        self.assertEqual(select_members(rows, "cluster_id", [0, 1, 2], "all")[2].tolist(), [])
        a = select_members(rows, "cluster_id", [0], "random2", seed=3)[0]
        self.assertTrue(np.array_equal(a, select_members(rows, "cluster_id", [0], "random2", seed=3)[0]))


class TestScores(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.v, self.t = unit(rng.randn(30, 16)), unit(rng.randn(8, 16))
        self.R = similarity(self.v, self.t)
        self.grp = np.repeat(np.arange(3), 10)
        self.members = [np.flatnonzero(self.grp == k) for k in range(3)]
        self.A = aggregate(self.R, self.v, self.members)["A"]

    def scores(self, cols):
        return G.score_clusters(self.A, self.R, self.grp, self.A, self.R, np.asarray(cols), T_dev=0.05)

    def test_column_permutation_gives_the_same_labels_and_scores(self):
        cols = np.array([0, 2, 3, 5, 7])
        perm = cols[[3, 0, 4, 1, 2]]
        a, b = self.scores(cols), self.scores(perm)
        for m in G.METHODS:
            self.assertTrue(np.array_equal(cols[a[m][0]], perm[b[m][0]]), m)
            np.testing.assert_allclose(a[m][1], b[m][1], rtol=1e-6, err_msg=m)

    def test_definitions(self):
        cols = np.arange(8)
        s = self.scores(cols)
        k = 0
        a = self.A[k]
        self.assertAlmostEqual(s["max_mean_sim"][1][k], a.max(), places=6)
        self.assertAlmostEqual(s["margin"][1][k], np.sort(a)[-1] - np.sort(a)[-2], places=6)
        p = np.exp((a - a.max()) / 0.01); p /= p.sum()
        self.assertAlmostEqual(s["msp_pilot"][1][k], p.max(), places=6)
        votes = np.argmax(self.R[self.grp == k], 1)
        self.assertAlmostEqual(s["vote_fraction"][1][k], np.mean(votes == np.argmax(a)), places=6)
        img_mcm = np.exp(self.R[self.grp == k]); img_mcm = (img_mcm / img_mcm.sum(1, keepdims=True)).max(1)
        self.assertAlmostEqual(s["mcm_image_mean"][1][k], img_mcm.mean(), places=6)


class TestMetrics(unittest.TestCase):
    def test_tied_clusters_are_accepted_together(self):
        rc = M.risk_coverage(np.array([0.9, 0.5, 0.5, 0.1]), np.array([10, 5, 5, 10]), np.array([10, 5, 0, 0]))
        self.assertEqual(rc["coverage"].tolist(), [10 / 30, 20 / 30, 1.0])        # the tie enters as one block
        self.assertAlmostEqual(rc["risk"][1], 5 / 20)

    def test_cluster_auroc_equals_image_auroc_with_ties(self):
        s = np.array([0.9, 0.5, 0.5, 0.1, 0.7])
        pos, neg = np.array([3, 1, 0, 2, 0]), np.array([1, 2, 1, 4, 2])
        img_s = np.concatenate([[x] * (p + q) for x, p, q in zip(s, pos, neg)])
        img_y = np.concatenate([[1] * p + [0] * q for p, q in zip(pos, neg)])
        self.assertAlmostEqual(M.auroc_clusters(s, pos, neg), roc_auc_score(img_y, img_s))
        self.assertAlmostEqual(M.auroc_clusters(s, pos, neg, np.ones((2, 5)))[1], roc_auc_score(img_y, img_s))

    def test_thresholds_and_absent_classes(self):
        s = np.array([0.9, 0.6, 0.2])
        n, na, nc = np.array([10, 10, 10]), np.array([10, 10, 0]), np.array([9, 2, 0])
        t = M.choose_thresholds(s, n, na, nc)
        # accept the first two: (9 + 2 correct) + 10 absent rejected = 21/30, better than 19/30 at 0.9,
        # because rejecting an available image is an error too
        self.assertEqual(t["tau_e2e"], 0.6)
        self.assertAlmostEqual(t["calibration_e2e_at_tau"], 21 / 30)
        self.assertEqual(t["tau_95"], 0.6)         # accepts 20/20 available images
        r = M.at_threshold(s, n, na, nc, 0.9)
        self.assertAlmostEqual(r["end_to_end_accuracy"], 19 / 30)
        none_absent = M.at_threshold(s, n, n, nc, 0.5)
        self.assertTrue(np.isnan(none_absent["false_acceptance_absent"]))


if __name__ == "__main__":
    unittest.main()
