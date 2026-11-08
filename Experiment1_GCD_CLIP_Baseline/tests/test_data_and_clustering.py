"""Partition rules, manifests, the repaired SS-k-means and the v2 metric."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e1 import upstream
from e1.config import REPO_ROOT
from e1.manifests import (Manifest, build_smoke, cub_class_split, imagenet100_source_indices, split_heldout,
                          upstream_labelled_subsample)
from e1.provenance import write_immutable
from e1.sskmeans import assign, fit_mix


class TestPartitions(unittest.TestCase):
    def test_imagenet100_selection_matches_pilot_class_list(self):
        pilot = json.loads((REPO_ROOT / "imagenet100_classes.json").read_text())
        self.assertEqual(imagenet100_source_indices().tolist(), [c["imagenet_id"] for c in pilot])
        self.assertEqual([c["gcd_id"] for c in pilot], list(range(100)))

    def test_labelled_subsample_is_upstream_subsample_instances(self):
        from data.data_utils import subsample_instances

        for n in (7, 120, 1301):
            self.assertEqual(upstream_labelled_subsample(n).tolist(),
                             subsample_instances(list(range(n)), prop_indices_to_subsample=0.5).tolist())

    def test_cub_ssb_split(self):
        known, novel = cub_class_split()
        self.assertEqual((len(known), len(novel)), (100, 100))
        self.assertEqual(sorted(known + novel), list(range(200)))

    def test_heldout_split_halves_each_class(self):
        pools = split_heldout({0: ["a", "b", "c"], 1: ["d", "e", "f", "g"]}, seed=0)
        dev = [s for s, p in pools.items() if p == "heldout_dev"]
        self.assertEqual(len(dev), 1 + 2)
        self.assertEqual(split_heldout({0: ["a", "b", "c"], 1: ["d", "e", "f", "g"]}, seed=0), pools)


class TestManifest(unittest.TestCase):
    def test_smoke_manifest_permissions_and_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = build_smoke(Path(tmp) / "data", Path(tmp) / "images", n_classes=4, train_per_class=6,
                               heldout_per_class=4)
            m = Manifest(Path(tmp) / "data")
            self.assertNotIn("class_id", m.table.columns)          # no labels in the manifest
            lab, held, ev = m.labels("labelled"), m.labels("heldout"), m.labels("eval")
            self.assertEqual(set(lab.index), set(m.ids("labelled")))
            self.assertEqual(set(ev.index), set(m.ids("unlabelled")))
            self.assertTrue(set(lab.index).isdisjoint(ev.index))
            self.assertTrue(set(held.index).isdisjoint(set(lab.index) | set(ev.index)))
            self.assertTrue(set(lab.to_numpy()) <= set(m.known_class_ids))
            self.assertEqual(meta["checks"]["files"]["missing"], 0)
            build_smoke(Path(tmp) / "data", Path(tmp) / "images", n_classes=4, train_per_class=6, heldout_per_class=4)
            with self.assertRaises(RuntimeError):
                write_immutable(Path(tmp) / "data" / "manifest.csv", "changed\n")


class TestBuildersAgainstUpstream(unittest.TestCase):
    """Miniature copies of the real layouts; the upstream dataset code must give the same partition."""

    def test_imagenet100_kaggle_layout(self):
        from e1.manifests import build_imagenet100, synthetic_image
        from e1.upstream_check import check_imagenet

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "imagenet-object-localization-challenge"
            cls = root / "ILSVRC" / "Data" / "CLS-LOC"
            wnids = [f"n{10000000 + i:08d}" for i in range(1000)]
            sel = set(imagenet100_source_indices().tolist())
            png = synthetic_image(np.random.RandomState(0), (1, 2, 3), 8)
            for i, w in enumerate(wnids):
                (cls / "train" / w).mkdir(parents=True)
                for j in range((3 + i % 4) if i in sel else 1):
                    (cls / "train" / w / f"{w}_{j:02d}.JPEG").write_bytes(png)
            (cls / "val").mkdir()
            sol = []
            for k, i in enumerate(sorted(sel)):
                for j in range(3):
                    name = f"ILSVRC2012_val_{3 * k + j:08d}"
                    (cls / "val" / f"{name}.JPEG").write_bytes(png)
                    sol.append((name, f"{wnids[i]} 1 2 3 4"))
            pd.DataFrame(sol, columns=["ImageId", "PredictionString"]).to_csv(root / "LOC_val_solution.csv", index=False)
            (root / "LOC_synset_mapping.txt").write_text("\n".join(f"{w} lemma{i}, alias{i}" for i, w in enumerate(wnids)))
            meta = build_imagenet100(Path(d), Path(d) / "out", workers=1)
            self.assertEqual(meta["counts"]["heldout_dev"] + meta["counts"]["heldout_calib"], 150)
            check = check_imagenet(Path(d), Manifest(Path(d) / "out"))
            self.assertTrue(check["passed"], check)

    def test_cub_layout(self):
        from e1.manifests import build_cub, synthetic_image
        from e1.upstream_check import check_cub

        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "CUB_200_2011"
            png = synthetic_image(np.random.RandomState(0), (1, 2, 3), 8)
            images, labels, split, classes = [], [], [], []
            img_id = 1
            for c in range(200):
                folder = f"{c + 1:03d}.Bird_{c}"
                classes.append(f"{c + 1} {folder}")
                (base / "images" / folder).mkdir(parents=True)
                for j in range(4 + c % 3):
                    rel = f"{folder}/img_{j}.jpg"
                    (base / "images" / rel).write_bytes(png)
                    images.append(f"{img_id} {rel}"), labels.append(f"{img_id} {c + 1}")
                    split.append(f"{img_id} {int(j < 3)}")
                    img_id += 1
            for name, lines in (("images.txt", images), ("image_class_labels.txt", labels),
                                ("train_test_split.txt", split), ("classes.txt", classes)):
                (base / name).write_text("\n".join(lines) + "\n")
            meta = build_cub(Path(d), Path(d) / "out", workers=1)
            self.assertEqual(meta["counts"]["labelled"] + meta["counts"]["unlabelled"], 600)
            check = check_cub(Path(d), Manifest(Path(d) / "out"))
            self.assertTrue(check["passed"], check)


def upstream_kmeans():
    from joblib import Parallel, delayed, effective_n_jobs
    from sklearn.utils import check_random_state

    path = REPO_ROOT / "methods" / "clustering" / "faster_mix_k_means_pytorch.py"
    ns = upstream.load_definitions(path, ["pairwise_distance", "K_Means"],
                                   {"check_random_state": check_random_state, "Parallel": Parallel, "delayed": delayed,
                                    "effective_n_jobs": effective_n_jobs, "copy": __import__("copy"),
                                    "random": __import__("random")})
    return ns["K_Means"]


def blobs(seed=0, k=6, per=30, dim=8):
    rng = np.random.RandomState(seed)
    centres = rng.randn(k, dim) * 6
    x = np.concatenate([c + rng.randn(per, dim) * 0.3 for c in centres]).astype(np.float32)
    y = np.repeat(np.arange(k), per)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, y


class TestSSKMeans(unittest.TestCase):
    def test_matches_upstream_on_well_separated_data(self):
        x, y = blobs()
        lab = (y < 3) & (np.arange(len(y)) % 2 == 0)
        ours = fit_mix(x[~lab], x[lab], y[lab], k=6, n_init=3, max_iter=100, seed=5)
        km = upstream_kmeans()(k=6, tolerance=1e-4, max_iterations=100, init="k-means++", n_init=3,
                               random_state=5, n_jobs=None, pairwise_batch_size=None)
        km.fit_mix(torch.from_numpy(x[~lab]), torch.from_numpy(x[lab]), torch.from_numpy(y[lab]))
        up = km.labels_.numpy()
        self.assertTrue(np.array_equal(up[: lab.sum()], ours.labels_labelled))
        self.assertTrue(np.array_equal(up[lab.sum():], ours.labels_unlabelled))

    def test_anchors_fixed_and_state_consistent(self):
        x, y = blobs(1)
        lab = y < 3
        r = fit_mix(x[~lab], x[lab], y[lab] + 10, k=6, n_init=2, seed=0)
        self.assertEqual(r.labelled_classes.tolist(), [10, 11, 12])
        self.assertTrue(np.array_equal(r.labels_labelled, y[lab]))
        lab_u, _ = assign(torch.from_numpy(x[~lab]), torch.from_numpy(r.centres))
        self.assertTrue(np.array_equal(lab_u.numpy(), r.labels_unlabelled))
        self.assertEqual(r.cluster_sizes.sum(), len(x))

    def test_empty_and_singleton_clusters(self):
        # 3 distinct unlabelled points but K=5: some clusters must be empty; upstream would make NaN centres
        u = np.repeat(np.eye(3, 4, dtype=np.float32), 4, axis=0)
        l = np.array([[0, 0, 0, 1]], dtype=np.float32)
        r = fit_mix(u, l, np.array([0]), k=5, n_init=2, seed=0)
        self.assertFalse(np.isnan(r.centres).any())
        self.assertGreater((r.cluster_sizes == 0).sum(), 0)
        self.assertEqual(r.cluster_sizes[0], 1)                  # the labelled singleton keeps its cluster
        self.assertEqual(r.cluster_sizes.sum(), 13)


class TestV2(unittest.TestCase):
    def test_v2_uses_one_global_assignment(self):
        # cluster 0 holds class 0 (old) and class 2 (new); a per-subset Hungarian would let both claim cluster 0
        y = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        p = np.array([0, 0, 0, 1, 1, 0, 0, 0])
        old = y < 2
        all_acc, old_acc, new_acc = upstream.split_cluster_acc_v2(y, p, old)
        self.assertAlmostEqual(all_acc, 5 / 8)
        self.assertEqual((old_acc, new_acc), (1.0, 0.0))
        v1 = upstream._metrics["split_cluster_acc_v1"](y, p, old)
        self.assertEqual(v1[2], 1.0)                             # v1 maps class 2 to cluster 0 separately


if __name__ == "__main__":
    unittest.main()
