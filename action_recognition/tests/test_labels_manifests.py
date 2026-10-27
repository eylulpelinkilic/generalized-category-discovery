import ast
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ovar.benchmark import (load_benchmark, nec_action_id, read_labels, source_group,
                            build_dataset, label_matches_path)
from ovar.config import project_path
from ovar.inference import class_texts, subset_per_class
from ovar.manifest import load_manifest, load_vocab, validate
from ovar.config import load_experiment
from ovar.native import compare_openvclip, export_openvclip, import_tcclip
from ovar.text import TEMPLATE_SETS
from tests.official_source import THIRD_PARTY, require

TABLE_I = {"ucf101": (50, 51), "hmdb51": (33, 18), "arid": (6, 5), "necdrone": (7, 9)}


class TestOfficialManifests(unittest.TestCase):
    def setUp(self):
        self.bench = load_benchmark()
        require(project_path(self.bench["xov_repo"]))

    def test_manifests_validate_against_official_lists_and_table_i(self):
        xov = project_path(self.bench["xov_repo"])
        for name, spec in self.bench["datasets"].items():
            report = validate(load_manifest(name, spec["split"]), load_vocab(name), expected=spec["expected"],
                              official_lists={k: xov / v for k, v in spec["lists"].items()})
            self.assertEqual(report["errors"], [], name)
            counts = report["counts"]
            self.assertEqual((counts["familiar_classes"], counts["unfamiliar_classes"]), TABLE_I[name])

    def test_committed_manifests_match_a_fresh_build(self):
        xov = project_path(self.bench["xov_repo"])
        for name, spec in self.bench["datasets"].items():
            fresh, vocab, _ = build_dataset(name, spec, xov)
            stored = load_manifest(name, spec["split"])
            pd.testing.assert_frame_equal(fresh.reset_index(drop=True), stored, check_dtype=False)

    def test_hmdb_ids_follow_xov_not_openvclip(self):
        xov = read_labels(project_path(self.bench["xov_repo"]) / "labels" / "hmdb51_labels_corrected.csv")
        ovc = {int(k): v for k, v in json.loads(
            require(THIRD_PARTY / "Open-VCLIP" / "zs_label_db" / "hmdb-index2cls.json").read_text()).items()}
        self.assertEqual((xov[5], ovc[5]), ("climb stairs", "climb"))
        manifest = load_manifest("hmdb51", "test")
        self.assertEqual(set(manifest.loc[manifest["video_path"].str.startswith("climb_stairs/"),
                                          "action_id"]), {5})

    def test_vocabulary_uses_full_target_label_set(self):
        for name, spec in self.bench["datasets"].items():
            self.assertEqual(len(load_vocab(name)), spec["expected"]["classes"])


class TestFilenameConventions(unittest.TestCase):
    def test_nec_action_codes(self):
        self.assertEqual(nec_action_id("NEC-Phantom-002_A0007_P0014_V0007.mp4"), 6)
        self.assertEqual(nec_action_id("NEC-Phantom-002_B0001_P0014P0015_V0008.mp4"), 10)

    def test_source_groups(self):
        self.assertEqual(source_group("ucf101", "Archery/v_Archery_g01_c03.avi")[0], "ucf101:Archery_g01")
        self.assertEqual(source_group("hmdb51", "brush_hair/brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED"
                                                "!!!!_brush_hair_h_cm_np1_le_goo_3.avi")[0],
                         "hmdb51:Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!")
        self.assertEqual(source_group("arid", "Drink/Drink_10_1.mp4")[0], "arid:s10")
        self.assertEqual(source_group("necdrone", "NEC-Phantom-001_A0002_P0016_V0003.mp4")[0],
                         "necdrone:Phantom-001_V0003")

    def test_no_unparsed_hmdb_groups(self):
        self.assertEqual(set(load_manifest("hmdb51", "test")["source_group_basis"]), {"hmdb51_source_title"})


class TestValidatorCatchesErrors(unittest.TestCase):
    def setUp(self):
        self.manifest, self.vocab = load_manifest("arid", "test"), load_vocab("arid")

    def errors(self, manifest, vocab=None):
        return validate(manifest, vocab if vocab is not None else self.vocab)["errors"]

    def test_clean(self):
        self.assertEqual(self.errors(self.manifest), [])

    def test_duplicate_and_label_and_status(self):
        dup = pd.concat([self.manifest, self.manifest.head(1)], ignore_index=True)
        self.assertTrue(any("duplicated video_id" in e for e in self.errors(dup)))
        wrong = self.manifest.copy()
        wrong.loc[0, "action_id"] = 3
        self.assertTrue(any("inconsistent labels" in e for e in self.errors(wrong)))
        status = self.manifest.copy()
        status.loc[0, "category_status"] = "unfamiliar"
        self.assertTrue(any("mixed status" in e for e in self.errors(status)))

    def test_missing_official_video_and_non_contiguous_vocab(self):
        xov = project_path(load_benchmark()["xov_repo"])
        spec = load_benchmark()["datasets"]["arid"]
        lists = {k: xov / v for k, v in spec["lists"].items()}
        report = validate(self.manifest.iloc[1:], self.vocab, official_lists=lists)
        self.assertTrue(any("split membership" in e for e in report["errors"]))
        vocab = self.vocab.copy()
        vocab.loc[0, "action_id"] = 99
        self.assertTrue(any("contiguous" in e for e in self.errors(self.manifest, vocab)))

    def test_missing_files_are_reported_not_dropped(self):
        report = validate(self.manifest, self.vocab, paths={"dataset_roots": {"arid": "/nonexistent"}},
                          check_files=True)
        self.assertEqual(report["files"]["missing"], len(self.manifest))
        self.assertEqual(len(report["failures"]), len(self.manifest))


class TestTemplates(unittest.TestCase):
    def test_openvclip28_is_verbatim(self):
        tree = ast.parse(require(THIRD_PARTY / "Open-VCLIP" / "slowfast" / "models" /
                                 "temporalclip_video_model.py").read_text())
        lists = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "text_aug" for t in n.targets) and isinstance(n.value, ast.List)]
        official = ["".join(v.value for v in e.values) for e in lists[0].elts]
        self.assertEqual(official, TEMPLATE_SETS["openvclip28"])


class TestClassTexts(unittest.TestCase):
    def test_default_is_text_name_and_raw_names_are_opt_in(self):
        vocab = load_vocab("ucf101")
        self.assertEqual(class_texts(vocab, {"prompts": {"template_set": "video_person"}})[0], "Apply Eye Makeup")
        raw = class_texts(vocab, {"prompts": {"template_set": "video_person", "class_names": "official_name"}})
        self.assertEqual(raw[0], "ApplyEyeMakeup")
        xov = read_labels(require(THIRD_PARTY / "XOV-Action" / "labels" / "ucf_101_labels.csv"))
        self.assertEqual(raw, [xov[i] for i in sorted(xov)])
        with self.assertRaises(ValueError):
            class_texts(vocab, {"prompts": {"class_names": "label"}})


class TestExportNative(unittest.TestCase):
    def test_hmdb51_paths_follow_the_folder_layout(self):
        for single, prefix in ((False, "brush_hair/brush_hair/"), (True, "brush_hair/")):
            with tempfile.TemporaryDirectory() as tmp:
                export_openvclip({"hmdb51": "test"}, tmp, per_class=1,
                                 paths={"hmdb51_single_class_folder": single})
                first = (Path(tmp) / "hmdb51" / "test.csv").read_text().splitlines()[0]
                self.assertTrue(first.startswith(prefix) and not first.startswith(prefix + "brush_hair/"), first)


class TestImportTcclip(unittest.TestCase):
    """Synthetic scores file: software check of the import, not a TC-CLIP result."""

    def _setup(self, tmp):
        paths = Path(tmp) / "paths.yaml"
        paths.write_text("checkpoints:\n  tcclip_zero_shot: /x/ckpt.pth\n")
        cfg = load_experiment("tcclip_vitb16__tcclip_native_16x2x1", ["datasets=[hmdb51]"], paths)
        manifest = subset_per_class(load_manifest("hmdb51", "test"), 1, 0).reset_index(drop=True)
        rng = np.random.default_rng(0)
        scores = rng.random((len(manifest), 51)).astype(np.float32)
        order = rng.permutation(len(manifest))          # the official sampler shuffles
        return cfg, manifest, scores, order

    def _save(self, path, file_id, label, scores):
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "hmdb51.npz", file_id=file_id, label=label, scores=scores, meta=json.dumps(
            {"num_frames": 16, "num_clip": 2, "num_crop": 1, "seed": 1024, "wise_ft": 0.0, "opt_level": "O1"}))

    def test_rows_return_to_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, manifest, scores, order = self._setup(tmp)
            labels = manifest["action_id"].to_numpy()
            self._save(Path(tmp) / "scores", order, labels[order], scores[order])
            run = import_tcclip(cfg, Path(tmp) / "scores", Path(tmp) / "run", per_class=1)
            pred = pd.read_csv(run / "hmdb51" / "predictions.csv", keep_default_na=False)
            self.assertEqual(pred["video_id"].tolist(), manifest["video_id"].tolist())
            self.assertEqual(pred["pred_id"].tolist(), scores.argmax(1).tolist())
            self.assertEqual(json.loads((run / "run.json").read_text())["comparison"], "native")

    def test_rejects_wrong_labels_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, manifest, scores, order = self._setup(tmp)
            labels = manifest["action_id"].to_numpy()
            bad = labels[order].copy()
            bad[0] = (bad[0] + 1) % 51
            self._save(Path(tmp) / "a", order, bad, scores[order])
            with self.assertRaisesRegex(ValueError, "labels differ"):
                import_tcclip(cfg, Path(tmp) / "a", Path(tmp) / "run", per_class=1)
            self._save(Path(tmp) / "b", order[1:], labels[order][1:], scores[order][1:])
            with self.assertRaisesRegex(ValueError, "once each"):
                import_tcclip(cfg, Path(tmp) / "b", Path(tmp) / "run", per_class=1)


class TestCompareNative(unittest.TestCase):
    def test_agreement_and_disagreements(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "ucf101").mkdir(parents=True)
            pd.DataFrame({"video_id": ["a", "b", "c"], "action_id": [0, 1, 2], "pred_id": [1, 2, 0],
                          "status": ["ok", "ok", "failed"], "correct": [0, 0, 0]}).to_csv(
                run / "ucf101" / "predictions.csv", index=False)
            official = np.array([[9, 0, 0], [0, 0, 9], [0, 0, 9]], dtype=np.float32)
            with open(Path(tmp) / "preds.pkl", "wb") as f:
                pickle.dump([official, np.array([0, 1, 2])], f)
            with self.assertRaises(PermissionError):
                compare_openvclip(Path(tmp) / "preds.pkl", run, "ucf101")
            out = compare_openvclip(Path(tmp) / "preds.pkl", run, "ucf101", trust_pickle=True)
        self.assertEqual(out["top1_agreement"], 0.5)        # failed videos excluded from agreement
        self.assertEqual(out["adapter_failed"], 1)
        self.assertAlmostEqual(out["official_accuracy"], 2 / 3)
        self.assertEqual(out["adapter_accuracy_failures_as_errors"], 0.0)
        self.assertEqual(out["disagreements"],
                         [{"video_id": "a", "action_id": 0, "official_pred": 0, "adapter_pred": 1}])


if __name__ == "__main__":
    unittest.main()
