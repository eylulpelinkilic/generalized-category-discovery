import ast
import json
import unittest

import pandas as pd

from ovar.benchmark import (load_benchmark, nec_action_id, read_labels, source_group,
                            build_dataset, label_matches_path)
from ovar.config import project_path
from ovar.manifest import load_manifest, load_vocab, validate
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


if __name__ == "__main__":
    unittest.main()
