import types
import unittest

import numpy as np
import torch

from ovar.preprocess import CLIP_MEAN, CLIP_STD, preprocess
from ovar.sampling import slowfast_dense, tsn_midpoints
from tests.official_source import THIRD_PARTY, NumpyIntShim, functions


class TestTSN(unittest.TestCase):
    def test_known_values(self):
        idx = tsn_midpoints(20, 16)[0].tolist()
        self.assertEqual(idx, [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19])

    def test_short_video_repeats_but_stays_in_range(self):
        idx = tsn_midpoints(5, 16)[0]
        self.assertTrue((np.diff(idx) >= 0).all())
        self.assertEqual(set(idx.tolist()), set(range(5)))

    def test_matches_official_mmaction_sampler(self):
        fn = functions(THIRD_PARTY / "tc-clip" / "datasets" / "pipeline.py", ["_get_test_clips"],
                       {"np": NumpyIntShim()}, class_name="SampleFrames")["_get_test_clips"]
        for n in (1, 7, 16, 17, 100, 301):
            for t in (8, 16):
                me = types.SimpleNamespace(clip_len=1, frame_interval=1, num_clips=t, twice_sample=False)
                self.assertEqual(tsn_midpoints(n, t)[0].tolist(), fn(me, n).tolist(), (n, t))


class TestSlowFastDense(unittest.TestCase):
    def test_matches_official_decoder_functions(self):
        fns = functions(THIRD_PARTY / "Open-VCLIP" / "slowfast" / "datasets" / "decoder.py",
                        ["get_start_end_idx", "temporal_sampling"])
        for n in (20, 64, 128, 129, 300, 1000):
            for fps in (25.0, 29.97, 30.0):
                for views in (1, 3):
                    ours = slowfast_dense(n, fps, 8, 16, views)
                    clip_size = max(1.0, 16 * 8 / 30.0 * fps)
                    frames = torch.arange(n).view(n, 1, 1, 1)
                    for v in range(views):
                        s, e = fns["get_start_end_idx"](n, clip_size, v, views, use_offset=True)[:2]
                        theirs = fns["temporal_sampling"](frames, s, e, 8).flatten().tolist()
                        self.assertEqual(ours[v].tolist(), theirs, (n, fps, views, v))

    def test_short_video_clamps_to_last_frame(self):
        idx = slowfast_dense(40, 30.0, 8, 16, 3)
        self.assertTrue(all((v <= 39).all() for v in idx))
        self.assertTrue(all((v == idx[0]).all() for v in idx), "views coincide when video < window")


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in
                       ((240, 320), (320, 240), (224, 224), (100, 300), (1080, 1920))]

    def test_clip_center_equals_open_clip_transform(self):
        import open_clip
        from PIL import Image

        ref = open_clip.image_transform(224, is_train=False, mean=CLIP_MEAN, std=CLIP_STD,
                                        interpolation="bicubic", resize_mode="shortest")
        for f in self.frames:
            ours = preprocess(f[None], {"preset": "clip_center", "size": 224})[0]
            theirs = ref(Image.fromarray(f))
            self.assertLess((ours - theirs).abs().max().item(), 1e-6)

    def test_slowfast_center_equals_official_transforms(self):
        path = THIRD_PARTY / "Open-VCLIP" / "slowfast" / "datasets" / "transform.py"
        fns = functions(path, ["random_short_side_scale_jitter", "uniform_crop"])
        mean, std = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
        for f in self.frames:
            # PySlowFast: /255, normalise (THWC), permute to CTHW, short-side scale, centre crop.
            x = (torch.from_numpy(f[None]).float() / 255.0 - mean) / std
            x = x.permute(0, 3, 1, 2)  # T C H W (the functions treat dim 0 as frames)
            x, _ = fns["random_short_side_scale_jitter"](x, 224, 224)
            x, _ = fns["uniform_crop"](x, 224, 1)
            ours = preprocess(f[None], {"preset": "slowfast_center", "size": 224})
            self.assertEqual(ours.shape, x.shape)
            self.assertLess((ours - x).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
