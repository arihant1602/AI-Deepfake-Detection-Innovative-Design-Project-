"""
Unit tests for Layer 2: live sensor-noise test (default) and the optional reference-PRNU API.
"""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from camera_sensor_noise_profiling import (
    CameraSensorNoiseProfiler,
    FingerprintStore,
    LayerResult,
    compute_pce_and_ncc,
    pce_with_shift_search,
    extract_noise_residual,
    zero_mean_normalize,
)

H = W = 256


def make_scene(n, moving=True, prnu=None, seed=0, noise=2.0, warp_photo=False):
    """Textured disc moving over a static background; optional sensor PRNU."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.uniform(60, 200, (H, W)).astype(np.float32), (0, 0), 3)
    tex = cv2.GaussianBlur(rng.uniform(80, 220, (H, W)).astype(np.float32), (0, 0), 1.5)
    yy, xx = np.mgrid[:H, :W]
    frames = []
    for t in range(n):
        cx = W / 2 + (60 * np.sin(t / 8) if moving else 0)
        cy = H / 2 + (30 * np.cos(t / 11) if moving else 0)
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 < 70 ** 2
        img = bg.copy()
        img[disk] = np.roll(tex, (int(cy - H / 2), int(cx - W / 2)), (0, 1))[disk]
        if prnu is not None:
            img = img * (1 + prnu)
        img = img + rng.normal(0, noise, img.shape)
        frames.append(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    if warp_photo:  # single captured photo, re-animated with a slow pan + zoom (non-periodic)
        src = frames[0]
        frames = []
        for t in range(n):
            m = cv2.getRotationMatrix2D((W / 2, H / 2), 0.1 * t, 1 + 0.004 * t)
            m[0, 2] += 1.5 * t
            m[1, 2] += 0.5 * t
            frames.append(cv2.warpAffine(src, m, (W, H), borderMode=cv2.BORDER_REFLECT))
    return frames


SENSOR_K = np.random.default_rng(42).normal(0, 0.01, (H, W)).astype(np.float32)
OTHER_K = np.random.default_rng(7).normal(0, 0.01, (H, W)).astype(np.float32)


class TestPrimitives(unittest.TestCase):
    def test_residual_shape_dtype_and_saturation(self):
        img = np.random.default_rng(0).integers(0, 255, (100, 100)).astype(np.uint8)
        img[:10, :10] = 255
        res = extract_noise_residual(img.astype(np.float32))
        self.assertEqual(res.shape, (100, 100))
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.all(res[:10, :10] == 0.0))

    def test_zero_mean_normalization(self):
        arr = np.random.default_rng(1).normal(50.0, 15.0, (80, 80)).astype(np.float32)
        norm = zero_mean_normalize(arr)
        self.assertAlmostEqual(float(np.mean(norm)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(norm)), 1.0, places=3)
        self.assertTrue(np.all(zero_mean_normalize(np.full((5, 5), 3.0)) == 0.0))

    def test_pce_matching_vs_uncorrelated(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, (64, 64)).astype(np.float32)
        pce, ncc = compute_pce_and_ncc(a, a + rng.normal(0, 0.1, (64, 64)).astype(np.float32))
        self.assertGreater(ncc, 0.9)
        self.assertGreater(pce, 500.0)
        noise = rng.normal(0, 1, (64, 64)).astype(np.float32)
        pce, ncc = compute_pce_and_ncc(a, noise)
        self.assertLess(abs(ncc), 0.1)
        self.assertLess(pce, 30.0)
        self.assertLess(pce_with_shift_search(a, noise)[0], 30.0)
        self.assertAlmostEqual(pce_with_shift_search(a, a)[0], compute_pce_and_ncc(a, a)[0], places=3)


def noisy_stream(n=30, kind="camera", moving=False, seed=0):
    """
    Static textured scene plus per-frame noise of a given kind:
      camera  spatially correlated (demosaic-like blur) and partially shared across channels
      grey    white noise identical in all channels (digitally added grain)
      colour  white noise independent per channel
      codec   no fresh noise: static content repeats bit-for-bit (inter-frame codec)
    """
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.uniform(40, 210, (240, 320, 3)).astype(np.float32), (0, 0), 4)
    frames = []
    for t in range(n):
        img = np.roll(base, 9 * t, axis=1) if moving else base.copy()
        if kind == "camera":
            common = rng.normal(0, 1.0, (240, 320, 1))
            own = rng.normal(0, 0.8, (240, 320, 3))
            img = img + cv2.GaussianBlur((common + own).astype(np.float32), (0, 0), 0.7) * 2.2
        elif kind == "grey":
            img = img + rng.normal(0, 2, (240, 320, 1))
        elif kind == "colour":
            img = img + rng.normal(0, 2, (240, 320, 3))
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


class TestLiveNoiseMode(unittest.TestCase):
    def setUp(self):
        self.p = CameraSensorNoiseProfiler()

    def test_camera_like_noise_is_present(self):
        r = self.p.analyze_frames(noisy_stream(kind="camera"))
        self.assertEqual(r.verdict, "PRESENT", r.components)
        self.assertFalse(r.flagged)

    def test_codec_stream_without_fresh_noise_is_absent(self):
        r = self.p.analyze_frames(noisy_stream(kind="codec"))
        self.assertEqual(r.verdict, "ABSENT")
        self.assertIn("no codec copying", r.components["failed_checks"])
        self.assertIn("noise present", r.components["failed_checks"])

    def test_grey_grain_is_absent(self):
        r = self.p.analyze_frames(noisy_stream(kind="grey"))
        self.assertEqual(r.verdict, "ABSENT")
        self.assertTrue(set(r.components["failed_checks"]) & {"sensor-like colour", "sensor-like texture"})

    def test_white_colour_grain_is_absent(self):
        r = self.p.analyze_frames(noisy_stream(kind="colour"))
        self.assertEqual(r.verdict, "ABSENT")
        self.assertIn("sensor-like texture", r.components["failed_checks"])

    def test_panning_textured_video_without_noise_is_not_present(self):
        # A panning clip with no fresh noise (e.g. generated video): its shifting fine texture
        # must not be mistaken for sensor noise.
        rng = np.random.default_rng(3)
        tex = cv2.GaussianBlur(rng.uniform(30, 220, (240, 400, 3)).astype(np.float32), (0, 0), 1.2)
        frames = [np.clip(np.roll(tex, 3 * t, axis=1)[:, :320], 0, 255).astype(np.uint8) for t in range(30)]
        r = self.p.analyze_frames(frames)
        self.assertNotEqual(r.verdict, "PRESENT", r.components)

    def test_too_few_frames_abstains(self):
        r = self.p.analyze_frames(noisy_stream(n=8))
        self.assertEqual(r.verdict, "INCONCLUSIVE")

    def test_missing_video_file(self):
        r = self.p.analyze("nonexistent_video_path.mp4")
        self.assertEqual(r.frames_analyzed, 0)
        self.assertIn("Could not open", r.explanation)


class TestReferenceMode(unittest.TestCase):
    def setUp(self):
        self.p = CameraSensorNoiseProfiler()

    def test_same_sensor_matches(self):
        r = self.p.analyze_frames(make_scene(40, prnu=SENSOR_K, seed=5), ref_fingerprint=SENSOR_K)
        self.assertEqual(r.verdict, "PRESENT")
        self.assertGreater(r.components["pce_score"], 60)

    def test_other_sensor_and_rendered_do_not_match(self):
        for k in (OTHER_K, None):
            r = self.p.analyze_frames(make_scene(40, prnu=k, seed=5), ref_fingerprint=SENSOR_K)
            self.assertEqual(r.verdict, "ABSENT")
            self.assertTrue(r.flagged)
            self.assertLess(r.components["pce_score"], 60)

    def test_enrolled_estimate_matches_new_session(self):
        enrolled = self.p.estimate_fingerprint_from_frames(make_scene(90, prnu=SENSOR_K, seed=1))
        r = self.p.analyze_frames(make_scene(40, prnu=SENSOR_K, seed=9), ref_fingerprint=enrolled)
        self.assertEqual(r.verdict, "PRESENT")

    def test_small_alignment_offset_is_found(self):
        # Genuine webcam sessions were measured with the PCE peak up to 2 px off (0, 0).
        frames = [np.roll(f, (1, -2), axis=(0, 1)) for f in make_scene(40, prnu=SENSOR_K, seed=5)]
        r = self.p.analyze_frames(frames, ref_fingerprint=SENSOR_K)
        self.assertEqual(r.verdict, "PRESENT")
        self.assertEqual(r.components["peak_offset"], [1, -2])

    def test_sensor_mode_mismatch_is_inconclusive(self):
        r = self.p.analyze_frames(make_scene(40, prnu=SENSOR_K), ref_fingerprint=np.zeros((128, 128), np.float32))
        self.assertEqual(r.verdict, "INCONCLUSIVE")
        self.assertFalse(r.flagged)

    def test_bundled_simulation_clips(self):
        if not all(os.path.exists(p) for p in ("real_sim.mp4", "fake_sim.mp4", "camera_fingerprint.npy")):
            self.skipTest("run generate_test_videos.py first")
        self.assertEqual(self.p.analyze("real_sim.mp4", "camera_fingerprint.npy").verdict, "PRESENT")
        self.assertEqual(self.p.analyze("fake_sim.mp4", "camera_fingerprint.npy").verdict, "ABSENT")


class TestStoreAndSerialization(unittest.TestCase):
    def test_fingerprint_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(d)
            self.assertFalse(store.has("usb-5986-216a-1-3", W, H))
            store.save("usb-5986-216a-1-3", SENSOR_K, {"frames": 90})
            self.assertTrue(store.has("usb-5986-216a-1-3", W, H))
            self.assertTrue(np.allclose(store.load("usb-5986-216a-1-3", W, H), SENSOR_K))
            self.assertEqual(store.metadata("usb-5986-216a-1-3", W, H)["frames"], 90)
            self.assertIsNone(store.load("usb-5986-216a-1-3", 640, 480))
            store.delete("usb-5986-216a-1-3", W, H)
            self.assertFalse(store.has("usb-5986-216a-1-3", W, H))

    def test_json_serialization(self):
        res = LayerResult(frames_analyzed=90, score=0.12, verdict="PRESENT", components={"z_score": 12.5})
        parsed = json.loads(res.to_json())
        self.assertEqual(parsed["layer"], "camera_sensor_noise_profiling")
        self.assertEqual(parsed["verdict"], "PRESENT")


if __name__ == "__main__":
    unittest.main()
