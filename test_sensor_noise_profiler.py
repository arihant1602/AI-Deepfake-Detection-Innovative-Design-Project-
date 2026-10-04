"""
Unit tests for Layer 2: Camera Sensor Noise Profiling (PRNU Analysis)
"""

import json
import os
import unittest
import numpy as np

from camera_sensor_noise_profiling import (
    extract_noise_residual,
    zero_mean_normalize,
    compute_pce_and_ncc,
    CameraSensorNoiseProfiler,
    LayerResult,
)


class TestPRNUResidualExtraction(unittest.TestCase):
    def test_residual_shape_and_dtype(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        res = extract_noise_residual(img)
        self.assertEqual(res.shape, (100, 100))
        self.assertTrue(np.issubdtype(res.dtype, np.floating))

    def test_zero_mean_normalization(self):
        arr = np.random.normal(50.0, 15.0, (80, 80)).astype(np.float32)
        norm = zero_mean_normalize(arr)
        self.assertAlmostEqual(float(np.mean(norm)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(norm)), 1.0, places=3)

    def test_flat_array_normalization(self):
        flat = np.full((50, 50), 128.0, dtype=np.float32)
        norm = zero_mean_normalize(flat)
        self.assertTrue(np.all(norm == 0.0))


class TestCorrelationAndPCE(unittest.TestCase):
    def test_matching_fingerprints(self):
        rng = np.random.default_rng(42)
        pattern = rng.normal(0, 1.0, (64, 64)).astype(np.float32)
        # Add small perturbation
        noisy_copy = pattern + rng.normal(0, 0.1, (64, 64)).astype(np.float32)
        pce, ncc = compute_pce_and_ncc(pattern, noisy_copy)
        self.assertGreater(ncc, 0.90)
        self.assertGreater(pce, 500.0)

    def test_uncorrelated_noise(self):
        rng = np.random.default_rng(99)
        noise1 = rng.normal(0, 1.0, (64, 64)).astype(np.float32)
        noise2 = rng.normal(0, 1.0, (64, 64)).astype(np.float32)
        pce, ncc = compute_pce_and_ncc(noise1, noise2)
        self.assertLess(abs(ncc), 0.10)
        self.assertLess(pce, 35.0)

    def test_dimension_mismatch_resizing(self):
        p1 = np.random.normal(0, 1, (64, 64)).astype(np.float32)
        p2 = np.random.normal(0, 1, (128, 128)).astype(np.float32)
        pce, ncc = compute_pce_and_ncc(p1, p2)
        self.assertIsInstance(pce, float)
        self.assertIsInstance(ncc, float)


class TestProfilerPipeline(unittest.TestCase):
    def setUp(self):
        self.profiler = CameraSensorNoiseProfiler()
        self.real_clip = "real_sim.mp4"
        self.fake_clip = "fake_sim.mp4"
        self.ref_fp = "camera_fingerprint.npy"

    def test_missing_video_file(self):
        result = self.profiler.analyze("nonexistent_video_path.mp4")
        self.assertEqual(result.frames_analyzed, 0)
        self.assertIn("Could not open", result.explanation)

    def test_real_video_analysis(self):
        if not os.path.exists(self.real_clip):
            self.skipTest(f"{self.real_clip} not found; run generate_test_videos.py first")
        result = self.profiler.analyze(self.real_clip)
        self.assertEqual(result.layer, "camera_sensor_noise_profiling")
        self.assertEqual(result.frames_analyzed, 90)
        self.assertFalse(result.flagged)
        self.assertLess(result.score, 0.25)
        self.assertGreater(result.components["inter_frame_persistence"], 0.20)
        self.assertGreater(result.components["pce_score"], 45.0)

    def test_fake_video_analysis(self):
        if not os.path.exists(self.fake_clip):
            self.skipTest(f"{self.fake_clip} not found; run generate_test_videos.py first")
        result = self.profiler.analyze(self.fake_clip)
        self.assertEqual(result.layer, "camera_sensor_noise_profiling")
        self.assertEqual(result.frames_analyzed, 90)
        self.assertTrue(result.flagged)
        self.assertGreaterEqual(result.score, 0.55)
        self.assertLess(result.components["pce_score"], 25.0)

    def test_reference_fingerprint_matching(self):
        if not (os.path.exists(self.real_clip) and os.path.exists(self.ref_fp)):
            self.skipTest("Reference or test video not found")
        res_real = self.profiler.analyze(self.real_clip, ref_fingerprint_path=self.ref_fp)
        res_fake = self.profiler.analyze(self.fake_clip, ref_fingerprint_path=self.ref_fp)
        self.assertFalse(res_real.flagged)
        self.assertTrue(res_fake.flagged)

    def test_json_serialization(self):
        res = LayerResult(
            frames_analyzed=90,
            frames_with_face=90,
            score=0.12,
            flagged=False,
            confidence=1.0,
            components={"pce_score": 120.5},
            explanation="Test explanation",
        )
        json_str = res.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["layer"], "camera_sensor_noise_profiling")
        self.assertEqual(parsed["score"], 0.12)
        self.assertFalse(parsed["flagged"])


if __name__ == "__main__":
    unittest.main()
