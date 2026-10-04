"""
Unit tests for Layer 3: Temporal Consistency & Frequency Analysis
"""

import json
import os
import unittest
import numpy as np

from temporal_consistency_analysis import (
    high_frequency_ratio,
    edge_density,
    flow_coherence,
    FaceLocator,
    TemporalConsistencyAnalyzer,
    LayerResult,
)


class TestTemporalFeatureExtraction(unittest.TestCase):
    def test_high_frequency_ratio_flat_image(self):
        flat = np.full((128, 128), 128, dtype=np.uint8)
        hf = high_frequency_ratio(flat, 0.25)
        self.assertAlmostEqual(hf, 0.0, places=4)

    def test_high_frequency_ratio_noisy_image(self):
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        hf = high_frequency_ratio(noise, 0.25)
        self.assertGreater(hf, 0.1)

    def test_edge_density_flat_image(self):
        flat = np.full((100, 100), 50, dtype=np.uint8)
        density = edge_density(flat)
        self.assertEqual(density, 0.0)

    def test_edge_density_patterned_image(self):
        pattern = np.zeros((100, 100), dtype=np.uint8)
        pattern[::4, :] = 255  # horizontal bars
        density = edge_density(pattern)
        self.assertGreater(density, 0.05)

    def test_flow_coherence_identical_frames(self):
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        coherence = flow_coherence(img, img)
        self.assertGreaterEqual(coherence, 0.95)

    def test_flow_coherence_random_frames(self):
        img1 = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        coherence = flow_coherence(img1, img2)
        self.assertIsInstance(coherence, float)


class TestFaceLocator(unittest.TestCase):
    def test_face_locator_init(self):
        locator = FaceLocator(max_missed_frames=3)
        self.assertEqual(locator.max_missed_frames, 3)
        self.assertIsNone(locator.last_box)

    def test_face_locator_fallback(self):
        locator = FaceLocator(max_missed_frames=2)
        locator.last_box = (10, 10, 50, 50)
        # Empty frame won't match Haar cascade, fallback should work for 2 calls
        blank = np.zeros((200, 200), dtype=np.uint8)
        box1 = locator.locate(blank)
        self.assertEqual(box1, (10, 10, 50, 50))
        box2 = locator.locate(blank)
        self.assertEqual(box2, (10, 10, 50, 50))
        # 3rd call exceeds max_missed_frames
        box3 = locator.locate(blank)
        self.assertIsNone(box3)


class TestTemporalConsistencyPipeline(unittest.TestCase):
    def setUp(self):
        self.analyzer = TemporalConsistencyAnalyzer()
        self.real_clip = "real_sim.mp4"
        self.fake_clip = "fake_sim.mp4"

    def test_missing_video_raises_io_error(self):
        with self.assertRaises(IOError):
            self.analyzer.analyze("nonexistent_video_path.mp4")

    def test_real_video_analysis(self):
        if not os.path.exists(self.real_clip):
            self.skipTest(f"{self.real_clip} not found; run generate_test_videos.py first")
        result = self.analyzer.analyze(self.real_clip)
        self.assertEqual(result.layer, "temporal_consistency_frequency_analysis")
        self.assertEqual(result.frames_analyzed, 90)
        self.assertFalse(result.flagged)
        self.assertLess(result.score, 0.60)
        self.assertIn("flicker_rate", result.components)

    def test_json_serialization(self):
        res = LayerResult(
            layer="temporal_consistency_frequency_analysis",
            frames_analyzed=90,
            frames_with_face=90,
            score=0.25,
            flagged=False,
            confidence=1.0,
            components={"flicker_rate": 0.12},
            explanation="Test temporal explanation",
        )
        json_str = res.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["layer"], "temporal_consistency_frequency_analysis")
        self.assertEqual(parsed["score"], 0.25)
        self.assertFalse(parsed["flagged"])


if __name__ == "__main__":
    unittest.main()
