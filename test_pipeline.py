"""
Integration and End-to-End Pipeline Unit Tests (Layer 5 Gated Fusion)
"""

import os
import unittest
import numpy as np

from pipeline import (
    check_hardware_attestation,
    check_sensor_noise,
    check_temporal_coherence,
    create_fast_sample_clip,
    extract_frames,
    run_detection_pipeline,
)


class TestHardwareAttestationGate(unittest.TestCase):
    def test_clean_device(self):
        hw = check_hardware_attestation("Clean Physical Device")
        self.assertTrue(hw["passed"])
        self.assertFalse(hw["blocked"])
        self.assertEqual(hw["verdict"], "PASS")

    def test_rooted_device_blocks(self):
        hw = check_hardware_attestation("Compromised / Rooted Android (Magisk/SU)")
        self.assertFalse(hw["passed"])
        self.assertTrue(hw["blocked"])
        self.assertEqual(hw["verdict"], "BLOCK")

    def test_emulator_blocks(self):
        hw = check_hardware_attestation("Android Emulator (Goldfish/QEMU)")
        self.assertFalse(hw["passed"])
        self.assertTrue(hw["blocked"])
        self.assertEqual(hw["verdict"], "BLOCK")

    def test_virtual_camera_flagged_for_review(self):
        hw = check_hardware_attestation("Virtual Camera Injection (OBS / Hooked Driver)")
        self.assertTrue(hw["passed"])
        self.assertFalse(hw["blocked"])
        self.assertTrue(hw["virtual_camera_detected"])
        self.assertEqual(hw["verdict"], "FLAG_FOR_REVIEW")


class TestPipelineFastSamplingAndFrameExtraction(unittest.TestCase):
    def setUp(self):
        self.real_clip = "real_sim.mp4"

    def test_fast_sample_clip(self):
        if not os.path.exists(self.real_clip):
            self.skipTest(f"{self.real_clip} not found")
        sample_path = create_fast_sample_clip(self.real_clip, max_frames=15)
        self.assertTrue(os.path.exists(sample_path))
        frames = extract_frames(sample_path, max_frames=30)
        self.assertLessEqual(len(frames), 15)
        os.remove(sample_path)

    def test_extract_frames_shape_and_rgb(self):
        if not os.path.exists(self.real_clip):
            self.skipTest(f"{self.real_clip} not found")
        frames = extract_frames(self.real_clip, max_frames=5)
        self.assertGreaterEqual(len(frames), 1)
        self.assertEqual(len(frames[0].shape), 3)
        self.assertEqual(frames[0].shape[2], 3)  # RGB 3-channel


class TestEndToEndGatedPipeline(unittest.TestCase):
    def setUp(self):
        self.real_clip = "real_sim.mp4"
        self.fake_clip = "fake_sim.mp4"

    def test_early_termination_at_gate_1_rooted(self):
        if not os.path.exists(self.real_clip):
            self.skipTest(f"{self.real_clip} not found")
        res = run_detection_pipeline(self.real_clip, attestation_mode="Compromised / Rooted Android (Magisk/SU)")
        self.assertEqual(res["gate"], 1)
        self.assertIn("DIGITAL INJECTION DETECTED", res["verdict"])
        self.assertTrue(res["details"]["blocked"])

    def test_early_termination_at_gate_1_emulator(self):
        if not os.path.exists(self.real_clip):
            self.skipTest(f"{self.real_clip} not found")
        res = run_detection_pipeline(self.real_clip, attestation_mode="Android Emulator (Goldfish/QEMU)")
        self.assertEqual(res["gate"], 1)
        self.assertIn("DIGITAL INJECTION DETECTED", res["verdict"])

    def test_gate_2_sensor_noise_flagging_fake_clip(self):
        if not os.path.exists(self.fake_clip):
            self.skipTest(f"{self.fake_clip} not found")
        res = run_detection_pipeline(self.fake_clip, attestation_mode="Clean Physical Device")
        self.assertEqual(res["gate"], 2)
        self.assertEqual(res["verdict"], "DIGITAL INJECTION DETECTED")

    def test_authentic_stream_all_gates_pass(self):
        if not os.path.exists(self.real_clip):
            self.skipTest(f"{self.real_clip} not found")
        res = run_detection_pipeline(self.real_clip, attestation_mode="Clean Physical Device")
        self.assertEqual(res["verdict"], "AUTHENTIC LIVE STREAM")
        self.assertIsNone(res["gate"])
        self.assertIn("frames", res)
        self.assertIn("details", res)
        self.assertTrue(res["details"]["attestation"]["passed"])
        self.assertTrue(res["details"]["prnu"]["passed"])
        self.assertTrue(res["details"]["temporal"]["passed"])


if __name__ == "__main__":
    unittest.main()
