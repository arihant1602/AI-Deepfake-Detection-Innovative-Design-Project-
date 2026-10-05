"""
Integration tests for the gated pipeline (Layer 5).
"""

import os
import unittest

from pipeline import (
    VERDICT_AUTHENTIC, VERDICT_INJECTION_ENV, VERDICT_INJECTION_PRNU, VERDICT_NO_ANOMALY,
    _fuse, check_hardware_attestation, create_fast_sample_clip, extract_frames, load_frames,
    run_detection_pipeline,
)

REAL, FAKE, FP = "real_sim.mp4", "fake_sim.mp4", "camera_fingerprint.npy"
HAVE_SIMS = all(os.path.exists(p) for p in (REAL, FAKE, FP))


class TestGate1(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(check_hardware_attestation("Clean Physical Device")["verdict"], "PASS")
        for preset in ("Compromised / Rooted Android (Magisk/SU)", "Android Emulator (Goldfish/QEMU)",
                       "Virtual Camera Injection (OBS / Hooked Driver)"):
            hw = check_hardware_attestation(preset)
            self.assertTrue(hw["blocked"], preset)
            self.assertEqual(hw["verdict"], "BLOCK")

    def test_live_probe_structure(self):
        hw = check_hardware_attestation("live")
        self.assertIn(hw["verdict"], ("PASS", "FLAG_FOR_REVIEW", "BLOCK"))
        for key in ("diagnostics_summary", "hardware_telemetry", "camera_audit", "anti_tampering", "camera_id"):
            self.assertIn(key, hw)
        self.assertLess(hw["latency_ms"], 500.0)

    def test_live_probe_targets_requested_node(self):
        hw = check_hardware_attestation("live", camera_node="/dev/video0")
        self.assertEqual(hw["camera_audit"]["target_node"], "/dev/video0")

    def test_nonexistent_node_blocks(self):
        hw = check_hardware_attestation("live", camera_node="/dev/video_does_not_exist")
        self.assertTrue(hw["blocked"])

    @unittest.skipUnless(HAVE_SIMS, "run generate_test_videos.py first")
    def test_file_upload_is_unattested(self):
        hw = check_hardware_attestation("file_upload", video_path=REAL)
        self.assertEqual(hw["verdict"], "UNATTESTED_ORIGIN")
        self.assertFalse(hw["blocked"])


@unittest.skipUnless(HAVE_SIMS, "run generate_test_videos.py first")
class TestFileHelpers(unittest.TestCase):
    def test_load_frames_is_lossless_decode(self):
        frames = load_frames(REAL, 10)
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[0].ndim, 3)

    def test_preview_clip_and_rgb_extraction(self):
        path = create_fast_sample_clip(REAL, max_frames=15)
        self.assertLessEqual(len(extract_frames(path, max_frames=30)), 15)
        if path != REAL:
            os.remove(path)


@unittest.skipUnless(HAVE_SIMS, "run generate_test_videos.py first")
class TestEndToEnd(unittest.TestCase):
    def test_gate1_termination_skips_pixel_analysis(self):
        res = run_detection_pipeline(REAL, attestation_mode="Android Emulator (Goldfish/QEMU)")
        self.assertEqual(res["gate"], 1)
        self.assertEqual(res["verdict"], VERDICT_INJECTION_ENV)
        self.assertNotIn("gate2_ms", res["timings"])

    def test_reference_mismatch_terminates_at_gate2(self):
        res = run_detection_pipeline(FAKE, attestation_mode="Clean Physical Device", ref_fingerprint_path=FP)
        self.assertEqual(res["gate"], 2)
        self.assertEqual(res["verdict"], VERDICT_INJECTION_PRNU)

    def test_blind_prnu_never_blocks(self):
        res = run_detection_pipeline(FAKE, attestation_mode="file_upload")
        self.assertNotEqual(res["gate"], 2)
        if res["gate"] is None:
            self.assertTrue(any("advisory" in l for l in res["limitations"]))

    def test_genuine_file_with_reference_passes_but_is_not_called_live(self):
        res = run_detection_pipeline(REAL, attestation_mode="Clean Physical Device", ref_fingerprint_path=FP)
        self.assertIsNone(res["gate"])
        self.assertEqual(res["verdict"], VERDICT_NO_ANOMALY)
        self.assertEqual(res["details"]["prnu"]["verdict"], "PRESENT")
        self.assertTrue(res["details"]["prnu"]["enforced"])
        self.assertEqual(len(res["limitations"]), 1)
        self.assertIn("frames", res)


class TestFusion(unittest.TestCase):
    hw = {"verdict": "PASS"}
    ch = {"verdict": "PASS"}
    prnu = {"mode": "reference", "verdict": "PRESENT"}
    temporal = {"abstained": False, "frames_with_face": 60}

    def test_authentic_requires_every_live_check(self):
        self.assertEqual(_fuse(self.hw, self.ch, self.prnu, self.temporal, live=True)["verdict"], VERDICT_AUTHENTIC)

    def test_each_missing_check_downgrades(self):
        cases = [
            ({"verdict": "FLAG_FOR_REVIEW", "details": "x"}, self.ch, self.prnu, self.temporal),
            (self.hw, {"verdict": "UNSUPPORTED"}, self.prnu, self.temporal),
            (self.hw, self.ch, {"mode": "blind", "verdict": "PRESENT"}, self.temporal),
            (self.hw, self.ch, self.prnu, {"abstained": True, "frames_with_face": 2}),
        ]
        for hw, ch, pr, te in cases:
            out = _fuse(hw, ch, pr, te, live=True)
            self.assertEqual(out["verdict"], VERDICT_NO_ANOMALY)
            self.assertEqual(len(out["limitations"]), 1)

    def test_files_are_never_authentic_live(self):
        self.assertEqual(_fuse(self.hw, None, self.prnu, self.temporal, live=False)["verdict"], VERDICT_NO_ANOMALY)


if __name__ == "__main__":
    unittest.main()
