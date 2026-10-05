"""
Unit tests for Layer 1: host & camera attestation (no camera or root needed).
Software-camera cases use a fake sysfs tree that mirrors the real kernel layout.
"""

import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

import host_integrity as h

UVC_IOCTL = {"driver": "uvcvideo", "card": "Integrated Camera", "bus_info": "usb-0000:05:00.3-3", "caps": 0x84A00001,
             "device_caps": 0x04200001, "is_capture": True, "is_output": False, "is_metadata": False,
             "is_streaming": True, "error": None}
UVC_META_IOCTL = {**UVC_IOCTL, "device_caps": 0x04A00000, "is_capture": False, "is_metadata": True}
LOOPBACK_IOCTL = {"driver": "v4l2 loopback", "card": "Integrated Camera", "bus_info": "platform:v4l2loopback-000",
                  "caps": 0x85208003, "device_caps": 0x05208003, "is_capture": True, "is_output": True,
                  "is_metadata": False, "is_streaming": True, "error": None}


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def build_fake_sysfs(root):
    """video0/video1: UVC camera (capture + metadata) on USB; video2: v4l2loopback (virtual)."""
    usb_dev = os.path.join(root, "devices/pci0000:00/0000:00:08.1/0000:05:00.3/usb1/1-3")
    usb_if = os.path.join(usb_dev, "1-3:1.0")
    for k, v in {"idVendor": "5986", "idProduct": "216a", "product": "Integrated Camera", "manufacturer": "SunplusIT",
                 "serial": "", "speed": "480"}.items():
        _write(os.path.join(usb_dev, k), v)
    os.makedirs(os.path.join(root, "bus/usb/drivers/uvcvideo"))
    os.makedirs(usb_if, exist_ok=True)
    os.symlink(os.path.join(root, "bus/usb/drivers/uvcvideo"), os.path.join(usb_if, "driver"))
    cls = os.path.join(root, "class/video4linux")
    os.makedirs(cls)
    for n in ("video0", "video1"):
        real = os.path.join(usb_if, "video4linux", n)
        _write(os.path.join(real, "name"), "Integrated Camera: Integrated C")
        os.symlink(usb_if, os.path.join(real, "device"))
        os.symlink(real, os.path.join(cls, n))
    virt = os.path.join(root, "devices/virtual/video4linux/video2")
    _write(os.path.join(virt, "name"), "Integrated Camera")  # attacker-chosen card label
    os.symlink(virt, os.path.join(cls, "video2"))


def fake_ioctl(node):
    return {"video0": UVC_IOCTL, "video1": UVC_META_IOCTL, "video2": LOOPBACK_IOCTL}.get(
        os.path.basename(node), {**UVC_IOCTL, "error": "open failed", "is_capture": False})


class TestDeviceClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sys = os.path.join(self.tmp.name, "sys")
        build_fake_sysfs(self.sys)
        self.patch = mock.patch.object(h, "_query_v4l2_ioctl", side_effect=fake_ioctl)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def devices(self):
        return {os.path.basename(d["node"]): d for d in h.enumerate_video_devices(self.sys, "/dev")}

    def test_physical_uvc_capture_node(self):
        d = self.devices()["video0"]
        self.assertTrue(d["is_physical"])
        self.assertFalse(d["is_virtual"])
        self.assertEqual(d["bus"], "usb")
        self.assertEqual(d["kernel_driver"], "uvcvideo")
        self.assertEqual(d["usb"]["vid"], "5986")
        self.assertEqual(d["camera_id"], "usb-5986-216a-1-3")

    def test_metadata_node_is_not_a_camera(self):
        d = self.devices()["video1"]
        self.assertFalse(d["is_capture"])
        self.assertFalse(d["is_physical"])
        self.assertTrue(d["is_metadata"])

    def test_loopback_with_spoofed_label_is_virtual(self):
        d = self.devices()["video2"]
        self.assertTrue(d["is_virtual"])
        self.assertFalse(d["is_physical"])
        joined = " ".join(d["virtual_reasons"])
        self.assertIn("/sys/devices/virtual", joined)
        self.assertIn("v4l2 loopback", joined)

    def test_default_target_prefers_physical_capture(self):
        audit = h.inspect_camera_hardware(None, self.sys, "/dev", proc_root=self.tmp.name)
        self.assertEqual(audit["target_node"], "/dev/video0")
        self.assertEqual(audit["other_virtual_nodes"], ["/dev/video2"])

    def test_probe_blocks_virtual_and_metadata_nodes(self):
        for node in ("/dev/video1", "/dev/video2", "/dev/video7"):
            r = h.probe_host_integrity(node, sysfs_root=self.sys, dev_root="/dev", proc_root=self.tmp.name)
            if r["emulator_detected"] or r["debugger_detected"] or r["root_detected"]:
                self.skipTest("test host itself is virtualised / hooked / root")
            self.assertEqual(r["verdict"], "BLOCK", node)
            self.assertIn("not a physical camera", r["details"])

    def test_probe_flags_physical_camera_beside_software_camera(self):
        r = h.probe_host_integrity("/dev/video0", sysfs_root=self.sys, dev_root="/dev", proc_root=self.tmp.name)
        if r["emulator_detected"] or r["debugger_detected"] or r["root_detected"]:
            self.skipTest("test host itself is virtualised / hooked / root")
        self.assertEqual(r["verdict"], "FLAG_FOR_REVIEW")
        self.assertTrue(r["passed"])
        self.assertEqual(r["camera_id"], "usb-5986-216a-1-3")


class TestHostScans(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proc = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def add_proc(self, pid, comm, cmdline):
        _write(os.path.join(self.proc, str(pid), "comm"), comm + "\n")
        with open(os.path.join(self.proc, str(pid), "cmdline"), "wb") as f:
            f.write(cmdline.replace(" ", "\x00").encode())

    def test_exact_process_matching(self):
        self.add_proc(100, "obsidian", "/usr/lib/obsidian/obsidian --type=renderer")
        self.add_proc(101, "jobs-worker", "jobs-worker")
        self.add_proc(102, "obs", "/usr/bin/obs")
        self.add_proc(103, "ffmpeg", "ffmpeg -re -i fake.mp4 -f v4l2 /dev/video2")
        self.add_proc(104, "python3", "python3 main.py run DeepFaceLive")
        self.add_proc(105, "ffmpeg", "ffmpeg -i in.mp4 out.mp4")
        hits = " | ".join(h.scan_injection_processes(self.proc))
        self.assertNotIn("PID 100", hits)
        self.assertNotIn("PID 101", hits)
        self.assertIn("PID 102", hits)
        self.assertIn("PID 103", hits)
        self.assertIn("PID 104", hits)
        self.assertNotIn("PID 105", hits)

    def test_module_scan(self):
        _write(os.path.join(self.proc, "modules"),
               "uvcvideo 188416 0 - Live 0x0\nv4l2loopback 49152 0 - Live 0x0\nvideodev 430080 2 - Live 0x0\n")
        self.assertEqual(h.scan_loaded_modules(self.proc), ["v4l2loopback"])


class TestSensorChallengeAnalysis(unittest.TestCase):
    cfg = h.CHALLENGE_CONFIG

    def command(self, seq):
        return [s for s in seq for _ in range(self.cfg["frames_per_slot"])]

    def test_sequence_is_balanced_with_transitions(self):
        rng = random.Random(3)
        for _ in range(50):
            seq = h.make_challenge_sequence(self.cfg["slots"], self.cfg["min_sign_changes"], rng)
            self.assertEqual(sum(seq), 0)
            self.assertGreaterEqual(sum(1 for a, b in zip(seq, seq[1:]) if a != b), self.cfg["min_sign_changes"])

    def test_responsive_sensor_passes_with_lag_and_drift(self):
        rng = np.random.default_rng(0)
        seq = h.make_challenge_sequence(self.cfg["slots"], self.cfg["min_sign_changes"], random.Random(1))
        cmd = np.array(self.command(seq), dtype=float)
        lagged = np.concatenate([[cmd[0]], cmd[:-1]])  # one frame of latency
        luma = 110 + 25 * lagged + np.linspace(0, 12, len(cmd)) + rng.normal(0, 1.0, len(cmd))  # AE drift + noise
        r = h.analyze_challenge_response(luma, cmd)
        self.assertTrue(r["passed"], r)
        self.assertEqual(r["lag"], 1)

    def test_unresponsive_stream_fails(self):
        rng = np.random.default_rng(1)
        fails = 0
        for i in range(200):
            seq = h.make_challenge_sequence(self.cfg["slots"], self.cfg["min_sign_changes"], random.Random(i))
            luma = 120 + np.cumsum(rng.normal(0, 1.5, len(seq) * self.cfg["frames_per_slot"]))  # natural drift, no response
            fails += not h.analyze_challenge_response(luma, self.command(seq))["passed"]
        self.assertEqual(fails, 200)

    def test_brightness_shift_ignores_a_moving_subject(self):
        # A subject covering a third of the frame moves while a +20 offset is applied:
        # the median shift must still report the offset, not the subject.
        rng = np.random.default_rng(3)
        base = rng.uniform(60, 180, (480, 640)).astype(np.float32)
        ref = h._small_gray(base)
        frame = base + 20.0
        frame[100:380, 200:420] = rng.uniform(0, 255, (280, 220))
        self.assertAlmostEqual(h.brightness_shift(frame, ref), 20.0, delta=1.0)

    def test_tiny_response_fails_effect_threshold(self):
        seq = h.make_challenge_sequence(self.cfg["slots"], self.cfg["min_sign_changes"], random.Random(5))
        cmd = np.array(self.command(seq), dtype=float)
        r = h.analyze_challenge_response(120 + 1.0 * cmd, cmd)
        self.assertFalse(r["passed"])


class TestClientPayloads(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(h.evaluate_client_attestation("Clean Physical Device")["verdict"], "PASS")
        self.assertEqual(h.evaluate_client_attestation("Android Emulator (Goldfish/QEMU)")["verdict"], "BLOCK")
        self.assertEqual(h.evaluate_client_attestation("Virtual Camera Injection (OBS / Hooked Driver)")["verdict"], "BLOCK")

    def test_json_payloads(self):
        self.assertEqual(h.evaluate_client_attestation('{"isRooted": true}')["verdict"], "BLOCK")
        self.assertEqual(h.evaluate_client_attestation('{"isVirtualCamera": true}')["verdict"], "FLAG_FOR_REVIEW")
        self.assertEqual(h.evaluate_client_attestation('{"isRooted": false}')["verdict"], "PASS")
        self.assertEqual(h.evaluate_client_attestation('{"isRooted": ')["verdict"], "BLOCK")


if __name__ == "__main__":
    unittest.main()
