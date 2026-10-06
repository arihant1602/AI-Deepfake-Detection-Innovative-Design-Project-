"""
Real-time detection engine behind the live page.

A capture thread owns the camera: it streams frames into a ring buffer and runs the
Gate 1b sensor challenge every CHALLENGE_INTERVAL seconds (and immediately when live sensor
noise disappears). An analysis thread re-runs Gate 2 (live sensor noise) and Gate 3
(GenD face-deepfake probability) once per second on a rolling window and records every
threshold crossing as an event.

The camera is released when stop() is called, when the UI stops sending heartbeats for
HEARTBEAT_TIMEOUT seconds (tab closed, page left), when another engine takes over the
same node, or when the process exits.

For demonstrations, set_injection() replaces the frames *after* the camera driver (what a
hooked capture call does) with a replay or an AI video, while the physical camera keeps
receiving the challenge.
"""

from __future__ import annotations

import atexit
import glob
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

import host_integrity as h
from camera_sensor_noise_profiling import CameraSensorNoiseProfiler
from pipeline import get_deepfake_detector

ROOT = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_INTERVAL = 10.0     # seconds between liveness challenges
HEARTBEAT_TIMEOUT = 8.0       # release the camera if the UI stops polling for this long
WINDOW = 20                   # frames for the sensor-noise test (19 consecutive pairs; 8 needed)
NOISE_T = 0.6                 # live-noise level threshold (check 1 of Gate 2), for the graph
DEEPFAKE_FRAMES = 4           # face crops scored by GenD per tick (judged on a 3-tick median)
INJECTIONS = {
    "none": "Off",
    "replay": "Replay of this camera's last 3 s",
    "ai": "AI-generated video",
    "photo": "Animated photo",
}

_registry: dict = {}
_registry_lock = threading.Lock()


def _shutdown_all():
    with _registry_lock:
        engines = list(_registry.values())
    for e in engines:
        e.stop("server shutting down")


atexit.register(_shutdown_all)


class LiveEngine:
    def __init__(self, node: str, width: int = 640, height: int = 480):
        self.node, self.width, self.height = node, width, height
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.challenge_now = threading.Event()
        # Held by the analysis thread while it computes and by the challenge while it runs, so the
        # capture thread is never starved of CPU during the timing-sensitive liveness check.
        self.compute_lock = threading.Lock()
        self.buffer: deque = deque(maxlen=150)   # (t, frame, during_challenge, injected)
        self.latest: Optional[np.ndarray] = None
        self.latest_t = 0.0
        self.face_box = None
        self.last_seen = time.time()
        self.started = time.time()
        self.running = False
        self.stop_reason: Optional[str] = None
        self.fps = 0.0
        self.in_challenge = False
        self.next_challenge = 0.0

        self.hw: Optional[dict] = None
        self.camera_id: Optional[str] = None
        self.noise_hist: deque = deque(maxlen=120)     # (t, noise_sigma, verdict, failed_checks)
        self.deepfake_hist: deque = deque(maxlen=120)  # (t, P(fake) or None, abstained)
        self.deepfake_threshold: Optional[float] = None
        self.challenges: deque = deque(maxlen=20)      # (t, result)
        self.events: deque = deque(maxlen=40)          # (t, level, text)

        self.injection = "none"
        self._inject_frames: list = []
        self._inject_i = 0

        self._prnu = CameraSensorNoiseProfiler()
        self._threads: list = []

    # ------------------------------------------------------------------ control

    def start(self) -> bool:
        with _registry_lock:
            old = _registry.get(self.node)
            if old is not None and old is not self:
                old.stop("another session took over the camera")
                for t in old._threads:
                    t.join(timeout=3)
            _registry[self.node] = self
        self.hw = h.probe_host_integrity(self.node)
        self.camera_id = self.hw.get("camera_id")
        self.log("info", f"Device check: {self.hw['verdict'].replace('_', ' ').lower()} ({self.hw['latency_ms']:.0f} ms)")
        if self.hw["blocked"]:
            self.log("block", self.hw["details"])
            self.stop_reason = "blocked by the device check; camera never opened"
            return False
        self.running = True
        self._threads = [threading.Thread(target=self._capture_loop, daemon=True, name="capture"),
                         threading.Thread(target=self._analysis_loop, daemon=True, name="analysis")]
        for t in self._threads:
            t.start()
        return True

    def stop(self, reason: str = "stopped"):
        if not self.stop_evt.is_set():
            self.stop_reason = reason
            self.stop_evt.set()

    def heartbeat(self):
        self.last_seen = time.time()

    def request_challenge(self):
        self.challenge_now.set()

    def set_injection(self, mode: str):
        if mode == self.injection:
            return
        frames: list = []
        if mode == "replay":
            with self.lock:
                frames = [f for (_, f, ch, inj) in list(self.buffer)[-90:] if not ch and not inj]
        elif mode == "ai":
            clip = sorted(glob.glob(os.path.join(ROOT, "benchmarks", "data", "fakes", "T2V_grok*")))
            if clip:
                cap = cv2.VideoCapture(clip[0])
                while len(frames) < 120:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frames.append(cv2.resize(f, (self.width, self.height)))
                cap.release()
            if not frames:
                mode = "photo"
        if mode == "photo":
            src = self.latest.copy() if self.latest is not None else np.zeros((self.height, self.width, 3), np.uint8)
            for t in range(120):
                m = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), 3 * np.sin(t / 9), 1 + 0.04 * np.sin(t / 13))
                m[0, 2] += 10 * np.sin(t / 7)
                frames.append(cv2.warpAffine(src, m, (self.width, self.height), borderMode=cv2.BORDER_REFLECT))
        with self.lock:
            self.injection = mode if (mode == "none" or frames) else "none"
            self._inject_frames, self._inject_i = frames, 0
        if self.injection == "none":
            self.log("info", "Simulated attack off: frames come from the camera again")
        else:
            self.log("warn", f"Simulated attack on: {INJECTIONS[self.injection]} replaces the camera frames")

    def log(self, level: str, text: str):
        with self.lock:
            self.events.appendleft((time.time(), level, text))

    # ------------------------------------------------------------------ capture

    def _read(self, cap, during_challenge: bool, with_timestamp: bool = False):
        ok, real = cap.read()
        if not ok:
            return (None, None) if with_timestamp else None
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        with self.lock:
            injected = self.injection != "none" and bool(self._inject_frames)
            if injected:
                frame = self._inject_frames[self._inject_i % len(self._inject_frames)]
                self._inject_i += 1
            else:
                frame = real
            now = time.time()
            self.buffer.append((now, frame, during_challenge, injected))
            self.latest, self.latest_t = frame, now
        return (frame, ts) if with_timestamp else frame

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.node, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                self.log("block", f"Could not open {self.node}")
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            for _ in range(10):
                self._read(cap, False)
            self.next_challenge = time.time()
            n, t0 = 0, time.time()
            while not self.stop_evt.is_set():
                if time.time() - self.last_seen > HEARTBEAT_TIMEOUT:
                    self.stop("page not visible; camera released")
                    break
                if time.time() >= self.next_challenge or self.challenge_now.is_set():
                    self.challenge_now.clear()
                    self._run_challenge(cap)
                    self.next_challenge = time.time() + CHALLENGE_INTERVAL
                    continue
                if self._read(cap, False) is None:
                    self.log("block", "Camera stopped delivering frames")
                    break
                n += 1
                if time.time() - t0 >= 1.0:
                    self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
        finally:
            cap.release()
            self.running = False
            self.log("info", f"Camera released ({self.stop_reason or 'stopped'})")
            with _registry_lock:
                if _registry.get(self.node) is self:
                    del _registry[self.node]

    def _run_challenge(self, cap):
        with self.compute_lock:
            self.in_challenge = True
            try:
                for _ in range(3):  # drop frames queued while waiting for the lock
                    self._read(cap, True)
                res = h.run_sensor_challenge(self.node, lambda: self._read(cap, True, with_timestamp=True),
                                             cfg={"attempts": 2})
            finally:
                self.in_challenge = False
        with self.lock:
            prev = self.challenges[0][1]["verdict"] if self.challenges else None
            self.challenges.appendleft((time.time(), res))
        if res["verdict"] == "FAIL":
            self.log("block", f"Liveness check failed: frames ignored the brightness pattern (r = {res.get('corr', 0):.2f})")
        elif res["verdict"] == "PASS" and prev != "PASS":
            self.log("ok", f"Liveness check passed (r = {res.get('corr', 0):.2f})")
        elif res["verdict"] == "UNSUPPORTED":
            self.log("warn", res["details"])

    # ------------------------------------------------------------------ analysis

    def _window(self, n: int, allow_injected: bool = True) -> list:
        # Consecutive non-challenge frames only: the live-noise test compares neighbours.
        with self.lock:
            items = [x for x in self.buffer if not x[2] and (allow_injected or not x[3])]
        return [f for (_, f, _, _) in items[-n:]]

    def _analysis_loop(self):
        self.log("info", "Loading the deepfake model…")
        det = get_deepfake_detector()
        if isinstance(det, Exception):
            self.log("warn", f"Deepfake model unavailable, Gate 3 off: {det}")
            det = None
        else:
            self.deepfake_threshold = det.cfg["fake_threshold"]
            self.log("info", f"Deepfake model ready (GenD {det.cfg['backbone'].upper()} on {det.device})")
        # Two staggered half-second steps, each holding the compute lock only for its own work,
        # so the capture thread and the liveness check are never starved.
        while not self.stop_evt.wait(0.5):
            with self.compute_lock:
                self._noise_step(det)
            if self.stop_evt.wait(0.5):
                break
            if det is not None:
                with self.compute_lock:
                    self._deepfake_step(det)

    def _noise_step(self, det):
        frames = self._window(WINDOW)
        if len(frames) < WINDOW:
            return
        now = time.time()
        if det is not None:
            found = det.faces.detect(frames[-1])
            self.face_box = tuple(found[0]) if found is not None else None
        r = self._prnu.analyze_frames(frames)
        c = r.components
        prev = self.noise_hist[-1][2] if self.noise_hist else None
        with self.lock:
            self.noise_hist.append((now, c.get("noise_sigma"), r.verdict, c.get("failed_checks") or []))
        if r.verdict == "ABSENT" and prev != "ABSENT":
            self.log("block", r.explanation)
            self.request_challenge()
        elif r.verdict == "PRESENT" and prev == "ABSENT":
            self.log("ok", "Live sensor noise back")

    def _deepfake_step(self, det):
        frames = self._window(WINDOW)
        if len(frames) < WINDOW:
            return
        step = max(1, len(frames) // DEEPFAKE_FRAMES)
        r = det.analyze_frames(frames[::step][:DEEPFAKE_FRAMES], min_faces=2)
        was = self.deepfake_smoothed()
        with self.lock:
            self.deepfake_hist.append((time.time(), r.fake_probability, r.abstained))
        cur = self.deepfake_smoothed()
        thr = self.deepfake_threshold
        if cur is not None and cur >= thr and (was is None or was < thr):
            self.log("block", f"Face looks synthetic: P(fake) {cur:.2f} over the last 3 s (flagged at {thr:.2f})")
        elif cur is not None and cur < thr and was is not None and was >= thr:
            self.log("ok", f"Face looks real again: P(fake) {cur:.2f}")

    def deepfake_smoothed(self) -> Optional[float]:
        """Median P(fake) of the last 3 scored seconds: one noisy window cannot flag a real user."""
        vals = [p for (_, p, abst) in list(self.deepfake_hist)[-3:] if not abst and p is not None]
        return float(np.median(vals)) if len(vals) >= 2 else None

    # ------------------------------------------------------------------ snapshot for the UI

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running, "stop_reason": self.stop_reason, "fps": self.fps,
                "in_challenge": self.in_challenge, "next_challenge_in": max(0.0, self.next_challenge - time.time()),
                "hw": self.hw, "camera_id": self.camera_id,
                "noise": list(self.noise_hist), "deepfake": list(self.deepfake_hist),
                "deepfake_threshold": self.deepfake_threshold,
                "deepfake_smoothed": self.deepfake_smoothed(),
                "challenges": list(self.challenges), "events": list(self.events),
                "injection": self.injection, "face_box": self.face_box,
                "started": self.started,
            }

    def frame(self) -> tuple:
        with self.lock:
            return self.latest, self.latest_t
