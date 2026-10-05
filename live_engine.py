"""
Real-time detection engine behind the live page.

A capture thread owns the camera: it streams frames into a ring buffer and runs the
Gate 1b sensor challenge every CHALLENGE_INTERVAL seconds (and immediately when the
fingerprint score drops). An analysis thread re-runs Gate 2 (reference PRNU) and Gate 3
(temporal) once per second on a rolling window and records every threshold crossing as
an event.

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
from camera_sensor_noise_profiling import CameraSensorNoiseProfiler, FingerprintStore
from pipeline import DEFAULT_FINGERPRINT_DIR, MIN_TEMPORAL_FACE_FRAMES
from temporal_consistency_analysis import FaceLocator, TemporalConsistencyAnalyzer

ROOT = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_INTERVAL = 10.0     # seconds between liveness challenges
HEARTBEAT_TIMEOUT = 8.0       # release the camera if the UI stops polling for this long
WINDOW = 45                   # frames per rolling analysis window (1.5 s at 30 fps)
ENROLL_FRAMES = 150
PCE_T = 60.0
TEMPORAL_T = 0.60
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
        self.buffer: deque = deque(maxlen=ENROLL_FRAMES + 60)   # (t, frame, during_challenge, injected)
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
        self.store = FingerprintStore(DEFAULT_FINGERPRINT_DIR)
        self.fingerprint: Optional[np.ndarray] = None

        self.pce_hist: deque = deque(maxlen=120)       # (t, pce)
        self.temporal_hist: deque = deque(maxlen=120)  # (t, score, abstained)
        self.challenges: deque = deque(maxlen=20)      # (t, result)
        self.events: deque = deque(maxlen=40)          # (t, level, text)

        self.injection = "none"
        self._inject_frames: list = []
        self._inject_i = 0
        self.enroll_state: Optional[dict] = None

        self._prnu = CameraSensorNoiseProfiler()
        self._temporal = TemporalConsistencyAnalyzer()
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
        self.fingerprint = self.store.load(self.camera_id, self.width, self.height) if self.camera_id else None
        if self.fingerprint is None:
            self.log("warn", "No enrolled fingerprint for this camera yet: press Enroll")
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

    def request_enroll(self):
        if self.injection != "none":
            self.log("warn", "Turn the simulated attack off before enrolling")
            return
        with self.lock:
            self.enroll_state = {"since": time.time(), "frames": []}
        self.log("info", "Enrolling: keep moving the camera slowly for 5 s")

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
        with self.lock:
            items = [x for x in self.buffer if not x[2] and (allow_injected or not x[3])]
        return [f for (_, f, _, _) in items[-n:]]

    def _analysis_loop(self):
        locator = FaceLocator(max_missed_frames=0)
        while not self.stop_evt.wait(1.0):
            with self.compute_lock:
                self._analyse_once(locator)

    def _analyse_once(self, locator):
        if self.enroll_state is not None:
            self._step_enrollment()
            return
        frames = self._window(WINDOW)
        if len(frames) < 30:
            return
        now = time.time()
        g = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
        self.face_box = locator.locate(g)

        if self.fingerprint is not None:
            r = self._prnu.analyze_frames(frames, ref_fingerprint=self.fingerprint)
            pce = r.components.get("pce_score")
            if pce is not None:
                prev = self.pce_hist[-1][1] if self.pce_hist else None
                with self.lock:
                    self.pce_hist.append((now, pce))
                if pce < PCE_T and (prev is None or prev >= PCE_T):
                    self.log("block", f"Sensor fingerprint lost: score {pce:,.1f} (needs {PCE_T:.0f})")
                    self.request_challenge()
                elif pce >= PCE_T and prev is not None and prev < PCE_T:
                    self.log("ok", f"Sensor fingerprint back: score {pce:,.0f}")

        t = self._temporal.analyze_frames(frames)
        abstained = t.frames_with_face < MIN_TEMPORAL_FACE_FRAMES
        prev_t = self.temporal_hist[-1] if self.temporal_hist else None
        with self.lock:
            self.temporal_hist.append((now, float(t.score), abstained))
        if not abstained and t.flagged and (prev_t is None or prev_t[2] or prev_t[1] < TEMPORAL_T):
            self.log("block", f"Motion anomaly: score {t.score:.2f} (flagged at {TEMPORAL_T:.2f})")

    def _step_enrollment(self):
        st_ = self.enroll_state
        with self.lock:
            new = [f for (t, f, ch, inj) in self.buffer if t > st_["since"] and not ch and not inj]
        if len(new) < ENROLL_FRAMES:
            return
        fp = self._prnu.estimate_fingerprint_from_frames(new[:ENROLL_FRAMES])
        last = self.challenges[0][1]["verdict"] if self.challenges else None
        if last != "PASS":
            self.log("block", "Enrollment refused: the last liveness check did not pass")
        else:
            self.store.save(self.camera_id, fp, {"frames": ENROLL_FRAMES, "node": self.node, "source": "live engine"})
            self.fingerprint = fp
            with self.lock:
                self.pce_hist.clear()
            self.log("ok", f"Enrolled fingerprint for {self.camera_id} at {self.width}x{self.height}")
        self.enroll_state = None

    # ------------------------------------------------------------------ snapshot for the UI

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running, "stop_reason": self.stop_reason, "fps": self.fps,
                "in_challenge": self.in_challenge, "next_challenge_in": max(0.0, self.next_challenge - time.time()),
                "hw": self.hw, "camera_id": self.camera_id, "enrolled": self.fingerprint is not None,
                "pce": list(self.pce_hist), "temporal": list(self.temporal_hist),
                "challenges": list(self.challenges), "events": list(self.events),
                "injection": self.injection, "face_box": self.face_box,
                "enrolling": self.enroll_state is not None,
                "enroll_progress": (sum(1 for (t, _, ch, inj) in self.buffer if self.enroll_state and t > self.enroll_state["since"]
                                        and not ch and not inj) / ENROLL_FRAMES) if self.enroll_state else 0.0,
                "started": self.started,
            }

    def frame(self) -> tuple:
        with self.lock:
            return self.latest, self.latest_t
