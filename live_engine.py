"""
Real-time detection engine behind the live page.

A capture thread owns the camera: it streams frames into a ring buffer and runs the
Gate 1b multi-control sensor challenge every CHALLENGE_INTERVAL seconds (and immediately when
live sensor noise disappears or the sensor pattern changes). An analysis thread re-runs Gate 2
(live sensor noise) and Gate 3 (GenD face-deepfake probability) once per second on a rolling
window, and every SAME_SENSOR_INTERVAL seconds matches the PRNU of the current frames against
the frames attested by the last passed challenge (Gate 2 same-sensor check). Every threshold
crossing is recorded as an event.

On request, request_screen_flash() schedules a random two-colour sequence that the browser
shows full-screen; the light the screen throws on the user's face must follow it (screen-flash
reflection check, part of Gate 1b). A replay held up to the camera, or an injected video, shows
a face that was never lit by this screen.

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
import secrets
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

import host_integrity as h
from camera_sensor_noise_profiling import CameraSensorNoiseProfiler, same_sensor_check
from pipeline import get_deepfake_detector

ROOT = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_INTERVAL = 10.0     # seconds between liveness challenges
MIN_CHALLENGE_GAP = 5.0       # seconds after a challenge before another can be requested
HEARTBEAT_TIMEOUT = 8.0       # release the camera if the UI stops polling for this long
WINDOW = 20                   # frames for the sensor-noise test (19 consecutive pairs; 8 needed)
NOISE_T = 0.6                 # live-noise level threshold (check 1 of Gate 2), for the graph
SAME_SENSOR_INTERVAL = 3.0    # seconds between same-sensor (PRNU split-and-compare) checks
SAME_SENSOR_FRAMES = 30       # current frames compared against the attested challenge frames
DEEPFAKE_FRAMES = 4           # face crops scored by GenD per tick (judged on a 3-tick median)
# Screen flash: slots of 0.4 s keep colour changes at or below 2.5 per second (WCAG 2.3.1 allows
# at most 3 flashes per second); no saturated red. Pairs are (RGB +, RGB -).
FLASH_SLOTS = 12            # 12 independent slots: a random face passes r >= 0.7 by chance < 1 % of the time
FLASH_SLOT_S = 0.4
FLASH_LEAD_S = 0.8            # time for the browser to receive the plan before the first slot
FLASH_PAIRS = [((255, 208, 0), (32, 64, 255)), ((0, 224, 255), (255, 64, 255))]
FLASH_MAX_LAG_S = 0.35        # display + exposure + transfer latency searched
FLASH_MIN_CORR = 0.7
FLASH_MIN_EFFECT = 1.0        # grey levels along the colour axis between + and - slots, on the face
INJECTIONS = {
    "none": "Off",
    "replay": "Replay of this camera's last 3 s",
    "ai": "AI-generated video",
    "photo": "Animated photo",
}

_registry: dict = {}
_registry_lock = threading.Lock()


def get_engine(node: str) -> Optional["LiveEngine"]:
    """The engine currently holding `node`, if any (used by the MJPEG route in server.py)."""
    with _registry_lock:
        return _registry.get(node)


def _shutdown_all():
    with _registry_lock:
        engines = list(_registry.values())
    for e in engines:
        e.stop("server shutting down")


atexit.register(_shutdown_all)


def analyze_screen_flash(items: list, plan: dict, face_box) -> dict:
    """
    Did the light from the screen show up on the face? `items` are (read time, frame) pairs
    around the flash, `plan` the schedule shown by the browser, `face_box` (x, y, w, h).
    The mean face colour is projected on the axis between the two screen colours and correlated
    with the shown sequence (controlling for a linear trend, i.e. auto-exposure drift) over
    0..FLASH_MAX_LAG_S of latency. PASS, NO_REFLECTION, or INCONCLUSIVE (no face, too few frames).
    """
    out = {"verdict": "INCONCLUSIVE", "corr": None, "effect": None, "lag_ms": None, "frames": len(items),
           "colours": plan["colours"]}
    if face_box is None:
        out["details"] = "Screen flash inconclusive: no face in view to reflect the screen."
        return out
    if len(items) < 3 * FLASH_SLOTS:
        out["details"] = f"Screen flash inconclusive: only {len(items)} frames during the flash."
        return out
    x, y, w, hh = (int(v) for v in face_box)
    plus, minus = (np.array(c[::-1], np.float64) for c in plan["rgb"])  # BGR
    axis = (plus - minus) / np.linalg.norm(plus - minus)
    ts = np.array([t for t, _ in items])
    vals = np.array([float(f[max(0, y):y + hh, max(0, x):x + w].reshape(-1, 3).mean(axis=0) @ axis) for _, f in items])
    seq, slot = np.array(plan["seq"], np.float64), plan["slot_s"]
    best = {"corr": -1.0, "effect": 0.0, "lag": 0.0}
    for lag in np.arange(0.0, FLASH_MAX_LAG_S + 1e-9, 0.033):
        k = np.floor((ts - lag - plan["t0"]) / slot).astype(int)
        ok = (k >= 0) & (k < len(seq))
        if ok.sum() < 2 * FLASH_SLOTS:
            continue
        v, c = vals[ok], seq[k[ok]]
        z = np.column_stack([np.ones(len(v)), ts[ok] - ts[ok][0]])
        proj = z @ np.linalg.pinv(z)
        rv, rc = v - proj @ v, c - proj @ c
        if np.linalg.norm(rv) < 1e-9 or np.linalg.norm(rc) < 1e-9:
            continue
        r = float(rv @ rc / (np.linalg.norm(rv) * np.linalg.norm(rc)))
        if r > best["corr"]:
            best = {"corr": r, "effect": 2.0 * float(rv @ rc / (rc @ rc)), "lag": float(lag)}
    passed = best["corr"] >= FLASH_MIN_CORR and best["effect"] >= FLASH_MIN_EFFECT
    out.update(verdict="PASS" if passed else "NO_REFLECTION", corr=round(best["corr"], 3),
               effect=round(best["effect"], 2), lag_ms=round(best["lag"] * 1000))
    out["details"] = (f"Screen light reflected on the face (r = {best['corr']:.2f}, {best['effect']:.1f} grey levels)."
                      if passed else
                      f"No screen reflection on the face (r = {best['corr']:.2f}, {best['effect']:.1f} grey levels): "
                      "a replayed or injected face, or a room too bright for the screen to light it.")
    return out


class LiveEngine:
    def __init__(self, node: str, width: int = 640, height: int = 480):
        self.node, self.width, self.height = node, width, height
        # Capability for the MJPEG route: only the browser that started the engine knows it.
        self.stream_token = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.challenge_now = threading.Event()
        # Held by the analysis thread while it computes and by the challenge while it runs, so the
        # capture thread is never starved of CPU during the timing-sensitive liveness check.
        self.compute_lock = threading.Lock()
        self.buffer: deque = deque(maxlen=200)   # (t, frame, during_challenge, injected); holds a whole screen flash at 30 fps
        self.latest: Optional[np.ndarray] = None
        self.latest_t = 0.0
        self.face_box = None
        self.last_seen = time.time()
        self.started = time.time()
        self.running = False
        self.stop_reason: Optional[str] = None
        self.fps = 0.0
        self._frames_read = 0
        self.in_challenge = False
        self.next_challenge = 0.0
        self._last_challenge_end = 0.0

        self.hw: Optional[dict] = None
        self.camera_id: Optional[str] = None
        self.noise_hist: deque = deque(maxlen=120)     # (t, noise_sigma, verdict, failed_checks)
        self.deepfake_hist: deque = deque(maxlen=120)  # (t, P(fake) or None, abstained)
        self.deepfake_threshold: Optional[float] = None
        self.challenges: deque = deque(maxlen=20)      # (t, result)
        self.same_sensor: deque = deque(maxlen=60)     # (t, result of same_sensor_check)
        self._attested_fp: Optional[np.ndarray] = None  # PRNU of the frames read in the last passed challenge
        self._attested_n = 0
        self._next_same_sensor = 0.0
        self.flash_plan: Optional[dict] = None
        self.screen_flashes: deque = deque(maxlen=10)  # (t, result)
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

    def request_challenge(self, manual: bool = False):
        # Automatic requests are ignored right after a challenge: the noise and same-sensor checks
        # both ask for one when an attack starts, and back-to-back challenges would starve the
        # analysis thread. A press of the button always counts.
        if manual or (not self.in_challenge and time.time() - self._last_challenge_end >= MIN_CHALLENGE_GAP):
            self.challenge_now.set()

    def request_screen_flash(self) -> Optional[dict]:
        """Schedules a screen-flash sequence and returns the plan the browser renders (epoch seconds)."""
        if not self.running or self.flash_plan is not None:
            return None
        rng = secrets.SystemRandom()
        plus, minus = FLASH_PAIRS[rng.randrange(len(FLASH_PAIRS))]
        t0 = time.time() + FLASH_LEAD_S
        plan = {"id": secrets.token_hex(4), "t0": t0, "slot_s": FLASH_SLOT_S,
                "seq": h.make_challenge_sequence(FLASH_SLOTS, 5, rng),
                "colours": ["#%02x%02x%02x" % plus, "#%02x%02x%02x" % minus], "rgb": [plus, minus]}
        with self.lock:
            self.flash_plan = plan
            # Keep the camera challenge out of the way: its control changes would mask the reflection.
            self.next_challenge = max(self.next_challenge, t0 + FLASH_SLOTS * FLASH_SLOT_S + 1.5)
        self.log("info", "Screen flash: watching the light from the screen on the face")
        return {k: v for k, v in plan.items() if k != "rgb"}

    def _screen_flash_step(self):
        plan = self.flash_plan
        end = plan["t0"] + FLASH_SLOTS * plan["slot_s"]
        if time.time() < end + FLASH_MAX_LAG_S + 0.2:
            return
        with self.lock:
            items = [(t, f) for (t, f, ch, _) in self.buffer if plan["t0"] - 0.1 <= t <= end + FLASH_MAX_LAG_S + 0.1]
            box = self.face_box
            self.flash_plan = None
        res = analyze_screen_flash(items, plan, box)
        with self.lock:
            self.screen_flashes.appendleft((time.time(), res))
        level = {"PASS": "ok", "NO_REFLECTION": "warn"}.get(res["verdict"], "info")
        self.log(level, res["details"])

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
            self._frames_read += 1
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
            n0, t0 = self._frames_read, time.time()
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
                if time.time() - t0 >= 1.0:  # every frame read, challenge frames included
                    self.fps = (self._frames_read - n0) / (time.time() - t0)
                    n0, t0 = self._frames_read, time.time()
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
                attested: list = []
                res = h.run_sensor_challenge(self.node, lambda: self._read(cap, True, with_timestamp=True),
                                             cfg={"attempts": 2, "on_challenge_frame": attested.append})
                if res["verdict"] == "PASS" and len(attested) >= 20:
                    # These frames followed the random control commands, so they are the sensor's own:
                    # their PRNU is the reference the frames between challenges must match.
                    self._attested_fp = self._prnu.estimate_fingerprint_from_frames(attested)
                    self._attested_n = len(attested)
                    self._next_same_sensor = 0.0
            finally:
                self.in_challenge = False
                self._last_challenge_end = time.time()
        with self.lock:
            prev = self.challenges[0][1]["verdict"] if self.challenges else None
            self.challenges.appendleft((time.time(), res))
        if res["verdict"] == "FAIL":
            missed = [n for n, c in (res.get("channels") or {}).items() if not c["passed"]]
            self.log("block", f"Liveness check failed: frames ignored the {', '.join(missed) or 'control'} pattern "
                              f"(lowest r = {res.get('corr', 0):.2f})")
        elif res["verdict"] == "PASS" and prev != "PASS":
            self.log("ok", f"Liveness check passed: {res['control']} (lowest r = {res.get('corr', 0):.2f})")
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
                if time.time() >= self._next_same_sensor:
                    self._same_sensor_step()
                if self.flash_plan is not None:
                    self._screen_flash_step()
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

    def _same_sensor_step(self):
        if self._attested_fp is None:
            return
        frames = self._window(SAME_SENSOR_FRAMES)
        if len(frames) < SAME_SENSOR_FRAMES:
            return
        self._next_same_sensor = time.time() + SAME_SENSOR_INTERVAL
        r = same_sensor_check([], frames, reference_fingerprint=self._attested_fp)
        r["reference_frames"] = self._attested_n
        prev = self.same_sensor[-1][1]["verdict"] if self.same_sensor else None
        with self.lock:
            self.same_sensor.append((time.time(), r))
        if r["verdict"] == "MISMATCH" and prev != "MISMATCH":
            self.log("block", r["explanation"])
            self.request_challenge()
        elif r["verdict"] == "MATCH" and prev == "MISMATCH":
            self.log("ok", f"Same sensor as the liveness check again (PCE {r['pce']:.0f})")

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
                "same_sensor": list(self.same_sensor),
                "screen_flashes": list(self.screen_flashes),
                "flash_plan": {k: v for k, v in self.flash_plan.items() if k != "rgb"} if self.flash_plan else None,
                "injection": self.injection, "face_box": self.face_box,
                "started": self.started,
            }

    def frame(self) -> tuple:
        with self.lock:
            return self.latest, self.latest_t

    def annotated(self, frame: np.ndarray) -> np.ndarray:
        """`frame` with the face box, liveness-check border, attack banner and frame rate drawn on."""
        img = frame.copy()
        hgt, wid = img.shape[:2]
        if self.face_box is not None:
            x, y, w, hh = self.face_box
            cv2.rectangle(img, (x, y), (x + w, y + hh), (214, 120, 42), 2)
        if self.in_challenge:
            cv2.rectangle(img, (2, 2), (wid - 3, hgt - 3), (52, 104, 235), 6)
            cv2.putText(img, "LIVENESS CHECK", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 104, 235), 2, cv2.LINE_AA)
        if self.injection != "none":
            cv2.rectangle(img, (0, hgt - 40), (wid, hgt), (40, 40, 200), -1)
            cv2.putText(img, f"SIMULATED ATTACK: {INJECTIONS[self.injection]}", (14, hgt - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{self.fps:.0f} fps", (wid - 86, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2,
                    cv2.LINE_AA)
        return img
