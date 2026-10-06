"""
Gated Verification Pipeline & Orchestration Engine (Layer 5)
============================================================
Sequential, fail-fast fusion of
  Gate 1  host & camera attestation (passive probe + active sensor challenge)
  Gate 2  live sensor noise - blind test for fresh, sensor-like noise with no codec
          fingerprint (enforced in live sessions; advisory for files)
  Gate 3  face deepfake detection (GenD, WACV 2026) on aligned face crops

Cheap gates run first and terminate the session as soon as a definitive anomaly is
found, so the expensive pixel analysis is only spent on sessions that survive.
Frames are analysed exactly as captured/decoded; they are never re-encoded (a lossy
re-encode erases the sensor noise Gate 2 measures).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

import host_integrity

try:
    from camera_sensor_noise_profiling import (
        CameraSensorNoiseProfiler, read_video_frames, VERDICT_INCONCLUSIVE, VERDICT_PRESENT,
    )
    HAS_LAYER_2 = True
    prnu_profiler = CameraSensorNoiseProfiler()
except ImportError:
    HAS_LAYER_2 = False
    prnu_profiler = None

try:
    from temporal_consistency_analysis import TemporalConsistencyAnalyzer
    HAS_LAYER_3 = True
    temporal_analyzer = TemporalConsistencyAnalyzer()
except ImportError:
    HAS_LAYER_3 = False
    temporal_analyzer = None

MIN_TEMPORAL_FACE_FRAMES = 8  # legacy temporal heuristic abstains below this many face frames
_deepfake_detector = None

VERDICT_AUTHENTIC = "AUTHENTIC LIVE STREAM"
VERDICT_NO_ANOMALY = "NO ANOMALY DETECTED"
VERDICT_INJECTION_ENV = "DIGITAL INJECTION DETECTED (ENVIRONMENT COMPROMISED)"
VERDICT_INJECTION_CHALLENGE = "DIGITAL INJECTION DETECTED (SENSOR CHALLENGE FAILED)"
VERDICT_INJECTION_PRNU = "DIGITAL INJECTION DETECTED"
VERDICT_DEEPFAKE = "DEEPFAKE DETECTED"


# --------------------------------------------------------------------------- #
# Frame helpers
# --------------------------------------------------------------------------- #

def load_frames(video_path: str, max_frames: int) -> List[np.ndarray]:
    """Decodes the first `max_frames` BGR frames without re-encoding."""
    frames, _ = read_video_frames(video_path, max_frames)
    return frames


def extract_frames(video_path: str, max_frames: int = 60) -> List[Any]:
    """Decoded RGB frames for the UI gallery."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def create_fast_sample_clip(input_path: str, max_frames: int = 45) -> str:
    """
    Writes the first N frames to a temporary MP4 (for previews only).
    NOT used for analysis: the lossy mp4v re-encode destroys sensor noise.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if 0 < total <= max_frames:
        cap.release()
        return input_path
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1
    cap.release()
    out.release()
    return tmp.name


def write_preview_clip(frames: Sequence[np.ndarray], fps: float = 30.0) -> Optional[str]:
    """Browser-playable preview of captured frames (display only, never analysed)."""
    if not frames:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    h, w = frames[0].shape[:2]
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            return tmp.name
    return None


# --------------------------------------------------------------------------- #
# Gate 1
# --------------------------------------------------------------------------- #

def check_hardware_attestation(
    attestation_input: Optional[Union[str, Dict[str, Any]]] = None,
    video_path: Optional[str] = None,
    camera_node: Optional[str] = None,
) -> Dict[str, Any]:
    """
    None / 'live'                      -> passive probe of this host and `camera_node`
    'file_upload' / 'unattested'       -> uploaded file: origin cannot be attested
    dict or JSON string                -> Android SDK attestation payload
    simulation preset name             -> predefined test vector
    """
    if attestation_input in ("file_upload", "unattested", "Standalone File Upload (No Client Attestation)"):
        if video_path and os.path.exists(video_path):
            return host_integrity.probe_uploaded_file_provenance(video_path)
        return {"passed": True, "blocked": False, "verdict": "UNATTESTED_ORIGIN", "root_detected": False,
                "emulator_detected": False, "virtual_camera_detected": False, "is_file_upload": True,
                "latency_ms": 0.0, "details": "Uploaded file: no live device to attest."}
    if attestation_input is None or attestation_input in ("live", "live_host", "Real-Time Live Host Probe"):
        return host_integrity.probe_host_integrity(target_camera_node=camera_node)
    return host_integrity.evaluate_client_attestation(attestation_input)


# --------------------------------------------------------------------------- #
# Gate 2
# --------------------------------------------------------------------------- #

def check_sensor_noise(
    video_path: Optional[str] = None,
    ref_fingerprint_path: Optional[str] = None,
    frames: Optional[Sequence[np.ndarray]] = None,
    ref_fingerprint: Optional[np.ndarray] = None,
    max_frames: int = 90,
    enforce: bool = False,
) -> Dict[str, Any]:
    """
    Runs Layer 2. The default blind live-noise test is enforced when `enforce` is set
    (live sessions): ABSENT terminates the session. For files it is advisory, because any
    encoded video lacks live sensor noise by construction. A reference fingerprint, when
    given, is always enforced.
    """
    if not HAS_LAYER_2 or prnu_profiler is None:
        return {"passed": True, "enforced": False, "verdict": "UNAVAILABLE", "mode": None, "score": None,
                "explanation": "PRNU module could not be imported.", "static_noise_detected": None}

    if ref_fingerprint is None and ref_fingerprint_path:
        try:
            ref_fingerprint = np.load(ref_fingerprint_path).astype(np.float32)
        except (OSError, ValueError) as e:
            return {"passed": True, "enforced": False, "verdict": "UNAVAILABLE", "mode": "reference", "score": None,
                    "explanation": f"Could not load reference fingerprint: {e}", "static_noise_detected": None}

    t0 = time.perf_counter()
    if frames is None:
        try:
            frames = load_frames(video_path, max_frames)
        except IOError as e:
            return {"passed": True, "enforced": False, "verdict": "UNAVAILABLE", "mode": None, "score": None,
                    "explanation": str(e), "static_noise_detected": None}
    res = prnu_profiler.analyze_frames(frames, ref_fingerprint=ref_fingerprint)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    c = res.components
    enforced = res.verdict != VERDICT_INCONCLUSIVE and (res.mode == "reference" or enforce)
    return {
        "passed": not (enforced and res.flagged),
        "enforced": enforced,
        "flagged": res.flagged,
        "verdict": res.verdict,
        "mode": res.mode,
        "static_noise_detected": c.get("static_noise_detected"),
        "score": res.score,
        "confidence": res.confidence,
        "pce_score": c.get("pce_score"),
        "noise_sigma": c.get("noise_sigma"),
        "exact_repeat": c.get("exact_repeat"),
        "spatial_corr": c.get("spatial_corr"),
        "colour_corr": c.get("colour_corr"),
        "failed_checks": c.get("failed_checks"),
        "frames_analyzed": res.frames_analyzed,
        "latency_ms": round(latency_ms, 1),
        "explanation": res.explanation,
        "raw_result": res,
    }


# --------------------------------------------------------------------------- #
# Gate 3
# --------------------------------------------------------------------------- #

def check_temporal_coherence(video_path: Optional[str] = None, frames: Optional[Sequence[np.ndarray]] = None,
                             max_frames: int = 90) -> Dict[str, Any]:
    if not HAS_LAYER_3 or temporal_analyzer is None:
        return {"passed": True, "abstained": True, "score": None, "explanation": "Temporal module could not be imported."}
    t0 = time.perf_counter()
    try:
        if frames is None:
            frames = load_frames(video_path, max_frames)
        res = temporal_analyzer.analyze_frames(frames)
    except Exception as e:  # Layer 3 is third-party code from the pipeline's point of view
        return {"passed": True, "abstained": True, "score": None, "explanation": f"Temporal analysis error: {e}"}
    latency_ms = (time.perf_counter() - t0) * 1000.0
    c = res.components or {}
    return {
        "passed": not res.flagged,
        "abstained": res.frames_with_face < MIN_TEMPORAL_FACE_FRAMES,
        "score": res.score,
        "confidence": res.confidence,
        "frames_analyzed": res.frames_analyzed,
        "frames_with_face": res.frames_with_face,
        "flicker_rate": c.get("flicker_rate", 0.0),
        "flow_incoherence": c.get("flow_incoherence", 0.0),
        "periodicity_anomaly": c.get("temporal_periodicity_anomaly", 0.0),
        "hf_cv": c.get("hf_energy_coefficient_of_variation", 0.0),
        "latency_ms": round(latency_ms, 1),
        "explanation": res.explanation,
        "raw_result": res,
    }


def get_deepfake_detector():
    """GenD detector, loaded once per process (~6 s, ~1.2 GB GPU memory). Returns the error if unavailable."""
    global _deepfake_detector
    if _deepfake_detector is None:
        try:
            import deepfake_detector as dd
            _deepfake_detector = dd.DeepfakeDetector(dd.CONFIG["backbone"])
        except Exception as e:  # torch / weights / GPU missing: Gate 3 abstains instead of crashing
            _deepfake_detector = e
    return _deepfake_detector


def check_deepfake(video_path: Optional[str] = None, frames: Optional[Sequence[np.ndarray]] = None,
                   max_frames: int = 90) -> Dict[str, Any]:
    """Gate 3: GenD face-deepfake probability averaged over aligned face crops."""
    det = get_deepfake_detector()
    if isinstance(det, Exception):
        return {"passed": True, "abstained": True, "score": None, "frames_with_face": 0, "frames_analyzed": 0,
                "model": "GenD (unavailable)", "explanation": f"Deepfake model unavailable: {det}"}
    if frames is None:
        frames = load_frames(video_path, max_frames)
    r = det.analyze_frames(frames)
    return {
        "passed": not r.flagged,
        "abstained": r.abstained,
        "score": r.fake_probability,
        "threshold": det.cfg["fake_threshold"],
        "frames_with_face": r.frames_with_face,
        "frames_analyzed": r.frames_analyzed,
        "frame_probabilities": r.frame_probabilities,
        "model": f"GenD {det.cfg['backbone'].upper()}",
        "latency_ms": r.latency_ms,
        "explanation": r.explanation,
        "raw_result": r,
    }


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

def _fuse(hw: Dict[str, Any], challenge: Optional[Dict[str, Any]], prnu: Dict[str, Any],
          temporal: Dict[str, Any], live: bool) -> Dict[str, Any]:
    """Final verdict for a session that survived every gate."""
    limits: List[str] = []
    if hw.get("verdict") == "UNATTESTED_ORIGIN":
        limits.append("origin unattested (uploaded file: no live device to verify)")
    elif hw.get("verdict") == "FLAG_FOR_REVIEW":
        limits.append(f"Gate 1 flag: {hw.get('details')}")
    if not live:
        limits.append("file input: no live sensor challenge possible, so liveness of the source is not proven")
    elif not challenge or challenge.get("verdict") != "PASS":
        limits.append("active sensor challenge not performed / unsupported by this camera")
    if prnu.get("verdict") != VERDICT_PRESENT:
        if not live and prnu.get("mode") == "live_noise":
            limits.append(f"file input: sensor-noise result is advisory ({prnu.get('verdict')}; any encoded video lacks live noise)")
        else:
            limits.append(f"live sensor noise not confirmed ({prnu.get('verdict')})")
    if temporal.get("abstained"):
        limits.append(f"Gate 3 abstained ({temporal.get('explanation') or 'no face'})")
    verdict = VERDICT_AUTHENTIC if (live and not limits) else VERDICT_NO_ANOMALY
    return {"verdict": verdict, "limitations": limits}


def run_detection_pipeline(
    video_path: str,
    attestation_mode: Optional[Union[str, Dict[str, Any]]] = None,
    ref_fingerprint_path: Optional[str] = None,
    fast_sample: bool = True,
    max_sample_frames: int = 60,
    camera_node: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gated pipeline for a video file (uploaded clip, or a recording with a simulated /
    client-supplied attestation). `fast_sample` limits analysis to the first
    `max_sample_frames` decoded frames (no re-encoding).
    """
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    hw = check_hardware_attestation(attestation_mode, video_path=video_path, camera_node=camera_node)
    timings["gate1_ms"] = hw.get("latency_ms", 0.0)
    if hw["blocked"]:
        return {"verdict": VERDICT_INJECTION_ENV, "gate": 1, "details": hw, "timings": timings}

    try:
        frames = load_frames(video_path, max_sample_frames if fast_sample else 100000)
    except IOError as e:
        return {"verdict": "ERROR", "gate": None, "details": {"explanation": str(e)}, "timings": timings}

    prnu = check_sensor_noise(frames=frames, ref_fingerprint_path=ref_fingerprint_path)
    timings["gate2_ms"] = prnu.get("latency_ms", 0.0)
    if not prnu["passed"]:
        return {"verdict": VERDICT_INJECTION_PRNU, "gate": 2, "details": prnu, "attestation": hw, "timings": timings}

    temporal = check_deepfake(frames=frames)
    timings["gate3_ms"] = temporal.get("latency_ms", 0.0)
    if not temporal["passed"]:
        return {"verdict": VERDICT_DEEPFAKE, "gate": 3, "details": temporal, "attestation": hw, "prnu": prnu,
                "timings": timings}

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    fused = _fuse(hw, None, prnu, temporal, live=False)
    return {
        "verdict": fused["verdict"],
        "limitations": fused["limitations"],
        "gate": None,
        "frames": [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[:32]],
        "details": {"attestation": hw, "prnu": prnu, "temporal": temporal},
        "timings": timings,
    }


def run_live_session(
    camera_node: str,
    n_frames: int = 90,
    width: int = 640,
    height: int = 480,
    fourcc: str = "MJPG",
    attestation_mode: Optional[Union[str, Dict[str, Any]]] = None,
    on_frame: Optional[Callable[[np.ndarray, int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Live KYC-style session on a local camera:
      1. passive Gate 1 probe of `camera_node` (BLOCK => the camera is never opened)
      2. open the node, warm up, run the active sensor challenge on the same stream (FAIL => BLOCK)
      3. capture `n_frames` raw frames
      4. Gate 2 live sensor-noise test on the raw frames (enforced)
      5. Gate 3 on the same raw frames
    """
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    if attestation_mode in (None, "live", "live_host", "Real-Time Live Host Probe"):
        hw = host_integrity.probe_host_integrity(target_camera_node=camera_node)
    else:
        hw = host_integrity.evaluate_client_attestation(attestation_mode)
        hw.setdefault("camera_id", None)
    timings["gate1_passive_ms"] = hw.get("latency_ms", 0.0)
    if hw["blocked"]:
        return {"verdict": VERDICT_INJECTION_ENV, "gate": 1, "details": hw, "timings": timings, "frames_bgr": []}

    cap = host_integrity.capture_attested_frames(camera_node, n_frames, width, height, fourcc, on_frame=on_frame)
    challenge = cap.get("challenge")
    if challenge:
        timings["gate1_challenge_ms"] = challenge.get("latency_ms", 0.0)
    if cap.get("error"):
        return {"verdict": "ERROR", "gate": None, "details": {"explanation": cap["error"]}, "timings": timings,
                "frames_bgr": []}
    frames = cap["frames"]
    if challenge and challenge.get("verdict") == "FAIL":
        return {"verdict": VERDICT_INJECTION_CHALLENGE, "gate": 1, "details": {**hw, "challenge": challenge,
                "details": challenge["details"], "blocked": True, "verdict": "BLOCK"},
                "timings": timings, "frames_bgr": frames}
    if len(frames) < 16:
        return {"verdict": "ERROR", "gate": None, "details": {"explanation": f"Only {len(frames)} frames captured."},
                "timings": timings, "frames_bgr": frames}

    prnu = check_sensor_noise(frames=frames, enforce=True)
    prnu["camera_id"] = hw.get("camera_id")
    timings["gate2_ms"] = prnu.get("latency_ms", 0.0)
    if not prnu["passed"]:
        return {"verdict": VERDICT_INJECTION_PRNU, "gate": 2, "details": prnu, "attestation": {**hw, "challenge": challenge},
                "timings": timings, "frames_bgr": frames}

    temporal = check_deepfake(frames=frames)
    timings["gate3_ms"] = temporal.get("latency_ms", 0.0)
    if not temporal["passed"]:
        return {"verdict": VERDICT_DEEPFAKE, "gate": 3, "details": temporal, "attestation": {**hw, "challenge": challenge},
                "prnu": prnu, "timings": timings, "frames_bgr": frames}

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    fused = _fuse(hw, challenge, prnu, temporal, live=True)
    return {
        "verdict": fused["verdict"],
        "limitations": fused["limitations"],
        "gate": None,
        "frames": [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[:32]],
        "frames_bgr": frames,
        "capture": {k: v for k, v in cap.items() if k not in ("frames", "challenge")},
        "details": {"attestation": {**hw, "challenge": challenge}, "prnu": prnu, "temporal": temporal},
        "timings": timings,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k not in ("raw_result", "frames", "frames_bgr", "all_devices", "devices")}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return clean({k: v for k, v in result.items() if k not in ("frames", "frames_bgr")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gated injection & deepfake detection pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Video file to analyse")
    src.add_argument("--live", metavar="NODE", help="Run a live session on a V4L2 node, e.g. /dev/video0")
    parser.add_argument("--attestation-mode", default="file_upload",
                        help="For --video: 'file_upload' (default), 'live', a preset name, or a JSON payload")
    parser.add_argument("--ref-fingerprint", help="Reference fingerprint (.npy) for --video")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", help="Write the JSON report here")
    args = parser.parse_args()

    if args.live:
        result = run_live_session(args.live, n_frames=args.frames, width=args.width, height=args.height)
    else:
        result = run_detection_pipeline(args.video, attestation_mode=args.attestation_mode,
                                        ref_fingerprint_path=args.ref_fingerprint, max_sample_frames=args.frames)
    report = json.dumps(_serializable(result), indent=2, default=str)
    print(report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
