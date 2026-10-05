"""
Gated Verification Pipeline & Orchestration Engine (Layer 5)
============================================================
Unifies Gate 1 (Host & Device Integrity Attestation),
        Gate 2 (Camera Sensor Noise Profiling - PRNU),
        Gate 3 (Temporal Consistency & Frequency Analysis)
into a sequential, fail-fast forensic verification flow compliant with CEN/TS 18099.
"""

from __future__ import annotations
import argparse
import json
import os
import tempfile
import cv2
from typing import Any, Dict, List, Optional, Union

# Layer 1: Live Host & Device Attestation
import host_integrity

# Layer 2: Camera Sensor Noise Profiling (Arihant)
try:
    from camera_sensor_noise_profiling import CameraSensorNoiseProfiler
    HAS_LAYER_2 = True
    prnu_profiler = CameraSensorNoiseProfiler()
except ImportError:
    HAS_LAYER_2 = False
    prnu_profiler = None

# Layer 3: Temporal Consistency & Frequency Analysis (Vibha)
try:
    from temporal_consistency_analysis import TemporalConsistencyAnalyzer
    HAS_LAYER_3 = True
    temporal_analyzer = TemporalConsistencyAnalyzer()
except ImportError:
    HAS_LAYER_3 = False
    temporal_analyzer = None


def create_fast_sample_clip(input_path: str, max_frames: int = 45) -> str:
    """
    Extracts the first N frames into a lightweight temporary MP4 file
    so PRNU cross-correlation and temporal FFT analysis complete in ~1-2 seconds.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # If the video is already short, reuse directly
    if 0 < total_frames <= max_frames:
        cap.release()
        return input_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_out.name
    temp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1

    cap.release()
    out.release()
    return temp_path


def extract_frames(video_path: str, max_frames: int = 60) -> List[Any]:
    """Extracts decoded RGB frames for visual pipeline review and gallery display."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


# Gate 1: Hardware Integrity Check (Aarya's Layer 1 Attestation Engine)
def check_hardware_attestation(
    attestation_input: Optional[Union[str, Dict[str, Any]]] = None,
    video_path: Optional[str] = None,
    camera_node: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates hardware and OS integrity per CEN/TS 18099 Gate 1:
    - If attestation_input is None or 'live': executes real-time live host system and video device probes.
    - If attestation_input is 'file_upload' or 'unattested': inspects container format, stream tags, and encoder provenance.
    - If attestation_input is a dict/JSON string: evaluates client attestation payload.
    - If attestation_input is a simulation preset string: evaluates predefined test vector.
    """
    if attestation_input in ("file_upload", "unattested", "Standalone File Upload (No Client Attestation)"):
        if video_path and os.path.exists(video_path):
            return host_integrity.probe_uploaded_file_provenance(video_path)
        return {
            "passed": True,
            "blocked": False,
            "root_detected": False,
            "emulator_detected": False,
            "virtual_camera_detected": False,
            "is_file_upload": True,
            "latency_ms": 0.1,
            "verdict": "UNATTESTED_ORIGIN",
            "details": "Standalone file upload: no live client attestation attached. Deferring to Gate 2/3 forensics.",
        }

    if attestation_input is None or attestation_input in ("live", "live_host", "Real-Time Live Host Probe"):
        return host_integrity.probe_host_integrity(target_camera_node=camera_node)
    return host_integrity.evaluate_client_attestation(attestation_input)


# Gate 2: Camera Sensor Noise Profiling (Arihant's Layer 2 Module)
def check_sensor_noise(video_path: str, ref_fingerprint_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Photo-Response Non-Uniformity (PRNU) noise analysis to detect
    whether a stationary, static physical sensor noise pattern exists across frames.
    Distinguishes genuine CMOS cameras from synthetic/injected streams without requiring pre-enrolled matching.
    """
    if not HAS_LAYER_2 or prnu_profiler is None:
        return {
            "passed": False,
            "static_noise_detected": False,
            "score": 1.0,
            "pce_score": 0.0,
            "persistence": 0.0,
            "explanation": "PRNU Engine unavailable: camera_sensor_noise_profiling module could not be imported.",
        }

    try:
        res = prnu_profiler.analyze(video_path, ref_fingerprint_path=ref_fingerprint_path)
    except Exception as e:
        return {
            "passed": False,
            "static_noise_detected": False,
            "score": 1.0,
            "pce_score": 0.0,
            "persistence": 0.0,
            "explanation": f"PRNU Execution Error: {str(e)}",
        }

    flagged = getattr(res, "flagged", False)
    score = getattr(res, "score", 0.0)
    explanation = getattr(res, "explanation", "PRNU analysis complete.")

    components = getattr(res, "components", {})
    if isinstance(components, dict):
        pce = components.get("pce_score", 0.0)
        persistence = components.get("inter_frame_persistence", 0.0)
        static_noise_detected = components.get("static_noise_detected", not flagged)
    else:
        pce = getattr(components, "pce_score", 0.0)
        persistence = getattr(components, "inter_frame_persistence", 0.0)
        static_noise_detected = getattr(components, "static_noise_detected", not flagged)

    return {
        "passed": not flagged,
        "static_noise_detected": static_noise_detected,
        "score": score,
        "pce_score": pce,
        "persistence": persistence,
        "explanation": explanation,
        "raw_result": res,
    }


# Gate 3: Temporal Consistency & Frequency Analysis (Vibha's Layer 3 Module)
def check_temporal_coherence(video_path: str) -> Dict[str, Any]:
    """
    Executes Farneback optical flow coherence, high-frequency energy ratio,
    Canny edge stability, and 1D temporal FFT power spectral density analysis.
    """
    if not HAS_LAYER_3 or temporal_analyzer is None:
        return {
            "passed": False,
            "score": 1.0,
            "confidence": 0.0,
            "flicker_rate": 0.0,
            "flow_incoherence": 0.0,
            "periodicity_anomaly": 0.0,
            "explanation": "Temporal Engine unavailable: temporal_consistency_analysis module could not be imported.",
        }

    try:
        res = temporal_analyzer.analyze(video_path)
    except Exception as e:
        return {
            "passed": False,
            "score": 1.0,
            "confidence": 0.0,
            "flicker_rate": 0.0,
            "flow_incoherence": 0.0,
            "periodicity_anomaly": 0.0,
            "explanation": f"Temporal Analysis Error: {str(e)}",
        }

    flagged = getattr(res, "flagged", False)
    score = getattr(res, "score", 0.0)
    explanation = getattr(res, "explanation", "Temporal consistency check complete.")
    confidence = getattr(res, "confidence", 1.0)

    components = getattr(res, "components", {})
    if isinstance(components, dict):
        flicker = components.get("flicker_rate", 0.0)
        flow_incoherence = components.get("flow_incoherence", 0.0)
        periodicity = components.get("temporal_periodicity_anomaly", 0.0)
        hf_cv = components.get("hf_energy_coefficient_of_variation", 0.0)
    else:
        flicker = getattr(components, "flicker_rate", 0.0)
        flow_incoherence = getattr(components, "flow_incoherence", 0.0)
        periodicity = getattr(components, "temporal_periodicity_anomaly", 0.0)
        hf_cv = getattr(components, "hf_energy_coefficient_of_variation", 0.0)

    return {
        "passed": not flagged,
        "score": score,
        "confidence": confidence,
        "flicker_rate": flicker,
        "flow_incoherence": flow_incoherence,
        "periodicity_anomaly": periodicity,
        "hf_cv": hf_cv,
        "explanation": explanation,
        "raw_result": res,
    }


# Layer 5: Gated Execution Fusion
def run_detection_pipeline(
    video_path: str,
    attestation_mode: Optional[Union[str, Dict[str, Any]]] = None,
    ref_fingerprint_path: Optional[str] = None,
    fast_sample: bool = True,
    max_sample_frames: int = 45,
    camera_node: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the CEN/TS 18099 Gated Detection Pipeline:
      1. Gate 1: Hardware & Device Attestation (fail-fast termination)
      2. Gate 2: PRNU Sensor Noise Profiling (fail-fast termination)
      3. Gate 3: Temporal & Frequency Consistency (fail-fast termination)
      4. Complete verification verdict & frame buffer extraction
    """
    # Gate 1: Hardware / Environment Attestation
    hw = check_hardware_attestation(attestation_mode, video_path=video_path, camera_node=camera_node)
    if hw["blocked"]:
        return {
            "verdict": "DIGITAL INJECTION DETECTED (ENVIRONMENT COMPROMISED)",
            "gate": 1,
            "details": hw,
        }

    # Fast frame sampling for sub-3s response time
    if fast_sample:
        sample_clip_path = create_fast_sample_clip(video_path, max_frames=max_sample_frames)
    else:
        sample_clip_path = video_path

    try:
        # Gate 2: Sensor Noise Profiling (Arihant)
        prnu = check_sensor_noise(sample_clip_path, ref_fingerprint_path=ref_fingerprint_path)
        if not prnu["passed"]:
            return {
                "verdict": "DIGITAL INJECTION DETECTED",
                "gate": 2,
                "details": prnu,
                "attestation": hw,
            }

        # Gate 3: Temporal & Frequency Coherence (Vibha)
        temporal = check_temporal_coherence(sample_clip_path)
        if not temporal["passed"]:
            return {
                "verdict": "DEEPFAKE DETECTED",
                "gate": 3,
                "details": temporal,
                "attestation": hw,
            }

    finally:
        if sample_clip_path != video_path and os.path.exists(sample_clip_path):
            try:
                os.remove(sample_clip_path)
            except OSError:
                pass

    frames = extract_frames(video_path, max_frames=32)

    return {
        "verdict": "AUTHENTIC LIVE STREAM",
        "gate": None,
        "frames": frames,
        "details": {
            "attestation": hw,
            "prnu": prnu,
            "temporal": temporal,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Gated Deepfake & Injection Attack Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video stream (.mp4, .mov, .avi)")
    parser.add_argument(
        "--attestation-mode",
        default="Clean Physical Device",
        help="Simulated client attestation preset ('live' for live host probe, or preset name)",
    )
    parser.add_argument("--ref-fingerprint", help="Optional path to reference camera PRNU fingerprint (.npy)")
    parser.add_argument("--output", help="Optional path to write JSON forensic report")
    args = parser.parse_args()

    print(f"Running gated forensic pipeline on: {args.video} (Attestation: {args.attestation_mode})")
    result = run_detection_pipeline(
        args.video,
        attestation_mode=args.attestation_mode,
        ref_fingerprint_path=args.ref_fingerprint,
    )

    # Clean serializable dictionary
    serializable_details = {}
    if "details" in result:
        for k, v in result["details"].items():
            if isinstance(v, dict):
                serializable_details[k] = {sk: sv for sk, sv in v.items() if sk != "raw_result"}
            else:
                serializable_details[k] = v

    report = {
        "verdict": result["verdict"],
        "gate": result["gate"],
        "details": serializable_details,
    }

    report_json = json.dumps(report, indent=2, default=str)
    print(report_json)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json)
        print(f"Wrote forensic report to: {args.output}")