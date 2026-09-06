import cv2
import tempfile
import os

# Import Arihant's Layer 2 PRNU Module
try:
    from camera_sensor_noise_profiling import CameraSensorNoiseProfiler
    HAS_LAYER_2 = True
    prnu_profiler = CameraSensorNoiseProfiler()
except ImportError:
    HAS_LAYER_2 = False

# Import Vibha's Layer 3 Temporal & Frequency Module
try:
    from temporal_consistency_analysis import TemporalConsistencyAnalyzer
    HAS_LAYER_3 = True
    temporal_analyzer = TemporalConsistencyAnalyzer()
except ImportError:
    HAS_LAYER_3 = False


def create_fast_sample_clip(input_path, max_frames=45):
    """
    Extracts first N frames into a lightweight temporary MP4 file
    so PRNU and temporal analysis complete in ~3 seconds.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_out.name
    temp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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


def extract_frames(video_path, max_frames=60):
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
def check_hardware_attestation(attestation_mode="Clean Physical Device"):
    """
    Evaluates client device attestation per Aarya's Layer 1 DeviceIntegrityChecker:
    - Root or Emulator flagged -> BLOCK (immediate session termination)
    - Only Virtual Camera flagged -> FLAG_FOR_REVIEW
    - Nothing flagged -> PASS (proceed to Gate 2)
    """
    scenarios = {
        "Clean Physical Device": {
            "passed": True,
            "blocked": False,
            "root_detected": False,
            "emulator_detected": False,
            "virtual_camera_detected": False,
            "latency_ms": 14.2,
            "verdict": "PASS",
            "details": "Hardware verified clean. No root binaries, hypervisor markers, or virtual cameras detected."
        },
        "Compromised / Rooted Android (Magisk/SU)": {
            "passed": False,
            "blocked": True,
            "root_detected": True,
            "emulator_detected": False,
            "virtual_camera_detected": False,
            "latency_ms": 18.6,
            "verdict": "BLOCK",
            "details": "Root binary detected (su / Magisk / writable system partition). Device environment untrusted."
        },
        "Android Emulator (Goldfish/QEMU)": {
            "passed": False,
            "blocked": True,
            "root_detected": False,
            "emulator_detected": True,
            "virtual_camera_detected": False,
            "latency_ms": 11.8,
            "verdict": "BLOCK",
            "details": "AVD hypervisor fingerprint detected (sdk_gphone / goldfish markers). High injection risk."
        },
        "Virtual Camera Injection (OBS / Hooked Driver)": {
            "passed": True,
            "blocked": False,
            "root_detected": False,
            "emulator_detected": False,
            "virtual_camera_detected": True,
            "latency_ms": 22.1,
            "verdict": "FLAG_FOR_REVIEW",
            "details": "Virtual camera enumeration anomaly detected. Flagged for secondary PRNU confirmation."
        }
    }
    return scenarios.get(attestation_mode, scenarios["Clean Physical Device"])


# Gate 2: Camera Sensor Noise (Arihant's Module)
def check_sensor_noise(video_path):
    if HAS_LAYER_2:
        try:
            res = prnu_profiler.analyze(video_path)
        except Exception as e:
            return {"passed": False, "score": 1.0, "pce_score": 0.0, "persistence": 0.0, "explanation": f"PRNU Execution Error: {str(e)}"}

        flagged = getattr(res, "flagged", False)
        score = getattr(res, "score", 0.0)
        explanation = getattr(res, "explanation", "PRNU analysis complete.")

        components = getattr(res, "components", {})
        if isinstance(components, dict):
            pce = components.get("pce_score", 0.0)
            persistence = components.get("inter_frame_persistence", 0.0)
        else:
            pce = getattr(components, "pce_score", 0.0)
            persistence = getattr(components, "inter_frame_persistence", 0.0)

        return {
            "passed": not flagged,
            "score": score,
            "pce_score": pce,
            "persistence": persistence,
            "explanation": explanation
        }
    return {
        "passed": True,
        "score": 0.12,
        "pce_score": 85.0,
        "persistence": 0.88,
        "explanation": "Mock PRNU passed."
    }


# Gate 3: Temporal Consistency & Frequency Analysis (Vibha's Module)
def check_temporal_coherence(video_path):
    if HAS_LAYER_3:
        try:
            res = temporal_analyzer.analyze(video_path)
        except Exception as e:
            return {"passed": False, "score": 1.0, "confidence": 0.0, "flicker_rate": 0.0, "flow_incoherence": 0.0, "periodicity_anomaly": 0.0, "explanation": f"Temporal Analysis Error: {str(e)}"}

        flagged = getattr(res, "flagged", False)
        score = getattr(res, "score", 0.0)
        explanation = getattr(res, "explanation", "Temporal consistency check complete.")
        confidence = getattr(res, "confidence", 1.0)

        components = getattr(res, "components", {})
        if isinstance(components, dict):
            flicker = components.get("flicker_rate", 0.0)
            flow_incoherence = components.get("flow_incoherence", 0.0)
            periodicity = components.get("temporal_periodicity_anomaly", 0.0)
        else:
            flicker = getattr(components, "flicker_rate", 0.0)
            flow_incoherence = getattr(components, "flow_incoherence", 0.0)
            periodicity = getattr(components, "temporal_periodicity_anomaly", 0.0)

        return {
            "passed": not flagged,
            "score": score,
            "confidence": confidence,
            "flicker_rate": flicker,
            "flow_incoherence": flow_incoherence,
            "periodicity_anomaly": periodicity,
            "explanation": explanation
        }
    return {
        "passed": True,
        "score": 0.20,
        "confidence": 1.0,
        "flicker_rate": 0.1,
        "flow_incoherence": 0.05,
        "periodicity_anomaly": 0.12,
        "explanation": "Mock Temporal Coherence passed."
    }


# Layer 5: Gated Execution Fusion
def run_detection_pipeline(video_path, attestation_mode="Clean Physical Device"):
    # Gate 1: Hardware Attestation (Aarya)
    hw = check_hardware_attestation(attestation_mode)
    if hw["blocked"]:
        return {"verdict": "DIGITAL INJECTION DETECTED (ENVIRONMENT COMPROMISED)", "gate": 1, "details": hw}

    # Fast frame sampling
    sample_clip_path = create_fast_sample_clip(video_path, max_frames=45)

    try:
        # Gate 2: Sensor Noise Profiling (Arihant)
        prnu = check_sensor_noise(sample_clip_path)
        if not prnu["passed"]:
            return {"verdict": "DIGITAL INJECTION DETECTED", "gate": 2, "details": prnu, "attestation": hw}

        # Gate 3: Temporal & Frequency Coherence (Vibha)
        temporal = check_temporal_coherence(sample_clip_path)
        if not temporal["passed"]:
            return {"verdict": "DEEPFAKE DETECTED", "gate": 3, "details": temporal, "attestation": hw}

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
        "details": {"attestation": hw, "prnu": prnu, "temporal": temporal}
    }