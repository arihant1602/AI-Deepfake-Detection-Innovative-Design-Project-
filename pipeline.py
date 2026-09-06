import cv2
import tempfile
import os

# Import Arihant's Layer 2 PRNU Module
try:
    from camera_sensor_noise_profiling import CameraSensorNoiseProfiler
    HAS_LAYER_2 = True
    profiler = CameraSensorNoiseProfiler()
except ImportError:
    HAS_LAYER_2 = False

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

# Layer 1: Hardware Integrity Check (Mocked until Aarya integrates)
def check_hardware_attestation():
    return {"passed": True, "details": "Physical device verified; no virtual camera detected."}

# Layer 2: Camera Sensor Noise (Arihant's Module)
# Layer 2: Camera Sensor Noise (Arihant's Module)
def check_sensor_noise(video_path):
    if HAS_LAYER_2:
        res = profiler.analyze(video_path)
        
        # res is a LayerResult object, access via attributes
        flagged = getattr(res, "flagged", False)
        score = getattr(res, "score", 0.0)
        explanation = getattr(res, "explanation", "PRNU analysis complete.")
        
        # Extract nested components if available
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
    else:
        return {
            "passed": True,
            "score": 0.12,
            "pce_score": 85.0,
            "persistence": 0.88,
            "explanation": "Mock PRNU passed."
        }

# Layer 3: Temporal Coherence (Mocked until Vibha integrates)
def check_temporal_coherence(frames):
    return {"passed": True, "coherence_score": 91.8}

# Layer 5: Gated Execution Fusion
def run_detection_pipeline(video_path):
    # Gate 1: Hardware Attestation
    hw = check_hardware_attestation()
    if not hw["passed"]:
        return {"verdict": "DIGITAL INJECTION DETECTED", "gate": 1, "details": hw}

    # Gate 2: Sensor Noise Profiling (Evaluates the raw video file)
    prnu = check_sensor_noise(video_path)
    if not prnu["passed"]:
        return {"verdict": "DIGITAL INJECTION DETECTED", "gate": 2, "details": prnu}

    # Frame Extraction (Only executed if Gate 1 & 2 pass to save compute)
    frames = extract_frames(video_path)
    if not frames:
        return {"verdict": "ERROR", "gate": 0, "details": "Unable to extract frames."}

    # Gate 3: Temporal and Frequency Coherence
    temporal = check_temporal_coherence(frames)
    if not temporal["passed"]:
        return {"verdict": "DEEPFAKE DETECTED", "gate": 3, "details": temporal}

    return {
        "verdict": "AUTHENTIC",
        "gate": None,
        "frames": frames,
        "details": {"attestation": hw, "prnu": prnu, "temporal": temporal}
    }