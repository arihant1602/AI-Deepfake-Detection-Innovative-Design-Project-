import cv2
import numpy as np

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

def analyze_sensor_noise(frames):
    """
    Heuristic check for camera sensor noise (PRNU placeholder).
    Synthetic/AI media tends to be mathematically over-smoothed compared to real CMOS sensors.
    """
    if not frames or len(frames) < 2:
        return {"passed": False, "prnu_confidence": 0.0, "raw_metric": 0.0}

    noise_levels = []
    for frame in frames[:10]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_levels.append(laplacian_var)

    avg_noise = float(np.mean(noise_levels))
    
    # Real camera feeds generally have natural sensor grain (variance > 15.0)
    is_valid = avg_noise > 15.0
    confidence = min(99.0, max(20.0, avg_noise * 1.8))

    return {
        "passed": is_valid,
        "prnu_confidence": round(confidence, 1),
        "raw_metric": round(avg_noise, 2)
    }

def analyze_temporal_coherence(frames):
    """
    Heuristic check for temporal continuity across sequential frames.
    Measures frame-to-frame variance to detect AI jitter/micro-flickering.
    """
    if len(frames) < 2:
        return {"passed": False, "coherence_score": 0.0, "temporal_jitter": 0.0}

    diffs = []
    for i in range(len(frames) - 1):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        mse = np.mean((prev.astype("float") - curr.astype("float")) ** 2)
        diffs.append(mse)

    std_diff = float(np.std(diffs))
    
    # Large variance indicates unnatural frame flickering
    is_coherent = std_diff < 45.0
    coherence_score = round(max(0.1, 1.0 - (std_diff / 100.0)), 2)

    return {
        "passed": is_coherent,
        "coherence_score": coherence_score,
        "temporal_jitter": round(std_diff, 2)
    }