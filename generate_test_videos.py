"""
Layer 2: Test Video Generator for Camera Sensor Noise Profiling (PRNU)
=====================================================================

Builds synthetic clips and reference camera fingerprints to evaluate the PRNU detector:

  - real_sim.mp4 : Simulates a genuine CMOS camera sensor capture.
                   A stationary microscopic PRNU pattern K (fixed to the physical
                   sensor grid) modulates scene light across all frames, along with
                   natural physical sensor shot noise.

  - fake_sim.mp4 : Simulates an injected deepfake video stream.
                   Digitally rendered / virtual-camera injection that lacks a physical
                   CMOS sensor's stationary PRNU (uncorrelated frame-by-frame generative
                   noise and compression smoothing).

  - camera_fingerprint.npy : Pre-extracted reference PRNU fingerprint of the genuine camera.
"""

import cv2
import numpy as np

W, H = 320, 320
N_FRAMES = 90
FPS = 30


def make_base_frame(t: int, w: int, h: int) -> np.ndarray:
    """Generates base clean scene irradiance (unmarred by sensor noise) with moving face."""
    frame = np.full((h, w), 90.0, dtype=np.float32)  # Ambient background

    # Face center motion (smooth head movement)
    cx = w // 2 + int(20 * np.sin(t / 15.0))
    cy = h // 2 + int(10 * np.cos(t / 20.0))
    radius = 65
    lighting = 160.0 + 15.0 * np.sin(t / 30.0)

    # Face disk
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    face_mask = dist_sq <= (radius ** 2)
    frame[face_mask] = lighting

    # Eyes
    eye_r_sq = 8 ** 2
    left_eye = ((x - (cx - 22)) ** 2 + (y - (cy - 12)) ** 2) <= eye_r_sq
    right_eye = ((x - (cx + 22)) ** 2 + (y - (cy - 12)) ** 2) <= eye_r_sq
    frame[left_eye | right_eye] = 40.0

    # Mouth
    mouth_mask = (np.abs(x - cx) <= 20) & (np.abs(y - (cy + 25)) <= 5)
    frame[mouth_mask] = 60.0

    return frame


def write_video(path: str, fake: bool, sensor_prnu: np.ndarray):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))

    rng = np.random.default_rng(101 if not fake else 202)

    for t in range(N_FRAMES):
        base_irradiance = make_base_frame(t, W, H)

        if not fake:
            # Genuine capture: light is modulated by physical sensor PRNU K + shot noise
            shot_noise = rng.normal(0.0, 3.0, (H, W)).astype(np.float32)
            frame = base_irradiance * (1.0 + sensor_prnu) + shot_noise
        else:
            # Injected deepfake / digitally synthesized stream:
            # Lacks stationary CMOS PRNU; noise is independent per-frame generative/rendering noise
            gen_noise = rng.normal(0.0, 6.0, (H, W)).astype(np.float32)
            frame = base_irradiance + gen_noise

        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)

    writer.release()


if __name__ == "__main__":
    # Generate stationary physical CMOS PRNU sensor fingerprint
    rng_sensor = np.random.default_rng(42)
    sensor_prnu = rng_sensor.normal(0.0, 0.06, (H, W)).astype(np.float32)
    np.save("camera_fingerprint.npy", sensor_prnu)

    write_video("real_sim.mp4", fake=False, sensor_prnu=sensor_prnu)
    write_video("fake_sim.mp4", fake=True, sensor_prnu=sensor_prnu)
    print("Wrote real_sim.mp4, fake_sim.mp4, and camera_fingerprint.npy")
