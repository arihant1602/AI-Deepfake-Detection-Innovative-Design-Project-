"""
Generates two short synthetic clips so the analyzer can be sanity-checked
without needing real "genuine" vs "deepfake" footage on hand:

  - real_sim.mp4  : a face-like blob that moves smoothly, with slowly
                    drifting lighting (mimics a natural camera capture)
  - fake_sim.mp4  : the same base motion, but with per-frame high-frequency
                    noise injected + occasional lighting/texture jumps
                    (mimics frame-by-frame generative synthesis artifacts)

This is only a synthetic smoke test standing in for real calibration
footage - it exercises the pipeline end-to-end, it does not validate
real-world accuracy.
"""

import cv2
import numpy as np

W, H = 320, 320
N_FRAMES = 90
FPS = 30


def make_base_frame(t: int, w: int, h: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cx = w // 2 + int(20 * np.sin(t / 15.0))
    cy = h // 2 + int(10 * np.cos(t / 20.0))
    lighting = 150 + int(20 * np.sin(t / 40.0))  # slow natural lighting drift

    # face-like blob
    cv2.circle(frame, (cx, cy), 70, (lighting, lighting, lighting), -1)
    # eyes
    cv2.circle(frame, (cx - 25, cy - 15), 8, (30, 30, 30), -1)
    cv2.circle(frame, (cx + 25, cy - 15), 8, (30, 30, 30), -1)
    # mouth
    cv2.ellipse(frame, (cx, cy + 30), (25, 10), 0, 0, 180, (60, 60, 60), 3)
    return frame


def write_video(path: str, fake: bool):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    rng = np.random.default_rng(42 if not fake else 7)

    for t in range(N_FRAMES):
        frame = make_base_frame(t, W, H)

        if fake:
            # inject per-frame high-frequency noise (uncorrelated across frames)
            noise = rng.normal(0, 18, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # occasional abrupt lighting/texture jump every ~10 frames
            if t % 10 == 0:
                frame = np.clip(frame.astype(np.int16) + 35, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    write_video("real_sim.mp4", fake=False)
    write_video("fake_sim.mp4", fake=True)
    print("Wrote real_sim.mp4 and fake_sim.mp4")
