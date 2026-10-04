"""
<<<<<<< HEAD
Test Video & Fingerprint Generator
====================================
Unified generator supporting synthetic clips and camera sensor fingerprints for:
  - Layer 2: Camera Sensor Noise Profiling (PRNU Analysis)
  - Layer 3: Temporal Consistency & Frequency Analysis
  - End-to-End Gated Verification Pipeline

Modes:
  - 'combined' (default): Generates clips testing both Layer 2 (stationary CMOS PRNU)
    and Layer 3 (temporal consistency and optical flow stability) simultaneously.
  - 'prnu': Layer 2 specific synthetic clips (stationary CMOS PRNU vs uncorrelated noise).
  - 'temporal': Layer 3 specific synthetic clips (smooth motion vs generative high-freq noise & jumps).
  - 'all': Generates combined, PRNU-specific, and temporal-specific clips.
"""

from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

W, H = 320, 240
FPS = 30
N_FRAMES = 90


def make_base_frame(t: int, w: int = W, h: int = H) -> np.ndarray:
    """
    Generates base scene irradiance with a smoothly moving face-like target
    and natural ambient illumination drift over time.
    """
    frame = np.full((h, w, 3), 90, dtype=np.uint8)
    cx = w // 2 + int(20 * np.sin(t / 15.0))
    cy = h // 2 + int(10 * np.cos(t / 20.0))
    lighting = int(160 + 15 * np.sin(t / 30.0))

    # Face disk
    cv2.circle(frame, (cx, cy), 65, (lighting, lighting, lighting), -1)
    # Eyes
    cv2.circle(frame, (cx - 22, cy - 12), 8, (40, 40, 40), -1)
    cv2.circle(frame, (cx + 22, cy - 12), 8, (40, 40, 40), -1)
    # Mouth
    cv2.ellipse(frame, (cx, cy + 25), (20, 5), 0, 0, 180, (60, 60, 60), 3)
    return frame


def generate_prnu_fingerprint(w: int = W, h: int = H, seed: int = 42) -> np.ndarray:
    """Generates a stationary 2D CMOS sensor Photo-Response Non-Uniformity matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.06, (h, w)).astype(np.float32)


def write_clip(
    path: str,
    fake: bool,
    mode: str = "combined",
    sensor_prnu: np.ndarray | None = None,
    n_frames: int = N_FRAMES,
    w: int = W,
    h: int = H,
    fps: int = FPS,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    if sensor_prnu is None and not fake and mode in ("combined", "prnu"):
        sensor_prnu = generate_prnu_fingerprint(w, h)

    rng = np.random.default_rng(101 if not fake else 202)

    for t in range(n_frames):
        base = make_base_frame(t, w, h).astype(np.float32)

        if mode == "prnu":
            # Pure PRNU mode (grayscale scene modulated by 2D sensor PRNU)
            gray_base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if not fake:
                shot_noise = rng.normal(0.0, 3.0, (h, w)).astype(np.float32)
                frame_2d = gray_base * (1.0 + sensor_prnu) + shot_noise
            else:
                gen_noise = rng.normal(0.0, 6.0, (h, w)).astype(np.float32)
                frame_2d = gray_base + gen_noise
            frame = cv2.cvtColor(np.clip(frame_2d, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        elif mode == "temporal":
            # Pure Temporal mode (RGB scene with high-frequency noise & lighting jumps in fake)
            if fake:
                noise = rng.normal(0, 18, base.shape).astype(np.float32)
                base = base + noise
                if t % 10 == 0:
                    base = base + 35.0
            frame = np.clip(base, 0, 255).astype(np.uint8)

        else:
            # Combined mode: genuine has stationary PRNU + shot noise;
            # fake has uncorrelated generative noise + temporal lighting jumps + no PRNU
            if not fake:
                prnu_3d = np.repeat(sensor_prnu[:, :, np.newaxis], 3, axis=2)
                shot_noise = rng.normal(0.0, 3.0, (h, w, 3)).astype(np.float32)
                frame = np.clip(base * (1.0 + prnu_3d) + shot_noise, 0, 255).astype(np.uint8)
            else:
                gen_noise = rng.normal(0.0, 16.0, (h, w, 3)).astype(np.float32)
                frame_data = base + gen_noise
                if t % 10 == 0:
                    frame_data += 30.0
                frame = np.clip(frame_data, 0, 255).astype(np.uint8)
=======
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
>>>>>>> temporal-freq-analysis

        writer.write(frame)

    writer.release()
<<<<<<< HEAD
    print(f"Generated: {path} ({n_frames} frames, mode={mode}, fake={fake})")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation videos for deepfake detection.")
    parser.add_argument(
        "--mode",
        choices=["combined", "prnu", "temporal", "all"],
        default="combined",
        help="Generation mode (default: combined)",
    )
    parser.add_argument("--frames", type=int, default=N_FRAMES, help="Number of frames per clip")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sensor_prnu = generate_prnu_fingerprint(W, H)

    fp_path = os.path.join(args.out_dir, "camera_fingerprint.npy")
    np.save(fp_path, sensor_prnu)
    print(f"Saved reference fingerprint: {fp_path}")

    if args.mode in ("combined", "all"):
        real_p = os.path.join(args.out_dir, "real_sim.mp4")
        fake_p = os.path.join(args.out_dir, "fake_sim.mp4")
        write_clip(real_p, fake=False, mode="combined", sensor_prnu=sensor_prnu, n_frames=args.frames)
        write_clip(fake_p, fake=True, mode="combined", sensor_prnu=sensor_prnu, n_frames=args.frames)

    if args.mode in ("prnu", "all"):
        real_prnu = os.path.join(args.out_dir, "real_prnu.mp4")
        fake_prnu = os.path.join(args.out_dir, "fake_prnu.mp4")
        write_clip(real_prnu, fake=False, mode="prnu", sensor_prnu=sensor_prnu, n_frames=args.frames)
        write_clip(fake_prnu, fake=True, mode="prnu", sensor_prnu=sensor_prnu, n_frames=args.frames)

    if args.mode in ("temporal", "all"):
        real_temp = os.path.join(args.out_dir, "real_temporal.mp4")
        fake_temp = os.path.join(args.out_dir, "fake_temporal.mp4")
        write_clip(real_temp, fake=False, mode="temporal", sensor_prnu=sensor_prnu, n_frames=args.frames)
        write_clip(fake_temp, fake=True, mode="temporal", sensor_prnu=sensor_prnu, n_frames=args.frames)

    print("All test clips generated successfully.")


if __name__ == "__main__":
    main()
=======


if __name__ == "__main__":
    write_video("real_sim.mp4", fake=False)
    write_video("fake_sim.mp4", fake=True)
    print("Wrote real_sim.mp4 and fake_sim.mp4")
>>>>>>> temporal-freq-analysis
