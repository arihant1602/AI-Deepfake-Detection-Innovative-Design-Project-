"""
Test Video & Fingerprint Generator
====================================
Unified generator supporting synthetic clips and camera sensor fingerprints for:
  - Layer 2: Camera Sensor Noise Profiling (PRNU Analysis)
  - Layer 3: Temporal Consistency & Frequency Analysis
  - End-to-End Gated Verification Pipeline

Modes:
  - 'combined' (default): Generates clips with stationary CMOS PRNU and realistic motion/lighting,
    matching unit test benchmarks for both Layer 2 and Layer 3.
  - 'prnu': Layer 2 specific synthetic clips (stationary CMOS PRNU vs uncorrelated noise).
  - 'temporal': Layer 3 specific synthetic clips (smooth motion vs generative high-freq noise & jumps).
  - 'all': Generates combined, PRNU-specific, and temporal-specific clips.
"""

from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

W, H = 320, 320
FPS = 30
N_FRAMES = 90


def make_base_frame(t: int, w: int = W, h: int = H) -> np.ndarray:
    """
    Generates base clean scene irradiance (unmarred by sensor noise)
    with a moving face-like target and ambient illumination drift over time.
    """
    frame = np.full((h, w), 90.0, dtype=np.float32)

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

    if sensor_prnu is None and not fake:
        sensor_prnu = generate_prnu_fingerprint(w, h)

    rng = np.random.default_rng(101 if not fake else 202)

    for t in range(n_frames):
        base_irradiance = make_base_frame(t, w, h)

        if not fake:
            # Genuine capture: light is modulated by physical sensor PRNU K + shot noise
            shot_noise = rng.normal(0.0, 3.0, (h, w)).astype(np.float32)
            frame = base_irradiance * (1.0 + sensor_prnu) + shot_noise
        else:
            # Injected deepfake / digitally synthesized stream:
            # Lacks stationary CMOS PRNU; noise is independent per-frame generative/rendering noise
            gen_noise = rng.normal(0.0, 6.0, (h, w)).astype(np.float32)
            frame = base_irradiance + gen_noise
            if mode in ("temporal", "combined") and t % 10 == 0:
                frame += 15.0

        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)

    writer.release()
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
