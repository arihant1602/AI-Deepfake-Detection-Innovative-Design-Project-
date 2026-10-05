"""
Captures genuine webcam sessions for the Layer 2 / Layer 1 benchmark.

Frames are stored exactly as delivered by the camera (grayscale uint8, the only channel
any layer uses) in compressed .npz files - never through a lossy video codec, which
would erase the sensor noise being measured.

    python benchmarks/capture_webcam_sessions.py --node /dev/video0 --sessions 4 --gap 30
    python benchmarks/capture_webcam_sessions.py --node /dev/video0 --width 1280 --height 720

Vary lighting, pose, distance and background between sessions; that is what makes the
genuine-session numbers representative. Files contain images of whoever is in front of
the camera: they are git-ignored and should not be published.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import host_integrity  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "webcam")


def capture(node, n, width, height, fourcc, warmup=15):
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {node}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frames = []
    for i in range(warmup + n):
        ok, f = cap.read()
        if not ok:
            break
        if i >= warmup:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    real_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "replace")
    cap.release()
    return np.stack(frames), real_fourcc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", default="/dev/video0")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--gap", type=float, default=30.0, help="seconds between sessions")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--tag", default="", help="free-text label stored with the session (e.g. 'lamp-off')")
    args = ap.parse_args()

    hw = host_integrity.probe_host_integrity(args.node)
    if hw["blocked"]:
        raise SystemExit(f"Gate 1 blocks {args.node}: {hw['details']}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in range(args.sessions):
        if s:
            time.sleep(args.gap)
        frames, fourcc = capture(args.node, args.frames, args.width, args.height, args.fourcc)
        h, w = frames.shape[1:]
        name = f"{hw['camera_id']}_{w}x{h}_{fourcc}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
        np.savez_compressed(os.path.join(OUT_DIR, name), frames=frames, camera_id=hw["camera_id"],
                            fourcc=fourcc, tag=args.tag, captured_at=time.time())
        print(f"saved {name}: {frames.shape}")


if __name__ == "__main__":
    main()
