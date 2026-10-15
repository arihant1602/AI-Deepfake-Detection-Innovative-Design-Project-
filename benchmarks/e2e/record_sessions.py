"""
Records labelled end-to-end trials for the presentation-attack evaluation (ISO/IEC 30107-3).

One trial = one run of the live path on a real camera node:
  1. Gate 1a passive probe of the node (stored)
  2. Gate 1b randomised multi-control challenge on the open stream (stored, with the attested
     challenge frames used by the same-sensor check)
  3. --frames colour frames, stored losslessly (Gate 2, same-sensor check and Gate 3 are scored
     offline by score_sessions.py, so their thresholds can be studied without re-recording)
  4. optional screen flash (--flash): a full-screen two-colour sequence on this display while the
     camera watches the face; the reflection result is stored

Bona fide trials: vary subject, lighting, distance, glasses, number of people.

    python benchmarks/e2e/record_sessions.py --subject s01 --lighting daylight --flash
    python benchmarks/e2e/record_sessions.py --subject s02 --lighting lamp --people 2 --flash

Attacks:
  physical, in front of the real camera (you perform them, then label the trial):
    --label screen_replay    a phone/monitor replaying a video of a face
    --label printed_photo    a printed photo
    --label mask             a mask or cut-out
  virtual camera (OBS / v4l2loopback / DroidCam): select the virtual node, e.g.
    --node /dev/video10 --label virtual_camera --force     (--force records past a Gate 1a block)
  hooked capture, simulated in software after the driver while the real challenge runs:
    --inject replay|ai|photo|adaptive        (label becomes hooked_<mode>)
    adaptive = replay that reads the brightness control and fakes its effect

Trials go to benchmarks/data/e2e/ (git-ignored: they contain faces). Never publish them.
"""

import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import host_integrity as h  # noqa: E402
import live_engine as le  # noqa: E402

OUT = os.path.join(ROOT, "benchmarks", "data", "e2e")
BONA_FIDE = "bona_fide"
PHYSICAL = ("screen_replay", "printed_photo", "mask", "virtual_camera", "other_attack")


class Camera:
    """The capture stream, optionally hooked: after the driver, frames are replaced like live_engine does."""

    def __init__(self, node, width, height, fourcc):
        self.node = node
        self.cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {node}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.inject, self.fake, self.i, self.fd = None, [], 0, None

    def read(self):
        ok, real = self.cap.read()
        if not ok:
            return None, None, None
        ts, t = self.cap.get(cv2.CAP_PROP_POS_MSEC), time.time()
        if not self.inject:
            return real, ts, t
        f = self.fake[self.i % len(self.fake)]
        self.i += 1
        if self.inject == "adaptive":
            b = h._get_ctrl(self.fd, h.V4L2_CID_BRIGHTNESS)
            f = np.clip(f.astype(np.int16) + int(1.6 * (b - self.b0)), 0, 255).astype(np.uint8)
        return f, ts, t

    def hook(self, mode):
        w, hgt = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if mode in ("replay", "adaptive"):
            self.fake = [self.read()[0] for _ in range(90)]
        elif mode == "ai":
            clips = sorted(glob.glob(os.path.join(ROOT, "benchmarks", "data", "fakes", "T2V_*")))
            if not clips:
                raise SystemExit("--inject ai needs benchmarks/data/fakes/T2V_* (run fetch_datasets.py)")
            c = cv2.VideoCapture(random.choice(clips))
            while len(self.fake) < 150:
                ok, f = c.read()
                if not ok:
                    break
                self.fake.append(cv2.resize(f, (w, hgt)))
        elif mode == "photo":
            src = self.read()[0]
            for t in range(150):
                m = cv2.getRotationMatrix2D((w / 2, hgt / 2), 3 * np.sin(t / 9), 1 + 0.04 * np.sin(t / 13))
                m[0, 2] += 10 * np.sin(t / 7)
                self.fake.append(cv2.warpAffine(src, m, (w, hgt), borderMode=cv2.BORDER_REFLECT))
        if mode == "adaptive":
            self.fd = os.open(self.node, os.O_RDONLY | os.O_NONBLOCK)
            self.b0 = h._get_ctrl(self.fd, h.V4L2_CID_BRIGHTNESS)
        self.inject = mode

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
        self.cap.release()


def screen_flash(cam, face_detector):
    """Shows a random two-colour sequence full-screen and scores the reflection on the face."""
    rng = random.SystemRandom()
    plus, minus = le.FLASH_PAIRS[rng.randrange(len(le.FLASH_PAIRS))]
    plan = {"t0": time.time() + 1.0, "slot_s": le.FLASH_SLOT_S,
            "seq": h.make_challenge_sequence(le.FLASH_SLOTS, 5, rng),
            "colours": ["#%02x%02x%02x" % plus, "#%02x%02x%02x" % minus], "rgb": [plus, minus]}
    win = "flash"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    blank = np.zeros((90, 160, 3), np.uint8)
    end = plan["t0"] + le.FLASH_SLOTS * plan["slot_s"] + le.FLASH_MAX_LAG_S + 0.1
    items, box, shown = [], None, None
    while time.time() < end:
        k = int(np.floor((time.time() - plan["t0"]) / plan["slot_s"]))
        want = None if k < 0 or k >= le.FLASH_SLOTS else (plus if plan["seq"][k] > 0 else minus)
        if want != shown:
            img = blank.copy() if want is None else np.full_like(blank, want[::-1])
            cv2.imshow(win, cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_NEAREST))
            shown = want
        cv2.waitKey(1)
        f, _, t = cam.read()
        if f is None:
            break
        if box is None and t < plan["t0"]:
            found = face_detector.detect(f)
            box = tuple(int(v) for v in found[0]) if found is not None else None
        if plan["t0"] - 0.1 <= t:
            items.append((t, f))
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return le.analyze_screen_flash(items, plan, box)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", default="/dev/video0")
    ap.add_argument("--label", default=BONA_FIDE, choices=(BONA_FIDE,) + PHYSICAL)
    ap.add_argument("--inject", choices=("replay", "ai", "photo", "adaptive"),
                    help="simulate a hooked capture call (sets the label to hooked_<mode>)")
    ap.add_argument("--subject", required=True, help="pseudonym, e.g. s01 (never a real name)")
    ap.add_argument("--lighting", default="", help="e.g. daylight, lamp, dim, backlit")
    ap.add_argument("--people", type=int, default=1)
    ap.add_argument("--note", default="")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--flash", action="store_true", help="also run the screen-flash reflection check")
    ap.add_argument("--force", action="store_true", help="record even when Gate 1a blocks the node")
    args = ap.parse_args()
    label = f"hooked_{args.inject}" if args.inject else args.label

    hw = h.probe_host_integrity(args.node)
    print(f"Gate 1a: {hw['verdict']} - {hw['details']}")
    if hw["blocked"] and not args.force:
        raise SystemExit("Gate 1a blocks this node (a rejected trial). Re-run with --force to record it anyway.")

    cam = Camera(args.node, args.width, args.height, args.fourcc)
    try:
        for _ in range(15):  # auto-exposure warm-up
            cam.read()
        if args.inject:
            cam.hook(args.inject)
        attested = []
        print("Gate 1b: camera challenge…")
        challenge = h.run_sensor_challenge(args.node, lambda: cam.read()[:2],
                                           cfg={"on_challenge_frame": attested.append})
        print(f"Gate 1b: {challenge['verdict']} - {challenge['details']}")
        for _ in range(4):
            cam.read()
        frames, times = [], []
        for _ in range(args.frames):
            f, _, t = cam.read()
            if f is None:
                break
            frames.append(f)
            times.append(t)
        flash = None
        if args.flash:
            from deepfake_detector import FaceDetector
            print("Screen flash: look at the screen…")
            flash = screen_flash(cam, FaceDetector())
            print(f"Screen flash: {flash['verdict']} - {flash['details']}")
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
    finally:
        cam.close()

    meta = {"label": label, "attack": label != BONA_FIDE, "subject": args.subject, "lighting": args.lighting,
            "people": args.people, "note": args.note, "node": args.node, "camera_id": hw.get("camera_id"),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "fps": round(fps, 1),
            "hw": {k: hw.get(k) for k in ("verdict", "blocked", "details")},
            "challenge": {k: v for k, v in challenge.items() if k != "attempts"},
            "flash": flash}
    os.makedirs(os.path.join(OUT, label), exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{args.subject}"
    path = os.path.join(OUT, label, name + ".npz")
    np.savez_compressed(path, frames=np.array(frames), times=np.array(times),
                        challenge_frames=np.array(attested) if attested else np.zeros((0,), np.uint8),
                        meta=json.dumps(meta, default=str))
    with open(os.path.join(OUT, "trials.jsonl"), "a") as f:
        f.write(json.dumps({"file": os.path.relpath(path, OUT), **meta}, default=str) + "\n")
    print(f"saved {os.path.relpath(path, ROOT)} ({len(frames)} frames, {len(attested)} challenge frames, {fps:.0f} fps)")


if __name__ == "__main__":
    main()
