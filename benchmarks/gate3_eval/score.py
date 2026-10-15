"""Scores every benchmark clip per frame with GenD-PE and GenD-CLIP (plus horizontal flip) and saves raw probabilities."""
import glob, json, os, sys, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from deepfake_detector import DeepfakeDetector

N = 32
D = os.path.join(ROOT, "benchmarks", "data")
OUT = os.path.join(D, "gate3_eval")

def frames_of(path):
    if path.endswith(".npz"):
        fr = np.load(path)["frames"]
        fr = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f for f in fr]
    else:
        cap, fr = cv2.VideoCapture(path), []
        while len(fr) < 300:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        cap.release()
    if not fr: return []
    idx = np.linspace(0, len(fr) - 1, min(N, len(fr))).astype(int)
    return [fr[i] for i in idx]

clips = []
for p in sorted(glob.glob(f"{D}/faces_fake/*")): clips.append((p, "fake", os.path.basename(p).split("__")[0]))
for p in sorted(glob.glob(f"{D}/fakes/*")):
    b = os.path.basename(p).split("_"); clips.append((p, "fake", f"{b[0]}_{b[1]}"))
for p in sorted(glob.glob(f"{D}/faces_real/*")): clips.append((p, "real", "celebvhq"))
for p in sorted(glob.glob(f"{D}/vision/*")): clips.append((p, "real", "vision_" + ("WA" if "WA" in p else "native")))
for p in sorted(glob.glob(f"{D}/webcam/*")): clips.append((p, "real", "webcam_gray"))

import gc, torch
pe = DeepfakeDetector("pe")
out, crops_all = [], []
t0 = time.time()
for i, (p, label, group) in enumerate(clips):
    fr = frames_of(p)
    crops, boxes = pe.crops(fr)
    rec = {"file": os.path.basename(p), "label": label, "group": group, "n_frames": len(fr),
           "face_px": [int(b[2]) for b in boxes]}
    if crops:
        flips = [c[:, ::-1].copy() for c in crops]
        rec.update(pe=pe.score_crops(crops), pe_flip=pe.score_crops(flips))
    out.append(rec); crops_all.append(crops)
    if i % 40 == 0: print(i, len(clips), f"{time.time()-t0:.0f}s", flush=True)
del pe; gc.collect(); torch.cuda.empty_cache()
cl = DeepfakeDetector("clip")
for rec, crops in zip(out, crops_all):
    if crops:
        rec.update(clip_scores=cl.score_crops(crops), clip_flip_scores=cl.score_crops([c[:, ::-1].copy() for c in crops]))
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, "scores.json"), "w"))
print("done", len(out), f"{time.time()-t0:.0f}s")
