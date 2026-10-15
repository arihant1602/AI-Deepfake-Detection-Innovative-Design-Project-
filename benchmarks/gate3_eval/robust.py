"""Gate 3 robustness: re-scores aligned face crops after realistic degradations; logs why abstained clips abstain; saves example crops."""
import glob, json, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from deepfake_detector import DeepfakeDetector, align_face
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gate3_eval")
D = os.path.join(ROOT, "benchmarks", "data")
S = json.load(open(os.path.join(HERE, "scores.json")))
det = DeepfakeDetector("pe")

def frames_of(path, n):
    if path.endswith(".npz"):
        fr = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in np.load(path)["frames"]]
    else:
        cap, fr = cv2.VideoCapture(path), []
        while len(fr) < 300:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        cap.release()
    idx = np.linspace(0, len(fr) - 1, min(n, len(fr))).astype(int) if fr else []
    return [fr[i] for i in idx]

def path_of(rec):
    for sub in ("faces_fake", "fakes", "faces_real", "vision", "webcam"):
        p = os.path.join(D, sub, rec["file"])
        if os.path.exists(p): return p

def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)

# Degradations applied to the full frame before detection/alignment, so they hit the pipeline as live input would.
DEG = {
    "none": lambda f: f,
    "jpeg_q30": lambda f: jpeg(f, 30),
    "half_res": lambda f: cv2.resize(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), (f.shape[1], f.shape[0])),
    "blur": lambda f: cv2.GaussianBlur(f, (0, 0), 1.6),
    "low_light": lambda f: np.clip(np.clip(f.astype(np.float32) * 0.35, 0, 255) + np.random.default_rng(0).normal(0, 6, f.shape), 0, 255).astype(np.uint8),
    "gray": lambda f: cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
}

usable = [r for r in S if "pe" in r and len(r["pe"]) >= 6]
rob = {k: {"fake": [], "real": []} for k in DEG}
for r in usable:
    fr = frames_of(path_of(r), 12)
    for k, fn in DEG.items():
        crops, _ = det.crops([fn(f) for f in fr])
        if len(crops) >= 3:
            rob[k][r["label"]].append({"clip": r["file"], "group": r["group"], "p": float(np.mean(det.score_crops(crops)))})

# Why abstained clips abstain: raw detections without the size/confidence gate.
why = []
for r in S:
    if "pe" in r and len(r["pe"]) >= 6: continue
    sizes, confs, none = [], [], 0
    for f in frames_of(path_of(r), 32):
        d = det.faces.detect(f)
        if d is None: none += 1; continue
        sizes.append(int(d[0][2])); confs.append(d[2])
    why.append({"clip": r["file"], "group": r["group"], "frames": r["n_frames"], "no_face": none,
                "face_px_median": int(np.median(sizes)) if sizes else None,
                "conf_median": round(float(np.median(confs)), 2) if confs else None,
                "too_small": int(sum(s < 90 for s in sizes)), "low_conf": int(sum(c < 0.85 for c in confs))})

# Example crops for the report: worst misses (lowest P fakes) and worst false flags (highest P reals).
os.makedirs(os.path.join(HERE, "crops"), exist_ok=True)
def save_crop(rec, name):
    fr = frames_of(path_of(rec), 32)
    crops, _ = det.crops(fr)
    if not crops: return None
    probs = det.score_crops(crops)
    i = int(np.argsort(probs)[len(probs) // 2])  # median-scoring frame: representative, not cherry-picked
    cv2.imwrite(os.path.join(HERE, "crops", name), cv2.cvtColor(cv2.resize(crops[i], (160, 160)), cv2.COLOR_RGB2BGR))
    return round(probs[i], 2)

ex = {"missed": [], "false_flag": []}
fakes = sorted([r for r in usable if r["label"] == "fake"], key=lambda r: np.mean(r["pe"]))
seen = set()
for r in fakes:
    if r["group"] in seen: continue
    seen.add(r["group"])
    if np.mean(r["pe"]) >= 0.5 or len(ex["missed"]) >= 6: continue
    name = f"miss_{len(ex['missed'])}.jpg"
    ex["missed"].append({"clip": r["file"], "group": r["group"], "p": round(float(np.mean(r["pe"])), 2), "img": name, "frame_p": save_crop(r, name)})
reals = sorted([r for r in usable if r["label"] == "real"], key=lambda r: -np.mean(r["pe"]))
for r in reals[:6]:
    name = f"ff_{len(ex['false_flag'])}.jpg"
    ex["false_flag"].append({"clip": r["file"], "group": r["group"], "p": round(float(np.mean(r["pe"])), 2), "img": name, "frame_p": save_crop(r, name)})

json.dump({"robust": rob, "abstain": why, "examples": ex}, open(os.path.join(HERE, "extra.json"), "w"), indent=1)
print("ok")
