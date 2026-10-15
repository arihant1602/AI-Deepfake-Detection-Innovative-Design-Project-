"""Gate 3 accuracy summary from scores.json: abstentions, AUROC per variant, operating points, per-generator recall, live-style stability.

Run order: score.py -> robust.py -> analyze.py / build_report.py. Outputs go to benchmarks/data/gate3_eval/ (gitignored: holds dataset face crops)."""
import json, os, numpy as np
from collections import defaultdict
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gate3_eval")
R = json.load(open(os.path.join(HERE, "scores.json")))
MIN = 6
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sig = lambda x: 1 / (1 + np.exp(-x))

def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return float("nan")
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1.0
    # average ranks for ties
    from collections import Counter
    vals = {}
    for v, rk in zip(allv, r): vals.setdefault(v, []).append(rk)
    r = np.array([np.mean(vals[v]) for v in allv])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def score(rec, key, agg):
    if key == "ens":
        p = sig((logit(np.array(rec["pe"])) + logit(np.array(rec["clip_scores"]))) / 2)
    elif key == "pe_tta":
        p = sig((logit(np.array(rec["pe"])) + logit(np.array(rec["pe_flip"]))) / 2)
    elif key == "ens_tta":
        p = sig(np.mean([logit(np.array(rec[k])) for k in ("pe", "pe_flip", "clip_scores", "clip_flip_scores")], 0))
    else:
        p = np.array(rec[key])
    if agg == "mean": return p.mean()
    if agg == "median": return np.median(p)
    if agg == "logit_mean": return sig(logit(p).mean())
    if agg == "trim": s = np.sort(p); k = len(s) // 5; return s[k:len(s) - k].mean() if len(s) > 4 else s.mean()
    if agg == "p75": return np.percentile(p, 75)
    if agg == "max": return p.max()

usable = [r for r in R if "pe" in r and len(r["pe"]) >= MIN]
abst = defaultdict(lambda: [0, 0])
for r in R:
    abst[r["group"]][1] += 1
    if r not in usable: abst[r["group"]][0] += 1
print("abstained (fewer than 6 usable faces of 32):")
for g, (a, n) in sorted(abst.items()): print(f"  {g:24s} {a}/{n}")

fake = [r for r in usable if r["label"] == "fake"]
real = [r for r in usable if r["label"] == "real"]
celeb = [r for r in real if r["group"] == "celebvhq"]
cam = [r for r in real if r["group"].startswith(("vision", "webcam"))]
print(f"\nusable: {len(fake)} fake, {len(real)} real ({len(celeb)} celebvhq, {len(cam)} camera-original)")

print("\nAUROC  [all reals | camera-original reals (VISION + webcam) | CelebV-HQ]")
for key in ("pe", "pe_tta", "clip_scores", "ens", "ens_tta"):
    for agg in ("mean", "median", "logit_mean", "trim", "p75", "max"):
        f = [score(r, key, agg) for r in fake]
        a = auroc(f, [score(r, key, agg) for r in real]); b = auroc(f, [score(r, key, agg) for r in cam]); c = auroc(f, [score(r, key, agg) for r in celeb])
        print(f"  {key:8s} {agg:10s} {a:.3f} | {b:.3f} | {c:.3f}")

def op(key, agg, thr):
    f = np.array([score(r, key, agg) for r in fake]); c = np.array([score(r, key, agg) for r in celeb]); m = np.array([score(r, key, agg) for r in cam])
    return (f >= thr).mean(), (c >= thr).mean(), (m >= thr).mean()

print("\nOperating points (recall on fakes / FPR CelebV-HQ / FPR camera reals)")
for key, agg in (("pe", "mean"), ("pe_tta", "logit_mean"), ("ens_tta", "logit_mean"), ("ens", "logit_mean")):
    for thr in (0.3, 0.4, 0.5):
        tp, fc, fm = op(key, agg, thr)
        print(f"  {key:8s} {agg:10s} thr {thr}: recall {tp:.0%}  FPR celeb {fc:.0%}  FPR camera {fm:.0%}")

print("\nPer-generator recall at thr 0.5: pe/mean  vs  ens_tta/logit_mean   (median clip score)")
by = defaultdict(list)
for r in fake: by[r["group"]].append(r)
for g, rs in sorted(by.items()):
    a = np.array([score(r, "pe", "mean") for r in rs]); b = np.array([score(r, "ens_tta", "logit_mean") for r in rs])
    print(f"  {g:24s} n={len(rs):2d}  {(a>=.5).mean():4.0%} -> {(b>=.5).mean():4.0%}   median {np.median(a):.2f} -> {np.median(b):.2f}")

print("\nReal-clip scores (pe/mean): ", {g: [round(score(r, 'pe', 'mean'), 2) for r in real if r['group'] == g][:12] for g in sorted({r['group'] for r in real})})
print("\nface px median: fake", int(np.median([np.median(r["face_px"]) for r in fake])), " celeb", int(np.median([np.median(r["face_px"]) for r in celeb])), " camera", int(np.median([np.median(r["face_px"]) for r in cam])) if cam else None)

# per-frame stability: how many frames does a single 4-crop live tick need? simulate live: random 4-frame subsets, median of 3 ticks
rng = np.random.default_rng(0)
def live_sim(rec, key="pe"):
    p = np.array(rec[key]); ticks = []
    for _ in range(3):
        ticks.append(p[rng.choice(len(p), min(4, len(p)), replace=False)].mean())
    return np.median(ticks)
f = [live_sim(r) for r in fake]; c = [live_sim(r) for r in celeb]
print(f"\nlive-style decision (3 ticks x 4 crops, median): AUROC {auroc(f, c):.3f}, recall@0.5 {np.mean(np.array(f)>=.5):.0%}, FPR celeb {np.mean(np.array(c)>=.5):.0%}")
