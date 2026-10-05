"""
Reproducible evaluation of every gate. Produces benchmarks/results/benchmark_results.json
and benchmarks/results/BENCHMARK_RESULTS.md (the numbers quoted in docs/ARCHITECTURE.md).

Inputs (all under benchmarks/data/, git-ignored):
  webcam/*.npz   genuine sessions from capture_webcam_sessions.py (grayscale, lossless)
  vision/*.mkv   VISION native + WhatsApp phone videos   (fetch_datasets.py)
  fakes/*        DF40 deepfakes + text-to-video clips     (fetch_datasets.py)

    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --live /dev/video0 --live-trials 20   # adds live challenge trials
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import host_integrity as h  # noqa: E402
from camera_sensor_noise_profiling import CameraSensorNoiseProfiler  # noqa: E402
from temporal_consistency_analysis import TemporalConsistencyAnalyzer  # noqa: E402

DATA = os.path.join(ROOT, "benchmarks", "data")
OUT = os.path.join(ROOT, "benchmarks", "results")
PCE_T = 60.0
RNG = np.random.default_rng(2026)

FAMILY = {
    "deepfacelab": "face-swap", "faceswap": "face-swap", "inswap": "face-swap", "simswap": "face-swap",
    "uniface": "face-swap", "mobileswap": "face-swap", "facedancer": "face-swap",
    "MRAA": "reenactment", "fomm": "reenactment",
    "sadtalker": "talking-head", "wav2lip": "talking-head", "heygen": "talking-head",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def read_gray(path, n):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(out) if out else None


def webcam_sessions():
    sessions = []
    for p in sorted(glob.glob(os.path.join(DATA, "webcam", "*.npz"))):
        z = np.load(p)
        fr = z["frames"]
        sessions.append({"name": os.path.basename(p)[:-4], "camera_id": str(z["camera_id"]),
                         "mode": f"{fr.shape[2]}x{fr.shape[1]}", "fourcc": str(z["fourcc"]), "frames": fr})
    return sessions


def corpus_clips(n):
    clips = []
    for p in sorted(glob.glob(os.path.join(DATA, "vision", "*.mkv"))):
        b = os.path.basename(p)[:-4]
        grp = "real: phone WhatsApp (VISION)" if "WA" in b else "real: phone native (VISION)"
        clips.append((grp, b, read_gray(p, n)))
    for p in sorted(glob.glob(os.path.join(DATA, "fakes", "*"))):
        b = os.path.basename(p)
        src = b.split("__")[0]
        if src.startswith("T2V"):
            grp = "fake: text-to-video"
        else:
            grp = f"fake: {FAMILY.get(src.replace('DF40_', ''), 'other')} (DF40)"
        fr = read_gray(p, n)
        if fr is not None and len(fr) >= 16:
            clips.append((grp, b, fr))
    return clips


def photo_reanimation(frame, n, grain=2.0):
    """Single genuine frame animated with smooth sub-pixel motion (photo-to-video attack)."""
    hh, ww = frame.shape
    out = []
    for t in range(n):
        m = cv2.getRotationMatrix2D((ww / 2, hh / 2), 2 * np.sin(t / 9), 1 + 0.03 * np.sin(t / 13))
        m[0, 2] += 6 * np.sin(t / 7)
        m[1, 2] += 4 * np.cos(t / 11)
        x = cv2.warpAffine(frame, m, (ww, hh), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        out.append(np.clip(x + RNG.normal(0, grain, x.shape), 0, 255).astype(np.uint8))
    return np.stack(out)


def static_replay(frame, n, grain=2.0):
    """Single genuine frame looped with fresh synthetic grain."""
    return np.stack([np.clip(frame.astype(np.float32) + RNG.normal(0, grain, frame.shape), 0, 255).astype(np.uint8)
                     for _ in range(n)])


def codec_roundtrip(frames, codec, quality):
    """Lossy re-encode (x264 CRF or OpenCV mp4v) and decode, as a recorded replay would be."""
    tmpdir = tempfile.mkdtemp()
    dst = os.path.join(tmpdir, "dst.mp4")
    hh, ww = frames.shape[1:]
    if codec == "mp4v":
        vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), 30, (ww, hh))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        vw.release()
    else:
        raw = os.path.join(tmpdir, "src.raw")
        frames.tofile(raw)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{ww}x{hh}", "-r", "30",
                        "-i", raw, "-c:v", "libx264", "-crf", str(quality), "-preset", "veryfast", "-pix_fmt", "yuv420p", dst],
                       check=True)
    out = read_gray(dst, len(frames))
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
    return out


def center_crop(frames, mode):
    ww, hh = (int(v) for v in mode.split("x"))
    H, W = frames.shape[1:]
    if H < hh or W < ww:
        return None
    y, x = (H - hh) // 2, (W - ww) // 2
    return np.ascontiguousarray(frames[:, y:y + hh, x:x + ww])


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.array(xs, dtype=float)
    return {"n": len(a), "min": round(float(a.min()), 2), "median": round(float(np.median(a)), 2), "max": round(float(a.max()), 2)}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def exp_reference(sessions, clips, n_test):
    """Leave-one-session-out enrollment; genuine vs impostor vs replay PCE."""
    prof = CameraSensorNoiseProfiler()
    by_mode = defaultdict(list)
    for s in sessions:
        by_mode[(s["camera_id"], s["mode"])].append(s)
    results = {}
    for (cam, mode), group in by_mode.items():
        if len(group) < 2:
            continue
        rec = {"sessions": [g["name"] for g in group], "genuine": [], "impostor": [], "replay": defaultdict(list),
               "attack": defaultdict(list), "impostor_by_group": defaultdict(list)}
        fps = {g["name"]: prof.estimate_fingerprint_from_frames(g["frames"]) for g in group}
        t_lat = []
        for enr in group:
            k = fps[enr["name"]]
            for test in group:
                if test is enr:
                    continue
                t0 = time.perf_counter()
                r = prof.analyze_frames(test["frames"][:n_test], ref_fingerprint=k)
                t_lat.append((time.perf_counter() - t0) * 1000)
                rec["genuine"].append(r.components["pce_score"])
        k0 = fps[group[0]["name"]]
        for test in group[1:]:
            f = test["frames"][:n_test]
            for codec, q in (("x264", 23), ("x264", 32), ("mp4v", 0)):
                rec["replay"][f"{codec}{'-crf' + str(q) if q else ''}"].append(
                    prof.analyze_frames(codec_roundtrip(f, codec, q), ref_fingerprint=k0).components["pce_score"])
            rec["attack"]["photo re-animation of a genuine frame"].append(
                prof.analyze_frames(photo_reanimation(f[0], n_test), ref_fingerprint=k0).components["pce_score"])
            rec["attack"]["static replay of a genuine frame"].append(
                prof.analyze_frames(static_replay(f[0], n_test), ref_fingerprint=k0).components["pce_score"])
        for grp, name, fr in clips:
            c = center_crop(fr[:n_test], mode)
            if c is None:
                continue
            pce = prof.analyze_frames(c, ref_fingerprint=k0).components["pce_score"]
            rec["impostor"].append(pce)
            rec["impostor_by_group"][grp].append(pce)
        gen, imp = rec["genuine"], rec["impostor"]
        results[f"{cam} {mode}"] = {
            "sessions": len(group),
            "genuine_pce": stats(gen),
            "genuine_accept_rate": round(sum(p >= PCE_T for p in gen) / len(gen), 4),
            "impostor_pce": stats(imp),
            "impostor_accept_rate": round(sum(p >= PCE_T for p in imp) / len(imp), 4) if imp else None,
            "impostor_by_group": {g: stats(v) for g, v in rec["impostor_by_group"].items()},
            "impostor_accept_by_group": {g: sum(p >= PCE_T for p in v) for g, v in rec["impostor_by_group"].items()},
            "replay_pce": {k: stats(v) for k, v in rec["replay"].items()},
            "replay_accept_rate": {k: round(sum(p >= PCE_T for p in v) / len(v), 3) for k, v in rec["replay"].items()},
            "attack_pce": {k: stats(v) for k, v in rec["attack"].items()},
            "attack_accept_rate": {k: round(sum(p >= PCE_T for p in v) / len(v), 3) for k, v in rec["attack"].items()},
            "latency_ms_per_test": stats(t_lat),
        }
        print(f"[reference] {cam} {mode}: genuine {results[f'{cam} {mode}']['genuine_pce']}, "
              f"impostor {results[f'{cam} {mode}']['impostor_pce']}", flush=True)
    return results


def exp_blind(sessions, clips, ns=(45, 90)):
    prof = CameraSensorNoiseProfiler()
    items = [("real: webcam (this machine)", s["name"], s["frames"]) for s in sessions]
    items += clips
    for s in sessions[:3]:
        items.append(("attack: photo re-animation", s["name"] + "-anim", photo_reanimation(s["frames"][0], 90)))
        items.append(("attack: static replay", s["name"] + "-static", static_replay(s["frames"][0], 90)))
    out = {}
    for n in ns:
        table = defaultdict(Counter)
        for grp, name, fr in items:
            table[grp][prof.analyze_frames(fr[:n]).verdict] += 1
        out[str(n)] = {g: dict(c) for g, c in sorted(table.items())}
        print(f"[blind N={n}] {out[str(n)]}", flush=True)
    return out


def exp_temporal(sessions, clips, n=90):
    ana = TemporalConsistencyAnalyzer()
    items = [("real: webcam (this machine)", s["name"], s["frames"]) for s in sessions] + clips
    table = defaultdict(Counter)
    lat = []
    for grp, name, fr in items:
        bgr = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in fr[:n]]
        t0 = time.perf_counter()
        r = ana.analyze_frames(bgr)
        lat.append(((time.perf_counter() - t0) * 1000, fr.shape[2] * fr.shape[1]))
        state = "abstained" if r.frames_with_face < 8 else ("flagged" if r.flagged else "passed")
        table[grp][state] += 1
    out = {g: dict(c) for g, c in sorted(table.items())}
    print(f"[temporal] {out}", flush=True)
    return out


def exp_challenge_null(sessions, clips, trials_per_trace=300):
    """
    Frames that do NOT respond to the challenge (recorded real and fake clips) scored against
    random challenge sequences, with exactly the statistic used live: the median brightness
    shift of each frame relative to the first frame of the window.
    """
    cfg = h.CHALLENGE_CONFIG
    L = cfg["slots"] * cfg["frames_per_slot"]
    thumbs = [np.stack([h._small_gray(f) for f in s["frames"]]) for s in sessions]
    thumbs += [np.stack([h._small_gray(f) for f in fr]) for _, _, fr in clips]
    thumbs = [t for t in thumbs if len(t) > L]
    rng = random.Random(7)
    n = fp = 0
    corrs = []
    for th in thumbs:
        for _ in range(trials_per_trace):
            s = rng.randrange(0, len(th) - L)
            trace = [float(np.median(th[i] - th[s])) for i in range(s, s + L)]
            seq = h.make_challenge_sequence(cfg["slots"], cfg["min_sign_changes"], rng)
            r = h.analyze_challenge_response(trace, [v for v in seq for _ in range(cfg["frames_per_slot"])],
                                             cfg["max_lag"], cfg["min_corr"], cfg["min_effect"])
            n += 1
            fp += r["passed"]
            corrs.append(r["corr"])
    out = {"traces": len(thumbs), "trials": n, "false_passes": fp, "false_pass_rate": fp / n,
           "rule_of_three_upper_95": (3.0 / n) if fp == 0 else None,
           "null_corr_p99": round(float(np.percentile(corrs, 99)), 3), "null_corr_max": round(float(np.max(corrs)), 3),
           "statistic": "median per-pixel brightness shift vs first frame (80x60 thumbnail)",
           "config": {k: cfg[k] for k in ("slots", "frames_per_slot", "min_corr", "min_effect", "max_lag", "attempts")}}
    print(f"[challenge null] {out}", flush=True)
    return out


def exp_challenge_live(node, trials, sessions):
    passes, corrs, effects, lats = 0, [], [], []
    for _ in range(trials):
        cap = h.capture_attested_frames(node, 1)
        c = cap["challenge"]
        passes += c["verdict"] == "PASS"
        corrs.append(c.get("corr"))
        effects.append(c.get("effect"))
        lats.append(c.get("latency_ms"))
    neg = 0
    for s in sessions[: min(len(sessions), trials)]:
        it = iter([cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in s["frames"]] * 3)
        neg += h.run_sensor_challenge(node, lambda: next(it))["verdict"] == "FAIL"
    out = {"node": node, "trials": trials, "genuine_pass": passes, "corr": stats(corrs), "effect_grey_levels": stats(effects),
           "latency_ms": stats(lats), "replay_trials": min(len(sessions), trials), "replay_rejected": neg}
    print(f"[challenge live] {out}", flush=True)
    return out


def exp_gate1_latency(runs=20):
    lat = [h.probe_host_integrity()["latency_ms"] for _ in range(runs)]
    return stats(lat)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_markdown(res, path):
    L = ["# Benchmark results", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `benchmarks/run_benchmark.py`.", ""]
    L += ["## Gate 2 - enrolled-reference PRNU (decision: PCE >= 60)", ""]
    for k, r in res["reference"].items():
        L += [f"### {k} ({r['sessions']} genuine sessions, leave-one-session-out enrollment)", "",
              "| Stream | n | PCE min | PCE median | PCE max | accepted |", "|---|---|---|---|---|---|"]
        g = r["genuine_pce"]
        L.append(f"| genuine (other sessions) | {g['n']} | {g['min']} | {g['median']} | {g['max']} | {r['genuine_accept_rate']:.1%} |")
        for grp, s in sorted(r["impostor_by_group"].items()):
            acc = r["impostor_accept_by_group"][grp]
            L.append(f"| impostor - {grp} (center crop) | {s['n']} | {s['min']} | {s['median']} | {s['max']} | {acc}/{s['n']} |")
        for kk, s in r["replay_pce"].items():
            L.append(f"| replay of genuine capture via {kk} | {s['n']} | {s['min']} | {s['median']} | {s['max']} | {r['replay_accept_rate'][kk]:.0%} |")
        for kk, s in r["attack_pce"].items():
            L.append(f"| {kk} | {s['n']} | {s['min']} | {s['median']} | {s['max']} | {r['attack_accept_rate'][kk]:.0%} |")
        L += ["", f"Impostor accept rate: {r['impostor_accept_rate']:.2%} of {r['impostor_pce']['n']} clips; "
              f"latency per test {r['latency_ms_per_test']['median']} ms (median).", ""]
    L += ["## Gate 2 - blind motion-gated PRNU (advisory)", ""]
    for n, table in res["blind"].items():
        L += [f"N = {n} frames", "", "| Group | PRESENT | ABSENT | INCONCLUSIVE |", "|---|---|---|---|"]
        for g, c in table.items():
            L.append(f"| {g} | {c.get('PRESENT', 0)} | {c.get('ABSENT', 0)} | {c.get('INCONCLUSIVE', 0)} |")
        L.append("")
    L += ["## Gate 3 - temporal consistency (unchanged module), N = 90", "", "| Group | flagged | passed | abstained (<8 face frames) |",
          "|---|---|---|---|"]
    for g, c in res["temporal"].items():
        L.append(f"| {g} | {c.get('flagged', 0)} | {c.get('passed', 0)} | {c.get('abstained', 0)} |")
    cn = res["challenge_null"]
    L += ["", "## Gate 1 - active sensor challenge", "",
          f"Null (no response) false-pass rate: {cn['false_passes']} / {cn['trials']} trials over {cn['traces']} real luma traces"
          + (f" (95% upper bound {cn['rule_of_three_upper_95']:.2e})" if cn["rule_of_three_upper_95"] else "")
          + f"; null correlation p99 = {cn['null_corr_p99']}, max = {cn['null_corr_max']} (threshold {cn['config']['min_corr']}).", ""]
    if res.get("challenge_live"):
        cl = res["challenge_live"]
        L += [f"Live on {cl['node']}: {cl['genuine_pass']}/{cl['trials']} genuine passes "
              f"(corr {cl['corr']['min']}-{cl['corr']['max']}, effect {cl['effect_grey_levels']['median']} grey levels, "
              f"{cl['latency_ms']['median']} ms); replayed frames rejected {cl['replay_rejected']}/{cl['replay_trials']}.", ""]
    L += [f"Gate 1 passive probe latency: median {res['gate1_latency_ms']['median']} ms "
          f"(min {res['gate1_latency_ms']['min']}, max {res['gate1_latency_ms']['max']}).", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", help="V4L2 node for live challenge trials (optional)")
    ap.add_argument("--live-trials", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=45, help="frames per verification in reference mode")
    ap.add_argument("--skip-temporal", action="store_true")
    ap.add_argument("--only", choices=["challenge"], help="re-run only these experiments and update the saved results")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sessions = webcam_sessions()
    clips = corpus_clips(90)
    print(f"{len(sessions)} webcam sessions, {len(clips)} corpus clips", flush=True)

    if args.only == "challenge":
        with open(os.path.join(OUT, "benchmark_results.json")) as f:
            res = json.load(f)
        res["challenge_null"] = exp_challenge_null(sessions, clips)
        if args.live:
            res["challenge_live"] = exp_challenge_live(args.live, args.live_trials, sessions)
        with open(os.path.join(OUT, "benchmark_results.json"), "w") as f:
            json.dump(res, f, indent=2, default=_json_default)
        write_markdown(res, os.path.join(OUT, "BENCHMARK_RESULTS.md"))
        print("updated challenge results")
        return

    res = {"generated": time.time(), "n_webcam_sessions": len(sessions), "n_corpus_clips": len(clips),
           "corpus_groups": dict(Counter(g for g, _, _ in clips))}
    res["gate1_latency_ms"] = exp_gate1_latency()
    res["reference"] = exp_reference(sessions, clips, args.n_test)
    res["blind"] = exp_blind(sessions, clips)
    res["challenge_null"] = exp_challenge_null(sessions, clips)
    if args.live:
        res["challenge_live"] = exp_challenge_live(args.live, args.live_trials, sessions)
    if not args.skip_temporal:
        res["temporal"] = exp_temporal(sessions, clips)
    else:
        res["temporal"] = {}

    with open(os.path.join(OUT, "benchmark_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=_json_default)
    write_markdown(res, os.path.join(OUT, "BENCHMARK_RESULTS.md"))
    print(f"wrote {OUT}/benchmark_results.json and BENCHMARK_RESULTS.md")


if __name__ == "__main__":
    main()
