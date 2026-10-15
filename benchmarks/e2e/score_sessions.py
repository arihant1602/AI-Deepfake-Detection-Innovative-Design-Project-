"""
Scores the trials recorded by record_sessions.py per gate and end to end, and reports
ISO/IEC 30107-3 metrics:

  APCER  attack presentation classification error rate: share of attack trials of one attack
         type that the system accepted as bona fide (reported per type; the worst one is the
         system's APCER).
  BPCER  bona fide presentation classification error rate: share of bona fide trials rejected.

Each gate is replayed the way the live app runs it:
  1a  stored probe verdict (BLOCK rejects)
  1b  stored multi-control challenge (FAIL rejects; UNSUPPORTED passes through, as live)
  2   live-noise test on consecutive live_engine.WINDOW-frame windows, and the same-sensor check
      of live_engine.SAME_SENSOR_FRAMES-frame windows against the attested challenge frames
  3   GenD on live_engine.DEEPFAKE_FRAMES crops per window, median of 3 windows, as live
      (Gate 3 is called, never changed)
  flash  screen-flash reflection, advisory in the app; reported as if it were enforced

A gate rejects a trial when most of its windows reject ("any" also reports the share of trials
where at least one window rejected: what a user may see flicker in the live console).

    python benchmarks/e2e/score_sessions.py              # writes benchmarks/results/e2e_results.{json,md}
    python benchmarks/e2e/score_sessions.py --no-gate3   # skip GenD (no GPU)
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import live_engine as le  # noqa: E402
from camera_sensor_noise_profiling import CameraSensorNoiseProfiler, same_sensor_check  # noqa: E402

DATA = os.path.join(ROOT, "benchmarks", "data", "e2e")
OUT = os.path.join(ROOT, "benchmarks", "results")
GATES = ("1a", "1b", "2 noise", "2 same sensor", "3", "flash")


def windows(frames, n, step=None):
    step = step or n
    return [frames[i:i + n] for i in range(0, len(frames) - n + 1, step)]


def score_trial(z, meta, det, prnu):
    frames = list(z["frames"])
    out = {}
    out["1a"] = {"reject": bool(meta["hw"]["blocked"]), "any": bool(meta["hw"]["blocked"]), "value": meta["hw"]["verdict"]}
    ch = meta["challenge"]
    out["1b"] = {"reject": ch["verdict"] == "FAIL", "any": ch["verdict"] == "FAIL", "value": ch["verdict"],
                 "corr": ch.get("corr"), "controls": ch.get("controls")}

    noise = [prnu.analyze_frames(w).verdict for w in windows(frames, le.WINDOW, le.WINDOW // 2)]
    absent = [v == "ABSENT" for v in noise if v != "INCONCLUSIVE"]
    out["2 noise"] = {"reject": bool(absent) and sum(absent) > len(absent) / 2, "any": any(absent),
                      "value": f"{sum(absent)}/{len(absent)} windows without live noise"}

    ref = z["challenge_frames"]
    if ch["verdict"] == "PASS" and len(ref) >= 20:
        fp = prnu.estimate_fingerprint_from_frames(list(ref))
        ss = [same_sensor_check([], w, reference_fingerprint=fp) for w in windows(frames, le.SAME_SENSOR_FRAMES)]
        mism = [r["verdict"] == "MISMATCH" for r in ss if r["verdict"] != "INCONCLUSIVE"]
        out["2 same sensor"] = {"reject": bool(mism) and sum(mism) > len(mism) / 2, "any": any(mism),
                                "value": f"PCE {', '.join(str(round(r['pce'] or 0)) for r in ss)}"}
    else:
        out["2 same sensor"] = {"reject": False, "any": False, "value": "not run (no attested frames)"}

    if det is not None:
        ticks = []
        for w in windows(frames, le.WINDOW):
            step = max(1, len(w) // le.DEEPFAKE_FRAMES)
            r = det.analyze_frames(w[::step][:le.DEEPFAKE_FRAMES], min_faces=2)
            ticks.append(None if r.abstained else r.fake_probability)
        thr = det.cfg["fake_threshold"]
        smoothed = []
        for i in range(len(ticks)):
            vals = [p for p in ticks[max(0, i - 2):i + 1] if p is not None]
            if len(vals) >= 2:
                smoothed.append(float(np.median(vals)))
        flagged = [p >= thr for p in smoothed]
        out["3"] = {"reject": bool(flagged) and sum(flagged) > len(flagged) / 2, "any": any(flagged),
                    "value": f"P(fake) {', '.join(f'{p:.2f}' for p in smoothed) or 'abstained'}"}
    else:
        out["3"] = {"reject": False, "any": False, "value": "skipped"}

    fl = meta.get("flash")
    out["flash"] = ({"reject": fl["verdict"] == "NO_REFLECTION", "any": fl["verdict"] == "NO_REFLECTION",
                     "value": f"{fl['verdict']} r={fl.get('corr')}"} if fl else None)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - half), 3), round(min(1.0, c + half), 3))


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"errors": k, "trials": n, "rate": round(k / n, 3) if n else None, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gate3", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(DATA, "*", "*.npz")))
    if not paths:
        raise SystemExit(f"No trials under {os.path.relpath(DATA, ROOT)}/; record some with record_sessions.py.")
    det = None
    if not args.no_gate3:
        from pipeline import get_deepfake_detector
        det = get_deepfake_detector()
        if isinstance(det, Exception):
            print(f"Gate 3 unavailable ({det}); scoring without it.")
            det = None
    prnu = CameraSensorNoiseProfiler()

    trials = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        g = score_trial(z, meta, det, prnu)
        core = [k for k in GATES if k != "flash"]
        rejected_by = [k for k in core if g[k]["reject"]]
        trials.append({"file": os.path.relpath(p, DATA), "label": meta["label"], "attack": meta["attack"],
                       "subject": meta["subject"], "lighting": meta["lighting"], "people": meta["people"],
                       "gates": g, "rejected_by": rejected_by, "accepted": not rejected_by,
                       "accepted_with_flash": not rejected_by and not (g["flash"] and g["flash"]["reject"])})
        print(f"{meta['label']:16s} {meta['subject']:5s} {'ACCEPT' if not rejected_by else 'reject (' + ', '.join(rejected_by) + ')'}")

    labels = sorted({t["label"] for t in trials})
    report = {"trials": len(trials), "per_label": {}, "gates": {}}
    for lab in labels:
        ts = [t for t in trials if t["label"] == lab]
        attack = ts[0]["attack"]
        err = (lambda t: t["accepted"]) if attack else (lambda t: not t["accepted"])
        err_fl = (lambda t: t["accepted_with_flash"]) if attack else (lambda t: not t["accepted_with_flash"])
        report["per_label"][lab] = {
            "metric": "APCER" if attack else "BPCER",
            "end_to_end": rate(sum(map(err, ts)), len(ts)),
            "end_to_end_with_flash": rate(sum(map(err_fl, [t for t in ts if t["gates"]["flash"]])),
                                          sum(1 for t in ts if t["gates"]["flash"])),
        }
        per_gate = {}
        for gname in GATES:
            vals = [t["gates"][gname] for t in ts if t["gates"][gname] is not None]
            if not vals:
                continue
            # Per gate: for attacks the share NOT rejected (gate alone misses it); for bona fide the share rejected.
            k = sum((not v["reject"]) if attack else v["reject"] for v in vals)
            k_any = sum((not v["any"]) if attack else v["any"] for v in vals)
            per_gate[gname] = {**rate(k, len(vals)), "any_window": rate(k_any, len(vals))}
        report["gates"][lab] = per_gate
    attacks = [lab for lab in labels if report["per_label"][lab]["metric"] == "APCER"]
    report["APCER_max"] = max((report["per_label"][lab]["end_to_end"]["rate"] for lab in attacks), default=None)
    bona = report["per_label"].get("bona_fide")
    report["BPCER"] = bona["end_to_end"]["rate"] if bona else None
    report["trial_results"] = trials

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e2e_results.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    lines = ["# End-to-end evaluation (ISO/IEC 30107-3 style)", "",
             f"{len(trials)} trials. Generated by `benchmarks/e2e/score_sessions.py`.", "",
             "| Presentation | Metric | Trials | End to end | 95% CI | With screen flash enforced |",
             "|---|---|---|---|---|---|"]
    for lab in labels:
        r = report["per_label"][lab]
        e, fl = r["end_to_end"], r["end_to_end_with_flash"]
        lines.append(f"| {lab} | {r['metric']} | {e['trials']} | {e['rate']:.0%} ({e['errors']}) | "
                     f"{e['ci95'][0]:.2f}–{e['ci95'][1]:.2f} | "
                     + (f"{fl['rate']:.0%} ({fl['errors']}/{fl['trials']})" if fl["trials"] else "n/a") + " |")
    lines += ["", "Per gate on its own (attacks: share the gate misses; bona fide: share the gate rejects).", "",
              "| Presentation | " + " | ".join(GATES) + " |", "|---|" + "---|" * len(GATES)]
    for lab in labels:
        cells = []
        for gname in GATES:
            v = report["gates"][lab].get(gname)
            cells.append(f"{v['rate']:.0%} ({v['errors']}/{v['trials']})" if v else "–")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT, "E2E_RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAPCER (worst attack type) {report['APCER_max']}, BPCER {report['BPCER']}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'E2E_RESULTS.md'), ROOT)}")


if __name__ == "__main__":
    main()
