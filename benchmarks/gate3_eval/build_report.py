"""Builds the Gate 3 review (HTML with inline SVG charts and crops) from scores.json and extra.json."""
import base64, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gate3_eval")
S = json.load(open(os.path.join(HERE, "scores.json")))
X = json.load(open(os.path.join(HERE, "extra.json")))
BLUE, CORAL, INK, SOFT, FAINT, LINE = "#226192", "#E58557", "#18222E", "#4A5562", "#7B8490", "#D9D4CA"

usable = [r for r in S if r.get("pe") and len(r["pe"]) >= 6]
fake = [r for r in usable if r["label"] == "fake"]
celeb = [r for r in usable if r["group"] == "celebvhq"]
cam = [r for r in usable if r["group"].startswith(("vision", "webcam"))]
m = lambda r: float(np.mean(r["pe"]))


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


recall = np.mean([m(r) >= 0.5 for r in fake])
fpr = np.mean([m(r) >= 0.5 for r in celeb])
auc = auroc([m(r) for r in fake], [m(r) for r in celeb])

# Frame-level instability: clips whose individual frames land on both sides of 0.5.
straddle_f = np.mean([min(np.mean(np.array(r["pe"]) >= .5), np.mean(np.array(r["pe"]) < .5)) >= 0.2 for r in fake])
straddle_r = np.mean([min(np.mean(np.array(r["pe"]) >= .5), np.mean(np.array(r["pe"]) < .5)) >= 0.2 for r in celeb])

# Live-style decision: 3 ticks x 4 random crops, median.
rng = np.random.default_rng(0)
def live(r):
    p = np.array(r["pe"])
    return np.median([p[rng.choice(len(p), min(4, len(p)), replace=False)].mean() for _ in range(3)])
live_auc = auroc([live(r) for r in fake], [live(r) for r in celeb])

# Threshold sweep: recall needed vs false flags.
def at(thr):
    return np.mean([m(r) >= thr for r in fake]), np.mean([m(r) >= thr for r in celeb])
sweep = [(t, *at(t)) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]

# Per-generator.
FAMILY = {"THB": "TalkingHeadBench", "DF40": "DF40", "T2V": "Text-to-video"}
NAMES = {"THB_extra_Hallo3": "Hallo3", "THB_extra_MAGI-1": "MAGI-1", "DF40_MRAA": "MRAA", "DF40_fomm": "FOMM",
         "T2V_kling": "Kling", "T2V_veo3": "Veo 3", "DF40_heygen": "HeyGen", "DF40_sadtalker": "SadTalker",
         "THB_AniPortraitAudio": "AniPortrait (audio)", "THB_AniPortraitVideo": "AniPortrait (video)",
         "THB_EmoPortrait": "EmoPortrait", "THB_Hallo": "Hallo", "THB_Hallo2": "Hallo2", "THB_LivePortrait": "LivePortrait",
         "DF40_inswap": "InSwap", "DF40_deepfacelab": "DeepFaceLab", "DF40_mobileswap": "MobileSwap", "DF40_wav2lip": "Wav2Lip",
         "DF40_simswap": "SimSwap", "DF40_facedancer": "FaceDancer", "DF40_uniface": "UniFace", "DF40_faceswap": "FaceSwap"}
by = defaultdict(list)
for r in fake:
    by[r["group"]].append(m(r))
gens = sorted(((NAMES.get(g, g.split("_", 1)[1].replace("_", " ")), g.split("_")[0], np.mean(np.array(v) >= .5), len(v), np.median(v))
               for g, v in by.items()), key=lambda x: (x[2], x[4]))

# Robustness.
rob = X["robust"]
base = {lab: {d["clip"]: d["p"] for d in rob["none"][lab]} for lab in ("fake", "real")}
ROB_NAMES = {"none": "As recorded", "jpeg_q30": "JPEG re-save, quality 30", "half_res": "Half resolution",
             "blur": "Slight blur (σ 1.6 px)", "low_light": "Low light (35% exposure + sensor noise)", "gray": "Grayscale"}
robrows = []
for k, v in rob.items():
    f = [d for d in v["fake"] if d["clip"] in base["fake"]]
    rr = [d for d in v["real"] if d["clip"] in base["real"] and d["group"] == "celebvhq"]
    robrows.append(dict(
        name=ROB_NAMES[k], recall=np.mean([d["p"] >= .5 for d in f]), fpr=np.mean([d["p"] >= .5 for d in rr]),
        lost=sum(1 for d in f if base["fake"][d["clip"]] >= .5 and d["p"] < .5),
        caught=sum(1 for d in f if base["fake"][d["clip"]] >= .5),
        newff=sum(1 for d in rr if base["real"][d["clip"]] < .5 and d["p"] >= .5)))

ab = defaultdict(lambda: [0, 0])
for r in S:
    ab[r["group"]][1] += 1
    if r not in usable:
        ab[r["group"]][0] += 1
webcam_total, webcam_abst = ab["webcam_gray"][1], ab["webcam_gray"][0]
t2v_total = sum(v[1] for g, v in ab.items() if g.startswith("T2V"))
t2v_abst = sum(v[0] for g, v in ab.items() if g.startswith("T2V"))

pct = lambda x: f"{x * 100:.0f}%"


def img64(name):
    with open(os.path.join(HERE, "crops", name), "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


# ----------------------------------------------------------------------------------------- charts (SVG)

def gen_chart():
    """Horizontal bars: recall per generator, worst first; 50% guide; value labelled on every bar."""
    row, w, lab, bar_w = 19, 640, 160, 330
    h = row * len(gens) + 34
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Recall per generator at the 0.5 threshold">']
    x0 = lab
    for t in (0, .25, .5, .75, 1):
        x = x0 + bar_w * t
        out.append(f'<line x1="{x}" x2="{x}" y1="4" y2="{h - 22}" stroke="{LINE}" stroke-width="1" {"stroke-dasharray=\"3 3\"" if t == .5 else ""}/>')
        out.append(f'<text x="{x}" y="{h - 8}" font-size="9" fill="{FAINT}" text-anchor="middle" font-family="JetBrains Mono">{int(t * 100)}%</text>')
    for i, (name, fam, rec, n, med) in enumerate(gens):
        y = 6 + i * row
        col = CORAL if rec < 0.75 else BLUE
        bw = max(2, bar_w * rec)
        out.append(f'<text x="{lab - 8}" y="{y + 11}" font-size="10.5" fill="{INK}" text-anchor="end" font-family="Archivo">{name}</text>')
        out.append(f'<text x="0" y="{y + 11}" font-size="8" fill="{FAINT}" font-family="JetBrains Mono">{fam}</text>')
        out.append(f'<rect x="{x0}" y="{y + 2}" width="{bw:.1f}" height="12" rx="2" fill="{col}"/>')
        out.append(f'<text x="{x0 + bw + 6:.1f}" y="{y + 11.5}" font-size="9.5" fill="{INK}" font-family="JetBrains Mono">{pct(rec)}  <tspan fill="{FAINT}">n={n} · median P {med:.2f}</tspan></text>')
    out.append("</svg>")
    return "".join(out)


def strip_chart():
    """Clip-level P(fake): each dot one clip; reals vs fakes grouped by family; 0.5 threshold."""
    groups = [("Real · CelebV-HQ", [m(r) for r in celeb], "real"),
              ("Real · webcam (gray)", [m(r) for r in cam], "real"),
              ("Fake · DF40 swaps", [m(r) for r in fake if r["group"].startswith("DF40") and r["group"] not in ("DF40_heygen", "DF40_sadtalker")], "fake"),
              ("Fake · DF40 HeyGen, SadTalker", [m(r) for r in fake if r["group"] in ("DF40_heygen", "DF40_sadtalker")], "fake"),
              ("Fake · TalkingHeadBench", [m(r) for r in fake if r["group"].startswith("THB")], "fake"),
              ("Fake · text-to-video", [m(r) for r in fake if r["group"].startswith("T2V")], "fake")]
    w, lab, pw, row = 640, 170, 440, 30
    h = row * len(groups) + 30
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Clip-level fake probability by group">']
    for t in (0, .25, .5, .75, 1):
        x = lab + pw * t
        out.append(f'<line x1="{x}" x2="{x}" y1="2" y2="{h - 20}" stroke="{INK if t == .5 else LINE}" stroke-width="1" {"stroke-dasharray=\"4 3\"" if t == .5 else ""}/>')
        out.append(f'<text x="{x}" y="{h - 6}" font-size="9" fill="{FAINT}" text-anchor="middle" font-family="JetBrains Mono">{t:.2f}</text>')
    jr = np.random.default_rng(3)
    for i, (name, vals, kind) in enumerate(groups):
        y = 6 + i * row + row / 2 - 4
        wrong = sum((v >= .5) if kind == "real" else (v < .5) for v in vals)
        out.append(f'<text x="{lab - 10}" y="{y + 3}" font-size="10.5" fill="{INK}" text-anchor="end" font-family="Archivo">{name}</text>')
        out.append(f'<text x="{lab - 10}" y="{y + 14}" font-size="8.5" fill="{FAINT}" text-anchor="end" font-family="JetBrains Mono">{wrong}/{len(vals)} wrong</text>')
        for v in vals:
            bad = (v >= .5) if kind == "real" else (v < .5)
            out.append(f'<circle cx="{lab + pw * v:.1f}" cy="{y + jr.uniform(-6, 6):.1f}" r="3.6" fill="{CORAL if bad else BLUE}" stroke="#fff" stroke-width="1" opacity=".92"/>')
    out.append("</svg>")
    return "".join(out)


def rob_chart():
    """Paired bars per degradation: recall on fakes (should stay high) and false flags on reals (should stay low)."""
    w, lab, bw, row = 640, 220, 300, 34
    h = row * len(robrows) + 40
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Recall and false-flag rate under degradations">']
    for t in (0, .25, .5, .75, 1):
        x = lab + bw * t
        out.append(f'<line x1="{x}" x2="{x}" y1="4" y2="{h - 34}" stroke="{LINE}" stroke-width="1"/>')
        out.append(f'<text x="{x}" y="{h - 22}" font-size="9" fill="{FAINT}" text-anchor="middle" font-family="JetBrains Mono">{int(t * 100)}%</text>')
    for i, r in enumerate(robrows):
        y = 6 + i * row
        out.append(f'<text x="{lab - 10}" y="{y + 14}" font-size="10.5" fill="{INK}" text-anchor="end" font-family="Archivo">{r["name"]}</text>')
        out.append(f'<rect x="{lab}" y="{y + 2}" width="{max(2, bw * r["recall"]):.1f}" height="11" rx="2" fill="{BLUE}"/>')
        out.append(f'<text x="{lab + bw * r["recall"] + 6:.1f}" y="{y + 11}" font-size="9.5" fill="{INK}" font-family="JetBrains Mono">{pct(r["recall"])} fakes caught</text>')
        out.append(f'<rect x="{lab}" y="{y + 15}" width="{max(2, bw * r["fpr"]):.1f}" height="11" rx="2" fill="{CORAL}"/>')
        out.append(f'<text x="{lab + bw * r["fpr"] + 6:.1f}" y="{y + 24}" font-size="9.5" fill="{INK}" font-family="JetBrains Mono">{pct(r["fpr"])} reals flagged</text>')
    ly = h - 8
    out.append(f'<rect x="{lab}" y="{ly - 8}" width="10" height="8" rx="1" fill="{BLUE}"/><text x="{lab + 14}" y="{ly}" font-size="9" fill="{SOFT}" font-family="Archivo">Fakes caught (want high)</text>')
    out.append(f'<rect x="{lab + 170}" y="{ly - 8}" width="10" height="8" rx="1" fill="{CORAL}"/><text x="{lab + 184}" y="{ly}" font-size="9" fill="{SOFT}" font-family="Archivo">Real faces flagged as fake (want low)</text>')
    out.append("</svg>")
    return "".join(out)


def crop_row(items, kind):
    cells = []
    for e in items:
        name = e["group"].split("_", 1)[1] if "_" in e["group"] and e["group"] != "celebvhq" else "CelebV-HQ"
        name = NAMES.get(e["group"], name)
        cells.append(f'<figure><img src="{img64(e["img"])}" alt="{kind} example: {name}"/>'
                     f'<figcaption><b>{name}</b><span>clip P(fake) {e["p"]:.2f}</span></figcaption></figure>')
    return '<div class="crops">' + "".join(cells) + "</div>"


jq, low, blur, gray = (next(r for r in robrows if r["name"].startswith(s)) for s in ("JPEG", "Low", "Slight", "Gray"))
weak = [g for g in gens if g[2] < 0.5]

# ----------------------------------------------------------------------------------------- page

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Gate 3 review</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
@page {{ size: A4; margin: 16mm 16mm 18mm; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Archivo, sans-serif; font-size: 9.6pt; line-height: 1.5; color: {INK}; }}
h1 {{ font-variation-settings: "wdth" 125; font-weight: 900; font-size: 25pt; line-height: 1; margin: 0 0 6pt; color: {BLUE}; }}
h2 {{ font-variation-settings: "wdth" 118; font-weight: 800; font-size: 13pt; margin: 20pt 0 4pt; break-after: avoid; }}
h2 .n {{ color: {CORAL}; margin-right: 6pt; }}
h3 {{ font-weight: 700; font-size: 10pt; margin: 12pt 0 3pt; break-after: avoid; }}
p {{ margin: 0 0 6pt; max-width: 165mm; }}
.lede {{ font-family: "Instrument Serif", serif; font-size: 14pt; line-height: 1.3; color: {SOFT}; margin: 0 0 10pt; }}
.meta {{ font-family: "JetBrains Mono", monospace; font-size: 8pt; color: {FAINT}; border-top: 1.5pt solid {INK}; border-bottom: 1px solid {LINE}; padding: 5pt 0; margin-bottom: 12pt; display: flex; gap: 18pt; flex-wrap: wrap; }}
.mono, code {{ font-family: "JetBrains Mono", monospace; font-size: 8.4pt; }}
.why {{ font-family: "Instrument Serif", serif; font-size: 11.5pt; line-height: 1.35; color: {SOFT}; }}
.box {{ border-left: 3pt solid {CORAL}; background: #FBF1EB; padding: 8pt 10pt; margin: 8pt 0 10pt; }}
.box.blue {{ border-left-color: {BLUE}; background: #EEF3F8; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE}; margin: 6pt 0 12pt; }}
.kpi {{ padding: 7pt 8pt; border-left: 1px solid {LINE}; }} .kpi:first-child {{ border-left: 0; padding-left: 0; }}
.kpi b {{ display: block; font-variation-settings: "wdth" 125; font-weight: 900; font-size: 18pt; line-height: 1.05; }}
.kpi span {{ font-size: 8pt; color: {SOFT}; }}
.bad {{ color: #B4552A; }}
ol.issues {{ padding-left: 14pt; margin: 4pt 0; }} ol.issues li {{ margin-bottom: 4pt; }}
table {{ width: 100%; border-collapse: collapse; font-size: 8.6pt; margin: 6pt 0 10pt; break-inside: avoid; }}
th {{ text-align: left; font-variation-settings: "wdth" 118; font-weight: 700; font-size: 8pt; color: {SOFT}; border-bottom: 1pt solid {INK}; padding: 3pt 8pt 3pt 0; }}
td {{ border-bottom: 1px solid {LINE}; padding: 3pt 8pt 3pt 0; vertical-align: top; }}
td.num {{ font-family: "JetBrains Mono", monospace; font-size: 8.2pt; white-space: nowrap; }}
figure.chart {{ margin: 6pt 0 10pt; break-inside: avoid; }}
figure.chart figcaption {{ font-size: 8pt; color: {SOFT}; margin-top: 2pt; }}
.crops {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 6pt; margin: 6pt 0 4pt; break-inside: avoid; }}
.crops figure {{ margin: 0; }} .crops img {{ width: 100%; display: block; }}
.crops figcaption {{ font-size: 7.6pt; line-height: 1.3; margin-top: 2pt; }} .crops figcaption span {{ display: block; font-family: "JetBrains Mono", monospace; color: {FAINT}; }}
.fix {{ break-inside: avoid; border-top: 1px solid {LINE}; padding: 6pt 0 4pt; display: grid; grid-template-columns: 22pt 1fr; gap: 0 6pt; }}
.fix .k {{ font-variation-settings: "wdth" 125; font-weight: 900; font-size: 13pt; color: {BLUE}; }}
.fix p {{ margin: 0 0 3pt; }} .fix .done {{ font-family: "JetBrains Mono", monospace; font-size: 8pt; color: {SOFT}; }}
.page-break {{ break-before: page; }}
.small {{ font-size: 8pt; color: {SOFT}; }}
</style></head><body>

<h1>Gate 3 review: where the face deepfake model fails</h1>
<p class="lede">GenD-PE-L as wired into the pipeline today, measured on the project's own benchmark data. It misses the generators a KYC attacker would actually use, it can be evaded by re-saving the video as JPEG, and it flags dim or blurry real faces.</p>
<div class="meta"><span>For: Gate 3 owner</span><span>From: Arihant (Gate 2)</span><span>15 Oct 2026</span><span>deepfake_detector.py · GenD-PE-L · threshold 0.5</span></div>

<div class="kpis">
  <div class="kpi"><b class="bad">{pct(recall)}</b><span>of fakes caught at the 0.5 threshold ({len(fake)} usable fake clips)</span></div>
  <div class="kpi"><b class="bad">{pct(fpr)}</b><span>of real CelebV-HQ faces flagged as fake ({len(celeb)} clips)</span></div>
  <div class="kpi"><b class="bad">{pct(jq["recall"])}</b><span>of fakes caught after a JPEG re-save at quality 30</span></div>
  <div class="kpi"><b>{auc:.2f}</b><span>AUROC, fakes vs CelebV-HQ reals (live-window setting: {live_auc:.2f})</span></div>
</div>

<h3>The problems, in order of severity</h3>
<ol class="issues">
  <li><b>Trivial evasion.</b> Re-saving a fake as JPEG (quality 30) drops recall from {pct(recall)} to {pct(jq["recall"])}: {jq["lost"]} of {jq["caught"]} caught fakes get through. Low light drops it to {pct(low["recall"])}.</li>
  <li><b>Blind to modern talking-head generators.</b> {", ".join(f"{g[0]} ({pct(g[2])}, n={g[3]})" for g in weak)} are caught at most half the time. The misses are clean, frontal, well-lit faces: what a KYC selfie looks like.</li>
  <li><b>False flags on genuine faces with blur or poor light.</b> {pct(fpr)} of real faces are flagged as recorded. Slight blur raises that to {pct(blur["fpr"])} and low light to {pct(low["fpr"])}. Webcams are usually soft and often dim.</li>
  <li><b>Never validated on live webcam faces.</b> {webcam_abst} of the {webcam_total} webcam sessions contain no usable face, and all were stored in grayscale. The project's "0 of 11 webcam faces flagged" therefore rests on 2 grayscale clips. Grayscale on its own loses {gray["lost"]} of {gray["caught"]} catches.</li>
  <li><b>Uncalibrated threshold and noisy live decision.</b> 0.5 is the model's default, not a calibrated value (the config comment says otherwise). Live mode decides on 3 ticks of 4 crops; that lowers AUROC from {auc:.3f} to {live_auc:.3f}.</li>
</ol>

<h2>How this was measured</h2>
<p>Every clip under <code>benchmarks/data/</code> was decoded (first 300 frames), 32 frames were sampled evenly, and each was run through the exact production path: <code>DeepfakeDetector.crops()</code> (YuNet, face ≥ 90 px, confidence ≥ 0.85, GenD alignment) then <code>score_crops()</code> with GenD-PE-L in fp16 on the RTX 4050. A clip is judged on the mean P(fake) of its crops, as <code>analyze_frames()</code> does, and abstains below 6 usable faces.</p>
<table>
<tr><th>Set</th><th>Clips</th><th>Usable (≥ 6 faces)</th><th>Notes</th></tr>
<tr><td>TalkingHeadBench fakes</td><td class="num">72</td><td class="num">72</td><td>8 generators: Hallo, Hallo2, Hallo3, AniPortrait (audio, video), EmoPortrait, LivePortrait, MAGI-1</td></tr>
<tr><td>DF40 fakes</td><td class="num">48</td><td class="num">45</td><td>12 generators, 4 clips each: swaps (SimSwap, InSwap, UniFace…), reenactment, HeyGen, SadTalker, Wav2Lip</td></tr>
<tr><td>Text-to-video fakes</td><td class="num">{t2v_total}</td><td class="num">{t2v_total - t2v_abst}</td><td>Veo 3, Kling, Hunyuan, Grok. Most contain no face at all (dancers, objects), so abstaining is correct</td></tr>
<tr><td>CelebV-HQ reals</td><td class="num">60</td><td class="num">60</td><td>YouTube faces, compressed: the only usable real set</td></tr>
<tr><td>Webcam reals (this laptop)</td><td class="num">{webcam_total}</td><td class="num">{webcam_total - webcam_abst}</td><td>Grayscale; 9 sessions show no face</td></tr>
<tr><td>VISION phone reals</td><td class="num">34</td><td class="num">0</td><td>Scene footage, no faces; useful for Gate 2 only</td></tr>
</table>
<p class="small">Robustness: the same clips at 12 frames each, with the degradation applied to the full frame before detection, so it reaches the model exactly as live input would. Scripts: <span class="mono">score.py</span>, <span class="mono">robust.py</span>, <span class="mono">analyze.py</span>, <span class="mono">build_report.py</span> (sent alongside on request). Per-generator counts are small (3–10 clips), so treat single-generator percentages as ±20 points; the robustness results are over all 120 fakes and 60 reals.</p>

<h2><span class="n">1</span>A JPEG re-save defeats it, and low light nearly does</h2>
<p>An attacker controls the encoding of the video they inject. One re-save at JPEG quality 30, which is visually fine at webcam resolution, removes {jq["lost"]} of the {jq["caught"]} fakes the model caught. Low light (35% exposure plus sensor-like noise, a dim room) removes {low["lost"]} and adds {low["newff"]} false flags on real faces. Blur works the other way: it pushes everything toward "fake", so real users get flagged.</p>
<figure class="chart">{rob_chart()}<figcaption>Same {len(fake)} fakes and {len(celeb)} CelebV-HQ reals under each degradation; threshold 0.5. Each pair of bars is one condition.</figcaption></figure>
<table>
<tr><th>Condition</th><th>Fakes caught</th><th>Caught → missed</th><th>Reals flagged</th><th>New false flags</th></tr>
{"".join(f'<tr><td>{r["name"]}</td><td class="num">{pct(r["recall"])}</td><td class="num">{r["lost"]} / {r["caught"]}</td><td class="num">{pct(r["fpr"])}</td><td class="num">{r["newff"]}</td></tr>' for r in robrows)}
</table>
<p class="why">Why it matters: the model is reading compression and sharpness statistics, not only face-synthesis artefacts. JPEG blocking hides the generator's high-frequency fingerprint; blur removes detail the model associates with "real". Both are things a webcam pipeline does on its own, and an attacker can do on purpose.</p>

<h2><span class="n">2</span>It misses the generators that matter for KYC</h2>
<p>Classic face swaps (SimSwap, UniFace, FaceSwap, MRAA, FOMM, Wav2Lip) are caught 75–100% of the time. Audio-driven and diffusion talking-head generators, which produce a whole talking selfie from one photo, are not.</p>
<figure class="chart">{gen_chart()}<figcaption>Share of each generator's clips with mean P(fake) ≥ 0.5, worst first. Coral: below 75%. Dashed line: 50%.</figcaption></figure>
<h3>Missed fakes: the median-scoring frame from the worst clip of each generator</h3>
{crop_row(X["examples"]["missed"], "Missed fake")}
<p class="small">Each is a clean, frontal, evenly lit face scoring P(fake) ≤ 0.08. This is the exact look of a KYC selfie: the case Gate 3 exists for.</p>

<h2><span class="n">3</span>It flags real people, mostly in dim or soft footage</h2>
<p>{pct(fpr)} of real CelebV-HQ clips are flagged at 0.5; lowering the threshold to catch more fakes makes this much worse (table below). The false flags are not random: they are dim, grainy, colour-graded or motion-blurred, which matches the blur and low-light results in section 1.</p>
{crop_row(X["examples"]["false_flag"], "False flag")}
<p class="small">The six highest-scoring real clips (median-scoring frame shown). All are genuine faces from CelebV-HQ.</p>
<figure class="chart">{strip_chart()}<figcaption>Each dot is one clip's mean P(fake). Blue: correct decision at 0.5 (dashed). Coral: wrong (real flagged, or fake missed).</figcaption></figure>

<h3>No threshold fixes it</h3>
<table>
<tr><th>Threshold</th><th>Fakes caught</th><th>Real faces flagged</th></tr>
{"".join(f'<tr><td class="num">{t:.1f}{" (current)" if t == .5 else ""}</td><td class="num">{pct(a)}</td><td class="num">{pct(b)}</td></tr>' for t, a, b in sweep)}
</table>
<p>The score distributions overlap too much: catching 80% of fakes costs a quarter of real users. This is a model problem, not a tuning problem.</p>

<h2><span class="n">4</span>It has not been tested on the input it actually gets</h2>
<p>Live, Gate 3 sees colour 640×480 webcam frames of a person close to the camera. None of the benchmark data matches that:</p>
<ul>
  <li>{webcam_abst} of {webcam_total} webcam sessions have no detectable face (YuNet finds nothing in 32 of 32 frames). The camera was not pointed at a person.</li>
  <li>All webcam sessions are stored grayscale (<code>frames</code> is <code>(120, 480, 640)</code>), and grayscale alone turns {gray["lost"]} of {gray["caught"]} catches into misses.</li>
  <li>The only real-face set with enough clips is CelebV-HQ: professionally shot YouTube video, re-encoded, mostly larger faces (median {int(np.median([np.median(r['face_px']) for r in celeb]))} px) than a webcam produces.</li>
  <li>The text-to-video set contains almost no faces, so Gate 3 has effectively never seen a modern text-to-video person (Veo 3, Kling talking heads).</li>
</ul>
<p>The project claim "0 of 11 webcam faces flagged (max P 0.18)" should be withdrawn until it is re-measured on colour webcam sessions with a face in view.</p>

<h2><span class="n">5</span>Smaller issues in the live path</h2>
<ul>
  <li><b>Threshold.</b> <code>CONFIG["fake_threshold"] = 0.5</code> is commented "calibrated in benchmarks", but no benchmark sets it. It should be chosen from held-out live reals for a target false-flag rate.</li>
  <li><b>Too few crops per decision.</b> <code>live_engine.DEEPFAKE_FRAMES = 4</code> crops per tick, median of 3 ticks: AUROC {live_auc:.3f} vs {auc:.3f} with 32 frames. Single frames swing widely: in {pct(straddle_f)} of fake clips and {pct(straddle_r)} of real clips at least a fifth of the frames land on the other side of 0.5.</li>
  <li><b>Face-size gate.</b> Faces of 77–86 px (DF40 FaceDancer, FaceSwap, SadTalker clips) are skipped by the 90 px floor. That is right for accuracy, but the UI must say "move closer" rather than treat it as a pass.</li>
</ul>

<h2>What did not help (so nobody repeats it)</h2>
<table>
<tr><th>Tried</th><th>AUROC fakes vs all reals</th><th>Verdict</th></tr>
<tr><td>Current: PE backbone, mean of frames</td><td class="num">0.851</td><td>baseline</td></tr>
<tr><td>Median, trimmed mean, logit mean, 75th percentile</td><td class="num">0.851 – 0.856</td><td>no change</td></tr>
<tr><td>Max over frames</td><td class="num">0.839</td><td>worse</td></tr>
<tr><td>Horizontal-flip test-time augmentation</td><td class="num">0.850 – 0.854</td><td>no change</td></tr>
<tr><td>GenD CLIP-L backbone</td><td class="num">0.801</td><td>worse, and 0.70 on camera reals</td></tr>
<tr><td>PE + CLIP ensemble (with and without flips)</td><td class="num">0.861 – 0.863</td><td>+0.01, doubles GPU cost; not worth it</td></tr>
</table>

<h2>What to do, in order</h2>
<div class="fix"><span class="k">1</span><div><p><b>Record a real evaluation set before touching the model.</b> Colour webcam sessions, 10+ people, face in view, normal, dim and backlit rooms, at least 2 webcams. Fix the capture script to keep colour. Add face-bearing text-to-video clips (Veo 3 / Kling talking heads) and HeyGen, Hallo3 and SadTalker outputs made from photos of those same people.</p><p class="done">Done when: ≥ 200 live real windows and ≥ 200 fake windows exist, split by person into train and test.</p></div></div>
<div class="fix"><span class="k">2</span><div><p><b>Fine-tune GenD on that data with augmentation.</b> GenD trains only the LayerNorm parameters and the linear head, so this fits the RTX 4050 at small batch sizes. Train with JPEG quality 20–90, blur, low light + noise, grayscale and downscaling applied to both reals and fakes, so none of them can signal the label.</p><p class="done">Done when: leave-one-generator-out recall ≥ 85% on every generator in section 2, and the section 1 table shows ≤ 10 points of recall loss and ≤ 5 points of extra false flags under every condition.</p></div></div>
<div class="fix"><span class="k">3</span><div><p><b>Calibrate the threshold on held-out live reals.</b> Pick it for a fixed false-flag rate (e.g. 2% of genuine sessions) and report recall at that point, not at 0.5.</p><p class="done">Done when: the threshold in <code>CONFIG</code> comes from a script in <code>benchmarks/</code> and the report states its false-flag rate with a confidence interval.</p></div></div>
<div class="fix"><span class="k">4</span><div><p><b>Make the live decision on more evidence.</b> Raise <code>DEEPFAKE_FRAMES</code> to 8 (there is GPU headroom) and decide over a 5 s window instead of 3 ticks.</p><p class="done">Done when: the live-window AUROC matches the 32-frame AUROC within 0.005.</p></div></div>
<div class="fix"><span class="k">5</span><div><p><b>Fix the project claims.</b> Replace the webcam false-flag claim and the 69% recall figure with numbers from steps 1–3, per generator, with the robustness table.</p></div></div>

<div class="box blue"><p style="margin:0"><b>Context for the system.</b> In the full pipeline these fakes are also stopped earlier: Gate 2 rejected 136 of 136 fake clips and Gate 1b every replay. Gate 3 matters when an attacker gets past those, for example by injecting raw frames with synthetic sensor noise. In that case Gate 3 is the last line, and the problems above are exactly how that attacker would get past it.</p></div>

<p class="small" style="margin-top:14pt">Face images are frames from the TalkingHeadBench, DF40 and CelebV-HQ research datasets, included only to show failure cases. Internal project use; please don't redistribute.</p>
</body></html>"""

open(os.path.join(HERE, "gate3_review.html"), "w").write(html)
print("recall", recall, "fpr", fpr, "auc", auc, "live", live_auc, "straddle", straddle_f, straddle_r)
