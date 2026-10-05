"""
Gate-by-gate demonstration views for the Streamlit portal.

Each tab states the question a gate answers, runs that gate on a real clip from this
machine's camera (or a recorded session), and draws the evidence the decision is based
on. The attack lab runs each attack class through the pipeline and shows which gate
stops it.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import time
import uuid

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import host_integrity as h
from camera_sensor_noise_profiling import (
    CameraSensorNoiseProfiler,
    FingerprintStore,
    extract_noise_residual,
    reference_correlation_surface,
    to_gray_float,
)
from pipeline import DEFAULT_FINGERPRINT_DIR, check_temporal_coherence, enroll_live_camera, run_live_session
from temporal_consistency_analysis import CONFIG as TEMPORAL_CONFIG
from temporal_consistency_analysis import FaceLocator, flow_coherence, high_frequency_ratio

ROOT = os.path.dirname(os.path.abspath(__file__))
WEBCAM_DIR = os.path.join(ROOT, "benchmarks", "data", "webcam")
FAKES_DIR = os.path.join(ROOT, "benchmarks", "data", "fakes")
VISION_DIR = os.path.join(ROOT, "benchmarks", "data", "vision")
RESULTS_JSON = os.path.join(ROOT, "benchmarks", "results", "benchmark_results.json")
W, H = 640, 480
PCE_T = 60.0

# Reference categorical palette (slot 1 / slot 2); status is always shown with a text badge too.
BLUE, ORANGE = "#2a78d6", "#eb6834"
store = FingerprintStore(DEFAULT_FINGERPRINT_DIR)

STATUS_STYLE = {
    "PASS": ("green", ":material/check_circle:"),
    "BLOCK": ("red", ":material/block:"),
    "FLAG": ("orange", ":material/flag:"),
    "ADVISORY": ("blue", ":material/info:"),
    "ABSTAINED": ("gray", ":material/help:"),
    "SKIPPED": ("gray", ":material/fast_forward:"),
    "N/A": ("gray", ":material/remove:"),
}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def badge(status: str, label: str | None = None):
    color, icon = STATUS_STYLE.get(status, ("gray", None))
    st.badge(label or status, color=color, icon=icon)


def render_strip(stages: list[dict]):
    """One card per gate, left to right, so early termination is visible at a glance."""
    cols = st.columns(len(stages))
    for col, s in zip(cols, stages):
        with col.container(border=True):
            st.caption(s["gate"])
            st.markdown(f"**{s['name']}**")
            badge(s["status"], s.get("label"))
            if s.get("value"):
                st.markdown(s["value"])
            if s.get("ms") is not None:
                st.caption(f"{s['ms']:,.0f} ms")


def stages_from_result(result: dict) -> list[dict]:
    """Converts a pipeline result into strip stages (Gate 1a, 1b, 2, 3, verdict)."""
    gate, details, t = result.get("gate"), result.get("details", {}) or {}, result.get("timings", {})
    if gate == 1:
        hw = details
    else:
        hw = details.get("attestation") or result.get("attestation") or {}
    ch = hw.get("challenge") or {}
    challenge_failed = ch.get("verdict") == "FAIL"

    if gate == 1 and not challenge_failed:
        g1a = {"status": "BLOCK", "value": hw.get("details", "")[:140]}
    else:
        v = hw.get("verdict", "PASS")
        g1a = {"status": "FLAG" if v == "FLAG_FOR_REVIEW" else ("PASS" if v != "UNATTESTED_ORIGIN" else "N/A"),
               "label": "UNATTESTED" if v == "UNATTESTED_ORIGIN" else None,
               "value": f"`{hw.get('camera_id') or '-'}`"}
    g1a.update(gate="Gate 1a", name="Real camera?", ms=t.get("gate1_passive_ms"))

    if gate == 1 and not challenge_failed:
        g1b = {"status": "SKIPPED"}
    elif ch:
        g1b = {"status": {"PASS": "PASS", "FAIL": "BLOCK"}.get(ch.get("verdict"), "N/A"),
               "value": f"r = {ch.get('corr', 0):.2f}" if ch.get("corr") is not None else ch.get("details", "")[:80]}
    else:
        g1b = {"status": "N/A", "value": "no live camera"}
    g1b.update(gate="Gate 1b", name="Sensor challenge", ms=t.get("gate1_challenge_ms"))

    prnu = details.get("prnu") or result.get("prnu") or (details if gate == 2 else None)
    if gate == 1:
        g2 = {"status": "SKIPPED"}
    elif not prnu:
        g2 = {"status": "SKIPPED"}
    elif prnu.get("mode") == "reference":
        g2 = {"status": "BLOCK" if gate == 2 else ("PASS" if prnu.get("verdict") == "PRESENT" else "ABSTAINED"),
              "value": f"PCE = {prnu.get('pce_score')}"}
    else:
        g2 = {"status": "ADVISORY", "label": f"blind: {prnu.get('verdict')}", "value": "no enrolled fingerprint"}
    g2.update(gate="Gate 2", name="Sensor fingerprint", ms=t.get("gate2_ms"))

    temporal = details.get("temporal") or (details if gate == 3 else None)
    if gate in (1, 2) or not temporal:
        g3 = {"status": "SKIPPED"}
    elif gate == 3:
        g3 = {"status": "BLOCK", "value": f"score {temporal.get('score')}"}
    elif temporal.get("abstained"):
        g3 = {"status": "ABSTAINED", "value": f"{temporal.get('frames_with_face', 0)} face frames"}
    else:
        g3 = {"status": "PASS", "value": f"score {temporal.get('score')}"}
    g3.update(gate="Gate 3", name="Temporal consistency", ms=t.get("gate3_ms"))

    verdict = result.get("verdict", "")
    final = {"gate": "Verdict", "name": verdict.title(),
             "status": "PASS" if verdict == "AUTHENTIC LIVE STREAM" else ("ADVISORY" if verdict == "NO ANOMALY DETECTED" else "BLOCK"),
             "label": "authentic" if verdict == "AUTHENTIC LIVE STREAM" else ("incomplete evidence" if verdict == "NO ANOMALY DETECTED" else "attack stopped"),
             "ms": t.get("total_ms")}
    return [g1a, g1b, g2, g3, final]


def memo(key, fn):
    """Per-clip memo in session state (avoids re-hashing large frame arrays on every rerun)."""
    demo = st.session_state.get("demo")
    cache = st.session_state.setdefault("demo_memo", {})
    k = (demo["id"] if demo else None, key)
    if k not in cache:
        cache[k] = fn()
    return cache[k]


def read_bgr(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def recorded_sessions() -> list[str]:
    return sorted(glob.glob(os.path.join(WEBCAM_DIR, f"*_{W}x{H}_*.npz")))


def load_recorded(path: str) -> dict:
    z = np.load(path)
    return {"frames": [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in z["frames"][:90]],
            "camera_id": str(z["camera_id"]), "source": f"recorded session `{os.path.basename(path)}`",
            "node": None, "challenge": None}


def to_display(gray_float: np.ndarray, clip_sigma: float = 3.0) -> np.ndarray:
    s = float(np.std(gray_float)) or 1.0
    x = np.clip(gray_float / (clip_sigma * s), -1, 1)
    return ((x + 1) * 127.5).astype(np.uint8)


@st.cache_resource
def simulated_loopback_node() -> dict:
    """
    A v4l2loopback node exactly as the kernel lays it out (OBS Virtual Camera on Linux uses
    v4l2loopback): registered under /sys/devices/virtual, driver 'v4l2 loopback', bus
    'platform:v4l2loopback-000', label chosen by the attacker. Classified by the real code.
    """
    root = tempfile.mkdtemp(prefix="demo_sysfs_")
    virt = os.path.join(root, "devices/virtual/video4linux/video2")
    os.makedirs(virt)
    with open(os.path.join(virt, "name"), "w") as f:
        f.write("Integrated Camera")
    os.makedirs(os.path.join(root, "class/video4linux"))
    os.symlink(virt, os.path.join(root, "class/video4linux/video2"))
    ioctl = {"driver": "v4l2 loopback", "card": "Integrated Camera", "bus_info": "platform:v4l2loopback-000",
             "caps": 0x85208003, "device_caps": 0x05208003, "is_capture": True, "is_output": True,
             "is_metadata": False, "is_streaming": True, "error": None}
    return h.classify_video_node(os.path.join(root, "class/video4linux/video2"), ioctl_info=ioctl)


# --------------------------------------------------------------------------- #
# Clip source
# --------------------------------------------------------------------------- #

def source_picker():
    st.markdown("#### :material/videocam: Step 0: get a clip to demonstrate on")
    devices = [d for d in h.enumerate_video_devices(capture_only=True) if d["is_physical"]]
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    node = None
    if devices:
        labels = {d["node"]: f"{d['node']}: {d['name']}" for d in devices}
        node = c1.selectbox("Camera", list(labels), format_func=labels.get, key="demo_node")
    else:
        c1.info("No physical camera found; use a recorded session.")
    if c2.button("Capture 3 s from camera", type="primary", icon=":material/radio_button_checked:",
                 disabled=node is None, width="stretch"):
        view = st.empty()

        def on_frame(frame, i, total, phase):
            if i % 3 == 0:
                view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"{phase} {i}/{total}",
                           alt="Camera viewfinder", width=360)
        with st.spinner("Attesting camera, running the sensor challenge, capturing..."):
            hw = h.probe_host_integrity(node)
            cap = h.capture_attested_frames(node, 90, W, H, on_frame=on_frame)
        view.empty()
        if cap.get("error") or len(cap["frames"]) < 30:
            st.error(cap.get("error") or "Too few frames captured.")
        else:
            st.session_state["demo"] = {"id": uuid.uuid4().hex, "frames": cap["frames"], "camera_id": hw["camera_id"],
                                        "source": f"live capture from `{node}`", "node": node,
                                        "challenge": cap["challenge"], "hw": hw}
    recs = recorded_sessions()
    if c3.button("Use a recorded session", icon=":material/folder_open:", disabled=not recs, width="stretch"):
        d = load_recorded(recs[0])
        d["id"] = uuid.uuid4().hex
        st.session_state["demo"] = d
    st.caption("Tip: have the person turn their head slowly during capture. The picture flickers briefly while "
               "Gate 1b runs; that flicker *is* the test.")
    demo = st.session_state.get("demo")
    if demo:
        st.success(f"Demo clip: {demo['source']} · {len(demo['frames'])} frames · camera `{demo['camera_id']}`")
    return demo


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def tab_overview():
    st.markdown("### One question per gate, cheapest first")
    st.markdown(
        "Each gate can **stop the session on its own**. Later, more expensive gates only run if the earlier "
        "ones pass, so most attacks are rejected in milliseconds, before a single frame is analysed."
    )
    gates = [
        ("Gate 1a", "Is it a real camera?", "~10 ms", "virtual camera · VM · hook · root"),
        ("Gate 1b", "Does the sensor obey a random command?", "~1.6 s", "frames don't follow the command"),
        ("Gate 2", "Do the pixels carry THIS sensor's fingerprint?", "~0.2 s", "other camera · AI video · animated photo"),
        ("Gate 3", "Is the face temporally consistent?", "~0.3 s", "flicker · incoherent motion"),
    ]
    cols = st.columns([5, 1, 5, 1, 5, 1, 5, 1, 4], vertical_alignment="center")
    for i, (g, q, cost, stops) in enumerate(gates):
        with cols[2 * i].container(border=True):
            st.markdown(f"**{g}**")
            st.markdown(f"#### {q}")
            st.caption(f":material/timer: {cost}")
            st.markdown(f":red[:material/block: **Stops:** {stops}]")
        cols[2 * i + 1].markdown("## :material/arrow_forward:")
    with cols[8].container(border=True):
        st.markdown("**All passed**")
        st.markdown(":green[:material/verified: Authentic live stream]")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Gate 1 knows *which* camera**")
        st.caption("It reads the operating system's own record of the device: USB vendor, product and port. "
                   "A virtual camera has no physical parent device, whatever name it pretends to have.")
    with c2.container(border=True):
        st.markdown("**Gate 1b proves the camera is *live***")
        st.caption("A random brightness pattern is sent to the physical sensor. Only frames coming from that sensor "
                   "*right now* can follow it. Recordings and injected streams cannot.")
    with c3.container(border=True):
        st.markdown("**Gate 2 proves the *pixels* came from it**")
        st.caption("Every sensor has a microscopic, unique noise pattern (PRNU). The frames must carry the pattern "
                   "enrolled for the camera Gate 1 identified.")
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            r = json.load(f)
        ref = r.get("reference", {})
        gen = sum(v["genuine_pce"]["n"] for v in ref.values())
        gen_ok = sum(round(v["genuine_accept_rate"] * v["genuine_pce"]["n"]) for v in ref.values())
        imp = sum(v["impostor_pce"]["n"] for v in ref.values())
        imp_ok = sum(round((v["impostor_accept_rate"] or 0) * v["impostor_pce"]["n"]) for v in ref.values())
        cn = r.get("challenge_null", {})
        st.markdown("#### Measured on the benchmark")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genuine sessions accepted (Gate 2)", f"{gen_ok}/{gen}")
        m2.metric("Other cameras / deepfakes accepted", f"{imp_ok}/{imp}")
        m3.metric("Challenge false passes", f"{cn.get('false_passes', '-')}/{cn.get('trials', '-'):,}")
        cl = r.get("challenge_live") or {}
        m4.metric("Replays rejected by challenge", f"{cl.get('replay_rejected', '-')}/{cl.get('replay_trials', '-')}")
        st.caption("Source: `benchmarks/results/BENCHMARK_RESULTS.md`. Limitations: `docs/ARCHITECTURE.md` §8.")


def tab_gate1a(demo):
    st.markdown("### Gate 1a: is the frame source a physical camera?")
    st.markdown(
        "Injection attacks need a **software camera** (OBS Virtual Camera, ManyCam, DeepFaceLive), a **virtual "
        "machine**, or a **hook** inside the app. All of these leave traces in the operating system, which Gate 1a "
        "reads in about 10 ms, *before the camera is even opened*."
    )
    node = (demo or {}).get("node") or "/dev/video0"
    hw = (demo or {}).get("hw") or h.probe_host_integrity(node)
    sel = hw["camera_audit"]["selected_device"]
    fake = simulated_loopback_node()

    def row(d):
        if d is None:
            return ["-"] * 6
        return [
            d["name"],
            "yes" if d["is_capture"] else "no",
            "virtual (no hardware)" if "/devices/virtual/" in d["node_sysfs"] else "hardware device",
            f"{d['bus']} · {d['bus_info'] or '-'}",
            d["driver"],
            f"{d['usb']['vid']}:{d['usb']['pid']}" if d.get("usb") else "none",
        ]

    checks = ["Name it reports", "Delivers video frames?", "Where the kernel registered it", "Bus",
              "Kernel driver", "USB vendor:product"]
    table = pd.DataFrame({"Check": checks, f"This machine: {node}": row(sel),
                          "OBS virtual camera (v4l2loopback)": row(fake)})
    st.dataframe(table, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown(f"**{node}**")
        badge("PASS" if sel and sel["is_physical"] else "BLOCK",
              "physical camera" if sel and sel["is_physical"] else "not a physical camera")
        st.caption(f"Identity used by Gate 2: `{hw.get('camera_id')}`")
    with c2.container(border=True):
        st.markdown("**OBS virtual camera, even when named \"Integrated Camera\"**")
        badge("BLOCK", "software camera")
        for r in fake["virtual_reasons"]:
            st.markdown(f"- {r}")
        st.caption("Same detection code as live, run on the exact sysfs layout v4l2loopback creates "
                   "(the driver is not installed on this laptop).")

    st.markdown("#### Environment checks")
    for d in hw["diagnostics_summary"]:
        c_a, c_b = st.columns([1, 4])
        with c_a:
            badge({"PASS": "PASS", "WARN": "FLAG", "BLOCK": "BLOCK"}.get(d["status"], "N/A"), d["status"])
        c_b.markdown(f"**{d['name']}**: `{d['value']}`")
    st.caption(f"Probe time: {hw['latency_ms']:.1f} ms")


def challenge_chart(attempt: dict, title: str):
    n = len(attempt["luma"])
    lo, hi = attempt["levels"]
    df_cmd = pd.DataFrame({"frame": range(n), "level": [hi if c > 0 else lo for c in attempt["command"]]})
    df_luma = pd.DataFrame({"frame": range(n), "luma": attempt["luma"]})
    x = alt.X("frame:Q", title="frame", scale=alt.Scale(domain=[0, n - 1]))
    cmd = alt.Chart(df_cmd).mark_line(interpolate="step-after", strokeWidth=2, color=ORANGE).encode(
        x=x, y=alt.Y("level:Q", title="brightness sent", scale=alt.Scale(zero=False)),
        tooltip=["frame", "level"]).properties(height=110, title="Command sent to the physical sensor")
    lum = alt.Chart(df_luma).mark_line(strokeWidth=2, color=BLUE, point=alt.OverlayMarkDef(size=20, color=BLUE)).encode(
        x=x, y=alt.Y("luma:Q", title="mean brightness", scale=alt.Scale(zero=False)),
        tooltip=["frame", alt.Tooltip("luma:Q", format=".1f")]).properties(height=150, title=title)
    st.altair_chart(alt.vconcat(cmd, lum).resolve_scale(x="shared"), width="stretch",
                    alt=f"Challenge command versus {title}")


def tab_gate1b(demo):
    st.markdown("### Gate 1b: does the camera obey a random command right now?")
    st.markdown(
        "1. Draw a **random** up/down pattern (12 steps) after the session starts. Nobody can know it in advance.\n"
        "2. Send it to the **physical sensor's brightness control** (the camera Gate 1a identified).\n"
        "3. Measure the brightness of the frames the app actually receives. If they follow the pattern, the frames "
        "come from that sensor, live. A virtual camera, a recording or an AI video ignores it."
    )
    if not demo or not demo.get("challenge") or not demo["challenge"].get("attempts"):
        st.info("Capture a clip from the camera (Step 0) to see the live challenge. Recorded sessions have no live sensor.")
        return
    ch = demo["challenge"]
    a = next((x for x in ch["attempts"] if x["passed"]), ch["attempts"][-1])
    c1, c2 = st.columns([3, 1])
    with c1:
        challenge_chart(a, "Brightness of the frames received (live camera)")
    with c2.container(border=True):
        st.markdown("**Live camera**")
        badge("PASS" if ch["passed"] else "BLOCK", "follows the pattern" if ch["passed"] else "does not follow")
        st.metric("Match (correlation)", f"{a['corr']:.2f}", help="Needs at least 0.90")
        st.metric("Response", f"{a['effect']:.0f} grey levels", help="Needs at least 6")
        st.metric("Delay", f"{a['lag']} frame(s)")

    st.markdown("#### Now the attack: feed a *recording* while the real sensor is challenged")
    st.caption("This is what a virtual camera or a hooked capture call does: frames come from somewhere else "
               "while the physical camera receives the command.")
    if st.button("Run replay attack against the challenge", icon=":material/replay:", disabled=not demo.get("node")):
        it = iter(list(demo["frames"]) * 3)
        with st.spinner("Challenging the physical sensor while replaying recorded frames..."):
            st.session_state["demo_replay"] = h.run_sensor_challenge(demo["node"], lambda: next(it))
    rep = st.session_state.get("demo_replay")
    if rep and rep.get("attempts"):
        b = rep["attempts"][-1]
        c1, c2 = st.columns([3, 1])
        with c1:
            challenge_chart(b, "Brightness of the frames received (replayed recording)")
        with c2.container(border=True):
            st.markdown("**Replayed recording**")
            badge("PASS" if rep["passed"] else "BLOCK", "follows the pattern" if rep["passed"] else "ignores the pattern")
            st.metric("Match (correlation)", f"{b['corr']:.2f}", help="Needs at least 0.90")
            st.metric("Response", f"{b['effect']:.1f} grey levels")


def fingerprint_for(demo):
    fp = store.load(demo["camera_id"], W, H) if demo.get("camera_id") else None
    return fp


def attack_options(demo) -> dict:
    opts = {"Photo re-animation (one real frame, animated)": "anim"}
    grok = sorted(glob.glob(os.path.join(FAKES_DIR, "T2V_grok*")))
    if grok:
        opts["AI-generated video (Grok)"] = grok[0]
    other = sorted(glob.glob(os.path.join(VISION_DIR, "D04_indoor_move.mkv")))
    if other:
        opts["Real video from a different camera (LG phone)"] = other[0]
    return opts


def attack_frames(demo, choice: str) -> list[np.ndarray]:
    if choice == "anim":
        f = demo["frames"][0]
        out = []
        for t in range(45):
            m = cv2.getRotationMatrix2D((W / 2, H / 2), 2 * np.sin(t / 9), 1 + 0.03 * np.sin(t / 13))
            m[0, 2] += 6 * np.sin(t / 7)
            out.append(cv2.warpAffine(f, m, (W, H), borderMode=cv2.BORDER_REFLECT))
        return out
    frames = read_bgr(choice, 45)
    hh, ww = frames[0].shape[:2]
    if (ww, hh) != (W, H):
        y, x = max(0, (hh - H) // 2), max(0, (ww - W) // 2)
        frames = [cv2.resize(f, (W, H)) if (hh < H or ww < W) else f[y:y + H, x:x + W] for f in frames]
    return [np.ascontiguousarray(f) for f in frames]


def surface_chart(surface: np.ndarray, peak: tuple, title: str):
    hw = surface.shape[0] // 2
    yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    df = pd.DataFrame({"dx": xx.ravel(), "dy": yy.ravel(), "v": surface.ravel()})
    vmax = max(5.0, float(surface[hw + peak[0], hw + peak[1]]))
    base = alt.Chart(df).mark_rect().encode(
        x=alt.X("dx:O", title="horizontal shift (px)", axis=alt.Axis(values=[-hw, 0, hw], labelAngle=0)),
        y=alt.Y("dy:O", title="vertical shift (px)", axis=alt.Axis(values=[-hw, 0, hw])),
        color=alt.Color("v:Q", title="correlation (× background)",
                        scale=alt.Scale(scheme="blues", domain=[0, vmax], clamp=True)),
        tooltip=["dx", "dy", alt.Tooltip("v:Q", format=".1f")],
    )
    mark = alt.Chart(pd.DataFrame({"dx": [peak[1]], "dy": [peak[0]]})).mark_point(
        shape="circle", size=320, filled=False, color=ORANGE, strokeWidth=2.5).encode(x="dx:O", y="dy:O")
    st.altair_chart((base + mark).properties(title=title, height=320), width="stretch", alt=title)


def tab_gate2(demo):
    st.markdown("### Gate 2: were the pixels formed on *this* camera's sensor?")
    st.markdown(
        "Every sensor's pixels differ microscopically in sensitivity: a fixed, unique pattern (**PRNU**) about 1% "
        "strong, invisible to the eye. It is extracted once at **enrollment** and stored under the camera identity "
        "from Gate 1. A session's frames must contain *that* pattern. AI video, animated photos and other cameras "
        "don't."
    )
    if not demo:
        st.info("Get a clip first (Step 0).")
        return
    fp = fingerprint_for(demo)
    g0 = to_gray_float(demo["frames"][0])
    res = memo("residual", lambda: extract_noise_residual(g0))
    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(demo["frames"][0], cv2.COLOR_BGR2RGB), caption="1. Camera frame", alt="Camera frame")
    c2.image(to_display(res), caption="2. Noise left after removing the picture (contrast boosted)",
             alt="Noise residual image")
    if fp is not None:
        crop = fp[H // 2 - 48: H // 2 + 48, W // 2 - 64: W // 2 + 64]
        c3.image(cv2.resize(to_display(crop), (W // 2, int(W // 2 * 96 / 128)), interpolation=cv2.INTER_NEAREST),
                 caption="3. Enrolled sensor fingerprint (centre, enlarged). Visible room outlines are scene texture "
                         "that leaked in because enrollment was filmed in one static scene; enrolling while moving the "
                         "camera gives a cleaner pattern.", alt="Enrolled sensor fingerprint")
    else:
        with c3.container(border=True):
            st.markdown("**3. No fingerprint enrolled** for this camera at 640×480.")
            if demo.get("node"):
                st.caption("During enrollment, slowly move or tilt the camera so the scene keeps changing.")
                if st.button("Enroll this camera now (~6 s)", type="primary", icon=":material/fingerprint:"):
                    with st.spinner("Gate 1a + challenge, then 150 frames..."):
                        r = enroll_live_camera(demo["node"], 150, W, H)
                    st.toast("Enrolled." if r["enrolled"] else f"Refused: {r['reason']}")
                    st.rerun()
            else:
                others = recorded_sessions()[1:]
                if others and st.button("Enroll from another recorded session", icon=":material/fingerprint:"):
                    z = np.load(others[0])
                    k = CameraSensorNoiseProfiler().estimate_fingerprint_from_frames(z["frames"])
                    store.save(demo["camera_id"], k, {"frames": len(z["frames"]), "source": os.path.basename(others[0])})
                    st.rerun()
        return

    st.markdown("#### The test: slide the fingerprint over the frames and look for a spike")
    opts = attack_options(demo)
    choice = st.selectbox("Compare the live clip against", list(opts), key="g2_attack")
    gen = memo("g2_gen", lambda: reference_correlation_surface(demo["frames"][:45], fp, half_width=15))
    att = memo(("g2_att", choice), lambda: reference_correlation_surface(attack_frames(demo, opts[choice]), fp, half_width=15))
    c1, c2 = st.columns(2)
    for col, (pce, surf, peak), title in ((c1, gen, "Live clip from this camera"), (c2, att, choice)):
        with col:
            surface_chart(surf, peak, title)
            ok = pce >= PCE_T
            badge("PASS" if ok else "BLOCK", f"PCE {pce:,.1f}: {'match' if ok else 'no match'}")
            st.caption(f"Strongest peak (orange ring) at shift ({peak[0]}, {peak[1]}) px.")
    st.caption("A dark spot at (or within 3 px of) zero shift that stands far above everything else means the frames "
               "carry this sensor's pattern. PCE (peak-to-correlation energy) measures how far it stands out; "
               f"the decision threshold is {PCE_T:.0f}. A few-pixel offset is normal: the camera's scaler phase "
               "shifts slightly between sessions.")
    df = pd.DataFrame({"stream": ["Live clip (this camera)", choice], "pce": [max(gen[0], 0.1), max(att[0], 0.1)]})
    x = alt.X("pce:Q", scale=alt.Scale(type="log", domain=[0.1, max(1000.0, gen[0] * 3)]), title="PCE (log scale)",
              axis=alt.Axis(values=[0.1, 1, 10, 60, 100, 1000, 10000, 100000], grid=False))
    dots = alt.Chart(df).mark_circle(size=260, color=BLUE, opacity=1).encode(
        x=x, y=alt.Y("stream:N", title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=["stream", alt.Tooltip("pce:Q", format=",.1f")])
    labels = alt.Chart(df).mark_text(dx=14, align="left", fontSize=13).encode(
        x=x, y="stream:N", text=alt.Text("pce:Q", format=",.1f"))
    rule = alt.Chart(pd.DataFrame({"t": [PCE_T]})).mark_rule(color=ORANGE, strokeDash=[5, 4], strokeWidth=2).encode(x="t:Q")
    rule_label = alt.Chart(pd.DataFrame({"t": [PCE_T], "txt": ["decision threshold 60"]})).mark_text(
        align="left", dx=5, dy=-58, color=ORANGE, fontSize=12).encode(x="t:Q", text="txt:N")
    st.altair_chart((rule + dots + labels + rule_label).properties(height=150), width="stretch",
                    alt="PCE of the live clip and the attack clip on a log scale, with the decision threshold")


def tab_gate3(demo):
    st.markdown("### Gate 3: is the face temporally consistent?")
    st.markdown(
        "Face-swap and generated video often look right frame by frame but **flicker** in fine texture, or move "
        "**incoherently**, from one frame to the next. Gate 3 tracks the face and measures four signals over time."
    )
    if not demo:
        st.info("Get a clip first (Step 0).")
        return

    def compute():
        loc = FaceLocator()
        boxes, hf, flow, prev = [], [], [], None
        for f in demo["frames"]:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            b = loc.locate(g)
            boxes.append(b)
            if b is None:
                hf.append(None)
                flow.append(None)
                prev = None
                continue
            x, y, w, hh = b
            roi = cv2.resize(g[y:y + hh, x:x + w], (TEMPORAL_CONFIG["face_roi_size"],) * 2)
            hf.append(high_frequency_ratio(roi, TEMPORAL_CONFIG["high_freq_radius_frac"]))
            flow.append(flow_coherence(prev, roi) if prev is not None else None)
            prev = roi
        return boxes, hf, flow, check_temporal_coherence(frames=demo["frames"])

    boxes, hf, flow, res = memo("g3", compute)
    found = sum(b is not None for b in boxes)
    cols = st.columns(4)
    for i, col in enumerate(cols):
        idx = min(i * (len(boxes) // 4), len(boxes) - 1)
        img = cv2.cvtColor(demo["frames"][idx], cv2.COLOR_BGR2RGB).copy()
        if boxes[idx] is not None:
            x, y, w, hh = boxes[idx]
            cv2.rectangle(img, (x, y), (x + w, y + hh), (42, 120, 214), 3)
        col.image(img, caption=f"Frame {idx}: {'face tracked' if boxes[idx] is not None else 'no face'}",
                  alt=f"Frame {idx} with face box")
    st.caption(f"Face tracked in {found} of {len(boxes)} frames.")
    if found < 8:
        st.warning("Too few frames with a face for Gate 3 to judge, so it abstains (it never guesses). Capture again "
                   "with someone facing the camera.", icon=":material/face:")
        return

    df = pd.DataFrame({"frame": range(len(hf)), "hf": hf, "flow": flow})
    c1, c2 = st.columns(2)
    xs = alt.X("frame:Q", title="frame", scale=alt.Scale(domain=[0, len(hf) - 1]))
    with c1:
        st.markdown("**Fine-texture energy of the face, per frame**")
        st.altair_chart(alt.Chart(df.dropna(subset=["hf"])).mark_line(strokeWidth=2, color=BLUE, point=True).encode(
            x=xs, y=alt.Y("hf:Q", title="high-frequency energy", scale=alt.Scale(zero=False)),
            tooltip=["frame", alt.Tooltip("hf:Q", format=".3f")]).properties(height=200), width="stretch",
            alt="High-frequency energy per frame")
    with c2:
        st.markdown("**How uniformly the face moves between frames**")
        st.altair_chart(alt.Chart(df.dropna(subset=["flow"])).mark_line(strokeWidth=2, color=BLUE, point=True).encode(
            x=xs, y=alt.Y("flow:Q", title="motion coherence (0-1)", scale=alt.Scale(domain=[0, 1])),
            tooltip=["frame", alt.Tooltip("flow:Q", format=".2f")]).properties(height=200), width="stretch",
            alt="Optical-flow coherence per frame")
    st.caption("Gaps are frames where no face was tracked.")

    c1, c2 = st.columns([2, 1])
    with c1:
        comp = pd.DataFrame({
            "signal": ["Flicker", "Incoherent motion", "Periodic flicker", "Texture instability"],
            "value": [res.get("flicker_rate", 0), res.get("flow_incoherence", 0), res.get("periodicity_anomaly", 0),
                      min(1.0, (res.get("hf_cv") or 0) / 0.5)],
            "weight": [0.30, 0.25, 0.25, 0.20]})
        comp["contribution"] = comp["value"] * comp["weight"]
        st.markdown("**What the anomaly score is made of** (score = sum of contributions; flagged at 0.60)")
        comp["label"] = comp.apply(lambda r: f"{r['contribution']:.2f} (value {r['value']:.2f} × weight {r['weight']:.2f})", axis=1)
        y = alt.Y("signal:N", title=None, sort=None, axis=alt.Axis(labelLimit=200, labelOverlap=False))
        bars = alt.Chart(comp).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("contribution:Q", title="contribution to anomaly score", scale=alt.Scale(domain=[0, 0.45])), y=y,
            tooltip=["signal", alt.Tooltip("value:Q", format=".3f"), "weight", alt.Tooltip("contribution:Q", format=".3f")])
        text = alt.Chart(comp).mark_text(align="left", dx=6, fontSize=12).encode(
            x="contribution:Q", y=y, text="label:N")
        st.altair_chart((bars + text).properties(height=200), width="stretch",
                        alt="Contribution of each temporal signal to the anomaly score")
    with c2.container(border=True):
        st.metric("Anomaly score", f"{res.get('score') or 0:.2f}", help="Flagged at 0.60 or above")
        if res.get("abstained"):
            badge("ABSTAINED", "too few face frames")
        else:
            badge("BLOCK" if not res["passed"] else "PASS", "flagged" if not res["passed"] else "consistent")
    st.caption("Honest status: this gate's thresholds are still uncalibrated. On the benchmark it flagged 2 of 64 "
               "fakes, so the architecture's security claim rests on Gates 1a, 1b and 2 (docs/ARCHITECTURE.md §7.5).")


SCENARIOS = {
    "Genuine user": ("A real person in front of the real webcam.", ":material/person:"),
    "Virtual camera (OBS)": ("The attacker installs OBS Virtual Camera and selects it as the webcam, feeding a deepfake.",
                             ":material/videocam_off:"),
    "Replayed recording": ("Frames from an earlier genuine recording are injected while the app thinks it reads "
                           "the webcam.", ":material/replay:"),
    "AI video via hooked capture": ("A hook inside the app swaps the webcam frames for an AI-generated or animated video.",
                                    ":material/smart_toy:"),
    "Emulator / virtual machine": ("The app runs inside an emulator or VM whose camera is fully software-controlled.",
                                   ":material/devices:"),
}


def tab_attack_lab(demo):
    st.markdown("### Attack lab: run an attack, see which gate stops it")
    scenario = st.segmented_control("Scenario", list(SCENARIOS), default="Genuine user", key="lab_scenario")
    desc, icon = SCENARIOS[scenario]
    st.markdown(f"{icon} {desc}")
    node = (demo or {}).get("node")
    run = st.button(f"Run: {scenario}", type="primary", icon=":material/play_arrow:")
    results = st.session_state.setdefault("lab_results", {})

    if run:
        if scenario == "Genuine user":
            if not node:
                st.warning("Needs a live camera: capture a clip in Step 0 first.")
            else:
                with st.spinner("Running the full live pipeline..."):
                    r = run_live_session(node, 90, W, H)
                results[scenario] = {"stages": stages_from_result(r), "note": "; ".join(r.get("limitations", []))}
        elif scenario == "Virtual camera (OBS)":
            t0 = time.perf_counter()
            fake = simulated_loopback_node()
            ms = (time.perf_counter() - t0) * 1000
            results[scenario] = {"stages": [
                {"gate": "Gate 1a", "name": "Real camera?", "status": "BLOCK", "value": "; ".join(fake["virtual_reasons"][:2]), "ms": ms},
                {"gate": "Gate 1b", "name": "Sensor challenge", "status": "SKIPPED"},
                {"gate": "Gate 2", "name": "Sensor fingerprint", "status": "SKIPPED"},
                {"gate": "Gate 3", "name": "Temporal consistency", "status": "SKIPPED"},
                {"gate": "Verdict", "name": "Digital injection detected", "status": "BLOCK", "label": "attack stopped"}],
                "note": "Stopped before the camera is opened; zero frames analysed. (Run on the sysfs layout "
                        "v4l2loopback creates; the driver is not installed here.)"}
        elif scenario in ("Replayed recording", "AI video via hooked capture"):
            if not node or not demo:
                st.warning("Needs a live camera: capture a clip in Step 0 first.")
            else:
                hw = h.probe_host_integrity(node)
                if scenario == "Replayed recording":
                    frames = list(demo["frames"])
                else:
                    opts = attack_options(demo)
                    key = "AI-generated video (Grok)" if "AI-generated video (Grok)" in opts else list(opts)[0]
                    frames = attack_frames(demo, opts[key])
                it = iter(frames * 4)
                with st.spinner("Challenging the physical sensor while the injected frames are delivered..."):
                    ch = h.run_sensor_challenge(node, lambda: next(it))
                fp = fingerprint_for(demo)
                pce = reference_correlation_surface(frames[:45], fp)[0] if fp is not None else None
                g2 = ({"status": "BLOCK" if pce < PCE_T else "FLAG", "value": f"PCE = {pce:,.1f}",
                       "label": "would also block" if pce < PCE_T else "would accept: same camera"}
                      if pce is not None else {"status": "N/A", "value": "enroll the camera to see Gate 2"})
                results[scenario] = {"stages": [
                    {"gate": "Gate 1a", "name": "Real camera?", "status": "PASS", "value": "hook is inside the app", "ms": hw["latency_ms"]},
                    {"gate": "Gate 1b", "name": "Sensor challenge", "status": "BLOCK" if not ch["passed"] else "PASS",
                     "value": f"r = {ch.get('corr', 0):.2f}", "ms": ch.get("latency_ms")},
                    {"gate": "Gate 2 (if 1b were bypassed)", "name": "Sensor fingerprint", **g2},
                    {"gate": "Gate 3", "name": "Temporal consistency", "status": "SKIPPED"},
                    {"gate": "Verdict", "name": "Digital injection detected" if not ch["passed"] else "Not detected",
                     "status": "BLOCK" if not ch["passed"] else "PASS", "label": "attack stopped" if not ch["passed"] else "missed"}],
                    "note": ("A replay of this camera's own footage still carries its fingerprint, so Gate 2 alone would "
                             "accept it; the live challenge is what stops it." if scenario == "Replayed recording" else
                             "Two independent gates reject it: the frames ignore the sensor command, and they lack "
                             "the sensor's fingerprint.")}
        else:
            r = h.evaluate_client_attestation("Android Emulator (Goldfish/QEMU)")
            plat = h.probe_host_integrity(node)["hardware_telemetry"] if node else {}
            results[scenario] = {"stages": [
                {"gate": "Gate 1a", "name": "Real camera?", "status": "BLOCK", "value": "hypervisor / emulator markers",
                 "ms": r["latency_ms"]},
                {"gate": "Gate 1b", "name": "Sensor challenge", "status": "SKIPPED"},
                {"gate": "Gate 2", "name": "Sensor fingerprint", "status": "SKIPPED"},
                {"gate": "Gate 3", "name": "Temporal consistency", "status": "SKIPPED"},
                {"gate": "Verdict", "name": "Environment compromised", "status": "BLOCK", "label": "attack stopped"}],
                "note": "Simulated attestation (this laptop is bare metal: "
                        f"{'no hypervisor bit, no VM DMI strings' if plat.get('is_bare_metal') else 'see inspector'}). "
                        "Inside a real VM the live check finds the CPU hypervisor flag and the virtual DMI vendor."}

    if scenario in results:
        render_strip(results[scenario]["stages"])
        if results[scenario].get("note"):
            st.caption(results[scenario]["note"])


def render_demo():
    st.subheader(":material/school: Gate-by-gate demonstration")
    demo = source_picker()
    st.divider()
    tabs = st.tabs(["Overview", "Gate 1a: real camera?", "Gate 1b: sensor challenge", "Gate 2: sensor fingerprint",
                    "Gate 3: temporal", "Attack lab"], key="demo_tabs", on_change="rerun")
    views = [tab_overview, lambda: tab_gate1a(demo), lambda: tab_gate1b(demo), lambda: tab_gate2(demo),
             lambda: tab_gate3(demo), lambda: tab_attack_lab(demo)]
    for tab, view in zip(tabs, views):
        if tab.open:  # only the selected tab computes
            with tab:
                view()
