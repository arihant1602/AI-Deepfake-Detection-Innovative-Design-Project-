"""
AI Deepfake Detection & Injection Attack Defense - Streamlit Portal
===================================================================
  1. Live camera session: Gate 1 attestation + active sensor challenge, raw capture,
     Gate 2 PRNU against the enrolled fingerprint of the attested camera, Gate 3 temporal.
  2. Pre-recorded video analysis (origin unattested; PRNU enforced only with a reference).
  3. System & attestation inspector.
"""

import os
import tempfile
import time

import cv2
import streamlit as st

import host_integrity
from demo import render_demo, render_strip, stages_from_result
from camera_sensor_noise_profiling import FingerprintStore
from pipeline import (
    DEFAULT_FINGERPRINT_DIR,
    VERDICT_AUTHENTIC,
    VERDICT_NO_ANOMALY,
    enroll_live_camera,
    run_detection_pipeline,
    run_live_session,
)

st.set_page_config(
    page_title="Injection & deepfake detection",
    page_icon=":material/shield:",
    layout="wide",
)

for key in ("live_result", "prerecorded_result", "enroll_result"):
    st.session_state.setdefault(key, None)

SIM_NONE = "None (use live hardware attestation)"
SIM_VECTORS = [SIM_NONE] + list(host_integrity.SIMULATION_PRESETS)
store = FingerprintStore(DEFAULT_FINGERPRINT_DIR)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header(":material/tune: Pipeline")
    with st.container(border=True):
        st.subheader(":material/memory: Gate 1: host telemetry")
        hw_quick = host_integrity.probe_host_integrity()
        plat = hw_quick.get("hardware_telemetry") or {}
        cam = hw_quick.get("camera_audit") or {}
        icon = {"PASS": ":green[PASS]", "FLAG_FOR_REVIEW": ":orange[FLAG]", "BLOCK": ":red[BLOCK]"}.get(hw_quick["verdict"], hw_quick["verdict"])
        st.markdown(f"**Host status:** {icon} ({hw_quick['latency_ms']:.1f} ms)")
        st.caption(hw_quick["details"])
        st.markdown(f"- **Platform:** `{plat.get('vendor', 'unknown')} {plat.get('product', '')}`")
        st.markdown(f"- **Default camera:** `{cam.get('target_node')}` · `{cam.get('primary_driver')}` · `{cam.get('primary_bus')}`")
        st.markdown(f"- **Camera ID:** `{hw_quick.get('camera_id')}`")

    with st.expander(":material/science: Attack simulation vectors", expanded=False):
        st.caption("Replace Gate 1 with a synthetic attestation result to demonstrate early termination.")
        sim_vector = st.selectbox("Simulated attestation", SIM_VECTORS, index=0)

    with st.container(border=True):
        st.subheader(":material/info: Decision rules")
        st.markdown(
            "- **Gate 1:** hook / VM / root / non-physical capture node / failed sensor challenge → `BLOCK`\n"
            "- **Gate 2:** enrolled-reference PCE < 60 → `BLOCK`; blind test is advisory only\n"
            "- **Gate 3:** temporal anomaly score ≥ 0.60 → `BLOCK`\n"
            "- **Authentic** only when every live check ran and passed"
        )

st.title(":material/shield: Injection attack & deepfake detection")
st.caption("Camera attestation · active sensor challenge · sensor-bound PRNU · temporal & spectral consistency")

app_mode = st.segmented_control(
    "Operating mode",
    ["Gate-by-gate demo", "Live camera session", "Pre-recorded video", "System inspector"],
    default="Gate-by-gate demo",
    label_visibility="collapsed",
)
st.divider()


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def render_gate1(hw: dict):
    st.markdown(f"**Verdict:** `{hw.get('verdict')}` · {hw.get('latency_ms', 0.0):.1f} ms")
    st.caption(hw.get("details", ""))
    for d in hw.get("diagnostics_summary", []):
        mark = {"PASS": ":green[PASS]", "WARN": ":orange[WARN]", "BLOCK": ":red[BLOCK]"}.get(d["status"], d["status"])
        st.markdown(f"- {mark} **{d['name']}**: `{d['value']}`")
    ch = hw.get("challenge")
    if ch:
        st.markdown(f"**Sensor challenge:** `{ch['verdict']}` ({ch.get('control')}) · {ch.get('latency_ms', 0):.0f} ms")
        st.caption(ch.get("details", ""))
    if hw.get("is_file_upload"):
        st.caption(f"Container encoder: `{hw.get('encoder')}`")


def render_gate2(pr: dict):
    mode = pr.get("mode")
    st.markdown(f"**Mode:** `{mode}` · **verdict:** `{pr.get('verdict')}` · "
                f"{'enforced' if pr.get('enforced') else 'advisory'} · {pr.get('latency_ms', 0):.0f} ms")
    if mode == "reference":
        st.markdown(f"- PCE vs enrolled fingerprint: `{pr.get('pce_score')}` (threshold 60)")
    else:
        st.markdown(f"- z-score: `{pr.get('z_score')}` · rho: `{pr.get('rho')}` · changed area: `{pr.get('changed_fraction')}`")
    if pr.get("camera_id"):
        st.markdown(f"- Camera: `{pr.get('camera_id')}` @ `{pr.get('sensor_mode')}`")
    st.caption(pr.get("explanation", ""))


def render_gate3(t: dict):
    st.markdown(f"**Score:** `{t.get('score')}` · face frames `{t.get('frames_with_face')}`/{t.get('frames_analyzed')} · "
                f"{'abstained' if t.get('abstained') else ('flagged' if not t.get('passed') else 'passed')}")
    st.markdown(f"- flicker `{t.get('flicker_rate', 0):.3f}` · flow incoherence `{t.get('flow_incoherence', 0):.3f}` · "
                f"periodicity `{t.get('periodicity_anomaly', 0):.3f}`")
    st.caption(t.get("explanation", ""))


def render_result(result: dict):
    verdict, gate = result.get("verdict"), result.get("gate")
    if verdict != "ERROR":
        render_strip(stages_from_result(result))
    if verdict == VERDICT_AUTHENTIC:
        st.success(f"### :material/check_circle: {verdict}")
    elif verdict == VERDICT_NO_ANOMALY:
        st.info(f"### :material/help: {verdict}")
        st.markdown("No gate detected an attack, but the evidence is incomplete:")
        for lim in result.get("limitations", []):
            st.markdown(f"- {lim}")
    else:
        st.error(f"### :material/error: {verdict}" + (f" (terminated at gate {gate})" if gate else ""))

    timings = result.get("timings", {})
    if timings:
        st.caption(" · ".join(f"{k.replace('_ms', '')}: {v:.0f} ms" for k, v in timings.items()))

    details = result.get("details", {})
    if gate == 1:
        render_gate1(details)
        return
    hw = details.get("attestation") or result.get("attestation") or {}
    prnu = details.get("prnu") or result.get("prnu") or (details if gate == 2 else {})
    temporal = details.get("temporal") or (details if gate == 3 else {})
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.subheader("Gate 1: attestation")
        render_gate1(hw)
    with c2.container(border=True):
        st.subheader("Gate 2: sensor noise")
        if prnu:
            render_gate2(prnu)
    with c3.container(border=True):
        st.subheader("Gate 3: temporal")
        if temporal:
            render_gate3(temporal)
        else:
            st.caption("Not reached.")

    frames = result.get("frames") or [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in result.get("frames_bgr", [])[:32]]
    if frames:
        st.subheader(":material/collections: Analysed frames")
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = min(i * max(1, len(frames) // 4), len(frames) - 1)
            col.image(frames[idx], caption=f"Frame {idx}", alt=f"Analysed frame {idx}")


# --------------------------------------------------------------------------- #
# Mode 0: gate-by-gate demonstration
# --------------------------------------------------------------------------- #
if app_mode == "Gate-by-gate demo":
    render_demo()

# --------------------------------------------------------------------------- #
# Mode 1: live camera session
# --------------------------------------------------------------------------- #
elif app_mode == "Live camera session":
    st.subheader(":material/videocam: Live camera session")
    devices = host_integrity.enumerate_video_devices(capture_only=True)
    if not devices:
        st.warning("No V4L2 video-capture nodes found on this host.")
        st.stop()

    c1, c2, c3 = st.columns([2, 1, 1])
    labels = {d["node"]: f"{d['node']}: {d['name']} ({d['driver']}, {d['bus']}{', SOFTWARE CAMERA' if d['is_virtual'] else ''})"
              for d in devices}
    node = c1.selectbox("Capture device", list(labels), format_func=labels.get)
    res = c2.selectbox("Sensor mode", ["640x480", "1280x720", "320x240"])
    n_frames = c3.selectbox("Frames", [45, 90], index=1, format_func=lambda n: f"{n} (~{n / 30:.1f} s)")
    width, height = (int(v) for v in res.split("x"))

    dev = next(d for d in devices if d["node"] == node)
    enrolled = store.metadata(dev["camera_id"], width, height)
    if enrolled:
        st.caption(f":material/fingerprint: Fingerprint enrolled for `{dev['camera_id']}` @ {res} "
                   f"({enrolled.get('frames')} frames, {time.strftime('%Y-%m-%d %H:%M', time.localtime(enrolled['enrolled_at']))}). "
                   "Gate 2 is enforced.")
    else:
        st.caption(f":material/fingerprint: No fingerprint enrolled for `{dev['camera_id']}` @ {res}. "
                   "Gate 2 runs the blind test (advisory). Enroll once from a trusted session.")

    b1, b2 = st.columns(2)
    run_btn = b1.button("Run live verification", type="primary", icon=":material/play_arrow:", width="stretch")
    enroll_btn = b2.button("Enroll this camera", icon=":material/fingerprint:", width="stretch",
                           help="Captures 150 frames after Gate 1 and the sensor challenge pass. Move slowly / vary the scene.")
    st.caption("Ask the subject to move (turn the head slowly) during capture. The image briefly "
               "flickers while the brightness challenge runs.")

    view = st.empty()
    status = st.empty()

    def on_frame(frame, i, total, phase):
        if i % 3 == 0 or i == total:
            view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"{phase} {i}/{total}",
                       alt="Live camera viewfinder", width=480)

    attestation = None if sim_vector == SIM_NONE else sim_vector
    if run_btn:
        with st.spinner("Running gated session..."):
            st.session_state["live_result"] = run_live_session(node, n_frames, width, height, attestation_mode=attestation,
                                                               on_frame=on_frame)
    if enroll_btn:
        with st.spinner("Enrolling camera fingerprint..."):
            st.session_state["enroll_result"] = enroll_live_camera(node, 150, width, height, on_frame=on_frame)
        er = st.session_state["enroll_result"]
        if er["enrolled"]:
            status.success(f"Enrolled `{er['camera_id']}` @ {er['sensor_mode']} from {er['frames']} frames.")
        else:
            status.error(f"Enrollment refused: {er['reason']}")

    if st.session_state["live_result"]:
        st.divider()
        render_result(st.session_state["live_result"])

# --------------------------------------------------------------------------- #
# Mode 2: pre-recorded video
# --------------------------------------------------------------------------- #
elif app_mode == "Pre-recorded video":
    st.subheader(":material/movie: Pre-recorded video analysis")
    st.caption("A file has no live device to attest, so its origin is unattested. Gate 2 is enforced only "
               "when a reference fingerprint for the claimed camera is supplied.")

    source = st.segmented_control("Video source", ["Upload", "Bundled simulations"], default="Upload")
    video_path = None
    if source == "Upload":
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm"])
        if up is not None:
            suffix = os.path.splitext(up.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(up.read())
                video_path = f.name
    else:
        choice = st.radio("Simulation clip", ["real_sim.mp4", "fake_sim.mp4"], horizontal=False)
        if os.path.exists(choice):
            video_path = choice
        else:
            st.error(f"`{choice}` not found. Run `python generate_test_videos.py` first.")

    ref_up = st.file_uploader("Optional reference fingerprint (.npy)", type=["npy"])
    max_frames = st.select_slider("Frames analysed", options=[45, 60, 90, 150], value=60)

    if video_path:
        col_v, col_a = st.columns(2)
        col_v.video(video_path)
        with col_a:
            if st.button("Run verification", type="primary", icon=":material/play_arrow:"):
                ref_path = None
                if ref_up is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f:
                        f.write(ref_up.read())
                        ref_path = f.name
                attestation = "file_upload" if sim_vector == SIM_NONE else sim_vector
                with st.spinner("Running gated pipeline..."):
                    st.session_state["prerecorded_result"] = run_detection_pipeline(
                        video_path, attestation_mode=attestation, ref_fingerprint_path=ref_path, max_sample_frames=max_frames)
        if st.session_state["prerecorded_result"]:
            st.divider()
            render_result(st.session_state["prerecorded_result"])

# --------------------------------------------------------------------------- #
# Mode 3: system inspector
# --------------------------------------------------------------------------- #
else:
    st.subheader(":material/security: Host & camera attestation audit")
    hw = host_integrity.probe_host_integrity()
    m1, m2, m3 = st.columns(3)
    m1.metric("Host verdict", hw["verdict"])
    m2.metric("Probe latency", f"{hw['latency_ms']:.1f} ms")
    m3.metric("Execution", "Bare metal" if hw["hardware_telemetry"]["is_bare_metal"] else "Virtualised")

    with st.container(border=True):
        st.subheader(":material/fact_check: Diagnostics")
        render_gate1(hw)

    with st.container(border=True):
        st.subheader(":material/videocam: Video nodes")
        for d in hw["devices"]:
            kind = "software camera" if d["is_virtual"] else ("physical camera" if d["is_physical"] else
                                                              ("metadata node" if d["is_metadata"] else "non-capture node"))
            st.markdown(f"`{d['node']}` **{d['name']}**: {kind}")
            st.caption(f"driver `{d['driver']}` · bus `{d['bus_info'] or d['bus']}` · device_caps `0x{d['device_caps']:08x}` · "
                       f"id `{d['camera_id']}`" + (f" · reasons: {'; '.join(d['virtual_reasons'])}" if d["virtual_reasons"] else ""))

    with st.container(border=True):
        st.subheader(":material/fingerprint: Enrolled fingerprints")
        files = sorted(f for f in os.listdir(DEFAULT_FINGERPRINT_DIR) if f.endswith(".json")) if os.path.isdir(DEFAULT_FINGERPRINT_DIR) else []
        if files:
            for f in files:
                st.markdown(f"- `{f[:-5]}`")
        else:
            st.caption("None yet.")

    st.subheader(":material/phone_android: Android client attestation payload")
    sample = '{\n  "isRooted": false,\n  "isEmulator": false,\n  "isVirtualCamera": false,\n  "details": "Android SDK verdict"\n}'
    payload = st.text_area("Payload JSON", value=sample, height=140)
    if st.button("Evaluate payload"):
        r = host_integrity.evaluate_client_attestation(payload)
        {"BLOCK": st.error, "FLAG_FOR_REVIEW": st.warning}.get(r["verdict"], st.success)(f"{r['verdict']}: {r['details']}")
        st.json(r)
