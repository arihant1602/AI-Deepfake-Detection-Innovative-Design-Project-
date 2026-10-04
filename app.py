"""
AI Deepfake Detection & Injection Attack Defense - Streamlit Portal
===================================================================
Compliant with CEN/TS 18099 Gated Presentation & Injection Attack Detection.

Supports:
  1. Live Camera Stream: Real-time webcam frame buffer capture & live forensic execution.
  2. Pre-recorded Video Analysis: Upload custom streams or run bundled benchmark simulations.
  3. System & Attestation Inspector: Live OS hardware, video device, and virtualization audit.
"""

import os
import tempfile
import time
import cv2
import numpy as np
import streamlit as st

import host_integrity
from pipeline import (
    check_hardware_attestation,
    check_sensor_noise,
    check_temporal_coherence,
    extract_frames,
    run_detection_pipeline,
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Forensics Portal | Deepfake & Injection Attack Defense",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INITIALIZE SESSION STATE ---
if "live_clip_path" not in st.session_state:
    st.session_state["live_clip_path"] = None
if "live_result" not in st.session_state:
    st.session_state["live_result"] = None
if "prerecorded_result" not in st.session_state:
    st.session_state["prerecorded_result"] = None

# --- SIDEBAR: SYSTEM AUDIT & SETTINGS ---
with st.sidebar:
    st.header(":material/tune: Forensic Pipeline Config")
    st.caption("CEN/TS 18099 Gated Verification Engine")

    with st.container(border=True):
        st.subheader(":material/memory: Gate 1 Attestation Source")
        attestation_source = st.radio(
            "Attestation Source:",
            [
                "Auto (Live Probe for Camera / Provenance for File Upload)",
                "Simulation: Clean Physical Device",
                "Simulation: Rooted Android (Magisk/SU)",
                "Simulation: Android Emulator (QEMU/Goldfish)",
                "Simulation: Virtual Camera Injection (OBS)",
            ],
            index=0,
            help="In Auto mode: probes live hardware for webcam capture, or analyzes container provenance for file uploads. Simulation options demonstrate early termination for specific attack scenarios.",
        )

    with st.container(border=True):
        st.subheader(":material/sensors: Gate 2: PRNU Detection Mode")
        st.caption(
            "**Autonomous Static Noise Detection:** Evaluates whether a persistent, stationary "
            "spatial noise field exists across the sensor grid (confirming a physical CMOS camera) "
            "without requiring any pre-enrolled camera fingerprint."
        )

    with st.container(border=True):
        st.subheader(":material/info: Verification Thresholds")
        st.markdown(
            """
            - **Gate 1 (Device):** Root / VM = `BLOCK`
            - **Gate 2 (PRNU):** PCE Peak > 45.0, Anomaly < 0.55
            - **Gate 3 (Temporal):** Flicker < 0.65, Anomaly < 0.60
            - **Target Latency:** Sub-3.5 seconds
            """
        )

# --- HEADER ---
st.title("🛡️ Next-Gen Injection Attack & Deepfake Detection")
st.caption("Autonomous multi-layered forensics: Client Attestation • CMOS Sensor PRNU • Optical Flow & Spectral Dynamics")

# --- MAIN MODE SELECTION ---
app_mode = st.segmented_control(
    "Select Operating Mode:",
    [
        "Live Camera Stream",
        "Pre-recorded Video Analysis",
        "System & Attestation Inspector",
    ],
    default="Live Camera Stream",
    label_visibility="collapsed",
)

st.divider()


# =============================================================================
# HELPER: FORMAT ATTESTATION ARGUMENT
# =============================================================================
def get_attestation_arg(selection: str, is_live: bool = False):
    if "Auto" in selection:
        return "live" if is_live else "file_upload"
    prefix = "Simulation: "
    if selection.startswith(prefix):
        raw = selection[len(prefix):]
        # Map to known preset keys
        mapping = {
            "Clean Physical Device": "Clean Physical Device",
            "Rooted Android (Magisk/SU)": "Compromised / Rooted Android (Magisk/SU)",
            "Android Emulator (QEMU/Goldfish)": "Android Emulator (Goldfish/QEMU)",
            "Virtual Camera Injection (OBS)": "Virtual Camera Injection (OBS / Hooked Driver)",
        }
        return mapping.get(raw, raw)
    return selection


# =============================================================================
# HELPER: RENDER FORENSIC REPORT CARDS
# =============================================================================
def render_forensic_results(result: dict, video_path: str):
    verdict = result.get("verdict", "UNKNOWN")
    gate = result.get("gate")
    details = result.get("details", {})

    if verdict != "AUTHENTIC LIVE STREAM":
        st.error(f"### :material/error: {verdict}")
        trigger_reason = details.get("details", details.get("explanation", "Verification threshold violated."))
        st.warning(f"**Forensic Trigger Reason:** {trigger_reason}")

        if gate == 1:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Gate 1 Status", details.get("verdict", "BLOCK"), delta="Terminated Early", delta_color="inverse")
            with c2:
                st.metric("Attestation Latency", f"{details.get('latency_ms', 0.0):.2f} ms", delta="Sub-50ms Target", delta_color="normal")
            with c3:
                st.metric("Root / VM Flag", f"Root={details.get('root_detected', False)} | VM={details.get('emulator_detected', False)}")

        elif gate == 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Gate 2: Static Noise", "ABSENT", delta="Synthetic Stream Detected", delta_color="inverse")
            with c2:
                st.metric("PRNU Anomaly Score", f"{details.get('score', 0.0):.3f}", delta="Flagged (≥ 0.55)", delta_color="inverse")
            with c3:
                st.metric("Internal Static PCE", f"{details.get('pce_score', 0.0):.1f}", delta="Below Authentic Threshold (<45.0)", delta_color="inverse")

        elif gate == 3:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Gate 3 Status", "Temporal Anomaly", delta="Terminated at Gate 3", delta_color="inverse")
            with c2:
                st.metric("Temporal Anomaly Score", f"{details.get('score', 0.0):.3f}", delta="Flagged (≥ 0.60)", delta_color="inverse")
            with c3:
                st.metric("Flicker Rate", f"{details.get('flicker_rate', 0.0):.3f}", delta="Elevated Synthesis Jitter", delta_color="inverse")

    else:
        st.success("### :material/check_circle: Verdict: AUTHENTIC LIVE STREAM")
        st.caption("All verification gates passed. Hardware integrity, static CMOS silicon PRNU noise, and temporal coherence confirmed.")

        hw_data = details.get("attestation", {})
        prnu_data = details.get("prnu", {})
        temp_data = details.get("temporal", {})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            if hw_data.get("is_file_upload"):
                enc_tag = str(hw_data.get("encoder", "N/A"))[:12]
                st.metric("Gate 1: Origin", "UNATTESTED", delta=f"Enc: {enc_tag}")
            else:
                st.metric("Gate 1: Device", hw_data.get("verdict", "PASS"), delta=f"{hw_data.get('latency_ms', 0.0):.2f} ms")
        with m2:
            st.metric("Gate 2: Static Noise", "DETECTED", delta="Physical Sensor Verified")
        with m3:
            st.metric("Gate 2: Static PCE", f"{prnu_data.get('pce_score', 0.0):.1f}", delta="≥ 45.0 Authentic")
        with m4:
            st.metric("Gate 3: Temporal Score", f"{temp_data.get('score', 0.0):.3f}", delta="Clean (< 0.60)")

        # Detailed breakdown in expander
        with st.expander(":material/analytics: Comprehensive Forensic Metrics Breakdown", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Layer 2: Static Sensor Noise (PRNU) Detection**")
                st.write(f"- **Static Sensor Noise Exists:** `{'YES' if prnu_data.get('passed', False) else 'NO'}`")
                st.write(f"- **Internal Static PCE Energy:** `{prnu_data.get('pce_score', 0.0):.2f}` (Threshold: ≥ 45.0)")
                st.write(f"- **Inter-frame Noise Persistence:** `{prnu_data.get('persistence', 0.0):.4f}`")
                st.write(f"- **Composite PRNU Anomaly Score:** `{prnu_data.get('score', 0.0):.4f}`")
                st.write(f"- **Analysis Note:** {prnu_data.get('explanation', '')}")
            with col_b:
                st.write("**Layer 3: Temporal & Frequency Consistency**")
                st.write(f"- **Flicker Rate:** `{temp_data.get('flicker_rate', 0.0):.4f}`")
                st.write(f"- **Flow Incoherence:** `{temp_data.get('flow_incoherence', 0.0):.4f}`")
                st.write(f"- **Spectral Periodicity Anomaly:** `{temp_data.get('periodicity_anomaly', 0.0):.4f}`")
                st.write(f"- **Composite Temporal Score:** `{temp_data.get('score', 0.0):.4f}`")
                st.write(f"- **Analysis Note:** {temp_data.get('explanation', '')}")

    # Display Extracted Frame Buffer Gallery
    frames = result.get("frames", [])
    if not frames and os.path.exists(video_path):
        frames = extract_frames(video_path, max_frames=32)

    if frames:
        st.write("---")
        st.subheader(":material/collections: Extracted Frame Pipeline Buffer")
        st.caption("Sampled frames inspected by the optical flow and spatial denoising residual extractors:")
        f_cols = st.columns(4)
        for idx, col in enumerate(f_cols):
            sample_idx = min(idx * 7, len(frames) - 1)
            col.image(frames[sample_idx], caption=f"Frame {sample_idx}", width="stretch")


# =============================================================================
# MODE 1: LIVE CAMERA STREAM
# =============================================================================
if app_mode == "Live Camera Stream":
    st.subheader(":material/videocam: Real-Time Live Camera Stream Verification")
    st.markdown(
        "Captures a live frame sequence directly from your connected camera hardware, "
        "streams it to the UI, and executes the 3 programmed verification tests in real time."
    )

    # Enumerate live video devices
    available_devices = host_integrity.enumerate_video_devices()

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1])

    with ctrl_col1:
        if available_devices:
            dev_options = [f"{d['node']}: {d['name']}" for d in available_devices]
            selected_dev_str = st.selectbox("Select Video Capture Device:", dev_options, index=0)
            selected_cam_node = selected_dev_str.split(":")[0].strip()
            # Extract device index number (e.g. /dev/video0 -> 0)
            try:
                selected_cam_idx = int(selected_cam_node.replace("/dev/video", ""))
            except ValueError:
                selected_cam_idx = 0
        else:
            st.warning("No `/dev/video*` devices detected in sysfs. Defaulting to camera index 0.")
            selected_cam_idx = st.number_input("Camera Index:", min_value=0, max_value=4, value=0)

    with ctrl_col2:
        capture_frames_target = st.selectbox(
            "Live Frame Sequence Buffer:",
            [45, 60, 90],
            index=0,
            format_func=lambda x: f"{x} frames (~{x // 30:.1f}s)",
            help="Higher frame counts increase statistical confidence for PRNU cross-correlation.",
        )

    with ctrl_col3:
        st.write("")
        st.write("")
        start_capture_btn = st.button(
            "Start Live Capture & Verification",
            type="primary",
        )

    live_container = st.container(border=True)

    with live_container:
        viewfinder_col, preview_col = st.columns([1, 1], gap="medium")

        viewfinder_placeholder = viewfinder_col.empty()
        status_placeholder = viewfinder_col.empty()
        preview_placeholder = preview_col.empty()

        # Handle Live Capture Execution
        if start_capture_btn:
            status_placeholder.info("Initializing camera device...")
            cap = cv2.VideoCapture(selected_cam_idx)

            if not cap.isOpened():
                status_placeholder.error(
                    f"Could not open camera device `{selected_cam_idx}`. "
                    "Ensure no other application is locking the webcam, or try Pre-recorded Video Analysis."
                )
            else:
                captured_frames = []
                progress_bar = viewfinder_col.progress(0, text="Capturing live camera buffer...")

                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

                # Capture frames live
                for i in range(capture_frames_target):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    captured_frames.append(frame_bgr)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    viewfinder_placeholder.image(
                        frame_rgb,
                        caption=f"Live Feed • Frame {i + 1}/{capture_frames_target} ({actual_w}x{actual_h})",
                        width="stretch",
                    )
                    progress_bar.progress((i + 1) / capture_frames_target, text=f"Captured {i + 1}/{capture_frames_target} frames...")
                    time.sleep(0.015)

                cap.release()
                progress_bar.empty()

                if len(captured_frames) < 15:
                    status_placeholder.error("Captured insufficient frames from camera. Verification aborted.")
                else:
                    status_placeholder.success(f"Successfully captured {len(captured_frames)} live frames.")

                    # Write captured frames to temporary MP4
                    temp_live = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_live_path = temp_live.name
                    temp_live.close()

                    h, w, _ = captured_frames[0].shape
                    writer = cv2.VideoWriter(temp_live_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                    for f in captured_frames:
                        writer.write(f)
                    writer.release()

                    st.session_state["live_clip_path"] = temp_live_path

                    # Execute the 3 Programmed Tests
                    attestation_arg = get_attestation_arg(attestation_source, is_live=True)

                    with st.status("Executing 3-stage gated forensic pipeline on live capture...", expanded=True) as status:
                        st.write("Gate 1: Probing live host environment & video device integrity...")
                        res = run_detection_pipeline(
                            temp_live_path,
                            attestation_mode=attestation_arg,
                            ref_fingerprint_path=None,
                            fast_sample=False,
                        )

                        if res["gate"] == 1:
                            status.update(label="❌ Terminated at Gate 1 (Host/Device Attestation Block)", state="error")
                        elif res["gate"] == 2:
                            status.update(label="❌ Terminated at Gate 2 (No Static Sensor Noise / Synthetic Video)", state="error")
                        elif res["gate"] == 3:
                            status.update(label="❌ Terminated at Gate 3 (Temporal Incoherence / Deepfake Detected)", state="error")
                        else:
                            st.write("✓ Hardware and OS environment verified clean.")
                            st.write("✓ Stationary CMOS sensor noise pattern detected (physical camera verified).")
                            st.write("✓ Optical flow stability and high-frequency spectral continuity verified.")
                            status.update(label="✅ All Verification Gates Passed: Authentic Live Stream", state="complete")

                    st.session_state["live_result"] = res

        # Display previous or newly captured live results
        if st.session_state.get("live_clip_path") and os.path.exists(st.session_state["live_clip_path"]):
            with preview_placeholder.container():
                st.subheader(":material/play_circle: Captured Live Stream Preview")
                st.video(st.session_state["live_clip_path"])

    if st.session_state.get("live_result") and st.session_state.get("live_clip_path"):
        st.divider()
        st.subheader(":material/assignment: Live Forensic Verification Report")
        render_forensic_results(st.session_state["live_result"], st.session_state["live_clip_path"])


# =============================================================================
# MODE 2: PRE-RECORDED VIDEO ANALYSIS
# =============================================================================
elif app_mode == "Pre-recorded Video Analysis":
    st.subheader(":material/movie: Pre-recorded Video Stream Analysis")
    st.markdown(
        "Analyze pre-recorded, uploaded, or synthetic benchmark video streams "
        "through the full 3-layer gated forensic verification pipeline."
    )

    source_type = st.segmented_control(
        "Choose Video Source:",
        ["Upload Video File", "Bundled Benchmark Simulations"],
        default="Upload Video File",
    )

    video_to_analyze = None

    if source_type == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload video stream (.mp4, .mov, .avi, .mkv, .webm):",
            type=["mp4", "mov", "avi", "mkv", "webm"],
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_to_analyze = tfile.name
            tfile.close()

    else:
        sample_choice = st.radio(
            "Select Benchmark Clip:",
            [
                "real_sim.mp4 — Genuine Physical Camera Simulation (CMOS PRNU + Smooth Motion)",
                "fake_sim.mp4 — Synthetic Deepfake Injection Simulation (No PRNU + Temporal Jitter)",
            ],
            index=0,
        )
        selected_file = "real_sim.mp4" if "real_sim.mp4" in sample_choice else "fake_sim.mp4"
        if os.path.exists(selected_file):
            video_to_analyze = selected_file
        else:
            st.error(f"Sample file `{selected_file}` not found on disk. Run `python generate_test_videos.py` first.")

    if video_to_analyze is not None:
        col_preview, col_action = st.columns([1, 1], gap="large")

        with col_preview:
            st.subheader(":material/smart_display: Video Stream Preview")
            st.video(video_to_analyze)

            # Metadata inspection
            cap_meta = cv2.VideoCapture(video_to_analyze)
            if cap_meta.isOpened():
                w_m = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_m = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_m = cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
                total_f = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                dur = total_f / fps_m if fps_m else 0.0
                cap_meta.release()
                st.caption(f"**Resolution:** {w_m}x{h_m} | **Frames:** {total_f} | **FPS:** {fps_m:.1f} | **Duration:** {dur:.2f}s")

        with col_action:
            st.subheader(":material/play_arrow: Gated Verification Execution")
            st.markdown(
                "Executes Gate 1 (Host Attestation), Gate 2 (PRNU Sensor Noise Profiling), "
                "and Gate 3 (Temporal & Spectral Coherence) with early termination."
            )

            fast_sample_check = st.checkbox(
                "Enable fast frame sampling (first 45 frames for sub-2s latency)",
                value=True,
                help="Recommended: preserves forensic accuracy while reducing processing time on long video clips.",
            )

            run_btn = st.button("Run Forensic Verification", type="primary")

            if run_btn:
                attestation_arg = get_attestation_arg(attestation_source, is_live=False)

                with st.status("Executing gated verification pipeline...", expanded=True) as status:
                    st.write(f"Gate 1: Evaluating client/host attestation ({attestation_source})...")
                    res = run_detection_pipeline(
                        video_to_analyze,
                        attestation_mode=attestation_arg,
                        ref_fingerprint_path=None,
                        fast_sample=fast_sample_check,
                    )

                    if res["gate"] == 1:
                        status.update(label="❌ Terminated at Gate 1 (Hardware/OS Level Block)", state="error")
                    elif res["gate"] == 2:
                        status.update(label="❌ Terminated at Gate 2 (No Static Sensor Noise / Synthetic Video)", state="error")
                    elif res["gate"] == 3:
                        status.update(label="❌ Terminated at Gate 3 (Temporal Coherence Anomaly)", state="error")
                    else:
                        st.write("✓ Hardware and OS integrity verified.")
                        st.write("✓ Stationary CMOS sensor noise pattern detected (physical camera verified).")
                        st.write("✓ Optical flow stability and high-frequency spectral ratios verified.")
                        status.update(label="✅ All Verification Gates Passed", state="complete")

                st.session_state["prerecorded_result"] = res

        if st.session_state.get("prerecorded_result"):
            st.divider()
            st.subheader(":material/assignment: Forensic Verification Report")
            render_forensic_results(st.session_state["prerecorded_result"], video_to_analyze)


# =============================================================================
# MODE 3: SYSTEM & ATTESTATION INSPECTOR
# =============================================================================
elif app_mode == "System & Attestation Inspector":
    st.subheader(":material/security: Real-Time Host & Video Hardware Integrity Audit")
    st.markdown(
        "Performs live, un-mocked diagnostic probing of your host operating system, "
        "video device drivers, hypervisor markers, and execution privileges."
    )

    t_probe_start = time.perf_counter()
    live_host_data = host_integrity.probe_host_integrity()
    probe_latency = (time.perf_counter() - t_probe_start) * 1000.0

    top_m1, top_m2, top_m3, top_m4 = st.columns(4)
    with top_m1:
        st.metric(
            "Overall Host Verdict",
            live_host_data["verdict"],
            delta="Session Trusted" if live_host_data["passed"] else "Untrusted",
        )
    with top_m2:
        st.metric(
            "Measured Probe Latency",
            f"{live_host_data['latency_ms']:.2f} ms",
            delta="Sub-1ms Target",
        )
    with top_m3:
        st.metric(
            "Hypervisor / VM",
            "Detected" if live_host_data["emulator_detected"] else "Bare Metal",
            delta="Clean" if not live_host_data["emulator_detected"] else "Risk Flag",
            delta_color="normal" if not live_host_data["emulator_detected"] else "inverse",
        )
    with top_m4:
        st.metric(
            "Virtual Camera Drivers",
            "Detected" if live_host_data["virtual_camera_detected"] else "None Detected",
            delta="Physical Camera" if not live_host_data["virtual_camera_detected"] else "Injection Risk",
            delta_color="normal" if not live_host_data["virtual_camera_detected"] else "inverse",
        )

    st.write("---")

    card1, card2 = st.columns(2, gap="large")

    with card1:
        with st.container(border=True):
            st.subheader(":material/videocam: Enumerated Video Capture Devices")
            devices = live_host_data.get("devices", [])
            if devices:
                for idx, d in enumerate(devices):
                    icon = ":material/videocam_off:" if d["is_virtual"] else ":material/videocam:"
                    v_badge = "**[VIRTUAL LOOPBACK]**" if d["is_virtual"] else "**[PHYSICAL UVC]**"
                    st.markdown(f"{icon} `{d['node']}` — **{d['name']}** {v_badge}")
            else:
                st.info("No video devices discovered in `/sys/class/video4linux`.")

    with card2:
        with st.container(border=True):
            st.subheader(":material/terminal: Host Privileges & Hypervisor Diagnostics")
            vm_info = live_host_data.get("vm_info", {})
            priv = live_host_data.get("privileges", {})

            st.write(f"- **System Platform / DMI:** `{vm_info.get('vendor', 'Unknown')}`")
            st.write(f"- **Containerized Environment:** `{vm_info.get('is_container', False)}`")
            st.write(f"- **Process Effective UID:** `{priv.get('uid', 'N/A')}` (`is_root={priv.get('is_root', False)}`)")
            st.write(f"- **SU Binary Presence:** `{priv.get('su_present', False)}`")

    st.write("---")
    st.subheader(":material/phone_android: Mobile Client Attestation Payload Tester")
    st.caption("Paste JSON payload generated by Aarya's Android Attestation SDK (`DetectionResult.kt`):")

    sample_client_json = (
        '{\n  "isRooted": false,\n  "isEmulator": false,\n  "isVirtualCamera": false,\n'
        '  "riskScore": 0.05,\n  "details": "Client hardware verified clean via Android Attestation SDK."\n}'
    )
    user_json = st.text_area("Client Attestation JSON Payload:", value=sample_client_json, height=120)

    if st.button("Evaluate Client Attestation Payload"):
        client_eval = host_integrity.evaluate_client_attestation(user_json)
        st.json(client_eval)
        if client_eval["blocked"]:
            st.error(f"Gate 1 Verdict: BLOCK — {client_eval['details']}")
        elif client_eval["virtual_camera_detected"]:
            st.warning(f"Gate 1 Verdict: FLAG FOR REVIEW — {client_eval['details']}")
        else:
            st.success(f"Gate 1 Verdict: PASS — {client_eval['details']}")