import os
import tempfile

import streamlit as st

import host_integrity
from pipeline import run_detection_pipeline
from ui import render_result

st.session_state.setdefault("file_result", None)

st.title("Analyze a video file")
st.caption("A file has no live camera behind it, so the device and live-sensor checks cannot run. The fingerprint "
           "check only decides when you supply the claimed camera's enrolled fingerprint.")

source = st.segmented_control("Source", ["Upload", "Sample clips"], default="Upload", required=True)
video_path = None
if source == "Upload":
    up = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"], label_visibility="collapsed")
    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1] or ".mp4") as f:
            f.write(up.getbuffer())
            video_path = f.name
else:
    sample = st.pills("Sample", ["real_sim.mp4", "fake_sim.mp4"], default="real_sim.mp4", required=True,
                      label_visibility="collapsed")
    if os.path.exists(sample):
        video_path = sample
    else:
        st.warning(f"`{sample}` is missing. Run `python generate_test_videos.py`.")

with st.expander("Options"):
    ref_up = st.file_uploader("Reference fingerprint (.npy)", type=["npy"])
    max_frames = st.select_slider("Frames to analyse", options=[45, 60, 90, 150], value=60)
    sim = st.selectbox("Attach a simulated device attestation",
                       ["None (unattested file)"] + list(host_integrity.SIMULATION_PRESETS))

if video_path:
    left, right = st.columns([1, 1], gap="medium")
    left.video(video_path)
    with right:
        if st.button("Analyze", type="primary", icon=":material/play_arrow:"):
            ref_path = None
            if ref_up is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f:
                    f.write(ref_up.getbuffer())
                    ref_path = f.name
            with st.spinner("Analyzing..."):
                st.session_state["file_result"] = run_detection_pipeline(
                    video_path, attestation_mode="file_upload" if sim.startswith("None") else sim,
                    ref_fingerprint_path=ref_path, max_sample_frames=max_frames)
        if st.session_state["file_result"]:
            render_result(st.session_state["file_result"])
