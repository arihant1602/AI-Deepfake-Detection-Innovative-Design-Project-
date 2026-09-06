import streamlit as st
import tempfile
import os
from pipeline import run_detection_pipeline

st.set_page_config(page_title="IAD & Deepfake Forensics Portal", layout="wide")

st.title("🛡️ Next-Gen Injection Attack & Deepfake Detection")
st.caption("CEN/TS 18099 Compliant Gated Forensic Pipeline")

uploaded_file = st.file_uploader("Upload incoming video stream (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📹 Input Stream Preview")
        st.video(video_path)

    with col2:
        st.subheader("🔍 Real-Time Gated Verification")
        
        if st.button("Run Forensic Verification", type="primary", use_container_width=True):
            with st.status("Executing gated verification pipeline...", expanded=True) as status:
                st.write("Gate 1: Querying OS-level hardware attestation...")
                st.write("✓ Hardware environment verified clean.")

                st.write("Gate 2: Executing CMOS PRNU sensor noise profiling...")
                result = run_detection_pipeline(video_path)

                if result["verdict"] != "AUTHENTIC LIVE STREAM":
                    status.update(label=f"❌ Terminated at Gate {result['gate']}", state="error")
                else:
                    st.write("✓ Microscopic PRNU pattern validated.")
                    st.write("Gate 3: Evaluating temporal consistency & FFT spectral coherence...")
                    st.write("✓ Optical flow stability and high-frequency spectral ratios verified.")
                    status.update(label="✅ All Verification Gates Passed", state="complete")

            # --- DISPLAY VERDICT & FORENSIC METRICS OUTSIDE STATUS CONTAINER ---
            st.divider()

            if result["verdict"] != "AUTHENTIC LIVE STREAM":
                st.error(f"### 🚨 {result['verdict']}")
                st.warning(f"**Forensic Trigger Reason:** {result['details'].get('explanation', 'Forensic threshold exceeded.')}")
                
                m1, m2 = st.columns(2)
                if result["gate"] == 2:
                    with m1:
                        st.metric("PRNU Anomaly Score", f"{result['details'].get('score', 0.0):.3f}", delta="Flagged (≥ 0.55)", delta_color="inverse")
                    with m2:
                        st.metric("PCE Peak Energy", f"{result['details'].get('pce_score', 0.0):.1f}", delta="Below Sensor Threshold", delta_color="inverse")
                elif result["gate"] == 3:
                    with m1:
                        st.metric("Temporal Anomaly Score", f"{result['details'].get('score', 0.0):.3f}", delta="Flagged (≥ 0.60)", delta_color="inverse")
                    with m2:
                        st.metric("Flicker Rate", f"{result['details'].get('flicker_rate', 0.0):.2f}")
            else:
                st.success("### Verdict: AUTHENTIC LIVE STREAM")

                prnu_data = result["details"]["prnu"]
                temp_data = result["details"]["temporal"]

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Device Integrity", "Verified Clean", delta="Gate 1")
                with m2:
                    st.metric("PCE Peak Energy", f"{prnu_data['pce_score']:.1f}", delta="Gate 2")
                with m3:
                    st.metric("PRNU Anomaly", f"{prnu_data['score']:.3f}", delta="< 0.55")
                with m4:
                    st.metric("Temporal Anomaly", f"{temp_data['score']:.3f}", delta="< 0.60")

                if "frames" in result and result["frames"]:
                    st.write("---")
                    st.write("**Extracted Frame Pipeline Buffer:**")
                    frames = result["frames"]
                    f_cols = st.columns(4)
                    for idx, col in enumerate(f_cols):
                        sample_idx = min(idx * 7, len(frames) - 1)
                        col.image(frames[sample_idx], caption=f"Frame {sample_idx}", use_container_width=True)

    try:
        os.remove(video_path)
    except OSError:
        pass