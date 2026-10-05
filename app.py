"""
Streamlit app for the gated injection & deepfake detector.

    streamlit run app.py

The live page (app_pages/live.py) runs every check continuously on the camera through
live_engine.py. The camera is released on Stop, when the page is left, or when the
browser stops polling.
"""

import streamlit as st

st.set_page_config(page_title="Live video verification", page_icon=":material/verified_user:", layout="wide")

live = st.Page("app_pages/live.py", title="Live", icon=":material/sensors:", default=True)
page = st.navigation(
    [
        live,
        st.Page("app_pages/analyze_file.py", title="Analyze a file", icon=":material/movie:"),
        st.Page("app_pages/device.py", title="Device", icon=":material/memory:"),
    ],
    position="top",
)

# Leaving the live page releases the camera immediately.
engine = st.session_state.get("engine")
if page.url_path != live.url_path and engine is not None and engine.running:
    engine.stop("left the live page")

page.run()
