import os

import streamlit as st

import host_integrity
from pipeline import DEFAULT_FINGERPRINT_DIR
from ui import badge

st.title("This device")
st.caption("What the device check sees on this machine right now.")

hw = host_integrity.probe_host_integrity()
row = st.container(horizontal=True, vertical_alignment="center", gap="small")
with row:
    badge({"PASS": "PASS", "FLAG_FOR_REVIEW": "FLAG"}.get(hw["verdict"], "BLOCK"), hw["verdict"].replace("_", " ").lower())
    st.markdown(hw["details"])
st.caption(f"Checked in {hw['latency_ms']:.1f} ms")

st.subheader("Video devices")
st.dataframe(
    [{"node": d["node"], "name": d["name"],
      "kind": "software camera" if d["is_virtual"] else ("physical camera" if d["is_physical"] else
              ("metadata" if d["is_metadata"] else "not a camera")),
      "driver": d["driver"], "bus": d["bus_info"] or d["bus"], "identity": d["camera_id"],
      "why": "; ".join(d["virtual_reasons"])} for d in hw["devices"]],
    hide_index=True, width="stretch")

st.subheader("Environment")
st.dataframe([{"check": d["name"], "result": d["status"], "value": d["value"]} for d in hw["diagnostics_summary"]],
             hide_index=True, width="stretch")

st.subheader("Enrolled fingerprints")
files = sorted(f[:-5] for f in os.listdir(DEFAULT_FINGERPRINT_DIR) if f.endswith(".json")) \
    if os.path.isdir(DEFAULT_FINGERPRINT_DIR) else []
if files:
    st.dataframe([{"camera and mode": f} for f in files], hide_index=True, width="stretch")
else:
    st.caption("None yet. Press Enroll camera on the Live page.")

with st.expander("Check an Android client attestation"):
    payload = st.text_area("Payload (JSON from the Android SDK)", height=130,
                           value='{\n  "isRooted": false,\n  "isEmulator": false,\n  "isVirtualCamera": false\n}')
    if st.button("Evaluate"):
        r = host_integrity.evaluate_client_attestation(payload)
        {"BLOCK": st.error, "FLAG_FOR_REVIEW": st.warning}.get(r["verdict"], st.success)(f"{r['verdict']}: {r['details']}")
