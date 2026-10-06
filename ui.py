"""
Shared presentation helpers for the Streamlit app: status badges, the gate strip,
and the verdict / details view used by every page that shows a pipeline result.
"""

from __future__ import annotations

import cv2
import streamlit as st

from pipeline import VERDICT_AUTHENTIC, VERDICT_NO_ANOMALY

STATUS_STYLE = {
    "PASS": ("green", ":material/check:"),
    "BLOCK": ("red", ":material/close:"),
    "FLAG": ("orange", ":material/priority_high:"),
    "ADVISORY": ("gray", None),
    "ABSTAINED": ("gray", None),
    "SKIPPED": ("gray", None),
    "N/A": ("gray", None),
}

GATE_NAMES = {
    "1a": "Real camera",
    "1b": "Live sensor",
    "2": "Sensor noise",
    "3": "Face (deepfake model)",
}


def badge(status: str, label: str | None = None):
    color, icon = STATUS_STYLE.get(status, ("gray", None))
    st.badge(label or status.lower(), color=color, icon=icon)


def render_strip(stages: list[dict]):
    """One compact card per gate, left to right; skipped gates make early termination visible."""
    row = st.container(horizontal=True, gap="small")
    for s in stages:
        with row.container(border=True, width="stretch"):
            st.caption(s["gate"])
            st.markdown(f"**{s['name']}**")
            badge(s["status"], s.get("label"))
            meta = " · ".join(x for x in (s.get("value"), f"{s['ms']:,.0f} ms" if s.get("ms") is not None else None) if x)
            if meta:
                st.caption(meta)


def stages_from_result(result: dict) -> list[dict]:
    """Converts a pipeline result into strip stages: Gate 1a, 1b, 2, 3."""
    gate, details, t = result.get("gate"), result.get("details", {}) or {}, result.get("timings", {})
    hw = details if gate == 1 else (details.get("attestation") or result.get("attestation") or {})
    ch = hw.get("challenge") or {}
    challenge_failed = ch.get("verdict") == "FAIL"

    if gate == 1 and not challenge_failed:
        g1a = {"status": "BLOCK", "label": "blocked"}
    else:
        v = hw.get("verdict", "PASS")
        g1a = {"status": {"FLAG_FOR_REVIEW": "FLAG", "UNATTESTED_ORIGIN": "N/A"}.get(v, "PASS"),
               "label": {"FLAG_FOR_REVIEW": "review", "UNATTESTED_ORIGIN": "not checked (file)"}.get(v, "physical camera")}
    g1a.update(gate="Gate 1a", name=GATE_NAMES["1a"], ms=t.get("gate1_passive_ms"))

    if gate == 1 and not challenge_failed:
        g1b = {"status": "SKIPPED", "label": "not run"}
    elif ch.get("verdict") == "PASS":
        g1b = {"status": "PASS", "label": "responded", "value": f"r = {ch.get('corr', 0):.2f}"}
    elif challenge_failed:
        g1b = {"status": "BLOCK", "label": "did not respond", "value": f"r = {ch.get('corr', 0):.2f}"}
    else:
        g1b = {"status": "N/A", "label": "not possible" if not ch else "unsupported"}
    g1b.update(gate="Gate 1b", name=GATE_NAMES["1b"], ms=t.get("gate1_challenge_ms"))

    prnu = details.get("prnu") or result.get("prnu") or (details if gate == 2 else None)
    if gate == 1 or not prnu:
        g2 = {"status": "SKIPPED", "label": "not run"}
    elif prnu.get("mode") == "reference" and prnu.get("verdict") in ("PRESENT", "ABSENT"):
        ok = prnu.get("verdict") == "PRESENT"
        g2 = {"status": "PASS" if ok else "BLOCK", "label": "match" if ok else "no match",
              "value": f"PCE {prnu.get('pce_score'):,.0f}"}
    elif prnu.get("verdict") == "PRESENT":
        g2 = {"status": "PASS", "label": "live noise", "value": f"level {prnu.get('noise_sigma') or 0:.2f}"}
    elif prnu.get("verdict") == "ABSENT":
        g2 = {"status": "BLOCK" if prnu.get("enforced") else "ADVISORY",
              "label": "no live noise" + ("" if prnu.get("enforced") else " (advisory)"),
              "value": ", ".join(prnu.get("failed_checks") or [])}
    else:
        g2 = {"status": "ADVISORY", "label": "inconclusive"}
    g2.update(gate="Gate 2", name=GATE_NAMES["2"], ms=t.get("gate2_ms"))

    temporal = details.get("temporal") or (details if gate == 3 else None)
    if gate in (1, 2) or not temporal:
        g3 = {"status": "SKIPPED", "label": "not run"}
    elif gate == 3:
        g3 = {"status": "BLOCK", "label": "synthetic face", "value": f"P(fake) {temporal.get('score'):.2f}"}
    elif temporal.get("abstained"):
        g3 = {"status": "ABSTAINED", "label": "no face"}
    else:
        g3 = {"status": "PASS", "label": "real face", "value": f"P(fake) {temporal.get('score'):.2f}"}
    g3.update(gate="Gate 3", name=GATE_NAMES["3"], ms=t.get("gate3_ms"))
    return [g1a, g1b, g2, g3]


VERDICT_COPY = {
    VERDICT_AUTHENTIC: ("PASS", "Authentic live stream", "Every check ran and passed."),
    VERDICT_NO_ANOMALY: ("ADVISORY", "No attack detected", "Nothing failed, but not every check could run."),
}


def render_verdict(result: dict):
    verdict, gate = result.get("verdict", ""), result.get("gate")
    if verdict == "ERROR":
        st.error(result.get("details", {}).get("explanation", "The session could not run."))
        return
    status, title, line = VERDICT_COPY.get(
        verdict, ("BLOCK", verdict.capitalize(), f"Stopped at gate {gate}; later gates did not run."))
    with st.container(border=True):
        top = st.container(horizontal=True, vertical_alignment="center", gap="small")
        with top:
            badge(status, {"PASS": "authentic", "ADVISORY": "incomplete", "BLOCK": "attack"}[status])
            st.markdown(f"### {title}")
        st.caption(line)
        for lim in result.get("limitations", []):
            st.markdown(f"- {lim}")
        render_strip(stages_from_result(result))
        total = result.get("timings", {}).get("total_ms")
        if total:
            st.caption(f"Total {total / 1000:.1f} s, mostly spent capturing frames.")


def render_details(result: dict):
    """Per-gate details, collapsed by default."""
    details, gate = result.get("details", {}) or {}, result.get("gate")
    hw = details if gate == 1 else (details.get("attestation") or result.get("attestation") or {})
    prnu = details.get("prnu") or result.get("prnu") or (details if gate == 2 else {})
    temporal = details.get("temporal") or (details if gate == 3 else {})

    with st.expander("Gate 1: device and environment checks"):
        st.write(hw.get("details", ""))
        rows = [{"check": d["name"], "result": d["status"], "value": d["value"]} for d in hw.get("diagnostics_summary", [])]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        ch = hw.get("challenge")
        if ch:
            st.write(f"Sensor challenge ({ch.get('control')}): {ch.get('details')}")
        if hw.get("is_file_upload"):
            st.write(f"Container encoder: `{hw.get('encoder')}`")
    if prnu:
        with st.expander("Gate 2: sensor fingerprint"):
            st.write(prnu.get("explanation", ""))
            st.caption(f"Mode: {prnu.get('mode')} · {'enforced' if prnu.get('enforced') else 'advisory'} · "
                       f"{prnu.get('frames_analyzed', '?')} frames · {prnu.get('latency_ms', 0):.0f} ms")
    if temporal:
        with st.expander("Gate 3: face deepfake model"):
            st.write(temporal.get("explanation", ""))
            st.caption(f"{temporal.get('model', 'GenD')} · P(fake) {temporal.get('score')} (flagged at "
                       f"{temporal.get('threshold', 0.5)}) · face in {temporal.get('frames_with_face')} of "
                       f"{temporal.get('frames_analyzed')} sampled frames")
    frames = result.get("frames") or [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in result.get("frames_bgr", [])[:32]]
    if frames:
        with st.expander("Frames analysed"):
            row = st.container(horizontal=True, gap="small")
            for i in range(4):
                idx = min(i * max(1, len(frames) // 4), len(frames) - 1)
                row.image(frames[idx], caption=f"frame {idx}", alt=f"Analysed frame {idx}", width=220)


def render_result(result: dict):
    render_verdict(result)
    if result.get("verdict") != "ERROR":
        render_details(result)
