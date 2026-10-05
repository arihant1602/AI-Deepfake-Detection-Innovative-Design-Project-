import time

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import host_integrity
from live_engine import INJECTIONS, PCE_T, TEMPORAL_T, LiveEngine
from ui import badge

BLUE, ORANGE, RED, GREY = "#2a78d6", "#eb6834", "#e34948", "#9aa1ad"
HISTORY_S = 60

eng: LiveEngine | None = st.session_state.get("engine")
running = bool(eng and eng.running)


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #

def start_engine(node: str):
    old = st.session_state.get("engine")
    if old is not None:
        old.stop("restarted")
    e = LiveEngine(node)
    e.start()
    st.session_state["engine"] = e
    st.session_state["attack"] = "none"


def stop_engine():
    e = st.session_state.get("engine")
    if e is not None:
        e.stop("stopped by user")


def change_attack():
    e = st.session_state.get("engine")
    if e is not None and e.running:
        e.set_injection(st.session_state["attack"])


devices = [d for d in host_integrity.enumerate_video_devices(capture_only=True)]
labels = {d["node"]: f"{d['name']} ({d['node']})" + (" · software camera" if d["is_virtual"] else "") for d in devices}

bar = st.container(horizontal=True, vertical_alignment="bottom", gap="small")
bar.markdown("### Live verification", width="content")
if devices:
    node = bar.selectbox("Camera", list(labels), format_func=labels.get, width=300, disabled=running,
                         label_visibility="collapsed")
else:
    node = None
    bar.caption("No video-capture device found.")
if running:
    bar.button("Stop", icon=":material/stop:", on_click=stop_engine, type="primary")
else:
    bar.button("Start", icon=":material/play_arrow:", on_click=start_engine, args=(node,), type="primary",
               disabled=node is None)
bar.button("Check liveness now", disabled=not running, on_click=lambda: eng.request_challenge())
bar.button("Enroll camera", disabled=not running, on_click=lambda: eng.request_enroll(),
           help="Learns this camera's sensor fingerprint from the next 5 s. Move the camera slowly while it runs.")
bar.selectbox("Simulate an attack", list(INJECTIONS), format_func=INJECTIONS.get, key="attack",
              on_change=change_attack, disabled=not running, width=270,
              help="Replaces the camera frames after the driver, like a hooked capture call, while the real camera "
                   "keeps receiving the liveness pattern.")

left, right = st.columns([10, 11], gap="medium")


# --------------------------------------------------------------------------- #
# Status logic
# --------------------------------------------------------------------------- #

def gate_states(s: dict) -> list[dict]:
    hw = s["hw"] or {}
    v = hw.get("verdict", "PASS")
    g1a = {"gate": "1a", "name": "Real camera", "status": {"PASS": "PASS", "FLAG_FOR_REVIEW": "FLAG"}.get(v, "BLOCK"),
           "label": {"PASS": "physical", "FLAG_FOR_REVIEW": "review"}.get(v, "blocked"),
           "value": f"`{s['camera_id']}`" if s["camera_id"] else ""}

    if s["challenges"]:
        t, ch = s["challenges"][0]
        ok = ch["verdict"] == "PASS"
        g1b = {"status": "PASS" if ok else ("BLOCK" if ch["verdict"] == "FAIL" else "N/A"),
               "label": "live" if ok else ("not live" if ch["verdict"] == "FAIL" else "unsupported"),
               "value": f"r = {ch.get('corr') or 0:.2f} · {time.time() - t:.0f} s ago"}
    else:
        g1b = {"status": "ABSTAINED", "label": "first check…", "value": ""}
    if s["in_challenge"]:
        g1b["label"], g1b["status"] = "checking now", "FLAG"
    g1b.update(gate="1b", name="Live sensor")

    if s["enrolling"]:
        g2 = {"status": "FLAG", "label": f"enrolling {s['enroll_progress']:.0%}", "value": "keep moving the camera"}
    elif not s["enrolled"]:
        g2 = {"status": "ABSTAINED", "label": "not enrolled", "value": "press Enroll camera"}
    elif s["pce"]:
        p = s["pce"][-1][1]
        g2 = {"status": "PASS" if p >= PCE_T else "BLOCK", "label": "match" if p >= PCE_T else "no match",
              "value": f"score {p:,.0f}"}
        if PCE_T <= p < 3 * PCE_T:
            g2.update(status="FLAG", label="weakening")
    else:
        g2 = {"status": "ABSTAINED", "label": "warming up", "value": ""}
    g2.update(gate="2", name="Sensor fingerprint")

    if s["temporal"]:
        _, sc, abst = s["temporal"][-1]
        if abst:
            g3 = {"status": "ABSTAINED", "label": "no face", "value": "face the camera"}
        else:
            g3 = {"status": "BLOCK" if sc >= TEMPORAL_T else ("FLAG" if sc >= 0.75 * TEMPORAL_T else "PASS"),
                  "label": "anomaly" if sc >= TEMPORAL_T else ("rising" if sc >= 0.75 * TEMPORAL_T else "consistent"),
                  "value": f"score {sc:.2f}"}
    else:
        g3 = {"status": "ABSTAINED", "label": "warming up", "value": ""}
    g3.update(gate="3", name="Motion")
    return [g1a, g1b, g2, g3]


def overall(states: list[dict]) -> tuple[str, str, str]:
    blocked = [g for g in states if g["status"] == "BLOCK"]
    if blocked:
        return "BLOCK", "Attack detected", " · ".join(f"{g['name']}: {g['label']}" for g in blocked)
    if all(g["status"] == "PASS" for g in states):
        return "PASS", "Authentic live stream", "Every check is passing right now."
    waiting = [f"{g['name']}: {g['label']}" for g in states if g["status"] != "PASS"]
    return "FLAG" if any(g["status"] == "FLAG" for g in states) else "ABSTAINED", "Watching", " · ".join(waiting)


# --------------------------------------------------------------------------- #
# Live video (fast refresh)
# --------------------------------------------------------------------------- #

@st.fragment(run_every=0.15 if running else None)
def video():
    e: LiveEngine | None = st.session_state.get("engine")
    if e is not None and not e.running and running:
        st.rerun(scope="app")  # the engine stopped itself: switch the whole page to idle (stops polling)
    if e is None or not e.running:
        blank = np.full((480, 640, 3), 24, np.uint8)
        msg = "Camera off" if e is None else f"Camera off: {e.stop_reason or 'stopped'}"
        cv2.putText(blank, msg, (24, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2, cv2.LINE_AA)
        st.image(blank, channels="BGR", output_format="JPEG", width="stretch", alt="Camera off")
        return
    e.heartbeat()
    frame, _ = e.frame()
    if frame is None:
        return
    img = frame.copy()
    hgt, wid = img.shape[:2]
    if e.face_box is not None:
        x, y, w, hh = e.face_box
        cv2.rectangle(img, (x, y), (x + w, y + hh), (214, 120, 42), 2)
    if e.in_challenge:
        cv2.rectangle(img, (2, 2), (wid - 3, hgt - 3), (52, 104, 235), 6)
        cv2.putText(img, "LIVENESS CHECK", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (52, 104, 235), 2, cv2.LINE_AA)
    if e.injection != "none":
        cv2.rectangle(img, (0, hgt - 40), (wid, hgt), (40, 40, 200), -1)
        cv2.putText(img, f"SIMULATED ATTACK: {INJECTIONS[e.injection]}", (14, hgt - 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{e.fps:.0f} fps", (wid - 86, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    st.image(img, channels="BGR", output_format="JPEG", width="stretch", alt="Live camera")


# --------------------------------------------------------------------------- #
# Gate tiles and graphs (1 s refresh)
# --------------------------------------------------------------------------- #

@st.fragment(run_every=1.0 if running else None)
def tiles():
    e: LiveEngine | None = st.session_state.get("engine")
    if e is None:
        st.caption("Press **Start**. The camera stays on only while this page is open and running.")
        return
    s = e.snapshot()
    row = st.container(horizontal=True, gap="small")
    for g in gate_states(s):
        with row.container(border=True, width="stretch"):
            st.caption(f"Gate {g['gate']} · {g['name']}")
            badge(g["status"], g["label"])
            st.caption(g["value"] or " ")
    if s["running"]:
        st.caption(f"Next liveness check in {s['next_challenge_in']:.0f} s · the picture flashes briefly during it")


def timeline(points, value_label, threshold, log=False, height=118):
    now = time.time()
    df = pd.DataFrame(points, columns=["t", "v"] + (["abst"] if points and len(points[0]) == 3 else []))
    if df.empty:
        st.caption("Waiting for data…")
        return
    df["ago"] = df["t"] - now
    df = df[df["ago"] >= -HISTORY_S]
    if log:
        df["v"] = df["v"].clip(lower=0.5)
    df["state"] = np.where(df["v"] >= threshold, "above", "below") if not log else np.where(df["v"] >= threshold, "ok", "fail")
    x = alt.X("ago:Q", title=None, scale=alt.Scale(domain=[-HISTORY_S, 0]),
              axis=alt.Axis(values=[-60, -45, -30, -15, 0], labelExpr="datum.value == 0 ? 'now' : datum.value + ' s'"))
    yscale = alt.Scale(type="log", domain=[0.5, max(5000.0, float(df["v"].max()) * 1.5)]) if log else alt.Scale(domain=[0, 1])
    y = alt.Y("v:Q", title=value_label, scale=yscale, axis=alt.Axis(tickCount=3))
    line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2).encode(x=x, y=y)
    bad = (df["v"] < threshold) if log else (df["v"] >= threshold)
    dots = alt.Chart(df[bad]).mark_circle(color=RED, size=40, opacity=1).encode(x=x, y=y)
    rule = alt.Chart(pd.DataFrame({"v": [threshold]})).mark_rule(color=ORANGE, strokeDash=[4, 3]).encode(y="v:Q")
    layers = [rule, line, dots]
    if "abst" in df:
        layers.append(alt.Chart(df[df["abst"]]).mark_circle(color=GREY, size=22).encode(x=x, y=y))
    st.altair_chart(alt.layer(*layers).properties(height=height), width="stretch", alt=value_label)


@st.fragment(run_every=1.0 if running else None)
def panel():
    e: LiveEngine | None = st.session_state.get("engine")
    if e is None:
        with st.container(border=True):
            st.markdown("**How it works**")
            st.markdown(
                "Four checks run continuously on the live camera:\n\n"
                "- **1a Real camera**: the operating system confirms a physical USB/PCI camera, with no virtual camera, "
                "VM or hook.\n"
                "- **1b Live sensor**: every 10 s a random brightness pattern is sent to the camera; real frames follow it.\n"
                "- **2 Sensor fingerprint**: the frames must carry this camera's unique pixel-noise pattern.\n"
                "- **3 Motion**: the face's texture and motion must stay temporally consistent.\n\n"
                "Use **Simulate an attack** while it runs to replace the frames with a replay or an AI video and "
                "watch the checks react.")
        return
    s = e.snapshot()
    status, title, line = overall(gate_states(s))
    with st.container(border=True):
        head = st.container(horizontal=True, vertical_alignment="center", gap="small")
        with head:
            badge(status, {"PASS": "authentic", "BLOCK": "attack", "FLAG": "deviating", "ABSTAINED": "watching"}[status])
            st.markdown(f"#### {title}")
        st.caption(line if s["running"] else f"Stopped: {s['stop_reason']}")

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown("**Sensor fingerprint** · score (log), match above 60")
        if s["enrolled"]:
            timeline(s["pce"], "score", PCE_T, log=True)
        else:
            st.caption("Enroll the camera to see this graph.")
    with c2:
        st.markdown("**Motion** · anomaly score, flagged at 0.60")
        timeline(s["temporal"], "score", TEMPORAL_T)

    st.markdown("**Live sensor** · last brightness pattern sent (orange) vs frame brightness (blue)")
    if s["challenges"] and s["challenges"][0][1].get("attempts"):
        a = s["challenges"][0][1]["attempts"][-1]
        n = len(a["luma"])
        lo, hi = a["levels"]
        cmd = pd.DataFrame({"f": range(n), "v": [hi if c > 0 else lo for c in a["command"]]})
        lum = pd.DataFrame({"f": range(n), "v": a["luma"]})
        x = alt.X("f:Q", title=None, axis=None)
        c_cmd = alt.Chart(cmd).mark_line(interpolate="step-after", color=ORANGE, strokeWidth=2).encode(
            x=x, y=alt.Y("v:Q", axis=None, scale=alt.Scale(zero=False))).properties(height=34)
        c_lum = alt.Chart(lum).mark_line(color=BLUE, strokeWidth=2).encode(
            x=x, y=alt.Y("v:Q", title="brightness", scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=3))).properties(height=70)
        st.altair_chart(alt.vconcat(c_cmd, c_lum, spacing=2), width="stretch", alt="Last liveness pattern versus frame brightness")
    else:
        st.caption("First check runs a second after Start.")

    st.markdown("**Events**")
    with st.container(height=150, border=True):
        icons = {"block": ":red[■]", "warn": ":orange[■]", "ok": ":green[■]", "info": ":gray[■]"}
        for t, level, text in s["events"][:12]:
            st.markdown(f"{icons[level]} `{time.strftime('%H:%M:%S', time.localtime(t))}` {text}")


with left:
    video()
    tiles()
with right:
    panel()
