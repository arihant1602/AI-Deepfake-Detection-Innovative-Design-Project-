"""
Web server for the gated injection & deepfake detector: a JSON API over live_engine.py and
pipeline.py, the live MJPEG stream, and the React front end built into web/dist.

    cd web && npm install && npm run build && cd ..
    python server.py                      # http://localhost:8501

During front-end development run `npm run dev` in web/ as well; Vite proxies /api here.

A live session is identified by its engine's stream token, a per-session secret returned by
/api/live/start. Every live call (state, controls, video) needs it, so only the browser that
started the camera can see or control it. The camera is released on Stop, when the page sends
its unload beacon, or when the page stops polling for live_engine.HEARTBEAT_TIMEOUT seconds.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import os
import secrets
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import host_integrity
import live_engine
from live_engine import INJECTIONS, NOISE_T, LiveEngine
from pipeline import run_detection_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, "web", "dist")
FINGERPRINT_DIR = os.path.join(ROOT, "fingerprints")   # CameraFingerprintStore's default root
SAMPLES = ("real_sim.mp4", "fake_sim.mp4")
VIDEO_TYPES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
BOUNDARY = "frame"
POLL_S = 0.004       # how often to look for a new camera frame
JPEG_QUALITY = 80

_sessions: dict[str, LiveEngine] = {}   # stream token -> engine
_sessions_lock = threading.Lock()


def jsonable(v):
    """Pipeline results mix numpy scalars/arrays into plain dicts; make them JSON-safe."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items() if k != "raw_result"}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 4096 else None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return None   # model objects and other internals are not part of the API


def ok(data) -> JSONResponse:
    return JSONResponse(jsonable(data))


def _engine(token: str) -> LiveEngine:
    with _sessions_lock:
        eng = next((e for t, e in _sessions.items() if secrets.compare_digest(t, token)), None)
    if eng is None:
        raise HTTPException(404, "No live session with this token. Press Start again.")
    return eng


async def _body(request: Request) -> dict:
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- #
# Live
# --------------------------------------------------------------------------- #

def _device_row(d: dict) -> dict:
    kind = ("software camera" if d["is_virtual"] else "physical camera" if d["is_physical"]
            else "metadata" if d["is_metadata"] else "not a camera")
    return {"node": d["node"], "name": d["name"], "kind": kind, "is_virtual": d["is_virtual"],
            "driver": d["driver"], "bus": d["bus_info"] or d["bus"], "camera_id": d["camera_id"],
            "why": "; ".join(d["virtual_reasons"])}


async def meta(request):
    devices = await asyncio.to_thread(host_integrity.enumerate_video_devices, capture_only=True)
    return ok({"devices": [_device_row(d) for d in devices], "injections": INJECTIONS, "noise_threshold": NOISE_T,
               "samples": [s for s in SAMPLES if os.path.exists(os.path.join(ROOT, s))],
               "presets": list(host_integrity.SIMULATION_PRESETS),
               "challenge_interval": live_engine.CHALLENGE_INTERVAL})


async def live_start(request):
    node = (await _body(request)).get("node", "")
    capture_nodes = {d["node"] for d in await asyncio.to_thread(host_integrity.enumerate_video_devices,
                                                               capture_only=True)}
    if node not in capture_nodes:
        raise HTTPException(400, f"{node or 'No camera'} is not a video-capture device on this machine.")
    eng = LiveEngine(node)
    with _sessions_lock:
        # One session per camera: a new start replaces the previous one (start() stops its engine).
        for t in [t for t, e in _sessions.items() if e.node == node]:
            del _sessions[t]
        _sessions[eng.stream_token] = eng
    await asyncio.to_thread(eng.start)
    return ok({"token": eng.stream_token, "node": node, "width": eng.width, "height": eng.height})


def _snapshot(eng: LiveEngine) -> dict:
    s = eng.snapshot()
    hw = s.pop("hw") or {}
    s["hw"] = {k: hw.get(k) for k in ("verdict", "details", "latency_ms")}
    # Only the latest challenge's last attempt is drawn; older raw traces stay on the server.
    chs = []
    for i, (t, ch) in enumerate(s["challenges"]):
        ch = {k: v for k, v in ch.items() if k != "attempts"} | (
            {"attempt": ch["attempts"][-1]} if i == 0 and ch.get("attempts") else {})
        chs.append({"t": t, **ch})
    s["challenges"] = chs
    s["noise"] = [{"t": t, "sigma": v, "verdict": verdict, "failed": failed} for t, v, verdict, failed in s["noise"]]
    s["deepfake"] = [{"t": t, "p": p, "abstained": a} for t, p, a in s["deepfake"]]
    s["events"] = [{"t": t, "level": lvl, "text": txt} for t, lvl, txt in s["events"]]
    s["node"] = eng.node
    s["now"] = time.time()   # the client draws "seconds ago" against server time
    return s


async def live_state(request):
    eng = _engine(request.query_params.get("token", ""))
    eng.heartbeat()
    return ok(_snapshot(eng))


async def live_stop(request):
    # Also reached by navigator.sendBeacon on page unload, which posts text/plain.
    token = request.query_params.get("token") or (await _body(request)).get("token", "")
    _engine(token).stop("stopped by user")
    return ok({"stopped": True})


async def live_challenge(request):
    eng = _engine((await _body(request)).get("token", ""))
    eng.request_challenge()
    return ok({"requested": True})


async def live_attack(request):
    body = await _body(request)
    eng, mode = _engine(body.get("token", "")), body.get("mode", "none")
    if mode not in INJECTIONS:
        raise HTTPException(400, f"Unknown attack '{mode}'.")
    if eng.running:
        await asyncio.to_thread(eng.set_injection, mode)
    return ok({"injection": eng.injection})


async def live_mjpeg(request):
    try:
        eng = _engine(request.query_params.get("token", ""))
    except HTTPException:
        return Response("No live session with this token.", status_code=404)

    async def frames():
        last_t = 0.0
        while eng.running and live_engine.get_engine(eng.node) is eng:
            frame, t = eng.frame()
            if frame is None or t == last_t:
                await asyncio.sleep(POLL_S)
                continue
            last_t = t
            ok_, jpg = await asyncio.to_thread(
                lambda: cv2.imencode(".jpg", eng.annotated(frame), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))
            if ok_:
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
                       + jpg.tobytes() + b"\r\n")

    return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------- #
# File analysis
# --------------------------------------------------------------------------- #

def _thumbs(frames: list, n: int = 4) -> list[dict]:
    out = []
    for i in range(min(n, len(frames))):
        idx = min(i * max(1, len(frames) // n), len(frames) - 1)
        ok_, jpg = cv2.imencode(".jpg", cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok_:
            out.append({"index": idx, "src": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()})
    return out


def _save_upload(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        return f.name


async def analyze(request):
    form = await request.form()
    temps = []
    try:
        sample, upload = form.get("sample"), form.get("video")
        if upload is not None and hasattr(upload, "read"):
            suffix = os.path.splitext(upload.filename or "")[1].lower() or ".mp4"
            if suffix not in VIDEO_TYPES:
                raise HTTPException(400, f"{suffix} files are not supported. Use {', '.join(VIDEO_TYPES)}.")
            video_path = _save_upload(await upload.read(), suffix)
            temps.append(video_path)
        elif sample in SAMPLES and os.path.exists(os.path.join(ROOT, sample)):
            video_path = os.path.join(ROOT, sample)
        else:
            raise HTTPException(400, "Choose a video file or a sample clip.")

        ref, ref_path = form.get("reference"), None
        if ref is not None and hasattr(ref, "read"):
            ref_path = _save_upload(await ref.read(), ".npy")
            temps.append(ref_path)
        try:
            max_frames = int(form.get("max_frames") or 60)
        except ValueError:
            max_frames = 60
        max_frames = min(max(max_frames, 16), 300)
        sim = form.get("attestation") or ""
        mode = sim if sim in host_integrity.SIMULATION_PRESETS else "file_upload"

        result = await asyncio.to_thread(run_detection_pipeline, video_path, attestation_mode=mode,
                                         ref_fingerprint_path=ref_path, max_sample_frames=max_frames)
        thumbs = _thumbs(result.get("frames") or [])
        result = {k: v for k, v in result.items() if k not in ("frames", "frames_bgr")}
        for d in (result.get("details") or {}, (result.get("details") or {}).get("attestation") or {},
                  result.get("attestation") or {}):
            for k in ("devices", "camera_audit", "hardware_telemetry"):
                d.pop(k, None)
        return ok({**result, "thumbs": thumbs})
    finally:
        for p in temps:
            try:
                os.unlink(p)
            except OSError:
                pass


async def sample_video(request):
    name = request.path_params["name"]
    path = os.path.join(ROOT, name)
    if name not in SAMPLES or not os.path.exists(path):
        raise HTTPException(404, "No such sample.")
    return FileResponse(path, media_type="video/mp4")


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #

async def device(request):
    hw = await asyncio.to_thread(host_integrity.probe_host_integrity)
    prints = sorted(f[:-5] for f in os.listdir(FINGERPRINT_DIR) if f.endswith(".json")) \
        if os.path.isdir(FINGERPRINT_DIR) else []
    return ok({"verdict": hw["verdict"], "details": hw["details"], "latency_ms": hw["latency_ms"],
               "devices": [_device_row(d) for d in hw["devices"]],
               "diagnostics": hw["diagnostics_summary"], "fingerprints": prints})


async def attestation(request):
    payload = (await _body(request)).get("payload", "")
    if not isinstance(payload, str) or not payload.strip().startswith("{"):
        raise HTTPException(400, "Paste the JSON object the Android SDK produced.")
    r = host_integrity.evaluate_client_attestation(payload)
    return ok({"verdict": r["verdict"], "details": r["details"]})


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

async def http_error(request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


class SPAFiles(StaticFiles):
    """Serves web/dist, falling back to index.html for client-side routes."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as e:
            if e.status_code != 404 or path.startswith("api"):
                raise
            return await super().get_response("index.html", scope)


def _preload():
    from pipeline import get_deepfake_detector
    get_deepfake_detector()
    # Move the model's long-lived objects out of the collector's view, so Python's own
    # periodic full collections stay short and never stall the capture thread.
    gc.freeze()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_preload, daemon=True, name="preload-gend").start()
    yield
    live_engine._shutdown_all()


routes = [
    Route("/api/meta", meta),
    Route("/api/live/start", live_start, methods=["POST"]),
    Route("/api/live/state", live_state),
    Route("/api/live/stop", live_stop, methods=["POST"]),
    Route("/api/live/challenge", live_challenge, methods=["POST"]),
    Route("/api/live/attack", live_attack, methods=["POST"]),
    Route("/api/live.mjpg", live_mjpeg),
    Route("/api/analyze", analyze, methods=["POST"]),
    Route("/api/samples/{name}", sample_video),
    Route("/api/device", device),
    Route("/api/attestation", attestation, methods=["POST"]),
]
if os.path.isdir(DIST):
    routes.append(Mount("/", SPAFiles(directory=DIST, html=True)))

app = Starlette(routes=routes, lifespan=lifespan, exception_handlers={HTTPException: http_error})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the detector web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not os.path.isdir(DIST):
        print("web/dist is missing: run `npm install && npm run build` in web/ (or `npm run dev` for development).")
    uvicorn.run(app, host=args.host, port=args.port)
