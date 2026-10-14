"""
Server entry point for the Streamlit app (the UI itself is streamlit_app.py).

    streamlit run app.py

Adds two things a plain Streamlit script cannot do:
  * /api/live.mjpg?node=/dev/videoN  streams the live camera as MJPEG straight from the
    engine's frame buffer, at the camera's frame rate. Pushing frames through Streamlit
    reruns (st.image in a fragment) capped the video at a few fps and stalled whenever a
    chart fragment was rendering.
  * the deepfake model starts loading in the background as soon as the server starts, so
    the first live session does not wait for it.
"""

import asyncio
import gc
import os
import threading
from contextlib import asynccontextmanager

import cv2
import streamlit as st
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

import live_engine

BOUNDARY = "frame"
POLL_S = 0.004       # how often to look for a new camera frame
JPEG_QUALITY = 80


async def live_mjpeg(request):
    node = request.query_params.get("node", "")
    if live_engine.get_engine(node) is None:
        return Response("No live session on this camera.", status_code=404)

    async def frames():
        last_t = 0.0
        while True:
            eng = live_engine.get_engine(node)
            if eng is None or not eng.running:
                return
            frame, t = eng.frame()
            if frame is None or t == last_t:
                await asyncio.sleep(POLL_S)
                continue
            last_t = t
            ok, jpg = await asyncio.to_thread(
                lambda: cv2.imencode(".jpg", eng.annotated(frame), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))
            if ok:
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
                       + jpg.tobytes() + b"\r\n")

    return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers={"Cache-Control": "no-store"})


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


app = st.App(os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py"),
             routes=[Route("/api/live.mjpg", live_mjpeg)], lifespan=lifespan)
