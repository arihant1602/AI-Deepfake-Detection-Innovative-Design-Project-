# AI Deepfake & Injection Attack Detection

*Innovative Design Project.* A gated, multi-layer pipeline that decides whether a video feed comes
from a **physical camera sensor** in an untampered environment, and whether its content is
temporally consistent. It runs on a laptop CPU in a few hundred milliseconds of compute per session,
with no learned model in Gates 1–2.

The full design, threat model, measured results and limitations are in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. That document is the authoritative reference.

## Gates

| Gate | Module | What it decides | Blocks when |
|---|---|---|---|
| 1a passive attestation | `host_integrity.py` | Is the capture node a physical USB/PCI camera? Is the environment clean? | software camera node (v4l2loopback, akvcam, vivid…), metadata node, VM/container, ptrace / `LD_PRELOAD` hook, root |
| 1b active sensor challenge | `host_integrity.py` | Do the frames respond to a random brightness challenge sent to that sensor? | partial correlation < 0.9 (frames not produced by the challenged sensor) |
| 2 sensor noise (PRNU) | `camera_sensor_noise_profiling.py` | Do the pixels carry the enrolled fingerprint of *the attested camera*? | PCE < 60 against the fingerprint enrolled for that camera ID + sensor mode |
| 3 temporal / spectral | `temporal_consistency_analysis.py` | Is facial texture, lighting and motion temporally consistent? | anomaly score ≥ 0.60 |
| 5 gated fusion | `pipeline.py`, `server.py` | Orchestration, early termination, final verdict | n/a |

Layer 4 (rPPG) is specified but not implemented. The Android SDK in `app/src/main/java/...` is the mobile
Layer 1 prototype; its JSON verdict can be evaluated with `host_integrity.evaluate_client_attestation`.

## Quickstart (Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python deepfake_detector.py --fetch    # once: GenD weights (Gate 3) into models/hf/; later loads are offline
cd web && npm install && npm run build && cd ..   # build the web UI (Node 20+)
python server.py                       # http://localhost:8501
```

Gates 1 and 2 have no learned weights. The app preloads the Gate 3 model in the background at startup
(~3 s on a GPU), and the live video is streamed as MJPEG from `/api/live.mjpg`, at the camera's frame rate.

For UI work, run `python server.py` and `npm run dev` in `web/` together: Vite serves the UI on
:5173 with hot reload and proxies `/api` to the server. The UI is React + TypeScript with plain CSS
(`web/src/styles/` for tokens and shared rules, one stylesheet beside each component and page).

The **Live** page runs every check continuously on the camera, on one screen: live video, the verdict,
a card per gate, the camera-grain and AI-face scores over the last minute, the latest flash test,
and an event log. Press **Start checking**. **Simulate an attack** swaps the frames for a replay, an AI
video or an animated photo mid-stream so you can watch the checks react. The camera is released on
**Stop checking**, when you leave the page, when the tab closes, or within 8 s of the page going quiet.
**Analyze a file** and **Device** are secondary pages.

Live session from the CLI:

```bash
python pipeline.py --live /dev/video0 --enroll      # once: enroll the camera's fingerprint (150 frames)
python pipeline.py --live /dev/video0               # verification session
python pipeline.py --video clip.mp4                 # uploaded file (origin unattested)
```

The camera must be accessible to the user (`video` group or logind ACL). Enrolled fingerprints are
stored in `fingerprints/` (git-ignored).

## Tests and benchmark

```bash
python run_tests.py                                  # 57 unit / integration tests, no camera needed
python benchmarks/fetch_datasets.py                  # VISION phone videos + DF40 deepfakes + text-to-video samples
python benchmarks/capture_webcam_sessions.py --sessions 4 --gap 30
python benchmarks/run_benchmark.py --live /dev/video0
```

Results are written to `benchmarks/results/BENCHMARK_RESULTS.md`.

## Repository layout

```
server.py                           Web server: JSON API, live MJPEG stream, model preload, serves web/dist
web/                                React + TypeScript UI (Vite); src/pages = Live, Analyze a file, Device
deepfake_detector.py                Gate 3: GenD face-deepfake model; --fetch caches weights in models/hf/
live_engine.py                      Real-time engine: capture + challenge thread, rolling analysis thread
pipeline.py                         Gated fusion: run_live_session, enroll_live_camera, run_detection_pipeline
host_integrity.py                   Gate 1: V4L2/sysfs attestation, host checks, active sensor challenge
camera_sensor_noise_profiling.py    Gate 2: PRNU residuals, fingerprints, reference PCE, blind motion-gated test
temporal_consistency_analysis.py    Gate 3: optical flow, HF spectral ratio, temporal FFT
generate_test_videos.py             Synthetic sanity-check clips (real_sim.mp4, fake_sim.mp4, camera_fingerprint.npy)
test_*.py, run_tests.py             Test suite
benchmarks/                         Dataset fetcher, webcam capture, benchmark runner, results
docs/ARCHITECTURE.md                Architecture, threat model, evaluation, limitations, paper notes
docs/LAYER*.md                      Per-layer notes from the original branches
app/src/main/                       Android Layer 1 prototype (Kotlin)
```

## Contributors

- **Aarya Jadhav**: Layer 1, Android device & environment attestation
- **Arihant**: Layer 2, camera sensor noise profiling
- **Vibha**: Layer 3, temporal consistency & frequency analysis
- **Ramya**: Layer 5, verification portal & gated fusion
