# Live Video Authenticity Verification — Project Description

BACSE291 Innovative Design Project · Fall 2026 · state as of 16 Oct 2026

## 1. Problem

Remote identity checks (video KYC, account recovery, exam proctoring) assume the video comes from a
real camera filming a real person. Two attacks break this:

- **Injection:** the attacker feeds video into the app without a physical lens: a virtual camera
  (OBS, v4l2loopback), a hooked capture call inside the app, or an emulator/VM.
- **Synthetic faces:** face swap, reenactment, talking-head or text-to-video generators, usually
  delivered through an injection channel.

Presentation-attack detection (ISO/IEC 30107-3) looks for masks and screens held up to a lens. It
cannot see an injected stream. CEN/TS 18099 (2024) defines injection-attack detection as a separate
problem.

## 2. Approach

Four checks run in sequence on the live camera. Cheap checks run first; any one can end the session.

| Gate | Question | How | Cost per tick |
|---|---|---|---|
| 1a | Is the source a physical camera in a clean environment? | Kernel device records (V4L2 capabilities, sysfs bus, driver, USB identity); VM, container, debugger, `LD_PRELOAD`, root and injection-tool checks | ~10 ms |
| 1b | Is that camera producing these frames right now? | Brightness plus two random controls (contrast, saturation, hue, white balance), each with its own random 12-step pattern; frames must follow every one. Optional screen flash: the screen's light must show on the face | ~1.6–5 s every 10 s |
| 2 | Do the frames carry live sensor noise, from the challenged sensor? | Four blind tests on still pixels: fresh noise, no codec block-copying, sensor-like texture, sensor-like colour. Same-sensor check: the PRNU of the current frames must match the frames read during the last passed challenge | ~0.35 s (+0.2 s every 3 s) |
| 3 | Does the face look real? | GenD deepfake detector (WACV 2026, Perception Encoder L) on aligned face crops | ~0.37 s |

The project's claim: the gates cover each other's blind spots, and the combination detects injected and
synthetic video in real time on a laptop without per-camera enrollment.

## 3. How each gate works

### Gate 1a — physical camera (`host_integrity.py`)
- A node counts as a camera only if its per-node `device_caps` include video capture, its sysfs entry is
  not under `/sys/devices/virtual`, it has a USB/PCI parent, and no layer reports a software-camera
  driver (`v4l2 loopback`, `akvcam`, `vivid`, …). The device's reported name is never trusted.
- It builds a camera identity from the USB VID:PID and port, e.g. `usb-5986-216a-1-3`.
- It BLOCKs on: a debugger or `LD_PRELOAD`, a VM or container, root, or a non-physical capture node.
  It FLAGs when software-camera modules or injection tools are present alongside a real camera.

### Gate 1b — multi-control sensor challenge
- Each attempt drives brightness (or gamma/gain) plus **two controls drawn at random** from contrast,
  saturation, hue and white-balance temperature through `VIDIOC_S_CTRL`, all at once while the stream
  under test is read. Every control has its own random balanced ±sequence; pairwise sequence correlation
  is ≤ 0.34, so the responses can be separated. Saturation and hue are drawn only when the scene has
  colour; white balance switches auto white balance off for the test.
- **Features**, one per control, against the first frame: median brightness shift (a moving person
  doesn't disturb it unless they cover more than half the image), log ratio of the 10–90% luma range
  (contrast), log ratio of mean chroma (saturation), pixel-wise rotation of the chroma plane (hue), and
  change of log(blue/red) (white balance).
- Each frame is labelled with the commands active **at its driver capture timestamp**, which removes
  queue latency at low frame rates.
- **Pass:** for *every* driven control, the partial correlation between its feature and its own command,
  controlling for drift and the other commands, is ≥ 0.85, and the response clears a per-control
  minimum (brightness ≥ 6 grey levels). Hue and white-balance direction differ between cameras, so those
  are judged on magnitude. Two attempts; all original values and auto white balance are restored.
- A stream that only follows brightness (an attacker reading the brightness control and faking it)
  fails the other two controls.
- The frames read during a passed challenge are kept as the attested PRNU reference for Gate 2's
  same-sensor check.

### Screen flash (part of Gate 1b, advisory)
- On request (the "Screen flash" button), the server draws a random 12-slot sequence over one of two
  colour pairs (blue/yellow, cyan/magenta) and the browser shows it full-screen, 0.4 s per slot
  (≤ 2.5 changes per second, below the WCAG 2.3.1 limit of 3 flashes per second; no saturated red).
- The mean colour of the detected face, projected on the axis between the two colours, must correlate
  with the shown sequence (r ≥ 0.7, controlling for drift, 0–350 ms latency searched). A face replayed on
  a phone held up to the camera, or an injected face, was never lit by this screen.
- A miss only flags: in a bright room the screen may not light the face enough. Not yet calibrated on
  people; see Limitations.

### Gate 2 — live sensor noise (`camera_sensor_noise_profiling.py`)
The test needs no enrollment. It runs on pixels that are still between consecutive frames: a pixel is
"still" if its change is small relative to its local gradient, so moving texture can't pose as noise.
The stream must pass all four checks:
1. **Noise present:** frame-to-frame noise ≥ 0.6 grey levels. Generators and video codecs largely remove it.
2. **No codec copying:** ≤ 0.1% of 8×8 blocks repeat bit-for-bit. Inter-frame codecs copy whole
   macroblocks; live sensor noise never repeats a whole block.
3. **Sensor-like texture:** noise neighbour correlation ≥ 0.15. Demosaicing and JPEG correlate it;
   added grain is white.
4. **Sensor-like colour:** blue–red noise correlation between 0.30 and 0.97. Grey grain is 1.0 and
   per-channel grain is 0.

**Same-sensor check (split and compare, no enrollment).** The frames read during the last passed
Gate 1b challenge followed random hardware commands, so they come from the physical sensor. Every 3 s
the live engine estimates a PRNU fingerprint from the last 30 frames and matches it against the
fingerprint of those attested frames (PCE ≥ 60, peak searched within ±2 px). A stream that switches to
another source after the challenge (a hook that passes real frames only while controls change) carries
another sensor's pattern, or none; a mismatch fails Gate 2 and triggers a new challenge. The one-shot
pipeline runs the same check between its challenge frames and its capture frames.

Unlike the old within-clip split (odd vs even frames of one clip, which measured static scene texture),
this compares two *sources*: an injected clip's scene does not line up with the camera's, so scene
texture cannot make it match. It can make content from this camera match: a replay of this camera's
own frames, or an animated photo taken from it, is caught by Gate 1b and the live-noise test instead.

The earlier designs were reference PRNU matching (needs enrollment) and a blind PRNU-presence test
(not discriminative). They are kept in the module as a research API and are not used by the app.

### Gate 3 — face deepfake model (`deepfake_detector.py`)
- **Pipeline:** YuNet face detection → 5-point alignment exactly as in GenD (1.3× margin, 256 px) →
  GenD-PE-L → P(fake).
- **Face-quality gate:** a face is scored only if it is ≥ 90 px wide with detection confidence ≥ 0.85.
  Below that, the gate abstains.
- **Live decision:** the median P(fake) of the last 3 seconds, flagged at ≥ 0.5.
- **Model choice:** GenD has the best published cross-dataset AUROC (91.2–91.6% mean over 14 benchmarks).
  - CLIP-L was tested and rejected: it flagged 65% of real webcam windows.
  - DINOv3-L requires Meta's licence approval and was not tested.

### Live engine (`live_engine.py`) and app (`server.py`, `web/`)
- A capture thread owns the camera and runs Gate 1b. An analysis thread alternates Gate 2 and Gate 3
  every 0.5 s and runs the same-sensor check every 3 s.
- A compute lock pauses analysis during a sensor challenge. Automatic re-challenges (noise lost, sensor
  pattern changed) are spaced at least 5 s apart; the button always runs one.
- The camera is released on Stop, when the user leaves the page, when the browser stops polling for
  8 s, when another session takes it over, or at exit.
- "Simulate an attack" replaces frames after the driver with a replay, an AI video or an animated photo.
- The server binds to `localhost` only.

## 4. Results (one laptop webcam, public datasets)

Data:
- 15 webcam sessions on one camera
- VISION phone videos: 11 phone models
- DF40: 48 clips from 12 generators
- TalkingHeadBench: 72 clips from 8 generators
- 16 text-to-video clips (Veo 3, Kling, Hunyuan, Grok)
- 60 CelebV-HQ YouTube clips

| Gate | Measurement | Result |
|---|---|---|
| 1b | Genuine live checks (brightness only, before 16 Oct) | 20/20 pass (r median 1.00) |
| 1b | Replayed frames while the real sensor is challenged | 11/11 rejected |
| 1b | Null false passes (non-responding footage, random patterns) | 0 / 32,700 (95% bound 9×10⁻⁵) |
| 1b | 10 fps (dim light) | 5/5 pass, r ≥ 0.99 |
| 1b | Multi-control, genuine (16 Oct, one camera, one room) | 6/6 pass (1 on the 2nd attempt); every control r ≥ 0.91 |
| 1b | Multi-control vs plain replay / replay faking brightness | 3/3 and 3/3 rejected (brightness r 0.996, other controls ≤ 0.41) |
| 2 | Same-sensor PCE: same camera, later frames | 9,600–20,000 |
| 2 | Same-sensor PCE: AI clips, other cameras, animated photo (offline), AI + re-added noise | ≤ 14 |
| 2 | Same-sensor PCE: frozen frame of this camera + fresh noise | ~500 (matches; caught by 1b and the noise test) |
| 2 | Same-sensor PCE: animated photo of this camera in the live engine | 4,155 (matches; caught by 1b) |
| 2 | Real webcam windows (20 and 45 frames) | 104/104 PRESENT |
| 2 | Talking-head fakes / DF40 + text-to-video | 72/72 and 64/64 ABSENT |
| 2 | Grey, colour and blurred synthetic grain on a still photo | 18/18 ABSENT |
| 2 | x264 replays of webcam sessions | 6/6 ABSENT |
| 2 | YouTube real clips (encoded) | 58/60 ABSENT (by design: not a live stream) |
| 3 | Live colour webcam faces | **Not validated.** The stored webcam sessions are grayscale and 9 of 11 contain no face; the earlier "0/11 flagged" rested on 2 grey clips and is withdrawn |
| 3 | Re-evaluation 15 Oct (live scoring path, 120 fakes vs 60 CelebV-HQ reals): AUROC | 0.85 (live-style 3 ticks × 4 crops: 0.84) |
| 3 | Fakes caught / CelebV-HQ reals flagged at P ≥ 0.5 | 67% / 13% |
| 3 | Fakes caught after JPEG q30 re-save | 25% |
| 3 (old heuristic) | Fakes caught | 2/64 (replaced) |

Cost on the test laptop (RTX 4050):
- Gate 2: 348 ms per tick; Gate 3: 365 ms per tick
- Process memory: ~1.5–1.9 GB resident
- "17.8 GB" in task managers is virtual address space reserved by CUDA, not RAM in use

## 5. Limitations

- **One camera, one room.** All genuine data, and every threshold, comes from this laptop's webcam, so
  thresholds were tuned and tested on the same data. A publishable study needs ≥ 10 webcams, several
  people, varied lighting and a held-out test split.
- **No real virtual camera was tested.** v4l2loopback is not installed. Gate 1a was verified on a
  replica of its sysfs layout.
- **Adaptive attackers:**
  - A brightness-only follower now fails Gate 1b. An attacker with local access who reads *every*
    control and emulates the camera's response to each (curves, colour processing, latency) could
    still pass it.
  - Someone who injects *raw* frames with realistic sensor-like noise could pass Gate 2's noise test;
    the same-sensor check stops this unless the injected frames carry this camera's PRNU.
  - A replay of raw frames from this same camera passes Gate 2, including the same-sensor check;
    Gate 1b is what stops it.
- **Gate 2 false rejects seen live (16 Oct).** On the test webcam in a dim room at 10 fps, 2 of 4
  twenty-frame windows showed 1–2% bit-identical 8×8 blocks (limit 0.1%) with no duplicate frames, and
  the console briefly showed a real user as fake. MJPEG likely quantises flat, denoised areas identically.
  The check needs re-tuning on the end-to-end set below.
- **Screen flash is uncalibrated.** Its thresholds come from synthetic tests; it has not yet been
  measured with people, rooms of different brightness, or a phone replay.
- **Kernel-level compromise is out of scope.**
- **Gate 3** catches about 67% of fakes at 0.5 and flags 13% of CelebV-HQ faces; the 0.5 threshold is
  the model default, not a calibrated value. It is one signal, not a proof.
- **Platform:** Linux/V4L2 only. The Android SDK (`app/`) is a separate Layer 1 prototype; its JSON
  verdict is accepted but it does not stream frames.

## 6. End-to-end evaluation (ISO/IEC 30107-3)

`benchmarks/e2e/record_sessions.py` records labelled trials through the real capture path: the Gate 1a
probe, the live multi-control challenge with its attested frames, lossless colour frames, and
optionally the screen flash. Labels: `bona_fide`; physical attacks in front of the camera
(`screen_replay`, `printed_photo`, `mask`); `virtual_camera` (OBS / v4l2loopback node); and simulated
hooked capture (`--inject replay|ai|photo|adaptive`, frames swapped after the driver while the real
camera is challenged). `benchmarks/e2e/score_sessions.py` replays each gate as the live app runs it and
reports APCER per attack type, BPCER, per-gate error rates and Wilson 95% intervals in
`benchmarks/results/E2E_RESULTS.md`. Trials stay in `benchmarks/data/e2e/` (git-ignored: they contain faces).
No trials have been collected yet.

The earlier in-app model-loading stall was resolved by preloading GenD at server start (commit `029730b`).

## 7. Run

```bash
source .venv/bin/activate
cd web && npm install && npm run build && cd ..   # once, and after UI changes
python server.py                          # http://localhost:8501; Start → checks run continuously
python run_tests.py                       # 68 tests
python benchmarks/run_benchmark.py        # Gate 1b / reference-PRNU benchmark
python benchmarks/e2e/record_sessions.py --subject s01 --lighting daylight --flash   # one labelled trial
python benchmarks/e2e/score_sessions.py   # APCER / BPCER over all recorded trials
```

Gate 3 needs PyTorch (CUDA build for NVIDIA GPUs) and downloads GenD-PE-L (~1.2 GB) on first use.

## 8. Repository

| Path | Purpose |
|---|---|
| `host_integrity.py` | Gate 1a + 1b |
| `camera_sensor_noise_profiling.py` | Gate 2 (live noise; optional reference PRNU) |
| `deepfake_detector.py` | Gate 3 (YuNet + GenD) |
| `pipeline.py` | Gate orchestration for files and one-shot live sessions |
| `live_engine.py` | Real-time engine (capture + challenge thread, rolling analysis thread) |
| `server.py` | JSON API, live MJPEG stream, serves the built web app |
| `web/` | React + TypeScript front end: Live, Analyze a file, Device |
| `temporal_consistency_analysis.py` | Original Gate 3 heuristic (kept, no longer used) |
| `benchmarks/` | Dataset fetchers, capture script, benchmark runner, results; `e2e/` trial recorder and APCER/BPCER scorer; `gate3_eval/` Gate 3 accuracy scripts |
| `docs/ARCHITECTURE.md` | Long-form design history and earlier measurements |
| `app/` | Android Layer 1 prototype |

## 9. Next steps

1. Record the end-to-end set (§6): several people, lighting conditions and cameras, plus physical and
   virtual-camera attacks. Re-tune Gate 2 (block-copy false rejects) and calibrate the screen flash on a
   training split; report APCER/BPCER on a held-out split.
2. Run real attacks: v4l2loopback + OBS, DeepFaceLive, and an `LD_PRELOAD` frame swap.
3. Move the challenge and verification server-side (server-issued pattern, signed client attestation)
   so a compromised client cannot skip the checks.
4. Test the adaptive attacks in §5.
