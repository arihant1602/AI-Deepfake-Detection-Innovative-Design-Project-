# Layer 5: Forensic Verification Portal & Gated Fusion Engine

**Component:** Layer 5 (Frontend & Integration Orchestrator)  
**Architecture:** CEN/TS 18099 Gated Presentation Attack & Injection Detection  
**Target Latency:** Sub-5 seconds end-to-end verification

---

## Overview

This module implements Layer 5 of the five-layer detection pipeline. It provides a real-time Streamlit forensic dashboard and an orchestration engine that unifies client-side attestation, CMOS sensor noise profiling, and temporal consistency checks into a sequential, gated execution flow.

Rather than running heavy spatial, frequency, and optical flow transformations simultaneously across entire video files, the pipeline enforces **early termination**:
- Failure at any intermediate gate halts execution immediately to preserve computational resources.
- Downstream mathematical transforms run only if preceding hardware and physical sensor criteria pass.

---

## The 5-Layer Gated Architecture

```text
[ Incoming Video Stream & Attestation Payload ]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Gate 1: Hardware & Environment Attestation (Layer 1)   │
│  - Root & Magisk detection                             │
│  - Android Emulator / QEMU hypervisor fingerprinting   │
│  - Virtual camera driver hooking detection             │
└──────────────────────────┬─────────────────────────────┘
                           │
             [ Flagged / Compromised? ] ──► YES ──► [ 🛑 EARLY TERMINATION: Digital Injection ]
                           │                               (< 20 ms Latency)
                          NO (PASS)
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Gate 2: CMOS Sensor Noise Profiling (Layer 2)          │
│  - Wavelet denoising & PRNU residual extraction        │
│  - Peak-to-Correlation Energy (PCE) validation         │
│  - Stationary sensor noise consistency check           │
└──────────────────────────┬─────────────────────────────┘
                           │
             [ Flagged / PCE < 0.55? ] ──► YES ──► [ 🛑 EARLY TERMINATION: Synthetic Sensor ]
                           │
                          NO (PASS)
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Gate 3: Temporal & Frequency Analysis (Layer 3)        │
│  - Facial ROI 2D FFT high-frequency energy ratios      │
│  - Farnebäck dense optical flow boundary coherence     │
│  - Temporal FFT micro-flicker periodicity              │
└──────────────────────────┬─────────────────────────────┘
                           │
             [ Flagged / Score ≥ 0.60? ] ─► YES ──► [ 🛑 EARLY TERMINATION: Deepfake Detected ]
                           │
                          NO (PASS)
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Gate 5: Gated Decision Fusion & Forensics (Layer 5)    │
│  - Real-time telemetry dashboard                       │
│  - CEN/TS 18099 compliance feedback                    │
│  - Final Verdict: ✅ AUTHENTIC LIVE STREAM             │
└────────────────────────────────────────────────────────┘
```
## Performance & Optimization Engineering
* Standard forensic video algorithms often require 1–3 minutes per video when processing every frame. To meet real-time operational requirements:

* Dynamic Burst Pre-Sampling: Implemented create_fast_sample_clip to extract a 45-frame burst (~1.5 seconds of video) for heavy mathematical analysis (PRNU wavelet decomposition and spatial/temporal FFTs). This reduces total verification latency to ~3 to 5 seconds.

* Zero-Compute Early Rejection: Compromised Android environments (Root/Emulator) are halted at Gate 1 in < 20 ms, eliminating unnecessary CPU/GPU load.

* Granular Forensic Telemetry: Surfaced raw metrics—including PCE Peak Energy, PRNU Anomaly Score, Attestation Latency, and Temporal Flicker—for full forensic explainability.

## File Structure
```text
.
├── app.py                            # Streamlit forensic dashboard & telemetry display
├── pipeline.py                       # Layer 5 orchestration engine & dynamic frame sampler
├── camera_sensor_noise_profiling.py  # Layer 2 PRNU profiler (Arihant)
├── temporal_consistency_analysis.py  # Layer 3 Temporal/FFT analyzer (Vibha)
├── generate_test_videos.py           # Synthetic video generator for validation testing
└── README.md                         # Pipeline documentation
```
## Getting Started
### Prerequisites
Ensure the required dependencies are installed:

```powershell
pip install streamlit opencv-python numpy scipy
```
### Running the Dashboard
Launch the interactive portal:

```powershell
streamlit run app.py
```
* Open http://localhost:8501 in your browser.

* Select the client device attestation state in the sidebar (to test hardware gating).

* Upload a video clip (.mp4, .mov, .avi) or test synthetic clips (fake_sim.mp4 / real_sim.mp4).

* Click Run Forensic Verification to inspect real-time gated execution and forensic metrics.
