# 🛡️ AI Deepfake Detection & Injection Attack Defense

> **Next-Generation Multi-Layered Forensic Verification Pipeline Compliant with CEN/TS 18099**  
> *Innovative Design Project (IDP)*

---

## 📌 Executive Summary

Remote identity verification, video KYC (Know-Your-Customer), and biometric authentication systems face unprecedented threats from **digital injection attacks** and **generative AI deepfakes**. Rather than physically presenting a fake face or photo mask to an authentic camera lens, sophisticated adversaries bypass the physical sensor entirely: they inject synthetic or manipulated video streams directly into the operating system or application runtime using virtual cameras, hooked camera HALs, or hypervisor emulators.

This project delivers an end-to-end, multi-layered, **gated forensic detection pipeline** designed to distinguish genuine physical camera streams from injected synthetic streams in real time. Following the **CEN/TS 18099** biometric presentation and injection attack detection standard, the system enforces **early termination** across sequential gates: lightweight client-level checks execute first (<50 ms), followed by physical sensor noise profiling and temporal/spectral coherence transforms, culminating in a unified forensic verification dashboard.

---

## 🏛️ Gated 5-Layer Forensic Architecture

```
[ Incoming Video Stream & Attestation Payload ]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  GATE 1: Layer 1 — Device & OS Integrity Attestation   │
│  Owner: Aarya Jadhav | Latency: ~14 ms                 │
│  - Root & Superuser Binary Probing (su, Magisk, Zygisk)│
│  - Android Emulator / Hypervisor Markers (QEMU/Goldfish│
│  - Virtual Camera Packages & Hooked HAL Drivers        │
└────────────────────────────────────────────────────────┘
                      │
            [ PASS ] ─┴─► [ BLOCK if Root/Emulator Flagged ]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  GATE 2: Layer 2 — Camera Sensor Noise Profiling (PRNU)│
│  Owner: Arihant | Latency: ~1.2 s (Fast-Sampled)       │
│  - Microscopic Photo-Response Non-Uniformity (PRNU)    │
│  - 2D Wiener Spatial Denoising Residual Extraction     │
│  - 2D FFT Circular Cross-Correlation & PCE Peak Energy │
│  - Inter-Frame Silicon Noise Persistence Correlation   │
└────────────────────────────────────────────────────────┘
                      │
            [ PASS ] ─┴─► [ FLAG if PRNU Missing / Uncorrelated ]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  GATE 3: Layer 3 — Temporal Consistency & Frequency    │
│  Owner: Vibha | Latency: ~1.8 s (Fast-Sampled)         │
│  - Farneback Dense Optical Flow Coherence              │
│  - Inter-Frame High-Frequency Ratio & Edge Stability   │
│  - 1D Temporal FFT Power Spectral Density Anomaly      │
│  - Generative Synthesis Jitter & Flicker Rate Analysis │
└────────────────────────────────────────────────────────┘
                      │
            [ PASS ] ─┴─► [ FLAG if Temporal Anomaly ≥ 0.60 ]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  GATE 4: Layer 4 — Biometric Liveness (rPPG Pulse)     │
│  Status: Architecture Specification / Future Expansion │
│  - Remote Photoplethysmography micro-vascular pulse     │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  GATE 5: Layer 5 — Gated Fusion Portal & Orchestration │
│  Owner: Ramya | Total Latency: < 3.5 s                 │
│  - Fast Sampling Frame Buffer Engine                   │
│  - Interactive Streamlit Forensic Operations Dashboard │
│  - Composite Anomaly Scoring, Metrics, & Frame Gallery │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
        ✅ AUTHENTIC LIVE STREAM VERDICT
```

---

## 🧩 Detailed Layer Breakdown

| Layer | Focus Area | Primary Methods / Signals | Key Metrics & Thresholds | Module / Source | Owner |
|---|---|---|---|---|---|
| **Layer 1** | Client & Device Attestation | Android PackageManager checks, su binary paths, test-keys, hypervisor build tags, CameraCharacteristics enumeration | Latency < 50ms; Hard BLOCK on Root/Emulator | [`app/src/main/`](app/src/main/) | Aarya Jadhav |
| **Layer 2** | Camera Sensor Noise Profiling | 2D Wiener noise residual extraction, 2D FFT circular cross-correlation, PCE, reference fingerprint matching | PCE > 45.0, Persistence > 0.20, Anomaly < 0.55 | [`camera_sensor_noise_profiling.py`](camera_sensor_noise_profiling.py) | Arihant |
| **Layer 3** | Temporal & Frequency Analysis | Farneback dense optical flow, high-frequency energy ratios, 1D temporal FFT spectral density, flicker rate | Flicker < 0.65, Score < 0.60, Confidence > 0.8 | [`temporal_consistency_analysis.py`](temporal_consistency_analysis.py) | Vibha |
| **Layer 4** | Biometric Liveness & Physiology | Remote photoplethysmography (rPPG) biological pulse wave detection from subtle facial skin color fluctuations | Heart rate coherence, SNR | *Standard Reference* | Research Roadmap |
| **Layer 5** | Gated Fusion & Interactive UI | Early termination orchestration, fast clip frame sampling, forensic reporting, Streamlit web portal | End-to-end latency < 5.0s, Fail-Fast Gating | [`pipeline.py`](pipeline.py), [`app.py`](app.py) | Ramya |

---

## 📂 Project Repository Structure

```
AI-Deepfake-Detection-Innovative-Design-Project-/
├── README.md                                 # Master unified project documentation (this file)
├── requirements.txt                          # Pinned Python package dependencies
├── .gitignore                                # Git ignore rules for virtualenvs, caches, & media
├── app.py                                    # Streamlit forensic verification portal (Layer 5 UI)
├── pipeline.py                               # Gated verification orchestration engine (Layer 5)
├── camera_sensor_noise_profiling.py          # PRNU sensor noise profiler & PCE engine (Layer 2)
├── temporal_consistency_analysis.py          # Optical flow & 1D temporal FFT analyzer (Layer 3)
├── generate_test_videos.py                   # Calibrated synthetic test stream & PRNU generator
├── test_sensor_noise_profiler.py             # Layer 2 PRNU unit test suite (11 test cases)
├── test_temporal_consistency.py              # Layer 3 Temporal consistency unit tests (11 test cases)
├── test_pipeline.py                          # Layer 5 Gated pipeline integration tests (10 test cases)
├── run_tests.py                              # Unified test runner executing all 32 test cases
├── camera_fingerprint.npy                    # Pre-extracted genuine CMOS sensor reference fingerprint
├── real_sim.mp4                              # Baseline genuine camera capture simulation clip
├── fake_sim.mp4                              # Baseline synthetic deepfake / injection simulation clip
├── result.json                               # Sample forensic evaluation JSON output
├── benchmark_results/                        # Preserved layer benchmark results
│   ├── prnu_benchmark_result.json            # Layer 2 PRNU evaluation benchmark
│   └── temporal_benchmark_result.json        # Layer 3 Temporal evaluation benchmark
├── docs/                                     # In-depth architectural documentation per layer
│   ├── LAYER1_DEVICE_INTEGRITY.md            # Layer 1 Android attestation deep-dive (Aarya)
│   ├── LAYER2_PRNU_PROFILING.md              # Layer 2 PRNU mathematics & forensics deep-dive (Arihant)
│   ├── LAYER3_TEMPORAL_ANALYSIS.md           # Layer 3 Temporal & FFT analysis deep-dive (Vibha)
│   └── LAYER5_GATED_FUSION_PORTAL.md         # Layer 5 Architecture & CEN/TS 18099 deep-dive (Ramya)
└── app/                                      # Android Client Attestation SDK (Layer 1)
    └── src/main/
        ├── AndroidManifest.xml               # Camera & storage permissions, activity configuration
        └── java/com/idp/deviceintegrity/
            ├── DetectionResult.kt            # Structured attestation result data class
            ├── DeviceIntegrityChecker.kt     # Attestation coordinator & verdict engine
            ├── EmulatorDetector.kt           # QEMU / Goldfish / hypervisor marker detection
            ├── MainActivity.kt               # Sample Android UI rendering attestation status
            ├── RootDetector.kt               # SU binary, Magisk mount, & test-keys detection
            └── VirtualCameraDetector.kt      # CameraManager enumeration & virtual driver detection
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation

Ensure Python 3.10+ is installed. Clone the repository and install required packages:

```bash
# Clone the unified repository
git clone https://github.com/arihant1602/AI-Deepfake-Detection-Innovative-Design-Project-.git
cd AI-Deepfake-Detection-Innovative-Design-Project-

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Forensic Portal (Layer 5)

Launch the interactive web portal:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.
1. Use the sidebar to simulate different **Layer 1 Client Attestation States** (*Clean Physical Device*, *Rooted Android*, *Emulator*, or *Virtual Camera Injection*).
2. Upload a video stream (`real_sim.mp4` or `fake_sim.mp4`).
3. Click **Run Forensic Verification** to witness the gated evaluation with real-time status updates and forensic metric breakdowns.

---

## 🔬 CLI Usage & Autonomous Analysis

### Run the End-to-End Gated Pipeline

```bash
# Evaluate genuine clip
python pipeline.py --video real_sim.mp4

# Evaluate synthetic injection clip
python pipeline.py --video fake_sim.mp4
```

### Run Standalone Layer 2: PRNU Profiler

```bash
# Autonomous blind consistency analysis
python camera_sensor_noise_profiling.py --video real_sim.mp4

# Reference camera fingerprint matching
python camera_sensor_noise_profiling.py --video fake_sim.mp4 --ref-fingerprint camera_fingerprint.npy
```

### Run Standalone Layer 3: Temporal Consistency Analyzer

```bash
python temporal_consistency_analysis.py --video fake_sim.mp4 --output result_temporal.json
```

### Generate Synthetic Test Streams & Camera Fingerprints

```bash
# Generate default calibrated combined test clips (real_sim.mp4, fake_sim.mp4, camera_fingerprint.npy)
python generate_test_videos.py

# Generate clips for all individual test modes
python generate_test_videos.py --mode all
```

---

## 🧪 Comprehensive Test Suite

The project includes 32 automated unit and integration tests across all layers:

```bash
# Run the complete test suite across Layer 2, Layer 3, and Layer 5
python run_tests.py
```

Output:
```text
======================================================================
AI Deepfake Detection & Injection Attack Defense - Test Suite
======================================================================
 Loaded module: test_sensor_noise_profiler (11 test cases)
 Loaded module: test_temporal_consistency (11 test cases)
 Loaded module: test_pipeline (10 test cases)
----------------------------------------------------------------------
Ran 32 tests in ~6.0s

OK
======================================================================
SUCCESS: All 32 tests passed successfully!
```

Individual test suites can also be executed independently:
- `python test_sensor_noise_profiler.py` (Layer 2 PRNU tests)
- `python test_temporal_consistency.py` (Layer 3 Temporal tests)
- `python test_pipeline.py` (Layer 5 Gated pipeline integration tests)

---

## 📚 Technical Documentation Index

Deep-dive documentation for each individual layer is available in the [`docs/`](docs/) directory:
- [📖 Layer 1: Device & Environment Integrity Attestation](docs/LAYER1_DEVICE_INTEGRITY.md)
- [📖 Layer 2: Camera Sensor Noise Profiling (PRNU Analysis)](docs/LAYER2_PRNU_PROFILING.md)
- [📖 Layer 3: Temporal Consistency & Frequency Analysis](docs/LAYER3_TEMPORAL_ANALYSIS.md)
- [📖 Layer 5: Forensic Verification Portal & Gated Fusion](docs/LAYER5_GATED_FUSION_PORTAL.md)

---

## 👥 Contributors & Subsystem Ownership

- **Aarya Jadhav**: Layer 1 — Device & Environment Integrity Attestation (Android SDK)
- **Arihant**: Layer 2 — Camera Sensor Noise Profiling (PRNU Analysis & PCE Engine)
- **Vibha**: Layer 3 — Temporal Consistency & Frequency Analysis (Optical Flow & 1D FFT)
- **Ramya**: Layer 5 — Interactive Verification Portal & Gated Execution Fusion Orchestrator
