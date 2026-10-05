# Layer 2 - Camera Sensor Noise Profiling (PRNU)

Owner: **Arihant** · Module: `camera_sensor_noise_profiling.py`

This note summarises the implementation. Rationale, measurements and limitations are in
[ARCHITECTURE.md](ARCHITECTURE.md) (Sections 4.3, 6 and 7).

## Model

A CMOS sensor outputs `I = I0 (1 + K) + Θ`, where `K` is the pixel-wise PRNU gain pattern and `Θ` is
temporal noise. `K` is locked to the pixel grid and differs between sensors (Lukáš, Fridrich & Goljan 2006).

## Pipeline

1. **Residual** `W = I − F(I)`. `F` is a spatially adaptive local Wiener filter: local variance is the
   minimum over 3/5/7/9 windows (Mihçak-style), and residual is attenuated by `σ0²/var` with `σ0 = 3`.
   Saturated pixels are zeroed.
2. **Fingerprint** (MLE): `K = Σ W·I / Σ I²`, then row/column zero-meaning and DFT-domain peak
   clamping. Peak clamping removes the JPEG/H.264 block grid and CFA periodicities that every camera and
   codec share.
3. **Reference mode (enforced)**: correlate the clip's summed residual with `I·K_ref` and score the
   **PCE**, taken at the strongest peak within ±2 px of zero shift (session-to-session crop offsets).
   The clip matches the enrolled camera if PCE ≥ 60. Fingerprints are stored per **Layer 1
   camera ID + sensor mode** (`FingerprintStore`), so the PRNU check is bound to the attested
   physical device.
4. **Blind mode (advisory)**: estimate `K_A` and `K_B` from temporally disjoint halves and correlate them
   **only on pixels whose low-passed content changed between the halves**. Significance comes from a
   circular-shift null (48 shifts) → z-score.
   - PRESENT: z ≥ 5 and ρ ≥ 0.05
   - ABSENT: z < 3 or ρ < 0.04
   - INCONCLUSIVE: less than 4% of the frame changed (highlights above 230 excluded), or anything in between

## Why the previous version produced false positives / false negatives

- Odd/even frame interleaving plus whole-frame correlation measured **static scene texture**, not
  PRNU. Real webcams scored PCE 7k–63k, and the Grok AI video scored 79k and passed.
- The app saved live frames with OpenCV `mp4v` before analysis, and file mode re-encoded a "fast
  sample" the same way. That re-encode erases PRNU: genuine frames dropped from PCE 300+ to 14–22.
  All analysis now runs on raw decoded frames.
- A blind "is there a noise pattern?" test cannot be a hard gate. Native phone videos (stabilisation,
  ISP denoising) often show none, while generators and face-swap backgrounds often show one (see the
  benchmark). Only the reference match is enforced.

## CLI

```bash
python camera_sensor_noise_profiling.py --video clip.mp4                        # blind
python camera_sensor_noise_profiling.py --video calib.mp4 --enroll-ref fp.npy   # enroll
python camera_sensor_noise_profiling.py --video clip.mp4 --ref-fingerprint fp.npy
```
