# Layer 3 - Temporal Consistency & Frequency Analysis

Owner: **Vibha**
Part of: Injection Attack & Deepfake Detection pipeline
(Layer 1 = Aarya's hardware attestation, Layer 2 = Arihant's PRNU sensor noise
profiling, Layer 3 = this module, Layer 5 = gated fusion in Ramya's pipeline)

## Files

| File | Purpose |
|---|---|
| `temporal_consistency_analysis.py` | The Layer 3 detector. Run it on a video and get a JSON verdict. |
| `generate_test_videos.py` | Builds two tiny synthetic clips (`real_sim.mp4`, `fake_sim.mp4`) so you can sanity-check the analyzer without needing real deepfake footage yet. |

## How it works

1. **Per frame**, locate the face and compute:
   - a 2D FFT **high-frequency energy ratio** of the face crop
   - **edge density** (Canny) as a texture-sharpness proxy
   - **mean luminance** as a lighting proxy
2. **Across a rolling window of frames**, look for the tell-tale signs of
   frame-by-frame generative synthesis:
   - **Flicker** - how much those three signals jump around frame-to-frame,
     relative to their own average level
   - **Optical-flow incoherence** - real motion is smooth and spatially
     consistent; face-swap boundaries often aren't
   - **Temporal-FFT periodicity** - take the high-frequency-ratio signal
     *across time* and FFT it again; a strong periodic peak flags flicker a
     human eye wouldn't notice
3. Combine all four into one **anomaly score in [0, 1]** with a human-readable
   explanation, in a JSON shape meant to be consumed directly by the
   pipeline/UI and fused with the other layers.

## Run it

```bash
pip install opencv-python numpy scipy --break-system-packages   # if not already installed

python temporal_consistency_analysis.py --video clip.mp4
python temporal_consistency_analysis.py --video clip.mp4 --output result.json
```

## Sanity-check without real footage

```bash
python generate_test_videos.py
python temporal_consistency_analysis.py --video real_sim.mp4
python temporal_consistency_analysis.py --video fake_sim.mp4
```

On these synthetic clips the "fake" one (per-frame injected noise +
occasional lighting jumps) currently scores noticeably higher than the
"real" one — confirming the pipeline is directionally sound.

## Output shape (what Ramya's portal should expect)

```json
{
  "layer": "temporal_consistency_frequency_analysis",
  "frames_analyzed": 90,
  "frames_with_face": 90,
  "score": 0.43,
  "flagged": false,
  "confidence": 1.0,
  "components": {
    "flicker_rate": 1.0,
    "mean_flow_coherence": 0.65,
    "flow_incoherence": 0.0,
    "temporal_periodicity_anomaly": 0.46,
    "hf_energy_coefficient_of_variation": 0.03
  },
  "explanation": "..."
}
```

`score >= 0.6` (configurable) sets `flagged = true`. `confidence` reflects
how many frames actually had a detected face vs. the target window size.

## Important - this is a heuristic prototype, not a calibrated detector

There's no labeled real-vs-deepfake dataset behind this yet. All thresholds
live in the `CONFIG` dict at the top of `temporal_consistency_analysis.py`
and are documented placeholders. **Before relying on `flagged` for anything
real**, run this against a batch of genuine clips and a batch of known
deepfakes (FaceForensics++ is a reasonable public dataset to start from) and
retune:

- `hf_instability_norm`, `edge_instability_norm`, `luma_instability_norm`
- `flow_coherence_thresh`
- `temporal_fft_peak_thresh`
- `score_flag_thresh`
- the component `weights`

## Integration note for Ramya / Layer 5 fusion

This module only needs a video file path in and returns one JSON object -
easy to call as a subprocess or import `TemporalConsistencyAnalyzer` directly
and call `.analyze(path)` to get a `LayerResult` object. It doesn't talk to
Layer 1 (Aarya) or Layer 2 (Arihant) at all — per the gated-fusion design in
the write-up, this layer only needs to run if Layers 1-2 didn't already
terminate the session.
