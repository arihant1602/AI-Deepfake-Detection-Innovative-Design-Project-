"""
Layer 3: Temporal Consistency and Frequency Analysis
=====================================================

Owner: Vibha

What this layer does
---------------------
Deepfake / face-swap generators produce individual frames that look convincing
in isolation, but struggle to keep facial texture, lighting, and geometry
*smoothly continuous from one frame to the next*. This module looks for that
breakdown by combining two signal families over a rolling window of frames:

  1. FREQUENCY-DOMAIN features (per frame)
     - High-frequency energy ratio from a 2D FFT of the face crop
       (GAN/diffusion upsampling artifacts show up as excess high-frequency
       energy relative to natural video, and it is not stable frame-to-frame).
     - Edge density (Canny) as a texture-sharpness proxy.
     - Mean luminance, as a lighting proxy.

  2. TEMPORAL-COHERENCE features (across the frame window)
     - Frame-to-frame "flicker": how much the frequency-domain features jump
       around instead of drifting smoothly.
     - Optical-flow coherence inside the face region: real video has smooth,
       spatially consistent motion; face-swap boundaries often show
       incoherent or jittery flow.
     - A *second* FFT, this time taken along the TIME axis of the
       high-frequency-ratio signal, to catch periodic "flicker" that a human
       eye/PAD system would miss but which is mathematically distinct from a
       real camera capture.

These are combined into a single anomaly score in [0, 1] with an explanation,
in a JSON shape designed to be consumed directly by Ramya's pipeline/UI layer
and fused with the other layers (Aarya's hardware attestation, Arihant's PRNU
check) at Gated Inference Fusion (Layer 5).

This is a *prototype/heuristic* scorer (no labeled training set yet) -
thresholds are documented in CONFIG below and are the first thing to tune
once we have real vs. fake sample videos to calibrate against.

Usage
-----
    python temporal_consistency_analysis.py --video path/to/clip.mp4
    python temporal_consistency_analysis.py --video clip.mp4 --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration (tune these once real/fake calibration clips are available)
# --------------------------------------------------------------------------- #

CONFIG = {
    "face_roi_size": 256,           # face crop is resized to this (square) size
    "window_frames": 30,            # rolling window (~1s at 30fps) for temporal stats
    "high_freq_radius_frac": 0.35,  # fraction of Nyquist radius considered "high frequency"
    "flow_coherence_thresh": 0.55,  # below this = incoherent motion (0-1, higher=more coherent)
    "temporal_fft_peak_thresh": 3.0,# how many std-devs above baseline a periodic flicker peak must be
    "score_flag_thresh": 0.6,       # final anomaly score above which we flag the clip
    # --- flicker normalization constants -----------------------------------
    # "instability" = mean frame-to-frame delta of a signal, relative to its
    # own mean level. These *_norm values are the instability level we'd
    # expect from ordinary camera/compression noise on a genuine capture;
    # instability above this starts contributing to the anomaly score.
    # PLACEHOLDER VALUES - re-tune against real labeled genuine/deepfake
    # clips once available; do not treat as production-ready thresholds.
    "hf_instability_norm": 0.03,
    "edge_instability_norm": 0.03,
    "luma_instability_norm": 0.05,
    "weights": {                    # relative weight of each anomaly component in final score
        "flicker": 0.30,
        "flow_incoherence": 0.25,
        "temporal_periodicity": 0.25,
        "hf_energy_instability": 0.20,
    },
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class FrameMetrics:
    frame_idx: int
    face_found: bool
    hf_ratio: float = 0.0        # high-frequency spectral energy ratio
    edge_density: float = 0.0    # canny edge pixel fraction
    mean_luma: float = 0.0       # mean luminance of face ROI


@dataclass
class LayerResult:
    layer: str = "temporal_consistency_frequency_analysis"
    frames_analyzed: int = 0
    frames_with_face: int = 0
    score: float = 0.0
    flagged: bool = False
    confidence: float = 0.0
    components: dict = field(default_factory=dict)
    explanation: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


# --------------------------------------------------------------------------- #
# Face detection / tracking (lightweight - Haar cascade, ships with OpenCV)
# --------------------------------------------------------------------------- #

class FaceLocator:
    """Detects a face bounding box each frame, falling back to the last known
    box for a few frames if detection briefly fails (keeps the ROI stable so
    we're measuring the same region over time)."""

    def __init__(self, max_missed_frames: int = 5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.last_box: Optional[tuple] = None
        self.missed = 0
        self.max_missed_frames = max_missed_frames

    def locate(self, gray_frame: np.ndarray) -> Optional[tuple]:
        faces = self.detector.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            # take the largest detected face (assume single subject)
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            self.last_box = tuple(faces[0])
            self.missed = 0
            return self.last_box

        self.missed += 1
        if self.last_box is not None and self.missed <= self.max_missed_frames:
            return self.last_box
        return None


# --------------------------------------------------------------------------- #
# Per-frame frequency-domain analysis
# --------------------------------------------------------------------------- #

def high_frequency_ratio(gray_roi: np.ndarray, radius_frac: float) -> float:
    """2D FFT of the face crop; returns the fraction of spectral energy that
    lies outside a low-frequency disk of radius `radius_frac` * Nyquist."""
    f = np.fft.fft2(gray_roi.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray_roi.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    mask_high = dist > (radius_frac * max_dist)

    total_energy = magnitude.sum() + 1e-8
    high_energy = magnitude[mask_high].sum()
    return float(high_energy / total_energy)


def edge_density(gray_roi: np.ndarray) -> float:
    edges = cv2.Canny(gray_roi, 100, 200)
    return float(np.count_nonzero(edges)) / edges.size


# --------------------------------------------------------------------------- #
# Optical-flow coherence (motion should be smooth & spatially consistent)
# --------------------------------------------------------------------------- #

def flow_coherence(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Returns a 0-1 coherence score: 1 = very smooth/consistent flow field,
    0 = chaotic/incoherent motion (a sign of face-swap boundary artifacts)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    angle = np.arctan2(fy, fx)

    # Coherence = how tightly the flow directions cluster, weighted by
    # whether there's meaningful motion at all (avoid rewarding a frozen frame)
    mean_mag = magnitude.mean()
    if mean_mag < 1e-3:
        return 1.0  # no motion => trivially coherent, don't penalize

    # circular variance of the angle distribution (0 = all aligned, 1 = random)
    sin_mean = np.sin(angle).mean()
    cos_mean = np.cos(angle).mean()
    resultant_length = np.sqrt(sin_mean ** 2 + cos_mean ** 2)  # 0..1
    return float(resultant_length)


# --------------------------------------------------------------------------- #
# Core analyzer
# --------------------------------------------------------------------------- #

class TemporalConsistencyAnalyzer:
    def __init__(self, config: dict = CONFIG):
        self.cfg = config
        self.face_locator = FaceLocator()

    def analyze(self, video_path: str) -> LayerResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        roi_size = self.cfg["face_roi_size"]
        frame_metrics: list[FrameMetrics] = []
        flow_scores: list[float] = []
        prev_gray_roi: Optional[np.ndarray] = None
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = self.face_locator.locate(gray)

            if box is None:
                frame_metrics.append(FrameMetrics(frame_idx, face_found=False))
                prev_gray_roi = None
                frame_idx += 1
                continue

            x, y, w, h = box
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (roi_size, roi_size))

            hf_ratio = high_frequency_ratio(roi, self.cfg["high_freq_radius_frac"])
            edges = edge_density(roi)
            luma = float(roi.mean())

            frame_metrics.append(FrameMetrics(
                frame_idx=frame_idx, face_found=True,
                hf_ratio=hf_ratio, edge_density=edges, mean_luma=luma,
            ))

            if prev_gray_roi is not None:
                flow_scores.append(flow_coherence(prev_gray_roi, roi))

            prev_gray_roi = roi
            frame_idx += 1

        cap.release()
        return self._score(frame_metrics, flow_scores)

    # ------------------------------------------------------------------ #

    def _score(self, frame_metrics: list[FrameMetrics], flow_scores: list[float]) -> LayerResult:
        result = LayerResult()
        result.frames_analyzed = len(frame_metrics)
        valid = [m for m in frame_metrics if m.face_found]
        result.frames_with_face = len(valid)

        if len(valid) < 8:
            result.explanation = (
                "Not enough frames with a detected face to make a reliable "
                "temporal-consistency judgment."
            )
            result.confidence = 0.1
            return result

        hf_series = np.array([m.hf_ratio for m in valid])
        edge_series = np.array([m.edge_density for m in valid])
        luma_series = np.array([m.mean_luma for m in valid])

        # --- Component 1: frame-to-frame "flicker" in frequency/edge/luma features ---
        # Uses the *magnitude* of frame-to-frame change relative to each
        # signal's own mean level (not a self-relative z-score) - a video
        # where every frame is uniformly noisy still needs to score as
        # unstable even though no single delta is a statistical "outlier"
        # relative to the others.
        def instability(series: np.ndarray) -> float:
            deltas = np.abs(np.diff(series))
            return float(deltas.mean() / (abs(series.mean()) + 1e-8))

        hf_instab = instability(hf_series)
        edge_instab = instability(edge_series)
        luma_instab = instability(luma_series)

        flicker_rate = float(np.clip(np.mean([
            hf_instab / self.cfg["hf_instability_norm"],
            edge_instab / self.cfg["edge_instability_norm"],
            luma_instab / self.cfg["luma_instability_norm"],
        ]), 0.0, 1.0))

        # --- Component 2: optical flow incoherence ---
        if flow_scores:
            mean_flow_coherence = float(np.mean(flow_scores))
            flow_incoherence = max(0.0, self.cfg["flow_coherence_thresh"] - mean_flow_coherence) \
                / self.cfg["flow_coherence_thresh"]
        else:
            mean_flow_coherence = 1.0
            flow_incoherence = 0.0

        # --- Component 3: temporal FFT of the hf_ratio signal -> periodic flicker ---
        temporal_periodicity = self._temporal_fft_anomaly(hf_series)

        # --- Component 4: raw instability (coefficient of variation) of hf energy ---
        hf_cv = float(hf_series.std() / (hf_series.mean() + 1e-8))
        hf_instability = float(np.clip(hf_cv / 0.5, 0.0, 1.0))  # 0.5 CV ~ saturates score

        w = self.cfg["weights"]
        score = (
            w["flicker"] * flicker_rate
            + w["flow_incoherence"] * flow_incoherence
            + w["temporal_periodicity"] * temporal_periodicity
            + w["hf_energy_instability"] * hf_instability
        )
        score = float(np.clip(score, 0.0, 1.0))

        result.score = round(score, 4)
        result.flagged = score >= self.cfg["score_flag_thresh"]
        result.confidence = round(min(1.0, len(valid) / self.cfg["window_frames"]), 2)
        result.components = {
            "flicker_rate": round(flicker_rate, 4),
            "mean_flow_coherence": round(mean_flow_coherence, 4),
            "flow_incoherence": round(flow_incoherence, 4),
            "temporal_periodicity_anomaly": round(temporal_periodicity, 4),
            "hf_energy_coefficient_of_variation": round(hf_cv, 4),
        }
        result.explanation = self._explain(result)
        return result

    def _temporal_fft_anomaly(self, signal: np.ndarray) -> float:
        """FFT along time of the hf_ratio signal. Real video's hf-ratio drifts
        slowly; a strong peak at a mid/high temporal frequency suggests
        periodic per-frame flicker typical of frame-by-frame generative
        synthesis."""
        n = len(signal)
        if n < 8:
            return 0.0
        detrended = signal - np.mean(signal)
        spectrum = np.abs(np.fft.rfft(detrended))
        if len(spectrum) < 4:
            return 0.0
        # ignore the DC/very-low-frequency bins (that's just natural drift)
        ac_spectrum = spectrum[2:]
        baseline = np.median(ac_spectrum) + 1e-8
        peak = ac_spectrum.max()
        ratio = peak / baseline
        anomaly = (ratio - self.cfg["temporal_fft_peak_thresh"]) / self.cfg["temporal_fft_peak_thresh"]
        return float(np.clip(anomaly, 0.0, 1.0))

    def _explain(self, r: LayerResult) -> str:
        c = r.components
        reasons = []
        if c["flicker_rate"] > 0.15:
            reasons.append("elevated frame-to-frame flicker in texture/lighting features")
        if c["flow_incoherence"] > 0.2:
            reasons.append("incoherent optical flow around the face region")
        if c["temporal_periodicity_anomaly"] > 0.2:
            reasons.append("periodic high-frequency spectral peak over time (synthesis artifact)")
        if c["hf_energy_coefficient_of_variation"] > 0.3:
            reasons.append("unstable high-frequency spectral energy across frames")

        if not reasons:
            return "Facial texture, lighting, and motion remained smoothly consistent across frames."
        verdict = "flagged as likely synthetic" if r.flagged else "within normal variability but worth monitoring"
        return f"Clip {verdict}: " + "; ".join(reasons) + "."


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Layer 3: Temporal Consistency & Frequency Analysis")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default=None, help="Optional path to write JSON result")
    args = parser.parse_args()

    analyzer = TemporalConsistencyAnalyzer()
    result = analyzer.analyze(args.video)
    output_json = result.to_json()

    print(output_json)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)


if __name__ == "__main__":
    sys.exit(main())
