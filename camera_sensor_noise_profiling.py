"""
Layer 2: Camera Sensor Noise Profiling (PRNU Analysis)
======================================================

Owner: Arihant
Part of: Injection Attack & Deepfake Detection pipeline
(Layer 1 = Aarya's hardware attestation, Layer 2 = this module,
 Layer 3 = Vibha's temporal & frequency analysis, Layer 5 = gated fusion in Ramya's pipeline)

What this layer does
---------------------
Every physical CMOS camera sensor exhibits a unique, microscopic
Photo-Response Non-Uniformity (PRNU) noise pattern, akin to a digital
fingerprint. This pattern arises from silicon substrate manufacturing
tolerances and remains stationary across all frames captured by that physical sensor.

Injected deepfakes, which are digitally rendered or aggressively compressed,
lack accurate, continuous PRNU:
  1. Lack of Inter-Frame PRNU Persistence: In a genuine camera stream, the
     underlying sensor noise pattern is physically bound to the pixel grid across
     all frames. Generative models render frames independently or synthesize
     uncorrelated noise, yielding near-zero inter-frame residual correlation.
  2. Sub-threshold Peak-to-Correlation Energy (PCE): Authentic sensor fingerprints
     produce a sharp delta peak on 2D circular cross-correlation surfaces (PCE > 50-100).
     Synthetically generated or injected video streams lack this spatial coherence,
     exhibiting flat cross-correlation surfaces (PCE < 20).
  3. Spatial Discrepancy & Face Region PRNU Absence: In facial injection attacks
     (e.g., face swaps or synthetic overlays), the face region lacks the sensor's
     stationary fingerprint while surrounding camera context may retain it.
  4. Reference Fingerprint Matching (Hardware Attestation): When calibrated against
     an enrolled camera fingerprint (e.g., verified via Layer 1 hardware attestation),
     injected streams fail both Normalized Cross-Correlation (NCC) and PCE tests.

This module extracts sensor noise residuals using edge-attenuated spatial filtering,
measures inter-frame PRNU persistence, analyzes 2D FFT cross-correlation surfaces and PCE,
and outputs a normalized anomaly score in [0, 1] with human-readable explanations.

Usage
-----
    # Autonomous blind analysis (no reference camera fingerprint needed)
    python camera_sensor_noise_profiling.py --video path/to/clip.mp4
    python camera_sensor_noise_profiling.py --video clip.mp4 --output result.json

    # Enroll camera sensor fingerprint from a genuine calibration clip
    python camera_sensor_noise_profiling.py --video genuine_clip.mp4 --enroll-ref fingerprint.npy

    # Verification against enrolled camera sensor fingerprint
    python camera_sensor_noise_profiling.py --video clip.mp4 --ref-fingerprint fingerprint.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration (tune against benchmark genuine / deepfake datasets)
# --------------------------------------------------------------------------- #

CONFIG = {
    "window_frames": 30,             # sliding analysis window (approx 1s at 30fps)
    "max_frames_to_accumulate": 90,  # frames used for estimating sensor fingerprint
    "filter_ksize": 5,               # Gaussian smoothing kernel size for base scene estimate
    "edge_thresh": 12.0,             # Sobel gradient threshold for structural scene edges
    "edge_dilation_ksize": 7,        # dilation kernel to mask edge transition skirts
    "pce_peak_exclusion_radius": 5,  # radius around center excluded when computing background energy
    "pce_authentic_thresh": 45.0,    # PCE score above which sensor match is deemed genuine
    "ncc_authentic_thresh": 0.05,    # zero-shift NCC above which noise correlation is significant
    "persistence_authentic_thresh": 0.20, # inter-frame noise residual correlation threshold
    "score_flag_thresh": 0.55,       # overall anomaly score threshold for flagging deepfake
    "weights": {
        "inter_frame_persistence": 0.35,  # lack of stationary sensor noise across frames
        "pce_correlation_energy": 0.35,   # absence of sharp cross-correlation peak (PCE)
        "normalized_cross_corr": 0.15,    # raw PRNU correlation magnitude
        "noise_floor_adequacy": 0.15,     # unnatural noise suppression / generative smoothing
    },
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class LayerResult:
    layer: str = "camera_sensor_noise_profiling"
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
# Noise Residual Extraction (Edge-Attenuated Spatial Filtering)
# --------------------------------------------------------------------------- #

def extract_noise_residual(
    gray_img: np.ndarray,
    ksize: int = 5,
    edge_thresh: float = 12.0,
    edge_dilation: int = 7,
) -> np.ndarray:
    """
    Extracts sensor noise residual W = (I - F(I)) * Mask_flat.
    Uses structural edge masking via smoothed Sobel gradients and dilation to ensure
    that moving object contours (face boundary, eyes, lips) do NOT contaminate the residual.
    Zero-means rows and columns to suppress readout/banding noise.
    """
    img_f = gray_img.astype(np.float32)

    # 1. Pre-smooth to detect genuine macro scene edges rather than fine sensor noise
    smoothed = cv2.GaussianBlur(img_f, (7, 7), sigmaX=2.0, sigmaY=2.0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # 2. Dilate edge mask to strictly eliminate edge transition skirts
    kernel = np.ones((edge_dilation, edge_dilation), np.uint8)
    edge_mask = cv2.dilate((grad_mag > edge_thresh).astype(np.uint8), kernel) > 0

    # 3. Spatial denoising to extract microscopic noise
    denoised = cv2.GaussianBlur(img_f, (ksize, ksize), sigmaX=1.2, sigmaY=1.2)
    residual = img_f - denoised

    # 4. Zero out all scene edges
    residual[edge_mask] = 0.0

    # 5. Subtract row and column means to suppress readout banding
    residual -= np.mean(residual, axis=0, keepdims=True)
    residual -= np.mean(residual, axis=1, keepdims=True)

    return residual


def zero_mean_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizes array to zero-mean and unit variance."""
    std = np.std(arr)
    if std < 1e-7:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - np.mean(arr)) / std).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2D Cross-Correlation & PCE (Peak-to-Correlation Energy)
# --------------------------------------------------------------------------- #

def compute_pce_and_ncc(
    residual: np.ndarray,
    fingerprint: np.ndarray,
    peak_radius: int = 5,
) -> Tuple[float, float]:
    """
    Computes 2D circular cross-correlation surface via 2D FFT:
    - NCC (Normalized Cross Correlation at zero shift)
    - PCE (Peak-to-Correlation Energy): ratio of squared zero-shift center peak
      to the average energy in the cross-correlation surface outside the peak neighborhood.
    """
    if residual.shape != fingerprint.shape:
        h, w = residual.shape[:2]
        fingerprint = cv2.resize(fingerprint, (w, h), interpolation=cv2.INTER_LINEAR)

    r_norm = zero_mean_normalize(residual)
    f_norm = zero_mean_normalize(fingerprint)

    h, w = r_norm.shape
    n_pixels = h * w

    # Zero-shift NCC
    ncc = float(np.sum(r_norm * f_norm) / (n_pixels + 1e-8))

    # 2D circular cross-correlation via FFT
    f_r = np.fft.fft2(r_norm)
    f_f = np.fft.fft2(f_norm)
    cross_corr = np.fft.ifft2(f_r * np.conj(f_f)).real / (n_pixels + 1e-8)
    cross_corr = np.fft.fftshift(cross_corr)

    # Center coordinates correspond to (0, 0) spatial shift
    center_y, center_x = h // 2, w // 2
    peak_val = cross_corr[center_y, center_x]

    # Mask excluding peak neighborhood around (0, 0)
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
    non_peak_mask = dist_sq > (peak_radius ** 2)

    non_peak_values = cross_corr[non_peak_mask]
    if len(non_peak_values) == 0:
        return 0.0, ncc

    mean_sq_energy = float(np.mean(non_peak_values ** 2))
    if mean_sq_energy < 1e-12:
        pce = float(peak_val ** 2 / 1e-12)
    else:
        pce = float((peak_val ** 2) / mean_sq_energy)

    return float(pce), float(ncc)


# --------------------------------------------------------------------------- #
# Face Detection & Tracking
# --------------------------------------------------------------------------- #

class FaceTracker:
    """Lightweight Haar-cascade face detector with bounding box temporal smoothing."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.last_box: Optional[Tuple[int, int, int, int]] = None

    def detect(self, gray_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )
        if len(faces) == 0:
            return self.last_box

        largest = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = [int(v) for v in largest]

        if self.last_box is not None:
            alpha = 0.75
            lx, ly, lw, lh = self.last_box
            x = int(alpha * x + (1 - alpha) * lx)
            y = int(alpha * y + (1 - alpha) * ly)
            w = int(alpha * w + (1 - alpha) * lw)
            h = int(alpha * h + (1 - alpha) * lh)

        self.last_box = (x, y, w, h)
        return self.last_box


# --------------------------------------------------------------------------- #
# Main Analyzer: Camera Sensor Noise Profiler
# --------------------------------------------------------------------------- #

class CameraSensorNoiseProfiler:
    """
    Analyzes video frames for physical CMOS camera sensor PRNU noise characteristics.
    Detects digital face injection and synthetic deepfakes by evaluating:
      1. Inter-frame PRNU persistence across the video sequence
      2. 2D cross-correlation peak-to-correlation energy (PCE) at zero shift
      3. Zero-shift normalized cross-correlation (NCC)
      4. High-frequency sensor noise floor adequacy
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = {**CONFIG, **(config or {})}
        self.face_tracker = FaceTracker()

    def estimate_sensor_fingerprint(self, residuals: list[np.ndarray], frames: list[np.ndarray]) -> np.ndarray:
        """
        Estimates maximum-likelihood camera PRNU sensor fingerprint K:
        K = sum(W_k * I_k) / sum(I_k^2)
        """
        if not residuals:
            return np.zeros((320, 320), dtype=np.float32)

        numerator = np.zeros_like(residuals[0], dtype=np.float32)
        denominator = np.zeros_like(residuals[0], dtype=np.float32)

        for res, frame in zip(residuals, frames):
            frame_f = frame.astype(np.float32)
            numerator += res * frame_f
            denominator += (frame_f ** 2)

        fingerprint = np.divide(numerator, denominator + 1e-6)
        fingerprint -= np.mean(fingerprint)
        return fingerprint

    def enroll_fingerprint_from_video(self, video_path: str, output_npy_path: str) -> bool:
        """
        Extracts and saves the camera PRNU sensor fingerprint from a genuine calibration video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}", file=sys.stderr)
            return False

        residuals = []
        frames_acc = []
        count = 0

        while count < self.cfg["max_frames_to_accumulate"]:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = extract_noise_residual(
                gray,
                ksize=self.cfg["filter_ksize"],
                edge_thresh=self.cfg["edge_thresh"],
                edge_dilation=self.cfg["edge_dilation_ksize"],
            )
            residuals.append(res)
            frames_acc.append(gray)
            count += 1

        cap.release()

        if not residuals:
            print("Error: No frames could be processed for enrollment", file=sys.stderr)
            return False

        fingerprint = self.estimate_sensor_fingerprint(residuals, frames_acc)
        np.save(output_npy_path, fingerprint)
        print(f"Enrolled camera PRNU fingerprint ({fingerprint.shape}) saved to {output_npy_path}")
        return True

    def analyze(self, video_path: str, ref_fingerprint_path: Optional[str] = None) -> LayerResult:
        """
        Main analysis method. Analyzes video frames and returns structured LayerResult.
        """
        cap = cv2.VideoCapture(video_path)
        result = LayerResult()

        if not cap.isOpened():
            result.explanation = f"Could not open input video: {video_path}"
            return result

        ref_fingerprint = None
        if ref_fingerprint_path:
            try:
                ref_fingerprint = np.load(ref_fingerprint_path).astype(np.float32)
            except Exception as e:
                print(f"Warning: Failed to load reference fingerprint: {e}", file=sys.stderr)

        residuals: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        face_vars: list[float] = []

        total_frames = 0
        frames_with_face = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = self.face_tracker.detect(gray)

            res = extract_noise_residual(
                gray,
                ksize=self.cfg["filter_ksize"],
                edge_thresh=self.cfg["edge_thresh"],
                edge_dilation=self.cfg["edge_dilation_ksize"],
            )

            if len(residuals) < self.cfg["max_frames_to_accumulate"]:
                residuals.append(res)
                frames.append(gray)

            if box is not None:
                x, y, w, h = box
                ih, iw = gray.shape
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(iw, x + w), min(ih, y + h)

                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    frames_with_face += 1
                    face_pixels = res[y1:y2, x1:x2]
                    valid_pixels = face_pixels[face_pixels != 0.0]
                    if valid_pixels.size > 0:
                        face_vars.append(float(np.var(valid_pixels)))

        cap.release()

        result.frames_analyzed = total_frames
        result.frames_with_face = frames_with_face

        if total_frames == 0:
            result.explanation = "Input video contained 0 readable frames."
            return result

        # ------------------------------------------------------------------- #
        # Sensor PRNU Analysis
        # ------------------------------------------------------------------- #
        has_ref = ref_fingerprint is not None

        if has_ref:
            # Mode A: Known camera reference fingerprint matching
            acc_res = np.mean(residuals, axis=0)
            pce_score, ncc_score = compute_pce_and_ncc(
                acc_res,
                ref_fingerprint,
                peak_radius=self.cfg["pce_peak_exclusion_radius"],
            )
            inter_frame_persistence = ncc_score
        else:
            # Mode B: Autonomous blind split-half PRNU consistency analysis
            odd_res = residuals[0::2]
            odd_frames = frames[0::2]
            even_res = residuals[1::2]
            even_frames = frames[1::2]

            fp_odd = self.estimate_sensor_fingerprint(odd_res, odd_frames)
            fp_even = self.estimate_sensor_fingerprint(even_res, even_frames)

            pce_score, ncc_score = compute_pce_and_ncc(
                fp_odd,
                fp_even,
                peak_radius=self.cfg["pce_peak_exclusion_radius"],
            )

            # Inter-frame persistence across consecutive frames
            corrs = []
            step = max(1, len(residuals) // 30)
            for i in range(0, len(residuals) - 1, step):
                r1 = zero_mean_normalize(residuals[i])
                r2 = zero_mean_normalize(residuals[i + 1])
                corrs.append(float(np.mean(r1 * r2)))
            inter_frame_persistence = float(np.mean(corrs)) if corrs else 0.0

        # Anomaly components in [0, 1]
        # 1. Inter-frame persistence anomaly: real camera > 0.20; deepfake ~ 0.0
        persistence_thresh = self.cfg["persistence_authentic_thresh"]
        persistence_anomaly = float(np.clip(1.0 - (inter_frame_persistence / persistence_thresh), 0.0, 1.0))

        # 2. PCE anomaly: real camera > 45.0; deepfake < 20.0
        pce_thresh = self.cfg["pce_authentic_thresh"]
        pce_anomaly = float(np.clip(1.0 - (pce_score / pce_thresh), 0.0, 1.0))

        # 3. NCC anomaly: real camera > 0.05; deepfake ~ 0.0
        ncc_thresh = self.cfg["ncc_authentic_thresh"]
        ncc_anomaly = float(np.clip(1.0 - (ncc_score / ncc_thresh), 0.0, 1.0))

        # 4. Noise floor adequacy (checks for generative over-smoothing)
        mean_face_var = float(np.mean(face_vars)) if face_vars else 1.0
        noise_floor_adequacy = float(np.clip(mean_face_var / 0.5, 0.0, 1.0))
        smoothing_anomaly = float(1.0 - noise_floor_adequacy)

        # Weighted aggregate anomaly score in [0, 1]
        w = self.cfg["weights"]
        score = (
            w["inter_frame_persistence"] * persistence_anomaly
            + w["pce_correlation_energy"] * pce_anomaly
            + w["normalized_cross_corr"] * ncc_anomaly
            + w["noise_floor_adequacy"] * smoothing_anomaly
        )
        score = float(np.clip(score, 0.0, 1.0))

        valid_ratio = frames_with_face / max(1, min(total_frames, self.cfg["window_frames"]))
        confidence = float(np.clip(min(1.0, total_frames / 15.0) * (0.5 + 0.5 * valid_ratio), 0.1, 1.0))

        result.score = round(score, 4)
        result.flagged = score >= self.cfg["score_flag_thresh"]
        result.confidence = round(confidence, 2)
        result.components = {
            "inter_frame_persistence": round(inter_frame_persistence, 4),
            "persistence_anomaly": round(persistence_anomaly, 4),
            "pce_score": round(pce_score, 2),
            "pce_anomaly": round(pce_anomaly, 4),
            "prnu_correlation": round(ncc_score, 4),
            "ncc_anomaly": round(ncc_anomaly, 4),
            "noise_floor_adequacy": round(noise_floor_adequacy, 4),
        }
        result.explanation = self._explain(result, has_ref)
        return result

    def _explain(self, r: LayerResult, has_ref_fingerprint: bool) -> str:
        c = r.components
        reasons = []

        if c["persistence_anomaly"] > 0.4:
            reasons.append(f"lack of stationary CMOS sensor PRNU noise persistence (corr={c['inter_frame_persistence']:.3f})")
        if c["pce_anomaly"] > 0.4:
            reasons.append(f"absence of sharp cross-correlation energy peak (PCE={c['pce_score']:.1f})")
        if c["ncc_anomaly"] > 0.4:
            reasons.append(f"near-zero PRNU correlation coefficient ({c['prnu_correlation']:.4f})")
        if c["noise_floor_adequacy"] < 0.6:
            reasons.append("unnatural facial noise suppression characteristic of generative synthesis")

        if not reasons:
            return "Authentic CMOS sensor PRNU fingerprint confirmed across video frames."

        verdict = "flagged as likely synthetic / injected deepfake" if r.flagged else "minor sensor noise irregularities detected"
        return f"Clip {verdict}: " + "; ".join(reasons) + "."


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Layer 2: Camera Sensor Noise Profiling (PRNU Deepfake Detection)"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default=None, help="Optional path to save JSON result")
    parser.add_argument(
        "--ref-fingerprint",
        default=None,
        help="Optional path to enrolled reference camera PRNU .npy fingerprint",
    )
    parser.add_argument(
        "--enroll-ref",
        default=None,
        help="Optional path to save camera PRNU fingerprint extracted from genuine --video",
    )

    args = parser.parse_args()
    profiler = CameraSensorNoiseProfiler()

    if args.enroll_ref:
        success = profiler.enroll_fingerprint_from_video(args.video, args.enroll_ref)
        if not success:
            sys.exit(1)
        return 0

    result = profiler.analyze(args.video, ref_fingerprint_path=args.ref_fingerprint)
    output_json = result.to_json()

    print(output_json)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
