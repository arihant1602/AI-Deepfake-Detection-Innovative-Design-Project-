"""
Layer 2: Camera Sensor Noise Profiling (PRNU Analysis)
======================================================

Owner: Arihant
Part of: Injection Attack & Deepfake Detection pipeline
(Layer 1 = hardware attestation, Layer 2 = this module,
 Layer 3 = temporal & frequency analysis, Layer 5 = gated fusion)

What this layer does
--------------------
Every CMOS sensor multiplies the incoming light by a fixed, pixel-specific gain
pattern K (Photo-Response Non-Uniformity, PRNU). A frame therefore carries a
noise component I * K that is *locked to the pixel grid*. Frames rendered by a
generator, re-animated from a photo, or resampled/warped by a face-swap model do
not carry a pixel-locked pattern that survives from one part of the clip to the
next.

Two modes are provided:

  Mode B - blind, motion-gated PRNU persistence (default, no enrollment)
    The clip is split into two temporally disjoint halves A and B and a
    fingerprint K_A, K_B is estimated from each (MLE estimator, Lukas/Fridrich/
    Goljan 2006; Chen et al. 2008). With a static camera, static background
    *texture* is also pixel-locked and would masquerade as PRNU, so the two
    fingerprints are compared **only over pixels whose scene content changed
    between A and B** (the moving subject). Correlation that survives there can
    only come from a pattern fixed to the sensor grid. Significance is measured
    against a circular-shift null distribution, so spatially correlated residue
    and codec block-grid artifacts do not inflate the result.

    Outcomes:
      PRESENT       pixel-locked sensor pattern found under content change
      ABSENT        enough content change, but no pixel-locked pattern -> flag
      INCONCLUSIVE  too little content change to decide -> never flagged

  Mode A - enrolled reference fingerprint (camera bound to Layer 1 identity)
    The clip's residuals are correlated against a fingerprint previously
    enrolled for the *attested* physical camera and scored with the standard
    Peak-to-Correlation Energy (PCE; Goljan et al. 2009). A stream that did not
    come from that sensor - including a replay of genuine footage recorded on a
    different camera - fails the match.

Every step is O(pixels x frames) with box filters and FFTs (no learned model).

Usage
-----
    python camera_sensor_noise_profiling.py --video clip.mp4
    python camera_sensor_noise_profiling.py --video calib.mp4 --enroll-ref fp.npy
    python camera_sensor_noise_profiling.py --video clip.mp4 --ref-fingerprint fp.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CONFIG = {
    "max_frames": 90,                 # frames analysed (first N of the clip)
    "min_frames": 16,                 # below this the layer abstains
    "denoise_sigma": 3.0,             # sigma_0 of the local Wiener denoiser (8-bit units)
    "denoise_windows": (3, 5, 7, 9),  # Mihcak-style minimum-variance window set
    "saturation_low": 5,              # pixels at/below this carry no PRNU
    "saturation_high": 250,           # pixels at/above this carry no PRNU
    "dft_peak_factor": 3.0,           # spectral peaks above k x local median are clamped
    # --- blind mode (motion gating) ---
    "change_blur_sigma": 3.0,         # low-pass applied before measuring content change
    "change_thresh": 10.0,            # grey-level change of the low-passed mean image
    "change_erode": 5,                # erosion kernel applied to the change mask
    "change_max_intensity": 230,      # highlights (ISP tone-curve shoulder) carry little usable PRNU
    "min_changed_fraction": 0.04,     # need >=4% of the frame to have changed ...
    "min_changed_pixels": 4000,       # ... and at least this many pixels
    "null_shifts": 48,                # circular-shift null samples
    "null_min_shift": 6,              # shifts closer than this to (0,0) are not used
    "null_max_shift": 40,
    "z_present": 5.0,                 # z-score at/above which PRNU is PRESENT ...
    "rho_present": 0.05,              # ... provided the correlation is also at least this
    "z_absent": 3.0,                  # z-score below which PRNU is ABSENT
    "rho_absent": 0.04,               # correlation below which PRNU is ABSENT
    # --- reference mode ---
    "pce_peak_radius": 2,
    "pce_search_radius": 2,           # the peak may sit a few pixels off (0,0): sensor-mode scaler/crop phase
    "pce_match_thresh": 60.0,         # standard PCE decision threshold (Goljan 2009)
    "random_seed": 1234,
}

VERDICT_PRESENT = "PRESENT"
VERDICT_ABSENT = "ABSENT"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class LayerResult:
    layer: str = "camera_sensor_noise_profiling"
    mode: str = "blind"
    verdict: str = VERDICT_INCONCLUSIVE
    frames_analyzed: int = 0
    frames_with_face: int = 0          # kept for schema compatibility; PRNU does not use faces
    score: float = 0.5                 # anomaly score in [0, 1]; 0.5 when inconclusive
    flagged: bool = False              # True only when verdict == ABSENT / reference mismatch
    confidence: float = 0.0
    components: dict = field(default_factory=dict)
    explanation: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


# --------------------------------------------------------------------------- #
# Noise residual and fingerprint estimation
# --------------------------------------------------------------------------- #

def to_gray_float(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32)


def extract_noise_residual(
    gray: np.ndarray,
    sigma0: float = CONFIG["denoise_sigma"],
    windows: Sequence[int] = CONFIG["denoise_windows"],
    sat_low: float = CONFIG["saturation_low"],
    sat_high: float = CONFIG["saturation_high"],
) -> np.ndarray:
    """
    Noise residual W = I - F(I) using a spatially adaptive local Wiener filter.

    The local signal variance is the minimum over several window sizes (as in
    Mihcak's wavelet denoiser used by the PRNU literature), so textured regions
    are attenuated by sigma0^2 / var and flat regions keep their full residual.
    Saturated / black pixels are zeroed because they carry no PRNU.
    """
    g = gray.astype(np.float32)
    min_var = None
    for k in windows:
        mu = cv2.boxFilter(g, -1, (k, k), borderType=cv2.BORDER_REFLECT)
        var = cv2.boxFilter(g * g, -1, (k, k), borderType=cv2.BORDER_REFLECT) - mu * mu
        min_var = var if min_var is None else np.minimum(min_var, var)
    mu3 = cv2.boxFilter(g, -1, (3, 3), borderType=cv2.BORDER_REFLECT)
    attenuation = np.minimum(1.0, (sigma0 ** 2) / np.maximum(min_var, 1e-6))
    residual = (g - mu3) * attenuation
    residual[(g <= sat_low) | (g >= sat_high)] = 0.0
    return residual.astype(np.float32)


def zero_mean_rows_cols(x: np.ndarray) -> np.ndarray:
    """Removes row/column means (readout banding, linear-pattern artifacts)."""
    x = x - x.mean(axis=0, keepdims=True)
    return x - x.mean(axis=1, keepdims=True)


def suppress_periodic_artifacts(x: np.ndarray, peak_factor: float = CONFIG["dft_peak_factor"]) -> np.ndarray:
    """
    Wiener-in-DFT style cleanup: spectral peaks (JPEG/H.264 block grids, CFA
    interpolation, demosaicing periodicities) are shared by every camera and every
    codec, so they are clamped to `peak_factor` x the local median magnitude.
    """
    spec = np.fft.fft2(x)
    mag = np.abs(spec).astype(np.float32)
    local = np.fft.ifftshift(cv2.medianBlur(np.fft.fftshift(mag), 5))
    scale = np.minimum(1.0, (peak_factor * local + 1e-9) / (mag + 1e-9))
    return np.real(np.fft.ifft2(spec * scale)).astype(np.float32)


def zero_mean_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizes array to zero mean and unit variance."""
    std = np.std(arr)
    if std < 1e-7:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - np.mean(arr)) / std).astype(np.float32)


def estimate_fingerprint(grays: Sequence[np.ndarray], residuals: Sequence[np.ndarray]) -> np.ndarray:
    """MLE PRNU estimate K = sum(W_k I_k) / sum(I_k^2), followed by artifact removal."""
    num = np.zeros_like(grays[0], dtype=np.float32)
    den = np.zeros_like(grays[0], dtype=np.float32)
    for g, w in zip(grays, residuals):
        num += w * g
        den += g * g
    k = num / (den + 1.0)
    return suppress_periodic_artifacts(zero_mean_rows_cols(k))


def compute_pce_and_ncc(a: np.ndarray, b: np.ndarray, peak_radius: int = CONFIG["pce_peak_radius"]) -> Tuple[float, float]:
    """
    Circular cross-correlation of `a` and `b` (via FFT).
    Returns (PCE, NCC) where PCE = sign(C0) * C0^2 / mean(C^2 outside the peak
    neighbourhood) and NCC is the normalized correlation at zero shift.
    """
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0, 0.0
    corr = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    peak = corr[0, 0]
    h, w = corr.shape
    yy, xx = np.ogrid[:h, :w]
    dy = np.minimum(yy, h - yy)
    dx = np.minimum(xx, w - xx)
    outside = (dy ** 2 + dx ** 2) > peak_radius ** 2
    energy = float(np.mean(corr[outside] ** 2))
    if energy < 1e-30:
        return 0.0, float(peak / denom)
    return float(np.sign(peak) * peak ** 2 / energy), float(peak / denom)


def reference_correlation_surface(frames: Sequence[np.ndarray], ref: np.ndarray, half_width: int = 20,
                                  cfg: Optional[dict] = None) -> Tuple[float, np.ndarray, Tuple[int, int]]:
    """
    Same statistic as reference mode, but also returns the centre of the circular
    cross-correlation surface (in units of its off-peak RMS) for visualisation: a
    sensor match shows one sharp spike at zero shift, a mismatch is flat noise.
    """
    cfg = {**CONFIG, **(cfg or {})}
    grays = [to_gray_float(f) for f in frames]
    w_sum = np.zeros_like(grays[0])
    ik_sum = np.zeros_like(grays[0])
    for g in grays:
        w_sum += extract_noise_residual(g, cfg["denoise_sigma"], cfg["denoise_windows"])
        ik_sum += g * ref
    a = suppress_periodic_artifacts(zero_mean_rows_cols(w_sum)).astype(np.float64)
    b = ik_sum.astype(np.float64)
    pce, pdy, pdx = pce_with_shift_search(a, b, cfg["pce_search_radius"], cfg["pce_peak_radius"])
    a -= a.mean()
    b -= b.mean()
    corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))))
    h, w = corr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    background = corr[((yy - cy) ** 2 + (xx - cx) ** 2) > cfg["pce_peak_radius"] ** 2]
    rms = float(np.sqrt(np.mean(background ** 2))) or 1.0
    return pce, corr[cy - half_width: cy + half_width + 1, cx - half_width: cx + half_width + 1] / rms, (pdy, pdx)


def pce_with_shift_search(a: np.ndarray, b: np.ndarray, search_radius: int = CONFIG["pce_search_radius"],
                          peak_radius: int = CONFIG["pce_peak_radius"]) -> Tuple[float, int, int]:
    """
    PCE at the strongest peak within +/-search_radius pixels of zero shift (Goljan et al.
    search the peak when the alignment is not exactly known). Returns (pce, dy, dx).
    Genuine webcam sessions were measured with peaks at (0, +/-2) and (-1, 0): the
    camera's scaler/crop phase differs slightly from one stream start to the next.
    """
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    corr = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    h, w = corr.shape
    sq = corr ** 2
    total, n = float(sq.sum()), corr.size
    offsets = [(dy, dx) for dy in range(-peak_radius, peak_radius + 1) for dx in range(-peak_radius, peak_radius + 1)
               if dy * dy + dx * dx <= peak_radius * peak_radius]
    best = (-np.inf, 0, 0)
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            peak = corr[dy % h, dx % w]
            if peak <= 0:
                continue
            excl = sum(sq[(dy + oy) % h, (dx + ox) % w] for oy, ox in offsets)
            energy = (total - excl) / (n - len(offsets))
            pce = peak * peak / max(energy, 1e-30)
            if pce > best[0]:
                best = (float(pce), dy, dx)
    if best[0] == -np.inf:
        peak = corr[0, 0]
        excl = sum(sq[oy % h, ox % w] for oy, ox in offsets)
        energy = (total - excl) / (n - len(offsets))
        return float(np.sign(peak) * peak * peak / max(energy, 1e-30)), 0, 0
    return best


# --------------------------------------------------------------------------- #
# Frame I/O
# --------------------------------------------------------------------------- #

def read_video_frames(video_path: str, max_frames: int) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #

class CameraSensorNoiseProfiler:
    """Blind motion-gated PRNU persistence test, plus enrolled-reference PCE matching."""

    def __init__(self, config: Optional[dict] = None):
        self.cfg = {**CONFIG, **(config or {})}

    # ---------------------------- public API ---------------------------- #

    def analyze(self, video_path: str, ref_fingerprint_path: Optional[str] = None) -> LayerResult:
        try:
            frames, _ = read_video_frames(video_path, self.cfg["max_frames"])
        except IOError as e:
            return LayerResult(explanation=str(e))
        ref = None
        if ref_fingerprint_path:
            try:
                ref = np.load(ref_fingerprint_path).astype(np.float32)
            except Exception as e:
                print(f"Warning: failed to load reference fingerprint: {e}", file=sys.stderr)
        return self.analyze_frames(frames, ref_fingerprint=ref)

    def analyze_frames(self, frames: Sequence[np.ndarray], ref_fingerprint: Optional[np.ndarray] = None) -> LayerResult:
        frames = list(frames)[: self.cfg["max_frames"]]
        result = LayerResult(mode="reference" if ref_fingerprint is not None else "blind")
        result.frames_analyzed = len(frames)
        if len(frames) == 0:
            result.explanation = "Input video contained 0 readable frames."
            return result
        if len(frames) < self.cfg["min_frames"]:
            result.explanation = (
                f"Only {len(frames)} frames available; at least {self.cfg['min_frames']} are needed "
                "to estimate a sensor pattern. Layer abstains."
            )
            return result

        grays = [to_gray_float(f) for f in frames]
        residuals = [
            extract_noise_residual(
                g, self.cfg["denoise_sigma"], self.cfg["denoise_windows"],
                self.cfg["saturation_low"], self.cfg["saturation_high"],
            )
            for g in grays
        ]

        if ref_fingerprint is not None:
            return self._reference_match(result, grays, residuals, ref_fingerprint)
        return self._blind_gated(result, grays, residuals)

    def estimate_fingerprint_from_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        grays = [to_gray_float(f) for f in frames]
        residuals = [extract_noise_residual(g, self.cfg["denoise_sigma"], self.cfg["denoise_windows"]) for g in grays]
        return estimate_fingerprint(grays, residuals)

    def enroll_fingerprint_from_video(self, video_path: str, output_npy_path: str, max_frames: int = 300) -> bool:
        """Estimates and saves the camera fingerprint from a genuine calibration clip."""
        try:
            frames, _ = read_video_frames(video_path, max_frames)
        except IOError as e:
            print(f"Error: {e}", file=sys.stderr)
            return False
        if len(frames) < self.cfg["min_frames"]:
            print("Error: not enough frames for enrollment", file=sys.stderr)
            return False
        fp = self.estimate_fingerprint_from_frames(frames)
        np.save(output_npy_path, fp)
        print(f"Enrolled camera PRNU fingerprint {fp.shape} from {len(frames)} frames -> {output_npy_path}")
        return True

    # ---------------------------- blind mode ---------------------------- #

    def _change_mask(self, grays_a: Sequence[np.ndarray], grays_b: Sequence[np.ndarray]) -> np.ndarray:
        """
        Pixels whose low-passed scene content differs between the two halves. Only a
        *between-half* difference guarantees that the scene texture leaking into K_A and
        K_B is decorrelated: motion that merely oscillates (both halves revisit the same
        content) leaks the same texture into both fingerprints and would mimic PRNU, so
        such pixels are excluded. Each frame is gain-normalised first, so a global
        exposure change of a static scene does not count as content change.
        """
        sigma = self.cfg["change_blur_sigma"]

        def gain_normalised_mean(grays):
            acc = None
            ref = None
            for g in grays:
                lp = cv2.GaussianBlur(g, (0, 0), sigma)
                m = float(lp.mean())
                ref = m if ref is None else ref
                lp *= ref / max(m, 1e-6)
                acc = lp if acc is None else acc + lp
            return acc / float(len(grays))

        mean_a = gain_normalised_mean(grays_a)
        mean_b = gain_normalised_mean(grays_b)
        mean_b *= float(mean_a.mean()) / max(float(mean_b.mean()), 1e-6)
        mask = np.abs(mean_a - mean_b) > self.cfg["change_thresh"]
        k = self.cfg["change_erode"]
        mask = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        lo, hi = self.cfg["saturation_low"], self.cfg["change_max_intensity"]
        mask &= (mean_a > lo) & (mean_a < hi) & (mean_b > lo) & (mean_b < hi)
        return mask

    def _masked_shift_test(self, fp_a: np.ndarray, fp_b: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
        a = fp_a[mask].astype(np.float64)
        a -= a.mean()
        a_norm = np.linalg.norm(a)

        def ncc(b_full: np.ndarray) -> float:
            b = b_full[mask].astype(np.float64)
            b -= b.mean()
            return float(np.dot(a, b) / (a_norm * np.linalg.norm(b) + 1e-12))

        rho = ncc(fp_b)
        rng = np.random.default_rng(self.cfg["random_seed"])
        lo, hi = self.cfg["null_min_shift"], self.cfg["null_max_shift"]
        nulls = []
        while len(nulls) < self.cfg["null_shifts"]:
            dy, dx = (int(v) for v in rng.integers(-hi, hi + 1, size=2))
            if abs(dy) < lo and abs(dx) < lo:
                continue
            nulls.append(ncc(np.roll(fp_b, (dy, dx), axis=(0, 1))))
        null_mean, null_std = float(np.mean(nulls)), float(np.std(nulls))
        z = (rho - null_mean) / (null_std + 1e-12)
        return rho, z, null_mean, null_std

    def _blind_gated(self, result: LayerResult, grays, residuals) -> LayerResult:
        cfg = self.cfg
        half = len(grays) // 2
        fp_a = estimate_fingerprint(grays[:half], residuals[:half])
        fp_b = estimate_fingerprint(grays[half:], residuals[half:])
        mask = self._change_mask(grays[:half], grays[half:])

        n_changed = int(mask.sum())
        frac_changed = n_changed / mask.size
        components = {
            "changed_fraction": round(frac_changed, 4),
            "changed_pixels": n_changed,
            "frames_per_half": half,
        }

        enough_change = frac_changed >= cfg["min_changed_fraction"] and n_changed >= cfg["min_changed_pixels"]
        if not enough_change:
            result.verdict = VERDICT_INCONCLUSIVE
            result.score = 0.5
            result.flagged = False
            result.confidence = round(min(1.0, frac_changed / cfg["min_changed_fraction"]) * 0.3, 2)
            result.components = {**components, "rho": None, "z_score": None, "static_noise_detected": None}
            result.explanation = (
                f"Inconclusive: only {frac_changed:.1%} of the frame changed content during the clip. "
                "With a static scene, background texture is as pixel-locked as sensor noise, so sensor "
                "PRNU cannot be isolated. Ask the subject to move (e.g. turn head) and re-capture."
            )
            return result

        rho, z, null_mean, null_std = self._masked_shift_test(fp_a, fp_b, mask)
        components.update({
            "rho": round(rho, 4),
            "z_score": round(z, 2),
            "null_mean": round(null_mean, 5),
            "null_std": round(null_std, 5),
        })

        if z >= cfg["z_present"] and rho >= cfg["rho_present"]:
            verdict = VERDICT_PRESENT
        elif z < cfg["z_absent"] or rho < cfg["rho_absent"]:
            verdict = VERDICT_ABSENT
        else:
            verdict = VERDICT_INCONCLUSIVE

        # Anomaly score: logistic in z centred between the ABSENT and PRESENT thresholds.
        z_mid = 0.5 * (cfg["z_absent"] + cfg["z_present"])
        score = float(1.0 / (1.0 + np.exp(np.clip(z - z_mid, -50, 50))))
        if verdict == VERDICT_ABSENT and rho < cfg["rho_absent"]:
            score = max(score, 0.75)

        result.verdict = verdict
        result.flagged = verdict == VERDICT_ABSENT
        result.score = round(score, 4)
        coverage = min(1.0, n_changed / (10 * cfg["min_changed_pixels"]))
        result.confidence = round(float(np.clip(0.4 + 0.6 * coverage, 0.0, 1.0)) if verdict != VERDICT_INCONCLUSIVE else 0.3, 2)
        components["static_noise_detected"] = verdict == VERDICT_PRESENT
        result.components = components
        result.explanation = self._explain_blind(verdict, components)
        return result

    @staticmethod
    def _explain_blind(verdict: str, c: dict) -> str:
        stats = f"rho={c['rho']:.3f}, z={c['z_score']:.1f} over {c['changed_fraction']:.1%} of the frame"
        if verdict == VERDICT_PRESENT:
            return (
                "Pixel-locked sensor noise (PRNU) persists across disjoint halves of the clip in regions "
                f"whose content changed ({stats}): consistent with a physical CMOS sensor."
            )
        if verdict == VERDICT_ABSENT:
            return (
                "No pixel-locked sensor pattern survives in the moving regions of the clip "
                f"({stats}). Frames were rendered, re-animated, warped or resampled rather than "
                "captured natively by a camera sensor."
            )
        return f"Weak sensor-pattern evidence ({stats}); neither PRESENT nor ABSENT thresholds reached."

    # -------------------------- reference mode -------------------------- #

    def _reference_match(self, result: LayerResult, grays, residuals, ref: np.ndarray) -> LayerResult:
        if ref.shape != grays[0].shape:
            result.verdict = VERDICT_INCONCLUSIVE
            result.explanation = (
                f"Enrolled fingerprint is {ref.shape[1]}x{ref.shape[0]} but the stream is "
                f"{grays[0].shape[1]}x{grays[0].shape[0]}; PRNU is only comparable in the same sensor mode."
            )
            result.components = {"reference_shape": list(ref.shape), "stream_shape": list(grays[0].shape)}
            return result

        w_sum = np.zeros_like(grays[0], dtype=np.float32)
        ik_sum = np.zeros_like(grays[0], dtype=np.float32)
        for g, w in zip(grays, residuals):
            w_sum += w
            ik_sum += g * ref
        w_clean = suppress_periodic_artifacts(zero_mean_rows_cols(w_sum))
        _, ncc = compute_pce_and_ncc(w_clean, ik_sum)
        pce, dy, dx = pce_with_shift_search(w_clean, ik_sum, self.cfg["pce_search_radius"], self.cfg["pce_peak_radius"])

        matched = pce >= self.cfg["pce_match_thresh"]
        result.verdict = VERDICT_PRESENT if matched else VERDICT_ABSENT
        result.flagged = not matched
        result.score = round(float(1.0 / (1.0 + np.exp(np.clip((np.log10(max(pce, 1e-3)) - np.log10(self.cfg["pce_match_thresh"])) * 4.0, -50, 50)))), 4)
        result.confidence = 0.9
        result.components = {
            "pce_score": round(pce, 2),
            "peak_offset": [dy, dx],
            "ncc": round(ncc, 4),
            "pce_match_thresh": self.cfg["pce_match_thresh"],
            "static_noise_detected": matched,
        }
        if matched:
            result.explanation = f"Stream matches the enrolled sensor fingerprint (PCE={pce:.1f} >= {self.cfg['pce_match_thresh']:.0f})."
        else:
            result.explanation = (
                f"Stream does NOT match the enrolled sensor fingerprint (PCE={pce:.1f} < "
                f"{self.cfg['pce_match_thresh']:.0f}): frames did not originate from the attested camera."
            )
        return result


# --------------------------------------------------------------------------- #
# Fingerprint store (binds Layer 2 to the Layer 1 camera identity)
# --------------------------------------------------------------------------- #

class FingerprintStore:
    """
    Enrolled fingerprints keyed by the physical camera identity reported by Layer 1
    (e.g. 'usb-5986-216a-1-3') and the sensor mode (WxH). PRNU is only comparable
    within one sensor mode, so each resolution is enrolled separately.
    """

    def __init__(self, root: str = "fingerprints"):
        self.root = root

    def _base(self, camera_id: str, width: int, height: int) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in camera_id)
        return os.path.join(self.root, f"{safe}_{width}x{height}")

    def has(self, camera_id: str, width: int, height: int) -> bool:
        return os.path.exists(self._base(camera_id, width, height) + ".npy")

    def load(self, camera_id: str, width: int, height: int) -> Optional[np.ndarray]:
        path = self._base(camera_id, width, height) + ".npy"
        return np.load(path).astype(np.float32) if os.path.exists(path) else None

    def metadata(self, camera_id: str, width: int, height: int) -> Optional[dict]:
        path = self._base(camera_id, width, height) + ".json"
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def save(self, camera_id: str, fingerprint: np.ndarray, meta: Optional[dict] = None) -> str:
        os.makedirs(self.root, exist_ok=True)
        h, w = fingerprint.shape
        base = self._base(camera_id, w, h)
        np.save(base + ".npy", fingerprint.astype(np.float32))
        with open(base + ".json", "w") as f:
            json.dump({"camera_id": camera_id, "width": w, "height": h, "enrolled_at": time.time(), **(meta or {})}, f, indent=2)
        return base + ".npy"

    def delete(self, camera_id: str, width: int, height: int) -> None:
        for ext in (".npy", ".json"):
            path = self._base(camera_id, width, height) + ext
            if os.path.exists(path):
                os.remove(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Layer 2: Camera Sensor Noise Profiling (PRNU)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default=None, help="Optional path to save JSON result")
    parser.add_argument("--ref-fingerprint", default=None, help="Enrolled reference fingerprint (.npy)")
    parser.add_argument("--enroll-ref", default=None, help="Save a fingerprint estimated from --video to this path")
    args = parser.parse_args()

    profiler = CameraSensorNoiseProfiler()
    if args.enroll_ref:
        return 0 if profiler.enroll_fingerprint_from_video(args.video, args.enroll_ref) else 1

    result = profiler.analyze(args.video, ref_fingerprint_path=args.ref_fingerprint)
    output_json = result.to_json()
    print(output_json)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
