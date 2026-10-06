"""
Layer 2: Camera Sensor Noise
============================

Owner: Arihant
Part of: Injection Attack & Deepfake Detection pipeline
(Layer 1 = hardware attestation, Layer 2 = this module,
 Layer 3 = temporal & frequency analysis, Layer 5 = gated fusion)

What this layer does
--------------------
Decides whether the frames carry *live sensor noise*: the fresh, physically produced
noise of a camera sensor that is capturing right now. No enrollment is needed.

Default mode - blind live-noise test
  Computed on pixels that are static between consecutive frames, so scene motion does
  not interfere. A stream has live sensor noise only if all four hold:

    1. noise present    the frame-to-frame difference of the high-passed image has a
                        robust std >= 0.6 grey levels. A camera re-samples its noise every
                        frame; generators and video codecs largely do not.
    2. no codec copying at most 0.1% of 8x8 blocks repeat bit-for-bit between frames.
                        Inter-frame codecs (H.264/HEVC/VP9/AV1) copy unchanged macroblocks
                        exactly; with live sensor noise all 64 pixels of a block never repeat.
    3. textured noise   the lag-1 spatial autocorrelation of that noise is >= 0.15:
                        demosaicing and JPEG make sensor noise spatially correlated,
                        digitally added grain is white.
    4. sensor colour    the blue/red noise correlation lies in [0.30, 0.98]: demosaicing
                        partially correlates the colour channels; grey grain is identical
                        in all channels (1.0), colour grain independent (0.0).

  Measured (docs/ARCHITECTURE.md section 4.3): live webcam windows have exactly 0 repeated
  8x8 blocks; codec-processed or generated clips (DF40, talking-head generators,
  text-to-video, YouTube, x264 replays) mostly have 0.2-98%, and those with none have
  noise <= 0.46; grey, colour and blurred synthetic grain each fail check 3 or 4.

  Outcomes: PRESENT, ABSENT (fails a check), INCONCLUSIVE (too little static area).

Optional mode - reference fingerprint (research API, not used by the app)
  PCE (Goljan et al. 2009) of the clip's PRNU residual against a fingerprint estimated
  earlier from the same camera.

Usage
-----
    python camera_sensor_noise_profiling.py --video clip.mp4                       # live-noise test
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
    # --- live-noise test ---
    "noise_window": 45,               # frames analysed (consecutive pairs)
    "noise_blur_sigma": 2.0,          # low-pass removed before measuring noise
    "static_thresh": 1.5,             # max grey-level change (sigma=1 blur) of a static pixel ...
    "static_grad_frac": 0.2,          # ... and max change relative to the local gradient (~0.2 px shift)
    "min_static_pixels": 3000,        # per frame pair
    "min_pairs": 8,                   # frame pairs with enough static pixels
    "min_noise_sigma": 0.6,           # check 1
    "max_block_repeat": 0.001,        # check 2: share of 8x8 blocks repeated bit-for-bit
    "min_spatial_corr": 0.15,         # check 3
    "colour_corr_range": (0.30, 0.97),  # check 4
    # --- reference mode ---
    "pce_peak_radius": 2,
    "pce_search_radius": 2,           # the peak may sit a few pixels off (0,0): sensor-mode scaler/crop phase
    "pce_match_thresh": 60.0,         # standard PCE decision threshold (Goljan 2009)
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


def live_noise_features(frames: Sequence[np.ndarray], cfg: Optional[dict] = None) -> Optional[dict]:
    """
    Temporal-noise statistics over static pixels of consecutive frames (BGR or grey).
    Returns None when fewer than `min_pairs` frame pairs have enough static pixels.
    """
    cfg = {**CONFIG, **(cfg or {})}
    sig = cfg["noise_blur_sigma"]
    colour = frames[0].ndim == 3
    weights = np.array([0.114, 0.587, 0.299], np.float32)

    def motion_view(y):
        # Lightly blurred luma and its gradient magnitude. A shift of d pixels changes it by
        # about d * gradient, so "static" must be judged relative to the local gradient, or
        # moving fine texture would be mistaken for noise.
        m = cv2.GaussianBlur(y, (0, 0), 1.0)
        g = np.sqrt(cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3) ** 2) / 8.0
        return m, g

    d_y, d_b, d_r, n0, n1, repeats = [], [], [], [], [], []
    prev = None
    for fr in frames:
        f = fr.astype(np.float32)
        y = f @ weights if colour else f
        lp = cv2.GaussianBlur(y, (0, 0), sig)
        hp = f - (cv2.GaussianBlur(f, (0, 0), sig) if colour else lp)
        m, grad = motion_view(y)
        cur = (fr, m, hp)
        if prev is not None:
            pfr, pm, php = prev
            change = np.abs(m - pm)
            static = (change < cfg["static_thresh"]) & (change < cfg["static_grad_frac"] * grad + 0.6)
            static &= (lp > 8) & (lp < 247)
            static[:, -1] = False
            if static.sum() >= cfg["min_static_pixels"]:
                d = hp - php
                dy = d @ weights if colour else d
                d_y.append(dy[static])
                pair = static & np.roll(static, -1, axis=1)
                n0.append(dy[pair])
                n1.append(np.roll(dy, -1, axis=1)[pair])
                if colour:
                    d_b.append(d[..., 0][static])
                    d_r.append(d[..., 2][static])
            raw = np.abs(fr.astype(np.int16) - pfr.astype(np.int16))
            same = (raw.max(axis=2) if colour else raw) == 0
            hh, ww = (same.shape[0] // 8) * 8, (same.shape[1] // 8) * 8
            blocks = same[:hh, :ww].reshape(hh // 8, 8, ww // 8, 8).all(axis=(1, 3))
            level = lp[:hh, :ww].reshape(hh // 8, 8, ww // 8, 8).mean(axis=(1, 3))
            usable = (level > 12) & (level < 243)  # clipped black/white blocks repeat legitimately
            if usable.sum() > 50:
                repeats.append(float(blocks[usable].mean()))
        prev = cur
    if len(d_y) < cfg["min_pairs"]:
        return None

    def robust_std(v):
        return float(1.4826 * np.median(np.abs(v)))

    def corr(a, b):
        ca, cb = 5 * robust_std(a) + 1e-6, 5 * robust_std(b) + 1e-6
        a, b = np.clip(a, -ca, ca), np.clip(b, -cb, cb)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    dy = np.concatenate(d_y)
    out = {
        "noise_sigma": float(robust_std(dy) / np.sqrt(2)),
        "block_repeat": float(np.median(repeats)) if repeats else 0.0,
        "spatial_corr": corr(np.concatenate(n0), np.concatenate(n1)),
        "colour_corr": corr(np.concatenate(d_b), np.concatenate(d_r)) if colour else None,
        "pairs": len(d_y),
    }
    return out


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
    """Blind live-sensor-noise test (default) plus optional reference-fingerprint PCE matching."""

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
        result = LayerResult(mode="reference" if ref_fingerprint is not None else "live_noise")
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

        if ref_fingerprint is None:
            return self._live_noise(result, frames[-self.cfg["noise_window"]:])

        grays = [to_gray_float(f) for f in frames]
        residuals = [
            extract_noise_residual(
                g, self.cfg["denoise_sigma"], self.cfg["denoise_windows"],
                self.cfg["saturation_low"], self.cfg["saturation_high"],
            )
            for g in grays
        ]
        return self._reference_match(result, grays, residuals, ref_fingerprint)

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

    # ------------------------- live-noise test ------------------------- #

    def _live_noise(self, result: LayerResult, frames) -> LayerResult:
        cfg = self.cfg
        f = live_noise_features(frames, cfg)
        if f is None:
            result.verdict = VERDICT_INCONCLUSIVE
            result.explanation = ("Too little of the scene is still between frames to measure sensor noise "
                                  "(subject moving across the whole frame, or a very dark/bright scene).")
            result.components = {"static_noise_detected": None}
            return result
        lo, hi = cfg["colour_corr_range"]
        checks = {
            "noise present": bool(f["noise_sigma"] >= cfg["min_noise_sigma"]),
            "no codec copying": bool(f["block_repeat"] <= cfg["max_block_repeat"]),
            "sensor-like texture": bool(f["spatial_corr"] >= cfg["min_spatial_corr"]),
        }
        if f["colour_corr"] is not None:
            checks["sensor-like colour"] = bool(lo <= f["colour_corr"] <= hi)
        failed = [k for k, ok in checks.items() if not ok]
        present = not failed
        result.verdict = VERDICT_PRESENT if present else VERDICT_ABSENT
        result.flagged = not present
        result.score = round(len(failed) / len(checks), 3)
        result.confidence = round(min(1.0, f["pairs"] / 20), 2)
        result.components = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()},
                             "checks": checks, "failed_checks": failed, "static_noise_detected": present}
        why = {
            "noise present": f"almost no frame-to-frame noise ({f['noise_sigma']:.2f} < {cfg['min_noise_sigma']})",
            "no codec copying": f"{f['block_repeat']:.1%} of image blocks repeat exactly, a video-codec fingerprint",
            "sensor-like texture": f"noise is white like added grain (spatial corr {f['spatial_corr']:.2f})",
            "sensor-like colour": f"colour channels do not behave like a sensor (corr {f['colour_corr'] or 0:.2f})",
        }
        if present:
            result.explanation = (f"Live sensor noise present: noise {f['noise_sigma']:.2f}, "
                                  f"{f['block_repeat']:.1%} repeated blocks, texture {f['spatial_corr']:.2f}"
                                  + (f", colour {f['colour_corr']:.2f}." if f["colour_corr"] is not None else "."))
        else:
            result.explanation = "No live sensor noise: " + "; ".join(why[k] for k in failed) + "."
        return result

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
