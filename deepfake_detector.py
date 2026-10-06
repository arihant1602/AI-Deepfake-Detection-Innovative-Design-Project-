"""
Layer 3: Face deepfake detection with GenD
==========================================

Replaces the hand-tuned temporal heuristic (temporal_consistency_analysis.py, which
flagged 2 of 64 fakes on the benchmark) with GenD, the strongest openly released
generalizable face-deepfake detector at the time of writing:

  A. Yermakov, J. Cech, J. Matas, M. Fritz. "Deepfake Detection that Generalizes Across
  Benchmarks." WACV 2026. https://github.com/yermandy/GenD (MIT licence)
  Mean cross-dataset AUROC 91.2-91.6% over 14 benchmarks (2019-2025), ahead of Effort
  (ICML 2025, 88.5%) and ForensicsAdapter (CVPR 2025, 88.4%).

GenD fine-tunes only the LayerNorm parameters of a frozen foundation vision encoder
(CLIP ViT-L/14 or Meta Perception Encoder L) and classifies L2-normalised features with a
linear head. It scores single aligned face crops; a clip is scored by averaging frames.

Face pipeline (matches GenD's preprocessing): detect the face and 5 landmarks, align to
GenD's landmark template with a 1.3x margin into a 256x256 crop, then the backbone's own
resize/normalisation to 224x224. Detection uses OpenCV's YuNet (MIT, ships with OpenCV)
instead of InsightFace RetinaFace, whose weights are non-commercial.

    python deepfake_detector.py --video clip.mp4 [--backbone clip|pe]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(ROOT, "models", "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
BACKBONES = {"clip": "yermandy/GenD_CLIP_L_14", "pe": "yermandy/GenD_PE_L"}

CONFIG = {
    "backbone": "pe",
    "frames_per_clip": 16,       # frames scored per clip / window (spread evenly)
    "min_faces": 6,              # fewer usable faces than this -> abstain
    "min_face_px": 90,           # faces narrower than this are too small to judge (scored only above it)
    "min_face_score": 0.85,      # YuNet confidence required for a face to be scored
    "fake_threshold": 0.5,       # mean P(fake) at/above which the clip is flagged (calibrated in benchmarks)
    "face_score_threshold": 0.7, # YuNet detection confidence
}


# --------------------------------------------------------------------------- #
# GenD model (vendored from github.com/yermandy/GenD, src/hf/modeling_gend.py, MIT)
# --------------------------------------------------------------------------- #

def _build_gend_classes():
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import PretrainedConfig, PreTrainedModel

    class LinearProbe(nn.Module):
        def __init__(self, input_dim, num_classes, normalize_inputs=False):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)
            self.normalize_inputs = normalize_inputs

        def forward(self, x, **kwargs):
            if self.normalize_inputs:
                x = F.normalize(x, p=2, dim=1)
            return self.linear(x)

    class CLIPEncoder(nn.Module):
        def __init__(self, model_name="openai/clip-vit-large-patch14"):
            super().__init__()
            # Built from the config only: the GenD checkpoint carries all vision-tower weights.
            from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel
            self._preprocess = CLIPImageProcessor.from_pretrained(model_name)
            # transformers 5: CLIPVisionModel has the vision transformer's submodules at top level,
            # so parameter names match the checkpoint (feature_extractor.vision_model.embeddings...).
            self.vision_model = CLIPVisionModel(CLIPConfig.from_pretrained(model_name).vision_config)
            self.features_dim = self.vision_model.config.hidden_size

        def preprocess(self, image):
            return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

        def forward(self, x):
            return self.vision_model(x).pooler_output

        def get_features_dim(self):
            return self.features_dim

    class PerceptionEncoder(nn.Module):
        def __init__(self, model_name="vit_pe_core_large_patch14_336"):
            super().__init__()
            import timm
            # Built without pretrained weights: the GenD checkpoint carries the whole backbone.
            self.backbone = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
            data_config = timm.data.resolve_model_data_config(self.backbone)
            data_config["input_size"] = (3, 224, 224)
            self._preprocess = timm.data.create_transform(**data_config, is_training=False)
            self.backbone.head = nn.Identity()
            self.features_dim = self.backbone.num_features

        def preprocess(self, image):
            return self._preprocess(image)

        def forward(self, x):
            return self.backbone(x)

        def get_features_dim(self):
            return self.features_dim

    class GenDConfig(PretrainedConfig):
        model_type = "GenD"

        def __init__(self, backbone="openai/clip-vit-large-patch14", head="linear", **kwargs):
            super().__init__(**kwargs)
            self.backbone = backbone
            self.head = head

    class GenD(PreTrainedModel):
        config_class = GenDConfig

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            b = config.backbone.lower()
            if "clip" in b:
                self.feature_extractor = CLIPEncoder(config.backbone)
            elif "vit_pe" in b:
                self.feature_extractor = PerceptionEncoder(config.backbone)
            else:
                raise ValueError(f"Unsupported backbone: {config.backbone}")
            dim = self.feature_extractor.get_features_dim()
            self.model = LinearProbe(dim, 2, normalize_inputs=(config.head == "LinearNorm"))

        def forward(self, inputs):
            return self.model(self.feature_extractor(inputs))

    return GenD, GenDConfig


def load_gend(repo: str):
    """
    Builds GenD from its config and loads the released checkpoint explicitly, failing loudly
    if any parameter is missing or unexpected (a silent partial load would be meaningless).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    GenD, GenDConfig = _build_gend_classes()
    model = GenD(GenDConfig.from_pretrained(repo))
    state = load_file(hf_hub_download(repo, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    # CLIP's text-alignment projection is stored but unused: GenD classifies pooled vision features.
    unexpected = [k for k in unexpected if k != "feature_extractor.visual_projection.weight"]
    if missing or unexpected:
        raise RuntimeError(f"GenD checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model


# --------------------------------------------------------------------------- #
# Face detection and GenD alignment
# --------------------------------------------------------------------------- #

GEND_TEMPLATE = np.array([[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], np.float32)


def align_face(img: np.ndarray, landmarks: np.ndarray, size: int = 256, scale: float = 1.3) -> np.ndarray:
    """GenD's alignment (detector.py: align_face): similarity transform onto a 5-point template."""
    dst = GEND_TEMPLATE.copy() * size
    margin = size * (scale - 1) / 2.0
    dst += margin
    dst *= size / (size + 2 * margin)
    m = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR)


class FaceDetector:
    """OpenCV YuNet: bounding box + 5 landmarks (eyes, nose tip, mouth corners)."""

    def __init__(self, score_threshold: float = CONFIG["face_score_threshold"]):
        if not os.path.exists(YUNET_PATH):
            import urllib.request
            os.makedirs(os.path.dirname(YUNET_PATH), exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        self.det = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, frame_bgr: np.ndarray):
        """Largest face as (box xywh, landmarks 5x2 in image order, confidence) or None."""
        h, w = frame_bgr.shape[:2]
        # Detect on a copy at most 320 px wide (4x fewer pixels at 640x480); coordinates are scaled
        # back so alignment still uses the full-resolution frame.
        k = min(1.0, 320.0 / w)
        small = cv2.resize(frame_bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame_bgr
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3]).copy()
        f[:14] /= k
        lm = f[4:14].reshape(5, 2)
        eyes = lm[:2][np.argsort(lm[:2, 0])]        # image-left eye first, as in the template
        mouth = lm[3:5][np.argsort(lm[3:5, 0])]
        return f[:4].astype(int), np.vstack([eyes, lm[2:3], mouth]), float(f[14])


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

@dataclass
class DeepfakeResult:
    layer: str = "gend_face_deepfake"
    backbone: str = ""
    frames_analyzed: int = 0
    frames_with_face: int = 0
    fake_probability: Optional[float] = None
    frame_probabilities: List[float] = field(default_factory=list)
    flagged: bool = False
    abstained: bool = True
    latency_ms: float = 0.0
    explanation: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


class DeepfakeDetector:
    def __init__(self, backbone: str = CONFIG["backbone"], device: Optional[str] = None, config: Optional[dict] = None):
        import torch
        self.cfg = {**CONFIG, **(config or {}), "backbone": backbone}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_gend(BACKBONES[backbone]).eval().to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.faces = FaceDetector()

    def crops(self, frames: Sequence[np.ndarray]) -> tuple:
        """
        Aligned RGB crops of faces that are large and confident enough to judge. Small or
        partially visible faces were the source of most false flags on real webcam footage.
        """
        out, boxes = [], []
        for f in frames:
            det = self.faces.detect(f)
            if det is None:
                continue
            box, lm, conf = det
            if box[2] < self.cfg["min_face_px"] or conf < self.cfg["min_face_score"]:
                continue
            out.append(cv2.cvtColor(align_face(f, lm), cv2.COLOR_BGR2RGB))
            boxes.append(box)
        return out, boxes

    def score_crops(self, crops: Sequence[np.ndarray]) -> List[float]:
        import torch
        from PIL import Image
        if not crops:
            return []
        x = torch.stack([self.model.feature_extractor.preprocess(Image.fromarray(c)) for c in crops]).to(self.device)
        if self.device == "cuda":
            x = x.half()
        with torch.inference_mode():
            p = self.model(x).float().softmax(-1)[:, 1]
        return [float(v) for v in p.cpu()]

    def analyze_frames(self, frames: Sequence[np.ndarray], min_faces: Optional[int] = None) -> DeepfakeResult:
        t0 = time.perf_counter()
        min_faces = self.cfg["min_faces"] if min_faces is None else min_faces
        frames = list(frames)
        n = self.cfg["frames_per_clip"]
        idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int) if frames else []
        sample = [frames[i] for i in idx]
        crops, _ = self.crops(sample)
        probs = self.score_crops(crops)
        r = DeepfakeResult(backbone=self.cfg["backbone"], frames_analyzed=len(sample), frames_with_face=len(crops),
                           frame_probabilities=[round(p, 4) for p in probs])
        if len(crops) < min_faces:
            r.explanation = (f"A clear face (at least {self.cfg['min_face_px']} px wide) in {len(crops)} of {len(sample)} "
                             f"sampled frames; {min_faces} needed. Move closer and face the camera.")
        else:
            r.abstained = False
            r.fake_probability = round(float(np.mean(probs)), 4)
            r.flagged = r.fake_probability >= self.cfg["fake_threshold"]
            r.explanation = (f"{'Likely deepfake' if r.flagged else 'Face looks authentic'}: mean P(fake) "
                             f"{r.fake_probability:.2f} over {len(crops)} faces (flagged at {self.cfg['fake_threshold']:.2f}).")
        r.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return r

    def analyze(self, video_path: str) -> DeepfakeResult:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        return self.analyze_frames(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--backbone", choices=list(BACKBONES), default=CONFIG["backbone"])
    args = ap.parse_args()
    print(DeepfakeDetector(args.backbone).analyze(args.video).to_json())


if __name__ == "__main__":
    sys.exit(main())
