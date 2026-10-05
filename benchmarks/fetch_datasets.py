"""
Downloads the public evaluation subsets used in docs/ARCHITECTURE.md (Section 7).

  VISION (Shullani et al., EURASIP J. Inf. Security 2017) - native smartphone videos
      first 90 frames of indoor 'still' / 'move' videos and their WhatsApp versions,
      stored losslessly (FFV1, grayscale). ffmpeg reads only the bytes it needs.
  DF40 (Yan et al., NeurIPS 2024) - face-swap / reenactment / talking-head fakes
  Text-to-video samples (Veo 3, Kling, HunyuanVideo) from Hugging Face '34data/gen-videos-*'
      both are zip archives; individual members are extracted with HTTP range requests,
      so only a few MB per clip are transferred.

Check each dataset's licence / terms of use before redistributing anything.

    python benchmarks/fetch_datasets.py              # defaults used in the paper draft
    python benchmarks/fetch_datasets.py --vision-step 1 --vision-devices 35 --df40-per-method 10
"""

import argparse
import io
import os
import random
import subprocess
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VISION_ROOT = "https://lesc.dinfo.unifi.it/VISION/dataset/"
HF = "https://huggingface.co/datasets"
DF40_METHODS = ["MRAA", "deepfacelab", "faceswap", "inswap", "simswap", "heygen", "sadtalker", "wav2lip",
                "fomm", "facedancer", "uniface", "mobileswap"]
T2V = {"veo3": "34data/gen-videos-veo3", "hunyuan": "34data/gen-videos-hunyuanvideo", "kling": "34data/gen-videos-kling"}


class HTTPRange(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_zip_members(url, out_dir, k, prefix, seed=0):
    z = zipfile.ZipFile(io.BufferedReader(HTTPRange(url), buffer_size=1 << 20))
    vids = [i for i in z.infolist() if i.filename.lower().endswith((".mp4", ".mov", ".avi", ".webm"))
            and i.file_size > 20000 and not os.path.basename(i.filename).startswith("._")]
    random.Random(seed).shuffle(vids)
    got = 0
    for info in vids[:k]:
        out = os.path.join(out_dir, f"{prefix}__{os.path.basename(info.filename)}")
        if not os.path.exists(out):
            with z.open(info) as src, open(out, "wb") as dst:
                dst.write(src.read())
        got += 1
    return f"{prefix}: {got} clips"


def vision_devices():
    with urllib.request.urlopen(VISION_ROOT, timeout=60) as r:
        html = r.read().decode("utf-8", "ignore")
    import re
    return sorted(set(re.findall(r'href="(D\d+_[^"/]+)/?"', html)))


def fetch_vision_clip(device, category, motion, out_dir, frames=90):
    n = device.split("_")[0]
    out = os.path.join(out_dir, f"{n}_{category}_{motion}.mkv")
    if os.path.exists(out):
        return f"{out} (cached)"
    for ext in ("mp4", "mov"):
        url = f"{VISION_ROOT}{device}/videos/{category}/{n}_V_{category}_{motion}_0001.{ext}"
        res = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", url, "-frames:v", str(frames), "-an",
                              "-c:v", "ffv1", "-pix_fmt", "gray", out], capture_output=True)
        if res.returncode == 0:
            return out
    return f"missing {device} {category} {motion}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-step", type=int, default=3, help="take every k-th VISION device (3 -> D01, D04, ..., D34)")
    ap.add_argument("--vision-devices", type=int, default=12, help="maximum number of VISION devices")
    ap.add_argument("--df40-per-method", type=int, default=4)
    ap.add_argument("--t2v-per-model", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    vdir, fdir = os.path.join(DATA, "vision"), os.path.join(DATA, "fakes")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    devs = vision_devices()
    devs = devs[:: args.vision_step][: args.vision_devices]
    jobs = [(d, c, m) for d in devs for c, m in (("indoor", "still"), ("indoor", "move"), ("indoorWA", "still"))]
    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(lambda j: fetch_vision_clip(*j, vdir), jobs):
            print("VISION", msg, flush=True)

    zips = [(f"{HF}/ohjoonhee/DF40-Fake-Videos/resolve/main/test/{m}.zip", args.df40_per_method, f"DF40_{m}") for m in DF40_METHODS]
    zips += [(f"{HF}/{repo}/resolve/main/data_001.zip", args.t2v_per_model, f"T2V_{name}") for name, repo in T2V.items()]
    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(lambda z: fetch_zip_members(z[0], fdir, z[1], z[2]), zips):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
