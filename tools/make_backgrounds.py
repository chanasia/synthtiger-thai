"""Real-scene backgrounds for the det_page generator: photos from the detector's own TRAINING sets with every text
region painted out (label boxes + ignore boxes + a low-threshold pass of the current detector, dilated, cv2.inpaint),
so the synthetic pages get real shelves / bottles / tables / hands / glare instead of texture tiles.
Eval images are never used (the jsonl is a train split). Scans (gazette, thbud, sroie), synthetic and id-card sets are
skipped (nothing to learn from a blank scan; id cards carry faces).

    python tools/make_backgrounds.py --jsonl /mnt/d/win-ocr/data/det_v1/train_v18.jsonl --images /mnt/d/win-ocr/data/det_v1/images \
        --det /mnt/d/win-ocr/data/det_v18.onnx --dst /mnt/d/win-ocr/data/synth_bg_real -w 12
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

SKIP = ("synth", "gazette", "thbud_doc", "sroie", "idcard_yolo", "pseudo_idcard_rf", "pseudo_idcard_web", "weak_idcard")
LONG = 1024            # output long side (the generator crops a window of it anyway)
_det = None


def _detector(path):
    global _det
    if _det is None:
        sys.path.insert(0, os.path.join(os.path.dirname(path), "..", "realdata"))
        from det_runtime import Detector
        _det = Detector(path, threads=1, long=640, adapt=0)   # coarse pass is enough for a mask
    return _det


def fill(img, mask):
    """Normalised convolution: masked pixels take the blurred colour of the unmasked pixels around them (text ghosts
    never leak in). Growing Gaussian scales so small holes keep local colour and big ones still close; the filled
    area is smoothed once more so no block edges remain. ~50x faster than cv2.inpaint."""
    keep = (mask == 0).astype(np.float32)
    hole0 = keep == 0
    out = img.astype(np.float32)
    for k in (21, 61, 181):
        num = cv2.GaussianBlur(out * keep[..., None], (0, 0), k / 3); den = cv2.GaussianBlur(keep, (0, 0), k / 3)[..., None]
        est = num / np.maximum(den, 1e-3)
        hole = (keep == 0) & (den[..., 0] > 0.02)
        out[hole] = est[hole]; keep[hole] = 1.0
    out[keep == 0] = cv2.GaussianBlur(out, (0, 0), 80)[keep == 0]
    soft = cv2.GaussianBlur(out, (0, 0), 6)
    out[hole0] = soft[hole0]
    return np.clip(out, 0, 255).astype(np.uint8)


def work(args):
    rec, images, det_path, dst = args
    try:
        im = Image.open(os.path.join(images, os.path.basename(rec["image"]))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    w, h = im.size
    s = LONG / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    img = np.asarray(im)
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for q in list(rec.get("boxes", [])) + list(rec.get("ignore", [])):
        pts = (np.asarray(q, np.float32).reshape(4, 2) * s).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        hh = max(3, int(0.35 * (abs(pts[3][1] - pts[0][1]) + abs(pts[2][1] - pts[1][1])) / 2))
        cv2.polylines(mask, [pts], True, 255, thickness=hh * 2)         # dilate by ~0.35 h around every box
    if det_path:                                                         # belt and braces: whatever the detector sees
        d = _detector(det_path)
        m, sc, _ = d.mask(im)
        m = (m.astype(np.uint8) * 255)[: int(H * sc) + 1, : int(W * sc) + 1]
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, np.ones((15, 15), np.uint8))
        mask = np.maximum(mask, m)
    frac = float(mask.mean()) / 255
    if frac > 0.6:                                                       # nothing but text: no scene left to use
        return None
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    out = fill(img[..., ::-1], mask)
    name = os.path.splitext(os.path.basename(rec["image"]))[0] + ".jpg"
    cv2.imwrite(os.path.join(dst, name), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return name, round(frac, 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--det", default=None, help="detector onnx (optional, masks text the labels missed)")
    ap.add_argument("--dst", required=True); ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    recs = [json.loads(l) for l in open(a.jsonl, encoding="utf8")]
    recs = [r for r in recs if r.get("source") not in SKIP]
    if a.limit:
        recs = recs[: a.limit]
    print(len(recs), "photos", flush=True)
    done = 0
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, [(r, a.images, a.det, a.dst) for r in recs], chunksize=8)):
            done += r is not None
            if i % 1000 == 0:
                print(i, "done", done, flush=True)
    print("backgrounds", done, "->", a.dst)
