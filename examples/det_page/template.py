"""
synthtiger-thai: detection pages.

Several text lines (Thai / English, HarfBuzz-shaped) are placed on a real photo background, each with its own colour,
optional border / shadow and a small perspective / rotation / skew, then the page gets photo-like degradation.
Every line is saved with its INK-TIGHT quad (the alpha extent of the glyphs, marks included, mapped through the line's
transform), its font-metric quad and one quad per grapheme cluster, so a detector's geometry head can be trained on
exact targets instead of thresholded guesses.

    synthtiger -o out -w 4 -v examples/det_page/template.py DetPage examples/det_page/config.yaml -c 1000
"""

import json
import os

import cv2
import numpy as np
from PIL import Image

from synthtiger import components, layers, templates, utils


class DetPage(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.page_size = config.get("page_size", [[640, 1280], [640, 1280]])        # [[w_min, w_max], [h_min, h_max]]
        self.lines = config.get("lines", [3, 14])
        self.margin = config.get("margin", 0.05)                                     # page border kept empty (fraction)
        self.gap = config.get("gap", 0.4)                                            # min gap between lines (fraction of line height)
        self.quality = config.get("quality", [60, 95])
        self.corpus = components.Selector(
            [components.BaseCorpus(), components.BaseCorpus()], **config.get("corpus", {})
        )
        self.font = components.BaseFont(**config.get("font", {}))
        self.background = components.BaseTexture(**config.get("background", {}))
        self.bg_color = components.Switch(components.Gray(), **config.get("bg_color", {}))   # plain / gradient-free pages
        self.color = components.RGB(**config.get("color", {}))
        self.style = components.Switch(
            components.Selector([components.TextBorder(), components.TextShadow()]), **config.get("style", {})
        )
        self.transform = components.Switch(
            components.Selector([components.Perspective(), components.Rotate(), components.Skew()]),
            **config.get("transform", {}),
        )
        self.postprocess = components.Iterator(
            [
                components.Switch(components.GaussianBlur()),
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.Resample()),
                components.Switch(components.JpegCompression()),
            ],
            **config.get("postprocess", {}),
        )

    # ------------------------------------------------------------------ generation
    def generate(self):
        width = np.random.randint(self.page_size[0][0], self.page_size[0][1] + 1)
        height = np.random.randint(self.page_size[1][0], self.page_size[1][1] + 1)
        bg_layer = layers.RectLayer((width, height), (255, 255, 255, 255))
        bg_meta = self.bg_color.sample()
        if bg_meta["state"]:
            self.bg_color.apply([bg_layer], bg_meta)
        else:                                                  # a random window (40-100 %) of a background photo
            meta = self.background.sample({"crop": False, "alpha": 1.0})
            W0, H0 = meta["w"], meta["h"]
            cw, ch = np.random.randint(int(W0 * 0.4), W0 + 1), np.random.randint(int(H0 * 0.4), H0 + 1)
            meta.update(crop=True, w=cw, h=ch, x=np.random.randint(0, W0 - cw + 1), y=np.random.randint(0, H0 - ch + 1))
            self.background.apply([bg_layer], meta)

        n_lines = np.random.randint(self.lines[0], self.lines[1] + 1)
        placed, records = [], []
        for _ in range(n_lines):
            line = self._make_line(width, height)
            if line is None:
                continue
            layer, rec = line
            if self._place(layer, rec, width, height, placed):
                placed.append(layer)
                records.append(rec)
        if not records:
            raise RuntimeError("no line placed")

        image = layers.Group([*placed, bg_layer]).output(bbox=bg_layer.bbox)
        page = layers.Layer(image)
        self.postprocess.apply([page])
        image = page.output()
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        return {"image": image, "quality": quality, "width": width, "height": height, "lines": records}

    def _make_line(self, width, height):
        text = self.corpus.data(self.corpus.sample())
        if not text.strip():
            return None
        font = self.font.sample({"text": text})
        # a line must fit the page: cap the font size by the page width
        max_size = max(8, int(width * 0.9 / max(1, len(text)) * 1.8))
        font["size"] = int(min(font["size"], max_size))
        layer = layers.TextLayer(text, **font)
        if layer.notdef:                                       # the font lacks a glyph of this text (would draw boxes)
            return None
        alpha = layer.image[..., 3]
        if alpha.max() <= 0:
            return None
        ink = _tight_rect(alpha)
        if ink is None:
            return None
        clusters = []
        gidx = [c[0] for c in layer.clusters] + [len(text)]
        for (cl, x0, x1), nxt in zip(layer.clusters, gidx[1:]):
            r = _tight_rect(alpha[:, int(x0):int(np.ceil(x1))])
            if r is None:
                continue
            clusters.append((text[cl:nxt], [r[0] + int(x0), r[1], r[2] + int(x0), r[3]]))
        font_rect = [0, 0, alpha.shape[1], alpha.shape[0]]

        glyph = layer.copy()                                   # geometry is measured on the bare glyphs
        self.color.apply([layer])
        self.style.apply([layer])                              # border / shadow may pad the image: shift the rects
        shift = glyph.topleft - layer.topleft
        self.transform.apply([layer])
        M = _rect_to_quad(layer.image.shape[1], layer.image.shape[0], layer.quad)
        rec = {
            "text": text,
            "quad": _map_rect([ink[0] + shift[0], ink[1] + shift[1], ink[2] + shift[0], ink[3] + shift[1]], M),
            "font_quad": _map_rect([font_rect[0] + shift[0], font_rect[1] + shift[1], font_rect[2] + shift[0], font_rect[3] + shift[1]], M),
            "clusters": [{"text": t, "quad": _map_rect([r[0] + shift[0], r[1] + shift[1], r[2] + shift[0], r[3] + shift[1]], M)} for t, r in clusters],
            "font": os.path.basename(font["path"]),
            "size": int(font["size"]),
        }
        return layer, rec

    def _place(self, layer, rec, width, height, placed):
        bw, bh = layer.size
        m = self.margin
        if bw > width * (1 - 2 * m) or bh > height * (1 - 2 * m):
            return False
        gap = self.gap * bh
        for _ in range(40):
            x = np.random.uniform(width * m, width * (1 - m) - bw)
            y = np.random.uniform(height * m, height * (1 - m) - bh)
            box = np.array([x - gap, y - gap, x + bw + gap, y + bh + gap])
            if all(not _overlap(box, [p.left, p.top, p.right, p.bottom]) for p in placed):
                delta = np.array([x, y], dtype=np.float32) - layer.topleft
                layer.quad = layer.quad + delta
                for key in ("quad", "font_quad"):
                    rec[key] = [round(float(v + delta[i % 2]), 1) for i, v in enumerate(rec[key])]
                for c in rec["clusters"]:
                    c["quad"] = [round(float(v + delta[i % 2]), 1) for i, v in enumerate(c["quad"])]
                return True
        return False

    # ------------------------------------------------------------------ saving
    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        self.gt_file = open(os.path.join(root, "pages.jsonl"), "w", encoding="utf-8")

    def save(self, root, data, idx):
        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        image_path = os.path.join(root, image_key)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        Image.fromarray(data["image"][..., :3].astype(np.uint8)).save(image_path, quality=int(data["quality"]))
        row = {"image": image_key.replace("\\", "/"), "w": int(data["width"]), "h": int(data["height"]), "lines": data["lines"]}
        self.gt_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def end_save(self, root):
        self.gt_file.close()


# ---------------------------------------------------------------------- geometry helpers
def _tight_rect(alpha, thr=0):
    """(x1, y1, x2, y2) of the non-zero alpha, x2 / y2 exclusive; None when empty."""
    ys = np.nonzero(alpha.max(1) > thr)[0]
    xs = np.nonzero(alpha.max(0) > thr)[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    return [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]


def _rect_to_quad(w, h, quad):
    """Perspective matrix taking the layer's image rectangle (w x h) onto its placed quad (what paste_image does)."""
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, np.asarray(quad, dtype=np.float32))


def _map_rect(rect, M):
    x1, y1, x2, y2 = rect
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, M).reshape(4, 2)
    return [round(float(v), 1) for v in out.reshape(-1)]


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
