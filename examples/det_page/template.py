"""
synthtiger-thai: detection pages (v2).

Several text lines (Thai / English, HarfBuzz-shaped) are placed on a real photo background, each with its own colour,
optional border / shadow, an optional cylinder curve (bottle / can labels) and a small perspective / rotation / skew,
then the page gets photo-like degradation (glare, shadow gradient, vignette, motion blur, brightness / contrast,
blur, noise, low resolution, jpeg). Three layouts: scattered lines, a dense paragraph in one font, and 1-3 big words.
Every line is saved with its INK-TIGHT quad (the alpha extent of the glyphs, marks included, mapped through the
line's transforms), its font-metric quad and one quad per grapheme cluster, so a detector's geometry head can be
trained on exact targets instead of thresholded guesses. Text whose colour is too close to the background under it
is re-inked black or white (an invisible line must not be labelled as text).

    synthtiger -o out -w 4 -v examples/det_page/template.py DetPage examples/det_page/config.yaml -c 1000
"""

import json
import os
import re

import cv2
import numpy as np
from PIL import Image

from synthtiger import components, layers, templates, utils
from synthtiger.layers.hb_text import graphemes


DANGLING_MARK = re.compile(r"(?:^|[^ก-ฮัิ-ฺ็-๎])[ัิ-ฺ็-๎]")   # mark without a base (OCR-label noise)


class DetPage(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.page_size = config.get("page_size", [[640, 1280], [640, 1280]])        # [[w_min, w_max], [h_min, h_max]]
        self.lines = config.get("lines", [3, 14])
        self.margin = config.get("margin", 0.05)                                     # page border kept empty (fraction)
        self.gap = config.get("gap", 0.4)                                            # min gap between lines (fraction of line height)
        self.quality = config.get("quality", [60, 95])
        self.modes = config.get("modes", {"scatter": 0.5, "paragraph": 0.3, "big": 0.2})
        self.paragraph = config.get("paragraph", {"size": [14, 44], "lines": [6, 30], "gap": [0.15, 0.8]})
        self.big = config.get("big", {"size": [60, 220], "lines": [1, 3], "graphemes": [1, 8]})
        self.curve = config.get("curve", {"prob": 0.25, "sag": [0.05, 0.35], "bend": [0.0, 1.1]})
        self.min_contrast = config.get("min_contrast", 70)
        self.page_fx = config.get("page_fx", {"glare": 0.3, "shadow": 0.35, "vignette": 0.25})
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
                components.Switch(components.MotionBlur()),
                components.Switch(components.Brightness()),
                components.Switch(components.Contrast()),
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
        bg = bg_layer.output()[..., :3]

        modes = list(self.modes.keys()); p = np.array([self.modes[m] for m in modes], dtype=np.float64)
        mode = modes[np.random.choice(len(modes), p=p / p.sum())]
        placed, records = [], []
        if mode == "paragraph":
            self._paragraph(width, height, bg, placed, records)
        else:
            n_lines = np.random.randint(self.lines[0], self.lines[1] + 1) if mode == "scatter" else np.random.randint(self.big["lines"][0], self.big["lines"][1] + 1)
            for _ in range(n_lines):
                line = self._make_line(width, height, big=(mode == "big"))
                if line is None:
                    continue
                layer, rec = line
                if self._place(layer, rec, width, height, placed):
                    self._contrast(layer, bg)
                    placed.append(layer)
                    records.append(rec)
        if not records:
            raise RuntimeError("no line placed")

        image = layers.Group([*placed, bg_layer]).output(bbox=bg_layer.bbox)
        image = self._page_fx(image)
        page = layers.Layer(image)
        self.postprocess.apply([page])
        image = page.output()
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        return {"image": image, "quality": quality, "width": width, "height": height, "mode": mode, "lines": records}

    # ---- one line: glyphs -> ink rects -> colour / style -> curve -> transform
    def _make_line(self, width, height, big=False, font_meta=None):
        text = self.corpus.data(self.corpus.sample())
        if big:                                                # a word or a few graphemes, large
            g = graphemes(text.strip())
            n = np.random.randint(self.big["graphemes"][0], self.big["graphemes"][1] + 1)
            text = "".join(g[:n]).strip() if " " not in text.strip() else text.strip().split()[0][:24]
        if not text.strip() or DANGLING_MARK.search(text):
            return None
        meta = {"text": text}
        if font_meta:
            meta.update(path=font_meta["path"], size=font_meta["size"], bold=font_meta["bold"])
        elif big:
            meta["size"] = int(np.random.randint(self.big["size"][0], self.big["size"][1] + 1))
        font = self.font.sample(meta)
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
        rects = [ink, font_rect] + [r for _, r in clusters]
        rects = [[r[0] + shift[0], r[1] + shift[1], r[2] + shift[0], r[3] + shift[1]] for r in rects]
        if np.random.rand() < self.curve["prob"]:
            h_ink = ink[3] - ink[1]
            sag = np.random.uniform(*self.curve["sag"]) * h_ink * np.random.choice([-1, 1])
            bend = np.random.uniform(*self.curve["bend"])
            layer, rects = _cylinder(layer, rects, sag, bend)
        self.transform.apply([layer])
        M = _rect_to_quad(layer.image.shape[1], layer.image.shape[0], layer.quad)
        rec = {
            "text": text,
            "quad": _map_rect(rects[0], M),
            "font_quad": _map_rect(rects[1], M),
            "clusters": [{"text": t, "quad": _map_rect(r, M)} for (t, _), r in zip(clusters, rects[2:])],
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
                _move(layer, rec, x, y)
                return True
        return False

    def _paragraph(self, width, height, bg, placed, records):
        """One font, left-aligned lines top to bottom with a constant gap (document / receipt / label look)."""
        m = self.margin
        probe = self.corpus.data(self.corpus.sample())           # the font is picked for the language of the paragraph
        font_meta = self.font.sample({"text": probe, "size": int(np.random.randint(self.paragraph["size"][0], self.paragraph["size"][1] + 1))})
        gap = np.random.uniform(*self.paragraph["gap"])
        n = np.random.randint(self.paragraph["lines"][0], self.paragraph["lines"][1] + 1)
        x0 = np.random.uniform(width * m, width * 0.3)
        y = np.random.uniform(height * m, height * 0.25)
        indent = np.random.rand() < 0.3
        for i in range(n):
            line = self._make_line(width, height, font_meta=font_meta)
            if line is None:
                continue
            layer, rec = line
            bw, bh = layer.size
            x = x0 + (np.random.uniform(0, width * 0.15) if indent and np.random.rand() < 0.3 else 0)
            if x + bw > width * (1 - m) or y + bh > height * (1 - m):
                break
            _move(layer, rec, x, y)
            self._contrast(layer, bg)
            placed.append(layer); records.append(rec)
            y += bh + gap * bh

    def _contrast(self, layer, bg):
        """Re-ink the line black or white when its colour is too close to the background it sits on."""
        x1, y1, x2, y2 = [int(v) for v in (layer.left, layer.top, layer.right, layer.bottom)]
        H, W = bg.shape[:2]
        patch = bg[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if patch.size == 0:
            return
        a = layer.image[..., 3] > 0
        if not a.any():
            return
        ink = layer.image[..., :3][a].mean(0); under = patch.reshape(-1, 3).mean(0)
        if np.abs(ink - under).sum() >= self.min_contrast:
            return
        new = np.array([0, 0, 0] if under.mean() > 128 else [255, 255, 255], dtype=np.float32)
        layer.image[..., :3][a] = new

    def _page_fx(self, image):
        """Photo lighting on the composed page (no pixel moves, quads stay exact): glare, shadow gradient, vignette."""
        H, W = image.shape[:2]
        rgb = image[..., :3]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        if np.random.rand() < self.page_fx.get("shadow", 0):      # one side of the page darker (hand / lamp)
            ang = np.random.uniform(0, 2 * np.pi); t = (xx * np.cos(ang) + yy * np.sin(ang)); t = (t - t.min()) / max(1e-6, t.max() - t.min())
            lo = np.random.uniform(0.35, 0.8); rgb *= (lo + (1 - lo) * t ** np.random.uniform(0.7, 2.0))[..., None]
        if np.random.rand() < self.page_fx.get("vignette", 0):
            r = np.hypot((xx - W / 2) / W, (yy - H / 2) / H); lo = np.random.uniform(0.4, 0.85)
            rgb *= (1 - (1 - lo) * np.clip(r / 0.7, 0, 1) ** 2)[..., None]
        if np.random.rand() < self.page_fx.get("glare", 0):        # specular blob (flash on plastic / glossy paper)
            cx, cy = np.random.uniform(0, W), np.random.uniform(0, H); rad = np.random.uniform(0.15, 0.6) * max(W, H)
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2)) * np.random.uniform(60, 220)
            rgb += g[..., None]
        image[..., :3] = np.clip(rgb, 0, 255)
        return image

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
        row = {"image": image_key.replace("\\", "/"), "w": int(data["width"]), "h": int(data["height"]), "mode": data["mode"], "lines": data["lines"]}
        self.gt_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def end_save(self, root):
        self.gt_file.close()


# ---------------------------------------------------------------------- geometry helpers
def _move(layer, rec, x, y):
    delta = np.array([x, y], dtype=np.float32) - layer.topleft
    layer.quad = layer.quad + delta
    for key in ("quad", "font_quad"):
        rec[key] = [round(float(v + delta[i % 2]), 1) for i, v in enumerate(rec[key])]
    for c in rec["clusters"]:
        c["quad"] = [round(float(v + delta[i % 2]), 1) for i, v in enumerate(c["quad"])]


def _cylinder(layer, rects, sag, bend):
    """Bend the line like a label on a bottle: columns shift vertically along a parabola (sag px at the ends relative
    to the centre, sign = convex / concave) and are compressed towards the ends (bend = cylinder angle, 0 = none).
    Returns the warped layer (same quad origin) and the rects' bounding boxes after the warp."""
    img = layer.image
    h, w = img.shape[:2]
    cx, hw = (w - 1) / 2.0, max(1.0, (w - 1) / 2.0)
    pad = int(np.ceil(abs(sag))) + 1

    def fwd_x(x):                                                  # compress towards the ends (cylinder seen from the front)
        if bend < 1e-3:
            return x
        return cx + hw * np.sin((x - cx) / hw * bend) / np.sin(bend)

    def inv_x(xp):
        if bend < 1e-3:
            return xp
        return cx + hw * np.arcsin(np.clip((xp - cx) / hw * np.sin(bend), -1, 1)) / bend

    def dy(x):
        return sag * ((x - cx) / hw) ** 2 - (sag if sag < 0 else 0)   # >= 0 everywhere: the image only needs one pad

    H2 = h + pad
    xp, yp = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(H2, dtype=np.float32))
    xs = inv_x(xp).astype(np.float32)
    ys = (yp - dy(xs)).astype(np.float32)
    out = cv2.remap(img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    new = []
    for x1, y1, x2, y2 in rects:
        xs_e = np.linspace(x1, x2, 33)
        ys_top, ys_bot = y1 + dy(xs_e), y2 + dy(xs_e)
        new.append([float(fwd_x(x1)), float(ys_top.min()), float(fwd_x(x2)), float(ys_bot.max())])
    tl = layer.topleft
    layer.image = out
    layer.bbox = [tl[0], tl[1], w, H2]
    return layer, new


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
