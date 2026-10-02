"""
HarfBuzz text rasterizer for TextLayer (synthtiger-thai).

Pillow's basic layout engine drops stacked Thai marks (a tone mark above an upper vowel, e.g. "ปู่", "ที่นี่") and Raqm
is not available in every Pillow build. This module shapes the string with HarfBuzz (uharfbuzz, GSUB/GPOS of the
font) and rasterizes the glyphs itself, so every mark lands where the font says. It also returns the per-grapheme-
cluster horizontal extents, so a detection template can derive character boxes without a word segmenter (Thai has
no spaces).

render(text, font_path, size, color, bold) -> (image float32 RGBA (H, W, 4), bbox [0, -ascent, width, height], clusters, notdef)
    notdef: number of glyphs the font does not have (gid 0, drawn as boxes) - a caller should drop such lines.
    clusters: list of (start_char_index, x0, x1) in image pixels, one per HarfBuzz cluster, left to right.
"""

from functools import lru_cache

import numpy as np
import regex
import uharfbuzz as hb


@lru_cache(maxsize=256)
def _font(path, size):
    face = hb.Face(hb.Blob.from_file_path(path), 0)
    font = hb.Font(face)
    font.scale = (face.upem, face.upem)
    return font, face.upem


def _shape(text, font):
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    return buf


def render(text, path, size, color=(0, 0, 0, 255), bold=False):
    font, upem = _font(path, size)
    s = size / upem
    buf = _shape(text, font)
    ext = font.get_font_extents("ltr")
    ascent, descent = int(round(ext.ascender * s)), int(round(-ext.descender * s))

    # pen walk in font units; glyphs may extend left of the origin / right of the advance (marks, italics), so the
    # canvas is the advance width plus a margin of one em on each side which is trimmed back below
    pen = 0
    placed = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        placed.append((info.codepoint, info.cluster, pen + pos.x_offset, pos.y_offset))
        pen += pos.x_advance
    adv_w = max(1, int(round(pen * s)))
    margin = size
    W, H = adv_w + 2 * margin, ascent + descent + 2 * margin
    base_y = margin + ascent

    rd = hb.RasterDraw()
    for gid, _, gx, gy in placed:
        rd.transform = (s, 0.0, 0.0, -s, margin + gx * s, base_y - gy * s)
        rd.draw_glyph(font, gid)
    rd.extents = hb.RasterExtents(0, 0, W, H, W)
    img = rd.render()
    alpha = np.frombuffer(img.buffer, dtype=np.uint8).reshape(H, img.extents.stride)[:, :W].astype(np.float32)
    if bold:                                              # synthetic emboldening: 1 px dilation of the coverage
        a = alpha
        alpha = np.maximum.reduce([a, np.roll(a, 1, 1), np.roll(a, -1, 1), np.roll(a, 1, 0), np.roll(a, -1, 0)])

    # trim the horizontal margin to the ink (keep the full line height so the layout keeps the font's ascent/descent)
    cols = np.nonzero(alpha.max(0) > 0)[0]
    x0 = min(margin, int(cols[0])) if len(cols) else margin
    x1 = max(margin + adv_w, int(cols[-1]) + 1) if len(cols) else margin + adv_w
    alpha = alpha[margin:margin + ascent + descent, x0:x1]
    height, width = alpha.shape

    image = np.zeros((height, width, 4), dtype=np.float32)
    image[..., :3] = np.asarray(color[:3], dtype=np.float32)
    image[..., 3] = alpha * (color[3] / 255.0 if len(color) > 3 else 1.0)
    bbox = [0, -ascent, width, height]

    # cluster extents: HarfBuzz cluster = index of the first char of the grapheme/ligature; x range = union of its glyphs'
    # advance boxes (in image pixels, after the trim)
    spans = {}
    pen = 0
    for (gid, cl, gx, gy), pos in zip(placed, buf.glyph_positions):
        a0 = margin + gx * s - x0
        a1 = margin + (pen + pos.x_advance) * s - x0
        lo, hi = spans.get(cl, (a0, a1))
        spans[cl] = (min(lo, a0), max(hi, a1))
        pen += pos.x_advance
    clusters = sorted((cl, float(lo), float(hi)) for cl, (lo, hi) in spans.items())
    notdef = sum(1 for gid, _, _, _ in placed if gid == 0)
    return image, bbox, clusters, notdef


def graphemes(text):
    """Extended grapheme clusters of text (regex \\X): what a human calls one Thai character incl. its marks."""
    return regex.findall(r"\X", text)


if __name__ == "__main__":
    import os
    import sys

    from PIL import Image

    path = sys.argv[1] if len(sys.argv) > 1 else "C:/Windows/Fonts/LeelawUI.ttf"
    text = "ป้ฤๅ ปู่ ฏ๊ กิ่ง น้ำ ที่นี่ ผู้ใหญ่ Hello 123"
    image, bbox, clusters, notdef = render(text, path, 48)
    assert notdef == 0, notdef
    assert render('ก 漢字', path, 48)[3] > 0, 'notdef not detected'
    rgb = np.full(image.shape[:2] + (3,), 255, np.float32)
    a = image[..., 3:4] / 255.0
    rgb = rgb * (1 - a) + image[..., :3] * a
    out = Image.fromarray(rgb.astype(np.uint8))
    out_path = os.path.join(os.environ.get("TEMP", "."), "hb_text_test.png")
    out.save(out_path)
    print("bbox", bbox, "| clusters", len(clusters), "graphemes", len(graphemes(text)), "|", clusters[:4])
    assert image.shape[2] == 4 and bbox[2] == image.shape[1] and bbox[3] == image.shape[0]
    assert image[..., 3].max() > 0, "no ink rendered"
    print("ok ->", out_path)
