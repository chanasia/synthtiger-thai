"""Audit fonts for Thai mark positioning and move the broken ones out.

A font that lacks GPOS mark attachment draws tone marks / vowels as spacing glyphs (they float beside the base) or
stacks a tone mark on top of an upper vowel at the same height (overlap). Pages rendered with such fonts teach a
detector that marks sit away from the body - wrong. Checks, shaped with HarfBuzz (per-character clusters):
  1. coverage: every char of the probe strings has a glyph (no .notdef);
  2. marks are non-spacing: each mark glyph has x_advance == 0;
  3. stacking: the ink top of the tone mark in "กิ่ง" is higher than in "ก่ง" (raised over the upper vowel);
  4. a lower vowel's ink lies below the baseline ("ปู่").
Fonts without Thai glyphs are kept for Latin text (the generator picks fonts by charset).

    python tools/check_thai_fonts.py resources/font_th [--move resources/font_bad] [--thai-only]
"""
import argparse
import os
import shutil

import uharfbuzz as hb

MARKS = set("ัิีึืฺุู็่้๊๋์ํ๎")
PROBES = ["กิ่ง", "ก่ง", "ปู่", "น้า", "ที่นี่", "ผู้ใหญ่", "ป้ฤๅ", "ฏ๊"]   # no SARA AM: its decomposition merges clusters


def shape(font, text):
    buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
    buf.cluster_level = hb.BufferClusterLevel.MONOTONE_CHARACTERS      # one cluster per character: marks stay identifiable
    hb.shape(font, buf)
    return list(zip(buf.glyph_infos, buf.glyph_positions))


def ink_box(font, gid, pos):
    """(top, bottom) of the glyph's ink in font units, y up, after its positioning offset; None when no extents."""
    ext = font.get_glyph_extents(gid)
    if ext is None:
        return None
    top = pos.y_offset + ext.y_bearing
    return top, top + ext.height                      # height is negative in HarfBuzz


def mark_ink(font, text, mark):
    for g, p in shape(font, text):
        if g.cluster < len(text) and text[g.cluster] == mark:
            return ink_box(font, g.codepoint, p)
    return None


def _unused_render_mass(path, text, size=64):
    import importlib.util
    global _hbt
    try:
        _hbt
    except NameError:
        spec = importlib.util.spec_from_file_location("hb_text", os.path.join(os.path.dirname(__file__), "..", "synthtiger", "layers", "hb_text.py"))
        _hbt = importlib.util.module_from_spec(spec); spec.loader.exec_module(_hbt)
    image, _, _, notdef = _hbt.render(text, path, size)
    return float(image[..., 3].sum()) / 255.0, notdef


TONES = set("่้๊๋")
STACKS = [("กิ่ง", "ิ", "่"), ("ที่", "ี", "่"), ("ชื่อ", "ื", "่"), ("น้า", "", "้"), ("ปู่", "", "่")]


def _glyph_ink(font, text, char):
    """(top, bottom, has_ink) of the glyph shaped for `char` in `text`, font units y-up, after positioning; None if absent."""
    for g, p in shape(font, text):
        if g.cluster < len(text) and text[g.cluster] == char:
            ext = font.get_glyph_extents(g.codepoint)
            if ext is None or ext.height == 0:
                return None, None, False
            top = p.y_offset + ext.y_bearing
            return top, top + ext.height, True
    return None


def audit(path):
    """-> (has_thai, problems: list of str)."""
    face = hb.Face(hb.Blob.from_file_path(path), 0); font = hb.Font(face); font.scale = (face.upem, face.upem)
    if font.get_nominal_glyph(ord("ก")) in (None, 0):
        return False, []
    problems = set()
    for text in PROBES:
        glyphs = shape(font, text)
        if any(g.codepoint == 0 for g, _ in glyphs):
            problems.add(f"notdef in {text}"); continue
        for g, p in glyphs:
            if g.cluster < len(text) and text[g.cluster] in MARKS and p.x_advance != 0:
                problems.add(f"spacing mark {text[g.cluster]!r} in {text}")
    # a mark must carry ink at all (legacy UPC fonts map marks to empty glyphs)
    for base, mark in (("ก", "่"), ("น", "้"), ("ป", "ุ")):
        r = _glyph_ink(font, base + mark, mark)
        if r is None or not r[2]:
            problems.add(f"mark {mark!r} has no glyph ink")
    base_top = _glyph_ink(font, "ก", "ก")
    base_top = base_top[0] if base_top else 0.6 * face.upem
    for text, vowel, tone in STACKS:
        t = _glyph_ink(font, text, tone)
        if t is None:                                        # the pair was swapped for one precomposed glyph: check its height
            v = _glyph_ink(font, text, vowel) if vowel else None
            v0 = _glyph_ink(font, text.replace(tone, ""), vowel) if vowel else None
            if not (v and v0 and v[2] and v0[2] and v[0] > v0[0] + 0.02 * face.upem):
                problems.add(f"stacked tone mark missing ({text})")
            continue
        if not t[2]:
            problems.add(f"stacked tone mark empty ({text})"); continue
        if vowel:
            v = _glyph_ink(font, text, vowel)
            if v and v[2] and t[1] < v[0] - 0.03 * face.upem:   # tone bottom below the vowel top = overlap
                problems.add(f"tone mark overlaps the vowel ({text})")
        elif t[1] < base_top - 0.03 * face.upem:               # bare consonant: tone must sit above the consonant
            problems.add(f"tone mark inside the consonant ({text})")
    # horizontal placement: a mark's ink must sit over its base consonant (a bad anchor floats it beside the word)
    for text in ("ก่", "น้", "ปู่", "กิ่ง", "ชื่อ"):
        pen = 0; base = None; marks = []
        for g, p in shape(font, text):
            ge = font.get_glyph_extents(g.codepoint)
            if ge is not None and ge.height != 0 and g.cluster < len(text):
                x0 = pen + p.x_offset + ge.x_bearing; x1 = x0 + ge.width
                if text[g.cluster] in MARKS:
                    marks.append((text[g.cluster], (x0 + x1) / 2))
                elif base is None:
                    base = (x0, x1)
            pen += p.x_advance
        if base:
            w = base[1] - base[0]
            for ch, cx in marks:
                if cx < base[0] - 0.35 * w or cx > base[1] + 0.35 * w:
                    problems.add(f"mark {ch!r} floats beside the base ({text})")
    return True, sorted(problems)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("font_dir"); ap.add_argument("--move", default=None, help="move failing fonts (and their .txt charset) here")
    ap.add_argument("--thai-only", action="store_true", help="also move fonts without Thai glyphs")
    a = ap.parse_args()
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.font_dir, followlinks=True) for f in fs if f.lower().endswith((".ttf", ".otf")))
    ok = bad = latin = 0
    for p in paths:
        try:
            has_thai, problems = audit(p)
        except Exception as e:  # noqa: BLE001
            has_thai, problems = True, [f"error {type(e).__name__}: {e}"]
        if not has_thai:
            latin += 1
            if a.thai_only and a.move:
                os.makedirs(a.move, exist_ok=True); shutil.move(p, os.path.join(a.move, os.path.basename(p)))
            continue
        if problems:
            bad += 1; print(f"BAD  {os.path.relpath(p, a.font_dir)}: " + "; ".join(problems))
            if a.move:
                os.makedirs(a.move, exist_ok=True); shutil.move(p, os.path.join(a.move, os.path.basename(p)))
                cs = os.path.splitext(p)[0] + ".txt"
                if os.path.exists(cs):
                    shutil.move(cs, os.path.join(a.move, os.path.basename(cs)))
        else:
            ok += 1
    print(f"thai ok {ok}, thai bad {bad}, no thai {latin}" + (f" -> bad moved to {a.move}" if a.move else ""))
