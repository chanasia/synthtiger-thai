"""Convert a det_page run (pages.jsonl + images/) into the win-ocr detector record format:

    {"image": "images/synthp_000123.jpg", "w", "h", "source": "synth_photo",
     "boxes": [quad x 8 ...], "ink": [same quads], "ink_ok": [1 ...], "texts": [...], "clusters": [[(text, quad), ...] ...]}

"boxes" = "ink" = the ink-tight quad of every line (the geometry head's target is exact here, so no separate label box);
the font-metric quad is kept as "font_boxes" for reference. Images are copied (or hard-linked) into <dst_images>.

    python tools/det_page_to_jsonl.py --run out --dst-jsonl det_v1/train_synthp.jsonl --dst-images det_v1/images --prefix synthp_
"""
import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="synthtiger output dir (pages.jsonl + images/)")
ap.add_argument("--dst-jsonl", required=True)
ap.add_argument("--dst-images", required=True, help="directory the images are copied into")
ap.add_argument("--prefix", default="synthp_")
ap.add_argument("--rel", default="images", help="image path prefix written into the jsonl (relative to the det root)")
ap.add_argument("--min-lines", type=int, default=1)
a = ap.parse_args()

os.makedirs(a.dst_images, exist_ok=True)
n_pages = n_lines = 0
with open(os.path.join(a.run, "pages.jsonl"), encoding="utf8") as f, open(a.dst_jsonl, "w", encoding="utf8", newline="\n") as out:
    for l in f:
        if not l.strip():
            continue
        r = json.loads(l)
        if len(r["lines"]) < a.min_lines:
            continue
        idx = os.path.splitext(os.path.basename(r["image"]))[0]
        name = f"{a.prefix}{int(idx):06d}.jpg"
        src = os.path.join(a.run, r["image"]); dst = os.path.join(a.dst_images, name)
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        row = {
            "image": f"{a.rel}/{name}", "w": r["w"], "h": r["h"], "source": "synth_photo",
            "boxes": [ln["quad"] for ln in r["lines"]],
            "ink": [ln["quad"] for ln in r["lines"]],
            "ink_ok": [1] * len(r["lines"]),
            "font_boxes": [ln["font_quad"] for ln in r["lines"]],
            "texts": [ln["text"] for ln in r["lines"]],
            "clusters": [[[c["text"], c["quad"]] for c in ln["clusters"]] for ln in r["lines"]],
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); n_pages += 1; n_lines += len(r["lines"])
print(f"{n_pages} pages, {n_lines} lines -> {a.dst_jsonl}")
