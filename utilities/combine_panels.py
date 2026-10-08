#!/usr/bin/env python
"""Combine single-panel plots into one multi-panel figure with PLOS-style panel letters (A, B, ...).

PLOS removes the figures from the manuscript and typesets each one from its own uploaded file, so a
figure made of LaTeX subfigures loses its sub-captions: every panel has to be in one image, lettered
in the image itself, with the panels described in the main caption ("A: California. B: New
Mexico."). The panels are each script's own output at its intended page width (see
plos_compbio_style.py), so they are placed at their native size -- not rescaled, which would change
their 8-12pt text -- in a grid read row by row, each with its letter in bold Arial above its top-left
corner. The horizontal gaps are widened so that the figure fills the full page width (6.5in) when
the panels are narrower than that.

The output is RGB on white (PLOS doesn't accept transparency) at the panels' resolution: a .png for
the LaTeX draft, or a .tif (LZW-compressed) for the submission.

Example, four half-width panels in two rows:
  combine_panels.py --cols 2 -o ca-p01-main.png \\
      ca-p01-main-exposed.png ca-p01-main-cum-exposed.png ca-p01-main-hospitalized.png ca-p01-main-dead.png
"""

import argparse
import os
import string
import sys

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import FULL_PAGE_WIDTH_IN  # noqa: E402

# PLOS figure text is 8-12pt; the letters sit at the top of that range so they stand out from the
# panels' own 8-9pt text.
LETTER_PT = 11
MIN_GAP_IN = 0.08
ROW_GAP_IN = 0.06


def load_panel(path):
    """The panel as RGB on white, and its resolution in dpi (300 if the file doesn't say)."""
    im = Image.open(path)
    dpi = im.info.get("dpi", (300, 300))[0]
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    return im, round(dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("panels", nargs="+", help="Panel images, in reading order (row by row)")
    parser.add_argument("--cols", type=int, default=2, help="Panels per row (default: 2)")
    parser.add_argument("--width", type=float, default=FULL_PAGE_WIDTH_IN,
                        help=f"Width in inches to fill by widening the gaps between columns, if the panels "
                             f"are narrower (default: {FULL_PAGE_WIDTH_IN}, the full page)")
    parser.add_argument("--output", "-o", required=True, help="Output file, .png or .tif")
    args = parser.parse_args()

    panels = [load_panel(p) for p in args.panels]
    dpis = {dpi for _, dpi in panels}
    if len(dpis) > 1:
        sys.exit(f"Panels have different resolutions ({sorted(dpis)} dpi): placing them at native size "
                 "would give them different text sizes")
    dpi = dpis.pop()
    images = [im for im, _ in panels]

    cols = min(args.cols, len(images))
    rows = [images[i:i + cols] for i in range(0, len(images), cols)]
    col_w = [max(row[c].width for row in rows if c < len(row)) for c in range(cols)]
    min_gap = round(MIN_GAP_IN * dpi)
    gap = min_gap if cols == 1 else max(min_gap, (round(args.width * dpi) - sum(col_w)) // (cols - 1))

    font_path = font_manager.findfont(font_manager.FontProperties(family="Arial", weight="bold"))
    if "arial" not in os.path.basename(font_path).lower():
        print(f"WARNING: Arial Bold not found, using {font_path}; PLOS figures need Arial, Times or Symbol")
    font = ImageFont.truetype(font_path, round(LETTER_PT / 72 * dpi))
    # A strip above each row holds its letters, so they never cover a panel's own text
    strip = round(1.25 * LETTER_PT / 72 * dpi)
    row_gap = round(ROW_GAP_IN * dpi)

    width = sum(col_w) + gap * (cols - 1)
    heights = [strip + max(im.height for im in row) for row in rows]
    height = sum(heights) + row_gap * (len(rows) - 1)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    letters = iter(string.ascii_uppercase)
    y = 0
    for row, row_h in zip(rows, heights):
        x = 0
        for c, im in enumerate(row):
            draw.text((x, y), next(letters), font=font, fill="black")
            out.paste(im, (x, y + strip))
            x += col_w[c] + gap
        y += row_h + row_gap

    kwargs = {"compression": "tiff_lzw"} if args.output.lower().endswith((".tif", ".tiff")) else {}
    out.save(args.output, dpi=(dpi, dpi), **kwargs)
    print(f"Wrote {args.output}: {len(images)} panels, {width}x{height} px "
          f"({width / dpi:.2f}x{height / dpi:.2f} in at {dpi} dpi)")


if __name__ == "__main__":
    main()
