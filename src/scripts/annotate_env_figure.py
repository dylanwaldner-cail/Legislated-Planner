"""Re-draw the annotation overlay on the environment figure (Jurix Fig. 2) in the PAPER'S TYPEFACE.

WHY THIS EXISTS. `Images/env.png` was originally annotated by a throwaway script that is not in the
repo, using PIL's default face -- DejaVu Sans **Bold**. Every other piece of text in the paper is
Times: the prose and both matplotlib panels (`plot_sign_lawset.py`, `plot_cushion_sweep_lawset.py`)
render in Nimbus Roman, the URW Times clone that `mathptmx` actually ships. Figure 2 was the only
place a reader met a different family, and it was the one figure whose text is baked into pixels and
so cannot be fixed in LaTeX. This script rebuilds that overlay; see FONT_BOLD for the face it now uses and why.

GEOMETRY IS NOT GUESSED. Every rectangle, leader endpoint and swatch below was measured off the
original by diffing it against its own un-annotated base render
(`env_hi_touch.png` vs `env_hi_touch_annotated.png`) and taking connected components of the changed
pixels: box interiors from the near-white fills, border colours from the ring just outside them,
leader anchors from the far end of each leader stroke, legend swatches from the saturated blobs. The
only intentional departures from the original are the typeface and a 3x supersample of the canvas so
the glyphs are not the limiting resolution in print (the figure sits in a 0.39\\textwidth minipage,
where the 512px base is already only ~270 dpi).

Box widths/heights are recomputed from the Nimbus Roman text extent rather than copied, because
Times is narrower than DejaVu Sans Bold and reusing the old rects would leave visible slack. Each
box keeps its original PINNED CORNER (`anchor`), so nothing drifts across the scene.

    python scripts/annotate_env_figure.py
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "env_hi_touch.png")
OUT_LOCAL = os.path.join(HERE, "env_hi_touch_annotated.png")
OUT_PAPER = "/newdata2/dylantw/Legislated-Planner/Jurix_paper/Images/env.png"

# MATCH THE VIDEO'S VOICE. This was Nimbus Sans, chosen to match the diagram's own cards (main.tex
# sets its nodes in \sffamily, which `helvet` maps to Nimbus Sans). That is a defensible match and
# it is also Helvetica, which is the most anonymous type available -- the callouts read as a system
# dialog sitting on the render. Lato Black is the face the intro video already titles in
# (scripts/video/common.py:_LATO) and the one dino-wm.github.io uses, so the panel now speaks with
# the same voice as everything else built around this figure.
#
# TRADE-OFF, on purpose: inside Fig. 1 the callouts no longer match the cards beside them. They are
# a photographic overlay rather than diagram furniture, so reading as a different register is the
# intent, not a slip. Put NimbusSans-Bold.otf back here to undo it -- nothing else needs changing,
# the box widths are recomputed from the text extent.
#
# Black, not Bold: these print at roughly 5pt in the paper, and the heavier weight is what survives
# at that size. Lato Bold measurably thins out there; Black holds.
FONT_BOLD = "/usr/share/fonts/truetype/lato/Lato-Black.ttf"

S = 3                     # supersample: draw the overlay at 3x, keep the composite at 3x
# FS drives everything downstream: the bigger the type IN the render, the smaller the panel can be
# in Fig. 1 for the same printed point size, which is width the diagram gets back. At FS=25 and a
# 65mm panel the callouts print at ~5pt -- the size they were at in the old 0.38\textwidth fig:env.
FS = 23                   # callout font size, in 1x pixels
LINE = 26.0               # baseline-to-baseline, in 1x pixels
PADX, PADY = 7, 4         # box padding, in 1x pixels
RADIUS = 7                # rounded-corner radius, in 1x pixels
BORDER = 5                # box outline, in SUPERSAMPLED px -> ~1.0pt at the printed panel size
ACCENT_W = 5              # width of the colour accent bar, in 1x pixels

# Callout chrome is now NEUTRAL: white fill, one grey border everywhere. Colour appears only as an
# accent -- the leader, its dot, and a bar inside the box -- so the cell colours it encodes (yellow
# 3/5, red 4) read as data rather than as seven competing box styles.
GREY = (60, 60, 60)
BORDER_GREY = (110, 116, 128)   # inkmute, the diagram's neutral line colour
GOLD = (200, 155, 0)
RED = (200, 30, 30)
INK = (20, 20, 20)
WHITE = (255, 255, 255)

# label, text, pinned corner, (x, y) of that corner, leader anchor dot, colour
# `anchor` says which corner of the box is FIXED: left-side callouts keep their top-left, right-side
# ones keep their top-right, so a narrower typeface eats into the scene instead of off the canvas.

# The "Franka arm" and "Cube" callouts were dropped: this render now sits inside Fig. 1 at a
# fraction of a text width, where seven callouts are unreadable clutter, and both are named in the
# prose anyway. "Gripper (paddle) on the cube" already points at the arm-cube contact, so nothing
# in the scene is left unlabelled. `y5` moved up to sit level with its own leader dot, which also
# buys clearance against the Cell IDs panel below it.
# The last field is the ACCENT: the leader/dot colour, and a bar inside the box. None = no bar, for
# the two callouts that name a part rather than a cell colour.
# Positions are re-pitched for FS=25: at the old FS=20 pitch these boxes collide with each other
# (center/y5) and with the sign octagon. The sign label is shortened for the same reason -- the full
# colour list no longer fits left of the octagon, and the prose carries it.
CALLOUTS = [
    ("paddle",  "Paddle",                          "tl", (26, 128),  (163, 186), GREY, None),
    ("y3",      "Yellow cell 3",                   "tr", (476, 140), (252, 220), GOLD, GOLD),
    ("center",  "Illegal center\n(cell 4)",        "tr", (492, 226), (262, 260), RED,  RED),
    ("y5",      "Yellow cell 5",                   "tr", (486, 300), (266, 300), GOLD, GOLD),
    ("sign",    "Rule sign\n(Flips green,\nyellow, red)", "tl", (14, 378), (250, 424), GREY, None),
]

# Cell-ID legend, measured off the original: panel interior, then the 3x3 swatch grid.
PANEL = (382, 350, 512, 499)          # x1 == 512 is the image edge in the original; preserved
GRID_X = (389, 429, 469)              # swatch left edges; each swatch is 36px, 4px gutter
GRID_Y = (373, 413, 453)
CELL = 36
GREEN_SW = (120, 200, 90)
YELLOW_SW = (240, 225, 70)
RED_SW = (202, 65, 56)
# row-major from the TOP row down, matching the original: 6 7 8 / 3 4 5 / 0 1 2
LEGEND = [[("6", GREEN_SW), ("7", GREEN_SW), ("8", GREEN_SW)],
          [("3", YELLOW_SW), ("4", RED_SW), ("5", YELLOW_SW)],
          [("0", GREEN_SW), ("1", GREEN_SW), ("2", GREEN_SW)]]


def _lines(text):
    return text.split("\n")


def _text_extent(draw, text, font):
    """(w, h) of a possibly multi-line label, in supersampled pixels."""
    w = 0
    for ln in _lines(text):
        x0, _, x1, _ = draw.textbbox((0, 0), ln, font=font)
        w = max(w, x1 - x0)
    return w, int(round(LINE * S * len(_lines(text))))


def _edge_point(box, target):
    """Where a leader leaves the box: the point on the box outline closest to `target`.

    Clamping the target into the rect and then snapping to the nearest side gives the same
    "line emerges from the corner facing its subject" look the original had, without hand-placing
    eight attachment points that would then have to be re-tuned whenever a box resizes.
    """
    x0, y0, x1, y1 = box
    tx, ty = target
    cx = min(max(tx, x0), x1)
    cy = min(max(ty, y0), y1)
    # Leave by the side the subject actually lies off: pick the axis with the LARGER displacement,
    # which reproduces the original's attachment on all seven callouts (nearest-side does not --
    # it would send "Cube" out of the top edge instead of the right).
    if min(abs(tx - x0), abs(tx - x1)) > min(abs(ty - y0), abs(ty - y1)):
        return (x0 if abs(tx - x0) < abs(tx - x1) else x1), cy
    return cx, (y0 if abs(ty - y0) < abs(ty - y1) else y1)


def main():
    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    img = base.resize((W * S, H * S), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_BOLD, FS * S)
    legend_font = ImageFont.truetype(FONT_BOLD, int(round(FS * 1.05 * S)))

    for _name, text, anchor, (ax, ay), dot, col, accent in CALLOUTS:
        tw, th = _text_extent(d, text, font)
        # an accent bar steals width from the text side, so the box has to grow to hold both
        lead = (ACCENT_W + 4) * S if accent else 0
        bw, bh = tw + 2 * PADX * S + lead, th + 2 * PADY * S
        y0 = ay * S
        x0 = ax * S if anchor == "tl" else ax * S - bw
        box = (x0, y0, x0 + bw, y0 + bh)
        ex, ey = _edge_point(box, (dot[0] * S, dot[1] * S))
        d.line([(ex, ey), (dot[0] * S, dot[1] * S)], fill=col, width=2 * S)
        d.rounded_rectangle(box, radius=RADIUS * S, fill=WHITE, outline=BORDER_GREY, width=BORDER)
        if accent:
            bx = x0 + PADX * S
            d.rounded_rectangle((bx, y0 + PADY * S, bx + ACCENT_W * S, y0 + bh - PADY * S),
                                radius=int(ACCENT_W * S / 2), fill=accent)
        for i, ln in enumerate(_lines(text)):
            d.text((x0 + PADX * S + lead, y0 + PADY * S + int(round(i * LINE * S))), ln,
                   font=font, fill=INK)
        r = 4 * S
        d.ellipse([dot[0] * S - r, dot[1] * S - r, dot[0] * S + r, dot[1] * S + r], fill=col)

    px0, py0, px1, py1 = [v * S for v in PANEL]
    d.rounded_rectangle((px0, py0, px1 - 1, py1), radius=RADIUS * S,
                        fill=WHITE, outline=BORDER_GREY, width=BORDER)
    d.text((px0 + 6 * S, py0 + 3 * S), "Cell IDs", font=font, fill=INK)
    for row, gy in zip(LEGEND, GRID_Y):
        for (label, swatch), gx in zip(row, GRID_X):
            cx0, cy0 = gx * S, gy * S
            cx1, cy1 = cx0 + CELL * S, cy0 + CELL * S
            d.rounded_rectangle((cx0, cy0, cx1, cy1), radius=3 * S, fill=swatch,
                                outline=BORDER_GREY, width=1 * S)
            tb = d.textbbox((0, 0), label, font=legend_font)
            d.text((cx0 + (CELL * S - (tb[2] - tb[0])) / 2 - tb[0],
                    cy0 + (CELL * S - (tb[3] - tb[1])) / 2 - tb[1]),
                   label, font=legend_font, fill=INK)

    for path in (OUT_LOCAL, OUT_PAPER):
        img.save(path)
        print("wrote", path, img.size)


if __name__ == "__main__":
    main()
