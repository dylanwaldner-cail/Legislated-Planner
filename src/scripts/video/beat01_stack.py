"""Beat 1 -- the stack assembles itself.

Cross-dissolves the cumulative Figure 1 stages built by build_fig1_stages.py, so the pipeline
diagram draws itself one layer at a time. A short label names each layer as it lands; the detail
for every box arrives in its own later beat, so the captions here stay to two or three words.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.video import common as c  # noqa: E402

STAGE_DIR = Path("/newdata2/dylantw/tmp/claude-1727/"
                 "-newdata2-dylantw-Legislated-Planner-src/"
                 "8f348aa2-3a2b-4256-9903-13229c0043e0/scratchpad/fig1_stages")

# (stage file, caption, caption colour)
SEQ = [
    ("bands",      "Three layers",                 c.MUTE),
    ("env",        "The environment",              c.ENV),
    ("law",        "The law",                      c.LAW),
    ("perception", "Perception",                   c.LATENT),
    ("planning",   "Planning",                     c.PLAN),
    ("loop",       "A closed loop",                c.MUTE),
    ("full",       "The Legislation Planning Stack", c.INK),
]

FIG_W, FIG_TOP = 1800, 135      # figure box; the caption sits under it
CAP_Y = 890
# Captions never overlap: the figure crossfades bare, then the caption fades in over the hold and
# back out before the next transition. Blending two captioned frames ghosts the old words.
FADE, CAP_IN, HOLD, CAP_OUT, TAIL = 10, 5, 10, 5, 30


def _plate(name: str) -> Image.Image:
    """One stage, composited onto the full canvas at a fixed position (so stages register)."""
    img = Image.open(STAGE_DIR / f"stage_{name}.png").convert("RGB")
    img = c.fit_image(img, FIG_W, 900)
    f = c.new_frame()
    f.paste(img, ((c.W - img.width) // 2, FIG_TOP))
    return f


def _captioned(plate: Image.Image, label: str, colour, alpha: float) -> Image.Image:
    if alpha <= 0.01:
        return plate
    f = plate.copy()
    d = ImageDraw.Draw(f)
    # fade the caption by blending toward the background rather than using an alpha layer
    col = tuple(int(bg + (fg - bg) * alpha) for fg, bg in zip(colour, c.BG))
    c.text(d, (c.W // 2, CAP_Y), label, kind="roman_b", size=58, fill=col, anchor="ma")
    return f


def build() -> list:
    plates = [(_plate(n), lab, col) for n, lab, col in SEQ]
    frames: list = []
    frames += [c.blend(c.new_frame(), plates[0][0], c.ease((k + 1) / FADE)) for k in range(FADE)]
    for i, (plate, label, colour) in enumerate(plates):
        if i > 0:                                        # bare figure crossfade -- no caption yet
            prev = plates[i - 1][0]
            frames += [c.blend(prev, plate, c.ease((k + 1) / FADE)) for k in range(FADE)]
        frames += [_captioned(plate, label, colour, c.ease((k + 1) / CAP_IN)) for k in range(CAP_IN)]
        frames += [_captioned(plate, label, colour, 1.0)] * HOLD
        if i < len(plates) - 1:                          # clear the caption before the next stage
            frames += [_captioned(plate, label, colour, 1.0 - c.ease((k + 1) / CAP_OUT))
                       for k in range(CAP_OUT)]
    frames += [frames[-1]] * TAIL
    return frames


if __name__ == "__main__":
    fr = build()
    out = sys.argv[1] if len(sys.argv) > 1 else str(STAGE_DIR.parent / "beat01_stack.mp4")
    c.write_mp4(fr, out)
