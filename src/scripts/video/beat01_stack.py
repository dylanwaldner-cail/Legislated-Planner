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
# The caption stays up the whole beat and dissolves straight into the next word -- no fade to
# nothing. Keeping the dissolve SHORT relative to the hold is what stops the two words ghosting:
# most of the time exactly one caption is at full opacity.
FADE, HOLD, TAIL = 8, 26, 28


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
    """Each stage is a fully captioned plate; consecutive plates cross-dissolve into one another."""
    plates = [_captioned(_plate(n), lab, col, 1.0) for n, lab, col in SEQ]
    frames: list = [c.blend(c.new_frame(), plates[0], c.ease((k + 1) / 12)) for k in range(12)]
    frames += [plates[0]] * HOLD
    for prev, nxt in zip(plates, plates[1:]):
        frames += [c.blend(prev, nxt, c.ease((k + 1) / FADE)) for k in range(FADE)]
        frames += [nxt] * HOLD
    frames += [frames[-1]] * TAIL
    return frames


if __name__ == "__main__":
    fr = build()
    out = sys.argv[1] if len(sys.argv) > 1 else str(STAGE_DIR.parent / "beat01_stack.mp4")
    c.write_mp4(fr, out)
