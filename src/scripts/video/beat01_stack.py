"""Beat 1 -- the stack assembles itself.

Cross-dissolves the cumulative Figure 1 stages from build_fig1_stages.py, so the pipeline diagram
draws itself one layer at a time in the paper's own vector output.

CAPTION HANDOFF. A straight cross-dissolve of two words sits at 50/50 in the middle, which reads as
both words washing out rather than one becoming the other. Instead the outgoing word fades while
rising and the incoming word fades in from below, with only a brief low-alpha overlap -- so a word
is legible at essentially all times and the change reads as a replacement. The FIGURE still
cross-dissolves underneath; only the caption is handed off this way.
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

# The first caption names exactly the THREE bands that appear, so the environment -- which is not a
# layer -- cannot be miscounted as a fourth.
SEQ = [
    ("bands",      "Law, perception, planning",     c.MUTE),
    ("env",        "The environment",               c.ENV),
    ("law",        "The law",                       c.LAW),
    ("perception", "Perception",                    c.LATENT),
    ("planning",   "Planning",                      c.PLAN),
    ("loop",       "A closed loop",                 c.MUTE),
    ("full",       "The Legislation Planning Stack", c.INK),
]

FIG_W, FIG_TOP = 1820, 120
CAP_Y, CAP_SIZE, CAP_TRACK, CAP_RISE = 948, 84, 1, 26   # CAP_Y is the BASELINE
FADE, HOLD, TAIL = 16, 26, 30


def _plate(name: str) -> Image.Image:
    img = c.fit_image(Image.open(STAGE_DIR / f"stage_{name}.png").convert("RGB"), FIG_W, 880)
    f = c.new_frame()
    f.paste(img, ((c.W - img.width) // 2, FIG_TOP))
    return f


def _caption(base: Image.Image, label, colour, alpha: float, dy: float) -> Image.Image:
    """Composite one caption over `base` at the given opacity and vertical offset."""
    if alpha <= 0.004:
        return base
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    c.text_tracked(ImageDraw.Draw(layer), (c.W // 2, CAP_Y + dy), label,
                   kind="display", size=CAP_SIZE, track=CAP_TRACK,
                   fill=colour + (int(255 * min(1.0, alpha)),))
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def _handoff(p: float):
    """progress p in [0,1] -> (out_alpha, out_dy, in_alpha, in_dy). Brief low-alpha overlap only."""
    o = c.ease(min(1.0, p / 0.55))                 # outgoing word: fades while rising
    i = c.ease(max(0.0, (p - 0.45) / 0.55))        # incoming word: fades in from below
    return 1.0 - o, -CAP_RISE * o, i, CAP_RISE * (1.0 - i)


def build() -> list:
    plates = [_plate(n) for n, _, _ in SEQ]
    frames: list = []

    first = _caption(plates[0], SEQ[0][1], SEQ[0][2], 1.0, 0.0)
    frames += [c.blend(c.new_frame(), first, c.ease((k + 1) / 12)) for k in range(12)]
    frames += [first] * HOLD

    for i in range(len(SEQ) - 1):
        (_, la, ca), (_, lb, cb) = SEQ[i], SEQ[i + 1]
        for k in range(FADE):
            p = (k + 1) / FADE
            plate = c.blend(plates[i], plates[i + 1], c.ease(p))     # figure cross-dissolves
            oa, ody, ia, idy = _handoff(p)
            f = _caption(plate, la, ca, oa, ody)
            f = _caption(f, lb, cb, ia, idy)
            frames.append(f)
        frames += [_caption(plates[i + 1], lb, cb, 1.0, 0.0)] * HOLD

    frames += [frames[-1]] * TAIL
    return frames


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(STAGE_DIR.parent / "beat01_stack.mp4")
    c.write_mp4(build(), out)
