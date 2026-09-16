"""Beat 13 -- the closing statement and the end card.

The claim the paper actually makes, not a grander one: robot governance has to act BEFORE the
illegal action executes, and there is no finished robot law -- it is iterative, like the human legal
system. Both gaps are measurement problems, which is why they were measured rather than asserted.
Wording tracks Jurix_paper/main.tex sec:intro ("no end goal for robot law ... an iterative process
akin to the human legal system") and sec:conc.
"""
from __future__ import annotations

import sys

from PIL import ImageDraw

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.video import common as c  # noqa: E402

LINES = [
    ("Law is how a society shapes", c.MUTE),
    ("the behavior of populations.", c.MUTE),
]
LINES2 = [
    ("For robots it has to act", c.MUTE),
    ("before the action executes.", c.INK),
]
LINES3 = [
    ("There is no finished robot law.", c.INK),
    ("Like our own, it is iterative:", c.MUTE),
    ("measure the gaps, close them,", c.MUTE),
    ("legislate again.", c.LAW),
]

TITLE = "Legislating World-Model-Based\nPlanning with Legal Reasoning"
AUTHORS = "Dylan Waldner · Yiannis Kantaros · Guido Governatori\nRisto Miikkulainen · Amir Banifatemi"
AFFIL = "Cognizant AI Lab · Washington University · Central Queensland University · UT Austin"
LINKS = "arxiv.org/abs/2609.15113"
PAGE = "dylanwaldner-cail.github.io/Legislated-Planner"


def _statement(lines, y0=400, size=62, gap=86):
    f = c.new_frame()
    d = ImageDraw.Draw(f)
    for i, (txt, col) in enumerate(lines):
        c.text_tracked(d, (c.W // 2, y0 + i * gap), txt, kind="display", size=size,
                       track=1, fill=col + (255,))
    return f


def _endcard():
    f = c.new_frame()
    d = ImageDraw.Draw(f)
    c.text(d, (c.W // 2, 250), TITLE, kind="roman_b", size=66, fill=c.INK,
           anchor="ma", align="center", spacing=16)
    c.text(d, (c.W // 2, 470), AUTHORS, kind="roman", size=38, fill=c.MUTE,
           anchor="ma", align="center", spacing=12)
    c.text(d, (c.W // 2, 590), AFFIL, kind="roman_i", size=28, fill=c.FAINT,
           anchor="ma", align="center")
    d.line([(c.W // 2 - 220, 680), (c.W // 2 + 220, 680)], fill=c.RULE, width=2)
    c.text_tracked(d, (c.W // 2, 776), LINKS, kind="display", size=46, track=1, fill=c.LAW + (255,))
    c.text(d, (c.W // 2, 820), PAGE, kind="roman", size=32, fill=c.MUTE, anchor="ma")
    return f


def build() -> list:
    frames: list = []
    cards = [_statement(LINES), _statement(LINES2), _statement(LINES3, y0=330), _endcard()]
    holds = [42, 48, 78, 110]
    for i, (card, hold) in enumerate(zip(cards, holds)):
        prev = frames[-1] if frames else c.new_frame()
        frames += [c.blend(prev, card, c.ease((k + 1) / 14)) for k in range(14)]
        frames += [card] * hold
    return frames


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ("/newdata2/dylantw/tmp/claude-1727/"
        "-newdata2-dylantw-Legislated-Planner-src/8f348aa2-3a2b-4256-9903-13229c0043e0/"
        "scratchpad/beat13_closing.mp4")
    c.write_mp4(build(), out)
