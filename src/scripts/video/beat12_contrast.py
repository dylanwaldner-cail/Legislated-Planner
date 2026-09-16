"""Beat 12 -- what the legal layer buys, as a controlled pair.

THE COMPARISON IS MATCHED WHERE IT CAN BE. Both clips should come from the same law_eval scenario
bank, the same task and the same scene_offset/index, so the start pose, the goal and the seed are
identical and the ONLY difference is legislation.mode. If MATCHED is False the pair is not yet
matched and the beat says so on screen rather than implying a controlled comparison it does not have.

The pooled rates are the recorded 400-episode-per-agent run (results/no_yaw/sign_change); they
reproduce the paper's Q1 numbers and were checked against it (social 0.647 vs realistic 0.110
swept abidance, i.e. 5.9x).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.video import common as c  # noqa: E402

DOCS = Path("/newdata2/dylantw/Legislated-Planner/docs/assets/video")
LEFT = dict(clip=DOCS / "ep_blatant.mp4", label="No legal layer",
            agent="realistic", colour=c.PLAN, outcome="drives through the center")
RIGHT = dict(clip=DOCS / "ep_abiding.mp4", label="Legislated", 
             agent="social", colour=c.LAW, outcome="detours around it")
MATCHED = False        # flip to True once both clips are the same task + scene index
MATCH_NOTE = "same task · same start and goal · only the legal layer differs"
UNMATCHED_NOTE = "different scenarios — illustrative, not a controlled pair"

PANEL = 470
PX = (620, 1300)
PY = 470
POOLED = [("realistic", "11%", c.PLAN), ("social", "65%", c.LAW)]


def build() -> list:
    a = c.read_mp4(LEFT["clip"])
    b = c.read_mp4(RIGHT["clip"])
    n = max(len(a), len(b))
    a += [a[-1]] * (n - len(a))                      # hold the shorter clip's last frame
    b += [b[-1]] * (n - len(b))

    frames: list = []
    for k in range(n):
        f = c.new_frame()
        d = ImageDraw.Draw(f)
        c.text_tracked(d, (c.W // 2, 96), "What the law buys", kind="display", size=64,
                       track=1, fill=c.INK + (255,))
        c.text(d, (c.W // 2, 132), MATCH_NOTE if MATCHED else UNMATCHED_NOTE,
               kind="roman_i", size=30, fill=c.MUTE if MATCHED else c.DEV, anchor="ma")
        for side, clip in ((LEFT, a[k]), (RIGHT, b[k])):
            x = PX[0] if side is LEFT else PX[1]
            im = c.fit_image(clip, PANEL, PANEL)
            f.paste(im, (x - im.width // 2, PY - im.height // 2))
            d.rectangle([x - im.width // 2 - 2, PY - im.height // 2 - 2,
                         x + im.width // 2 + 1, PY + im.height // 2 + 1],
                        outline=side["colour"], width=4)
            c.text(d, (x, PY - PANEL // 2 - 74), side["label"], kind="display", size=44,
                   fill=side["colour"], anchor="ma")
            c.text(d, (x, PY + PANEL // 2 + 26), side["outcome"], kind="roman_i", size=32,
                   fill=c.MUTE, anchor="ma")
        # pooled result, always on screen so the single pair is never mistaken for the evidence
        c.text(d, (c.W // 2, 800), "law abidance, pooled over 400 episodes each",
               kind="sans_b", size=26, fill=c.FAINT, anchor="ma")
        for i, (name, pct, col) in enumerate(POOLED):
            x = PX[i]
            c.text_tracked(d, (x, 916), pct, kind="display", size=92, track=1, fill=col + (255,))
            c.text(d, (x, 946), name, kind="roman_i", size=30, fill=c.MUTE, anchor="ma")
        c.text(d, (c.W // 2, 916), "vs", kind="roman_i", size=34, fill=c.FAINT, anchor="mm")
        frames.append(f)
    frames += [frames[-1]] * 40
    return frames


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ("/newdata2/dylantw/tmp/claude-1727/"
        "-newdata2-dylantw-Legislated-Planner-src/8f348aa2-3a2b-4256-9903-13229c0043e0/"
        "scratchpad/beat12_contrast.mp4")
    c.write_mp4(build(), out)
