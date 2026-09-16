"""Beat 6 -- the DDL engine, as a before/after on one added fact.

EVERY STRING HERE IS REAL ENGINE OUTPUT. Reproduce with:
    python -m legislation.engine_trace --lawset geometric_laws --state "in_cell(4)"
    python -m legislation.engine_trace --lawset geometric_laws --state "in_cell(4) sign(green)"

The beat holds the cube's PHYSICAL STATE FIXED (in_cell(4)) and adds one fact, sign(green). The
verdict inverts: the prohibition on the centre cell and the contrary-to-duty repair duty both
vanish, because `green_sign_permits_center > no_center_cell` outranks the prohibition. That is
defeasibility, superiority and CTD compensation in a single side-by-side, with nothing staged --
the rule `no_center_cell` is APPLICABLE in both columns; it simply loses in the second.

The engine's raw answer set also contains ~100 `refuted(...)` atoms and 9 `discarded(reach_goal_N)`
copies. Those are omitted here as noise; the applicable rules and the verdict are verbatim.
"""
from __future__ import annotations

import sys

from PIL import ImageDraw

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.video import common as c  # noqa: E402

# --- verbatim from engine_trace -------------------------------------------------------------
RULES = [                       # (label, rendered .dl text, applicable in phase>=?)
    ("no_center_cell",            "cube ⇒ [O]¬in_cell(4)",                     0),
    ("center_ctd",                "cube ⇒ [O]¬in_cell(4) ⊗ exit_cell(4)", 0),
    ("no_off_grid",               "cube ⇒ [O]¬off_grid",                       0),
    ("green_sign_permits_center", "sign(green) ⇒ [P]in_cell(4)",                    3),
]
FACTS_A = ["cube", "in_cell(4)"]
FACTS_B = ["cube", "in_cell(4)", "sign(green)"]
VERDICT_A = [("prohibitions", ["in_cell(4)", "off_grid"], c.PLAN),
             ("obligations",  ["exit_cell(4)"],           c.LAW),
             ("violations",   ["non(in_cell(4))"],        c.PLAN)]
VERDICT_B = [("prohibitions", ["off_grid"], c.PLAN),
             ("obligations",  [],           c.LAW),
             ("violations",   [],           c.LAW)]

COL_F, COL_R, COL_V = 90, 620, 1420
TOP = 250
PHASES = [(0, 30), (1, 45), (2, 55), (3, 40), (4, 50), (5, 90)]   # (phase, n_frames)


def _panel(d, x, y, w, h, title):
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, outline=c.RULE, width=2, fill=c.PANEL)
    c.text(d, (x + 22, y + 18), title, kind="sans_b", size=26, fill=c.FAINT)


def frame(phase: int):
    f = c.new_frame()
    d = ImageDraw.Draw(f)
    c.text(d, (c.W // 2, 70), "The reasoner", kind="roman_b", size=62, anchor="ma")
    sub = ("same cube, same cell" if phase < 3 else
           "one fact added — the verdict inverts")
    c.text(d, (c.W // 2, 150), sub, kind="roman_i", size=36, fill=c.MUTE, anchor="ma")

    # ---- facts
    facts = FACTS_B if phase >= 3 else FACTS_A
    _panel(d, COL_F, TOP, 440, 300, "FACTS")
    for i, a in enumerate(facts):
        new = (phase >= 3 and a == "sign(green)")
        c.text(d, (COL_F + 30, TOP + 78 + i * 58), a, kind="mono", size=34,
               fill=(0, 0x88, 0) if new else c.INK)

    # ---- rules
    _panel(d, COL_R, TOP, 740, 420, "RULES APPLICABLE")
    yy = TOP + 78
    for label, body, appears in RULES:
        if phase < 1 or phase < appears:
            yy += 92
            continue
        struck = (phase >= 4 and label == "no_center_cell")
        col = c.FAINT if struck else c.INK
        c.text(d, (COL_R + 28, yy), label, kind="sans_b", size=26, fill=c.FAINT if struck else c.LAW)
        c.text(d, (COL_R + 28, yy + 34), body, kind="mono", size=30, fill=col)
        if struck:
            w, _ = c.measure(body, "mono", 30)
            d.line([COL_R + 24, yy + 50, COL_R + 40 + w, yy + 50], fill=c.GAP_ONTO, width=4)
        yy += 92
    if phase >= 4:
        c.text(d, (COL_R + 28, TOP + 360),
               "green_sign_permits_center  >  no_center_cell",
               kind="mono_b", size=27, fill=c.GAP_ONTO)

    # ---- verdict
    verdict = VERDICT_B if phase >= 5 else (VERDICT_A if phase >= 2 else None)
    _panel(d, COL_V, TOP, 410, 420, "VERDICT")
    if verdict:
        yy = TOP + 78
        for name, items, colour in verdict:
            c.text(d, (COL_V + 26, yy), name, kind="sans_b", size=24, fill=c.FAINT)
            if items:
                for it in items:
                    yy += 40
                    c.text(d, (COL_V + 26, yy), it, kind="mono", size=28, fill=colour)
            else:
                yy += 40
                c.text(d, (COL_V + 26, yy), "—", kind="mono", size=28, fill=c.FAINT)
            yy += 64
    if phase == 2:
        c.text(d, (COL_V + 205, TOP + 460), "in violation", kind="roman_b", size=32,
               fill=c.PLAN, anchor="ma")
    if phase == 5:
        c.text(d, (COL_V + 205, TOP + 460), "no violation", kind="roman_b", size=32,
               fill=(0, 0x88, 0), anchor="ma")
    return f


def build() -> list:
    frames: list = []
    for ph, n in PHASES:
        target = frame(ph)
        if frames:
            frames += [c.blend(frames[-1], target, c.ease((k + 1) / 8)) for k in range(8)]
        else:
            frames += [c.blend(c.new_frame(), target, c.ease((k + 1) / 10)) for k in range(10)]
        frames += [target] * n
    return frames


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ("/newdata2/dylantw/tmp/claude-1727/"
        "-newdata2-dylantw-Legislated-Planner-src/8f348aa2-3a2b-4256-9903-13229c0043e0/"
        "scratchpad/beat06_ddl.mp4")
    c.write_mp4(build(), out)
