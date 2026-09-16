"""Beat 11 -- the ONTOLOGICAL isomorphism gap, shown with deliberately constructed pushes.

WHY CONSTRUCTED AND NOT MINED FROM THE DATA. A real episode happens to fall on one side of each
reading; it cannot isolate WHY. These two pushes are built so that exactly one choice flips the
verdict, which is the whole claim: one law, several faithful readings, different answers.

Every verdict badge is COMPUTED at build time with the repo's own predicate
(probes.probe_cube_cells.swept_cells, the same function planning_metrics and the Q5 scripts use),
not typed in. If the geometry stopped doing what the caption says, the badge would change and the
build would disagree with itself -- see the assertions in build().

HOW THE READINGS ARE DRAWN, per Dylan's spec:
  centre    -> ONE line tracing the cube's centre
  footprint -> TWO lines a full cube-width apart (the swept band the law is really enforced against)
  full path -> the continuous trace between rest poses
  at rest   -> only the poses the planner actually perceives

Overhead, no arm: the robot is irrelevant to the question and only adds occlusion.

SCENE A  straight through the centre, starting and ending clear of it.
         Both AT-REST readings acquit a cube that was driven straight through the forbidden cell.
SCENE B  a corner clip. The footprint crosses; the centre never does.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from probes.probe_cube_cells import CUBE_HALF, swept_cells  # noqa: E402
from probes.probe_cube_position import gm  # noqa: E402
from scripts.video import common as c, grid  # noqa: E402

FOOT, CENT = CUBE_HALF, 0.0
READINGS = [("centre", CENT, "full path", True), ("centre", CENT, "at rest", False),
            ("footprint", FOOT, "full path", True), ("footprint", FOOT, "at rest", False)]

# Three constructed pushes, all clear of the grid edge, all verified in build().
#   1) corner clip in ONE stroke   -> only footprint x full-path convicts
#   2) the SAME corner clip, SAME endpoints, in TWO strokes (it rests at the corner)
#      -> at-rest now convicts too. Nothing physical changed; the planner just stopped there.
#   3) straight THROUGH the cell, starting and ending clear of it
#      -> both at-rest readings acquit a cube driven right through the forbidden cell.
SCENES = [
    dict(path=[(0.152, 0.03), (0.03, 0.152)],
         title="Clipping the corner",
         sub="one stroke \u00b7 the centre never enters",
         note="only the swept footprint sees it"),
    dict(path=[(0.152, 0.03), (0.09, 0.09), (0.03, 0.152)],
         title="The same corner, two strokes",
         sub="same start, same end \u00b7 it merely STOPS at the corner",
         note="stopping there is what makes it illegal at rest"),
    dict(path=[(-0.155, 0.0), (0.155, 0.0)],
         title="Pushed straight through",
         sub="one stroke \u00b7 starts and ends clear of the cell",
         note="at rest, a cube driven through the cell looks clean"),
]

PANEL, PX, PY = 320, (700, 1290), (430, 812)
LBL_X = 330


def verdict(path, half, swept) -> bool:
    """True = ILLEGAL under this reading. Uses the repo's predicate, not a re-derivation."""
    P = [np.asarray(p, float) for p in path]
    if swept:
        return any(bool(swept_cells(P[t], P[t + 1], half)[4]) for t in range(len(P) - 1))
    return any(bool(swept_cells(p, p, half)[4]) for p in P)


def _band_edges(path, half):
    """The two rails a cube of half-width `half` sweeps: each segment offset +-half perpendicular."""
    rails = []
    P = [np.asarray(p, float) for p in path]
    for t in range(len(P) - 1):
        d = P[t + 1] - P[t]
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        perp = np.array([-d[1], d[0]]) / n * half
        rails.append((P[t] + perp, P[t + 1] + perp))
        rails.append((P[t] - perp, P[t] - perp + (P[t + 1] - P[t])))
    return rails


def panel(d, g, scene, half, swept, progress: float, show_verdict: bool):
    g.draw_grid(d, label_cells=False)
    P = [np.asarray(p, float) for p in scene["path"]]

    # total arclength parameterisation so the cube moves at constant speed through a turn
    segs = [np.linalg.norm(P[t + 1] - P[t]) for t in range(len(P) - 1)]
    total = sum(segs)
    want = progress * total
    pts, acc = [P[0]], 0.0
    for t, L in enumerate(segs):
        if acc + L <= want:
            pts.append(P[t + 1]); acc += L
        else:
            f = max(0.0, (want - acc) / L) if L > 1e-9 else 0.0
            pts.append(P[t] + (P[t + 1] - P[t]) * f)
            break
    col = c.GAP_ONTO if (show_verdict and verdict(scene["path"], half, swept)) else c.LAW

    if swept:
        if half > 0:                                        # footprint -> two rails, a cube apart
            for a, b in _band_edges(pts, half):
                g.draw_edge(d, a, b, col, width=5)
        else:                                               # centre -> one line
            for t in range(len(pts) - 1):
                g.draw_edge(d, pts[t], pts[t + 1], col, width=6)
    else:
        reached = [p for k, p in enumerate(P) if k == 0 or sum(segs[:k]) <= want + 1e-9]
        for p in reached:
            if half > 0:
                g.draw_footprint(d, *p, col, width=5)
            else:
                g.draw_node(d, *p, col, r=9)

    # the moving cube itself, always shown so the motion is legible
    cur = pts[-1]
    g.draw_footprint(d, *cur, (90, 94, 100), width=2)
    g.draw_node(d, *cur, (60, 64, 70), r=5)


def frame(si: int, progress: float, show_verdict: bool):
    sc = SCENES[si]
    f = c.new_frame()
    d = ImageDraw.Draw(f)
    c.text(d, (c.W // 2, 54), sc["title"], kind="display", size=60, fill=c.INK, anchor="ma")
    c.text(d, (c.W // 2, 132), sc["sub"], kind="roman_i", size=34, fill=c.MUTE, anchor="ma")

    for cx, lab in zip(PX, ("FULL PATH", "AT REST")):
        c.text(d, (cx, 235), lab, kind="sans_b", size=28, fill=c.FAINT, anchor="ma")
    for cy, lab in zip(PY, ("CENTRE", "FOOTPRINT")):
        c.text(d, (LBL_X, cy), lab, kind="sans_b", size=28, fill=c.FAINT, anchor="rm")

    for k, (body, half, when, swept) in enumerate(READINGS):
        g = grid.GridCanvas(PX[k % 2], PY[k // 2], PANEL)
        panel(d, g, sc, half, swept, progress, show_verdict)
        if show_verdict:
            bad = verdict(sc["path"], half, swept)
            txt = "ILLEGAL" if bad else "legal"
            bx, by = PX[k % 2], PY[k // 2] + PANEL // 2 - 34
            w, _ = c.measure(txt, "display" if bad else "roman_i", 32)
            d.rounded_rectangle([bx - w / 2 - 16, by - 8, bx + w / 2 + 16, by + 44],
                                radius=10, fill=(255, 255, 255),
                                outline=c.GAP_ONTO if bad else (0, 0x88, 0), width=2)
            c.text(d, (bx, by), txt, kind="display" if bad else "roman_i", size=32,
                   fill=c.GAP_ONTO if bad else (0, 0x88, 0), anchor="ma")
    c.text(d, (c.W // 2, 992), sc.get("note", ""), kind="roman_b", size=32,
           fill=c.GAP_ONTO, anchor="ma")
    c.text(d, (c.W // 2, 1042), "one law · four faithful readings",
           kind="roman_i", size=26, fill=c.FAINT, anchor="ma")
    return f


def build() -> list:
    # the beat only makes its point if the readings actually disagree -- check, do not assume
    for sc in SCENES:
        vs = [verdict(sc["path"], h, s) for _, h, _, s in READINGS]
        assert len(set(vs)) > 1, f"scene {sc['title']!r} has no disagreement: {vs}"
        print(f"[beat11] {sc['title']:<26} " +
              "  ".join(f"{b}/{w}={'ILLEGAL' if v else 'legal'}"
                        for (b, _, w, _), v in zip(READINGS, vs)))
    frames: list = []
    for si in range(len(SCENES)):
        N = 50
        frames += [frame(si, (k + 1) / N, False) for k in range(N)]
        tgt = frame(si, 1.0, True)
        frames += [c.blend(frames[-1], tgt, c.ease((k + 1) / 8)) for k in range(8)]
        frames += [tgt] * 155
    return frames


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ("/newdata2/dylantw/tmp/claude-1727/"
        "-newdata2-dylantw-Legislated-Planner-src/8f348aa2-3a2b-4256-9903-13229c0043e0/"
        "scratchpad/beat11_ontology.mp4")
    c.write_mp4(build(), out)
