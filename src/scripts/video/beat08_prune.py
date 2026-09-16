"""Beat 8 -- the verdict becomes a planning constraint: candidate strokes get pruned.

EVERY NUMBER AND POSITION IS REAL. Read straight from the planner's own trace sidecars
(`rrt_tree.npz`, `rrt_candidates.npz`) written by plan.py:_dump_trace_sidecars during a social run:

    results/yawlock/OFF_A/social/0_8/batch_000   step 1, eval 6

That step expands four times from a 4-node tree -- the "about five nodes" scale, chosen because it
is legible, not because it flatters the method. For each expansion the trace stores BOTH winners:
`free_*`, the best stroke ignoring legality, and `pick_*`, the best stroke the rule actually allows,
plus `n_viol/n_cands`, how many of the 64 sampled strokes the DDL layer killed. `pick_viol == -1`
means every one of the 64 was illegal and no node was added at all.

HONEST LIMITATION: only those two representative strokes per expansion were saved, not the full fan
of 64 (trace_cfg.py notes the full fan would cost 64x). So the fan is not drawn -- only the rule-blind
winner, the legal winner, and the true counter. Nothing here implies we are showing all 64.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.video import common as c, grid  # noqa: E402

RUN = "results/yawlock/OFF_A/social/0_8/batch_000"
STEP, EVAL = 1, 6
GX, GY, GS = 560, 600, 740
TX = 1080
T_TARGET, T_FREE, T_RESOLVE = 12, 22, 26


def _load():
    t = np.load(f"{RUN}/rrt_tree.npz"); q = np.load(f"{RUN}/rrt_candidates.npz")
    ti = {str(k): i for i, k in enumerate(t["columns"])}
    ci = {str(k): i for i, k in enumerate(q["columns"])}
    T, C = t["rows"], q["rows"]
    nodes = T[(T[:, ti["step"]] == STEP - 1) & (T[:, ti["eval"]] == EVAL)]
    nodes = nodes[np.argsort(nodes[:, ti["age"]])]
    pos = {int(r[ti["age"]]): (float(r[ti["x"]]), float(r[ti["y"]])) for r in nodes}
    cand = C[(C[:, ci["step"]] == STEP) & (C[:, ci["eval"]] == EVAL)]
    ex = []
    for r in cand:
        ex.append(dict(near=int(r[ci["near_age"]]),
                       target=(float(r[ci["target_x"]]), float(r[ci["target_y"]])),
                       free=(float(r[ci["free_x"]]), float(r[ci["free_y"]])),
                       free_viol=int(r[ci["free_viol"]]),
                       pick=(float(r[ci["pick_x"]]), float(r[ci["pick_y"]])),
                       pick_viol=int(r[ci["pick_viol"]]),
                       n_viol=int(r[ci["n_viol"]]), n_cands=int(r[ci["n_cands"]])))
    return pos, ex


POS, EX = _load()


def frame(i: int, sub: str, killed: int, added: list):
    f = c.new_frame()
    d = ImageDraw.Draw(f)
    g = grid.GridCanvas(GX, GY, GS)

    c.text(d, (GX, 96), "Pruning the search", kind="display", size=64, fill=c.INK, anchor="ma")
    g.draw_grid(d)

    # committed edges so far
    for a, b in added:
        g.draw_edge(d, POS[a], POS[b], grid.TREE_EDGE, width=5)
    for k in sorted(POS):
        if k == 0 or any(b == k for _, b in added):
            g.draw_node(d, *POS[k], c.LAW if k == 0 else grid.TREE_NODE, r=10)

    e = EX[i]
    near = POS[e["near"]]
    if sub in ("target", "free", "resolve"):
        g.draw_cross(d, *e["target"], (150, 154, 160), r=11, width=3)
    if sub in ("free", "resolve"):
        col = c.PLAN if e["free_viol"] else (0, 0x88, 0)
        g.draw_edge(d, near, e["free"], col, width=5, dash=True)
        g.draw_footprint(d, *e["free"], col, width=4, dash=True)
    if sub == "resolve" and e["pick_viol"] != -1:
        g.draw_edge(d, near, e["pick"], grid.TREE_EDGE, width=6)
        g.draw_node(d, *e["pick"], grid.TREE_NODE, r=10)

    # ---- right column
    c.text(d, (TX, 250), "EXPANSION %d of %d" % (i + 1, len(EX)), kind="sans_b", size=28,
           fill=c.FAINT)
    if sub in ("free", "resolve"):
        lab = ("the planner's best stroke\nwould cross the forbidden cell"
               if e["free_viol"] else "the planner's best stroke\nis already legal")
        c.text(d, (TX, 310), lab, kind="roman", size=38,
               fill=c.PLAN if e["free_viol"] else c.INK, spacing=12)
    if sub == "resolve":
        if e["pick_viol"] == -1:
            c.text(d, (TX, 440), "all %d strokes illegal" % e["n_cands"],
                   kind="display", size=52, fill=c.PLAN)
            c.text(d, (TX, 510), "no node added", kind="roman_i", size=38, fill=c.MUTE)
        else:
            c.text(d, (TX, 440), "%d of %d pruned" % (e["n_viol"], e["n_cands"]),
                   kind="display", size=52, fill=c.LAW)
            c.text(d, (TX, 510), "the legal best is taken instead", kind="roman_i",
                   size=36, fill=c.MUTE)
    c.text(d, (TX, 700), "strokes killed by the law", kind="sans_b", size=26, fill=c.FAINT)
    c.text(d, (TX, 740), str(killed), kind="display", size=96, fill=c.GAP_ONTO)

    c.text(d, (TX, 960), "real trace · social agent · task 8→0",
           kind="roman_i", size=26, fill=c.FAINT)
    return f


def build() -> list:
    frames, killed, added = [], 0, []
    for i, e in enumerate(EX):
        for sub, n in (("target", T_TARGET), ("free", T_FREE), ("resolve", T_RESOLVE)):
            if sub == "resolve":
                killed += e["n_viol"]
            tgt = frame(i, sub, killed, list(added))
            frames += [c.blend(frames[-1], tgt, c.ease((k + 1) / 6)) for k in range(6)] if frames \
                else [c.blend(c.new_frame(), tgt, c.ease((k + 1) / 10)) for k in range(10)]
            frames += [tgt] * n
        if e["pick_viol"] != -1:
            added.append((e["near"], max(POS) if False else _age_of(e["pick"])))
    frames += [frames[-1]] * 26
    return frames


def _age_of(pt):
    for k, v in POS.items():
        if abs(v[0] - pt[0]) < 1e-6 and abs(v[1] - pt[1]) < 1e-6:
            return k
    return max(POS)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ("/newdata2/dylantw/tmp/claude-1727/"
        "-newdata2-dylantw-Legislated-Planner-src/8f348aa2-3a2b-4256-9903-13229c0043e0/"
        "scratchpad/beat08_prune.mp4")
    c.write_mp4(build(), out)
