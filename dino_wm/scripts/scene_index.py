"""Index the dataset by the cube's cell + colour, to select scenarios for specific law tests.

Tags every frame with the cube's CENTER cell, that cell's colour (from CELL_COLORS:
yellow={3,5}, red={4}, green={0,1,2,6,7,8}), and per-cell occupancy. Lets you (a) see how many
frames/episodes meet a condition and (b) pull ready-to-use (episode, init_frame, goal_frame)
tuples for a specific law test -- e.g. goal in a yellow cell, init in green.

States-only (no probe / no sim).

    python scripts/scene_index.py --data_dir data/isaaclab_stroke_1500                  # summary
    python scripts/scene_index.py --goal_H 5 --goal_color yellow --init_color green --dump scenarios.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
# load grid_metadata by file path so we don't trigger env/isaaclab/__init__ (-> no IsaacLab import)
_spec = importlib.util.spec_from_file_location("grid_metadata", _REPO / "env" / "isaaclab" / "grid_metadata.py")
gm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gm)

CUBE_OFF = gm.STATE_FIRST_CUBE_OFFSET_SINGLE  # 18
CUBE_HALF = 0.045
COLOR_BY_CELL = tuple(gm.CELL_COLORS[c // gm.N_COLS][c % gm.N_COLS] for c in range(gm.N_CELLS))


def cell_color(cell_id):
    return "off-grid" if cell_id == gm.OFF_GRID else COLOR_BY_CELL[int(cell_id)]


def _occupancy(xy, cube_half=CUBE_HALF):
    """(...,2) cube centers -> (...,N_CELLS) bool occupancy (AABB overlap)."""
    centers = np.array([gm.cell_center(c) for c in range(gm.N_CELLS)], np.float32)   # (9,2)
    H = gm.CELL / 2.0 + cube_half
    d = np.abs(xy[..., None, :] - centers)                                           # (...,9,2)
    return (d[..., 0] < H) & (d[..., 1] < H)                                          # (...,9)


def _seg_aabb_hit(p0, p1, lo, hi):
    """Does the 2D segment p0->p1 intersect the axis-aligned box [lo,hi]? (Liang-Barsky)."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            ta, tb = (lo[i] - p0[i]) / d[i], (hi[i] - p0[i]) / d[i]
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


def path_through(c0, c1, cell, cube_half=CUBE_HALF):
    """Does the cube footprint sweep `cell` along the straight init->goal segment c0->c1?
    (the 'the direct route crosses this cell' condition — e.g. via_cell=4 for the center)."""
    cx, cy = gm.cell_center(cell)
    H = gm.CELL / 2.0 + cube_half
    return _seg_aabb_hit(np.asarray(c0, float), np.asarray(c1, float),
                         np.array([cx - H, cy - H]), np.array([cx + H, cy + H]))


def build_index_from_states(states, seq, cube_half=CUBE_HALF):
    """states (E,T,31), seq (E,) -> index dict (cube cell + occupancy per frame). Use this when
    you already hold the states (e.g. plan.py's valid SPLIT) so the episode indices line up."""
    states = np.asarray(states)
    seq = np.asarray(seq).astype(int)
    xy = states[..., CUBE_OFF:CUBE_OFF + 2]                                           # (E,T,2)
    return {"xy": xy, "cell": gm.which_cell(xy), "occ": _occupancy(xy, cube_half),
            "seq": seq, "E": states.shape[0]}


def build_index(data_dir, cube_half=CUBE_HALF):
    p = Path(data_dir)
    states = torch.load(p / "states.pth").float().numpy()                            # (E,T,31)
    seq = torch.load(p / "seq_lengths.pth").numpy()
    return build_index_from_states(states, seq, cube_half)


def frames_with_color(idx, color, mode="center"):
    """(e,f) where the cube's CENTER cell (mode='center') or ANY occupied cell ('occupied')
    has `color`. Respects per-episode seq lengths."""
    hits = []
    for e in range(idx["E"]):
        for f in range(int(idx["seq"][e])):
            if mode == "center":
                ok = cell_color(idx["cell"][e, f]) == color
            else:
                ok = any(COLOR_BY_CELL[c] == color for c in np.where(idx["occ"][e, f])[0])
            if ok:
                hits.append((e, f))
    return hits


def goal_pairs(idx, goal_H, init_color=None, goal_color=None, init_cell=None, goal_cell=None,
               via_cell=None, goal_not_in_cells=None, cube_half=CUBE_HALF, require_move=True):
    """(e, init_f, goal_f=init_f+goal_H) in one episode, filtered by init/goal cell colour or id,
    and/or whether the straight init->goal path crosses `via_cell`. require_move drops pairs
    whose center cell doesn't change.

    goal_not_in_cells: drop pairs whose GOAL-cube FOOTPRINT overlaps any listed cell (not just its
    centroid). Use to exclude goals that sit partly in a forbidden cell -- otherwise closing
    euclidean distance to such a goal drags the cube INTO that cell, unfairly penalising the law."""
    _excl = {int(c) for c in goal_not_in_cells} if goal_not_in_cells else set()
    out = []
    for e in range(idx["E"]):
        T = int(idx["seq"][e])
        for f0 in range(T - goal_H):
            ic, gc = int(idx["cell"][e, f0]), int(idx["cell"][e, f0 + goal_H])
            if require_move and ic == gc:
                continue
            if init_color and cell_color(ic) != init_color:
                continue
            if goal_color and cell_color(gc) != goal_color:
                continue
            if init_cell is not None and ic != init_cell:
                continue
            if goal_cell is not None and gc != goal_cell:
                continue
            if via_cell is not None and not path_through(idx["xy"][e, f0], idx["xy"][e, f0 + goal_H],
                                                         via_cell, cube_half):
                continue
            if _excl and any(bool(idx["occ"][e, f0 + goal_H, c]) for c in _excl):
                continue                                         # goal footprint overlaps an excluded cell
            out.append((e, f0, f0 + goal_H))
    return out


def select_pairs(data_dir, goal_H, cube_half=CUBE_HALF, **filters):
    """Convenience: build the index from `data_dir` and return the filtered goal pairs."""
    return goal_pairs(build_index(data_dir, cube_half), goal_H, cube_half=cube_half, **filters)


def select_pairs_from_states(states, seq, goal_H, cube_half=CUBE_HALF, **filters):
    """Like select_pairs but from in-memory states (e.g. plan.py's valid split) so the returned
    episode indices match the caller's dataset indexing."""
    return goal_pairs(build_index_from_states(states, seq, cube_half), goal_H, cube_half=cube_half, **filters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--cube_half", type=float, default=CUBE_HALF)
    ap.add_argument("--goal_H", type=int, default=None, help="if set, list (init,goal) pairs goal_H apart")
    ap.add_argument("--init_color", default=None)
    ap.add_argument("--goal_color", default=None)
    ap.add_argument("--init_cell", type=int, default=None)
    ap.add_argument("--goal_cell", type=int, default=None)
    ap.add_argument("--via_cell", type=int, default=None, help="straight init->goal path must cross this cell (e.g. 4=center)")
    ap.add_argument("--goal_not_in_cells", type=int, nargs="+", default=None,
                    help="drop pairs whose GOAL-cube footprint overlaps any of these cells (e.g. --goal_not_in_cells 4)")
    ap.add_argument("--dump", default=None, help="write matching {episode, init_frame, goal_frame} to JSON")
    args = ap.parse_args()

    idx = build_index(args.data_dir, args.cube_half)
    ccol, ncells = Counter(), Counter()
    for e in range(idx["E"]):
        for f in range(int(idx["seq"][e])):
            ccol[cell_color(idx["cell"][e, f])] += 1
            ncells[int(idx["occ"][e, f].sum())] += 1
    total = sum(ccol.values())
    print(f"[scene index] {idx['E']} episodes, {total} valid frames")
    print(f"  cell colours: yellow={[c for c in range(gm.N_CELLS) if COLOR_BY_CELL[c]=='yellow']} "
          f"red={[c for c in range(gm.N_CELLS) if COLOR_BY_CELL[c]=='red']} "
          f"green={[c for c in range(gm.N_CELLS) if COLOR_BY_CELL[c]=='green']}")
    print("  cube center-cell colour:", {k: f"{v} ({100*v/total:.0f}%)" for k, v in ccol.most_common()})
    print("  cells occupied/frame:", dict(sorted(ncells.items())))

    if args.goal_H is not None:
        pairs = goal_pairs(idx, args.goal_H, args.init_color, args.goal_color,
                           args.init_cell, args.goal_cell, via_cell=args.via_cell,
                           goal_not_in_cells=args.goal_not_in_cells, cube_half=args.cube_half)
        eps = sorted({e for e, _, _ in pairs})
        print(f"\n[goal pairs] goal_H={args.goal_H} init={args.init_color or args.init_cell} "
              f"goal={args.goal_color or args.goal_cell}: {len(pairs)} pairs across {len(eps)} episodes")
        for e, f0, fg in pairs[:12]:
            print(f"  ep {e}: init f{f0} (cell {int(idx['cell'][e,f0])} {cell_color(idx['cell'][e,f0])}) "
                  f"-> goal f{fg} (cell {int(idx['cell'][e,fg])} {cell_color(idx['cell'][e,fg])})")
        if args.dump:
            with open(args.dump, "w") as fp:
                json.dump([{"episode": e, "init_frame": f0, "goal_frame": fg} for e, f0, fg in pairs], fp, indent=2)
            print(f"  [dumped] {len(pairs)} scenarios -> {args.dump}")


if __name__ == "__main__":
    main()
