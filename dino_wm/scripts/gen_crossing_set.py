"""Generate a GREEN->GREEN CROSSING benchmark: controlled (init, goal) scenarios where the cube
starts in a green cell and must reach a green cell on the OPPOSITE side, so its route ALWAYS crosses
the middle (yellow/red) row -- there is no all-green path. Sibling of gen_law_eval_set.py; same
output format, so it plugs straight into `eval_sweep --law_eval`.

The grid colours (env/isaaclab/grid_metadata.CELL_COLORS) are: green top & bottom rows, and a middle
row of yellow | RED | yellow. Green cells therefore live only in rows 0 and 2, so every (top green ->
bottom green) or (bottom green -> top green) pair forces a crossing of the middle row. WHICH middle
cell the straight route transits is the crossing severity -- yellow (side columns 3/5) vs red (centre
4) -- recorded per pair in metadata as `crosses`/`crosses_color` so violation levels can be colour-
coded downstream (grading itself is intentionally NOT done here).

Default set: all 6 green cells as starts, each paired with the 3 green cells in the opposite row ->
18 ordered pairs, n_var x n_var position variants per pair (centroid jittered in-cell, footprint kept
clear of the red centre). n_var=3 -> 9/pair -> 162 scenarios.

    # run in the container (needs the sim to render); pick a free GPU
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh scripts/gen_crossing_set.py --out data/crossing_set --n_var 3

Output (self-describing, identical layout to gen_law_eval_set):
    <init>_<goal>/states.pth   (V, 2, 31)  [:,0]=init [:,1]=goal
    <init>_<goal>/proprio.pth  (V, 2, 18)
    <init>_<goal>/obses/scene_NNNNN.pth  (2, H, W, 3) uint8   [init, goal] images
    <init>_<goal>/metadata.json  +  top-level metadata.json (benchmark spec)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import provenance

CUBE_HALF = 0.045                                  # matches probes.probe_cube_cells.CUBE_HALF
_CUBE_XY = slice(18, 20)
_CUBE_VEL = slice(25, 31)                          # lin+ang vel -> zero for a statically-placed cube


def _green_cross_pairs(gm):
    """All ordered (start, goal) pairs of GREEN cells in OPPOSITE green rows. Green cells occupy only
    the top and bottom rows, so every such pair's straight route crosses the middle (yellow/red) row.
    Returns [(start, goal), ...] sorted; 6 greens x 3 opposite-row greens = 18 ordered pairs."""
    green = [r * gm.N_COLS + c for r, row in enumerate(gm.CELL_COLORS)
             for c, col in enumerate(row) if col == "green"]
    row_of = {cell: cell // gm.N_COLS for cell in green}
    return sorted((s, g) for s in green for g in green if row_of[s] != row_of[g])


def _crossed_cell(ic, gc, gm):
    """The middle-row cell the straight init->goal route transits (via the segment midpoint), plus
    its colour -- the crossing's severity label. e.g. 0->6 crosses 3 (yellow), 1->7 crosses 4 (red)."""
    ix, iy = gm.cell_center(ic)
    gx, gy = gm.cell_center(gc)
    mid = np.array([[0.5 * (ix + gx), 0.5 * (iy + gy)]], dtype=np.float32)
    c = int(np.atleast_1d(gm.which_cell(mid))[0])
    r, col = c // gm.N_COLS, c % gm.N_COLS
    return c, gm.CELL_COLORS[r][col]


def _sample_xy(cell, gm, rng, clear_cell, jitter_margin=0.012):
    """Cube center jittered within `cell`, with the FOOTPRINT clamped clear of `clear_cell` (the red
    centre) so a green endpoint can't graze it. Mirrors gen_law_eval_set._sample_xy."""
    cx, cy = gm.cell_center(cell)
    icx, icy = gm.cell_center(clear_cell)
    h = gm.CELL / 2.0 - jitter_margin
    x = cx + rng.uniform(-h, h)
    y = cy + rng.uniform(-h, h)
    clr = gm.CELL / 2.0 + CUBE_HALF                # min center-gap for footprints to NOT overlap
    if abs(x - icx) < clr and abs(y - icy) < clr:  # footprint would touch the clear cell -> push out
        if abs(cx - icx) >= abs(cy - icy):         # clamp along the axis this cell is offset on
            x = icx + np.sign(cx - icx or 1.0) * clr
        else:
            y = icy + np.sign(cy - icy or 1.0) * clr
    return float(x), float(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/crossing_set")
    ap.add_argument("--template", default="data/isaaclab_stroke_1500",
                    help="dataset to borrow the parked-arm + cube z/quat template state from")
    ap.add_argument("--n_var", type=int, default=3, help="position variants per cell (n_var init x n_var goal per pair)")
    ap.add_argument("--clear_cell", type=int, default=4, help="keep endpoint footprints clear of this cell (red centre)")
    ap.add_argument("--batch", type=int, default=8, help="envs per render pass (<= the RTX ceiling)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from env.isaaclab.grid_venv import GridVectorEnv
    from env.isaaclab import grid_metadata as gm
    from env.isaaclab.app_launcher import close_or_exit

    pairs = _green_cross_pairs(gm)
    rng = np.random.RandomState(args.seed)
    # Template: a real resting frame (valid cube z/quat) with the ARM overwritten by the CANONICAL
    # PARKED pose (states[0,0]'s arm is a pre-park transient -> OOD render). See gen_law_eval_set.
    _tmpl = torch.load(Path(args.template) / "states.pth").float().numpy()   # (E,T,31)
    template = _tmpl[0, 0].copy()                                            # valid resting cube z/quat
    template[0:9] = np.median(_tmpl.reshape(-1, 31), axis=0)[0:9]            # arm -> canonical parked pose
    template[9:18] = 0.0                                                     # joint velocities -> at rest

    def make_state(xy):
        s = template.copy()
        s[_CUBE_XY] = xy
        s[_CUBE_VEL] = 0.0
        return s

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = GridVectorEnv(num_envs=args.batch, device=args.device)
    seeds = [args.seed] * args.batch
    n_per_pair = args.n_var * args.n_var
    print(f"[gen] {len(pairs)} green->green opposite-row pairs x {args.n_var}x{args.n_var} = "
          f"{len(pairs)*n_per_pair} scenarios -> one subdir per pair under {out}")

    img_hw = 0
    pair_dirs, pair_meta = [], []
    try:
        for (ic, gc) in pairs:
            crossed, crossed_color = _crossed_cell(ic, gc, gm)
            scen = []
            for vi in range(args.n_var):
                ixy = _sample_xy(ic, gm, rng, args.clear_cell)
                for vg in range(args.n_var):
                    gxy = _sample_xy(gc, gm, rng, args.clear_cell)
                    scen.append((make_state(ixy), make_state(gxy),
                                 {"init_cell": ic, "goal_cell": gc, "init_var": vi, "goal_var": vg,
                                  "init_xy": list(ixy), "goal_xy": list(gxy),
                                  "crosses": crossed, "crosses_color": crossed_color}))
            V = len(scen)
            pdir = out / f"{ic}_{gc}"                                   # per-pair subdir
            (pdir / "obses").mkdir(parents=True, exist_ok=True)
            states = np.zeros((V, 2, 31), dtype=np.float32)
            proprio = np.zeros((V, 2, 18), dtype=np.float32)
            for b0 in range(0, V, args.batch):
                grp = scen[b0:b0 + args.batch]
                m = len(grp)
                init_b = np.stack([g[0] for g in grp] + [grp[-1][0]] * (args.batch - m))   # pad to batch
                goal_b = np.stack([g[1] for g in grp] + [grp[-1][1]] * (args.batch - m))
                obs_i, st_i = env.prepare(seeds, init_b)
                obs_g, st_g = env.prepare(seeds, goal_b)
                vis_i = np.asarray(obs_i["visual"]); vis_g = np.asarray(obs_g["visual"])
                pro_i = np.asarray(obs_i["proprio"]); pro_g = np.asarray(obs_g["proprio"])
                img_hw = vis_i.shape[1]
                for j in range(m):
                    idx = b0 + j
                    states[idx, 0] = np.asarray(st_i[j]); states[idx, 1] = np.asarray(st_g[j])
                    proprio[idx, 0] = pro_i[j]; proprio[idx, 1] = pro_g[j]
                    torch.save(torch.from_numpy(np.stack([vis_i[j], vis_g[j]]).astype(np.uint8)),
                               pdir / "obses" / f"scene_{idx:05d}.pth")
            torch.save(torch.from_numpy(states), pdir / "states.pth")
            torch.save(torch.from_numpy(proprio), pdir / "proprio.pth")
            (pdir / "metadata.json").write_text(json.dumps({
                "pair": [ic, gc], "init_cell": ic, "goal_cell": gc, "n_scenarios": V,
                "crosses": crossed, "crosses_color": crossed_color, "metric_cell": crossed,
                "scenarios": [g[2] for g in scen],
            }, indent=2))
            pair_dirs.append(f"{ic}_{gc}")
            pair_meta.append({"pair": [ic, gc], "crosses": crossed, "crosses_color": crossed_color})
            print(f"  pair {ic}->{gc}: {V} scenarios, crosses cell {crossed} ({crossed_color}) -> {pdir}")

        (out / "metadata.json").write_text(json.dumps({           # top-level benchmark manifest
            "benchmark": "crossing_set",
            "description": "Green->green cube scenarios whose route ALWAYS crosses the middle "
                           "(yellow/red) row; the crossed cell's colour is the crossing severity. "
                           "One subdir per init->goal pair. Violation grading is done downstream.",
            "pairs": pairs, "pair_dirs": pair_dirs, "pair_crossings": pair_meta, "n_var": args.n_var,
            "n_scenarios_per_pair": n_per_pair, "n_scenarios_total": len(pairs) * n_per_pair,
            "clear_cell": args.clear_cell, "cell_colors": gm.CELL_COLORS,
            "cube_half": CUBE_HALF, "cell": gm.CELL, "grid_half": gm.GRID_HALF, "n_cells": gm.N_CELLS,
            "placement": "centroid jittered in-cell, footprint clamped clear of the red centre",
            "template": args.template, "seed": args.seed, "img_hw": int(img_hw),
            "layout": "<init>_<goal>/{states.pth (V,2,31)[init,goal], proprio.pth (V,2,18), obses/scene_NNNNN.pth (2,H,W,3)}",
            "eval": "per pair: plan.py goal_source=law_eval goal_file_path=<out>/<init>_<goal> (batched via scene_offset)",
        }, indent=2))
        print(f"[gen] wrote benchmark to {out}/  ({len(pairs)} pair subdirs + metadata.json)")
        provenance.write(out, __file__, args=args, repo=_REPO)
    finally:
        close_or_exit(env)


if __name__ == "__main__":
    main()
