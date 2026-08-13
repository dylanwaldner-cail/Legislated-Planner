"""Generate the GOAL-CELL BANK: one canonical goal state per grid cell, cube placed DEAD-CENTER in
the cell. This is the swappable target bank a POSITIVE obligation retrieves from -- [O]in_cell(k)
(or [O]in_yellow_cell -> {3,5}) looks up the cell's goal image/state and hands it to the planner's
objective as a target (see the obligation->objective bridge). Goal-image-as-subgoal has precedent in
hierarchical latent world models (HWM): a long-horizon target latent steers short-horizon planning.

Placement is EXACT: cube (x,y) = grid_metadata.cell_center(k), zero velocity, canonical parked arm
(same template trick as gen_law_eval_set.py -- median parked joints + a real resting cube z/quat, so
the arm/paddle is in-distribution for the WM). Rendering needs the sim, so run it in the container on
a FREE GPU (not one hosting a live sweep):

    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh scripts/gen_goal_cell_bank.py \
        --out data/goal_cell_bank --template data/isaaclab_stroke_5k

Output (self-describing; the bridge encodes the images with the SAME WM preprocessor as live goals):
    states.pth   (9, 31)  cell k -> its centered goal state (index == cell_id)
    proprio.pth  (9, 18)
    obses/cell_KK.pth  (H, W, 3) uint8   the goal image (cube centered in cell K)
    metadata.json  cell_id -> center xy, geometry, template, provenance
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/goal_cell_bank")
    ap.add_argument("--template", default="data/isaaclab_stroke_5k",
                    help="dataset to borrow the parked-arm + cube z/quat template state from")
    ap.add_argument("--batch", type=int, default=8, help="envs per render pass (<= the RTX ceiling)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from env.isaaclab.grid_venv import GridVectorEnv
    from env.isaaclab import grid_metadata as gm
    from env.isaaclab.app_launcher import close_or_exit

    # Canonical template: a real resting frame (valid cube z/quat) with the ARM overwritten by the
    # median PARKED pose + zero joint velocity. Identical to gen_law_eval_set.py -- see its comment for
    # why the raw states[0,0] arm is an out-of-distribution pre-park transient.
    _tmpl = torch.load(Path(args.template) / "states.pth").float().numpy()   # (E,T,31)
    template = _tmpl[0, 0].copy()
    template[0:9] = np.median(_tmpl.reshape(-1, 31), axis=0)[0:9]            # arm -> canonical parked pose
    template[9:18] = 0.0                                                     # joint velocities -> at rest

    def make_state(xy):
        s = template.copy()
        s[_CUBE_XY] = xy
        s[_CUBE_VEL] = 0.0
        return s

    centers = [gm.cell_center(k) for k in range(gm.N_CELLS)]                 # exact cell centers, index == cell_id
    scen = [make_state(c) for c in centers]
    K = len(scen)

    out = Path(args.out)
    (out / "obses").mkdir(parents=True, exist_ok=True)
    env = GridVectorEnv(num_envs=args.batch, device=args.device)
    seeds = [args.seed] * args.batch
    print(f"[gen] {K} centered goal states (cube at each cell center) -> {out}")

    states = np.zeros((K, 31), dtype=np.float32)
    proprio = np.zeros((K, 18), dtype=np.float32)
    img_hw = 0
    try:
        for b0 in range(0, K, args.batch):
            grp = scen[b0:b0 + args.batch]
            m = len(grp)
            batch_in = np.stack(grp + [grp[-1]] * (args.batch - m))         # pad the last pass to batch
            obs, st = env.prepare(seeds, batch_in)
            vis = np.asarray(obs["visual"]); pro = np.asarray(obs["proprio"])
            img_hw = vis.shape[1]
            for j in range(m):
                k = b0 + j
                states[k] = np.asarray(st[j]); proprio[k] = pro[j]
                torch.save(torch.from_numpy(vis[j].astype(np.uint8)), out / "obses" / f"cell_{k:02d}.pth")
                # sanity: the rendered/settled cube should still be centered in cell k
                got = gm.which_cell(states[k, _CUBE_XY])
                tag = "" if got == k else f"  !! settled into cell {got}, not {k}"
                print(f"  cell {k}: center=({centers[k][0]:+.4f},{centers[k][1]:+.4f}){tag}")
        torch.save(torch.from_numpy(states), out / "states.pth")
        torch.save(torch.from_numpy(proprio), out / "proprio.pth")
        (out / "metadata.json").write_text(json.dumps({
            "bank": "goal_cell_bank",
            "description": "One centered goal state per grid cell (cube at cell_center); the target a "
                           "positive obligation [O]in_cell(k) retrieves for the planner objective.",
            "index_is_cell_id": True, "n_cells": gm.N_CELLS,
            "cell_centers": {str(k): list(map(float, centers[k])) for k in range(gm.N_CELLS)},
            "cube_half": CUBE_HALF, "cell": gm.CELL, "grid_half": gm.GRID_HALF,
            "template": args.template, "seed": args.seed, "img_hw": int(img_hw),
            "layout": "states.pth (9,31)[cell], proprio.pth (9,18), obses/cell_KK.pth (H,W,3) uint8",
        }, indent=2))
        print(f"[gen] wrote goal-cell bank to {out}/")
        provenance.write(out, __file__, args=args, repo=_REPO)
    finally:
        close_or_exit(env)


if __name__ == "__main__":
    main()
