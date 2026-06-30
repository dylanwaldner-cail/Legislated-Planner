"""Reach-calibration harness for the single-robot grid (planar-stroke task).

Answers: "can the arm push the cube from EVERY part of the grid?" For each of the
9 cell centers it teleports the cube there and attempts a contact stroke (start
behind the cube, push through it) in N evenly-spaced directions, then reports per
(cell, direction):
    reached_push : the executor got to the PUSH phase (vs stalling in approach,
                   which means it couldn't get behind the cube = UNREACHABLE)
    moved        : the cube actually advanced along the push direction (> thresh)

The N directions are batched across num_envs (one direction per env) via the
vectorized StrokeExecutor, so each cell is a single execute_stroke. Only STATES
are read (cube/EE come from physics, not the camera), so run it cheap:
    python scripts/reach_calibration.py --render_mode RaytracedLighting --spp 4

Workflow: run it on the current geometry for a baseline (the near column will
fail), then edit the robot base x / GRID_HALF / home pose / REACH_X_MIN and
re-run until every cell passes from all directions. GRID_HALF must be changed in
BOTH env/isaaclab/grid_metadata.py AND IsaacLab/.../dinowm_grid/grid_assets.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from env.isaaclab.grid_wrapper_single import GridWrapperSingle
from env.isaaclab.grid_metadata import GRID_HALF, N_CELLS, cell_center, cell_id_to_label
from env.isaaclab.app_launcher import close_or_exit

_SPAWN_Z = 0.046  # 9cm cube center resting on the table (matches grid_wrapper_single)


_EDGES = (  # (label, outward unit vector); the cube is placed past this edge
    ("+x(away from base)", np.array([1.0, 0.0], np.float32)),
    ("-x(toward base)",    np.array([-1.0, 0.0], np.float32)),
    ("+y",                 np.array([0.0, 1.0], np.float32)),
    ("-y",                 np.array([0.0, -1.0], np.float32)),
)


def run_offgrid(args):
    """Recovery test: place the cube `dist` past each edge and push it back toward
    center. Reports per (edge, dist) whether the arm got behind it (reached PUSH)
    and moved it inward -> how far past each edge the arm can recover a cube."""
    dists = list(args.offgrid_dists)
    combos = [(lbl, e, d) for (lbl, e) in _EDGES for d in dists]
    N = len(combos)
    env = GridWrapperSingle(
        num_envs=N, device=args.device, render_mode=args.render_mode, spp=args.spp,
        stroke_max_steps=args.stroke_max_steps,
    )
    print(f"[offgrid] GRID_HALF={GRID_HALF:.4f}  edges x dists = {len(_EDGES)}x{len(dists)}={N}")
    arm = env.reset()[1][:, :18].copy()

    # cube placed at edge_unit*(GRID_HALF+dist); recovery pushes toward center.
    cube0 = np.stack([e * (GRID_HALF + d) for (_, e, d) in combos]).astype(np.float32)  # (N,2)
    cube_block = np.zeros((N, 13), np.float32)
    cube_block[:, 0:2] = cube0
    cube_block[:, 2] = _SPAWN_Z
    cube_block[:, 3] = 1.0  # quat w
    env.set_init_state(np.concatenate([arm, cube_block], axis=1))

    cstart = env.get_cube_positions()[:, :2].astype(np.float32)         # (N,2) actual placed pos
    in_dir = np.stack([-e for (_, e, _) in combos]).astype(np.float32)  # toward center
    # start BEHIND the cube (further out), push well past the edge back onto the grid
    back = np.stack([e * args.back for (_, e, _) in combos]).astype(np.float32)
    start = cstart + back
    end = cstart + in_dir * (np.array([d for (_, _, d) in combos], np.float32)[:, None] + 0.14)
    strokes = np.concatenate([start, end], axis=1).astype(np.float32)

    _, state_after = env.execute_stroke(strokes)
    reached = (env._executor.phase_idx == 2)
    inward = np.einsum("nd,nd->n", state_after[:, 18:20] - cstart, in_dir)  # progress toward center
    recovered = reached & (inward > args.move_thresh)

    print(f"\n=== off-grid recovery (cube placed past edge, pushed back) ===")
    per_edge_max = {}
    for (lbl, _e, d), rc, rch, inw in zip(combos, recovered, reached, inward):
        tag = "RECOVERED" if rc else ("reached but no inward move" if rch else "UNREACHABLE (no get-behind)")
        print(f"  {lbl:20s} +{d:.2f}m past edge: {tag}  (inward move {inw:+.3f}m)")
        if rc:
            per_edge_max[lbl] = max(per_edge_max.get(lbl, 0.0), d)
    print("\n=== max recoverable distance past each edge ===")
    for lbl, _e in _EDGES:
        m = per_edge_max.get(lbl, None)
        print(f"  {lbl:20s}: {('up to +%.2fm' % m) if m is not None else 'NONE recovered'}")
    safe = min(per_edge_max.values()) if len(per_edge_max) == len(_EDGES) else 0.0
    print(f"\nSuggested start_margin (min across edges, the limiting +x usually): {safe:.2f}m")
    print("Set collect --start_margin to ~this so the sampler only commands reachable contact points.")
    close_or_exit(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_dirs", type=int, default=8, help="push directions per cell (also num_envs)")
    ap.add_argument("--back", type=float, default=0.10, help="stroke start this far behind the cube")
    ap.add_argument("--push", type=float, default=0.14, help="stroke length through the cube")
    ap.add_argument("--move_thresh", type=float, default=0.02, help="min forward cube move to count as pushed (m)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render_mode", default="RaytracedLighting", choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--spp", type=int, default=4, help="images are unused here; keep low/fast")
    ap.add_argument("--stroke_max_steps", type=int, default=240)
    ap.add_argument("--repeats", type=int, default=1,
                    help="run the full sweep this many times; report per-(cell,dir) pass "
                    "RATE across repeats to separate consistent failures from physics noise.")
    ap.add_argument("--offgrid", action="store_true",
                    help="Instead of the 9-cell sweep, test RECOVERY: teleport the cube to "
                    "several distances PAST each edge and try to push it back onto the grid. "
                    "Tells us how far past each edge the arm can reach (sets start_margin).")
    ap.add_argument("--offgrid_dists", type=float, nargs="+", default=[0.04, 0.08, 0.12, 0.16],
                    help="distances past the grid edge (m) to place the cube for --offgrid.")
    args = ap.parse_args()

    if args.offgrid:
        run_offgrid(args)
        return

    N = args.num_dirs
    thetas = np.array([2 * np.pi * k / N for k in range(N)], np.float32)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1).astype(np.float32)  # (N,2)

    env = GridWrapperSingle(
        num_envs=N, device=args.device, render_mode=args.render_mode, spp=args.spp,
        stroke_max_steps=args.stroke_max_steps,
    )
    print(f"[reach] GRID_HALF={GRID_HALF:.4f}  num_dirs={N}  back={args.back} push={args.push}")

    # Home-pose arm state to reuse for every teleport (cube swapped per cell).
    template = env.reset()[1]            # (N, 31)
    arm = template[:, :18].copy()

    R = args.repeats
    # passes[cell] = (N,) int count of how many of the R repeats passed each direction.
    passes = {cell: np.zeros(N, np.int64) for cell in range(N_CELLS)}
    try:
        for rep in range(R):
            for cell in range(N_CELLS):
                cx, cy = cell_center(cell)
                cube_block = np.tile(
                    np.array([cx, cy, _SPAWN_Z, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.float32), (N, 1)
                )
                state = np.concatenate([arm, cube_block], axis=1)  # (N,31)
                env.set_init_state(state)

                cube0 = env.get_cube_positions()[:, :2].astype(np.float32)        # (N,2) all == (cx,cy)
                start = cube0 - dirs * args.back
                end = cube0 + dirs * args.push
                strokes = np.concatenate([start, end], axis=1).astype(np.float32)  # (N,4)

                obs, state_after = env.execute_stroke(strokes)
                reached = (env._executor.phase_idx == 2)                          # (N,) got to PUSH
                fwd = np.einsum("nd,nd->n", state_after[:, 18:20] - cube0, dirs)   # signed progress
                ok = reached & (fwd > args.move_thresh)
                passes[cell] += ok.astype(np.int64)
            print(f"[reach] repeat {rep+1}/{R} done")

        # ---- report: pass RATE across repeats per (cell, dir) ----
        print(f"\n=== per-cell reachability across {R} repeat(s) (pass/{R} per direction) ===")
        mean_rate = {}
        consistent_fail = []   # (cell, deg): 0/R every repeat -> genuine unreachable
        noisy = []             # (cell, deg): 1..R-1 /R -> marginal / physics noise
        for cell in range(N_CELLS):
            label, r = cell_id_to_label(cell)
            cnt = passes[cell]
            mean_rate[cell] = float(cnt.sum()) / (N * R)
            bad = []
            for i in range(N):
                if cnt[i] < R:
                    deg = int(round(np.degrees(thetas[i])))
                    bad.append(f"{deg:+d}:{cnt[i]}/{R}")
                    (consistent_fail if cnt[i] == 0 else noisy).append((cell, deg))
            cxy = cell_center(cell)
            print(f"  cell {cell} ({label},row{r}) @({cxy[0]:+.3f},{cxy[1]:+.3f}): "
                  f"mean {cnt.sum()}/{N*R}" + (f"  | <full: {', '.join(bad)}" if bad else "  | all dirs reliable"))

        print(f"\n=== mean pass-rate grid (cell_id=row*3+col; rows front->back) ===")
        for r in range(3):
            print("  " + "  ".join(f"{mean_rate[r*3+c]:.2f}" for c in range(3)))

        print(f"\nCONSISTENT failures (0/{R} -> genuine reach limit): "
              f"{consistent_fail if consistent_fail else 'none'}")
        print(f"NOISY/marginal (sometimes pass -> physics noise, not a hard limit): "
              f"{noisy if noisy else 'none'}")
    finally:
        close_or_exit(env)


if __name__ == "__main__":
    main()
