"""Replay executed episodes in the simulator to recover the CUBE ORIENTATION per frame.

WHY. Law abidance is scored with `probe_cube_cells.swept_cells`, which models the cube as an
AXIS-ALIGNED square of half-width CUBE_HALF (0.045 m). The cube is never axis-aligned: it spawns at
23.56 deg off-axis in every law_eval scenario, and it rotates while being pushed (16.5 deg per step
on average in the training data, p90 33 deg). At 45 deg the true axis-aligned extent is
0.045*sqrt(2) = 0.0636 m, 41% larger than the model. The error is one-directional -- too small a body
UNDER-detects cell entry -- so reported abidance is an upper bound on true compliance.

The per-frame orientation is not in any saved artefact: `cube_xy_frames` holds x,y only, and the
eval `states.pth` holds just the init and goal poses. But `executed_actions.npy` IS saved, so the
episodes can be re-simulated exactly and the quaternion read off. That is what this does.

`GridWrapperSingle.rollout` already returns the full (N, T+1, 31) state including the cube quaternion
at offsets 21..24, so nothing in the environment needs changing.

DETERMINISM IS CHECKED, NOT ASSUMED. For every replayed batch the script compares the replayed cube
(x,y) against the stored `cube_xy_frames` and reports the max absolute deviation. If the replay does
not reproduce the original trajectory the recovered orientations describe a different rollout and are
worthless -- so run with --limit 1 first and read that number before committing to the full sweep.

The sign colour is NOT reproduced: it is a shader property, not physics state (grid_venv.py:44), so
it cannot affect the trajectory. Sign verdicts continue to come from the recorded ledger.

Run INSIDE the IsaacLab container, via isaaclab.sh (not the raw kit python, or `yaml` fails to
import):

  docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
      /workspace/dino_wm/scripts/replay_orientation.py --limit 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/workspace/dino_wm")

CUBE_O = 18          # STATE_FIRST_CUBE_OFFSET_SINGLE
DEFAULT_RUN = "/workspace/dino_wm/results/aug20/sign_change"
DEFAULT_OUT = "/workspace/dino_wm/results/aug20/sign_change_replay"


def batch_dirs(run, agents):
    for a in agents:
        for p in sorted(glob.glob(f"{run}/{a}/*/batch_*")):
            if os.path.exists(f"{p}/executed_actions.npy") and \
               os.path.exists(f"{p}/eval_metrics.json"):
                yield a, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--agents", default="off,social,deviant")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="stop after N batches (0 = all)")
    args = ap.parse_args()

    agents = [a for a in args.agents.split(",") if a]
    todo = list(batch_dirs(args.run, agents))
    if args.limit:
        todo = todo[:args.limit]
    print(f"[replay] {len(todo)} batches from {args.run} -> {args.out}", flush=True)

    from env.isaaclab.grid_venv import GridVectorEnv
    env = None
    worst_overall, done, t0 = 0.0, 0, time.time()

    for k, (agent, bdir) in enumerate(todo):
        d = json.load(open(f"{bdir}/eval_metrics.json"))
        acts = np.load(f"{bdir}/executed_actions.npy")           # (N, T, 4)
        init = np.asarray(d["init_state"], dtype=np.float32)     # (N, 31)
        n = int(d["n_evals"])
        if env is None:
            env = GridVectorEnv(num_envs=n, device=args.device)
            print(f"[replay] env up (num_envs={n}, {args.device})", flush=True)
        elif len(env) != n:
            print(f"[replay] SKIP {bdir}: n_evals={n} != env {len(env)}")
            continue

        _obs, states = env.rollout([d.get("seed")], init, acts)   # states (N, T+1, 31)
        states = np.asarray(states)

        # ---- determinism check against the stored ground-truth xy ----
        stored = d.get("cube_xy_frames", [])
        worst = 0.0
        for i, P in enumerate(stored):
            P = np.asarray(P, dtype=float)
            Q = states[i, :P.shape[0], CUBE_O:CUBE_O + 2]
            if Q.shape[0] == P.shape[0]:
                worst = max(worst, float(np.abs(Q - P).max()))
        worst_overall = max(worst_overall, worst)

        rel = os.path.relpath(bdir, args.run)
        dst = os.path.join(args.out, rel)
        os.makedirs(dst, exist_ok=True)
        np.savez_compressed(
            os.path.join(dst, "replay_states.npz"),
            states=states.astype(np.float32),
            cube_xy=states[..., CUBE_O:CUBE_O + 2].astype(np.float32),
            cube_quat=states[..., CUBE_O + 3:CUBE_O + 7].astype(np.float32),
            n_steps=np.asarray(d.get("n_steps", [])),
            max_xy_dev=np.float32(worst),
        )
        done += 1
        el = time.time() - t0
        print(f"[replay] {done}/{len(todo)} {agent}/{rel}  max|dxy|={worst:.6f} m  "
              f"({el/done:.1f}s/batch, {el/60:.1f} min elapsed)", flush=True)

    print(f"\n[replay] DONE {done} batches. WORST xy deviation vs stored: {worst_overall:.6f} m")
    print("[replay] a faithful replay should be ~1e-4 m or less; anything larger means the "
          "recovered orientations belong to a DIFFERENT trajectory and must not be used.")


if __name__ == "__main__":
    main()
