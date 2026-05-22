"""Collect rollouts from IsaacLab DinoWM grid env into dino_wm .pth format.

Writes ${output_dir}/:
  states.pth          (E, T, 75)
  actions_left.pth    (E, T, 7)        actions_right.pth   (E, T, 7)
  proprio_left.pth    (E, T, 18)       proprio_right.pth   (E, T, 18)
  cell_labels.pth     (E, T, 3) int64
  seq_lengths.pth     (E,) int64
  obses/{overhead,front}/episode_NNN.pth   (T, H, W, 3) uint8

Random independent per-agent action policy.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from env.isaaclab.grid_wrapper import ACTION_DIM_PER_AGENT, GridWrapper, PROPRIO_DIM_OWN
from env.isaaclab.grid_metadata import cell_labels_from_states

CAMERAS = ("overhead", "front")
IMG_HW = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-v0")
    ap.add_argument("--num_episodes", type=int, default=100)
    ap.add_argument("--episode_len", type=int, default=120)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--output_dir", default=os.environ.get("DATASET_DIR", "./data") + "/isaaclab_grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0",
                    help="Must be 'cuda:0' for this Isaac Sim build. Mask with CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--cooperative", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--renderer", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    for cam in CAMERAS:
        (out / "obses" / cam).mkdir(parents=True, exist_ok=True)

    env = GridWrapper(
        task_id=args.task_id, num_envs=args.num_envs, device=args.device,
        cooperative=args.cooperative, renderer=args.renderer,
    )
    rng = np.random.RandomState(args.seed)
    T, A, N, P = args.episode_len, ACTION_DIM_PER_AGENT, args.num_envs, PROPRIO_DIM_OWN

    all_states, all_aL, all_aR, all_pL, all_pR, all_cells, all_lens = [], [], [], [], [], [], []
    ep_idx = 0

    while ep_idx < args.num_episodes:
        env.reset()
        ep_states = np.zeros((N, T, env.state_dim), dtype=np.float32)
        ep_aL = np.zeros((N, T, A), dtype=np.float32)
        ep_aR = np.zeros((N, T, A), dtype=np.float32)
        ep_pL = np.zeros((N, T, P), dtype=np.float32)
        ep_pR = np.zeros((N, T, P), dtype=np.float32)
        ep_vis = {cam: np.zeros((N, T, IMG_HW, IMG_HW, 3), dtype=np.uint8) for cam in CAMERAS}

        for t in range(T):
            a_left = rng.uniform(-1.0, 1.0, size=(N, A)).astype(np.float32)
            a_right = rng.uniform(-1.0, 1.0, size=(N, A)).astype(np.float32)
            obs, _, _, info = env.step({"left": a_left, "right": a_right})

            ep_states[:, t] = info["state"]
            ep_aL[:, t] = a_left
            ep_aR[:, t] = a_right
            if env.cooperative:
                # In cooperative mode both keys are the same 36-D vector.
                ep_pL[:, t] = obs["proprio"]["left"][:, :P]
                ep_pR[:, t] = obs["proprio"]["left"][:, P:]
            else:
                ep_pL[:, t] = obs["proprio"]["left"]
                ep_pR[:, t] = obs["proprio"]["right"]
            for cam in CAMERAS:
                ep_vis[cam][:, t] = obs["visual"][cam]

        ep_cells = cell_labels_from_states(ep_states)

        for i in range(N):
            if ep_idx >= args.num_episodes:
                break
            all_states.append(ep_states[i])
            all_aL.append(ep_aL[i])
            all_aR.append(ep_aR[i])
            all_pL.append(ep_pL[i])
            all_pR.append(ep_pR[i])
            all_cells.append(ep_cells[i])
            all_lens.append(T)
            for cam in CAMERAS:
                torch.save(torch.from_numpy(ep_vis[cam][i]), out / "obses" / cam / f"episode_{ep_idx:03d}.pth")
            ep_idx += 1
            print(f"[collect] episode {ep_idx}/{args.num_episodes}")

    torch.save(torch.from_numpy(np.stack(all_states)), out / "states.pth")
    torch.save(torch.from_numpy(np.stack(all_aL)), out / "actions_left.pth")
    torch.save(torch.from_numpy(np.stack(all_aR)), out / "actions_right.pth")
    torch.save(torch.from_numpy(np.stack(all_pL)), out / "proprio_left.pth")
    torch.save(torch.from_numpy(np.stack(all_pR)), out / "proprio_right.pth")
    torch.save(torch.from_numpy(np.stack(all_cells)), out / "cell_labels.pth")
    torch.save(torch.tensor(all_lens, dtype=torch.int64), out / "seq_lengths.pth")
    env.close()
    print(f"[collect] wrote {ep_idx} episodes to {out}")


if __name__ == "__main__":
    main()
