"""Collect rollouts from the IsaacLab stub env into dino_wm's .pth format.

Writes:
  ${output_dir}/states.pth         (num_eps, T_max, state_dim) float32
  ${output_dir}/actions.pth        (num_eps, T_max, action_dim) float32
  ${output_dir}/seq_lengths.pth    (num_eps,) int64
  ${output_dir}/obses/episode_NNN.pth   (T_max, H, W, 3) uint8

Random-action policy by default. Matches PointMazeDataset's on-disk layout
(see datasets/point_maze_dset.py:14-22).
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

from env.isaaclab.isaaclab_wrapper import ACTION_DIM, IsaacLabWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMStub-v0")
    ap.add_argument("--num_episodes", type=int, default=100)
    ap.add_argument("--episode_len", type=int, default=120)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--output_dir", default=os.environ.get("DATASET_DIR", "./data") + "/isaaclab_stub")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "CUDA device string. Must be 'cuda:0' for this Isaac Sim build "
            "(USDRT scenegraph only supports cuda:0). To use a non-zero "
            "physical GPU, mask it via CUDA_VISIBLE_DEVICES=<idx> and keep "
            "--device cuda:0."
        ),
    )
    ap.add_argument(
        "--renderer",
        default=None,
        help=(
            "Override Omniverse render delegate. Leave unset to use the "
            "default (RTX, requires driver >= 535.129). Pass 'pxr' to try "
            "Hydra Storm (OpenGL-based, no RTX driver requirement) when the "
            "host driver is too old for RTX."
        ),
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    (out / "obses").mkdir(parents=True, exist_ok=True)

    env = IsaacLabWrapper(
        task_id=args.task_id,
        num_envs=args.num_envs,
        device=args.device,
        renderer=args.renderer,
    )

    rng = np.random.RandomState(args.seed)
    T = args.episode_len
    A = ACTION_DIM

    all_states, all_actions, all_lens = [], [], []
    ep_idx = 0
    while ep_idx < args.num_episodes:
        env.reset()
        ep_states = np.zeros((args.num_envs, T, env.state_dim), dtype=np.float32)
        ep_actions = np.zeros((args.num_envs, T, A), dtype=np.float32)
        ep_visuals = np.zeros((args.num_envs, T, 224, 224, 3), dtype=np.uint8)

        for t in range(T):
            action = rng.uniform(-1.0, 1.0, size=(args.num_envs, A)).astype(np.float32)
            obs, _, _, info = env.step(action)
            ep_states[:, t] = info["state"]
            ep_actions[:, t] = action
            ep_visuals[:, t] = obs["visual"]

        for i in range(args.num_envs):
            if ep_idx >= args.num_episodes:
                break
            all_states.append(ep_states[i])
            all_actions.append(ep_actions[i])
            all_lens.append(T)
            torch.save(
                torch.from_numpy(ep_visuals[i]),
                out / "obses" / f"episode_{ep_idx:03d}.pth",
            )
            ep_idx += 1
            print(f"[collect] episode {ep_idx}/{args.num_episodes}")

    torch.save(torch.from_numpy(np.stack(all_states)), out / "states.pth")
    torch.save(torch.from_numpy(np.stack(all_actions)), out / "actions.pth")
    torch.save(torch.tensor(all_lens, dtype=torch.int64), out / "seq_lengths.pth")
    env.close()
    print(f"[collect] wrote {ep_idx} episodes to {out}")


if __name__ == "__main__":
    main()
