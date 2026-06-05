"""Collect rollouts from IsaacLab DinoWM grid env into dino_wm .pth format.

Writes ${output_dir}/:
  states.pth          (E, T, 62)
  actions_left.pth    (E, T, 7)        actions_right.pth   (E, T, 7)
  proprio_left.pth    (E, T, 18)       proprio_right.pth   (E, T, 18)
  cell_labels.pth     (E, T, 2) int64  (per-cube: red, blue)
  seq_lengths.pth     (E,) int64
  obses/left/episode_NNN.pth    (T, H, W, 3) uint8
  obses/right/episode_NNN.pth   (T, H, W, 3) uint8

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
from env.isaaclab.app_launcher import close_or_exit
from env.isaaclab.expert_policy import ExpertPickPlace

CAMERAS = ("left", "right")  # per-robot OTS views; each robot's WM trains on its own side
IMG_HW = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-v0")
    ap.add_argument("--num_episodes", type=int, default=100)
    ap.add_argument("--episode_len", type=int, default=200)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--output_dir", default=os.environ.get("DATASET_DIR", "./data") + "/isaaclab_grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0",
                    help="Must be 'cuda:0' for this Isaac Sim build. Mask with CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--cooperative", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--render_mode", default="PathTracing",
                    choices=("PathTracing", "RaytracedLighting"),
                    help="RTX render mode. PathTracing = clean but slower; "
                    "RaytracedLighting = real-time but noisier.")
    ap.add_argument("--spp", type=int, default=128,
                    help="Samples-per-pixel for PathTracing. 128 default "
                    "(fine for training data); bump to 256+ only for demo "
                    "videos. Ignored when --render_mode=RaytracedLighting.")
    ap.add_argument("--mix_ratio", type=float, default=0.25,
                    help="Fraction of episodes that use the deterministic "
                    "expert (single pick-and-place at one assigned cell, "
                    "then idle). The rest use noisy_expert (looped "
                    "pick-and-place with random targets + position-delta "
                    "noise). 0.0 = all noisy_expert; 1.0 = all expert; "
                    "0.25 (default) = 1 in 4 episodes is a clean demo, "
                    "the rest are noisy loops.")
    ap.add_argument("--noise_std_min", type=float, default=0.05,
                    help="For noisy_expert episodes: lower bound of the "
                    "per-episode noise std sampled uniformly each episode. "
                    "Lower = cleaner trajectories.")
    ap.add_argument("--noise_std_max", type=float, default=0.3,
                    help="For noisy_expert episodes: upper bound. At 0.3+ "
                    "some episodes will produce missed grasps and dropped "
                    "cubes (useful failure-mode data). Sampling across "
                    "[min, max] per episode gives the dataset a mix of "
                    "near-clean and very wobbly trajectories.")
    ap.add_argument("--bias_strength", type=float, default=0.5,
                    help="RESERVED / currently unused. Placeholder for a "
                    "future biased policy that may want a similar knob.")
    args = ap.parse_args()
    # ExpertPickPlace is single-env (one phase machine). Force num_envs=1.
    # A batched expert would let collection scale, but that's a follow-up
    # refactor — for now, expert-flavored collection runs serially.
    if args.num_envs != 1:
        print(f"[collect] expert policies require --num_envs 1; "
              f"overriding from {args.num_envs}")
        args.num_envs = 1
    if not (0.0 <= args.mix_ratio <= 1.0):
        raise ValueError(f"--mix_ratio must be in [0, 1], got {args.mix_ratio}")

    out = Path(args.output_dir)
    for cam in CAMERAS:
        (out / "obses" / cam).mkdir(parents=True, exist_ok=True)

    env = GridWrapper(
        task_id=args.task_id, num_envs=args.num_envs, device=args.device,
        cooperative=args.cooperative, render_mode=args.render_mode, spp=args.spp,
    )
    rng = np.random.RandomState(args.seed)
    T, A, N, P = args.episode_len, ACTION_DIM_PER_AGENT, args.num_envs, PROPRIO_DIM_OWN

    all_states, all_aL, all_aR, all_pL, all_pR, all_cells, all_lens = [], [], [], [], [], [], []
    ep_idx = 0

    # Two expert instances — same code, different mode. Pick per episode
    # based on --mix_ratio.
    expert_det = ExpertPickPlace(rng, env=env, verbose=False, chase_random=False)
    expert_noisy = ExpertPickPlace(rng, env=env, verbose=False, chase_random=True)
    while ep_idx < args.num_episodes:
        # Pre-step convention: at index t the saved arrays hold the
        # obs/state from which action[t] is taken (action[t] transitions
        # state[t] -> state[t+1]).
        obs, state = env.reset()
        # Pick this episode's policy. mix_ratio fraction → expert (clean,
        # one cycle then idle). The rest → noisy_expert (looped pick-place
        # with random targets + per-step position noise).
        use_expert = rng.random() < args.mix_ratio
        expert = expert_det if use_expert else expert_noisy
        expert.reset()
        # noise_std irrelevant for the deterministic expert; for
        # noisy_expert, sample once per episode for consistent wobble.
        ep_noise_std = 0.0 if use_expert else float(
            rng.uniform(args.noise_std_min, args.noise_std_max)
        )
        policy_label = "expert" if use_expert else "noisy_expert"
        ep_states = np.zeros((N, T, env.state_dim), dtype=np.float32)
        ep_aL = np.zeros((N, T, A), dtype=np.float32)
        ep_aR = np.zeros((N, T, A), dtype=np.float32)
        ep_pL = np.zeros((N, T, P), dtype=np.float32)
        ep_pR = np.zeros((N, T, P), dtype=np.float32)
        ep_vis = {cam: np.zeros((N, T, IMG_HW, IMG_HW, 3), dtype=np.uint8) for cam in CAMERAS}

        for t in range(T):
            # Save the CURRENT (pre-step) obs/state at index t.
            ep_states[:, t] = state
            if env.cooperative:
                ep_pL[:, t] = obs["proprio"]["left"][:, :P]
                ep_pR[:, t] = obs["proprio"]["left"][:, P:]
            else:
                ep_pL[:, t] = obs["proprio"]["left"]
                ep_pR[:, t] = obs["proprio"]["right"]
            for cam in CAMERAS:
                ep_vis[cam][:, t] = obs["visual"][cam]

            # Compute and save the action taken FROM this state.
            actions = expert(env.get_ee_positions(), env.get_cube_positions())
            if not use_expert:
                for side in ("left", "right"):
                    a = actions[side]
                    noise = rng.normal(0.0, ep_noise_std,
                                       size=a[:, :3].shape).astype(np.float32)
                    a[:, :3] = np.clip(a[:, :3] + noise, -1.0, 1.0)
                    actions[side] = a
            a_left = actions["left"]
            a_right = actions["right"]
            ep_aL[:, t] = a_left
            ep_aR[:, t] = a_right

            # Step the env; obs/state now reflect the post-action result,
            # which becomes the pre-step observation for iteration t+1.
            obs, _, _, info = env.step({"left": a_left, "right": a_right})
            state = info["state"]

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
            print(f"[collect] episode {ep_idx}/{args.num_episodes} "
                  f"({policy_label}, noise_std={ep_noise_std:.3f})")

    torch.save(torch.from_numpy(np.stack(all_states)), out / "states.pth")
    torch.save(torch.from_numpy(np.stack(all_aL)), out / "actions_left.pth")
    torch.save(torch.from_numpy(np.stack(all_aR)), out / "actions_right.pth")
    torch.save(torch.from_numpy(np.stack(all_pL)), out / "proprio_left.pth")
    torch.save(torch.from_numpy(np.stack(all_pR)), out / "proprio_right.pth")
    torch.save(torch.from_numpy(np.stack(all_cells)), out / "cell_labels.pth")
    torch.save(torch.tensor(all_lens, dtype=torch.int64), out / "seq_lengths.pth")
    print(f"[collect] wrote {ep_idx} episodes to {out}")
    close_or_exit(env)


if __name__ == "__main__":
    main()
