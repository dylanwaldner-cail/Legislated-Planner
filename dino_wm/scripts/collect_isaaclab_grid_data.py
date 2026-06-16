"""Collect single-robot push rollouts from the DinoWM-Single grid env into
dino_wm .pth format.

Fixed-length collection for a one-step world model: many SHORT trajectories
(default 60 env-steps -> 60-20+1 = 41 training windows each at frameskip 5 /
num_frames 4), each a FRESH random cube spawn driven by random-direction pushes
(PushExpert push_mode="random"), with the stop-sign recolored randomly per
episode. Coverage of the (cube-position, push-direction) space is emergent from
the random spawns + random pushes — no goal/cell structure (a one-step dynamics
model needs none).

Writes ${output_dir}/:
  states.pth        (E, T, 31)
  actions.pth       (E, T, 7)
  proprio.pth       (E, T, 18)
  cell_labels.pth   (E, T) int64   (single cube)
  seq_lengths.pth   (E,) int64
  sign_colors.pth   (E,) int64     (per-episode sign color, indexes SIGN_PALETTE)
  metadata.json     collection params (provenance)
  obses/episode_NNNNN.pth   (T, H, W, 3) uint8

The .pth arrays are re-saved every --save_every episodes and on Ctrl-C, so an
interrupted run leaves a valid dataset of everything collected so far.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from env.isaaclab.grid_wrapper_single import ACTION_DIM, GridWrapperSingle, PROPRIO_DIM
from env.isaaclab.grid_metadata import cell_labels_from_states_single
from env.isaaclab.app_launcher import close_or_exit
from env.isaaclab.expert_policy import PushExpert

IMG_HW = 224
EP_PAD = 5  # episode index zero-pad width (episode_00000.pth); must match the loader

# Stop-sign colors, randomized per episode (visual diversity + sets up the
# future sign-as-rule signal). Index recorded per episode in sign_colors.pth.
SIGN_PALETTE = [
    ("white", (1.0, 1.0, 1.0)),
    ("red", (1.0, 0.0, 0.0)),
    ("yellow", (1.0, 1.0, 0.0)),
    ("green", (0.0, 0.8, 0.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-Single-v0")
    ap.add_argument("--num_episodes", type=int, default=500)
    ap.add_argument("--episode_len", type=int, default=60,
                    help="env-steps per trajectory (fixed). >= num_frames*frameskip "
                    "(=20) or the slicer drops it; 60 gives ~41 windows each.")
    ap.add_argument("--output_dir", default=os.environ.get("DATASET_DIR", "./data") + "/isaaclab_grid_single")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0",
                    help="Must be 'cuda:0' for this Isaac Sim build. Mask with CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--render_mode", default="PathTracing",
                    choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--spp", type=int, default=128,
                    help="Samples-per-pixel for PathTracing (128 fine for training; "
                    "256+ only for demo videos).")
    ap.add_argument("--noise_std", type=float, default=0.0,
                    help="Per-step push-noise std (0 = clean pushes). Bump for "
                    "exploration variety.")
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="ON by default: at reset, drive the EE (unrecorded) until it "
                    "engages the cube, so EVERY recorded trajectory starts with the "
                    "pusher AT the (randomly placed) block. Cheap now that the raised "
                    "base makes reaching quick. --no-warmup to disable.")
    ap.add_argument("--warmup_max", type=int, default=10,
                    help="Cap on unrecorded warmup steps; if it can't engage by then "
                    "(unreachable spawn) recording starts anyway.")
    ap.add_argument("--save_every", type=int, default=25,
                    help="Re-save the .pth arrays every N episodes (crash safety).")
    ap.add_argument("--debug", action="store_true",
                    help="per-step expert diagnostics (heading, orientation error, "
                    "EE/cube progress, joints near limits). Use with a few episodes "
                    "to diagnose freezes; it's very verbose.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    (out / "obses").mkdir(parents=True, exist_ok=True)

    env = GridWrapperSingle(
        task_id=args.task_id, num_envs=1, device=args.device,
        render_mode=args.render_mode, spp=args.spp,
    )
    env.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    T = args.episode_len

    expert = PushExpert(rng, env=env, verbose=False, push_mode="random")
    expert.debug = args.debug
    all_states, all_actions, all_proprio, all_cells, all_lens, all_sign = [], [], [], [], [], []

    def save_arrays():
        """Persist everything collected so far (valid dataset at any checkpoint)."""
        n = len(all_states)
        if n == 0:
            return
        torch.save(torch.from_numpy(np.stack(all_states)), out / "states.pth")
        torch.save(torch.from_numpy(np.stack(all_actions)), out / "actions.pth")
        torch.save(torch.from_numpy(np.stack(all_proprio)), out / "proprio.pth")
        torch.save(torch.from_numpy(np.stack(all_cells)), out / "cell_labels.pth")
        torch.save(torch.tensor(all_lens, dtype=torch.int64), out / "seq_lengths.pth")
        torch.save(torch.tensor(all_sign, dtype=torch.int64), out / "sign_colors.pth")
        meta = {
            "num_episodes": n, "episode_len": T, "seed": args.seed,
            "state_dim": env.state_dim, "action_dim": ACTION_DIM, "proprio_dim": PROPRIO_DIM,
            "img_hw": IMG_HW, "noise_std": args.noise_std, "task_id": args.task_id,
            "render_mode": args.render_mode, "spp": args.spp, "ep_pad": EP_PAD,
            "sign_palette": [name for name, _ in SIGN_PALETTE],  # sign_colors.pth indexes this
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    t0 = time.perf_counter()
    try:
        for ep_idx in range(args.num_episodes):
            obs, state = env.reset()  # fresh random cube spawn
            # Randomize the stop-sign color this episode, then refresh obs so the
            # recorded frames show it.
            sign_idx = int(rng.randint(0, len(SIGN_PALETTE)))
            env.set_sign_color(SIGN_PALETTE[sign_idx][1])
            obs, state = env._scene_outputs()

            expert.noise_std = args.noise_std
            expert.reset()
            warmed = 0
            if args.warmup:  # unrecorded warmup until the pusher engages the cube
                for warmed in range(1, args.warmup_max + 1):
                    action = expert(env.get_ee_positions(), env.get_cube_positions())
                    obs, _, _, info = env.step(action)
                    state = info["state"]
                    if expert.PHASES[expert.state["phase_idx"]] == "push":
                        break

            ep_states = np.zeros((T, env.state_dim), dtype=np.float32)
            ep_actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
            ep_proprio = np.zeros((T, PROPRIO_DIM), dtype=np.float32)
            ep_vis = np.zeros((T, IMG_HW, IMG_HW, 3), dtype=np.uint8)

            for t in range(T):
                # Pre-step convention: save the obs/state FROM which action[t] is taken.
                ep_states[t] = state[0]
                ep_proprio[t] = obs["proprio"][0]
                ep_vis[t] = obs["visual"][0]

                action = expert(env.get_ee_positions(), env.get_cube_positions())
                ep_actions[t] = action[0]

                obs, _, _, info = env.step(action)
                state = info["state"]

            all_states.append(ep_states)
            all_actions.append(ep_actions)
            all_proprio.append(ep_proprio)
            all_cells.append(cell_labels_from_states_single(ep_states))  # (T,)
            all_lens.append(T)
            all_sign.append(sign_idx)
            torch.save(torch.from_numpy(ep_vis), out / "obses" / f"episode_{ep_idx:0{EP_PAD}d}.pth")

            done = ep_idx + 1
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (args.num_episodes - done)
            print(f"[collect] episode {done}/{args.num_episodes} "
                  f"(push warmup={warmed}, sign={SIGN_PALETTE[sign_idx][0]})  "
                  f"{elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m")
            if done % args.save_every == 0:
                save_arrays()
                print(f"[collect] checkpoint saved ({done} episodes)")

        save_arrays()
        print(f"[collect] DONE: wrote {len(all_states)} episodes (T={T}) to {out}")
    except KeyboardInterrupt:
        print(f"\n[collect] interrupted — saving {len(all_states)} episodes collected so far")
        save_arrays()
    finally:
        close_or_exit(env)


if __name__ == "__main__":
    main()
