"""Smoke test for the single-robot base-case env.

Instantiates GridWrapperSingle (num_envs=1), renders frames, checks shapes, and
exercises the runtime sign-color API. Save PNGs to eyeball framing (single
robot + 3x3 colored grid + octagonal sign to the robot's right, ~45deg down)
and to confirm the sign recolors.

Usage:
    ./IsaacLab/isaaclab.sh -p inspect_single.py
    ./IsaacLab/isaaclab.sh -p inspect_single.py --seed 3 --num_steps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image

from env.isaaclab.app_launcher import close_or_exit
from env.isaaclab.grid_metadata import STATE_DIM_SINGLE, which_cell
from env.isaaclab.grid_wrapper_single import ACTION_DIM, GridWrapperSingle
from env.isaaclab.expert_policy import PushExpert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-Single-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_steps", type=int, default=10,
                    help="hold-action steps in the static-check section")
    ap.add_argument("--policy", choices=("none", "random", "noisy_random", "cells"),
                    default="random",
                    help="none = just static checks; random/noisy_random = cube "
                    "random-walk push (the data-collection driver); cells = "
                    "goal-directed Manhattan push to --target_cell. Rolls out the "
                    "expert and saves a frame per step to <out>/traj/")
    ap.add_argument("--traj_steps", type=int, default=250,
                    help="expert rollout length")
    ap.add_argument("--target_cell", type=int, default=None,
                    help="--policy cells: pin the push target cell (0-8); default random")
    ap.add_argument("--noise_std", type=float, default=0.1,
                    help="exploration-noise std for --policy noisy_random")
    ap.add_argument("--debug", action="store_true",
                    help="per-step expert diagnostics (heading, orientation error, "
                    "EE/cube progress, joints near their limits) — to diagnose freezes")
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="drive the EE to the cube (unrecorded) before the rollout so "
                    "it starts engaged at the block; --no-warmup to disable")
    ap.add_argument("--warmup_max", type=int, default=10,
                    help="cap on unrecorded warmup steps")
    ap.add_argument("--output_dir", default=str(_REPO_ROOT / "single_inspect"))
    ap.add_argument("--render_mode", default="PathTracing",
                    choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--spp", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()

    env = GridWrapperSingle(
        task_id=args.task_id, num_envs=1, device=args.device,
        render_mode=args.render_mode, spp=args.spp,
    )
    env.seed(args.seed)

    def save(name):
        Image.fromarray(obs["visual"][0]).save(out / f"{name}.png")

    obs, state = env.reset()
    # --- shape checks ---
    vis = obs["visual"]
    print(f"[shapes] visual={vis.shape} (expect (1,224,224,3))")
    print(f"[shapes] proprio={obs['proprio'].shape} (expect (1,18))")
    print(f"[shapes] state={state.shape} (expect (1,{STATE_DIM_SINGLE}))")
    assert vis.shape[1:] == (224, 224, 3), vis.shape
    assert state.shape[-1] == STATE_DIM_SINGLE, state.shape
    print(f"[startup] ee(world-local)={env.get_ee_positions()[0]}  "
          f"cube(local)={env.get_cube_positions()[0]}")
    save("00_reset")

    # --- sign recolor API ---
    for name, rgb in (("01_sign_red", (1.0, 0.0, 0.0)),
                      ("02_sign_green", (0.0, 1.0, 0.0)),
                      ("03_sign_blue", (0.0, 0.0, 1.0))):
        env.set_sign_color(rgb)
        obs, _ = env._scene_outputs()
        save(name)
        print(f"[sign] set {rgb} -> saved {name}.png")

    # --- step a few hold actions (no-op: zero delta, gripper open) ---
    hold = np.zeros((1, ACTION_DIM), dtype=np.float32)
    hold[0, 6] = 1.0  # gripper open
    for i in range(args.num_steps):
        obs, _, _, info = env.step(hold)
        if i == args.num_steps - 1:
            save("04_after_steps")
    print(f"[steps] held {args.num_steps} steps; cube(local)={env.get_cube_positions()[0]}")

    if args.policy == "none":
        print(f"[done] saved PNGs to {out}")
        close_or_exit(env)
        return

    # --- push-expert rollout: one frame per step to <out>/traj/ ---
    traj = out / "traj"
    traj.mkdir(parents=True, exist_ok=True)
    for f in traj.glob("frame_*.png"):
        f.unlink()

    rng = np.random.RandomState(args.seed)
    obs, state = env.reset()  # fresh episode for the rollout
    cells_mode = (args.policy == "cells")
    expert = PushExpert(rng, env=env, push_mode=("cells" if cells_mode else "random"))
    expert.noise_std = args.noise_std if args.policy == "noisy_random" else 0.0
    expert.debug = args.debug
    expert.reset(target_cell=args.target_cell if cells_mode else None)

    # Unrecorded warmup: drive the EE to the cube so the recording starts engaged.
    if not cells_mode and args.warmup:
        for w in range(args.warmup_max):
            a = expert(env.get_ee_positions(), env.get_cube_positions())
            obs, _, _, _ = env.step(a)
            if expert.PHASES[expert.state["phase_idx"]] == "push":
                print(f"[push] warmup engaged the cube in {w + 1} steps")
                break

    start_cell = int(which_cell(env.get_cube_positions()[0][:2]))
    print(f"[push] rollout ({args.policy}): cube starts in cell {start_cell}")
    Image.fromarray(obs["visual"][0]).save(traj / "frame_0000.png")
    last_cell = start_cell
    for t in range(1, args.traj_steps + 1):
        action = expert(env.get_ee_positions(), env.get_cube_positions())
        obs, _, _, info = env.step(action)
        Image.fromarray(obs["visual"][0]).save(traj / f"frame_{t:04d}.png")
        cell = int(which_cell(env.get_cube_positions()[0][:2]))
        if cell != last_cell:
            print(f"[push] step {t}: cube entered cell {cell}")
            last_cell = cell
        if expert.state["done"]:
            print(f"[push] expert done at step {t}; cube in cell {cell}")
            break

    final_cell = int(which_cell(env.get_cube_positions()[0][:2]))
    if cells_mode:
        print(f"[push] final cube cell {final_cell} (target {expert.state['target_cell']}) "
              f"-> {'HIT' if final_cell == expert.state['target_cell'] else 'miss'}")
    else:
        print(f"[push] final cube cell {final_cell} after {args.traj_steps} random pushes")
    print(f"[done] saved static PNGs to {out}, trajectory frames to {traj}")
    close_or_exit(env)


if __name__ == "__main__":
    main()
