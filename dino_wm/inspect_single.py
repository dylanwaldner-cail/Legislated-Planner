"""Smoke test / visualizer for the single-robot base-case env (PLANAR-STROKE action).

Instantiates GridWrapperSingle (num_envs=1), renders frames, checks shapes, and
exercises the runtime sign-color API. With --policy strokes it rolls out a
sequence of high-level push strokes (the same primitive the WM/planner use) via
GridWrapperSingle.execute_stroke and saves one boundary frame per stroke, so you
can eyeball that the blade faces each push direction and the cube tracks it.

Usage:
    ./IsaacLab/isaaclab.sh -p inspect_single.py
    ./IsaacLab/isaaclab.sh -p inspect_single.py --seed 3 --policy strokes --traj_steps 12
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
from env.isaaclab.grid_wrapper_single import GridWrapperSingle
from env.isaaclab.stroke_sampler import StrokeSampler

_CUBE_XY = slice(18, 20)  # cube (x,y) within the 31-D state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-Single-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("none", "strokes"), default="strokes",
                    help="none = static checks only; strokes = roll a sequence of "
                    "push strokes and save one boundary frame per stroke to <out>/traj/")
    ap.add_argument("--traj_steps", type=int, default=12, help="number of strokes to roll")
    ap.add_argument("--aimed_frac", type=float, default=1.0,
                    help="fraction of strokes that aim at the cube; 1.0 = always aim (clear viz)")
    ap.add_argument("--push_max", type=float, default=0.08,
                    help="per-axis push displacement bound (m) for the uniform strokes")
    ap.add_argument("--stroke_max_steps", type=int, default=320,
                    help="cap on internal IK sim steps per stroke (4 phases incl. retract)")
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
        render_mode=args.render_mode, spp=args.spp, stroke_max_steps=args.stroke_max_steps,
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

    # --- one zero-displacement "hold" stroke at the cube (near no-op): confirms step() ---
    cube_xy = state[0, _CUBE_XY]
    obs, _, _, info = env.step(np.concatenate([cube_xy, np.zeros(2)]).astype(np.float32))
    save("04_after_hold")
    print(f"[hold] zero-length stroke; cube(local)={env.get_cube_positions()[0]}")

    if args.policy == "none":
        print(f"[done] saved PNGs to {out}")
        close_or_exit(env)
        return

    # --- stroke rollout: one frame per stroke to <out>/traj/ ---
    traj = out / "traj"
    traj.mkdir(parents=True, exist_ok=True)
    for f in traj.glob("frame_*.png"):
        f.unlink()

    rng = np.random.RandomState(args.seed)
    sampler = StrokeSampler(rng, aimed_frac=args.aimed_frac, push_max=args.push_max)
    obs, state = env.reset()  # fresh episode for the rollout
    mode = sampler.reset_episode()
    start_cell = int(which_cell(state[0, _CUBE_XY]))
    print(f"[strokes] rollout ({mode}): cube starts in cell {start_cell}")
    Image.fromarray(obs["visual"][0]).save(traj / "frame_0000.png")
    last_cell = start_cell
    for t in range(1, args.traj_steps + 1):
        stroke = sampler.sample(state[0, _CUBE_XY])
        before = state[0, _CUBE_XY].copy()
        obs, _, _, info = env.step(stroke)
        state = info["state"]
        moved = float(np.linalg.norm(state[0, _CUBE_XY] - before))
        Image.fromarray(obs["visual"][0]).save(traj / f"frame_{t:04d}.png")
        cell = int(which_cell(state[0, _CUBE_XY]))
        tag = f" -> cell {cell}" if cell != last_cell else ""
        print(f"[strokes] stroke {t}: cube moved {moved:.3f}m{tag}")
        last_cell = cell

    final_cell = int(which_cell(state[0, _CUBE_XY]))
    print(f"[strokes] final cube cell {final_cell} after {args.traj_steps} strokes")
    print(f"[done] saved static PNGs to {out}, stroke frames to {traj}")
    close_or_exit(env)


if __name__ == "__main__":
    main()
