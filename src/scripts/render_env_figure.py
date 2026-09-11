"""Render ONE high-quality figure frame of the grid env (parked arm + cube), for the paper.

The WM camera is 224x224 (what DINO trains on), so the paper's env.png is a soft upscale. This script
overrides the camera to a high-res square and path-traces at high spp, at the SAME camera pose, then
places the cube where you want and saves a single crisp frame. Run it in the container (RTX renderer):

    ./IsaacLab/isaaclab.sh -p scripts/render_env_figure.py --spp 512 --width 1024 --height 1024 \
        --cube_cell 3 --sign white --out env_hi.png

Knobs: --cube_cell N (teleport cube to cell N's center) OR --cube_xy X Y (exact env-local xy); omit
both to use the seeded reset position. --sign {white,yellow,green,red}. Bump --spp for less noise.
Non-destructive: writes a NEW png (default scripts/env_hi.png); it never touches Images/env.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image

_CUBE_XY = slice(18, 20)  # cube (x,y) within the 31-D state
_SIGN_RGB = {"white": (1.0, 1.0, 1.0), "yellow": (1.0, 0.9, 0.0),
             "green": (0.0, 1.0, 0.0), "red": (1.0, 0.0, 0.0)}


def _touch_frame(env, cube_xy, args):
    """Drive the paddle down beside the cube and STOP at first contact, so the rendered frame shows the
    blade visibly on the cube (not the parked-at-home arm). Reuses the real StrokeExecutor: descend the
    paddle `back_off` behind the cube along push_dir, push toward it, and break the instant the cube is
    nudged (>= contact_move) -- a geometry-agnostic contact detector that leaves the cube in its cell.
    The arm is frozen at the contact pose (NO home-snap) and the cube's velocity zeroed, then one full-
    spp frame is rendered via the wrapper's normal boundary path (_write_state + _materialize_state)."""
    push_dir = np.asarray(args.push_dir, dtype=np.float32)
    push_dir = push_dir / max(float(np.linalg.norm(push_dir)), 1e-6)
    start_xy = cube_xy - push_dir * args.back_off        # descend clearly BEHIND the cube
    end_xy = cube_xy + push_dir * args.push_len          # push target past it (loop stops at contact)

    ex = env._get_executor()                             # captures the current wrist-down ref quat
    ex.begin(start_xy[None], end_xy[None])
    inner = env._env
    orig_ri = inner.unwrapped.cfg.sim.render_interval
    inner.unwrapped.cfg.sim.render_interval = 10**9      # skip per-step render; we render once at the end
    moved = 0.0
    try:
        for _ in range(args.stroke_max_steps):
            a7 = ex.compute_action(env.get_ee_positions(), env.get_cube_positions())
            inner.step(env._action_tensor(a7))
            env._freeze_episode_clock()
            env._clamp_cube()
            cube_now = env.get_cube_positions()[0, :2]
            moved = float(np.linalg.norm(cube_now - cube_xy))
            if int(ex.phase_idx[0]) >= 2 and moved >= args.contact_move:  # push phase + cube nudged = contact
                break
            if ex.all_done():
                break
    finally:
        inner.unwrapped.cfg.sim.render_interval = orig_ri

    ee = env.get_ee_positions()[0, :2]
    print(f"[touch] stopped at contact: cube moved {moved*1000:.1f}mm, "
          f"wrist-cube planar dist {float(np.linalg.norm(ee - cube_now)):.3f}m")

    # Freeze the contact pose: _write_state holds the arm at its CURRENT joints (target==state) instead of
    # snapping home, and we zero the cube's lin/ang velocity so the single render step doesn't drift it.
    obs, state = env._scene_outputs()
    state[0, 9:18] = 0.0                                 # zero arm joint vel so the render step holds the pose
    state[0, 25:31] = 0.0                                # cube block [pos3,quat4,linvel3,angvel3] @18 -> vel @25:31
    env._write_state(state)
    env._materialize_state()                             # one physics step + full-spp render at the contact pose
    return env._scene_outputs()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=512, help="path-tracing samples/pixel (noise ~1/sqrt(spp))")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cube_cell", type=int, default=None, help="teleport cube to this cell's center (0..8)")
    ap.add_argument("--cube_xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="teleport cube to this exact env-local xy (overrides --cube_cell)")
    ap.add_argument("--sign", default="white", choices=tuple(_SIGN_RGB))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render_mode", default="PathTracing", choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--stroke_max_steps", type=int, default=320)
    ap.add_argument("--touch", action="store_true",
                    help="drive the paddle down BESIDE the cube and stop at first contact (paddle "
                         "visibly on the cube) instead of parking the arm at home")
    ap.add_argument("--push_dir", type=float, nargs=2, default=(0.0, 1.0), metavar=("DX", "DY"),
                    help="--touch approach direction (env-local); (0,1)=+y=image-right beside the cube")
    ap.add_argument("--back_off", type=float, default=0.10,
                    help="--touch: how far behind the cube (along push_dir) the paddle descends (m)")
    ap.add_argument("--push_len", type=float, default=0.08,
                    help="--touch: push target distance past the cube (m); the loop stops at contact")
    ap.add_argument("--contact_move", type=float, default=0.004,
                    help="--touch: cube displacement (m) that counts as first contact -> stop")
    ap.add_argument("--out", default=str(_REPO / "scripts" / "env_hi.png"))
    args = ap.parse_args()

    from env.isaaclab.app_launcher import close_or_exit
    from env.isaaclab.grid_wrapper_single import GridWrapperSingle
    from env.isaaclab import grid_metadata as gm

    env = GridWrapperSingle(
        num_envs=1, device=args.device, render_mode=args.render_mode, spp=args.spp,
        stroke_max_steps=args.stroke_max_steps, cam_wh=(args.width, args.height),
    )
    env.seed(args.seed)

    obs, state = env.reset()
    cube_xy = state[0, _CUBE_XY].astype(np.float32).copy()
    if args.cube_xy is not None:
        cube_xy = np.asarray(args.cube_xy, dtype=np.float32)
    elif args.cube_cell is not None:
        cube_xy = np.asarray(gm.cell_center(args.cube_cell), dtype=np.float32)
    state[0, _CUBE_XY] = cube_xy
    env._write_state(state)                       # teleport cube (arm stays at reset/home)

    env.set_sign_color(_SIGN_RGB[args.sign])

    if args.touch:
        obs = _touch_frame(env, cube_xy, args)
    else:
        # zero-displacement "hold" at the cube: renders the boundary frame at full spp and leaves the
        # arm parked at home (retract phase) -- the same parked-arm look as the paper's env.png.
        obs, _, _, _ = env.step(np.concatenate([cube_xy, np.zeros(2)]).astype(np.float32))

    frame = obs["visual"][0]
    Image.fromarray(frame).save(args.out)
    print(f"[render] {frame.shape} @ {args.spp}spp {args.render_mode}, sign={args.sign}, "
          f"cube=({cube_xy[0]:+.3f},{cube_xy[1]:+.3f}) -> {args.out}")
    close_or_exit(env)


if __name__ == "__main__":
    main()
