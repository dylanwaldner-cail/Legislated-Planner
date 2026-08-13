"""Render the frames for a two-row ROLLOUT FILMSTRIP (prospective figure) in ONE IsaacLab boot.

Two matched recorded episodes on task 0->8 (base_025, geometry law: center cell 4 forbidden), same
start (cell 0) and goal (cell 8), differing only at the middle stroke:
  * realistic (off)  cells [0, 4, 7, 8]  -- plows STRAIGHT THROUGH the forbidden centre (panel 2 on the red tile)
  * social           cells [0, 3, 7, 8]  -- DETOURS around it via cell 3 (panel 2 far left)
For each recorded cube rest position we teleport the cube there, park the arm at home, and render one
full-spp frame (same non-touch path as scripts/render_env_figure.py -> the env.png look). Frames land in
--out_dir as film_{row}_{k}.png; scripts/compose_filmstrip.py stitches them on the host.

Run INSIDE the container (RTX renderer), from /workspace/dino_wm:
    ./IsaacLab/isaaclab.sh -p scripts/render_filmstrip.py --spp 128 --width 640 --height 640 --device cuda:1
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
_WHITE = (1.0, 1.0, 1.0)

# Recorded cube rest positions (env-local xy), source episodes noted above.
ROWS = {
    "realistic": [(-0.1247, -0.0862), (-0.0032, 0.0520), (0.0039, 0.1135), (0.1314, 0.1437)],  # off b000 ep8 [0,4,7,8]
    "social":    [(-0.1268, -0.1091), (-0.1442, 0.0439), (0.0062, 0.1559), (0.1263, 0.1316)],  # social b000 ep0 [0,3,7,8]
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render_mode", default="PathTracing", choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--out_dir", default=str(_REPO / "scripts" / "filmstrip_frames"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from env.isaaclab.app_launcher import close_or_exit
    from env.isaaclab.grid_wrapper_single import GridWrapperSingle

    env = GridWrapperSingle(
        num_envs=1, device=args.device, render_mode=args.render_mode, spp=args.spp,
        cam_wh=(args.width, args.height),
    )
    env.seed(args.seed)
    _, base_state = env.reset()          # base_state carries the parked/home arm joints we reuse every frame
    env.set_sign_color(_WHITE)

    # WARMUP: the first path-traced frame after boot comes back unconverged (near-empty). Render one
    # throwaway so every saved panel below is a fully converged frame.
    _warm = base_state.copy(); _warm[0, _CUBE_XY] = np.asarray(ROWS["realistic"][0], dtype=np.float32)
    env._write_state(_warm)
    env.step(np.concatenate([np.asarray(ROWS["realistic"][0], dtype=np.float32), np.zeros(2)]).astype(np.float32))

    for row, xys in ROWS.items():
        for k, xy in enumerate(xys):
            state = base_state.copy()                                   # fresh home-arm pose each frame
            state[0, _CUBE_XY] = np.asarray(xy, dtype=np.float32)       # teleport cube to the recorded rest point
            env._write_state(state)
            # zero-displacement "hold": renders the boundary frame at full spp, arm parked home (env.png look)
            obs, _, _, _ = env.step(np.concatenate([np.asarray(xy, dtype=np.float32), np.zeros(2)]).astype(np.float32))
            frame = obs["visual"][0]
            out = out_dir / f"film_{row}_{k}.png"
            Image.fromarray(frame).save(out)
            print(f"[render] {row} panel {k} cube=({xy[0]:+.3f},{xy[1]:+.3f}) -> {out}")

    close_or_exit(env)


if __name__ == "__main__":
    main()
