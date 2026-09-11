"""Render the Q4 audit-trace filmstrip: one full-spp panel per decision step, sign colour included.

Panels are the recorded GT cube rest positions of ONE episode, up to the step where the taint
sanction fires. Same non-touch render path as scripts/render_filmstrip.py / render_env_figure.py
(teleport the cube, park the arm at home, zero-displacement hold) so every panel has the clean
`env.png` look rather than a mid-stroke arm pose.

DEFAULT EPISODE -- results/aug20/sign_change social 1_7 batch_004 ep3, the Q4 trace candidate:
  step 0  cell 1   white   planner predicts a 4.09 cm nudge landing in cell 1
  step 1  cell 4   yellow  it travelled 11.16 cm (2.73x overshoot, 7.87 cm WM error) into the centre
  step 2  cell 1   yellow  R10 contrary-to-duty fires -> [O]exit_cell(4), goal bank waypoints to 1, exit succeeds
  step 3  cell 0   yellow  resumes toward yellow cell 3 to satisfy [O]in_yellow_cell
  step 4  cell 0   RED     reaches yellow cell 3 while visited(4)-tainted -> R7b flips red -> R9 freezes it

SIGN COLOUR PER PANEL uses the ledger's EFFECTIVE sign (the colour the verdict was assessed under),
not the raw rendered colour. At step 4 those differ: raw is still yellow, effective is red, because
R7b derives the flip. The figure is about the normative state, so effective is the honest choice --
but it is a CHOICE, and `--sign_source raw` renders the other one.

Run INSIDE the container, from /workspace/src:
    ./IsaacLab/isaaclab.sh -p scripts/render_trace_filmstrip.py --spp 256 --width 640 --height 640
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image

_CUBE_XY = slice(18, 20)
_CUBE_QUAT = slice(21, 25)      # (w, x, y, z) within the 31-D state; 18:21 pos, 25:31 lin/ang vel
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="results/aug20/sign_change/social/1_7/batch_004/normative_ledger.json")
    ap.add_argument("--ep", default="3", help="episode key within the ledger")
    ap.add_argument("--first_step", type=int, default=0,
                    help="first step to render (lets you add one panel without redoing the others)")
    ap.add_argument("--steps", type=int, default=5, help="render steps [first_step, steps)")
    ap.add_argument("--predicted", action="store_true", default=True,
                    help="also render, per panel, the cube at the PREVIOUS step's predicted landing, "
                         "plus one cube-free background frame. compose_trace_filmstrip.py differences "
                         "them to locate the predicted cube in image space and draw a marker, which "
                         "needs no camera calibration.")
    ap.add_argument("--keep_orientation", action="store_true",
                    help="keep the reset pose's cube yaw. Default writes an IDENTITY quaternion so the "
                         "cube sits square to the grid: per-step orientation is not recorded anywhere "
                         "(the ledger stores only gt_xy), so a yawed cube would be an arbitrary artifact "
                         "of env.reset() rather than the episode, and it reads as balanced on a corner.")
    ap.add_argument("--sign_source", default="effective", choices=("effective", "raw"),
                    help="'effective' = the colour the verdict was assessed under (R7b-derived); "
                         "'raw' = the colour actually rendered in the scene that step")
    ap.add_argument("--spp", type=int, default=256, help="raise for publication (128 is the data default)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render_mode", default="PathTracing", choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--out_dir", default=str(_REPO / "scripts" / "trace_filmstrip_frames"))
    args = ap.parse_args()

    from env.isaaclab.grid_venv import SIGN_PALETTE

    allrecs = json.loads((_REPO / args.ledger).read_text())[args.ep]["records"]
    panels = []
    for i in range(args.first_step, min(args.steps, len(allrecs))):
        r = allrecs[i]
        raw = next((f[len("sign("):-1] for f in r["facts"] if f.startswith("sign(")), "white")
        colour = r.get("effective_sign", raw) if args.sign_source == "effective" else raw
        # The PREDICTED landing for THIS panel is what the PREVIOUS step's committed stroke aimed at,
        # i.e. where the planner thought the cube would be when this frame was taken. Indexed off the
        # FULL record list, so rendering a later panel alone still gets its predecessor's prediction.
        prev = allrecs[i - 1].get("committed") if i else None
        pred = np.asarray((prev or {}).get("predicted_final_pos"), dtype=np.float32) if (prev or {}).get(
            "predicted_final_pos") else None
        panels.append((r["step"], np.asarray(r["gt_xy"], dtype=np.float32), colour, pred))
    print(f"[render] {len(panels)} panels from {args.ledger} ep{args.ep} "
          f"(sign_source={args.sign_source}, spp={args.spp}, predicted={args.predicted})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from env.isaaclab.app_launcher import close_or_exit
    from env.isaaclab.grid_wrapper_single import GridWrapperSingle

    env = GridWrapperSingle(num_envs=1, device=args.device, render_mode=args.render_mode,
                            spp=args.spp, cam_wh=(args.width, args.height))
    env.seed(args.seed)
    _, base_state = env.reset()          # carries the parked/home arm joints reused for every panel

    # WARMUP: the first path-traced frame after boot returns unconverged (near-empty, ~33-byte PNG).
    # Render throwaways until the frame is a plausible image so no saved panel is the boot artifact.
    warm = base_state.copy()
    warm[0, _CUBE_XY] = panels[0][1]
    for i in range(3):
        env._write_state(warm)
        obs, _, _, _ = env.step(np.concatenate([panels[0][1], np.zeros(2)]).astype(np.float32))
        f = obs["visual"][0]
        if f is not None and getattr(f, "shape", None) == (args.height, args.width, 3) and f.std() > 1.0:
            print(f"[render] warmup converged after {i+1} throwaway frame(s)")
            break
    else:
        print("[render] WARNING: warmup never produced a plausible frame -- panels may be unconverged")

    def shoot(xy, colour):
        """Teleport the cube to `xy` under sign `colour` and return one converged full-spp frame.

        Deliberately does NOT call env.step(): a zero-displacement "hold" stroke has its START at the
        cube's own xy, so the paddle descends onto the cube and TIPS it -- which is why some panels
        came out with the cube on a corner and others flat, depending on the approach geometry. We
        write the state, flush it to PhysX, and read the scene directly instead, so nothing touches
        the cube between the teleport and the shutter.
        """
        env.set_sign_color(SIGN_PALETTE[colour])
        state = base_state.copy()                       # fresh home-arm pose so no arm occlusion
        state[0, _CUBE_XY] = xy
        if not args.keep_orientation:
            state[0, _CUBE_QUAT] = _IDENTITY_QUAT       # square to the grid, not the reset pose's yaw
            state[0, 25:31] = 0.0                       # zero lin/ang velocity so the flush cannot drift it
        env._write_state(state)
        env._materialize_state()                        # write_data_to_sim + one render step, no stroke
        obs, _ = env._scene_outputs()
        return obs["visual"][0]

    # One clean plate PER SIGN COLOUR. A single plate is not enough: differencing a yellow-sign frame
    # against a white-sign plate lights up the whole octagon, which swamps the cube and drags the
    # centroid onto the sign (~26k diff pixels instead of the cube's few hundred).
    if args.predicted:
        for colour in sorted({c for _, _, c, _ in panels}):
            bg_path = out_dir / f"trace_background_{colour}.png"
            Image.fromarray(shoot(np.array([1.5, 1.5], dtype=np.float32), colour)).save(bg_path)
            print(f"[render] background plate (sign={colour}) -> {bg_path}")

    for step, xy, colour, pred in panels:
        out = out_dir / f"trace_{step:02d}_{colour}.png"
        Image.fromarray(shoot(xy, colour)).save(out)
        print(f"[render] step {step} cube=({xy[0]:+.4f},{xy[1]:+.4f}) sign={colour} -> {out}")
        if args.predicted and pred is not None:
            pout = out_dir / f"trace_{step:02d}_pred.png"
            Image.fromarray(shoot(pred, colour)).save(pout)
            print(f"[render]   predicted landing=({pred[0]:+.4f},{pred[1]:+.4f}) -> {pout}")

    close_or_exit(env)


if __name__ == "__main__":
    main()
