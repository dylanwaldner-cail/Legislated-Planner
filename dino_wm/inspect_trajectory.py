"""Run one trajectory and save every frame as PNG for inspection.

Supports two policies:
  random — uniform random actions per agent.
  expert — scripted pick-and-place: each arm picks a random cube at episode
           start and moves it to a random target cell, then idles. Phase
           machine: approach_above -> descend -> grasp -> lift -> move ->
           descend2 -> release -> done.

Usage:
    ./IsaacLab/isaaclab.sh -p inspect_trajectory.py
    ./IsaacLab/isaaclab.sh -p inspect_trajectory.py --policy expert
    ./IsaacLab/isaaclab.sh -p inspect_trajectory.py --policy expert --seed 42 --num_steps 300
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

from env.isaaclab.grid_wrapper import ACTION_DIM_PER_AGENT, GridWrapper
from env.isaaclab.app_launcher import close_or_exit
from env.isaaclab.grid_metadata import GRID_HALF  # noqa: F401 (reserved; was used by biased_random target sampling)
from env.isaaclab.expert_policy import ExpertPickPlace, _CUBE_NAMES, _SIDES  # noqa: F401

# World->base delta is now computed via env.world_to_base_delta() which uses
# IsaacLab's quat_inv + quat_apply (library functions, not hand-derived
# signs). The previous _BASE_X_SIGN / _BASE_Y_SIGN hardcoded flips made
# theoretically correct sign choices but the empirical IK behavior showed
# X-axis motion going opposite the commanded direction. Using the library
# transform eliminates any hand-derivation error.


# Desired starting EE positions in world frame. The env's ready joint pose
# can land the EE wherever; this is the "true" starting pose we drive the
# arms to via IK before the policy begins.
START_EE_WORLD = {
    "left":  np.array([-0.08, 0.0, 0.30], dtype=np.float32),
    "right": np.array([+0.08, 0.0, 0.30], dtype=np.float32),
}


# _world_to_base_delta is replaced by env.world_to_base_delta(side, delta).
# Callers use that directly now to get library-validated frame conversion.


def _drive_to_target_ee(env, target_ee_world, max_steps=120, dist_threshold=0.015):
    """Drive both arms to the given world-frame EE positions using the env's
    IK action interface. Used once at episode start to put the arms at a
    declarative start pose instead of relying on ready-pose joint angles."""
    last_obs = None
    for step in range(max_steps):
        ee = env.get_ee_positions()
        actions = {}
        all_close = True
        for side in _SIDES:
            target = target_ee_world[side]
            current = ee[side][0]
            dist = float(np.linalg.norm(target - current))
            if dist > dist_threshold:
                all_close = False
            delta_world = (target - current) / _IK_SCALE
            delta_base = np.clip(_world_to_base_delta(side, delta_world), -1.0, 1.0)
            action = np.zeros((1, ACTION_DIM_PER_AGENT), dtype=np.float32)
            action[0, :3] = delta_base
            # action[0, 3:6] = 0 by zeros init (no rotation change)
            action[0, 6] = +1.0  # gripper open (positive = open)
            actions[side] = action
        last_obs, _, _, _ = env.step(actions)
        if all_close:
            print(f"[startup] arms settled to target EE in {step + 1} steps")
            return last_obs
    print(f"[startup] WARN: arms did not converge in {max_steps} steps "
          f"(left dist={np.linalg.norm(target_ee_world['left'] - env.get_ee_positions()['left'][0]):.3f}, "
          f"right dist={np.linalg.norm(target_ee_world['right'] - env.get_ee_positions()['right'][0]):.3f})")
    return last_obs



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-v0")
    ap.add_argument("--num_steps", type=int, default=200)  # noisy_expert loops
                                                            # phase + tighter threshold
                                                            # need more total steps
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=["expert", "noisy_expert"], default="noisy_expert")
    ap.add_argument("--place_cells", default=None,
                    help="Pin the deterministic expert's place targets to fixed "
                    "cells, as 'left,right' cell ids 0-8 (e.g. '2,6' = each "
                    "robot's far corner nearest its own camera). Only applies "
                    "to --policy expert; default = random within each robot's "
                    "allowed row-half.")
    ap.add_argument("--noise_std", type=float, default=0.1,
                    help="For noisy_expert: standard deviation of gaussian "
                    "noise added to expert position deltas before clipping. "
                    "0 = deterministic expert; ~0.1 = subtle perturbation "
                    "(still picks blocks up reliably); 0.3+ = expert sometimes "
                    "misses grasps, useful for failure-mode data.")
    ap.add_argument("--bias_strength", type=float, default=0.5,
                    help="RESERVED / currently unused. Was for biased_random "
                    "(now removed). Kept as a placeholder for a future "
                    "policy that wants a similar blend-toward-target knob.")
    ap.add_argument("--output_dir", default=str(_REPO_ROOT / "trajectory_inspect"))
    ap.add_argument("--render_mode", default="PathTracing",
                    choices=("PathTracing", "RaytracedLighting"),
                    help="RTX render mode. PathTracing = clean but slower.")
    ap.add_argument("--spp", type=int, default=128,
                    help="Samples-per-pixel for PathTracing. 128 default; "
                    "use 256 or 512 for demo videos you plan to share.")
    ap.add_argument("--show_markers", action="store_true",
                    help="Render cyan IK target spheres and magenta cube "
                    "spheres in the scene (debug aid). Off by default so "
                    "demo videos are clean.")
    ap.add_argument("--label_test", action="store_true",
                    help="Diagnostic: render a colored sphere 6cm above each "
                    "cube at the position reported for that label (black/blue). "
                    "If the colored sphere floats above the wrong cube, the "
                    "wrapper's cube_black/blue label-to-prim mapping is swapped.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    env = GridWrapper(
        task_id=args.task_id, num_envs=1, device=args.device,
        render_mode=args.render_mode, spp=args.spp,
    )
    env.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    obs, _ = env.reset()
    cameras = list(obs["visual"].keys())
    for cam in cameras:
        cam_dir = out / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f in cam_dir.glob("frame_*.png"):
            f.unlink()

    # NOTE: previously used _drive_to_target_ee here to declaratively set the
    # start EE position via IK. Dropped because position-mode IK has huge null
    # space and chose flat-on-table joint configurations to reach the target.
    # The env_cfg's ready_joint_pos now controls the start pose directly — it
    # gives a classic Franka "crane" shape (upper arm tilted forward, elbow
    # bent, wrist down) which is what we actually want visually. If you need
    # to move the EE, tune ready_joint_pos in dinowm_grid_env_cfg.py.

    # Sanity-check final starting positions (after the IK drive-to-target).
    ee0 = env.get_ee_positions()
    cubes0 = env.get_cube_positions()
    for side in _SIDES:
        print(f"[startup] {side:5s} start EE (world): {ee0[side][0]}")
    for k in _CUBE_NAMES:
        print(f"[startup] {k:11s} start pos (world): {cubes0[k][0]}")

    expert = None
    if args.policy in ("expert", "noisy_expert"):
        fixed_cells = None
        if args.place_cells:
            l, r = (int(c) for c in args.place_cells.split(","))
            fixed_cells = {"left": l, "right": r}
        expert = ExpertPickPlace(rng, env=env,
                                  chase_random=(args.policy == "noisy_expert"),
                                  fixed_cells=fixed_cells)
        expert.noise_std = args.noise_std if args.policy == "noisy_expert" else 0.0
        expert.reset()

    # Visualization markers for per-arm IK targets and cube positions.
    # Off by default so demo videos are clean — pass --show_markers to
    # turn them back on for debugging IK/policy.
    target_markers = None
    cube_markers = None
    label_markers = None  # {cube_name -> VisualizationMarkers} for --label_test
    if (expert is not None and args.show_markers) or args.label_test:
        import torch as _torch
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as _sim_utils
    if expert is not None and args.show_markers:
        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/IKTargets",
            markers={
                "target": _sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
            },
        )
        target_markers = VisualizationMarkers(target_marker_cfg)
        cube_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CubeMarkers",
            markers={
                "cube": _sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                ),
            },
        )
        cube_markers = VisualizationMarkers(cube_marker_cfg)
    if args.label_test:
        # One marker per cube label, colored to match the label name. Floats
        # 6cm above the cube position the wrapper reports for that name. If
        # the visual cube under the sphere isn't the matching color, the
        # label-to-prim mapping inside the wrapper is wrong.
        label_markers = {}
        _LABEL_COLORS = {
            # cube_black renders near-black; use a mid-gray marker so the
            # diagnostic sphere is actually visible above it.
            "cube_black": (0.4, 0.4, 0.4),
            "cube_blue": (0.0, 0.3, 1.0),  # slightly less pure blue so it's distinguishable from the blue cube under it
        }
        for cube_name, color in _LABEL_COLORS.items():
            cfg = VisualizationMarkersCfg(
                prim_path=f"/Visuals/CubeLabel_{cube_name}",
                markers={
                    "marker": _sim_utils.SphereCfg(
                        radius=0.020,
                        visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    ),
                },
            )
            label_markers[cube_name] = VisualizationMarkers(cfg)

    def _save(step):
        for cam in cameras:
            img = obs["visual"][cam][0]
            Image.fromarray(img).save(out / cam / f"frame_{step:04d}.png")

    step = 0
    zero = np.zeros((1, ACTION_DIM_PER_AGENT), dtype=np.float32)
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step({"left": zero, "right": zero})
        _save(step)
        step += 1

    for _ in range(args.num_steps):
        # Exploration noise (noisy_expert) is applied inside the expert, scaled
        # per phase (full on transit/place, a little on the pickup); rotation/
        # gripper channels stay clean. See expert.noise_std / PHASE_NOISE_SCALE.
        actions = expert(env.get_ee_positions(), env.get_cube_positions())

        obs, _, _, _ = env.step(actions)
        _save(step)

        # Label-test markers: float a color-matched sphere above each cube
        # at the position the wrapper reports for that cube name. This is
        # the visual sanity check for cube_black/blue prim-to-label
        # mapping. Updated every step so the marker tracks moving cubes.
        if label_markers is not None:
            cubes_now = env.get_cube_positions()
            for cube_name, marker in label_markers.items():
                pos = cubes_now[cube_name][0].copy()
                pos[2] += 0.06  # 6cm above cube top
                marker.visualize(
                    translations=_torch.tensor(
                        [list(pos)], device=env.device, dtype=_torch.float32,
                    )
                )

        # Update target markers and emit a diagnostic every 5 steps so we
        # can see if target/EE are tracking each other or drifting apart.
        if expert is not None:
            tL = expert.last_target.get("left")
            tR = expert.last_target.get("right")
            if tL is not None and tR is not None and target_markers is not None:
                target_markers.visualize(
                    translations=_torch.tensor(
                        [list(tL), list(tR)], device=env.device, dtype=_torch.float32,
                    )
                )
            # Also render markers at the actual cube positions so the cyan
            # target marker can be visually compared to the magenta cube
            # marker. Each arm's assigned cube is rendered.
            if cube_markers is not None and expert is not None:
                cubes_now = env.get_cube_positions()
                cube_l = cubes_now[expert.state["left"]["cube"]][0]
                cube_r = cubes_now[expert.state["right"]["cube"]][0]
                cube_markers.visualize(
                    translations=_torch.tensor(
                        [list(cube_l), list(cube_r)], device=env.device, dtype=_torch.float32,
                    )
                )
            if step % 5 == 0:
                ee = env.get_ee_positions()
                cubes_diag = env.get_cube_positions()
                if tL is not None:
                    cube_l = cubes_diag[expert.state["left"]["cube"]][0]
                    print(f"[diag step={step:3d}] left  target_w={tL}  cube_w={cube_l}  ee_w={ee['left'][0]}  d={np.linalg.norm(tL - ee['left'][0]):.3f}")
                if tR is not None:
                    cube_r = cubes_diag[expert.state["right"]["cube"]][0]
                    print(f"[diag step={step:3d}] right target_w={tR}  cube_w={cube_r}  ee_w={ee['right'][0]}  d={np.linalg.norm(tR - ee['right'][0]):.3f}")

        step += 1
        if step % 20 == 0:
            print(f"[traj] step {step}/{args.settle_steps + args.num_steps}")

    print(f"[traj] saved {step} frames per camera ({', '.join(cameras)}) to {out}")
    close_or_exit(env)


if __name__ == "__main__":
    main()
