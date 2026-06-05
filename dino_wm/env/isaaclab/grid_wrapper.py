"""Adapter: IsaacLab DinoWMGrid -> dino_wm contract, per-agent dict I/O.

step({"left": (N,7), "right": (N,7)}) -> (obs, reward_dict, done, info).
Wrapper concatenates per-agent actions into IsaacLab's 14-D action.

State layout (62-D); cube positions stored in env-local frame (root_pos_w minus
env_origins) so cell-label math is correct for any num_envs:
    [left_jpos(9), left_jvel(9), right_jpos(9), right_jvel(9),
     cube_red(13), cube_blue(13)]
each cube block = [pos(3) env-local, quat(4 wxyz), linvel(3), angvel(3)].

cooperative=True  -> obs["proprio"] = {"left": both(36), "right": both(36)}
cooperative=False -> obs["proprio"] = {"left": own(18),  "right": own(18)}
Visual obs is {"left": (N,H,W,3), "right": (N,H,W,3)} either way — one
camera per robot (each looks over its own shoulder at the grid). Each
robot's world model trains on its own side's view.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .app_launcher import get_app
from .grid_metadata import GRID_CENTER_XY


_ARM_JOINT_DIM = 9
_CUBE_DIM = 13
_CUBE_KEYS = ("cube_red", "cube_blue")
# Cube placement zones (env-local x offset from grid center): left / right —
# one per cube now that there are two. Each reset randomly permutes the 2 cubes
# across these zones so which colour the left/right robots grab rotates
# episode-to-episode. With the bases at x=±0.45, a side cube at x=±0.10 sits
# ~0.35m from its own base (a comfortable grasp) and clearly on that robot's
# side. x-jitter (±0.05) < offset so a cube never crosses sides; y-jitter
# (±0.20) varies the row.
_CUBE_ZONE_DX = (-0.10, 0.10)
_CUBE_ZONE_JITTER_X = 0.05
_CUBE_ZONE_JITTER_Y = 0.20
_CUBE_SPAWN_Z = 0.026
_CAMERA_KEYS = ("camera_left", "camera_right")
_VIS_KEYS = ("left", "right")
# Per-robot cameras looking at the grid center. Left cam is front-left;
# right cam is mirrored across the robot-to-robot line (the x-axis) to the
# back-right (y: -0.75 -> +0.75), so each camera frames its own robot's side
# from opposite y-sides. Same xy-radius (0.901m), height (z=1.0), and
# down-angle for both — only camera_right's y is flipped.
_CAM_VIEWS = {
    "camera_left":  ((-0.5, -0.75, 1.0), (0.0, 0.0, 0.0)),
    "camera_right": (( 0.5,  0.75, 1.0), (0.0, 0.0, 0.0)),
}

PROPRIO_DIM_OWN = 2 * _ARM_JOINT_DIM  # 18
STATE_DIM = 2 * PROPRIO_DIM_OWN + len(_CUBE_KEYS) * _CUBE_DIM  # 62 (2 cubes)
# Per-agent action: [pos_delta_x, pos_delta_y, pos_delta_z,
#                    rot_delta_x, rot_delta_y, rot_delta_z, gripper] = 7-D.
# IK is in pose-mode (position + orientation); rot delta of (0,0,0) keeps
# the gripper at its current orientation each step. This is required to
# prevent the IK from drifting the wrist into bad orientations while
# tracking position targets.
ACTION_DIM_PER_AGENT = 7
ACTION_DIM_JOINT = 2 * ACTION_DIM_PER_AGENT  # 14 (= 6 arm + 1 gripper, per arm)


class GridWrapper:
    def __init__(
        self,
        task_id: str = "Isaac-DinoWMGrid-v0",
        num_envs: int = 1,
        device: str = "cuda:0",
        cooperative: bool = True,
        headless: bool = True,
        render_mode: str = "PathTracing",
        spp: int = 128,
    ):
        # render_mode: "PathTracing" (clean, slower) or "RaytracedLighting"
        # (noisier, faster). Both are RTX modes; IsaacLab Camera needs RTX.
        # spp: samples-per-pixel for PathTracing — 128 for collection, 256+
        # for demo videos. Ignored when render_mode is RaytracedLighting.
        get_app(headless=headless, enable_cameras=True, render_mode=render_mode, spp=spp)

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.dinowm_grid  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self.num_envs = num_envs
        self.device = device
        self.cooperative = cooperative
        self.action_dim_per_agent = ACTION_DIM_PER_AGENT
        self.action_dim_joint = ACTION_DIM_JOINT
        self.state_dim = STATE_DIM
        self.proprio_dim = 2 * PROPRIO_DIM_OWN if cooperative else PROPRIO_DIM_OWN

        env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
        self._env = gym.make(task_id, cfg=env_cfg)
        self._scene = self._env.unwrapped.scene
        self._setup_cameras()

    def _setup_cameras(self):
        """Pose cameras via look-at so we don't depend on guessing the
        right OffsetCfg quaternion. Eye/target are env-local; env_origins
        offsets per env so multi-env setups also work."""
        origins = self._scene.env_origins  # (N, 3)
        for name, (eye, target) in _CAM_VIEWS.items():
            try:
                cam = self._scene[name]
            except KeyError:
                continue
            eye_t = torch.tensor(eye, device=self.device, dtype=torch.float32)
            target_t = torch.tensor(target, device=self.device, dtype=torch.float32)
            eyes = eye_t.unsqueeze(0).expand(self.num_envs, 3) + origins
            targets = target_t.unsqueeze(0).expand(self.num_envs, 3) + origins
            try:
                cam.set_world_poses_from_view(eyes, targets)
            except AttributeError:
                pass  # older IsaacLab without this method; OffsetCfg fallback

    def _arm_proprio(self, name):
        d = self._scene[name].data
        return torch.cat([d.joint_pos, d.joint_vel], dim=-1).reshape(self.num_envs, -1)

    def _scene_outputs(self):
        """Single pass over the scene: returns (obs_dict, state_np)."""
        left = self._arm_proprio("robot_left")
        right = self._arm_proprio("robot_right")

        # Convert cube positions to env-local so cell_labels math works for num_envs > 1.
        origins = self._scene.env_origins  # (num_envs, 3) world-frame env origins
        cube_parts = []
        for k in _CUBE_KEYS:
            d = self._scene[k].data
            cube_parts.extend([d.root_pos_w - origins, d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_w])
        cubes = torch.cat([p.reshape(self.num_envs, -1) for p in cube_parts], dim=-1)

        state = torch.cat([left, right, cubes], dim=-1).detach().cpu().numpy()
        left_np = left.detach().cpu().numpy()
        right_np = right.detach().cpu().numpy()

        visual = {
            v: self._scene[c].data.output["rgb"].detach().cpu().numpy()
            for v, c in zip(_VIS_KEYS, _CAMERA_KEYS)
        }
        if self.cooperative:
            both = np.concatenate([left_np, right_np], axis=-1)
            proprio = {"left": both, "right": both}
        else:
            proprio = {"left": left_np, "right": right_np}
        return {"visual": visual, "proprio": proprio}, state

    def _write_state(self, state):
        t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        i = 0
        for name in ("robot_left", "robot_right"):
            self._scene[name].write_joint_state_to_sim(
                t[:, i : i + _ARM_JOINT_DIM],
                t[:, i + _ARM_JOINT_DIM : i + 2 * _ARM_JOINT_DIM],
            )
            i += 2 * _ARM_JOINT_DIM
        # State stores env-local cube pos; sim write_root_state_to_sim wants world frame.
        origins = self._scene.env_origins
        for k in _CUBE_KEYS:
            block = t[:, i : i + _CUBE_DIM].clone()
            block[:, :3] = block[:, :3] + origins
            self._scene[k].write_root_state_to_sim(block)
            i += _CUBE_DIM

    def _action_tensor(self, action_dict):
        left = np.atleast_2d(np.asarray(action_dict["left"], dtype=np.float32))
        right = np.atleast_2d(np.asarray(action_dict["right"], dtype=np.float32))
        a = torch.as_tensor(np.concatenate([left, right], axis=-1), device=self.device)
        if a.shape[0] == 1 and self.num_envs > 1:
            a = a.expand(self.num_envs, -1)
        return a

    def get_ee_positions(self) -> dict:
        """Returns {"left": (N,3), "right": (N,3)} env-local panda_hand
        (wrist) positions. Matches the IK action term's control point after
        the body_offset was removed — both the IK and this getter now
        reference the wrist body, so policy deltas are consistent with what
        the IK is trying to achieve. The gripper tip sits ~10.7cm below the
        wrist along the gripper axis; policies should target wrist heights
        accordingly (e.g. wrist at cube_z+0.107 to put gripper tip at cube).
        """
        origins = self._scene.env_origins  # (N, 3)
        out = {}
        for side in ("left", "right"):
            robot = self._scene[f"robot_{side}"]
            hand_idx = robot.body_names.index("panda_hand")
            data = robot.data
            if hasattr(data, "body_pos_w"):
                hand_pos = data.body_pos_w[:, hand_idx, :]
            else:
                hand_pos = data.body_link_pose_w[:, hand_idx, :3]
            out[side] = (hand_pos - origins).detach().cpu().numpy()
        return out

    def get_cube_positions(self) -> dict:
        """Returns {cube_red: (N,3), cube_blue: (N,3)} env-local cube xyz positions."""
        origins = self._scene.env_origins
        return {
            k: (self._scene[k].data.root_pos_w - origins).detach().cpu().numpy()
            for k in _CUBE_KEYS
        }

    def world_to_base_delta(self, side: str, delta_world) -> np.ndarray:
        """Rotate world-frame xyz delta(s) into the given arm's base frame.

        Accepts (3,) for a single delta (broadcast to all envs) or (N, 3)
        for per-env deltas. Returns the same shape as input.
        """
        import isaaclab.utils.math as math_utils
        robot = self._scene[f"robot_{side}"]
        root_quat_inv = math_utils.quat_inv(robot.data.root_quat_w)  # (N, 4)
        delta_w_t = torch.as_tensor(
            np.asarray(delta_world, dtype=np.float32), device=self.device
        )
        is_1d = (delta_w_t.ndim == 1)
        if is_1d:
            delta_w_t = delta_w_t.unsqueeze(0)
        if delta_w_t.shape[0] == 1 and root_quat_inv.shape[0] > 1:
            delta_w_t = delta_w_t.expand(root_quat_inv.shape[0], -1)
        delta_b = math_utils.quat_apply(root_quat_inv, delta_w_t)
        out = delta_b.detach().cpu().numpy()
        return out[0] if is_1d else out

    def _ee_quat_base_t(self, side: str):
        """Current panda_hand orientation in the arm's base frame, (N, 4) wxyz tensor."""
        import isaaclab.utils.math as math_utils
        robot = self._scene[f"robot_{side}"]
        hand_idx = robot.body_names.index("panda_hand")
        data = robot.data
        if hasattr(data, "body_quat_w"):
            ee_quat_w = data.body_quat_w[:, hand_idx, :]
        else:
            ee_quat_w = data.body_link_pose_w[:, hand_idx, 3:7]
        return math_utils.quat_mul(math_utils.quat_inv(data.root_quat_w), ee_quat_w)

    def get_ee_quats_base(self) -> dict:
        """{"left": (N,4), "right": (N,4)} base-frame panda_hand quaternions (wxyz).
        Used to capture a reference orientation to hold the gripper at."""
        return {s: self._ee_quat_base_t(s).detach().cpu().numpy() for s in ("left", "right")}

    def orientation_delta_base(self, side: str, ref_quat_base) -> np.ndarray:
        """Axis-angle (N,3) in the base frame that rotates the current EE
        orientation onto ref_quat_base. Fed to the IK's relative-pose rotation
        command so the gripper is actively held at the reference orientation
        (closed loop) instead of drifting — a zero rotation command provides no
        orientation feedback at all, which let the wrist drift/spin."""
        import isaaclab.utils.math as math_utils
        cur = self._ee_quat_base_t(side)  # (N, 4)
        ref = torch.as_tensor(np.asarray(ref_quat_base, dtype=np.float32), device=self.device)
        if ref.ndim == 1:
            ref = ref.unsqueeze(0).expand(cur.shape[0], -1)
        delta = math_utils.quat_mul(ref, math_utils.quat_inv(cur))
        return math_utils.axis_angle_from_quat(delta).detach().cpu().numpy()

    # ---------- dino_wm contract ----------

    def _randomize_cube_zones(self):
        """Reset hook: randomly permute the 3 cubes across the left / center
        (spare) / right zones so the colour each robot grabs rotates per
        episode. Writes env-local cube root states (zone x + jitter, random
        yaw, zero velocity); overrides the per-cube jitter from EventCfg.
        Only runs on the reset() path — set_init_state() writes explicit
        states and is unaffected."""
        N, dev = self.num_envs, self.device
        origins = self._scene.env_origins            # (N, 3) world
        cx, cy = GRID_CENTER_XY
        # (N, n_cubes): each row a random permutation of the zone indices, so
        # the cubes occupy the zones (one each) in random order. n_cubes ==
        # len(_CUBE_ZONE_DX) == len(_CUBE_KEYS) (2 cubes -> 2 zones).
        n_cubes = len(_CUBE_KEYS)
        perms = torch.argsort(torch.rand(N, n_cubes, device=dev), dim=1)
        zone_dx = torch.tensor(_CUBE_ZONE_DX, device=dev)
        for k, key in enumerate(_CUBE_KEYS):
            dx = zone_dx[perms[:, k]]                 # (N,)
            jx = (torch.rand(N, device=dev) * 2 - 1) * _CUBE_ZONE_JITTER_X
            jy = (torch.rand(N, device=dev) * 2 - 1) * _CUBE_ZONE_JITTER_Y
            x = cx + dx + jx
            y = cy + jy
            z = torch.full((N,), _CUBE_SPAWN_Z, device=dev)
            pos_world = torch.stack([x, y, z], dim=1) + origins   # (N, 3)
            yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
            qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)
            zeros = torch.zeros_like(qw)
            quat = torch.stack([qw, zeros, zeros, qz], dim=1)     # (N, 4) wxyz
            vel = torch.zeros(N, 6, device=dev)
            block = torch.cat([pos_world, quat, vel], dim=1)      # (N, 13)
            self._scene[key].write_root_state_to_sim(block)

    def reset(self):
        self._env.reset()
        self._randomize_cube_zones()
        return self._scene_outputs()

    def step(self, action_dict):
        _, _, terminated, truncated, _ = self._env.step(self._action_tensor(action_dict))
        done = (terminated | truncated).detach().cpu().numpy()
        obs, state = self._scene_outputs()
        zero = np.zeros((self.num_envs,), dtype=np.float32)
        return obs, {"left": zero, "right": zero}, done, {"state": state}

    def set_init_state(self, init_state):
        self._env.reset()
        self._write_state(np.asarray(init_state))
        return self._scene_outputs()[0]

    def prepare(self, seed, init_state):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        self.set_init_state(init_state)
        return self._scene_outputs()

    def rollout(self, seed, init_state, actions_dict):
        """actions_dict: {"left": (T,7), "right": (T,7)} or {"left": (N,T,7), "right": (N,T,7)}."""
        obs, state = self.prepare(seed, init_state)
        visuals = {k: [obs["visual"][k]] for k in _VIS_KEYS}
        proprios = {k: [obs["proprio"][k]] for k in ("left", "right")}
        states = [state]

        left = np.asarray(actions_dict["left"])
        right = np.asarray(actions_dict["right"])
        if left.ndim == 2:  # (T, 7) -> (1, T, 7)
            left = left[None]
            right = right[None]
        for t in range(left.shape[1]):
            o, _, _, info = self.step({"left": left[:, t], "right": right[:, t]})
            for k in _VIS_KEYS:
                visuals[k].append(o["visual"][k])
            for k in ("left", "right"):
                proprios[k].append(o["proprio"][k])
            states.append(info["state"])
        return (
            {
                "visual": {k: np.stack(v, axis=1) for k, v in visuals.items()},
                "proprio": {k: np.stack(v, axis=1) for k, v in proprios.items()},
            },
            np.stack(states, axis=1),
        )

    def update_env(self, env_info):
        pass

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        return [seed]
