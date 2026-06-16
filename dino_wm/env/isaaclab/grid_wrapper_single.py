"""Adapter: IsaacLab DinoWMGrid-Single -> dino_wm contract (single robot).

Single-robot base case: ONE Franka, ONE cube, ONE 45deg camera facing the
robot across the 3x3 grid, plus a static octagonal "rule" sign (recolored at
runtime via set_sign_color; placeholder, no behavioral effect yet).

step(action(N,7)) -> (obs, reward(N,), done(N,), info). action is the 7-D
[pos_dx, pos_dy, pos_dz, rot_dx, rot_dy, rot_dz, gripper] vector (IK pose-mode
+ binary gripper), same per-agent layout as the two-arm wrapper.

State layout (31-D); cube position stored env-local (root_pos_w - env_origins)
so cell-label math is correct for any num_envs:
    [arm_jpos(9), arm_jvel(9), cube(13)]
cube block = [pos(3) env-local, quat(4 wxyz), linvel(3), angvel(3)].

obs = {"visual": (N,H,W,3) rgb, "proprio": (N,18) arm jpos+jvel}.

Mirrors grid_wrapper.py (GridWrapper) reduced to a single robot.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .app_launcher import get_app
from .grid_metadata import GRID_CENTER_XY, GRID_HALF, REACH_X_MIN

_ARM_JOINT_DIM = 9
_CUBE_DIM = 13
_CUBE_KEY = "cube"
_CAMERA_KEY = "camera"
# Single cube placement: uniform within the grid, with a margin so the 9cm cube
# doesn't spawn half-off the grid edge. Re-randomized each reset().
_CUBE_RAND_HALF = GRID_HALF - 0.06   # margin >= cube half (0.045) keeps it on-grid
_CUBE_SPAWN_Z = 0.046                # 9cm cube center (half 0.045) resting on the table
# Camera look-at (env-local): eye in front of the grid at +x, ~45deg down,
# facing the robot at -x. Tuned in the smoke test. Eye dollied back 15% from
# the original (0.75,0.75) along the same view direction -> ~15% zoom out,
# same angle/framing.
_CAM_EYE = (0.8625, 0.0, 0.8625)
_CAM_TARGET = (0.0, 0.0, 0.0)
# Sign shader prim (under each env) whose diffuseColor we set at runtime.
_SIGN_SHADER_SUBPATH = "Sign/Looks/SignMaterial/Surface"

PROPRIO_DIM = 2 * _ARM_JOINT_DIM            # 18
STATE_DIM = PROPRIO_DIM + _CUBE_DIM         # 31
ACTION_DIM = 7                              # 6 arm (pose delta) + 1 gripper


class GridWrapperSingle:
    def __init__(
        self,
        task_id: str = "Isaac-DinoWMGrid-Single-v0",
        num_envs: int = 1,
        device: str = "cuda:0",
        headless: bool = True,
        render_mode: str = "PathTracing",
        spp: int = 128,
    ):
        # renderer/physics GPU follows the wrapper's device (Vulkan ignores
        # CUDA_VISIBLE_DEVICES; AppLauncher derives the active GPU from device).
        get_app(headless=headless, enable_cameras=True, render_mode=render_mode, spp=spp, device=device)

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.dinowm_grid  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self.num_envs = num_envs
        self.device = device
        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
        self.proprio_dim = PROPRIO_DIM

        env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
        self._env = gym.make(task_id, cfg=env_cfg)
        self._scene = self._env.unwrapped.scene
        self._setup_camera()

    def _setup_camera(self):
        """Pose the camera via look-at (eye/target env-local; env_origins offsets
        per env). Avoids guessing an OffsetCfg quaternion."""
        try:
            cam = self._scene[_CAMERA_KEY]
        except KeyError:
            return
        origins = self._scene.env_origins  # (N, 3)
        eye = torch.tensor(_CAM_EYE, device=self.device, dtype=torch.float32)
        target = torch.tensor(_CAM_TARGET, device=self.device, dtype=torch.float32)
        eyes = eye.unsqueeze(0).expand(self.num_envs, 3) + origins
        targets = target.unsqueeze(0).expand(self.num_envs, 3) + origins
        try:
            cam.set_world_poses_from_view(eyes, targets)
        except AttributeError:
            pass  # older IsaacLab without this method; OffsetCfg fallback

    def _arm_proprio(self):
        d = self._scene["robot"].data
        return torch.cat([d.joint_pos, d.joint_vel], dim=-1).reshape(self.num_envs, -1)

    def _scene_outputs(self):
        """Single pass over the scene: returns (obs_dict, state_np)."""
        arm = self._arm_proprio()  # (N, 18)

        origins = self._scene.env_origins  # (N, 3) world-frame env origins
        d = self._scene[_CUBE_KEY].data
        cube = torch.cat(
            [d.root_pos_w - origins, d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_w], dim=-1
        ).reshape(self.num_envs, -1)  # (N, 13)

        state = torch.cat([arm, cube], dim=-1).detach().cpu().numpy()  # (N, 31)
        visual = self._scene[_CAMERA_KEY].data.output["rgb"].detach().cpu().numpy()
        proprio = arm.detach().cpu().numpy()
        return {"visual": visual, "proprio": proprio}, state

    def _write_state(self, state):
        t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        self._scene["robot"].write_joint_state_to_sim(
            t[:, :_ARM_JOINT_DIM],
            t[:, _ARM_JOINT_DIM : 2 * _ARM_JOINT_DIM],
        )
        # State stores env-local cube pos; write_root_state_to_sim wants world frame.
        origins = self._scene.env_origins
        block = t[:, 2 * _ARM_JOINT_DIM : 2 * _ARM_JOINT_DIM + _CUBE_DIM].clone()
        block[:, :3] = block[:, :3] + origins
        self._scene[_CUBE_KEY].write_root_state_to_sim(block)

    def _materialize_state(self):
        """Flush written state to PhysX, step+render once, refresh sensor
        buffers so obs reflect the written/randomized state (not the stale
        pre-write reset frame). See GridWrapper._materialize_state."""
        self._scene.write_data_to_sim()
        sim = self._env.unwrapped.sim
        sim.step(render=True)
        self._scene.update(sim.get_physics_dt())

    def _action_tensor(self, action):
        a = np.atleast_2d(np.asarray(action, dtype=np.float32))  # (N,7) or (1,7)
        t = torch.as_tensor(a, device=self.device)
        if t.shape[0] == 1 and self.num_envs > 1:
            t = t.expand(self.num_envs, -1)
        return t

    # ---------- sign rule API (placeholder) ----------

    def set_sign_color(self, rgb):
        """Set the octagonal sign's diffuseColor (per env) at runtime, then
        render so the change is visible. rgb is a 3-tuple in [0,1]. Placeholder
        "new rules" signal — no behavioral effect yet."""
        from pxr import Gf, Sdf, UsdShade

        sim = self._env.unwrapped.sim
        stage = sim.stage
        col = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        for i in range(self.num_envs):
            path = f"/World/envs/env_{i}/{_SIGN_SHADER_SUBPATH}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            shader = UsdShade.Shader(prim)
            inp = shader.GetInput("diffuseColor")
            if not inp:
                inp = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            inp.Set(col)
        sim.step(render=True)
        self._scene.update(sim.get_physics_dt())

    # ---------- IK / geometry helpers (single robot) ----------

    def get_ee_positions(self) -> np.ndarray:
        """(N,3) env-local panda_hand (wrist) position. The IK action term
        references the wrist body (body_offset removed), so policy deltas are
        consistent with this. Gripper tip sits ~10.7cm below the wrist."""
        origins = self._scene.env_origins
        robot = self._scene["robot"]
        hand_idx = robot.body_names.index("panda_hand")
        data = robot.data
        if hasattr(data, "body_pos_w"):
            hand_pos = data.body_pos_w[:, hand_idx, :]
        else:
            hand_pos = data.body_link_pose_w[:, hand_idx, :3]
        return (hand_pos - origins).detach().cpu().numpy()

    def get_cube_positions(self) -> np.ndarray:
        """(N,3) env-local cube xyz position."""
        origins = self._scene.env_origins
        return (self._scene[_CUBE_KEY].data.root_pos_w - origins).detach().cpu().numpy()

    def get_joint_diag(self):
        """Debug: (joint_pos (J,), names list, limits (J,2) or None) for env 0 —
        to see which joints pin at their limits during a push."""
        robot = self._scene["robot"]
        d = robot.data
        jp = d.joint_pos[0].detach().cpu().numpy()
        names = list(robot.joint_names)
        lim = None
        for attr in ("soft_joint_pos_limits", "joint_pos_limits"):
            if hasattr(d, attr):
                lim = getattr(d, attr)[0].detach().cpu().numpy()
                break
        return jp, names, lim

    def world_to_base_delta(self, delta_world) -> np.ndarray:
        """Rotate world-frame xyz delta(s) into the robot's base frame. Accepts
        (3,) (broadcast to all envs) or (N,3); returns the input shape."""
        import isaaclab.utils.math as math_utils
        robot = self._scene["robot"]
        root_quat_inv = math_utils.quat_inv(robot.data.root_quat_w)  # (N,4)
        delta_w_t = torch.as_tensor(np.asarray(delta_world, dtype=np.float32), device=self.device)
        is_1d = (delta_w_t.ndim == 1)
        if is_1d:
            delta_w_t = delta_w_t.unsqueeze(0)
        if delta_w_t.shape[0] == 1 and root_quat_inv.shape[0] > 1:
            delta_w_t = delta_w_t.expand(root_quat_inv.shape[0], -1)
        delta_b = math_utils.quat_apply(root_quat_inv, delta_w_t)
        out = delta_b.detach().cpu().numpy()
        return out[0] if is_1d else out

    def _ee_quat_base_t(self):
        """Current panda_hand orientation in the base frame, (N,4) wxyz tensor."""
        import isaaclab.utils.math as math_utils
        robot = self._scene["robot"]
        hand_idx = robot.body_names.index("panda_hand")
        data = robot.data
        if hasattr(data, "body_quat_w"):
            ee_quat_w = data.body_quat_w[:, hand_idx, :]
        else:
            ee_quat_w = data.body_link_pose_w[:, hand_idx, 3:7]
        return math_utils.quat_mul(math_utils.quat_inv(data.root_quat_w), ee_quat_w)

    def get_ee_quats_base(self) -> np.ndarray:
        """(N,4) base-frame panda_hand quaternion (wxyz). Reference orientation
        to hold the gripper at."""
        return self._ee_quat_base_t().detach().cpu().numpy()

    def orientation_delta_base(self, ref_quat_base) -> np.ndarray:
        """Axis-angle (N,3) in the base frame rotating the current EE orientation
        onto ref_quat_base. Fed to the IK relative-pose rotation command to hold
        the gripper at the reference orientation (closed loop)."""
        import isaaclab.utils.math as math_utils
        cur = self._ee_quat_base_t()  # (N,4)
        ref = torch.as_tensor(np.asarray(ref_quat_base, dtype=np.float32), device=self.device)
        if ref.ndim == 1:
            ref = ref.unsqueeze(0).expand(cur.shape[0], -1)
        delta = math_utils.quat_mul(ref, math_utils.quat_inv(cur))
        return math_utils.axis_angle_from_quat(delta).detach().cpu().numpy()

    # ---------- dino_wm contract ----------

    def _randomize_cube(self):
        """Reset hook: place the single cube uniformly within the grid (random
        yaw, zero velocity). Overrides the per-cube jitter from EventCfg. Only on
        the reset() path; set_init_state() writes explicit states unaffected."""
        N, dev = self.num_envs, self.device
        origins = self._scene.env_origins  # (N,3) world
        cx, cy = GRID_CENTER_XY
        # x lower-bounded at REACH_X_MIN so the cube never spawns in the near-base
        # column the arm can't get behind (would just stall/drift).
        x_lo = max(cx - _CUBE_RAND_HALF, REACH_X_MIN)
        x_hi = cx + _CUBE_RAND_HALF
        x = x_lo + torch.rand(N, device=dev) * (x_hi - x_lo)
        y = cy + (torch.rand(N, device=dev) * 2 - 1) * _CUBE_RAND_HALF
        z = torch.full((N,), _CUBE_SPAWN_Z, device=dev)
        pos_world = torch.stack([x, y, z], dim=1) + origins  # (N,3)
        yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)
        zeros = torch.zeros_like(qw)
        quat = torch.stack([qw, zeros, zeros, qz], dim=1)  # (N,4) wxyz
        vel = torch.zeros(N, 6, device=dev)
        block = torch.cat([pos_world, quat, vel], dim=1)  # (N,13)
        self._scene[_CUBE_KEY].write_root_state_to_sim(block)

    def reset(self):
        self._env.reset()
        self._randomize_cube()
        self._materialize_state()
        return self._scene_outputs()

    def step(self, action):
        _, _, terminated, truncated, _ = self._env.step(self._action_tensor(action))
        done = (terminated | truncated).detach().cpu().numpy()
        obs, state = self._scene_outputs()
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        return obs, reward, done, {"state": state}

    def set_init_state(self, init_state):
        self._env.reset()
        self._write_state(np.asarray(init_state))
        self._materialize_state()
        return self._scene_outputs()[0]

    def prepare(self, seed, init_state):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        self.set_init_state(init_state)
        return self._scene_outputs()

    def rollout(self, seed, init_state, actions):
        """actions: (T,7) or (N,T,7)."""
        obs, state = self.prepare(seed, init_state)
        visuals = [obs["visual"]]
        proprios = [obs["proprio"]]
        states = [state]

        a = np.asarray(actions)
        if a.ndim == 2:  # (T,7) -> (1,T,7)
            a = a[None]
        for t in range(a.shape[1]):
            o, _, _, info = self.step(a[:, t])
            visuals.append(o["visual"])
            proprios.append(o["proprio"])
            states.append(info["state"])
        return (
            {
                "visual": np.stack(visuals, axis=1),
                "proprio": np.stack(proprios, axis=1),
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
