"""Adapter: IsaacLab DinoWMGrid -> dino_wm contract, per-agent dict I/O.

step({"left": (N,7), "right": (N,7)}) -> (obs, reward_dict, done, info).
Wrapper concatenates per-agent actions into IsaacLab's 14-D action.

State layout (75-D); cube positions stored in env-local frame (root_pos_w minus
env_origins) so cell-label math is correct for any num_envs:
    [left_jpos(9), left_jvel(9), right_jpos(9), right_jvel(9),
     cube_red(13), cube_yellow(13), cube_blue(13)]
each cube block = [pos(3) env-local, quat(4 wxyz), linvel(3), angvel(3)].

cooperative=True  -> obs["proprio"] = {"left": both(36), "right": both(36)}
cooperative=False -> obs["proprio"] = {"left": own(18),  "right": own(18)}
Visual obs is {"overhead": (N,H,W,3), "front": (N,H,W,3)} either way.
"""
from __future__ import annotations

import numpy as np
import torch

from .app_launcher import get_app


_ARM_JOINT_DIM = 9
_CUBE_DIM = 13
_CUBE_KEYS = ("cube_red", "cube_yellow", "cube_blue")
_CAMERA_KEYS = ("camera_overhead", "camera_front")
_VIS_KEYS = ("overhead", "front")

PROPRIO_DIM_OWN = 2 * _ARM_JOINT_DIM  # 18
STATE_DIM = 2 * PROPRIO_DIM_OWN + len(_CUBE_KEYS) * _CUBE_DIM  # 75
ACTION_DIM_PER_AGENT = 7
ACTION_DIM_JOINT = 2 * ACTION_DIM_PER_AGENT  # 14


class GridWrapper:
    def __init__(
        self,
        task_id: str = "Isaac-DinoWMGrid-v0",
        num_envs: int = 1,
        device: str = "cuda:0",
        cooperative: bool = True,
        headless: bool = True,
        renderer: str | None = None,
    ):
        get_app(headless=headless, enable_cameras=True, renderer=renderer)

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

    # ---------- dino_wm contract ----------

    def reset(self):
        self._env.reset()
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
