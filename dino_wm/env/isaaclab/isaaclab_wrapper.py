"""Adapter: IsaacLab ManagerBasedRLEnv -> dino_wm env contract.

A single wrapper holds an IsaacLab env with `num_envs` GPU-batched sims. All
methods accept and return numpy on the CPU side; tensors live on GPU internally.

State layout (DinoWMStub):
    [arm_left_joint_pos(9), arm_left_joint_vel(9),
     arm_right_joint_pos(9), arm_right_joint_vel(9),
     cube_red(13), cube_green(13), cube_blue(13)]
    -> state_dim = 75
Each cube block = [pos(3), quat(4 wxyz), linvel(3), angvel(3)].
"""

from __future__ import annotations

import numpy as np
import torch

from .app_launcher import get_app


_ARM_JOINT_DIM = 9
_CUBE_DIM = 13
_CUBE_KEYS = ("cube_red", "cube_green", "cube_blue")
STATE_DIM = 2 * (2 * _ARM_JOINT_DIM) + len(_CUBE_KEYS) * _CUBE_DIM  # 75
ACTION_DIM = 14  # 6 EE delta + 1 gripper per arm, two arms


class IsaacLabWrapper:
    def __init__(
        self,
        task_id: str = "Isaac-DinoWMStub-v0",
        num_envs: int = 1,
        image_size: int = 64,
        device: str = "cuda:0",
        headless: bool = True,
        renderer: str | None = None,
    ):
        get_app(headless=headless, enable_cameras=True, renderer=renderer)

        # Import only after app boot.
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  registers task ids
        import isaaclab_tasks.dinowm_stub  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self.num_envs = num_envs
        self.image_size = image_size
        self.device = device
        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
        self.proprio_dim = 2 * (2 * _ARM_JOINT_DIM)  # arm joints only

        env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
        self._env = gym.make(task_id, cfg=env_cfg)

    # ---------- state <-> sim ----------

    def _read_state(self) -> np.ndarray:
        scene = self._env.unwrapped.scene
        parts = [
            scene["robot_left"].data.joint_pos,
            scene["robot_left"].data.joint_vel,
            scene["robot_right"].data.joint_pos,
            scene["robot_right"].data.joint_vel,
        ]
        for k in _CUBE_KEYS:
            d = scene[k].data
            parts.extend([d.root_pos_w, d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_w])
        flat = torch.cat([p.reshape(self.num_envs, -1) for p in parts], dim=-1)
        return flat.detach().cpu().numpy()

    def _write_state(self, state: np.ndarray) -> None:
        scene = self._env.unwrapped.scene
        t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)

        i = 0
        for name in ("robot_left", "robot_right"):
            jp = t[:, i : i + _ARM_JOINT_DIM]
            jv = t[:, i + _ARM_JOINT_DIM : i + 2 * _ARM_JOINT_DIM]
            scene[name].write_joint_state_to_sim(jp, jv)
            i += 2 * _ARM_JOINT_DIM
        for k in _CUBE_KEYS:
            block = t[:, i : i + _CUBE_DIM]
            root_state = torch.cat([block[:, :7], block[:, 7:13]], dim=-1)  # pos(3)+quat(4)+linvel(3)+angvel(3)
            scene[k].write_root_state_to_sim(root_state)
            i += _CUBE_DIM

    def _read_proprio(self) -> np.ndarray:
        scene = self._env.unwrapped.scene
        parts = [
            scene["robot_left"].data.joint_pos,
            scene["robot_left"].data.joint_vel,
            scene["robot_right"].data.joint_pos,
            scene["robot_right"].data.joint_vel,
        ]
        flat = torch.cat([p.reshape(self.num_envs, -1) for p in parts], dim=-1)
        return flat.detach().cpu().numpy()

    def _read_visual(self) -> np.ndarray:
        cam = self._env.unwrapped.scene["camera"]
        img = cam.data.output["rgb"]  # (N, H, W, 3) uint8
        return img.detach().cpu().numpy()

    def _obs(self) -> dict:
        return {"visual": self._read_visual(), "proprio": self._read_proprio()}

    # ---------- dino_wm contract ----------

    def reset(self):
        self._env.reset()
        return self._obs(), self._read_state()

    def step(self, action):
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0).expand(self.num_envs, -1)
        _, reward, terminated, truncated, _ = self._env.step(a)
        done = (terminated | truncated).detach().cpu().numpy()
        info = {"state": self._read_state()}
        return self._obs(), reward.detach().cpu().numpy(), done, info

    def set_init_state(self, init_state):
        self._env.reset()
        self._write_state(np.asarray(init_state))
        return self._obs()

    def prepare(self, seed, init_state):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        self.set_init_state(init_state)
        return self._obs(), self._read_state()

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)
        obses = {"visual": [obs["visual"]], "proprio": [obs["proprio"]]}
        states = [state]
        for a in actions:
            o, _, _, info = self.step(a)
            obses["visual"].append(o["visual"])
            obses["proprio"].append(o["proprio"])
            states.append(info["state"])
        obses = {k: np.stack(v, axis=0) for k, v in obses.items()}
        return obses, np.stack(states, axis=0)

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)
        # Conservative ranges: arm joint pos/vel near defaults, cubes on the table.
        init = self._random_state(rs)
        goal = self._random_state(rs)
        return init, goal

    def _random_state(self, rs: np.random.RandomState) -> np.ndarray:
        s = np.zeros(STATE_DIM, dtype=np.float32)
        # Arms near zero (defaults); skip noise for the stub.
        offset = 4 * _ARM_JOINT_DIM
        for k_idx, _ in enumerate(_CUBE_KEYS):
            base = offset + k_idx * _CUBE_DIM
            s[base + 0] = rs.uniform(-0.15, 0.15)  # x
            s[base + 1] = rs.uniform(-0.20, 0.20)  # y
            s[base + 2] = 0.06                     # z (on table)
            s[base + 3] = 1.0                      # quat w
        return s

    def eval_state(self, goal_state, cur_state):
        goal = np.asarray(goal_state)
        cur = np.asarray(cur_state)
        # Distance on cube positions only.
        offset = 4 * _ARM_JOINT_DIM
        dists = []
        for k_idx, _ in enumerate(_CUBE_KEYS):
            base = offset + k_idx * _CUBE_DIM
            dists.append(np.linalg.norm(goal[base : base + 3] - cur[base : base + 3]))
        state_dist = float(np.mean(dists))
        return {"success": state_dist < 0.05, "state_dist": state_dist}

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
