"""Vector env shim for DinoWMGrid. One process, GPU-batched sims."""
from __future__ import annotations

import numpy as np

from .grid_wrapper import GridWrapper


class GridVectorEnv:
    def __init__(self, num_envs, task_id="Isaac-DinoWMGrid-v0",
                 device="cuda:0", cooperative=True):
        self.env_num = num_envs
        self.cooperative = cooperative
        self._w = GridWrapper(
            task_id=task_id, num_envs=num_envs,
            device=device, cooperative=cooperative,
        )

    def __len__(self):
        return self.env_num

    def reset(self):
        return self._w.reset()

    def step(self, action_dict):
        return self._w.step(action_dict)

    def prepare(self, seeds, init_states):
        seeds = np.asarray(seeds)
        if seeds.size > 0 and seeds[0] is not None:
            self._w.seed(int(seeds[0]))
        return self._w.prepare(None, np.asarray(init_states))

    def rollout(self, seeds, init_states, actions_dict, id=None):
        s = seeds[0] if hasattr(seeds, "__getitem__") else seeds
        return self._w.rollout(s, init_states, actions_dict)

    def update_env(self, env_info):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self._w.seed(int(seed if np.isscalar(seed) else seed[0]))
        return [seed]

    def render(self, **kwargs):
        return [None] * self.env_num

    def close(self):
        self._w.close()
