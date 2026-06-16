"""Vector env shim for DinoWMGrid. One process, GPU-batched sims."""
from __future__ import annotations

import numpy as np

from .grid_wrapper import GridWrapper, ACTION_DIM_PER_AGENT
from .grid_metadata import cell_labels_from_states


class GridVectorEnv:
    def __init__(self, num_envs, task_id="Isaac-DinoWMGrid-v0",
                 device="cuda:0", cooperative=True, camera="left"):
        self.env_num = num_envs
        self.cooperative = cooperative
        # Which camera stream the world model was trained on (dataset `camera`).
        self.camera = camera
        self._w = GridWrapper(
            task_id=task_id, num_envs=num_envs,
            device=device, cooperative=cooperative,
        )

    def _flatten_obs(self, obs):
        """Collapse the wrapper's nested per-camera/per-agent obs into the flat
        {"visual": ndarray, "proprio": ndarray} the WM/planner expects.

        - visual: select the single trained camera (self.camera).
        - proprio: the joint vector. In cooperative mode both "left"/"right"
          entries are the same concatenated [left, right] (36-D), matching the
          dataset's cat([proprio_left, proprio_right]).
        """
        return {
            "visual": obs["visual"][self.camera],
            "proprio": obs["proprio"]["left"],
        }

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
        obs, state = self._w.prepare(None, np.asarray(init_states))
        return self._flatten_obs(obs), state

    def rollout(self, seeds, init_states, actions_dict, id=None):
        s = seeds[0] if hasattr(seeds, "__getitem__") else seeds
        # The generic planner passes one concatenated action array
        # (..., 14) laid out as [left(7), right(7)] (see datasets/isaaclab_grid_dset
        # which builds actions via cat([actions_left, actions_right], dim=-1)).
        # GridWrapper.rollout expects {"left": (...,7), "right": (...,7)}.
        if not isinstance(actions_dict, dict):
            a = np.asarray(actions_dict)
            d = ACTION_DIM_PER_AGENT
            actions_dict = {"left": a[..., :d], "right": a[..., d:2 * d]}
        obs, states = self._w.rollout(s, init_states, actions_dict)
        return self._flatten_obs(obs), states

    def eval_state(self, goal_state, cur_state):
        """Score goal-reaching per env. Batched over n_evals.

        Success = both cubes occupy their goal grid cells (discrete 3x3 cell
        match, the natural pick-and-place criterion). Velocity/quaternion parts
        of the 62-D state are intentionally ignored for success.

        goal_state, cur_state: (b, 62). Returns dict of (b,) arrays.
        """
        goal_cells = cell_labels_from_states(np.asarray(goal_state))  # (b, 2)
        cur_cells = cell_labels_from_states(np.asarray(cur_state))    # (b, 2)
        cube_match = goal_cells == cur_cells                          # (b, 2)
        success = np.all(cube_match, axis=-1)                         # (b,)
        cubes_correct = np.sum(cube_match, axis=-1)                   # (b,)
        state_dist = np.linalg.norm(
            np.asarray(goal_state) - np.asarray(cur_state), axis=-1
        )                                                             # (b,)
        return {
            "success": success,
            "cubes_correct": cubes_correct,
            "state_dist": state_dist,
        }

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
