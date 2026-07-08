"""Vector-env shim for DinoWMGrid-Single (single robot). One process, GPU-batched.

Wraps GridWrapperSingle so the dino_wm planner/evaluator see the flat
{"visual": (N,H,W,3), "proprio": (N,18)} obs, a single 4-D planar-stroke action
[x_start, y_start, dx, dy] (end = start + displacement), and 31-D state it expects. One stroke = one
macro-step. Success = the (single) cube reaches its goal grid cell.
"""
from __future__ import annotations

import numpy as np

from .grid_wrapper_single import GridWrapperSingle, ACTION_DIM
from .grid_metadata import cell_labels_from_states_single, STATE_FIRST_CUBE_OFFSET_SINGLE


class GridVectorEnv:
    def __init__(self, num_envs, task_id="Isaac-DinoWMGrid-Single-v0",
                 device="cuda:0", tiled_camera=False, **kwargs):
        # **kwargs absorbs any legacy env-cfg keys (e.g. cooperative/camera) so an
        # old training config doesn't break construction. tiled_camera=True swaps the per-env
        # Camera for a TiledCamera (eval-only render-memory speedup; see GridWrapperSingle).
        self.env_num = num_envs
        self._w = GridWrapperSingle(task_id=task_id, num_envs=num_envs, device=device,
                                    tiled_camera=tiled_camera)

    def __len__(self):
        return self.env_num

    def reset(self):
        return self._w.reset()

    def step(self, action):
        # action: (N, 4) or (4,) stroke. GridWrapperSingle returns (obs, reward, done, info).
        return self._w.step(action)

    def prepare(self, seeds, init_states):
        seeds = np.asarray(seeds)
        if seeds.size > 0 and seeds.flat[0] is not None:
            self._w.seed(int(seeds.flat[0]))
        # GridWrapperSingle.prepare returns the already-flat (obs, state).
        return self._w.prepare(None, np.asarray(init_states))

    def rollout(self, seeds, init_states, actions, id=None, frame_sink=None):
        # The planner passes one action array (..., 4) — single robot, no left/right
        # split. GridWrapperSingle.rollout accepts (T,4) or (N,T,4) strokes.
        # frame_sink: optional list -> per-step (N,H,W,3) frames for a smooth plan video.
        s = seeds[0] if hasattr(seeds, "__getitem__") else seeds
        return self._w.rollout(s, np.asarray(init_states), np.asarray(actions), frame_sink=frame_sink)

    def eval_state(self, goal_state, cur_state):
        """Score goal-reaching per env (batched over n_evals).

        Success = the cube occupies its goal grid cell (discrete 3x3 match).
        goal_state, cur_state: (b, 31). Returns dict of (b,) arrays.
        """
        goal_cell = cell_labels_from_states_single(np.asarray(goal_state))  # (b,)
        cur_cell = cell_labels_from_states_single(np.asarray(cur_state))    # (b,)
        success = goal_cell == cur_cell                                     # (b,)
        state_dist = np.linalg.norm(
            np.asarray(goal_state) - np.asarray(cur_state), axis=-1
        )                                                                   # (b,)
        # SIM-GROUND-TRUTH cube planar L2 (m): the honest yardstick for the probe
        # objective — reported in logs so we can see whether minimizing the probe's
        # latent-space distance actually reduces the real cube->goal distance.
        o = STATE_FIRST_CUBE_OFFSET_SINGLE
        gxy = np.asarray(goal_state)[..., o:o + 2]
        cxy = np.asarray(cur_state)[..., o:o + 2]
        cube_l2 = np.linalg.norm(gxy - cxy, axis=-1)                        # (b,)
        return {
            "success": success,
            "cubes_correct": success.astype(np.int64),  # 0/1 (single cube)
            "state_dist": state_dist,
            "cube_l2": cube_l2,
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
