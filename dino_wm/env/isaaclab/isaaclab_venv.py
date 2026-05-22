"""Vector env shim: one IsaacLab process, one GPU, N batched sims.

Satisfies the subset of BaseVectorEnv that plan.py / planning/evaluator.py
actually call (reset, step, rollout, prepare, sample_random_init_goal_states,
eval_state, update_env, seed, render, close, env_num, __len__).

NOT a subprocess vector env — there is no fork. The "vector" axis is the
GPU-batched num_envs dimension of a single IsaacLab env.
"""

from __future__ import annotations

import numpy as np

from .isaaclab_wrapper import IsaacLabWrapper


class IsaacLabVectorEnv:
    def __init__(
        self,
        num_envs: int,
        task_id: str = "Isaac-DinoWMStub-v0",
        image_size: int = 224,
        device: str = "cuda:0",
    ):
        self.env_num = num_envs
        self._w = IsaacLabWrapper(
            task_id=task_id, num_envs=num_envs, image_size=image_size, device=device
        )

    def __len__(self):
        return self.env_num

    def reset(self):
        return self._w.reset()

    def step(self, action):
        return self._w.step(action)

    def prepare(self, seeds, init_states):
        # seeds: (N,), init_states: (N, state_dim)
        seeds = np.asarray(seeds)
        if seeds.size > 0 and seeds[0] is not None:
            self._w.seed(int(seeds[0]))
        return self._w.prepare(None, np.asarray(init_states))

    def rollout(self, seeds, init_states, action, id=None):
        # action: (N, T, action_dim)
        obs, state = self.prepare(seeds, init_states)
        obses = {"visual": [obs["visual"]], "proprio": [obs["proprio"]]}
        states = [state]
        action = np.asarray(action)
        T = action.shape[1]
        for t in range(T):
            o, _, _, info = self._w.step(action[:, t, :])
            obses["visual"].append(o["visual"])
            obses["proprio"].append(o["proprio"])
            states.append(info["state"])
        obses = {k: np.stack(v, axis=1) for k, v in obses.items()}  # (N, T+1, ...)
        states = np.stack(states, axis=1)  # (N, T+1, state_dim)
        return obses, states

    def sample_random_init_goal_states(self, seed):
        # seed: list/array of per-env seeds
        inits, goals = [], []
        for s in np.asarray(seed).ravel():
            i, g = self._w.sample_random_init_goal_states(int(s))
            inits.append(i)
            goals.append(g)
        return np.stack(inits), np.stack(goals)

    def eval_state(self, goal_state, cur_state):
        goal = np.asarray(goal_state)
        cur = np.asarray(cur_state)
        results = [self._w.eval_state(goal[i], cur[i]) for i in range(self.env_num)]
        return {k: np.array([r[k] for r in results]) for k in results[0]}

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
