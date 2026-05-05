import sys
sys.path.insert(0, '/newdata2/dylantw/Legislative-Harness/dino_wm')

import numpy as np
import pickle
import torch
import gym
from env.pointmaze.point_maze_wrapper import PointMazeWrapper

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

env = PointMazeWrapper(maze_spec=U_MAZE)
env.prepare_for_render()

init_state = np.array([1.0, 1.0, 0.0, 0.0])
goal_state = np.array([3.0, 1.0, 0.0, 0.0])

obs_0, state_0 = env.prepare(seed=99, init_state=init_state)
obs_g, state_g = env.prepare(seed=99, init_state=goal_state)

# add batch and time dimensions — n_evals=5
n_evals = 5
plan_targets = {
    "obs_0": {k: np.stack([np.expand_dims(v, 0)] * n_evals) for k, v in obs_0.items()},
    "obs_g": {k: np.stack([np.expand_dims(v, 0)] * n_evals) for k, v in obs_g.items()},
    "state_0": np.stack([init_state] * n_evals),
    "state_g": np.stack([goal_state] * n_evals),
    "gt_actions": None,
    "goal_H": 5,
}

with open("plan_targets.pkl", "wb") as f:
    pickle.dump(plan_targets, f)

print("Saved plan_targets.pkl")
