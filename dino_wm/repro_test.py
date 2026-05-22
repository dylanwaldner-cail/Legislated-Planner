import gym
import numpy as np
import env  # registers point_maze in the gym registry

e = gym.make("point_maze")
obs, state = e.prepare(seed=99, init_state=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
print("OK:", type(obs), type(state))
