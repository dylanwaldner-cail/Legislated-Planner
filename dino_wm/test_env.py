import gym
from env.venv import SubprocVectorEnv
from omegaconf import OmegaConf

train_cfg = OmegaConf.load("checkpoints/outputs/point_maze/hydra.yaml")
env = SubprocVectorEnv(
    [lambda: gym.make(train_cfg.env.name, *train_cfg.env.args, **train_cfg.env.kwargs)]
)
print("env created")
obs, state = env.prepare([0], [[1.0, 1.0, 0.0, 0.0]])
print("obs:", obs)
