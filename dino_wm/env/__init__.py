from gym.envs.registration import register

# point_maze depends on gym.envs.mujoco -> mujoco_py, which is not installed in
# the IsaacLab docker container. Make this registration opt-out so other envs
# (including isaaclab_grid) can still load when mujoco_py is missing.
try:
    from .pointmaze import U_MAZE
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
except Exception:
    # gym raises gym.error.DependencyNotInstalled (not ImportError) when
    # mujoco_py is missing, so catch broadly.
    pass

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# Opt-in: only register when ISAACLAB_AVAILABLE is set. Entry point is a string
# so importing env/__init__.py does not import IsaacLab or boot Omniverse.
import os
if os.environ.get("ISAACLAB_AVAILABLE"):
    register(
        id="isaaclab_grid",
        entry_point="env.isaaclab.grid_wrapper:GridWrapper",
        max_episode_steps=300,
    )