"""Physics-only (no-camera) grid env for the SIM-ORACLE baseline.

Subclasses GridWrapperSingle but boots `enable_cameras=False` and returns cube STATE straight from
physics (no render). That removes the RTX render-product ceiling, so RRT can roll B candidate strokes
FROM A TREE NODE in one physics batch and read their true end positions -- the "perfect world model"
that the sim-oracle RRT plans with. NOTHING in the main pipeline is modified: this only subclasses +
overrides the three camera-touching methods and adds a rollout helper.

De-risk before building the rest: this file must first be shown to boot + step physics without a
camera in the container (see the smoke test in the module docstring of run_sim_oracle.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]          # .../dino_wm
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from env.isaaclab.app_launcher import get_app
from env.isaaclab.grid_wrapper_single import (       # reuse -- do NOT copy
    GridWrapperSingle, ACTION_DIM, STATE_DIM, PROPRIO_DIM, _CUBE_KEY)

_CUBE_XY = slice(18, 20)


class PhysGridEnv(GridWrapperSingle):
    """GridWrapperSingle with the camera stripped: physics only, state-only outputs, no render.

    Inherits the stroke machinery unchanged (execute_stroke / StrokeExecutor / _write_state /
    _clamp_cube / arm-park snap). Only the three methods that touch the camera or render are
    overridden, plus a `rollout` helper the sim-oracle RRT calls per tree node."""

    def __init__(self, num_envs, device="cuda:0",
                 task_id="Isaac-DinoWMGrid-Single-v0", stroke_max_steps=320):
        # Physics-only boot: enable_cameras=False -> no render products -> no RTX num_envs ceiling.
        # get_app is idempotent, so this MUST be the first (only) sim boot in the process.
        get_app(headless=True, enable_cameras=False, device=device)

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.dinowm_grid  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self.num_envs = num_envs
        self.device = device
        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
        self.proprio_dim = PROPRIO_DIM
        self.stroke_max_steps = stroke_max_steps
        self.fast_stroke_render = True     # inherited execute_stroke skips per-step render; we never render
        self._executor = None
        self._home_ee = None
        self._home_jp = None

        env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
        # Drop the camera sensor: physics-only. With enable_cameras=False the sensor is not rendered;
        # removing it from the cfg avoids any render-product allocation and keeps _scene[camera] absent
        # (so the inherited _setup_camera no-ops via its KeyError guard).
        if getattr(env_cfg.scene, "camera", None) is not None:
            env_cfg.scene.camera = None
        self._env = gym.make(task_id, cfg=env_cfg)
        self._scene = self._env.unwrapped.scene
        # No _setup_camera() call -- there is no camera to pose.

    # ---- overrides: never touch the camera / render ----

    def _materialize_state(self):
        """Flush written state to PhysX + step physics with NO render, then refresh buffers.
        (Base steps with render=True to update the camera; we have none.)"""
        self._scene.write_data_to_sim()
        sim = self._env.unwrapped.sim
        sim.step(render=False)
        self._scene.update(sim.get_physics_dt())

    def _scene_outputs(self):
        """State-only (N,31): arm proprio + cube block from physics. No visual (no camera)."""
        arm = self._arm_proprio()                                  # (N,18)
        origins = self._scene.env_origins
        d = self._scene[_CUBE_KEY].data
        cube = torch.cat([d.root_pos_w - origins, d.root_quat_w,
                          d.root_lin_vel_w, d.root_ang_vel_w], dim=-1).reshape(self.num_envs, -1)  # (N,13)
        state = torch.cat([arm, cube], dim=-1).detach().cpu().numpy()   # (N,31)
        return {"visual": None, "proprio": arm.detach().cpu().numpy()}, state

    # ---- sim-oracle helper ----

    def roll_strokes(self, state, strokes):
        """Restore `state` to ALL envs, apply per-env `strokes`, return the resulting states (N,31).
        (NOT an override of GridWrapperSingle.rollout -- distinct name + signature to avoid confusion.)

        state:   (31,) restored to every env (a tree node's full state; broadcast by _write_state).
        strokes: (N,4) per-env aimed strokes [x_start,y_start,dx,dy] (N == num_envs; for a single
                 executed stroke, pass it broadcast and read row 0). GT physics, no render, no probe.
        The arm is snapped back to the episode's home joints after the push (inherited execute_stroke),
        so returned states carry the parked arm + true pushed cube -- matching the training frames."""
        self._write_state(np.asarray(state, dtype=np.float32))
        self._materialize_state()
        _, state_out = self.execute_stroke(np.asarray(strokes, dtype=np.float32))
        return state_out
