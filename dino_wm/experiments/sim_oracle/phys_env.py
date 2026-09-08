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

import os
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
                 task_id="Isaac-DinoWMGrid-Single-v0", stroke_max_steps=320,
                 grid_away_shift=None, lock_cube_yaw=None):
        # grid_away_shift: override the robot-base away-from-grid shift (m) for THIS env only, WITHOUT
        # editing the shared cfg (dinowm_grid_env_cfg._GRID_AWAY_SHIFT). None = use the cfg default
        # (currently 0.025 -> base x=-0.475). 0.0 = the pre-shift -0.45 geometry. Lets the oracle
        # ceiling be measured at a chosen geometry (e.g. reproduce the pre-shift ceiling, or A/B the
        # shift) instead of silently inheriting whatever the cfg constant happens to be.
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
        # YAW LOCK. Resolved exactly as GridWrapperSingle does (explicit arg, else env var, else off).
        # REQUIRED even when off: this __init__ bypasses GridWrapperSingle.__init__, but the inherited
        # execute_stroke() and set_init_state() both read self.lock_cube_yaw, so leaving it unset makes
        # the oracle raise AttributeError on its first stroke.
        self.lock_cube_yaw = (bool(int(os.environ.get("DINOWM_LOCK_CUBE_YAW", "0")))
                              if lock_cube_yaw is None else bool(lock_cube_yaw))

        env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
        # Drop the camera sensor: physics-only. With enable_cameras=False the sensor is not rendered;
        # removing it from the cfg avoids any render-product allocation and keeps _scene[camera] absent
        # (so the inherited _setup_camera no-ops via its KeyError guard).
        if getattr(env_cfg.scene, "camera", None) is not None:
            env_cfg.scene.camera = None

        if self.lock_cube_yaw:
            rp = getattr(getattr(getattr(env_cfg.scene, "cube", None), "spawn", None),
                         "rigid_props", None)
            if rp is None:
                raise RuntimeError(f"lock_cube_yaw=True but {task_id} has no scene.cube.spawn."
                                   "rigid_props to clamp -- refusing to run half-locked.")
            rp.max_angular_velocity = 0.0
            rp.angular_damping = 1000.0
            print("[PhysGridEnv] cube rotation LOCKED (solver clamp + per-step re-pin)", flush=True)

        # ---- optional geometry intervention: move the robot base (reachability A/B) ----
        # The base x is a class-def-time default baked from dinowm_grid_env_cfg._ROBOT_BASE_X
        # (= -0.45 - _GRID_AWAY_SHIFT), so it can't be changed by setting the module var post-import.
        # Instead we mutate the INSTANTIATED cfg here (same pattern as the camera drop above).
        from isaaclab_tasks.dinowm_grid import dinowm_grid_env_cfg as _cfgmod
        self.grid_away_shift = _cfgmod._GRID_AWAY_SHIFT if grid_away_shift is None else float(grid_away_shift)
        _orig_base_x = _cfgmod._ROBOT_BASE_X + _cfgmod._GRID_AWAY_SHIFT   # recover the un-shifted base (-0.45)
        self.robot_base_x = _orig_base_x - self.grid_away_shift
        if grid_away_shift is not None:
            r = env_cfg.scene.robot
            p = r.init_state.pos
            r.init_state.pos = (float(self.robot_base_x), p[1], p[2])   # only the arm base; pedestal is cosmetic (camera-off)
            print(f"[PhysGridEnv] GEOMETRY OVERRIDE: grid_away_shift {_cfgmod._GRID_AWAY_SHIFT:.3f} -> "
                  f"{self.grid_away_shift:.3f} => robot base x {p[0]:.3f} -> {self.robot_base_x:.3f} "
                  f"(pedestal left in place, camera-off)", flush=True)

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

    def roll_strokes(self, state, strokes, pos_sink=None):
        """Restore `state` to ALL envs, apply per-env `strokes`, return the resulting states (N,31).
        (NOT an override of GridWrapperSingle.rollout -- distinct name + signature to avoid confusion.)

        state:   (31,) restored to every env (a tree node's full state; broadcast by _write_state).
        strokes: (N,4) per-env aimed strokes [x_start,y_start,dx,dy] (N == num_envs; for a single
                 executed stroke, pass it broadcast and read row 0). GT physics, no render, no probe.
        The arm is snapped back to the episode's home joints after the push (inherited execute_stroke),
        so returned states carry the parked arm + true pushed cube -- matching the training frames.

        RESIDUE CLEAR (the reset + double write below). A single write+materialize does NOT land PhysX
        in a state consistent with the written values: contact/solver state from the PREVIOUS roll
        survives and moves THIS roll's outcome by up to 67mm, so the same (state, stroke) rolled twice
        gave different answers depending on where it sat in the call sequence. That is what made the
        sim-oracle prune on one endpoint and execute another (58/58 committed strokes of
        results/no_yaw/sign_change/oracle entered a cell its own verdict forbade).

        Measured with experiments/sim_oracle/probe_batch_determinism.py --variants, B=64, same args
        rolled twice:
            nothing (the old body)              67.095 mm   23/64 exact
            _env.reset() only                   67.095 mm   25/64      <- reset ALONE does nothing
            2x _materialize_state               67.095 mm   12/64      <- extra steps ALONE do nothing
            _write_state + _materialize_state    8.719 mm   61/64
            reset + write + materialize          1.070 mm   62/64      <- == prepare(), the fix
        1.07mm is the true GPU-solver floor. Cost is ~0.1%: reset is 4.5ms against a 5543ms roll.

        This is exactly what set_init_state (grid_wrapper_single.py:538-544) does, which is why the WM
        arm was never affected -- mpc.py rolls each committed stroke through env.rollout(), which opens
        with prepare(). Only the oracle's tree build teleports repeatedly through THIS path.
        _home_ee/_home_jp are deliberately NOT recaptured: the park pose belongs to the episode."""
        st = np.asarray(state, dtype=np.float32)
        self._env.reset()
        self._write_state(st)
        self._materialize_state()
        self._write_state(st)
        self._materialize_state()
        _, state_out = self.execute_stroke(np.asarray(strokes, dtype=np.float32), pos_sink=pos_sink)
        return state_out
