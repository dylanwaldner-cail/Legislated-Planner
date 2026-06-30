"""Adapter: IsaacLab DinoWMGrid-Single -> dino_wm contract (single robot).

Single-robot base case: ONE Franka, ONE cube, ONE 45deg camera facing the
robot across the 3x3 grid, plus a static octagonal "rule" sign (recolored at
runtime via set_sign_color; placeholder, no behavioral effect yet).

step(action(N,4)) -> (obs, reward(N,), done(N,), info). action is the 4-D
HIGH-LEVEL planar stroke [x_start, y_start, dx, dy] (env-local grid meters;
end = start + displacement). ONE stroke == ONE macro-step: StrokeExecutor (stroke_executor.py)
"compiles" it into the per-step 7-D IK commands the env actually consumes
(DifferentialIKController, pose/relative), running many internal sim steps and
returning a single boundary observation.

State layout (31-D); cube position stored env-local (root_pos_w - env_origins)
so cell-label math is correct for any num_envs:
    [arm_jpos(9), arm_jvel(9), cube(13)]
cube block = [pos(3) env-local, quat(4 wxyz), linvel(3), angvel(3)].

obs = {"visual": (N,H,W,3) rgb, "proprio": (N,18) arm jpos+jvel}.

Mirrors grid_wrapper.py (GridWrapper) reduced to a single robot.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .app_launcher import get_app
from .grid_metadata import GRID_CENTER_XY, GRID_HALF, REACH_X_MIN

_ARM_JOINT_DIM = 9
_CUBE_DIM = 13
_CUBE_KEY = "cube"
_CAMERA_KEY = "camera"
# Single cube placement: uniform within the grid, with a margin so the 9cm cube
# doesn't spawn half-off the grid edge. Re-randomized each reset().
_CUBE_RAND_HALF = GRID_HALF - 0.06   # margin >= cube half (0.045) keeps it on-grid
_CUBE_SPAWN_Z = 0.046                # 9cm cube center (half 0.045) resting on the table
# Hard containment box (env-local |x|,|y| <= this) for the cube. After each internal
# stroke step the cube is projected back inside, so it CANNOT leave the arm's reachable
# region -- no physical wall, so nothing for the paddle to hit. = GRID_HALF (cube center
# stays on-grid). The soft keep-redirect keeps it central; this is the hard backstop.
_CUBE_CLAMP_HALF = GRID_HALF
# Cube spawn edge-bias exponent (<1 pushes spawns toward grid edges/corners, >1 toward
# center, 1 = uniform). The keep-redirect makes the cube DWELL central, so an edge-biased
# spawn flattens overall cell coverage (corners were under-represented).
_SPAWN_EDGE_K = 0.5
# Camera look-at (env-local): eye in front of the grid at +x, ~45deg down,
# facing the robot at -x. Tuned in the smoke test. Eye dollied back 15% from
# the original (0.75,0.75) along the same view direction -> ~15% zoom out,
# same angle/framing.
_CAM_EYE = (0.8625, 0.0, 0.8625)
_CAM_TARGET = (0.0, 0.0, 0.0)
# Sign shader prim (under each env) whose diffuseColor we set at runtime.
_SIGN_SHADER_SUBPATH = "Sign/Looks/SignMaterial/Surface"

PROPRIO_DIM = 2 * _ARM_JOINT_DIM            # 18
STATE_DIM = PROPRIO_DIM + _CUBE_DIM         # 31
ACTION_DIM = 4                              # planar stroke [x_start, y_start, dx, dy]
_IK_CMD_DIM = 7                             # internal low-level cmd: 6 arm (pose delta) + 1 gripper


class GridWrapperSingle:
    def __init__(
        self,
        task_id: str = "Isaac-DinoWMGrid-Single-v0",
        num_envs: int = 1,
        device: str = "cuda:0",
        headless: bool = True,
        render_mode: str = "PathTracing",
        spp: int = 128,
        stroke_max_steps: int = 320,
        fast_stroke_render: bool = True,
    ):
        # renderer/physics GPU follows the wrapper's device (Vulkan ignores
        # CUDA_VISIBLE_DEVICES; AppLauncher derives the active GPU from device).
        get_app(headless=headless, enable_cameras=True, render_mode=render_mode, spp=spp, device=device)

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.dinowm_grid  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self.num_envs = num_envs
        self.device = device
        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
        self.proprio_dim = PROPRIO_DIM
        # one stroke = up to this many internal IK sim steps (4 phases x ~80:
        # approach, descend, push, retract-to-home)
        self.stroke_max_steps = stroke_max_steps
        # speedup: skip the per-step camera path-trace during a stroke, render only
        # the boundary frame we actually record (~30-80x fewer renders/stroke).
        self.fast_stroke_render = fast_stroke_render
        self._executor = None  # lazily built StrokeExecutor (needs a live scene)
        self._home_ee = None   # (N,3) env-local wrist pose at reset -> the stroke "park" pose
        self._home_jp = None   # (N,J) reset joint config -> teleported back after each stroke (exact park)

        env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs)
        self._env = gym.make(task_id, cfg=env_cfg)
        self._scene = self._env.unwrapped.scene
        self._setup_camera()

    def _setup_camera(self):
        """Pose the camera via look-at (eye/target env-local; env_origins offsets
        per env). Avoids guessing an OffsetCfg quaternion."""
        try:
            cam = self._scene[_CAMERA_KEY]
        except KeyError:
            return
        origins = self._scene.env_origins  # (N, 3)
        eye = torch.tensor(_CAM_EYE, device=self.device, dtype=torch.float32)
        target = torch.tensor(_CAM_TARGET, device=self.device, dtype=torch.float32)
        eyes = eye.unsqueeze(0).expand(self.num_envs, 3) + origins
        targets = target.unsqueeze(0).expand(self.num_envs, 3) + origins
        try:
            cam.set_world_poses_from_view(eyes, targets)
        except AttributeError:
            pass  # older IsaacLab without this method; OffsetCfg fallback

    def _arm_proprio(self):
        d = self._scene["robot"].data
        return torch.cat([d.joint_pos, d.joint_vel], dim=-1).reshape(self.num_envs, -1)

    def _scene_outputs(self):
        """Single pass over the scene: returns (obs_dict, state_np)."""
        arm = self._arm_proprio()  # (N, 18)

        origins = self._scene.env_origins  # (N, 3) world-frame env origins
        d = self._scene[_CUBE_KEY].data
        cube = torch.cat(
            [d.root_pos_w - origins, d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_w], dim=-1
        ).reshape(self.num_envs, -1)  # (N, 13)

        state = torch.cat([arm, cube], dim=-1).detach().cpu().numpy()  # (N, 31)
        visual = self._scene[_CAMERA_KEY].data.output["rgb"].detach().cpu().numpy()
        proprio = arm.detach().cpu().numpy()
        return {"visual": visual, "proprio": proprio}, state

    def _write_state(self, state):
        t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        self._scene["robot"].write_joint_state_to_sim(
            t[:, :_ARM_JOINT_DIM],
            t[:, _ARM_JOINT_DIM : 2 * _ARM_JOINT_DIM],
        )
        # State stores env-local cube pos; write_root_state_to_sim wants world frame.
        origins = self._scene.env_origins
        block = t[:, 2 * _ARM_JOINT_DIM : 2 * _ARM_JOINT_DIM + _CUBE_DIM].clone()
        block[:, :3] = block[:, :3] + origins
        self._scene[_CUBE_KEY].write_root_state_to_sim(block)

    def _materialize_state(self):
        """Flush written state to PhysX, step+render once, refresh sensor
        buffers so obs reflect the written/randomized state (not the stale
        pre-write reset frame). See GridWrapper._materialize_state."""
        self._scene.write_data_to_sim()
        sim = self._env.unwrapped.sim
        sim.step(render=True)
        self._scene.update(sim.get_physics_dt())

    def _action_tensor(self, action):
        a = np.atleast_2d(np.asarray(action, dtype=np.float32))  # (N,7) or (1,7)
        t = torch.as_tensor(a, device=self.device)
        if t.shape[0] == 1 and self.num_envs > 1:
            t = t.expand(self.num_envs, -1)
        return t

    # ---------- sign rule API (placeholder) ----------

    def set_sign_color(self, rgb):
        """Set the octagonal sign's diffuseColor (per env) at runtime, then render
        so the change is visible. `rgb` is either a single 3-tuple in [0,1] (same
        color for all envs) OR a sequence of num_envs 3-tuples (per-env color, for
        parallel collection). Placeholder "new rules" signal — no behavior yet."""
        from pxr import Gf, Sdf, UsdShade

        cols = np.asarray(rgb, dtype=np.float32)
        if cols.ndim == 1:                       # one color -> all envs
            cols = np.tile(cols, (self.num_envs, 1))
        sim = self._env.unwrapped.sim
        stage = sim.stage
        for i in range(self.num_envs):
            path = f"/World/envs/env_{i}/{_SIGN_SHADER_SUBPATH}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            shader = UsdShade.Shader(prim)
            inp = shader.GetInput("diffuseColor")
            if not inp:
                inp = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            inp.Set(Gf.Vec3f(float(cols[i][0]), float(cols[i][1]), float(cols[i][2])))
        sim.step(render=True)
        self._scene.update(sim.get_physics_dt())

    # ---------- IK / geometry helpers (single robot) ----------

    def get_ee_positions(self) -> np.ndarray:
        """(N,3) env-local panda_hand (wrist) position. The IK action term
        references the wrist body (body_offset removed), so policy deltas are
        consistent with this. Gripper tip sits ~10.7cm below the wrist."""
        origins = self._scene.env_origins
        robot = self._scene["robot"]
        hand_idx = robot.body_names.index("panda_hand")
        data = robot.data
        if hasattr(data, "body_pos_w"):
            hand_pos = data.body_pos_w[:, hand_idx, :]
        else:
            hand_pos = data.body_link_pose_w[:, hand_idx, :3]
        return (hand_pos - origins).detach().cpu().numpy()

    def get_cube_positions(self) -> np.ndarray:
        """(N,3) env-local cube xyz position."""
        origins = self._scene.env_origins
        return (self._scene[_CUBE_KEY].data.root_pos_w - origins).detach().cpu().numpy()

    def get_joint_diag(self):
        """Debug: (joint_pos (J,), names list, limits (J,2) or None) for env 0 —
        to see which joints pin at their limits during a push."""
        robot = self._scene["robot"]
        d = robot.data
        jp = d.joint_pos[0].detach().cpu().numpy()
        names = list(robot.joint_names)
        lim = None
        for attr in ("soft_joint_pos_limits", "joint_pos_limits"):
            if hasattr(d, attr):
                lim = getattr(d, attr)[0].detach().cpu().numpy()
                break
        return jp, names, lim

    def world_to_base_delta(self, delta_world) -> np.ndarray:
        """Rotate world-frame xyz delta(s) into the robot's base frame. Accepts
        (3,) (broadcast to all envs) or (N,3); returns the input shape."""
        import isaaclab.utils.math as math_utils
        robot = self._scene["robot"]
        root_quat_inv = math_utils.quat_inv(robot.data.root_quat_w)  # (N,4)
        delta_w_t = torch.as_tensor(np.asarray(delta_world, dtype=np.float32), device=self.device)
        is_1d = (delta_w_t.ndim == 1)
        if is_1d:
            delta_w_t = delta_w_t.unsqueeze(0)
        if delta_w_t.shape[0] == 1 and root_quat_inv.shape[0] > 1:
            delta_w_t = delta_w_t.expand(root_quat_inv.shape[0], -1)
        delta_b = math_utils.quat_apply(root_quat_inv, delta_w_t)
        out = delta_b.detach().cpu().numpy()
        return out[0] if is_1d else out

    def _ee_quat_base_t(self):
        """Current panda_hand orientation in the base frame, (N,4) wxyz tensor."""
        import isaaclab.utils.math as math_utils
        robot = self._scene["robot"]
        hand_idx = robot.body_names.index("panda_hand")
        data = robot.data
        if hasattr(data, "body_quat_w"):
            ee_quat_w = data.body_quat_w[:, hand_idx, :]
        else:
            ee_quat_w = data.body_link_pose_w[:, hand_idx, 3:7]
        return math_utils.quat_mul(math_utils.quat_inv(data.root_quat_w), ee_quat_w)

    def get_ee_quats_base(self) -> np.ndarray:
        """(N,4) base-frame panda_hand quaternion (wxyz). Reference orientation
        to hold the gripper at."""
        return self._ee_quat_base_t().detach().cpu().numpy()

    def orientation_delta_base(self, ref_quat_base) -> np.ndarray:
        """Axis-angle (N,3) in the base frame rotating the current EE orientation
        onto ref_quat_base. Fed to the IK relative-pose rotation command to hold
        the gripper at the reference orientation (closed loop)."""
        import isaaclab.utils.math as math_utils
        cur = self._ee_quat_base_t()  # (N,4)
        ref = torch.as_tensor(np.asarray(ref_quat_base, dtype=np.float32), device=self.device)
        if ref.ndim == 1:
            ref = ref.unsqueeze(0).expand(cur.shape[0], -1)
        delta = math_utils.quat_mul(ref, math_utils.quat_inv(cur))
        return math_utils.axis_angle_from_quat(delta).detach().cpu().numpy()

    # ---------- dino_wm contract ----------

    def _randomize_cube(self):
        """Reset hook: place the single cube within the grid (random yaw, zero
        velocity), EDGE-BIASED (_SPAWN_EDGE_K) to flatten cell coverage. Overrides the
        per-cube jitter from EventCfg. Only on the reset() path; set_init_state()
        writes explicit states unaffected."""
        N, dev = self.num_envs, self.device
        origins = self._scene.env_origins  # (N,3) world
        cx, cy = GRID_CENTER_XY
        # x lower-bounded at REACH_X_MIN so the cube never spawns in the near-base
        # column the arm can't get behind (would just stall/drift).
        x_lo = max(cx - _CUBE_RAND_HALF, REACH_X_MIN)
        x_hi = cx + _CUBE_RAND_HALF
        def _edge(u):  # warp U(0,1) toward 0/1 (k<1 = edge bias, 1 = uniform)
            s = 2 * u - 1
            return 0.5 + 0.5 * torch.sign(s) * s.abs() ** _SPAWN_EDGE_K
        x = x_lo + _edge(torch.rand(N, device=dev)) * (x_hi - x_lo)
        y = cy + (2 * _edge(torch.rand(N, device=dev)) - 1) * _CUBE_RAND_HALF
        z = torch.full((N,), _CUBE_SPAWN_Z, device=dev)
        pos_world = torch.stack([x, y, z], dim=1) + origins  # (N,3)
        yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)
        zeros = torch.zeros_like(qw)
        quat = torch.stack([qw, zeros, zeros, qz], dim=1)  # (N,4) wxyz
        vel = torch.zeros(N, 6, device=dev)
        block = torch.cat([pos_world, quat, vel], dim=1)  # (N,13)
        self._scene[_CUBE_KEY].write_root_state_to_sim(block)

    def reset(self):
        self._env.reset()
        self._randomize_cube()
        self._materialize_state()
        self._home_ee = self.get_ee_positions()   # park pose for this episode's strokes
        self._home_jp = self._scene["robot"].data.joint_pos.clone()  # exact joint config to park at
        return self._scene_outputs()

    def _get_executor(self):
        """Lazily build the StrokeExecutor (needs a live scene for the wrist ref)
        and capture the current wrist-down orientation as this stroke's reference.
        Lazy import avoids an import cycle with stroke_executor."""
        if self._executor is None:
            from .stroke_executor import StrokeExecutor
            self._executor = StrokeExecutor(self)
        self._executor.refresh_ref_quat()
        return self._executor

    def execute_stroke(self, strokes, frame_sink=None):
        """strokes: (N,4) or (4,) planar [x_start, y_start, dx, dy] env-local meters
        (start point + push displacement; end = start + (dx,dy)). Runs
        approach->descend->push via the internal 7-D IK term until the executor
        reports done for all envs (or stroke_max_steps elapse), then SNAPS the arm to
        the episode's home joint config so the BOUNDARY frame shows a fixed arm pose.
        Returns the BOUNDARY (obs_dict, state_np) after the stroke completes.

        frame_sink: optional list. If given, every internal step's env-0 RGB frame is
        appended (forces per-step rendering, ignoring fast_stroke_render) so a caller
        can assemble a SMOOTH dynamics preview video. Leave None for normal collection
        (boundary-only, fast)."""
        recording = frame_sink is not None
        s = np.atleast_2d(np.asarray(strokes, dtype=np.float32))  # (N,4) or (1,4)
        if s.shape[0] == 1 and self.num_envs > 1:
            s = np.repeat(s, self.num_envs, axis=0)
        start_xy = s[:, 0:2]
        end_xy = s[:, 0:2] + s[:, 2:4]            # action = [start_xy, displacement]
        ex = self._get_executor()
        ex.begin(start_xy, end_xy)   # no IK retract; we snap the arm to home after the loop
        env = self._env.unwrapped
        # Speedup: a stroke runs ~30-80 internal env.step()s, each of which would
        # path-trace the camera (gated by cfg.sim.render_interval, read every
        # substep in ManagerBasedRLEnv.step). We only keep the BOUNDARY frame, so
        # bump render_interval huge to skip rendering in the loop; physics +
        # scene.update still run (the executor reads EE/cube from physics, not the
        # camera), then render ONCE at the end. Toggle via fast_stroke_render.
        orig_ri = env.cfg.sim.render_interval
        if self.fast_stroke_render and not recording:
            env.cfg.sim.render_interval = 10**9       # skip per-step render (boundary only)
        try:
            for _ in range(self.stroke_max_steps):
                a7 = ex.compute_action(self.get_ee_positions(), self.get_cube_positions())
                self._env.step(self._action_tensor(a7))  # internal low-level IK step
                self._freeze_episode_clock()              # prevent mid-stroke time_out reset
                self._clamp_cube()                        # contain the cube to the reachable box
                if recording:                             # grab this step's per-env frame (smooth preview / plan video)
                    rgb = self._scene[_CAMERA_KEY].data.output["rgb"].detach().cpu().numpy()
                    frame_sink.append(np.asarray(rgb).copy())   # (N,H,W,3), all envs
                if ex.all_done():
                    break
        finally:
            env.cfg.sim.render_interval = orig_ri
        # Final containment, then render the boundary frame so image AND state both
        # reflect the clamped cube (a stroke runs render_interval high, so we render
        # once here). scene.update refreshes the obs/state buffers we read below.
        self._clamp_cube()
        # SNAP the arm to exactly the reset joint config so every recorded boundary frame
        # has an identical arm. Set BOTH the joint state AND the position target to home:
        # writing state alone left the PD actuator holding its last IK target, so the
        # settle step yanked the arm and it curled. With target==state==home the arm holds
        # home. _materialize_state does one physics step so the link poses (hence the
        # render) reflect the snap; the cube is at rest + clamped so it doesn't drift.
        if self._home_jp is not None:
            robot = self._scene["robot"]
            robot.write_joint_state_to_sim(self._home_jp, torch.zeros_like(self._home_jp))
            robot.set_joint_position_target(self._home_jp)
            self._materialize_state()
        else:
            env.sim.render()
            self._scene.update(env.sim.get_physics_dt())
        return self._scene_outputs()

    def _clamp_cube(self):
        """Hard containment: project the cube back into the +/-_CUBE_CLAMP_HALF box
        (env-local) so it can't leave the arm's reachable region. No physical wall ->
        nothing for the paddle to hit. Outward velocity is zeroed on a clamped cube so
        it doesn't retain momentum and fight the projection. No-op when the cube is
        inside (the common case), so per-step overhead is just a read + compare."""
        cube = self._scene[_CUBE_KEY]
        origins = self._scene.env_origins                       # (N,3) world
        d = cube.data
        local = d.root_pos_w - origins                          # (N,3) env-local
        h = _CUBE_CLAMP_HALF
        cl = local.clone()
        cl[:, 0] = local[:, 0].clamp(-h, h)
        cl[:, 1] = local[:, 1].clamp(-h, h)
        outside = (cl[:, 0] != local[:, 0]) | (cl[:, 1] != local[:, 1])  # (N,)
        if not bool(outside.any()):
            return
        lin = d.root_lin_vel_w.clone()
        lin[outside] = 0.0                                      # stop the clamped cubes
        block = torch.cat([cl + origins, d.root_quat_w, lin, d.root_ang_vel_w], dim=-1)  # (N,13)
        cube.write_root_state_to_sim(block)

    def _freeze_episode_clock(self):
        """A stroke runs up to stroke_max_steps internal env.step calls, far past
        the env's max_episode_length (~125). The only DoneTerm is `time_out`
        (episode_length_buf >= max), and ManagerBasedRLEnv auto-resets done envs
        INSIDE step() — which would re-randomize the scene mid-stroke. Episode
        boundaries are managed externally here (prepare/reset), so zero the counter
        after every internal step: it re-increments to 1 next step and never reaches
        the limit. No-op if the attribute is absent (non-RL env)."""
        env = self._env.unwrapped
        buf = getattr(env, "episode_length_buf", None)
        if buf is not None:
            buf.zero_()

    def step(self, action):
        """action: (N,4) or (4,) stroke. ONE stroke == ONE macro-step. `done` is
        not episode-meaningful here (success is judged via eval_state)."""
        obs, state = self.execute_stroke(action)
        done = np.zeros((self.num_envs,), dtype=bool)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        return obs, reward, done, {"state": state}

    def set_init_state(self, init_state):
        self._env.reset()
        self._write_state(np.asarray(init_state))
        self._materialize_state()
        self._home_ee = self.get_ee_positions()   # park pose for this episode's strokes
        self._home_jp = self._scene["robot"].data.joint_pos.clone()  # exact joint config to park at
        return self._scene_outputs()[0]

    def prepare(self, seed, init_state):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        self.set_init_state(init_state)
        return self._scene_outputs()

    def rollout(self, seed, init_state, actions, frame_sink=None):
        """actions: (T,4) or (N,T,4) strokes. One boundary obs/state per stroke.
        frame_sink: optional list -> every internal sim step's (N,H,W,3) frame is
        appended (forces per-step rendering) for a SMOOTH plan/preview video, starting
        with the initial rest frame. Leave None for the fast boundary-only rollout."""
        obs, state = self.prepare(seed, init_state)
        visuals = [obs["visual"]]
        proprios = [obs["proprio"]]
        states = [state]
        if frame_sink is not None:
            frame_sink.append(np.asarray(obs["visual"]).copy())   # initial (rest) frame

        a = np.asarray(actions)
        if a.ndim == 2:  # (T,4) -> (1,T,4)
            a = a[None]
        for t in range(a.shape[1]):
            o, st = self.execute_stroke(a[:, t], frame_sink=frame_sink)
            visuals.append(o["visual"])
            proprios.append(o["proprio"])
            states.append(st)
        return (
            {
                "visual": np.stack(visuals, axis=1),
                "proprio": np.stack(proprios, axis=1),
            },
            np.stack(states, axis=1),
        )

    def update_env(self, env_info):
        pass

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        return [seed]
