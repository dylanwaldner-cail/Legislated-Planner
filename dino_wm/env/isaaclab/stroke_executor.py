"""Open-loop planar-stroke executor for the DinoWMGrid-Single env.

Compiles ONE high-level stroke (given as explicit start_xy/end_xy endpoints; the
4-D env action [x_start, y_start, dx, dy] is converted to endpoints by the wrapper,
end = start + disp) into the per-step 7-D IK commands the env consumes (DifferentialIKController,
pose/relative, scale=2.0). One stroke == one env macro-step == one recorded WM frame.

Phase machine approach_above -> descend -> push, OPEN-LOOP to the commanded
endpoints: the pusher goes to start_xy, descends to a fixed push height, then pushes
straight to end_xy with the blade yawed to atan2(dy, dx). (An OPTIONAL 4th `retract`
phase to a home pose runs only if begin() is given home_xyz; the wrapper currently
leaves it off and instead SNAPS the arm to a fixed home joint config after the stroke
-- IK-retracting the redundant arm could curl it.) Parking the arm at a fixed pose
makes every recorded boundary frame show only the cube changing, matching how the
DINO-WM deformable dataset is generated -- no arm-motion/occlusion for the WM to model.
No slip/re-approach recovery, so a stroke is a deterministic function of its
endpoints (the cube may or may not move). Cube position is read only for the
push-height z reference and the stall watchdog. The action carries no z; heights are
wrist offsets above cube_z (the gripper TCP sits ~10.7cm below the IK-aimed wrist).

VECTORIZED across envs: all per-env state is length-N arrays, so the same executor
drives collection (num_envs=1) and planning (num_envs=n_evals); the wrapper helpers
it calls (get_ee_positions, world_to_base_delta, orientation_delta_base) accept/return (N, .).
"""
from __future__ import annotations

import numpy as np

# Must match the IK scale in dinowm_grid_env_cfg.py (_ik scale=2.0).
_IK_SCALE = 2.0
# The env consumes a 7-D IK command [pos_delta(3), rot_aa(3), gripper]; that is
# the LOW-LEVEL interface, distinct from the 4-D stroke action the WM sees.
_IK_CMD_DIM = 7


# --- minimal quaternion helpers (wxyz, to match IsaacLab) ---
def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def _quat_rotate(q, v):
    """Rotate 3-vec v by quaternion q (wxyz)."""
    w, x, y, z = q
    qv = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(qv, v)
    return (v + w * t + np.cross(qv, t)).astype(np.float32)


def _quat_yaw(angle):
    """Quaternion (wxyz) for a rotation of `angle` about +z."""
    h = 0.5 * angle
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float32)


class StrokeExecutor:
    """Drives ONE planar stroke per env to completion via the env's IK term.

    Usage (the wrapper macro-step):
        ex.refresh_ref_quat()          # capture the wrist-down ref each stroke
        ex.begin(start_xy, end_xy)     # (N,2) or (2,) env-local meters
        while not ex.all_done() and steps < cap:
            a7 = ex.compute_action(ee_pos(N,3), cube_pos(N,3))   # (N,7) IK cmd
            env._env.step(...a7...)
    """

    PHASES = ("approach_above", "descend", "push", "retract")

    # Blade flat-face normal in panda_hand local frame (thin axis = local-x).
    LOCAL_FACE_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Wrist target offsets above cube_z (cube CENTER); the pusher tip is ~13cm below.
    APPROACH_HEIGHT = 0.27   # wrist high enough that the tip clears the cube top
    PUSH_HEIGHT = 0.10       # wrist height while descending/pushing

    # Phase-advance distances (loose so motion flows phase-to-phase, not a full stop).
    APPROACH_ARRIVE = 0.12
    DESCEND_ARRIVE = 0.06
    PUSH_ARRIVE = 0.03       # EE within this of end_xy => push done
    RETRACT_ARRIVE = 0.06    # EE within this of home_xyz => retract (park) done

    MAX_STEPS_PER_PHASE = 80
    # Per-step EE displacement caps (m, world) near the cube. These USED to be tiny to
    # stop a hard push coasting the cube ~1m off-grid -- but the hard cube clamp
    # (GridWrapperSingle._clamp_cube) now contains the cube to the grid every step, so
    # overshoot can no longer launch it; we can move much faster. DESCEND is a near-
    # VERTICAL drop at start_xy (no lateral approach to the cube), so it's the fastest.
    # PUSH still has a cap so the cube gets a controlled shove (not a full-speed lunge),
    # but it's no longer the launch-prevention knob. Approach (0) and retract (3) uncapped.
    PUSH_STEP = 0.06         # phase 2: controlled shove; clamp contains any overshoot
    DESCEND_STEP = 0.15      # phase 1: fast vertical drop (no lateral cube risk)
    # Approach/descend stall watchdog: if EE-to-target distance stops shrinking
    # (arm pinned a joint / unreachable start) abort the stroke rather than freeze.
    # BUT only once the blade is roughly aimed -- a big yaw to face the push dir
    # rotates the wrist in place (EE position barely moves), which is NOT a stall;
    # gating on ori_err lets the aim finish before the watchdog can fire.
    EE_PLATEAU_WINDOW = 10
    EE_PLATEAU_TOL = 0.012   # min EE-distance progress over the window (m)
    ORI_ALIGN_THRESH = 0.18  # rad (~10deg): below this the blade is aimed, so a position plateau is a real stall

    # Cube z guard: if the cube tipped/fell, clamp the height reference so the
    # blade doesn't dive. 0.046 = 9cm-cube center resting on the table.
    _CUBE_Z_REF = 0.046

    def __init__(self, env):
        self._env = env
        self.N = int(env.num_envs)
        self.ref_quat = None          # (N,4) base-frame wrist orientation to hold
        # per-env stroke state (filled by begin)
        self.start_xy = np.zeros((self.N, 2), np.float32)
        self.end_xy = np.zeros((self.N, 2), np.float32)
        self.home_xyz = None          # (N,3) fixed park pose; None -> no retract phase
        self._has_retract = False
        self.target_quat = None       # (N,4)
        self.phase_idx = np.zeros(self.N, np.int64)
        self.step_in_phase = np.zeros(self.N, np.int64)
        self.done = np.zeros(self.N, bool)
        self._ee_hist = np.full((self.N, self.EE_PLATEAU_WINDOW), np.inf, np.float32)
        self.ori_err = np.zeros(self.N, np.float32)  # (N,) blade-aim error (rad), set each step

    # ---------- setup ----------

    def refresh_ref_quat(self):
        """Capture the current (wrist-down) base-frame orientation to hold during
        the stroke; the blade is yawed off this each begin()."""
        self.ref_quat = np.asarray(self._env.get_ee_quats_base(), np.float32)  # (N,4)

    def _target_quat_for_dir(self, push_dir_i, ref_quat_i):
        """Base-frame wrist quat that yaws the blade face along push_dir_i (single
        env). The blade's ±x faces are both flat, so wrap to the NEARER face
        ([-pi/2, pi/2]) -- at most a 90deg spin."""
        n0 = _quat_rotate(ref_quat_i, self.LOCAL_FACE_AXIS)
        heading0 = float(np.arctan2(n0[1], n0[0]))
        desired = float(np.arctan2(push_dir_i[1], push_dir_i[0]))
        yaw = (desired - heading0 + np.pi / 2) % np.pi - np.pi / 2
        return _quat_mul(_quat_yaw(yaw), ref_quat_i)

    def begin(self, start_xy, end_xy, home_xyz=None):
        """Set the stroke endpoints (env-local grid meters) and reset the phase
        machine. start_xy/end_xy: (N,2) or (2,) broadcast to all envs.
        home_xyz: (N,3) or (3,) env-local wrist pose to PARK at after the push (adds a
        retract phase, so every recorded boundary frame has the arm at this fixed pose
        -- matches the deformable dataset). None -> no retract (push ends the stroke)."""
        if self.ref_quat is None:
            self.refresh_ref_quat()
        s = np.asarray(start_xy, np.float32).reshape(-1, 2)
        e = np.asarray(end_xy, np.float32).reshape(-1, 2)
        if s.shape[0] == 1 and self.N > 1:
            s = np.repeat(s, self.N, axis=0)
        if e.shape[0] == 1 and self.N > 1:
            e = np.repeat(e, self.N, axis=0)
        self.start_xy, self.end_xy = s, e

        self._has_retract = home_xyz is not None
        if self._has_retract:
            hz = np.asarray(home_xyz, np.float32).reshape(-1, 3)
            if hz.shape[0] == 1 and self.N > 1:
                hz = np.repeat(hz, self.N, axis=0)
            self.home_xyz = hz

        d = e - s                                   # (N,2)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        push_dir = np.where(n > 1e-6, d / np.maximum(n, 1e-6),
                            np.array([[1.0, 0.0]], np.float32))  # (N,2)
        # target_quat per env (cold path; once per stroke) via the scalar helper.
        self.target_quat = np.stack([
            self._target_quat_for_dir(push_dir[i], self.ref_quat[i])
            for i in range(self.N)
        ], axis=0).astype(np.float32)               # (N,4)

        self.phase_idx[:] = 0
        self.step_in_phase[:] = 0
        self.done[:] = False
        self._ee_hist[:] = np.inf

    # ---------- per-step control ----------

    def all_done(self):
        return bool(self.done.all())

    def _phase_target(self, ee_pos, cube_pos):
        """(N,3) wrist target for the current per-env phase. approach/descend aim
        at start_xy (high then low), push aims at end_xy. Done envs hold (target =
        ee_pos -> zero delta)."""
        z_cube = np.clip(cube_pos[:, 2], self._CUBE_Z_REF, self._CUBE_Z_REF + 0.05)  # (N,)
        is_approach = self.phase_idx == 0
        is_push = self.phase_idx == 2
        target_xy = np.where(is_push[:, None], self.end_xy, self.start_xy)           # (N,2)
        height = np.where(is_approach, self.APPROACH_HEIGHT, self.PUSH_HEIGHT)       # (N,)
        target = np.concatenate([target_xy, (z_cube + height)[:, None]], axis=1)     # (N,3)
        if self._has_retract:
            is_retract = self.phase_idx == 3
            if is_retract.any():
                target[is_retract] = self.home_xyz[is_retract]   # park at the fixed home pose
        if self.done.any():
            target[self.done] = ee_pos[self.done]   # hold: zero positional delta
        return target.astype(np.float32)

    def compute_action(self, ee_positions, cube_positions):
        """Returns the (N,7) IK command for the current step and advances each
        env's phase machine. ee_positions/cube_positions: (N,3) env-local."""
        ee_pos = np.asarray(ee_positions, np.float32)
        cube_pos = np.asarray(cube_positions, np.float32)
        target = self._phase_target(ee_pos, cube_pos)                # (N,3)

        # --- build the 7-D IK command (vectorized): pos delta + blade-aim rot + open gripper ---
        raw_world = target - ee_pos                                  # (N,3) desired EE displacement (m)
        # gentle push: cap the displacement during the PUSH phase so the paddle creeps
        # (low cube release velocity -> minimal coast). Other phases uncapped.
        mag = np.linalg.norm(raw_world, axis=1, keepdims=True)       # (N,1)
        # per-phase cap: descend faster than push; approach(0)/retract(3) uncapped.
        is_descend = (self.phase_idx == 1)[:, None]
        is_push = (self.phase_idx == 2)[:, None]
        cap = np.where(is_descend, self.DESCEND_STEP,
                       np.where(is_push, self.PUSH_STEP, mag))
        raw_world = raw_world * np.minimum(1.0, cap / np.maximum(mag, 1e-9))
        delta_world = raw_world / _IK_SCALE                          # (N,3)
        delta_base = np.asarray(self._env.world_to_base_delta(delta_world), np.float32)
        delta_base = np.clip(delta_base, -1.0, 1.0)
        rot_aa = np.asarray(self._env.orientation_delta_base(self.target_quat), np.float32)  # (N,3) axis-angle (rad)
        self.ori_err = np.linalg.norm(rot_aa, axis=1)                 # (N,) blade-aim error before IK scaling
        rot_aa = np.clip(rot_aa / _IK_SCALE, -1.0, 1.0)
        action = np.zeros((self.N, _IK_CMD_DIM), np.float32)
        action[:, :3] = delta_base
        action[:, 3:6] = rot_aa
        action[:, 6] = +1.0                                          # gripper OPEN (fingers clear of blade)

        self._advance(ee_pos, target)
        return action

    def _advance(self, ee_pos, target):
        """Vectorized phase machine update. approach/descend -> next phase on
        arrival or timeout (or done on stall); push -> done on arrival or timeout."""
        active = ~self.done
        self.step_in_phase[active] += 1
        ee_dist = np.linalg.norm(target - ee_pos, axis=1)            # (N,)

        # rolling EE-distance window for the stall watchdog
        self._ee_hist[:, :-1] = self._ee_hist[:, 1:]
        self._ee_hist[:, -1] = ee_dist

        is_approach = self.phase_idx == 0
        is_descend = self.phase_idx == 1
        is_push = self.phase_idx == 2
        is_retract = self.phase_idx == 3
        arrive = np.where(is_approach, self.APPROACH_ARRIVE,
                          np.where(is_descend, self.DESCEND_ARRIVE,
                                   np.where(is_push, self.PUSH_ARRIVE, self.RETRACT_ARRIVE)))
        arrived = ee_dist < arrive
        timed_out = self.step_in_phase >= self.MAX_STEPS_PER_PHASE

        filled = self.step_in_phase >= self.EE_PLATEAU_WINDOW
        span = self._ee_hist.max(axis=1) - self._ee_hist.min(axis=1)
        # Real stall only in approach/descend, and only once the blade is aimed: while
        # ori_err is large the arm is legitimately yawing in place, so don't abort --
        # MAX_STEPS_PER_PHASE bounds it. (push/retract never abort on a plateau.)
        stalled = (filled & (span < self.EE_PLATEAU_TOL) & (ee_dist > arrive)
                   & (is_approach | is_descend) & (self.ori_err < self.ORI_ALIGN_THRESH))

        # The stroke ENDS on its last phase: retract (park-at-home) when enabled, else
        # push. Earlier phases advance forward; push advances into retract when on.
        if self._has_retract:
            finish = is_retract & active & (arrived | timed_out)
            advance = (is_approach | is_descend | is_push) & active & (arrived | timed_out)
        else:
            finish = is_push & active & (arrived | timed_out)
            advance = (is_approach | is_descend) & active & (arrived | timed_out)
        self.done[finish] = True

        # approach/descend: abort on stall (only if not already advancing)
        abort = (is_approach | is_descend) & active & stalled & ~advance
        self.done[abort] = True

        adv = advance & ~self.done
        self.phase_idx[adv] += 1
        self.step_in_phase[adv] = 0
        if adv.any():
            self._ee_hist[adv] = np.inf   # restart stall window for the new phase
