"""Scripted single-robot PUSH expert for the DinoWMGrid-Single env.

Single env, single arm, single cube. A rigid cylindrical pusher rod is welded
under the hand (see _pusher in the env cfg); the arm holds a fixed wrist-down
orientation and drives the rod to push the cube along a random monotone-Manhattan
path of within-cell waypoints from its start cell to a target cell. The gripper
is parked OPEN so the fingers stay clear of the rod.

Per push segment (cube -> next waypoint) the phase machine is:
    approach_above -> descend -> push
On reaching a waypoint it advances to the next; if the cube slips off the
paddle or stops progressing it does a corrective re-approach (back to
approach_above behind the cube's new position). After the final waypoint it is
`done` (deterministic) or resamples a fresh target and loops (chase_random).

Used by inspect_single.py (visualization) and
scripts/collect_isaaclab_grid_data.py (data collection). Single env only.

All HEIGHT constants are wrist-target offsets relative to cube_z: the IK aims
`panda_hand` (the wrist) at the target, and the gripper TCP sits ~10.7cm below
the wrist, so PUSH_HEIGHT ~0.10 puts the closed paddle low on the cube side.
"""
from __future__ import annotations

import numpy as np

from .grid_wrapper_single import ACTION_DIM
from .grid_metadata import (
    GRID_CENTER_XY,
    GRID_HALF,
    N_CELLS,
    REACH_X_MIN,
    monotone_manhattan_cells,
    random_point_in_cell,
    which_cell,
)

# Must match the IK scale in dinowm_grid_env_cfg.py (_ik scale=2.0).
_IK_SCALE = 2.0


# --- minimal quaternion helpers (wxyz, to match IsaacLab) for yawing the blade ---
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


class PushExpert:
    """Single-arm push phase machine. Single env only.

    Two waypoint-generation modes (push_mode):
      "random" (default) -> the cube does a random walk: each segment picks a
          random direction + distance from the cube's current spot (clamped to
          stay on the grid) and pushes it there, then picks another, forever.
          This is the WM data-collection driver: dense, diverse contact
          transitions, no goal structure (a one-step dynamics model needs none).
      "cells" -> goal-directed: random monotone-Manhattan path of within-cell
          waypoints from a start cell to a target cell (deterministic ends, or
          chase_random to loop to fresh target cells). Kept for goal-directed
          demos / eval-goal generation.

    Either way the control is the same: approach_above (move behind the cube and
    rotate the blade to face the push direction simultaneously) -> descend ->
    push, with corrective re-approach on slip/stall. Stores `last_target` (world
    frame) for optional marker rendering.
    """

    PHASES = ("approach_above", "descend", "push")

    # The blade's flat-face normal in panda_hand local frame (thin axis = local-x;
    # see _pusher in the env cfg). The wrist is yawed so this points along the
    # push direction.
    LOCAL_FACE_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Don't leave approach_above (blade up, clear of the cube) until the blade has
    # rotated to within this of facing the push direction — otherwise the slow IK
    # wrist rotation isn't finished by contact and the cube gets hit with the side.
    ORI_ALIGN_THRESH = 0.18  # rad (~10 deg)

    # Random-mode push: each segment shoves the cube a random distance in a
    # random heading, clamped to stay `RAND_MARGIN` inside the grid edge.
    RAND_PUSH_MIN = 0.06
    RAND_PUSH_MAX = 0.18
    RAND_MARGIN = 0.06   # >= cube half (0.045) so pushes keep the 9cm cube on-grid

    # Heights are wrist (panda_hand) target offsets above cube_z (= cube CENTER).
    # The pusher tip (paddle bottom) sits ~0.13m below the wrist (short pole +
    # paddle). With the base at 0.15m, the push wrist (~0.15) is ~at base height
    # (elbow-comfortable) while the tip reaches the cube's lower side / CoM.
    APPROACH_HEIGHT = 0.27   # wrist high enough that the tip (~0.13 below) clears the cube top
    PUSH_HEIGHT = 0.10       # wrist height while pushing (tip ~just above table, wrist ~base height)
    BEHIND_DIST = 0.12       # how far behind the cube (along -push_dir) the blade
                             # sits; clears the 9cm cube (half 0.045) on descent,
                             # even yawed, so it descends beside the cube not on top
    CONTACT_OFFSET = 0.05    # EE push-target = waypoint - push_dir*CONTACT_OFFSET
                             # (~blade_half + cube_half, so the cube center lands on wp)

    APPROACH_ARRIVE = 0.12   # loose so it cuts the corner into descend (no full stop)
    DESCEND_ARRIVE = 0.06    # loose so it flows into the push instead of stopping low
    CUBE_ARRIVE = 0.03       # cube within this of the waypoint => segment done
    SLIP_THRESH = 0.06       # cube lateral deviation from the push line => re-approach

    MAX_STEPS_PER_PHASE = 80
    MAX_REAPPROACH = 4       # per segment; then force-advance the waypoint
    PLATEAU_WINDOW = 8
    PLATEAU_TOL = 0.004      # cube-to-waypoint distance change over the window
    # Approach/descend stall watchdog: if the EE-to-target distance stops shrinking
    # (the arm pinned a joint / can't reach this target), abandon the push and try
    # a new direction instead of freezing the whole trajectory.
    EE_PLATEAU_WINDOW = 10
    EE_PLATEAU_TOL = 0.012   # min EE-distance progress over the window (m)

    # Per-phase scale on exploration noise. Push gets full noise for path
    # variety; the re-positioning phases get less so the paddle still lands
    # behind the cube. Default 1.0.
    PHASE_NOISE_SCALE = {"approach_above": 0.5, "descend": 0.3, "push": 1.0}

    def __init__(self, rng, env=None, verbose=True, push_mode="random",
                 chase_random=False, reachable_cells=None):
        """push_mode -> "random" (default; cube random-walk, loops forever) or
            "cells" (goal-directed Manhattan path; see class docstring).
        chase_random -> "cells" mode only: loop to fresh target cells instead of
            stopping (done) at the first target. Ignored in "random" mode (always
            loops).
        reachable_cells -> "cells" mode only: cell_ids the target may be drawn
            from (default all 9). Restrict once the smoke test shows which cells
            the single arm can reach.
        """
        self.rng = rng
        self._env = env
        self.verbose = verbose
        self.push_mode = push_mode
        self.chase_random = chase_random
        self.reachable_cells = tuple(reachable_cells) if reachable_cells is not None \
            else tuple(range(N_CELLS))
        self.noise_std = 0.0          # set per episode by the caller; 0 = clean
        self.ref_quat = None          # base-frame wrist orientation to hold
        self.last_target = None
        self.state = None
        self.debug = False            # per-step diagnostic prints (inspect --debug sets True)

    # ---------- path / target setup ----------

    def _cube_xy(self):
        return self._env.get_cube_positions()[0][:2].astype(np.float32)

    def _pick_target_cell(self, start_cell):
        choices = [c for c in self.reachable_cells if c != start_cell] \
            or [c for c in range(N_CELLS) if c != start_cell]
        return int(choices[self.rng.randint(0, len(choices))])

    def _build_waypoints(self, start_cell, target_cell):
        """One random within-cell waypoint per cell on the monotone-Manhattan
        path AFTER the start cell (the last lands in the target cell = goal)."""
        path = monotone_manhattan_cells(start_cell, target_cell, self.rng)
        cells = path[1:] if len(path) > 1 else [target_cell]
        return [random_point_in_cell(c, self.rng) for c in cells]

    def _random_waypoint(self):
        """A single push target a random distance in a random heading from the
        cube's current spot, clamped to stay inside the grid (RAND_MARGIN from
        the edge). This is what makes the cube move in random directions."""
        cube_xy = self._cube_xy()
        theta = self.rng.uniform(-np.pi, np.pi)
        dist = self.rng.uniform(self.RAND_PUSH_MIN, self.RAND_PUSH_MAX)
        wp = cube_xy + dist * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        cx, cy = GRID_CENTER_XY
        lim = GRID_HALF - self.RAND_MARGIN
        # x lower-bounded at REACH_X_MIN so pushes don't drive the cube into the
        # near-base column the arm can't get behind.
        wp[0] = float(np.clip(wp[0], max(cx - lim, REACH_X_MIN), cx + lim))
        wp[1] = float(np.clip(wp[1], cy - lim, cy + lim))
        return wp.astype(np.float32)

    def _target_quat_for_dir(self, push_dir):
        """Base-frame wrist quaternion that yaws the blade face normal to point
        along push_dir (rotation about vertical, preserving the captured
        wrist-down pose). Assumes the robot base is at identity yaw, so base-z is
        world vertical and push_dir (env-local xy) is already in the base frame."""
        n0 = _quat_rotate(self.ref_quat, self.LOCAL_FACE_AXIS)  # blade normal at ref
        heading0 = float(np.arctan2(n0[1], n0[0]))
        desired = float(np.arctan2(push_dir[1], push_dir[0]))
        # The blade is symmetric (both ±x faces are flat), so align the NEARER
        # face: wrap the yaw to [-pi/2, pi/2] -> max 90deg spin instead of 180.
        yaw = (desired - heading0 + np.pi / 2) % np.pi - np.pi / 2
        return _quat_mul(_quat_yaw(yaw), self.ref_quat)

    def _begin_segment(self, recompute_dir_from_cube=True):
        """(Re)compute the push direction for the current waypoint from the
        cube's current position, set the wrist orientation that faces the blade
        along it, and reset the per-phase counters."""
        st = self.state
        if recompute_dir_from_cube:
            wp = st["waypoints"][st["wp_idx"]]
            d = wp - self._cube_xy()
            n = float(np.linalg.norm(d))
            st["push_dir"] = (d / n) if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        st["target_quat"] = self._target_quat_for_dir(st["push_dir"])
        st["phase_idx"] = 0
        st["step_in_phase"] = 0
        st["recent_cube_dists"] = []
        st["recent_ee"] = []

    def reset(self, start_cell=None, target_cell=None):
        """Init episode state. In "cells" mode start_cell/target_cell may be
        pinned (else derived/random). In "random" mode they're ignored."""
        self.ref_quat = self._env.get_ee_quats_base()[0].copy()
        self.state = {
            "wp_idx": 0,
            "phase_idx": 0,
            "step_in_phase": 0,
            "push_dir": np.array([1.0, 0.0], dtype=np.float32),
            "reapproaches": 0,
            "recent_cube_dists": [],
            "cycle": 0,
            "done": False,
        }
        if self.push_mode == "random":
            self.state["waypoints"] = [self._random_waypoint()]
            self.state["target_cell"] = -1  # n/a in random mode
            if self.verbose:
                wp = self.state["waypoints"][0]
                print(f"[push] random mode: first target ({wp[0]:+.3f},{wp[1]:+.3f})")
        else:
            cube_xy = self._cube_xy()
            if start_cell is None:
                start_cell = int(which_cell(cube_xy))
                if start_cell < 0:
                    start_cell = 4  # cube off-grid -> fall back to center
            if target_cell is None:
                target_cell = self._pick_target_cell(start_cell)
            self.state["start_cell"] = int(start_cell)
            self.state["target_cell"] = int(target_cell)
            self.state["waypoints"] = self._build_waypoints(start_cell, target_cell)
            if self.verbose:
                print(f"[push] cells mode: cell {start_cell} -> cell {target_cell}; "
                      f"{len(self.state['waypoints'])} waypoints")
        self._begin_segment()

    # ---------- per-step control ----------

    def __call__(self, ee_positions, cube_positions):
        return self._step(ee_positions[0].astype(np.float32),
                          cube_positions[0].astype(np.float32))

    def _action(self, target_xyz, ee_pos, phase):
        """Build the 7-D action that drives the wrist toward target_xyz while
        yawing it so the blade face points along the current push direction
        (state['target_quat']). Gripper parked OPEN so the fingers stay clear of
        the blade."""
        OPEN = +1.0
        self.last_target = np.asarray(target_xyz, dtype=np.float32).copy()
        delta_world = (np.asarray(target_xyz, dtype=np.float32) - ee_pos) / _IK_SCALE
        delta_base = np.asarray(self._env.world_to_base_delta(delta_world), dtype=np.float32)
        scale = self.PHASE_NOISE_SCALE.get(phase, 1.0)
        if self.noise_std > 0.0 and scale > 0.0:
            delta_base = delta_base + self.rng.normal(
                0.0, self.noise_std * scale, size=3).astype(np.float32)
        delta_base = np.clip(delta_base, -1.0, 1.0).astype(np.float32)
        action = np.zeros((1, ACTION_DIM), dtype=np.float32)
        action[0, :3] = delta_base
        target_quat = self.state.get("target_quat", self.ref_quat) \
            if self.state is not None else self.ref_quat
        rot_aa = self._env.orientation_delta_base(target_quat)[0]
        if self.state is not None:
            self.state["ori_err"] = float(np.linalg.norm(rot_aa))  # rad, for phase gating
        action[0, 3:6] = np.clip(rot_aa / _IK_SCALE, -1.0, 1.0).astype(np.float32)
        action[0, 6] = OPEN
        return action

    def _step(self, ee_pos, cube_pos):
        st = self.state
        if st["done"]:
            return self._action(ee_pos, ee_pos, "done")  # hold

        cube_xy = cube_pos[:2]
        z_cube = float(cube_pos[2])
        wp = st["waypoints"][st["wp_idx"]]
        d = st["push_dir"]
        phase = self.PHASES[st["phase_idx"]]
        behind_xy = cube_xy - d * self.BEHIND_DIST

        if phase == "approach_above":
            # Move behind the cube (up high) and rotate to face the push dir at
            # once; descent waits until both arrived and aimed.
            target = np.array([behind_xy[0], behind_xy[1], z_cube + self.APPROACH_HEIGHT])
        elif phase == "descend":
            target = np.array([behind_xy[0], behind_xy[1], z_cube + self.PUSH_HEIGHT])
        else:  # push
            push_xy = wp - d * self.CONTACT_OFFSET
            target = np.array([push_xy[0], push_xy[1], z_cube + self.PUSH_HEIGHT])

        action = self._action(target, ee_pos, phase)
        st["step_in_phase"] += 1
        ee_dist = float(np.linalg.norm(target - ee_pos))
        cube_to_wp = float(np.linalg.norm(wp - cube_xy))

        if phase in ("approach_above", "descend"):
            # Advance on POSITION only (loose) — the blade keeps rotating to face
            # the push dir every step via target_quat, but we never WAIT for the
            # yaw. That wait was the "complete stop" between phases and the cause
            # of trajectories that approached but never pushed. It contacts mostly
            # aimed and keeps correcting during the push.
            arrive = self.APPROACH_ARRIVE if phase == "approach_above" else self.DESCEND_ARRIVE
            st["recent_ee"].append(ee_dist)
            if len(st["recent_ee"]) > self.EE_PLATEAU_WINDOW:
                st["recent_ee"].pop(0)
            stalled = (
                len(st["recent_ee"]) >= self.EE_PLATEAU_WINDOW
                and max(st["recent_ee"]) - min(st["recent_ee"]) < self.EE_PLATEAU_TOL
                and ee_dist > arrive
            )
            if ee_dist < arrive or st["step_in_phase"] >= self.MAX_STEPS_PER_PHASE:
                self._advance_phase()
            elif stalled:
                # Arm can't reach this target (pinned a joint, e.g. the elbow at a
                # far-lateral/low pose) -> abandon and try a different push dir
                # instead of freezing the trajectory.
                if self.verbose:
                    print(f"[push] {phase} STALLED (unreachable) wp {st['wp_idx']} "
                          f"eeD={ee_dist:.3f} -> new push")
                self._advance_waypoint("stuck")
        else:  # push
            self._push_logic(cube_xy, wp, d, cube_to_wp)

        if self.debug:
            self._debug_print(phase, d, ee_dist, cube_to_wp)
        return action

    def _debug_print(self, phase, d, ee_dist, cube_to_wp):
        """Per-step diagnostic: push heading, orientation error, EE/cube progress,
        and any joints pinned near their limits (the freeze suspect)."""
        st = self.state
        heading = float(np.degrees(np.arctan2(d[1], d[0])))
        ori_err_deg = float(np.degrees(st.get("ori_err", 0.0)))
        near = ""
        try:
            jp, names, lim = self._env.get_joint_diag()
            if lim is not None:
                flags = []
                for k, nm in enumerate(names):
                    lo, hi = float(lim[k][0]), float(lim[k][1])
                    margin = min(jp[k] - lo, hi - jp[k])
                    if margin < 0.15:  # within ~8.5deg of a joint limit
                        flags.append(f"{nm.replace('panda_', '')}={jp[k]:+.2f}(m{margin:.2f})")
                if flags:
                    near = " LIMIT:" + ",".join(flags)
        except Exception:
            pass
        print(f"[diag s{st['step_in_phase']:02d}] {phase:13s} head={heading:+4.0f} "
              f"oriErr={ori_err_deg:3.0f} eeD={ee_dist:.3f} cubeD={cube_to_wp:.3f}{near}")

    def _push_logic(self, cube_xy, wp, d, cube_to_wp):
        st = self.state
        # Reached the waypoint -> next segment.
        if cube_to_wp < self.CUBE_ARRIVE:
            self._advance_waypoint("reached")
            return
        # Lateral deviation of the cube from the push line -> slipped off paddle.
        off = cube_xy - wp
        slip = float(np.linalg.norm(off - np.dot(off, d) * d))
        # Progress plateau: cube no longer closing on the waypoint.
        st["recent_cube_dists"].append(cube_to_wp)
        if len(st["recent_cube_dists"]) > self.PLATEAU_WINDOW:
            st["recent_cube_dists"].pop(0)
        plateaued = (
            len(st["recent_cube_dists"]) >= self.PLATEAU_WINDOW
            and max(st["recent_cube_dists"]) - min(st["recent_cube_dists"]) < self.PLATEAU_TOL
        )
        if slip > self.SLIP_THRESH or plateaued or st["step_in_phase"] >= self.MAX_STEPS_PER_PHASE:
            st["reapproaches"] += 1
            if st["reapproaches"] > self.MAX_REAPPROACH:
                self._advance_waypoint("gave_up")
            else:
                if self.verbose:
                    reason = "slip" if slip > self.SLIP_THRESH else (
                        "plateau" if plateaued else "timeout")
                    print(f"[push] re-approach ({reason}) wp {st['wp_idx']} "
                          f"slip={slip:.3f} d_wp={cube_to_wp:.3f} "
                          f"#{st['reapproaches']}")
                self._begin_segment()  # back to approach_above, recompute push_dir

    def _advance_phase(self):
        st = self.state
        st["phase_idx"] = min(st["phase_idx"] + 1, len(self.PHASES) - 1)
        st["step_in_phase"] = 0
        st["recent_cube_dists"] = []

    def _advance_waypoint(self, reason):
        st = self.state
        if self.verbose:
            print(f"[push] wp {st['wp_idx']} done ({reason}); cube cell "
                  f"{int(which_cell(self._cube_xy()))}")
        st["reapproaches"] = 0
        st["cycle"] += 1

        # Random mode: just shove the cube somewhere new, forever.
        if self.push_mode == "random":
            st["waypoints"] = [self._random_waypoint()]
            st["wp_idx"] = 0
            self._begin_segment()
            return

        # Cells mode: advance along the Manhattan path.
        st["wp_idx"] += 1
        if st["wp_idx"] < len(st["waypoints"]):
            self._begin_segment()
            return
        # Final waypoint reached.
        if self.chase_random:
            start_cell = int(which_cell(self._cube_xy()))
            if start_cell < 0:
                start_cell = st["target_cell"]
            target_cell = self._pick_target_cell(start_cell)
            st["wp_idx"] = 0
            st["target_cell"] = target_cell
            st["waypoints"] = self._build_waypoints(start_cell, target_cell)
            self._begin_segment()
            if self.verbose:
                print(f"[push] cycle {st['cycle']}: new target cell {target_cell}")
        else:
            st["done"] = True
