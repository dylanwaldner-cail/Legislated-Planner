"""Scripted pick-and-place expert for the DinoWMGrid env.

Single-env phase machine: each arm grabs its nearest cube and moves it to a
random cell. Phases:
    approach_above -> descend -> grasp -> lift -> move -> hover -> descend2
    -> release -> done

Used by inspect_trajectory.py (visualization) and
scripts/collect_isaaclab_grid_data.py (data collection). For batched data
collection the caller must force num_envs=1; a batched refactor would move
the per-arm state from scalars to length-N arrays.
"""
from __future__ import annotations

import numpy as np

from .grid_wrapper import ACTION_DIM_PER_AGENT
from .grid_metadata import GRID_CENTER_XY, GRID_HALF, CELL


# Must match the IK scale in dinowm_grid_env_cfg.py
_IK_SCALE = 0.5
_CUBE_NAMES = ("cube_red", "cube_blue")
_SIDES = ("left", "right")

# Per-side allowed place cells (cell_id = row*3 + col; col grows with +x, row
# with +y, so row 0 = front/-y, row 2 = back/+y). With the bases at ±0.55 each
# robot can reach across most of the grid width (the absolute far corner is
# near-singular), so columns are unrestricted; the split is by camera row-half:
# camera_left sits at -y, so the
# left robot owns the front rows {0,1} (all columns); camera_right sits at +y,
# so the right robot owns the back rows {1,2} (all columns). The middle row
# (cells 3,4,5) is shared, but neither robot enters the other's exclusive far
# row (left never row 2, right never row 0).
_LEFT_CELLS = (0, 1, 2, 3, 4, 5)   # front rows 0-1, all columns
_RIGHT_CELLS = (3, 4, 5, 6, 7, 8)  # back rows 1-2, all columns
_SIDE_CELLS = {"left": _LEFT_CELLS, "right": _RIGHT_CELLS}


class ExpertPickPlace:
    """Per-arm phase machine for pick-and-place. Single env only.

    All HEIGHT constants are wrist-target offsets relative to cube_z. The IK
    aims `panda_hand` (the wrist body) at the target, but the gripper TCP sits
    ~10.7cm below the wrist along the gripper axis. So a wrist target of
    cube_z + 10.7cm puts the fingertips at cube center for grasping.

    Stores `last_target` per side (world frame) so callers can render the
    commanded target with VisualizationMarkers and diff against EE position.
    """

    # Deterministic pick-and-place phases (used by --policy expert).
    PHASES_DETERMINISTIC = (
        "approach_above", "descend", "grasp",
        "lift", "move", "hover", "descend2", "release", "done",
    )
    # Looped pick-and-place phases (used by --policy noisy_expert): same
    # phases as deterministic but without "done" at the end. After release,
    # phase_idx wraps to 0 (approach_above) and a fresh random target xy
    # is sampled. The arm continuously picks up the cube, places it at a
    # random spot, picks it back up, places it elsewhere, etc. Each cycle
    # yields a grasp event + release event with a different placement, so
    # an episode produces many transitions instead of just one.
    PHASES_LOOP = (
        "approach_above", "descend", "grasp",
        "lift", "move", "hover", "descend2", "release",
    )

    APPROACH_HEIGHT = 0.25
    PICK_HEIGHT = 0.107
    LIFT_Z = 0.32   # was 0.26; raised so the carried block clears the grid
                    # instead of dragging when an arm reaches a far cell on the
                    # enlarged grid. Used by lift / move / hover phases.
    PLACE_Z = 0.13
    DIST_THRESHOLD = 0.010  # default / precise-phase arrival threshold
    # Per-phase arrival threshold. Transit phases (approach_above, lift, move)
    # advance loosely so the arm doesn't creep the last cm at each waypoint —
    # the next precise phase re-aims and converges tightly (descend re-targets
    # the cube before grasp; descend2 re-targets the place spot), so loose
    # transit doesn't cost grasp/place accuracy. Phases not listed (descend,
    # and the step-hold phases) fall back to DIST_THRESHOLD.
    PHASE_DIST_THRESHOLD = {
        "approach_above": 0.05,
        "lift": 0.05,
        "move": 0.05,
        "descend2": 0.015,  # placement — looser than grasp but still tidy
    }
    MAX_STEPS_PER_PHASE = 80
    GRASP_HOLD_STEPS = 10
    HOVER_HOLD_STEPS = 12
    RELEASE_HOLD_STEPS = 12
    PLATEAU_WINDOW = 6
    PLATEAU_TOL = 0.005

    # Random placement-target bounds for looped (noisy_expert) mode.
    RAND_XY_HALF = GRID_HALF  # legacy full-half bound; superseded by PLACE_* below.
    # Place-target band (env-local). With the bases at ±0.45 and the smaller
    # grid, x spans [-PLACE_X_HALF, +PLACE_X_HALF] for both robots; the band is
    # kept ~= the grid corner extent so continuous targets don't reach past the
    # deterministic far corner. The front/back (y) split stays: each robot only
    # places on its own camera's row-half — left robot front (y<=0, near
    # camera_left at -y), right robot back (y>=0, near camera_right at +y),
    # sharing the center row.
    PLACE_X_HALF = 0.19125  # scaled with the grid (~23.5% smaller than original)
    PLACE_Y_MIN = 0.0
    PLACE_Y_MAX = 0.19125

    def __init__(self, rng, env=None, verbose=True, chase_random=False, fixed_cells=None):
        """chase_random=False -> deterministic pick-and-place (expert):
            one fixed cell, run once, then done.
        chase_random=True  -> looped pick-and-place (noisy_expert):
            place at a fresh random xy each cycle, then loop back to
            re-grasp the just-released cube; never terminates.
        fixed_cells -> optional {side: cell_id} forcing the deterministic
            place target (e.g. {"left": 2, "right": 6}); overrides the random
            cell draw. Used to pin a specific placement for a demo run.
        """
        self.rng = rng
        self.state = {}
        self.last_target = {"left": None, "right": None}
        self._env = env
        self.verbose = verbose
        self.chase_random = chase_random
        self.fixed_cells = fixed_cells or {}
        # Reference gripper orientation (base frame, per side) captured at reset.
        # The IK gets a rotation command each step driving the wrist back to this
        # so it can't drift forward or spin (a zero rotation command gives the IK
        # no orientation feedback, which caused the wrist-forward / barrel-roll).
        self.ref_quat = {}
        self.PHASES = self.PHASES_LOOP if chase_random else self.PHASES_DETERMINISTIC

    def _sample_target_xy(self, side):
        # Markers land anywhere across the grid width (x full range), but each
        # robot stays on its own camera's row-half (y): left robot front
        # (y<=0, near camera_left), right robot back (y>=0, near camera_right).
        x = self.rng.uniform(-self.PLACE_X_HALF, self.PLACE_X_HALF)
        mag_y = self.rng.uniform(self.PLACE_Y_MIN, self.PLACE_Y_MAX)
        y = -mag_y if side == "left" else mag_y
        return np.array([x, y], dtype=np.float32)

    def reset(self):
        cube_pos = self._env.get_cube_positions()
        ee_pos = self._env.get_ee_positions()
        # Capture the ready-pose gripper orientation (base frame) to hold the
        # wrist at throughout the episode — this is the "wrist down" crane pose.
        ee_quats = self._env.get_ee_quats_base()
        for side in _SIDES:
            self.ref_quat[side] = ee_quats[side][0].copy()
        available = list(_CUBE_NAMES)

        def _nearest(side, choices):
            ee = ee_pos[side][0]
            return min(choices, key=lambda c: float(np.linalg.norm(cube_pos[c][0] - ee)))

        assignments = {}
        assignments["left"] = _nearest("left", available)
        available.remove(assignments["left"])
        assignments["right"] = _nearest("right", available)

        for side in _SIDES:
            cube = assignments[side]
            # Deterministic-mode place cell: a fixed_cells override wins (pins a
            # specific target for a demo run); otherwise draw a random cell from
            # this side's allowed region (its camera's row-half, any column).
            # Used only in deterministic mode.
            if side in self.fixed_cells:
                cell = int(self.fixed_cells[side])
            else:
                allowed = _SIDE_CELLS[side]
                cell = int(allowed[self.rng.randint(0, len(allowed))])
            target_xy = self._sample_target_xy(side) if self.chase_random else None
            self.state[side] = {
                "phase_idx": 0,
                "step_in_phase": 0,
                "cube": cube,
                "cell": cell,
                "target_xy": target_xy,  # used only in chase_random mode
                "cycle": 0,
                "recent_dists": [],
            }
            if self.verbose:
                if self.chase_random:
                    print(f"[expert] {side:5s}: pick {cube} (nearest) -> "
                          f"loop place/regrasp, first target_xy = "
                          f"({target_xy[0]:+.3f}, {target_xy[1]:+.3f})")
                else:
                    r, c = cell // 3, cell % 3
                    print(f"[expert] {side:5s}: pick {cube} (nearest) -> place at cell (row {r+1}, col {c})")

    def __call__(self, ee_positions, cube_positions):
        return {side: self._step_arm(side, ee_positions[side][0], cube_positions) for side in _SIDES}

    @staticmethod
    def _cell_xy(cell_id):
        cx, cy = GRID_CENTER_XY
        row = cell_id // 3
        col = cell_id % 3
        return np.array([
            cx - GRID_HALF + CELL / 2 + col * CELL,
            cy - GRID_HALF + CELL / 2 + row * CELL,
        ], dtype=np.float32)

    def _step_arm(self, side, ee_pos, cube_positions):
        st = self.state[side]
        phase = self.PHASES[st["phase_idx"]]
        cube_pos = cube_positions[st["cube"]][0]
        # Place target: random per-cycle xy in chase_random mode, otherwise
        # the fixed assigned cell center.
        place_xy = st["target_xy"] if self.chase_random else self._cell_xy(st["cell"])
        z_cube = float(cube_pos[2])

        OPEN, CLOSE = +1.0, -1.0
        if phase == "approach_above":
            target = np.array([cube_pos[0], cube_pos[1], z_cube + self.APPROACH_HEIGHT])
            gripper = OPEN
        elif phase == "descend":
            target = np.array([cube_pos[0], cube_pos[1], z_cube + self.PICK_HEIGHT])
            gripper = OPEN
        elif phase == "grasp":
            target = np.array([cube_pos[0], cube_pos[1], z_cube + self.PICK_HEIGHT])
            gripper = CLOSE
        elif phase == "lift":
            target = np.array([cube_pos[0], cube_pos[1], self.LIFT_Z])
            gripper = CLOSE
        elif phase == "move":
            target = np.array([place_xy[0], place_xy[1], self.LIFT_Z])
            gripper = CLOSE
        elif phase == "hover":
            target = np.array([place_xy[0], place_xy[1], self.LIFT_Z])
            gripper = CLOSE
        elif phase == "descend2":
            target = np.array([place_xy[0], place_xy[1], self.PLACE_Z])
            gripper = CLOSE
        elif phase == "release":
            target = np.array([place_xy[0], place_xy[1], self.PLACE_Z])
            gripper = OPEN
        else:  # done
            target = ee_pos
            gripper = OPEN

        self.last_target[side] = target.copy()
        delta_world = (target - ee_pos) / _IK_SCALE
        delta_base = self._env.world_to_base_delta(side, delta_world)
        delta_base = np.clip(delta_base, -1.0, 1.0).astype(np.float32)
        action = np.zeros((1, ACTION_DIM_PER_AGENT), dtype=np.float32)
        action[0, :3] = delta_base
        # Closed-loop orientation hold: command the rotation that drives the
        # wrist back to the captured reference orientation each step (instead of
        # a zero delta, which gave the IK no orientation feedback and let the
        # wrist drift forward / spin). /_IK_SCALE matches the action term's 0.5
        # scale, same as the position channel; clip to the action range.
        rot_aa = self._env.orientation_delta_base(side, self.ref_quat[side])[0]
        action[0, 3:6] = np.clip(rot_aa / _IK_SCALE, -1.0, 1.0).astype(np.float32)
        action[0, 6] = gripper

        st["step_in_phase"] += 1
        dist = float(np.linalg.norm(target - ee_pos))
        thresh = self.PHASE_DIST_THRESHOLD.get(phase, self.DIST_THRESHOLD)

        st["recent_dists"].append(dist)
        if len(st["recent_dists"]) > self.PLATEAU_WINDOW:
            st["recent_dists"].pop(0)
        plateaued = (
            len(st["recent_dists"]) >= self.PLATEAU_WINDOW
            and max(st["recent_dists"]) - min(st["recent_dists"]) < self.PLATEAU_TOL
        )

        if phase == "grasp":
            advance = st["step_in_phase"] >= self.GRASP_HOLD_STEPS
        elif phase == "hover":
            advance = st["step_in_phase"] >= self.HOVER_HOLD_STEPS
        elif phase == "release":
            advance = st["step_in_phase"] >= self.RELEASE_HOLD_STEPS
        elif phase == "done":
            advance = False
        else:
            advance = (
                dist < thresh
                or plateaued
                or st["step_in_phase"] >= self.MAX_STEPS_PER_PHASE
            )

        if advance:
            if self.chase_random:
                # Wrap from last phase (release) back to approach_above and
                # resample a fresh target for the next cycle.
                next_idx = (st["phase_idx"] + 1) % len(self.PHASES)
                if next_idx == 0:
                    st["target_xy"] = self._sample_target_xy(side)
                    st["cycle"] += 1
                    if self.verbose:
                        tx, ty = st["target_xy"]
                        print(f"[expert] {side:5s}: cycle {st['cycle']} start, "
                              f"new target_xy = ({tx:+.3f}, {ty:+.3f})")
            else:
                next_idx = min(st["phase_idx"] + 1, len(self.PHASES) - 1)
            if self.verbose:
                reason = (
                    "reached" if dist < thresh
                    else "plateau" if plateaued
                    else "timeout"
                )
                print(
                    f"[expert] {side:5s}: {phase} -> {self.PHASES[next_idx]} "
                    f"({reason}, steps={st['step_in_phase']}, dist={dist:.3f}, "
                    f"ee_z={ee_pos[2]:.3f}, target_z={target[2]:.3f})"
                )
            st["phase_idx"] = next_idx
            st["step_in_phase"] = 0
            st["recent_dists"] = []

        return action
