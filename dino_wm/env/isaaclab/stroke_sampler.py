"""Mixed planar-stroke sampler for single-robot data collection / visualization.

Shared by scripts/collect_isaaclab_grid_data.py and inspect_single.py so they generate
strokes identically. Each stroke is INDEPENDENT (no coherent walks / cell targeting),
and per stroke we pick one of two regimes by `aimed_frac`:

  aimed   : a CONTACT stroke -- start behind the cube along a random heading and push
            THROUGH it. Teaches the cube dynamics (the ~frames the WM must get right).
  uniform : a contact point sampled uniformly over the workspace + a SHORT bounded push,
            blind to the cube. Covers the planner's whole action space, INCLUDING strokes
            that miss -> at plan time whatever (start, push) CEM/GD proposes is in-dist.

The DINO-WM deformable dataset is pure-uniform (memory: deformable-datagen-tricks), but
that works because granular material FILLS the workspace so uniform rarely misses. Our
cube is a small point-object, so pure-uniform would be ~90% no-ops -> we add the aimed
regime to recover contact density, keeping uniform for action-space coverage.

Action = [x_start, y_start, dx, dy] in env-local grid-plane meters: the pusher goes to
(x_start, y_start) and pushes by the displacement (dx, dy) (end = start + disp). This is
the deformable [start, displacement] parameterization (NOT [start, end]) so a box bound
on (dx, dy) at plan time caps the push to the in-distribution range. NOTE: an aimed
stroke starts `back` BEHIND the cube and pushes through it, so its displacement is
`back + push` (larger than a uniform short push) while the CUBE only travels ~`push` --
aimed and uniform thus have comparable cube motion despite different EE displacement.

Off-grid drift is contained by the wrapper's hard cube clamp, not by the sampler.
"""
from __future__ import annotations

import numpy as np

from .grid_metadata import GRID_CENTER_XY, GRID_HALF


class StrokeSampler:
    """Per-stroke mixed (aimed-contact / uniform) generator. reset_episode() is a no-op
    regime selector (kept for the collection loop's logging); sample(cube_xy) returns a
    fresh independent stroke each call."""

    _BARRIER_EPS = 0.03      # cube within this of the clamp edge (GRID_HALF) counts as "on the barrier"
    _RECENTER_MARGIN = 0.05  # cube center within this of GRID_HALF -> part of the cube hangs off the grid
    _RECENTER_JITTER = 0.4   # rad of heading noise around the to-center direction when recentering

    def __init__(self, rng, aimed_frac=0.6, push_max=0.08, start_margin=0.06,
                 back_range=(0.08, 0.12), aim_push_range=(0.05, 0.09), aim_offset_sd=0.0):
        self.rng = rng
        self.aimed_frac = aimed_frac             # P(stroke aims at the cube) vs uniform
        self.push_max = push_max                 # uniform: per-axis push displacement bound (m)
        self.back_range = back_range             # aimed: how far behind the cube the push starts
        self.aim_push_range = aim_push_range     # aimed: how far the cube is pushed (cube travel)
        # NEAR-MISS: aimed strokes aim at a point offset laterally from the TRUE cube by
        # ~N(0, aim_offset_sd) m, mimicking the planner aiming at its probe estimate (off by the
        # perception error). |offset| < cube_half (0.045) still contacts; larger grazes -> misses.
        # 0.0 = exact-contact aiming (old behavior). Set ~probe-error sd (~0.03) to match the planner.
        self.aim_offset_sd = float(aim_offset_sd)
        cx, cy = GRID_CENTER_XY
        h = GRID_HALF + start_margin             # start box: grid + margin so the pusher
        self.sx_lo, self.sx_hi = cx - h, cx + h  # can get behind a cube at/just past an edge
        self.sy_lo, self.sy_hi = cy - h, cy + h

    def reset_episode(self):
        """No per-episode regime (strokes are independent). Returns a mode string for
        the collection loop's logging."""
        return "mixed"

    def _keep_off_barrier(self, cube_xy, vec):
        """If the cube is ON a clamp edge (|cube[ax]| >= GRID_HALF - eps), flip any push
        component of `vec` that points further OUTWARD on that axis, so no stroke drives
        the cube into the barrier (where the clamp would just pin it). Interior cube ->
        unchanged. (On the +x edge the inward push is unreachable for this single arm,
        so it stall-aborts -> a no-op; still better than pinning the cube outward.)"""
        v = np.array(vec, np.float32)
        on = np.abs(cube_xy) >= (GRID_HALF - self._BARRIER_EPS)   # (2,) per-axis on-edge
        outward = np.sign(v) == np.sign(cube_xy)                  # push goes further out
        flip = on & outward
        v[flip] = -v[flip]
        return v

    def _uniform(self, cube_xy=None):
        sx = self.rng.uniform(self.sx_lo, self.sx_hi)
        sy = self.rng.uniform(self.sy_lo, self.sy_hi)
        disp = np.array([self.rng.uniform(-self.push_max, self.push_max),
                         self.rng.uniform(-self.push_max, self.push_max)], np.float32)
        if cube_xy is not None:
            disp = self._keep_off_barrier(cube_xy, disp)
        return np.array([sx, sy, disp[0], disp[1]], np.float32)

    def _aimed(self, cube_xy):
        partly_off = np.abs(cube_xy) >= (GRID_HALF - self._RECENTER_MARGIN)
        if partly_off.any():
            # part of the cube hangs off the grid -> aim toward the grid CENTER (with
            # jitter) so it's brought back IN, instead of being slid along the edge.
            base = float(np.arctan2(-cube_xy[1], -cube_xy[0]))
            theta = base + float(self.rng.normal(0.0, self._RECENTER_JITTER))
        else:
            theta = float(self.rng.uniform(-np.pi, np.pi))
        dirv = np.array([np.cos(theta), np.sin(theta)], np.float32)
        dirv = self._keep_off_barrier(cube_xy, dirv)  # safety net: never aim a push into the barrier
        back = float(self.rng.uniform(*self.back_range))
        push = float(self.rng.uniform(*self.aim_push_range))
        # NEAR-MISS: aim at a FALSE point offset laterally (⊥ to the push heading) from the true
        # cube, ~N(0, aim_offset_sd). Reproduces the planner aiming at its probe estimate: |offset|
        # < cube_half still contacts, larger grazes -> clean miss. sd=0 -> exact-contact (old).
        aim = cube_xy
        if self.aim_offset_sd > 0.0:
            perp = np.array([-dirv[1], dirv[0]], np.float32)   # unit ⊥ to the push heading
            aim = cube_xy + perp * float(self.rng.normal(0.0, self.aim_offset_sd))
        start = aim - dirv * back                     # descend behind the (false) aim point
        disp = dirv * (back + push)                   # push along dir; may pass beside the cube -> miss
        return np.concatenate([start, disp]).astype(np.float32)

    def sample(self, cube_xy=None):
        """One independent stroke [x_start, y_start, dx, dy] (env-local m). Aimed w.p.
        aimed_frac when cube_xy is given; uniform otherwise. When the cube sits on a
        clamp edge, the push is redirected inward so it's never driven into the barrier."""
        if cube_xy is not None:
            cube_xy = np.asarray(cube_xy, np.float32)
            if self.rng.rand() < self.aimed_frac:
                return self._aimed(cube_xy)
        return self._uniform(cube_xy)
