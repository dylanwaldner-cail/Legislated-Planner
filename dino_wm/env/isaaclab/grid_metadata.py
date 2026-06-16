"""Cell geometry for the DinoWM grid task. Standalone (no IsaacLab imports).

3x3 grid centered on origin. Cell colours follow CELL_COLORS; rows 1-3 go
front->back. cell_id = row * 3 + col in {0..8}; off-grid = -1. CUBE_ORDER fixes
the cube axis of cell_labels.pth (two cubes now: black, blue).

Keep GRID_HALF / CELL / CELL_COLORS in sync with
IsaacLab/.../dinowm_grid/grid_assets.py.
"""
from __future__ import annotations

import numpy as np

GRID_HALF = 0.2581875  # ~51.6cm grid (shrunk ~23.5%; kept in sync with grid_assets.py)
CELL = 0.172125        # = 2*GRID_HALF/3
# Grid center in env-local coords. Must match _GRID_CENTER_XY in
# IsaacLab/.../dinowm_grid/dinowm_grid_env_cfg.py.
GRID_CENTER_XY = (0.0, 0.0)
### HARNESS EDIT ### single-robot reachable zone: keep the cube out of the near-
# base column (x < this). The robot base is at x=-0.45; a cube in the nearest
# column (x~-0.17) can't be pushed because "behind" it is into the base. The
# spawn (wrapper) and push waypoints (expert) both clamp x >= REACH_X_MIN.
# Tune toward 0 if more of the near area is reachable, more negative if less.
REACH_X_MIN = -0.04
# Legacy per-column colour order — superseded by CELL_COLORS for cell labels.
# Kept only so existing imports don't break; not used for labeling anymore.
COLOR_ORDER = ("black", "blue")
CUBE_ORDER = ("black", "blue")
# Per-cell grid colours, indexed [row][col]; MUST match CELL_COLORS in
# grid_assets.py. Robots are left/right -> grid is green except the center
# row, whose left/right cells are yellow accents and center cell is red.
CELL_COLORS = (
    ("green",  "green", "green"),
    ("yellow", "red",   "yellow"),
    ("green",  "green", "green"),
)
N_COLS = N_ROWS = 3
N_CELLS = N_COLS * N_ROWS
OFF_GRID = -1

STATE_DIM = 62  # 2 arms * 18 (jpos+jvel) + 2 cubes * 13
STATE_CUBE_BLOCK_DIM = 13
STATE_FIRST_CUBE_OFFSET = 36

### HARNESS EDIT ### single-robot base case: [arm_jpos(9), arm_jvel(9), cube(13)] = 31
STATE_DIM_SINGLE = 31              # 1 arm * 18 (jpos+jvel) + 1 cube * 13
STATE_FIRST_CUBE_OFFSET_SINGLE = 18
CUBE_ORDER_SINGLE = ("cube",)


def which_cell(xy):
    """xy as (x, y) or (..., 2) array -> cell_id(s); -1 if off-grid.

    xy is in env-local frame; we subtract GRID_CENTER_XY internally so the
    rest of the math is in grid-local frame.
    """
    xy = np.asarray(xy)
    x = xy[..., 0] - GRID_CENTER_XY[0]
    y = xy[..., 1] - GRID_CENTER_XY[1]
    in_grid = (x >= -GRID_HALF) & (x < GRID_HALF) & (y >= -GRID_HALF) & (y < GRID_HALF)
    col = np.clip(np.floor((x + GRID_HALF) / CELL).astype(np.int64), 0, N_COLS - 1)
    row = np.clip(np.floor((y + GRID_HALF) / CELL).astype(np.int64), 0, N_ROWS - 1)
    out = np.where(in_grid, row * N_COLS + col, OFF_GRID).astype(np.int64)
    return out.item() if xy.ndim == 1 else out


def _rc(cell_id):
    return cell_id // N_COLS, cell_id % N_COLS


def cell_id_to_label(cell_id):
    if cell_id == OFF_GRID:
        return None
    r, c = _rc(cell_id)
    return CELL_COLORS[r][c], r + 1


def cell_center(cell_id):
    """Returns the cell center in env-local frame (includes GRID_CENTER_XY offset)."""
    if cell_id == OFF_GRID:
        return None
    r, c = _rc(cell_id)
    cx, cy = GRID_CENTER_XY
    return cx - GRID_HALF + CELL / 2 + c * CELL, cy - GRID_HALF + CELL / 2 + r * CELL


def cell_labels_from_states(states):
    """(..., 62) states -> (..., 2) int64 cell ids per cube (black, blue)."""
    states = np.asarray(states)
    offsets = STATE_FIRST_CUBE_OFFSET + np.arange(len(CUBE_ORDER)) * STATE_CUBE_BLOCK_DIM
    xy = np.stack([states[..., offsets], states[..., offsets + 1]], axis=-1)
    return which_cell(xy)


### HARNESS EDIT ### single-robot helpers
def cell_labels_from_states_single(states):
    """(..., 31) single-robot states -> (...,) int64 cell id for the one cube."""
    states = np.asarray(states)
    o = STATE_FIRST_CUBE_OFFSET_SINGLE
    xy = np.stack([states[..., o], states[..., o + 1]], axis=-1)
    return which_cell(xy)


def monotone_manhattan_cells(start_cell, target_cell, rng):
    """Random monotone Manhattan cell path from start to target (inclusive of
    both). Steps only toward the target along grid-adjacent cells, randomly
    interleaving row and column moves. rng is an np.random.RandomState."""
    r0, c0 = _rc(int(start_cell))
    r1, c1 = _rc(int(target_cell))
    dr = 1 if r1 >= r0 else -1
    dc = 1 if c1 >= c0 else -1
    moves = ["r"] * abs(r1 - r0) + ["c"] * abs(c1 - c0)
    rng.shuffle(moves)
    cells = [int(start_cell)]
    r, c = r0, c0
    for m in moves:
        if m == "r":
            r += dr
        else:
            c += dc
        cells.append(r * N_COLS + c)
    return cells


def random_point_in_cell(cell_id, rng, margin=0.03):
    """Random env-local (x, y) inside the cell, kept `margin` from the cell
    edges so the point (and a cube centered there) stays unambiguously in-cell.
    rng is an np.random.RandomState."""
    cx, cy = cell_center(int(cell_id))
    half = CELL / 2 - margin
    return np.array(
        [cx + rng.uniform(-half, half), cy + rng.uniform(-half, half)],
        dtype=np.float32,
    )


CELL_TABLE = tuple(
    {
        "cell_id": cid,
        "color": CELL_COLORS[cid // N_COLS][cid % N_COLS],
        "row": cid // N_COLS + 1,
        "center_xy": cell_center(cid),
        "extent": CELL,
    }
    for cid in range(N_CELLS)
)
