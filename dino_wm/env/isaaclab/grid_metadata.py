"""Cell geometry for the DinoWM grid task. Standalone (no IsaacLab imports).

3x3 grid centered on origin. Columns = COLOR_ORDER (red, yellow, blue);
rows 1-3 go front->back. cell_id = row * 3 + col in {0..8}; off-grid = -1.
CUBE_ORDER fixes the cube axis of cell_labels.pth.

Keep GRID_HALF / CELL / COLOR_ORDER in sync with
IsaacLab/.../dinowm_grid/grid_assets.py.
"""
from __future__ import annotations

import numpy as np

GRID_HALF = 0.15
CELL = 0.10
COLOR_ORDER = ("red", "yellow", "blue")
CUBE_ORDER = ("red", "yellow", "blue")
N_COLS = N_ROWS = 3
N_CELLS = N_COLS * N_ROWS
OFF_GRID = -1

STATE_DIM = 75
STATE_CUBE_BLOCK_DIM = 13
STATE_FIRST_CUBE_OFFSET = 36


def which_cell(xy):
    """xy as (x, y) or (..., 2) array -> cell_id(s); -1 if off-grid."""
    xy = np.asarray(xy)
    x, y = xy[..., 0], xy[..., 1]
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
    return COLOR_ORDER[c], r + 1


def cell_center(cell_id):
    if cell_id == OFF_GRID:
        return None
    r, c = _rc(cell_id)
    return -GRID_HALF + CELL / 2 + c * CELL, -GRID_HALF + CELL / 2 + r * CELL


def cell_labels_from_states(states):
    """(..., 75) states -> (..., 3) int64 cell ids per cube (red, yellow, blue)."""
    states = np.asarray(states)
    offsets = STATE_FIRST_CUBE_OFFSET + np.arange(3) * STATE_CUBE_BLOCK_DIM
    xy = np.stack([states[..., offsets], states[..., offsets + 1]], axis=-1)
    return which_cell(xy)


CELL_TABLE = tuple(
    {
        "cell_id": cid,
        "color": COLOR_ORDER[cid % N_COLS],
        "row": cid // N_COLS + 1,
        "center_xy": cell_center(cid),
        "extent": CELL,
    }
    for cid in range(N_CELLS)
)
