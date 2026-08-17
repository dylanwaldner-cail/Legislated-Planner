"""Shared geometry + calibration primitives for the conformal cushion work.

WHY THIS FILE EXISTS
--------------------
Every conformal script in this directory answers the same question with a different
statistical tool:

    "How much must we inflate the cube's footprint (the CUSHION, delta, in metres) so that
     the planner's BELIEVED swept footprint is a reliable over-approximation of the TRUE one?"

They therefore all need the same two things, which live here:
  1. the geometry that turns four logged points into ONE scalar per stroke (`nonconformity`), and
  2. the finite-sample conformal quantile (`conformal_quantile`).

THE NONCONFORMITY SCORE -- read this before touching anything else
------------------------------------------------------------------
Per stroke we have four planar points (see dump_probe_residuals.py):

    probe_start, probe_end   what the PLANNER believed (probe on encoded / WM-predicted latent)
    gt_start,    gt_end      what actually happened  (sim ground truth from states.pth)

The cube sweeps from start to end, so the footprint of a stroke is the Minkowski sum of the
cube square (half-extent h) with the segment [start, end]. Define, for a given forbidden cell:

    d_bel = signed distance from the BELIEVED swept footprint to the forbidden region
    d_true = signed distance from the TRUE swept footprint to the forbidden region

with the usual sign convention: POSITIVE = clear of the region, NEGATIVE = overlapping it
(magnitude = penetration depth).

The pruner with cushion delta rejects a stroke when the delta-inflated believed footprint
touches the region, i.e. when `d_bel <= delta`. A real violation happens when the true
footprint touches it, i.e. when `d_true < 0`. A LEAK is the bad case: pruner passed it
(`d_bel > delta`) but reality violated (`d_true < 0`).

Now the key algebra. Define the score

    s = d_bel - d_true

If delta >= s then d_bel = d_true + s <= d_true + delta, so whenever d_true < 0 we get
d_bel < delta and the stroke IS pruned. In words: **delta >= s is sufficient to prevent a
leak on that stroke.** Therefore choosing delta as the (1-alpha) conformal quantile of s over
an exchangeable calibration set certifies

    P(leak on a fresh stroke) <= alpha

That is the whole trick, and it is why `s` -- not a raw endpoint L2 error -- is the right
nonconformity score. Note that `s` depends on BOTH endpoints: the grounding error at the
start and the WM prediction error at the end both move d_bel away from d_true. Calibrating on
endpoint error alone silently prices only half the problem.

`s` is also monotone in delta by construction (larger delta can only prune more), which is
exactly the condition Conformal Risk Control requires -- see conformal_risk_control.py.

GEOMETRY SHORTCUT (exact, not an approximation)
-----------------------------------------------
Distance from a Minkowski sum to a set can be pushed onto the other operand:

    dist( segment (+) square(h),  box B )  ==  dist( segment,  B (+) square(h) )

and for an axis-aligned box B, `B (+) square(h)` is just B grown by h on every side. So we
never build the swept polygon at all: we compute the signed distance from the raw SEGMENT to
the forbidden cell box GROWN BY h. That is exact, and cheap.

The only discretisation is that we minimise the box SDF along the segment by dense sampling
(`SEGMENT_SAMPLES`). At the default resolution the error is far below a micrometre for our
stroke lengths (<= 0.09 m), i.e. orders of magnitude under the millimetre-scale effects we
report. Raise it if you ever lengthen strokes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------
# Grid / cube constants. These MUST match the deployment geometry, so we import them from
# the single source of truth rather than re-declaring magic numbers here.
# ---------------------------------------------------------------------------------------
import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from env.isaaclab import grid_metadata as gm      # noqa: E402  cell_center / CELL / which_cell
from probes.probe_cube_cells import CUBE_HALF     # noqa: E402  0.045 m -- the ONE true half-extent

#: How finely we march along a stroke segment when minimising the box SDF. 512 samples over a
#: <=0.09 m stroke is a <0.2 mm step, and the SDF is 1-Lipschitz, so the induced error is
#: bounded by half a step. Well under anything we report.
SEGMENT_SAMPLES = 512

#: Default output location. Everything reproducible lands here so a reviewer can re-run the
#: statistics without re-running the (expensive) encode pass.
CONFORMAL_DIR = _REPO / "data" / "conformal"


# =========================================================================================
# Geometry
# =========================================================================================
def cell_box(cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (centre, half_extent) of grid `cell` as float arrays of shape (2,).

    Delegates to env.isaaclab.grid_metadata so this file cannot drift from the geometry the
    planner and the law checker actually use. A cell is a CELL x CELL square, so its
    half-extent is CELL/2 on both axes.
    """
    centre = np.asarray(gm.cell_center(int(cell)), dtype=np.float64).reshape(2)
    half = np.full(2, gm.CELL / 2.0, dtype=np.float64)
    return centre, half


def _box_sdf(points: np.ndarray, centre: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Exact signed distance from `points` (..., 2) to an axis-aligned box.

    Standard 2-D AABB SDF. Negative inside (magnitude = penetration depth), positive outside
    (magnitude = Euclidean distance to the boundary).
    """
    q = np.abs(points - centre) - half                      # per-axis overshoot, negative if inside
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)    # distance once we're beyond a face
    inside = np.minimum(np.max(q, axis=-1), 0.0)             # deepest axis penetration (<=0)
    return outside + inside


def _square_box_penetration(points: np.ndarray, centre: np.ndarray, half: np.ndarray,
                            cube_half: float) -> np.ndarray:
    """Exact penetration of a CUBE SQUARE centred at `points` into an axis-aligned box.

    Needed because the Minkowski shortcut used by `swept_signed_distance` is exact for
    SEPARATION but not for PENETRATION DEPTH: growing the box by h and measuring the bare
    segment overstates how deep the swept square actually reaches (verified against brute
    force: sign always agrees, magnitude differs by up to ~4 cm on overlapping strokes).

    The exact form is cheap because the per-axis cube offsets are INDEPENDENT. We want

        min_{o in [-h,h]^2}  max_i ( |x_i + o_i - c_i| - e_i )

    and since term i depends only on o_i, the min of the max equals the max of the per-axis
    minima. Choosing o_i to pull x_i toward the box centre gives |x_i - c_i| -> max(0, |x_i - c_i| - h),
    hence the expression below. Returns <= 0 when the square overlaps the box.
    """
    d = np.maximum(np.abs(points - centre) - float(cube_half), 0.0) - half
    return np.max(d, axis=-1)


def swept_signed_distance(start_xy, end_xy, cell: int, cube_half: float) -> np.ndarray:
    """Signed distance from the SWEPT CUBE FOOTPRINT of stroke [start->end] to `cell`.

    Uses the Minkowski identity documented at the top of this file: instead of building the
    swept polygon and measuring it against the cell box, we grow the cell box by the cube's
    half-extent and measure the bare segment against that. Exact, and much simpler.

    Args:
        start_xy: (N, 2) stroke start positions, metres.
        end_xy:   (N, 2) stroke end positions, metres.
        cell:     index of the forbidden grid cell.
        cube_half: cube half-extent in metres. Pass the TRUE half-extent for d_true and for
            d_bel alike -- the cushion delta is NOT folded in here. Keeping delta out of the
            geometry is what lets the calibration scripts sweep delta afterwards without
            recomputing anything.

    Returns:
        (N,) signed distances. Negative => the footprint overlaps the cell.
    """
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(-1, 2)
    end_xy = np.asarray(end_xy, dtype=np.float64).reshape(-1, 2)
    centre, half = cell_box(cell)
    grown = half + float(cube_half)                          # B (+) square(h)

    t = np.linspace(0.0, 1.0, SEGMENT_SAMPLES).reshape(1, -1, 1)   # (1, S, 1)
    seg = start_xy[:, None, :] + t * (end_xy - start_xy)[:, None, :]  # (N, S, 2)
    return _box_sdf(seg, centre, grown).min(axis=1)          # deepest point along the stroke


def swept_penetration(start_xy, end_xy, cell: int, cube_half: float) -> np.ndarray:
    """How deep the swept footprint actually reaches INTO `cell`, in metres (0 if clear).

    Use this -- not `-swept_signed_distance(...)` -- whenever the MAGNITUDE of an overlap
    matters (e.g. a graded severity loss). `swept_signed_distance` is exact in sign and exact
    for separation, but its negative branch overstates depth because the Minkowski shortcut
    does not preserve penetration; `_square_box_penetration` computes the true value.

    Returns (N,) non-negative depths.
    """
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(-1, 2)
    end_xy = np.asarray(end_xy, dtype=np.float64).reshape(-1, 2)
    centre, half = cell_box(cell)

    t = np.linspace(0.0, 1.0, SEGMENT_SAMPLES).reshape(1, -1, 1)
    seg = start_xy[:, None, :] + t * (end_xy - start_xy)[:, None, :]
    deepest = _square_box_penetration(seg, centre, half, cube_half).min(axis=1)
    return np.maximum(0.0, -deepest)


def nonconformity(probe_start, probe_end, gt_start, gt_end, cell: int,
                  cube_half: float, mode: str = "tight") -> np.ndarray:
    """Per-stroke nonconformity score, in metres. `delta >= s` means the pruner catches it.

    IS CONFORMAL SEVERITY-AWARE? No -- and deliberately so. The quantile COUNTS strokes above a
    threshold, so a 1 mm intrusion and a 5 cm drive-through contribute equally. If you want the
    margin to care about how bad a breach was, that is a different object: an explicitly graded
    loss under Conformal Risk Control (conformal_risk_control.py --loss severity). Keep the two
    separate; do not let severity leak into the default score.

    mode='tight' (DEFAULT, use this)
        s = d_bel                     on strokes that actually violated (d_true < 0)
        s = -inf                      otherwise (such a stroke cannot leak, so it constrains
                                      nothing)
        Then {s > delta} is EXACTLY the leak event {d_bel > delta AND d_true < 0}, so the
        (1-alpha) conformal quantile certifies P(leak) <= alpha with no slack, and the score is
        severity-blind: only the planner's BELIEVED clearance sets the margin.

    mode='gap'  (the looser variant, kept for comparison only)
        s = d_bel - d_true
        Also sufficient -- delta >= s implies delta >= d_bel whenever d_true < 0 -- but the
        slack it adds is precisely the penetration depth, so deep violations inflate the
        cushion more than shallow ones. That is severity-weighting by accident. Strictly more
        conservative than 'tight'; useful only to show how much the tightening buys.

    A NOTE ON THE DENOMINATOR (decide this consciously)
    ----------------------------------------------------
    P(leak) is per stroke over whatever population you calibrate on. Calibrating over ALL
    strokes -- most of which are nowhere near the forbidden cell -- gives a very low base rate
    and therefore a trivially satisfiable alpha. That is honest but weak. Restricting to
    strokes the planner would plausibly consider near the cell gives a conditional, far more
    meaningful rate. Choose with --near_only in the calibration scripts and SAY which one the
    reported alpha refers to.
    """
    d_bel = swept_signed_distance(probe_start, probe_end, cell, cube_half)
    d_true = swept_signed_distance(gt_start, gt_end, cell, cube_half)
    if mode == "gap":
        return d_bel - d_true
    if mode != "tight":
        raise ValueError(f"unknown nonconformity mode {mode!r} (expected 'tight' or 'gap')")
    s = np.full(d_bel.shape, -np.inf, dtype=np.float64)
    violated = d_true < 0.0
    s[violated] = d_bel[violated]
    return s


# =========================================================================================
# Conformal calibration
# =========================================================================================
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of `scores` at level 1-alpha.

    The correction that makes this conformal rather than a plain empirical quantile is the
    index: we take the ceil((n+1)(1-alpha))-th smallest score, not the (1-alpha) empirical
    one. That +1 is what buys the guarantee

        P(s_{new} <= qhat) >= 1 - alpha

    for an exchangeable calibration set, with NO distributional assumptions -- which is
    precisely why this tool suits our heavy-tailed WM error, where bounded-disturbance methods
    (HJ reachability, CBFs) would need a worst case we do not have.

    Returns +inf when n is too small to support the requested alpha; that is the honest
    answer (the data cannot certify that level), not an error to paper over.
    """
    # NOTE: drop NaN only. -inf entries (non-violating strokes under the 'tight' score) MUST be
    # retained -- they are legitimate calibration points that sort to the bottom, and removing
    # them would shrink n and silently break the coverage guarantee.
    s = np.asarray(scores, dtype=np.float64)
    s = s[~np.isnan(s)]
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")                                   # need at least ceil((n+1)(1-a)) points
    return float(np.sort(s)[k - 1])


def empirical_coverage(scores: np.ndarray, qhat: float) -> float:
    """Fraction of `scores` at or below `qhat` -- i.e. realised coverage of a chosen cushion.

    Use this on a HELD-OUT set (never the calibration set) to check the guarantee empirically.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[~np.isnan(s)]                       # keep -inf: those strokes ARE covered
    return float((s <= qhat).mean()) if s.size else float("nan")


def leak_rate(scores: np.ndarray, delta: float) -> float:
    """Fraction of strokes that would LEAK at cushion `delta` (i.e. s > delta).

    This is the monotone loss used by Conformal Risk Control: non-increasing in delta by
    construction, which is the condition CRC needs to invert it into a certified bound.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[~np.isnan(s)]                       # keep -inf: those strokes simply never leak
    return float((s > delta).mean()) if s.size else float("nan")


# =========================================================================================
# Grouping (for Mondrian / group-conditional conformal)
# =========================================================================================
def push_direction_bin(start_xy, end_xy, n_bins: int = 8) -> np.ndarray:
    """Bin each stroke's heading into one of `n_bins` equal angular sectors.

    Direction matters because the WM's error is anisotropic -- the known +x blind spot means a
    single global cushion over-inflates easy headings and under-inflates hard ones.
    """
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(-1, 2)
    end_xy = np.asarray(end_xy, dtype=np.float64).reshape(-1, 2)
    d = end_xy - start_xy
    theta = np.arctan2(d[:, 1], d[:, 0])                      # (-pi, pi]
    frac = (theta + np.pi) / (2.0 * np.pi)                    # -> [0, 1)
    return np.clip((frac * n_bins).astype(int), 0, n_bins - 1)


def group_keys(cells, dir_bins) -> np.ndarray:
    """Combine (cell, direction bin) into one integer group id per stroke."""
    cells = np.asarray(cells, dtype=int).reshape(-1)
    dir_bins = np.asarray(dir_bins, dtype=int).reshape(-1)
    return cells * 1000 + dir_bins


# =========================================================================================
# IO
# =========================================================================================
def load_residuals(path) -> dict:
    """Load a residual dump written by dump_probe_residuals.py.

    Returns a plain dict of arrays plus a 'meta' dict carrying provenance (checkpoint, probe,
    git sha, ...). Always check `meta` before reporting numbers -- it records exactly which
    WM/probe produced the predictions.
    """
    with np.load(Path(path), allow_pickle=True) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return out


def save_result(name: str, payload: dict, outdir=None) -> Path:
    """Write one calibration result as JSON under data/conformal/ for reproducibility."""
    outdir = Path(outdir) if outdir is not None else CONFORMAL_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[conformal] wrote {path}")
    return path
