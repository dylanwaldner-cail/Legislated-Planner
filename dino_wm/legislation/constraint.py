"""Translate DDL verdicts into an explicit, inspectable planning constraint.

reasoner.assess(facts) returns deontic verdicts as SYMBOLS — prohibitions / obligations /
permissions over arbitrary literals (in_cell(4), passed_through(5), near(human),
pushing_hard, ...). A Constraint turns those into a per-candidate violation check the planner
applies.

EXTENSIBILITY is the point — this must not calcify around the current cell/probe paradigm:
  * Verdicts are parsed GENERICALLY into (predicate_name, args). No predicate is special-cased.
  * Each predicate name maps to a registered CHECKER (the @checker registry) that knows how to
    evaluate it from probe outputs. The cell checker is the FIRST; new law vocabulary = a new
    @checker, with zero changes to Constraint or the planner.
  * Probes are passed as a DICT {name: Probe}, so a new checker just reads whatever probe(s) it
    needs (orientation, other objects, effort, ...) — not a fixed cube_cells/cube_position pair.
  * Prohibitions are enforced now (predicate must NOT hold -> violation). Obligations/permissions
    are parsed and stored for when we add their enforcement (soft reward / goal / relaxation).
Unsupported predicates are reported, not silently dropped.
"""
import re
from functools import partial

import numpy as np
import torch

from probes.probe_cube_cells import swept_cells, CUBE_HALF, gm

_GRID_HALF = gm.GRID_HALF   # workspace half-extent (m); "off the grid" = cube center beyond this


# ----------------------------------------------------------------- predicate parsing
def parse_literal(lit):
    """'in_cell(4)' -> ('in_cell', ['4']);  'pushing' -> ('pushing', []). Multi-arg tolerated
    (checkers read args[-1] as the cell), so a subject-carrying literal still parses."""
    m = re.match(r"^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*$", str(lit).strip())
    if not m:
        return (str(lit).strip(), [])
    name, argstr = m.group(1), m.group(2)
    return (name, [a.strip() for a in argstr.split(",")] if argstr else [])


# ----------------------------------------------------------------- checker registry
# A checker answers, for each candidate: does this predicate HOLD across the predicted
# trajectory? Signature:
#   (args, visual_latents (B,L,P,D), n_pred, probes:{name:Probe}, cube_half, occ_thresh,
#    skip_current=False) -> (B,) bool
# skip_current: exclude frame 0 (the cube's CURRENT position, given as context) from the check.
# Register new law vocabulary here; Constraint and the planner need no changes.
CHECKERS = {}


def checker(*names):
    def deco(fn):
        for n in names:
            CHECKERS[n] = fn
        return fn
    return deco


def _swept_hits_cell(pos, cell, cube_half):
    """(B,S,2) per-candidate CENTER path -> (B,) bool: does ANY segment pos[:,s]->pos[:,s+1] intersect
    cell `cell`'s AABB (center +/- (CELL/2 + cube_half))? Vectorized Liang-Barsky, EXACTLY equivalent to
    `probes.probe_cube_cells.swept_cells(pos[b,s], pos[b,s+1], cube_half)[cell]` OR-ed over segments s
    (verified against that per-segment reference). Replaces the per-candidate Python double loop."""
    pos = np.asarray(pos, np.float64)
    B, S = pos.shape[0], pos.shape[1]
    if S < 2:
        return np.zeros(B, dtype=bool)
    H = gm.CELL / 2.0 + cube_half
    cx, cy = gm.cell_center(int(cell))
    box_lo = np.array([cx - H, cy - H]); box_hi = np.array([cx + H, cy + H])
    p0 = pos[:, :-1, :]                                   # (B,S-1,2) segment starts
    d = pos[:, 1:, :] - p0                                # (B,S-1,2) segment deltas
    t0 = np.zeros((B, S - 1)); t1 = np.ones((B, S - 1))   # Liang-Barsky clip window per segment
    ok = np.ones((B, S - 1), dtype=bool)                  # not-yet-rejected
    for i in (0, 1):                                      # x, y axes
        di = d[..., i]; p0i = p0[..., i]
        par = np.abs(di) < 1e-12                          # segment parallel to axis i (matches scalar's 1e-12)
        # parallel axis: no hit unless the segment lies within the slab [lo,hi] on this axis
        ok &= ~(par & ((p0i < box_lo[i]) | (p0i > box_hi[i])))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (box_lo[i] - p0i) / di
            tb = (box_hi[i] - p0i) / di
        tmin = np.minimum(ta, tb); tmax = np.maximum(ta, tb)   # entry/exit t on axis i (nan where parallel)
        t0 = np.where(par, t0, np.maximum(t0, tmin))      # only tighten the window on non-parallel axes
        t1 = np.where(par, t1, np.minimum(t1, tmax))
    hit_seg = ok & (t0 <= t1)                             # (B,S-1) segment intersects the box
    return hit_seg.any(axis=1)                            # (B,) any segment along the path


def _footprint_in_cell(xy, cell, cube_half):
    """(B,2) cube centers -> (B,) bool: does the footprint (center +/- cube_half) overlap cell `cell`'s
    AABB (center +/- (CELL/2 + cube_half))? Single-point containment matching _swept_hits_cell's box.
    `xy` is the PROBE-read cube position (cube_position probe on the latent; GT only in the sim-oracle);
    gm.CELL / gm.cell_center are static grid coordinates. Used to detect a cube ALREADY in the cell at
    the current step, so its escape is grandfathered."""
    xy = np.asarray(xy, np.float64)
    H = gm.CELL / 2.0 + cube_half
    cx, cy = gm.cell_center(int(cell))
    return (np.abs(xy[:, 0] - cx) < H) & (np.abs(xy[:, 1] - cy) < H)


def _cell_presence(args, latents, n_pred, probes, cube_half, occ_thresh, use_occ, use_transit,
                   skip_current=False, actions=None, positions=None):
    """Does the cube REST IN (cell probe) and/or TRANSIT (position probe + swept_cells) the cell
    named by the last arg, anywhere in the predicted trajectory? Missing probes are skipped.

    skip_current: frame 0 is the cube's CURRENT (grounded) position. When True, occupancy is not
    flagged there (checked on frames [1,L) only), AND a candidate whose frame-0 footprint is ALREADY
    in the cell is EXEMPT from the transit check for that cell -- so a cube sitting in the cell can
    still plan an escape (grandfathered per-step, matching the metric's spawn grandfather in spirit).
    Otherwise the transit sweep covers the FULL root->end path, so a stroke that CLIPS through the cell
    from a clear start IS pruned (matching what the metric penalises). A stroke that ENDS in the cell
    is always pruned by occupancy, escaping-or-not."""
    cell = int(args[-1])
    B, L = latents.shape[:2]
    lo = 1 if skip_current else (L - n_pred)                                      # first frame to enforce on
    viol = torch.zeros(B, dtype=torch.bool, device=latents.device)
    if use_occ and probes.get("cube_cells") is not None:
        cp = probes["cube_cells"]
        for t in range(lo, L):
            viol |= cp(latents[:, t])[:, cell] > occ_thresh                       # in_cell logic
    if use_transit and (positions is not None or probes.get("cube_position") is not None):
        # Full PROBE-read root->end path (frame 0 included) so a CLIP through the cell from a clear
        # start is caught; skip_current then exempts only candidates ALREADY in the cell (escape).
        if positions is not None:                                                # SHARED positions (probed once in violations)
            allpos = positions.detach().cpu().numpy()                            # (B, L, 2)
        else:                                                                     # fallback: probe here
            pp = probes["cube_position"]
            allpos = np.stack([pp(latents[:, t]).detach().cpu().numpy() for t in range(L)], axis=1)
        if skip_current:
            # The cushion zone (cell inflated by the margin) is KEEP-CLEAR: no stroke's ENDPOINT may
            # rest in it. From a CLEAR start the whole path must avoid it (no pass-through). If the cube
            # already STARTS inside the cushion it is not frozen but is FORCED OUT -- its endpoint must
            # land OUTSIDE the cushion. The TRUE cell is never entered, except a cube genuinely inside
            # it may transit it while escaping.
            start_in_cushion = _footprint_in_cell(allpos[:, 0], cell, cube_half)    # start in annulus OR cell
            start_in_true = _footprint_in_cell(allpos[:, 0], cell, CUBE_HALF)       # start genuinely in the real cell
            end_in_cushion = _footprint_in_cell(allpos[:, -1], cell, cube_half)     # endpoint rests in the margin
            enters_cushion = _swept_hits_cell(allpos, cell, cube_half)              # path touches the margin (clear start)
            transits_true = _swept_hits_cell(allpos, cell, CUBE_HALF) & ~start_in_true  # enters the real cell (not escaping)
            hit = np.where(start_in_cushion,
                           end_in_cushion | transits_true,     # start inside: must EXIT the margin + not cross the cell
                           enters_cushion)                     # start outside: never enter the margin
        else:                                                                     # legacy (non-RRT) callers: prior window
            hit = _swept_hits_cell(allpos[:, max(L - n_pred - 1, 0):], cell, cube_half)
        viol |= torch.from_numpy(hit).to(latents.device)
    return viol


# in_cell prohibition = "never be in cell C" -> rest OR transit (full in-cell + cube-pos logic).
checker("in_cell")(partial(_cell_presence, use_occ=True, use_transit=True))
# passed_through prohibition = transit only.
checker("passed_through")(partial(_cell_presence, use_occ=False, use_transit=True))


def _off_grid(args, latents, n_pred, probes, cube_half, occ_thresh, skip_current=False, actions=None, positions=None):
    """Cube must not leave the workspace (|x| or |y| > GRID_HALF). Enforced two ways, OR'd:

      (a) OUTCOME  -- the predicted cube CENTER is off-grid at any predicted frame (position probe).
          CENTER-based on purpose: a footprint bound (GRID_HALF - cube_half) prunes the legal detour
          corners and freezes the plan.
      (b) INTENT (action space) -- the stroke's intended endpoint (start + disp) is off-grid. This
          catches the BOUNCE-BACK blind spot: the sim/WM shove the cube back onto the grid, so the
          OUTCOME never reads off-grid, yet the ACTION still INTENDED to push it off. `actions` is the
          candidate strokes in METERS [start_x,start_y,disp_x,disp_y]; start+disp is the pusher
          endpoint ~= the cube's intended endpoint for an aimed stroke.

    skip_current excludes frame 0 (the current position), matching the cell checkers."""
    B, L = latents.shape[:2]
    pp = probes.get("cube_position")
    if positions is None and pp is None and actions is None:
        raise RuntimeError("off_grid checker needs the 'cube_position' probe / shared positions (outcome) "
                           "or the candidate actions (intent); none given -- the workspace bound can't be enforced.")
    viol = torch.zeros(B, dtype=torch.bool, device=latents.device)
    if positions is not None or pp is not None:                                   # (a) OUTCOME
        lo = 1 if skip_current else (L - n_pred)
        if positions is not None:                                                 # SHARED positions (probed once in violations)
            viol |= (positions[:, lo:].abs() > _GRID_HALF).any(dim=-1).any(dim=1)  # any axis off-grid at any frame
        else:
            for t in range(lo, L):
                viol |= (pp(latents[:, t]).abs() > _GRID_HALF).any(dim=1)          # any axis off-grid
    if actions is not None:                                                       # (b) INTENT
        a = torch.as_tensor(actions, device=latents.device, dtype=torch.float32)
        end = a[..., :2] + a[..., 2:4]                                            # pusher endpoint ~= cube endpoint
        viol |= (end.abs() > _GRID_HALF).any(dim=-1)
    return viol


# off_grid prohibition = the cube must stay on the workspace grid.
checker("off_grid")(_off_grid)


def _moving(args, latents, n_pred, probes, cube_half, occ_thresh, skip_current=False, actions=None, positions=None):
    """FREEZE (stop-sign) checker: treat EVERY candidate as a violation. When a `moving`
    prohibition is active -- e.g. the red-sign law `sign(red) => [O]~moving` -- this prunes ALL
    planner candidates, so the RRT tree cannot extend past its root and the robot HOLDS at its
    current position (a full stop). Deliberately UNCONDITIONAL: an RRT candidate IS a stroke, i.e.
    a motion by construction, so `moving` holds for every one; skip_current is irrelevant because no
    motion at all is permitted. This is the pure "prune all nodes" freeze the stop-sign rule wants."""
    return torch.ones(latents.shape[0], dtype=torch.bool, device=latents.device)


# moving prohibition = a full stop (freeze): every candidate motion is illegal -> prune everything.
checker("moving")(_moving)


# prohibitions whose checker reads the cube-CENTER position -> violations() probes it ONCE and shares it
_POS_CHECKERS = {"in_cell", "passed_through", "off_grid"}


# ----------------------------------------------------------------- the constraint object
class Constraint:
    """Deontic verdict -> per-candidate violation check, handed to the planner. Dispatches each
    parsed prohibition to its registered checker and ORs the results. Add predicates via @checker;
    add probes by putting them in the probes dict — no edits here."""

    def __init__(self, prohibitions, probes, cube_half=CUBE_HALF, occ_thresh=0.5,
                 obligations=None, permissions=None):
        self.prohibitions = [parse_literal(p) for p in prohibitions]
        self.obligations = [parse_literal(o) for o in (obligations or [])]    # parsed for future
        self.permissions = [parse_literal(p) for p in (permissions or [])]    # enforcement (not yet acted on)
        self.probes = probes
        self.cube_half = cube_half
        self.occ_thresh = occ_thresh
        self.unsupported = sorted({n for n, _ in self.prohibitions if n not in CHECKERS})
        if self.unsupported:
            print(f"[constraint] no checker for prohibition predicate(s) {self.unsupported} "
                  f"-> NOT enforced. Add a @checker in legislation/constraint.py.")

    @classmethod
    def from_reasoner(cls, reasoner, facts, probes, **kw):
        v = reasoner.assess(facts)
        return cls(v["prohibitions"], probes, obligations=v["obligations"],
                   permissions=v["permissions"], **kw)

    def __repr__(self):
        fmt = lambda lst: [f"{n}({','.join(a)})" if a else n for n, a in lst]
        s = f"Constraint(prohibit={fmt(self.prohibitions)}"
        if self.obligations:
            s += f", oblige={fmt(self.obligations)}"
        if self.unsupported:
            s += f", UNSUPPORTED={self.unsupported}"
        return s + ")"

    @torch.no_grad()
    def violations(self, visual_latents, n_pred, skip_current=False, actions=None, positions=None):
        """(B,) bool: candidate violates if ANY enforced prohibition predicate holds across its
        predicted trajectory. skip_current excludes frame 0 (the cube's CURRENT position) from the
        check -- see _cell_presence. `actions` (candidate strokes in METERS, optional) lets checkers
        reason in ACTION space too (e.g. off_grid INTENT that survives WM bounce-back). `positions`
        (B,L,2) is the cube CENTER per frame; if not supplied it is probed ONCE here and shared across
        the position-using checkers (transit + off_grid), instead of each re-probing every frame.
        Obligations are not yet enforced (would be a soft reward / goal)."""
        B, L = visual_latents.shape[:2]
        if positions is None:
            pp = self.probes.get("cube_position")
            if pp is not None and any(n in _POS_CHECKERS for n, _ in self.prohibitions):
                positions = torch.stack([pp(visual_latents[:, t]) for t in range(L)], dim=1)   # (B,L,2)
        viol = torch.zeros(B, dtype=torch.bool, device=visual_latents.device)
        for name, args in self.prohibitions:
            fn = CHECKERS.get(name)
            if fn is not None:
                viol |= fn(args, visual_latents, n_pred, self.probes, self.cube_half, self.occ_thresh,
                           skip_current=skip_current, actions=actions, positions=positions)
        return viol
