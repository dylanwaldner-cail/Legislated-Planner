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
    (verified against that per-segment reference). Replaces the per-candidate Python double loop.

    `cube_half` is a scalar OR a per-candidate (B,) array of half-extents."""
    pos = np.asarray(pos, np.float64)
    B, S = pos.shape[0], pos.shape[1]
    if S < 2:
        return np.zeros(B, dtype=bool)
    # cube_half may be a SCALAR (one body model for every candidate) or a per-candidate (B,) array
    # (orientation-aware: h_eff = h*(|cos yaw|+|sin yaw|) read off each candidate's own latent). Build
    # the slab bounds at (B,1) so they broadcast against the (B,S-1) per-segment arrays below; a bare
    # np.array([cx-H, cy-H]) with an (B,) H would come out (2,B) and silently mis-broadcast.
    H = np.broadcast_to(np.asarray(cube_half, np.float64).reshape(-1), (B,))[:, None] + gm.CELL / 2.0
    cx, cy = gm.cell_center(int(cell))
    box_lo = (cx - H, cy - H); box_hi = (cx + H, cy + H)  # each (B,1)
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


def _obb_overlaps_cell(xy, yaw, cell, half):
    """(B,2) centers + (B,) yaws -> (B,) bool: does the ORIENTED square (half-side `half`, rotated by
    `yaw`) overlap cell `cell`'s axis-aligned rectangle (half CELL/2)?

    Exact, via the separating-axis theorem. The Minkowski shortcut the axis-aligned path uses --
    inflate the cell by cube_half and test the centre as a point -- is NOT available here: the
    Minkowski sum of a rotated square and a rectangle is an octagon, not a rectangle, so treating it
    as one would over-report near the corners by exactly the amount we are trying to stop
    approximating. Four candidate axes suffice for two convex quads (the box normals of each):
      a = x, y     -> cube radius h(|cos|+|sin|),  cell radius CELL/2
      a = u, v     -> cube radius h,               cell radius (CELL/2)(|cos|+|sin|)
    where u=(cos,sin), v=(-sin,cos). At yaw=0 both reduce to the axis-aligned test (asserted in
    tests). Overlap iff NO axis separates."""
    xy = np.asarray(xy, np.float64)
    yaw = np.broadcast_to(np.asarray(yaw, np.float64).reshape(-1), (xy.shape[0],))
    cx, cy = gm.cell_center(int(cell))
    dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
    c, sn = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
    R = gm.CELL / 2.0
    sep = (np.abs(dx) > R + half * (c + sn)) | (np.abs(dy) > R + half * (c + sn))
    cu, su = np.cos(yaw), np.sin(yaw)
    sep |= np.abs(dx * cu + dy * su) > half + R * (c + sn)      # axis u
    sep |= np.abs(-dx * su + dy * cu) > half + R * (c + sn)     # axis v
    return ~sep


def _obb_swept_hits_cell(pos, yaws, cell, half, n_sub=16):
    """(B,S,2) centre path + (B,S) yaws -> (B,) bool: does the ORIENTED cube touch cell `cell`
    anywhere along the path?

    The cube both translates AND rotates between recorded poses, and the exact swept region of a
    rotating square has curved boundaries. Rather than approximate that region, this SAMPLES n_sub
    intermediate poses per segment (linear in position, shortest-arc in yaw) and tests each exactly.
    Sampling can only ever MISS a graze between poses, never invent one, so the residual is a small
    one-directional under-detection. Measured at realistic stroke lengths (0.05-0.09 m) against the
    exact Liang-Barsky sweep at yaw=0: n_sub=4 -> 0.208%, 8 -> 0.100%, 16 -> 0.031%, 32 -> 0.022%.
    16 puts 4.4 mm between poses, against a 4.22 mm yaw-probe extent error and 38 mm mean WM+probe
    position error -- so the sampling is nowhere near the binding approximation.

    Yaw is interpolated on the SHORTEST ARC so a wrap across +/-pi does not sweep the long way round
    (which would drag the cube through orientations it never held)."""
    pos = np.asarray(pos, np.float64)
    yaws = np.asarray(yaws, np.float64)
    B, S = pos.shape[0], pos.shape[1]
    if S < 2:
        # MUST match _swept_hits_cell: with no segment there is nothing to sweep, so it reports
        # nothing -- returning the single-pose overlap here instead would make the oriented path
        # disagree with the axis-aligned one even at yaw=0, breaking opt-in-by-absence. Reachable:
        # _swept(allpos[:, 1:], ...) has S=1 whenever the trajectory is 2 frames long.
        return np.zeros(B, dtype=bool)
    hit = np.zeros(B, dtype=bool)
    for s in range(S - 1):
        p0, p1 = pos[:, s], pos[:, s + 1]
        y0 = yaws[:, s]
        dy = (yaws[:, s + 1] - y0 + np.pi) % (2 * np.pi) - np.pi      # shortest arc
        for k in range(n_sub + 1):
            t = k / n_sub
            hit |= _obb_overlaps_cell(p0 + t * (p1 - p0), y0 + t * dy, cell, half)
        if hit.all():
            break
    return hit


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
                   skip_current=False, actions=None, positions=None, strict_escape=False,
                   yaws=None, **_kw):
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
    # ORIENTATION. With a yaw estimate the cube is its true rotated square (exact SAT); without one
    # it falls back to the axis-aligned model every published run used. Both closures take the same
    # (positions, half) so the skip_current logic below is written once.
    if yaws is not None:
        _yw = yaws.detach().cpu().numpy() if hasattr(yaws, "detach") else np.asarray(yaws)
        _swept = lambda P, h: _obb_swept_hits_cell(P, _yw[:, -P.shape[1]:], cell, h)
        _foot = lambda xy, h, k: _obb_overlaps_cell(xy, _yw[:, k], cell, h)
    else:
        _swept = lambda P, h: _swept_hits_cell(P, cell, h)
        _foot = lambda xy, h, k: _footprint_in_cell(xy, cell, h)
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
            start_in_cushion = _foot(allpos[:, 0], cube_half, 0)    # start in annulus OR cell
            end_in_cushion = _foot(allpos[:, -1], cube_half, -1)     # endpoint rests in the margin
            enters_cushion = _swept(allpos, cube_half)              # path touches the margin (clear start)
            # RE-ENTRY. The grandfather exists ONLY to let a cube that INHERITED an illegal state escape,
            # so it exempts exactly ONE segment: frame 0 -> 1. Every later segment is enforced like any
            # clear-start candidate, so a path that escapes and then dips back in is pruned. Previously
            # this read `_swept_hits_cell(allpos, ...) & ~start_in_true`, which anchored the exemption at
            # the TREE ROOT and therefore handed it to every descendant -- a candidate could leave the
            # cell and re-enter freely as long as its endpoint was clear. Note the exemption is only ever
            # reachable at the root: from a clear start any entry is pruned on the spot, so no legal
            # descendant is ever inside. Escaping in one stroke is feasible -- measured over every run and
            # every delta, the deepest root needed 0.134 m to clear vs ~0.152 m p90 stroke travel.
            if strict_escape:
                inside_rule = end_in_cushion | _swept(allpos[:, 1:], cube_half)
            else:                                # LEGACY (default): endpoint-only once the root is inside
                start_in_true = _foot(allpos[:, 0], CUBE_HALF, 0)
                inside_rule = end_in_cushion | (_swept(allpos, CUBE_HALF) & ~start_in_true)
            hit = np.where(start_in_cushion,
                           inside_rule,                        # start inside: must EXIT the margin
                           enters_cushion)                     # start outside: never enter the margin
        else:                                                                     # legacy (non-RRT) callers: prior window
            hit = _swept(allpos[:, max(L - n_pred - 1, 0):], cube_half)
        viol |= torch.from_numpy(hit).to(latents.device)
    return viol


# in_cell prohibition = "never be in cell C" -> rest OR transit (full in-cell + cube-pos logic).
checker("in_cell")(partial(_cell_presence, use_occ=True, use_transit=True))
# passed_through prohibition = transit only.
checker("passed_through")(partial(_cell_presence, use_occ=False, use_transit=True))


def _off_grid(args, latents, n_pred, probes, cube_half, occ_thresh, skip_current=False, actions=None, positions=None, **_kw):
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


def _moving(args, latents, n_pred, probes, cube_half, occ_thresh, skip_current=False, actions=None, positions=None, **_kw):
    """FREEZE (stop-sign) checker: treat EVERY candidate as a violation. When a `moving`
    prohibition is active -- e.g. the red-sign law `sign(red) => [O]~moving` -- this prunes ALL
    planner candidates, so the RRT tree cannot extend past its root and the robot HOLDS at its
    current position (a full stop). Deliberately UNCONDITIONAL: an RRT candidate IS a stroke, i.e.
    a motion by construction, so `moving` holds for every one; skip_current is irrelevant because no
    motion at all is permitted. This is the pure "prune all nodes" freeze the stop-sign rule wants."""
    return torch.ones(latents.shape[0], dtype=torch.bool, device=latents.device)


# UNCONDITIONAL: this checker's answer does not depend on the candidate, so a planner can read the
# verdict alone and skip generating candidates at all (see Constraint.blocks_everything).
_moving.unconditional_violation = True

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
                 obligations=None, permissions=None, strict_escape=False):
        self.prohibitions = [parse_literal(p) for p in prohibitions]
        self.obligations = [parse_literal(o) for o in (obligations or [])]    # parsed for future
        self.permissions = [parse_literal(p) for p in (permissions or [])]    # enforcement (not yet acted on)
        self.probes = probes
        self.cube_half = cube_half
        self.occ_thresh = occ_thresh
        # STRICT ESCAPE (default False = the behaviour every published run was produced under).
        # False: once the ROOT is inside the keep-clear zone, only the candidate's ENDPOINT is
        #        checked -- it may leave and re-enter freely (see _cell_presence).
        # True:  the grandfather covers ONLY the frame 0->1 escape segment; every later segment is
        #        enforced, so a path that escapes and dips back in is pruned. Appropriate where a
        #        compensatory duty (R10 exit_cell) exists to retarget the planner at the exit;
        #        without one the objective still points across the cell and the step can freeze.
        self.strict_escape = bool(strict_escape)
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

    @property
    def blocks_everything(self):
        """True when some ENFORCED prohibition rejects every candidate regardless of its trajectory
        (today only `moving`, the stop-sign freeze -- flagged via the checker's
        `unconditional_violation` attribute). A pruning planner can then skip its search entirely
        instead of rediscovering the same total rejection once per sample. This is an OPTIMISATION
        ONLY: the plan it replaces is the identical empty plan (no legal stroke exists), so it cannot
        change which actions are taken -- only how long it takes to conclude none are legal."""
        return any(getattr(CHECKERS.get(n), "unconditional_violation", False)
                   for n, _ in self.prohibitions)

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
        _needs_pos = any(n in _POS_CHECKERS for n, _ in self.prohibitions)
        if positions is None:
            pp = self.probes.get("cube_position")
            if pp is not None and _needs_pos:
                positions = torch.stack([pp(visual_latents[:, t]) for t in range(L)], dim=1)   # (B,L,2)
        # ORIENTATION (opt-in, by ABSENCE): with no "cube_yaw" probe in the stack this stays None and
        # every checker uses the axis-aligned body model exactly as before -- so no existing run moves
        # unless the probe is registered. Probed once here and shared, like positions. Decoded from the
        # (sin 4t, cos 4t) head, which is why it is atan2(...)/4 and not a raw output.
        yaws = None
        yp = self.probes.get("cube_yaw")
        if yp is not None and _needs_pos:
            sc = torch.stack([yp(visual_latents[:, t]) for t in range(L)], dim=1)              # (B,L,2)
            yaws = torch.atan2(sc[..., 0], sc[..., 1]) / 4.0                                   # (B,L)
        viol = torch.zeros(B, dtype=torch.bool, device=visual_latents.device)
        # PER-PROHIBITION ATTRIBUTION (instrumentation only -- `viol` is unchanged, so enabling this
        # cannot alter planner behaviour). `last_pruned_by` counts candidates each prohibition flags;
        # a candidate flagged by two prohibitions counts under both. `last_pruned_solely_by` counts
        # candidates that ONLY that prohibition caught, which is the marginal contribution of the law.
        masks = {}
        for name, args in self.prohibitions:
            fn = CHECKERS.get(name)
            if fn is not None:
                m = fn(args, visual_latents, n_pred, self.probes, self.cube_half, self.occ_thresh,
                       skip_current=skip_current, actions=actions, positions=positions,
                       strict_escape=self.strict_escape, yaws=yaws)
                atom = f"{name}({','.join(map(str, args))})" if args else name
                masks[atom] = m
                viol |= m
        self.last_pruned_by = {a: int(m.sum()) for a, m in masks.items()}
        self.last_pruned_solely_by = {
            a: int((m & ~torch.stack([o for b, o in masks.items() if b != a]).any(dim=0)).sum())
               if len(masks) > 1 else int(m.sum())
            for a, m in masks.items()
        }
        self.last_considered = int(B)
        return viol
