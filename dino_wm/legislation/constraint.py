"""Translate DDL verdicts into an explicit, inspectable planning constraint.

reasoner.assess(facts) returns deontic verdicts as SYMBOLS — prohibitions / obligations /
permissions over arbitrary literals (in_cell(4), passed_through(cube,5), near(human),
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

_GRID_HALF = gm.GRID_HALF   # workspace half-extent (m); "off the grid" = cube centre beyond this


# ----------------------------------------------------------------- predicate parsing
def parse_literal(lit):
    """'in_cell(cube,4)' -> ('in_cell', ['cube','4']);  'pushing' -> ('pushing', [])."""
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


def _cell_presence(args, latents, n_pred, probes, cube_half, occ_thresh, use_occ, use_transit,
                   skip_current=False, actions=None):
    """Does the cube REST IN (cell probe) and/or TRANSIT (position probe + swept_cells) the cell
    named by the last arg, anywhere in the predicted trajectory? Missing probes are skipped.

    skip_current: frame 0 is the cube's CURRENT position (given as context, not a prediction). When
    True it is NOT counted -- the prohibition governs where the cube GOES, not where it already IS.
    Occupancy is then checked on frames [1,L); transit only AMONG the predicted frames [1,L). Without
    this, a current footprint merely GRAZING the cell flags every candidate and freezes the planner
    (it can't even move out, because every escape stroke's sweep starts from the grazing frame 0)."""
    cell = int(args[-1])
    B, L = latents.shape[:2]
    lo = 1 if skip_current else (L - n_pred)                                      # first frame to enforce on
    viol = torch.zeros(B, dtype=torch.bool, device=latents.device)
    if use_occ and probes.get("cube_cells") is not None:
        cp = probes["cube_cells"]
        for t in range(lo, L):
            viol |= cp(latents[:, t])[:, cell] > occ_thresh                       # in_cell logic
    if use_transit and probes.get("cube_position") is not None:
        pp = probes["cube_position"]
        start = lo if skip_current else max(L - n_pred - 1, 0)                    # skip=frame 1; else include current frame
        pos = np.stack([pp(latents[:, t]).detach().cpu().numpy() for t in range(start, L)], axis=1)
        hit = np.zeros(B, dtype=bool)
        for b in range(B):                                                        # cube-pos / swept-path logic
            for s in range(pos.shape[1] - 1):
                if swept_cells(pos[b, s], pos[b, s + 1], cube_half)[cell]:
                    hit[b] = True
                    break
        viol |= torch.from_numpy(hit).to(latents.device)
    return viol


# in_cell prohibition = "never be in cell C" -> rest OR transit (full in-cell + cube-pos logic).
checker("in_cell")(partial(_cell_presence, use_occ=True, use_transit=True))
# passed_through prohibition = transit only.
checker("passed_through")(partial(_cell_presence, use_occ=False, use_transit=True))


def _off_grid(args, latents, n_pred, probes, cube_half, occ_thresh, skip_current=False, actions=None):
    """Cube must not leave the workspace (|x| or |y| > GRID_HALF). Enforced two ways, OR'd:

      (a) OUTCOME  -- the predicted cube CENTRE is off-grid at any predicted frame (position probe).
          CENTRE-based on purpose: a footprint bound (GRID_HALF - cube_half) prunes the legal detour
          corners and freezes the plan.
      (b) INTENT (action space) -- the stroke's intended endpoint (start + disp) is off-grid. This
          catches the BOUNCE-BACK blind spot: the sim/WM shove the cube back onto the grid, so the
          OUTCOME never reads off-grid, yet the ACTION still INTENDED to push it off. `actions` is the
          candidate strokes in METERS [start_x,start_y,disp_x,disp_y]; start+disp is the pusher
          endpoint ~= the cube's intended endpoint for an aimed stroke.

    skip_current excludes frame 0 (the current position), matching the cell checkers."""
    B, L = latents.shape[:2]
    pp = probes.get("cube_position")
    if pp is None and actions is None:
        raise RuntimeError("off_grid checker needs either the 'cube_position' probe (outcome) or the "
                           "candidate actions (intent); neither given -- the workspace bound can't be enforced.")
    viol = torch.zeros(B, dtype=torch.bool, device=latents.device)
    if pp is not None:                                                            # (a) OUTCOME
        lo = 1 if skip_current else (L - n_pred)
        for t in range(lo, L):
            viol |= (pp(latents[:, t]).abs() > _GRID_HALF).any(dim=1)             # any axis off-grid
    if actions is not None:                                                       # (b) INTENT
        a = torch.as_tensor(actions, device=latents.device, dtype=torch.float32)
        end = a[..., :2] + a[..., 2:4]                                            # pusher endpoint ~= cube endpoint
        viol |= (end.abs() > _GRID_HALF).any(dim=-1)
    return viol


# off_grid prohibition = the cube must stay on the workspace grid.
checker("off_grid")(_off_grid)


def _moving(args, latents, n_pred, probes, cube_half, occ_thresh, skip_current=False, actions=None):
    """FREEZE (stop-sign) checker: treat EVERY candidate as a violation. When a `moving`
    prohibition is active -- e.g. the red-sign law `sign(red) => [O]~moving` -- this prunes ALL
    planner candidates, so the RRT tree cannot extend past its root and the robot HOLDS at its
    current position (a full stop). Deliberately UNCONDITIONAL: an RRT candidate IS a stroke, i.e.
    a motion by construction, so `moving` holds for every one; skip_current is irrelevant because no
    motion at all is permitted. This is the pure "prune all nodes" freeze the stop-sign rule wants."""
    return torch.ones(latents.shape[0], dtype=torch.bool, device=latents.device)


# moving prohibition = a full stop (freeze): every candidate motion is illegal -> prune everything.
checker("moving")(_moving)


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
    def violations(self, visual_latents, n_pred, skip_current=False, actions=None):
        """(B,) bool: candidate violates if ANY enforced prohibition predicate holds across its
        predicted trajectory. skip_current excludes frame 0 (the cube's CURRENT position) from the
        check -- see _cell_presence. `actions` (candidate strokes in METERS, optional) lets checkers
        reason in ACTION space too (e.g. off_grid INTENT that survives WM bounce-back). Obligations
        are not yet enforced (would be a soft reward / goal)."""
        viol = torch.zeros(visual_latents.shape[0], dtype=torch.bool, device=visual_latents.device)
        for name, args in self.prohibitions:
            fn = CHECKERS.get(name)
            if fn is not None:
                viol |= fn(args, visual_latents, n_pred, self.probes, self.cube_half, self.occ_thresh,
                           skip_current=skip_current, actions=actions)
        return viol
