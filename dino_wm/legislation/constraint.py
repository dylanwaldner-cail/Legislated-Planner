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

from probes.probe_cube_cells import swept_cells, CUBE_HALF


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
#   (args, visual_latents (B,L,P,D), n_pred, probes:{name:Probe}, cube_half, occ_thresh) -> (B,) bool
# Register new law vocabulary here; Constraint and the planner need no changes.
CHECKERS = {}


def checker(*names):
    def deco(fn):
        for n in names:
            CHECKERS[n] = fn
        return fn
    return deco


def _cell_presence(args, latents, n_pred, probes, cube_half, occ_thresh, use_occ, use_transit):
    """Does the cube REST IN (cell probe) and/or TRANSIT (position probe + swept_cells) the cell
    named by the last arg, anywhere in the predicted trajectory? Missing probes are skipped."""
    cell = int(args[-1])
    B, L = latents.shape[:2]
    viol = torch.zeros(B, dtype=torch.bool, device=latents.device)
    if use_occ and probes.get("cube_cells") is not None:
        cp = probes["cube_cells"]
        for t in range(L - n_pred, L):
            viol |= cp(latents[:, t])[:, cell] > occ_thresh                       # in_cell logic
    if use_transit and probes.get("cube_position") is not None:
        pp = probes["cube_position"]
        start = max(L - n_pred - 1, 0)                                            # last history frame = current cube
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
    def violations(self, visual_latents, n_pred):
        """(B,) bool: candidate violates if ANY enforced prohibition predicate holds across its
        predicted trajectory. Obligations are not yet enforced (would be a soft reward / goal)."""
        viol = torch.zeros(visual_latents.shape[0], dtype=torch.bool, device=visual_latents.device)
        for name, args in self.prohibitions:
            fn = CHECKERS.get(name)
            if fn is not None:
                viol |= fn(args, visual_latents, n_pred, self.probes, self.cube_half, self.occ_thresh)
        return viol
