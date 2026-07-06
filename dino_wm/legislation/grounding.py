"""Grounding: probe-registry outputs -> DDL ground facts.

Sits between the probe registry (perception) and the DDL reasoner. A Grounder receives a dict
of probe outputs for ONE state and derives every normatively-relevant predicate it can,
returning the fact list the reasoner consumes. The idiosyncratic geometry lives HERE (Python),
not in DDL (which can't do continuous geometry) and not in env/planning (never imported here).
Goal: maximize the normatively-relevant facts up front; the reasoner/laws decide what to use.

Each derived predicate is its OWN method (add a method per case we care about). Facts are
cube-indexed: predicate(cube, cell). Composite predicates (e.g. stroke overlap) use the
continuous position probe to SUPPLEMENT the discrete cell probe.

    from probes.registry import ProbeRegistry
    from legislation.grounding import Grounder
    from legislation.reasoner import LegislativeReasoner

    out = {}
    out.update(reg.forward(enc_tokens,  source="encoded"))     # cube_cells, cube_position
    out.update(reg.forward(pred_tokens, source="predicted"))   # cube_position_pred (end of stroke)
    facts = Grounder(out).ground()                             # ['in_cell(cube,4)', 'passed_through(cube,1)', ...]
    LegislativeReasoner().assess(facts)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probes.probe_cube_cells import swept_cells, CUBE_HALF   # cell geometry (numpy)


def _as_cube_rows(x):
    """probe output (tensor/array) -> numpy with a leading CUBE axis.
    (2,)->(1,2), (9,)->(1,9); (n_cubes, ·) kept as-is."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x[None] if x.ndim == 1 else x


class Grounder:
    """Derive DDL facts from one state's probe outputs. Each method emits one predicate family;
    `ground()` runs them all and returns the sorted fact list for the reasoner.

    probe_outputs: dict {probe_name: output} from the probe registry (merge the encoded and
        predicted passes into one dict). Methods read the keys they need and no-op if absent."""

    def __init__(self, probe_outputs, cube_names=("cube",), cube_half=CUBE_HALF, occ_thresh=0.5):
        self.out = dict(probe_outputs)
        self.cube_names = cube_names
        self.cube_half = cube_half
        self.occ_thresh = occ_thresh
        self.facts = set()

    # --- atomic: resting per-cell occupancy from the cell-occupancy probe ---
    def in_cell(self, key="cube_cells"):
        """cube_cells probe (per-cell sigmoid) -> in_cell(cube, c) for occupied cells."""
        occ = self.out.get(key)
        if occ is None:
            return
        for cube, row in zip(self.cube_names, _as_cube_rows(occ)):
            for c in np.where(row > self.occ_thresh)[0]:
                self.facts.add(f"in_cell({cube},{int(c)})")

    # --- composite: cells the cube SWEEPS over during the stroke (start obs -> predicted end) ---
    def stroke_overlap(self, start_key="cube_position", end_key="cube_position_pred"):
        """position probe at the stroke START (encoded obs) + END (predicted latent) ->
        passed_through(cube, c) for every cell the footprint crosses along the stroke. Catches
        mid-stroke transit invisible to the boundary-only occupancy. Kept DISTINCT from in_cell
        so the rest-vs-transit distinction survives; a DDL constitutive rule can union them
        (e.g. touched(C) :- in_cell(C); touched(C) :- passed_through(C)) if a law wants that."""
        c0, c1 = self.out.get(start_key), self.out.get(end_key)
        if c0 is None or c1 is None:
            return
        c0, c1 = _as_cube_rows(c0), _as_cube_rows(c1)
        for i, cube in enumerate(self.cube_names):
            for c in np.where(swept_cells(c0[i], c1[i], self.cube_half))[0]:
                self.facts.add(f"passed_through({cube},{int(c)})")

    # --- atomic: sign colour from the sign-colour classifier (scene fact for conditional laws) ---
    def sign_color(self, key="sign_color", names=("white", "red", "yellow", "green")):
        """sign-colour probe (class probs) -> sign(<colour>) for the argmax class. Enables laws
        conditioned on the sign, e.g. `sign(red) => [O]~in_cell(4)`. No-op if the probe is absent
        (not yet registered in probes.yaml). `names` must match the sign probe's class order."""
        probs = self.out.get(key)
        if probs is None:
            return
        for row in _as_cube_rows(probs):
            self.facts.add(f"sign({names[int(np.argmax(row))]})")

    def ground(self):
        """Run every grounder method and return the sorted DDL fact list."""
        self.in_cell()
        self.stroke_overlap()
        self.sign_color()
        return sorted(self.facts)
