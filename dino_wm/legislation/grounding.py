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

import functools
import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probes.probe_cube_cells import swept_cells, CUBE_HALF   # cell geometry (numpy)

# Load grid_metadata by file path (like probe_cube_position) so grounding stays importable in
# plain python without pulling env/isaaclab/__init__ (IsaacLab). Gives CELL / N_CELLS / cell_center.
_gm_spec = importlib.util.spec_from_file_location(
    "grid_metadata", _REPO / "env" / "isaaclab" / "grid_metadata.py")
gm = importlib.util.module_from_spec(_gm_spec)
_gm_spec.loader.exec_module(gm)


@functools.lru_cache(maxsize=2)
def _grid_border_pairs(diagonal=False):
    """Static grid adjacency: directed (N, M) pairs where cell M borders cell N. ORTHOGONAL by
    default (centres one CELL apart in x XOR y); set diagonal=True to also include corner
    neighbours (8-connectivity). Computed from cell_center geometry, memoized (constant)."""
    cs = [np.asarray(gm.cell_center(c), float) for c in range(gm.N_CELLS)]
    tol = gm.CELL * 0.25
    pairs = []
    for n in range(gm.N_CELLS):
        for m in range(gm.N_CELLS):
            if n == m:
                continue
            dx, dy = np.abs(cs[n] - cs[m])
            ortho = (abs(dx - gm.CELL) < tol and dy < tol) or (abs(dy - gm.CELL) < tol and dx < tol)
            diag = diagonal and abs(dx - gm.CELL) < tol and abs(dy - gm.CELL) < tol
            if ortho or diag:
                pairs.append((n, m))
    return tuple(pairs)


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

    # --- goal: the goal cell, from the SAME cell probe run on the GOAL latent (see set_goal) ---
    def goal_cell(self, key="cube_cells"):
        """cube_cells probe on the GOAL latent -> goal_cell(k) for the goal cube's DOMINANT cell
        (single argmax; the goal is a resting centroid). 1-ary to match the in_cell law vocabulary.
        Returned DIRECTLY (not merged into ground()) so goal grounding stays isolated from the
        current-state facts -- a goal is not a place the cube currently occupies."""
        occ = self.out.get(key)
        if occ is None:
            return []
        return [f"goal_cell({int(np.argmax(_as_cube_rows(occ)[0]))})"]

    def ground(self):
        """Run every grounder method and return the sorted DDL fact list."""
        self.in_cell()
        self.stroke_overlap()
        self.sign_color()
        return sorted(self.facts)
