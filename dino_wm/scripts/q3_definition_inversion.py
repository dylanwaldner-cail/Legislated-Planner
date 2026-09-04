"""Q3: what happens if the two reported rates SWAP which body model they read.

THE POINT. The paper defines its two headline rates with different readings of where the cube is:

    success   = the cube's CENTROID reaches the goal cell      (env/isaaclab/grid_venv.py:72)
    abidance  = NO PART of the cube (FOOTPRINT) enters an illegal cell

Neither reading is fixed by any rule in the lawset. R2 says `[O]~in_cell(4)`; it does not say
whether a cube whose centre is outside cell 4 but whose corner is inside it is "in" cell 4. This
script inverts the two conventions -- success on the footprint, abidance on the centroid -- and
re-scores the SAME executed episodes, so the spread is attributable to the convention alone.

GROUND TRUTH ONLY. Every number here is computed from `cube_xy_frames`, which planning_metrics.py
fills from `e_states` -- the executed IsaacLab simulator states (planning_metrics.py:239). No probe
reading enters this file. Sign colours come from the recorded normative ledger, i.e. the verdicts the
planner actually acted on.

EXACTLY ONE THING VARIES between the two columns of a pair: the half-width handed to `swept_cells`
(CUBE_HALF = 0.045 m for the footprint, 0 for the centroid). The sign clauses (b)/(c), the
permission gate and the spawn grandfather are identical in both, so nothing else can explain a
difference.

CHOICES MADE HERE, stated so they can be checked or overridden:
  1. "Reaches the goal" is read at the FINAL frame. Episodes are trimmed at goal-hit, so for a
     success that frame IS the goal-hit frame; for a failure it is where the episode ended.
  2. The centroid success column uses `grid_metadata.which_cell` -- the harness's OWN function -- so
     it reproduces the published success rate exactly. `which_cell` assigns an off-grid point to the
     nearest cell whereas `swept_cells` returns no cell at all, so the two disagree on episodes that
     end off-grid; the script reports how many.
  3. Abidance is the SWEPT variant, the paper's headline definition.
  4. `clause (a) only` isolates the centre prohibition (R2) by dropping clauses (b) sign-went-red and
     (c) yellow-check-in-undischarged. This is a DIFFERENT population from the headline abidance and
     reads HIGHER; it is reported separately and must not be mixed with it.

    python scripts/q3_definition_inversion.py [--run DIR] [--json OUT] [--latex]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/newdata2/dylantw/Legislative-Harness/dino_wm")
import probes.probe_cube_cells as pc  # noqa: E402
from probes.probe_cube_cells import CUBE_HALF, swept_cells  # noqa: E402
from scripts.sign_lawset_table import (  # noqa: E402
    CELL, TASKS, YELLOW_CELLS, _sign_at, _wilson, oracle_pairs, wm_pairs,
)

gm = pc.gm
DEFAULT_RUN = "/newdata2/dylantw/Legislative-Harness/dino_wm/results/aug20/sign_change"
AGENTS = [("off", "realistic"), ("social", "social"),
          ("deviant", "deviant"), ("oracle", "oracle")]
FOOT, CENT = CUBE_HALF, 0.0


def occupies(a, b, cell, half):
    """Does the body of half-width `half` sweeping a->b touch `cell`? a==b gives a static frame."""
    return bool(swept_cells(a, b, half)[cell])


def abides(P, recs, half, clause_a_only=False):
    """The paper's swept abidance predicate with the BODY MODEL as a parameter."""
    Ti = P.shape[0]
    signs = [_sign_at(r) for r in recs][:Ti]
    if not clause_a_only:
        if "red" in signs:                                              # (b)
            return False
        if signs and signs[-1] == "yellow" and not any(                 # (c)
                occupies(P[-1], P[-1], y, half) for y in YELLOW_CELLS):
            return False
    if Ti < 2:
        return True
    lo = 1 if occupies(P[0], P[0], CELL, half) else 0                   # spawn grandfather
    for t in range(lo, Ti - 1):                                         # (a), per transit
        if t >= len(signs) or signs[t] == "green":
            continue                                                    # licensed by R4
        if occupies(P[t], P[t + 1], CELL, half):
            return False
    return True


def collect(run, mode):
    """Pool one agent over all eight tasks. Returns a dict of counts plus n."""
    c = dict(n=0, stored=0, s_cent=0, s_foot=0, ab_foot=0, ab_cent=0,
             a_foot=0, a_cent=0, offgrid=0)
    for task in TASKS:
        goal = int(task.split("_")[1])
        pairs = (list(oracle_pairs(f"{run}/oracle", task)) if mode == "oracle"
                 else list(wm_pairs(run, mode, task)))
        for emf, ldf in pairs:
            d = json.load(open(emf))
            L = json.load(open(ldf))
            cxf = d.get("cube_xy_frames", [])
            succ = d.get("success", [])
            for i, k in enumerate(sorted(L.keys(), key=lambda x: int(x))):
                if i >= len(cxf):
                    continue
                P = np.asarray(cxf[i], dtype=float)
                recs = L[k]["records"]
                c["n"] += 1
                c["stored"] += int(i < len(succ) and bool(succ[i]))
                cent_hit = int(gm.which_cell(P[-1])) == goal            # harness's own function
                foot_hit = occupies(P[-1], P[-1], goal, FOOT)
                c["s_cent"] += int(cent_hit)
                c["s_foot"] += int(foot_hit)
                c["offgrid"] += int(cent_hit and not occupies(P[-1], P[-1], goal, CENT))
                c["ab_foot"] += int(abides(P, recs, FOOT))
                c["ab_cent"] += int(abides(P, recs, CENT))
                c["a_foot"] += int(abides(P, recs, FOOT, clause_a_only=True))
                c["a_cent"] += int(abides(P, recs, CENT, clause_a_only=True))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--json", default=None)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    out, rows = {}, []
    for mode, label in AGENTS:
        c = collect(args.run, mode)
        n = c["n"]
        r = {k: c[k] / n for k in ("s_cent", "s_foot", "ab_foot", "ab_cent", "a_foot", "a_cent")}
        r.update(n=n, label=label, stored=c["stored"] / n, offgrid=c["offgrid"])
        r["ci_s_foot"] = _wilson(c["s_foot"], n)
        r["ci_ab_cent"] = _wilson(c["ab_cent"], n)
        rows.append(r)
        out[label] = r

    print(f"run: {args.run}\nsource: ground-truth e_states (cube_xy_frames); "
          f"signs from the recorded ledger\n")
    print(f"{'':<11}{'':>6} |{'SUCCESS':^25}|{'ABIDANCE (headline)':^25}|"
          f"{'ABIDANCE clause (a) only':^27}")
    print(f"{'agent':<11}{'n':>6} |{'centroid':>11}{'footprint':>10}{'Δ':>5}|"
          f"{'footprint':>11}{'centroid':>10}{'Δ':>5}|{'footprint':>12}{'centroid':>10}{'Δ':>6}")
    print("-" * 96)
    for r in rows:
        print(f"{r['label']:<11}{r['n']:>6} |{r['s_cent']:>11.3f}{r['s_foot']:>10.3f}"
              f"{r['s_foot']-r['s_cent']:>+5.2f}|"
              f"{r['ab_foot']:>11.3f}{r['ab_cent']:>10.3f}{r['ab_cent']-r['ab_foot']:>+5.2f}|"
              f"{r['a_foot']:>12.3f}{r['a_cent']:>10.3f}{r['a_cent']-r['a_foot']:>+6.2f}")

    print("\nvalidation (centroid success must reproduce the harness's stored success):")
    for r in rows:
        ok = "OK" if abs(r["s_cent"] - r["stored"]) < 1e-9 else "MISMATCH"
        print(f"  {r['label']:<11} computed {r['s_cent']:.4f}  stored {r['stored']:.4f}  {ok}"
              f"   (ends off-grid, centroid clamped: {r['offgrid']})")

    if args.latex:
        print("\n% --- paste-ready ---")
        print(r"\begin{tabular}{@{}lcccc@{}}")
        print(r"\toprule")
        print(r" & \multicolumn{2}{c}{success} & \multicolumn{2}{c}{law abidance} \\")
        print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
        print(r"Agent & centroid & footprint & footprint & centroid \\")
        print(r"\midrule")
        for r in rows:
            print(f"{r['label']} & {r['s_cent']:.3f} & {r['s_foot']:.3f} & "
                  f"{r['ab_foot']:.3f} & {r['ab_cent']:.3f} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"run": args.run, "agents": out}, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
