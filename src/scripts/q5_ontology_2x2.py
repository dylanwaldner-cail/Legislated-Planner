"""Q5: the FULL 2x2 of ontological readings of R2, on one predicate with two parameters.

WHY THIS EXISTS. The paper's Table 3 reports four abidance numbers, but the two existing scripts
each vary only ONE axis and so cover only three of the four corners:

    scripts/q3_definition_inversion.py   body model {footprint, centroid}, ALWAYS swept
    scripts/q3_frame_vs_swept.py         temporal  {swept, frame},       ALWAYS footprint

The centroid x at-rest corner is their intersection and is computed by neither. This script
unifies both into a single `abides(P, recs, half, swept)` so all four corners come from ONE
predicate, and only the two parameters change.

SELF-VALIDATION IS THE POINT. The script recomputes the THREE corners the published scripts do
produce and asserts they match. If the shared predicate were wrong, the known corners would drift
and the run ABORTS -- so the fourth number is only ever reported alongside proof that the machinery
reproduces the paper.

CLAUSE (a) ONLY. Table 3 reports the bare geometric clause, WITHOUT the sign clauses (b) red-taint
and (c) yellow check-in duty. Verified: clause-(a) footprint/swept = 0.655 social matches the table,
whereas the headline (a+b+c) predicate gives 0.647. The sign clauses convict most episodes first and
hide the disagreement entirely, so the 2x2 must be computed on clause (a) -- that is a CHOICE, made
to match the published table, and it is stated here rather than buried.

GROUND TRUTH ONLY: `cube_xy_frames` (executed IsaacLab states); signs from the recorded ledger.

    python scripts/q5_ontology_2x2.py [--run DIR] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from probes.probe_cube_cells import CUBE_HALF, swept_cells  # noqa: E402
from scripts.sign_lawset_table import (  # noqa: E402
    CELL, TASKS, YELLOW_CELLS, _sign_at, oracle_pairs, wm_pairs,
)

DEFAULT_RUN = "/newdata2/dylantw/Legislated-Planner/src/results/no_yaw/sign_change"
AGENTS = [("off", "realistic"), ("social", "social"),
          ("deviant", "deviant"), ("oracle", "oracle")]
FOOT, CENT = CUBE_HALF, 0.0

# ALL TWELVE values published in Table 3 of the paper (clause (a) only). Three corners are also
# produced by the existing scripts; the centroid x at-rest corner is published but computed by NO
# script -- reproducing it here is the point of this file.
#   (agent, half, swept) -> published rate
KNOWN = {
    ("realistic", FOOT, True):  0.113, ("realistic", CENT, True):  0.138,
    ("realistic", FOOT, False): 0.130, ("realistic", CENT, False): 0.352,
    ("social",    FOOT, True):  0.655, ("social",    CENT, True):  0.948,
    ("social",    FOOT, False): 0.833, ("social",    CENT, False): 0.978,
    ("deviant",   FOOT, True):  0.448, ("deviant",   CENT, True):  0.557,
    ("deviant",   FOOT, False): 0.497, ("deviant",   CENT, False): 0.680,
}
# Published values are 3 d.p.; n=400 makes one episode 0.0025, so an exact half (e.g. 0.1125 vs a
# printed 0.113) is a rounding artefact, not a disagreement. Tolerance = one unit in the last
# published place.
TOL = 1e-3


def occupies(a, b, cell, half):
    """Body of half-width `half` sweeping a->b touches `cell`? a==b gives the static frame test."""
    return bool(swept_cells(a, b, half)[cell])


def abides(P, recs, half, swept, clause_a_only=True):
    """The paper's abidance predicate with BOTH readings as parameters.

    half  : CUBE_HALF -> footprint, 0.0 -> centroid   (the body model)
    swept : True -> transits t->t+1, False -> rest frames only   (the temporal reading)

    Every other clause is byte-identical to the two published scripts, so nothing but these two
    parameters can explain a difference between corners.
    """
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
    hi = Ti - 1 if swept else Ti                                        # transits vs frames
    for t in range(lo, hi):
        if t >= len(signs) or signs[t] == "green":
            continue                                                    # licensed by R4
        if occupies(P[t], P[t + 1] if swept else P[t], CELL, half):
            return False
    return True


CORNERS = [(FOOT, True), (CENT, True), (FOOT, False), (CENT, False)]


def collect(run, mode):
    n = 0
    hits = {c: 0 for c in CORNERS}
    for task in TASKS:
        pairs = (list(oracle_pairs(f"{run}/oracle", task)) if mode == "oracle"
                 else list(wm_pairs(run, mode, task)))
        for emf, ldf in pairs:
            d = json.load(open(emf))
            led = json.load(open(ldf))
            for i in range(int(d["n_evals"])):
                P = np.asarray(d["cube_xy_frames"][i], float)
                if P.ndim != 2 or P.shape[0] == 0:
                    continue
                recs = led.get(str(i), {}).get("records", [])
                n += 1
                for half, swept in CORNERS:
                    if abides(P, recs, half, swept):
                        hits[(half, swept)] += 1
    return n, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print(f"run: {args.run}")
    print("predicate: clause (a) only -- bare geometry, no sign clauses (matches Table 3)\n")
    print(f"{'agent':<11}{'n':>5} | {'full path':^19} | {'at rest':^19}")
    print(f"{'':<11}{'':>5} | {'centroid':>9}{'footprint':>10} | {'centroid':>9}{'footprint':>10}")
    print("-" * 70)

    out, failures = {}, []
    for mode, label in AGENTS:
        n, hits = collect(args.run, mode)
        if not n:
            continue
        r = {f"{'cent' if h == CENT else 'foot'}_{'swept' if s else 'frame'}": hits[(h, s)] / n
             for h, s in CORNERS}
        out[label] = dict(n=n, **r)
        print(f"{label:<11}{n:>5} | {r['cent_swept']:>9.3f}{r['foot_swept']:>10.3f} | "
              f"{r['cent_frame']:>9.3f}{r['foot_frame']:>10.3f}")
        for (half, swept), want in ((k[1:], v) for k, v in KNOWN.items() if k[0] == label):
            got = hits[(half, swept)] / n
            if abs(got - want) > TOL:
                failures.append(f"{label} half={half} swept={swept}: got {got:.4f} want {want:.3f}")

    print("\nvalidation against the three published corners:")
    if failures:
        for f in failures:
            print("  MISMATCH", f)
        sys.exit("\nABORT: the shared predicate does not reproduce the published table; "
                 "the fourth corner is therefore untrustworthy and is NOT reported.")
    print(f"  all {len(KNOWN)} published values reproduce within {TOL} -> predicate confirmed")
    print("\n  centroid x at-rest -- published in Table 3 but computed by NO existing script:")
    for label in out:
        print(f"    {label:<11} {out[label]['cent_frame']:.3f}")

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
