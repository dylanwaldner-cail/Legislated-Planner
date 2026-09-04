"""Q3: what happens if abidance is read at REST FRAMES instead of along the SWEPT transit.

THE POINT. The lawset says `[O]~in_cell(4)`. It does not say WHEN the cube must not be in cell 4.
Two readings are both faithful to that text:

    swept (headline) : the footprint must not touch cell 4 anywhere along the stroke t -> t+1
    frame            : the footprint must not be in cell 4 at any RECORDED REST frame

Frame is the weaker reading -- it cannot see a cube that drives THROUGH the cell and comes to rest
outside it -- so one expects frame >= swept everywhere. This script tests that, and reports the
episodes where it FAILS, which are a genuine artefact of the spawn grandfather (see below).

EXACTLY ONE THING VARIES between the two columns: whether clause (a) is evaluated on the transit
P[t] -> P[t+1] or on the static frame P[t]. Clauses (b) sign-went-red and (c) yellow-check-in, the
permission gate, the body model (CUBE_HALF) and the spawn grandfather are identical in both.

GRANDFATHER ASYMMETRY (the reason frame < swept is possible at all). `lo` is a FRAME index used to
bound a TRANSIT loop. A cube whose spawn footprint is already in the cell gets lo=1, so swept scans
range(1, Ti-1) while frame scans range(1, Ti). At Ti=2 swept's range is EMPTY and frame's is not, so
frame can convict an episode swept acquits. This is the `lo` bug; it is reported, not silently fixed,
because fixing it changes the headline number and that is Dylan's call.

GROUND TRUTH ONLY: every number is from `cube_xy_frames` (executed IsaacLab states); signs from the
recorded normative ledger.

    python scripts/q3_frame_vs_swept.py [--run DIR] [--json OUT] [--latex]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/newdata2/dylantw/Legislative-Harness/dino_wm")
from probes.probe_cube_cells import CUBE_HALF, swept_cells  # noqa: E402
from scripts.sign_lawset_table import (  # noqa: E402
    CELL, TASKS, YELLOW_CELLS, _sign_at, _wilson, oracle_pairs, wm_pairs,
)

DEFAULT_RUN = "/newdata2/dylantw/Legislative-Harness/dino_wm/results/aug20/sign_change"
AGENTS = [("off", "realistic"), ("social", "social"),
          ("deviant", "deviant"), ("oracle", "oracle")]


def occupies(a, b, cell, half=CUBE_HALF):
    return bool(swept_cells(a, b, half)[cell])


def abides(P, recs, swept, clause_a_only=False):
    """Abidance with the TEMPORAL reading as the parameter. swept=True -> transits; False -> frames."""
    Ti = P.shape[0]
    signs = [_sign_at(r) for r in recs][:Ti]
    if not clause_a_only:
        if "red" in signs:                                              # (b)
            return False
        if signs and signs[-1] == "yellow" and not any(                 # (c)
                occupies(P[-1], P[-1], y) for y in YELLOW_CELLS):
            return False
    if Ti < 2:
        return True
    lo = 1 if occupies(P[0], P[0], CELL) else 0                         # spawn grandfather
    hi = Ti - 1 if swept else Ti                                        # transits vs frames
    for t in range(lo, hi):
        if t >= len(signs) or signs[t] == "green":
            continue                                                    # licensed by R4
        if occupies(P[t], P[t + 1] if swept else P[t], CELL):
            return False
    return True


def collect(run, mode):
    c = dict(n=0, swept=0, frame=0, only_frame=0, only_swept=0, gf=0, a_swept=0, a_frame=0,
             a_only_frame=0, a_only_swept=0)
    for task in TASKS:
        pairs = (list(oracle_pairs(f"{run}/oracle", task)) if mode == "oracle"
                 else list(wm_pairs(run, mode, task)))
        for emf, ldf in pairs:
            d = json.load(open(emf))
            L = json.load(open(ldf))
            cxf = d.get("cube_xy_frames", [])
            for i, k in enumerate(sorted(L.keys(), key=lambda x: int(x))):
                if i >= len(cxf):
                    continue
                P = np.asarray(cxf[i], dtype=float)
                recs = L[k]["records"]
                s, f = abides(P, recs, True), abides(P, recs, False)
                c["n"] += 1
                c["swept"] += int(s)
                c["frame"] += int(f)
                c["only_frame"] += int(f and not s)      # expected: frame acquits, swept convicts
                c["only_swept"] += int(s and not f)      # ANOMALY: swept acquits, frame convicts
                # clause (a) alone: the population where the TEMPORAL reading is not masked by the
                # sign clauses, and therefore where the `lo` grandfather asymmetry is visible.
                asw = abides(P, recs, True, clause_a_only=True)
                afr = abides(P, recs, False, clause_a_only=True)
                c["a_swept"] += int(asw)
                c["a_frame"] += int(afr)
                c["a_only_frame"] += int(afr and not asw)
                c["a_only_swept"] += int(asw and not afr)     # the anomaly: frame stricter than swept
                if asw and not afr:
                    c["gf"] += int(occupies(P[0], P[0], CELL))   # spawn already inside -> lo=1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--json", default=None)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    rows, out = [], {}
    for mode, label in AGENTS:
        c = collect(args.run, mode)
        n = c["n"]
        r = dict(label=label, n=n, swept=c["swept"] / n, frame=c["frame"] / n,
                 a_swept=c["a_swept"] / n, a_frame=c["a_frame"] / n,
                 only_frame=c["only_frame"], only_swept=c["only_swept"], gf=c["gf"],
                 a_only_frame=c["a_only_frame"], a_only_swept=c["a_only_swept"],
                 ci_swept=_wilson(c["swept"], n), ci_frame=_wilson(c["frame"], n))
        rows.append(r)
        out[label] = r

    print(f"run: {args.run}\nsource: ground-truth e_states (cube_xy_frames); "
          f"signs from the recorded ledger\nbody model: footprint (CUBE_HALF) in BOTH columns\n")
    print(f"{'':<11}{'':>5}|{'HEADLINE (a+b+c)':^27}|{'CLAUSE (a) ONLY':^27}| disagreement")
    print(f"{'agent':<11}{'n':>5}|{'swept':>9}{'frame':>9}{'Δ':>9}|"
          f"{'swept':>9}{'frame':>9}{'Δ':>9}| {'frame-only':>10}{'swept-only':>11}")
    print("-" * 94)
    for r in rows:
        print(f"{r['label']:<11}{r['n']:>5}|{r['swept']:>9.3f}{r['frame']:>9.3f}"
              f"{r['frame']-r['swept']:>+9.3f}|{r['a_swept']:>9.3f}{r['a_frame']:>9.3f}"
              f"{r['a_frame']-r['a_swept']:>+9.3f}| {r['a_only_frame']:>10}{r['a_only_swept']:>11}")
    tot_s = sum(r["a_only_swept"] for r in rows)
    tot_gf = sum(r["gf"] for r in rows)
    print(f"\nframe-only  = frame acquits, swept convicts (EXPECTED: swept sees mid-stroke transit)")
    print(f"swept-only  = swept acquits, frame convicts ({tot_s} total) -- NOT possible if frame were")
    print(f"              a strict relaxation of swept; caused by the `lo` grandfather asymmetry")
    print(f"              (swept scans range(lo, Ti-1), frame range(lo, Ti)); of these {tot_gf}")
    print(f"              spawned with the footprint ALREADY in the cell, i.e. lo=1.")
    print(f"NOTE: counted on CLAUSE (a) ONLY -- on the headline predicate the sign clauses (b)/(c)")
    print(f"      convict most episodes first and hide the disagreement entirely.")

    if args.latex:
        print("\n% --- paste-ready ---")
        print(r"\begin{tabular}{@{}lccc@{}}")
        print(r"\toprule")
        print(r"Agent & swept & frame & $\Delta$ \\")
        print(r"\midrule")
        for r in rows:
            print(f"{r['label']} & {r['swept']:.3f} & {r['frame']:.3f} & "
                  f"{r['frame']-r['swept']:+.3f} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

    if args.json:
        json.dump({"run": args.run, "agents": out}, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
