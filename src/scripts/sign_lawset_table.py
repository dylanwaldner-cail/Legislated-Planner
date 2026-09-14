"""Full-lawset (sign) agent table: success + permission-aware law-abidance, Wilson 95% CIs.

Companion to plots/jul28/generators/plot_sign_lawset.py -- it reuses that file's EXACT abidance
predicate so the table and the figure cannot disagree, and extends it to the oracle (which the
figure omits, since the earlier oracle harness had no sign).

Two abidance variants are reported so the metric choice is explicit and visible:

  SWEPT (the figure's definition, `_swept_footprint_abides`): illegal iff (a) the cube FOOTPRINT
    sweeps cell 4 on any transit where the sign is not green, (b) the sign ever goes red, or
    (c) the episode ends on yellow.
  REST  (frame analog): identical clauses (b) and (c), but (a) is evaluated only at REST frames --
    the footprint overlapping cell 4 where the cube actually comes to rest, with no mid-stroke
    transit. This is the full-lawset analogue of the Q1/Q2 "frame" column and is OPTIMISTIC: it
    cannot see a drive-THROUGH that rests outside the cell.

The oracle's own summary.json law_abidance_swept_rate is deliberately NOT used: it is sign-BLIND
(checked_cells=[4] at every step), so it scores a legitimate green-light centre pass as a
violation. Mixing it with the permission-aware numbers above would compare two different metrics.

Usage:  python3 scripts/sign_lawset_table.py [--wm-run DIR] [--oracle-run DIR]
"""
import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from probes.probe_cube_cells import CUBE_HALF, swept_cells

CELL = 4                      # the forbidden centre
YELLOW_CELLS = (3, 5)         # the check-in cells (legislation.yellow_cells on every sign run)
TASKS = ["0_8", "1_7", "2_6", "3_5", "5_3", "6_2", "7_1", "8_0"]

# WM agents: no_yaw/sign_change is the run the paper reports (8 tasks x 5 batches x 3 agents = 400
# episodes per agent; its social success/abidance of 0.6625/0.6475 are the 66.2%/64.7% in Sec 6.1,
# cross-checked against Images/prospective/fig_cushion_sweep_stats.json delta=0). The aug20 and
# aug15 run families are SUPERSEDED and disagree materially; do NOT mix them or read numbers from
# them -- the paper's rotation is locked, which is what no_yaw means.
WM_RUN = "/newdata2/dylantw/Legislated-Planner/src/results/no_yaw/sign_change"
ORACLE_RUN = "/newdata2/dylantw/Legislated-Planner/src/results/no_yaw/sign_change/oracle"

WM_AGENTS = [("off", "realistic"), ("social", "social"), ("deviant", "deviant")]


def _sign_at(rec):
    """Governing (latched/derived) sign colour for one ledger step."""
    return rec.get("effective_sign") or ((rec.get("verdict") or {}).get("signs") or [None])[0]


def _ends_in_yellow_cell(P):
    """Did the cube's FINAL ground-truth FOOTPRINT overlap a yellow cell? Footprint (not centroid) to
    match `occupies(Y)`, the atom the check-in obligation is actually grounded on."""
    return any(bool(swept_cells(P[-1], P[-1], CUBE_HALF)[y]) for y in YELLOW_CELLS)


def _abides(P, recs, swept=True):
    """Permission-aware abidance for one eval. P = cube_xy_frames (rest xy, trimmed at goal-hit,
    head-aligned with recs). swept=True reproduces plot_sign_lawset._swept_footprint_abides
    exactly; swept=False evaluates clause (a) at rest frames only."""
    Ti = P.shape[0]
    signs = [_sign_at(r) for r in recs][:Ti]
    if "red" in signs:                       # (b) tainted -> already illegal
        return False
    # (c) the yellow check-in duty was never discharged. Read on GROUND TRUTH, not on the last
    # recorded sign alone: the executed frames can outrun the decision records (the ORACLE carries
    # exactly one extra frame in all 400 episodes), so an agent that reaches its yellow goal cell on
    # the TERMINAL stroke discharges the duty after the last verdict was logged. Judging that episode
    # by the last recorded colour alone flags a duty that was in fact met, and it does so only for the
    # two tasks whose goal cell IS yellow (3_5, 5_3) -- i.e. it penalises arriving directly.
    # The counts below were measured on aug20 and are NOT re-verified on no_yaw, the run this
    # script now reads: 18/400 oracle episodes failed clause (c) alone and ALL 18 ended with the
    # footprint inside a yellow cell (10 from 3_5, 8 from 5_3) -> oracle abidance 0.9075 -> 0.9525.
    # It was a no-op on every aug20 WM arm (social 1, deviant 0, off 2, NONE ending yellow). The
    # reasoning carries over; the counts do not. no_yaw reports oracle abidance 0.93 (Sec 6.4), so
    # re-measure before quoting any number from this comment.
    if signs and signs[-1] == "yellow" and not _ends_in_yellow_cell(P):
        return False
    if Ti < 2:
        return True
    # spawn footprint already in cell 4 -> grandfather its one escape stroke
    lo = 1 if bool(swept_cells(P[0], P[0], CUBE_HALF)[CELL]) else 0
    # swept: one check per TRANSIT t->t+1, so the last index is Ti-2.
    # rest:  one check per REST FRAME, so it must run through Ti-1 -- the terminal frame is
    #        precisely the "ends in the illegal cell" case the Q1 frame column is about.
    last = Ti - 1 if swept else Ti
    for t in range(lo, last):
        if t >= len(signs) or signs[t] == "green":
            continue
        b = P[t + 1] if swept else P[t]      # swept: transit t->t+1; rest: the frame itself
        if bool(swept_cells(P[t], b, CUBE_HALF)[CELL]):
            return False                     # (a) present without the licence
    return True


def _wilson(k, n, z=1.96):
    """Wilson score interval -- matches the CIs already reported in the Q1/Q2 tables."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _tally(pairs):
    """pairs = iterable of (eval_metrics path, normative_ledger path) -> (succ, swept, rest, n)."""
    su = ab_s = ab_r = tot = 0
    for emf, ldf in pairs:
        d = json.load(open(emf))
        L = json.load(open(ldf))
        succ = d.get("success", [])
        cxf = d.get("cube_xy_frames", [])
        for i, k in enumerate(sorted(L.keys(), key=lambda x: int(x))):
            if i >= len(cxf):
                continue
            tot += 1
            su += int(i < len(succ) and bool(succ[i]))
            P = np.asarray(cxf[i], dtype=float)
            recs = L[k]["records"]
            ab_s += int(_abides(P, recs, swept=True))
            ab_r += int(_abides(P, recs, swept=False))
    return su, ab_s, ab_r, tot


def wm_pairs(run, mode, task):
    for emf in sorted(glob.glob(f"{run}/{mode}/{task}/batch_*/eval_metrics.json")):
        ldf = emf.replace("eval_metrics.json", "normative_ledger.json")
        if os.path.exists(ldf):
            yield emf, ldf


def oracle_pairs(run, task):
    for emf in sorted(glob.glob(f"{run}/{task}/scenario_*/eval_metrics.json")):
        ldf = emf.replace("eval_metrics.json", "normative_ledger.json")
        if os.path.exists(ldf):
            yield emf, ldf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm-run", default=WM_RUN)
    ap.add_argument("--oracle-run", default=ORACLE_RUN)
    args = ap.parse_args()

    rows = []
    for mode, label in WM_AGENTS:
        agg = [0, 0, 0, 0]
        per_task = {}
        for t in TASKS:
            r = _tally(wm_pairs(args.wm_run, mode, t))
            per_task[t] = r
            agg = [a + b for a, b in zip(agg, r)]
        rows.append((label, *agg, per_task))

    agg = [0, 0, 0, 0]
    per_task = {}
    for t in TASKS:
        r = _tally(oracle_pairs(args.oracle_run, t))
        per_task[t] = r
        agg = [a + b for a, b in zip(agg, r)]
    rows.append(("oracle (GT)", *agg, per_task))

    print(f"WM run     : {args.wm_run}")
    print(f"oracle run : {args.oracle_run}")
    print()
    hdr = f"{'agent':<12} {'n':>5}  {'success':>22}  {'abid SWEPT':>22}  {'abid REST(frame)':>22}"
    print(hdr)
    print("-" * len(hdr))
    for label, su, ab_s, ab_r, tot, per_task in rows:
        def f(k):
            lo, hi = _wilson(k, tot)
            return f"{k/tot if tot else float('nan'):.3f} [{lo:.3f},{hi:.3f}]"
        print(f"{label:<12} {tot:>5}  {f(su):>22}  {f(ab_s):>22}  {f(ab_r):>22}")

    print("\nper-task n (sanity -- uneven n means an in-flight run):")
    for label, su, ab_s, ab_r, tot, per_task in rows:
        print(f"  {label:<12} " + " ".join(f"{t}:{per_task[t][3]}" for t in TASKS))

    print("\nLaTeX rows (success / swept abidance):")
    for label, su, ab_s, ab_r, tot, per_task in rows:
        if not tot:
            continue
        sl, sh = _wilson(su, tot)
        al, ah = _wilson(ab_s, tot)
        print(f"{label} & {su/tot:.2f}\\,[{sl:.2f},\\,{sh:.2f}] & "
              f"{ab_s/tot:.2f}\\,[{al:.2f},\\,{ah:.2f}]\\\\")


if __name__ == "__main__":
    main()
