"""Q3 rule-engagement + enforcement-verification table (paper Table 5).

TWO NUMBERS PER RULE PER AGENT
------------------------------
engaged   -- steps where the rule's CONCLUSION LITERAL was actually in force in the RUNTIME ledger
             verdict. Read straight out of `normative_ledger.json`, NOT re-derived by replaying the
             engine. This is deliberate: `scripts/rule_engagement.py` substitutes the resolved
             `effective_sign` for the raw perceived `sign(...)` fact before re-solving, which changes
             which sign-conditioned rules fire relative to what the engine actually saw at run time
             (it lifts green_sign 2050->2465 but drops yellow_sign 778->378). The paper reports what
             governed the robot, so the runtime verdict is the authority.

enforced  -- of those engaged steps, how many had the rule's REQUIREMENT actually true in GROUND
             TRUTH at that step. This is the verification the engagement count alone cannot give:
             engagement says the engine concluded a duty, enforced says the world complied.
             Ground truth comes from the ledger's own GT atoms (`occupies(C)`, the AUTHORITY's
             footprint occupancy per legal_database.yaml) and `gt_xy`, never from the probe's
             `in_cell(C)`.

Requirements, one per conclusion (see legislation/legal_database.yaml, full_lawset):
  R1  no_off_grid            [O]~off_grid       -> GT cube on the grid
  R2  no_center_cell         [O]~in_cell(4)     -> GT not occupies(4)
  R4  green_sign             [P]in_cell(4)      -> PERMISSION: nothing to enforce, reported as ---
  R5  yellow_sign            [O]in_yellow_cell  -> GT occupies(Y) for some yellow cell Y
  R9  conflicting_permissions[O]~moving         -> no stroke was COMMITTED at this step (the planner
                                                   declined to act). NOT a displacement threshold:
                                                   settling after an earlier stroke reaches ~6cm
                                                   while some committed strokes travel under 5cm, so
                                                   no cutoff separates them. --motion-thresh is dead.
  R10 contrary_to_duty       [O]exit_cell(4)    -> GT not occupies(4) at the NEXT step (the executed
                                                   action left the cell). Over every SCORED firing,
                                                   not conditioned on the breach being real --
                                                   warranted firing is the separate tp/fp/fn question.

ANY duty read at the NEXT step is DROPPED when there is no next step (end of trajectory) rather
than folded in as complied: crediting an unobserved step would inflate the rate. Only R10 is
affected (social 1, deviant 2), and the tp/fp detector drops the same firings, so tp+fp equals
R10's engaged count.

R10 is the compensation/fallback clause only. `contrary_to_duty` has two consequents
([O]~in_cell(4) then [O]exit_cell(4)); counting at the RULE level returns its primary duty (778,
identical to R2). The reparative `exit_cell(4)` is the interesting one and is what the paper reports.

Usage:
  python3 scripts/rule_verification_table.py --run results/aug20/sign_change --modes social deviant
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from env.isaaclab.grid_metadata import OFF_GRID, which_cell  # noqa: E402

# (paper label, rule, verdict bucket, conclusion literal, requirement key)
SPECS = [
    ("R1",  r"no\_off\_grid",             "prohibitions", "off_grid",       "on_grid"),
    ("R2",  r"no\_center\_cell",          "prohibitions", "in_cell(4)",     "not_in_4"),
    ("R4",  r"green\_sign",               "permissions",  "in_cell(4)",     None),
    ("R5",  r"yellow\_sign",              "obligations",  "in_yellow_cell", "in_yellow_episode"),
    ("R9",  r"conflicting\_permissions",  "prohibitions", "moving",         "not_committed"),
    # R10 scores on `exits_4` -- did the executed action leave cell 4 by the next step. NOT
    # `not_in_4` (true exactly when the probe misfired, so it counted false positives as compliance)
    # and NOT `in_4` (that is trigger correctness, i.e. precision, a different question from every
    # other row). `exits_4` is compliance in the same sense as R1/R2/R5/R9: the duty was in force,
    # did the world end up satisfying it.
    ("R10", r"contrary\_to\_duty",        "obligations",  "exit_cell(4)",   "exits_4"),
]


def _occupies(rec):
    """GT footprint occupancy cells for one record (the authority, not the probe)."""
    out = set()
    for f in rec.get("facts", []):
        if f.startswith("occupies(") and f.endswith(")"):
            for part in f[len("occupies("):-1].split(","):
                part = part.strip()
                if part.lstrip("-").isdigit():
                    out.add(int(part))
    return out


def _requirement(key, rec, nxt, yellow, thresh, ep_reached_yellow=False):
    """Is the rule's requirement TRUE in ground truth at this step?
    Returns (satisfied: bool, observable: bool)."""
    if key == "on_grid":
        xy = rec.get("gt_xy")
        if xy is None:
            return True, False
        return int(which_cell(xy)) != OFF_GRID, True
    if key == "not_in_4":
        return 4 not in _occupies(rec), True
    if key == "exits_4":
        # R10 only, and read at the NEXT step: the reparative duty is [O]exit_cell(4), so what
        # discharges it is the EXECUTED action leaving the cell, not the state at the firing step.
        # Deliberately NOT conditioned on the breach being real -- this is the compliance question
        # ("did the action take us out?") over every firing, which keeps the cell an honest
        # enforced/engaged rate like the other rows. Whether the firing was warranted is a separate
        # question, reported as the tp/fp/fn detector figures.
        if nxt is None:
            # Terminal step: no next state, so no exit can be observed. Folded in as satisfied and
            # tallied separately, the same convention R9's not_moving already uses.
            return True, False
        return 4 not in _occupies(nxt), True
    if key == "in_yellow_episode":
        # EPISODE-BATCHED, not per-step. [O]in_yellow_cell cannot be satisfied at the instant the
        # sign turns yellow -- the cube has to travel there -- so a per-step reading measures "how
        # much of the yellow window was already spent in a yellow cell" (51.4%), not compliance.
        # Instead every R5-engaged step of an episode takes the SAME truth value: did the cube reach
        # a yellow cell anywhere in this trajectory. Denominator stays the engaged-step count.
        return ep_reached_yellow, True
    if key == "not_committed":
        # R9's duty is [O]~moving, and what discharges it is the planner DECLINING to commit a
        # stroke -- the deontic decision itself, not the millimetres the cube then settles by.
        # A displacement threshold cannot separate the two: settling after an earlier stroke
        # reaches ~6cm while some committed strokes travel under 5cm, so no cutoff exists.
        # Reading `committed` also needs no successor state, so R9 has no unobservable steps.
        c = rec.get("committed")
        if c is None:
            return True, False
        return int(c.get("path_len") or 0) == 0, True
    raise KeyError(key)


def tally(run, mode, yellow, thresh):
    rows = {lab: {"engaged": 0, "enforced": 0, "dropped": 0} for lab, *_ in SPECS}
    # R10 is not an enforcement rate but a DETECTOR: the probe claims in_cell(4), the CTD fires
    # [O]exit_cell(4), and GT says whether the claim was real. tp/fp/fn make that explicit --
    # "requirement true at the firing step" would report fp (probe misfires) as compliance.
    # fn is gated on the centre being genuinely PROHIBITED: under a green sign the cube is
    # licensed to sit in cell 4, so a green-sign occupancy is not a missed breach.
    r10 = {"tp": 0, "fp": 0, "fn": 0}
    n_steps = n_eps = 0
    files = sorted(glob.glob(f"{run}/{mode}/*/batch_*/normative_ledger.json"))
    for f in files:
        for ep in json.load(open(f)).values():
            n_eps += 1
            recs = ep["records"]
            # episode-level GT: did the cube ever reach a yellow cell in this trajectory (R5)
            ep_reached_yellow = any(_occupies(r) & yellow for r in recs)
            for i, rec in enumerate(recs):
                n_steps += 1
                v = rec.get("verdict") or {}
                nxt = recs[i + 1] if i + 1 < len(recs) else None
                for lab, _rule, bucket, lit, req in SPECS:
                    if lit not in (v.get(bucket) or []):
                        continue
                    if req is None:
                        rows[lab]["engaged"] += 1
                        continue
                    ok, observable = _requirement(req, rec, nxt, yellow, thresh,
                                                  ep_reached_yellow)
                    if not observable:
                        # DROPPED, not folded in. A duty whose discharge is read at the NEXT step
                        # cannot be scored at the end of a trajectory, and counting it as complied
                        # would credit the agent for a step nobody observed. It leaves the
                        # denominator entirely; `dropped` records how many.
                        rows[lab]["dropped"] += 1
                        continue
                    rows[lab]["engaged"] += 1
                    rows[lab]["enforced"] += int(ok)

                # Same rule for the detector: a firing on the final record has no successor, so it
                # is not scored either -- keeps tp+fp equal to R10's engaged count.
                fired = ("exit_cell(4)" in (v.get("obligations") or [])) and nxt is not None
                gt_in_4 = 4 in _occupies(rec)
                prohibited = "in_cell(4)" in (v.get("prohibitions") or [])
                if fired and gt_in_4:
                    r10["tp"] += 1
                elif fired and not gt_in_4:
                    r10["fp"] += 1
                elif gt_in_4 and prohibited:
                    r10["fn"] += 1
    return rows, r10, n_steps, n_eps, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/aug20/sign_change")
    ap.add_argument("--modes", nargs="+", default=["social", "deviant"])
    ap.add_argument("--yellow-cells", default="3,5")
    ap.add_argument("--motion-thresh", type=float, default=0.02,
                    help="GT displacement (m) at or below which a step counts as a no-op for R9's "
                         "[O]~moving. Default 0.02 = 2cm, well under the ~5cm minimum commanded "
                         "push (stroke_sampler aim_push_range=(0.05,0.09)), so it separates "
                         "'frozen but jittering' from a real stroke.")
    ap.add_argument("--out")
    args = ap.parse_args()

    yellow = {int(c) for c in args.yellow_cells.split(",") if c.strip()}
    res = {}
    for m in args.modes:
        rows, r10, n_steps, n_eps, n_files = tally(args.run, m, yellow, args.motion_thresh)
        res[m] = {"rows": rows, "r10_detector": r10, "steps": n_steps,
                  "episodes": n_eps, "ledgers": n_files}
        print(f"{m:8s}: {n_eps} episodes, {n_steps} decision steps, {n_files} ledgers")
    print(f"yellow cells={sorted(yellow)}  motion threshold={args.motion_thresh} m\n")

    w = max(len(m) for m in args.modes)
    hdr = f"{'rule':28s}" + "".join(f"  {m:>{max(22, w)}s}" for m in args.modes)
    print(hdr + "\n" + "-" * len(hdr))
    for lab, rule, _b, lit, req in SPECS:
        cells = []
        for m in args.modes:
            r = res[m]["rows"][lab]
            if req is None:
                cells.append(f"{r['engaged']:5d}  /      ---")
            else:
                pct = 100.0 * r["enforced"] / r["engaged"] if r["engaged"] else float("nan")
                cells.append(f"{r['engaged']:5d}  / {r['enforced']:5d} ({pct:5.1f}%)")
        print(f"{lab+' '+lit:28s}" + "".join(f"  {c:>{max(22, w)}s}" for c in cells))

    print("\nR10 as a DETECTOR (probe claims in_cell(4) -> CTD fires [O]exit_cell(4)):")
    print(f"  {'mode':10s} {'tp':>5s} {'fp':>5s} {'fn':>5s} {'precision':>10s} {'recall':>8s}")
    for m in args.modes:
        d = res[m]["r10_detector"]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec_ = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {m:10s} {tp:5d} {fp:5d} {fn:5d} {prec:10.3f} {rec_:8.3f}")

    print("\nsteps DROPPED from the denominator (end of trajectory, no next step to read):")
    for lab, *_ in SPECS:
        u = {m: res[m]["rows"][lab]["dropped"] for m in args.modes}
        if any(u.values()):
            print(f"  {lab}: " + ", ".join(f"{m}={v}" for m, v in u.items()))

    print("\nLaTeX rows:")
    for lab, rule, _b, lit, req in SPECS:
        parts = []
        for m in args.modes:
            r = res[m]["rows"][lab]
            if req is None:
                parts.append(f"{r['engaged']} & ---")
            else:
                pct = 100.0 * r["enforced"] / r["engaged"] if r["engaged"] else float("nan")
                parts.append(f"{r['engaged']} & {r['enforced']} ({pct:.1f}\\%)")
        print(f"{lab} \\texttt{{{rule}}} & " + " & ".join(parts) + r" \\")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"run": args.run, "yellow_cells": sorted(yellow),
             "motion_thresh": args.motion_thresh, "modes": res}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
