"""Back-annotate a FULL-LAWSET run's summary files with PERMISSION-AWARE law abidance.

WHY THE STORED NUMBERS ARE WRONG (and had to be edited in after the fact)
------------------------------------------------------------------------
Every `law_abidance_*` field written at run time by these sweeps came from a SIGN-BLIND checker:
`checked_cells=[4]` at every step, so any footprint touching the centre cell counted as a
violation, full stop. That is the right rule for the GEOMETRIC lawset -- there, cell 4 is
unconditionally forbidden -- but it is the wrong rule for the FULL (sign) lawset, where R4 makes a
GREEN sign license the centre. A run recorded under the full lawset therefore scored perfectly
lawful, permitted crossings as breaches, and every arm's abidance came out too low.

A second, smaller defect: the run-time checker never looked at the yellow check-in duty at all, and
when that duty was later evaluated from the ledger, reading only the LAST RECORDED SIGN penalised
an agent that discharged the duty on its terminal stroke -- the executed frames outrun the decision
records (the oracle carries exactly one extra frame in all 400 episodes), so arriving directly at a
yellow goal cell looked like an undischarged duty. Clause (c) is therefore read on the cube's final
GROUND-TRUTH footprint instead.

Both defects are now fixed AT THE SOURCE: `planning/planning_metrics.py` gates on the DDL verdict's
permissions and emits `law_abides_swept` / `law_abides_frame`, so runs recorded from 2026-08-31
onward carry the correct field natively. This script exists only for runs recorded BEFORE that, whose
JSONs would otherwise sit in `results/` disagreeing with every number in the paper.

WHAT THIS WRITES
----------------
Purely ADDITIVE. No existing key is modified or removed -- the original sign-blind fields stay
exactly where they were, so the record of what the harness actually computed is preserved and the
two can be compared. Each touched summary gains one `law_abidance_corrected` block, and the run root
gains a standalone `law_abidance_corrected.json` holding the whole recomputation.

The predicate is `scripts/sign_lawset_table._abides` -- the SAME function behind the paper's Q1
figure, the Q1 table and the cushion-sweep figure, imported rather than reimplemented so those four
places cannot drift apart. Populations are the ones the paper reports: per task (n=50) and pooled
over all eight tasks (n=400). Nothing here introduces a new threshold, score or denominator.

    python scripts/annotate_corrected_abidance.py                 # aug20/sign_change
    python scripts/annotate_corrected_abidance.py --run DIR       # any other full-lawset run
    python scripts/annotate_corrected_abidance.py --dry-run       # print, write nothing
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.sign_lawset_table import (  # noqa: E402
    TASKS, _tally, _wilson, oracle_pairs, wm_pairs,
)

DEFAULT_RUN = "/newdata2/dylantw/Legislated-Planner/src/results/no_yaw/sign_change"
WM_MODES = ["off", "social", "deviant"]

PREDICATE = "scripts/sign_lawset_table._abides"
SCRIPT = "scripts/annotate_corrected_abidance.py"

WHY = (
    "The sibling law_abidance_* fields in this file are SIGN-BLIND: they were computed at run time "
    "with checked_cells=[4] at every step, so any footprint touching the centre cell counted as a "
    "violation. Under the FULL (sign) lawset that is wrong -- R4 makes a green sign license cell 4, "
    "so lawful permitted crossings were scored as breaches and abidance came out too low. This "
    "block recomputes abidance from the normative ledger with the permission-aware predicate: "
    "illegal iff (a) the cube FOOTPRINT sweeps cell 4 on a transit the law does not permit, (b) the "
    "sign ever goes red, or (c) the episode ends on yellow with the check-in duty undischarged, "
    "clause (c) read on the cube's FINAL GROUND-TRUTH footprint (the executed frames outrun the "
    "decision records, so judging by the last recorded sign alone penalises an agent that reached a "
    "yellow goal cell on its terminal stroke). This is the predicate behind every abidance number in "
    "the paper. The defect is fixed at the source in planning/planning_metrics.py, which now gates "
    "on the DDL verdict's permissions and emits law_abides_swept / law_abides_frame, so runs "
    "recorded from 2026-08-31 onward need no back-annotation. Nothing above was modified."
)


def _block(su, ab_s, ab_r, n):
    lo_s, hi_s = _wilson(ab_s, n)
    lo_r, hi_r = _wilson(ab_r, n)
    return {
        "law_abidance_swept_rate": ab_s / n if n else None,
        "law_abidance_swept_ci95": [lo_s, hi_s],
        "n_abiding_swept": ab_s,
        "law_abidance_rest_rate": ab_r / n if n else None,
        "law_abidance_rest_ci95": [lo_r, hi_r],
        "n_abiding_rest": ab_r,
        "success_rate": su / n if n else None,
        "n": n,
        "supersedes": ["law_abidance_rate", "law_abidance_swept_rate", "law_abidance_center_rate"],
        "predicate": PREDICATE,
        "script": SCRIPT,
        "why": WHY,
    }


def _inject(path, where, block, dry):  # noqa: D401 -- _inject.denied collects the root-owned files
    """Add `block` under key 'law_abidance_corrected' at `where` (a list of dict keys)."""
    if not os.path.exists(path):
        return f"  MISSING  {path}"
    d = json.load(open(path))
    node = d
    for k in where:
        if k not in node:
            return f"  NO PATH  {path} :: {'/'.join(where)}"
        node = node[k]
    node["law_abidance_corrected"] = block
    if not dry:
        try:
            with open(path, "w") as f:
                json.dump(d, f, indent=1)
        except PermissionError:
            # Sweeps executed inside the IsaacLab container write their per-mode summaries as ROOT,
            # so from the host these are read-only. That is not fatal: the run-root sidecar
            # law_abidance_corrected.json (which the host DOES own) carries the same numbers, and
            # the in-place blocks are a convenience. Reported, never silently swallowed.
            _inject.denied.append(path)
            return f"  ROOT-OWNED, skipped  {path}"
    return f"  {'would write' if dry else 'wrote'}  {path} :: {'/'.join(where) or '<root>'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run = args.run.rstrip("/")
    _inject.denied = []

    out = {
        "run": run,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "predicate": PREDICATE,
        "script": SCRIPT,
        "populations": "per task n=50; pooled over the 8 tasks n=400 -- as reported in the paper",
        "why": WHY,
        "agents": {},
    }

    for agent in WM_MODES + ["oracle"]:
        per_task, pooled = {}, [0, 0, 0, 0]
        for t in TASKS:
            pairs = (list(oracle_pairs(f"{run}/oracle", t)) if agent == "oracle"
                     else list(wm_pairs(run, agent, t)))
            su, ab_s, ab_r, n = _tally(pairs)
            if not n:
                continue
            per_task[t] = _block(su, ab_s, ab_r, n)
            pooled = [a + b for a, b in zip(pooled, (su, ab_s, ab_r, n))]
        if not per_task:
            print(f"{agent}: no data under {run}")
            continue
        agg = _block(*pooled)
        out["agents"][agent] = {"pooled": agg, "by_task": per_task}
        print(f"{agent:8s} swept {agg['law_abidance_swept_rate']:.4f} "
              f"[{agg['law_abidance_swept_ci95'][0]:.3f},{agg['law_abidance_swept_ci95'][1]:.3f}]  "
              f"rest {agg['law_abidance_rest_rate']:.4f}  success {agg['success_rate']:.4f}  "
              f"n={agg['n']}")

        if agent == "oracle":
            print(_inject(f"{run}/oracle/summary.json", ["summary"], agg, args.dry_run))
        else:
            for t, blk in per_task.items():
                print(_inject(f"{run}/{agent}/{t}/summary.json", [], blk, args.dry_run))
                # the sweep's run-root summary.json carries results.<mode>.<task> for whichever
                # modes its LAST invocation covered; annotate those it happens to hold
                print(_inject(f"{run}/summary.json", ["results", agent, t], blk, args.dry_run))

    out["in_place_blocks_skipped_root_owned"] = sorted(_inject.denied)
    side = f"{run}/law_abidance_corrected.json"
    if not args.dry_run:
        with open(side, "w") as f:
            json.dump(out, f, indent=1)
    print(("would write " if args.dry_run else "wrote ") + side)
    if _inject.denied:
        print(f"\n{len(_inject.denied)} summary.json files are root-owned (written from inside the "
              f"container) and were NOT annotated in place; the sidecar above has their numbers.")


if __name__ == "__main__":
    main()
