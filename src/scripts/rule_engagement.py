"""Per-rule engagement counts for Q4's Table 1, read from the ENGINE rather than inferred from atoms.

WHY
---
The ledger stores each step's verdict as sets of atoms (obligations / prohibitions / permissions),
not the rule that produced them. Counting "steps engaged" by pattern-matching atoms back to rules is
an inference and can be wrong: two rules may produce the same atom, a rule may fire and be defeated,
and DDL `permission` can hold as a WEAK permission (nothing forbids it) rather than because a
permissive rule fired.

This script removes the inference. It replays each recorded step's facts through the same DDL engine
via `LegislativeReasoner.solve_all()`, which returns the FULL answer set including `applicable(R,...)`
(antecedent satisfied), `defeated`, `discarded`, and per-rule `obligation(R,X,N)`. A rule is counted
as engaged on a step when the engine says it was applicable there.

FAITHFUL REPLAY
---------------
The engine's input is NOT just the Grounder facts stored per record. `NormativeMemory.derived_facts()`
adds history-dependent atoms, notably `visited(C)`, computed as the union of prior `occupies(C)` and
`passed_through(C)`. Without those, R7b (yellow -> red on taint) can never fire and the counts are
wrong. We therefore rebuild a NormativeMemory per episode and append records in order, so step t sees
exactly the facts step t saw when the run happened.

    /newdata2/dylantw/envs/dino_wm/bin/python scripts/rule_engagement.py \
        --root results/no_yaw/sign_change/social --lawset full_lawset
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from legislation.memory import NormativeMemory        # noqa: E402
from legislation.reasoner import LegislativeReasoner  # noqa: E402

import re                                             # noqa: E402

_EXPANDED = re.compile(r"_\d+$")


def _base_rule(label):
    """reach_goal_8 -> reach_goal. Schematic rules are expanded per binding by `expand` in the
    YAML; the paper reports the law, not each grounding of it."""
    return _EXPANDED.sub("", label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/no_yaw/sign_change/social")
    ap.add_argument("--lawset", default="full_lawset")
    ap.add_argument("--yellow_cells", default="3,5",
                    help="must match legislation.yellow_cells for the run being replayed")
    ap.add_argument("--limit", type=int, default=None, help="cap episodes (smoke test)")
    ap.add_argument("--out", default=None, help="optional JSON dump of the counts")
    args = ap.parse_args()

    reasoner = LegislativeReasoner(active_lawsets=[args.lawset])
    base = {"cube"} | {f"yellow_cell({int(c)})" for c in args.yellow_cells.split(",") if c.strip()}

    def goal_of(ep):
        """goal_cell(k) is a per-eval base fact; recover it from the recorded obligations, where
        reach_goal surfaces it as in_cell(k) on the episode's very first step."""
        for rec in ep["records"]:
            for o in (rec.get("verdict") or {}).get("obligations") or []:
                if o.startswith("in_cell(") and o != "in_cell(4)":
                    return {f"goal_cell({o[len('in_cell('):-1]})"}
        return set()
    applicable, obliged, defeated = Counter(), Counter(), Counter()
    n_steps = n_eps = 0

    files = sorted(glob.glob(str(_REPO / args.root / "*" / "batch_*" / "normative_ledger.json")))
    for f in files:
        for ek, ep in json.load(open(f)).items():
            if args.limit and n_eps >= args.limit:
                break
            n_eps += 1
            mem = NormativeMemory()
            goal = goal_of(ep)
            for rec in ep["records"]:
                # derived_facts() reads the records appended SO FAR, so append first, then solve:
                # step t must see the taint accumulated over steps 0..t, exactly as it did at runtime.
                mem.append(rec["facts"], rec.get("verdict"), rec.get("gt_xy"))
                # BASE facts matter: enforcement.py:146 feeds base + goal + current + derived. Without
                # `cube` the always-applicable rules (R1/R2) never fire, and without yellow_cell(Y) the
                # sign constitutive rules never fire -- both would silently read as zero engagement.
                # RESOLVED SIGN, not the raw perceived one. enforcement.py:140-142 swaps sign(...) for
                # the latched/derived colour before solving, so replaying the raw fact understates
                # every sign-conditioned rule (green_sign came out 2050 vs 2465 permissions granted).
                cur = {f for f in rec["facts"] if not f.startswith("sign(")}
                eff = rec.get("effective_sign")
                if eff:
                    cur.add(f"sign({eff})")
                facts = sorted(cur | set(mem.derived_facts()) | base | goal)
                atoms = reasoner.solve_all(facts)
                n_steps += 1
                # Count each rule ONCE per step. applicable(R,X) is emitted per HEAD LITERAL, so a
                # two-consequent rule like contrary_to_duty would otherwise score 2x. Schematic rules
                # are also collapsed back to their base label (reach_goal_8 -> reach_goal), since the
                # expansion is an implementation detail of `expand`, not a separate law.
                for key, ctr in (("applicable", applicable), ("defeated", defeated)):
                    seen = {_base_rule(a[len(key) + 1:-1].split(",")[0]) for a in atoms.get(key, [])}
                    for r in seen:
                        ctr[r] += 1
                seen_o = {_base_rule(a[len("obligation("):-1].split(",")[0])
                          for a in atoms.get("obligation", [])
                          if len(a[len("obligation("):-1].split(",")) == 3}
                for r in seen_o:
                    obliged[r] += 1
        if args.limit and n_eps >= args.limit:
            break

    print(f"{n_eps} episodes, {n_steps} decision steps, lawset={args.lawset}\n")
    print(f"  {'rule':32s} {'applicable':>10s} {'defeated':>9s} {'obligation(R,X,N)':>18s}")
    for r, c in applicable.most_common():
        print(f"  {r:32s} {c:10d} {defeated[r]:9d} {obliged[r]:18d}")
    extra = set(defeated) | set(obliged) - set(applicable)
    for r in sorted(extra - set(applicable)):
        print(f"  {r:32s} {'0':>10s} {defeated[r]:9d} {obliged[r]:18d}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"episodes": n_eps, "steps": n_steps, "lawset": args.lawset,
             "applicable": dict(applicable), "defeated": dict(defeated),
             "obligation_per_rule": dict(obliged)}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
