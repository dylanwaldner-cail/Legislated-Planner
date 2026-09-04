"""Q5 Table: what the legislation apparatus costs, per DEONTIC DECISION (paper's tab:legbreakdown).

WHY THIS DENOMINATOR AND NOT "PER ACTION"
-----------------------------------------
`runtime_breakdown` is written once per BATCH, so it cannot be resolved into "seconds per executed
action": a batch's totals also cover (a) evals that already reached the goal -- mpc.py:135 loops
`while not np.all(is_success)` and rrt.plan() re-plans EVERY eval each iteration -- and (b) steps
frozen by [F]moving. In the 40-batch full-lawset run those are 1774 and 1175 of 4500 eval-steps
respectively, i.e. only 1551 of 4500 are productive actions. A regression of batch totals on the
three step counts cannot separate them either (it returns a NEGATIVE seconds/step for the
post-success class -- unidentified, the counts being collinear with iteration count).

The legislation REASON path does not have that problem, because `leg_n_observe` is recorded and
LawEvaluator.observe() runs exactly once per (eval, MPC iteration). So probe / ground / clingo /
build divided by n_observe is EXACT -- it is the cost of one deontic decision, whatever the planner
then did with it. That is the quantity this script reports. The same denominator makes
`legislation_prune_s` exact as mean enforcement cost per decision.

WHAT IS NOT REPORTED HERE, DELIBERATELY
---------------------------------------
Anything divided by "executed actions". See above -- that needs the per-(eval,step) timing added to
runtime_breakdown["per_step"] (planning/rrt.py) and a fresh run.

THE CLINGO NUMBER IS AMORTISED OVER A CACHE
-------------------------------------------
LegislativeReasoner.assess() memoizes by frozenset(facts) (reasoner.py:225) and the cache lives on
the reasoner instance for the whole batch process. The recorded per-decision clingo time is
therefore (real solves x solve cost) / (all decisions). This script also measures the UNCACHED cost
directly -- replaying the run's own fact-sets through a cache-cleared reasoner -- and reports the
cache hit rate, so both the amortised and the worst-case figure are available. The two are
cross-checked against each other (uncached_mean x miss_rate should reproduce the recorded average).

    /newdata2/dylantw/envs/dino_wm/bin/python scripts/legislation_cost_table.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The four stages LawEvaluator.timing splits observe() into, in pipeline order.
STAGES = [("leg_probe_s", "Probe forward pass (perception)"),
          ("leg_ground_s", r"Grounding (probe reads $\to$ DDL facts)"),
          ("leg_logic_s", "Clingo DDL solve"),
          ("leg_build_s", "Constraint build")]


def _facts_of(rec, mem, base, goal):
    """Rebuild the fact-set observe() actually handed to assess(), for cache accounting.

    enforcement.py feeds base + goal + current + history-derived facts, with the raw perceived
    sign(...) replaced by the LATCHED/derived colour. `effective_sign` is exactly that governing
    colour as recorded, so we substitute it. The caller must have already appended `rec` to `mem`,
    since derived_facts() must include this step's own taint."""
    cur = {f for f in rec["facts"] if not f.startswith("sign(")}
    if rec.get("effective_sign"):
        cur.add(f"sign({rec['effective_sign']})")
    return frozenset(cur | set(mem.derived_facts()) | base | goal)


def _goal_of(ep):
    """goal_cell(k) is a per-eval base fact; recover it from the recorded obligations, where the
    reach_goal rule surfaces it as in_cell(k) (cell 4 excluded -- that one is the prohibition)."""
    for rec in ep["records"]:
        for o in (rec.get("verdict") or {}).get("obligations") or []:
            if o.startswith("in_cell(") and o != "in_cell(4)":
                return {f"goal_cell({o[len('in_cell('):-1]})"}
    return set()


def _boot(per_batch_ms, n=10000, seed=0):
    """95% CI on the pooled mean, resampling BATCHES (the independent unit: each is its own
    plan.py process on its own scenes; decisions within a batch share a reasoner + its cache)."""
    a = np.asarray(per_batch_ms, dtype=float)
    rng = np.random.RandomState(seed)
    bs = a[rng.randint(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/aug20/sign_change/social",
                    help="the social full-lawset run the table describes")
    ap.add_argument("--lawset", default="full_lawset")
    ap.add_argument("--yellow_cells", default="3,5")
    ap.add_argument("--sample", type=int, default=400, help="distinct fact-sets to time uncached")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(str(_REPO / args.root / "*" / "batch_*" / "eval_metrics.json")))
    if not files:
        raise SystemExit(f"no eval_metrics.json under {args.root}")

    # ---------------------------------------------------------------- recorded per-decision costs
    tot = {k: 0.0 for k, _ in STAGES}
    tot["legislation_reason_s"] = tot["legislation_prune_s"] = 0.0
    n_obs = n_batches = n_eps = 0
    per_batch = {k: [] for k in list(tot)}
    for f in files:
        m = json.loads(Path(f).read_text())
        r = m.get("runtime_breakdown") or {}
        if "leg_n_observe" not in r:
            continue
        nb = int(r["leg_n_observe"])
        # ASSERT the denominator rather than trusting it: observe() runs once per (eval, MPC
        # iteration), so leg_n_observe must equal the total number of ledger records in this batch.
        led = Path(f).parent / "normative_ledger.json"
        if led.exists():
            L = json.loads(led.read_text())
            recs = sum(len(ep["records"] if isinstance(ep, dict) else ep) for ep in L.values())
            assert recs == nb, f"{f}: leg_n_observe={nb} but ledger holds {recs} records"
            n_eps += len(L)
        n_batches += 1
        n_obs += nb
        for k in tot:
            v = float(r.get(k, 0.0))
            tot[k] += v
            per_batch[k].append(v / nb * 1000.0)          # ms per decision, this batch

    print(f"source: {args.root}")
    print(f"{n_batches} batches | {n_eps} episodes | {n_obs} deontic decisions "
          f"(= observe() calls = ledger records; asserted per batch)\n")
    print(f"  {'stage':44s} {'ms/decision':>12s}   {'CI95':>18s}")
    sub = 0.0
    for k, label in STAGES:
        ms = tot[k] / n_obs * 1000.0
        sub += ms
        lo, hi = _boot(per_batch[k], args.boot)
        print(f"  {label:44s} {ms:12.3f}   [{lo:7.3f}, {hi:7.3f}]")
    reason_ms = tot["legislation_reason_s"] / n_obs * 1000.0
    lo, hi = _boot(per_batch["legislation_reason_s"], args.boot)
    print(f"  {'-- sum of the four stages':44s} {sub:12.3f}")
    print(f"  {'Symbolic reasoning subtotal (measured timer)':44s} {reason_ms:12.3f}   [{lo:7.3f}, {hi:7.3f}]")
    print(f"     (the {reason_ms - sub:.3f} ms gap is set_goal's once-per-episode goal perception,\n"
          f"      which the four stage timers do not cover)")
    prune_ms = tot["legislation_prune_s"] / n_obs * 1000.0
    lo, hi = _boot(per_batch["legislation_prune_s"], args.boot)
    print(f"  {'Per-candidate legality pruning (enforcement)':44s} {prune_ms:12.3f}   [{lo:7.3f}, {hi:7.3f}]")

    # ---------------------------------------------------------------- cache accounting + true solve
    from legislation.memory import NormativeMemory
    from legislation.reasoner import LegislativeReasoner
    base = {"cube"} | {f"yellow_cell({int(c)})" for c in args.yellow_cells.split(",") if c.strip()}
    classes = {"productive": [0, 0], "frozen": [0, 0], "post-success": [0, 0]}
    universe = set()
    for f in files:
        led = Path(f).parent / "normative_ledger.json"
        if not led.exists():
            continue
        ns = (json.loads(Path(f).read_text()).get("n_steps") or [])
        seen = set()                                      # cache scope == one batch process
        for ek, ep in json.load(open(led)).items():
            mem, goal = NormativeMemory(), _goal_of(ep)
            cap = ns[int(ek)] if int(ek) < len(ns) else len(ep["records"])
            for i, rec in enumerate(ep["records"]):
                mem.append(rec["facts"], rec.get("verdict"), rec.get("gt_xy"))
                fs = _facts_of(rec, mem, base, goal)
                pro = (rec.get("verdict") or {}).get("prohibitions") or []
                cls = ("post-success" if i >= cap else
                       "frozen" if "moving" in pro else "productive")
                classes[cls][0] += 1
                classes[cls][1] += (fs not in seen)
                seen.add(fs)
                universe.add(fs)

    misses = sum(v[1] for v in classes.values())
    total = sum(v[0] for v in classes.values())
    print(f"\nclingo cache (assess() memoizes by frozenset(facts); cache scope = one batch):")
    print(f"  {total} decisions -> {misses} real solves, hit rate {1 - misses/total:.1%}")

    r = LegislativeReasoner(active_lawsets=[args.lawset])
    keys = sorted(universe, key=lambda s: (len(s), sorted(s)))    # deterministic order (no Math.random equiv)
    rng = np.random.RandomState(0)
    samp = [keys[i] for i in rng.choice(len(keys), size=min(args.sample, len(keys)), replace=False)]
    r.assess(samp[0])                                             # warm clingo / theory build
    t = []
    for k in samp:
        r._cache.clear()                                          # force a REAL solve every time
        t0 = time.perf_counter(); r.assess(k); t.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(t)
    for k in samp:
        r.assess(k)                                               # populate, then time pure hits
    t0 = time.perf_counter()
    for _ in range(20):
        for k in samp:
            r.assess(k)
    hit_ms = (time.perf_counter() - t0) / (20 * len(samp)) * 1000.0
    print(f"  uncached assess(): n={len(t)}  mean={t.mean():.2f} ms  median={np.median(t):.2f} ms  "
          f"p90={np.percentile(t,90):.2f} ms")
    print(f"  cached  assess(): {hit_ms:.5f} ms")
    implied = t.mean() * misses / total
    logic_ms = tot["leg_logic_s"] / n_obs * 1000.0
    print(f"  CROSS-CHECK: uncached_mean x miss_rate = {implied:.2f} ms  vs recorded {logic_ms:.2f} ms "
          f"({abs(implied-logic_ms)/logic_ms:.1%} apart)")

    print(f"\n  {'class':14s} {'decisions':>10s} {'solves':>8s} {'miss rate':>10s} {'amortised clingo':>18s}")
    for k, (n, mss) in classes.items():
        print(f"  {k:14s} {n:10d} {mss:8d} {mss/n:10.1%} {t.mean()*mss/n:15.2f} ms")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "root": args.root, "batches": n_batches, "episodes": n_eps, "decisions": n_obs,
            "ms_per_decision": {k: tot[k] / n_obs * 1000.0 for k in tot},
            "stage_sum_ms": sub,
            "cache": {"hit_rate": 1 - misses / total, "solves": misses, "decisions": total,
                      "uncached_mean_ms": float(t.mean()), "uncached_median_ms": float(np.median(t)),
                      "uncached_p90_ms": float(np.percentile(t, 90)), "cached_ms": hit_ms},
            "by_class": {k: {"decisions": v[0], "solves": v[1]} for k, v in classes.items()},
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
