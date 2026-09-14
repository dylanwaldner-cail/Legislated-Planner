"""Q5 runtime table: planning wall-clock per PRODUCTIVE action, with bootstrap CIs.

WHY A CUSTOM DENOMINATOR
------------------------
"Seconds per action" is only meaningful if the denominator counts actions the planner actually
worked on. Two kinds of step must be excluded or the cost is deflated:

  1. FROZEN steps ([F]moving in the verdict). Under the full lawset R9 freezes a tainted agent, so
     1257 of 4500 observes produce no action at all. Counting them as "actions" made the full-lawset
     plan time look 30% cheaper than the geometric one (17.67 vs 24.52 s/action) -- an artifact, not
     a result. The geometric lawset has ZERO frozen steps (verified: 1665 observes in base_x2), so
     this only bites the sign runs.

     CAVEAT for runs collected BEFORE the freeze fast-path (planning/rrt.py, added 2026-08-18): a
     frozen step used to run a FULL max_samples RRT search that pruned everything, so it was the most
     expensive step in the system, not a free one -- batches with a high frozen fraction spent 22.9 s
     of RRT per observe vs 10.5 s for low-freeze batches (corr +0.983). `runtime_breakdown` is
     recorded PER BATCH, so that time cannot be removed from the NUMERATOR here; only the denominator
     can be corrected. The Q3 seconds-per-action below therefore carries the frozen steps' search cost
     on the productive actions and is an UPPER BOUND for the current code, which skips that search.
  2. POST-SUCCESS HOLDS. `success_hold` keeps a solved eval alive with zero-displacement holds.
     `n_steps` in eval_metrics already counts only strokes up to success (or the full length if
     unsolved), so using it as the ceiling handles this.

The denominator is therefore: per eval, the steps within `n_steps` that are not under [F]moving.
When a tree has no ledger (final/base and cushion delta_0.00 were stripped to metrics-only) we fall
back to `n_steps`, which is exact there precisely because the geometric lawset never freezes.

CIs are bootstrapped over BATCHES, not steps: batches are the independent unit (each is its own
plan.py process on its own scenes), and steps within a batch share a planner instance.

    /newdata2/dylantw/envs/dino_wm/bin/python scripts/runtime_table.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

# Column -> eval_metrics glob patterns. Q1 baseline, Q2 cushion sweep, Q3 full lawset.
COLUMNS = {
    "Q1 geometric (base+x2)": [
        "results/final/base/social/*/batch_*/eval_metrics.json",
        "results/final/base_x2/social/*/batch_*/eval_metrics.json",
    ],
    "Q2 cushion sweep": [
        "results/final/social_cushion/*/social/*/batch_*/eval_metrics.json",
    ],
    "Q3 full lawset": [
        "results/no_yaw/sign_change/social/*/batch_*/eval_metrics.json",
    ],
}
COMPONENTS = ["plan_total_s", "rrt_s", "legislation_s", "legislation_reason_s", "legislation_prune_s"]


def _records(ep):
    """Ledger episodes come in two schemas: {records: [...], ...} (current) and a bare list (older
    cushion runs). Normalise both to a list of step records."""
    return ep["records"] if isinstance(ep, dict) else ep


def productive_steps(metrics_path: Path, n_steps):
    """Steps the MPC loop actually planned an action on, per eval: run to whichever terminal comes
    FIRST -- freeze ([F]moving), task success, or exhausting the step budget without arriving.

      * post-success holds: `n_steps` from eval_metrics is already the count up to success (or the
        full length when unsolved), so capping at it drops them.
      * freeze: truncate at the FIRST [F]moving step. Under the full lawset taint is monotone, so a
        freeze is absorbing -- truncating and simply dropping frozen steps should agree, and the
        caller checks that they do rather than assuming it.

    Returns (truncate_at_freeze, drop_frozen_only, had_ledger). Without a ledger both fall back to
    sum(n_steps), which is exact there precisely because the geometric lawset never freezes."""
    led = metrics_path.parent / "normative_ledger.json"
    if not led.exists():
        n = int(sum(v for v in n_steps if v))
        return n, n, False
    L = json.loads(led.read_text())
    trunc = drop = 0
    for k, ep in L.items():
        recs = _records(ep)
        cap = n_steps[int(k)] if int(k) < len(n_steps) and n_steps[int(k)] else len(recs)
        frozen = ["moving" in ((s.get("verdict") or {}).get("prohibitions") or [])
                  for s in recs[:int(cap)]]
        trunc += frozen.index(True) if True in frozen else len(frozen)
        drop += sum(1 for f in frozen if not f)
    return trunc, drop, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples over batches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON dump")
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    results = {}
    for col, pats in COLUMNS.items():
        rows, n_led, n_fb, n_drop, disagree = [], 0, 0, 0, 0
        for pat in pats:
            for f in sorted(glob.glob(str(_REPO / pat))):
                m = json.loads(Path(f).read_text())
                r = m.get("runtime_breakdown") or {}
                if not r or "plan_total_s" not in r:
                    continue
                d, d2, had = productive_steps(Path(f), m.get("n_steps") or [])
                if d <= 0:
                    continue
                n_led += had
                n_fb += (not had)
                n_drop += d2
                disagree += (d != d2)      # freeze not absorbing in this batch -> the two rules differ
                rows.append(({c: r.get(c, 0.0) for c in COMPONENTS}, d))
        if not rows:
            print(f"[warn] {col}: no batches found")
            continue

        num = {c: np.array([x[0][c] for x in rows]) for c in COMPONENTS}
        den = np.array([x[1] for x in rows], dtype=float)
        pooled = {c: num[c].sum() / den.sum() for c in COMPONENTS}

        # Bootstrap the POOLED ratio over batches (resample batches with replacement).
        idx = rng.randint(0, len(rows), size=(args.boot, len(rows)))
        ci = {}
        for c in COMPONENTS:
            bs = num[c][idx].sum(axis=1) / den[idx].sum(axis=1)
            ci[c] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

        results[col] = {"batches": len(rows), "productive_steps": int(den.sum()),
                        "ledger_batches": n_led, "fallback_batches": n_fb,
                        "pooled": pooled, "ci95": ci}

        print(f"\n{col}:  {len(rows)} batches, {int(den.sum())} productive actions"
              f"  ({n_led} via ledger, {n_fb} via n_steps fallback)")
        if disagree:
            print(f"   [!] {disagree} batch(es): truncate-at-first-freeze ({int(den.sum())}) != "
                  f"drop-frozen-steps ({n_drop}) -- a freeze was NOT absorbing there")
        for c in COMPONENTS:
            lo, hi = ci[c]
            unit = (1000.0, "ms") if c == "legislation_reason_s" else (1.0, "s")
            print(f"   {c:24s} {pooled[c]*unit[0]:9.3f} {unit[1]}"
                  f"   [{lo*unit[0]:8.3f}, {hi*unit[0]:8.3f}]")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
