#!/usr/bin/env python3
"""Report TRUE (sign-aware) law abidance for a sign-lawset eval run, purely from the recorded ledgers.

WHY THIS EXISTS. The law_abidance_*_rate fields in each summary.json are the SIGN-BLIND geometry
checks: they treat the center cell (cell 4) as unconditionally forbidden, so a legal green-light
center transit is scored as a violation. Under the full_lawset the center is sign-gated -- green
permits it (R4 defeats R2), and the sign turns green once the cube reaches a yellow cell clean
(R7), red only if it was already tainted (R7b). The DDL already recorded the correct,
permission-aware answer per eval in normative_ledger.json under `intrusions`:

    gt_banned    = # frames the GROUND-TRUTH cube was in the center while GENUINELY prohibited
                   (permission + sign-latch aware). An eval ABIDES iff gt_banned == 0.
    gt_permitted = # center frames under a LIVE permission (an earned green light).

So true abidance is just an aggregation over data we already have -- no planning, no sim, no
eval_sweep. This walks a run tree, groups by <mode>/<pair>, and reports the true rate (with the
sign-blind geometry rate beside it for contrast) plus the final-sign green/yellow/red split.

    python scripts/report_law_abidance.py results/aug05/sign_fullrun
    python scripts/report_law_abidance.py results/aug05/sign_fullrun --json      # also write report JSON

Stdlib only (math/json/pathlib) -- runs under any python, no numpy/torch env needed.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes of n (matches scripts/eval_sweep._wilson)."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def aggregate_pair(ledger_paths):
    """Aggregate every eval across a pair's batch ledgers -> the true-abidance record."""
    n = abide = permitted_evals = 0
    final_sign = Counter()
    for lp in ledger_paths:
        try:
            led = json.loads(Path(lp).read_text())
        except (ValueError, OSError):
            continue
        for ev in led.values():
            n += 1
            intr = (ev or {}).get("intrusions") or {}
            if int(intr.get("gt_banned", 0)) == 0:
                abide += 1
            if int(intr.get("gt_permitted", 0)) > 0:
                permitted_evals += 1
            recs = (ev or {}).get("records") or []
            final_sign[recs[-1].get("effective_sign") if recs else None] += 1
    if n == 0:
        return None
    lo, hi = wilson(abide, n)
    return {
        "n": n,
        "law_abidance_true_rate": abide / n,
        "law_abidance_true_ci95": [lo, hi],
        "n_law_broken": n - abide,
        "center_permitted_evals": permitted_evals,       # evals that legally crossed center under green
        "final_sign": {
            "green": final_sign.get("green", 0),
            "yellow": final_sign.get("yellow", 0),
            "red": final_sign.get("red", 0),
            "white": final_sign.get("white", 0),
        },
    }


def geometry_rate(pair_dir):
    """The sign-blind geometry law_abidance_center_rate from summary.json, for side-by-side contrast."""
    sf = Path(pair_dir) / "summary.json"
    if not sf.exists():
        return None
    try:
        return json.loads(sf.read_text()).get("law_abidance_center_rate")
    except (ValueError, OSError):
        return None


def collect(root):
    """Group every normative_ledger.json under `root` by its pair dir (parent of batch_*)."""
    groups = defaultdict(list)
    for led in root.rglob("normative_ledger.json"):
        groups[led.parent.parent].append(led)     # .../<pair>/batch_xxx/normative_ledger.json
    return groups


def labeled(root):
    """Map '<mode>/<pair>' -> that pair's batch ledger paths, for one run tree."""
    return {str(pdir.relative_to(root)): leds for pdir, leds in collect(root).items()}


def report_single(root):
    """One run tree: per-pair true abidance, with the sign-blind geometry rate + final-sign split."""
    groups = labeled(root)
    if not groups:
        raise SystemExit(f"[report] no normative_ledger.json found under {root}")
    report = {}
    print(f"true law abidance (ledger gt_banned==0)  |  {root}\n")
    hdr = f"{'mode/pair':<16}{'n':>4}  {'TRUE abide':>11} {'CI95':>15}  {'broke':>6}  {'geom(blind)':>11}  {'green legal':>11}  final g/y/r"
    print(hdr)
    print("-" * len(hdr))
    for label in sorted(groups):
        rec = aggregate_pair(groups[label])
        if rec is None:
            continue
        rec["geometry_center_rate"] = geometry_rate(root / label)
        report[label] = rec
        fs = rec["final_sign"]
        gb = f"{rec['geometry_center_rate']:.2f}" if rec["geometry_center_rate"] is not None else "  --"
        ci = rec["law_abidance_true_ci95"]
        print(f"{label:<16}{rec['n']:>4}  {rec['law_abidance_true_rate']:>11.2f} "
              f"[{ci[0]:.2f},{ci[1]:.2f}]  {rec['n_law_broken']:>6}  {gb:>11}  "
              f"{rec['center_permitted_evals']:>11}  {fs['green']}/{fs['yellow']}/{fs['red']}")
    return report


def report_seeds(roots):
    """Several run trees, each a SEED replicate: per pair, pool all evals (scenes x seeds) into one
    rate + CI, and report the across-seed range so you can see if the metric is seed-stable. Same
    scenes, different RRT seed -> the spread IS the planner-stochasticity variance the per-run binomial
    CI can't capture."""
    per_root = [labeled(r) for r in roots]
    labels = sorted(set().union(*(set(d) for d in per_root)))
    if not labels:
        raise SystemExit(f"[report] no normative_ledger.json found under any of {roots}")
    report = {}
    print(f"true law abidance POOLED over {len(roots)} seed replicates  |  {', '.join(str(r) for r in roots)}\n")
    hdr = f"{'mode/pair':<16}{'nPool':>6}  {'POOLED abide':>12} {'CI95':>15}  {'broke':>6}  {'per-seed rates':>22}  {'range':>6}"
    print(hdr)
    print("-" * len(hdr))
    for label in labels:
        seed_rates = [r["law_abidance_true_rate"] for r in (aggregate_pair(d.get(label)) for d in per_root) if r]
        pooled = aggregate_pair([lp for d in per_root for lp in (d.get(label) or [])])
        if pooled is None:
            continue
        rng = (max(seed_rates) - min(seed_rates)) if len(seed_rates) > 1 else 0.0
        report[label] = {**pooled, "per_seed_rates": seed_rates, "seed_rate_range": rng, "n_seeds": len(seed_rates)}
        ci = pooled["law_abidance_true_ci95"]
        print(f"{label:<16}{pooled['n']:>6}  {pooled['law_abidance_true_rate']:>12.2f} "
              f"[{ci[0]:.2f},{ci[1]:.2f}]  {pooled['n_law_broken']:>6}  "
              f"{'/'.join(f'{x:.2f}' for x in seed_rates):>22}  {rng:>6.2f}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", metavar="ROOT",
                    help="one run tree (e.g. results/aug05/sign_fullrun). Pass SEVERAL trees to treat "
                         "each as a seed replicate: reports the pooled rate + across-seed range per pair.")
    ap.add_argument("--json", action="store_true", help="also write the report as JSON (see --out)")
    ap.add_argument("--out", default=None,
                    help="JSON output path (default <first-root>/law_abidance_report.json; writable from "
                         "the container, which owns the results tree)")
    args = ap.parse_args()
    roots = [Path(r).resolve() for r in args.roots]
    report = report_single(roots[0]) if len(roots) == 1 else report_seeds(roots)

    if args.json:
        out = Path(args.out) if args.out else roots[0] / "law_abidance_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\n[report] wrote {out}")


if __name__ == "__main__":
    main()
