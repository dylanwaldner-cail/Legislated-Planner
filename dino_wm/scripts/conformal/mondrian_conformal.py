"""MONDRIAN (group-conditional) CONFORMAL -- a cushion that varies by WHERE and WHICH WAY you push.

THE PROBLEM WITH ONE GLOBAL DELTA
----------------------------------
split_conformal.py prices the whole grid at the worst region's rate. Our WM error is known to
be anisotropic and spatially structured -- a +x blind spot, a corner grounding bias, a heavy
p99 tail concentrated near the forbidden cell -- so a single delta simultaneously
    OVER-cushions the easy cells (needless detours; the hidden over-conservatism cost), and
    UNDER-cushions the hard ones (the leaks we actually care about).

THE FIX
-------
Partition strokes into groups and calibrate a SEPARATE quantile per group (Vovk's Mondrian
conformal; Bostrom & Johansson). Here the group is

    (start cell) x (push-direction bin)

which yields a literal hot/cold error map over the grid. Within each group the guarantee is
the ordinary split-conformal one, so we get GROUP-CONDITIONAL coverage:

    P(leak | group g) <= alpha    for every g with enough calibration data

That conditional-by-group property is also what makes this the right tool for our covariate
shift: RRT's deployment distribution REWEIGHTS the groups relative to the offline dataset
(it pushes toward goals and detours around cell 4), but a per-group guarantee is invariant to
reweighting across groups. Marginal calibration is not. This is the single most important
reason to prefer Mondrian here over split_conformal.

Do this BEFORE full CQR: it is the discrete, one-afternoon version of the same idea, and it
should be reported regardless because it is trivially auditable -- a reviewer can read the
per-group table.

THIN GROUPS
-----------
A group needs at least ceil((n+1)(1-alpha)) - 1 points to certify level alpha at all; below
that the conformal quantile is +inf (correctly: "this data cannot certify that"). Rather than
emit an infinite cushion we FALL BACK to the global quantile for thin groups and mark them
`fallback: true`, so the table shows exactly where the calibration set was too sparse. Never
silently drop them -- a dropped group is an unpriced region.

HONEST CAVEAT
-------------
Group-conditional is not point-conditional. Exact per-point conditional coverage is impossible
(Barber, Candes, Ramdas, Tibshirani 2021); Mondrian approximates it by partition. Report the
per-group empirical coverage table -- that is the evidence the approximation is adequate.

USAGE
-----
    python scripts/conformal/mondrian_conformal.py --alpha 0.05 --n_dir_bins 8
    python scripts/conformal/mondrian_conformal.py --group_by cell      # cells only, coarser
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from scripts.conformal.common import (  # noqa: E402
    conformal_quantile, empirical_coverage, leak_rate, push_direction_bin, save_result,
)
from scripts.conformal.split_conformal import add_common_args, load_scores, split_by_episode  # noqa: E402


def build_groups(d, how: str, n_dir_bins: int):
    """Return (group_id per stroke, human-readable label per group id).

    `how` selects the partition granularity:
      'cell'      -- start cell only (9 groups; coarse but always well-populated)
      'direction' -- push heading only (n_dir_bins groups; isolates the +x weakness)
      'both'      -- their product (the full hot/cold map; finest, thinnest groups)
    """
    cells = np.asarray(d["start_cell"], dtype=int)
    dirs = push_direction_bin(d["probe_start"], d["probe_end"], n_bins=n_dir_bins)
    if how == "cell":
        gid = cells
        label = {int(g): f"cell{g}" for g in np.unique(gid)}
    elif how == "direction":
        gid = dirs
        label = {int(g): f"dir{g}" for g in np.unique(gid)}
    elif how == "both":
        gid = cells * 1000 + dirs
        label = {int(g): f"cell{g // 1000}/dir{g % 1000}" for g in np.unique(gid)}
    else:
        raise SystemExit(f"unknown --group_by {how!r}")
    return gid, label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--alpha", type=float, default=0.05, help="target per-group leak probability")
    ap.add_argument("--group_by", default="both", choices=["cell", "direction", "both"])
    ap.add_argument("--n_dir_bins", type=int, default=8, help="angular sectors for push heading")
    ap.add_argument("--min_group", type=int, default=30,
                    help="below this many calibration points a group falls back to the global qhat")
    ap.add_argument("--out", default="mondrian_conformal", help="basename under data/conformal/")
    args = ap.parse_args()

    s, d = load_scores(args)
    gid, label = build_groups(d, args.group_by, args.n_dir_bins)
    is_cal, is_test = split_by_episode(d["episode"], args.seed)

    # Global quantile: both the comparison baseline and the fallback for thin groups.
    q_global = conformal_quantile(s[is_cal], args.alpha)
    print(f"[mondrian] {s.size} strokes, {len(label)} groups, alpha={args.alpha}")
    print(f"[mondrian] GLOBAL qhat = {q_global*100:.3f} cm  (the split-conformal baseline)\n")

    rows = []
    for g in sorted(label):
        in_g = gid == g
        s_cal_g, s_test_g = s[in_g & is_cal], s[in_g & is_test]
        thin = s_cal_g.size < args.min_group
        q_g = q_global if thin else conformal_quantile(s_cal_g, args.alpha)
        if not np.isfinite(q_g):                       # too few points even above min_group
            q_g, thin = q_global, True
        rows.append({
            "group": label[g], "group_id": int(g),
            "n_cal": int(s_cal_g.size), "n_test": int(s_test_g.size),
            "qhat_m": float(q_g), "fallback": bool(thin),
            "empirical_coverage_holdout": empirical_coverage(s_test_g, q_g),
            "leak_rate_holdout": leak_rate(s_test_g, q_g),
        })

    # Sort the printed table by cushion so the hot regions are immediately visible.
    for r in sorted(rows, key=lambda r: -r["qhat_m"]):
        flag = " (fallback)" if r["fallback"] else ""
        cov = r["empirical_coverage_holdout"]
        print(f"  {r['group']:<16} n_cal={r['n_cal']:<6} delta={r['qhat_m']*100:7.3f} cm  "
              f"holdout cov={cov:.4f}{flag}")

    # The headline comparison: does the adaptive cushion buy anything over one global number?
    finite = [r for r in rows if r["n_test"] > 0]
    w = np.array([r["n_test"] for r in finite], dtype=float)
    mean_delta = float(np.average([r["qhat_m"] for r in finite], weights=w)) if w.sum() else float("nan")
    print(f"\n[mondrian] test-weighted mean cushion = {mean_delta*100:.3f} cm "
          f"vs global {q_global*100:.3f} cm  "
          f"({100*(1 - mean_delta/q_global):+.1f}% tighter on average)" if np.isfinite(mean_delta)
          else "")
    n_fallback = sum(r["fallback"] for r in rows)
    if n_fallback:
        print(f"[mondrian] WARNING: {n_fallback}/{len(rows)} groups fell back to the global qhat "
              f"(too few calibration points) -- these regions are NOT group-certified.")

    save_result(args.out, {
        "method": "mondrian_group_conditional_conformal",
        "residuals": str(args.residuals), "meta": d.get("meta", {}),
        "alpha": args.alpha, "group_by": args.group_by, "n_dir_bins": args.n_dir_bins,
        "min_group": args.min_group, "cell": args.cell, "seed": args.seed,
        "qhat_global_m": float(q_global),
        "mean_qhat_m": mean_delta, "n_fallback_groups": int(n_fallback),
        "groups": rows,
    })


if __name__ == "__main__":
    main()
