"""INVERSE CALIBRATION -- what risk level alpha does our HAND-TUNED cushion actually correspond to?

THE POINT
---------
Every other script here answers "given a risk budget alpha, what cushion delta do I need?".
This one runs the map backwards: "we already deployed delta = 0.03 m -- what does that
GUARANTEE?" That is the cheapest high-value framing move available, because it converts a
knob that currently has no probabilistic meaning into a risk statement, without changing a
single line of the planner.

THE CONTROL-THEORY FRAMING (this is what to cite)
--------------------------------------------------
Chance-Constrained RRT (Luders, Kothari, How, 2010) is RRT where every node must satisfy
P(collision) <= alpha, enforced by inflating the obstacle by a margin derived from a risk
bound. Our cushion IS exactly that construction with the risk bound left implicit. Backing
out the implicit alpha makes the correspondence explicit and lets us write

    "delta = 0.03 m corresponds to a certified per-stroke violation probability <= alpha_hat"

Related framings worth naming in the same breath: an HJ-reachability tracking-error tube
(Bansal/Tomlin) inflates obstacles by a bound on deviation; a CBF needs the same. All four --
safety margin, chance constraint, reachability tube, conformal radius -- are the same object
(an inflation of the forbidden set sized by uncertainty) and differ only in the ASSUMPTION
that sizes it. Ours is distribution-free by necessity: the WM error is heavy-tailed
(p99 >> mean), so the bounded-worst-case methods do not cleanly apply, and a conformal
quantile is the principled choice. That sentence is the one that shows command of the
landscape -- see research_lessons.md, Lesson 15.

HOW alpha_hat IS COMPUTED
-------------------------
Given the calibration scores s (see common.py), the deployed delta fails on exactly the
strokes with s > delta. We report:

  alpha_empirical   the raw holdout leak rate at delta -- what we observed
  alpha_certified   the CRC-corrected upper bound (n*Rhat + 1)/(n+1) -- what we can CLAIM

Always quote the certified number in the paper; the empirical one is the point estimate that
the certificate is built on, and the gap between them is just finite-sample honesty.

We also invert the other way -- the smallest delta certifying each of a few standard alphas --
so the table shows both directions at once.

USAGE
-----
    python scripts/conformal/chance_constraint_alpha.py --deltas 0.0 0.01 0.02 0.03 0.04 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from scripts.conformal.common import conformal_quantile, leak_rate, save_result  # noqa: E402
from scripts.conformal.split_conformal import add_common_args, load_scores, split_by_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                    help="cushions to price, in metres (defaults match the run cushion sweep)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10],
                    help="risk levels to invert back into a required cushion")
    ap.add_argument("--out", default="chance_constraint_alpha", help="basename under data/conformal/")
    args = ap.parse_args()

    s, d = load_scores(args)
    is_cal, is_test = split_by_episode(d["episode"], args.seed)
    s_cal, s_test = s[is_cal], s[is_test]
    n = int(s_cal.size)

    # ---- forward: deployed delta -> implied risk ----------------------------------------
    print(f"[cc-alpha] n_cal={n}  n_test={s_test.size}\n")
    print("  delta(cm)   alpha_empirical(holdout)   alpha_certified(CRC bound)")
    fwd = []
    for delta in args.deltas:
        emp = leak_rate(s_test, delta)
        rhat = leak_rate(s_cal, delta)
        cert = (n * rhat + 1.0) / (n + 1)
        fwd.append({"delta_m": float(delta), "alpha_empirical": emp, "alpha_certified": cert})
        print(f"  {delta*100:8.2f}   {emp:23.5f}   {cert:25.5f}")

    # ---- inverse: target risk -> required delta -------------------------------------------
    print("\n  alpha    required delta (cm)")
    inv = []
    for a in args.alphas:
        q = conformal_quantile(s_cal, a)
        inv.append({"alpha": a, "required_delta_m": q})
        shown = f"{q*100:.3f}" if np.isfinite(q) else "inf (n too small to certify)"
        print(f"  {a:<7} {shown}")

    save_result(args.out, {
        "method": "chance_constraint_inverse_calibration",
        "residuals": str(args.residuals), "meta": d.get("meta", {}),
        "cell": args.cell, "cube_half": args.cube_half, "seed": args.seed,
        "n_cal": n, "n_test": int(s_test.size),
        "delta_to_alpha": fwd,
        "alpha_to_delta": inv,
        "citation": "Luders, Kothari, How -- Chance-Constrained RRT under Uncertainty (2010)",
        "scope_note": ("PER-STROKE risk, on the offline stroke distribution. See "
                       "dump_probe_residuals.py for why this is not yet the deployment "
                       "distribution, and mondrian_conformal.py for the fix."),
    })


if __name__ == "__main__":
    main()
