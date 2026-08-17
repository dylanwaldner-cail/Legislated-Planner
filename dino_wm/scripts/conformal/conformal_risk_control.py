"""CONFORMAL RISK CONTROL (CRC) -- certify the EXPECTED VIOLATION RATE, not just a coverage set.

WHY THIS AND NOT PLAIN CONFORMAL
---------------------------------
Split conformal certifies a coverage event: "the score is below qhat with prob >= 1-alpha".
But the thing a safety reviewer asks about is a RISK: "what is the expected violation rate?"
CRC (Angelopoulos, Bates, Fisch, Lei, Schuster -- Conformal Risk Control, ICLR 2024,
arXiv:2208.02814) generalises conformal from that 0/1 coverage event to ANY loss that is
monotone in the tuning parameter. Our safety target IS such a loss, so CRC is the more natural
framing and gives the number directly.

Concretely, for cushion delta define a per-stroke loss L_i(delta) that is non-increasing in
delta. CRC picks

    delta_hat = inf { delta :  (n * Rhat(delta) + B) / (n + 1)  <=  alpha }

where Rhat(delta) is the mean loss on n calibration points and B bounds the loss. Then the
expected loss on a fresh exchangeable point satisfies  E[L(delta_hat)] <= alpha.

TWO LOSSES ARE IMPLEMENTED -- the second is the reason to bother with CRC
-------------------------------------------------------------------------
  --loss leak      L_i = 1{ s_i > delta }            (B = 1)
      The binary "did the pruner leak" loss. With this loss CRC nearly coincides with split
      conformal -- worth stating honestly rather than presenting it as a new guarantee.

  --loss severity  L_i = min(pen_i(delta) / B, 1)     (B = --severity_scale, metres)
      The GRADED loss: how far the true footprint actually intruded into the forbidden cell on
      strokes the pruner let through. This is where CRC genuinely goes beyond conformal, and it
      matches the legal intuition that a 1 mm graze and a full drive-through are not the same
      breach. Bounding E[severity] is a stronger, more informative statement than bounding the
      count of breaches.

RELATION TO THE EXISTING DELTA-SWEEP
-------------------------------------
The cushion sweep already run in results/final/{social,oracle}_cushion IS this risk-vs-delta
curve measured empirically. CRC's contribution is to re-label its axis with a CERTIFIED bound
instead of an observed rate, and to say which delta the certificate picks. Plot the two
together: the empirical episode-level curve, and the CRC per-stroke certificate.

SCOPE OF THE GUARANTEE -- do not overstate this
------------------------------------------------
The certificate is PER STROKE. Converting it to an episode-level violation rate needs the
trajectory distribution, which itself depends on delta (a bigger cushion reroutes the planner),
so the two are complementary, not interchangeable. Keep the empirical delta-sweep as the
episode-level evidence and present CRC as the per-stroke certificate.

USAGE
-----
    python scripts/conformal/conformal_risk_control.py --alpha 0.05 --loss leak
    python scripts/conformal/conformal_risk_control.py --alpha 0.01 --loss severity
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
    save_result, swept_signed_distance,
)
from scripts.conformal.split_conformal import add_common_args, load_scores, split_by_episode  # noqa: E402


def leak_loss(s: np.ndarray, delta: float) -> np.ndarray:
    """L_i = 1{ s_i > delta } -- binary, non-increasing in delta, bounded by B = 1."""
    return (s > delta).astype(np.float64)


def severity_loss(s: np.ndarray, penetration: np.ndarray, delta: float, scale: float) -> np.ndarray:
    """Graded loss: normalised penetration depth, but ONLY on strokes the pruner let through.

    A stroke is caught when s_i <= delta (see the derivation in common.py), so it contributes
    zero loss. A leaked stroke contributes how deep the TRUE footprint actually went into the
    forbidden cell, normalised by `scale` and clipped to 1 so the loss stays bounded (CRC needs
    a finite B).

    `penetration` MUST come from common.swept_penetration, not from -d_true: the signed
    distance's negative branch overstates depth (the Minkowski shortcut preserves the overlap
    predicate but not the depth), which would inflate this loss by centimetres on exactly the
    strokes that matter.

    Non-increasing in delta because raising delta can only move strokes from leaked to caught.
    """
    leaked = s > delta
    return np.clip(np.asarray(penetration) / float(scale), 0.0, 1.0) * leaked


def crc_threshold(deltas, risk_fn, alpha: float, n: int, B: float = 1.0):
    """Smallest delta on the grid whose CRC-corrected empirical risk clears alpha.

    The correction (n*Rhat + B)/(n+1) is what upgrades an empirical mean into a bound on the
    expected loss at a fresh point. Returns (delta_hat, curve) where curve carries both the raw
    and corrected risk at every grid point so the trade-off can be plotted.
    """
    curve = []
    chosen = None
    for d in deltas:
        rhat = float(np.mean(risk_fn(d)))
        corrected = (n * rhat + B) / (n + 1)
        curve.append({"delta_m": float(d), "risk_empirical": rhat, "risk_certified": corrected})
        if chosen is None and corrected <= alpha:
            chosen = float(d)
    return chosen, curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--alpha", type=float, default=0.05, help="risk budget to certify")
    ap.add_argument("--loss", default="leak", choices=["leak", "severity"])
    ap.add_argument("--severity_scale", type=float, default=0.045,
                    help="metres of penetration mapped to loss 1.0 (default: one cube half-extent)")
    ap.add_argument("--delta_max", type=float, default=0.10, help="largest cushion on the grid (m)")
    ap.add_argument("--delta_steps", type=int, default=201)
    ap.add_argument("--out", default="conformal_risk_control", help="basename under data/conformal/")
    args = ap.parse_args()

    s, d = load_scores(args)
    is_cal, is_test = split_by_episode(d["episode"], args.seed)

    # True penetration depth, needed by the severity loss. Computed with the EXACT routine, not
    # from the signed distance's negative branch -- see severity_loss().
    pen_true = swept_penetration(d["gt_start"], d["gt_end"], args.cell, args.cube_half)

    s_cal, s_test = s[is_cal], s[is_test]
    dt_cal, dt_test = pen_true[is_cal], pen_true[is_test]
    n = int(s_cal.size)

    if args.loss == "leak":
        B = 1.0
        risk_cal = lambda dd: leak_loss(s_cal, dd)              # noqa: E731
        risk_test = lambda dd: leak_loss(s_test, dd)            # noqa: E731
    else:
        B = 1.0                                                  # loss already clipped to [0,1]
        risk_cal = lambda dd: severity_loss(s_cal, dt_cal, dd, args.severity_scale)    # noqa: E731
        risk_test = lambda dd: severity_loss(s_test, dt_test, dd, args.severity_scale)  # noqa: E731

    deltas = np.linspace(0.0, args.delta_max, args.delta_steps)
    delta_hat, curve = crc_threshold(deltas, risk_cal, args.alpha, n, B=B)

    print(f"[crc] loss={args.loss}  alpha={args.alpha}  n_cal={n}  n_test={s_test.size}")
    if delta_hat is None:
        print(f"[crc] NO delta <= {args.delta_max} m certifies alpha={args.alpha}. "
              f"Either the risk budget is too tight for this WM, or the grid is too short "
              f"-- raise --delta_max, but treat a very large cushion as evidence the "
              f"perception, not the margin, is the bottleneck.")
    else:
        realised = float(np.mean(risk_test(delta_hat)))
        print(f"[crc] certified delta_hat = {delta_hat*100:.3f} cm")
        print(f"[crc] holdout realised risk at delta_hat = {realised:.5f}  (budget {args.alpha})")

    # A few readable points off the trade-off curve -- this is the Pareto/pressure-test table.
    print("\n  delta(cm)   empirical risk   certified bound")
    for row in curve[:: max(1, len(curve) // 12)]:
        print(f"  {row['delta_m']*100:8.2f}   {row['risk_empirical']:14.5f}   "
              f"{row['risk_certified']:15.5f}")

    save_result(args.out, {
        "method": "conformal_risk_control",
        "residuals": str(args.residuals), "meta": d.get("meta", {}),
        "loss": args.loss, "alpha": args.alpha, "severity_scale": args.severity_scale,
        "cell": args.cell, "cube_half": args.cube_half, "seed": args.seed,
        "n_cal": n, "n_test": int(s_test.size),
        "delta_hat_m": delta_hat,
        "holdout_risk_at_delta_hat": (float(np.mean(risk_test(delta_hat)))
                                      if delta_hat is not None else None),
        "curve": curve,
        "scope_note": ("PER-STROKE certificate. Episode-level violation rate depends on the "
                       "delta-dependent trajectory distribution -- keep the empirical cushion "
                       "sweep as the episode-level evidence."),
    })


if __name__ == "__main__":
    main()
