"""CONFORMAL QUANTILE REGRESSION (CQR) -- a cushion that varies CONTINUOUSLY with the situation.

THE PROGRESSION
---------------
    split_conformal.py   one number for the whole grid
    mondrian_conformal.py one number per (cell, direction) BIN
    cqr.py (this file)   a smooth function delta(x) of the situation x

Mondrian already adapts, but only as a step function over hand-chosen bins: it cannot use push
LENGTH, distance to the cell boundary, or anything continuous, and it wastes data by splitting
it into disjoint buckets. CQR (Romano, Patterson, Candes -- Conformalized Quantile Regression,
NeurIPS 2019, arXiv:1905.03222) instead fits a conditional quantile of the score and then
applies a conformal correction to whatever that regressor gets wrong.

HOW IT WORKS
------------
1. On a TRAINING split, fit qhi(x) ~ the (1-alpha) conditional quantile of the score s, using
   the pinball (quantile) loss. Any regressor works; we use a small MLP in torch to avoid a
   new dependency.
2. On a disjoint CALIBRATION split, form the one-sided conformity residual

       E_i = s_i - qhi(x_i)

   and take its (1-alpha) conformal quantile, qhat.
3. Deploy the cushion

       delta(x) = qhi(x) + qhat

This retains EXACT distribution-free marginal coverage no matter how bad the quantile
regressor is (step 2 repairs it), while inheriting locally-adaptive width when the regressor
is good. That is the appealing property: the model can only help, never break the guarantee.

ONE-SIDED ON PURPOSE
--------------------
Textbook CQR builds a two-sided interval. We only ever need an UPPER margin -- a cushion that
is too large is merely conservative, not unsafe -- so we conformalise the upper quantile only.
This is tighter than taking the upper half of a two-sided interval at the same alpha.

FEATURES
--------
Deliberately kept simple and interpretable so the learned cushion can be audited:
    - one-hot start cell (9)                  where on the grid we are
    - sin/cos of push heading (2)             direction, smooth and wrap-around safe
    - push length (1)                         longer strokes accumulate more WM error
    - signed distance from the BELIEVED swept footprint to the forbidden cell (1)
      This last one is the important feature: error matters most when the planner already
      thinks it is close to the boundary.

Note the features must be computable AT PLAN TIME from believed quantities only -- never from
ground truth -- or the cushion could not actually be applied online. That is why the distance
feature uses probe positions, not GT.

CAVEAT
------
Same as Mondrian: this approximates conditional coverage, it does not achieve it exactly
(impossible -- Barber, Candes, Ramdas, Tibshirani 2021). Report per-region empirical coverage.

USAGE
-----
    python scripts/conformal/cqr.py --alpha 0.05 --epochs 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from scripts.conformal.common import (  # noqa: E402
    conformal_quantile, empirical_coverage, push_direction_bin, save_result,
    swept_signed_distance,
)
from scripts.conformal.split_conformal import add_common_args, load_scores, split_by_episode  # noqa: E402


def build_features(d, cell: int, cube_half: float) -> np.ndarray:
    """Assemble the plan-time feature matrix (see FEATURES in the module docstring).

    Every column is derived from BELIEVED quantities (probe positions), never ground truth, so
    the fitted cushion is actually deployable inside the planner.
    """
    ps = np.asarray(d["probe_start"], dtype=np.float64).reshape(-1, 2)
    pe = np.asarray(d["probe_end"], dtype=np.float64).reshape(-1, 2)
    disp = pe - ps

    cells = np.asarray(d["start_cell"], dtype=int)
    onehot = np.zeros((len(cells), 9), dtype=np.float64)
    valid = (cells >= 0) & (cells < 9)                      # off-grid is -1; leave its row zero
    onehot[np.arange(len(cells))[valid], cells[valid]] = 1.0

    theta = np.arctan2(disp[:, 1], disp[:, 0])
    length = np.linalg.norm(disp, axis=1)
    d_bel = swept_signed_distance(ps, pe, cell, cube_half)  # believed clearance -- the key feature

    return np.column_stack([onehot, np.sin(theta), np.cos(theta), length, d_bel])


class QuantileMLP(nn.Module):
    """Small MLP trained with the pinball loss to predict a conditional quantile."""

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pinball_loss(pred, target, tau: float):
    """Pinball / quantile loss. Minimised by the conditional tau-quantile of `target`.

    Asymmetric: under-prediction is penalised with weight tau, over-prediction with (1-tau).
    At tau = 0.95 that means missing high is 19x worse than missing low, which is what pushes
    the fit up to the upper quantile rather than the mean.
    """
    err = target - pred
    return torch.maximum(tau * err, (tau - 1.0) * err).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu", help="tiny model; cpu is fine and keeps GPUs free")
    ap.add_argument("--out", default="cqr", help="basename under data/conformal/")
    args = ap.parse_args()

    s, d = load_scores(args)
    X = build_features(d, args.cell, args.cube_half)

    # THREE disjoint episode-level splits: fit / calibrate / test. Reusing the fit split for
    # calibration would invalidate the conformal step, which is the whole point of CQR.
    is_a, is_rest = split_by_episode(d["episode"], args.seed, frac=0.4)          # fit
    sub, _ = split_by_episode(d["episode"][is_rest], args.seed + 1, frac=0.5)    # split the rest
    is_cal = np.zeros_like(is_a)
    is_cal[np.where(is_rest)[0][sub]] = True
    is_test = is_rest & ~is_cal

    dev = torch.device(args.device)
    # Standardise features on the FIT split only -- calibration/test must not inform the scaler.
    mu, sd = X[is_a].mean(0), X[is_a].std(0) + 1e-8
    Xn = (X - mu) / sd
    xt = lambda m: torch.tensor(Xn[m], dtype=torch.float32, device=dev)   # noqa: E731
    st = lambda m: torch.tensor(s[m], dtype=torch.float32, device=dev)    # noqa: E731

    tau = 1.0 - args.alpha
    model = QuantileMLP(X.shape[1], args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    xa, sa = xt(is_a), st(is_a)
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = pinball_loss(model(xa), sa, tau)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"[cqr] epoch {ep+1}/{args.epochs}  pinball={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        q_cal = model(xt(is_cal)).cpu().numpy()
        q_test = model(xt(is_test)).cpu().numpy()

    # Conformal repair: the (1-alpha) quantile of the one-sided residual.
    E_cal = s[is_cal] - q_cal
    qhat = conformal_quantile(E_cal, args.alpha)
    delta_test = q_test + qhat                      # the deployable, situation-dependent cushion

    cov = float((s[is_test] <= delta_test).mean())
    print(f"\n[cqr] conformal correction qhat = {qhat*100:.3f} cm")
    print(f"[cqr] holdout coverage = {cov:.4f}  (target {1-args.alpha:.4f})")
    print(f"[cqr] cushion delta(x) on holdout: mean={delta_test.mean()*100:.3f} cm  "
          f"min={delta_test.min()*100:.3f}  max={delta_test.max()*100:.3f}")

    # The payoff claim: an adaptive cushion should be TIGHTER on average than the global one at
    # matched coverage. If it is not, say so -- a null result here is worth reporting.
    q_global = conformal_quantile(s[is_cal], args.alpha)
    print(f"[cqr] global split-conformal cushion = {q_global*100:.3f} cm  "
          f"-> adaptive is {100*(1 - delta_test.mean()/q_global):+.1f}% tighter on average"
          if np.isfinite(q_global) else "")

    # Per-cell coverage: the evidence that approximate conditional coverage actually holds.
    per_cell = []
    cells_test = np.asarray(d["start_cell"])[is_test]
    for c in sorted(set(cells_test.tolist())):
        m = cells_test == c
        per_cell.append({"cell": int(c), "n": int(m.sum()),
                         "coverage": float((s[is_test][m] <= delta_test[m]).mean()),
                         "mean_delta_m": float(delta_test[m].mean())})
        print(f"    cell {c}: n={per_cell[-1]['n']:<5} coverage={per_cell[-1]['coverage']:.4f}  "
              f"mean delta={per_cell[-1]['mean_delta_m']*100:.3f} cm")

    save_result(args.out, {
        "method": "conformalized_quantile_regression_one_sided",
        "residuals": str(args.residuals), "meta": d.get("meta", {}),
        "alpha": args.alpha, "cell": args.cell, "cube_half": args.cube_half, "seed": args.seed,
        "n_fit": int(is_a.sum()), "n_cal": int(is_cal.sum()), "n_test": int(is_test.sum()),
        "qhat_correction_m": float(qhat),
        "qhat_global_m": float(q_global),
        "holdout_coverage": cov,
        "delta_mean_m": float(delta_test.mean()),
        "delta_min_m": float(delta_test.min()), "delta_max_m": float(delta_test.max()),
        "per_cell_coverage": per_cell,
        "features": ["cell_onehot(9)", "sin(theta)", "cos(theta)", "push_len", "d_believed"],
    })


if __name__ == "__main__":
    main()
