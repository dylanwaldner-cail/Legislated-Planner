"""SPLIT CONFORMAL -- the baseline: one global cushion delta with a distribution-free guarantee.

WHAT IT DOES
------------
Takes the per-stroke nonconformity scores s = d_bel - d_true (derived in common.py), splits
them into a calibration half and a held-out half, and reports

    qhat(alpha) = the (1-alpha) conformal quantile of s on the calibration half

together with the coverage actually realised on the held-out half. `qhat` IS a cushion in
metres: setting delta = qhat certifies

    P(the pruner leaks a violation on a fresh stroke) <= alpha

WHY START HERE
--------------
This is the simplest thing that replaces a hand-tuned delta with a calibrated one, and it is
the reference every other script in this directory is measured against:
  - mondrian_conformal.py should beat it by being tighter where the WM is accurate,
  - cqr.py should beat Mondrian by adapting continuously rather than per bin,
  - conformal_risk_control.py generalises it from a 0/1 leak to a graded severity loss.

WHAT IT CANNOT DO (say this out loud in the paper)
--------------------------------------------------
The guarantee is MARGINAL: it holds on average over the calibration distribution, not
per-cell or per-direction. A single global delta therefore over-cushions easy regions and
under-cushions the hard ones (our +x blind spot, the corners near cell 4). That is exactly
what Mondrian/CQR fix, and exact per-point conditional coverage is provably impossible
(Barber, Candes, Ramdas, Tibshirani 2021), so the honest claim is marginal here and
approximate-conditional there.

Exchangeability caveat: the offline dataset's stroke distribution is not RRT's deployment
distribution. Read the module docstring of dump_probe_residuals.py before quoting any number.

USAGE
-----
    python scripts/conformal/split_conformal.py --residuals data/conformal/residuals_5k_shift025.npz
    python scripts/conformal/split_conformal.py --cell 4 --alphas 0.01 0.05 0.10 --exclude_train
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
    CONFORMAL_DIR, conformal_quantile, empirical_coverage, leak_rate,
    load_residuals, nonconformity, save_result,
)
from probes.probe_cube_cells import CUBE_HALF  # noqa: E402


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared by every calibration script here, so they stay consistent."""
    ap.add_argument("--residuals", default=str(CONFORMAL_DIR / "residuals_5k_shift025.npz"),
                    help="npz written by dump_probe_residuals.py")
    ap.add_argument("--cell", type=int, default=4, help="forbidden cell the law protects")
    ap.add_argument("--cube_half", type=float, default=CUBE_HALF,
                    help="TRUE cube half-extent; the cushion is separate and swept, not folded in here")
    ap.add_argument("--exclude_train", action="store_true",
                    help="drop strokes from episodes the WM or probe trained on (see the "
                         "contamination note in dump_probe_residuals.py)")
    ap.add_argument("--drop_padded", action="store_true",
                    help="drop strokes whose history window was left-padded (strict parity with "
                         "deployment, which always has a full num_hist context)")
    ap.add_argument("--seed", type=int, default=0, help="controls the calibration/holdout split")


def load_scores(args):
    """Load the dump, apply the requested filters, and return (scores, data_dict).

    Filtering happens HERE rather than in the dump so one expensive encode pass can serve every
    slicing decision.
    """
    d = load_residuals(args.residuals)
    keep = np.ones(len(d["episode"]), dtype=bool)
    if args.exclude_train:
        keep &= ~d["in_wm_train"].astype(bool)            # exact (torch randperm, seed 42)
        keep &= ~d["in_probe_train_guess"].astype(bool)   # GUESS -- probe split was never recorded
    if args.drop_padded:
        keep &= ~d["padded_history"].astype(bool)
    if not keep.any():
        raise SystemExit("no strokes left after filtering -- loosen --exclude_train/--drop_padded")

    d = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
         for k, v in d.items()}
    s = nonconformity(d["probe_start"], d["probe_end"], d["gt_start"], d["gt_end"],
                      cell=args.cell, cube_half=args.cube_half)
    return s, d


def split_by_episode(episodes: np.ndarray, seed: int, frac: float = 0.5):
    """Split BY EPISODE, never by stroke.

    Strokes inside one episode share frames and are strongly dependent, so a stroke-level split
    would leak calibration information into the holdout and flatter the coverage numbers.
    Exchangeability is at the episode level, so the split has to be too.
    """
    uniq = np.unique(episodes)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_cal = max(1, int(round(len(uniq) * frac)))
    cal_eps = set(uniq[:n_cal].tolist())
    is_cal = np.array([e in cal_eps for e in episodes])
    return is_cal, ~is_cal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20],
                    help="target leak probabilities to calibrate for")
    ap.add_argument("--out", default="split_conformal", help="basename under data/conformal/")
    args = ap.parse_args()

    s, d = load_scores(args)
    is_cal, is_test = split_by_episode(d["episode"], args.seed)
    s_cal, s_test = s[is_cal], s[is_test]

    print(f"[split] {s.size} strokes  ({s_cal.size} calibration / {s_test.size} holdout)")
    # TIGHT score: -inf on non-violating strokes, so report percentiles over the FULL vector (the
    # -inf mass is what makes the quantile a bound on P(leak)) but summarise magnitudes over the
    # finite entries only -- mean/max of a vector containing -inf is uninformative.
    _fin = s[np.isfinite(s)]
    print(f"[split] tight score s (m), {_fin.size} violating / {s.size} total:  "
          f"p50={np.median(_fin):+.5f}  p90={np.quantile(_fin, 0.90):+.5f}  "
          f"p99={np.quantile(_fin, 0.99):+.5f}  max={_fin.max():+.5f}")
    print(f"[split] marginal leak at delta=0: {float((s > 0).mean())*100:.3f}% "
          f"(this is why alpha >= that value yields a NEGATIVE delta)")

    rows = []
    for a in args.alphas:
        qhat = conformal_quantile(s_cal, a)
        rows.append({
            "alpha": a,
            "qhat_m": qhat,                                   # <-- the calibrated cushion, metres
            "target_coverage": 1.0 - a,
            "empirical_coverage_holdout": empirical_coverage(s_test, qhat),
            "leak_rate_holdout": leak_rate(s_test, qhat),
            "n_cal": int(s_cal.size), "n_test": int(s_test.size),
        })
        print(f"  alpha={a:<5} delta={qhat*100:7.3f} cm   holdout coverage="
              f"{rows[-1]['empirical_coverage_holdout']:.4f} (target {1-a:.4f})   "
              f"holdout leak={rows[-1]['leak_rate_holdout']:.4f}")

    save_result(args.out, {
        "method": "split_conformal_marginal",
        "residuals": str(args.residuals), "meta": d.get("meta", {}),
        "cell": args.cell, "cube_half": args.cube_half,
        "exclude_train": bool(args.exclude_train), "drop_padded": bool(args.drop_padded),
        "seed": args.seed,
        "score_summary": {"mean": float(s.mean()), "p50": float(np.median(s)),
                          "p90": float(np.quantile(s, 0.90)), "p99": float(np.quantile(s, 0.99)),
                          "max": float(s.max()), "n": int(s.size)},
        "results": rows,
    })


if __name__ == "__main__":
    main()
