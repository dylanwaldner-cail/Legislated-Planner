"""Self-contained validation of the conformal machinery. Run this instead of reading the paper.

    python scripts/conformal/test_conformal.py

WHY THIS FILE EXISTS
--------------------
The coverage guarantee behind `conformal_quantile` is a one-page theorem (Lei, G'Sell,
Rinaldo, Tibshirani, Wasserman, JASA 2018, Sec 2.2; the underlying exchangeability lemma is
Lemma 1 of Tibshirani, Barber, Candes, Ramdas, NeurIPS 2019). But reading the theorem only
confirms the FORMULA is right -- it says nothing about whether this repo implements it
correctly. That is what these tests are for.

The guarantee is unusually easy to check empirically, because it is distribution-free: for ANY
exchangeable score distribution, repeated trials must give

    P(s_new <= qhat)  >=  1 - alpha

So we throw deliberately nasty distributions at it -- heavy-tailed, discrete, massively tied,
degenerate -- and confirm coverage never falls below the target. If someone breaks the
implementation later, these fail.

WHAT EACH TEST PROVES
---------------------
  test_coverage_guarantee    the headline: empirical coverage >= 1-alpha across distributions
  test_ties_lower_bound      coverage still holds with a huge mass of tied -inf scores (which
                             the 'tight' nonconformity score produces by construction); note
                             only the LOWER bound is claimed -- the usual upper bound
                             1-alpha+1/(n+1) needs distinct scores and does NOT apply to us
  test_small_n_returns_inf   refuses to certify a level the sample size cannot support
  test_geometry_vs_bruteforce  the Minkowski shortcut matches an explicit swept-square sweep
  test_penetration_exact     the exact penetration formula matches brute force
  test_tight_score_identity  {s > delta} is EXACTLY the leak event -- the claim the whole
                             cushion argument rests on
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probes.probe_cube_cells import CUBE_HALF                       # noqa: E402
from scripts.conformal.common import (                              # noqa: E402
    cell_box, conformal_quantile, nonconformity, swept_penetration,
    swept_signed_distance,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------------------
def test_coverage_guarantee() -> None:
    """The headline property, across deliberately awkward distributions."""
    print("\ntest_coverage_guarantee (empirical coverage must be >= 1-alpha)")
    rng = np.random.RandomState(0)
    dists = {
        "gaussian": lambda n: rng.randn(n),
        "heavy_tail_cauchy": lambda n: rng.standard_cauchy(n),      # our WM error is heavy-tailed
        "lognormal": lambda n: rng.lognormal(0, 2, n),
        "discrete_5_values": lambda n: rng.choice([0., 1., 2., 3., 4.], n),
        "constant": lambda n: np.zeros(n),                          # fully degenerate
    }
    for alpha in (0.01, 0.05, 0.20):
        for dname, draw in dists.items():
            n, trials, covered = 200, 2000, 0
            for _ in range(trials):
                s = draw(n + 1)
                qhat = conformal_quantile(s[:n], alpha)             # calibrate on n
                covered += int(s[n] <= qhat)                        # test the (n+1)-th
            cov = covered / trials
            # Monte-Carlo slack: 4 sigma of a binomial proportion at this trial count.
            tol = 4.0 * np.sqrt(max(cov, 1e-9) * (1 - cov) / trials)
            check(f"alpha={alpha:<5} {dname:<18} coverage={cov:.4f}",
                  cov >= (1 - alpha) - tol, f"target >= {1-alpha:.4f}")


def test_ties_lower_bound() -> None:
    """The 'tight' score puts a large mass at -inf; coverage must still hold."""
    print("\ntest_ties_lower_bound (massive -inf ties, as produced by the tight score)")
    rng = np.random.RandomState(1)
    for viol_rate in (0.01, 0.10, 0.50):
        for alpha in (0.01, 0.05, 0.20):
            n, trials, covered = 500, 2000, 0
            for _ in range(trials):
                s = np.where(rng.rand(n + 1) < viol_rate, rng.randn(n + 1), -np.inf)
                qhat = conformal_quantile(s[:n], alpha)
                covered += int(s[n] <= qhat)
            cov = covered / trials
            tol = 4.0 * np.sqrt(max(cov, 1e-9) * (1 - cov) / trials)
            check(f"viol_rate={viol_rate:<5} alpha={alpha:<5} coverage={cov:.4f}",
                  cov >= (1 - alpha) - tol, f"target >= {1-alpha:.4f}")


def test_small_n_returns_inf() -> None:
    """Refuse to certify a level the sample cannot support, rather than silently under-covering."""
    print("\ntest_small_n_returns_inf")
    # ceil((n+1)(1-alpha)) > n  =>  cannot certify. For alpha=0.05 that means n < 19.
    check("n=5, alpha=0.05 -> inf", conformal_quantile(np.arange(5.0), 0.05) == np.inf)
    check("n=18, alpha=0.05 -> inf", conformal_quantile(np.arange(18.0), 0.05) == np.inf)
    check("n=19, alpha=0.05 -> finite", np.isfinite(conformal_quantile(np.arange(19.0), 0.05)))
    check("n=0 -> inf", conformal_quantile(np.array([]), 0.05) == np.inf)
    # exact index check: 1..100 at alpha=0.05 -> ceil(101*0.95)=96 -> the 96th smallest = 96
    check("index is ceil((n+1)(1-alpha))",
          conformal_quantile(np.arange(1, 101.0), 0.05) == 96.0,
          f"got {conformal_quantile(np.arange(1, 101.0), 0.05)}")


def _brute_swept_sdf(p, q, cell, cube_half, n_off=25, n_t=400):
    """Explicitly sweep the cube square along the segment and measure to the RAW cell box."""
    centre, half = cell_box(cell)
    off = np.array([[dx, dy]
                    for dx in np.linspace(-cube_half, cube_half, n_off)
                    for dy in np.linspace(-cube_half, cube_half, n_off)])
    ts = np.linspace(0, 1, n_t)
    pts = (np.asarray(p) + ts[:, None] * (np.asarray(q) - np.asarray(p)))[:, None, :] + off[None]
    d = np.abs(pts - centre) - half
    return (np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)).min()


def test_geometry_vs_bruteforce() -> None:
    """Separation must be exact; the overlap PREDICATE must always agree."""
    print("\ntest_geometry_vs_bruteforce")
    rng = np.random.RandomState(2)
    sign_ok, sep_worst = 0, 0.0
    N = 300
    for _ in range(N):
        p = rng.uniform(-0.22, 0.22, 2)
        q = p + rng.uniform(-0.09, 0.09, 2)
        fast = swept_signed_distance([p], [q], 4, CUBE_HALF)[0]
        brute = _brute_swept_sdf(p, q, 4, CUBE_HALF)
        sign_ok += int((fast < 0) == (brute < 0))
        if fast >= 0 and brute >= 0:
            sep_worst = max(sep_worst, abs(fast - brute))
    check(f"overlap predicate agrees {sign_ok}/{N}", sign_ok == N)
    check(f"separation exact (worst {sep_worst:.2e} m)", sep_worst < 1e-6)


def test_penetration_exact() -> None:
    """swept_penetration must match the brute-force swept square, within ITS grid resolution."""
    print("\ntest_penetration_exact")
    rng = np.random.RandomState(3)
    worst, n_overlap = 0.0, 0
    for _ in range(300):
        p = rng.uniform(-0.22, 0.22, 2)
        q = p + rng.uniform(-0.09, 0.09, 2)
        fast = swept_penetration([p], [q], 4, CUBE_HALF)[0]
        brute = max(0.0, -_brute_swept_sdf(p, q, 4, CUBE_HALF))
        worst = max(worst, abs(fast - brute))
        n_overlap += int(fast > 0)
    # The brute force samples the cube on a 25x25 grid (~3.75mm spacing), so it is ITSELF only
    # accurate to about half a step. Anything under 2mm here is the reference's error, not ours.
    check(f"penetration matches brute (worst {worst:.2e} m, {n_overlap} overlapping)",
          worst < 2e-3)


def test_tight_score_identity() -> None:
    """{s > delta} must be EXACTLY the leak event {d_bel > delta AND d_true < 0}.

    This is the claim the entire cushion argument rests on: if it holds, then bounding
    P(s > delta) IS bounding the leak probability, with no slack and no severity weighting.
    """
    print("\ntest_tight_score_identity")
    rng = np.random.RandomState(4)
    n = 4000
    gt_s = rng.uniform(-0.22, 0.22, (n, 2))
    gt_e = gt_s + rng.uniform(-0.09, 0.09, (n, 2))
    # believed = truth plus a perception error of realistic magnitude
    pb_s = gt_s + rng.normal(0, 0.01, (n, 2))
    pb_e = gt_e + rng.normal(0, 0.03, (n, 2))

    s = nonconformity(pb_s, pb_e, gt_s, gt_e, cell=4, cube_half=CUBE_HALF, mode="tight")
    d_bel = swept_signed_distance(pb_s, pb_e, 4, CUBE_HALF)
    d_true = swept_signed_distance(gt_s, gt_e, 4, CUBE_HALF)

    ok = True
    for delta in (0.0, 0.01, 0.02, 0.03, 0.05):
        leak = (d_bel > delta) & (d_true < 0)
        ok &= bool(np.array_equal(s > delta, leak))
    check("{s > delta} == leak event, for delta in {0,.01,.02,.03,.05}", ok)

    # And the 'gap' variant must be strictly more conservative (never misses a leak the tight
    # score catches), which is what justifies calling it the looser of the two.
    s_gap = nonconformity(pb_s, pb_e, gt_s, gt_e, cell=4, cube_half=CUBE_HALF, mode="gap")
    viol = d_true < 0
    check("gap score >= tight score on violating strokes",
          bool(np.all(s_gap[viol] >= s[viol] - 1e-12)))


if __name__ == "__main__":
    test_coverage_guarantee()
    test_ties_lower_bound()
    test_small_n_returns_inf()
    test_geometry_vs_bruteforce()
    test_penetration_exact()
    test_tight_score_identity()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all conformal checks passed")
