"""How fat is the tail of env.roll_strokes' run-to-run divergence?

After the residue fix in phys_env.roll_strokes, a single (state, stroke) rolled twice agreed to
0.26mm mean / 3.5mm max -- but that was 64 strokes from ONE spawn state. The smoke test then threw a
43mm divergence (6_2/scenario_000), which that floor cannot explain. So: sweep many spawn states x
many strokes and measure the DISTRIBUTION of |roll_1 - roll_2| for identical inputs.

Why it matters: it decides whether the sim-oracle should commit its planned rollout (_Node.state) or
keep re-rolling. A THIN tail means the residual divergence is harness noise and committing the plan
is cleanup. A FAT tail means it is real contact sensitivity -- the chosen stroke genuinely has a
violating tail -- and the oracle should stay exposed to it, because that is exactly what the cushion
(delta) sweep exists to measure. Committing the plan would define that phenomenon out of existence.

Reported two ways, because the denominator is a real choice:
  MARGINAL          -- every sampled stroke, including ones that miss the cube entirely (those are
                       trivially reproducible: the cube never moves, so they deflate the tail).
  GIVEN CONTACT     -- only strokes that actually moved the cube (> --contact_mm).
Both are printed; neither is privileged here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phys_env import PhysGridEnv                                  # noqa: E402

_CUBE_XY = slice(18, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/law_eval_center_no_yaw_400")
    ap.add_argument("--template", default="data/isaaclab_stroke_5k_no_yaw_merged")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--n_states", type=int, default=20, help="spawn states swept (2 rolls each)")
    ap.add_argument("--contact_mm", type=float, default=1.0, help="cube displacement counting as contact")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stroke_max_steps", type=int, default=320)
    args = ap.parse_args()

    B = args.B
    bench = Path(args.benchmark)
    pairs = sorted(p.name for p in bench.iterdir() if (p / "states.pth").exists())
    states = []
    for i in range(args.n_states):                       # round-robin across pairs, varied scenarios
        pd = pairs[i % len(pairs)]
        s = torch.load(bench / pd / "states.pth").float().numpy()
        states.append((pd, i // len(pairs), s[i // len(pairs), 0].astype(np.float32)))

    a = torch.load(Path(args.template) / "actions.pth").float().reshape(-1, 4)
    lo = a.min(dim=0).values.numpy(); hi = a.max(dim=0).values.numpy()

    env = PhysGridEnv(num_envs=B, device=args.device, stroke_max_steps=args.stroke_max_steps)
    env.prepare(0, states[0][2])

    D, DISP = [], []
    for k, (pd, sc, state) in enumerate(states):
        rng = np.random.RandomState(1000 + k)
        c0 = state[_CUBE_XY]
        strokes = np.empty((B, 4), np.float32)
        strokes[:, 0] = c0[0] + rng.uniform(-0.09, 0.09, B)
        strokes[:, 1] = c0[1] + rng.uniform(-0.09, 0.09, B)
        ang = rng.uniform(-np.pi, np.pi, B); mag = rng.uniform(0.05, 0.09, B)
        strokes[:, 2] = mag * np.cos(ang); strokes[:, 3] = mag * np.sin(ang)
        strokes = np.clip(strokes, lo, hi).astype(np.float32)

        sB = np.tile(state, (B, 1)).astype(np.float32)
        E1 = env.roll_strokes(sB, strokes).copy()
        E2 = env.roll_strokes(sB, strokes).copy()
        d = np.linalg.norm(E1[:, _CUBE_XY] - E2[:, _CUBE_XY], axis=1)
        disp = np.linalg.norm(E1[:, _CUBE_XY] - c0, axis=1)
        D.append(d); DISP.append(disp)
        print(f"  [{k+1}/{len(states)}] {pd} sc{sc}: max={1000*d.max():7.2f} mm  "
              f"mean={1000*d.mean():6.3f} mm  n_exact={int((d == 0).sum()):>2}/{B}  "
              f"moved={int((disp > args.contact_mm/1000).sum())}/{B}", flush=True)

    D = np.concatenate(D); DISP = np.concatenate(DISP)
    contact = DISP > args.contact_mm / 1000.0

    def report(x, label):
        if not len(x):
            print(f"  {label}: (empty)"); return
        x = 1000 * np.asarray(x)                                   # mm
        qs = [50, 90, 95, 99, 99.9, 100]
        print(f"  {label}  n={len(x)}  mean={x.mean():.3f} mm  exact={int((x == 0).sum())} "
              f"({100*(x == 0).mean():.1f}%)")
        print("     " + "  ".join(f"p{q}={np.percentile(x, q):.2f}" for q in qs))
        for thr in (1, 5, 10, 20, 43):
            print(f"     frac > {thr:>2} mm : {(x > thr).mean():.4f}", end="")
        print()

    print("\n================ ROLL-TO-ROLL DIVERGENCE, identical inputs ================")
    report(D, "MARGINAL      (all strokes)")
    print()
    report(D[contact], f"GIVEN CONTACT (cube moved > {args.contact_mm} mm)")
    print(f"\n  contact rate: {contact.mean():.3f}  ({int(contact.sum())}/{len(D)})")
    print(f"\n  for scale: the smoke-test outlier was 43 mm; WM pred err mean is 33.5 mm;")
    print(f"             single-state floor measured earlier was mean 0.26 mm / max 3.5 mm.")
    env.close()


if __name__ == "__main__":
    main()
