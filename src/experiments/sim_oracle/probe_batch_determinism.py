"""Is env.roll_strokes reproducible, and does it depend on BATCH COMPOSITION?

Why: sim_rrt evaluates a candidate stroke inside a block of B DIFFERENT strokes (tree build,
_pick_from_block prunes on that rolled endpoint), then rolls the winner a SECOND time replicated B
times to execute it (sim_rrt.py:498-505, "B copies identical"). If those two rolls disagree, the
planner prunes on one endpoint and executes another -- which is what put 58/58 committed strokes of
results/no_yaw/sign_change/oracle into a cell the verdict forbade.

Three trials, same state throughout:
  A  roll a block of B varied strokes                        -> EA
  B  roll the IDENTICAL block again                          -> EB   (pure repeatability)
  C  for selected j, roll strokes[j] replicated B times       -> EC   (the COMMIT layout)

  EA vs EB  differ  -> PhysX is not reproducible even call-to-call
  EA[j] vs EC       differ  -> the result depends on what ELSE is in the batch  (the re-roll bug)
  EC[0] vs EC[j]    differ  -> it depends on the env INDEX within the batch

Read-only w.r.t. results/. Physics only, no render.
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
    ap.add_argument("--pair", default="0_8")
    ap.add_argument("--scenario", type=int, default=0)
    ap.add_argument("--template", default="data/isaaclab_stroke_5k_no_yaw_merged")
    ap.add_argument("--B", type=int, default=64, help="block size (== sim_rrt num_envs)")
    ap.add_argument("--probe_j", type=int, nargs="*", default=[0, 7, 31, 63])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stroke_max_steps", type=int, default=320)
    ap.add_argument("--cost", action="store_true", help="time the candidate fix (reset before each roll)")
    ap.add_argument("--cost_iters", type=int, default=10)
    ap.add_argument("--variants", action="store_true", help="isolate which reset step clears the residue")
    args = ap.parse_args()

    B = args.B
    st = torch.load(Path(args.benchmark) / args.pair / "states.pth").float().numpy()
    state = st[args.scenario, 0].astype(np.float32)                # the real spawn state

    a = torch.load(Path(args.template) / "actions.pth").float().reshape(-1, 4)
    lo = a.min(dim=0).values.numpy(); hi = a.max(dim=0).values.numpy()

    rng = np.random.RandomState(0)
    c0 = state[_CUBE_XY]
    strokes = np.empty((B, 4), np.float32)
    strokes[:, 0] = c0[0] + rng.uniform(-0.09, 0.09, B)            # start near the cube
    strokes[:, 1] = c0[1] + rng.uniform(-0.09, 0.09, B)
    ang = rng.uniform(-np.pi, np.pi, B); mag = rng.uniform(0.05, 0.09, B)
    strokes[:, 2] = mag * np.cos(ang); strokes[:, 3] = mag * np.sin(ang)
    strokes = np.clip(strokes, lo, hi).astype(np.float32)

    env = PhysGridEnv(num_envs=B, device=args.device, stroke_max_steps=args.stroke_max_steps)
    env.prepare(0, state)                    # gym reset + set_init_state (run_sim_oracle does this too)
    states_B = np.tile(state, (B, 1)).astype(np.float32)

    EA = env.roll_strokes(states_B, strokes).copy()
    EB = env.roll_strokes(states_B, strokes).copy()

    dAB = np.linalg.norm(EA[:, _CUBE_XY] - EB[:, _CUBE_XY], axis=1)
    print("\n================ RESULTS ================")
    print(f"state = {args.benchmark}/{args.pair} scenario {args.scenario}, B={B}")
    print(f"\nA vs B  (same block rolled twice)  -- pure repeatability")
    print(f"   cube xy delta (m): max={dAB.max():.3e}  mean={dAB.mean():.3e}  "
          f"n_exact={int((dAB == 0).sum())}/{B}")

    print(f"\nA vs C  (same stroke, but alone-in-block vs among-B-different) -- the COMMIT re-roll")
    print(f"   {'j':>4} {'|EA[j]-EC[j]|':>16} {'|EA[j]-EC[0]|':>16} {'spread within EC':>18}")
    worst = 0.0
    for j in args.probe_j:
        if j >= B:
            continue
        EC = env.roll_strokes(states_B, np.tile(strokes[j], (B, 1)).astype(np.float32)).copy()
        d_jj = float(np.linalg.norm(EA[j, _CUBE_XY] - EC[j, _CUBE_XY]))
        d_j0 = float(np.linalg.norm(EA[j, _CUBE_XY] - EC[0, _CUBE_XY]))
        spread = float(np.linalg.norm(EC[:, _CUBE_XY] - EC[0, _CUBE_XY], axis=1).max())
        worst = max(worst, d_j0)
        print(f"   {j:>4} {d_jj:16.3e} {d_j0:16.3e} {spread:18.3e}")

    # ---- WHY is it non-reproducible: hidden carried-over state, or true nondeterminism? ----------
    # roll_strokes restores the 31-D state every call, but if something NOT in those 31 dims survives
    # (solver/contact state, controller internals, joint velocities), call N+1 starts from a different
    # hidden condition than call N even with identical arguments -> looks stochastic, but is FIXABLE.
    # prepare() does a full gym reset + set_init_state. So:
    #   E1 vs E3 (each taken immediately after prepare)  equal -> prepare clears it => HIDDEN STATE
    #                                                    differ -> TRUE nondeterminism (GPU/solver)
    #   E1 vs E2 (2nd roll in the same sequence)         the drift a repeated call accumulates
    print("\n===== WHY: hidden carried-over state vs true nondeterminism =====")
    env.prepare(0, state); E1 = env.roll_strokes(states_B, strokes).copy()
    E2 = env.roll_strokes(states_B, strokes).copy()
    env.prepare(0, state); E3 = env.roll_strokes(states_B, strokes).copy()
    E4 = env.roll_strokes(states_B, strokes).copy()
    def dd(X, Y):
        d = np.linalg.norm(X[:, _CUBE_XY] - Y[:, _CUBE_XY], axis=1)
        return f"max={d.max():.3e} mean={d.mean():.3e} n_exact={int((d == 0).sum())}/{B}"
    print(f"   E1 vs E3  (both right after prepare()) : {dd(E1, E3)}")
    print(f"   E2 vs E4  (both 2nd-in-sequence)       : {dd(E2, E4)}")
    print(f"   E1 vs E2  (same sequence, no reset)    : {dd(E1, E2)}")
    # prepare() does TWO things: reseeds torch/np AND does a gym reset + set_init_state. If the drift
    # is RNG consumed inside execute_stroke, a bare RESEED (no reset) is enough to restore it; if it is
    # carried-over sim state, only the reset is. Separate them -- the fix is completely different.
    torch.manual_seed(0); np.random.seed(0)
    E5 = env.roll_strokes(states_B, strokes).copy()                  # reseed ONLY, no gym reset
    torch.manual_seed(0); np.random.seed(0)
    E6 = env.roll_strokes(states_B, strokes).copy()
    print(f"\n   E5 vs E6  (reseed only, NO gym reset)  : {dd(E5, E6)}")
    print(f"   E1 vs E5  (reset+reseed vs reseed only): {dd(E1, E5)}")

    d13 = np.linalg.norm(E1[:, _CUBE_XY] - E3[:, _CUBE_XY], axis=1)
    d12 = np.linalg.norm(E1[:, _CUBE_XY] - E2[:, _CUBE_XY], axis=1)
    d56 = np.linalg.norm(E5[:, _CUBE_XY] - E6[:, _CUBE_XY], axis=1)
    print()
    if d13.max() < 0.01 * max(d12.max(), 1e-12):
        print(f"   => matched-position rolls agree to {1000*d13.max():.3f} mm while sequence-position")
        print(f"      changes the answer by {1000*d12.max():.1f} mm -- {d12.max()/max(d13.max(),1e-12):.0f}x.")
        print("      This is CARRIED-OVER STATE BETWEEN CALLS, not solver nondeterminism.")
        if d56.max() < 0.01 * max(d12.max(), 1e-12):
            print("      A bare RESEED is enough -> the carrier is RNG consumed inside execute_stroke.")
        else:
            print("      A reseed is NOT enough -> the carrier is SIM state that only a reset clears.")
    else:
        print("   => differs even at matched sequence position: TRUE nondeterminism (GPU solver order).")

    # ---- COST + CORRECTNESS of the candidate fix -------------------------------------------------
    # set_init_state (grid_wrapper_single.py:538-544) = self._env.reset() + _write_state +
    # _materialize_state. roll_strokes does the last two but NOT the reset. So the minimal fix is to
    # add the reset to roll_strokes. Two questions: does it actually restore reproducibility, and what
    # does it cost -- the tree build is ~100% of oracle wall time (sim_rrt.py:55), and it is one
    # roll_strokes per extend round, so a per-roll multiplier is ~the multiplier on total runtime.
    if args.cost:
        import time as _time

        def _sync():
            torch.cuda.synchronize()

        def _timeit(fn, n):
            fn(); _sync()                                       # warm up, not counted
            t0 = _time.perf_counter()
            for _ in range(n):
                fn()
            _sync()
            return (_time.perf_counter() - t0) / n

        N = args.cost_iters
        t_roll = _timeit(lambda: env.roll_strokes(states_B, strokes), N)
        t_reset = _timeit(lambda: env._env.reset(), N)
        t_both = _timeit(lambda: (env._env.reset(), env.roll_strokes(states_B, strokes)), N)
        print(f"\n===== COST of resetting before every roll (B={B}, n={N}) =====")
        print(f"   roll_strokes alone      : {1000*t_roll:9.2f} ms")
        print(f"   _env.reset() alone      : {1000*t_reset:9.2f} ms")
        print(f"   reset + roll_strokes    : {1000*t_both:9.2f} ms")
        print(f"   => multiplier on per-roll cost: {t_both/t_roll:.3f}x  "
              f"({100*(t_both/t_roll-1):+.1f}% wall on a sim-bound run)")

        # does it actually fix it?
        env._env.reset(); F1 = env.roll_strokes(states_B, strokes).copy()
        env._env.reset(); F2 = env.roll_strokes(states_B, strokes).copy()
        dF = np.linalg.norm(F1[:, _CUBE_XY] - F2[:, _CUBE_XY], axis=1)
        print(f"\n   reset-before-each-roll, same args twice: max={dF.max():.3e} mean={dF.mean():.3e} "
              f"n_exact={int((dF == 0).sum())}/{B}")
        print(f"   (unfixed, same comparison was: max={d12.max():.3e} n_exact={int((d12 == 0).sum())}/{B})")
        if dF.max() <= max(d13.max(), 1e-12):
            print("   => the reset RESTORES reproducibility to the solver-noise floor. Fix confirmed.")
        else:
            print(f"   => reset is NOT sufficient: still {1000*dF.max():.2f} mm. Residue lives elsewhere.")

    # ---- WHICH component clears the residue? ------------------------------------------------------
    # prepare() = _env.reset() + _write_state + _materialize_state + home capture, and it DOES restore
    # reproducibility. A bare _env.reset() does NOT. So isolate each piece: run every variant twice
    # with identical args and see which pair agrees.
    if args.variants:
        def _pair(pre, label):
            pre(); X = env.roll_strokes(states_B, strokes).copy()
            pre(); Y = env.roll_strokes(states_B, strokes).copy()
            d = np.linalg.norm(X[:, _CUBE_XY] - Y[:, _CUBE_XY], axis=1)
            print(f"   {label:52} max={1000*d.max():8.3f} mm  n_exact={int((d == 0).sum()):>2}/{B}")
            return d.max()

        print("\n===== WHICH step clears the carried-over residue? =====")
        _pair(lambda: None, "nothing (current roll_strokes)")
        _pair(lambda: env._env.reset(), "_env.reset() only")
        _pair(lambda: (env._write_state(state), env._materialize_state()),
              "_write_state + _materialize_state (extra settle)")
        _pair(lambda: (env._env.reset(), env._write_state(state), env._materialize_state()),
              "reset + write + materialize (== prepare, no home)")
        _pair(lambda: (env._materialize_state(), env._materialize_state()),
              "2x _materialize_state (extra sim steps only)")
        _pair(lambda: env.prepare(0, state), "prepare() [reference: known good]")

    print("\n----------------- verdict -----------------")
    if dAB.max() > 0:
        print(f"  PhysX is NOT call-to-call reproducible (max {dAB.max():.3e} m on an identical block).")
    else:
        print("  PhysX IS call-to-call reproducible on an identical block (bitwise).")
    if worst > 1e-9:
        print(f"  Result DEPENDS ON BATCH COMPOSITION: up to {worst:.3e} m ({1000*worst:.2f} mm) for the")
        print("  SAME (state, stroke) rolled alone-in-block vs among B different strokes.")
        print("  -> the tree-build endpoint and the committed endpoint are different physics.")
    else:
        print("  Result is INDEPENDENT of batch composition -- the re-roll is not the divergence source.")
    env.close()


if __name__ == "__main__":
    main()
