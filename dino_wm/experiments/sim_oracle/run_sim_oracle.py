"""Run the SIM-ORACLE RRT over a benchmark (law_eval / crossing pair dirs) and score it with the
SAME metrics as the WM pipeline (planning.planning_metrics.build_eval_metrics). One scenario at a
time; the perfect-world-model safety/efficiency ceiling to compare against the WM-RRT sweep.

    # smoke-test the no-camera env first (must boot before building on it):
    /isaac-sim/python.sh - <<'PY'
    import sys; sys.path.insert(0, "experiments/sim_oracle")
    import numpy as np, torch; from phys_env import PhysGridEnv
    env = PhysGridEnv(num_envs=8); st = torch.load("data/isaaclab_stroke_1500/states.pth").float().numpy()[0,0]
    _, s = env.prepare(0, st); c0 = s[0,18:20].copy()
    out = env.roll_strokes(st, np.tile([c0[0]-0.12,c0[1],0.15,0.0],(8,1)).astype("float32"))
    print("moved", float(np.linalg.norm(out[0,18:20]-c0)))
    PY

    # then run the oracle (SMALL --max first: each scenario is minutes at full --max_samples):
    /isaac-sim/python.sh experiments/sim_oracle/run_sim_oracle.py \
        --benchmark data/law_eval_center --mode social --metric_cell 4 --max 2 --max_samples 128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for p in (str(_HERE), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import provenance

from phys_env import PhysGridEnv
from sim_rrt import SimOracleRRT
from legislation.reasoner import LegislativeReasoner
from legislation.grounding import gm
from planning.planning_metrics import build_eval_metrics
from probes.probe_cube_cells import CUBE_HALF

_CUBE_XY = slice(18, 20)


def _last_metrics(final_state, goal_state, goal_cell):
    """The evaluator's rollout metrics for ONE finished episode (build_eval_metrics reads these)."""
    succ = int(np.atleast_1d(gm.which_cell(final_state[None, _CUBE_XY]))[0]) == goal_cell
    return {"success": [bool(succ)], "cubes_correct": [int(succ)],
            "cube_l2": [float(np.linalg.norm(final_state[_CUBE_XY] - goal_state[_CUBE_XY]))],
            "state_dist": [float(np.linalg.norm(final_state - goal_state))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/law_eval_center",
                    help="dir with <init>_<goal> pair subdirs (default: the original center set; also crossing_set)")
    ap.add_argument("--template", default="data/isaaclab_stroke_1500",
                    help="WM training set: its actions.pth per-dim min/max == the stroke clamp the WM-RRT applies")
    ap.add_argument("--mode", default="social", choices=["social", "deviant", "off"])
    ap.add_argument("--db_path", default=None, help="legislation DB (null=geom1; legislation/laws/geometric_2.yaml)")
    ap.add_argument("--metric_cell", type=int, default=None, help="override; else read per-pair metadata")
    ap.add_argument("--max", type=int, default=None, help="scenarios PER PAIR (default all) -- keep SMALL, it's slow")
    ap.add_argument("--max_samples", type=int, default=512, help="RRT extend attempts/plan step (512=WM parity; lower=faster)")
    ap.add_argument("--max_iter", type=int, default=12)
    ap.add_argument("--stroke_max_steps", type=int, default=320,
                    help="PhysX substeps per stroke (per-round cost scales with this). Most strokes "
                    "finish well before 320; ~160 roughly halves per-round time. Too low truncates "
                    "strokes (partial push -> corrupted GT), so verify the cube still moves as expected.")
    ap.add_argument("--num_envs", type=int, default=64, help="RRT batch_size B (candidates per extend)")
    ap.add_argument("--patience", type=int, default=0,
                    help="convergence early-stop: bail a tree-build after this many rounds with NO "
                    "goal-dist improvement in any active scenario (0 = off; try 30-50). Big speedup on "
                    "the far-from-goal MPC steps that otherwise run all max_samples.")
    ap.add_argument("--batch_scenarios", type=int, default=1,
                    help="K: run K scenarios IN PARALLEL sharing one K*B-env PhysX batch (~Kx faster on "
                    "one GPU; memory is free at B=64). 1 = sequential. Try 8-32 (env count = K*B).")
    ap.add_argument("--steer_cone_deg", type=float, default=90.0,
                    help="push-direction cone around the bearing to target (90 = WM-RRT parity; 360 = uniform A/B)")
    ap.add_argument("--cushion", type=float, default=0.0,
                    help="keep-clear MARGIN delta (m) added to the illegal cell: the planner keeps the cube "
                    "footprint delta beyond the true cell (cube_half -> CUBE_HALF+delta in the Constraint), "
                    "while the METRIC still scores the true cell. delta=0 = no cushion (the razor-edge arm "
                    "where jitter/batch-noise/tight-cornering graze). Sweep {0,0.02,0.03,0.04,0.05} to find "
                    "the knee: max swept-abidance before success drops (too big => no legal route past cell 4).")
    ap.add_argument("--push_min", type=float, default=0.05, help="per-stroke cube-travel MIN (m); lower = finer")
    ap.add_argument("--push_max", type=float, default=0.09, help="per-stroke cube-travel MAX (m)")
    ap.add_argument("--no_clamp", action="store_true",
                    help="drop the training-range action clamp so finer strokes (push_min<0.05) are not "
                    "clipped back up. Use WITH --push_min 0.03 to test the granularity hypothesis; note this "
                    "leaves the WM's training distribution, so it's a CEILING probe, not a WM-comparison.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid_away_shift", type=float, default=None,
                    help="GEOMETRY INTERVENTION (m): override the robot-base away-from-grid shift for THIS "
                    "oracle run only, WITHOUT editing the shared cfg. None (default) = the cfg's current "
                    "shift (0.025 -> base x=-0.475). 0.0 = the pre-shift -0.45 geometry. base_x = -0.45 - shift. "
                    "Use to reproduce a pre-shift ceiling or A/B the shift's effect on reachability.")
    ap.add_argument("--out", default="results/jul22/oracle_sim",
                    help="output DIR: writes <out>/<pair>/scenario_<i>/eval_metrics.json (FULL per-eval "
                    "dict, eval_sweep layout -> drop-in for the diagnostics/plots) + <out>/summary.json. "
                    "No renders/videos: the oracle env is camera-off (state-only).")
    ap.add_argument("--resume", action="store_true",
                    help="skip (pair, scenario) slots already complete under --out and FOLD their saved "
                    "metrics back into the summary, so a restarted run still reports over the whole "
                    "benchmark. A scenario counts as complete only if eval_metrics.json parses AND (on a "
                    "sign lawset) normative_ledger.json exists -- a half-written scenario is redone. Off "
                    "by default: without it a rerun recomputes everything exactly as before.")
    # SIGN-DEPENDENT LAWSETS. Selecting one (e.g. full_lawset) is itself the switch that puts the oracle
    # on the per-step verdict path (SimOracleRRT.dynamic); a sign-free lawset keeps the legacy static
    # path, so every previously reported oracle run reproduces unchanged.
    ap.add_argument("--active_lawsets", default=None,
                    help="comma-separated law CATEGORIES to enforce (e.g. full_lawset). Default: the "
                         "db's own active_lawsets (geometric_laws -> legacy static-verdict oracle).")
    ap.add_argument("--color", default=None, choices=["white", "yellow", "green", "red"],
                    help="raw sign colour asserted from --frame onward (exogenous authority, fed from "
                         "ground truth: the oracle has no renderer and no sign probe).")
    ap.add_argument("--frame", type=int, default=1,
                    help="executed step at which the sign turns --color (before it: white). "
                         "Matches eval_sweep.py --frame/--color.")
    ap.add_argument("--yellow_cells", default="",
                    help="comma-separated yellow-cell ids for full_lawset (R5b/R7/R7b), e.g. 3,5. MUST "
                         "match the WM run's legislation.yellow_cells: without them in_yellow_cell is "
                         "never DERIVED, so the sign can never flip to green or red (plan.py:335).")
    ap.add_argument("--strict_escape", action="store_true",
                    help="escape grandfather covers ONLY the frame 0->1 segment, so a candidate that "
                         "leaves the keep-clear zone and dips back in is pruned. OMIT to match every "
                         "existing run (aug15/sign_color, results/final/*) -- the arms must agree on this "
                         "or an oracle-vs-WM gap measures the enforcement rule, not perception error.")
    ap.add_argument("--goal_bank", default=None,
                    help="goal-cell bank for the obligation channel (R5/R8 target swap). Omit to leave "
                         "obligations unenforced on the goal, matching a prohibition-only oracle.")
    args = ap.parse_args()

    bench = Path(args.benchmark).resolve()
    top = json.loads((bench / "metadata.json").read_text())
    pair_dirs = top.get("pair_dirs") or [p.name for p in sorted(bench.iterdir()) if (p / "states.pth").exists()]

    # Stroke clamp = the WM's training action range (preprocessor.action_min/max == actions.pth per-dim
    # min/max, isaaclab_grid_dset.py). Load it so the sim-oracle samples the IDENTICAL candidate set.
    _a = torch.load(Path(args.template) / "actions.pth").float().reshape(-1, 4)
    action_min = _a.min(dim=0).values.numpy(); action_max = _a.max(dim=0).values.numpy()

    if args.no_clamp:                                                   # finer-stroke probe: don't clip to training range
        action_min = action_max = None
    _lawsets = args.active_lawsets.split(",") if args.active_lawsets else None
    _rkw = {"active_lawsets": _lawsets} if _lawsets else {}
    reasoner = (LegislativeReasoner(db_path=args.db_path, **_rkw) if args.db_path
                else LegislativeReasoner(**_rkw))
    B = args.num_envs; K = max(1, args.batch_scenarios)
    env = PhysGridEnv(num_envs=K * B, device=args.device, stroke_max_steps=args.stroke_max_steps,
                      grid_away_shift=args.grid_away_shift)  # K*B envs

    # Sign-dependent lawsets need a per-step verdict, so hand the oracle the SAME Enforcement object the
    # WM agent uses -- with the probe stack swapped for sim ground truth. Reusing it (rather than
    # reimplementing) is what makes the two arms share the sign latch, swept taint, visited() history and
    # verdict semantics, so an oracle-vs-WM gap is perception error and not a harness difference.
    enforcement = sign_schedule = None
    if _lawsets and any(s.strip() != "geometric_laws" for s in _lawsets):
        from legislation.enforcement import LawEvaluator
        _rrt_holder = {}
        sign_schedule = (lambda step: (args.color if (args.color and step >= args.frame) else "white"))
        _yc = [int(c) for c in args.yellow_cells.split(",") if c.strip()]
        enforcement = LawEvaluator(
            reasoner, registry=None, cube_half=CUBE_HALF + args.cushion,
            base_facts=["cube"] + [f"yellow_cell({c})" for c in _yc],   # plan.py:335 -- static config facts
            strict_escape=args.strict_escape,
            perceive_fn=lambda ev, xy: _rrt_holder["rrt"]._gt_facts(ev, xy),
            # OCCUPANCY BRANCH ON (matches the legacy static path). The transit test alone is too weak
            # for a candidate that STARTS inside the cell: its escape grandfather reduces legality to
            # "the endpoint clears", so a plan may leave and re-enter freely. The grandfather is only
            # meant to excuse an INHERITED illegal state, not to license dipping in and out, so the
            # per-frame occupancy check has to stay. Filled in with the GT occ fn once rrt exists.
            constraint_probes={"cube_cells": None})
        if not _yc:
            print("[sim-oracle] WARNING: --yellow_cells is empty; in_yellow_cell can never be derived, "
                  "so R7/R7b cannot flip the sign and the run degenerates to R1/R2/R8/R10.", flush=True)
        print(f"[sim-oracle] lawsets={_lawsets} -> PER-STEP verdict; sign: white then "
              f"{args.color} from step {args.frame}", flush=True)

    rrt = SimOracleRRT(env, reasoner, mode=args.mode, batch_size=B,
                       max_samples=args.max_samples, action_min=action_min, action_max=action_max,
                       push_min=args.push_min, push_max=args.push_max, cube_half=CUBE_HALF + args.cushion,
                       steer_cone_deg=args.steer_cone_deg, patience=args.patience, device=args.device,
                       enforcement=enforcement, sign_schedule=sign_schedule,
                       goal_bank_path=args.goal_bank)
    if enforcement is not None:
        _rrt_holder["rrt"] = rrt
        enforcement.constraint_probes = {"cube_cells": rrt._occ}   # GT footprint-occupancy, not a probe
    print(f"[sim-oracle] cushion delta={args.cushion:.3f} (cube_half {CUBE_HALF:.3f}->{CUBE_HALF+args.cushion:.3f}) "
          f"| push=[{args.push_min},{args.push_max}] | clamp={'OFF' if args.no_clamp else 'training-range'}", flush=True)

    # Flatten all (pair, scenario) so the batched runner can pack K at a time (a chunk may span pairs --
    # fine, run_batch grounds goal cell + constraint per scenario).
    flat = []
    for pd in pair_dirs:
        pdir = bench / pd
        states = torch.load(pdir / "states.pth").float().numpy()            # (V,2,31) [init, goal]
        pmeta = json.loads((pdir / "metadata.json").read_text()) if (pdir / "metadata.json").exists() else {}
        mc = args.metric_cell if args.metric_cell is not None else pmeta.get("metric_cell", 4)
        V = states.shape[0] if args.max is None else min(args.max, states.shape[0])
        for i in range(V):
            flat.append((pd, i, mc, states[i, 0].copy(), states[i, 1].copy()))

    acc, records = {}, []

    # ---- RESUME ------------------------------------------------------------------------------
    # Drop (pair, scenario) slots already on disk AND fold their saved metrics back into acc/records.
    # The fold-back is the load-bearing half: summary.json below is computed from `acc`, so a resume
    # that only skipped work would report every rate over the RESUMED SUBSET -- a wrong denominator,
    # not a smaller sample. With the fold-back a resumed run and a from-scratch run summarise the same
    # population. Completeness requires eval_metrics.json to PARSE (a kill can truncate it) and, on a
    # sign lawset, the ledger to exist too -- _record writes them in that order, so a scenario with the
    # first and not the second died between the two writes and must be redone.
    _ACC_KEYS = ("success", "law_violated", "law_violated_swept", "law_violated_center",
                 "illegal_frame_frac", "n_steps", "path_efficiency", "optimal_path_len", "cube_l2")
    if args.resume and args.out:
        def _loaded(pd, i):
            d = Path(args.out) / pd / f"scenario_{i:03d}"
            if enforcement is not None and not (d / "normative_ledger.json").exists():
                return None
            try:
                return json.loads((d / "eval_metrics.json").read_text())
            except (OSError, json.JSONDecodeError):
                return None                            # missing or truncated -> redo
        todo, skipped = [], 0
        for row in flat:
            pd, i, mc = row[0], row[1], row[2]
            m = _loaded(pd, i)
            if m is None:
                todo.append(row)
                continue
            skipped += 1
            for kk in _ACC_KEYS:
                acc.setdefault(kk, []).extend(v for v in m.get(kk, []) if v is not None)
            records.append({"pair": pd, "scenario": i, "metric_cell": mc,
                            "success": m["success"][0], "law_violated": m["law_violated"][0],
                            "steps": m["n_steps"][0]})
        print(f"[sim-oracle] RESUME: {skipped} scenario(s) already complete under {args.out} "
              f"(folded into the summary), {len(todo)} left to run", flush=True)
        flat = todo

    print(f"[sim-oracle] {args.mode} | {len(flat)} scenarios ({len(pair_dirs)} pairs) | "
          f"K={K} x B={B} = {K * B} envs | max_samples={args.max_samples}", flush=True)

    def _record(pd, i, mc, goal_s, e_states, constraint, alen, eval_index=0):
        """Full per-eval eval_metrics.json (eval_sweep layout) + accumulate acc/records. Returns m.
        On a sign-dependent lawset also writes normative_ledger.json in the eval_sweep layout, without
        which none of the Q4 scorers (plot_sign_lawset / plot_sign_overlap / plot_sign_prepost) can read
        this arm -- every one of their metrics gates on the per-step sign held in the ledger."""
        gc = int(np.atleast_1d(gm.which_cell(goal_s[None, _CUBE_XY]))[0])
        m = build_eval_metrics(
            e_states=e_states[None], action_len=np.array([alen], dtype=float),
            last_metrics=_last_metrics(e_states[-1], goal_s, gc),
            constraint=constraint, scene_filter={}, metric_cell=mc,
            scene_offset=0, pool_size=1, n_evals=1, seed=0, goal_states=goal_s[None])
        sc_dir = Path(args.out) / pd / f"scenario_{i:03d}"; sc_dir.mkdir(parents=True, exist_ok=True)
        (sc_dir / "eval_metrics.json").write_text(json.dumps(m, indent=2))   # drop-in for the diagnostics
        if enforcement is not None:                                          # sign-dependent lawset only
            led = enforcement.ledger(eval_index)
            (sc_dir / "normative_ledger.json").write_text(json.dumps({"0": {
                "records": led.records, "signs": led.signs,
                "intrusions": led.intrusions(mc)}}, indent=2, default=float))
        for kk in ("success", "law_violated", "law_violated_swept", "law_violated_center",
                   "illegal_frame_frac", "n_steps", "path_efficiency", "optimal_path_len", "cube_l2"):
            acc.setdefault(kk, []).extend(v for v in m.get(kk, []) if v is not None)
        records.append({"pair": pd, "scenario": i, "metric_cell": mc, "success": m["success"][0],
                        "law_violated": m["law_violated"][0], "steps": m["n_steps"][0]})
        return m

    if K == 1:                                                              # sequential (env num_envs == B)
        for (pd, i, mc, init_s, goal_s) in flat:
            print(f"[sim-oracle] >>> {pd} #{i} planning...", flush=True)
            e_states, constraint, alen, timing = rrt.run(init_s, goal_s, max_iter=args.max_iter)
            m = _record(pd, i, mc, goal_s, e_states, constraint, alen)
            acc.setdefault("wall_s", []).append(timing["wall_s"])
            print(f"  {pd} #{i}: success={m['success'][0]} law_violated={m['law_violated'][0]} "
                  f"steps={m['n_steps'][0]} | {timing['wall_s']}s ({timing['n_extends']} extends, "
                  f"sim {timing['sim_s']}s)", flush=True)
    else:                                                                   # batched (env num_envs == K*B)
        for c0 in range(0, len(flat), K):
            chunk = flat[c0:c0 + K]; realK = len(chunk)
            if realK < K:                                                   # pad tail so num_envs == K*B holds
                chunk = chunk + [chunk[-1]] * (K - realK)
            init_states = np.stack([c[3] for c in chunk]); goal_states = np.stack([c[4] for c in chunk])
            print(f"[sim-oracle] >>> batch {c0}..{c0 + realK}/{len(flat)} (K={K}) planning...", flush=True)
            e_list, constraints, alens, timing = rrt.run_batch(init_states, goal_states, max_iter=args.max_iter)
            for j in range(realK):                                          # skip padded duplicates
                pd, i, mc, _init, goal_s = chunk[j]
                _record(pd, i, mc, goal_s, e_list[j], constraints[j], alens[j], eval_index=j)
            acc.setdefault("wall_s", []).append(timing["wall_s"])
            print(f"  batch done: {timing['wall_s']}s / {realK} scenarios "
                  f"({timing['wall_s'] / max(realK, 1):.1f}s each) | {timing['n_extends']} extends "
                  f"sim {timing['sim_s']}s ({100 * timing['sim_s'] / max(timing['wall_s'], 1e-6):.0f}%)", flush=True)
    env.close()

    n = len(acc.get("success", []))
    rate = lambda k: (1.0 - float(np.mean(acc[k]))) if acc.get(k) else None   # abidance = 1 - violated
    summary = {
        "mode": args.mode, "db_path": args.db_path, "n": n,
        "success_rate": float(np.mean(acc["success"])) if n else None,
        "law_abidance_rate": rate("law_violated"),
        "law_abidance_swept_rate": rate("law_violated_swept"),
        "law_abidance_center_rate": rate("law_violated_center"),
        "illegal_frame_frac_mean": float(np.mean(acc["illegal_frame_frac"])) if acc.get("illegal_frame_frac") else None,
        "path_efficiency_mean": float(np.mean(acc["path_efficiency"])) if acc.get("path_efficiency") else None,
        "n_steps_mean": float(np.mean(acc["n_steps"])) if acc.get("n_steps") else None,
        "max_samples": args.max_samples,
        "cushion": args.cushion,
        "grid_away_shift": env.grid_away_shift,   # geometry provenance (see the silent-cfg-shift bug)
        "robot_base_x": env.robot_base_x,
    }
    print("[sim-oracle] SUMMARY:", json.dumps(summary, indent=2))
    if args.out:
        outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
        # sort so a --resume run and a from-scratch run emit byte-identical record ORDER (resumed
        # slots are folded in first, ahead of whatever this process actually ran).
        records.sort(key=lambda r: (str(r["pair"]), int(r["scenario"])))
        (outdir / "summary.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))
        provenance.write(outdir, __file__, args=args)
        print(f"[sim-oracle] wrote {outdir}/summary.json + per-scenario eval_metrics.json under {outdir}/")


if __name__ == "__main__":
    main()
