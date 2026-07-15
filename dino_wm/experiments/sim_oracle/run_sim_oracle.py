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

from phys_env import PhysGridEnv
from sim_rrt import SimOracleRRT
from legislation.reasoner import LegislativeReasoner
from legislation.grounding import gm
from planning.planning_metrics import build_eval_metrics

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
    ap.add_argument("--num_envs", type=int, default=64, help="== RRT batch_size (candidates per extend)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="write per-scenario + summary JSON here")
    args = ap.parse_args()

    bench = Path(args.benchmark).resolve()
    top = json.loads((bench / "metadata.json").read_text())
    pair_dirs = top.get("pair_dirs") or [p.name for p in sorted(bench.iterdir()) if (p / "states.pth").exists()]

    # Stroke clamp = the WM's training action range (preprocessor.action_min/max == actions.pth per-dim
    # min/max, isaaclab_grid_dset.py). Load it so the sim-oracle samples the IDENTICAL candidate set.
    _a = torch.load(Path(args.template) / "actions.pth").float().reshape(-1, 4)
    action_min = _a.min(dim=0).values.numpy(); action_max = _a.max(dim=0).values.numpy()

    reasoner = LegislativeReasoner(db_path=args.db_path) if args.db_path else LegislativeReasoner()
    env = PhysGridEnv(num_envs=args.num_envs, device=args.device)
    rrt = SimOracleRRT(env, reasoner, mode=args.mode, batch_size=args.num_envs,
                       max_samples=args.max_samples, action_min=action_min, action_max=action_max,
                       device=args.device)

    acc, records = {}, []
    print(f"[sim-oracle] {args.mode} | {len(pair_dirs)} pairs | max/pair={args.max} | max_samples={args.max_samples}")
    for pd in pair_dirs:
        pdir = bench / pd
        states = torch.load(pdir / "states.pth").float().numpy()            # (V,2,31) [init, goal]
        pmeta = json.loads((pdir / "metadata.json").read_text()) if (pdir / "metadata.json").exists() else {}
        metric_cell = args.metric_cell if args.metric_cell is not None else pmeta.get("metric_cell", 4)
        V = states.shape[0] if args.max is None else min(args.max, states.shape[0])
        for i in range(V):
            init_s, goal_s = states[i, 0], states[i, 1]
            goal_cell = int(np.atleast_1d(gm.which_cell(goal_s[None, _CUBE_XY]))[0])
            e_states, constraint, alen = rrt.run(init_s, goal_s, max_iter=args.max_iter)
            m = build_eval_metrics(
                e_states=e_states[None], action_len=np.array([alen], dtype=float),
                last_metrics=_last_metrics(e_states[-1], goal_s, goal_cell),
                constraint=constraint, scene_filter={}, metric_cell=metric_cell,
                scene_offset=0, pool_size=1, n_evals=1, seed=0, goal_states=goal_s[None])
            for k in ("success", "law_violated", "law_violated_swept", "law_violated_center",
                      "illegal_frame_frac", "n_steps", "path_efficiency", "optimal_path_len", "cube_l2"):
                acc.setdefault(k, []).extend(v for v in m.get(k, []) if v is not None)
            records.append({"pair": pd, "scenario": i, "metric_cell": metric_cell,
                            "success": m["success"][0], "law_violated": m["law_violated"][0],
                            "steps": m["n_steps"][0]})
            print(f"  {pd} #{i}: success={m['success'][0]} law_violated={m['law_violated'][0]} steps={m['n_steps'][0]}")
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
    }
    print("[sim-oracle] SUMMARY:", json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "records": records}, indent=2))
        print(f"[sim-oracle] wrote {args.out}")


if __name__ == "__main__":
    main()
