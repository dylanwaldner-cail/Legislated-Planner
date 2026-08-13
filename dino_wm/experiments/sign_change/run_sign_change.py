"""Sign-change experiment driver — does the agent RE-PERCEIVE and ADAPT when the stop-sign flips mid-rollout?

Runs the sign-conditioned lawset over a benchmark, flipping the physical sign at MPC step --frame to each
--colors value, and compares agent behavior across conditions on the SAME scenarios (same seed). The
adaptation is the DELTA vs the WHITE baseline:

  * WHITE (baseline, no flip): no sign law fires -> R2 `no_center_cell` holds -> social agent DETOURS.
  * ->GREEN : green PERMITS the center; R4 > R2 by superiority -> the prohibition is DEFEATED -> the agent
              may now CROSS cell 4.                                    (defeasibility / superiority demo)
  * ->YELLOW: the DDL-DRIVEN flip. On reaching a yellow cell (R7/R7b), the reasoner concludes GREEN if the
              trajectory is clean (permit) or RED if it already visited(4) (freeze) -- and the effective
              sign is RENDERED BACK to the env each step so the decision is physical and PERSISTS.
  * ->RED   : red obliges HALT (prohibition on `moving`) -> agent stops.  (state-dependent obligation)

INTERPRETATION (read carefully): the raw center-entry / law-abidance columns are scored on `metric_cell`
(default 4) against the *unconditional* prohibition. So after a GREEN flip, center-entry going UP is the
CORRECT adaptation (green makes it legal), NOT a regression. This driver reports the raw numbers; the
adaptation is the change relative to the white baseline. (A norm-aware "did it obey the EFFECTIVE law"
score is future work — see todo.md.)

=========================== WIRING (all now in place, 2026-08-03) =============================
  1. sign_color probe ENABLED in probes.yaml (balanced_acc 1.0 on the current data). Until enabled,
     sign(...) laws are DORMANT and every condition collapses to the white baseline.
  2. active_lawsets = [full_lawset] + yellow_cells = [3,5]: THIS DRIVER now sets both as Hydra overrides
     (--lawset / --yellow_cells) -> plan.py renders the R1-R10 sign system and injects yellow_cell(3/5).
  3. in_cell grounded from the position probe + geometry (grounding.py), so R7's in_cell(Y) guard and
     R7b's visited(4) taint actually fire live (cube_cells probe is retired).
  4. DDL render-back: mpc.py latches the reasoner-derived sign into the env each step -- GATED on
     sign_flip being set, so ONLY sign-experiment runs touch it. Standard law_eval / oracle runs are unaffected.
  5. a benchmark whose route crosses the sign-controlled band: use data/crossing_set (every pair crosses
     the middle row 3/4/5) -- law_eval_center works too but only some pairs traverse a yellow cell.

Runs in the CONTAINER (drives plan.py via scripts/eval_sweep.py, one subprocess per condition):
    /isaac-sim/python.sh experiments/sign_change/run_sign_change.py \
        --benchmark data/law_eval_center --frame 2 --colors green yellow red \
        --model_name wm_5k --epoch 30 \
        --probe probes/weights/cube_pos_encoded.pth --out results/sign_change
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent   # .../dino_wm
sys.path.insert(0, str(_REPO)); import provenance


def _aggregate(summary_path: Path, mode: str):
    """n-weighted (success, law-abidance, illegal-frac) over a condition's law-eval summary.json."""
    if not summary_path.exists():
        return None
    d = json.load(open(summary_path))
    pairs = (d.get("results", {}) or {}).get(mode, {})
    rows = [(s.get("n_total", 0), s.get("success_rate"), s.get("law_abidance_rate"),
             s.get("illegal_frame_frac_mean")) for s in pairs.values() if isinstance(s, dict)]
    rows = [r for r in rows if r[0]]
    if not rows:
        return None
    N = sum(r[0] for r in rows)
    w = lambda i: sum(r[0] * (r[i] or 0) for r in rows) / N
    return dict(n=N, succ=w(1), law=w(2), ill=w(3))


def run_condition(args, label: str, flip_color):
    """One eval_sweep run; flip_color=None => WHITE baseline (no flip)."""
    out = Path(args.out) / label
    overrides = [
        "planner=mpc_rrt", f"metric_cell={args.metric_cell}",
        f"model_name={args.model_name}", f"model_epoch={args.epoch}",
        f"objective.pos_probe_path={args.probe}",
        f"legislation.mode={args.mode}", f"legislation.constraint_margin={args.cushion}",
        # THE sign-experiment flags: select the R1-R10 sign lawset + designate the static yellow cells.
        # yellow_cells is a plain int list (Hydra can't parse `yellow_cell(3)`; plan.py expands it to facts).
        f"legislation.active_lawsets=[{args.lawset}]",
        f"legislation.yellow_cells=[{','.join(str(c) for c in args.yellow_cells)}]",
        "has_decoder=false", f"device={args.device}",
    ]
    cmd = [sys.executable, "scripts/eval_sweep.py",
           "--law_eval", args.benchmark, "--modes", args.mode,
           "--batch", str(args.batch), "--out", str(out)]
    if args.max is not None:
        cmd += ["--max", str(args.max)]
    if args.video:
        cmd += ["--video"]              # smooth per-sim-step executed video (shows the sign recolour)
    if flip_color is not None:
        cmd += ["--frame", str(args.frame), "--color", flip_color]
    cmd += ["--"] + overrides
    print(f"\n===== condition [{label}] "
          f"({'no flip' if flip_color is None else f'flip -> {flip_color} @ step {args.frame}'}) =====")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=_REPO, check=True)
    return _aggregate(out / "summary.json", args.mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/law_eval_center",
                    help="law-eval pair dir whose straight path crosses the sign-controlled cell")
    ap.add_argument("--frame", type=int, default=2, help="MPC step at which the sign flips")
    ap.add_argument("--colors", nargs="+", default=["green", "yellow"],
                    help="sign colours to flip TO (each a separate condition); white baseline always runs")
    ap.add_argument("--mode", default="social", choices=["social", "deviant", "off"])
    ap.add_argument("--lawset", default="full_lawset", help="law CATEGORY to enforce (full_lawset = the R1-R10 sign system)")
    ap.add_argument("--yellow_cells", nargs="+", type=int, default=[3, 5],
                    help="static yellow-cell ids (R5/R7/R7b). Cells flanking the center 4 by default.")
    ap.add_argument("--metric_cell", type=int, default=4, help="cell the sign governs (scored for entry)")
    ap.add_argument("--model_name", default="wm_5k")
    ap.add_argument("--epoch", default="30")
    ap.add_argument("--probe", default="probes/weights/cube_pos_encoded.pth",
                    help="cube-position probe (planner + legislation share it)")
    ap.add_argument("--cushion", type=float, default=0.0, help="constraint_margin (m); 0 = honest baseline")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max", type=int, default=None, help="cap evals/pair for a quick pass")
    ap.add_argument("--video", action="store_true",
                    help="save the smooth per-sim-step executed video (shows the sign recolour mid-rollout)")
    ap.add_argument("--out", default="results/sign_change")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("[sign_change] PREREQ CHECK — this driver assumes: (1) sign_color probe trained on the CURRENT "
          "data + enabled in probes.yaml, (2) active_lawsets=[full_lawset] in legal_database.yaml. If the "
          "sign colours don't change behaviour, one of those is unset (see the module docstring).")

    results = {"white": run_condition(args, "white", None)}
    for c in args.colors:
        results[c] = run_condition(args, c, c)

    print("\n===== SIGN-CHANGE SUMMARY (adaptation = change vs white baseline) =====")
    print(f"  {'condition':<10}{'n':>5}{'success':>9}{'center-abid':>13}{'ill-frac':>10}")
    base = results.get("white")
    for label, r in results.items():
        if r is None:
            print(f"  {label:<10}   (no summary — did the run finish?)"); continue
        delta = "" if (label == "white" or base is None) else f"   Δcenter-abid {r['law'] - base['law']:+.3f}"
        print(f"  {label:<10}{r['n']:>5}{r['succ']:>9.3f}{r['law']:>13.3f}{r['ill']:>10.3f}{delta}")
    print("\n  Reminder: under GREEN, LOWER center-abidance is the CORRECT adaptation (green permits cell "
          f"{args.metric_cell}); under RED expect success to drop (agent halts). See docstring.")
    (Path(args.out) / "sign_change_summary.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    provenance.write(Path(args.out), __file__, args=args)


if __name__ == "__main__":
    main()
