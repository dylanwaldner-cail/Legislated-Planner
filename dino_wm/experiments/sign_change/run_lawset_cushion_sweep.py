"""CUSHION SWEEP ON THE FULL SIGN LAWSET (R1-R10) -> results/aug25.

Q2's mitigation arm: how much of the compliance gap is recoverable by tightening the constraint the
PLANNER avoids-with (constraint_margin = delta, metres), given that the metric still scores reality
with the true cube_half? The existing social_cushion sweep (results/final/social_cushion) is on the
STATIC lawset (data/law_eval_center, geometric_laws) and therefore CANNOT be placed beside the
full-lawset Q1 numbers -- hence this run.

SETTINGS ARE PINNED TO results/aug20/sign_change so the arms are comparable: same benchmark
(law_eval_center_400), same seed (99), same sign schedule (--frame 1 --color yellow), same WM
(wm_5k @ epoch 30) and probe, same goal bank, same yellow cells. The ONLY thing that varies is
legislation.constraint_margin.

delta = 0 IS NOT RE-RUN. results/aug20/sign_change already IS the delta=0 arm under exactly these
settings, and it is the arm every cushioned result must be reported beside. Point --delta0-run at it
when aggregating (that is what `summarize()` does).

MODES -- `off` IS DELTA-INVARIANT AND IS EXCLUDED BY DEFAULT. rrt.py:161 gates the legality check on
`mode != "off"`, so the realistic agent's `viol` is all-False and the cushion never enters its
planner: five deltas would produce five identical result sets. Its single arm is already in aug20.
Pass `--modes social deviant off` to override if you want the invariance recorded explicitly.

COST. aug20 ran 400 evals/mode. At the documented ~10 GPU-h per 200 evals that is ~20 GPU-h per arm,
so the default 5 deltas x 2 modes = 10 arms ~= 200 GPU-h ~= 4 days wall on the 2 usable GPUs. Use
--max to cut N (e.g. --max 200 halves it) and --devices to split across both. ALWAYS --dry-run first.

Usage:
  python experiments/sign_change/run_lawset_cushion_sweep.py --dry-run
  python experiments/sign_change/run_lawset_cushion_sweep.py --devices cuda:0 cuda:1 --max 200
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Pinned to results/aug20/sign_change's manifest -- do not drift these without re-running delta=0.
_AUG20 = {
    "benchmark": "data/law_eval_center_400",
    "seed": 99,
    "batch": 10,
    "frame": 1,
    "color": "yellow",
    "model_name": "wm_5k",
    "epoch": 30,
    "probe": "probes/weights/cube_pos_encoded.pth",
    "goal_bank": "data/goal_cell_bank",
    "lawset": "full_lawset",
    "yellow_cells": [3, 5],
    "metric_cell": 4,
}


def build_cmd(a, mode, delta, device, out):
    overrides = [
        "planner=mpc_rrt", f"metric_cell={_AUG20['metric_cell']}",
        f"model_name={_AUG20['model_name']}", f"model_epoch={_AUG20['epoch']}",
        f"objective.pos_probe_path={_AUG20['probe']}",
        f"legislation.mode={mode}",
        f"legislation.constraint_margin={delta}",          # THE swept knob
        f"legislation.active_lawsets=[{_AUG20['lawset']}]",
        f"legislation.yellow_cells=[{','.join(str(c) for c in _AUG20['yellow_cells'])}]",
        f"legislation.goal_bank={_AUG20['goal_bank']}",
        "has_decoder=false", f"device={device}",
    ]
    cmd = [sys.executable, "scripts/eval_sweep.py",
           "--law_eval", _AUG20["benchmark"], "--modes", mode,
           "--batch", str(_AUG20["batch"]), "--seed", str(_AUG20["seed"]),
           "--out", str(out),
           "--frame", str(_AUG20["frame"]), "--color", _AUG20["color"]]
    if a.max is not None:
        cmd += ["--max", str(a.max)]
    if a.resume:
        cmd += ["--resume"]
    return cmd + ["--"] + overrides


def arm_dir(out_root, mode, delta):
    """One directory PER ARM. Deliberately not a shared delta_<d>/ root with mode subdirs: two arms at
    the same delta run CONCURRENTLY on different GPUs, and eval_sweep writes manifest.json/summary.json
    at the root it is given -- a shared root races them."""
    return Path(out_root) / f"delta_{delta:.2f}_{mode}"


def summarize(out_root, delta0_run, modes, deltas):
    """Aggregate the sweep, with the delta=0 arm pulled from the aug20 run (never re-run here)."""
    rows = []
    for mode in modes:
        for delta in [0.0] + list(deltas):
            d = Path(delta0_run) if delta == 0.0 else arm_dir(out_root, mode, delta)
            f = d / "summary.json"
            if not f.exists():
                rows.append({"mode": mode, "delta": delta, "status": "MISSING", "path": str(f)})
                continue
            s = json.load(open(f))
            rows.append({"mode": mode, "delta": delta, "status": "ok", "summary_keys": list(s)[:6]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.04, 0.05])
    ap.add_argument("--modes", nargs="+", default=["social", "deviant"],
                    help="off is delta-invariant (rrt.py:161) and excluded by default -- see module docstring")
    ap.add_argument("--out", default="results/aug25/cushion_lawset")
    ap.add_argument("--delta0-run", default="results/aug20/sign_change",
                    help="existing delta=0 arm; NOT re-run, only referenced when aggregating")
    ap.add_argument("--devices", nargs="+", default=["cuda:0"],
                    help="round-robin arms across these (container: cuda:0=GPU3, cuda:1=GPU7)")
    ap.add_argument("--max", type=int, default=None, help="cap evals/arm (400 = full aug20 N)")
    ap.add_argument("--resume", action="store_true", help="skip batches already on disk")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + commands, run nothing")
    ap.add_argument("--summarize", action="store_true", help="aggregate an existing sweep and exit")
    a = ap.parse_args()

    if a.summarize:
        for r in summarize(a.out, a.delta0_run, a.modes, a.deltas):
            print(r)
        return

    arms = [(m, d) for m in a.modes for d in a.deltas]
    n = a.max if a.max is not None else 400
    gpu_h = len(arms) * (n / 200.0) * 10.0
    print(f"=== full-lawset cushion sweep -> {a.out} ===")
    print(f"  modes   : {a.modes}" + ("" if "off" not in a.modes else
          "   [!] off is DELTA-INVARIANT (rrt.py:161) -- its arms will be identical"))
    print(f"  deltas  : {a.deltas}   (delta=0 NOT re-run; taken from {a.delta0_run})")
    print(f"  arms    : {len(arms)} x {n} evals")
    print(f"  EST COST: ~{gpu_h:.0f} GPU-h ~= {gpu_h/max(1,len(a.devices)):.0f} h wall on "
          f"{len(a.devices)} device(s)")
    print(f"  devices : {a.devices}\n")

    for i, (mode, delta) in enumerate(arms):
        device = a.devices[i % len(a.devices)]
        out = Path(a.out) / f"delta_{delta:.2f}"
        cmd = build_cmd(a, mode, delta, device, out)
        print(f"--- arm {i+1}/{len(arms)}: mode={mode} delta={delta} device={device} -> {out}")
        print("    " + " ".join(cmd))
        if not a.dry_run:
            subprocess.run(cmd, cwd=_REPO, check=True)

    if a.dry_run:
        print("\n[dry-run] nothing executed.")


if __name__ == "__main__":
    main()
