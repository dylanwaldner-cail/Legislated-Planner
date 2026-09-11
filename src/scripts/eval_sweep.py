"""Evaluate a planner over the ENTIRE matching scene pool, in OOM-safe batches, and accumulate
per-eval metrics into one set of statistics + a graph.

One eval per DISTINCT dataset segment: `scene_offset` walks the pool (pool[offset:offset+n_evals])
so every matching scenario is run exactly once -- no seed-repeat. plan.py is run once per batch as a
SUBPROCESS (each frees the GPU on exit, so batch size caps memory, not total). Each batch dumps
eval_metrics.json (per-eval success / cube_l2 / law_violated / illegal_frame_frac + pool_size).

    # social agent (never breaks the law) over every segment whose straight path crosses cell 4:
    python scripts/eval_sweep.py --batch 10 --out sweep_social -- \
        planner=mpc_rrt scene_filter.via_cell=4 legislation.mode=social

    # deviant (minimizes violations) / rational (off) baselines on the SAME pool:
    python scripts/eval_sweep.py --batch 10 --out sweep_deviant  -- planner=mpc_rrt scene_filter.via_cell=4 legislation.mode=deviant
    python scripts/eval_sweep.py --batch 10 --out sweep_rational -- planner=mpc_rrt scene_filter.via_cell=4 legislation.mode=off

MULTIPLE init->goal PAIRS: --pairs runs one SUB-SWEEP per (init_cell, goal_cell) pair over its own
pool, tracks each separately, and writes a per-pair breakdown + a comparison figure. Use metric_cell
(not via_cell) so law-abidance is scored on the forbidden cell across every pair:

    python scripts/eval_sweep.py --pairs 1:7 3:5 --batch 10 --out sweep_social -- \
        planner=mpc_rrt metric_cell=4 legislation.mode=social

Everything after `--` is passed to plan.py as hydra overrides. Writes <out>/summary.json and
<out>/sweep.png (single sweep), or <out>/summary.json + <out>/compare.png plus a per-pair
<out>/<init>_<goal>/ subdir for each pair.
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (scripts/ is one level down)
import provenance

import numpy as np

# Plotting lives in plot_sweep.py (one home for matplotlib; also runnable standalone to re-plot a
# finished sweep from its JSON). Guarded import works whether run as a script or a module.
try:
    from plot_sweep import plot_all
except ImportError:
    from scripts.plot_sweep import plot_all

_REPO = Path(__file__).resolve().parent.parent   # run plan.py from here
_ACC_KEYS = ("success", "cube_l2", "state_dist", "cubes_correct", "law_violated", "law_violated_swept",
             "law_violated_center",
             "illegal_frame_frac", "illegal_overlap_frac", "peak_frame_overlap", "peak_swept_overlap",
             "n_steps", "path_len_to_goal",
             "optimal_path_len", "path_efficiency",
             "cell_revisits", "boundary_contacts",
             "wm_pred_err", "wm_latent_err",
             # ragged per-eval step-lists (list of lists) -> aggregated BY STEP INDEX in summarize
             "wm_pred_err_steps", "wm_latent_err_steps")


def _run_batch(overrides, n_evals, offset, bdir, seed, video=False, introspect=False, resume=False):
    mf = bdir / "eval_metrics.json"
    if resume and mf.exists():                          # --resume: this batch already completed -> skip the re-run
        print(f"[resume] skip {bdir} (eval_metrics.json exists)")
        return json.loads(mf.read_text()), 0.0
    bdir.mkdir(parents=True, exist_ok=True)
    # NB: TiledCamera did NOT raise the batch ceiling -- the RTX descriptor/parameter-block pool is
    # bound by scene INSTANCE/material count (not the camera), so it caps num_envs in the mid-teens
    # regardless. Sweeps stay on the plain per-env Camera; pass tiled_camera=true yourself to opt in.
    extra = []
    # --introspect: RRT logs its sampled-candidate QUERY distribution (resim off -> no extra sim cost);
    # each batch drops rrt_introspect/summary.json, aggregated by plot_query_vs_train at the end.
    if introspect and not any(o.split("=", 1)[0].startswith("rrt_introspect") for o in overrides):
        extra += ["rrt_introspect.enabled=true", "rrt_introspect.resim=false"]
    cmd = [sys.executable, "plan.py", *overrides,
           f"n_evals={n_evals}", f"scene_offset={offset}", f"seed={seed}",
           f"video={'true' if video else 'false'}", *extra, f"hydra.run.dir={bdir}"]
    print(f"\n$ (cwd={_REPO}) {' '.join(cmd)}")
    _t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=_REPO)
    elapsed = time.perf_counter() - _t0
    print(f"  [batch runtime] {elapsed:.1f}s  ({n_evals} evals, offset {offset})")
    mf = bdir / "eval_metrics.json"
    return (json.loads(mf.read_text()) if mf.exists() else None), elapsed


def _wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def sweep_pool(overrides, batch, maxn, seed, out_dir, video=False, introspect=False, resume=False):
    """Run the WHOLE matching pool in OOM-safe batches. Returns (acc, pool_size, meta).
    acc maps each metric name -> a flat list of per-eval values across all batches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = {k: [] for k in _ACC_KEYS}
    batch_times = []
    rb_sum = {}
    def _acc_rb(mm):    # sum the per-batch planning-time breakdown (plan/rrt/legislation seconds)
        # Scalars only. runtime_breakdown also carries "per_step" -- a per-(eval, MPC step) list of
        # timing records written by rrt.plan() -- which is not summable and stays in the batch's own
        # eval_metrics.json for anyone who wants per-action costs.
        for _k, _v in ((mm or {}).get("runtime_breakdown") or {}).items():
            if isinstance(_v, (list, dict)):
                continue
            rb_sum[_k] = rb_sum.get(_k, 0.0) + float(_v or 0.0)
    # batch 0 also tells us the pool size + the checked cells
    m, dt = _run_batch(overrides, batch, 0, out_dir / "batch_000", seed, video=video, introspect=introspect, resume=resume)
    batch_times.append(dt); _acc_rb(m)
    if m is None:
        print(f"[warn] {out_dir.name}: batch 0 produced no eval_metrics.json -- skipping")
        return acc, 0, {}
    pool_size = m.get("pool_size") or 0
    if not pool_size:
        print(f"[warn] {out_dir.name}: no pool_size -- is the filter too strict? skipping")
        return acc, 0, {}
    total = pool_size if maxn is None else min(maxn, pool_size)
    print(f"[sweep:{out_dir.name}] pool_size={pool_size} -> {total} segments in batches of {batch}")
    for k in acc:
        acc[k].extend(m.get(k, []))
    meta = {"scene_filter": m.get("scene_filter"), "checked_cells": m.get("checked_cells")}

    offset, b = batch, 0
    while offset < total:
        b += 1
        n = min(batch, total - offset)
        m, dt = _run_batch(overrides, n, offset, out_dir / f"batch_{b:03d}", seed, video=video, introspect=introspect, resume=resume)
        batch_times.append(dt); _acc_rb(m)
        if m is not None:
            for k in acc:
                acc[k].extend(m.get(k, []))
        offset += batch
    # RUNTIME (wall-clock per subprocess batch, incl. sim boot) -> lands in summary.json via **meta
    meta["runtime_s"] = round(sum(batch_times), 1)
    meta["n_batches"] = len(batch_times)
    meta["mean_batch_s"] = round(sum(batch_times) / len(batch_times), 1) if batch_times else None
    if rb_sum:
        rb = {k: round(v, 1) for k, v in rb_sum.items()}
        rb["other_s"] = round(meta["runtime_s"] - rb.get("plan_total_s", 0.0), 1)  # boot + sim exec + render + scoring
        meta["runtime_breakdown_s"] = rb
    print(f"[sweep:{out_dir.name}] runtime {meta['runtime_s']}s over {meta['n_batches']} batches "
          f"(mean {meta['mean_batch_s']}s/batch) | breakdown {meta.get('runtime_breakdown_s')}")
    # per-task violation-geometry figure (WHY breaches happen: approach angle / start side / mechanism).
    # Reads the batches just written; never let a plot failure crash the sweep. Guarded import mirrors
    # plot_sweep above (works whether eval_sweep runs as a script or a module).
    try:
        try:
            from plot_violation_geometry import render_run
        except ImportError:
            from scripts.plot_violation_geometry import render_run
        render_run(out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[violation_geometry] {out_dir.name} skipped: {e}")
    return acc, pool_size, meta


def _mean(xs):
    xs = [v for v in (xs or []) if v is not None]
    return float(np.mean(xs)) if xs else None


def _by_step(list_of_lists):
    """Ragged per-eval step-lists -> per-STEP-INDEX mean across evals (only evals that reached step
    k contribute to step k). e.g. [0.08, 0.05, 0.04] => step 0 worst. [] if no data."""
    lls = [x for x in (list_of_lists or []) if x]
    if not lls:
        return []
    maxk = max(len(x) for x in lls)
    return [float(np.mean([x[k] for x in lls if len(x) > k])) for k in range(maxk)]


def _iqr(xs):
    """[median, q25, q75] over non-null values (robust summary for skewed per-eval metrics)."""
    xs = [v for v in (xs or []) if v is not None]
    if not xs:
        return None
    return [float(np.median(xs)), float(np.percentile(xs, 25)), float(np.percentile(xs, 75))]


def summarize(acc, pool_size, meta):
    succ = np.asarray(acc["success"], float)
    cube_l2 = np.asarray(acc["cube_l2"], float)
    violated = np.asarray(acc["law_violated"], float)          # boolean: entered illegal after frame 0
    sw = np.asarray(acc.get("law_violated_swept", []), float)  # boolean: FOOTPRINT swept THROUGH illegal (transit-aware)
    ct = np.asarray(acc.get("law_violated_center", []), float) # boolean: CENTER path swept THROUGH illegal (footprint ignored)
    ill_frac = np.asarray(acc["illegal_frame_frac"], float)    # per-eval fraction of frames illegal
    n = len(succ)
    return {
        "n_total": n, "pool_size": pool_size,
        "success_rate": float(succ.mean()) if n else None, "success_ci95": _wilson(succ.sum(), n),
        # BOOLEAN metric: fraction of episodes that NEVER entered a forbidden cell after frame 0
        "law_abidance_rate": float(1 - violated.mean()) if violated.size else None,
        "law_abidance_ci95": _wilson(int((1 - violated).sum()), violated.size) if violated.size else None,
        "n_violated": int(violated.sum()) if violated.size else None,
        # TRANSIT-AWARE (swept footprint between stroke boundaries) -- catches cubes that pass THROUGH
        # a cell mid-path; per-frame law_abidance can't (only boundaries are recorded). Use this as the
        # stricter/true law-abidance number.
        "law_abidance_swept_rate": float(1 - sw.mean()) if sw.size else None,
        "law_abidance_swept_ci95": _wilson(int((1 - sw).sum()), sw.size) if sw.size else None,
        # CENTER transit: cube CENTROID path through the cell (footprint ignored) -- the LEAST-strict
        # boolean (a graze where only the footprint clips the cell does NOT count). center >= swept abidance.
        "law_abidance_center_rate": float(1 - ct.mean()) if ct.size else None,
        "law_abidance_center_ci95": _wilson(int((1 - ct).sum()), ct.size) if ct.size else None,
        "n_violated_center": int(ct.sum()) if ct.size else None,
        # CONTINUOUS metrics: fraction of frames illegal, and FLAGRANCY (fraction of cube AREA in zone)
        "illegal_frame_frac_mean": float(ill_frac.mean()) if ill_frac.size else None,
        "illegal_overlap_frac_mean": _mean(acc.get("illegal_overlap_frac")),
        "illegal_overlap_frac_iqr": _iqr(acc.get("illegal_overlap_frac")),   # [median, q25, q75]
        # PEAK (worst-moment) intrusion: mean over evals of each eval's deepest footprint-area fraction
        # in a checked cell -- at a rest frame (frame) and along transit (swept). swept >= frame; a big
        # swept mean with a small frame mean = drive-throughs the per-frame metric misses.
        "peak_frame_overlap_mean": _mean(acc.get("peak_frame_overlap")),
        "peak_swept_overlap_mean": _mean(acc.get("peak_swept_overlap")),
        "cube_l2_mean": float(cube_l2.mean()) if n else None,
        "cube_l2_median": float(np.median(cube_l2)) if n else None,
        "cube_l2_std": float(cube_l2.std()) if n else None,
        # path efficiency: median + IQR (robust) alongside means
        "n_steps_mean": _mean(acc.get("n_steps")), "n_steps_iqr": _iqr(acc.get("n_steps")),
        # cube travel distance (m) until FIRST goal-cell entry, over evals that reached (None dropped)
        "path_len_to_goal_mean": _mean(acc.get("path_len_to_goal")),
        "path_len_to_goal_iqr": _iqr(acc.get("path_len_to_goal")),
        # geometric optimal law-respecting path + efficiency (taken/optimal, >=1; over reached evals)
        "optimal_path_len_mean": _mean(acc.get("optimal_path_len")),
        "path_efficiency_mean": _mean(acc.get("path_efficiency")),
        "path_efficiency_iqr": _iqr(acc.get("path_efficiency")),
        "cell_revisits_mean": _mean(acc.get("cell_revisits")), "cell_revisits_iqr": _iqr(acc.get("cell_revisits")),
        "boundary_contacts_mean": _mean(acc.get("boundary_contacts")),
        "wm_pred_err_mean": _mean(acc.get("wm_pred_err")),   # mean WM 1-step cube-pred error (m)
        "wm_latent_err_mean": _mean(acc.get("wm_latent_err")),  # mean WM 1-step latent MSE (decoder-free)
        # BY STEP INDEX (step 0 = first committed stroke): is step 0 systematically the worst?
        "wm_pred_err_by_step": _by_step(acc.get("wm_pred_err_steps")),
        "wm_latent_err_by_step": _by_step(acc.get("wm_latent_err_steps")),
        **meta,
    }


# plot_single / plot_pairs now live in scripts/plot_sweep.py (imported above).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10, help="n_evals per plan.py run (cap to avoid GPU OOM)")
    ap.add_argument("--max", type=int, default=None, help="cap total evals PER PAIR (default: the whole pool)")
    ap.add_argument("--seed", type=int, default=99, help="fixed seed -- pool coverage, not seed-repeat")
    ap.add_argument("--out", default="sweep_out")
    ap.add_argument("--pairs", nargs="+", default=None, metavar="INIT:GOAL",
                    help="one sub-sweep per init:goal cell pair, e.g. --pairs 1:7 3:5")
    ap.add_argument("--video", action="store_true",
                    help="save the smooth full-MPC executed video (per internal sim step; slower renders)")
    ap.add_argument("--introspect", action="store_true",
                    help="log RRT's sampled-candidate QUERY distribution (resim off) + draw query_vs_train.png")
    ap.add_argument("--resume", action="store_true",
                    help="skip any batch that already has eval_metrics.json (a completed run) -- lets a "
                         "killed sweep pick up where it left off without redoing finished batches")
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_5k_no_yaw_merged",
                    help="training data for the query-vs-train overlay (--introspect)")
    ap.add_argument("--law_eval", default=None, metavar="DIR",
                    help="run the law-eval benchmark (gen_law_eval_set.py output dir): one sub-sweep per "
                         "<init>_<goal> pair subdir x per --modes, via goal_source=law_eval")
    ap.add_argument("--modes", nargs="+", default=["social", "deviant"],
                    help="legislation modes for --law_eval (default: social deviant; add off for rational)")
    ap.add_argument("--frame", type=int, default=None,
                    help="MPC step at which to flip the stop sign to --color (sign_flip.frame). "
                         "Needs the sign_color probe registered for the sign laws to fire.")
    ap.add_argument("--color", default="yellow",
                    help="colour the sign flips TO at --frame (white|red|yellow|green; sign_flip.color)")
    ap.add_argument("overrides", nargs=argparse.REMAINDER,
                    help="hydra overrides for plan.py after `--` (planner=, scene_filter.*, metric_cell=, legislation.enforce=)")
    args = ap.parse_args()
    overrides = [o for o in args.overrides if o != "--"]
    # PROVENANCE payload shared by every manifest.json this sweep writes: the DATASET used and the
    # LAWS in play. eval_sweep doesn't resolve the active law CATEGORY (plan.py does) -> the resolved
    # active_lawsets/db_path land in each <mode>/<pair>/batch_*/manifest.json; here we record the source.
    prov_extra = {
        "benchmark": args.law_eval, "data_dir": args.data_dir, "modes": args.modes,
        "law_database": "legislation/legal_database.yaml (+ any legislation.* in overrides)",
        "resolved_laws_note": "active_lawsets/db_path resolved per run in each pair's plan.py manifest.json",
    }
    if args.frame is not None:                       # exogenous sign flip (same for every mode/pair)
        overrides += [f"sign_flip.frame={args.frame}", f"sign_flip.color={args.color}"]
    _cushion = next((o.split("=", 1)[1] for o in overrides if o.startswith("legislation.constraint_margin=")),
                    "config default (plan.yaml)")
    print(f"[eval_sweep] cushion δ (legislation.constraint_margin) = {_cushion}  |  modes={args.modes}  "
          f"|  benchmark={args.law_eval or args.data_dir}", flush=True)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---------- law-eval benchmark (per legislation mode x per <init>_<goal> pair subdir) ----------
    if args.law_eval:
        bench = Path(args.law_eval).resolve()
        pair_dirs = sorted(d for d in bench.iterdir() if d.is_dir() and re.fullmatch(r"\d+_\d+", d.name))
        if not pair_dirs:
            sys.exit(f"[fatal] no <init>_<goal> pair subdirs under {bench}")
        print(f"[law_eval] {len(pair_dirs)} pairs x modes {args.modes} from {bench}")
        results = {}
        for mode in args.modes:
            mode_pairs = {}
            for pdir in pair_dirs:
                pov = [*overrides, "goal_source=law_eval", f"goal_file_path={pdir}", f"legislation.mode={mode}"]
                odir = out / mode / pdir.name
                acc, pool_size, meta = sweep_pool(pov, args.batch, args.max, args.seed, odir,
                                                  video=args.video, introspect=args.introspect, resume=args.resume)
                if pool_size == 0:
                    print(f"[warn] [{mode}] {pdir.name}: empty pool -- skipping")
                    continue
                s = summarize(acc, pool_size, {**meta, "pair": pdir.name, "mode": mode, "overrides": pov})
                mode_pairs[pdir.name] = s
                (odir / "summary.json").write_text(json.dumps(s, indent=2))
                print(f"----- [{mode}] {pdir.name} (pool={pool_size}, n={s['n_total']}): "
                      f"success={_fmt(s['success_rate'])} law-abidance={_fmt(s['law_abidance_rate'])} "
                      f"illegal-frac={_fmt(s['illegal_frame_frac_mean'])}")
            # per-mode summary in the --pairs schema so plot_all can draw the per-mode comparison
            (out / mode).mkdir(parents=True, exist_ok=True)
            (out / mode / "summary.json").write_text(json.dumps(
                {"pairs": mode_pairs, "seed": args.seed, "batch": args.batch,
                 "max_per_pair": args.max, "base_overrides": [*overrides, f"legislation.mode={mode}"]}, indent=2))
            try:
                plot_all(out / mode)
            except Exception as e:  # noqa: BLE001
                print(f"[plot] {mode} skipped: {e}")
            results[mode] = mode_pairs
        mode_runtime = {mode: round(sum((s.get("runtime_s") or 0) for s in results.get(mode, {}).values()), 1)
                        for mode in args.modes}
        (out / "summary.json").write_text(json.dumps(
            {"benchmark": str(bench), "modes": args.modes, "results": results,
             "runtime_s_by_mode": mode_runtime, "total_runtime_s": round(sum(mode_runtime.values()), 1),
             "seed": args.seed, "batch": args.batch, "max_per_pair": args.max, "base_overrides": overrides}, indent=2))
        try:                                            # cross-mode AGENT comparison + frame-risk + shooting chart
            try:
                from plot_full_eval import plot_all_agents
            except ImportError:
                from scripts.plot_full_eval import plot_all_agents
            plot_all_agents(out)
        except Exception as e:  # noqa: BLE001
            print(f"[plot] agent comparison skipped: {e}")
        print("\n===== LAW-EVAL BENCHMARK SUMMARY =====")
        for mode in args.modes:
            print(f"  [{mode}]  (runtime {mode_runtime[mode]}s)")
            for label, s in results.get(mode, {}).items():
                print(f"    {label:>6}: n={s['n_total']:<4} success={_fmt(s['success_rate'])} "
                      f"law-abidance={_fmt(s['law_abidance_rate'])} illegal-frac={_fmt(s['illegal_frame_frac_mean'])} "
                      f"runtime={s.get('runtime_s')}s")
        provenance.write(out, __file__, args=args, extra=prov_extra)
        return

    # ---------- single sweep (no pairs) ----------
    if not args.pairs:
        acc, pool_size, meta = sweep_pool(overrides, args.batch, args.max, args.seed, out,
                                          video=args.video, introspect=args.introspect, resume=args.resume)
        if pool_size == 0:
            sys.exit("[fatal] empty pool -- pass a scene_filter/metric_cell so plan.py builds a pool")
        summary = summarize(acc, pool_size, {**meta, "overrides": overrides})
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\n===== SWEEP SUMMARY =====\n" + json.dumps(summary, indent=2))
        plot_all(out)   # read the written JSON + batch metrics and draw everything
        _maybe_query_plot(args, out)
        provenance.write(out, __file__, args=args, extra=prov_extra)
        return

    # ---------- multi-pair sweep ----------
    pairs = []
    for p in args.pairs:
        try:
            i, g = p.split(":")
            pairs.append((int(i), int(g)))
        except ValueError:
            sys.exit(f"[fatal] bad --pairs entry {p!r}; expected INIT:GOAL, e.g. 1:7")
    pair_summaries = {}
    for i, g in pairs:
        label = f"{i}->{g}"
        pov = [*overrides, f"scene_filter.init_cell={i}", f"scene_filter.goal_cell={g}"]
        acc, pool_size, meta = sweep_pool(pov, args.batch, args.max, args.seed, out / f"{i}_{g}",
                                          video=args.video, introspect=args.introspect, resume=args.resume)
        s = summarize(acc, pool_size, {**meta, "init_cell": i, "goal_cell": g, "overrides": pov})
        pair_summaries[label] = s
        (out / f"{i}_{g}" / "summary.json").write_text(json.dumps(s, indent=2))
        print(f"\n----- {label} (pool={pool_size}, n={s['n_total']}): "
              f"success={s['success_rate']}, law_abidance={s['law_abidance_rate']}, "
              f"illegal_frac={s['illegal_frame_frac_mean']}")

    combined = {"pairs": pair_summaries, "seed": args.seed, "batch": args.batch,
                "max_per_pair": args.max, "base_overrides": overrides}
    (out / "summary.json").write_text(json.dumps(combined, indent=2))
    print("\n===== PER-PAIR SUMMARY =====")
    for label, s in pair_summaries.items():
        print(f"  {label:>10}: n={s['n_total']:<4} success={_fmt(s['success_rate'])} "
              f"law-abidance={_fmt(s['law_abidance_rate'])} illegal-frac={_fmt(s['illegal_frame_frac_mean'])}")
    plot_all(out)   # read the written JSON + batch metrics and draw everything (compare + per-pair)
    _maybe_query_plot(args, out)
    provenance.write(out, __file__, args=args, extra=prov_extra)


def _maybe_query_plot(args, out):
    """With --introspect, draw the SEPARATE planner-query-vs-training overlay from the batch logs."""
    if not args.introspect:
        return
    try:
        from plot_query_vs_train import plot as _qvt
    except ImportError:
        from scripts.plot_query_vs_train import plot as _qvt
    _qvt(out, args.data_dir)


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "n/a"


if __name__ == "__main__":
    main()
