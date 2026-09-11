"""Harness instrumentation for the MPC loop — extracted from planning/mpc.py (behaviour-preserving).

planning/mpc.py is upstream DINO-WM code; this module collects the local additions that used to sit
inline as `### HARNESS EDIT ###` blocks so the planner reads close to the original. Every helper takes
the planner (or evaluator) explicitly and preserves the exact semantics of the block it replaced:

  * seed_actions_from_probe : warm-start the stroke at the probe-estimated cube (obs-only).
  * log_stroke_vs_cube      : per-eval [dbg] print — is the planned stroke near the cube?
  * stitch_executed         : accumulate executed frames across MPC steps (drop the duplicate boundary).
  * make_introspector       : build the optional RRT/MPC introspector (off by default).
  * save_reground_diag      : per-iter plan{iter}.png of the re-grounded imagination vs the real step.
  * WMErrorTracker          : per-eval WM 1-step prediction error + re-grounded imagination, finalised
                              onto the planner for eval_metrics.json / the closed-loop output_final.

Sign-colour control (exogenous flip + DDL render-back) lives separately in planning/sign_control.py.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def seed_actions_from_probe(planner, cur_obs_0, memo_actions):
    """Warm-start the stroke START at the cube ESTIMATED FROM THE CURRENT OBSERVATION via the probe
    (obs-only -- NO ground-truth state). The CEM's default init is the data-mean (~grid centre); for an
    off-centre cube ~all samples start far from it and MISS -> frozen/no-op. Seeding [start=cube_est,
    disp=0] centres the search on the cube so it samples CONTACTING strokes. Falls back to `memo_actions`
    (the sub-planner's carried tail) when there is no position probe or a warm tail already exists."""
    probe = getattr(planner.objective_fn, "position_probe", None)
    if probe is not None and (memo_actions is None or memo_actions.shape[1] == 0):
        trans = planner.preprocessor.transform_obs(cur_obs_0)
        with torch.no_grad():
            z = planner.wm.encode_obs({"visual": trans["visual"].to(planner.device),
                                       "proprio": trans["proprio"].to(planner.device)})
            cube = probe(z["visual"][:, -1]).detach().cpu().numpy()      # (b,2) meters, from OBS
        warm = np.concatenate([cube, np.zeros_like(cube)], axis=1)[:, None, :]   # (b,1,4)
        return planner.preprocessor.normalize_actions(torch.from_numpy(warm.astype(np.float32)))
    return memo_actions


def log_stroke_vs_cube(cur_state, exec_taken):
    """[dbg] print the first few evals' planned stroke vs the current cube xy (diagnose frozen/no-op)."""
    cxy = np.asarray(cur_state)[:, 18:20]
    for i in range(min(3, exec_taken.shape[0])):
        s = exec_taken[i, 0]
        print(f"  [dbg e{i}] cube=({cxy[i][0]:+.3f},{cxy[i][1]:+.3f})  start=({s[0]:+.3f},{s[1]:+.3f})"
              f"  disp=({s[2]:+.3f},{s[3]:+.3f})  |start-cube|={np.linalg.norm(s[:2] - cxy[i]):.3f}")


def stitch_executed(planner, e_obses, e_states):
    """Accumulate the newly-committed executed frames onto planner.executed_obses / executed_states so
    the final video/metrics reuse them (no full re-roll). Drops the duplicate boundary frame on later
    rolls (the first frame of roll k == the last frame of roll k-1)."""
    if planner.executed_obses is None:
        planner.executed_obses = {k: v for k, v in e_obses.items()}
        planner.executed_states = e_states
    else:
        for k in planner.executed_obses:
            planner.executed_obses[k] = np.concatenate([planner.executed_obses[k], e_obses[k][:, 1:]], axis=1)
        planner.executed_states = np.concatenate([planner.executed_states, e_states[:, 1:]], axis=1)


def make_introspector(planner):
    """Build the optional RRT/MPC introspector (rrt_introspect.enabled=true). OFF by default -> None.
    Also flips the sub-planner's `_log_queries` so it records its sampled-candidate distribution."""
    icfg = getattr(planner, "introspect_cfg", None)
    if not (icfg and icfg.get("enabled")):
        return None
    from planning.introspect import RRTIntrospector
    introspector = RRTIntrospector(
        out_dir=os.path.join(os.getcwd(), "rrt_introspect"),
        top_k=int(icfg.get("top_k", 5)),
        resim=bool(icfg.get("resim", True)),
        max_evals=int(icfg.get("max_evals", 3)))
    if hasattr(planner.sub_planner, "_log_queries"):
        planner.sub_planner._log_queries = True
    return introspector


def save_reground_diag(evaluator, cur_obs_0, cur_state, taken_actions, precomputed_env, iteration):
    """plan{iter}.png = what the planner ACTUALLY reasoned over this step: the RE-GROUNDED WM imagination
    (from cur_obs_0 over the committed action) vs the real executed step. Reuses the already-rendered
    frames (precomputed_env) -> no extra sim render. Best-effort: a plot/shape hiccup is logged and
    skipped, never kills the run."""
    try:
        evaluator.assign_init_cond(obs_0=cur_obs_0, state_0=cur_state)
        evaluator.eval_actions(taken_actions, filename=f"plan{iteration}", save_video=False,
                               full_video=True, precomputed_env=precomputed_env)
    except Exception as e:  # noqa: BLE001
        print(f"[diag] per-iter plan{iteration}.png skipped: {e}")


class WMErrorTracker:
    """Per-eval closed-loop WM diagnostics accumulated across MPC steps: the 1-step cube-prediction error
    vs the REAL executed cube, the decoder-free latent MSE, the pred-vs-real committed cube xy, and the
    re-grounded decoded imagination stitched into a closed-loop output_final. `record` is called once per
    committed step; `finalize` writes the aggregates back onto the planner (surfaced in eval_metrics.json).
    """

    def __init__(self):
        self.pred_err = {}     # eval_index -> [per-step WM 1-step cube-pred error (m)]
        self.latent_err = {}   # eval_index -> [per-step WM 1-step latent MSE (decoder-free)]
        self.pred_xy = {}      # eval_index -> [[x,y], ...] WM-PREDICTED committed-stroke cube
        self.real_xy = {}      # eval_index -> [[x,y], ...] REAL sim committed-stroke cube (pred-vs-real diag)
        self.probe_start_xy = {}  # eval_index -> [[x,y], ...] q = probe(encode(obs)): the cube the planner
                                  # GROUNDS legality on at each step's START (obs-only, NO GT). Lets a rerun
                                  # anchor the pred-vs-real figure at the planner's true (probe-space) start.
        self.imagined = []     # per-step re-grounded decoded frames -> closed-loop output_final

    def record(self, planner, cur_obs_0, taken_actions, e_final_state, e_final_obs, iteration):
        """One committed step: WM 1-step error vs the real executed cube (accumulated per eval) + the
        re-grounded decoded imagined frame. Best-effort — a hiccup is logged and skipped, never fatal."""
        try:
            from planning.planning_metrics import wm_regrounded_eval
            err, lat, imag, pred, real = wm_regrounded_eval(
                planner.wm, planner.preprocessor, planner.objective_fn, cur_obs_0, taken_actions,
                e_final_state, real_obs=e_final_obs,
                decode=(getattr(planner.wm, "decoder", None) is not None))
            if err is not None:
                for i in range(len(err)):
                    self.pred_err.setdefault(i, []).append(float(err[i]))
            if lat is not None:
                for i in range(len(lat)):
                    self.latent_err.setdefault(i, []).append(float(lat[i]))
            if pred is not None and real is not None:      # record pred-vs-real committed cube (x,y)
                for i in range(len(pred)):
                    self.pred_xy.setdefault(i, []).append([float(pred[i][0]), float(pred[i][1])])
                    self.real_xy.setdefault(i, []).append([float(real[i][0]), float(real[i][1])])
            if err is not None:
                print(f"[wm 1-step err] iter {iteration}: probe {float(err.mean()):.4f} m"
                      + (f" | latent-mse {float(lat.mean()):.4f}" if lat is not None else "")
                      + f" | per-eval probe {np.round(err, 3).tolist()}")
            if imag is not None:
                if not self.imagined:
                    self.imagined.append(imag[:, 0])       # frame 0 = recon(initial obs), once
                self.imagined.append(imag[:, -1])          # this step's predicted frame
        except Exception as e:  # noqa: BLE001
            print(f"[wm regrounded eval] iter {iteration} skipped: {e}")

    def record_start(self, planner, cur_obs_0):
        """Log q = probe(encode(cur_obs_0)): the cube xy the planner GROUNDS legality on THIS step (obs-only,
        no ground truth). Call once per committed MPC step, at the top of the loop, so probe_start_xy stays
        aligned with pred_xy/real_xy. This is the ONE quantity missing from past runs: the pruner checks
        legality from q, but only GT frames were saved, so a GT-anchored pred-vs-real arrow splices a true
        start onto a probe-space prediction. Best-effort -- a hiccup is logged and skipped, never fatal."""
        try:
            probe = getattr(planner.objective_fn, "position_probe", None)
            if probe is None:
                return
            trans = planner.preprocessor.transform_obs(cur_obs_0)
            with torch.no_grad():
                z = planner.wm.encode_obs({"visual": trans["visual"].to(planner.device),
                                           "proprio": trans["proprio"].to(planner.device)})
                q = probe(z["visual"][:, -1]).detach().cpu().numpy()   # (b,2) meters, obs-only
            for i in range(len(q)):
                self.probe_start_xy.setdefault(i, []).append([float(q[i][0]), float(q[i][1])])
        except Exception as e:  # noqa: BLE001
            print(f"[wm probe-start] iter skipped: {e}")

    def finalize(self, planner, n_evals):
        """Write the per-eval aggregates onto the planner (same attributes the old inline block set):
        *_mean (mean over steps), *_steps (ragged per-step lists, for by-step-index aggregation) and the
        stitched re-grounded imagination tensor for the closed-loop output_final."""
        planner.wm_pred_err_mean = [
            float(np.mean(self.pred_err[i])) if self.pred_err.get(i) else float("nan") for i in range(n_evals)]
        planner.wm_latent_err_mean = [
            float(np.mean(self.latent_err[i])) if self.latent_err.get(i) else float("nan") for i in range(n_evals)]
        planner.wm_pred_err_steps = [self.pred_err.get(i, []) for i in range(n_evals)]
        planner.wm_latent_err_steps = [self.latent_err.get(i, []) for i in range(n_evals)]
        planner.wm_pred_xy_steps = [self.pred_xy.get(i, []) for i in range(n_evals)]     # pred-vs-real diag
        planner.wm_real_xy_steps = [self.real_xy.get(i, []) for i in range(n_evals)]
        planner.wm_probe_start_xy_steps = [self.probe_start_xy.get(i, []) for i in range(n_evals)]  # PERCEIVED start
        planner.imagined_regrounded = (torch.stack(self.imagined, dim=1) if self.imagined else None)
