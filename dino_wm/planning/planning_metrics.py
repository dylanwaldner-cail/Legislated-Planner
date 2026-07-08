"""Per-eval planning metrics (dumped to eval_metrics.json, aggregated by scripts/eval_sweep.py).

Pulled out of plan.py to keep the planning workspace lean. `build_eval_metrics(...)` takes the
GROUND-TRUTH executed states + the run's context and returns the metrics dict. Everything here is
state-based (no probe assumption). All array fields are per-eval lists (length == n_evals):

  * task success / cube_l2 / state_dist / cubes_correct  -- from the evaluator's rollout metrics
  * law abidance (law_violated / illegal_frame_*)        -- cube FOOTPRINT in any CHECKED cell after
    frame 0 (frame 0 grandfathered). Checked cells = the constraint's forbidden cells UNION
    scene_filter.via_cell UNION metric_cell (so the SELFISH agent is still scored on the cell of
    interest, and one metric cell can span many init/goal pairs).
  * path efficiency: n_steps (strokes to goal), cell_revisits (backtracking), boundary_contacts.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.scene_index import _occupancy          # AABB FOOTPRINT occupancy (matches the law)
from probes.probe_cube_position import gm            # which_cell / GRID_HALF
from probes.probe_cube_cells import CUBE_HALF

_CUBE_XY = slice(18, 20)                              # cube (x,y) within the 31-D state


@torch.no_grad()
def wm_regrounded_eval(wm, preprocessor, objective_fn, cur_obs_0, taken_actions, e_final_state,
                       real_obs=None, decode=False):
    """Re-ground on cur_obs_0 and roll the WM over the COMMITTED stroke (closed-loop, 1 step).
    Returns (err, latent_err, imagined):
      err        : (N,) L2 meters between the position-probe's 1-step cube prediction and the REAL
                   executed cube -- the task-space 'is the committed prediction any good?' signal.
                   None if there's no position probe on the objective.
      latent_err : (N,) mean-squared error between the PREDICTED visual latent and the TRUE encoded
                   latent of the real executed frame (real_obs). Decoder-free -- the purest WM error,
                   exactly the quantity the predictor was trained to minimise. None if real_obs is None.
      imagined   : (N, L, 3, H, W) decoded frames of the re-grounded rollout if decode=True and the
                   WM has a decoder, else None. Frame 0 = recon(cur_obs_0); frame -1 = predicted step.
    One rollout serves all three, so the MPC loop pays a single 1-step WM pass per step."""
    dev = next(wm.parameters()).device
    t = preprocessor.transform_obs(cur_obs_0)
    obs0 = {"visual": t["visual"].to(dev), "proprio": t["proprio"].to(dev)}
    z = wm.rollout(obs_0=obs0, act=taken_actions.to(dev))[0]          # z_obses dict
    zp_vis = z["visual"][:, -1]                                       # (N, P, D) predicted visual latent
    err = None
    probe = getattr(objective_fn, "position_probe", None)
    if probe is not None:
        pred = probe(zp_vis).detach().cpu().numpy()                  # (N,2) WM-predicted cube
        real = np.asarray(e_final_state)[:, _CUBE_XY]                 # (N,2) sim ground-truth cube
        err = np.linalg.norm(pred - real, axis=1)                    # (N,)
    latent_err = None
    if real_obs is not None:                                         # pure-WM latent MSE (no decoder)
        tr = preprocessor.transform_obs(real_obs)
        zr = wm.encode_obs({"visual": tr["visual"].to(dev),
                            "proprio": tr["proprio"].to(dev)})["visual"][:, -1]   # (N,P,D) TRUE encoded
        latent_err = ((zp_vis - zr) ** 2).reshape(zp_vis.shape[0], -1).mean(dim=1).detach().cpu().numpy()
    imagined = None
    if decode and getattr(wm, "decoder", None) is not None:
        imagined = wm.decode_obs(z)[0]["visual"].detach().cpu()      # (N, L, 3, H, W) in [-1,1]
    return err, latent_err, imagined


def _checked_cells(constraint, scene_filter, metric_cell):
    """The cells scored for law abidance (see module docstring)."""
    check = set()
    for nm, ar in getattr(constraint, "prohibitions", []) or []:
        if nm in ("in_cell", "passed_through") and ar:
            try:
                check.add(int(ar[-1]))
            except (ValueError, TypeError):
                pass
    vc = (scene_filter or {}).get("via_cell")
    if vc is not None:
        check.add(int(vc))
    if metric_cell is not None:
        check.add(int(metric_cell))
    return check


def build_eval_metrics(*, e_states, action_len, last_metrics, constraint,
                       scene_filter, metric_cell, scene_offset, pool_size, n_evals, seed,
                       wm_pred_err=None, wm_latent_err=None,
                       wm_pred_err_steps=None, wm_latent_err_steps=None):
    """Return the per-eval metrics dict for eval_metrics.json. e_states: (n_evals, T, 31) executed
    ground-truth states (or None). action_len: (n_evals,) strokes-to-success (inf if unsolved)."""
    m = last_metrics or {}
    check = _checked_cells(constraint, scene_filter, metric_cell)

    # --- law abidance: cube footprint in any checked cell, per frame; frame 0 grandfathered ---
    violated, ill_frames, ill_frac = [], [], []
    if check and e_states is not None:
        occ = _occupancy(np.asarray(e_states)[..., _CUBE_XY])     # (n_evals, T, 9)
        inzone = np.zeros(occ.shape[:2], dtype=bool)
        for c in check:
            inzone |= occ[..., c]
        after = inzone[:, 1:]                                     # drop frame 0 (grandfather the start)
        den = max(after.shape[1], 1)
        violated = [bool(after[i].any()) for i in range(after.shape[0])]
        ill_frames = [int(after[i].sum()) for i in range(after.shape[0])]
        ill_frac = [float(after[i].sum()) / den for i in range(after.shape[0])]

    # --- path efficiency: strokes-to-goal, cell backtracking, wall contacts ---
    n_steps, revisits, bcontacts = [], [], []
    if e_states is not None:
        xy = np.asarray(e_states)[..., _CUBE_XY]                  # (n_evals, T, 2)
        for i in range(xy.shape[0]):
            seq = np.atleast_1d(gm.which_cell(xy[i])).astype(int).tolist()   # per-frame cell
            comp = [seq[0]]                                       # compress consecutive dups -> cell path
            for c in seq[1:]:
                if c != comp[-1]:
                    comp.append(c)
            revisits.append(int(len(comp) - len(set(comp))))     # times a cell was re-entered
            bcontacts.append(int(np.any(np.abs(xy[i]) >= (gm.GRID_HALF - CUBE_HALF), axis=1).sum()))
            n_steps.append(int(action_len[i]) if np.isfinite(action_len[i]) else int(xy.shape[1] - 1))

    arr = lambda x: np.asarray(x).tolist()
    return {
        "n_evals": int(n_evals), "seed": int(seed),
        "n_steps": n_steps,                # committed strokes to success (or full length if unsolved)
        "cell_revisits": revisits,         # # cells re-entered along the executed path (backtracking)
        "boundary_contacts": bcontacts,    # # executed frames with the cube footprint at the wall
        "wm_pred_err": arr(wm_pred_err) if wm_pred_err is not None else [],  # per-eval mean WM 1-step cube-pred err (m)
        "wm_latent_err": arr(wm_latent_err) if wm_latent_err is not None else [],  # per-eval mean WM 1-step latent MSE (decoder-free)
        # per-STEP (ragged: one list per eval, by MPC step) -> eval_sweep aggregates by step index
        "wm_pred_err_steps": [list(map(float, s)) for s in (wm_pred_err_steps or [])],
        "wm_latent_err_steps": [list(map(float, s)) for s in (wm_latent_err_steps or [])],
        "scene_offset": scene_offset,
        "pool_size": pool_size,
        "scene_filter": {k: v for k, v in (scene_filter or {}).items() if v is not None},
        "checked_cells": sorted(check),
        "success": arr(m.get("success", [])), "cubes_correct": arr(m.get("cubes_correct", [])),
        "state_dist": arr(m.get("state_dist", [])), "cube_l2": arr(m.get("cube_l2", [])),
        "law_violated": violated,          # BOOLEAN per eval: footprint entered a checked cell AFTER frame 0
        "illegal_frame_count": ill_frames, # # executed frames with footprint in a checked cell
        "illegal_frame_frac": ill_frac,    # fraction of post-frame-0 frames spent illegal
    }
