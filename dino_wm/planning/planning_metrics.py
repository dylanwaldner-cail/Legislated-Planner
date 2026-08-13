"""Per-eval planning metrics (dumped to eval_metrics.json, aggregated by scripts/eval_sweep.py).

Pulled out of plan.py to keep the planning workspace lean. `build_eval_metrics(...)` takes the
GROUND-TRUTH executed states + the run's context and returns the metrics dict. Everything here is
state-based (no probe assumption). All array fields are per-eval lists (length == n_evals):

  * task success / cube_l2 / state_dist / cubes_correct  -- from the evaluator's rollout metrics
  * law abidance (law_violated / illegal_frame_*)        -- cube FOOTPRINT in any CHECKED cell after
    frame 0 (frame 0 grandfathered). Checked cells = the constraint's forbidden cells UNION
    scene_filter.via_cell UNION metric_cell (so the RATIONAL agent is still scored on the cell of
    interest, and one metric cell can span many init/goal pairs).
  * path efficiency: n_steps (strokes to goal), cell_revisits (backtracking), boundary_contacts.
"""
from __future__ import annotations

import heapq

import numpy as np
import torch

from scripts.scene_index import _occupancy          # AABB FOOTPRINT occupancy (matches the law)
from probes.probe_cube_position import gm            # which_cell / GRID_HALF
from probes.probe_cube_cells import (CUBE_HALF, swept_cells, _seg_aabb_hit,  # footprint sweep + seg-AABB test
                                      stroke_breach_geometry)                 # per-stroke breach geometry

_CUBE_XY = slice(18, 20)                              # cube (x,y) within the 31-D state
_PEAK_NS = 40                                         # samples along each rest->rest transit for peak_swept_overlap


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
    err = pred = real = None
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
    return err, latent_err, imagined, pred, real



def _overlap_fraction(xy, cells, cube_half):
    """(...,2) cube centers -> (...,) fraction of the cube's AABB footprint AREA inside the union of
    `cells` (a 3x3 grid, cells disjoint so per-cell overlaps just sum). 0 = clear, 1 = cube fully in
    the zone. This is the FLAGRANCY of a violation (how much of the cube is in), vs the boolean
    'is any part in' that illegal_frame_frac uses."""
    xy = np.asarray(xy, float)
    ch = (2.0 * gm.GRID_HALF / 3.0) / 2.0                     # cell half-width (3x3 grid)
    cxlo, cxhi = xy[..., 0] - cube_half, xy[..., 0] + cube_half
    cylo, cyhi = xy[..., 1] - cube_half, xy[..., 1] + cube_half
    area = (2.0 * cube_half) ** 2
    frac = np.zeros(xy.shape[:-1], float)
    for c in cells:
        ccx, ccy = gm.cell_center(int(c))
        ix = np.clip(np.minimum(cxhi, ccx + ch) - np.maximum(cxlo, ccx - ch), 0, None)
        iy = np.clip(np.minimum(cyhi, ccy + ch) - np.maximum(cylo, ccy - ch), 0, None)
        frac = frac + (ix * iy) / area
    return np.clip(frac, 0.0, 1.0)


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


def _optimal_law_path_len(p0, p1, cells, cube_half, eps=1e-6):
    """Geometric shortest 2D path length p0->p1 that keeps the cube CENTER out of every cell in `cells`
    (each an axis-aligned obstacle = the cell AABB inflated by cube_half, so the FOOTPRINT never enters
    -- matches the in_cell prohibition). Visibility graph over the obstacle corners + Dijkstra: NO
    planner, NO sim, ~microseconds. A LOWER BOUND on the achievable cube path (ignores stroke
    discretization / dynamics / reachability). Straight-line fallback if an endpoint is inside an obstacle."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    straight = float(np.linalg.norm(p1 - p0))
    cells = list(cells or [])
    if not cells:
        return straight
    H = gm.CELL / 2.0 + cube_half
    rects = []
    for c in cells:
        cx, cy = gm.cell_center(int(c))
        rects.append((np.array([cx - H, cy - H]), np.array([cx + H, cy + H])))
    in_rect = lambda p, r: r[0][0] < p[0] < r[1][0] and r[0][1] < p[1] < r[1][1]
    if any(in_rect(p1, r) for r in rects):
        return straight                                      # GOAL inside a forbidden cell -> no legal path can END there
    # GRANDFATHER THE START: if p0 begins inside an illegal cell it may exit it (not the agent's fault,
    # matching the law-abidance spawn grandfather), so drop that cell from the obstacles -- the optimal
    # path escapes it, then routes around the rest. (Permissive is fine: this is a LOWER bound.)
    rects = [r for r in rects if not in_rect(p0, r)]
    if not rects:
        return straight                                      # only the start cell was in the way -> straight out
    # edge valid iff its segment doesn't cross any obstacle INTERIOR (rect shrunk by eps so corners pass)
    blocked = lambda a, b: any(_seg_aabb_hit(np.asarray(a, float), np.asarray(b, float), lo + eps, hi - eps)
                               for lo, hi in rects)
    if not blocked(p0, p1):
        return straight                                      # direct line already clears the law
    nodes = [p0, p1]                                         # visibility-graph nodes: endpoints + obstacle corners
    for lo, hi in rects:
        nodes += [np.array([lo[0], lo[1]]), np.array([lo[0], hi[1]]),
                  np.array([hi[0], lo[1]]), np.array([hi[0], hi[1]])]
    N = len(nodes)
    dist = [np.inf] * N; dist[0] = 0.0
    pq = [(0.0, 0)]
    while pq:                                                # Dijkstra to the goal node (index 1)
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in range(N):
            if v == u or blocked(nodes[u], nodes[v]):
                continue
            w = d + float(np.linalg.norm(nodes[v] - nodes[u]))
            if w < dist[v]:
                dist[v] = w
                heapq.heappush(pq, (w, v))
    return dist[1] if np.isfinite(dist[1]) else straight


def build_eval_metrics(*, e_states, action_len, last_metrics, constraint,
                       scene_filter, metric_cell, scene_offset, pool_size, n_evals, seed,
                       wm_pred_err=None, wm_latent_err=None,
                       wm_pred_err_steps=None, wm_latent_err_steps=None, runtime_breakdown=None,
                       wm_pred_xy_steps=None, wm_real_xy_steps=None, wm_probe_start_xy_steps=None,
                       goal_states=None):
    """Return the per-eval metrics dict for eval_metrics.json. e_states: (n_evals, T, 31) executed
    ground-truth states (or None). action_len: (n_evals,) strokes-to-success (inf if unsolved)."""
    m = last_metrics or {}
    check = _checked_cells(constraint, scene_filter, metric_cell)

    # --- law abidance (all ground-truth) --------------------------------------------------------
    # Violation detection is FOOTPRINT-based (edges included: _occupancy uses cell-half + cube-half).
    # Grandfather = FRAME 0 ONLY (the spawn / initial condition): every checked cell is scored from
    # frame 1 on, so the cube's STARTING footprint never counts (we don't punish where it was placed),
    # but any footprint in a checked cell from the first stroke onward does. A cube spawned in an
    # illegal cell gets exactly ONE stroke to clear it -- out by frame 1 or it's flagged.
    #   spawn in 1, drive THROUGH 4      -> violation
    #   spawn in 4, out by frame 1       -> NOT a violation (its one escape stroke)
    #   spawn in 4, still in at frame 1  -> violation (failed to escape in its one frame)
    violated, ill_frames, ill_frac, overlap, violated_swept, violated_center = [], [], [], [], [], []
    peak_frame_overlap, peak_swept_overlap = [], []             # per-eval WORST-moment intrusion (rest / transit); vs the mean `overlap`
    illegal_frames = []                                          # per-eval per-frame footprint-in-checked-cell mask (grandfather-aware)
    overlap_frames = []                                          # per-eval per-frame overlap AREA fraction (which strokes contributed the flagrancy)
    cube_xy_trim = []                                            # per-eval boundary xy, trimmed at goal-hit (drops holding frames)
    violation_approach = []                                      # per-eval: each violating stroke's approach angle (deg) to the cell CENTER
    violation_approach_edge = []                                 # per-eval: same, but angle to the NEAREST cell-boundary point
    violation_side = []                                          # per-eval list: signed lateral (m) of the cell center off the stroke line (+/- = side)
    if check and e_states is not None:
        xy_all = np.asarray(e_states)[..., _CUBE_XY]              # (n_evals, T, 2)
        occ = _occupancy(xy_all)                                 # (n_evals, T, 9) footprint occupancy
        T = occ.shape[1]
        checkl = sorted(check)
        for i in range(xy_all.shape[0]):
            # HOLDING-FRAME TRIM: once an eval succeeds it is HELD at the goal (success_hold), so those
            # trailing frames are not steps taken -- score only frames 0..success. action_len = strokes
            # to success (frame index of the goal); inf if unsolved -> keep the full trajectory. Ti = the
            # eval's effective frame count.
            Ti = min(T, int(action_len[i]) + 1) if np.isfinite(action_len[i]) else T
            Ti = max(Ti, 1)
            den_i = max(Ti - 1, 1)
            # GRANDFATHER FRAME 0 ONLY (the spawn = initial condition): every checked cell is scored
            # from frame 1 on, so the cube's INITIAL footprint (spawn cell, or an edge-graze at spawn)
            # never counts -- but a cube that spawned in an illegal cell gets exactly ONE stroke
            # (frame 0 -> 1) to clear it; still inside at frame 1 onward counts.
            counted = np.zeros(Ti, dtype=bool)                   # frames (0..Ti-1) genuinely in a checked cell
            for c in checkl:
                counted[1:] |= occ[i, 1:Ti, c]                   # drop frame 0 (spawn); count frames 1..Ti-1
            violated.append(bool(counted.any()))
            ill_frames.append(int(counted.sum()))
            ill_frac.append(float(counted.sum()) / den_i)
            illegal_frames.append([bool(x) for x in counted])    # per-frame illegal flag (for frame-risk + shooting-chart plots)
            ov = _overlap_fraction(xy_all[i, :Ti], checkl, CUBE_HALF)  # (Ti,) area fraction in checked cells
            ov_masked = np.where(counted, ov, 0.0)               # grandfather frame 0 + only counted frames contribute
            overlap.append(float(ov_masked[1:].mean()) if Ti > 1 else 0.0)
            overlap_frames.append([float(x) for x in ov_masked]) # per-frame overlap area (see if it's one stroke or several)
            cube_xy_trim.append(xy_all[i, :Ti].tolist())         # trimmed boundary xy (aligns with the trimmed frame arrays)
            # PEAK intrusion (worst-moment flagrancy, vs the mean `overlap` reports): deepest the
            # footprint pokes into any checked cell at a REST frame (peak_frame) and along the TRANSIT
            # between rest frames (peak_swept, _PEAK_NS samples/segment). peak_swept >= peak_frame; the
            # gap is a mid-stroke drive-through the per-frame metric misses -- separates a boundary GRAZE
            # (peak ~0) from a full drive-THROUGH (peak -> 1). Same frame-0 / spawn-cell grandfather as
            # the swept boolean below (was computed offline in plot_peak_swept_ab125.py; now recorded).
            pk_frame = float(ov_masked[1:].max()) if Ti > 1 else 0.0
            pk_swept = pk_frame
            for c in checkl:
                lo = 1 if occ[i, 0, c] > 0.5 else 0              # spawn footprint IN c -> grandfather its escape transit
                for t in range(lo, Ti - 1):
                    seg = xy_all[i, t] + np.linspace(0.0, 1.0, _PEAK_NS)[:, None] * (xy_all[i, t + 1] - xy_all[i, t])
                    pk_swept = max(pk_swept, float(_overlap_fraction(seg, [c], CUBE_HALF).max()))
            peak_frame_overlap.append(pk_frame)
            peak_swept_overlap.append(pk_swept)
            # SWEPT (transit-aware): catch a pass-through BETWEEN recorded stroke boundaries. Same grace.
            swept_hit = False
            for c in checkl:
                lo = 1 if occ[i, 0, c] > 0.5 else 0              # spawn footprint IN c -> grandfather its escape; else CHECK the first stroke's transit (clip)
                for t in range(lo, Ti - 1):
                    if bool(swept_cells(xy_all[i, t], xy_all[i, t + 1], CUBE_HALF)[c]):
                        swept_hit = True
                        break
                if swept_hit:
                    break
            violated_swept.append(swept_hit)
            # CENTER transit (footprint ignored): did the cube CENTROID path cross a checked cell?
            # swept_cells with cube_half=0 == the center segment vs the cell AABB. Same spawn grace.
            center_hit = False
            for c in checkl:
                lo = 1 if occ[i, 0, c] > 0.5 else 0              # spawn footprint IN c -> grandfather escape; else check first-stroke center transit
                for t in range(lo, Ti - 1):
                    if bool(swept_cells(xy_all[i, t], xy_all[i, t + 1], 0.0)[c]):
                        center_hit = True
                        break
                if center_hit:
                    break
            violated_center.append(center_hit)
            # PER-STROKE BREACH GEOMETRY (toward-vs-around / start-side diagnosis). For every breaching
            # stroke, record its approach angle to the illegal cell (0 = pushing straight AT it, 90 =
            # around/tangent, 180 = away) and the signed lateral offset of the cell off the push line
            # (which side). Same trimmed path + frame-0 grandfather as the swept check above, so these
            # attribute the eval's swept violation to the stroke(s) that caused it. Full per-stroke arrays
            # are recomputable from cube_xy_frames offline (scripts/plot_violation_geometry.py) -- we save
            # only the compact breaching-stroke lists here.
            geom = stroke_breach_geometry(xy_all[i, :Ti], checkl, CUBE_HALF)
            violation_approach.append([g["approach_deg"] for g in geom if g["breach"]])
            violation_approach_edge.append([g["approach_edge_deg"] for g in geom if g["breach"]])
            violation_side.append([g["side"] for g in geom if g["breach"]])

    # --- path efficiency: strokes-to-goal, cell backtracking, wall contacts ---
    n_steps, revisits, bcontacts, plen_to_goal, opt_len, path_eff = [], [], [], [], [], []
    straight_len, law_detour = [], []                            # straight-line (unconstrained) baseline + law-detour cost
    froze = []                                                   # FAILED and barely moved -> gave up / no legal move (vs overshoot)
    _succ_list = m.get("success", [])
    if e_states is not None:
        xy = np.asarray(e_states)[..., _CUBE_XY]                  # (n_evals, T, 2)
        gxy = np.asarray(goal_states)[..., _CUBE_XY] if goal_states is not None else None  # (n_evals, 2)
        gcells = np.atleast_1d(gm.which_cell(gxy)).astype(int) if gxy is not None else None  # goal cell per eval
        checkl = sorted(check)                                   # illegal cells = the law's obstacle
        for i in range(xy.shape[0]):
            seq = np.atleast_1d(gm.which_cell(xy[i])).astype(int)            # per-frame cell (array)
            comp = [int(seq[0])]                                 # compress consecutive dups -> cell path
            for c in seq[1:].tolist():
                if c != comp[-1]:
                    comp.append(c)
            revisits.append(int(len(comp) - len(set(comp))))     # times a cell was re-entered
            bcontacts.append(int(np.any(np.abs(xy[i]) >= (gm.GRID_HALF - CUBE_HALF), axis=1).sum()))
            n_steps.append(int(action_len[i]) if np.isfinite(action_len[i]) else int(xy.shape[1] - 1))
            _si = bool(_succ_list[i]) if i < len(_succ_list) else False
            _travel = float(np.linalg.norm(np.diff(xy[i], axis=0), axis=1).sum())  # total cube travel (m)
            froze.append(bool((not _si) and _travel < gm.CELL))  # failed + < ~1 cell of travel = stuck
            if gcells is not None:
                # TAKEN: cube travel distance (m) from spawn until it FIRST enters the goal cell -- path
                # length, NOT stroke count (n_steps) or final displacement (cube_l2). None if never entered.
                hit = np.where(seq == gcells[i])[0]
                plen = (float(np.linalg.norm(np.diff(xy[i, :int(hit[0]) + 1], axis=0), axis=1).sum())
                        if hit.size else None)
                plen_to_goal.append(plen)
                # THREE lengths init->goal: straight_line <= optimal_law <= actual.
                #   straight_line = unconstrained Euclidean (ignores the law) -- absolute lower bound
                #   optimal_law   = shortest LAW-RESPECTING center path (pure geometry) -- the law's floor
                #   actual (plen) = executed cube travel until it first enters the goal cell
                straight = float(np.linalg.norm(gxy[i] - xy[i, 0]))             # unconstrained lower bound (m)
                straight_len.append(straight)
                opt = _optimal_law_path_len(xy[i, 0], gxy[i], checkl, CUBE_HALF)
                opt_len.append(opt)
                law_detour.append(float(opt / straight) if straight > 1e-9 else None)      # cost of the law (>=1)
                path_eff.append(float(plen / opt) if (plen is not None and opt > 1e-9) else None)  # planner ineff. (>=1)

    arr = lambda x: np.asarray(x).tolist()
    return {
        "n_evals": int(n_evals), "seed": int(seed),
        "n_steps": n_steps,                # committed strokes to success (or full length if unsolved)
        "froze": froze,                    # BOOLEAN per eval: failed AND barely moved (gave up / no legal move)
        "path_len_to_goal": plen_to_goal,  # ACTUAL cube travel (m) until it FIRST enters the goal cell (None if never)
        "straight_line_len": straight_len, # UNCONSTRAINED straight-line init->goal center dist (m); absolute lower bound (ignores law)
        "optimal_path_len": opt_len,       # geometric shortest LAW-RESPECTING init->goal cube-center path (m); law's floor
        "law_detour_ratio": law_detour,    # optimal_law / straight_line (>=1; how much the LAW forces a detour)
        "path_efficiency": path_eff,       # actual / optimal_law (>=1; PLANNER inefficiency; None if unreached)
        "cell_revisits": revisits,         # # cells re-entered along the executed path (backtracking)
        "boundary_contacts": bcontacts,    # # executed frames with the cube footprint at the wall
        "wm_pred_err": arr(wm_pred_err) if wm_pred_err is not None else [],  # per-eval mean WM 1-step cube-pred err (m)
        "wm_latent_err": arr(wm_latent_err) if wm_latent_err is not None else [],  # per-eval mean WM 1-step latent MSE (decoder-free)
        # per-STEP (ragged: one list per eval, by MPC step) -> eval_sweep aggregates by step index
        "wm_pred_err_steps": [list(map(float, s)) for s in (wm_pred_err_steps or [])],
        # per committed MPC step: WM-PREDICTED cube (x,y) the planner judged legality on, and the REAL
        # sim (x,y) it landed at. The gap is what lets the agent circumvent the law (scripts/compare_pred_real.py).
        "wm_pred_xy_steps": [[[float(x), float(y)] for x, y in s] for s in (wm_pred_xy_steps or [])],
        "wm_real_xy_steps": [[[float(x), float(y)] for x, y in s] for s in (wm_real_xy_steps or [])],
        "wm_probe_start_xy_steps": [[[float(x), float(y)] for x, y in s] for s in (wm_probe_start_xy_steps or [])],
        "wm_latent_err_steps": [list(map(float, s)) for s in (wm_latent_err_steps or [])],
        "scene_offset": scene_offset,
        "pool_size": pool_size,
        # planning-time split for this run (seconds, summed over all MPC steps x evals): total plan(),
        # RRT search, and legislation (DDL reason + prune). See rrt.py reset() for the exact accounting.
        "runtime_breakdown": runtime_breakdown,
        "scene_filter": {k: v for k, v in (scene_filter or {}).items() if v is not None},
        "checked_cells": sorted(check),
        "success": arr(m.get("success", [])), "cubes_correct": arr(m.get("cubes_correct", [])),
        "state_dist": arr(m.get("state_dist", [])), "cube_l2": arr(m.get("cube_l2", [])),
        "illegal_overlap_frac": overlap,   # per-eval mean per-frame fraction of the cube AREA in the zone (flagrancy)
        "peak_frame_overlap": peak_frame_overlap,  # per-eval MAX footprint-area fraction in a checked cell at a REST frame (worst moment)
        "peak_swept_overlap": peak_swept_overlap,  # per-eval MAX along TRANSIT between rest frames (>= peak_frame; graze vs drive-through)
        "law_violated": violated,          # BOOLEAN per eval: footprint IN a checked cell at any boundary frame AFTER frame 0
        "law_violated_swept": violated_swept,  # BOOLEAN: footprint SWEPT THROUGH a checked cell between boundaries (transit-aware)
        "law_violated_center": violated_center,  # BOOLEAN: cube CENTROID path swept THROUGH a checked cell (transit-aware, footprint ignored)
        "illegal_frame_count": ill_frames, # # executed frames with footprint in a checked cell
        "illegal_frame_frac": ill_frac,    # fraction of post-frame-0 frames spent illegal (holding frames excluded)
        # per-frame raw data for the frame-risk curve + the top-down "shooting chart" (WHERE breaches happen).
        # All per-frame arrays are TRIMMED at goal-hit per eval (holding frames dropped) -> ragged lengths.
        "cube_xy_frames": (cube_xy_trim if (check and e_states is not None)  # trimmed (n_evals, <=T, 2) boundary xy
                           else (np.asarray(e_states)[..., _CUBE_XY].tolist() if e_states is not None else [])),
        "illegal_frames": illegal_frames,  # (n_evals, <=T) footprint-in-a-checked-cell per frame (grandfather-aware, goal-trimmed; [] if no checked cells)
        "illegal_overlap_frames": overlap_frames,  # (n_evals, <=T) per-frame overlap AREA fraction (which strokes contributed the flagrancy)
        # per-eval list of each BREACHING stroke's approach angle (deg): to the cell CENTER
        # (violation_approach) and to the NEAREST cell-boundary point (violation_approach_edge) -- both
        # tracked since which better predicts breaches isn't obvious. 0 = pushing straight toward, 90 =
        # around/tangent, 180 = away. violation_side = signed lateral offset (m) of the cell off the push
        # line (which side). Empty list = no breaching stroke. See scripts/plot_violation_geometry.py.
        "violation_approach": violation_approach,
        "violation_approach_edge": violation_approach_edge,
        "violation_side": violation_side,
    }
