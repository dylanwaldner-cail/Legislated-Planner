"""Sim-oracle RRT: byte-for-byte the same aimed-stroke RRT + legislation as planning/rrt.py +
planning/mpc.py, with EXACTLY ONE difference (the independent variable):

    WM rollout + probe reads   -->   SIM rollout (PhysGridEnv, physics only) + GROUND-TRUTH reads.

Everything else is matched to the WM pipeline: stroke sampling, the action-range clamp, the legal
check (Constraint occ+transit, skip_current=True, over the full root->end trajectory), social-prune
vs deviant-lexsort selection, the strict fewest-violation goal pick, and the closed-loop MPC (rebuild
the tree each step, commit the first stroke / a HOLD, success = goal-cell match, action_len=iter+1).
Hypers mirror conf/planner/mpc_rrt.yaml.

How the IV is realized without breaking the match:
  * root/goal cube come from the SIM STATE (GT), not objective_fn.position_probe(z).
  * the Constraint is reused UNCHANGED; its `cube_cells` occ path is fed a GT footprint-occupancy
    fn and its transit/off_grid paths are fed GT positions -- so the checker logic is identical,
    only the SOURCE of positions/occupancy is GT instead of WM+probe.
  * the sim resumes from each node's stored full state, so a candidate's full root->end trajectory is
    the node's stored boundary positions + the new GT end (deterministic sim => == a re-roll).

========================= REVIEWER GUIDE (read me first) =========================
This file is the CEILING baseline; its whole value is being a FAITHFUL mirror of the deployed
planner (planning/rrt.py + planning/mpc.py) with exactly ONE change (GT reads, not WM+probe).
When reviewing, verify each correspondence below rather than the code in isolation:

  1. CANDIDATE SET == rrt.py. `_sample_strokes` is byte-for-byte rrt.py `_extend`'s stroke math
     (steer cone, L~U[push_min,push_max], start=base-aim_back*dir, disp=dir*(aim_back+L)). The clamp
     is in RAW meters to actions.pth min/max; that equals rrt.py's normalized clamp to _aim_lo/_aim_hi
     because normalization is per-dim monotonic. If you change either, they must stay in lockstep.
  2. EDGE ROLLOUT == a WM rollout, but exact. rrt.py re-rolls the whole prefix through the WM every
     extend (and accumulates WM error); the oracle instead RESUMES from `near.state` (the exact GT
     sim state after the prefix) and rolls only the new stroke. Deterministic sim => identical to a
     re-roll, and strictly cleaner (no error). This is why every _Node stores its full (31,) state.
  3. LEGALITY == constraint.violations, GT-sourced. `_pick_from_block` calls the SAME Constraint with
     the SAME (skip_current=True, actions=, n_pred=full-length) as rrt.py:153. The ONLY difference:
     positions + occupancy come from GT (`positions=full_t`, occ=_GTOcc) instead of the WM latent +
     probes. Same checkers, same occ_thresh, same Liang-Barsky swept transit. This IS the independent
     variable -- do not add any other GT shortcut into the legal path.
  4. SELECTION == rrt.py. social: prune violators, argmin dist, None if all illegal. deviant: argmin
     (dist + lambda*cum_viol). Goal test + final tree pick mirror rrt.py `_build_tree` exactly.
  5. MPC == mpc.py. Commit the FIRST stroke (or a HOLD), execute, re-plan; success = goal-CELL match
     (== env.grid_venv.eval_state, which is a discrete 3x3 cell match, NOT cube_l2); action_len = the
     MPC step index at success (n_taken=1, so == mpc.py's (iter+1)*n_taken).
  6. METRICS == the SAME build_eval_metrics as the deployed sweep (see run_sim_oracle._record). The
     oracle feeds GT e_states; the swept/occupancy/path metrics are literally the deployed functions.

  CONFIG NOTE for interpreting results: `_make_constraint` passes cube_half=CUBE_HALF with NO margin
  inflation => this is the delta=0 (NO-CUSHION) arm. A cushion would raise abidance. Always report
  the no-cushion number as such.

  KNOWN STRUCTURAL GAP (not a bug): the planner prunes each candidate's PLANNED path per MPC step with
  skip_current grandfathering the CURRENT cube every step and commits only stroke 0; the metric scores
  the EXECUTED committed-stroke chain with a frame-0-only grandfather. So a chain of individually-legal
  strokes can, as an executed sequence, still swept-graze -- which is why a PERFECT WM still grazes.

  PERF NOTE: run_sim_oracle timing shows ~100% of wall is env.roll_strokes (physics); pruning is ~0.
  Speed knobs are all physics-side: stroke_max_steps (substeps/roll), max_samples + patience + max_iter
  (fewer rolls). Per-substep cost is env-count-INDEPENDENT (fixed Kit/manager overhead), so raising
  batch_scenarios K packs more scenarios per roll for ~free; lowering num_envs B does NOT speed a roll.
==================================================================================
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from legislation.constraint import Constraint
from legislation.grounding import gm                     # importlib-loaded grid_metadata (no heavy deps)
from probes.probe_cube_cells import CUBE_HALF

_CUBE_XY = slice(18, 20)


class _GTOcc:
    """GT stand-in for the cube_cells probe: footprint-cell occupancy from a cube (x,y), matching the
    probe's target (|x-cx| < CELL/2+cube_half AND |y-cy| < CELL/2+cube_half). The reused Constraint
    calls it on `latents[:, t]`, which the oracle sets to the GT cube positions -> occ is GT-exact
    instead of a probe read on a WM latent. Returns (B, N_CELLS) float in {0,1} (occ_thresh=0.5)."""

    def __init__(self, cube_half, device):
        self.centers = torch.tensor([gm.cell_center(c) for c in range(gm.N_CELLS)],
                                    dtype=torch.float32, device=device)          # (9,2)
        self.h = gm.CELL / 2.0 + cube_half

    def __call__(self, pos):                                                     # pos (B,2) = GT cube xy
        d = (pos[:, None, :].float() - self.centers).abs()                       # (B,9,2)
        return ((d[..., 0] < self.h) & (d[..., 1] < self.h)).float()             # (B,9)


class _Node:
    __slots__ = ("pos", "state", "prefix", "parent", "viol", "traj")

    def __init__(self, pos, state, prefix, parent, viol, traj):
        self.pos = pos          # (2,) GT cube xy at this node
        self.state = state      # (31,) full sim state (restore point for extensions)
        self.prefix = prefix    # list of (4,) strokes root -> this node
        self.parent = parent    # index into nodes (-1 = root)
        self.viol = viol        # cumulative law violations along the prefix
        self.traj = traj        # (P,2) boundary cube positions root -> this node (P = len(prefix)+1)


class SimOracleRRT:
    def __init__(self, env, reasoner, mode="social", *, action_min=None, action_max=None,
                 batch_size=64, max_samples=512, max_path=3, goal_tol=0.06, goal_bias=0.5,
                 push_min=0.05, push_max=0.09, aim_back=0.12, deviant_lambda=0.067,
                 steer_cone_deg=90.0, patience=0, cube_half=CUBE_HALF, device="cuda:0"):
        self.env = env                     # PhysGridEnv with num_envs == batch_size
        self.reasoner = reasoner
        self.mode = mode                   # social | deviant | off
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.max_path = max_path
        self.goal_tol = goal_tol
        self.goal_bias = goal_bias
        self.push_min, self.push_max = push_min, push_max
        self.aim_back = aim_back
        self.deviant_lambda = deviant_lambda   # DEVIANT scalar trade-off (m/violation); matches rrt.py
        self.steer_cone_deg = float(steer_cone_deg)   # match rrt.py: sample push dirs within +/-(cone/2)
                                                      # of the bearing to the target (>=360 -> uniform)
        self.patience = int(patience)                 # >0: bail a tree-build after this many rounds with NO
                                                      # improvement in the closest active tree's goal-dist
                                                      # (opt-in speedup; 0 = off, full max_samples)
        self.cube_half = cube_half
        self.device = device
        # WM-RRT clamps sampled strokes to the WM's training action range (rrt.py:115); replicate so
        # the CANDIDATE SET is identical. Clamp in raw meters == the normalized clamp (monotonic).
        self.action_min = None if action_min is None else np.asarray(action_min, np.float32)
        self.action_max = None if action_max is None else np.asarray(action_max, np.float32)
        self._occ = _GTOcc(cube_half, device)
        self._t_sim = self._t_prune = 0.0; self._n_extends = 0   # runtime accounting (reset per run)

    def _make_constraint(self, goal_cell):
        """DDL verdict from GT facts (goal cell perceived perfectly) -> reused Constraint whose occ
        path uses the GT footprint-occupancy fn. Static per episode (no sign flip in the oracle). None
        for mode=off (no pruning), matching the WM pipeline (no law_fn -> self.constraint unset)."""
        if self.mode == "off":
            return None
        v = self.reasoner.assess(["cube", f"goal_cell({int(goal_cell)})"])
        return Constraint(v["prohibitions"], {"cube_cells": self._occ}, cube_half=self.cube_half,
                          obligations=v["obligations"], permissions=v["permissions"])

    def _sample_strokes(self, base, target):
        """B aimed strokes from `base` toward `target` (steered cone + WM-range clamp; rrt.py:123-136).
        SHARED by the single _extend and the scenario-batched builder so candidate sets are identical."""
        B = self.batch_size
        if self.steer_cone_deg >= 360.0 or float(np.linalg.norm(target - base)) < 1e-6:
            theta = (np.random.rand(B).astype(np.float32) * 2 - 1) * np.pi
        else:
            bearing = float(np.arctan2(target[1] - base[1], target[0] - base[0]))
            half = np.radians(self.steer_cone_deg) / 2.0
            theta = (bearing + (np.random.rand(B).astype(np.float32) * 2 - 1) * half).astype(np.float32)
        Lp = np.random.rand(B).astype(np.float32) * (self.push_max - self.push_min) + self.push_min
        dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)                 # (B,2)
        start = base[None] - self.aim_back * dirs
        disp = dirs * (self.aim_back + Lp)[:, None]
        strokes = np.concatenate([start, disp], axis=1).astype(np.float32)      # (B,4) raw meters
        if self.action_min is not None:
            strokes = np.clip(strokes, self.action_min, self.action_max)        # == WM-RRT clamp to training range
        return strokes

    def _pick_from_block(self, near, target, constraint, out, strokes):
        """From a ROLLED candidate block (out (B,31) GT states + strokes (B,4)) off `near`, prune/rank
        with the reused Constraint over each candidate's full root->end GT trajectory (skip_current=True)
        and return the chosen _Node (parent unset) or None if all illegal. SHARED single/batched logic."""
        B = strokes.shape[0]
        end = out[:, _CUBE_XY]                                                 # (B,2) GT end cube
        if constraint is not None:
            P = near.traj.shape[0]
            full = np.concatenate([np.broadcast_to(near.traj[None], (B, P, 2)),
                                   end[:, None, :]], axis=1).astype(np.float32)  # (B,L,2), L=P+1
            full_t = torch.as_tensor(full, device=self.device)                 # doubles as `latents` (occ reads [:,t])
            _tp = time.perf_counter()
            viol = constraint.violations(                                      # SAME call as rrt.py, GT-sourced
                full_t, full.shape[1], skip_current=True,
                actions=torch.as_tensor(strokes, device=self.device),
                positions=full_t).detach().cpu().numpy()
            self._t_prune += time.perf_counter() - _tp
        else:
            viol = np.zeros(B, dtype=bool)
        tdist = np.linalg.norm(end - target, axis=1)
        cum = int(near.viol) + viol.astype(np.int64)
        if self.mode == "deviant":
            best = int(np.argmin(tdist + self.deviant_lambda * cum))           # dist + lambda*violations
        else:
            tdist = tdist.copy(); tdist[viol] = np.inf                         # social/off: prune violators
            if not np.isfinite(tdist).any():
                return None
            best = int(tdist.argmin())
        return _Node(end[best].copy(), out[best].copy(), near.prefix + [strokes[best].copy()],
                     -1, int(cum[best]), np.concatenate([near.traj, end[best][None]], axis=0))

    def _extend(self, near, target, constraint):
        """Single-scenario extend: sample B strokes, roll from near.state, pick best (== _sample + roll + _pick)."""
        strokes = self._sample_strokes(near.pos, target)
        _ts = time.perf_counter()
        out = self.env.roll_strokes(near.state, strokes)                        # (B,31) GT sim outcome
        self._t_sim += time.perf_counter() - _ts; self._n_extends += 1
        return self._pick_from_block(near, target, constraint, out, strokes)

    def _build_tree(self, root_state, goal_pos, constraint):
        """One tree from root_state; returns (final_node, nodes). Mirrors planning/rrt.py._build_tree."""
        lo = gm.GRID_CENTER_XY[0] - gm.GRID_HALF
        hi = gm.GRID_CENTER_XY[0] + gm.GRID_HALF
        root_pos = np.asarray(root_state[_CUBE_XY], np.float32)
        root = _Node(root_pos, np.asarray(root_state, np.float32), [], -1, 0, root_pos[None].copy())
        nodes = [root]
        for _ in range(self.max_samples):
            target = goal_pos if np.random.rand() < self.goal_bias else \
                np.random.uniform(lo, hi, size=2).astype(np.float32)
            arr = np.stack([n.pos for n in nodes])
            near_i = int(np.linalg.norm(arr - target, axis=1).argmin())
            if len(nodes[near_i].prefix) >= self.max_path:
                continue
            node = self._extend(nodes[near_i], target, constraint)
            if node is None:
                continue
            node.parent = near_i
            nodes.append(node)
            if float(np.linalg.norm(node.pos - goal_pos)) < self.goal_tol:
                if self.mode != "deviant":
                    return node, nodes                                        # social/off: first goal path
                if node.viol == 0:
                    return node, nodes                                        # deviant speed early-stop: 0-viol goal path (matches rrt.py)
        # No in-loop return -> rank the whole tree.
        if self.mode == "deviant":
            # DEVIANT scalar cost = dist-to-goal + lambda*violations over ALL nodes (matches rrt.py):
            # unifies reached/best-effort, no freeze, crosses only when it saves > lambda per violation.
            return min(nodes, key=lambda n: (float(np.linalg.norm(n.pos - goal_pos))
                                             + self.deviant_lambda * n.viol, len(n.prefix))), nodes
        return min(nodes, key=lambda n: float(np.linalg.norm(n.pos - goal_pos))), nodes  # social/off: closest legal

    def run(self, init_state, goal_state, max_iter=12, seed=0):
        """Closed-loop episode == MPCPlanner.plan with the sim as the world model. Rebuild the tree
        from the real state each step, commit the FIRST stroke (a HOLD if the path is empty), execute
        in the sim, re-plan. Success = goal-cell match (checked AFTER execution). Returns
        (executed_states (T,31), constraint, action_len)."""
        init_state = np.asarray(init_state, np.float32)
        self.env.prepare(seed, init_state)                                    # set the episode home-joint park pose
        goal_pos = np.asarray(goal_state[_CUBE_XY], np.float32)
        goal_cell = int(np.atleast_1d(gm.which_cell(goal_pos[None]))[0])
        constraint = self._make_constraint(goal_cell)
        cur = init_state.copy()
        executed = [cur.copy()]
        action_len, success, it = np.inf, False, 0
        self._t_sim = self._t_prune = 0.0; self._n_extends = 0                  # reset runtime accounting per episode
        _wall0 = time.perf_counter()
        while not success and it < max_iter:
            _step0 = time.perf_counter()
            final, nodes = self._build_tree(cur, goal_pos, constraint)
            stroke = (final.prefix[0] if final.prefix                          # commit first stroke; else HOLD (rrt.py pad)
                      else np.array([cur[18], cur[19], 0.0, 0.0], np.float32))
            _tr = time.perf_counter()
            cur = self.env.roll_strokes(cur, np.tile(stroke, (self.batch_size, 1)))[0].copy()
            self._t_sim += time.perf_counter() - _tr                           # committed-stroke exec is sim time too
            executed.append(cur.copy())
            it += 1
            success = int(np.atleast_1d(gm.which_cell(cur[None, 18:20]))[0]) == goal_cell
            if success:
                action_len = it                                              # (iter+1)*n_taken_actions, n_taken=1
            # per-MPC-step heartbeat (each step rebuilds the whole tree -> minutes; flush for live tracking)
            print(f"    [oracle step {it}/{max_iter}] tree={len(nodes)} path={len(final.prefix)} "
                  f"cube=({cur[18]:+.3f},{cur[19]:+.3f}) goal_dist={float(np.linalg.norm(cur[_CUBE_XY] - goal_pos)):.3f} "
                  f"step={time.perf_counter() - _step0:.1f}s{'  SUCCESS' if success else ''}", flush=True)
        wall = time.perf_counter() - _wall0
        timing = {"wall_s": round(wall, 2), "sim_s": round(self._t_sim, 2),
                  "prune_s": round(self._t_prune, 2), "n_extends": self._n_extends, "n_steps": it}
        return np.stack(executed), constraint, action_len, timing

    # ===================== SCENARIO-BATCHED (parallel across K scenarios) =====================
    def _build_trees_batched(self, cur, goal_pos, goal_cell, constraints, active):
        """Grow K RRT trees IN LOCKSTEP, one shared PhysX batch of K*B envs per extend round (env.num_envs
        MUST == K*B). Per-scenario logic (nearest / sample / prune / pick / goal-stop) is IDENTICAL to
        _build_tree; only the physics roll is shared. Returns finals[k] (_Node) per active scenario."""
        B = self.batch_size; K = cur.shape[0]
        lo = gm.GRID_CENTER_XY[0] - gm.GRID_HALF; hi = gm.GRID_CENTER_XY[0] + gm.GRID_HALF
        trees, finals = [], [None] * K
        for k in range(K):
            rp = np.asarray(cur[k, _CUBE_XY], np.float32)
            trees.append([_Node(rp, np.asarray(cur[k], np.float32), [], -1, 0, rp[None].copy())])
        done = ~np.asarray(active, bool)
        best_dist = np.full(K, np.inf); stall = 0                   # convergence early-stop bookkeeping
        for r in range(self.max_samples):
            if done.all():
                break
            states_KB = np.zeros((K * B, 31), np.float32)
            strokes_KB = np.zeros((K * B, 4), np.float32)
            ctx = [None] * K                                    # (near, near_i, strokes, target) this round
            for k in range(K):
                if done[k]:
                    continue
                target = goal_pos[k] if np.random.rand() < self.goal_bias else \
                    np.random.uniform(lo, hi, size=2).astype(np.float32)
                arr = np.stack([n.pos for n in trees[k]])
                near_i = int(np.linalg.norm(arr - target, axis=1).argmin())
                if len(trees[k][near_i].prefix) >= self.max_path:
                    continue                                    # capped -> skip this scenario this round (dummy block)
                near = trees[k][near_i]
                strokes = self._sample_strokes(near.pos, target)
                states_KB[k * B:(k + 1) * B] = near.state
                strokes_KB[k * B:(k + 1) * B] = strokes
                ctx[k] = (near, near_i, strokes, target)
            _ts = time.perf_counter()
            out = self.env.roll_strokes(states_KB, strokes_KB)  # (K*B,31) ONE shared PhysX batch
            self._t_sim += time.perf_counter() - _ts
            self._n_extends += sum(1 for c in ctx if c is not None)
            for k in range(K):
                if ctx[k] is None:
                    continue
                near, near_i, strokes, target = ctx[k]
                node = self._pick_from_block(near, target, constraints[k], out[k * B:(k + 1) * B], strokes)
                if node is None:
                    continue
                node.parent = near_i
                trees[k].append(node)
                if float(np.linalg.norm(node.pos - goal_pos[k])) < self.goal_tol:
                    if self.mode != "deviant" or node.viol == 0:    # social/off: first goal; deviant: first 0-viol
                        finals[k] = node; done[k] = True
            # per-tree best goal-dist this round (also drives the heartbeat + convergence early-stop)
            cur = np.array([min((float(np.linalg.norm(n.pos - goal_pos[k])) for n in trees[k]), default=np.inf)
                            if not done[k] else best_dist[k] for k in range(K)])
            improved = (cur < best_dist - 1e-4)
            best_dist = np.minimum(best_dist, cur)
            stall = 0 if improved.any() else stall + 1              # any active tree got closer -> reset
            if (r + 1) % 10 == 0 and not done.all():                # intra-build heartbeat (is progress happening?)
                act = ~done
                print(f"      [build {r + 1}/{self.max_samples}] trees_done={int(done.sum())}/{K} "
                      f"mean_best_goal_dist={float(np.mean(cur[act])):.3f} (goal_tol={self.goal_tol}) "
                      f"stall={stall} | extends={self._n_extends} sim={self._t_sim:.1f}s", flush=True)
            if self.patience and stall >= self.patience:            # opt-in: no tree improved in `patience` rounds
                print(f"      [build {r + 1}/{self.max_samples}] EARLY-STOP: no goal-dist improvement in "
                      f"{self.patience} rounds (trees_done={int(done.sum())}/{K})", flush=True)
                break
        for k in range(K):                                      # non-goal-reached -> rank whole tree (as _build_tree)
            if not active[k] or finals[k] is not None:
                continue
            if self.mode == "deviant":
                finals[k] = min(trees[k], key=lambda n: (float(np.linalg.norm(n.pos - goal_pos[k]))
                                                         + self.deviant_lambda * n.viol, len(n.prefix)))
            else:
                finals[k] = min(trees[k], key=lambda n: float(np.linalg.norm(n.pos - goal_pos[k])))
        return finals

    def run_batch(self, init_states, goal_states, max_iter=12, seed=0):
        """Closed-loop MPC over K scenarios IN PARALLEL, sharing one K*B-env PhysX batch (env.num_envs
        MUST == K*B). Algorithm per scenario == run(); the tree-build + committed-exec physics are
        batched for ~Kx throughput on one GPU. Returns (list of e_states (Ti,31) per scenario, list of
        constraints, action_len (K,), timing)."""
        init_states = np.asarray(init_states, np.float32); goal_states = np.asarray(goal_states, np.float32)
        K = init_states.shape[0]; B = self.batch_size
        assert self.env.num_envs == K * B, f"env.num_envs={self.env.num_envs} must equal K*B={K * B}"
        self.env.prepare(seed, np.repeat(init_states, B, axis=0))               # per-env home/park (arm identical)
        print(f"    [run_batch] env prepared ({K * B} envs); grounding {K} constraints...", flush=True)
        goal_pos = goal_states[:, _CUBE_XY]                                     # (K,2)
        goal_cell = np.atleast_1d(gm.which_cell(goal_pos)).astype(int)          # (K,)
        constraints = [self._make_constraint(int(goal_cell[k])) for k in range(K)]
        print(f"    [run_batch] constraints built; starting MPC (max_iter={max_iter})...", flush=True)
        cur = init_states.copy()
        executed = [[cur[k].copy()] for k in range(K)]
        success = np.zeros(K, bool); action_len = np.full(K, np.inf); it = 0
        self._t_sim = self._t_prune = 0.0; self._n_extends = 0
        _wall0 = time.perf_counter()
        while not success.all() and it < max_iter:
            active = ~success
            finals = self._build_trees_batched(cur, goal_pos, goal_cell, constraints, active)
            states_KB = np.zeros((K * B, 31), np.float32); strokes_KB = np.zeros((K * B, 4), np.float32)
            for k in range(K):                                  # commit first stroke (or HOLD) per scenario
                f = finals[k]
                stroke = (f.prefix[0] if (active[k] and f is not None and len(f.prefix))
                          else np.array([cur[k, 18], cur[k, 19], 0.0, 0.0], np.float32))
                states_KB[k * B:(k + 1) * B] = cur[k]; strokes_KB[k * B:(k + 1) * B] = stroke
            _ts = time.perf_counter()
            out = self.env.roll_strokes(states_KB, strokes_KB)  # execute all K committed strokes in one batch
            self._t_sim += time.perf_counter() - _ts
            it += 1
            for k in range(K):
                if success[k]:
                    continue
                cur[k] = out[k * B].copy()                       # first env of block k (B copies identical)
                executed[k].append(cur[k].copy())
                if int(np.atleast_1d(gm.which_cell(cur[k, None, 18:20]))[0]) == goal_cell[k]:
                    success[k] = True; action_len[k] = it
            print(f"    [oracle-batch step {it}/{max_iter}] active={int((~success).sum())}/{K} "
                  f"extends={self._n_extends} sim={self._t_sim:.1f}s prune={self._t_prune:.1f}s", flush=True)
        wall = time.perf_counter() - _wall0
        timing = {"wall_s": round(wall, 2), "sim_s": round(self._t_sim, 2), "prune_s": round(self._t_prune, 2),
                  "n_extends": self._n_extends, "n_steps": it, "K": K}
        return [np.stack(ex) for ex in executed], constraints, action_len, timing
