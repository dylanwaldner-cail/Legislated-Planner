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
"""
from __future__ import annotations

import sys
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
                 cube_half=CUBE_HALF, device="cuda:0"):
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
        self.cube_half = cube_half
        self.device = device
        # WM-RRT clamps sampled strokes to the WM's training action range (rrt.py:115); replicate so
        # the CANDIDATE SET is identical. Clamp in raw meters == the normalized clamp (monotonic).
        self.action_min = None if action_min is None else np.asarray(action_min, np.float32)
        self.action_max = None if action_max is None else np.asarray(action_max, np.float32)
        self._occ = _GTOcc(cube_half, device)

    def _make_constraint(self, goal_cell):
        """DDL verdict from GT facts (goal cell perceived perfectly) -> reused Constraint whose occ
        path uses the GT footprint-occupancy fn. Static per episode (no sign flip in the oracle). None
        for mode=off (no pruning), matching the WM pipeline (no law_fn -> self.constraint unset)."""
        if self.mode == "off":
            return None
        v = self.reasoner.assess(["cube", f"goal_cell({int(goal_cell)})"])
        return Constraint(v["prohibitions"], {"cube_cells": self._occ}, cube_half=self.cube_half,
                          obligations=v["obligations"], permissions=v["permissions"])

    def _extend(self, near, target, constraint):
        """Sample B aimed strokes from near.pos (rrt.py:107-115 recipe + the WM-range clamp), roll them
        through the SIM, and prune/rank with the reused Constraint over each candidate's full root->end
        GT trajectory (skip_current=True). Returns a new _Node (parent unset) or None if all illegal."""
        B = self.batch_size
        base = near.pos
        theta = (np.random.rand(B).astype(np.float32) * 2 - 1) * np.pi          # U[-pi,pi], as rrt.py
        Lp = np.random.rand(B).astype(np.float32) * (self.push_max - self.push_min) + self.push_min
        dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)                 # (B,2)
        start = base[None] - self.aim_back * dirs
        disp = dirs * (self.aim_back + Lp)[:, None]
        strokes = np.concatenate([start, disp], axis=1).astype(np.float32)      # (B,4) raw meters
        if self.action_min is not None:                                         # == WM-RRT's clamp to training range
            strokes = np.clip(strokes, self.action_min, self.action_max)

        out = self.env.roll_strokes(near.state, strokes)                            # (B,31) GT sim outcome
        end = out[:, _CUBE_XY]                                                 # (B,2) GT end cube

        if constraint is not None:
            # Full root->end boundary trajectory per candidate: near.traj (shared prefix) + GT end.
            P = near.traj.shape[0]
            full = np.concatenate([np.broadcast_to(near.traj[None], (B, P, 2)),
                                   end[:, None, :]], axis=1).astype(np.float32)  # (B, L, 2), L=P+1
            full_t = torch.as_tensor(full, device=self.device)                 # doubles as `latents` (occ reads [:,t])
            L = full.shape[1]
            viol = constraint.violations(                                      # SAME call as rrt.py:132, GT-sourced
                full_t, L, skip_current=True,
                actions=torch.as_tensor(strokes, device=self.device),
                positions=full_t).detach().cpu().numpy()
        else:
            viol = np.zeros(B, dtype=bool)

        tdist = np.linalg.norm(end - target, axis=1)
        cum = int(near.viol) + viol.astype(np.int64)
        if self.mode == "deviant":
            best = int(np.argmin(tdist + self.deviant_lambda * cum))           # SCALAR: dist + lambda*violations (matches rrt.py)
        else:
            tdist = tdist.copy(); tdist[viol] = np.inf                         # social/off: prune violators
            if not np.isfinite(tdist).any():
                return None
            best = int(tdist.argmin())
        return _Node(end[best].copy(), out[best].copy(), near.prefix + [strokes[best].copy()],
                     -1, int(cum[best]), np.concatenate([near.traj, end[best][None]], axis=0))

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
        while not success and it < max_iter:
            final, _ = self._build_tree(cur, goal_pos, constraint)
            stroke = (final.prefix[0] if final.prefix                          # commit first stroke; else HOLD (rrt.py pad)
                      else np.array([cur[18], cur[19], 0.0, 0.0], np.float32))
            cur = self.env.roll_strokes(cur, np.tile(stroke, (self.batch_size, 1)))[0].copy()
            executed.append(cur.copy())
            it += 1
            success = int(np.atleast_1d(gm.which_cell(cur[None, 18:20]))[0]) == goal_cell
            if success:
                action_len = it                                              # (iter+1)*n_taken_actions, n_taken=1
        return np.stack(executed), constraint, action_len
