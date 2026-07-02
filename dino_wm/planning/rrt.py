"""Kinodynamic RRT over the world model (same state/action space as the CEM/MPC planners).

State = the cube (x,y) read from the probe (objective_fn.position_probe), exactly as the CEM
planners use it. Action = a 4-D planar stroke [start, dx, dy], aimed-contact (start derived from
the push direction), same representation + bounds as the aimed CEM. Edges are WM rollouts: to
extend toward a sampled target we sample a BATCH of aimed strokes from the nearest node, roll
them through the WM (reusing the prefix-from-root rollout of the chained planner), read each
resulting cube via the probe, and keep the one closest to the target. The tree therefore explores
diverse multi-step pushes -- the breadth that escapes the local minima a distance-greedy CEM
gets stuck in.

This is plain RRT (no rewiring). LAW PRUNING is applied: when a Constraint is injected, every
predicted frame of every candidate is read by the probe, and the ENTIRE candidate is pruned if the
cube footprint ever overlaps an illegal cell (rest or transit). It plans an open-loop root->goal
path and returns it like the CEM planners (padded actions + per-eval length) for the evaluator.

Adapted from the pytorch_rrt KinodynamicRRT structure (sample target -> nearest -> batched
propagate -> PRUNE illegal -> pick-closest -> add node -> goal check).
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from einops import repeat

from utils import move_to_device
from .cem_aimed_contact import AimedContactCEMPlanner


@dataclass
class Node:
    """One RRT node, carrying the probe-syntaxed facts for (later) temporal/deontic reasoning.

    Everything here is what a SENSOR could see: pos from the position probe, cell from
    which_cell(pos). law records the prohibition set in effect WHEN THIS NODE WAS CREATED, so a
    runtime law change leaves older nodes tagged with the old law. age is the node's creation order
    within this re-plan -- a placeholder temporal stamp to be refined into real elapsed time later."""
    pos: np.ndarray        # cube (x,y) meters, from objective_fn.position_probe  (the probe fact)
    cell: int              # which_cell(pos)                                      (the cell fact)
    age: int               # creation order (0 = root); placeholder for a real temporal age
    law: tuple             # illegal cells in effect at creation                  (the law at the time)
    prefix: torch.Tensor   # (T,4) normalized actions, root -> this node
    parent: int            # parent node index (-1 for root)


class RRTPlanner(AimedContactCEMPlanner):
    """Kinodynamic RRT. Subclasses the aimed planner only to reuse its action bounds + caches;
    overrides plan() with the tree search (the CEM opt loop is unused)."""

    def __init__(self, wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run,
                 log_filename="logs.json", max_samples=256, batch_size=64, goal_tol=0.06,
                 goal_bias=0.2, push_min=0.05, push_max=0.09, max_path=20, **kwargs):
        # the CEM hyperparams are unused by RRT; pass placeholders so the parent __init__ is happy
        super().__init__(horizon=1, topk=1, num_samples=batch_size, var_scale=1, opt_steps=1,
                         eval_every=1, wm=wm, action_dim=action_dim, objective_fn=objective_fn,
                         preprocessor=preprocessor, evaluator=evaluator, wandb_run=wandb_run,
                         log_filename=log_filename, **kwargs)
        self.max_samples = int(max_samples)
        self.batch_size = int(batch_size)
        self.goal_tol = float(goal_tol)
        self.goal_bias = float(goal_bias)
        self.push_min = float(push_min)
        self.push_max = float(push_max)
        self.max_path = int(max_path)
        # closed-loop MEMORY: the executed trajectory so far (one entry per re-plan step). The root
        # of each re-plan is the current REAL cube, so appending roots == the executed path. Kept so
        # temporally-dependent laws (CTD / "at most once" / "having entered X you must now Y") can be
        # grounded over the FULL history+future, not just the current frame. Reset per episode.
        self._mem_cubes = []      # list of (n_evals, 2) np arrays, oldest first
        self._mem_latents = []    # list of (n_evals, P, D) tensors (encoded real frames)

    def reset(self):
        """Clear the closed-loop memory at the start of a new episode (called by MPCPlanner)."""
        self._mem_cubes = []
        self._mem_latents = []

    @property
    def memory(self):
        """The executed history so far, per eval. cubes (n_evals, T_hist, 2);
        latents (n_evals, T_hist, P, D). T_hist = number of re-plan steps taken. None if empty."""
        if not self._mem_cubes:
            return {"cubes": None, "latents": None}
        return {"cubes": np.stack(self._mem_cubes, axis=1),
                "latents": torch.stack(self._mem_latents, dim=1)}

    def _illegal_cells(self):
        """Cells forbidden by the CURRENT law (parsed from the injected Constraint's prohibitions),
        as a set of ints. Empty if no constraint -> RRT does not prune (selfish agent)."""
        c = getattr(self, "constraint", None)
        if c is None:
            return set()
        out = set()
        for name, args in getattr(c, "prohibitions", []):
            if name in ("in_cell", "passed_through") and args:
                try:
                    out.add(int(args[-1]))
                except (ValueError, TypeError):
                    pass
        return out

    @torch.no_grad()
    def _legal_mask(self, z_full):
        """Footprint law check: read every predicted frame with the probe and reject a candidate if
        the cube FOOTPRINT (half-extent, via swept_cells -> 'any part of the cube', not just the
        center) overlaps an illegal cell at any STROKE ENDPOINT or in transit between stroke
        endpoints. z_full: (B,L,P,D). Returns (legal (B,) bool, end_pos (B,2)).

        FRAME 0 (the cube's CURRENT position at the re-plan root) is deliberately NOT counted: the
        law governs where the cube GOES, and the 9cm footprint straddles the boundary of the cell it
        is leaving, so counting frame 0 would freeze the agent. (The old code instead grandfathered
        the whole cell out of enforcement -- which turned the prohibition OFF the moment the footprint
        grazed the illegal cell from the adjacent one, letting the cube walk straight through.)"""
        from probes.probe_cube_cells import swept_cells, CUBE_HALF
        B, L = z_full.shape[:2]
        pp = self.objective_fn.position_probe
        pos = np.stack([pp(z_full[:, t]).detach().cpu().numpy() for t in range(L)], axis=1)  # (B,L,2)
        end_pos = pos[:, -1]
        illegal = self._illegal
        if not illegal or L <= 1:
            return np.ones(B, dtype=bool), end_pos
        legal = np.ones(B, dtype=bool)
        for b in range(B):
            ok = True
            for t in range(1, L):                                                    # frames 1..L-1: where strokes LAND (skip root frame 0)
                occ = swept_cells(pos[b, t], pos[b, min(t + 1, L - 1)], CUBE_HALF)    # footprint over [frame t .. next]: endpoint + transit
                if any(occ[c] for c in illegal):
                    ok = False
                    break
            legal[b] = ok
        return legal, end_pos

    @torch.no_grad()
    def _extend(self, obs_e, near, target):
        """Sample batch_size aimed strokes from near.pos, roll them through the WM, PRUNE any whose
        predicted trajectory enters an illegal cell (every frame analyzed -- see _legal_mask), and
        return (end_pos, new_prefix) for the legal candidate closest to target. None if none legal."""
        B = self.batch_size
        dev = self.device
        base = torch.as_tensor(near.pos, dtype=torch.float32, device=dev)             # (2,)
        theta = (torch.rand(B, device=dev) * 2 - 1) * math.pi
        L = torch.rand(B, device=dev) * (self.push_max - self.push_min) + self.push_min
        dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)              # (B,2)
        start = base - self.aim_back * dirs                                           # behind the cube
        disp = dirs * (self.aim_back + L)[:, None]                                    # end = cube + dir*L
        strokes = torch.cat([start, disp], dim=-1)                                    # (B,4) raw meters
        strokes = (strokes - self._amean) / self._astd                               # normalize
        strokes = torch.clamp(strokes, self._aim_lo, self._aim_hi)                    # WM training range

        pref_b = near.prefix.unsqueeze(0).expand(B, -1, -1)                           # (B,len,4)
        act = torch.cat([pref_b, strokes.unsqueeze(1)], dim=1)                        # (B,len+1,4)
        obs_b = {k: repeat(v, "1 ... -> b ...", b=B) for k, v in obs_e.items()}
        z_full = self.wm.rollout(obs_0=obs_b, act=act)[0]["visual"]                   # (B,L,P,D) full trajectory
        legal, end_pos = self._legal_mask(z_full)                                     # PRUNE law-violating candidates (DDL)
        ongrid = np.all(np.abs(end_pos) <= self._grid_bound, axis=1)                 # keep the cube footprint on the grid (workspace bound)
        self._considered += int(legal.size)                                          # cumulative prune tally (per MPC iter)
        self._pruned += int((~legal).sum())                                          # law-pruned
        self._offgrid += int((legal & ~ongrid).sum())                                # off-grid-pruned (feasibility, not law)
        feasible = legal & ongrid
        tdist = np.linalg.norm(end_pos - target, axis=1)
        tdist[~feasible] = np.inf
        if not np.isfinite(tdist).any():
            return None                                                              # no legal + on-grid extension toward target
        best = int(tdist.argmin())
        new_prefix = torch.cat([near.prefix, strokes[best:best + 1]], dim=0)          # (len+1,4)
        return end_pos[best], new_prefix

    @torch.no_grad()
    def _build_tree(self, trans_obs_0, e, root_cube, goal_cube):
        """Grow one tree for eval e; return (path actions (T,4) tensor, final cube (2,), nodes)."""
        from probes.probe_cube_position import gm  # grid extent for sampling + which_cell
        lo, hi = gm.GRID_CENTER_XY[0] - gm.GRID_HALF, gm.GRID_CENTER_XY[0] + gm.GRID_HALF
        obs_e = {k: v[e:e + 1] for k, v in trans_obs_0.items()}                       # (1, ...)
        law = tuple(sorted(self._illegal))
        root = Node(pos=np.asarray(root_cube, dtype=np.float32),
                    cell=int(gm.which_cell(np.asarray(root_cube, dtype=np.float32))),
                    age=0, law=law, prefix=torch.zeros(0, self.action_dim, device=self.device), parent=-1)
        nodes = [root]

        for _ in range(self.max_samples):
            target = goal_cube if np.random.rand() < self.goal_bias else \
                np.random.uniform(lo, hi, size=2).astype(np.float32)
            arr = np.stack([n.pos for n in nodes])
            near_i = int(np.linalg.norm(arr - target, axis=1).argmin())
            if len(nodes[near_i].prefix) >= self.max_path:
                continue
            ext = self._extend(obs_e, nodes[near_i], target)
            if ext is None:                                                          # all extensions illegal -> drop
                continue
            pos, prefix = ext
            node = Node(pos=pos, cell=int(gm.which_cell(pos)), age=len(nodes), law=law,
                        prefix=prefix, parent=near_i)
            nodes.append(node)
            if float(np.linalg.norm(pos - goal_cube)) < self.goal_tol:
                return node.prefix, node.pos, nodes                                  # reached goal
        best = min(nodes, key=lambda n: float(np.linalg.norm(n.pos - goal_cube)))    # best effort
        return best.prefix, best.pos, nodes

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        probe = getattr(self.objective_fn, "position_probe", None)
        if probe is None:
            raise RuntimeError("RRTPlanner needs objective_fn.position_probe (use a probe objective).")
        # caches used by _extend (action normalization + bounds), mirroring the aimed planner
        self._amean = self.preprocessor.action_mean.to(self.device).reshape(-1).float()
        self._astd = self.preprocessor.action_std.to(self.device).reshape(-1).float()
        self._aim_lo, self._aim_hi = self._action_bounds()

        with torch.no_grad():
            z_root = self.wm.encode_obs(trans_obs_0)["visual"][:, -1]              # (n_evals, P, D)
            root_cube = probe(z_root).detach().cpu().numpy()
            goal_cube = probe(self.wm.encode_obs(trans_obs_g)["visual"][:, -1]).detach().cpu().numpy()
        n_evals = trans_obs_0["visual"].shape[0]

        # MEMORY: append this step's REAL root to the executed-trajectory history (see __init__).
        self._mem_cubes.append(root_cube)
        self._mem_latents.append(z_root.detach())

        from probes.probe_cube_cells import CUBE_HALF as _CUBE_HALF
        from probes.probe_cube_position import gm as _gm
        self._grid_bound = _gm.GRID_HALF - _CUBE_HALF   # cube-CENTER bound so the footprint stays on the grid
        self._illegal = self._illegal_cells()    # cells the current law forbids (drives the prune)
        self._trees = []                          # keep each eval's nodes (with pos/cell/age/law facts)
        self._pruned = 0                          # cumulative law-pruned candidate strokes this MPC iter
        self._offgrid = 0                         # cumulative off-grid-pruned (feasibility bound)
        self._considered = 0
        paths, finals = [], []
        for e in range(n_evals):
            path, final, nodes = self._build_tree(trans_obs_0, e, root_cube[e], goal_cube[e])
            self._trees.append(nodes)
            paths.append(path)
            finals.append(final)
            print(f"  [rrt e{e}] hist {len(self._mem_cubes)} steps | tree {len(nodes)} nodes | "
                  f"path {len(path)} strokes | illegal cells {sorted(self._illegal)} | "
                  f"start ({root_cube[e][0]:+.3f},{root_cube[e][1]:+.3f}) -> "
                  f"reached ({final[0]:+.3f},{final[1]:+.3f}) goal ({goal_cube[e][0]:+.3f},{goal_cube[e][1]:+.3f})")
        pct = (100.0 * self._pruned / self._considered) if self._considered else 0.0
        print(f"[rrt prune] MPC iter {len(self._mem_cubes)}: {self._pruned} law-pruned ({pct:.0f}%) "
              f"+ {self._offgrid} off-grid-pruned / {self._considered} candidates | law {sorted(self._illegal)}")

        # pad each path to T_max with a HOLD (zero-displacement stroke at the final cube)
        T_max = max((len(p) for p in paths), default=1) or 1
        out = torch.zeros(n_evals, T_max, self.action_dim, device=self.device)
        action_len = np.zeros(n_evals, dtype=np.int64)
        for e, (p, fc) in enumerate(zip(paths, finals)):
            action_len[e] = len(p)
            if len(p):
                out[e, :len(p)] = p
            hold = (torch.tensor([fc[0], fc[1], 0.0, 0.0], device=self.device) - self._amean) / self._astd
            if len(p) < T_max:
                out[e, len(p):] = hold
        return out, action_len
