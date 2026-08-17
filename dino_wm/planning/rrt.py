"""Kinodynamic RRT over the world model (same state/action space as the CEM/MPC planners).

State = the cube (x,y) read from the probe (objective_fn.position_probe), exactly as the CEM
planners use it. Action = a 4-D planar stroke [start, dx, dy], aimed-contact (start derived from
the push direction), same representation + bounds as the aimed CEM. Edges are WM rollouts: to
extend toward a sampled target we sample a BATCH of aimed strokes from the nearest node, roll
them through the WM (reusing the prefix-from-root rollout of the chained planner), read each
resulting cube via the probe, and keep the one closest to the target. The tree therefore explores
diverse multi-step pushes -- the breadth that escapes the local minima a distance-greedy CEM
gets stuck in.

This is plain RRT (no rewiring). LAW ENFORCEMENT is DELEGATED to the legislation pipeline -- RRT does
NO legal reasoning of its own: it calls constraint.violations() (constraint.py) to prune candidates,
and a per-step LawEvaluator.observe() perceives + records + re-derives the Constraint. The executed
normative memory lives in the legislation LEDGER (per eval), not here. Plans an open-loop root->goal
path, returned like the CEM planners (padded actions + per-eval length) for the evaluator.

Adapted from the pytorch_rrt KinodynamicRRT structure (sample target -> nearest -> batched
propagate -> PRUNE via constraint.violations -> pick-closest -> add node -> goal check).
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from einops import repeat

from utils import move_to_device
from .cem_aimed_contact import AimedContactCEMPlanner

from probes.probe_cube_position import gm  # grid extent for sampling + which_cell

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
    law: str               # str(Constraint) verdict in effect at creation        (the law at the time)
    prefix: torch.Tensor   # (T,4) normalized actions, root -> this node
    parent: int            # parent node index (-1 for root)
    viol: int = 0          # cumulative law violations along root->this node (# of law-breaking strokes)


class RRTPlanner(AimedContactCEMPlanner):
    """Kinodynamic RRT. Subclasses the aimed planner only to reuse its action bounds + caches;
    overrides plan() with the tree search (the CEM opt loop is unused)."""

    def __init__(self, wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run,
                 log_filename="logs.json", max_samples=256, batch_size=64, goal_tol=0.06,
                 goal_bias=0.2, push_min=0.05, push_max=0.09, max_path=20,
                 deviant_lambda=0.067, steer_cone_deg=90.0, **kwargs):
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
        # DEVIANT trade-off knob: meters-of-progress one law-violation is worth. Deviant minimizes a
        # SCALAR cost = distance-to-goal + deviant_lambda * cumulative_violations (extend + final pick),
        # so it crosses a forbidden region only when doing so buys > deviant_lambda per violation, and
        # always takes a 0-violation route when one exists (cost 0). lambda->inf == social (never
        # violate); lambda==0 == rational/off (ignore law). Default 0.067 = half a cell (CELL/2).
        self.deviant_lambda = float(deviant_lambda)
        # STEER cone (deg): _extend samples push directions within +/-(steer_cone_deg/2) of the analytic
        # bearing to the sampled target, instead of blind uniform [-pi,pi]. Concentrates the batch toward
        # the target (no wasted backward pushes) while the +/- spread still searches around the WM's
        # heading error. steer_cone_deg >= 360 -> full uniform (the old blind sampling, for A/B).
        self.steer_cone_deg = float(steer_cone_deg)
        # Normative MEMORY now lives in the legislation LEDGER (LawEvaluator.ledger, per eval), NOT
        # here -- the planner is stateless about law. RRT keeps only an episode step counter for logs.
        self._step = 0
        self._log_queries = False   # introspection: when True, _extend records every sampled candidate
        self._query_log = []        # -> the planner's QUERY distribution (aim/push/predicted move+cell)
        self._t_plan = self._t_reason = self._t_prune = 0.0   # runtime accounting (see reset)

    def reset(self):
        """New episode (called by MPCPlanner): reset the step counter and the legislation ledger."""
        self._step = 0
        self._query_log = []        # accumulate candidate stats across the whole episode's re-plans
        # RUNTIME accounting, accumulated over the episode's re-plans across ALL evals:
        #   _t_plan   = total wall-clock inside plan()   (RRT search + legislation)
        #   _t_reason = LawEvaluator.observe()+set_goal() (LEGISLATION: probe-perceive + ground facts
        #               + clingo DDL reasoning + build the Constraint), per eval per re-plan
        #   _t_prune  = constraint.violations() calls     (LEGISLATION: per-candidate legality check
        #               + its probe reads), per tree extension
        # RRT-only time = _t_plan - _t_reason - _t_prune  (WM rollouts + position-probe tree geometry +
        #               stroke sampling + nearest/tree bookkeeping + path build).
        self._t_plan = self._t_reason = self._t_prune = 0.0
        law_fn = getattr(self, "law_fn", None)
        if law_fn is not None:
            law_fn.reset()

    @torch.no_grad()
    def _extend(self, obs_e, near, target):
        """Sample batch_size aimed strokes from near.pos, roll them through the WM, ask the
        LEGISLATION PIPELINE which candidates are legal, and return (end_pos, new_prefix) for the
        legal candidate closest to target. None if none are legal.

        RRT does NO legal reasoning of its own: legality is entirely self.constraint.violations(...)
        (constraint.py / the DDL pipeline). n_pred = ALL frames -- no frame-0 skip; the current
        DINO-WM frame is handed to the constraint as data like any other. The end-of-stroke cube
        position is still probed HERE, but only for TREE geometry (nearest / target / goal), not law."""
        B = self.batch_size
        dev = self.device
        base = torch.as_tensor(near.pos, dtype=torch.float32, device=dev)             # (2,)
        # STEER: push directions in a cone centered on the analytic bearing to the target (was blind
        # uniform [-pi,pi], which wasted ~half the batch on backward pushes). >=360deg or degenerate
        # (near == target) -> uniform fallback. Exploration is preserved via target selection (goal-or-random).
        tgt = torch.as_tensor(target, dtype=torch.float32, device=dev)                # (2,)
        if self.steer_cone_deg >= 360.0 or float(torch.linalg.norm(tgt - base)) < 1e-6:
            theta = (torch.rand(B, device=dev) * 2 - 1) * math.pi
        else:
            bearing = torch.atan2(tgt[1] - base[1], tgt[0] - base[0])                 # analytic direction to target
            half = math.radians(self.steer_cone_deg) / 2.0                            # +/- half-cone (45deg for 90)
            theta = bearing + (torch.rand(B, device=dev) * 2 - 1) * half
        L = torch.rand(B, device=dev) * (self.push_max - self.push_min) + self.push_min
        dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)              # (B,2)
        start = base - self.aim_back * dirs                                           # behind the cube
        disp = dirs * (self.aim_back + L)[:, None]                                    # end = cube + dir*L
        strokes = torch.cat([start, disp], dim=-1)                                    # (B,4) raw meters
        strokes_m = strokes                                                           # keep METERS for the action-space law (off_grid INTENT)
        strokes = (strokes - self._amean) / self._astd                               # normalize
        strokes = torch.clamp(strokes, self._aim_lo, self._aim_hi)                    # WM training range

        pref_b = near.prefix.unsqueeze(0).expand(B, -1, -1)                           # (B,len,4)
        act = torch.cat([pref_b, strokes.unsqueeze(1)], dim=1)                        # (B,len+1,4)
        obs_b = {k: repeat(v, "1 ... -> b ...", b=B) for k, v in obs_e.items()}
        z_full = self.wm.rollout(obs_0=obs_b, act=act)[0]["visual"]                   # (B,L,P,D) full trajectory
        end_pos = self.objective_fn.position_probe(z_full[:, -1]).detach().cpu().numpy()  # (B,2) cube after stroke -- TREE geometry only

        # LEGALITY: delegated entirely to the injected Constraint (constraint.py). No constraint =
        # rational (no pruning). n_pred = full length -> the constraint sees every frame, current included.
        constraint = getattr(self, "constraint", None)
        # OFF (rational / observe-only): a Constraint may be present so the LawEvaluator can flip the
        # EXTERNAL sign + record the ledger, but the realistic agent must NOT be pruned or steered by it
        # -- skip the legality check so no candidate is ever marked violating (viol stays all-False).
        if constraint is not None and getattr(self, "mode", "social") != "off":
            # skip_current=True: frame 0 is the cube's CURRENT position (context), not a prediction --
            # enforce the law on where the strokes GO, so a grazing current footprint can't freeze it.
            # actions=strokes_m -> the off_grid checker can prune strokes whose INTENDED endpoint is
            # off-grid, even when the WM/sim bounce the cube back on-grid (the outcome-only blind spot).
            _tv = time.perf_counter()               # LEGISLATION (prune): per-candidate legality check + its probes
            viol = constraint.violations(z_full, z_full.shape[1], skip_current=True,
                                         actions=strokes_m).detach().cpu().numpy()
            self._t_prune += time.perf_counter() - _tv
        else:
            viol = np.zeros(B, dtype=bool)
        self._considered += int(viol.size)                                           # cumulative prune tally (per MPC iter)
        self._pruned += int(viol.sum())
        if getattr(self, "_log_queries", False):
            # INTROSPECTION: record every sampled candidate (the planner's QUERY distribution) -- aim
            # |start-cube|, push |disp|, WM-PREDICTED cube move + from/to cell, and whether pruned.
            # Same schema as scripts/stroke_transition_stats.py so it overlays on the TRAINING data.
            bn = np.asarray(near.pos, dtype=np.float32)
            sm = strokes_m.detach().cpu().numpy()                                    # (B,4) meters [start, disp]
            self._query_log.append({
                "aim": np.linalg.norm(sm[:, :2] - bn, axis=1),                       # |start - cube|
                "push": np.linalg.norm(sm[:, 2:4], axis=1),                          # action displacement
                "pred_move": np.linalg.norm(end_pos - bn, axis=1),                   # WM-predicted cube move
                "from_cell": np.full(len(end_pos), int(gm.which_cell(bn))),
                "to_cell": np.array([int(gm.which_cell(end_pos[b])) for b in range(len(end_pos))]),
                "viol": np.asarray(viol, dtype=bool),
            })
        tdist = np.linalg.norm(end_pos - target, axis=1)
        cum = int(near.viol) + viol.astype(np.int64)                                  # path violations if candidate i is taken
        if getattr(self, "mode", "social") == "deviant":
            # DEVIANT: never prune. SCALAR trade-off -- minimize (dist-to-target + lambda*cum_violations).
            # A crossing candidate (adds a violation) wins only if it gets > lambda closer to target than
            # the best non-crossing one; else the legal candidate wins. This lets the tree grow ACROSS a
            # forbidden region when progress justifies it (a strict viol-first sort would never cross).
            best = int(np.argmin(tdist + self.deviant_lambda * cum))
        else:
            # SOCIAL / off: prune law-breakers; pick the legal candidate closest to target.
            tdist[viol] = np.inf
            if not np.isfinite(tdist).any():
                return None                                                          # every extension violates the law
            best = int(tdist.argmin())
        new_prefix = torch.cat([near.prefix, strokes[best:best + 1]], dim=0)          # (len+1,4)
        return end_pos[best], new_prefix, int(cum[best])

    @staticmethod
    def _route_cells(final_node, nodes):
        """Cells along the planned path root->final_node (the agent's intended route) -- post-hoc."""
        chain, n = [], final_node
        while True:
            chain.append(int(n.cell))
            if n.parent < 0:
                break
            n = nodes[n.parent]
        chain.reverse()
        return chain

    @torch.no_grad()
    def _build_tree(self, trans_obs_0, e, root_cube, goal_cube):
        """Grow one tree for eval e; return (final Node, nodes). Caller reads node.prefix / node.pos
        and can parent-walk for the intended route."""
        lo, hi = gm.GRID_CENTER_XY[0] - gm.GRID_HALF, gm.GRID_CENTER_XY[0] + gm.GRID_HALF
        obs_e = {k: v[e:e + 1] for k, v in trans_obs_0.items()}                       # (1, ...)
        law = str(getattr(self, "constraint", None))     # the verdict in effect for this eval's tree
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
            pos, prefix, viol = ext
            node = Node(pos=pos, cell=int(gm.which_cell(pos)), age=len(nodes), law=law,
                        prefix=prefix, parent=near_i, viol=viol)
            nodes.append(node)
            if float(np.linalg.norm(pos - goal_cube)) < self.goal_tol:
                if getattr(self, "mode", "social") != "deviant":
                    return node, nodes                                               # SOCIAL/off: first goal path (already law-abiding)
                if node.viol == 0:
                    return node, nodes                                               # DEVIANT speed early-stop: a 0-viol goal path has cost<=goal_tol and (lambda>=goal_tol) no violating path beats it
                # DEVIANT reached goal WITH violations: keep growing -- a cheaper path (fewer violations
                # or closer) may still appear; the scalar-cost pick below decides at the end.
        # No in-loop return -> rank the whole tree.
        if getattr(self, "mode", "social") == "deviant":
            # DEVIANT: minimize the SCALAR cost over ALL nodes = dist-to-goal + lambda*violations. Unifies
            # "reached, fewest violations" and "best-effort, closest" into ONE criterion -- a violating
            # node that reaches beats a legal node that stops short iff it saves > lambda per violation.
            # Fewest strokes breaks near-ties. No freeze (progress is priced in, not lexicographically last).
            return min(nodes, key=lambda n: (float(np.linalg.norm(n.pos - goal_cube))
                                             + self.deviant_lambda * n.viol, len(n.prefix))), nodes
        # SOCIAL/off: any goal hit already returned in-loop; here nothing reached -> closest legal node.
        return min(nodes, key=lambda n: float(np.linalg.norm(n.pos - goal_cube))), nodes

    def _ensure_goal_bank(self):
        """Lazily load + encode the positive-obligation goal-cell bank (OPT-IN via self.goal_bank_path,
        injected by plan.py from legislation.goal_bank). Returns the GoalBank or None. Encoded once with
        THIS planner's wm/preprocessor/position-probe, so its target xy matches root_cube/goal_cube."""
        path = getattr(self, "goal_bank_path", None)
        if not path:
            return None
        bank = getattr(self, "_goal_bank", None)
        if bank is None:
            from legislation.goal_bank import GoalBank
            bank = GoalBank(path).encode(self.wm, self.preprocessor,
                                         self.objective_fn.position_probe, self.device)
            self._goal_bank = bank
            print(f"[rrt oblige] goal-cell bank loaded from {path} ({bank.n} cells)")
        return bank

    def plan(self, obs_0, obs_g, actions=None):
        _tp0 = time.perf_counter()          # RUNTIME: total plan() wall-clock (RRT search + legislation)
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
            z_goal = self.wm.encode_obs(trans_obs_g)["visual"][:, -1]              # (n_evals, P, D)
            root_cube = probe(z_root).detach().cpu().numpy()
            goal_cube = probe(z_goal).detach().cpu().numpy()
        n_evals = trans_obs_0["visual"].shape[0]

        self._step += 1                           # episode re-plan counter (memory itself lives in the ledger)
        law_fn = getattr(self, "law_fn", None)    # per-step, per-eval legislation evaluator (holds the ledger)

        self._trees = []                          # keep each eval's search nodes (planner-side)
        self._pruned = 0                          # cumulative law-pruned candidate strokes this MPC iter
        self._considered = 0
        paths, finals = [], []
        for e in range(n_evals):
            # STATE-DEPENDENT LAW: perceive eval e's current frame, record it in eval e's LEDGER, and
            # get the Constraint for eval e's current state. Legislation owns the memory + reasoning;
            # RRT just receives the constraint and prunes on it (constraint.violations, in _extend).
            if law_fn is not None:
                _tr = time.perf_counter()               # LEGISLATION (reason): perceive+ground+clingo+build Constraint
                if hasattr(law_fn, "set_goal"):
                    law_fn.set_goal(z_goal[e:e + 1], e)     # perceive the goal cell on the probe stack
                self.constraint = law_fn.observe(z_root[e:e + 1], e)
                self._t_reason += time.perf_counter() - _tr
                print(f"  [rrt law e{e}] facts {law_fn.ledger(e).last_facts()} -> {self.constraint}")
            # POSITIVE OBLIGATION -> objective switch: if a live obligation names a cell (and the real
            # goal doesn't already satisfy it, and it isn't discharged), steer this eval toward that
            # cell's goal image instead of the task goal (a waypoint). Same pipeline, swapped target;
            # re-evaluated every MPC step, so it tracks sign flips + discharges. Opt-in (bank is None else).
            tgt_cube = goal_cube[e]
            wp = None                                # obligation waypoint cell this eval/step (None = real goal)
            bank = self._ensure_goal_bank()
            if (bank is not None and getattr(self, "constraint", None) is not None and law_fn is not None
                    and getattr(self, "mode", "social") != "off"):   # off: never steer toward an obligation waypoint
                from legislation.goal_bank import visited_cells_from_ledger
                _obl = list(getattr(self.constraint, "obligations", []) or [])
                wp = bank.waypoint(_obl, visited_cells_from_ledger(law_fn.ledger(e)),
                                   root_cube[e], int(gm.which_cell(goal_cube[e])))
                if wp is not None:
                    tgt_cube = bank.pos[wp]
                    _exit = bank.exit_obligations(_obl)   # reparative exit_cell(k) duties, if any
                    _duty = (f"exit cells {sorted(_exit)}" if _exit
                             else f"obligation cells {sorted(bank.obligated_cells(_obl))}")
                    # SELECTED: greppable marker in plan.log + recorded in the ledger commit below.
                    print(f"[OBLIGE] step {self._step} e{e}: {_duty} -> steering to WAYPOINT cell {wp} "
                          f"({tgt_cube[0]:+.3f},{tgt_cube[1]:+.3f}) instead of goal cell "
                          f"{int(gm.which_cell(goal_cube[e]))}")
            final_node, nodes = self._build_tree(trans_obs_0, e, root_cube[e], tgt_cube)
            path, final = final_node.prefix, final_node.pos
            # POST-HOC: record the agent's INTENT (committed first stroke + predicted route) in the
            # ledger. Analysis only -- never read during planning; compared offline against the next
            # step's observed outcome (foreseeability / side-effect attribution).
            if law_fn is not None:
                law_fn.commit(e, {
                    "action": path[0].detach().cpu().tolist() if len(path) else None,
                    "intended_route_cells": self._route_cells(final_node, nodes),
                    "predicted_final_cell": int(final_node.cell),
                    "predicted_final_pos": [float(final[0]), float(final[1])],
                    "path_len": int(len(path)),
                    "obligation_waypoint": wp,   # cell the positive-obligation switch steered to (None = real goal)
                })
            self._trees.append(nodes)
            paths.append(path)
            finals.append(final)
            print(f"  [rrt e{e}] step {self._step} | tree {len(nodes)} nodes | path {len(path)} strokes | "
                  f"law {getattr(self, 'constraint', None)} | "
                  f"start ({root_cube[e][0]:+.3f},{root_cube[e][1]:+.3f}) -> "
                  f"reached ({final[0]:+.3f},{final[1]:+.3f}) goal ({goal_cube[e][0]:+.3f},{goal_cube[e][1]:+.3f})")
        pct = (100.0 * self._pruned / self._considered) if self._considered else 0.0
        print(f"[rrt prune] step {self._step}: pruned {self._pruned}/{self._considered} "
              f"candidate strokes ({pct:.0f}%) | law {getattr(self, 'constraint', None)}")

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
        self._t_plan += time.perf_counter() - _tp0
        return out, action_len
