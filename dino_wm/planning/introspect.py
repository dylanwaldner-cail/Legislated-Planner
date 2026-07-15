"""RRT/MPC introspection: expose what the planner reasoned over each re-plan step.

Turned on with `rrt_introspect.enabled=true` in conf/plan.yaml. MPCPlanner calls
`RRTIntrospector.record_step(...)` once per MPC iteration (see planning/mpc.py), AFTER the
committed stroke is executed but while `cur_state`/`cur_obs_0` still hold the tree ROOT, so we
can re-simulate branches from the same root the tree was built from.

Two tiers (both guarded -- a failure logs and skips, never crashes planning):

  TIER 1 (no extra sim): for the top-K branches of each eval's tree (ranked by final distance to
    goal) dump, per step:
      * probe-xy path (root -> leaf)   -- what the WM thinks the cube does
      * predicted cell path            -- which_cell of each node
      * committed first stroke         -- the ONLY stroke MPC executes
      * a 2D tree plot (nodes + edges, colored by goal-distance, goal starred, forbidden cell(s)
        shaded, committed branch highlighted)
      * decoded imagined strip         -- decode(WM rollout of the branch prefix): the imagined images

  TIER 2 (needs sim; resim=true): re-run the top-K branches through the REAL sim from the tree
    root and compare imagined vs realized:
      * final-position disparity   ||imagined_leaf - realized_leaf||   (branch-quality error)
      * first-stroke disparity     imagined vs realized cube after stroke 1 (what actually executes)
      * cell-path fidelity         imagined cells vs realized cells (incl. did imagined AVOID a
                                   forbidden cell that realized ENTERED -> legislation-fidelity gap)
      * ranking inversion          does argmin_k(imagined goal-dist) == argmin_k(realized goal-dist)?
                                   (the smoking gun for the WM ranking a bad branch best)

`finalize()` writes rrt_introspect/summary.json with per-step/per-eval metrics + cross-step
oscillation stats (committed direction vs goal direction, direction reversals, net progress).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from probes.probe_cube_position import gm  # which_cell / grid extent / cell_center


def _cells_of(xy_seq):
    return [int(gm.which_cell(np.asarray(p, dtype=np.float32))) for p in xy_seq]


def _branch(node, nodes):
    """Parent-walk root->node: returns (positions [P0..leaf], the leaf node)."""
    chain = []
    n = node
    while True:
        chain.append(n)
        if n.parent < 0:
            break
        n = nodes[n.parent]
    chain.reverse()
    return [np.asarray(c.pos, dtype=np.float32) for c in chain]


class RRTIntrospector:
    def __init__(self, out_dir, top_k=5, resim=True, max_evals=3):
        self.out_dir = out_dir
        self.top_k = int(top_k)
        self.resim = bool(resim)
        self.max_evals = int(max_evals)
        os.makedirs(out_dir, exist_ok=True)
        self.summary = []          # flat list of per-(step,eval) metric dicts
        self._committed_hist = {}   # eval -> list of (root_xy, committed_leaf_xy, goal_xy) per step

    # ------------------------------------------------------------------ helpers
    def _forbidden_cells(self, sub_planner):
        con = getattr(sub_planner, "constraint", None)
        cells = []
        for name, args in getattr(con, "prohibitions", []) or []:
            if name in ("in_cell", "passed_through") and args:
                try:
                    cells.append(int(args[-1]))
                except ValueError:
                    pass
        return cells

    @torch.no_grad()
    def _decode_branch(self, wm, preprocessor, obs_e, prefix, device):
        """Roll a branch prefix (T,4 normalized) from obs_e through the WM and decode ->
        (L,3,H,W) imagined visuals in [-1,1]. None if no decoder or empty prefix."""
        if getattr(wm, "decoder", None) is None or prefix is None or len(prefix) == 0:
            return None
        act = prefix.to(device).unsqueeze(0)                       # (1,T,4)
        z_obses, _ = wm.rollout(obs_0=obs_e, act=act)
        vis = wm.decode_obs(z_obses)[0]["visual"][0]              # (L,3,H,W)
        return vis.detach().cpu()

    def _save_strip(self, tensor_lchw, path):
        try:
            from torchvision import utils
            utils.save_image(tensor_lchw, path, nrow=tensor_lchw.shape[0],
                             normalize=True, value_range=(-1, 1))
        except Exception as e:  # noqa: BLE001
            print(f"[introspect] strip {path} skipped: {e}")

    def _plot_tree(self, nodes, goal_xy, forbidden, top_idx, committed_idx, path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001
            print(f"[introspect] plot skipped (no matplotlib): {e}")
            return
        pos = np.stack([n.pos for n in nodes])
        gd = np.linalg.norm(pos - goal_xy, axis=1)
        fig, ax = plt.subplots(figsize=(5, 5))
        half = gm.GRID_HALF
        cx, cy = gm.GRID_CENTER_XY
        ax.set_xlim(cx - half - 0.03, cx + half + 0.03)
        ax.set_ylim(cy - half - 0.03, cy + half + 0.03)
        # grid + shaded forbidden cells (3x3)
        cw = 2 * half / 3.0
        for c in forbidden:
            ccx, ccy = gm.cell_center(c)
            ax.add_patch(plt.Rectangle((ccx - cw / 2, ccy - cw / 2), cw, cw,
                                       color="red", alpha=0.15, zorder=0))
        for gl in (cx - half, cx - half + cw, cx - half + 2 * cw, cx + half):
            ax.axvline(gl, color="0.85", lw=0.8, zorder=0)
            ax.axhline(gl, color="0.85", lw=0.8, zorder=0)
        # edges
        for i, n in enumerate(nodes):
            if n.parent >= 0:
                p = nodes[n.parent].pos
                ax.plot([p[0], n.pos[0]], [p[1], n.pos[1]], color="0.6", lw=0.6, zorder=1)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=gd, cmap="viridis_r", s=18, zorder=2)
        plt.colorbar(sc, ax=ax, label="dist to goal (m)")
        # committed branch (bold)
        cb = _branch(nodes[committed_idx], nodes)
        cb = np.stack(cb)
        ax.plot(cb[:, 0], cb[:, 1], color="tab:orange", lw=2.2, zorder=3, label="committed")
        ax.scatter(*pos[0], marker="o", s=90, edgecolor="k", facecolor="white", zorder=4, label="root")
        ax.scatter(*goal_xy, marker="*", s=220, color="tab:green", zorder=5, label="goal")
        ax.set_title(os.path.basename(path).replace(".png", ""))
        ax.legend(loc="upper right", fontsize=7)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def record_step(self, *, step, sub_planner, wm, evaluator, objective_fn,
                    cur_obs_0, cur_state, obs_g):
        self._sub_planner = sub_planner   # kept for finalize()'s query-distribution aggregation
        try:
            self._record(step=step, sub_planner=sub_planner, wm=wm, evaluator=evaluator,
                         objective_fn=objective_fn, cur_obs_0=cur_obs_0, cur_state=cur_state,
                         obs_g=obs_g)
        except Exception as e:  # noqa: BLE001
            print(f"[introspect] step {step} skipped: {type(e).__name__}: {e}")

    @torch.no_grad()
    def _record(self, *, step, sub_planner, wm, evaluator, objective_fn,
                cur_obs_0, cur_state, obs_g):
        trees = getattr(sub_planner, "_trees", None)
        probe = getattr(objective_fn, "position_probe", None)
        if trees is None or probe is None:
            print("[introspect] no trees / probe -- skipping")
            return
        dev = next(wm.parameters()).device
        pre = evaluator.preprocessor
        forbidden = self._forbidden_cells(sub_planner)
        # goal cube per eval (probe on encoded goal frame)
        tg = pre.transform_obs(obs_g)
        tg = {k: (v.to(dev) if torch.is_tensor(v) else torch.as_tensor(v).to(dev)) for k, v in tg.items()}
        goal_cube = probe(wm.encode_obs(tg)["visual"][:, -1]).detach().cpu().numpy()  # (N,2)
        t0 = pre.transform_obs(cur_obs_0)
        t0 = {k: (v.to(dev) if torch.is_tensor(v) else torch.as_tensor(v).to(dev)) for k, v in t0.items()}

        n_evals = min(len(trees), self.max_evals)
        # ---- per eval: rank branches, dump tier-1, collect for tier-2 ----
        per_eval_topk = []   # list over evals of list of (leaf_node, branch_positions)
        for e in range(n_evals):
            nodes = trees[e]
            gxy = goal_cube[e]
            order = sorted(range(len(nodes)),
                           key=lambda i: float(np.linalg.norm(nodes[i].pos - gxy)))
            topk = order[: self.top_k]
            committed_idx = order[0]
            per_eval_topk.append([(nodes[i], _branch(nodes[i], nodes)) for i in topk])

            root_xy = np.asarray(nodes[0].pos, dtype=np.float32)
            self._committed_hist.setdefault(e, []).append(
                (root_xy.tolist(), np.asarray(nodes[committed_idx].pos).tolist(), gxy.tolist()))

            print(f"  [introspect s{step} e{e}] tree {len(nodes)} nodes | goal ({gxy[0]:+.3f},{gxy[1]:+.3f}) "
                  f"| forbidden {forbidden}")
            for rank, i in enumerate(topk):
                bxy = _branch(nodes[i], nodes)
                cells = _cells_of(bxy)
                first = None
                if len(nodes[i].prefix):
                    first = pre.denormalize_actions(nodes[i].prefix[:1].cpu().unsqueeze(0)).numpy()[0, 0].tolist()
                gd = float(np.linalg.norm(nodes[i].pos - gxy))
                tag = " <=COMMITTED" if i == committed_idx else ""
                print(f"    rank{rank} len{len(nodes[i].prefix)} goald {gd:.3f} cells {cells} "
                      f"first_stroke {None if first is None else [round(x,3) for x in first]}{tag}")

            # tree plot
            self._plot_tree(nodes, gxy, forbidden, topk, committed_idx,
                            os.path.join(self.out_dir, f"tree_s{step}_e{e}.png"))
            # imagined decode strips for top-k
            obs_e = {k: v[e:e + 1] for k, v in t0.items()}
            for rank, i in enumerate(topk):
                vis = self._decode_branch(wm, pre, obs_e, nodes[i].prefix, dev)
                if vis is not None:
                    self._save_strip(vis, os.path.join(
                        self.out_dir, f"imagined_s{step}_e{e}_rank{rank}.png"))

        # ---- tier 2: re-sim top-k branches from the tree root ----
        realized = None
        if self.resim:
            realized = self._resim_topk(step, sub_planner, evaluator, cur_state,
                                        per_eval_topk, n_evals)

        # ---- metrics ----
        for e in range(n_evals):
            gxy = goal_cube[e]
            rec = {"step": int(step), "eval": int(e), "forbidden": forbidden,
                   "goal_xy": gxy.tolist(), "branches": []}
            imag_gd, real_gd = [], []
            for rank, (leaf, bxy) in enumerate(per_eval_topk[e]):
                b = {"rank": rank, "path_len": len(leaf.prefix),
                     "imagined_leaf": np.asarray(leaf.pos).tolist(),
                     "imagined_cells": _cells_of(bxy),
                     "imagined_goal_dist": float(np.linalg.norm(leaf.pos - gxy))}
                imag_gd.append(b["imagined_goal_dist"])
                if realized is not None and realized[e].get(rank) is not None:
                    rr = realized[e][rank]           # dict: xy_path, leaf_xy, cells
                    b["realized_leaf"] = rr["leaf_xy"]
                    b["realized_cells"] = rr["cells"]
                    b["realized_goal_dist"] = float(np.linalg.norm(np.asarray(rr["leaf_xy"]) - gxy))
                    b["final_disparity"] = float(np.linalg.norm(np.asarray(rr["leaf_xy"]) - leaf.pos))
                    # first-stroke disparity (imagined pos after stroke1 vs realized after stroke1)
                    if len(bxy) > 1 and len(rr["xy_path"]) > 1:
                        b["first_stroke_disparity"] = float(
                            np.linalg.norm(np.asarray(rr["xy_path"][1]) - bxy[1]))
                    b["entered_forbidden_realized"] = bool(set(rr["cells"]) & set(forbidden))
                    b["avoided_forbidden_imagined"] = not bool(set(b["imagined_cells"]) & set(forbidden))
                    real_gd.append(b["realized_goal_dist"])
                else:
                    real_gd.append(None)
                rec["branches"].append(b)
            # ranking inversion: does the imagined-best rank == realized-best rank?
            if any(x is not None for x in real_gd):
                rg = [(k, v) for k, v in enumerate(real_gd) if v is not None]
                rec["imagined_best_rank"] = int(np.argmin(imag_gd))
                rec["realized_best_rank"] = int(min(rg, key=lambda kv: kv[1])[0])
                rec["ranking_inverted"] = rec["imagined_best_rank"] != rec["realized_best_rank"]
            self.summary.append(rec)

    @torch.no_grad()
    def _resim_topk(self, step, sub_planner, evaluator, cur_state, per_eval_topk, n_evals):
        """Run each rank's branch (all evals batched) through the real sim from cur_state.
        Returns realized[e][rank] = {xy_path, leaf_xy, cells} (None if that eval lacks the rank)."""
        pre = evaluator.preprocessor
        env = evaluator.env
        fs = evaluator.frameskip
        N = cur_state.shape[0] if hasattr(cur_state, "shape") else len(cur_state)
        realized = {e: {} for e in range(n_evals)}
        # max branch length across the evals we track
        max_len = max((len(per_eval_topk[e][r][0].prefix)
                       for e in range(n_evals) for r in range(len(per_eval_topk[e]))), default=0)
        if max_len == 0:
            return realized
        for rank in range(self.top_k):
            # build (N, max_len, 4) normalized actions; pad missing/short with HOLD (disp 0 at root cube)
            acts = torch.zeros(N, max_len, 4)
            lengths = {}
            for e in range(N):
                if e < n_evals and rank < len(per_eval_topk[e]):
                    leaf = per_eval_topk[e][rank][0]
                    pfx = leaf.prefix.cpu()
                    L = len(pfx)
                    lengths[e] = L
                    if L:
                        acts[e, :L] = pfx
                    root_xy = np.asarray(per_eval_topk[e][0][0].pos if per_eval_topk[e] else [0, 0])
                else:
                    root_xy = np.zeros(2, dtype=np.float32)
                    lengths[e] = 0
                # HOLD padding: [start=root cube, disp=0], normalized
                if lengths[e] < max_len:
                    hold = torch.tensor([root_xy[0], root_xy[1], 0.0, 0.0], dtype=torch.float32)
                    hold = pre.normalize_actions(hold.reshape(1, 1, 4))[0, 0]
                    acts[e, lengths[e]:] = hold
            act_env = pre.denormalize_actions(acts)                 # (N, max_len, 4)
            act_env = act_env.reshape(N, max_len * fs, 4).numpy() if fs > 1 else act_env.numpy()
            try:
                e_obses, e_states = env.rollout(evaluator.seed, cur_state, act_env)
            except Exception as ex:  # noqa: BLE001
                print(f"[introspect] resim rank{rank} skipped: {type(ex).__name__}: {ex}")
                continue
            st = np.asarray(e_states)                                # (N, T+1, 31)
            for e in range(n_evals):
                if rank >= len(per_eval_topk[e]):
                    continue
                L = lengths[e]
                xy = st[e, : (L * fs + 1), 18:20]                   # realized cube path (only real strokes)
                realized[e][rank] = {
                    "xy_path": xy.tolist(),
                    "leaf_xy": xy[-1].tolist(),
                    "cells": _cells_of(xy),
                }
            # save a realized strip for eval 0..n_evals of this rank (boundary frames)
            try:
                vis = np.asarray(e_obses["visual"])                  # (N, T+1, H, W, 3)
                for e in range(n_evals):
                    frames = vis[e]                                  # (T+1,H,W,3)
                    t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
                    self._save_strip(t, os.path.join(
                        self.out_dir, f"realized_s{step}_e{e}_rank{rank}.png"))
            except Exception as ex:  # noqa: BLE001
                print(f"[introspect] realized strip rank{rank} skipped: {ex}")
        return realized

    def finalize(self):
        # cross-step oscillation stats per eval
        osc = {}
        for e, hist in self._committed_hist.items():
            reversals, progress = 0, []
            prev_dir = None
            for (root, leaf, goal) in hist:
                root, leaf, goal = map(np.asarray, (root, leaf, goal))
                step_vec = leaf - root
                goal_vec = goal - root
                if np.linalg.norm(step_vec) > 1e-4:
                    d = step_vec / (np.linalg.norm(step_vec) + 1e-9)
                    if prev_dir is not None and float(np.dot(d, prev_dir)) < 0:
                        reversals += 1
                    prev_dir = d
                # net progress toward goal from this committed step
                progress.append(float(np.linalg.norm(root - goal) - np.linalg.norm(leaf - goal)))
            osc[str(e)] = {"n_steps": len(hist), "direction_reversals": reversals,
                           "mean_progress_per_step": float(np.mean(progress)) if progress else 0.0}
        qd = self._query_distribution()
        out = {"per_step_eval": self.summary, "oscillation": osc,
               "query_distribution": qd, "top_k": self.top_k, "resim": self.resim}
        path = os.path.join(self.out_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[introspect] wrote {path} | oscillation {osc}")
        if qd:
            print(f"[introspect] query dist: {qd['n_candidates']} candidates | "
                  f"aim~{qd['aim_median']:.3f} push~{qd['push_median']:.3f} "
                  f"pred_move~{qd['pred_move_median']:.3f} | predicted miss {100*qd['predicted_miss_rate']:.0f}% | "
                  f"pruned {100*qd['pruned_rate']:.0f}%  (compare to stroke_transition_stats.py on the training data)")
        return path

    def _query_distribution(self):
        """Aggregate RRT's logged sampled-candidate population (the planner's QUERY distribution) into
        the same schema as scripts/stroke_transition_stats.py, so it overlays on the TRAINING data:
        aim |start-cube|, push |disp|, WM-PREDICTED cube move + from/to-cell transition matrix, and
        predicted-miss / law-pruned rates. Empty if RRT wasn't logging (no _query_log)."""
        ql = getattr(getattr(self, "_sub_planner", None), "_query_log", None)
        if not ql:
            return {}
        cat = lambda k: np.concatenate([q[k] for q in ql])
        aim, push, pmove = cat("aim"), cat("push"), cat("pred_move")
        fc, tc, viol = cat("from_cell"), cat("to_cell"), cat("viol")
        NC = gm.N_CELLS
        ci = lambda c: NC if int(c) == gm.OFF_GRID else int(c)
        mat = np.zeros((NC + 1, NC + 1), dtype=int)
        for a, b in zip(fc, tc):
            mat[ci(a), ci(b)] += 1
        thr = 0.02
        # histograms (fixed edges) + raw COUNTS so batches aggregate EXACTLY across a sweep
        aim_e = np.linspace(0.0, 0.35, 15); push_e = np.linspace(0.0, 0.30, 15); move_e = np.linspace(0.0, 0.20, 15)
        h = lambda a, e: np.histogram(a, bins=e)[0].tolist()
        return {
            "n_candidates": int(aim.size),
            "n_pruned": int(viol.sum()), "n_pred_miss": int((pmove < thr).sum()),   # counts (sum across batches)
            "pruned_rate": float(viol.mean()) if viol.size else 0.0,
            "predicted_miss_rate": float((pmove < thr).mean()),         # WM predicts cube ~unmoved
            "aim_median": float(np.median(aim)), "aim_mean": float(aim.mean()),
            "push_median": float(np.median(push)), "push_mean": float(push.mean()),
            "pred_move_median": float(np.median(pmove)), "pred_move_mean": float(pmove.mean()),
            "aim_hist": h(aim, aim_e), "push_hist": h(push, push_e), "move_hist": h(pmove, move_e),
            "aim_edges": aim_e.tolist(), "push_edges": push_e.tolist(), "move_edges": move_e.tolist(),
            "predicted_transition_matrix": mat.tolist(),                # from-cell -> predicted to-cell
        }
