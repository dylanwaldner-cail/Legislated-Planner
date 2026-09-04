import torch
import hydra
import copy
import time  # per-iteration timing
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner
# Local additions to this upstream DINO-WM planner live in dedicated modules so plan() stays close to
# the original: sign-colour control (exogenous flip + DDL render-back) and the harness instrumentation
# (warm-start, executed-frame stitching, WM 1-step error tracking, introspection, per-iter diagnostics).
from .sign_control import SignController
from . import trace_cfg


def _collect_search(mpc):
    """Drain the sub-planner's per-iter search record into the MPC (opt-in; see trace_cfg).

    Tree nodes are kept as flat arrays, NOT Node objects: `prefix` is a live CUDA tensor and `law`
    repeats the same verdict string for every node in a step, so both are dropped here."""
    sp = mpc.sub_planner
    if trace_cfg.trace_tree():
        for e, nodes in enumerate(getattr(sp, "_trees", []) or []):
            for n in nodes:
                mpc._tree_log.append((int(mpc.iter), int(e), int(n.age), float(n.pos[0]),
                                      float(n.pos[1]), int(n.cell), int(n.parent), int(n.viol),
                                      int(len(n.prefix))))
    if getattr(sp, "_cand_log", None):
        mpc._cand_log.extend(sp._cand_log)
        sp._cand_log = []
from .mpc_harness import (
    WMErrorTracker, log_stroke_vs_cube, make_introspector, save_reground_diag,
    seed_actions_from_probe, stitch_executed,
)


class MPCPlanner(BasePlanner):
    """An online (receding-horizon) planner: feedback from the env is allowed between replans.

    Upstream DINO-WM planner with two LOCAL structural changes (see git 62d7e08 for the original):

      1. INCREMENTAL execution. The original re-rolled the CUMULATIVE action sequence from the initial
         condition every MPC step (`eval_actions(action_so_far, save_video=True)`) -> O(n^2) renders.
         This version tracks `cur_state` and rolls ONLY the newly-committed step from it
         (`env.rollout(cur_state, exec_taken)`), stitching the executed frames as it goes; the full
         trajectory is still rendered once at the end by perform_planning. Same executed trajectory and
         success outcome (the env is deterministic), computed without the quadratic re-roll. Note
         `_apply_success_mask` consequently takes `cur_state` (the original took only `actions`).

      2. Instrumentation FACTORED OUT. The per-step extras -- warm-start seeding, sign-colour control
         (exogenous flip + DDL render-back), executed-frame stitching, WM 1-step error tracking,
         introspection, per-iter diagnostics -- live in planning/mpc_harness.py and
         planning/sign_control.py, called as one-liners so plan() reads close to the upstream loop.
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        success_hold=False,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []
        self.success_hold = bool(success_hold)  # OFF by default -- see plan()/_apply_success_mask

    def _apply_success_mask(self, actions, cur_state):
        """Succeeded envs hold: execute a zero-DISPLACEMENT stroke at the CURRENT cube xy
        (start=cube, disp=0), a near no-op that doesn't disturb a solved cube. Gated by
        `success_hold` (OFF by default) because it reads the GROUND-TRUTH cube (cur_state[18:20]) --
        the one non-probe read in the planning loop -- and drags the per-step WM-error curve down
        (succeeded evals hold -> ~0 error -> survivorship). frameskip=1, so each taken step is one
        4-D stroke [x_start, y_start, dx, dy]; the hold is broadcast over the n_taken_actions horizon."""
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        if not mask.any():
            return actions
        cube_xy = np.asarray(cur_state)[:, 18:20].astype(np.float32)          # (N,2)
        raw_hold = np.concatenate([cube_xy, np.zeros_like(cube_xy)], axis=1)[:, None, :]  # (N,1,4) [start, 0-disp]
        norm_hold = self.preprocessor.normalize_actions(torch.from_numpy(raw_hold))
        actions[mask] = norm_hold[mask].to(device=device, dtype=actions.dtype)  # broadcast over taken steps
        return actions

    def plan(self, obs_0, obs_g, actions=None):
        """Closed-loop MPC: replan from the current observation, commit the first n_taken_actions,
        execute them incrementally in the env, and repeat until every eval succeeds (or max_iter).

        `actions` is NOT used (kept for the BasePlanner interface). Returns (planned_actions
        (B,T,action_dim), action_len). The local instrumentation -- warm-start, sign-colour control,
        executed-frame stitching, WM 1-step error tracking, introspection -- is factored into
        planning/mpc_harness.py + planning/sign_control.py so this method mirrors the upstream planner.
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        # closed-loop memory: clear the sub-planner's per-episode history (e.g. RRT's executed-trajectory
        # memory) so temporal laws start fresh each episode. No-op for stateless planners.
        if hasattr(self.sub_planner, "reset"):
            self.sub_planner.reset()
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        cur_state = init_state_0                  # track current executed state for incremental stepping
        memo_actions = None
        # executed-frame accumulators (final video/metrics reuse them -> no full re-roll)
        self.executed_obses = None
        self.executed_states = None
        self.executed_strokes = []                # per-iter DENORMALIZED committed strokes (b,T,4) -> replay/diagnostics
        self._smooth_frames = []                  # per-step (N,H,W,3) frames across iters -> smooth MPC video
        self._subframe_traces = []                # per-iter sub-frame cube xy+yaw (opt-in; see trace_cfg)
        self._tree_log = []                       # RRT nodes across iters (opt-in)
        self._cand_log = []                       # per-extend counterfactual rows across iters (opt-in)

        # --- harness instrumentation (opt-in / behaviour-preserving; see mpc_harness.py + sign_control.py) ---
        wm_tracker = WMErrorTracker()                                 # per-step WM 1-step error + imagination
        self._introspector = make_introspector(self)                 # RRT/MPC introspection (off by default)
        sign = SignController(getattr(self, "sign_flip", None),       # exogenous flip schedule + DDL render-back
                              self.evaluator.env, getattr(self.sub_planner, "law_fn", None), n_evals)

        # MID-RUN GOAL RETARGET (rule-insertion experiment): when the injected return duty fires for an
        # eval, that eval's GOAL becomes the goal-cell bank's CENTERED IMAGE of its start cell -- the same
        # 9-image bank the obligation waypoint already steers with, so the new goal is a real perceptual
        # target encoded exactly like the task goal, not a synthetic one. Swaps the eval's row of obs_g
        # (visual+proprio -> what the planner encodes into z_goal) AND state_g (-> what env.eval_state
        # scores), so "success" becomes "got back home" and the success-hold mask freezes the agent there.
        # LAW-DRIVEN, not scheduled: `off` ignores obligations (mode gate in rrt.py) so its goal never
        # moves -- a free control arm. Latched per eval, applied once.
        _retargeted = set()
        # DYNAMIC GOAL PANEL. (frame_index, (N,H,W,3) goal visual) checkpoints, frame_index counted in
        # _smooth_frames units. Seeded with the ORIGINAL goal so frames before any retarget still have
        # one: obs_g is MUTATED in place by the retarget, so by render time only the final goal
        # survives and a video built from it would show "home" from frame 0 -- i.e. it would hide the
        # very switch the experiment is about.
        self._goal_log = []
        if obs_g is not None and getattr(self.evaluator, "video", False):
            self._goal_log.append((0, np.asarray(obs_g["visual"][:, 0]).copy()))

        def _like(src, dst):
            """Goal-bank tensor -> the container type of `dst`. On the law_eval path obs_g's visual /
            proprio arrive as NUMPY arrays, so a bare `src.to(dst.dtype)` is handed a numpy dtype and
            raises TypeError; elsewhere they are torch. Both sides carry the same dtypes and ranges
            (visual uint8 0-255, proprio/state float32), so the cast is value-preserving either way."""
            if isinstance(dst, np.ndarray):
                return src.detach().cpu().numpy().astype(dst.dtype, copy=False)
            return src.to(dst.dtype)

        def _retarget_from_duty():
            duty = getattr(self.sub_planner, "_return_duty", None) or {}
            bank = getattr(self.sub_planner, "_goal_bank", None)
            if not duty or bank is None:
                return
            for e, cell in duty.items():
                if e in _retargeted:
                    continue
                # MUST write the LOCAL obs_g as well: plan()'s `obs_g` parameter is what gets handed to
                # sub_planner.plan() (and re-encoded into z_goal each step), and it is NOT guaranteed to
                # be the same object as self.evaluator.obs_g. Mutating only the evaluator's copy moves
                # the SUCCESS target and the video's goal panel while the PLANNER keeps aiming at the
                # original goal -- which looks exactly like "the goal image never changed".
                for tgt in (obs_g, self.evaluator.obs_g):
                    if tgt is None:
                        continue
                    tgt["visual"][e, 0] = _like(bank.visual[cell], tgt["visual"])
                    tgt["proprio"][e, 0] = _like(bank.proprio[cell], tgt["proprio"])
                self.evaluator.state_g[e] = _like(bank.states[cell], self.evaluator.state_g)
                _retargeted.add(e)
                # checkpoint the goal panel AT THIS FRAME so the video shows the switch, not the result
                if self._goal_log and obs_g is not None:
                    self._goal_log.append((len(self._smooth_frames),
                                           np.asarray(obs_g["visual"][:, 0]).copy()))
                print(f"[RETARGET] step {self.iter} e{e}: goal -> goal-bank image of START cell {cell} "
                      f"(success now means returning home)")

        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            wm_tracker.record_start(self, cur_obs_0)                  # PERCEIVED start q=probe(encode(obs)) this
            #                                                          step (obs-only, no GT) -- the start the pruner
            #                                                          grounds legality on; pairs with pred/real end
            _t_plan = time.perf_counter()
            # warm-start the stroke START at the probe-estimated cube (obs-only) so samples CONTACT it.
            seed_actions = seed_actions_from_probe(self, cur_obs_0, memo_actions)
            # AUTHORITY: hand the evaluator this frame's GROUND-TRUTH cube xy (state cols 18:20) so the
            # SIGN's constitutive flip (R7/R7b) is adjudicated on truth, not the perception probe -- a
            # probe error must not be able to fabricate a permission/taint. observe() (in sub_planner.plan)
            # reads it. Agent prohibition + planning stay probe-derived; only the sign uses GT.
            _law_fn = getattr(self.sub_planner, "law_fn", None)
            if _law_fn is not None and hasattr(_law_fn, "set_gt_cube"):
                _cs = cur_state.detach().cpu().numpy() if hasattr(cur_state, "detach") else np.asarray(cur_state)
                _law_fn.set_gt_cube(_cs[:, 18:20])
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=seed_actions,
            )  # (b, t, act_dim)
            # The duty is concluded INSIDE plan() (per-eval observe -> DDL verdict), so retarget right
            # after it returns: this iteration's success evaluation then already scores against home.
            _retarget_from_duty()
            _collect_search(self)                 # opt-in: this iter's RRT nodes + per-extend rows
            _t_plan = time.perf_counter() - _t_plan
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            if self.success_hold:   # OFF by default: reads GT cube + causes the per-step WM-err artifact
                self._apply_success_mask(taken_actions, cur_state)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            # incremental observe: roll ONLY the newly-committed step from cur_state (avoids the O(n^2)
            # re-roll). The full trajectory is still rendered once at the end by perform_planning.
            ev = self.evaluator
            _t_eval = time.perf_counter()
            exec_taken = rearrange(taken_actions.cpu(), "b t (f d) -> b (t f) d", f=ev.frameskip)
            exec_taken = ev.preprocessor.denormalize_actions(exec_taken).numpy()
            self.executed_strokes.append(exec_taken)                 # (b,T,4) denormalized -> saved per episode for replay
            log_stroke_vs_cube(cur_state, exec_taken)                # [dbg] is the planned stroke near the cube?
            sign.on_step_pre_roll(self.iter, n_evals)                # recolour BEFORE the roll (exogenous + DDL latch)
            _fs = self._smooth_frames if getattr(ev, "video", False) else None  # per-step frames only when video=true
            _ts = {} if trace_cfg.trace_subframe() else None   # sub-frame cube xy+yaw (opt-in)
            e_obses, e_states = ev.env.rollout(ev.seed, cur_state, exec_taken, frame_sink=_fs,
                                               trace_sink=_ts, trace_stride=trace_cfg.trace_stride())
            if _ts is not None:
                self._subframe_traces.append(_ts)             # one per MPC iter (= one committed stroke)
            _t_eval = time.perf_counter() - _t_eval
            stitch_executed(self, e_obses, e_states)                 # accumulate executed frames for the final video
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            e_final_state = e_states[:, -1]
            wm_tracker.record(self, cur_obs_0, taken_actions, e_final_state, e_final_obs, self.iter)

            eval_results = ev.env.eval_state(ev.state_g, e_final_state)
            successes = eval_results["success"]
            logs = {
                ("success_rate" if k == "success" else f"mean_{k}"):
                (np.mean(v.astype(float)) if k == "success" else float(np.mean(v)))
                for k, v in eval_results.items()
            }
            print("Success rate: ", logs["success_rate"])
            print(
                f"[TIMING iter {self.iter}] {self.sub_planner.__class__.__name__} sub_plan={_t_plan:.1f}s  "
                f"observe(incremental {exec_taken.shape[1]} env-steps)={_t_eval:.1f}s"
            )
            # plan{iter}.png: the re-grounded WM imagination vs the real executed step (reuses rendered frames).
            save_reground_diag(ev, cur_obs_0, cur_state, taken_actions, (e_obses, e_states), self.iter)
            # introspection: dump tree + top-K branches (cur_state/cur_obs_0 still hold THIS iter's root).
            if self._introspector is not None:
                self._introspector.record_step(
                    step=self.iter, sub_planner=self.sub_planner, wm=self.wm,
                    evaluator=self.evaluator, objective_fn=self.objective_fn,
                    cur_obs_0=cur_obs_0, cur_state=cur_state, obs_g=obs_g)
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # continue from the new executed state next iter
            cur_obs_0 = e_final_obs
            cur_state = e_final_state
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

        if self._introspector is not None:
            self._introspector.finalize()
        # surface per-eval WM 1-step error + regrounded imagination (eval_metrics.json / output_final)
        wm_tracker.finalize(self, n_evals)
        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )
        # smooth MPC video: every internal sim step captured across iters (decoder-free, [executed | goal]).
        if self._smooth_frames:
            vis = np.stack(self._smooth_frames, axis=1)  # (N, n_steps, H, W, 3)
            self.evaluator._save_executed_video(vis, self.is_success, "output_mpc_smooth",
                                                goal_log=self._goal_log)

        return planned_actions, self.action_len
