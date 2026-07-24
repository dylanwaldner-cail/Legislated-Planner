import os
import torch
import hydra
import copy
import time  ### HARNESS EDIT ### per-iteration timing
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
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
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        ### HARNESS EDIT ### closed-loop memory: clear the sub-planner's per-episode history (e.g. RRT's
        # executed-trajectory memory) so temporal laws start fresh each episode. No-op for stateless planners.
        if hasattr(self.sub_planner, "reset"):
            self.sub_planner.reset()
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        cur_state = init_state_0  ### HARNESS EDIT ### track current executed state for incremental stepping
        memo_actions = None
        ### HARNESS EDIT ### accumulate executed frames so the final video/metrics reuse them (no full re-roll)
        self.executed_obses = None
        self.executed_states = None
        self._smooth_frames = []  # per-step (N,H,W,3) frames across iters -> smooth MPC video
        ### END HARNESS EDIT ###
        self._wm_pred_err = {}   ### HARNESS EDIT ### eval_index -> list of per-step WM 1-step cube-pred errors (m)
        self._wm_latent_err = {} ### HARNESS EDIT ### eval_index -> list of per-step WM 1-step latent MSE (decoder-free)
        self._wm_pred_xy = {}    ### HARNESS EDIT ### eval_index -> [[x,y],...] WM-PREDICTED committed-stroke cube
        self._wm_real_xy = {}    ### HARNESS EDIT ### eval_index -> [[x,y],...] REAL sim committed-stroke cube (pred-vs-real diag)
        self._imagined_regrounded = []   ### HARNESS EDIT ### per-step re-grounded decoded frames -> closed-loop output_final
        ### HARNESS EDIT ### optional RRT/MPC introspection (rrt_introspect.enabled=true). OFF by default.
        self._introspector = None
        _icfg = getattr(self, "introspect_cfg", None)
        if _icfg and _icfg.get("enabled"):
            from planning.introspect import RRTIntrospector
            self._introspector = RRTIntrospector(
                out_dir=os.path.join(os.getcwd(), "rrt_introspect"),
                top_k=int(_icfg.get("top_k", 5)),
                resim=bool(_icfg.get("resim", True)),
                max_evals=int(_icfg.get("max_evals", 3)))
            if hasattr(self.sub_planner, "_log_queries"):
                self.sub_planner._log_queries = True   # RRT logs its sampled-candidate distribution
        ### END HARNESS EDIT ###
        ### HARNESS EDIT ### exogenous sign-flip schedule (conf sign_flip). `frame` = the MPC step at
        # which the sign becomes `color`; before that it holds `base_color`. Recolouring the exec env
        # BEFORE step (frame-1)'s rollout puts the new colour in that step's executed frame, which
        # becomes cur_obs_0 for step `frame` -> the legislation re-perceives it exactly at `frame`.
        # No-op unless configured AND the env supports recolouring (grid env only).
        _sf = getattr(self, "sign_flip", None)
        _sf_frame = None
        if _sf and _sf.get("frame") is not None and hasattr(self.evaluator.env, "set_sign_color"):
            _sf_frame = int(_sf["frame"])
            _sf_color = _sf.get("color", "yellow")
            self.evaluator.env.set_sign_color(_sf_color if _sf_frame == 0 else _sf.get("base_color", "white"))
        ### END HARNESS EDIT ###
        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            _t_plan = time.perf_counter()  ### HARNESS EDIT ### timing
            ### HARNESS EDIT ### warm-start the stroke START at the cube ESTIMATED FROM THE
            # CURRENT OBSERVATION via the probe (obs-only -- NO ground-truth state). The CEM's
            # default init is the data-mean (~grid center); for an off-center cube ~all samples
            # start far from it and MISS -> frozen/no-op. Seeding [start=cube_est, disp=0]
            # centers the search on the cube so it samples CONTACTING strokes.
            probe = getattr(self.objective_fn, "position_probe", None)
            if probe is not None and (memo_actions is None or memo_actions.shape[1] == 0):
                _trans = self.preprocessor.transform_obs(cur_obs_0)
                with torch.no_grad():
                    _z = self.wm.encode_obs({"visual": _trans["visual"].to(self.device),
                                             "proprio": _trans["proprio"].to(self.device)})
                    _cube = probe(_z["visual"][:, -1]).detach().cpu().numpy()   # (b,2) meters, from OBS
                _warm = np.concatenate([_cube, np.zeros_like(_cube)], axis=1)[:, None, :]  # (b,1,4)
                seed_actions = self.preprocessor.normalize_actions(torch.from_numpy(_warm.astype(np.float32)))
            else:
                seed_actions = memo_actions
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=seed_actions,
            )  # (b, t, act_dim)
            _t_plan = time.perf_counter() - _t_plan  ### HARNESS EDIT ### timing
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            if self.success_hold:   # OFF by default: reads GT cube + causes the per-step WM-err artifact
                self._apply_success_mask(taken_actions, cur_state)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            ### HARNESS EDIT ### incremental observe: roll ONLY the newly-committed step from cur_state
            # (was: re-roll the cumulative trajectory from init every iter -> O(n^2) PathTracing renders).
            # MPC only needs the new executed state to check success + seed the next sub-plan. The full
            # trajectory is still rendered once at the end by perform_planning (for the video + full metrics).
            ev = self.evaluator
            _t_eval = time.perf_counter()
            exec_taken = rearrange(taken_actions.cpu(), "b t (f d) -> b (t f) d", f=ev.frameskip)
            exec_taken = ev.preprocessor.denormalize_actions(exec_taken).numpy()
            ### DEBUG ### is the planned stroke even near the cube? (diagnose frozen/no-op)
            _cxy = np.asarray(cur_state)[:, 18:20]
            for _i in range(min(3, exec_taken.shape[0])):
                _s = exec_taken[_i, 0]
                print(f"  [dbg e{_i}] cube=({_cxy[_i][0]:+.3f},{_cxy[_i][1]:+.3f})  start=({_s[0]:+.3f},{_s[1]:+.3f})"
                      f"  disp=({_s[2]:+.3f},{_s[3]:+.3f})  |start-cube|={np.linalg.norm(_s[:2]-_cxy[_i]):.3f}")
            ### HARNESS EDIT ### sign flip: recolour so step `frame` PERCEIVES it (see above).
            if _sf_frame is not None and self.iter == _sf_frame - 1:
                ev.env.set_sign_color(_sf_color)
            _fs = self._smooth_frames if getattr(ev, "video", False) else None  # per-step frames only when video=true
            e_obses, e_states = ev.env.rollout(ev.seed, cur_state, exec_taken, frame_sink=_fs)
            _t_eval = time.perf_counter() - _t_eval
            ### HARNESS EDIT ### stitch executed frames (drop the duplicate boundary frame on later rolls)
            if self.executed_obses is None:
                self.executed_obses = {k: v for k, v in e_obses.items()}
                self.executed_states = e_states
            else:
                for k in self.executed_obses:
                    self.executed_obses[k] = np.concatenate(
                        [self.executed_obses[k], e_obses[k][:, 1:]], axis=1
                    )
                self.executed_states = np.concatenate(
                    [self.executed_states, e_states[:, 1:]], axis=1
                )
            ### END HARNESS EDIT ###
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            e_final_state = e_states[:, -1]
            ### HARNESS EDIT ### closed-loop WM eval per committed step (planning/planning_metrics.py):
            #   * 1-step cube-pred error vs the REAL executed cube (accumulated per eval)
            #   * the RE-GROUNDED decoded imagined frame -> stitched into a CLOSED-LOOP output_final
            #     (the open-loop rollout is misleading for a re-planning MPC system).
            try:
                from planning.planning_metrics import wm_regrounded_eval
                _err, _lat, _imag, _pred, _real = wm_regrounded_eval(
                    self.wm, self.preprocessor, self.objective_fn, cur_obs_0, taken_actions,
                    e_final_state, real_obs=e_final_obs,
                    decode=(getattr(self.wm, "decoder", None) is not None))
                if _err is not None:
                    for _i in range(len(_err)):
                        self._wm_pred_err.setdefault(_i, []).append(float(_err[_i]))
                if _lat is not None:
                    for _i in range(len(_lat)):
                        self._wm_latent_err.setdefault(_i, []).append(float(_lat[_i]))
                if _pred is not None and _real is not None:   # record pred-vs-real committed cube (x,y)
                    for _i in range(len(_pred)):
                        self._wm_pred_xy.setdefault(_i, []).append([float(_pred[_i][0]), float(_pred[_i][1])])
                        self._wm_real_xy.setdefault(_i, []).append([float(_real[_i][0]), float(_real[_i][1])])
                if _err is not None:
                    print(f"[wm 1-step err] iter {self.iter}: probe {float(_err.mean()):.4f} m"
                          + (f" | latent-mse {float(_lat.mean()):.4f}" if _lat is not None else "")
                          + f" | per-eval probe {np.round(_err, 3).tolist()}")
                if _imag is not None:
                    if not self._imagined_regrounded:
                        self._imagined_regrounded.append(_imag[:, 0])    # frame 0 = recon(initial obs), once
                    self._imagined_regrounded.append(_imag[:, -1])       # this step's predicted frame
            except Exception as _pe:  # noqa: BLE001
                print(f"[wm regrounded eval] iter {self.iter} skipped: {_pe}")
            ### END HARNESS EDIT ###
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
            ### END HARNESS EDIT ###
            ### HARNESS EDIT ### plan{iter}.png = what the planner ACTUALLY reasoned over this step:
            # the RE-GROUNDED WM imagination (from cur_obs_0 over the committed action) vs the real
            # executed step. Reuses the already-rendered frames (precomputed_env) -> no extra sim render.
            # Best-effort: a plot/shape hiccup is logged and skipped, never kills the run.
            try:
                ev.assign_init_cond(obs_0=cur_obs_0, state_0=cur_state)
                ev.eval_actions(
                    taken_actions, filename=f"plan{self.iter}",
                    save_video=False, full_video=True,
                    precomputed_env=(e_obses, e_states),
                )
            except Exception as _diag_e:
                print(f"[diag] per-iter plan{self.iter}.png skipped: {_diag_e}")
            ### END HARNESS EDIT ###
            ### HARNESS EDIT ### introspection: dump tree + top-K branches + imagined-vs-realized
            # (cur_state/cur_obs_0 still hold THIS iter's tree root here). Guarded internally.
            if self._introspector is not None:
                self._introspector.record_step(
                    step=self.iter, sub_planner=self.sub_planner, wm=self.wm,
                    evaluator=self.evaluator, objective_fn=self.objective_fn,
                    cur_obs_0=cur_obs_0, cur_state=cur_state, obs_g=obs_g)
            ### END HARNESS EDIT ###
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
        # per-eval mean WM 1-step prediction error (surfaced in eval_metrics.json / the sweep summary)
        self.wm_pred_err_mean = [
            float(np.mean(self._wm_pred_err[i])) if self._wm_pred_err.get(i) else float("nan")
            for i in range(n_evals)
        ]
        self.wm_latent_err_mean = [
            float(np.mean(self._wm_latent_err[i])) if self._wm_latent_err.get(i) else float("nan")
            for i in range(n_evals)
        ]
        # per-step (ragged: one list per eval, indexed by MPC step) so the sweep can aggregate BY
        # STEP INDEX -- e.g. is step 0 (the first big contact push) systematically the worst?
        self.wm_pred_err_steps = [self._wm_pred_err.get(i, []) for i in range(n_evals)]
        self.wm_latent_err_steps = [self._wm_latent_err.get(i, []) for i in range(n_evals)]
        self.wm_pred_xy_steps = [self._wm_pred_xy.get(i, []) for i in range(n_evals)]   # pred-vs-real diag
        self.wm_real_xy_steps = [self._wm_real_xy.get(i, []) for i in range(n_evals)]
        # stitched re-grounded imagination (N, 1+n_steps, 3, H, W) -> closed-loop output_final
        self.imagined_regrounded = (torch.stack(self._imagined_regrounded, dim=1)
                                    if self._imagined_regrounded else None)
        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        ### HARNESS EDIT ### smooth MPC video: every internal sim step captured across iters
        # (decoder-free, [executed | goal]). Writes output_mpc_smooth_<eval>_<tag>.mp4.
        if self._smooth_frames:
            vis = np.stack(self._smooth_frames, axis=1)  # (N, n_steps, H, W, 3)
            self.evaluator._save_executed_video(vis, self.is_success, "output_mpc_smooth")

        return planned_actions, self.action_len
