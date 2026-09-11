import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device

# NOTE: the legislative-probe (PointMaze illegal-region) version of this planner
# is preserved in planning/cem_legislative.py.bak. This is the clean CEM that
# uses only the passed objective_fn (e.g. the cube-position probe objective), so
# it works for the single-agent push task without loading wm_probe.pt.


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
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
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        ### HARNESS EDIT ### env_eval gates the in-optimization REAL-sim diagnostic rollout (logging + success early-stop only; CEM's topk ranking is WM-only). Default off so planning never touches the sim — mirrors gd.py. Each call does a full env.rollout (PathTracing) per opt step, so leaving it on is ~opt_steps sim rolls per solve. mpc_cem still rolls the committed actions once per iter in mpc.py.
        self.env_eval = kwargs.get("env_eval", False)
        # Trust-region constraint (meters; None=off): when warm-started at a cube estimate,
        # every sampled action's START coords are clamped to within start_window of that cube,
        # so the CEM can't explore far-start MISSES the WM hallucinates about. Push dims free.
        self.start_window = kwargs.get("start_window", None)
        self.logging_prefix = logging_prefix

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        # var_scale may be a scalar OR a per-dim list (len == action_dim). Per-dim lets us
        # keep the search TIGHT on the start coords (stay near a cube-seeded warm start, in
        # the WM's accurate regime) while exploring the push dims widely. e.g. [0.5,0.5,1,1].
        _vs = torch.as_tensor(self.var_scale, dtype=torch.float32).reshape(-1)
        sigma = _vs.reshape(1, 1, -1) * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def _action_bounds(self):
        """### HARNESS EDIT ### clamp bounds = the WM's TRAINING action range, so
        CEM's N(mu,sigma) samples stay in-distribution. Two cases:
          * preprocessor carries action_min/max (e.g. the stroke dataset): clamp to
            that raw per-dim range -> normalized (min-mean)/std .. (max-mean)/std.
            Already full action_dim (frameskip=1), so no tiling.
          * else (legacy envs whose raw actions were clipped to [-1,1]): keep the
            [(-1-mean)/std, (1-mean)/std] bounds, tiled over frameskip."""
        amean = self.preprocessor.action_mean.to(self.device).reshape(-1)
        astd = self.preprocessor.action_std.to(self.device).reshape(-1)
        amin = getattr(self.preprocessor, "action_min", None)
        amax = getattr(self.preprocessor, "action_max", None)
        if amin is not None and amax is not None:
            raw_lo = torch.as_tensor(amin, device=self.device, dtype=astd.dtype).reshape(-1)
            raw_hi = torch.as_tensor(amax, device=self.device, dtype=astd.dtype).reshape(-1)
            return (raw_lo - amean) / astd, (raw_hi - amean) / astd
        fs = self.evaluator.frameskip
        act_lo = ((-1.0 - amean) / astd).repeat(fs)
        act_hi = ((1.0 - amean) / astd).repeat(fs)
        return act_lo, act_hi

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        act_lo, act_hi = self._action_bounds()  ### HARNESS EDIT ### clamp range
        ### HARNESS EDIT ### near-cube START trust region: when warm-started (actions given =
        # the probe's cube estimate), clamp every sampled action's start coords to within
        # start_window (m) of that cube, so the CEM can't escape to far-start hallucinations.
        start_lo = start_hi = None
        if self.start_window is not None and actions is not None:
            _astd = self.preprocessor.action_std.to(self.device).reshape(-1)[:2]
            _w = self.start_window / _astd            # meters -> normalized half-window, (2,)
            _sc = mu[:, :, :2].clone()                # (n_evals, horizon, 2) = warm-start cube (normalized)
            start_lo, start_hi = _sc - _w, _sc + _w

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                action = torch.clamp(action, act_lo, act_hi)  ### HARNESS EDIT ### keep samples in the WM's training range
                action = self._constrain_start(action, traj, start_lo, start_hi)  ### HARNESS EDIT ### start constraint (overridable hook)
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            ### HARNESS EDIT ### gated on env_eval (default off) — was unconditional; each call does a full env.rollout (PathTracing) per opt step
            if self.env_eval and self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
            ### END HARNESS EDIT ###

        ### HARNESS EDIT ### per-solve diagnostic (gated by evaluator.cem_debug):
        # score SPREAD across sampled actions + the GT action's score vs the winner.
        if getattr(self.evaluator, "cem_debug", False):
            self._cem_debug(mu, sigma, trans_obs_0, z_obs_g, act_lo, act_hi)

        return mu, np.full(n_evals, np.inf)  # all actions are valid

    def _constrain_start(self, action, traj, start_lo, start_hi):
        """### HARNESS EDIT ### Overridable hook: project sampled actions' START dims
        onto the start constraint, in-loop, just after the global action clamp.
        BASE behavior == the previous inline trust-region clamp (a near-cube box around
        the warm-start cube; a no-op when start_window is off). Subclasses (see
        planning/cem_aimed_contact.AimedContactCEMPlanner) override this to DERIVE the
        start from the push direction instead of clamping it. Behavior here is unchanged."""
        if start_lo is not None:  # near-cube start trust region (push dims free)
            action[:, :, :2] = torch.maximum(torch.minimum(action[:, :, :2], start_hi[traj]), start_lo[traj])
        return action

    @torch.no_grad()
    def _cem_debug(self, mu, sigma, trans_obs_0, z_obs_g, act_lo, act_hi):
        """Diagnose whether the objective is FOCUSED. For each eval, sample num_samples
        actions around the converged (mu,sigma), score them via the WM+objective, and
        report: (1) the score SPREAD (win/mean/max/std -- if it's tiny, the objective
        can't tell actions apart -> the WM is 'unfocused'); (2) the GT action's score vs
        the winner (does the objective even RATE the truly-good action best?). gt_actions
        (normalized) + the MPC iter index arrive via the evaluator / logging_prefix."""
        gt = getattr(self.evaluator, "gt_actions", None)
        try:
            k = int(str(self.logging_prefix).rsplit("_", 1)[-1])
        except (ValueError, AttributeError):
            k = 0
        for traj in range(mu.shape[0]):
            cur_obs = {key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                       for key, arr in trans_obs_0.items()}
            cur_g = {key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                     for key, arr in z_obs_g.items()}
            # sample with the INITIAL exploration sigma (var_scale), NOT the passed sigma
            # which has converged to ~0 -> all samples would equal mu (fake spread=0).
            init_sigma = torch.as_tensor(self.var_scale, device=self.device, dtype=mu.dtype).reshape(1, 1, -1)
            action = torch.randn(self.num_samples, self.horizon, self.action_dim,
                                 device=self.device) * init_sigma + mu[traj]
            action = torch.clamp(action, act_lo, act_hi)
            i_z, _ = self.wm.rollout(obs_0=cur_obs, act=action)
            loss = self.objective_fn(i_z, cur_g).reshape(-1)
            lo, hi = float(loss.min()), float(loss.max())
            msg = (f"[cem dbg plan_{k} e{traj}] sample scores: win={lo:.4f} mean={float(loss.mean()):.4f} "
                   f"max={hi:.4f} std={float(loss.std()):.4f} spread={hi - lo:.4f}")
            if gt is not None:
                gtt = torch.as_tensor(gt, device=self.device).float()
                kk = max(0, min(k, gtt.shape[1] - self.horizon))
                gt_a = gtt[traj:traj + 1, kk:kk + self.horizon]
                one_obs = {key: arr[traj].unsqueeze(0) for key, arr in trans_obs_0.items()}
                one_g = {key: arr[traj].unsqueeze(0) for key, arr in z_obs_g.items()}
                i_z_gt, _ = self.wm.rollout(obs_0=one_obs, act=gt_a)
                gt_loss = float(self.objective_fn(i_z_gt, one_g).reshape(-1)[0])
                n_better = int((loss < gt_loss).sum())
                msg += (f"  | GT={gt_loss:.4f} ({n_better}/{self.num_samples} beat GT; "
                        f"GT {'BEATS winner' if gt_loss < lo else 'worse than winner'})")
            print(msg)
