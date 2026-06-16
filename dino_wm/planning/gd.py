import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
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
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        ### HARNESS EDIT ### env_eval gates the in-optimization REAL-sim diagnostic rollout (logging + early-stop only; the GD gradient is WM-only). Default off so planning never touches the sim.
        self.env_eval = kwargs.get("env_eval", False)
        ### HARNESS EDIT ### RIGHT-ROBOT FREEZE (two-arm grid env only)
        # We test ONLY the LEFT robot. The RIGHT robot must replay its recorded expert
        # trajectory (so it does its real task instead of flopping around and getting in
        # the left robot's way). freeze_right_gt is the per-eval recorded action sequence
        # (normalized, WM-grouped: (b, goal_H, frameskip*14), per-env action = [left7, right7]),
        # set externally by PlanWorkspace. gt_offset aligns it to the executed step (MPC sets
        # it = current iter each replan). When set, the right-arm dims of every planned action
        # are overwritten with the expert's, so the planner only optimizes the left arm.
        # None => normal joint planning (both arms optimized). See _apply_right_gt.
        self.freeze_right_gt = None
        self.gt_offset = 0
        ### END HARNESS EDIT ###
        self.logging_prefix = logging_prefix

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return self._apply_right_gt(actions)  ### HARNESS EDIT ### seed right arm with its expert trajectory

    ### HARNESS EDIT ### overwrite the RIGHT robot's action dims with its recorded expert trajectory
    def _apply_right_gt(self, actions):
        """Replace the right-arm dims of `actions` (normalized, WM-grouped
        (b, T, frameskip*14); per-env action = [left7, right7]) with the recorded
        expert trajectory in self.freeze_right_gt, aligned at self.gt_offset. Leaves
        the left arm untouched -> planner optimizes left only. No-op if not set.
        Holds the last expert frame if the lookahead runs past the recorded segment."""
        if self.freeze_right_gt is None:
            return actions
        fs = self.evaluator.frameskip
        env_dim = self.action_dim // fs       # 14 (= left7 + right7)
        agent = env_dim // 2                  # 7
        T = actions.shape[1]
        seg = self.freeze_right_gt.to(actions.device)[:, self.gt_offset:self.gt_offset + T]
        if seg.shape[1] < T:                  # ran past the recorded segment -> hold last
            seg = torch.cat([seg, seg[:, -1:].repeat(1, T - seg.shape[1], 1)], dim=1)
        actions = actions.clone()
        for f in range(fs):
            b0 = f * env_dim
            actions[:, :, b0 + agent: b0 + env_dim] = seg[:, :, b0 + agent: b0 + env_dim]
        return actions

    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        actions = self.init_actions(obs_0, actions).to(self.device)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_evals = actions.shape[0]

        for i in range(self.opt_steps):
            optimizer.zero_grad()
            i_z_obses, i_zs = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
            loss = self.objective_fn(i_z_obses, z_obs_g_detached)  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            with torch.no_grad():
                actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                actions_new += (
                    torch.randn_like(actions_new) * self.action_noise
                )  # Add Gaussian noise
                actions_new = self._apply_right_gt(actions_new)  ### HARNESS EDIT ### keep right arm on its expert trajectory (left-only optimization)
                actions.copy_(actions_new)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )
            ### HARNESS EDIT ### gated on env_eval (default off) — was unconditional
            if self.env_eval and self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
            ### END HARNESS EDIT ###
        return actions, np.full(n_evals, np.inf)  # all actions are valid
