"""Multi-step (chained) aimed-contact CEM planner (experimental, horizon > 1).

Extends AimedContactCEMPlanner WITHOUT modifying it. The horizon-1 aimed planner derives
every stroke's start from a single initial cube anchor; at horizon > 1 that aims strokes
2..H at where the cube WAS, not where it is after the earlier strokes move it -> they miss
and the multi-step plan silently collapses to horizon-1. This planner fixes that by rolling
the WM AUTOREGRESSIVELY and re-deriving each stroke's start from the cube PREDICTED after
the previous stroke:

    cube <- probe(encode(current obs))                 # sensors only, h=0 anchor
    for k in 1..H:
        start_k = cube - back * dir(d_k)               # d_k = the CEM-sampled push for step k
        roll the WM one more step with [start_k, d_k]
        cube <- probe(predicted latent)                # anchor for the next stroke

So every stroke in the imagined trajectory contacts the (predicted) cube. The intermediate
anchors use the ENCODED probe (objective_fn.position_probe = probe_cube_1500.pth), which is
the most accurate probe on the h<=3 predicted latents (see the chaining diagnostic).

Cost: the rollout is re-done from obs_0 at each step (O(H^2) predictor steps, H small), so
this is several times slower per CEM iteration than the single-shot horizon-1 planner. It is
also the RRT node-expansion primitive (predict -> probe cube -> derive aimed children).
"""
import numpy as np
import torch
from einops import repeat

from utils import move_to_device
from .cem_aimed_contact import AimedContactCEMPlanner


class ChainedAimedCEMPlanner(AimedContactCEMPlanner):
    def _anchor_cube(self, obs_0, actions):
        """Cube anchor (meters), ALWAYS from the probe on the current obs -- NOT the warm-start
        seed. At horizon>1 the MPC seed is the leftover strokes, whose start coords are not the
        current cube, so the parent's seed shortcut would anchor wrong. Sensors only."""
        probe = getattr(self.objective_fn, "position_probe", None)
        if probe is None:
            raise RuntimeError("ChainedAimedCEMPlanner needs objective_fn.position_probe "
                               "(use a probe objective) for the per-step cube anchors.")
        trans = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        with torch.no_grad():
            z = self.wm.encode_obs({"visual": trans["visual"], "proprio": trans["proprio"]})
            return probe(z["visual"][:, -1]).to(self.device).float()      # (n_evals, 2) meters

    def _derive_step(self, disp_norm, cube_m):
        """One stroke: given a normalized push (N,2) and the current cube anchor (N,2 meters),
        return the full normalized action (N,4) = [behind-cube start, barrier-aware push] by
        reusing the parent's _derive (which also applies keep-off-barrier + clamps)."""
        a = torch.zeros(disp_norm.shape[0], 1, self.action_dim, device=self.device)
        a[:, 0, 2:4] = disp_norm
        a = self._derive(a, cube_m[:, None, :])                           # (N,1,4)
        return a[:, 0]                                                    # (N,4)

    @torch.no_grad()
    def _chained_rollout(self, obs_0_rep, action, cube0):
        """Autoregressive roll of `action` (N,H,4; only push dims used) starting from anchor
        cube0 (N,2 meters). Returns (final H-step latent dict, realized action with per-step
        derived starts)."""
        realized = action.clone()
        cube = cube0
        z_seq = None
        anchor_probe = self.objective_fn.position_probe
        for k in range(self.horizon):
            realized[:, k] = self._derive_step(action[:, k, 2:4], cube)
            z_seq, _ = self.wm.rollout(obs_0=obs_0_rep, act=realized[:, :k + 1])
            if k < self.horizon - 1:                                      # no anchor needed past the last stroke
                cube = anchor_probe(z_seq["visual"][:, -1]).to(self.device).float()  # (N,2) predicted cube
        return z_seq, realized

    @torch.no_grad()
    def _realize_mu(self, trans_obs_0, mu):
        """Re-derive the FINAL mu's per-step starts via the same chained anchoring, so the
        EXECUTED strokes contact the predicted cube at each step (not the stale initial one)."""
        out = mu.clone()
        anchor_probe = self.objective_fn.position_probe
        for traj in range(mu.shape[0]):
            obs1 = {k: arr[traj].unsqueeze(0) for k, arr in trans_obs_0.items()}
            cube = self._cube_hat_m[traj].unsqueeze(0)                    # (1,2)
            for k in range(self.horizon):
                out[traj, k] = self._derive_step(mu[traj:traj + 1, k, 2:4], cube)[0]
                z_seq, _ = self.wm.rollout(obs_0=obs1, act=out[traj:traj + 1, :k + 1])
                if k < self.horizon - 1:
                    cube = anchor_probe(z_seq["visual"][:, -1]).to(self.device).float()
        return out

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        # caches used by _derive / _derive_step
        self._amean = self.preprocessor.action_mean.to(self.device).reshape(-1).float()
        self._astd = self.preprocessor.action_std.to(self.device).reshape(-1).float()
        self._aim_lo, self._aim_hi = self._action_bounds()
        self._cube_hat_m = self._anchor_cube(obs_0, actions)              # (n_evals, 2) meters

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        act_lo, act_hi = self._aim_lo, self._aim_hi

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_obs = {k: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                           for k, arr in trans_obs_0.items()}
                cur_g = {k: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                         for k, arr in z_obs_g.items()}
                action = (torch.randn(self.num_samples, self.horizon, self.action_dim,
                                      device=self.device) * sigma[traj] + mu[traj])
                action[0] = mu[traj]
                action = torch.clamp(action, act_lo, act_hi)
                cube0 = self._cube_hat_m[traj].unsqueeze(0).expand(self.num_samples, 2)
                with torch.no_grad():
                    i_z_obses, realized = self._chained_rollout(cur_obs, action, cube0)
                loss = self.objective_fn(i_z_obses, cur_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = realized[topk_idx]                          # realized -> carries derived starts + flipped pushes
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)
            self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1})

        mu = self._realize_mu(trans_obs_0, mu)   # final action on the chained-aimed manifold
        return mu, np.full(n_evals, np.inf)
