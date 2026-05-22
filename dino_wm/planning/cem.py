import os
import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device

# === HARNESS EDIT: imports for loading the legislative probe + logging ===
# Probe ckpt is trained by train_probe.py and saved to <orig_cwd>/wm_probe.pt
# (see train_probe.py:708-710). At plan time we re-instantiate the same
# LegislativeProbe class and load those weights. is_illegal_state is reused
# for executed-rollout illegality logging so the planner agrees with the
# training-time labelling rule.
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
from legislative_harness.probe import LegislativeProbe
from legislative_harness.utils import is_illegal_state
# === END HARNESS EDIT ===


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
        self.logging_prefix = logging_prefix

        # === HARNESS EDIT: load trained LegislativeProbe ===
        # input_dim=788 matches the training-time probe input:
        # [mean-pooled visual_t, proprio_t, mean-pooled visual_{t+1}, proprio_{t+1}]
        # (see train_probe.py:397-402). Probe is eval-only at plan time — no grads.
        probe_path = os.path.join(get_original_cwd(), "wm_probe.pt")
        self.probe = LegislativeProbe(input_dim=788).to(self.device)
        self.probe.load_state_dict(torch.load(probe_path, map_location=self.device))
        self.probe.eval()

        # Load illegal_region from the training config so this stays a single
        # source of truth (the same yaml that trained the probe defines what
        # "illegal" means for executed-rollout logging). Env-specific: if you
        # train a probe for a non-pointmaze env, update the filename here.
        train_cfg_path = os.path.join(
            get_original_cwd(), "conf", "train_probe_point_maze.yaml"
        )
        train_cfg = OmegaConf.load(train_cfg_path)
        self.illegal_region = OmegaConf.to_container(
            train_cfg.probe.illegal_region, resolve=True
        )
        # Threshold for counting a CEM (S, T) pair as "predicted illegal" in
        # the logs. Pure logging knob -- does not enter the loss.
        self.probe_illegal_threshold = 0.75
        # === END HARNESS EDIT ===

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
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

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            # === HARNESS EDIT: per-opt-step logging accumulators ===
            # Raw tensors so we can do IQR / percentile analysis at the end of
            # the opt step rather than reducing to mean() per traj-instance.
            probe_all_probs   = []   # per-(traj-instance) (S, T_pairs) probe probs, flattened
            loss_mse_only_raw = []   # per-(traj-instance) (S,) MSE-only loss tensors
            loss_total_raw    = []   # per-(traj-instance) (S,) MSE + probe loss tensors
            illegal_pair_count = 0   # number of (S, T) pairs with prob > threshold
            illegal_traj_count = 0   # number of CEM trajectories with at least one such pair
            total_pair_count   = 0   # denominator for fraction stats
            total_traj_count   = 0   # ditto
            topk_overlaps     = []   # per-(traj-instance) fraction of MSE-only topk preserved under MSE+probe
            spearman_corrs    = []   # per-(traj-instance) Spearman rank corr (loss_mse_only vs loss_total)
            # === END HARNESS EDIT ===
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
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                # === HARNESS EDIT: sliding-window probe over the rollout ===
                # Mirror the training-time probe input format
                # (train_probe.py:397-402): for every consecutive frame pair
                # (t, t+1) build [mean-pooled visual_t, proprio_t,
                # mean-pooled visual_{t+1}, proprio_{t+1}] -> 788 dims.
                # Shapes confirmed against visual_world_model.py:122-123 and
                # rollout's docstring at visual_world_model.py:289-290:
                #   i_z_obses["visual"]:  (S, T+1, num_patches, emb_dim)
                #   i_z_obses["proprio"]: (S, T+1, proprio_emb_dim)
                # Probe head is a logit head trained with BCEWithLogitsLoss,
                # so we sigmoid before averaging over time to get a per-traj
                # P(illegal) score in [0, 1] for the objective.
                with torch.no_grad():
                    v_pool = i_z_obses["visual"].mean(dim=2)  # (S, T+1, emb_dim)
                    p_full = i_z_obses["proprio"]              # (S, T+1, proprio_emb_dim)
                    probe_in = torch.cat(
                        [v_pool[:, :-1], p_full[:, :-1],
                         v_pool[:, 1:],  p_full[:, 1:]],
                        dim=-1,
                    )  # (S, T, 788)
                    S, T_pairs, D = probe_in.shape
                    probe_logits = self.probe(probe_in.reshape(S * T_pairs, D))
                    probe_probs = torch.sigmoid(probe_logits).reshape(S, T_pairs)
                    # Pass per-chunk probabilities through unaggregated; the
                    # objective decides the time-axis reduction (sum, in the
                    # current config -- see planning/objectives.py).
                    probe_out = probe_probs  # (S, T_pairs)
                # === END HARNESS EDIT ===

                # === HARNESS EDIT: compute MSE-only and MSE+probe loss for logging ===
                # Both forms so we can quantify how much the probe term shifts
                # the loss CEM ranks by. Cheap -- only the linear-probe matmul
                # and a sum are extra work.
                loss_mse_only = self.objective_fn(i_z_obses, cur_z_obs_g, probe_out=None)
                loss = self.objective_fn(i_z_obses, cur_z_obs_g, probe_out)
                loss_mse_only_raw.append(loss_mse_only.detach().cpu())
                loss_total_raw.append(loss.detach().cpu())
                # === END HARNESS EDIT ===

                # === HARNESS EDIT: CEM-side illegal-pair / illegal-trajectory counts ===
                with torch.no_grad():
                    probe_all_probs.append(probe_probs.detach().flatten().cpu())
                    illegal_pair_mask = probe_probs > self.probe_illegal_threshold
                    illegal_pair_count += int(illegal_pair_mask.sum().item())
                    illegal_traj_count += int(illegal_pair_mask.any(dim=1).sum().item())
                    total_pair_count += int(probe_probs.numel())
                    total_traj_count += int(probe_probs.shape[0])
                # === END HARNESS EDIT ===

                # === HARNESS EDIT: topk reranking diagnostics ===
                # Quantify how much the probe penalty changes CEM's elite
                # selection vs MSE-only ranking:
                #   - overlap : |topk_mse ∩ topk_total| / topk  (set-level)
                #   - spearman: rank correlation over all S samples (global)
                # Spearman is Pearson on ranks; argsort().argsort() converts
                # values -> ranks (see chat for derivation).
                with torch.no_grad():
                    topk_mse_idx   = torch.argsort(loss_mse_only)[: self.topk]
                    topk_total_idx = torch.argsort(loss)[: self.topk]
                    overlap = (
                        len(set(topk_mse_idx.cpu().tolist())
                            & set(topk_total_idx.cpu().tolist()))
                        / max(self.topk, 1)
                    )
                    rank_mse   = loss_mse_only.argsort().argsort().float()
                    rank_total = loss.argsort().argsort().float()
                    rm_c = rank_mse   - rank_mse.mean()
                    rt_c = rank_total - rank_total.mean()
                    spearman = (
                        (rm_c * rt_c).sum()
                        / (rm_c.norm() * rt_c.norm() + 1e-12)
                    ).item()
                    topk_overlaps.append(overlap)
                    spearman_corrs.append(spearman)
                # === END HARNESS EDIT ===

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            # === HARNESS EDIT: aggregate per-opt-step probe & loss stats ===
            probe_flat = torch.cat(probe_all_probs)
            probe_q = torch.quantile(
                probe_flat, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
            ).tolist()
            probe_stats = {
                f"{self.logging_prefix}/probe/mean": probe_flat.mean().item(),
                f"{self.logging_prefix}/probe/std":  probe_flat.std().item(),
                f"{self.logging_prefix}/probe/p05":  probe_q[0],
                f"{self.logging_prefix}/probe/p25":  probe_q[1],
                f"{self.logging_prefix}/probe/p50":  probe_q[2],
                f"{self.logging_prefix}/probe/p75":  probe_q[3],
                f"{self.logging_prefix}/probe/p95":  probe_q[4],
                f"{self.logging_prefix}/probe/frac_low":  (probe_flat < 0.1).float().mean().item(),
                f"{self.logging_prefix}/probe/frac_mid":  ((probe_flat >= 0.4) & (probe_flat <= 0.6)).float().mean().item(),
                f"{self.logging_prefix}/probe/frac_high": (probe_flat > 0.9).float().mean().item(),
            }

            illegal_stats = {
                f"{self.logging_prefix}/cem/illegal_pair_count":    illegal_pair_count,
                f"{self.logging_prefix}/cem/illegal_pair_fraction": illegal_pair_count / max(total_pair_count, 1),
                f"{self.logging_prefix}/cem/illegal_traj_count":    illegal_traj_count,
                f"{self.logging_prefix}/cem/illegal_traj_fraction": illegal_traj_count / max(total_traj_count, 1),
                f"{self.logging_prefix}/cem/illegal_threshold":     self.probe_illegal_threshold,
            }

            mse_flat   = torch.cat(loss_mse_only_raw)
            total_flat = torch.cat(loss_total_raw)
            delta_flat = total_flat - mse_flat
            qs = torch.tensor([0.25, 0.5, 0.75])

            def _iqr(t):
                q25, q50, q75 = torch.quantile(t, qs).tolist()
                return {"median": q50, "q25": q25, "q75": q75, "iqr": q75 - q25,
                        "min": t.min().item(), "max": t.max().item()}

            loss_stats = {}
            for name, flat in [("loss_mse_only", mse_flat),
                               ("loss_total",    total_flat),
                               ("loss_delta",    delta_flat)]:
                for k, v in _iqr(flat).items():
                    loss_stats[f"{self.logging_prefix}/{name}/{k}"] = v
            # === END HARNESS EDIT ===

            # === HARNESS EDIT: aggregate reranking diagnostics ===
            overlaps_t = torch.tensor(topk_overlaps)
            spearman_t = torch.tensor(spearman_corrs)
            rerank_stats = {
                f"{self.logging_prefix}/rerank/topk_overlap_median": overlaps_t.median().item(),
                f"{self.logging_prefix}/rerank/topk_overlap_min":    overlaps_t.min().item(),
                f"{self.logging_prefix}/rerank/topk_overlap_mean":   overlaps_t.mean().item(),
                f"{self.logging_prefix}/rerank/spearman_median":     spearman_t.median().item(),
                f"{self.logging_prefix}/rerank/spearman_min":        spearman_t.min().item(),
                f"{self.logging_prefix}/rerank/spearman_mean":       spearman_t.mean().item(),
            }
            # === END HARNESS EDIT ===

            self.wandb_run.log(
                {
                    f"{self.logging_prefix}/loss": np.mean(losses),
                    "step": i + 1,
                    # === HARNESS EDIT: probe + loss-spread + cem-illegal + rerank stats ===
                    **probe_stats,
                    **illegal_stats,
                    **loss_stats,
                    **rerank_stats,
                    # === END HARNESS EDIT ===
                }
            )

            # === HARNESS EDIT: console summary so we can sanity-check w/o wandb ===
            print(
                f"[CEM {self.logging_prefix} opt_step={i+1}] "
                f"probe p25/p50/p75={probe_q[1]:.3f}/{probe_q[2]:.3f}/{probe_q[3]:.3f}  "
                f"frac_low/mid/high="
                f"{probe_stats[f'{self.logging_prefix}/probe/frac_low']:.2f}/"
                f"{probe_stats[f'{self.logging_prefix}/probe/frac_mid']:.2f}/"
                f"{probe_stats[f'{self.logging_prefix}/probe/frac_high']:.2f}  "
                f"cem_illegal_traj={illegal_traj_count}/{total_traj_count} "
                f"({illegal_pair_count}/{total_pair_count} pairs)  "
                f"loss_delta med/max="
                f"{loss_stats[f'{self.logging_prefix}/loss_delta/median']:.3f}/"
                f"{loss_stats[f'{self.logging_prefix}/loss_delta/max']:.3f}  "
                f"topk_overlap={rerank_stats[f'{self.logging_prefix}/rerank/topk_overlap_median']:.2f} "
                f"spearman={rerank_stats[f'{self.logging_prefix}/rerank/spearman_median']:.2f}"
            )
            # === END HARNESS EDIT ===

            if self.evaluator is not None and i % self.eval_every == 0:
                # === HARNESS EDIT: capture e_states (4th return) for illegality logging
                # + enable MP4 export for demo videos ===
                logs, successes, _, e_states = self.evaluator.eval_actions(
                    mu,
                    filename=f"{self.logging_prefix}_output_{i+1}",
                    save_video=True,
                )
                # === END HARNESS EDIT ===
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})

                # === HARNESS EDIT: executed-rollout illegality logging ===
                # e_states comes from env.rollout in the evaluator. For PointMaze
                # this is (n_evals, T*frameskip + 1, 4) with [x, y, vx, vy] in
                # the last dim. is_illegal_state expects a 2D (N, state_dim)
                # tensor (uses states[:, :2] internally -- see
                # legislative_harness/utils.py:6-19), so we flatten to (B*Tsteps, D)
                # for the call and reshape the (B*Tsteps,) result back to (B, Tsteps).
                # If you change envs, recheck the x/y convention at dims 0/1.
                e_states_t = torch.as_tensor(e_states, dtype=torch.float32)
                B_exec, T_exec, D_exec = e_states_t.shape
                exec_illegal_per_step = is_illegal_state(
                    e_states_t.reshape(B_exec * T_exec, D_exec),
                    self.illegal_region,
                ).reshape(B_exec, T_exec)
                exec_illegal_traj_mask = exec_illegal_per_step.bool().any(dim=1)
                n_exec_traj  = int(exec_illegal_per_step.shape[0])
                n_exec_steps = int(exec_illegal_per_step.numel())
                exec_stats = {
                    f"{self.logging_prefix}/exec/illegal_step_count":    int(exec_illegal_per_step.sum().item()),
                    f"{self.logging_prefix}/exec/illegal_step_fraction": exec_illegal_per_step.float().mean().item(),
                    f"{self.logging_prefix}/exec/illegal_traj_count":    int(exec_illegal_traj_mask.sum().item()),
                    f"{self.logging_prefix}/exec/illegal_traj_fraction": exec_illegal_traj_mask.float().mean().item(),
                    f"{self.logging_prefix}/exec/n_traj":                n_exec_traj,
                    f"{self.logging_prefix}/exec/n_steps":               n_exec_steps,
                }
                logs.update(exec_stats)
                print(
                    f"[CEM {self.logging_prefix} eval@opt_step={i+1}] "
                    f"exec_illegal_traj={exec_stats[f'{self.logging_prefix}/exec/illegal_traj_count']}/{n_exec_traj} "
                    f"steps={exec_stats[f'{self.logging_prefix}/exec/illegal_step_count']}/{n_exec_steps}"
                )
                # === END HARNESS EDIT ===

                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid
