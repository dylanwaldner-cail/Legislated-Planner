import os
import random
import gym
import logging
import warnings
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
import torch
import numpy as np
from omegaconf import OmegaConf, open_dict
import time

from einops import repeat, rearrange
from custom_resolvers import replace_slash
from utils import cfg_to_dict, seed
from plan import load_model

from preprocessor import Preprocessor
from utils import move_to_device

from legislative_harness.probe import LegislativeProbe
from legislative_harness.utils import is_illegal_state, eval_probe, append_step
from planning.objectives import create_objective_fn
from env.venv import SubprocVectorEnv
from env.pointmaze.point_maze_wrapper import PointMazeWrapper
from env.pointmaze.maze_model import U_MAZE_EVAL, U_MAZE

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# FIDELITY THRESHOLD PLACEHOLDER                                       #
# This value (0.5) is a placeholder. Before fixing it, run this       #
# script once and observe the pixel_mse values printed per-step.      #
# The threshold should be set based on the empirical distribution of  #
# those values across multiple runs.                                   #
# ------------------------------------------------------------------ #
FIDELITY_THRESHOLD = 1.0

# Fixed init and goal states for PointMaze (x, y, dx, dy).
# Set based on domain knowledge of the maze layout.
INIT_STATE = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
GOAL_STATE = np.array([3.0, 3.0, 0.0, 0.0], dtype=np.float32)


class ProbeRolloutTrainer:
    """
    Trains a LegislativeProbe on WM rollout predictions grounded by env steps.

    For each training iteration:
      1. Get obs_0 and obs_g from fixed init/goal states via the env.
      2. Encode obs_g to get z_obs_g (goal in embedding space).
      3. Sample num_samples random action trajectories, run wm.rollout
         on all of them in a single batched call.
      4. Score trajectories against z_obs_g with objective_fn, keep topk.
      5. For each top-k trajectory, step through env one action at a time.
         At each step, compute pixel MSE (decoder) and embedding divergence
         (visual + proprio) as fidelity signals. Stop when pixel MSE exceeds
         FIDELITY_THRESHOLD.
      6. Train the probe on all fidelity-passing steps.
    """

    def __init__(self, cfg, wm, train_dset, env, data_preprocessor, device):
        self.cfg = cfg
        self.wm = wm
        self.train_dset = train_dset
        self.env = env
        self.data_preprocessor = data_preprocessor
        self.device = device

        self.illegal_region = OmegaConf.to_container(cfg.probe.illegal_region, resolve=True)

        # num_hist: context frames WM needs before predicting.
        self.num_hist: int = cfg.num_hist  # baked in from train_cfg at init

        self.rollout_horizon: int = cfg.training.rollout_horizon
        self.num_samples: int     = cfg.training.num_samples
        self.frameskip: int       = cfg.frameskip
        self.action_dim: int      = train_dset.action_dim * self.frameskip

        # Rewritten to use all samples, we just need to get more signal at this point
        self.topk: int = cfg.training.get("topk", max(1, self.num_samples))

        # Objective function for scoring trajectories against z_obs_g.
        # alpha=0 weights visual only, consistent with the empirical finding
        # that visual embeddings carry more illegal-state signal than proprio.
        self.objective_fn = create_objective_fn(
            alpha=cfg.training.get("alpha", 1.0),
            base=cfg.training.get("base", 1.0),
            mode=cfg.training.get("objective_mode", "last"),
        )

        self.probe = LegislativeProbe(input_dim=cfg.probe.input_dim).to(device)
        # Cold start: warm load disabled because input_dim changed to 788
        # (paired start/end state embeddings). Re-enable once a compatible
        # checkpoint exists.
        # probe_path = os.path.join(hydra.utils.get_original_cwd(), "final_probe.pt")
        # self.probe.load_state_dict(torch.load(probe_path, map_location=device))
        self.probe_optimizer = torch.optim.Adam(self.probe.parameters(), lr=cfg.probe.lr)

        self.var_scale = cfg.training.get("var_scale", 1.0)   # matches cem.yaml
        self.opt_steps = cfg.training.get("opt_steps", 30)    # matches cem.yaml

        # 5% of each iter's data is held out into this pool, never trained or
        # val'd on, then eval'd once after the full training loop in log_test.
        self.test_pool_z = []
        self.test_pool_labels = []

        # Persistent training buffer. Each iter appends its train split here,
        # then train_probe_epochs runs cfg.probe.epochs of mini-batch Adam
        # over the entire buffer-so-far. Kept on CPU to bound GPU memory.
        self.buffer_z: list = []
        self.buffer_y: list = []

    # ------------------------------------------------------------------ #
    # Observation helpers                                                  #
    # ------------------------------------------------------------------ #

    def get_obs_at_state(self, state, seed_val):
        obs, _ = self.env.prepare(seed_val[0], state)
        # obs is a dict with "visual" (H,W,C) and "proprio" (proprio_dim,)
        return obs

    def to_wm_obs(self, obs_np):
        """
        Transform a raw numpy obs dict to a WM-ready tensor dict with
        batch and time dims: visual (1,1,C,H,W), proprio (1,1,D).
        """
        obs_t = self.data_preprocessor.transform_obs(
            {k: v[np.newaxis, np.newaxis] for k, v in obs_np.items()}
        )
        return move_to_device(obs_t, self.device)

    def encode_env_frame(self, rollout_obses):
        """
        Encode env ground-truth obs into WM embedding space.
        Uses data_preprocessor.transform_obs before encode_obs, mirroring
        evaluator.py:L100 (e_obs = move_to_device(preprocessor.transform_obs(e_obs), ...))

        Returns z_env dict:
            "visual":  (1, 1, num_patches, emb_dim)
            "proprio": (1, 1, proprio_emb_dim)
        """
        e_obs = {
            "visual":  rollout_obses["visual"][0, -1:].unsqueeze(0),  # (1, 1, H, W, C)
            "proprio": rollout_obses["proprio"][0, -1:].unsqueeze(0), # (1, 1, proprio_dim)
        }
        e_obs = move_to_device(
            self.data_preprocessor.transform_obs(e_obs), self.device
        )
        with torch.no_grad():
            return self.wm.encode_obs(e_obs) 

    # ------------------------------------------------------------------ #
    # WM rollout + trajectory scoring                                      #
    # ------------------------------------------------------------------ #

    def sample_and_rollout(self, trans_obs_0, z_obs_g):
        """
        Run CEM with a small topk elite set, then return the full last-iter
        sample batch (all num_samples) and its WM rollouts. Probe data
        collection iterates every sample — elites give goal-directed paths,
        non-elites give off-goal trajectories that are more likely to hit
        illegal regions, so the union is class-diverse.

        Returns:
            actions:     (num_samples, rollout_horizon, action_dim)
            z_obses_all: dict with
                "visual"  (num_samples, 1+H+1, num_patches, emb_dim)
                "proprio" (num_samples, 1+H+1, proprio_emb_dim)
        """
        mu    = torch.zeros(self.rollout_horizon, self.action_dim).to(self.device)
        sigma = self.var_scale * torch.ones(self.rollout_horizon, self.action_dim).to(self.device)

        # Expand obs_0 to num_samples batch.
        batched_obs_0 = {
            k: repeat(v, "1 t ... -> n t ...", n=self.num_samples)
            for k, v in trans_obs_0.items()
        }
        z_obs_g_expanded = {
            k: repeat(v, "1 t ... -> n t ...", n=self.num_samples)
            for k, v in z_obs_g.items()
        }

        actions = None
        z_obses_all = None
        for _ in range(self.opt_steps):
            # Sample around current mu/sigma
            actions = torch.randn(
                self.num_samples, self.rollout_horizon, self.action_dim
            ).to(self.device) * sigma + mu
            actions[0] = mu  # set first action to the mean

            with torch.no_grad():
                z_obses_all, _ = self.wm.rollout(batched_obs_0, actions)

            # Score and update mu/sigma from topk elite only.
            losses = self.objective_fn(z_obses_all, z_obs_g_expanded)
            topk_idx = torch.argsort(losses)[:self.topk]
            topk_actions = actions[topk_idx]
            mu    = topk_actions.mean(dim=0)
            sigma = topk_actions.std(dim=0)

        return actions, z_obses_all


    # ------------------------------------------------------------------ #
    # Per-step fidelity check                                              #
    # ------------------------------------------------------------------ #

    def compute_fidelity_metrics(self, wm_z_visual_step, wm_z_proprio_step,
                         rollout_obses, rollout_states,
                         iteration, traj_idx, step, decoded_obs):
        """
        Compute three fidelity signals between WM prediction and env ground truth:

        (A) pixel_mse (secondary metric):
            Decode WM visual latents to pixels via wm.decode_obs
            (vworld_model.py:L115) and compare against env-rendered frame.
            Richer signal than proprio: (C*H*W) dims vs 4, and the decoder
            was trained specifically on these latents.

        (B) div_visual, div_proprio, div (PRIMARY for Threshold):
            Re-encode the env ground-truth obs and compare in embedding space.
            NOTE: we cannot compare wm_z_proprio_step to raw normalised state —
            wm_z_proprio_step is proprio_encoder output (dim=proprio_emb_dim=10)
            while raw state is 4-dimensional; completely different spaces.

        Returns:
            env_frame:   (C,H,W) tensor normalised to [0,1]
            env_state_t: (state_dim,) float tensor — raw post-action state
            pixel_mse:   float
            div_visual:  float
            div_proprio: float
            div:         float
            breached:    bool — True if div > FIDELITY_THRESHOLD
        """
        # (A) Decode WM latents → pixels.
        # decoded_obs["visual"]: (1, 1+H+1, C, H, W)
        # Index step+1 to get the predicted frame at this step
        decoded_frame = decoded_obs["visual"][0, step + 1]   # (C, H, W)

        # Get env-rendered post-action frame.
        # rollout_obses["visual"]: (1, F+1, H, W, C) NFHWC.
        env_frame_np = rollout_obses["visual"]          # (1, F+1, H, W, C)
        env_frame = self.data_preprocessor.transform_obs_visual(
            env_frame_np                # (1, F+1, H, W, C)
        )[0, -1].to(self.device)              # (C, H, W)

        pixel_mse = torch.nn.functional.mse_loss(decoded_frame, env_frame).item()

        # (B) Embedding-space divergence.
        z_env = self.encode_env_frame(rollout_obses)

        # RMS per-dim distance: L2 / sqrt(numel). Shape-invariant — for iid
        # noise of std σ, L2 grows like σ·sqrt(numel), so dividing by
        # sqrt(numel) recovers σ. Makes div_visual and div_proprio
        # directly comparable in the same units.
        div_visual = torch.norm(
            wm_z_visual_step - z_env["visual"].squeeze()
        ).item() / (wm_z_visual_step.numel() ** 0.5)

        div_proprio = torch.norm(
            wm_z_proprio_step - z_env["proprio"].squeeze()
        ).item() / (wm_z_proprio_step.numel() ** 0.5)

        div = div_visual + div_proprio

        log_every = self.cfg.training.get("log_every_traj", 100)
        should_log = (traj_idx % log_every == 0)
        if should_log:
            print(
                f"  [iter={iteration} traj={traj_idx} step={step}] "
                f"pixel_mse={pixel_mse:.6f}  "
                f"div_visual={div_visual:.6f}  div_proprio={div_proprio:.6f}  div={div:.6f}"
            )

        # ------------------------------------------------------------------ #
        # PLACEHOLDER THRESHOLD on div.                                 #
        # Observe the printed values above across several runs before fixing  #
        # this number. 0.5 is a placeholder guess.                            #
        # ------------------------------------------------------------------ #
        breached = div > FIDELITY_THRESHOLD
        if breached and False:
            print(
                f"\n{'!'*70}\n"
                f"  FIDELITY THRESHOLD BREACHED at "
                f"iter={iteration} traj={traj_idx} step={step}\n"
                f"  pixel_mse={pixel_mse:.6f}  "
                f"div_visual={div_visual:.6f}  div_proprio={div_proprio:.6f}  div={div:.6f}\n"
                f"threshold={FIDELITY_THRESHOLD}\n"
                f"  THIS THRESHOLD IS A PLACEHOLDER — observe MSE values\n"
                f"  printed above across multiple runs and set\n"
                f"  FIDELITY_THRESHOLD accordingly.\n"
                f"{'!'*70}\n"
            )

        env_state_t = torch.tensor(rollout_states[-1], dtype=torch.float32)
        return env_frame, env_state_t, pixel_mse, div_visual, div_proprio, div, breached

    # ------------------------------------------------------------------ #
    # Per-trajectory env stepping loop                                     #
    # ------------------------------------------------------------------ #

    def collect_trajectory_data(self, act_seq, z_obses, seed_val, iteration, traj_idx):
        """
        Run a single CONTINUOUS env rollout for the whole trajectory (matching
        evaluator.py:112-116), then slice the returned arrays per WM step to
        compute fidelity metrics and illegal labels. The previous path called
        env.rollout once per WM step, which triggers sim.reset between chunks
        and produces a different physical trajectory than the WM was scored
        against during CEM.

        act_seq: (rollout_horizon, action_dim_total) normalised actions,
                 where action_dim_total = frameskip * action_dim_inner.
        z_obses: dict "visual" (1+H+1, P, D), "proprio" (1+H+1, proprio_emb_dim)
                 — single trajectory already sliced from z_obses_all.

        Returns:
            z_inputs: list of (2*(emb_dim + proprio_emb_dim),) tensors
            labels:   list of scalar float tensors (0 or 1)
        """
        z_inputs = []
        labels   = []
        F = self.frameskip
        T = self.rollout_horizon

        z_for_decode = {
            "visual":  z_obses["visual"].unsqueeze(0),   # (1, 1+H+1, P, D)
            "proprio": z_obses["proprio"].unsqueeze(0),  # (1, 1+H+1, proprio_emb_dim)
        }
        with torch.no_grad():
            decoded_obs, _ = self.wm.decode_obs(z_for_decode)

        # Flatten T chunks of frameskip-bundled actions into one continuous
        # sequence: (T, F*d) -> (T*F, d). Mirrors evaluator.py's
        #   rearrange(actions, "b t (f d) -> b (t f) d")
        exec_actions_t  = rearrange(act_seq, "t (f d) -> (t f) d", f=F).cpu()
        exec_actions_np = self.data_preprocessor.denormalize_actions(exec_actions_t).numpy()

        # ONE env rollout — no sim.reset between WM steps. Returned arrays
        # prepend the start frame, so length is T*F + 1.
        rollout_obses_full, rollout_states_full = self.env.rollout(
            seed_val[0],
            INIT_STATE,
            exec_actions_np,
        )

        for step in range(T):
            # \033[K clears from cursor to end of line, so anything appended
            # later in the loop (e.g. "breached") gets wiped on the next refresh
            # instead of leaving a tail.
            print(
                f"\rIter {iteration+1}/{self.cfg.training.num_iterations}  "
                f"Traj {traj_idx+1}/{self.num_samples}  "
                f"Step {step+1}/{T}\033[K",
                end="", flush=True,
            )
            chunk_start = step * F
            chunk_end   = (step + 1) * F   # slice end is chunk_end + 1 (inclusive of last substep)

            # Per-chunk slice: F+1 entries (chunk-start frame + F substep frames),
            # matching the (1, F+1, ...) / (F+1, state_dim) shapes the old
            # per-call rollout produced.
            chunk_states = rollout_states_full[chunk_start : chunk_end + 1]
            chunk_obses_bt = {
                k: (v[chunk_start : chunk_end + 1].unsqueeze(0)
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v[chunk_start : chunk_end + 1]).unsqueeze(0))
                for k, v in rollout_obses_full.items()
            }

            start_env_state = chunk_states[0].copy()

            # WM predicted embeddings — pair the start (z_obses[step]) with
            # the end (z_obses[step+1]) of this chunk. Position 0 is the
            # encoded ground-truth obs_0; positions 1..H are WM predictions.
            wm_z_visual_start  = z_obses["visual"][step].unsqueeze(0)         # (1, P, emb_dim)
            wm_z_proprio_start = z_obses["proprio"][step].unsqueeze(0)        # (1, proprio_emb_dim)
            wm_z_visual_end    = z_obses["visual"][step + 1].unsqueeze(0)     # (1, P, emb_dim)
            wm_z_proprio_end   = z_obses["proprio"][step + 1].unsqueeze(0)    # (1, proprio_emb_dim)

            env_frame, env_state_t, pixel_mse, div_visual, div_proprio, div, breached = self.compute_fidelity_metrics(
                wm_z_visual_end, wm_z_proprio_end,
                chunk_obses_bt, chunk_states,
                iteration, traj_idx, step, decoded_obs
            )

            if breached:
                print(" breached", end="", flush=True)
                break

            # Step passed — build pair probe input.
            # concat order: start_visual_pool, start_proprio, end_visual_pool, end_proprio.
            z_input = torch.cat([
                wm_z_visual_start.mean(dim=1),   # (emb_dim,)
                wm_z_proprio_start,              # (proprio_emb_dim,)
                wm_z_visual_end.mean(dim=1),     # (emb_dim,)
                wm_z_proprio_end,                # (proprio_emb_dim,)
            ], dim=-1)                           # 2 * (emb_dim + proprio_emb_dim) = 788

            # Label = any sub-step in the frameskip chunk lands in illegal.
            # Captures pass-through cases where start and end are both legal
            # but an intermediary state crossed the illegal region.
            chunk_states_t = torch.tensor(chunk_states, dtype=torch.float32)
            illegal_per_substep = is_illegal_state(chunk_states_t, self.illegal_region)
            label = illegal_per_substep.bool().any().float().to(self.device)

            SHOULD_PRINT = False
            if traj_idx < 50 and step < 50 and SHOULD_PRINT:
                print(
                    f"[DEBUG iter={iteration} traj={traj_idx} step={step}] "
                    f"illegal_region=x[{self.illegal_region['x_min']}, {self.illegal_region['x_max']}] "
                    f"y[{self.illegal_region['y_min']}, {self.illegal_region['y_max']}] | "
                    f"x range=[{chunk_states_t[:,0].min():.3f}, {chunk_states_t[:,0].max():.3f}] "
                    f"y range=[{chunk_states_t[:,1].min():.3f}, {chunk_states_t[:,1].max():.3f}] "
                    f"illegal_per_substep={illegal_per_substep.tolist()} "
                    f"label={label.item()}"
                )

            z_inputs.append(z_input)
            labels.append(label)

            act_exec_chunk = torch.tensor(
                exec_actions_np[chunk_start:chunk_end], dtype=torch.float32
            )

            append_step(
                path=self.cfg.training.db_path,
                z_input=z_input,
                action=act_exec_chunk,
                illegal_label=label,
                illegal_region=self.illegal_region,
                pixel_mse=pixel_mse,
                div_visual=div_visual,
                div_proprio=div_proprio,
                div=div,
                start_env_state=start_env_state,
                new_env_state=chunk_states[-1],
                iteration=iteration,
                traj_idx=traj_idx,
                step=step,
                seed_val=seed_val[0],   # int, not list
            )

        return z_inputs, labels

    # ------------------------------------------------------------------ #
    # Probe training step                                                  #
    # ------------------------------------------------------------------ #

    def train_probe_epochs(self, buffer_z, buffer_y, num_epochs, batch_size):
        """
        Run num_epochs of mini-batch Adam over the full buffer.
        Per-batch pos_weight handles class imbalance locally.

        buffer_z: (N, D) CPU tensor
        buffer_y: (N,)   CPU tensor

        Returns metrics averaged over the final epoch:
            loss, acc, illegal_rate, illegal_acc (None if no positives seen).
        """
        n = len(buffer_z)
        last_losses, last_accs, last_illegal_rates, last_illegal_accs = [], [], [], []

        for epoch in range(num_epochs):
            perm = torch.randperm(n)
            epoch_losses, epoch_accs, epoch_illegal_rates, epoch_illegal_accs = [], [], [], []

            for i in range(0, n, batch_size):
                idx     = perm[i:i + batch_size]
                z_batch = buffer_z[idx].to(self.device)
                y_batch = buffer_y[idx].to(self.device)

                num_pos    = y_batch.sum()
                num_neg    = y_batch.numel() - num_pos
                pos_weight = num_neg / (num_pos + 1e-8)
                loss_fn    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                logits = self.probe(z_batch).squeeze()
                loss   = loss_fn(logits, y_batch)

                self.probe_optimizer.zero_grad()
                loss.backward()
                self.probe_optimizer.step()

                with torch.no_grad():
                    preds        = (torch.sigmoid(logits) >= 0.5).float()
                    acc          = (preds == y_batch).float().mean().item()
                    illegal_rate = y_batch.mean().item()
                    illegal_mask = (y_batch == 1)
                    illegal_acc  = (
                        (preds[illegal_mask] == y_batch[illegal_mask]).float().mean().item()
                        if illegal_mask.sum() > 0 else None
                    )

                epoch_losses.append(loss.item())
                epoch_accs.append(acc)
                epoch_illegal_rates.append(illegal_rate)
                if illegal_acc is not None:
                    epoch_illegal_accs.append(illegal_acc)

            last_losses, last_accs = epoch_losses, epoch_accs
            last_illegal_rates, last_illegal_accs = epoch_illegal_rates, epoch_illegal_accs

        avg_loss         = sum(last_losses) / len(last_losses)
        avg_acc          = sum(last_accs) / len(last_accs)
        avg_illegal_rate = sum(last_illegal_rates) / len(last_illegal_rates)
        avg_illegal_acc  = (sum(last_illegal_accs) / len(last_illegal_accs)) if last_illegal_accs else None

        return avg_loss, avg_acc, avg_illegal_rate, avg_illegal_acc

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def log_test(self):
        """Eval probe on the test pool accumulated across all iterations."""
        if len(self.test_pool_z) == 0:
            log.warning("Test pool is empty; skipping final test eval.")
            return

        test_z = torch.cat(self.test_pool_z, dim=0)
        test_labels = torch.cat(self.test_pool_labels, dim=0)
        test_metrics = self.eval_probe_on_latents(test_z, test_labels)

        log.info(
            f"Test Results "
            f"n_test={len(test_z)} "
            f"test_loss={test_metrics['eval_loss']:.4f} "
            f"test_acc={test_metrics['eval_acc']:.4f} "
            f"test_illegal_rate={test_metrics['eval_illegal_rate']:.4f} "
            f"test_illegal_acc={test_metrics['eval_illegal_acc'] if test_metrics['eval_illegal_acc'] is not None else 'n/a'}"
        )

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #

    def train(self):
        self.probe.train()
        log.info("Setup complete.")
        log.info(
            f"FIDELITY_THRESHOLD = {FIDELITY_THRESHOLD} "
            f"(placeholder — observe printed pixel_mse values first)"
        )

        log_every = 1
        running = dict(
            loss=0.0, acc=0.0, illegal_rate=0.0,
            illegal_acc=0.0, illegal_count=0,
            steps_used=0, time=0.0,
        )

        for iteration in range(self.cfg.training.num_iterations):
            start_time = time.time()
            seed_val = [self.cfg.training.seed + iteration]

            log.info(f"[iter={iteration}/{self.cfg.training.num_iterations}] Getting obs at init/goal states...")

            # 1. Get obs_0 and obs_g from fixed states.
            obs_0_np    = self.get_obs_at_state(INIT_STATE, seed_val)
            obs_g_np    = self.get_obs_at_state(GOAL_STATE, seed_val)
            trans_obs_0 = self.to_wm_obs(obs_0_np)
            trans_obs_g = self.to_wm_obs(obs_g_np)

            log.info(f"[iter={iteration}/{self.cfg.training.num_iterations}] Encoding Obs...")

            # 2. Encode goal once.
            with torch.no_grad():
                z_obs_g = self.wm.encode_obs(trans_obs_g)

            log.info(f"[iter={iteration}] Running CEM ({self.opt_steps} steps, {self.num_samples} samples)...")

            # 3. Sample actions and run batched wm.rollout.
            actions, z_obses_all = self.sample_and_rollout(trans_obs_0, z_obs_g)

            log.info(f"[iter={iteration}] Stepping {self.num_samples} trajectories through env...")

            # 4. Step every sampled trajectory through env with fidelity check.
            all_z_inputs       = []
            all_illegal_labels = []

            for traj_idx in range(self.num_samples):
                act_seq = actions[traj_idx]
                z_obses = {k: v[traj_idx] for k, v in z_obses_all.items()}
                z_inputs, labels = self.collect_trajectory_data(
                    act_seq, z_obses, seed_val, iteration, traj_idx
                )
                all_z_inputs.extend(z_inputs)
                all_illegal_labels.extend(labels)

            if len(all_z_inputs) == 0:
                log.warning(
                    f"iter={iteration}: all trajectories failed fidelity "
                    f"check at step 0 — skipping."
                )
                continue

            # 5. Carve val/test out of the iter's data; remainder is train.
            # Illegals: val and test each get up to 10 (floor goal); the surplus
            # stays in train, which is where they matter most for probe quality.
            # Legals: split 10% to val, 5% to test, rest to train.
            # Test slice is appended to self.test_pool_* and never trained on.
            all_z_inputs = torch.stack(all_z_inputs)           # [N, D]
            all_illegal_labels = torch.tensor(all_illegal_labels, dtype=torch.float32)

            illegal_indices = (all_illegal_labels == 1).nonzero(as_tuple=True)[0].tolist()
            legal_indices   = (all_illegal_labels == 0).nonzero(as_tuple=True)[0].tolist()

            random.shuffle(illegal_indices)
            random.shuffle(legal_indices)

            ILLEGAL_FLOOR = 10
            num_illegal_val  = min(ILLEGAL_FLOOR, len(illegal_indices))
            num_illegal_test = min(ILLEGAL_FLOOR, len(illegal_indices) - num_illegal_val)

            num_legal_val  = int(len(legal_indices) * 0.10)
            num_legal_test = int(len(legal_indices) * 0.05)

            val_illegal  = illegal_indices[:num_illegal_val]
            test_illegal = illegal_indices[num_illegal_val:num_illegal_val + num_illegal_test]
            val_legal    = legal_indices[:num_legal_val]
            test_legal   = legal_indices[num_legal_val:num_legal_val + num_legal_test]

            val_indices   = val_illegal  + val_legal
            test_indices  = test_illegal + test_legal
            train_indices = (
                illegal_indices[num_illegal_val + num_illegal_test:] +
                legal_indices[num_legal_val + num_legal_test:]
            )

            random.shuffle(val_indices)
            random.shuffle(test_indices)
            random.shuffle(train_indices)

            val_z        = all_z_inputs[val_indices]
            val_labels   = all_illegal_labels[val_indices]
            test_z       = all_z_inputs[test_indices]
            test_labels  = all_illegal_labels[test_indices]
            train_z      = all_z_inputs[train_indices]
            train_labels = all_illegal_labels[train_indices]

            if len(train_z) == 0:
                log.warning(f"iter={iteration}: no training data after val/test split — skipping.")
                continue

            if len(test_z) > 0:
                self.test_pool_z.append(test_z.detach().cpu())
                self.test_pool_labels.append(test_labels.detach().cpu())

            # 6. Append this iter's train split to the persistent buffer,
            # then train probe for cfg.probe.epochs over the entire buffer.
            self.buffer_z.append(train_z.detach().cpu())
            self.buffer_y.append(train_labels.detach().cpu())
            buffer_z_all = torch.cat(self.buffer_z, dim=0)
            buffer_y_all = torch.cat(self.buffer_y, dim=0)

            log.info(
                f"[iter={iteration}] Training probe for {self.cfg.probe.epochs} "
                f"epochs over buffer of {len(buffer_z_all)} steps "
                f"(this iter contributed {len(train_z)})..."
            )
            loss, acc, illegal_rate, illegal_acc = self.train_probe_epochs(
                buffer_z_all,
                buffer_y_all,
                num_epochs=self.cfg.probe.epochs,
                batch_size=self.cfg.probe.batch_size,
            )

            # 7. Eval probe on val split
            val_metrics = self.eval_probe_on_latents(val_z, val_labels)

            elapsed = time.time() - start_time

            log.info(
                f"[iter={iteration}] Done. "
                f"loss={loss:.4f} acc={acc:.4f} "
                f"illegal_rate={illegal_rate:.4f} "
                f"illegal_acc={illegal_acc if illegal_acc is not None else 'n/a'} "
                f"val_loss={val_metrics['eval_loss']:.4f} "
                f"val_acc={val_metrics['eval_acc']:.4f} "
                f"val_illegal_rate={val_metrics['eval_illegal_rate']:.4f} "
                f"val_illegal_acc={val_metrics['eval_illegal_acc'] if val_metrics['eval_illegal_acc'] is not None else 'n/a'} "
                f"elapsed={elapsed:.2f}s"
            )

            running["loss"]         += loss
            running["acc"]          += acc
            running["illegal_rate"] += illegal_rate
            running["steps_used"]   += len(all_z_inputs)
            running["time"]         += time.time() - start_time
            if illegal_acc is not None:
                running["illegal_acc"]   += illegal_acc
                running["illegal_count"] += 1

            if (iteration + 1) % log_every == 0:
                running = dict(
                    loss=0.0, acc=0.0, illegal_rate=0.0,
                    illegal_acc=0.0, illegal_count=0,
                    steps_used=0, time=0.0,
                )

        self.log_test()

        probe_save_path = os.path.join(get_original_cwd(), "wm_probe.pt")
        torch.save(self.probe.state_dict(), probe_save_path)
        log.info(f"Saved probe weights to {probe_save_path}")

        log.info(f"Learning Rate: {self.cfg.probe.lr}")
        log.info(f"Rollout Horizon: {self.rollout_horizon}")
        log.info(f"Fidelity Threshold (PLACEHOLDER): {FIDELITY_THRESHOLD}")


    def eval_probe_on_latents(self, z_inputs, labels):
        """Eval probe directly on collected latents (no dset needed)."""
        self.probe.eval()

        # Always coerce to self.device — the test pool is stored on CPU
        # (we .detach().cpu() before appending), so following z_inputs.device
        # would leave tensors on CPU while the probe lives on GPU.
        z_inputs = z_inputs.to(self.device)
        labels   = labels.to(self.device)

        with torch.no_grad():
            logits = self.probe(z_inputs).squeeze()
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss    = loss_fn(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            acc          = (preds == labels).float().mean().item()
            illegal_rate = labels.mean().item()

            illegal_mask = labels == 1
            illegal_acc  = (
                (preds[illegal_mask] == labels[illegal_mask]).float().mean().item()
                if illegal_mask.sum() > 0 else None
            )

        self.probe.train()

        return {
            "eval_loss":         loss.item(),
            "eval_acc":          acc,
            "eval_illegal_rate": illegal_rate,
            "eval_illegal_acc":  illegal_acc,
        }

# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

@hydra.main(config_path="conf", config_name="train_probe_point_maze")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    seed(cfg.training.seed)

    # Load world model 
    repo_root  = Path(get_original_cwd())
    model_path = repo_root / cfg.ckpt_base_path / "outputs" / cfg.model_name
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    train_cfg  = OmegaConf.load(model_path / "hydra.yaml")

    wm = load_model(
        model_ckpt=model_ckpt,
        train_cfg=train_cfg,
        num_action_repeat=train_cfg.num_action_repeat,
        device=device,
    )
    wm.eval()

    # Dataset — used only for Preprocessor normalisation stats.
    # We do NOT use dataset trajectories as action or state supervision.
    _, traj_dset = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )

    train_dset = traj_dset["train"]

    # Preprocessor.
    data_preprocessor = Preprocessor(
        action_mean=train_dset.action_mean,
        action_std=train_dset.action_std,
        state_mean=train_dset.state_mean,
        state_std=train_dset.state_std,
        proprio_mean=train_dset.proprio_mean,
        proprio_std=train_dset.proprio_std,
        transform=train_dset.transform,
    )

    # Env — kwargs mirror env/__init__.py:9-21 (the registration plan.py uses
    # via gym.make("point_maze")), so the planner's env behaves the same here.
    env = PointMazeWrapper(
        maze_spec=U_MAZE,
        reward_type="sparse",
        reset_target=False,
    )

    # Bake num_hist from train_cfg into cfg so the trainer can access it.
    with open_dict(cfg):
        cfg.num_hist = train_cfg.num_hist

    trainer = ProbeRolloutTrainer(
        cfg=cfg,
        wm=wm,
        train_dset=train_dset,
        env=env,
        data_preprocessor=data_preprocessor,
        device=device,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
