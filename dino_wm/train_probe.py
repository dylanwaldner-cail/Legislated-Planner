import os
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

from einops import repeat
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
from env.pointmaze.maze_model import U_MAZE

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# FIDELITY THRESHOLD PLACEHOLDER                                       #
# This value (0.5) is a placeholder. Before fixing it, run this       #
# script once and observe the pixel_mse values printed per-step.      #
# The threshold should be set based on the empirical distribution of  #
# those values across multiple runs.                                   #
# ------------------------------------------------------------------ #
FIDELITY_THRESHOLD = 0.5

# Fixed init and goal states for PointMaze (x, y, dx, dy).
# Set based on domain knowledge of the maze layout.
INIT_STATE = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
GOAL_STATE = np.array([3.0, 1.0, 0.0, 0.0], dtype=np.float32)


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

    def __init__(self, cfg, wm, train_dset, val_dset, env, data_preprocessor, device):
        self.cfg = cfg
        self.wm = wm
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.env = env
        self.data_preprocessor = data_preprocessor
        self.device = device

        self.illegal_region = OmegaConf.to_container(cfg.probe.illegal_region, resolve=True)

        # num_hist: context frames WM needs before predicting.
        self.num_hist: int = cfg.num_hist  # baked in from train_cfg at init

        self.rollout_horizon: int = cfg.training.rollout_horizon
        self.num_samples: int     = cfg.training.num_samples
        self.action_dim: int      = train_dset.action_dim

        # topk: one-shot goal-directed filter over num_samples trajectories.
        # Unlike CEM's topk which iteratively updates mu/sigma,
        # we just select the topk best trajectories as training data sources.
        self.topk: int = cfg.training.get("topk", max(1, self.num_samples // 4))

        # Objective function for scoring trajectories against z_obs_g.
        # alpha=0 weights visual only, consistent with the empirical finding
        # that visual embeddings carry more illegal-state signal than proprio.
        self.objective_fn = create_objective_fn(
            alpha=cfg.training.get("alpha", 1.0),
            base=cfg.training.get("base", 1.0),
            mode=cfg.training.get("objective_mode", "last"),
        )

        self.probe = LegislativeProbe(input_dim=cfg.probe.input_dim).to(device)
        self.probe_optimizer = torch.optim.Adam(self.probe.parameters(), lr=cfg.probe.lr)

        num_val = len(val_dset) // 2
        self.val_episodes  = list(range(num_val))
        self.test_episodes = list(range(num_val, len(val_dset)))

        self.var_scale = cfg.training.get("var_scale", 1.0)   # matches cem.yaml
        self.opt_steps = cfg.training.get("opt_steps", 30)    # matches cem.yaml

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
            "visual":  rollout_obses["visual"][1:2],    # (1, H, W, C)
            "proprio": rollout_obses["proprio"][1:2],   # (1, proprio_dim)
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
        Run CEM to find goal-directed action trajectories, then return the
        final mu and the WM rollout from it.

        Returns:
            actions:     (topk, rollout_horizon, action_dim) — top-k actions
                         from final CEM iteration
            z_obses_all: dict with
                "visual"  (topk, 1+H+1, num_patches, emb_dim)
                "proprio" (topk, 1+H+1, proprio_emb_dim)
        """
        mu    = torch.zeros(self.rollout_horizon, self.action_dim).to(self.device)
        sigma = self.var_scale * torch.ones(self.rollout_horizon, self.action_dim).to(self.device)

        # Expand obs_0 to num_samples batch — mirrors cem.py:L76-80.
        batched_obs_0 = {
            k: repeat(v, "1 t ... -> n t ...", n=self.num_samples)
            for k, v in trans_obs_0.items()
        }
        z_obs_g_expanded = {
            k: repeat(v, "1 t ... -> n t ...", n=self.num_samples)
            for k, v in z_obs_g.items()
        }

        for _ in range(self.opt_steps):
            # Sample around current mu/sigma — mirrors cem.py:L68-74.
            actions = torch.randn(
                self.num_samples, self.rollout_horizon, self.action_dim
            ).to(self.device) * sigma + mu
            actions[0] = mu  # mirrors cem.py:L75

            with torch.no_grad():
                z_obses_all, _ = self.wm.rollout(batched_obs_0, actions)

            # Score and update mu/sigma from topk — mirrors cem.py:L90-95.
            losses = self.objective_fn(z_obses_all, z_obs_g_expanded)
            topk_idx = torch.argsort(losses)[:self.topk]
            topk_actions = actions[topk_idx]
            mu    = topk_actions.mean(dim=0)
            sigma = topk_actions.std(dim=0)

        # Final rollout from top-k actions of last iteration.
        batched_obs_0_topk = {
            k: repeat(v, "1 t ... -> n t ...", n=self.topk)
            for k, v in trans_obs_0.items()
        }
        with torch.no_grad():
            z_obses_all, _ = self.wm.rollout(batched_obs_0_topk, topk_actions)

        return topk_actions, z_obses_all


    # ------------------------------------------------------------------ #
    # Per-step fidelity check                                              #
    # ------------------------------------------------------------------ #

    def compute_fidelity_metrics(self, wm_z_visual_step, wm_z_proprio_step,
                         rollout_obses, rollout_states,
                         iteration, traj_idx, step, decoded_obs):
        """
        Compute three fidelity signals between WM prediction and env ground truth:

        (A) pixel_mse (PRIMARY — used for threshold):
            Decode WM visual latents to pixels via wm.decode_obs
            (vworld_model.py:L115) and compare against env-rendered frame.
            Richer signal than proprio: (C*H*W) dims vs 4, and the decoder
            was trained specifically on these latents.

        (B) div_visual, div_proprio (secondary — logged only):
            Re-encode the env ground-truth obs and compare in embedding space,
            mirroring evaluator.py:L100-103 (div_visual_emb, div_proprio_emb).
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
        # rollout_obses["visual"]: (1, 2, H, W, C) uint8 — NHWC.
        # Index [0,1] = post-action frame.
        env_frame_np = rollout_obses["visual"][1]          # (H, W, C) uint8
        env_frame = self.data_preprocessor.transform_obs_visual(
            env_frame_np                # (H, W, C)
        ).to(self.device)              # (C, H, W)

        pixel_mse = torch.nn.functional.mse_loss(decoded_frame, env_frame).item()

        # (B) Embedding-space divergence.
        z_env = self.encode_env_frame(rollout_obses)

        div_visual = torch.norm(
            wm_z_visual_step - z_env["visual"].squeeze()
        ).item()
        div_proprio = torch.norm(
            wm_z_proprio_step - z_env["proprio"].squeeze()
        ).item()
        div = div_visual + div_proprio

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
        if breached:
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

        env_state_t = torch.tensor(rollout_states[1], dtype=torch.float32)

        return env_frame, env_state_t, pixel_mse, div_visual, div_proprio, div, breached

    # ------------------------------------------------------------------ #
    # Per-trajectory env stepping loop                                     #
    # ------------------------------------------------------------------ #

    def collect_trajectory_data(self, act_seq, z_obses, seed_val, iteration, traj_idx):
        """
        Step a single action trajectory through the env one action at a time,
        checking fidelity at each step. Collect (z_input, illegal_label) pairs
        for all steps that pass the fidelity check.

        act_seq: (rollout_horizon, action_dim) normalised actions
        z_obses: dict "visual" (1+H+1, P, D), "proprio" (1+H+1, proprio_emb_dim)
                 — single trajectory already sliced from z_obses_all

        Returns:
            z_inputs: list of (emb_dim + proprio_emb_dim,) tensors
            labels:   list of scalar float tensors (0 or 1)
        """
        z_inputs = []
        labels   = []
        current_state = INIT_STATE[np.newaxis, :]   # (1, state_dim)

        z_for_decode = {
            "visual":  z_obses["visual"].unsqueeze(0),   # (1, 1+H+1, P, D)
            "proprio": z_obses["proprio"].unsqueeze(0),  # (1, 1+H+1, proprio_emb_dim)
        }
        with torch.no_grad():
            decoded_obs, _ = self.wm.decode_obs(z_for_decode)


        for step in range(self.rollout_horizon):
            # Denormalise action for env — mirrors plan.py:L227.
            act_exec = self.data_preprocessor.denormalize_actions(
                act_seq[step].cpu().unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
            )

            # Step env one action at a time — mirrors plan.py:L228-233.
            rollout_obses, rollout_states = self.env.rollout(
                seed_val[0],
                current_state[0],       # (state_dim,) not (1, state_dim)
                act_exec.numpy()[0, 0]  # (action_dim,) not (1, 1, action_dim)
            )
            # rollout_states: (1, 2, state_dim) — [reset_state, post_action_state]

            # WM predicted embedding at this step.
            # z_obses has no batch dim (already sliced from z_obses_all).
            # Index step+1: position 0 = context frame.
            wm_z_visual_step  = z_obses["visual"][step + 1]    # (P, emb_dim)
            wm_z_proprio_step = z_obses["proprio"][step + 1]   # (proprio_emb_dim,)

            env_frame, env_state_t, pixel_mse, div_visual, div_proprio, div, breached = self.compute_fidelity_metrics(
                wm_z_visual_step, wm_z_proprio_step,
                rollout_obses, rollout_states,
                iteration, traj_idx, step, decoded_obs
            )

            if breached:
                break

            # Step passed — build probe input.
            # Mean-pool visual patches then concat proprio embedding.
            # Empirically visual carries more illegal-state signal than proprio.
            z_input = torch.cat([
                wm_z_visual_step.mean(dim=0),   # (emb_dim,)
                wm_z_proprio_step,              # (proprio_emb_dim,)
            ], dim=-1)                          # (emb_dim + proprio_emb_dim = 394)

            # Label from env ground-truth state, not WM prediction.
            label = is_illegal_state(
                env_state_t.unsqueeze(0), self.illegal_region
            ).float().to(self.device).squeeze(0)

            z_inputs.append(z_input)
            labels.append(label)

            current_state = rollout_states[1, :][np.newaxis, :]   # advance

            append_step(
                path=self.cfg.training.db_path,
                z_input=z_input,
                illegal_label=label,
                illegal_region=self.illegal_region,
                pixel_mse=pixel_mse,
                div_visual=div_visual,
                div_proprio=div_proprio,
                div=div,
                env_state=rollout_states[1],
                iteration=iteration,
                traj_idx=traj_idx,
                step=step,
                seed_val=seed_val[0],   # int, not list
            )

        return z_inputs, labels

    # ------------------------------------------------------------------ #
    # Probe training step                                                  #
    # ------------------------------------------------------------------ #

    def train_probe_step(self, all_z_inputs, all_illegal_labels):
        """
        Run one gradient step on the probe given a batch of (z_input, label) pairs.
        Uses per-batch pos_weight to handle class imbalance.

        Returns:
            loss:         float
            acc:          float
            illegal_rate: float
            illegal_acc:  float or None (if no positive examples in batch)
        """
        z_batch = torch.stack(all_z_inputs)        # (N, 394)
        y_batch = torch.stack(all_illegal_labels)  # (N,)

        num_pos    = y_batch.sum()
        num_neg    = y_batch.numel() - num_pos
        pos_weight = num_neg / (num_pos + 1e-8)
        loss_fn    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        logits = self.probe(z_batch).squeeze(-1)
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

        return loss.item(), acc, illegal_rate, illegal_acc

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def log_iteration(self, iteration, running, log_every):
        """Log running averages and a validation pass every log_every iterations."""
        avg_illegal_acc = (
            running["illegal_acc"] / running["illegal_count"]
            if running["illegal_count"] > 0 else 0.0
        )
        val_metrics = eval_probe(
            probe=self.probe,
            wm=self.wm,
            dset=self.val_dset,
            eval_episodes=self.val_episodes,
            illegal_region=self.illegal_region,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            device=self.device,
        )
        log.info(
            f"iter={iteration + 1} "
            f"steps_used={running['steps_used']} "
            f"loss={running['loss'] / log_every:.4f} "
            f"acc={running['acc'] / log_every:.4f} "
            f"illegal_rate={running['illegal_rate'] / log_every:.4f} "
            f"illegal_acc={avg_illegal_acc:.4f} "
            f"time/iter={running['time'] / log_every:.4f}s "
            f"val_loss={val_metrics['eval_loss']:.4f} "
            f"val_acc={val_metrics['eval_acc']:.4f} "
            f"val_illegal_rate={val_metrics['eval_illegal_rate']:.4f} "
            f"val_illegal_acc={val_metrics['eval_illegal_acc']:.4f}"
        )

    def log_test(self):
        """Run final evaluation on the held-out test split and log results."""
        test_metrics = eval_probe(
            probe=self.probe,
            wm=self.wm,
            dset=self.val_dset,
            eval_episodes=self.test_episodes,
            illegal_region=self.illegal_region,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            device=self.device,
        )
        log.info(
            f"Test Results "
            f"test_loss={test_metrics['eval_loss']:.4f} "
            f"test_acc={test_metrics['eval_acc']:.4f} "
            f"test_illegal_rate={test_metrics['eval_illegal_rate']:.4f} "
            f"test_illegal_acc={test_metrics['eval_illegal_acc']:.4f}"
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

        log_every = 50
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

            # 2. Encode goal once — mirrors cem.py:L83.
            with torch.no_grad():
                z_obs_g = self.wm.encode_obs(trans_obs_g)

            log.info(f"[iter={iteration}] Running CEM ({self.opt_steps} steps, {self.num_samples} samples)...")

            # 3. Sample actions and run batched wm.rollout.
            actions, z_obses_all = self.sample_and_rollout(trans_obs_0)

            log.info(f"[iter={iteration}] Stepping {self.topk} trajectories through env...")

            # 4. Step each top-k trajectory through env with fidelity check.
            all_z_inputs       = []
            all_illegal_labels = []

            for traj_idx in range(self.topk):
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

            # 5. Train probe on fidelity-passing steps.
            log.info(f"[iter={iteration}] Training probe on {len(all_z_inputs)} steps...")

            loss, acc, illegal_rate, illegal_acc = self.train_probe_step(
                all_z_inputs, all_illegal_labels
            )

            log.info(f"[iter={iteration}] Done. loss={loss:.4f} acc={acc:.4f} illegal_rate={illegal_rate:.4f} illegal_acc={illegal_acc if illegal_acc is not None else 'n/a'} elapsed={elapsed:.2f}s")

            running["loss"]         += loss
            running["acc"]          += acc
            running["illegal_rate"] += illegal_rate
            running["steps_used"]   += len(all_z_inputs)
            running["time"]         += time.time() - start_time
            if illegal_acc is not None:
                running["illegal_acc"]   += illegal_acc
                running["illegal_count"] += 1

            if (iteration + 1) % log_every == 0:
                self.log_iteration(iteration, running, log_every)
                running = dict(
                    loss=0.0, acc=0.0, illegal_rate=0.0,
                    illegal_acc=0.0, illegal_count=0,
                    steps_used=0, time=0.0,
                )

        self.log_test()
        log.info(f"Learning Rate: {self.cfg.probe.lr}")
        log.info(f"Rollout Horizon: {self.rollout_horizon}")
        log.info(f"Fidelity Threshold (PLACEHOLDER): {FIDELITY_THRESHOLD}")


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

    # Load world model — mirrors plan.py:L278-284.
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
    val_dset   = traj_dset["valid"]

    # Preprocessor — mirrors plan.py PlanWorkspace.__init__ (plan.py:L143-152).
    data_preprocessor = Preprocessor(
        action_mean=train_dset.action_mean,
        action_std=train_dset.action_std,
        state_mean=train_dset.state_mean,
        state_std=train_dset.state_std,
        proprio_mean=train_dset.proprio_mean,
        proprio_std=train_dset.proprio_std,
        transform=train_dset.transform,
    )

    # Env
    env = PointMazeWrapper(maze_spec=U_MAZE)

    # Bake num_hist from train_cfg into cfg so the trainer can access it.
    with open_dict(cfg):
        cfg.num_hist = train_cfg.num_hist

    trainer = ProbeRolloutTrainer(
        cfg=cfg,
        wm=wm,
        train_dset=train_dset,
        val_dset=val_dset,
        env=env,
        data_preprocessor=data_preprocessor,
        device=device,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
