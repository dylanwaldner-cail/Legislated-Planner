import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_objective_fn(alpha, base, mode="last", probe_lambda=1.0):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
        probe_lambda: float (HARNESS EDIT). Scales the per-traj probe penalty
            added to the loss. Set to 0.0 from the CLI to disable the probe
            entirely (standard MSE-only planning baseline). The probe still
            runs in cem.py for logging, but doesn't shift the loss ranking.
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    # === HARNESS EDIT: added probe_out arg, default None so non-CEM callers
    # (e.g. planning/gd.py) keep working with the old 2-arg signature ===
    def objective_fn_last(z_obs_pred, z_obs_tgt, probe_out=None):
    # === END HARNESS EDIT ===
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            probe_out: tensor (B, T_pairs) -- HARNESS EDIT, per-chunk P(illegal) from legislative probe
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        # === HARNESS EDIT: probe penalty (skipped if caller didn't pass probe_out) ===
        # probe_out is shape (B, T_pairs) of per-chunk P(illegal). We sum over
        # the time axis so that (a) a confidently-illegal chunk gives a ~1.0
        # jump (max-like behavior on its own), and (b) additional illegal
        # chunks add linearly (penalizes duration without saturating like
        # noisy-OR). No time-position weighting -- early-vs-late illegal is
        # equally bad for a safety signal.
        if probe_out is not None:
            loss = loss + probe_lambda * probe_out.sum(dim=1)
        # === END HARNESS EDIT ===
        return loss

    # === HARNESS EDIT: added probe_out arg, default None so non-CEM callers
    # (e.g. planning/gd.py) keep working with the old 2-arg signature ===
    def objective_fn_all(z_obs_pred, z_obs_tgt, probe_out=None):
    # === END HARNESS EDIT ===
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            probe_out: tensor (B, T_pairs) -- HARNESS EDIT, per-chunk P(illegal) from legislative probe
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        # === HARNESS EDIT: probe penalty (skipped if caller didn't pass probe_out) ===
        # probe_out is shape (B, T_pairs) of per-chunk P(illegal). We sum over
        # the time axis so that (a) a confidently-illegal chunk gives a ~1.0
        # jump (max-like behavior on its own), and (b) additional illegal
        # chunks add linearly (penalizes duration without saturating like
        # noisy-OR). No time-position weighting -- early-vs-late illegal is
        # equally bad for a safety signal.
        if probe_out is not None:
            loss = loss + probe_lambda * probe_out.sum(dim=1)
        # === END HARNESS EDIT ===
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError


# ============================================================================
# TWO DISTINCT PROBES live in this objective — don't conflate them:
#
#   1. POSITION probe (this section, `_PositionProbe` / pos_probe_*): reads the
#      cube (x,y) off the latent to define the GOAL-REACHING term ("get the cube
#      to the goal cell"). This is the TASK objective.
#   2. NORMATIVE probe (`probe_out` arg, weighted by `probe_lambda`): per-chunk
#      P(illegal) from the legislative/deontic probe — the CONSTRAINT term ("don't
#      do the forbidden thing"). It's produced & passed in by cem.py only; gd.py
#      leaves it None, so it's INERT under GD/mpc_gd. Defined elsewhere, not here.
#
# Together: position probe = where to go, normative probe = what's off-limits.
#
# HARNESS: probe-based planning objective (independent of create_objective_fn).
# Primary term = squared-L2 between the cube (x,y) read off the WM's PREDICTED
# latent and the cube (x,y) read off the GOAL latent (same frozen probe, so any
# constant probe bias cancels). Latent MSE stays as a SECONDARY manifold
# regularizer (limits the planner gaming the probe); proprio is TERTIARY.
# `use_probe=False` drops the position-probe term -> identical to the MSE
# baseline, for a clean A/B (run with objective.use_probe=true vs =false).
# ============================================================================
class _ProbeMLP(nn.Module):
    """Mirror of probe_cube_position.MLP so its saved state_dict (keys 'net.*') loads."""
    def __init__(self, d_in, hidden=256, d_out=2, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class _PositionProbe:
    """Differentiable cube (x, y) readout from DINO patch tokens (checkpoint from
    probe_cube_position.py). Frozen but differentiable, so GD's gradient flows
    action -> latent -> probe -> loss. NOTE: trained on ENCODED latents; applying
    it to PREDICTED latents is the OOD/overconfidence risk we flagged — watch the
    imagined-vs-real cube-L2 gap in the logs, escalate to a calibrated probe only
    if it shows up."""

    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location=device)
        self.grid = int(ck["pool_grid"])
        self.mlp = _ProbeMLP(int(ck["d_in"])).to(device).eval()
        self.mlp.load_state_dict(ck["state_dict"])
        for prm in self.mlp.parameters():
            prm.requires_grad_(False)
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.x_mu, self.x_sd = f(ck["x_mu"]), f(ck["x_sd"])   # (1, d_in)
        self.y_mu, self.y_sd = f(ck["y_mu"]), f(ck["y_sd"])   # (1, 2)
        self.device = device

    def __call__(self, tokens):
        """tokens: (B, P, D) DINO patch tokens for ONE frame -> (B, 2) cube xy (meters)."""
        b, p, d = tokens.shape
        s = int(round(p ** 0.5))
        g = tokens.to(self.device).float().reshape(b, s, s, d).permute(0, 3, 1, 2)
        g = F.adaptive_avg_pool2d(g, self.grid).reshape(b, -1)   # (B, grid*grid*D)
        y = self.mlp((g - self.x_mu) / self.x_sd)               # normalized xy
        return y * self.y_sd + self.y_mu                        # meters


def create_probe_objective_fn(alpha, base, mode="last", probe_lambda=1.0,
                              pos_probe_path=None, pos_probe_weight=50.0,
                              visual_weight=1.0, device="cuda", use_probe=True):
    """Like create_objective_fn but with the cube-position probe as the primary
    term. Set use_probe=False (or pos_probe_path=null) to fall back to the exact
    MSE baseline. pos_probe_weight is large because the probe term is squared-L2 in
    METERS^2 (~0.03 for a 1-cell error) while latent MSE is O(0.1-1)."""
    metric = nn.MSELoss(reduction="none")
    position_probe = None
    if use_probe and pos_probe_path:
        pp = pos_probe_path
        if not os.path.isabs(pp) and not os.path.exists(pp):
            try:  # plan.py runs under hydra (cwd = run dir); resolve vs launch dir
                from hydra.utils import get_original_cwd
                pp = os.path.join(get_original_cwd(), pp)
            except Exception:
                pass
        position_probe = _PositionProbe(pp, device)
        print(f"[objective] cube-position probe ON: {pp} | "
              f"weights pos={pos_probe_weight} visual={visual_weight} proprio(alpha)={alpha}")
    else:
        print(f"[objective] cube-position probe OFF (MSE baseline) | "
              f"weights visual={visual_weight} proprio(alpha)={alpha}")
    _dbg = {"n": 0}  # print component magnitudes for the first couple of calls

    def _probe_term(z_obs_pred, z_obs_tgt):
        pred_xy = position_probe(z_obs_pred["visual"][:, -1])    # (B, 2) meters
        goal_xy = position_probe(z_obs_tgt["visual"][:, -1])     # (B, 2) meters
        return ((pred_xy - goal_xy) ** 2).sum(dim=1)             # (B,) squared-L2 (m^2)

    def objective_fn_last(z_obs_pred, z_obs_tgt, probe_out=None):
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = visual_weight * loss_visual + alpha * loss_proprio
        if position_probe is not None:
            loss_pos = _probe_term(z_obs_pred, z_obs_tgt)
            if _dbg["n"] < 2:
                print(f"[objective dbg] pos*w={float((pos_probe_weight*loss_pos).mean()):.4f}  "
                      f"visual*w={float((visual_weight*loss_visual).mean()):.4f}  "
                      f"proprio*a={float((alpha*loss_proprio).mean()):.4f}")
                _dbg["n"] += 1
            loss = pos_probe_weight * loss_pos + loss
        if probe_out is not None:
            loss = loss + probe_lambda * probe_out.sum(dim=1)
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, probe_out=None):
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = visual_weight * loss_visual + alpha * loss_proprio
        if position_probe is not None:
            loss = pos_probe_weight * _probe_term(z_obs_pred, z_obs_tgt) + loss
        if probe_out is not None:
            loss = loss + probe_lambda * probe_out.sum(dim=1)
        return loss

    # expose the cube-position probe so the planner can ESTIMATE the cube from an
    # observation latent (e.g. MPC warm-start) -- obs-only, no ground-truth state.
    objective_fn_last.position_probe = position_probe
    objective_fn_all.position_probe = position_probe

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError
