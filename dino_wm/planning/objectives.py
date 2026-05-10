import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    # === HARNESS EDIT: probe penalty weight ===
    # Added: per-trajectory P(illegal) from the legislative probe (computed in
    # planning/cem.py:plan()) is added to the MSE loss with this scalar weight.
    # Higher P(illegal) -> higher loss -> trajectory is less likely to land in
    # the CEM topk elite set. lambda=1.0 is the starter value.
    probe_lambda = 1.0
    # === END HARNESS EDIT ===

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
