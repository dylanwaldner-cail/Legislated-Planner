"""Offline diagnostic: does the cube survive a PREDICTED WM step?

Localizes the planner failure (C+D in the failure map) without any sim. For known
(history, action) windows from the dataset, roll the WM forward and run the cube probe
on the WM's PREDICTED latent, comparing to the TRUE post-action cube position. Compare
against the probe on the ENCODED latent of the real frame (the floor, ~0.015 m).

  encoded-probe L2  ~= 0.015  (sanity: confirms transforms are right)
  predicted-probe L2:
     ~= encoded floor   -> WM moves the cube faithfully in its predicted latents AND the
                           probe reads them -> planner failure is the CEM/optimizer, not C/D.
     >> encoded floor   -> the predicted latent does NOT carry the moved cube -> C+D
                           confirmed; fix = train probe on predicted latents / add a
                           cube-position regression head to WM training.

No sim needed: WM (frozen DINO + predictor) + probe + the .pth dataset. Run with the
container python OR the host conda env (torch + cached dinov2 hub):
    python scripts/wm_cube_pred_check.py \
        --model_dir outputs/2026-06-25/16-46-57 --epoch 20 \
        --data_dir data/isaaclab_stroke_1500 --probe probe_cube_1500.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probe_cube_position import MLP, _spatial_pool_grid  # same probe head + pooling

CUBE_OFF = 18  # cube xy in the 31-D state


def load_wm(model_dir, epoch, device):
    train_cfg = OmegaConf.load(Path(model_dir) / ".hydra" / "config.yaml")
    ckpt = Path(model_dir) / "checkpoints" / f"model_{epoch}.pth"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    # frozen DINO encoder isn't saved in the ckpt -> re-instantiate from config (like load_model)
    encoder = payload["encoder"] if "encoder" in payload else hydra.utils.instantiate(train_cfg.encoder)
    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=payload["proprio_encoder"],
        action_encoder=payload["action_encoder"],
        predictor=payload["predictor"],
        decoder=None,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=train_cfg.num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device).eval()
    return model, train_cfg


def load_probe(path, device):
    p = torch.load(path, map_location=device, weights_only=False)
    mlp = MLP(p["d_in"]).to(device)
    mlp.load_state_dict(p["state_dict"])
    mlp.eval()
    return mlp, p


@torch.no_grad()
def probe_xy(mlp, pinfo, tokens, device):
    """tokens (b, P, D) -> cube (x,y) meters, replicating probe_cube_position."""
    X = _spatial_pool_grid(tokens, pinfo["pool_grid"])              # (b, grid*grid*D)
    mu = torch.as_tensor(pinfo["x_mu"], device=device, dtype=X.dtype)
    sd = torch.as_tensor(pinfo["x_sd"], device=device, dtype=X.dtype)
    pn = mlp((X - mu) / sd).cpu().numpy()
    return pn * pinfo["y_sd"] + pinfo["y_mu"]                       # (b, 2) meters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/2026-06-25/16-46-57")
    ap.add_argument("--epoch", default="20")
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--probe", default="probe_cube_1500.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=5, help="multi-step open-loop horizon")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device

    wm, tcfg = load_wm(args.model_dir, args.epoch, dev)
    num_hist = int(tcfg.num_hist)
    mlp, pinfo = load_probe(args.probe, dev)
    print(f"[load] WM num_hist={num_hist} | probe pool_grid={pinfo['pool_grid']} d_in={pinfo['d_in']}")

    p = Path(args.data_dir)
    states = torch.load(p / "states.pth").float().numpy()           # (E,T,31)
    actions = torch.load(p / "actions.pth").float().numpy()         # (E,T,4)
    proprio = torch.load(p / "proprio.pth").float().numpy()         # (E,T,18)
    E, T, _ = states.shape
    # normalization stats (same form as the dataset/preprocessor: (x-mean)/std)
    a_mean = actions.reshape(-1, 4).mean(0); a_std = actions.reshape(-1, 4).std(0) + 1e-6
    p_mean = proprio.reshape(-1, 18).mean(0); p_std = proprio.reshape(-1, 18).std(0) + 1e-6

    def load_vis(ei, f0, n):
        """frames f0..f0+n-1 of episode ei -> (n,3,224,224) transformed (/255, *2-1)."""
        vid = torch.load(p / "obses" / f"episode_{ei:05d}.pth")[f0:f0 + n]  # (n,H,W,3) uint8
        x = vid.float().permute(0, 3, 1, 2) / 255.0
        return (x * 2.0 - 1.0)                                       # Normalize(0.5,0.5)

    rng = np.random.RandomState(args.seed)
    H = args.horizon
    # a window needs num_hist init frames + H future actions/frames
    max_f = T - (num_hist + H)
    windows = [(int(rng.randint(0, E)), int(rng.randint(0, max_f + 1))) for _ in range(args.n_windows)]

    err_pred1, err_predH, err_enc = [], [], []   # 1-step pred, H-step pred, encoded floor
    move_true, move_pred1 = [], []               # how far the cube ACTUALLY moved vs predicted (1-step)
    d_start = []                                  # |stroke start - cube| (distance from the contact region)

    for i in range(0, len(windows), args.batch):
        batch = windows[i:i + args.batch]
        vis0 = torch.stack([load_vis(e, f, num_hist) for e, f in batch]).to(dev)        # (b,nh,3,H,W)
        pro0 = torch.stack([torch.tensor((proprio[e, f:f + num_hist] - p_mean) / p_std)
                            for e, f in batch]).float().to(dev)                          # (b,nh,18)
        # actions: init nh + H future (normalized). rollout takes act[:, :nh] as init, rest as future.
        acts = torch.stack([torch.tensor((actions[e, f:f + num_hist + H] - a_mean) / a_std)
                            for e, f in batch]).float().to(dev)                          # (b,nh+H,4)
        obs0 = {"visual": vis0, "proprio": pro0}

        # ---- 1-step: predict frame f+nh from the nh history frames + action a_{f+nh-1} ----
        with torch.no_grad():
            z1, _ = wm.rollout(obs_0=obs0, act=acts[:, :num_hist])    # adds 1 predicted frame
        pred1 = probe_xy(mlp, pinfo, z1["visual"][:, -1], dev)        # (b,2)
        # ---- H-step open-loop ----
        with torch.no_grad():
            zH, _ = wm.rollout(obs_0=obs0, act=acts)                  # predicts H frames ahead
        predH = probe_xy(mlp, pinfo, zH["visual"][:, -1], dev)

        for j, (e, f) in enumerate(batch):
            cube_hist = states[e, f + num_hist - 1, CUBE_OFF:CUBE_OFF + 2]   # last seen cube
            cube_next = states[e, f + num_hist, CUBE_OFF:CUBE_OFF + 2]       # true after 1 stroke
            cube_H = states[e, f + num_hist - 1 + H, CUBE_OFF:CUBE_OFF + 2]  # true after H strokes
            err_pred1.append(np.linalg.norm(pred1[j] - cube_next))
            err_predH.append(np.linalg.norm(predH[j] - cube_H))
            move_true.append(np.linalg.norm(cube_next - cube_hist))
            move_pred1.append(np.linalg.norm(pred1[j] - cube_hist))
            d_start.append(np.linalg.norm(actions[e, f + num_hist - 1, :2] - cube_hist))  # start dist from cube
            # encoded floor: probe the REAL frame f+nh
            ev = load_vis(e, f + num_hist, 1).to(dev)
            with torch.no_grad():
                ze = wm.encode_obs({"visual": ev[None],
                                    "proprio": torch.zeros(1, 1, 18, device=dev)})["visual"][:, 0]
            enc = probe_xy(mlp, pinfo, ze, dev)[0]
            err_enc.append(np.linalg.norm(enc - cube_next))

    f = lambda a: (np.mean(a), np.median(a))
    print(f"\n[probe floor]  ENCODED real frame -> true cube:  mean {f(err_enc)[0]:.4f}  median {f(err_enc)[1]:.4f} m"
          f"   (should be ~0.015; validates transforms)")
    print(f"[1-step pred]  PREDICTED latent  -> true cube:    mean {f(err_pred1)[0]:.4f}  median {f(err_pred1)[1]:.4f} m")
    print(f"[{H}-step pred] PREDICTED latent  -> true cube:    mean {f(err_predH)[0]:.4f}  median {f(err_predH)[1]:.4f} m")
    print(f"\n[motion]  true cube move/step:      mean {f(move_true)[0]:.4f} m")
    print(f"[motion]  predicted cube move/step: mean {f(move_pred1)[0]:.4f} m   "
          f"(ratio>1 => WM OVER-predicts push distance; <1 => under)")
    ratio = np.mean(move_pred1) / max(np.mean(move_true), 1e-6)
    print(f"[motion]  predicted/true move ratio: {ratio:.2f}  (1.0 = faithful; ~0 = WM keeps cube ~put)")

    # contact vs miss: aggregate 1-step err is dominated by MISS strokes (cube ~still ->
    # trivially predictable). Planning only relies on CONTACT strokes. Split them out.
    e1 = np.array(err_pred1); mt = np.array(move_true)
    contact = mt > 0.02
    print(f"\n[contact split]  1-step PREDICTED-probe err vs whether the stroke moved the cube:")
    if contact.any():
        print(f"  CONTACT (moved >2cm, n={int(contact.sum())}):  mean {e1[contact].mean():.4f} m"
              f"   <- the strokes the planner actually relies on")
    if (~contact).any():
        print(f"  MISS    (cube ~still, n={int((~contact).sum())}): mean {e1[~contact].mean():.4f} m")

    # bin 1-step prediction by START distance from the cube. The hallucination signature:
    # for FAR-start strokes (misses, true_move~0), does the WM predict the cube STAYS
    # (pred_move~0, honest) or MOVES (pred_move>0, hallucination)? If pred_err / pred_move
    # grow with distance, the WM is unreliable far from the cube -> near-cube trust region
    # is justified. (Caveat: these are DATASET starts, random; the CEM adversarially finds
    # the worst far-start combos, so this UNDER-states the planner's exploit.)
    d = np.array(d_start); mp = np.array(move_pred1)
    print(f"\n[start-distance bins]  |stroke_start - cube| -> WM behavior  (CELL={2*0.0667:.3f}m wide)")
    print(f"  {'bin (m)':<12}{'n':>5}{'pred_err':>10}{'true_move':>11}{'pred_move':>11}")
    for lo, hi in [(0, 0.06), (0.06, 0.12), (0.12, 0.20), (0.20, 0.35), (0.35, 9.0)]:
        m = (d >= lo) & (d < hi)
        if m.sum() == 0:
            continue
        print(f"  {f'{lo:.2f}-{hi:.2f}':<12}{int(m.sum()):>5}{e1[m].mean():>10.4f}"
              f"{mt[m].mean():>11.4f}{mp[m].mean():>11.4f}")


if __name__ == "__main__":
    main()
