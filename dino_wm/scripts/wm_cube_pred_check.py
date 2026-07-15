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
        --data_dir data/isaaclab_stroke_1500 --probe probes/weights/probe_cube_1500.pth
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

from probes.probe_cube_position import MLP, _spatial_pool_grid, gm  # same head + pooling; gm=grid_metadata

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
    # Two SEPARATE z-scores with two different stat sets (not "subtract then add back"):
    # x_mu/x_sd are FEATURE stats -> standardize the input so the MLP sees the same
    # distribution it trained on. y_mu/y_sd are LABEL stats.
    mu = torch.as_tensor(pinfo["x_mu"], device=device, dtype=X.dtype)
    sd = torch.as_tensor(pinfo["x_sd"], device=device, dtype=X.dtype)
    pn = mlp((X - mu) / sd).cpu().numpy()
    # The MLP was TRAINED to predict z-scored targets (probe_cube_position.py:272-273,293),
    # so its raw output is in standardized cube-position units, not meters. Invert that
    # target z-score (* y_sd + y_mu) to recover physical meters (comparable to CELL=0.133 m).
    return pn * pinfo["y_sd"] + pinfo["y_mu"]                       # (b, 2) meters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/2026-06-25/16-46-57")
    ap.add_argument("--epoch", default="20")
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--probe", default="probes/weights/probe_cube_1500.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split_ratio", type=float, default=0.9,
                    help="train fraction of the seed-42 split; eval runs on ALL held-out val episodes")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=5, help="multi-step open-loop horizon")
    ap.add_argument("--seed", type=int, default=0)
    # Restrict eval to windows whose transition stroke is IN the RRT planner's operating domain
    # (aimed contact: aim & push in range) -- the manifold the planner actually queries, so the WM
    # error reflects what matters at plan time rather than the broad train/val mix. Same definition
    # as train.py's planner_domain flag.
    ap.add_argument("--planner_domain", action="store_true",
                    help="only evaluate strokes in the RRT domain (aim & push in the ranges below)")
    ap.add_argument("--aim_lo", type=float, default=0.085)
    ap.add_argument("--aim_hi", type=float, default=0.125)
    ap.add_argument("--push_lo", type=float, default=0.14)
    ap.add_argument("--push_hi", type=float, default=0.21)
    args = ap.parse_args()
    dev = args.device

    wm, tcfg = load_wm(args.model_dir, args.epoch, dev)
    num_hist = int(tcfg.num_hist)
    mlp, pinfo = load_probe(args.probe, dev)
    print(f"[load] WM num_hist={num_hist} | PROBE={args.probe} "
          f"(source={pinfo.get('source', '?')} horizons={pinfo.get('pred_horizons', '?')} "
          f"pool_grid={pinfo['pool_grid']} d_in={pinfo['d_in']})")

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

    H = args.horizon
    # VALIDATION EPISODES ONLY -- reproduce training's seed-42 split (datasets/traj_dset.py
    # split_traj_datasets: randperm(E, seed=42); the last (1-train_fraction) episodes are val), so the
    # WM never trained on these windows -> an honest generalization number.
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(42)).tolist()
    val_eps = sorted(perm[int(args.split_ratio * E):])
    # a window needs num_hist init frames + H future frames; enumerate EVERY valid window over EVERY
    # val episode (per-episode length if seq_lengths.pth is present, else T) = the ENTIRE val set.
    seq_len = torch.load(p / "seq_lengths.pth").numpy() if (p / "seq_lengths.pth").exists() else None
    windows = []
    for e in val_eps:
        te = int(seq_len[e]) if seq_len is not None else T
        windows += [(e, f) for f in range(0, te - (num_hist + H) + 1)]
    print(f"[val split] {len(val_eps)}/{E} held-out episodes (seed 42, train_fraction={args.split_ratio}) "
          f"-> {len(windows)} windows (ENTIRE val set)")

    if args.planner_domain:
        # keep only windows whose transition stroke (frame f+num_hist-1) is in the RRT's domain:
        # aim = |start-cube|, push = |disp|, both in range. RAW meters (actions/states unnormalized here).
        def _in_domain(e, f):
            tf = f + num_hist - 1
            a = actions[e, tf]; cube = states[e, tf, CUBE_OFF:CUBE_OFF + 2]
            aim = float(np.linalg.norm(a[:2] - cube)); push = float(np.linalg.norm(a[2:4]))
            return args.aim_lo <= aim <= args.aim_hi and args.push_lo <= push <= args.push_hi
        n0 = len(windows)
        windows = [(e, f) for (e, f) in windows if _in_domain(e, f)]
        print(f"[planner_domain] restricted to RRT-domain strokes "
              f"(aim[{args.aim_lo},{args.aim_hi}] push[{args.push_lo},{args.push_hi}]): {len(windows)}/{n0} windows")

    err_enc = []                                 # encoded-floor: probe on the REAL frame (perception ceiling)
    err_h = {k: [] for k in range(1, H + 1)}     # endpoint error per horizon k (m)
    dir_h = {k: [] for k in range(1, H + 1)}     # heading error per horizon k (deg; NaN when cube ~still)
    move_h = {k: [] for k in range(1, H + 1)}    # true cumulative cube move at horizon k (m; contact/miss split)
    move_true, move_pred1 = [], []               # 1-step actual vs predicted move (motion ratio)
    axerr_enc = []                               # per-axis |err| for the encoded floor -> (N,2)
    axerr_h = {k: [] for k in range(1, H + 1)}   # per-axis |err| per horizon -> (N,2); for cushion δ sizing

    n_batches = (len(windows) + args.batch - 1) // args.batch
    for bi, i in enumerate(range(0, len(windows), args.batch)):
        batch = windows[i:i + args.batch]
        vis0 = torch.stack([load_vis(e, f, num_hist) for e, f in batch]).to(dev)        # (b,nh,3,H,W)
        pro0 = torch.stack([torch.tensor((proprio[e, f:f + num_hist] - p_mean) / p_std)
                            for e, f in batch]).float().to(dev)                          # (b,nh,18)
        # actions: init nh + H future (normalized). rollout takes act[:, :nh] as init, rest as future.
        acts = torch.stack([torch.tensor((actions[e, f:f + num_hist + H] - a_mean) / a_std)
                            for e, f in batch]).float().to(dev)                          # (b,nh+H,4)
        obs0 = {"visual": vis0, "proprio": pro0}

        # ONE open-loop rollout over all H future actions. Horizon k's prediction is frame index
        # num_hist+k-1 (the k-th action-driven frame); rollout returns num_hist+H+1 frames and the
        # trailing free-predict frame ([:, -1]) sits one step PAST horizon H -- not used. (The old
        # H-step read [:, -1], i.e. one frame too far; this indexes each horizon exactly.)
        with torch.no_grad():
            zf, _ = wm.rollout(obs_0=obs0, act=acts)                          # (b, num_hist+H+1, P, D)
        pred_k = {k: probe_xy(mlp, pinfo, zf["visual"][:, num_hist + k - 1], dev) for k in range(1, H + 1)}

        for j, (e, f) in enumerate(batch):
            cube_hist = states[e, f + num_hist - 1, CUBE_OFF:CUBE_OFF + 2]    # last seen cube
            for k in range(1, H + 1):
                true_k = states[e, f + num_hist - 1 + k, CUBE_OFF:CUBE_OFF + 2]   # true cube after k strokes
                pk = pred_k[k][j]
                err_h[k].append(np.linalg.norm(pk - true_k))
                axerr_h[k].append(np.abs(pk - true_k))       # per-axis (|dx|,|dy|) for δ sizing
                # heading error: angle between predicted & true CUMULATIVE displacement from cube_hist.
                # NaN unless the cube truly moved (>2cm, the contact threshold) -- a heading on sub-cm
                # jitter is meaningless (random angle ~90-145deg), so misses/near-still frames get NaN.
                td, pd = true_k - cube_hist, pk - cube_hist
                move_h[k].append(float(np.linalg.norm(td)))      # true cumulative move (contact/miss split)
                if np.linalg.norm(td) > 0.02 and np.linalg.norm(pd) > 1e-6:
                    cos = float(td @ pd) / (np.linalg.norm(td) * np.linalg.norm(pd))
                    dir_h[k].append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
                else:
                    dir_h[k].append(np.nan)
            # 1-step motion stats (motion ratio)
            cube_next = states[e, f + num_hist, CUBE_OFF:CUBE_OFF + 2]        # true after 1 stroke
            move_true.append(np.linalg.norm(cube_next - cube_hist))
            move_pred1.append(np.linalg.norm(pred_k[1][j] - cube_hist))
            # encoded floor: probe the REAL frame f+nh (perception ceiling, no prediction)
            ev = load_vis(e, f + num_hist, 1).to(dev)
            with torch.no_grad():
                ze = wm.encode_obs({"visual": ev[None],
                                    "proprio": torch.zeros(1, 1, 18, device=dev)})["visual"][:, 0]
            enc = probe_xy(mlp, pinfo, ze, dev)[0]
            err_enc.append(np.linalg.norm(enc - cube_next))
            axerr_enc.append(np.abs(enc - cube_next))        # per-axis for δ sizing
        done = min(i + args.batch, len(windows))
        print(f"  [progress] batch {bi + 1}/{n_batches}  ({done}/{len(windows)} windows)  "
              f"running 1-step err {np.mean(err_h[1]):.4f} m", flush=True)

    def md(a):                                                   # (mean, median) over finite values
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return (float(a.mean()), float(np.median(a))) if a.size else (float("nan"), float("nan"))

    # error vs true cube: the encoded floor (probe on the real frame = perception ceiling) + each
    # open-loop horizon, split into CONTACT (cube truly moved >2cm over that horizon -- the strokes
    # the planner relies on) vs MISS (cube ~still -> trivially predictable). dir = heading error,
    # meaningful on contacts only (a miss has no true direction).
    print(f"\n[prediction error vs true cube]  (encoded = perception ceiling ~0.015; dir = heading, contacts only)")
    print(f"  {'source':<16}{'n':>6}{'err mean':>10}{'err med':>9}{'dir(deg)':>10}")
    em, emd = md(err_enc)
    print(f"  {'encoded':<16}{len(err_enc):>6}{em:>10.4f}{emd:>9.4f}{'--':>10}")
    for k in range(1, H + 1):
        e = np.array(err_h[k]); mv = np.array(move_h[k]); dv = np.array(dir_h[k])
        for tag, mask in [("contact", mv > 0.02), ("miss", mv <= 0.02)]:
            if not mask.any():
                continue
            em, emd = md(e[mask])
            dok = dv[mask][np.isfinite(dv[mask])]
            dstr = f"{dok.mean():>10.1f}" if dok.size else f"{'--':>10}"
            print(f"  {f'{k}-step {tag}':<16}{int(mask.sum()):>6}{em:>10.4f}{emd:>9.4f}{dstr}")
    # === cushion δ sizing: per-axis max(|dx|,|dy|) percentiles, on the PREDICTED reads the constraint
    # transit check actually uses. A footprint breach is per-axis, so δ must cover max(x,y); to prevent
    # a breach in q% of frames set δ = pq of that row. CONTACT rows (cube truly moved) are the strokes
    # the planner relies on -> size δ off '1-step contact' (closed-loop commits the first stroke).
    def _pcts(a):                                            # (p50,p75,p90,p95,p99,max) or None
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return None if a.size == 0 else [np.percentile(a, q) for q in (50, 75, 90, 95, 99)] + [a.max()]

    def _prow(label, n, mx):
        row = _pcts(mx)
        if row is not None:
            print(f"  {label:<16}{n:>6}" + "".join(f"{v:>9.4f}" for v in row))

    print(f"\n[cushion δ sizing]  per-axis max(x,y) error (m) -- δ=pq prevents a breach in q% of frames. "
          f"CELL/2={gm.CELL / 2:.4f} m: δ≥this seals the cell. Size δ off '1-step contact'.")
    print(f"  {'source':<16}{'n':>6}{'p50':>9}{'p75':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>9}")
    _ae = np.asarray(axerr_enc)
    _prow("encoded", len(_ae), _ae.max(1) if _ae.size else _ae)
    for k in range(1, H + 1):
        ax = np.asarray(axerr_h[k]); mv = np.array(move_h[k])
        for tag, mask in [("contact", mv > 0.02), ("miss", mv <= 0.02)]:
            if mask.any():
                _prow(f"{k}-step {tag}", int(mask.sum()), ax[mask].max(1))

    print(f"\n[motion]  true cube move/step:      mean {md(move_true)[0]:.4f} m")
    print(f"[motion]  predicted cube move/step: mean {md(move_pred1)[0]:.4f} m   "
          f"(ratio>1 => WM OVER-predicts push distance; <1 => under)")
    ratio = np.mean(move_pred1) / max(np.mean(move_true), 1e-6)
    print(f"[motion]  predicted/true move ratio: {ratio:.2f}  (1.0 = faithful; ~0 = WM keeps cube ~put)")



if __name__ == "__main__":
    main()
