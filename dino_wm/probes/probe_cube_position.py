"""Probe whether the CUBE'S (x, y) POSITION is decodable from the WM's DINOv2 latent.

Companion to probe_sign_color.py. Motivation (the planning-objective fix): GD/MPC
currently optimize latent-MSE to a goal frame, which is dominated by the big arm +
scene while the cube is only a few of the 256 patches — so "cube closer to goal cell"
doesn't cleanly reduce latent distance. If an MLP can read the cube's continuous (x, y)
off the frozen DINOv2 latent, we can define a TASK-RELEVANT planning objective
(L2 from probed cube position to the goal cell center) and demote latent MSE to an
auxiliary "stay on the manifold" term. This file de-risks that: it measures whether
the position is there to be read.

Same frozen encoder + preprocessing the WM uses (Normalize(0.5,0.5) -> x*2-1), the same
spatial-pool-then-MLP recipe as the color probe, and the same BY-EPISODE split (no frame
leakage) + shuffle-label control. The head is a 2-output REGRESSION (x, y in env-local
meters); targets are read per-frame from states.pth[..., 18:20] (cube block offset 18).

Reported on held-out episodes:
  - L2 position error (meters) vs a predict-the-mean baseline,
  - per-axis MAE,
  - CELL accuracy: snap the predicted (x, y) to its 3x3 grid cell and compare to the
    true cell (the metric that actually matters for planning/success).
Read it as: L2 error well under CELL/2 (~0.086 m) and cell-acc >> chance -> position is
cleanly present; build the probe objective. Shuffle control should collapse L2 to the
baseline; if it doesn't, there's a leak/bug.

No sim needed — run with the container python (torch + cached dinov2 hub):
    python probe_cube_position.py --data_dir data/isaaclab_single_stroke --save_path probe_cube.pth
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent   # repo root (this file lives in probe/)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.dino import DinoV2Encoder

# Load grid_metadata by file path so we don't trigger env/isaaclab/__init__ (which
# pulls in IsaacLab) — this probe must run in plain container python, no sim.
_gm_spec = importlib.util.spec_from_file_location(
    "grid_metadata", _REPO_ROOT / "env" / "isaaclab" / "grid_metadata.py"
)
gm = importlib.util.module_from_spec(_gm_spec)
_gm_spec.loader.exec_module(gm)

CUBE_OFF = gm.STATE_FIRST_CUBE_OFFSET_SINGLE  # 18: cube xyz starts here in the 31-D state


def _spatial_pool_grid(tokens, grid):
    """(B, P, D) patch tokens -> (B, grid*grid*D). P must be a perfect square
    (16x16 for vits14@224). Keeping the grid (vs global mean-pool) preserves WHERE
    the cube is — essential for position regression; finer grid = finer localization."""
    b, p, d = tokens.shape
    s = int(round(p ** 0.5))
    assert s * s == p, f"patch count {p} not a square"
    g = tokens.reshape(b, s, s, d).permute(0, 3, 1, 2)        # (B, D, s, s)
    g = F.adaptive_avg_pool2d(g, grid)                         # (B, D, grid, grid)
    return g.reshape(b, -1)                                    # (B, grid*grid*D)


@torch.no_grad()
def encode_dataset(data_dir, encoder, device, frame_stride, batch_size, pool_grid, enc_res):
    """Encode every (strided, non-padded) frame -> pooled latent. Returns
    X (N, D), Y (N, 2) cube xy in meters, ep (N,) episode index, V (N,2) xy displacement."""
    p = Path(data_dir)
    resize = transforms.Resize(enc_res)  # match the WM's encoder_transform exactly
    states = torch.load(p / "states.pth").float().numpy()      # (E, T, 31)
    seq_lengths = torch.load(p / "seq_lengths.pth").numpy().astype(np.int64)  # (E,)
    files = sorted((p / "obses").glob("episode_*.pth"))
    assert len(files) == len(states), f"{len(files)} obs files vs {len(states)} states"

    feats, ys, eps, vels = [], [], [], []
    for ei, f in enumerate(files):
        T_real = int(seq_lengths[ei])                          # drop zero-padded tail frames
        vid = torch.load(f)[:T_real]                           # (T_real, H, W, 3) uint8
        idx = np.arange(0, T_real, frame_stride)
        sel = vid[idx].float() / 255.0
        x = sel.permute(0, 3, 1, 2)                            # (n,3,H,W)
        # Match the WM's encoder_transform EXACTLY: it resizes to encoder_image_size =
        # (img_size//16)*patch_size = 196 for vits14@224 -> a 14x14=196 patch grid. The
        # probe MUST train on the SAME token grid it reads off the WM latent at plan time,
        # NOT 224px/16x16=256 (which the WM never produces), or it's applied OOD.
        x = resize(x)                                          # (n,3,enc_res,enc_res)
        x = x * 2.0 - 1.0                                      # WM preprocessing (Normalize(0.5,0.5))
        for i in range(0, x.shape[0], batch_size):
            toks = encoder(x[i:i + batch_size].to(device))     # (b,P,D)
            feats.append(_spatial_pool_grid(toks, pool_grid).cpu())
        xy = states[ei, idx, CUBE_OFF:CUBE_OFF + 2].astype(np.float32)  # (n, 2) env-local meters
        ys.append(xy)
        v = np.zeros_like(xy)                                  # per-frame cube xy displacement
        v[1:] = xy[1:] - xy[:-1]                               # toward base = negative x (base at x=-0.45)
        vels.append(v)
        eps.append(np.full(len(idx), ei, dtype=np.int64))
        if (ei + 1) % 25 == 0:
            print(f"[encode] {ei + 1}/{len(files)} episodes")
    X = torch.cat(feats).numpy().astype(np.float32)
    Y = np.concatenate(ys)
    ep = np.concatenate(eps)
    V = np.concatenate(vels)
    print(f"[encode] {X.shape[0]} frames, latent dim {X.shape[1]}, target dim {Y.shape[1]}")
    return X, Y, ep, V


@torch.no_grad()
def predict_dataset(data_dir, wm, num_hist, device, pred_horizons, batch_size, pool_grid):
    """Build PREDICTED-latent features: for each (history, action) window roll the frozen
    WM `h` steps open-loop USING THE DATASET'S OWN ACTIONS, pool the PREDICTED tokens, and
    pair with the TRUE cube at that step (states.pth). The dataset actions make the GT cube
    well-defined at any h. This is the distribution the probe is applied to in the planning
    objective (the candidate's WM-rolled next state). Same return shape as encode_dataset.

    Feeds native-res frames straight to wm.rollout (its encoder resizes internally), exactly
    as plan.py does, so the pred head trains on the SAME tokens it reads at plan time."""
    p = Path(data_dir)
    states = torch.load(p / "states.pth").float().numpy()                       # (E,T,31)
    actions = torch.load(p / "actions.pth").float().numpy()                     # (E,T,A)
    proprio = torch.load(p / "proprio.pth").float().numpy()                     # (E,T,Pd)
    seq = torch.load(p / "seq_lengths.pth").numpy().astype(np.int64)            # (E,)
    files = sorted((p / "obses").glob("episode_*.pth"))
    A, Pd = actions.shape[-1], proprio.shape[-1]
    a_mean = actions.reshape(-1, A).mean(0); a_std = actions.reshape(-1, A).std(0) + 1e-6
    p_mean = proprio.reshape(-1, Pd).mean(0); p_std = proprio.reshape(-1, Pd).std(0) + 1e-6
    Hmax = max(pred_horizons)

    feats, ys, eps, vels = [], [], [], []
    for ei, f in enumerate(files):
        T = int(seq[ei])
        starts = list(range(0, T - (num_hist - 1 + Hmax)))   # f0 s.t. f0+num_hist-1+Hmax <= T-1
        if not starts:
            continue
        vid = (torch.load(f)[:T].float() / 255.0).permute(0, 3, 1, 2) * 2.0 - 1.0   # (T,3,H,W) Normalize(.5,.5)
        for h in pred_horizons:
            for i in range(0, len(starts), batch_size):
                bs = starts[i:i + batch_size]
                vis = torch.stack([vid[f0:f0 + num_hist] for f0 in bs]).to(device)              # (b,nh,3,H,W)
                pro = torch.stack([torch.tensor((proprio[ei, f0:f0 + num_hist] - p_mean) / p_std)
                                   for f0 in bs]).float().to(device)                            # (b,nh,Pd)
                act = torch.stack([torch.tensor((actions[ei, f0:f0 + num_hist + h - 1] - a_mean) / a_std)
                                   for f0 in bs]).float().to(device)                            # (b,nh+h-1,A)
                z = wm.rollout(obs_0={"visual": vis, "proprio": pro}, act=act)[0]["visual"][:, -1]  # (b,P,D) PREDICTED
                feats.append(_spatial_pool_grid(z, pool_grid).cpu())
                cube = np.stack([states[ei, f0 + num_hist - 1 + h, CUBE_OFF:CUBE_OFF + 2] for f0 in bs]).astype(np.float32)
                prev = np.stack([states[ei, f0 + num_hist - 1, CUBE_OFF:CUBE_OFF + 2] for f0 in bs]).astype(np.float32)
                ys.append(cube); vels.append(cube - prev)
                eps.append(np.full(len(bs), ei, dtype=np.int64))
        if (ei + 1) % 25 == 0:
            print(f"[predict] {ei + 1}/{len(files)} episodes")
    X = torch.cat(feats).numpy().astype(np.float32)
    Y = np.concatenate(ys); ep = np.concatenate(eps); V = np.concatenate(vels)
    print(f"[predict] {X.shape[0]} predicted-latent samples (horizons {pred_horizons}), latent dim {X.shape[1]}")
    return X, Y, ep, V


class MLP(nn.Module):
    def __init__(self, d_in, d_out=2, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _cells(xy):
    """(N,2) env-local xy -> (N,) cell id (-1 off-grid), via grid_metadata.which_cell."""
    return gm.which_cell(np.asarray(xy, dtype=np.float32))


def _err_percentiles(pred, true):
    """Position-error distribution for choosing a CONSTRAINT CUSHION delta (robust
    constraint tightening: the planner inflates each illegal cell by delta to stay clear
    despite perception error). The illegal-cell test is an AXIS-ALIGNED footprint box
    (|x-cx| < CELL/2 + cube_half), so a breach happens when the TRUE center crosses an
    edge the probe thought was clear -> delta must cover the PER-AXIS error, which is
    smaller than the L2. Read delta off a TAIL percentile (a chance constraint 'breach
    with prob <= eps' uses the (1-eps) quantile), not the mean. Calibrate on --source
    predicted (the latent the planner actually reads), NOT encoded."""
    d = pred - true                                    # (N,2) signed error, meters
    ax = np.abs(d)                                      # (N,2) per-axis magnitude
    l2 = np.linalg.norm(d, axis=1)                      # (N,) euclidean
    qs = [50, 75, 90, 95, 99]

    def row(v):
        return "  ".join(f"p{q}={np.percentile(v, q):.4f}" for q in qs) + f"  max={v.max():.4f}"

    print("\n[error percentiles] (meters) — pick the constraint cushion delta off the PER-AXIS tail")
    print(f"  bias (mean signed)  x={d[:, 0].mean():+.4f}  y={d[:, 1].mean():+.4f}   "
          "(a systematic offset — subtract it instead of cushioning it)")
    print(f"  |err| per-axis x    {row(ax[:, 0])}")
    print(f"  |err| per-axis y    {row(ax[:, 1])}")
    print(f"  |err| max(x,y)      {row(ax.max(1))}   <- symmetric delta covering BOTH axes (use this)")
    print(f"  L2                  {row(l2)}")
    print(f"  [guide] CELL/2={gm.CELL/2:.4f} m: a delta at/above CELL/2 seals cells shut. For "
          "border-of-goal (geom2) too large a delta can make the goal unreachable — sweep it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--source", choices=["encoded", "predicted", "both"], default="encoded",
                    help="feature source: 'encoded' = φ(real frame) -> GOAL/anchor head "
                    "(probe_cube_1500.pth); 'predicted' = WM-rolled latent -> objective head "
                    "(probe_cube_pred.pth); 'both' = train ONE head on the UNION of encoded ∪ "
                    "predicted latents (at --pred_horizons, default 1) -- matches the mixed latent "
                    "stream the constraint reads at plan time (encoded history + predicted future).")
    ap.add_argument("--model_dir", default="outputs/2026-06-25/16-46-57",
                    help="WM dir (only used for --source predicted)")
    ap.add_argument("--epoch", default="20", help="WM epoch (only used for --source predicted)")
    ap.add_argument("--pred_horizons", default="1",
                    help="comma list of open-loop horizons to build predicted features at. Use the "
                    "PLANNER'S lookahead depth: '1' for horizon-1 MPC; '1,2,3' if you go deeper. "
                    "Do NOT include 4-5 -- the latent has drifted past CELL/2 there, so the cube "
                    "isn't faithfully encoded and those become noisy training targets.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame_stride", type=int, default=1, help="use every Nth (non-padded) frame")
    ap.add_argument("--enc_batch", type=int, default=64)
    ap.add_argument("--pool_grid", type=int, default=8,
                    help="spatial pool size (8x8 keeps finer location than the color probe's 4x4 "
                    "— the cube moves across the whole grid, so localization needs resolution)")
    ap.add_argument("--enc_res", type=int, default=196,
                    help="encoder input resolution — MUST match the WM: (img_size//16)*patch_size "
                    "= 196 for vits14@224, giving a 14x14=196 token grid. Do NOT use 224 (that's "
                    "16x16=256 tokens, which the WM never produces -> probe is OOD at plan time).")
    ap.add_argument("--val_frac", type=float, default=0.2, help="fraction of EPISODES held out")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vel_thresh", type=float, default=0.003,
                    help="m of cube xy displacement / frame above which a frame counts as 'moving' "
                    "(for the motion-direction breakdown that tests the arm-occlusion hypothesis)")
    ap.add_argument("--boundary_margin", type=float, default=0.03,
                    help="m from nearest cell edge; frames closer than this are 'near-edge' (the "
                    "boundary breakdown that tests whether cell-acc loss is just edge discretization)")
    ap.add_argument("--save_path", default=None,
                    help="if set, save probe weights + normalization stats here for use at "
                    "planning time (plan.py reads cube position off the WM's predicted latent)")
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="CONTROL: permute the (x,y) targets across frames (break the image<->position "
                    "link). Real signal -> L2 error collapses to the predict-mean baseline; if it "
                    "stays low there's a leak/bug.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    parts = []   # each: (X, Y, ep, V); >1 only for --source both (encoded ∪ predicted)
    if args.source in ("encoded", "both"):
        encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
        for prm in encoder.parameters():
            prm.requires_grad_(False)
        parts.append(encode_dataset(args.data_dir, encoder, device, args.frame_stride,
                                    args.enc_batch, args.pool_grid, args.enc_res))
    if args.source in ("predicted", "both"):
        from scripts.wm_cube_pred_check import load_wm  # lazy: avoids a circular import at module load
        pred_h = [int(s) for s in str(args.pred_horizons).split(",") if s.strip()]
        wm, tcfg = load_wm(args.model_dir, args.epoch, device)
        print(f"[predict] WM {args.model_dir}@{args.epoch} num_hist={int(tcfg.num_hist)} horizons={pred_h}")
        parts.append(predict_dataset(args.data_dir, wm, int(tcfg.num_hist), device,
                                     pred_h, args.enc_batch, args.pool_grid))
    # 'both' concatenates the two sources into one training set. Each source tags frames with the
    # SAME episode index (both iterate sorted(files) identically), so the by-episode split below
    # draws val_eps from the union and holds a val episode out of BOTH halves -> no cross-source
    # leakage (a test episode's encoded frames can't sneak into train via its predicted frames).
    X = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    ep = np.concatenate([p[2] for p in parts])
    V = np.concatenate([p[3] for p in parts])
    if args.source == "both":
        print(f"[both] union = {parts[0][0].shape[0]} encoded + {parts[1][0].shape[0]} predicted "
              f"(h={args.pred_horizons}) = {X.shape[0]} samples, dim {X.shape[1]}")
        assert parts[0][0].shape[1] == parts[1][0].shape[1], "encoded/predicted feature dims differ"

    if args.shuffle_labels:
        perm = np.random.RandomState(args.seed + 1).permutation(len(Y))
        Y = Y[perm]
        print("[CONTROL] (x,y) targets permuted across frames — expect L2 to collapse to the "
              "predict-mean baseline; anything low means a leak/bug.")

    # --- split BY EPISODE (no frame leakage) ---
    ep_ids = np.unique(ep)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ep_ids)
    n_val = max(1, int(round(len(ep_ids) * args.val_frac)))
    val_eps = set(ep_ids[:n_val].tolist())
    test_mask = np.array([e in val_eps for e in ep])
    train_mask = ~test_mask
    print(f"[split] {len(ep_ids) - n_val} train / {n_val} test episodes "
          f"({train_mask.sum()} / {test_mask.sum()} frames)")

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    tr = torch.from_numpy(np.where(train_mask)[0])
    te = torch.from_numpy(np.where(test_mask)[0])

    # Standardize inputs AND targets on TRAIN frames only (no test stats leak).
    mu = Xt[tr].mean(0, keepdim=True)
    sd = Xt[tr].std(0, keepdim=True) + 1e-6
    Xn = ((Xt - mu) / sd).to(device)
    ymu = Yt[tr].mean(0, keepdim=True)
    ysd = Yt[tr].std(0, keepdim=True) + 1e-6
    Yn = ((Yt - ymu) / ysd).to(device)
    ymu_np, ysd_np = ymu.numpy(), ysd.numpy()

    probe = MLP(X.shape[1], d_out=2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    Y_te = Y[test_mask]                                        # meters, true xy
    true_cells_te = _cells(Y_te)
    ingrid = true_cells_te != gm.OFF_GRID                      # cell-acc only over in-grid truth
    # predict-the-mean baseline (the bar a useful probe must clear)
    base_pred = np.repeat(ymu_np, len(Y_te), axis=0)
    base_l2 = float(np.linalg.norm(base_pred - Y_te, axis=1).mean())
    base_cell_acc = float((_cells(base_pred)[ingrid] == true_cells_te[ingrid]).mean()) if ingrid.any() else 0.0

    def evaluate():
        probe.eval()
        with torch.no_grad():
            pn = probe(Xn[te]).cpu().numpy()
        pred = pn * ysd_np + ymu_np                            # unnormalize to meters
        l2 = float(np.linalg.norm(pred - Y_te, axis=1).mean())
        mae = np.abs(pred - Y_te).mean(0)                      # per-axis (x, y) MAE
        pred_cells = _cells(pred)
        cell_acc = float((pred_cells[ingrid] == true_cells_te[ingrid]).mean()) if ingrid.any() else 0.0
        return l2, mae, cell_acc

    print(f"[geom] CELL={gm.CELL:.3f} m (need L2 << CELL/2={gm.CELL/2:.3f} m for reliable cell hits)")
    print(f"[baseline] predict-mean: L2={base_l2:.4f} m  cell-acc={base_cell_acc:.3f}")

    best_l2 = float("inf")
    for epoch in range(1, args.epochs + 1):
        probe.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(probe(Xn[idx]), Yn[idx])
            loss.backward()
            opt.step()
        if epoch % 15 == 0 or epoch == args.epochs:
            l2, mae, cell_acc = evaluate()
            best_l2 = min(best_l2, l2)
            print(f"[probe] epoch {epoch:3d}  L2 {l2:.4f} m  MAE(x,y) ({mae[0]:.4f},{mae[1]:.4f})  "
                  f"cell-acc {cell_acc:.3f}")

    l2, mae, cell_acc = evaluate()
    print(f"\n[RESULT] L2 {l2:.4f} m (baseline {base_l2:.4f}) | MAE(x,y) ({mae[0]:.4f},{mae[1]:.4f}) | "
          f"cell-acc {cell_acc:.3f} (baseline {base_cell_acc:.3f})")
    print(f"         best test L2 during training = {min(best_l2, l2):.4f} m (optimistic — peeks at test)")
    if l2 < 0.5 * base_l2 and cell_acc > base_cell_acc + 0.2:
        print("         -> cube position is cleanly decodable; the probe-objective plan is viable.")
    else:
        print("         -> WEAK signal: position is hard to read off this latent — revisit before "
              "betting the planning objective on it (finer pool_grid? more epochs? leakage check?).")

    # === diagnostic breakdowns (real run only — meaningless when targets are shuffled) ===
    if not args.shuffle_labels:
        probe.eval()
        with torch.no_grad():
            pred = probe(Xn[te]).cpu().numpy() * ysd_np + ymu_np   # (Nte, 2) meters

        _err_percentiles(pred, Y_te)                               # cushion-sizing table

        def _bm(mask):
            n = int(mask.sum())
            if n == 0:
                return n, float("nan"), float("nan")
            l2 = float(np.linalg.norm(pred[mask] - Y_te[mask], axis=1).mean())
            mi = mask & ingrid
            ca = float((_cells(pred[mi]) == true_cells_te[mi]).mean()) if mi.any() else float("nan")
            return n, l2, ca

        # (1) motion-direction breakdown — does the probe fail when the cube is being
        # pushed toward the base/arm (where the arm is most likely to occlude it)?
        V_te = V[test_mask]
        speed = np.linalg.norm(V_te, axis=1)
        vx, vy = V_te[:, 0], V_te[:, 1]
        moving = speed > args.vel_thresh
        xdom = np.abs(vx) >= np.abs(vy)
        buckets = {
            "toward-base (-x push)": moving & xdom & (vx < 0),
            "away-from-base (+x)":   moving & xdom & (vx > 0),
            "lateral (±y push)":     moving & ~xdom,
            "static/near-still":     ~moving,
        }
        print(f"\n[motion breakdown]  vel_thresh={args.vel_thresh} m/frame  "
              "(tests the arm-occludes-cube-on-toward-self-push hypothesis)")
        for name, m in buckets.items():
            n, l2, ca = _bm(m)
            print(f"  {name:<23s} n={n:6d}  L2={l2:.4f} m  cell-acc={ca:.3f}")

        # (2) boundary breakdown — are the cell-acc errors just edge discretization of an
        # otherwise-accurate position? Distance from true xy to the nearest grid line.
        gx = Y_te[:, 0] - gm.GRID_CENTER_XY[0]
        gy = Y_te[:, 1] - gm.GRID_CENTER_XY[1]
        lines = -gm.GRID_HALF + gm.CELL * np.arange(4)         # 4 grid lines per axis (incl. outer)
        dx = np.min(np.abs(gx[:, None] - lines[None, :]), axis=1)
        dy = np.min(np.abs(gy[:, None] - lines[None, :]), axis=1)
        dbound = np.minimum(dx, dy)
        n_far, _, ca_far = _bm(ingrid & (dbound > args.boundary_margin))
        n_near, _, ca_near = _bm(ingrid & (dbound <= args.boundary_margin))
        print(f"\n[boundary breakdown]  margin={args.boundary_margin} m from nearest cell edge  "
              "(tests whether cell-acc loss is just edge discretization)")
        print(f"  interior  (> margin)  n={n_far:6d}  cell-acc={ca_far:.3f}")
        print(f"  near-edge (<= margin) n={n_near:6d}  cell-acc={ca_near:.3f}")

    if args.save_path:
        torch.save({
            "kind": "regression", "state_dict": probe.state_dict(),
            "x_mu": mu.numpy(), "x_sd": sd.numpy(),
            "y_mu": ymu_np, "y_sd": ysd_np,
            "pool_grid": args.pool_grid, "d_in": X.shape[1], "out_dim": 2, "hidden": 256,
            "encoder": "dinov2_vits14", "feature_key": "x_norm_patchtokens",
            "source": args.source, "pred_horizons": args.pred_horizons,  # provenance: encoded vs predicted head
        }, args.save_path)
        print(f"[save] probe + norm stats -> {args.save_path}")


if __name__ == "__main__":
    main()
