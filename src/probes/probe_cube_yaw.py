"""Probe whether the CUBE'S YAW is decodable from the WM's DINOv2 latent.

WHY. Abidance is scored with an AXIS-ALIGNED square footprint of half-width CUBE_HALF, but the cube
spawns at a random angle and rotates as it is pushed. A square of half h rotated by t has axis-aligned
half-extent h*(|cos t| + |sin t|), up to h*sqrt(2) at 45deg -- so the axis-aligned model UNDER-detects
entry, one-directionally, and every abidance number computed with it is an UPPER BOUND. Reading yaw
off the latent is the deployment-side fix named in the paper ("deployment would require an orientation
probe"): with yaw, enforcement can use the true oriented box instead of the axis-aligned one.

THE KEY TRICK: the cube is SQUARE, so its footprint is invariant under 90deg rotation. The probe does
not need yaw -- it needs yaw MOD 90deg. Predicting (sin 4t, cos 4t) and decoding atan2(s,c)/4 makes
the target range four times smaller, removes the wrap-around discontinuity entirely (a naive scalar
regressor has to represent -179deg and +179deg as far apart when they are 2deg apart), and makes a
40deg and a 130deg answer identical -- which they are, for a square.

Structure, split, controls and save format follow probes/probe_cube_position.py; only the target and
the metrics differ. Same frozen encoder + preprocessing as the WM, same BY-EPISODE split (no frame
leakage), same shuffle-label control.

Reported on held-out episodes:
  - circular error mod 90deg, in degrees, vs a predict-the-constant baseline (~22.5deg for near-
    uniform yaw, which is what the training set is: 22.83deg from the nearest axis on average);
  - FOOTPRINT extent error -- the number that actually matters. Compares three models of the cube's
    axis-aligned half-extent: the probe's, the current axis-aligned assumption (constant CUBE_HALF),
    and ground truth. If the probe's extent error is not well under the axis-aligned model's, the
    probe buys nothing and the oriented-box rescoring is not worth doing;
  - the UNDER-estimate rate, separately, because under-estimating the extent is the unsafe direction
    (it is what lets a real violation go undetected).

No sim needed -- run with the container python (torch + cached dinov2 hub):
    python probes/probe_cube_yaw.py --data_dir data/isaaclab_stroke_5k \
        --save_path probes/weights/probe_cube_yaw.pth
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import provenance
from models.dino import DinoV2Encoder

# Load grid_metadata by file path so we don't trigger env/isaaclab/__init__ (which pulls in
# IsaacLab) -- this probe must run in plain container python, no sim.
_gm_spec = importlib.util.spec_from_file_location(
    "grid_metadata", _REPO_ROOT / "env" / "isaaclab" / "grid_metadata.py")
gm = importlib.util.module_from_spec(_gm_spec)
_gm_spec.loader.exec_module(gm)

CUBE_OFF = gm.STATE_FIRST_CUBE_OFFSET_SINGLE      # 18: cube xyz starts here in the 31-D state
QUAT_OFF = CUBE_OFF + 3                            # 21:25 -- (w, x, y, z), verified unit-norm & planar
CUBE_HALF = 0.045                                  # matches probes/probe_cube_cells.CUBE_HALF


def _log(msg):
    print(msg, flush=True)                         # flush: this is a long job, keep the terminal live


def yaw_from_quat(q):
    """(N,4) wxyz -> (N,) yaw about world +z. Identical formula to GridWrapperSingle.get_cube_yaw.

    NOTE: this describes the FOOTPRINT's in-plane angle only while the cube sits flat. Once it tips
    onto a side face -- 16.1% of frames in isaaclab_stroke_5k -- the body is rotated about a
    horizontal axis and this angle no longer corresponds to the square seen from above. Kept for the
    env-parity check and for --label yaw; footprint_from_quat is the correct target."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_to_R(q):
    """(N,4) wxyz -> (N,3,3) rotation. Columns are the body axes expressed in world coordinates."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],     -1),
        np.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],     -1),
        np.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], -1)], -2)


def footprint_from_quat(q, half=None, tol_deg=5.0):
    """(N,4) wxyz -> (angle, is_square, half_extent): the cube's GROUND FOOTPRINT, derived from the
    FULL orientation rather than assuming the cube is flat.

    A cube resting on ANY face projects to a square of half-side `half`; which face is down only
    changes the in-plane angle. So the target is that angle mod 90deg, recovered by finding the body
    axis closest to world +z (the face normal pointing up) -- the other two span the ground plane and
    give the square's edge directions. This is exact for any face-up pose, which is 97.75% of frames.

    `is_square` is False when NO body axis is within tol_deg of vertical: the cube is caught mid-tip,
    the projected hull is a hexagon, and no single angle describes it (3.1% of frames). Those are not
    a square-footprint problem at all and are excluded from training rather than fitted with a target
    that cannot be right.

    `half_extent` is the EXACT axis-aligned half-extent of the projected hull (max over x and y),
    computed from all 8 vertices -- the ground truth the yaw model h(|cos|+|sin|) approximates."""
    q = np.asarray(q, np.float64)
    R = _quat_to_R(q)                                     # (N,3,3)
    upness = np.abs(R[:, 2, :])                           # (N,3): |world-z component| of each body axis
    k = upness.argmax(1)                                  # the body axis pointing (most) up
    is_square = upness.max(1) >= np.cos(np.radians(tol_deg))
    n = len(q)
    # one of the two in-plane body axes (either works: they differ by 90deg, which the mod folds away)
    j = (k + 1) % 3
    ex, ey = R[np.arange(n), 0, j], R[np.arange(n), 1, j]
    ang = wrap90(np.arctan2(ey, ex))
    if half is None:
        half = CUBE_HALF
    V = np.array([[a, b, c] for a in (-half, half) for b in (-half, half)
                  for c in (-half, half)], dtype=np.float64)          # (8,3)
    P = np.einsum("nij,vj->nvi", R, V)[..., :2]                       # (N,8,2) projected vertices
    return ang, is_square, np.abs(P).max(axis=1).max(axis=1)


def wrap90(a):
    """Fold an angle (rad) into (-45deg, +45deg]: the equivalence class a square cube lives in.
    Used for BOTH the target encoding and the error metric, so a 40deg/130deg confusion -- which is
    no confusion at all for a square -- is never counted as error."""
    q = np.pi / 2.0
    return (a + q / 2.0) % q - q / 2.0


def aabb_half_extent(yaw):
    """Axis-aligned half-extent of the square footprint at this yaw: h*(|cos|+|sin|), in meters.
    Ranges h..h*sqrt(2) (0.0450..0.0636). The scoring model currently assumes the constant h, which
    is the minimum of this function -- hence a one-directional under-detection of cell entry."""
    return CUBE_HALF * (np.abs(np.cos(yaw)) + np.abs(np.sin(yaw)))


def _spatial_pool_grid(tokens, grid):
    """(B, P, D) patch tokens -> (B, grid*grid*D). Keeping the grid (vs global mean-pool) preserves
    WHERE the cube is; orientation is a local property, so the grid matters here too."""
    b, p, d = tokens.shape
    s = int(round(p ** 0.5))
    assert s * s == p, f"patch count {p} not a square"
    g = tokens.reshape(b, s, s, d).permute(0, 3, 1, 2)
    g = F.adaptive_avg_pool2d(g, grid)
    return g.reshape(b, -1)


@torch.no_grad()
def encode_dataset(data_dir, encoder, device, frame_stride, batch_size, pool_grid, enc_res, log_every):
    """Encode every (strided, non-padded) frame -> pooled latent.
    Returns X (N,D), ANG (N,) footprint angle mod 90deg, SQ (N,) bool 'the hull IS a square',
    EXT (N,) EXACT projected half-extent (m), ep (N,) episode index."""
    p = Path(data_dir)
    resize = transforms.Resize(enc_res)                        # match the WM's encoder_transform exactly
    states = torch.load(p / "states.pth").float().numpy()
    seq_lengths = torch.load(p / "seq_lengths.pth").numpy().astype(np.int64)
    files = sorted((p / "obses").glob("episode_*.pth"))
    assert len(files) == len(states), f"{len(files)} obs files vs {len(states)} states"
    _log(f"[encode] {len(files)} episodes, stride {frame_stride}, enc_res {enc_res}, pool {pool_grid}")

    feats, angs, sqs, exts, eps = [], [], [], [], []
    t0 = time.time()
    for ei, f in enumerate(files):
        T_real = int(seq_lengths[ei])                          # drop zero-padded tail frames
        vid = torch.load(f)[:T_real]
        idx = np.arange(0, T_real, frame_stride)
        x = (vid[idx].float() / 255.0).permute(0, 3, 1, 2)
        # Match the WM's encoder_transform EXACTLY (196px -> 14x14 tokens), NOT 224/16x16 which the
        # WM never produces -- otherwise the probe is applied out of distribution at plan time.
        x = resize(x) * 2.0 - 1.0
        for i in range(0, x.shape[0], batch_size):
            toks = encoder(x[i:i + batch_size].to(device))
            feats.append(_spatial_pool_grid(toks, pool_grid).cpu())
        _a, _s, _e = footprint_from_quat(states[ei, idx, QUAT_OFF:QUAT_OFF + 4].astype(np.float64))
        angs.append(_a); sqs.append(_s); exts.append(_e)
        eps.append(np.full(len(idx), ei, dtype=np.int64))
        if (ei + 1) % log_every == 0 or ei + 1 == len(files):
            el = time.time() - t0
            rate = (ei + 1) / el
            _log(f"[encode] {ei + 1}/{len(files)} episodes  {el/60:.1f} min elapsed  "
                 f"{rate:.1f} ep/s  ETA {(len(files) - ei - 1) / max(rate, 1e-9) / 60:.1f} min")
    X = torch.cat(feats).numpy().astype(np.float32)
    ANG, SQ, EXT, ep = (np.concatenate(angs), np.concatenate(sqs),
                        np.concatenate(exts), np.concatenate(eps))
    _log(f"[encode] done: {X.shape[0]} frames, latent dim {X.shape[1]}, "
         f"square footprint on {SQ.mean():.2%}")
    return X, ANG, SQ, EXT, ep



@torch.no_grad()
def predict_dataset_yaw(data_dir, wm, num_hist, device, pred_horizons, batch_size, pool_grid,
                        keep_eps=None, log_every=25):
    """PREDICTED-latent features for EVAL ONLY: roll the frozen WM `h` steps open-loop using the
    dataset's own actions, pool the PREDICTED tokens, and pair with the TRUE yaw at that step.

    This mirrors probe_cube_position.predict_dataset, but the probe is NOT trained on these -- the
    registry's established pattern is train-on-encoded, apply-to-predicted (cube_position is enabled,
    cube_position_pred is not). This measures the gap that pattern silently accepts.

    `keep_eps` restricts to the held-out episodes, so the eval encodes ~20% of the set rather than
    all of it -- the training episodes would be meaningless here anyway."""
    p = Path(data_dir)
    states = torch.load(p / "states.pth").float().numpy()
    actions = torch.load(p / "actions.pth").float().numpy()
    proprio = torch.load(p / "proprio.pth").float().numpy()
    seq = torch.load(p / "seq_lengths.pth").numpy().astype(np.int64)
    files = sorted((p / "obses").glob("episode_*.pth"))
    A, Pd = actions.shape[-1], proprio.shape[-1]
    a_mean = actions.reshape(-1, A).mean(0); a_std = actions.reshape(-1, A).std(0) + 1e-6
    p_mean = proprio.reshape(-1, Pd).mean(0); p_std = proprio.reshape(-1, Pd).std(0) + 1e-6
    Hmax = max(pred_horizons)

    feats, angs, sqs, exts, eps = [], [], [], [], []
    todo = [ei for ei in range(len(files)) if keep_eps is None or ei in keep_eps]
    _log(f"[predict] rolling WM on {len(todo)} episodes, horizons {pred_horizons}")
    t0 = time.time()
    for n, ei in enumerate(todo):
        T = int(seq[ei])
        starts = list(range(0, T - (num_hist - 1 + Hmax)))
        if not starts:
            continue
        vid = (torch.load(files[ei])[:T].float() / 255.0).permute(0, 3, 1, 2) * 2.0 - 1.0
        for h in pred_horizons:
            for i in range(0, len(starts), batch_size):
                bs = starts[i:i + batch_size]
                vis = torch.stack([vid[f0:f0 + num_hist] for f0 in bs]).to(device)
                pro = torch.stack([torch.tensor((proprio[ei, f0:f0 + num_hist] - p_mean) / p_std)
                                   for f0 in bs]).float().to(device)
                act = torch.stack([torch.tensor((actions[ei, f0:f0 + num_hist + h - 1] - a_mean) / a_std)
                                   for f0 in bs]).float().to(device)
                z = wm.rollout(obs_0={"visual": vis, "proprio": pro}, act=act)[0]["visual"][:, -1]
                feats.append(_spatial_pool_grid(z, pool_grid).cpu())
                q = np.stack([states[ei, f0 + num_hist - 1 + h, QUAT_OFF:QUAT_OFF + 4] for f0 in bs])
                _a, _s, _e = footprint_from_quat(q.astype(np.float64))
                angs.append(_a); sqs.append(_s); exts.append(_e)
                eps.append(np.full(len(bs), ei, dtype=np.int64))
        if (n + 1) % log_every == 0 or n + 1 == len(todo):
            el = time.time() - t0
            _log(f"[predict] {n + 1}/{len(todo)} episodes  {el/60:.1f} min  "
                 f"ETA {(len(todo) - n - 1) / max((n + 1) / el, 1e-9) / 60:.1f} min")
    X = torch.cat(feats).numpy().astype(np.float32)
    _log(f"[predict] {X.shape[0]} predicted-latent samples, dim {X.shape[1]}")
    return (X, np.concatenate(angs), np.concatenate(sqs),
            np.concatenate(exts), np.concatenate(eps))


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


def _run_predicted(args, probe, mu, sd, device, val_eps, m, p90, ext_mae, under):
    """Evaluate an already-trained yaw probe on WM-PREDICTED latents.

    Shared by the full run and by --load_probe. `m`/`p90`/`ext_mae`/`under` are the ENCODED
    metrics, printed alongside for contrast; they are NaN on the --load_probe path, which
    does not recompute them."""
    # === PREDICTED-latent evaluation (eval only -- the probe above was trained on encoded) ==========
    # Enforcement never reads an encoded latent: it reads the WM's rolled prediction for a candidate
    # action. Everything above is therefore an upper bound on the probe's usable accuracy, and this
    # block measures the drop. cube_position accepts the same gap silently (registry: the encoded head
    # is enabled, the predicted one is not), but that was established for POSITION -- orientation is a
    # finer feature and WM rollout may smear it more.
    if not args.eval_predicted:
        return None
    from scripts.wm_cube_pred_check import load_wm     # lazy: avoids a circular import at load
    pred_h = [int(t) for t in str(args.pred_horizons).split(",") if t.strip()]
    wm, tcfg = load_wm(args.model_dir, args.epoch, device)
    _log(f"\n[predict] WM {args.model_dir}@{args.epoch} num_hist={int(tcfg.num_hist)} "
         f"horizons={pred_h}")
    pdd = args.pred_data_dir or args.data_dir
    # Same dataset -> must restrict to the probe's held-out episodes. Different dataset -> none of
    # it was in probe training, so use all of it.
    keep = val_eps if pdd == args.data_dir else None
    _log(f"[predict] dataset {pdd}" + ("  (held-out episodes only)" if keep else "  (all episodes)"))
    Xp, yaw_p, sq_p, ext_p, _ep_p = predict_dataset_yaw(pdd, wm, int(tcfg.num_hist), device,
                                                        pred_h, args.enc_batch, args.pool_grid,
                                                        keep_eps=keep, log_every=args.log_every)
    # SAME normalization as training -- mu/sd are train-split statistics and must not be recomputed
    Xpn = ((torch.from_numpy(Xp) - mu) / sd).to(device)
    probe.eval()
    with torch.no_grad():
        pn = probe(Xpn).cpu().numpy()
    yaw_hat_p = np.arctan2(pn[:, 0], pn[:, 1]) / 4.0
    err_p = np.degrees(np.abs(wrap90(yaw_hat_p - yaw_p)))
    # TRUTH is the EXACT projected hull, not the yaw model applied to the true angle -- otherwise the
    # metric would grade the probe against an approximation and hide the model's own error.
    ext_true_p = ext_p
    d_p = aabb_half_extent(yaw_hat_p) - ext_true_p
    aabb_p = float(np.abs(CUBE_HALF - ext_true_p).mean())
    _log(f"\n[PREDICTED] yaw error mod 90deg: mean {err_p.mean():.2f} deg  "
         f"p90 {np.percentile(err_p, 90):.2f} deg   (encoded was {m:.2f} / {p90:.2f})")
    _log(f"            extent MAE: probe {np.abs(d_p).mean()*1000:.2f} mm  vs fixed axis-aligned "
         f"box {aabb_p*1000:.2f} mm   (encoded probe was {ext_mae*1000:.2f} mm)")
    _log(f"            under-estimates on {(d_p < 0).mean():.1%} of samples "
         f"(encoded {under:.1%})   degradation {err_p.mean()/max(m,1e-9):.2f}x")
    _log("            signed residual (pred-true) mm: " +
         "  ".join(f"q{q}={np.percentile(d_p, q)*1000:+.2f}" for q in (1, 5, 10, 50, 90, 99)))
    _log("            margin to cover under-estimation: " +
         "  ".join(f"{c}%->{max(0.0, -np.percentile(d_p, 100-c))*1000:.2f}mm" for c in (90, 95, 99)))
    _sq = sq_p.astype(bool)
    _log(f"            square-footprint frames: {_sq.mean():.2%}  "
         f"(mid-tip frames have no single angle; extent MAE there is irreducible)")
    if _sq.any():
        _log(f"            SQUARE-ONLY  yaw err {err_p[_sq].mean():.2f} deg   "
             f"extent MAE {np.abs(d_p[_sq]).mean()*1000:.2f} mm")
    pred_metrics = {"n": int(len(yaw_p)), "horizons": pred_h,
                    "square_frac": float(_sq.mean()),
                    "square_only": {"yaw_err_deg_mean": float(err_p[_sq].mean()) if _sq.any() else None,
                                    "extent_mae_m": float(np.abs(d_p[_sq]).mean()) if _sq.any() else None},
                    "model_dir": args.model_dir, "epoch": str(args.epoch),
                    "pred_data_dir": args.pred_data_dir or args.data_dir,
                    "same_dataset_as_train": (args.pred_data_dir or args.data_dir) == args.data_dir,
                    "yaw_err_deg_mean": float(err_p.mean()),
                    "yaw_err_deg_p90": float(np.percentile(err_p, 90)),
                    "yaw_err_deg_p99": float(np.percentile(err_p, 99)),
                    "extent_mae_m": float(np.abs(d_p).mean()),
                    "extent_under_frac": float((d_p < 0).mean()),
                    # SIGNED residual (predicted - true), meters. NEGATIVE = the probe thinks the cube
                    # is SMALLER than it is -- the direction that lets a real violation go undetected.
                    # Kept as a full quantile table, and as the margin that would cover each coverage
                    # level, so the alpha stays a choice rather than something baked in here.
                    "extent_residual_m": {f"q{q}": float(np.percentile(d_p, q))
                                          for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
                    "extent_residual_mean_m": float(d_p.mean()),
                    "margin_for_coverage_m": {f"{c}pct": float(max(0.0, -np.percentile(d_p, 100 - c)))
                                              for c in (90, 95, 99)},
                    "fixed_aabb_extent_mae_m": aabb_p}

    return pred_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_5k")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame_stride", type=int, default=1, help="use every Nth (non-padded) frame")
    ap.add_argument("--enc_batch", type=int, default=64)
    ap.add_argument("--pool_grid", type=int, default=8,
                    help="spatial pool size; 8x8 matches the position probe. Orientation is a LOCAL "
                    "property of the cube, so if the probe is weak try 12 or 16 before more epochs.")
    ap.add_argument("--enc_res", type=int, default=196,
                    help="encoder input resolution -- MUST match the WM (196 = 14x14 tokens for "
                    "vits14@224). Do NOT use 224.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--encoder_name", default="dinov2_vits14")
    ap.add_argument("--val_frac", type=float, default=0.2, help="fraction of EPISODES held out")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=5, help="epochs between held-out evaluations")
    ap.add_argument("--log_every", type=int, default=100, help="episodes between encode progress lines")
    ap.add_argument("--load_probe", default=None,
                    help="Skip encoding + training and load a probe saved earlier, then run ONLY the "
                    "predicted-latent eval. The encoded pass costs ~30 min on the 5k set and its "
                    "result never changes; this makes iterating on --pred_data_dir / --pred_horizons "
                    "a two-minute job. Requires --eval_predicted.")
    ap.add_argument("--eval_predicted", action="store_true",
                    help="After training on ENCODED latents, ALSO evaluate the trained probe on "
                    "WM-PREDICTED latents -- the ones enforcement actually reads at plan time. Train "
                    "stays encoded-only (matching cube_position); this only measures the gap.")
    ap.add_argument("--pred_data_dir", default=None,
                    help="dataset for --eval_predicted (default: --data_dir). SET THIS to a set the "
                    "WM did NOT train on: wm_5k was trained on data/isaaclab_stroke_5k, so rolling it "
                    "there measures memorised dynamics and flatters the probe. data/conformal_calib_1k "
                    "is the clean dump. The probe itself is unaffected either way -- it trains on "
                    "frozen-DINOv2 ENCODED latents, which never saw any of this data.")
    ap.add_argument("--model_dir", default="outputs/wm_5k", help="WM dir for --eval_predicted")
    ap.add_argument("--epoch", default="30", help="WM epoch for --eval_predicted")
    ap.add_argument("--pred_horizons", default="1",
                    help="comma list of open-loop horizons for --eval_predicted. Use the planner's "
                    "lookahead depth (mpc_rrt is horizon-1 by default).")
    ap.add_argument("--metrics_json", default=None)
    ap.add_argument("--save_path", default=None)
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="CONTROL: permute yaw targets across frames. Real signal -> error collapses "
                    "to the constant baseline; if it stays low there is a leak/bug.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    _log(f"[cfg] data={args.data_dir} device={device} epochs={args.epochs} "
         f"pool_grid={args.pool_grid} hidden={args.hidden}")

    if args.load_probe:
        # EVAL-ONLY PATH. Rebuild the probe from the checkpoint and jump straight to the predicted
        # eval. The by-episode split is recomputed rather than stored: it is a pure function of
        # (seed, episode count), and episode ids are 0..N-1 in sorted-file order, so this reproduces
        # the original held-out set exactly -- asserted below against the saved feature width.
        if not args.eval_predicted:
            raise SystemExit("--load_probe only makes sense with --eval_predicted")
        ck = torch.load(args.load_probe, map_location="cpu")
        if ck.get("kind") != "yaw_mod90_sincos4":
            raise SystemExit(f"{args.load_probe} is kind={ck.get('kind')!r}, not a yaw probe")
        if int(ck["pool_grid"]) != args.pool_grid:
            raise SystemExit(f"--pool_grid {args.pool_grid} != checkpoint {ck['pool_grid']}; features "
                             "would not match the trained head")
        probe = MLP(int(ck["d_in"]), d_out=2, hidden=int(ck["hidden"])).to(device)
        probe.load_state_dict(ck["state_dict"]); probe.eval()
        mu, sd = torch.from_numpy(ck["x_mu"]), torch.from_numpy(ck["x_sd"])
        n_eps = len(torch.load(Path(args.data_dir) / "seq_lengths.pth"))
        ids = np.arange(n_eps); np.random.RandomState(args.seed).shuffle(ids)
        val_eps = set(ids[:max(1, int(round(n_eps * args.val_frac)))].tolist())
        _log(f"[load] {args.load_probe}  d_in={ck['d_in']} pool_grid={ck['pool_grid']} "
             f"hidden={ck['hidden']}  ({len(val_eps)} held-out episodes recomputed from seed)")
        m = p90 = ext_mae = under = float("nan")     # encoded metrics not recomputed on this path
        Xn = Yn = None
        X = np.empty((0, int(ck["d_in"])), dtype=np.float32)
        pm = _run_predicted(args, probe, mu, sd, device, val_eps, m, p90, ext_mae, under)
        # Persist. Without this the --load_probe path printed its results and dropped them: the whole
        # point of the flag is cheap iteration on --pred_data_dir / --pred_horizons, so every one of
        # those iterations must leave a record or the numbers live only in a terminal buffer.
        mj = args.metrics_json or (args.load_probe + ".predicted.json")
        pr = provenance.write(mj, "probes/probe_cube_yaw.py", args=args,
                              extra={"results": {"loaded_from": args.load_probe,
                                                 "predicted_latent_eval": pm}}, repo=_REPO_ROOT)
        if pr:
            _log(f"[metrics] results + provenance -> {pr}")
        return

    encoder = DinoV2Encoder(name=args.encoder_name, feature_key="x_norm_patchtokens").to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad_(False)
    X, YAW, SQ, EXT, ep = encode_dataset(args.data_dir, encoder, device, args.frame_stride,
                                         args.enc_batch, args.pool_grid, args.enc_res, args.log_every)

    if args.shuffle_labels:
        YAW = YAW[np.random.RandomState(args.seed + 1).permutation(len(YAW))]
        _log("[CONTROL] yaw targets permuted across frames -- expect error to collapse to baseline.")

    # TARGET: (sin 4t, cos 4t). The 4x folds the square's 90deg symmetry into a full circle, so the
    # regression target is continuous everywhere (no wrap seam) and the network is never asked to
    # distinguish orientations that are physically identical.
    Y = np.stack([np.sin(4.0 * YAW), np.cos(4.0 * YAW)], axis=1).astype(np.float32)

    # TRAIN ONLY ON SQUARE-FOOTPRINT FRAMES. A cube caught mid-tip projects to a hexagon, so no
    # single angle is the right answer and fitting one injects noise into every other frame. They are
    # still EVALUATED (below) -- excluding them from the test set would flatter the probe.
    ep_ids = np.unique(ep)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ep_ids)
    n_val = max(1, int(round(len(ep_ids) * args.val_frac)))
    val_eps = set(ep_ids[:n_val].tolist())
    test_mask = np.array([e in val_eps for e in ep])
    train_mask = (~test_mask) & SQ.astype(bool)
    _log(f"[label] footprint angle from the FULL quaternion; {SQ.mean():.2%} of frames have a square "
         f"hull. Non-square frames are dropped from TRAIN ({int((~SQ).sum())}) but kept in TEST.")
    _log(f"[split] {len(ep_ids) - n_val} train / {n_val} test episodes "
         f"({train_mask.sum()} / {test_mask.sum()} frames)")

    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    tr = torch.from_numpy(np.where(train_mask)[0])
    te = torch.from_numpy(np.where(test_mask)[0])
    mu = Xt[tr].mean(0, keepdim=True)
    sd = Xt[tr].std(0, keepdim=True) + 1e-6
    Xn = ((Xt - mu) / sd).to(device)
    Yn = Yt.to(device)                                  # already on the unit circle; do NOT standardize

    probe = MLP(X.shape[1], d_out=2, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    yaw_te = YAW[test_mask]
    ext_true = EXT[test_mask]      # EXACT projected hull, not the yaw model applied to the true angle
    # BASELINE 1: the circular mean of 4t on TRAIN -- the best constant angle.
    s_bar, c_bar = np.sin(4.0 * YAW[train_mask]).mean(), np.cos(4.0 * YAW[train_mask]).mean()
    yaw_const = np.arctan2(s_bar, c_bar) / 4.0
    base_deg = float(np.degrees(np.abs(wrap90(yaw_const - yaw_te))).mean())
    # BASELINE 2: the model the PAPER currently uses -- a fixed axis-aligned box, i.e. extent == h.
    ext_aabb_err = float(np.abs(CUBE_HALF - ext_true).mean())

    def evaluate():
        probe.eval()
        with torch.no_grad():
            pn = probe(Xn[te]).cpu().numpy()
        yaw_hat = np.arctan2(pn[:, 0], pn[:, 1]) / 4.0          # decode: atan2(sin4t, cos4t)/4
        err = np.degrees(np.abs(wrap90(yaw_hat - yaw_te)))
        ext_hat = aabb_half_extent(yaw_hat)
        d = ext_hat - ext_true
        return (float(err.mean()), float(np.percentile(err, 90)),
                float(np.abs(d).mean()), float((d < 0).mean()), err, d)

    _log(f"[geom] CUBE_HALF={CUBE_HALF:.4f} m, true half-extent ranges "
         f"{CUBE_HALF:.4f}..{CUBE_HALF*np.sqrt(2):.4f} m")
    _log(f"[baseline] constant-yaw ({np.degrees(yaw_const):+.1f} deg): {base_deg:.2f} deg mean error")
    _log(f"[baseline] fixed axis-aligned box (what the paper scores with today): "
         f"extent err {ext_aabb_err*1000:.2f} mm = {ext_aabb_err/CUBE_HALF:.1%} of CUBE_HALF, "
         f"and it under-estimates on {(CUBE_HALF < ext_true).mean():.1%} of frames")

    for epoch in range(1, args.epochs + 1):
        probe.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(probe(Xn[idx]), Yn[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m, p90, ext_mae, under, _, _ = evaluate()
            _log(f"[probe] epoch {epoch:3d}  train_mse {tot/len(perm):.4f}  "
                 f"yaw_err {m:5.2f} deg (p90 {p90:5.2f})  "
                 f"extent_mae {ext_mae*1000:5.2f} mm  under {under:.1%}")

    m, p90, ext_mae, under, err, d = evaluate()
    _log(f"\n[RESULT] yaw error mod 90deg: mean {m:.2f} deg  p90 {p90:.2f} deg  "
         f"(constant baseline {base_deg:.2f} deg)")
    _log(f"         footprint half-extent MAE: probe {ext_mae*1000:.2f} mm   "
         f"vs fixed axis-aligned box {ext_aabb_err*1000:.2f} mm   "
         f"-> {(1 - ext_mae/ext_aabb_err):+.1%} change")
    _log(f"         probe under-estimates the extent (the UNSAFE direction) on {under:.1%} of frames")
    if ext_mae < 0.5 * ext_aabb_err:
        _log("         -> yaw is cleanly readable; an oriented-box rescoring is worth doing.")
    else:
        _log("         -> WEAK: the probe's extent error is not much better than assuming a fixed "
             "axis-aligned box, so it would buy little. Try --pool_grid 12/16 before concluding.")

    # Free the training features before the WM rollout: Xn is the entire encoded feature matrix
    # on GPU (~10 GB for the 5k set at pool_grid 8) and is finished with once evaluate() has run.
    if args.eval_predicted and Xn is not None:
        del Xn, Yn
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    pred_metrics = _run_predicted(args, probe, mu, sd, device, val_eps, m, p90, ext_mae, under)

    metrics = {
        "d_in": int(X.shape[1]),
        "n_train_ep": int(len(ep_ids) - n_val), "n_test_ep": int(n_val),
        "n_train_frames": int(train_mask.sum()), "n_test_frames": int(test_mask.sum()),
        "baseline": {"constant_yaw_deg": float(np.degrees(yaw_const)),
                     "constant_yaw_err_deg": base_deg,
                     "fixed_aabb_extent_mae_m": ext_aabb_err},
        "result": {"yaw_err_deg_mean": m, "yaw_err_deg_p90": p90,
                   "yaw_err_deg_p99": float(np.percentile(err, 99)),
                   "extent_mae_m": ext_mae, "extent_under_frac": under,
                   "extent_bias_m": float(d.mean())},
    }
    if pred_metrics is not None:
        metrics["predicted_latent_eval"] = pred_metrics

    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            # kind is DELIBERATELY not "regression": the output is a point on the unit circle at 4t,
            # not a value in target units, so any consumer must decode with atan2(s,c)/4 and must not
            # apply a y_mu/y_sd unstandardization. Naming it distinctly makes a wrong load fail loudly.
            "kind": "yaw_mod90_sincos4", "state_dict": probe.state_dict(),
            "x_mu": mu.numpy(), "x_sd": sd.numpy(),
            "pool_grid": args.pool_grid, "d_in": int(X.shape[1]), "out_dim": 2,
            "hidden": args.hidden, "encoder": args.encoder_name,
            "feature_key": "x_norm_patchtokens", "source": "encoded",
            "cube_half": CUBE_HALF, "decode": "yaw = atan2(out[0], out[1]) / 4",
        }, args.save_path)
        _log(f"[save] probe + norm stats -> {args.save_path}")

    mj = args.metrics_json or (args.save_path + ".metrics.json" if args.save_path else None)
    if mj:
        p = provenance.write(mj, "probes/probe_cube_yaw.py", args=args,
                             extra={"results": metrics}, repo=_REPO_ROOT)
        if p:
            _log(f"[metrics] results + provenance -> {p}")


if __name__ == "__main__":
    main()
