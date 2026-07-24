"""Per-cell cube-occupancy probe (multi-label).

For each of the 9 grid cells, predict whether ANY part of the cube is in that cell -> a
length-9 binary vector per frame. Multiple cells can be 1 (a 9cm cube straddles cells), so
this is multi-label, trained with BCE. This mirrors the neuroscience contrast: the continuous
position probe is a grid/rate code, this per-cell probe is a place-cell code (one detector
per location).

Ground truth = AABB overlap between the cube and each cell. Cube center is state[18:20];
cube half-extent 0.045 m (9cm cube; grid_wrapper_single: _CUBE_SPAWN_Z=0.046 -> half 0.045).
A cell c (center cell_center(c), half CELL/2) is occupied iff
    |cube_x - cx| < CELL/2 + cube_half  AND  |cube_y - cy| < CELL/2 + cube_half.

Reuses the position probe's encoder + pooled-feature extraction (encode_dataset) and the same
MLP head (d_out=9). Saves with kind='multilabel' so probes/registry.py loads it generically.

    python probes/probe_cube_cells.py --data_dir data/isaaclab_stroke_1500 \
        --save_path probes/weights/probe_cube_cells.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probes.probe_cube_position import MLP, encode_dataset, predict_dataset, gm  # shared head + features + grid_metadata
from models.dino import DinoV2Encoder

CUBE_HALF = 0.045  # 9cm cube half-extent (grid_wrapper_single: _CUBE_SPAWN_Z=0.046 -> half 0.045)


def cube_cell_occupancy(xy, cube_half):
    """(N,2) cube centers (env-local m) -> (N, N_CELLS) binary occupancy via AABB overlap."""
    centers = np.array([gm.cell_center(c) for c in range(gm.N_CELLS)], dtype=np.float32)  # (9,2)
    half = gm.CELL / 2.0 + cube_half
    d = np.abs(xy[:, None, :] - centers[None, :, :])                                       # (N,9,2)
    return ((d[..., 0] < half) & (d[..., 1] < half)).astype(np.float32)                    # (N,9)


def _seg_aabb_hit(p0, p1, lo, hi):
    """Does the 2D segment p0->p1 intersect the axis-aligned box [lo,hi]? (Liang-Barsky)."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            ta, tb = (lo[i] - p0[i]) / d[i], (hi[i] - p0[i]) / d[i]
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


def swept_cells(c0, c1, cube_half):
    """(N_CELLS,) bool: cells the cube footprint (half cube_half) touches anywhere along the
    straight segment c0->c1 (segment vs each cell's AABB expanded by CELL/2 + cube_half).
    Run-time computable from probe positions alone -> supplements boundary-only occupancy with
    mid-stroke transit. Conservative: a corner clip counts (shrink to CELL/2 for center-only)."""
    H = gm.CELL / 2.0 + cube_half
    out = np.zeros(gm.N_CELLS, dtype=bool)
    c0 = np.asarray(c0, np.float64); c1 = np.asarray(c1, np.float64)
    for c in range(gm.N_CELLS):
        cx, cy = gm.cell_center(c)
        out[c] = _seg_aabb_hit(c0, c1, np.array([cx - H, cy - H]), np.array([cx + H, cy + H]))
    return out


STROKE_MOVE_EPS = 0.02  # min stroke displacement (m) to count as a real push (skip holding frames)


def _angle_deg(u, v):
    """Angle (deg, [0,180]) between 2D vectors u,v; NaN if either is ~zero."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)) / (nu * nv), -1.0, 1.0))))


def _seg_point_dist(c0, c1, p):
    """Closest-approach distance from segment c0->c1 to point p (m)."""
    c0, c1, p = (np.asarray(a, float) for a in (c0, c1, p))
    d = c1 - c0
    L2 = float(d @ d)
    if L2 < 1e-18:
        return float(np.linalg.norm(p - c0))
    t = float(np.clip((p - c0) @ d / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (c0 + t * d)))


def stroke_breach_geometry(cube_xy, cells, cube_half=CUBE_HALF, move_eps=STROKE_MOVE_EPS):
    """Per-stroke swept-breach geometry vs each checked cell, computed from a cube boundary path.

    cube_xy : (Ti,2) trimmed boundary path (exactly what build_eval_metrics saves as cube_xy_frames).
    cells   : iterable of checked cell ids (usually just the illegal cell). Frame-0 spawn is
              grandfathered PER CELL (a cube spawned inside cell c gets its 0->1 escape stroke for
              free), matching planning_metrics' swept check.

    Returns a list of dicts, one per (MOVING stroke t->t+1, checked cell c). "Toward the cell" is
    reported TWO ways (it's not obvious which is the better predictor, so we track both):
      approach_deg      : angle(heading, cell_CENTER - c0); 0 = pushing straight at the cell center
      approach_edge_deg : angle(heading, NEAREST-BOUNDARY-point - c0), where the boundary is the cell
                          AABB inflated by cube_half (H = CELL/2 + cube_half) -- the box the cube CENTER
                          must cross to breach. This reads edge-skirting strokes as more 'toward' than
                          the center angle does. NaN if c0 is already inside that box.
    both: 0 = toward, 90 = tangent/around, 180 = away.
      breach       : bool -- footprint swept through cell c on this stroke (swept_cells)
      dmin         : closest approach (m) of the stroke segment to the cell center (proximity control)
      side         : signed lateral offset (m) of the cell center off the push line (sign = which side)
      start_x/end_x, start_y/end_y : stroke endpoints (start-side analysis: arm reaches from -x)
      t, cell      : stroke index and cell id
    This is the single source of truth shared by planning_metrics (per-eval violation_approach/side),
    the offline analysis, and the tracking plots -- so their numbers cannot drift apart."""
    cube_xy = np.asarray(cube_xy, float)
    out = []
    cells = [int(c) for c in (cells or [])]
    if cube_xy.ndim != 2 or len(cube_xy) < 2 or not cells:
        return out
    H = gm.CELL / 2.0 + cube_half                                  # center-breach boundary half-width
    occ0 = cube_cell_occupancy(cube_xy[:1], cube_half)[0]           # frame-0 footprint occupancy (grandfather)
    for c in cells:
        cc = np.asarray(gm.cell_center(c), float)
        lo = 1 if occ0[c] > 0.5 else 0                             # spawn in c -> skip its one escape stroke
        for t in range(lo, len(cube_xy) - 1):
            c0, c1 = cube_xy[t], cube_xy[t + 1]
            head = c1 - c0
            if np.linalg.norm(head) < move_eps:                    # holding / no-op frame
                continue
            hn = head / np.linalg.norm(head)
            edge = np.clip(c0, cc - H, cc + H)                     # nearest point of the H-box to c0
            out.append(dict(
                t=int(t), cell=c,
                breach=bool(swept_cells(c0, c1, cube_half)[c]),
                approach_deg=_angle_deg(head, cc - c0),           # toward the CENTER
                approach_edge_deg=_angle_deg(head, edge - c0),    # toward the NEAREST BOUNDARY point (NaN if inside)
                dmin=_seg_point_dist(c0, c1, cc),
                side=float(hn[0] * (cc[1] - c0[1]) - hn[1] * (cc[0] - c0[0])),
                start_x=float(c0[0]), end_x=float(c1[0]),
                start_y=float(c0[1]), end_y=float(c1[1]),
            ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--source", choices=["encoded", "predicted"], default="encoded",
                    help="features from φ(real frame) (encoded) or WM-rolled latents (predicted). "
                    "Train one probe per source; the registry routes encoded probes to encoded "
                    "latents and predicted probes to WM-predicted latents.")
    ap.add_argument("--model_dir", default="outputs/reg_dino", help="WM dir (only for --source predicted)")
    ap.add_argument("--epoch", default="20", help="WM epoch (only for --source predicted)")
    ap.add_argument("--pred_horizons", default="1",
                    help="open-loop horizons to build predicted features at, e.g. '1,2,3,4,5'; match the "
                    "planner depth. Higher h has drifted latents -> noisier occupancy targets.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--enc_batch", type=int, default=64)
    ap.add_argument("--pool_grid", type=int, default=8)
    ap.add_argument("--enc_res", type=int, default=196)
    ap.add_argument("--cube_half", type=float, default=CUBE_HALF)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="CONTROL: permute occupancy across frames -> F1 should collapse to the prior baseline")
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device

    if args.source == "encoded":
        encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(dev).eval()
        for prm in encoder.parameters():
            prm.requires_grad_(False)
        X, Y, ep, _V = encode_dataset(args.data_dir, encoder, dev, args.frame_stride,
                                      args.enc_batch, args.pool_grid, args.enc_res)
    else:  # predicted: WM-rolled latents (dataset actions -> true cube at each horizon step)
        from scripts.wm_cube_pred_check import load_wm  # lazy: avoids a circular import at module load
        pred_h = [int(s) for s in str(args.pred_horizons).split(",") if s.strip()]
        wm, tcfg = load_wm(args.model_dir, args.epoch, dev)
        print(f"[predict] WM {args.model_dir}@{args.epoch} num_hist={int(tcfg.num_hist)} horizons={pred_h}")
        X, Y, ep, _V = predict_dataset(args.data_dir, wm, int(tcfg.num_hist), dev,
                                       pred_h, args.enc_batch, args.pool_grid)
    occ = cube_cell_occupancy(Y, args.cube_half)                                   # (N,9)
    n_cells = occ.shape[1]
    print(f"[occupancy] {occ.shape[0]} frames | mean cells occupied/frame={occ.sum(1).mean():.2f} | "
          f"per-cell prior={np.round(occ.mean(0), 3)}")
    if args.shuffle_labels:
        occ = occ[np.random.RandomState(args.seed + 1).permutation(len(occ))]
        print("[CONTROL] occupancy permuted across frames -> expect F1 ~ prior baseline")

    # split BY EPISODE (no frame leakage)
    ep_ids = np.unique(ep); rng = np.random.RandomState(args.seed); rng.shuffle(ep_ids)
    n_val = max(1, int(round(len(ep_ids) * args.val_frac)))
    val = set(ep_ids[:n_val].tolist())
    te_mask = np.array([e in val for e in ep]); tr_mask = ~te_mask
    print(f"[split] {len(ep_ids) - n_val} train / {n_val} test episodes "
          f"({int(tr_mask.sum())}/{int(te_mask.sum())} frames)")

    Xt = torch.from_numpy(X); Ot = torch.from_numpy(occ)
    tr = torch.from_numpy(np.where(tr_mask)[0]); te = torch.from_numpy(np.where(te_mask)[0])
    mu = Xt[tr].mean(0, keepdim=True); sd = Xt[tr].std(0, keepdim=True) + 1e-6
    Xn = ((Xt - mu) / sd).to(dev); On = Ot.to(dev)

    probe = MLP(X.shape[1], d_out=n_cells).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    O_te = occ[te_mask]
    prior = occ[tr_mask].mean(0)                                   # per-cell train prevalence

    def f1_per_cell(pred, gt):
        tp = ((pred == 1) & (gt == 1)).sum(0); fp = ((pred == 1) & (gt == 0)).sum(0)
        fn = ((pred == 0) & (gt == 1)).sum(0)
        return 2 * tp / (2 * tp + fp + fn + 1e-9)

    base_pred = np.tile((prior > 0.5).astype(np.float32), (len(O_te), 1))   # predict-majority baseline
    base_macro_f1 = float(f1_per_cell(base_pred, O_te).mean())

    def evaluate():
        probe.eval()
        with torch.no_grad():
            logits = probe(Xn[te]).cpu().numpy()
        pred = (logits > 0).astype(np.float32)                     # sigmoid > 0.5
        per_cell_acc = float((pred == O_te).mean())
        exact = float((pred == O_te).all(1).mean())                # all 9 cells right
        f1s = f1_per_cell(pred, O_te)
        return per_cell_acc, exact, float(f1s.mean()), f1s

    print(f"[baseline] predict-majority: macro-F1={base_macro_f1:.3f}")
    for epoch in range(1, args.epochs + 1):
        probe.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(); loss = lossf(probe(Xn[idx]), On[idx]); loss.backward(); opt.step()
        if epoch % 15 == 0 or epoch == args.epochs:
            acc, exact, mf1, _ = evaluate()
            print(f"[probe] epoch {epoch:3d}  per-cell-acc {acc:.3f}  exact-match {exact:.3f}  macro-F1 {mf1:.3f}")

    acc, exact, mf1, f1s = evaluate()
    print(f"\n[RESULT] per-cell-acc {acc:.3f}  exact-match {exact:.3f}  macro-F1 {mf1:.3f} "
          f"(baseline {base_macro_f1:.3f})")
    print(f"  per-cell F1 (cell = row*3+col, 0..8): {np.round(f1s, 3)}")

    # breakdown by HOW MANY cells the cube occupies: the hard cases are the multi-cell frames
    # (cube straddling boundaries/corners). recall = fraction of occupied cells detected.
    probe.eval()
    with torch.no_grad():
        pred_te = (probe(Xn[te]).cpu().numpy() > 0).astype(np.float32)        # (Nte, 9)
    ncells = O_te.sum(1).astype(int)
    print("\n[occupancy-count breakdown] test metrics by #cells the cube occupies:")
    print(f"  {'#cells':<7}{'n':>7}{'exact':>9}{'percell_acc':>13}{'recall':>9}{'precision':>11}")
    for k in sorted(set(ncells.tolist())):
        m = ncells == k
        pm, gt_m = pred_te[m], O_te[m]
        tp = float(((pm == 1) & (gt_m == 1)).sum()); fp = float(((pm == 1) & (gt_m == 0)).sum())
        fn = float(((pm == 0) & (gt_m == 1)).sum())
        rec = tp / (tp + fn + 1e-9) if k > 0 else float("nan")               # recall undefined when 0 cells occupied
        prec = tp / (tp + fp + 1e-9)
        print(f"  {k:<7}{int(m.sum()):>7}{(pm == gt_m).all(1).mean():>9.3f}"
              f"{(pm == gt_m).mean():>13.3f}{rec:>9.3f}{prec:>11.3f}")

    if args.save_path:
        torch.save({
            "kind": "multilabel", "state_dict": probe.state_dict(),
            "x_mu": mu.numpy(), "x_sd": sd.numpy(),
            "pool_grid": args.pool_grid, "d_in": X.shape[1], "out_dim": n_cells, "hidden": 256,
            "cube_half": args.cube_half, "cell_layout": "row*3+col, 0..8",
            "encoder": "dinov2_vits14", "feature_key": "x_norm_patchtokens",
            "source": args.source, "pred_horizons": args.pred_horizons,  # encoded vs predicted-trained
        }, args.save_path)
        print(f"[save] occupancy probe -> {args.save_path}")


if __name__ == "__main__":
    main()
