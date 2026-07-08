"""How good is the WM when you CHAIN predictions (t+1, t+2, ...) without re-observing?

Decides two coupled questions: (a) is multi-step lookahead (a tree / RRT in latent space)
viable at all, or must we stay with 1-step re-planning MPC? (b) would retraining on SHORTER
strokes help chaining, or only normative granularity?

No sim needed: frozen-DINO WM + (optional) cube probe + the .pth dataset. Shares the WM/probe
loaders with wm_cube_pred_check.py.

PRIMARY metric -- prediction -> next-state ENCODED error, per chaining horizon h:
    e_lat(h) = || ẑ_h - φ(o_real(h)) || / || φ(o_real(h)) ||
  ẑ_h = WM open-loop prediction h steps ahead; φ(o_real(h)) = encoded REAL frame at step h.
  Model-intrinsic (no probe); the quantity that governs lookahead viability.

Also:
  * e_probe(h)  probe cube error on ẑ_h vs true cube (task-space, m)         [cube data only]
  * drift(h)    ||ẑ_h|| / ||φ(o_real(h))||  -- manifold departure (~1 = on manifold)
  * Δ_chain(h)  e_lat_OPEN(h) - e_lat_CLOSED(h): chaining h steps open-loop MINUS the
                1-step error starting from the real frame at h-1 == the pure cost of NOT
                re-observing == the price a tree/RRT pays over re-planning MPC. Δ_chain(1)=0.
  * 1-step probe error binned by cube DISPLACEMENT ‖Δc‖ (the retrain-shorter proxy):
        err/‖Δc‖ DROPS for small moves -> shorter strokes proportionally more accurate
            -> retraining shorter would HELP chaining.
        err FLAT across ‖Δc‖ -> fixed per-step cost -> more (shorter) steps = more total
            error -> retraining shorter HURTS chaining (helps only granularity).

Usage:
    python scripts/wm_chaining_degradation.py \
        --model_dir outputs/2026-06-25/16-46-57 --epoch 20 \
        --data_dir data/isaaclab_stroke_1500 --probe probes/weights/probe_cube_1500.pth --horizon 5
    # one-time noisy baseline (latent metrics only; deformable WM has no cube probe):
    python scripts/wm_chaining_degradation.py --include_deformable
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import hydra

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# reuse the proven WM/probe loaders + probe forward from the sibling diagnostic
from scripts.wm_cube_pred_check import load_wm, load_probe, probe_xy, CUBE_OFF


def _frob(x):
    """Frobenius norm over all-but-batch dims -> (b,)."""
    return x.reshape(x.shape[0], -1).norm(dim=1)


# ------------------------------------------------------------------------------- adapters
# Each adapter yields fixed-length windows of W = num_hist + H frames as torch tensors,
# already in the WM's expected input form (visual in [-1,1], proprio/action normalized).
class IsaacLabAdapter:
    has_cube = True

    def __init__(self, data_dir):
        p = Path(data_dir)
        self.p = p
        self.states = torch.load(p / "states.pth").float().numpy()      # (E,T,31)
        self.actions = torch.load(p / "actions.pth").float().numpy()    # (E,T,A)
        self.proprio = torch.load(p / "proprio.pth").float().numpy()    # (E,T,Pd)
        self.E, self.T, _ = self.states.shape
        self.A = self.actions.shape[-1]
        self.pdim = self.proprio.shape[-1]
        self.a_mean = self.actions.reshape(-1, self.A).mean(0); self.a_std = self.actions.reshape(-1, self.A).std(0) + 1e-6
        self.p_mean = self.proprio.reshape(-1, self.pdim).mean(0); self.p_std = self.proprio.reshape(-1, self.pdim).std(0) + 1e-6

    def get(self, e, f0, W):
        vid = torch.load(self.p / "obses" / f"episode_{e:05d}.pth")[f0:f0 + W]   # (W,H,W,3) uint8
        vis = (vid.float().permute(0, 3, 1, 2) / 255.0) * 2.0 - 1.0              # (W,3,H,W), Normalize(.5,.5)
        pro = torch.tensor((self.proprio[e, f0:f0 + W] - self.p_mean) / self.p_std).float()
        act = torch.tensor((self.actions[e, f0:f0 + W] - self.a_mean) / self.a_std).float()
        cube = torch.tensor(self.states[e, f0:f0 + W, CUBE_OFF:CUBE_OFF + 2]).float()
        return vis, pro, act, cube


class DeformAdapter:
    """One-time noisy baseline. Particle-cloud env: NO cube, dummy proprio (dim 1)."""
    has_cube = False

    def __init__(self, data_dir, object_name, transform, normalize_action):
        from datasets.deformable_env_dset import DeformDataset
        self.dset = DeformDataset(data_path=data_dir, object_name=object_name,
                                  transform=transform, normalize_action=normalize_action)
        self.E = len(self.dset)
        self.T = int(self.dset.get_seq_length(0))
        self.A = self.dset.action_dim
        self.pdim = self.dset.proprio_dim

    def get(self, e, f0, W):
        obs, act, _state, _ = self.dset.get_frames(e, range(f0, f0 + W))
        return obs["visual"].float(), obs["proprio"].float(), act.float(), None


# ----------------------------------------------------------------------------- core sweep
@torch.no_grad()
def eval_chaining(wm, num_hist, H, adapter, mlp, pinfo, n_windows, batch, seed, device,
                  mlp2=None, pinfo2=None):
    """Open-loop chaining sweep h=1..H + the 1-step (h=1) probe/displacement diagnostics.
    mlp/pinfo = ENCODED-trained probe (floor + encH-on-pred column); mlp2/pinfo2 = optional
    PREDICTED-trained probe (predH-on-pred column, isolates WM drift from probe shift).
    Returns (per_h dict, one_step dict)."""
    rng = np.random.RandomState(seed)
    W = num_hist + H
    max_f = adapter.T - W
    if max_f < 0:
        raise ValueError(f"episodes too short (T={adapter.T}) for num_hist+H={W}")
    windows = [(int(rng.randint(0, adapter.E)), int(rng.randint(0, max_f + 1))) for _ in range(n_windows)]

    acc = {h: {"e_lat_open": [], "e_lat_closed": [], "drift": [], "e_probe": [], "e_probe_enc": [],
               "e_probe_pred": []}
           for h in range(1, H + 1)}
    one = {"e_probe1": [], "e_lat1": [], "move_true": [], "move_pred1": [], "d_start": []}

    for i in range(0, len(windows), batch):
        bw = windows[i:i + batch]
        vis = torch.stack([adapter.get(e, f, W)[0] for e, f in bw]).to(device)     # (b,W,3,H,W)
        pro = torch.stack([adapter.get(e, f, W)[1] for e, f in bw]).to(device)     # (b,W,pd)
        act = torch.stack([adapter.get(e, f, W)[2] for e, f in bw]).to(device)     # (b,W,A)
        cubes = None
        if adapter.has_cube:
            cubes = torch.stack([adapter.get(e, f, W)[3] for e, f in bw]).numpy()   # (b,W,2)

        # encode ALL real frames once -> z_true[:, t] = φ(real frame t)
        z_true = wm.encode_obs({"visual": vis, "proprio": pro})["visual"]          # (b,W,P,D)

        for h in range(1, H + 1):
            tgt = num_hist - 1 + h                                                  # target frame index
            zt = z_true[:, tgt]                                                     # (b,P,D) encoded real
            nt = _frob(zt).clamp_min(1e-6)
            # OPEN: chain h steps from the initial real history (frames 0..num_hist-1)
            z_open = wm.rollout(obs_0={"visual": vis[:, :num_hist], "proprio": pro[:, :num_hist]},
                                act=act[:, :num_hist + h - 1])[0]["visual"][:, -1]   # (b,P,D)
            # CLOSED: 1-step from the REAL history ending at the target's predecessor
            z_closed = wm.rollout(obs_0={"visual": vis[:, h - 1:h - 1 + num_hist],
                                         "proprio": pro[:, h - 1:h - 1 + num_hist]},
                                  act=act[:, h - 1:h - 1 + num_hist])[0]["visual"][:, -1]
            acc[h]["e_lat_open"].extend((_frob(z_open - zt) / nt).cpu().numpy())
            acc[h]["e_lat_closed"].extend((_frob(z_closed - zt) / nt).cpu().numpy())
            acc[h]["drift"].extend((_frob(z_open) / nt).cpu().numpy())
            if adapter.has_cube and mlp is not None:
                ct = cubes[:, tgt]                                                  # (b,2) true cube
                pe = probe_xy(mlp, pinfo, z_open, device)
                ee = probe_xy(mlp, pinfo, zt, device)
                acc[h]["e_probe"].extend(np.linalg.norm(pe - ct, axis=1))
                acc[h]["e_probe_enc"].extend(np.linalg.norm(ee - ct, axis=1))
                if mlp2 is not None:  # pred-head probe on the SAME predicted latent
                    pe2 = probe_xy(mlp2, pinfo2, z_open, device)
                    acc[h]["e_probe_pred"].extend(np.linalg.norm(pe2 - ct, axis=1))
                if h == 1:
                    one["e_probe1"].extend(np.linalg.norm(pe - ct, axis=1))
                    one["e_lat1"].extend((_frob(z_open - zt) / nt).cpu().numpy())
                    one["move_true"].extend(np.linalg.norm(cubes[:, num_hist] - cubes[:, num_hist - 1], axis=1))
                    one["move_pred1"].extend(np.linalg.norm(pe - cubes[:, num_hist - 1], axis=1))
                    # |stroke start - cube| (start = first 2 action dims, denormalized)
                    a0 = act[:, num_hist - 1, :2].cpu().numpy() * adapter.a_std[:2] + adapter.a_mean[:2]
                    one["d_start"].extend(np.linalg.norm(a0 - cubes[:, num_hist - 1], axis=1))
    return acc, one


def _mean(a):
    a = np.asarray(a)
    return a.mean() if a.size else float("nan")


def report(tag, acc, one, H, has_cube, has_pred=False):
    CELL = 2 * 0.0667
    print(f"\n================= {tag} =================")
    print(f"[chaining sweep]  predicted latent vs ENCODED real next-state, by horizon h")
    hdr = f"  {'h':>2}{'e_lat_open':>12}{'e_lat_closed':>14}{'Δ_chain':>10}{'drift':>8}"
    if has_cube:
        hdr += f"{'encH_pred':>11}"
        if has_pred:
            hdr += f"{'predH_pred':>12}"
        hdr += f"{'encH_real':>11}"
    print(hdr)
    Hstar = None
    for h in range(1, H + 1):
        lo = _mean(acc[h]["e_lat_open"]); lc = _mean(acc[h]["e_lat_closed"]); dr = _mean(acc[h]["drift"])
        row = f"  {h:>2}{lo:>12.4f}{lc:>14.4f}{lo - lc:>10.4f}{dr:>8.3f}"
        if has_cube:
            ep = _mean(acc[h]["e_probe"]); en = _mean(acc[h]["e_probe_enc"])
            row += f"{ep:>11.4f}"
            ref = ep                                          # which probe sets H*
            if has_pred:
                epp = _mean(acc[h]["e_probe_pred"]); row += f"{epp:>12.4f}"; ref = epp
            row += f"{en:>11.4f}"
            if Hstar is None and ref >= CELL / 2:
                Hstar = h - 1
        print(row)
    if has_cube:
        which = "predH_pred" if has_pred else "encH_pred"
        print(f"  reliable horizon H* ({which} < CELL/2={CELL / 2:.3f} m): {H if Hstar is None else Hstar} step(s)")
        if has_pred:
            print("  decompose:  shift penalty = encH_pred - predH_pred (recovered by the pred head);  "
                  "residual WM drift = predH_pred - encH_real (the part NO probe can fix -> the true horizon cap).")
    print("  encH/predH = encoded/predicted-trained probe; _pred = on the predicted latent, _real = on the real frame.")
    print("  e_lat=||pred-encoded||/||encoded|| (0=perfect). Δ_chain=open-loop chaining cost over re-observing.")

    if not has_cube or not one["e_probe1"]:
        return
    e1 = np.array(one["e_probe1"]); l1 = np.array(one["e_lat1"])
    mt = np.array(one["move_true"]); mp = np.array(one["move_pred1"]); ds = np.array(one["d_start"])
    contact = mt > 0.02
    print(f"\n[1-step contact split]  CONTACT (moved>2cm, n={int(contact.sum())}): "
          f"e_probe {e1[contact].mean():.4f} m  e_lat {l1[contact].mean():.4f}"
          + (f"   |   MISS (n={int((~contact).sum())}): e_probe {e1[~contact].mean():.4f} m" if (~contact).any() else ""))

    print(f"\n[displacement bins]  retrain-shorter proxy: 1-step error vs cube move ‖Δc‖")
    print(f"  {'‖Δc‖ (m)':<12}{'n':>5}{'e_probe':>10}{'e_lat':>9}{'e_probe/‖Δc‖':>14}")
    for lo, hi in [(0.0, 0.02), (0.02, 0.05), (0.05, 0.09), (0.09, 0.15), (0.15, 9.0)]:
        m = (mt >= lo) & (mt < hi)
        if not m.any():
            continue
        per = e1[m].mean() / max(mt[m].mean(), 1e-6)
        print(f"  {f'{lo:.2f}-{hi:.2f}':<12}{int(m.sum()):>5}{e1[m].mean():>10.4f}{l1[m].mean():>9.4f}{per:>14.2f}")

    print(f"\n[start-distance bins]  |stroke_start - cube| -> WM behavior (CELL={CELL:.3f} m)")
    print(f"  {'bin (m)':<12}{'n':>5}{'e_probe':>10}{'true_move':>11}{'pred_move':>11}")
    for lo, hi in [(0, 0.06), (0.06, 0.12), (0.12, 0.20), (0.20, 0.35), (0.35, 9.0)]:
        m = (ds >= lo) & (ds < hi)
        if not m.any():
            continue
        print(f"  {f'{lo:.2f}-{hi:.2f}':<12}{int(m.sum()):>5}{e1[m].mean():>10.4f}{mt[m].mean():>11.4f}{mp[m].mean():>11.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/2026-06-25/16-46-57")
    ap.add_argument("--epoch", default="20")
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--probe", default="probes/weights/probe_cube_1500.pth", help="ENCODED-trained probe (floor + encH-on-pred)")
    ap.add_argument("--probe_pred", default=None,
                    help="optional PREDICTED-trained probe (e.g. probe_cube_pred.pth) -> adds the "
                    "predH_pred column so we can isolate WM drift from probe distribution-shift")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=5, help="max open-loop chaining horizon")
    ap.add_argument("--seed", type=int, default=0)
    # one-time noisy baseline; OFF by default
    ap.add_argument("--include_deformable", action="store_true", default=False)
    ap.add_argument("--deformable_model_dir", default="outputs/2026-06-25/19-40-40")
    ap.add_argument("--deformable_epoch", default="20")
    ap.add_argument("--deformable_data_dir", default="checkpoints/deformable")
    ap.add_argument("--deformable_object", default="granular")
    args = ap.parse_args()
    dev = args.device

    # ---- primary: our cube WM ----
    wm, tcfg = load_wm(args.model_dir, args.epoch, dev)
    num_hist = int(tcfg.num_hist)
    mlp, pinfo = load_probe(args.probe, dev)
    mlp2 = pinfo2 = None
    if args.probe_pred:
        mlp2, pinfo2 = load_probe(args.probe_pred, dev)
    print(f"[load] cube WM num_hist={num_hist} | enc-probe d_in={pinfo['d_in']}"
          + (f" | pred-probe {args.probe_pred}" if mlp2 is not None else "") + f" | H={args.horizon}")
    iso = IsaacLabAdapter(args.data_dir)
    acc, one = eval_chaining(wm, num_hist, args.horizon, iso, mlp, pinfo,
                             args.n_windows, args.batch, args.seed, dev, mlp2=mlp2, pinfo2=pinfo2)
    report(f"OURS  {args.model_dir}@{args.epoch}", acc, one, args.horizon, has_cube=True, has_pred=mlp2 is not None)

    # ---- optional: deformable noisy baseline (latent metrics only) ----
    if args.include_deformable:
        try:
            dwm, dcfg = load_wm(args.deformable_model_dir, args.deformable_epoch, dev)
            d_nh = int(dcfg.num_hist)
            tfm = hydra.utils.instantiate(dcfg.env.dataset.transform)
            norm_a = bool(getattr(dcfg, "normalize_action", True))
            dadapt = DeformAdapter(args.deformable_data_dir, args.deformable_object, tfm, norm_a)
            print(f"\n[load] deformable WM num_hist={d_nh} object={args.deformable_object} (latent-only; no cube probe)")
            dacc, done = eval_chaining(dwm, d_nh, args.horizon, dadapt, None, None,
                                       min(args.n_windows, 128), args.batch, args.seed, dev)
            report(f"DEFORMABLE  {args.deformable_model_dir}@{args.deformable_epoch} (noisy baseline)",
                   dacc, done, args.horizon, has_cube=False)
        except Exception as ex:  # noqa: BLE001 -- baseline is best-effort; never block the primary run
            print(f"\n[deformable baseline SKIPPED] {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
