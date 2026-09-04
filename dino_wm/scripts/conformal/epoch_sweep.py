"""Does a different WM checkpoint generalise better? Measured in PROBE space, on HELD-OUT episodes.

WHY
---
`outputs/wm_5k/train.log` shows a textbook overfitting curve: val_loss plateaus around epoch 11
(~0.060) and never meaningfully improves, while train_loss keeps falling to 0.033 by epoch 50.
Measured in probe space on the seed-42 held-out split, the deployed e30 checkpoint has ~1.5x the
mean cube-prediction error of its own training episodes (2.45 -> 3.62 cm) and ~2x at p90
(4.24 -> 8.75 cm), which turns a 1.0% training-set leak rate at delta=0.03 into ~8.9% held out.

But val_loss is LATENT MSE, and what the legislation layer actually consumes is the PROBE's cube
position. Those two can diverge -- a checkpoint can keep reconstructing latents well while the
cube-position readout degrades, or vice versa. This script measures the quantity we care about,
directly, across checkpoints:

    held-out 1-step cube-position error, and the leak rate at each candidate cushion.

WHAT IT DOES NOT DECIDE
-----------------------
e30 is the DEPLOYED checkpoint: every result in results/final was produced with it. If an earlier
epoch turns out better here, that is a FINDING to report, not a free switch -- changing it
invalidates the whole results tree. Treat a small difference as "e30 is fine, and here is the
evidence"; only a large gap would justify re-running anything.

METHOD NOTES
------------
* HELD-OUT ONLY. The split is reproduced exactly as datasets/traj_dset.py does it -- torch
  randperm under a Generator seeded 42, int() truncation, train takes the leading slice. Using
  numpy here would silently select different episodes (see dump_probe_residuals.py).
* ENCODE ONCE. DINOv2 is frozen during WM training, so the encoder is identical in every
  checkpoint; only the predictor differs. We encode each episode once and feed the cached latents
  to every epoch via the `visual_cached` path, which makes the sweep cheap. The script ASSERTS
  encoder equality against the first checkpoint rather than assuming it.
* 1-FRAME CONTEXT + act of length exactly num_hist, matching deployment and avoiding the 2-step
  off-by-one documented in dump_probe_residuals.py.
* Batched per episode: all 19 transitions of an episode go through the predictor as one batch.

USAGE
-----
    CUDA_VISIBLE_DEVICES=6 python scripts/conformal/epoch_sweep.py --epochs 15 20 25 30 40 50
    CUDA_VISIBLE_DEVICES=6 python scripts/conformal/epoch_sweep.py --limit 100   # quick look
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probes.probe_cube_cells import CUBE_HALF                       # noqa: E402
from scripts.conformal.common import CONFORMAL_DIR, save_result, swept_signed_distance  # noqa: E402

_CUBE_XY = slice(18, 20)


def heldout_episodes(n_total: int, train_fraction: float, seed: int) -> list[int]:
    """Episodes the WM did NOT train on, reproduced exactly as traj_dset.py splits.

    Must use a torch Generator: torch.randperm(seed=42) and np.random.RandomState(42) give
    different orderings, so a numpy stand-in would evaluate on episodes the model trained on
    while looking perfectly plausible.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return sorted(perm[int(train_fraction * n_total):])          # train takes the LEADING slice


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/isaaclab_stroke_5k_shift025")
    ap.add_argument("--model_name", default="wm_5k")
    ap.add_argument("--epochs", type=int, nargs="+", default=[15, 20, 25, 30, 40, 50])
    ap.add_argument("--probe_path", default="probes/weights/cube_pos_encoded.pth")
    ap.add_argument("--limit", type=int, default=None, help="cap held-out episodes (quick look)")
    ap.add_argument("--device", default="cuda:0",
                    help="index WITHIN CUDA_VISIBLE_DEVICES; pin the card with that env var")
    ap.add_argument("--context_frames", type=int, default=1, help="1 = match deployment")
    ap.add_argument("--train_fraction", type=float, default=0.9)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.02, 0.03, 0.05])
    ap.add_argument("--cell", type=int, default=4)
    ap.add_argument("--out", default="epoch_sweep")
    args = ap.parse_args()

    import hydra  # noqa: F401  (imported for side effects in plan.load_model)
    from omegaconf import OmegaConf

    from datasets.isaaclab_grid_dset import IsaacLabSingleDataset as _DS
    from plan import load_model
    from probes.registry import ProbeRegistry

    dev = torch.device(args.device)
    root = _REPO / args.dataset
    ck_dir = _REPO / "outputs" / args.model_name / "checkpoints"
    cfg = OmegaConf.load(_REPO / "outputs" / args.model_name / "hydra.yaml")
    cfg.has_decoder = False

    A = torch.load(root / "actions.pth").float()
    S = torch.load(root / "states.pth").float()
    P = torch.load(root / "proprio.pth").float()
    L = torch.load(root / "seq_lengths.pth")
    am, asd = _DS._mean_std(A, L)
    pm, psd = _DS._mean_std(P, L)

    eps = heldout_episodes(A.shape[0], args.train_fraction, args.split_seed)
    if args.limit:
        eps = eps[:args.limit]
    NH = int(args.context_frames)
    print(f"[epoch_sweep] {len(eps)} HELD-OUT episodes (seed {args.split_seed}, "
          f"train_fraction {args.train_fraction}), context={NH} frame(s)")

    reg = ProbeRegistry(device=dev)
    reg.set_probe("cube_position", args.probe_path)
    probe = reg["cube_position"]

    # ---- encode every held-out episode ONCE with the first checkpoint's (frozen) encoder ----
    base_model = load_model(ck_dir / f"model_{args.epochs[0]}.pth", cfg, 1, dev)
    base_model.eval()
    enc_ref = next(base_model.encoder.parameters()).detach().clone()

    cache = {}
    for i, e in enumerate(eps):
        T = int(L[e])
        vid = (torch.load(root / "obses" / f"episode_{e:05d}.pth")[:T].float() / 255.0)
        vid = (vid.permute(0, 3, 1, 2) * 2.0 - 1.0).to(dev)
        prop = ((P[e, :T] - pm) / psd).to(dev)
        cache[e] = (base_model.encode_obs({"visual": vid.unsqueeze(0),
                                           "proprio": prop.unsqueeze(0)})["visual"].cpu(),
                    prop.cpu(), T)
        if (i + 1) % 100 == 0:
            print(f"  encoded {i+1}/{len(eps)} episodes", flush=True)
    del base_model
    torch.cuda.empty_cache()

    rows = []
    for ep in args.epochs:
        ck = ck_dir / f"model_{ep}.pth"
        if not ck.exists():
            print(f"[epoch_sweep] skip epoch {ep}: {ck} missing")
            continue
        model = load_model(ck, cfg, 1, dev)
        model.eval()
        # The cached latents are only reusable if the encoder really is frozen. Verify, don't assume.
        if not torch.allclose(next(model.encoder.parameters()).detach().cpu(), enc_ref.cpu()):
            raise SystemExit(f"epoch {ep}: encoder differs from epoch {args.epochs[0]} -- the "
                             f"encode-once optimisation is invalid for this run.")

        pe_all, ge_all, ps_all, gs_all = [], [], [], []
        for e in eps:
            vc_cpu, prop_cpu, T = cache[e]
            vc = vc_cpu.to(dev)
            prop = prop_cpu.to(dev)
            act = ((A[e, :T] - am) / asd).to(dev)
            ts = list(range(T - 1))
            idx = torch.tensor([[max(0, t - NH + 1 + k) for k in range(NH)] for t in ts], device=dev)
            obs = {"visual": vc[0][idx],                      # (B, NH, P, D)
                   "visual_cached": vc[0][idx],
                   "proprio": prop[idx]}                      # (B, NH, 18)
            z = model.rollout(obs_0=obs, act=act[idx])[0]["visual"]   # (B, NH+1, P, D)
            pe_all.append(probe(z[:, -1]).cpu().numpy())              # 1-step prediction
            ps_all.append(probe(z[:, NH - 1]).cpu().numpy())          # encoded current frame
            ge_all.append(S[e, 1:T, _CUBE_XY].numpy())
            gs_all.append(S[e, :T - 1, _CUBE_XY].numpy())
        pe = np.concatenate(pe_all); ge = np.concatenate(ge_all)
        pspred = np.concatenate(ps_all); gs = np.concatenate(gs_all)

        ee = np.linalg.norm(pe - ge, axis=1)
        es = np.linalg.norm(pspred - gs, axis=1)
        db = swept_signed_distance(pspred, pe, args.cell, CUBE_HALF)
        dt = swept_signed_distance(gs, ge, args.cell, CUBE_HALF)
        leaks = {f"{d:.2f}": float((dt[db >= d] < 0).mean() * 100) for d in args.deltas}

        rows.append({"epoch": ep, "n": int(ee.size),
                     "end_mean_cm": float(ee.mean() * 100), "end_p50_cm": float(np.median(ee) * 100),
                     "end_p90_cm": float(np.quantile(ee, .9) * 100),
                     "end_p99_cm": float(np.quantile(ee, .99) * 100),
                     "start_mean_cm": float(es.mean() * 100),
                     "leak_pct_by_delta": leaks})
        lk = "  ".join(f"d{k}={v:5.2f}%" for k, v in leaks.items())
        print(f"  epoch {ep:3d}: end mean={ee.mean()*100:5.2f} p90={np.quantile(ee,.9)*100:5.2f} "
              f"p99={np.quantile(ee,.99)*100:6.2f} | start mean={es.mean()*100:4.2f} | {lk}", flush=True)
        del model
        torch.cuda.empty_cache()

    best = min(rows, key=lambda r: r["end_p90_cm"]) if rows else None
    if best:
        print(f"\n[epoch_sweep] lowest held-out p90: epoch {best['epoch']} "
              f"({best['end_p90_cm']:.2f} cm). Deployed = e30. A small gap means e30 is fine; "
              f"only a large one would justify re-running results/final.")
    save_result(args.out, {"method": "wm_epoch_sweep_heldout", "model_name": args.model_name,
                           "dataset": args.dataset, "probe_path": args.probe_path,
                           "context_frames": NH, "n_heldout_episodes": len(eps),
                           "split": {"seed": args.split_seed, "train_fraction": args.train_fraction},
                           "cell": args.cell, "rows": rows})


if __name__ == "__main__":
    main()
