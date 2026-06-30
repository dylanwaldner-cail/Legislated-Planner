"""Compare WM latent-prediction drift: OUR cube WM vs a GRANULAR WM (same pipeline).

The question: is our cube WM's autoregressive drift (1-step ok, multi-step bad) ABNORMAL,
or is it just how DINO-WM behaves on contact tasks? Granular is a known-plannable task (the
paper plans it via MPC), so a granular WM trained with OUR pipeline is the fair reference.

Metric (comparable across datasets via per-WM normalization): roll the WM h steps open-loop
on REAL actions, compare predicted visual latent to the truly-encoded latent, normalize by
the latent's natural spread d_far (mean L2 between random frame pairs):
    norm_drift(h) = ||pred(h) - enc(h)|| / d_far     (~0 perfect; ~1 as far off as random)
If granular drifts like ours -> autoregressive drift is normal -> greedy 1-step MPC is the
right call. If granular holds up far better -> our cube WM/data is the weak link.

No sim. Run with the host conda env (or container). May compete for GPU with training -> use
--device cpu if it OOMs.
"""
from __future__ import annotations
import sys
import numpy as np
import torch
from einops import rearrange
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from scripts.wm_cube_pred_check import load_wm


@torch.no_grad()
def enc_visual(wm, vis, device):
    vis = vis.to(device)
    b = vis.shape[0]
    x = rearrange(vis, "b t c h w -> (b t) c h w")
    x = wm.encoder_transform(x)
    emb = wm.encoder.forward(x)
    return rearrange(emb, "(b t) p d -> b t p d", b=b)


def _trans(frames):                              # (n,H,W,3) -> (n,3,224,224) in [-1,1]
    return frames.float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0


@torch.no_grad()
def drift_curve(name, model_dir, epoch, obs_path_fn, n_eps, actions, proprio,
                fs, num_hist, device, H=5, n_windows=80, seed=0):
    wm, _ = load_wm(model_dir, epoch, device)
    raw = actions.shape[-1]; T = actions.shape[1]; n_model = T // fs
    a_mean = actions.reshape(-1, raw).mean(0); a_std = actions.reshape(-1, raw).std(0) + 1e-6
    pd = proprio.shape[-1]
    p_mean = proprio.reshape(-1, pd).mean(0); p_std = proprio.reshape(-1, pd).std(0) + 1e-6
    rng = np.random.RandomState(seed)

    def obs_frames(e, ms):
        vid = torch.load(obs_path_fn(e), map_location="cpu")        # (T,H,W,3)
        return _trans(vid[[m * fs for m in ms]])

    def m_act(e, m):
        return ((actions[e, m * fs:(m + 1) * fs] - a_mean) / a_std).reshape(-1)

    far = []
    for _ in range(48):
        e1, m1 = rng.randint(n_eps), rng.randint(n_model)
        e2, m2 = rng.randint(n_eps), rng.randint(n_model)
        z1 = enc_visual(wm, obs_frames(e1, [m1])[None], device).flatten()
        z2 = enc_visual(wm, obs_frames(e2, [m2])[None], device).flatten()
        far.append(float(torch.norm(z1 - z2)))
    d_far = float(np.mean(far))

    max_m0 = n_model - (num_hist + H)
    drift = {h: [] for h in range(1, H + 1)}
    for _ in range(n_windows):
        e = rng.randint(n_eps); m0 = rng.randint(max(1, max_m0 + 1))
        init_ms = list(range(m0, m0 + num_hist))
        vis0 = obs_frames(e, init_ms)[None].to(device)
        pro0 = torch.tensor(np.stack([(proprio[e, m * fs] - p_mean) / p_std for m in init_ms]))[None].float().to(device)
        init_a = [m_act(e, m) for m in init_ms]
        fut = [m_act(e, m0 + num_hist - 1 + k) for k in range(H)]
        for h in range(1, H + 1):
            act = torch.tensor(np.stack(init_a + fut[:h - 1]))[None].float().to(device)
            z, _ = wm.rollout(obs_0={"visual": vis0, "proprio": pro0}, act=act)
            pred = z["visual"][:, num_hist + h - 1].flatten()
            true_enc = enc_visual(wm, obs_frames(e, [m0 + num_hist - 1 + h])[None], device).flatten()
            drift[h].append(float(torch.norm(pred - true_enc)))

    print(f"\n=== {name}  (fs={fs}, num_hist={num_hist}, d_far={d_far:.1f}) ===")
    for h in range(1, H + 1):
        a = np.array(drift[h])
        print(f"  h={h}: abs {a.mean():8.1f}   norm_drift {a.mean() / d_far:.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--granular_run", default="outputs/2026-06-25/19-40-40")
    ap.add_argument("--granular_epoch", default="20")
    ap.add_argument("--n_windows", type=int, default=80)
    args = ap.parse_args()
    dev = args.device

    drift_curve(
        "OURS cube (1500ep)", "outputs/2026-06-25/16-46-57", "20",
        lambda e: f"data/isaaclab_stroke_1500/obses/episode_{e:05d}.pth", 1500,
        torch.load("data/isaaclab_stroke_1500/actions.pth").float().numpy(),
        torch.load("data/isaaclab_stroke_1500/proprio.pth").float().numpy(),
        fs=1, num_hist=3, device=dev, n_windows=args.n_windows,
    )
    drift_curve(
        "GRANULAR (our pipeline)", args.granular_run, args.granular_epoch,
        lambda e: f"checkpoints/deformable/granular/{e:06d}/obses.pth", 1000,
        torch.load("checkpoints/deformable/granular/actions.pth").float().numpy(),
        np.zeros((1000, 20, 1), dtype=np.float32),   # dummy proprio (loader uses zeros)
        fs=1, num_hist=3, device=dev, n_windows=args.n_windows,
    )


if __name__ == "__main__":
    main()
