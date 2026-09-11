#!/usr/bin/env python
"""Precompute + cache the frozen DINO visual latents for a stroke dataset.

DINO is frozen, so it produces the SAME patch tokens for a frame every epoch -- re-encoding
each epoch is the main per-epoch cost. This encodes every frame ONCE, replicating the WM's
visual path EXACTLY (dataset transform -> encoder_transform -> encoder), and saves per-episode
latents to <data_dir>/dino_latents_<enc>/episode_NNNNN.pth  (fp16, (T,P,D)).

--verify compares the cache to the REAL model.encode_obs (with real proprio) on a few episodes;
a silently-wrong cache would poison training, so max|cache - encode_obs| must be ~0.

Encoder-BOUND: valid only for the encoder it was built with (name is in the dir). Rebuild for
a different backbone (e.g. dinov2_vitb14). Run in the container python (torch + dino hub):
    python scripts/precompute_dino_latents.py --data_dir data/isaaclab_stroke_5k \
        --model_dir outputs/wm_5k --epoch 30            # build
    python scripts/precompute_dino_latents.py --data_dir data/isaaclab_stroke_5k \
        --model_dir outputs/wm_5k --epoch 30 --verify   # check == training
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import provenance
from scripts.wm_cube_pred_check import load_wm          # loads the frozen WM (encoder + encoder_transform)
from datasets.img_transforms import default_transform   # the SAME image transform the dataset applies


@torch.no_grad()
def _encode_visual(wm, transform, imgs, device):
    """(T,H,W,3) uint8 -> (T,P,D) patch tokens, replicating encode_obs's VISUAL path exactly."""
    x = rearrange(imgs.float() / 255.0, "t h w c -> t c h w")   # dataset getitem does this
    x = transform(x).to(device)                                 # dataset transform (-> 224, [0,1])
    x = wm.encoder_transform(x)                                 # encode_obs's transform (resize 196 + Normalize)
    return wm.encoder.forward(x).float().cpu()                  # (T, P, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_dir", default="outputs/wm_5k")
    ap.add_argument("--epoch", default="20")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--verify", action="store_true", help="check cache == real encode_obs")
    ap.add_argument("--verify_n", type=int, default=20,
                    help="with --verify: # episodes to check, spread evenly across the dataset")
    args = ap.parse_args()

    data = Path(args.data_dir)
    seq = torch.load(data / "seq_lengths.pth").numpy().astype(np.int64)
    files = sorted((data / "obses").glob("episode_*.pth"))
    assert len(files) == len(seq), f"{len(files)} obs files vs {len(seq)} seq_lengths"

    wm, _ = load_wm(args.model_dir, args.epoch, args.device)
    enc_name = getattr(getattr(wm, "encoder", None), "name", "enc")
    transform = default_transform(img_size=args.img_size)
    out = data / f"dino_latents_{enc_name}"
    out.mkdir(exist_ok=True)
    print(f"[precompute] WM {args.model_dir}@{args.epoch} enc={enc_name} -> {out}")

    if args.verify:
        proprios = torch.load(data / "proprio.pth").float()      # (E,T,Pd) for a real encode_obs
        n = min(args.verify_n, len(files))
        idxs = sorted(set(np.linspace(0, len(files) - 1, n).astype(int).tolist()))  # spread across the set
        maxdiff, worst = 0.0, -1
        for i in idxs:
            f = files[i]; T = int(seq[i]); imgs = torch.load(f)[:T]
            x = rearrange(imgs.float() / 255.0, "t h w c -> t c h w")
            x = transform(x)
            obs = {"visual": x[None].to(args.device), "proprio": proprios[i:i + 1, :T].to(args.device)}
            z_train = wm.encode_obs(obs)["visual"][0].float().cpu()   # what TRAINING actually computes
            z_cache = torch.load(out / f.name).float()
            d = float((z_train - z_cache).abs().max())
            if d > maxdiff:
                maxdiff, worst = d, i
        print(f"[verify] checked {len(idxs)} episodes spread over {len(files)}  |  "
              f"max|cache - encode_obs| = {maxdiff:.2e} (worst ep {worst})  "
              f"({'OK' if maxdiff < 1e-2 else 'MISMATCH!'})")
        return

    for i, f in enumerate(files):
        T = int(seq[i])
        z = _encode_visual(wm, transform, torch.load(f)[:T], args.device)   # (T,P,D)
        torch.save(z.half(), out / f.name)                                  # fp16 -> halve disk
        if (i + 1) % 100 == 0:
            print(f"[precompute] {i + 1}/{len(files)} episodes")
    print(f"[precompute] DONE: {len(files)} episodes -> {out}")
    provenance.write(out, __file__, args=args, repo=_REPO)


if __name__ == "__main__":
    main()
