"""Find frames where the per-cell occupancy probe is WRONG (pred != GT), save + print them.

Scans frames, runs the probe, and surfaces the exact-match FAILURES — the genuine ~10% errors.
For each mismatch it reports the offending cell(s): gt vs pred and the sigmoid prob, tagged
'miss' (truly-occupied cell that fell below 0.5) or 'false+' (empty cell that rose above 0.5).
Expected pattern: a single marginally-clipped corner cell sitting near the 0.5 boundary.

Saves the worst mismatch frames (PNG) + GT/pred/probs grids to --out, plus verify_mismatch.json.

    python probes/verify_cube_cells.py --probe probes/probe_cube_cells.pth --n 8
    python probes/verify_cube_cells.py --min_cells 4 --n 8     # only corner (4-cell) frames
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import image as mpimg
from torchvision import transforms

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probes.probe_cube_position import gm, CUBE_OFF             # grid metadata + cube state offset
from probes.probe_cube_cells import cube_cell_occupancy, CUBE_HALF
from probes.registry import Probe
from models.dino import DinoV2Encoder


@torch.no_grad()
def episode_probs(encoder, probe, frames_uint8, resize, enc_batch, dev):
    """(T,H,W,3) uint8 frames -> (T, 9) sigmoid occupancy probs (matches training preprocessing)."""
    x = torch.from_numpy(frames_uint8).float().permute(0, 3, 1, 2) / 255.0   # (T,3,H,W)
    x = resize(x) * 2.0 - 1.0
    outs = []
    for i in range(0, x.shape[0], enc_batch):
        toks = encoder(x[i:i + enc_batch].to(dev))                           # (b,P,D)
        outs.append(probe(toks).cpu())                                       # (b,9) sigmoid
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--probe", default="probes/probe_cube_cells.pth")
    ap.add_argument("--out", default="probes/cell_probe_mismatch")
    ap.add_argument("--n", type=int, default=8, help="how many mismatch frames to SAVE (worst first)")
    ap.add_argument("--scan_episodes", type=int, default=60, help="episodes to scan for mismatches")
    ap.add_argument("--min_cells", type=int, default=0, help="only consider frames with >= this many GT cells")
    ap.add_argument("--enc_res", type=int, default=196)
    ap.add_argument("--enc_batch", type=int, default=64)
    ap.add_argument("--cube_half", type=float, default=CUBE_HALF)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    p = Path(args.data_dir)
    states = torch.load(p / "states.pth").float().numpy()                    # (E,T,31)
    seq = torch.load(p / "seq_lengths.pth").numpy().astype(np.int64)

    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(dev).eval()
    for prm in encoder.parameters():
        prm.requires_grad_(False)
    probe = Probe("cube_cells", args.probe, device=dev)
    resize = transforms.Resize(args.enc_res)

    rng = np.random.RandomState(args.seed)
    ep_order = rng.permutation(states.shape[0])[:args.scan_episodes]
    mism, n_frames = [], 0
    for e in ep_order:
        e = int(e); T = int(seq[e])
        xy = states[e, :T, CUBE_OFF:CUBE_OFF + 2]
        gt = cube_cell_occupancy(xy, args.cube_half).astype(int)             # (T,9)
        vid = torch.load(p / "obses" / f"episode_{e:05d}.pth")[:T].numpy().astype(np.uint8)
        probs = episode_probs(encoder, probe, vid, resize, args.enc_batch, dev)   # (T,9)
        pred = (probs > 0.5).astype(int)
        n_frames += T
        for f in range(T):
            if args.min_cells and gt[f].sum() < args.min_cells:
                continue
            wrong = np.where(pred[f] != gt[f])[0]
            if len(wrong) == 0:
                continue
            mism.append({
                "episode": e, "frame": f,
                "cube_xy": [round(float(xy[f, 0]), 4), round(float(xy[f, 1]), 4)],
                "n_cells_gt": int(gt[f].sum()), "n_wrong": int(len(wrong)),
                "wrong_cells": [{"cell": int(c), "gt": int(gt[f, c]), "pred": int(pred[f, c]),
                                 "prob": round(float(probs[f, c]), 3),
                                 "type": "miss" if gt[f, c] == 1 else "false+"} for c in wrong],
                "gt_3x3": gt[f].reshape(3, 3).tolist(),
                "pred_3x3": pred[f].reshape(3, 3).tolist(),
                "probs_3x3": np.round(probs[f], 3).reshape(3, 3).tolist(),
            })

    rate = 100 * len(mism) / max(n_frames, 1)
    print(f"[scan] {n_frames} frames in {len(ep_order)} episodes | {len(mism)} mismatches ({rate:.1f}%)")
    if not mism:
        print("no mismatches found in the scanned frames"); return

    mism.sort(key=lambda m: -m["n_wrong"])                                   # worst (most wrong cells) first
    chosen = mism[:args.n]
    for i, m in enumerate(chosen):
        vid = torch.load(p / "obses" / f"episode_{m['episode']:05d}.pth")
        png = out / f"mismatch{i}_ep{m['episode']:05d}_f{m['frame']:03d}.png"
        mpimg.imsave(png, vid[m["frame"]].numpy().astype(np.uint8))
        m["png"] = png.name
        wc = ", ".join(f"cell{w['cell']}({w['type']}: gt{w['gt']}->pred{w['pred']}, p={w['prob']})"
                       for w in m["wrong_cells"])
        print(f"\n=== mismatch {i}: ep {m['episode']} frame {m['frame']} | gt {m['n_cells_gt']} cells | "
              f"{m['n_wrong']} wrong | cube=({m['cube_xy'][0]:+.3f},{m['cube_xy'][1]:+.3f}) ===  ({png.name})")
        print(f"  wrong: {wc}")
        print(f"  GT   (row*3+col): {m['gt_3x3']}")
        print(f"  pred            : {m['pred_3x3']}")
        print(f"  probs           : {m['probs_3x3']}")

    n_miss = sum(w["type"] == "miss" for m in mism for w in m["wrong_cells"])
    n_fp = sum(w["type"] == "false+" for m in mism for w in m["wrong_cells"])
    print(f"\n[error types over all {len(mism)} mismatches] miss (dropped occupied cell)={n_miss}  "
          f"false+ (spurious cell)={n_fp}")
    with open(out / "verify_mismatch.json", "w") as fp:
        json.dump(chosen, fp, indent=2)
    print(f"[saved] {len(chosen)} mismatch frames + verify_mismatch.json -> {out}")


if __name__ == "__main__":
    main()
