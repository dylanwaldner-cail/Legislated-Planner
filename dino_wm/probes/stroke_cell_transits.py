"""Detect strokes whose cube TRANSITS a cell mid-stroke without being there at the boundaries.

The dataset is boundary-only (one frame per stroke), so the probe/WM see the cube only at
stroke start/end. A stroke can push the cube THROUGH a cell (enter + exit within the stroke)
with neither endpoint showing it -> invisible to the boundary representation. For normative
laws ("never enter region X") this is a blind spot.

This scans consecutive recorded cube positions c0 -> c1, sweeps the cube footprint (half
CUBE_HALF) along the straight segment between them (a push moves ~straight), and flags cells
the footprint crosses that are NOT occupied at either endpoint. No WM/encoding -- states only.
Reports how often it happens (i.e. how big the blind spot is) + examples.

The `swept_cells` primitive is also the basis for the fix at plan time: ground laws on the
SWEPT cells of a candidate stroke (start cube from the probe on obs, end cube from the probe on
the WM-predicted latent), not just the endpoint occupancy.

    python probes/stroke_cell_transits.py --data_dir data/isaaclab_stroke_1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import image as mpimg

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from probes.probe_cube_position import gm, CUBE_OFF
from probes.probe_cube_cells import cube_cell_occupancy, swept_cells, CUBE_HALF  # swept_cells now lives with the cell geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--cube_half", type=float, default=CUBE_HALF)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--out", default="probes/stroke_transits", help="dir to save example start/end frame PNGs")
    args = ap.parse_args()

    p = Path(args.data_dir)
    states = torch.load(p / "states.pth").float().numpy()
    seq = torch.load(p / "seq_lengths.pth").numpy().astype(np.int64)

    n_strokes = n_with_transit = 0
    per_stroke_extra = []          # # invisible-transit cells per stroke
    examples = []
    for e in range(states.shape[0]):
        T = int(seq[e])
        xy = states[e, :T, CUBE_OFF:CUBE_OFF + 2]
        occ = cube_cell_occupancy(xy, args.cube_half).astype(bool)          # (T,9) endpoint occupancy
        for t in range(T - 1):
            c0, c1 = xy[t], xy[t + 1]
            n_strokes += 1
            sw = swept_cells(c0, c1, args.cube_half)
            extra = sw & ~occ[t] & ~occ[t + 1]                              # transited but at neither boundary
            k = int(extra.sum())
            per_stroke_extra.append(k)
            if k > 0:
                n_with_transit += 1
                if len(examples) < args.examples:
                    examples.append((e, t, c0.copy(), c1.copy(),
                                     np.where(occ[t])[0].tolist(), np.where(occ[t + 1])[0].tolist(),
                                     np.where(extra)[0].tolist()))

    arr = np.array(per_stroke_extra)
    print(f"[transit scan] {n_strokes} strokes | {n_with_transit} ({100 * n_with_transit / max(n_strokes,1):.1f}%) "
          f"cross >=1 cell that's invisible at both endpoints")
    print(f"  invisible-transit cells per stroke: mean {arr.mean():.3f}  max {int(arr.max())}  "
          f"hist {[int((arr == k).sum()) for k in range(int(arr.max()) + 1)]} (k=0,1,2,...)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"\n[examples] (cells = row*3+col)  saving start/end frames -> {out}")
    for i, (e, t, c0, c1, o0, o1, ex) in enumerate(examples):
        print(f"  ex{i} ep {e} stroke {t}: cube ({c0[0]:+.3f},{c0[1]:+.3f}) -> ({c1[0]:+.3f},{c1[1]:+.3f}) | "
              f"start cells {o0} end cells {o1} -> PASSED THROUGH {ex}")
        vid = torch.load(p / "obses" / f"episode_{e:05d}.pth")
        tag = "-".join(map(str, ex)) or "none"
        mpimg.imsave(out / f"ex{i}_ep{e:05d}_t{t:03d}_start_thru{tag}.png", vid[t].numpy().astype(np.uint8))
        mpimg.imsave(out / f"ex{i}_ep{e:05d}_t{t + 1:03d}_end_thru{tag}.png", vid[t + 1].numpy().astype(np.uint8))


if __name__ == "__main__":
    main()
