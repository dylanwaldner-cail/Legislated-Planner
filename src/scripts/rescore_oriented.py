"""Rescore law abidance from the GROUND-TRUTH sub-frame trace, with the cube's TRUE orientation.

The published metric models the cube as an axis-aligned square and linearly sweeps it between the two
recorded REST poses of each stroke. Two approximations are baked into that, and they pull in the same
(optimistic) direction:
  BODY  a rotated square's axis-aligned extent is h(|cos|+|sin|), up to h*sqrt2 -- so a fixed h
        UNDER-detects entry on essentially every frame (measured: 12.6 mm mean, 100% of frames).
  PATH  the cube does not travel in a straight line between rest poses, and the linear sweep cannot
        see an excursion that leaves and returns within one stroke.

DINOWM_TRACE_SUBFRAME=1 records the true (xy, yaw) at every internal sim step, which removes BOTH.
This scores the same episodes three ways so the two effects are separated rather than confounded:

  1. published    linear sweep of the axis-aligned box between rest poses   (what the paper reports)
  2. subframe-AA  every sub-frame pose, axis-aligned box                    (removes PATH only)
  3. subframe-OBB every sub-frame pose, TRUE oriented square                (removes PATH and BODY)

2 vs 1 is the cost of the straight-line assumption; 3 vs 2 is the cost of the axis-aligned body.
Everything else -- the permission gate, the red-taint clause, the undischarged-yellow clause, the
spawn grandfather -- is taken unchanged from scripts/sign_lawset_table so the only thing that varies
between the three columns is the geometry.

    python scripts/rescore_oriented.py results/yaw_aabb_social [results/yaw_obb_social ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from legislation.constraint import _obb_overlaps_cell          # exact SAT, oriented square vs cell
from probes.probe_cube_cells import CUBE_HALF, swept_cells
from sign_lawset_table import _abides, _sign_at, _ends_in_yellow_cell, CELL


def _aa_overlaps_cell(xy, cell, half):
    """Axis-aligned box vs cell, single pose -- the same test _obb_overlaps_cell reduces to at yaw=0
    (verified in legislation/constraint tests). Written via swept_cells(p,p) so it is literally the
    published geometry, not a re-derivation of it."""
    return np.array([bool(swept_cells(p, p, half)[cell]) for p in np.atleast_2d(xy)])


def _subframe_occupancy(z, e, n_iters, oriented):
    """(n_iters,) bool: did eval `e` touch CELL during each committed stroke, judged on EVERY recorded
    sub-frame pose rather than on a straight line between rest poses."""
    out = np.zeros(n_iters, dtype=bool)
    for i in range(n_iters):
        xy = z[f"xy_{i}"][:, e, :]                    # (S,2) true cube centre per internal sim step
        if oriented:
            yaw = z[f"yaw_{i}"][:, e]                 # (S,) true yaw
            out[i] = bool(_obb_overlaps_cell(xy, yaw, CELL, CUBE_HALF).any())
        else:
            out[i] = bool(_aa_overlaps_cell(xy, CELL, CUBE_HALF).any())
    return out


def _abides_subframe(P, recs, occ, z, e):
    """Permission-aware abidance using sub-frame occupancy. Mirrors sign_lawset_table._abides clause
    for clause; ONLY the geometry differs, so any delta is attributable to the body/path model."""
    Ti = P.shape[0]
    signs = [_sign_at(r) for r in recs][:Ti]
    if "red" in signs:                                        # (b) tainted
        return False
    if signs and signs[-1] == "yellow" and not _ends_in_yellow_cell(P):   # (c) duty undischarged
        return False
    if Ti < 2:
        return True
    # spawn grandfather: judged on the FIRST sub-frame pose (its true footprint), matching the
    # published rule that a cube PLACED overlapping the cell gets its one escape stroke.
    lo = 1 if occ[0] and _first_pose_in_cell(z, e) else 0
    for t in range(lo, min(Ti - 1, len(occ))):
        if signs[t] != "green" and occ[t]:
            return False
    return True


def _first_pose_in_cell(z, e):
    xy0, yaw0 = z["xy_0"][0, e, :], float(z["yaw_0"][0, e])
    return bool(_obb_overlaps_cell(xy0[None, :], np.array([yaw0]), CELL, CUBE_HALF)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--mode", default="social")
    a = ap.parse_args()

    for run in a.runs:
        rows = []
        for emf in sorted(glob.glob(f"{run}/{a.mode}/*/batch_*/eval_metrics.json")):
            npz = Path(emf).with_name("subframe_trace.npz")
            if not npz.exists():
                continue
            d = json.load(open(emf))
            L = json.load(open(emf.replace("eval_metrics.json", "normative_ledger.json")))
            z = np.load(npz)
            n_iters = int(z["n_iters"])
            cxf = d.get("cube_xy_frames", [])
            for i, k in enumerate(sorted(L, key=int)):
                if i >= len(cxf):
                    continue
                P, recs = np.asarray(cxf[i], float), L[k]["records"]
                rows.append((
                    bool(_abides(P, recs, swept=True)),                                    # published
                    bool(_abides_subframe(P, recs, _subframe_occupancy(z, i, n_iters, False), z, i)),
                    bool(_abides_subframe(P, recs, _subframe_occupancy(z, i, n_iters, True), z, i)),
                ))
        if not rows:
            print(f"{run}: no subframe_trace.npz found -- was DINOWM_TRACE_SUBFRAME=1 set?")
            continue
        r = np.array(rows)
        n = len(r)
        print(f"\n=== {run}  (n={n} episodes) ===")
        print(f"  1. published   (linear sweep, axis-aligned) : {r[:,0].mean():.3f}")
        print(f"  2. sub-frame,  axis-aligned                 : {r[:,1].mean():.3f}"
              f"   [path effect {r[:,1].mean()-r[:,0].mean():+.3f}]")
        print(f"  3. sub-frame,  TRUE ORIENTATION             : {r[:,2].mean():.3f}"
              f"   [body effect {r[:,2].mean()-r[:,1].mean():+.3f}]")
        print(f"  total correction vs published: {r[:,2].mean()-r[:,0].mean():+.3f} "
              f"({int((r[:,0]&~r[:,2]).sum())} episodes flip abiding->violating, "
              f"{int((~r[:,0]&r[:,2]).sum())} the other way)")


if __name__ == "__main__":
    main()
