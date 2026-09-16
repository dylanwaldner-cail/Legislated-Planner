"""Trim a planner-produced episode mp4 to the part where something happens, and play it at a chosen
multiple of REAL time.

TIMEBASE. The Single env runs sim.dt=0.01 with decimation=4 (dinowm_grid_env_cfg.DinoWMGridSingleEnvCfg),
so one captured substep frame is 0.04 s of simulated time -- real time is 25 fps. `--speed 2` means
twice real time, and the output fps is derived from that, not guessed. plan.py muxes these at 8 fps,
i.e. ~3x SLOWER than real, which is why the raw clips feel interminable.

TRIMMING. `full_video=true` keeps rendering after the goal is reached, so the tail is a parked arm
and a motionless cube. Rather than guess a cut point from stroke counts (substeps per stroke vary),
the cut is measured from the footage: the last frame whose executed panel differs from its
predecessor by more than --motion-eps, plus a short tail. The detected cut is printed so it can be
sanity-checked against the episode's recorded n_steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.video import common as c  # noqa: E402

SIM_DT = 0.04                      # s per captured substep frame (0.01 * decimation 4)
REAL_FPS = 1.0 / SIM_DT            # 25


def last_motion_frame(frames, eps: float) -> int:
    """Index of the last frame where the EXECUTED panel (left half) still changes."""
    half = frames[0].width // 2
    prev = np.asarray(frames[0].crop((0, 0, half, frames[0].height)), dtype=np.int16)
    last = 0
    for i, fr in enumerate(frames[1:], start=1):
        cur = np.asarray(fr.crop((0, 0, half, fr.height)), dtype=np.int16)
        if float(np.abs(cur - prev).mean()) > eps:
            last = i
        prev = cur
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--speed", type=float, default=2.0, help="multiple of real time")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame (1 = all, smoothest)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--trim", action="store_true", default=True,
                    help="cut the motionless tail after the goal is reached")
    ap.add_argument("--no-trim", dest="trim", action="store_false")
    ap.add_argument("--motion-eps", type=float, default=0.35)
    ap.add_argument("--tail", type=float, default=0.4, help="seconds of stillness to keep")
    a = ap.parse_args()

    frames = c.read_mp4(a.src)
    n0 = len(frames)
    if a.trim:
        last = last_motion_frame(frames, a.motion_eps)
        keep = min(n0, last + 1 + int(a.tail / SIM_DT))
        print(f"[clip] motion ends at frame {last}/{n0-1} "
              f"({last * SIM_DT:.1f}s sim) -> keeping {keep} frames")
        frames = frames[:keep]

    frames = frames[::a.stride]
    fps = max(1, int(round(REAL_FPS * a.speed / a.stride)))
    if a.scale > 1:
        w, h = frames[0].size
        frames = [f.resize((w * a.scale, h * a.scale)) for f in frames]
    print(f"[clip] {Path(a.src).name}: {n0} -> {len(frames)} frames @ {fps}fps "
          f"= {len(frames)/fps:.1f}s  ({a.speed:g}x real time)")
    c.write_mp4(frames, a.dst, fps=fps, crf=20)


if __name__ == "__main__":
    main()
