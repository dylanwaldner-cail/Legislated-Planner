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


def stroke_boundaries(frames, expected: int):
    """Frame indices where a stroke ends, found from the footage.

    At every stroke boundary the harness snaps the arm back to the home pose
    (grid_wrapper_single.execute_stroke), which produces a far larger frame-to-frame change than
    anything mid-push. So the boundaries are the `expected` strongest, well-separated peaks in the
    inter-frame difference. The count is checked against the batch's own stroke count and reported.
    """
    half = frames[0].width // 2
    A = np.stack([np.asarray(f.crop((0, 0, half, f.height)), dtype=np.int16) for f in frames])
    d = np.abs(np.diff(A, axis=0)).mean(axis=(1, 2, 3))
    order = np.argsort(d)[::-1]
    picked: list[int] = []
    for i in order:
        if len(picked) >= expected:
            break
        if all(abs(int(i) - p) > 15 for p in picked):      # suppress the cluster around each peak
            picked.append(int(i))
    return sorted(picked)


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
    # Default ON: the goal panel is a STATIC shot of the cube already at the goal, shown from frame
    # zero. Watched as a loop it reads as "it finished ages ago" and competes with the real action.
    ap.add_argument("--executed-only", action="store_true", default=True,
                    help="keep only the LEFT (executed) panel; drop the static goal panel (default)")
    ap.add_argument("--with-goal", dest="executed_only", action="store_false",
                    help="keep the [executed | goal] side-by-side layout")
    ap.add_argument("--metrics", default=None,
                    help="eval_metrics.json of the run; with --ep, cuts at THIS episode's own goal "
                         "instead of the batch's longest episode")
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--tail", type=float, default=0.4, help="seconds of stillness to keep")
    a = ap.parse_args()

    frames = c.read_mp4(a.src)
    n0 = len(frames)

    # The clip's length is set by the LONGEST episode in the batch -- a shorter episode keeps being
    # stepped with padded actions after it has already reached the goal, so it must be cut at its
    # own boundary, not where motion happens to stop.
    if a.metrics and a.ep is not None:
        import json
        ns = np.asarray(json.load(open(a.metrics))["n_steps"])
        mine, longest = int(ns[a.ep]), int(ns.max())
        if mine < longest:
            b = stroke_boundaries(frames, longest - 1)
            print(f"[clip] batch runs {longest} strokes, ep{a.ep} runs {mine}; "
                  f"found {len(b)} boundaries {b[:6]}{'...' if len(b) > 6 else ''}")
            cut = b[mine - 1] + int(a.tail / SIM_DT)
            frames = frames[:min(n0, cut)]
            print(f"[clip] cut at its own goal: frame {cut} of {n0}")
        else:
            print(f"[clip] ep{a.ep} IS the longest episode ({mine} strokes); no goal cut needed")

    # The two cuts COMPOSE: the goal cut removes padded strokes after this episode finished, and the
    # stillness cut removes the settle frames at the end of the last real stroke. Applying only one
    # leaves either padded pushes or a frozen tail.
    if a.trim:
        last = last_motion_frame(frames, a.motion_eps)
        keep = min(n0, last + 1 + int(a.tail / SIM_DT))
        print(f"[clip] motion ends at frame {last}/{n0-1} "
              f"({last * SIM_DT:.1f}s sim) -> keeping {keep} frames")
        frames = frames[:keep]

    if a.executed_only:
        w, h = frames[0].size
        frames = [f.crop((0, 0, w // 2, h)) for f in frames]
        print(f"[clip] dropped the goal panel -> {frames[0].size[0]}x{frames[0].size[1]}")

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
