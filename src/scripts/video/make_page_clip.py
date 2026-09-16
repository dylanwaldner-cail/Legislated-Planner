"""Re-time a planner-produced episode mp4 into a short loop for the project page.

plan.py's `video=true` clips capture EVERY internal sim substep and are muxed at 8 fps, so a 4-stroke
episode lands at ~860 frames / 108 s -- unwatchable on a page. This subsamples every Nth substep and
remuxes at 30 fps, which both speeds it up and shrinks it, exactly like the --speed flag on
render_episode_video.py. Frame content is untouched: no interpolation, no re-encode of pixels beyond
the x264 pass, so nothing is invented.

Layout note: the source is [executed | goal] side by side (evaluator._save_executed_video), so the
goal panel travels with the clip and the viewer can see what the task was.

    python scripts/video/make_page_clip.py IN.mp4 OUT.mp4 [--stride 3] [--fps 30]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--stride", type=int, default=4, help="keep every Nth substep frame")
    ap.add_argument("--fps", type=int, default=60,
                    help="playback rate. Speed is set HERE rather than by dropping more frames -- "
                         "a higher fps at a low stride keeps the motion smooth instead of choppy.")
    ap.add_argument("--scale", type=int, default=2, help="integer upscale for crisper playback")
    a = ap.parse_args()

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,nb_frames",
                            "-of", "default=noprint_wrappers=1:nokey=1", a.src],
                           capture_output=True, text=True).stdout.split()
    w, h, n = int(probe[0]), int(probe[1]), int(probe[2])
    kept = n // a.stride
    print(f"[clip] {Path(a.src).name}: {n} frames {w}x{h} -> keep every {a.stride} = {kept} "
          f"@ {a.fps}fps = {kept / a.fps:.1f}s")

    vf = (f"select='not(mod(n\\,{a.stride}))',setpts=N/{a.fps}/TB,"
          f"scale={w * a.scale}:{h * a.scale}:flags=lanczos")
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.src, "-vf", vf,
                        "-r", str(a.fps), "-an", "-c:v", "libx264", "-preset", "slow",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.dst])
    if r.returncode != 0:
        sys.exit("[clip] ffmpeg failed")
    print(f"[clip] wrote {a.dst}  {Path(a.dst).stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
