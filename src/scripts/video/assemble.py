"""Assemble the built beats into one mp4 and drop it on the project page.

Beats are listed in order; each module exposes build() -> list[PIL.Image]. Missing beats are
skipped with a warning so the preview always reflects exactly what exists today.
"""
from __future__ import annotations

import importlib
import sys

sys.path.insert(0, "/newdata2/dylantw/Legislated-Planner/src")
from scripts.video import common as c  # noqa: E402

ORDER = [
    ("beat01_stack",    "the stack assembles"),
    ("beat06_ddl",      "the reasoner"),
    ("beat08_prune",    "pruning the search"),
    ("beat11_ontology", "the ontological gap"),
    ("beat12_contrast",  "what the law buys"),
    ("beat13_closing",   "closing"),
]
XFADE = 14          # frames of crossfade between beats

OUT = "/newdata2/dylantw/Legislated-Planner/docs/assets/video/intro_preview.mp4"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    frames: list = []
    built = []
    for mod, label in ORDER:
        try:
            m = importlib.import_module(f"scripts.video.{mod}")
        except Exception as e:                                   # noqa: BLE001
            print(f"[assemble] SKIP {mod}: {type(e).__name__}: {e}")
            continue
        fr = m.build()
        print(f"[assemble] {mod:<16} {len(fr):>4} frames  {len(fr)/c.FPS:5.1f}s  ({label})")
        frames = c.crossfade(frames, fr, XFADE) if frames else fr
        built.append(label)
    if not frames:
        sys.exit("[assemble] nothing built")
    c.write_mp4(frames, out)
    print(f"[assemble] beats included: {', '.join(built)}")
    print(f"[assemble] total {len(frames)/c.FPS:.1f}s of a 90-120s target")


if __name__ == "__main__":
    main()
