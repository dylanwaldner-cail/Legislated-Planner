"""Render Figure 1 as a sequence of CUMULATIVE stages, straight from the paper's TikZ source.

WHY NOT PAN OVER THE PNG. The published figure is a TikZ picture (Jurix_paper/main.tex:129-202)
whose sections are already delimited by `%===========` markers: bands / environment / LAW row /
PERCEPTION row / PLANNING row / closed loop / part labels. Truncating the body at each marker gives
a diagram that BUILDS ITSELF one idea at a time, in the paper's own vector output -- perfectly
registered and resolution-independent, which a pan over a flat raster cannot be.

REGISTRATION. A tikzpicture sizes itself to its content, so a partial stage would have a smaller
bounding box and the stages would jitter. An oversized invisible `\path` rectangle is injected right
after the options block so every stage inherits the SAME bounding box. The script asserts all stages
rasterize to identical pixel dimensions and aborts otherwise -- silent misregistration would be
invisible in the code and obvious in the video.

ImageMagick cannot be used to rasterize (the Ubuntu policy blocks PDF); ghostscript does it directly.

    python scripts/video/build_fig1_stages.py [--dpi 200] [--out DIR]
"""
from __future__ import annotations

import argparse
import functools
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path("/newdata2/dylantw/Legislated-Planner/Jurix_paper")
BODY_FIRST, BODY_LAST = 129, 202          # the tikzpicture inside main.tex
OPTS_END = 24                              # line of the `]` closing the options block (1-based, in body)
BBOX = r"\path (-3.4,-5.6) rectangle (28.4,5.6);   % pinned bbox: keeps every stage the same size"

# VIDEO-ONLY TWEAK (the paper figure is untouched): slide the environment panel further left so the
# gap between it and the legal stack opens up and the connecting arrows read clearly on screen.
# At paper scale the panel nearly abuts the bands, which is fine in print and muddy in motion.
ENV_SHIFT = [(r"(world) at (5.1,0)", r"(world) at (3.2,0)")]

# (name, last body line included). Cumulative: each stage contains everything above it.
STAGES = [
    ("bands",      33),   # the three layer bands + their names
    ("env",        37),   # + the environment panel
    ("law",        44),   # + legal source text -> DDL rule base (classical isomorphism)
    ("perception", 53),   # + world model -> probes -> atoms (GROUNDING gap)
    ("planning",   64),   # + verdict -> constraint -> planner (ONTOLOGICAL gap)
    ("loop",       67),   # + the closed loop back to the world
    ("full",       74),   # + the (a)/(b) part labels
]

PREAMBLE = r"""\documentclass[border=3pt]{standalone}
\usepackage{graphicx}\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,calc,shadows,shadows.blur}
\usepackage{xcolor}\usepackage[scaled]{helvet}\usepackage[T1]{fontenc}
\definecolor{diagLaw}{HTML}{1D6FE0}\definecolor{diagPlan}{HTML}{E63946}
\definecolor{diagEnv}{HTML}{12A150}\definecolor{diagLatent}{HTML}{7C3AED}
\definecolor{diagProbe}{HTML}{E8890C}\definecolor{gapblue}{HTML}{00B4D8}
\definecolor{gapred}{HTML}{C2185B}\definecolor{inkmute}{HTML}{6B7280}
\begin{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default="/newdata2/dylantw/tmp/claude-1727/"
                    "-newdata2-dylantw-Legislated-Planner-src/"
                    "8f348aa2-3a2b-4256-9903-13229c0043e0/scratchpad/fig1_stages")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    work = out / "_tex"; work.mkdir(exist_ok=True)

    body = (PAPER / "main.tex").read_text().splitlines()[BODY_FIRST - 1:BODY_LAST]
    assert body[0].strip().startswith(r"\begin{tikzpicture}"), body[0]
    assert body[OPTS_END - 1].strip() == "]", f"line {OPTS_END} is not the options terminator: {body[OPTS_END-1]!r}"
    assert body[-1].strip() == r"\end{tikzpicture}", body[-1]

    body = [functools.reduce(lambda a, kv: a.replace(*kv), ENV_SHIFT, ln) for ln in body]
    head = body[:OPTS_END] + [BBOX]
    sizes = {}
    for name, last in STAGES:
        lines = head + body[OPTS_END:last]
        if lines[-1].strip() != r"\end{tikzpicture}":
            lines.append(r"\end{tikzpicture}")
        tex = work / f"stage_{name}.tex"
        tex.write_text(PREAMBLE + "\n".join(lines) + "\n\\end{document}\n")

        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={work}", str(tex)],
                           cwd=PAPER, capture_output=True, text=True)
        if r.returncode != 0:
            tail = "\n".join(r.stdout.splitlines()[-25:])
            sys.exit(f"[fig1] pdflatex FAILED on stage '{name}':\n{tail}")

        png = out / f"stage_{name}.png"
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
                        f"-r{args.dpi}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        f"-sOutputFile={png}", str(work / f"stage_{name}.pdf")], check=True)
        from PIL import Image
        with Image.open(png) as im:
            sizes[name] = im.size
        print(f"[fig1] stage {name:<11} -> {png.name}  {sizes[name][0]}x{sizes[name][1]}")

    uniq = set(sizes.values())
    if len(uniq) != 1:
        sys.exit(f"[fig1] ABORT: stages have different dimensions {sizes} -- they would jitter. "
                 f"Widen the pinned bbox.")
    print(f"[fig1] all {len(STAGES)} stages registered at {uniq.pop()}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
