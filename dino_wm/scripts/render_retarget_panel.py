"""Rule-injection demo panel: the goal the planner was AIMING AT changes mid-episode.

THE POINT. At `rule_injection.frame` the sign reverts to white and R11 (white_sign_return) is
injected, raising [O]return_cell(S). mpc._retarget_from_duty then overwrites obs_g with the goal
bank's cell-S image, so the planner's target switches from the episode's original goal to home.
The recorded mp4 cannot show this -- it is rendered once at the end and bakes in the FINAL obs_g
for every frame -- so this script rebuilds the composite with the correct goal PER FRAME.

BOTH goal images are the real ones the planner was handed; nothing is synthesised:
    frames < inject : plan_targets.pkl -> obs_g['visual'][eval]   (this episode's original goal)
    frames >= inject: goal_cell_bank/obses/cell_%02d.pth          (exactly what the retarget wrote)

Executed frames come from the recorded mp4 by default (`--source mp4`, no container needed); the
compositor's debug darkening is inverted so the panels read at their true brightness. Sign colour
per frame is the ledger's EFFECTIVE sign (the colour the verdict was assessed under).

    python scripts/render_retarget_panel.py [--batch DIR] [--eval N] [--inject 3] [--out FIG.png]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

BANK = "/newdata2/dylantw/Legislative-Harness/dino_wm/data/goal_cell_bank/obses"
DEF_BATCH = ("/newdata2/dylantw/Legislative-Harness/dino_wm/results/aug28/"
             "rule_insert/social/2_6/batch_000")
SIGN_RGB = {"white": "#E8E8E8", "yellow": "#F2C200", "green": "#2E9E4F", "red": "#D0342C"}
DARK = 0.3            # evaluator._plot_rollout_compare `correction`

# LAYOUT, verified empirically against this run's mp4 (not assumed from the compositor source):
# frames are (224, 448) = ONE row, [executed | goal], each 224x224. There is no imagined panel
# because these runs set has_decoder=false. Each 224x224 panel itself stacks two camera views, so
# its top and bottom halves are uncorrelated (r=0.19) -- that is the raw obs, not a composite seam.
# The goal panel is darkened by exactly 1*DARK (mean|diff| 0.110 vs the bank image, against 0.327
# at k=0 and 0.168 at k=2), and the executed panel carries the same single subtraction.
PANEL = 224


def _undark(panel_u8, n=1):
    """Invert the compositor's n*DARK subtraction. It wrote ((x - n*DARK) + 1)/2 * 255 with no
    rescale (min < 0 short-circuits the *2-1 branch), so x = pix/255*2 - 1 + n*DARK."""
    return np.clip((panel_u8.astype(np.float32) / 255.0) * 2.0 - 1.0 + n * DARK, 0.0, 1.0)


def _sign_at(rec):
    return rec.get("effective_sign") or ((rec.get("verdict") or {}).get("signs") or [None])[0]


def _bank_img(cell):
    """The goal bank's canonical image of `cell`. For a WAYPOINT this is a DEPICTION: the planner
    consumed only bank.pos[cell], a coordinate. Only return_cell ever loads an image (into obs_g)."""
    g = torch.load(f"{BANK}/cell_{int(cell):02d}.pth", map_location="cpu").numpy().astype(np.float32)
    return g / 255.0 if g.max() > 1.5 else g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=DEF_BATCH)
    ap.add_argument("--eval", type=int, default=3)
    ap.add_argument("--inject", type=int, default=3)
    ap.add_argument("--source", choices=["mp4"], default="mp4")
    ap.add_argument("--frames", type=int, default=0, help="panels to show; 0 = auto-trim")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="render ONLY these frame indices -- a compact strip for a half-width slot")
    ap.add_argument("--figw", type=float, default=1.05, help="inches per panel column")
    ap.add_argument("--no-trim", dest="trim", action="store_false")
    ap.add_argument("--out", default="retarget_panel.png")
    a = ap.parse_args()

    vid = glob.glob(f"{a.batch}/output_final_{a.eval}_*.mp4")
    if not vid:
        raise SystemExit(f"no video for eval {a.eval} in {a.batch}")
    frames = [f for f in imageio.get_reader(vid[0])]
    execs = [_undark(f[:, :PANEL]) for f in frames]   # left panel = executed rollout

    # TRIM. The video runs the full MPC horizon, but `cube_xy_frames` is trimmed at goal-hit, so its
    # length is the last frame where the trajectory is still doing something -- everything after is
    # the success hold repeating one pose. Trimming there keeps the figure about the episode rather
    # than about the padding. --frames overrides; --no-trim shows the raw video length.
    if a.frames:
        execs = execs[:a.frames]
    elif a.trim:
        try:
            Ti = len(json.load(open(f"{a.batch}/eval_metrics.json"))["cube_xy_frames"][a.eval])
            execs = execs[:max(a.inject + 3, min(Ti + 1, len(execs)))]
        except Exception:
            pass

    keep = list(range(len(execs))) if not a.only else [i for i in a.only if i < len(execs)]

    led = json.load(open(f"{a.batch}/normative_ledger.json"))
    recs = led[sorted(led, key=int)[a.eval]]["records"]
    signs = [_sign_at(r) for r in recs]

    # the two REAL goal images the planner held, before and after the retarget
    tp = pickle.load(open(f"{a.batch}/plan_targets.pkl", "rb"))
    g_pre = np.asarray(tp["obs_g"]["visual"][a.eval, 0]).astype(np.float32) / 255.0
    ret = next((int(o.split("(")[1].rstrip(")")) for r in recs[a.inject:]
                for o in ((r.get("verdict") or {}).get("obligations") or [])
                if o.startswith("return_cell")), None)
    if ret is None:
        raise SystemExit("no return_cell duty found at/after the injection frame")
    g_post = _bank_img(ret)
    # the ORIGINAL task goal cell, as the ledger recorded it before the retarget landed
    pre_cell = next((c.get("goal_cell") for r in recs[:a.inject]
                     for c in [r.get("committed") or {}] if c.get("goal_cell") is not None), "?")

    execs = [execs[i] for i in keep]
    recs = [recs[i] if i < len(recs) else {} for i in keep]
    idx = keep                                  # original frame numbers, for the titles
    n = len(execs)
    fig, axes = plt.subplots(2, n, figsize=(a.figw * n, 2.85),
                             gridspec_kw=dict(hspace=0.30, wspace=0.06))
    for i in range(n):
        s = signs[idx[i]] if idx[i] < len(signs) else None
        axes[0, i].imshow(execs[i])
        axes[0, i].set_title(f"{idx[i]}", fontsize=8, pad=2)
        # sign swatch: the colour the verdict was assessed under at this step. Frames can outnumber
        # ledger records (the final rest frame has no decision), so a missing sign renders grey.
        axes[0, i].add_patch(Rectangle((0, 0), PANEL, PANEL * 0.055, color=SIGN_RGB.get(s, "#999999"),
                                       zorder=5, ec="#444444", lw=0.4))
        # THE TARGET THE RRT ACTUALLY BUILT ITS TREE TOWARD. obs_g is NOT it whenever a positive
        # obligation is live: rrt.py:416 swaps tgt_cube = bank.pos[wp] and _build_tree uses only that
        # position (z_goal never enters the tree). Only `return_cell` rewrites obs_g, so a waypoint
        # duty -- yellow check-in, exit_cell -- steers the planner while leaving the goal image stale.
        wp = (recs[i].get("committed") or {}).get("obligation_waypoint")
        # R9 FREEZE: prohibitions become [F]moving, so the agent is not steering anywhere and a goal
        # panel here would imply motion that did not happen. Read from the verdict, not a distance
        # threshold. (in_cell(4) also drops out of the prohibitions -- the R2/R3 ambiguity block.)
        frozen = "moving" in ((recs[i].get("verdict") or {}).get("prohibitions") or [])
        if frozen:
            axes[1, i].imshow(np.ones((PANEL, PANEL, 3)) * 0.82)
            lab, col = "frozen\n[F]moving", "#666666"
        elif wp is not None:
            axes[1, i].imshow(_bank_img(wp)); lab, col = f"wp {wp}", "#E8820C"
        else:
            axes[1, i].imshow(g_pre if idx[i] < a.inject else g_post)
            lab, col = f"goal {ret if idx[i] >= a.inject else pre_cell}", "#C2185B"
        axes[1, i].set_xlabel(lab, fontsize=6.5, color=col, labelpad=1.5)
        for r in (0, 1):
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
            for sp in axes[r, i].spines.values():
                sp.set_edgecolor(col if r == 1 else "#BBBBBB")
                sp.set_linewidth(1.6 if r == 1 else 0.5)

    axes[0, 0].set_ylabel("executed", fontsize=8)
    axes[1, 0].set_ylabel("goal the\nplanner held", fontsize=8)
    # amendment marker
    _mk = next((j for j, f in enumerate(idx) if f >= a.inject), None)
    x = axes[1, _mk].get_position().x0 - 0.004 if _mk else None
    if x is not None:
        fig.lines.append(plt.Line2D([x, x], [0.06, 0.90], transform=fig.transFigure,
                                    color="#C2185B", lw=1.6, ls="--"))
    # A compact strip is destined for a half-width slot beside another figure, so the shared caption
    # carries the explanation; only the full-width strip footnotes itself.
    if n > 5:
        fig.text((x + 0.006) if x is not None else 0.1, 0.025,
                 f"amendment @ frame {a.inject}: sign reverts to white, R11 injected "
                 f"$\\Rightarrow$ [O]return_cell({ret}); goal retargets to home",
                 fontsize=8, color="#C2185B", va="bottom")
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}  ({n} frames, return_cell({ret}), signs={''.join((s or '?')[0] for s in signs[:n])})")


if __name__ == "__main__":
    main()
