#!/usr/bin/env python3
"""Paper tables from the BASE trichotomy (off/social/deviant), pooled over base_025 + base_x2 = 400
episodes/agent (two distinct 200-scenario benchmarks, law_eval_center + law_eval_center_s1; see
results/final tracking). Emits two booktabs LaTeX tables:

  tab_price_of_legality.tex  (Q5) -- the PATH cost of obeying: path efficiency (actual / legal-optimal
                                     travel) + strokes to goal, per agent. Companion to the runtime cost.
  tab_wm_error.tex           (Q2) -- WM one-step error (cube-xy metres + latent L2), per agent, showing
                                     it is intrinsic to the world model (agent-invariant).

Usage:
    python scripts/report_base_metrics.py [--out-dir DIR]
"""
import argparse, glob, json
import numpy as np

ROOTS = ["results/final/base_025", "results/final/base_x2"]
AGENTS = [("off", "realistic"), ("social", "social (legislated)"), ("deviant", "deviant (penalized)")]
CUBE_HALF = 0.045


def pool(agent, keys):
    acc = {k: [] for k in keys}
    for r in ROOTS:
        for f in glob.glob(f"{r}/{agent}/*/batch_*/eval_metrics.json"):
            d = json.load(open(f))
            for k in keys:
                for x in d.get(k, []):
                    if isinstance(x, (int, float)) and x is not None and np.isfinite(x):
                        acc[k].append(x)
    return {k: np.asarray(v, dtype=float) for k, v in acc.items()}


def _table(header_cols, rows, colspec, caption, label, size=r"\small"):
    L = [r"\begin{table}[H]\centering" + size, r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{@{}" + colspec + r"@{}}", r"\toprule",
         " & ".join(header_cols) + r" \\", r"\midrule"]
    L += [r" & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + r"}",
          r"\label{" + label + r"}", r"\end{table}"]
    return "\n".join(L) + "\n"


def price_table(out):
    keys = ("path_efficiency", "n_steps", "law_detour_ratio", "success")
    rows = []
    detour = []
    for ag, disp in AGENTS:
        a = pool(ag, keys)
        detour.append(np.nanmean(a["law_detour_ratio"]))
        rows.append([disp, f"{np.mean(a['path_efficiency']):.2f}", f"{np.median(a['path_efficiency']):.2f}",
                     f"{np.mean(a['n_steps']):.2f}"])
    cap = (r"The path cost of obeying, base trichotomy pooled over two benchmarks ($n{=}400$/agent). "
           r"Path efficiency is executed travel-to-goal divided by the shortest law-respecting path "
           r"(pure geometry); $<1$ means the agent took an illegal shortcut. The realistic agent ignores "
           r"the law and undercuts the legal optimum; the social agent pays $1.46\times$ to route around "
           r"the forbidden cell, while the deviant agent's cost reordering finds cheaper legal routes "
           r"($1.16\times$). The law itself forces at least $%.2f\times$ the straight-line path "
           r"(\texttt{law\_detour\_ratio}, agent-invariant geometry)." % np.mean(detour))
    tex = _table(["Agent", "path eff. (mean)", "(median)", "strokes"], rows, "lrrr", cap, "tab:price")
    open(out, "w").write(tex); print(f"wrote {out}\n{tex}")


def wm_table(out):
    keys = ("wm_pred_err", "wm_latent_err")
    rows = []
    for ag, disp in AGENTS:
        a = pool(ag, keys)
        rows.append([disp, f"{np.mean(a['wm_pred_err']):.3f} $\\pm$ {np.std(a['wm_pred_err']):.3f}",
                     f"{np.mean(a['wm_latent_err']):.3f} $\\pm$ {np.std(a['wm_latent_err']):.3f}"])
    cap = (r"World-model one-step error, base trichotomy ($n{=}400$/agent). Prediction error is the "
           r"cube-xy distance (metres) between the WM-imagined and executed next state; latent error is "
           r"the $L_2$ distance in DINO latent space. The error is essentially identical across agents: it "
           r"is intrinsic to the world model, not the planner or the law. At $\approx\!0.032$\,m it is "
           r"$\approx\!70\%$ of the cube half-width ($0.045$\,m), which is why WM error, not the deontic "
           r"logic, is the binding obstacle to ex ante governance (Q2's cushion compensates for it).")
    tex = _table(["Agent", "pred.\\ error (cube xy, m)", "latent error ($L_2$)"], rows, "lrr",
                 cap, "tab:wmerr")
    open(out, "w").write(tex); print(f"wrote {out}\n{tex}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/newdata2/dylantw/Legislative-Harness/full_paper/Images")
    a = ap.parse_args()
    price_table(f"{a.out_dir}/tab_price_of_legality.tex")
    wm_table(f"{a.out_dir}/tab_wm_error.tex")
