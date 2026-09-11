#!/usr/bin/env python3
"""Q5 runtime report: aggregate the per-plan() `runtime_breakdown` recorded in every eval_metrics.json and
show where wall-clock goes -- RRT/WM rollouts vs the LEGISLATION overhead (reason vs prune), plus the fine
reason split (probe / ground / clingo-logic / build) when present. Headline: the DDL apparatus is a
negligible slice; RRT/WM dominates.

UNIT: seconds PER EXECUTED ACTION (one MPC loop iteration = plan a tree, execute the first stroke,
re-perceive, replan). Denominator = total executed actions = sum(n_steps). Reasoning (observe) fires ~once
per loop; pruning fires per RRT candidate within the loop; both are reported amortized per executed action.
Scenarios plan SERIALLY (rrt.py `for e in range(n_evals)`), so the aggregate is a clean per-action mean.
Report ONE agent mode per column (the governed 'social' agent is the right "cost of enforcement"
reference; 'off' never prunes). Do NOT average across modes -- off/social/deviant are different workloads.

Coarse keys (in ALL past runs): plan_total_s, rrt_s, legislation_s, legislation_reason_s, legislation_prune_s.
Fine keys  (runs AFTER the enforcement timing patch): leg_probe_s, leg_ground_s, leg_logic_s, leg_build_s,
leg_n_observe -- decompose legislation_reason_s (isolating clingo `logic`); leg_n_observe = exact verdict count.

Usage:
    python scripts/report_runtime.py \
        "Q1--Q2 (geometric)=results/aug10/cushion_social/delta_0.03/**/eval_metrics.json" \
        "Q3 (full lawset)=results/aug05/sign_fullrun/social/**/eval_metrics.json" \
        [--unit action|scenario|plan] [--tex OUT.tex]
Each arg is 'LABEL=GLOB' (point each glob at ONE agent mode). Prints a console table; --tex writes a paper table.
"""
import argparse, glob, json
import numpy as np

COARSE = ["plan_total_s", "rrt_s", "legislation_s", "legislation_reason_s", "legislation_prune_s"]
FINE = ["leg_probe_s", "leg_ground_s", "leg_logic_s", "leg_build_s", "leg_n_observe"]
_PER = {"action": "executed action", "scenario": "scenario", "plan": "plan() call"}
_UCOL = {"action": r"s\,/\,action", "scenario": r"s\,/\,scenario", "plan": r"s\,/\,call"}


def aggregate(pattern):
    """Sum every runtime_breakdown key + n_evals + executed actions (sum n_steps) over matching files."""
    sums = {k: 0.0 for k in COARSE + FINE}
    n_scen = n_rec = n_act = 0
    for f in glob.glob(pattern, recursive=True):
        d = json.load(open(f))
        rb = d.get("runtime_breakdown")
        if not rb:
            continue
        n_rec += 1
        n_scen += int(d.get("n_evals", 0) or 0)
        n_act += int(sum(s for s in d.get("n_steps", []) if s is not None and np.isfinite(s)))
        for k in COARSE + FINE:
            sums[k] += float(rb.get(k, 0.0))
    if not n_rec:
        return None
    return {"sums": sums, "n_scen": n_scen, "n_rec": n_rec, "n_act": n_act,
            "has_fine": sums["leg_logic_s"] > 0 or sums["leg_n_observe"] > 0}


def _denom(m, unit):
    return {"action": m["n_act"], "scenario": m["n_scen"], "plan": m["n_rec"]}[unit]


def pct(x, tot):
    return 100.0 * x / tot if tot else 0.0


def console(cols, unit):
    for label, m in cols:
        if m is None:
            print(f"[{label}] no runtime_breakdown found\n"); continue
        d = _denom(m, unit) or 1
        val = {k: m["sums"][k] / d for k in COARSE}
        tot = val["plan_total_s"] or 1.0
        print(f"[{label}]  {m['n_rec']} records, {m['n_scen']} scenarios, {m['n_act']} actions"
              f"   (mean seconds / {_PER[unit]}, % of total)")
        for k in COARSE:
            print(f"    {k:24s} {val[k]:9.4f}s  {pct(val[k], tot):5.2f}%")
        if m["has_fine"]:
            nobs = m["sums"]["leg_n_observe"]
            print(f"    -- reason fine split ({nobs:.0f} verdicts) --")
            for k in ["leg_probe_s", "leg_ground_s", "leg_logic_s", "leg_build_s"]:
                v = m["sums"][k] / d
                perv = (1e3 * m["sums"][k] / nobs) if nobs else 0.0
                print(f"    {k:24s} {v:9.4f}s  {pct(v, tot):5.2f}%   ({perv:.3f} ms/verdict)")
        else:
            print("    -- fine split (probe/clingo/build) not recorded in this run --")
        print()


def latex(cols, unit, out, pending=()):
    """Booktabs table, rows = components, one column per config: 'sec (pct%)'. Columns whose label matches
    a --pending substring are rendered as red \\texttt{?} (their data is stale / awaiting a re-run) so no
    false numbers can be reported."""
    pend = [p.lower() for p in pending]

    def is_pend(label):
        return any(p in label.lower() for p in pend)

    def val(m, k):
        return m["sums"][k] / (_denom(m, unit) or 1)

    def cell(label, m, k):
        if is_pend(label):
            return r"\textcolor{red}{?}"
        if m is None:
            return "--"
        v = val(m, k); tot = val(m, "plan_total_s") or 1.0
        return f"{v:.3f} ({pct(v, tot):.2f}\\%)"
    labels = [l for l, _ in cols]
    L = [r"\begin{table}[H]\centering\small", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}", r"\toprule",
         "Component & " + " & ".join(labels) + r" \\", r"\midrule"]

    def row(name, key, indent=False):
        pre = r"\quad " if indent else ""
        return f"{pre}{name} & " + " & ".join(cell(l, m, key) for l, m in cols) + r" \\"
    warn = (r" \textcolor{red}{The full-lawset (Q3) column is pending a re-run after the swept-taint law "
            r"change and is shown as \texttt{?}; it must not be reported until refreshed.}") if pend else ""
    # Cite the reasoning-cost figure from VALID columns only. While Q3 is pending, quote the geometric
    # (Q1--Q2) number alone so no stale Q3-derived range leaks into the caption.
    reason_claim = (r"$\approx\!20$\,ms ($\approx\!0.07\%$ of the action) for the geometric lawset" if pend
                    else r"$\approx\!20$--$30$\,ms ($\approx\!0.1$--$0.3\%$ of the action)")
    L += [row("RRT search + WM rollouts", "rrt_s"),
          row("Legislation (total)", "legislation_s"),
          row("reasoning (probe+ground+DDL+build)", "legislation_reason_s", indent=True),
          row("pruning (per-candidate legality)", "legislation_prune_s", indent=True),
          r"\midrule", row("Plan total", "plan_total_s"), r"\bottomrule",
          r"\end{tabular}",
          r"\caption{Where planning wall-clock goes for the governed (social) agent, mean seconds per "
          r"executed action (one MPC loop: plan, execute a stroke, re-perceive, replan) with \% of total. "
          r"RRT search and world model rollouts dominate. The entire legislation layer, both the "
          r"defeasible deontic reasoning and the per-candidate legality pruning, is $\approx\!1\%$, and the "
          r"symbolic reasoning path (perception grounding, clingo DDL solve, constraint build) is only "
          + reason_claim + r". The deontic apparatus is not the bottleneck." + warn + r"}",
          r"\label{tab:runtime}", r"\end{table}"]
    tex = "\n".join(L) + "\n"
    if out:
        open(out, "w").write(tex)
        print(f"wrote {out}")
    print("\n" + tex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="LABEL=GLOB pairs (point each glob at ONE agent mode)")
    ap.add_argument("--unit", choices=["action", "scenario", "plan"], default="action")
    ap.add_argument("--pending", nargs="*", default=[],
                    help="column-label substrings whose (stale) data is awaiting a re-run -> render as red ?")
    ap.add_argument("--tex", default=None, help="write a LaTeX table to this path")
    a = ap.parse_args()
    cols = [(s.split("=", 1)[0], aggregate(s.split("=", 1)[1])) for s in a.specs]
    console(cols, a.unit)
    latex(cols, a.unit, a.tex, a.pending)
