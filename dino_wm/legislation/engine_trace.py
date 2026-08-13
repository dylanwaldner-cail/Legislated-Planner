"""Trace a law through the DDL engine, from input facts to the deontic verdict -- a paper/JURIX
reporting tool, not part of the planning loop.

For a given perceived state it dumps the WHOLE derivation the engine produces (not just the
verdict): which rules were APPLICABLE, which were DISCARDED / DEFEATED (and by which superior
rule), the resulting OBLIGATIONS / PERMISSIONS / VIOLATIONS, and any CONTRARY-TO-DUTY compensation.
Rule labels in the trace are annotated with their .dl text, so a reader can follow, e.g., how
`green_sign_permits_center > no_center_cell` defeats the centre-cell prohibition when sign(green)
holds. This is the "print the engine output to a log and report it in the paper" artifact.

Runs wherever `clingo` imports (host conda now; the container also works). From the repo root:

    python -m legislation.engine_trace                          # all preset scenarios, geometric_laws
    python -m legislation.engine_trace --lawset full_lawset     # ...for the full R1-R10 system
    python -m legislation.engine_trace --scenario green_defeats # one named preset
    python -m legislation.engine_trace --state "in_cell(4) sign(green)"   # a custom state
    python -m legislation.engine_trace --out results/engine_trace.log     # tee to a log file
    python -m legislation.engine_trace --verbose               # also dump EVERY engine predicate

`--state` facts are space/comma separated; `cube` is added automatically. `--lawset` selects which
law CATEGORY is compiled (geometric_laws | full_lawset); the presets are keyed to the lawset.
"""
from __future__ import annotations

import argparse
import re
import sys

from legislation.reasoner import LegislativeReasoner, _render_dl

# Canonical states worth tracing, per lawset. (name, one-line point, extra facts beyond `cube`).
# Chosen to exercise a bare prohibition, a superiority defeat, an ambiguity-block, and a CTD chain --
# the four DDL behaviours the paper claims. Add your own with --state.
_PRESETS = {
    "geometric_laws": [
        ("baseline", "no signs -> the lone centre-cell prohibition stands", []),
        ("in_center", "cube IS in cell 4 -> prohibition violated; center_ctd fires the exit duty",
         ["in_cell(4)"]),
        ("green_defeats", "sign(green): green_sign_permits_center > no_center_cell DEFEATS the "
         "prohibition -> cell 4 becomes permitted", ["sign(green)", "in_cell(4)"]),
        ("yellow_requires", "sign(yellow): yellow_sign_requires_center flips the prohibition into a "
         "positive obligation to enter cell 4", ["sign(yellow)"]),
        ("goal_is_forbidden", "goal_cell(4) vs no_center_cell, NO superiority -> the two "
         "ambiguity-block (neither holds) -> cell 4 unregulated", ["goal_cell(4)"]),
        ("goal_elsewhere", "goal_cell(0): reach_goal_0 obliges cell 0, prohibition on cell 4 stands "
         "-> a normal legal goal", ["goal_cell(0)"]),
    ],
    "full_lawset": [
        ("baseline", "no signs -> R2 prohibits the centre; R1 keeps it on-grid", []),
        ("goal", "goal_cell(0) + free to move -> reach_goal obliges in_cell(0). Shows the GOAL "
         "obligation, gated by may_move ([P]moving)", ["goal_cell(0)"]),
        ("red_freeze", "sign(red): R2 vs R3 ambiguity-block -> perm_conflict(4) -> R9 obliges a full "
         "stop [O]~moving", ["sign(red)"]),
        ("goal_suspended", "PERMISSION-IN-BODY showcase: sign(red) freezes ([O]~moving) so may_move "
         "fails -> the goal obligation is SUSPENDED (compare with `goal`: same goal_cell(0), no [O]in_cell(0))",
         ["sign(red)", "goal_cell(0)"]),
        ("contrary_to_duty", "cube in cell 4 -> contrary_to_duty violated -> compensation activates the "
         "reparative [O]exit_cell(4)", ["in_cell(4)"]),
        ("yellow_flip", "cube TRULY reaches yellow cell 3 under sign(yellow) + goal_cell(8): the "
         "AUTHORITY's occupies(3) (GT, not the probe) fires yellow_counts_as -> in_yellow_cell; "
         "yellow_to_green flips GREEN; reach_goal obliges in_cell(8)",
         ["sign(yellow)", "occupies(3)", "yellow_cell(3)", "goal_cell(8)"]),
        ("yellow_flip_tainted", "SAME as yellow_flip but the cube ALREADY passed through cell 4 "
         "(visited(4), from GT occupies history): R7b > R7 -> the flip goes RED, not green (history "
         "changes the verdict)",
         ["sign(yellow)", "occupies(3)", "yellow_cell(3)", "visited(4)", "goal_cell(8)"]),
        ("yellow_probe_cannot_flip", "AUTHORITY guard: the PROBE puts the cube in yellow cell 3 "
         "(in_cell(3)) but the GT occupies(3) is absent -> yellow_counts_as does NOT fire, in_yellow_cell "
         "is not derived, and the sign stays YELLOW. A perception error alone cannot flip the sign.",
         ["sign(yellow)", "in_cell(3)", "yellow_cell(3)", "goal_cell(8)"]),
        ("green_defeats", "sign(green): R4 > R2 defeats the centre prohibition", ["sign(green)", "in_cell(4)"]),
    ],
}

# The engine predicates that make up a READABLE derivation trace, in input->verdict order. Everything
# else (structural: literal/atom/opposes/body/rule/... plus weak-permission noise) is hidden unless
# --verbose. See Defeasible-Deontic-Logic/Deontic/*.asp for the full predicate semantics.
_TRACE_SECTIONS = [
    ("defeasible", "Defeasibly provable literals (the derived state)"),
    ("applicable", "Rules APPLICABLE (antecedent satisfied)"),
    ("discarded", "Rules DISCARDED (a body literal is refuted)"),
    ("defeated", "Rules DEFEATED (beaten by a superior opposing rule)"),
    ("rebutted", "Rules REBUTTED (defeated or discarded head)"),
    ("overruled", "Rules OVERRULED"),
    ("refuted", "Literals REFUTED (provably not derivable)"),
    ("obligation", "OBLIGATIONS  (obligation(X); obligation(Rule,X,N) = per-rule, N=CTD depth)"),
    ("compensate", "COMPENSATION chains (contrary-to-duty: primary (X) reparative)"),
    ("permission", "PERMISSIONS"),
    ("violation", "VIOLATIONS  (an obligation whose opposite is provable)"),
    ("weakViolation", "WEAK violations (an obligation whose target is refuted)"),
    ("terminalViolation", "TERMINAL violations (breach with no remedy)"),
    ("superior", "SUPERIORITY in effect (superior(Strong,Weak))"),
]


def _rule_text(dl_text):
    """label -> its rendered .dl rule text, from _render_dl output (skip `A > B` superiority lines).
    Longest labels first so annotate() prefers reach_goal_4 over reach_goal."""
    out = {}
    for line in dl_text.splitlines():
        if " => " in line and ": " in line:
            label, body = line.split(": ", 1)
            out[label.strip()] = body.strip()
    return dict(sorted(out.items(), key=lambda kv: -len(kv[0])))


def _annotate(atom, labels):
    """If a rule label appears (as a whole token) in the atom string, append its .dl text."""
    for lab, txt in labels.items():
        if re.search(rf"(?<![\w]){re.escape(lab)}(?![\w])", atom):
            return f"{atom:<44} % {lab}: {txt}"
    return atom


def _parse_state(s):
    """'in_cell(4), sign(green)' | 'in_cell(4) sign(green)' -> ['in_cell(4)', 'sign(green)'].
    Split on commas / whitespace but NOT inside parentheses (in_cell(4) stays intact)."""
    toks, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1; cur += ch
        elif ch == ")":
            depth -= 1; cur += ch
        elif depth == 0 and (ch.isspace() or ch == ","):
            if cur:
                toks.append(cur); cur = ""
        else:
            cur += ch
    if cur:
        toks.append(cur)
    return toks


def trace(reasoner, facts, labels, emit, verbose=False):
    """Print the full engine derivation for one state (`facts`) via `emit(line)`."""
    facts = ["cube"] + [f for f in facts if f != "cube"]
    emit("  input facts : " + ", ".join(f"fact({f})" for f in facts))
    allatoms = reasoner.solve_all(facts)
    shown = set()
    for name, title in _TRACE_SECTIONS:
        atoms = allatoms.get(name, [])
        shown.add(name)
        emit(f"\n  {title}")
        if not atoms:
            emit("      (none)")
        for a in atoms:
            emit("      " + _annotate(a, labels))
    if verbose:                                   # every remaining predicate (structural + weak-perm)
        emit("\n  --- other engine predicates (verbose) ---")
        for name in sorted(set(allatoms) - shown):
            emit(f"  {name}/ : " + ";  ".join(allatoms[name]))
    v = reasoner.assess(facts)                     # the structured verdict the planner consumes
    emit("\n  VERDICT (assess -> Constraint):")
    for k in ("obligations", "prohibitions", "permissions", "violations", "signs"):
        emit(f"      {k:12}: {v.get(k)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trace a law through the DDL engine.")
    ap.add_argument("--lawset", default="geometric_laws", choices=sorted(_PRESETS),
                    help="which law CATEGORY to compile + trace")
    ap.add_argument("--scenario", help="run one named preset (see --lawset's presets)")
    ap.add_argument("--state", help="custom facts (space/comma separated); 'cube' is added for you")
    ap.add_argument("--verbose", action="store_true", help="also dump every engine predicate")
    ap.add_argument("--out", help="also write the trace to this log file")
    args = ap.parse_args(argv)

    reasoner = LegislativeReasoner(active_lawsets=[args.lawset])
    dl_text = _render_dl(reasoner.db, [args.lawset])
    labels = _rule_text(dl_text)

    lines = []
    emit = lambda s="": (print(s), lines.append(s))
    emit("=" * 96)
    emit(f"DDL ENGINE TRACE   active lawset: {args.lawset}")
    emit("=" * 96)
    emit("\nRULES IN FORCE (rendered DDL .dl):")
    for line in dl_text.splitlines():
        emit(("    > " if " > " in line and " => " not in line else "    ") + line)

    if args.state is not None:
        scenarios = [("custom", args.state, _parse_state(args.state))]
    else:
        presets = _PRESETS[args.lawset]
        if args.scenario:
            presets = [p for p in presets if p[0] == args.scenario]
            if not presets:
                ap.error(f"no preset '{args.scenario}' for lawset {args.lawset}; "
                         f"have {[p[0] for p in _PRESETS[args.lawset]]}")
        scenarios = presets

    for name, desc, extra in scenarios:
        emit("\n" + "-" * 96)
        emit(f"SCENARIO: {name}")
        emit(f"  {desc}")
        emit("-" * 96)
        trace(reasoner, extra, labels, emit, verbose=args.verbose)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[wrote trace -> {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
