"""Legislative reasoning bridge: laws (defeasible deontic logic) -> deontic conclusions.

Reads the modular legal database (legal_database.yaml), renders it to Governatori DDL .dl
text, compiles it with Defeasible-Deontic-Logic/Python/parser.py, and runs clingo over the
DDL engine to derive what is obligatory / permitted / violated (and any constitutive
"counts-as" facts) from a perceived state.

DESIGN: this layer is deliberately law-AGNOSTIC and stops at the normative verdict. It does
NOT decide what a conclusion *does* to the planner (prune / reward / abort / replan) — that
enforcement mapping depends on the actual laws and is intentionally left downstream, so the
reasoner is not hardwired to "forbidden geometry -> obstacle". Conclusions range over
arbitrary literals (e.g. push_gently, in_place(a), trespass), not just cell predicates.

Runs in the container (needs `clingo` + the DDL .asp engine files).

    from legislation.reasoner import LegislativeReasoner
    r = LegislativeReasoner()                       # loads legal_database.yaml + compiles the theory
    r.assess(["cube", "in_cell(4)"])                # -> {'obligations':[...], 'prohibitions':[...],
                                                    #     'permissions':[...], 'violations':[...]}
    r.derive(["cube"], predicates=["obligation", "defeasible"])   # general query, any engine predicate
"""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

import yaml
import clingo

_HERE = Path(__file__).resolve().parent
_DDL_ROOT = _HERE.parent / "Defeasible-Deontic-Logic"
sys.path.insert(0, str(_DDL_ROOT / "Python"))   # so `import parser` resolves to the DDL parser
import parser as ddl_parser  # noqa: E402

# DDL engine: ambiguity-blocking defeasible inference + deontic w/ compensation.
# Mirrors Defeasible-Deontic-Logic/Python/ddl.py; resolved under _DDL_ROOT.
_ENGINE_FILES = [
    "language.asp",
    "Basic/language.asp",
    "Deontic/language.asp",
    "Deontic/defeasible-ab.asp",
    "Deontic/deontic-comp.asp",
]
_DEFAULT_DB = _HERE / "legal_database.yaml"
# The engine's deontic verdict predicates. `derive()` can request ANY predicate the engine
# produces (e.g. "defeasible" for constitutive/derived facts) — these are just the default.
_DEONTIC_PREDICATES = ("obligation", "permission", "violation", "weakViolation")


def load_legal_database(path=_DEFAULT_DB):
    with open(path, "r", encoding="utf-8") as f:   # YAML has non-ascii (em dashes); container default is ascii
        return yaml.safe_load(f)


def _expand_law(law):
    """SCHEMA expansion: emit one ground copy per binding, substituting each VAR (matched as a
    whole ASP term) everywhere in antecedent/consequent and suffixing the label. ASP forbids free
    variables in facts, so schematic laws must be expanded HERE. Reference an expanded copy in
    `superiority` by its suffixed label (e.g. ["reach_goal_4", "no_center_cell"]). Two forms:

      expand: {VAR: domain, ...}   -> CARTESIAN product of INDEPENDENT var domains (`reach_goal` +
          {N: "0..8"} -> reach_goal_0..8). domain = a list or a clingo-style "a..b" integer range.

      expand_pairs: {vars: [N, M], pairs: <list | "grid_borders">}  -> CORRELATED tuples: one copy
          per tuple with vars zipped to it. Use this instead of a relational guard predicate in the
          antecedent -- the DDL syntax separates antecedent literals by comma, so a 2-ary term like
          borders(N,M) would be mis-split. "grid_borders" pulls the grid adjacency (N, neighbour M)
          from grounding, so `goal_cell(N) => [O]~in_cell(M)` becomes a per-neighbour prohibition."""

    def emit(keys, combos):
        out = []
        for combo in combos:
            sub = dict(zip(keys, combo))

            def rep(s):
                for k, v in sub.items():
                    s = re.sub(rf"\b{k}\b", v, s)
                return s

            out.append({
                "label": law["label"] + "_" + "_".join(combo),
                "antecedent": [rep(a) for a in (law.get("antecedent", []) or [])],
                "consequent": [rep(c) for c in law["consequent"]],
            })
        return out

    pairs = law.get("expand_pairs")
    if pairs:
        vals = pairs.get("pairs")
        if vals in ("grid_borders", "grid_borders_diag"):
            from legislation.grounding import _grid_border_pairs
            vals = _grid_border_pairs(diagonal=(vals == "grid_borders_diag"))
        combos = [tuple(str(x) for x in t) for t in vals]
        return emit(list(pairs["vars"]), combos)

    exp = law.get("expand")
    if not exp:
        return [law]

    def domain(d):
        if isinstance(d, str) and ".." in d:
            a, b = d.split("..")
            return [str(i) for i in range(int(a), int(b) + 1)]
        return [str(v) for v in d]

    keys = list(exp.keys())
    return emit(keys, list(itertools.product(*[domain(exp[k]) for k in keys])))


def _render_dl(db):
    """YAML legal database -> Governatori DDL .dl text (one line per law / superiority).
    Each law: `<label>: <antecedent...> => <consequent...>` where literals may be plain (x),
    negated (~x), or deontic ([O]x, [P]x, [O]~x, ~[O]x). A leading [O]/[P] in the consequent
    makes the rule prescriptive/permissive; multiple [O] consequents form a compensation
    (contrary-to-duty) chain; no deontic operator => a constitutive ('counts-as') rule.

    `laws` may be a flat list, or a dict grouping laws into named categories
    (e.g. {geometric_laws: [...]}); categories are organisational only and are flattened
    here — the DDL theory is the union of every law across every group."""
    lines = []
    laws = db.get("laws", []) or []
    if isinstance(laws, dict):                       # {category: [law, ...]} -> flat list
        laws = [law for group in laws.values() for law in (group or [])]
    for law in laws:
        for g in _expand_law(law):
            ante = ", ".join(g.get("antecedent", []) or [])
            cons = ", ".join(g["consequent"])
            lines.append(f"{g['label']}: {ante} => {cons}")
    for pair in db.get("superiority", []) or []:
        lines.append(f"{pair[0]} > {pair[1]}")
    return "\n".join(lines)


class LegislativeReasoner:
    """Compiles the legal database once; queries it against a (perceived) state."""

    def __init__(self, db_path=_DEFAULT_DB, ddl_root=_DDL_ROOT):
        self.ddl_root = Path(ddl_root)
        self.db = load_legal_database(db_path)
        p = ddl_parser.DDLParser()
        p.parse(_render_dl(self.db))
        atom_decls = "\n".join(f"atom({a})." for a in (self.db.get("atoms", []) or []))
        self.theory_asp = atom_decls + "\n" + p.get_output()
        # assess() is a PURE function of `facts` (the theory is fixed here) and fact sets recur heavily
        # across a sweep -> memoize by frozenset(facts) to skip re-running clingo, which otherwise
        # RELOADS the 5 engine .asp files from disk + re-grounds the whole theory on EVERY call.
        self._cache = {}

    def _solve(self, facts, predicates):
        """Run the DDL engine on theory + the given ground facts; return the first answer
        set's atoms whose predicate name is in `predicates`, grouped by name (as Symbols)."""
        ctl = clingo.Control(["0", "--warn=no-atom-undefined"])
        for ef in _ENGINE_FILES:
            ctl.load(str(self.ddl_root / ef))
        ctl.add("base", [], self.theory_asp)
        ctl.add("base", [], "".join(f"fact({lit})." for lit in facts))
        ctl.ground([("base", ())])
        res = {p: [] for p in predicates}
        for model in ctl.solve(yield_=True):
            for sym in model.symbols(atoms=True):
                if sym.name in res:
                    res[sym.name].append(sym)
            break  # determinate theory + fixed facts -> single answer set
        return res

    def derive(self, facts, predicates=None):
        """General query. `facts`: ground literals true in the (perceived/predicted) state.
        `predicates`: engine predicate names to collect (default: the deontic verdicts; pass
        e.g. ['obligation','defeasible'] to also surface constitutive/derived facts). Returns
        {predicate: [atom strings]}."""
        predicates = tuple(predicates) if predicates else _DEONTIC_PREDICATES
        return {p: sorted(str(s) for s in v) for p, v in self._solve(facts, predicates).items()}

    def assess(self, facts):
        """Structured, content-agnostic deontic verdict of a state. Splits the net obligations
        into positive obligations O(X) and prohibitions O(~X)->X, plus permissions and
        violations. The literals are arbitrary (geometric or not). What each verdict DOES to
        the planner is intentionally NOT decided here."""
        key = frozenset(facts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        res = self._solve(facts, _DEONTIC_PREDICATES)

        def lits(symbols):  # 1-ary verdicts only (skip internal obligation(Rule,X,N) form)
            return [s.arguments[0] for s in symbols if len(s.arguments) == 1]

        obligations, prohibitions = [], []
        for c in lits(res["obligation"]):
            if c.name == "non" and len(c.arguments) == 1:
                prohibitions.append(str(c.arguments[0]))   # O(~X) -> X prohibited
            else:
                obligations.append(str(c))                 # O(X)  -> X obligatory
        result = {
            "obligations": sorted(set(obligations)),
            "prohibitions": sorted(set(prohibitions)),
            "permissions": sorted({str(c) for c in lits(res["permission"])}),
            "violations": sorted({str(c) for c in lits(res["violation"])}),
        }
        self._cache[key] = result
        return result


if __name__ == "__main__":
    r = LegislativeReasoner()
    print("[theory ASP]\n" + r.theory_asp + "\n")
    for state in (["cube"], ["cube", "in_cell(4)"], ["cube", "in_cell(1)"]):
        print(f"state {state} -> {r.assess(state)}")
