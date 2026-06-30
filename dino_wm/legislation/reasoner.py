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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _render_dl(db):
    """YAML legal database -> Governatori DDL .dl text (one line per law / superiority).
    Each law: `<label>: <antecedent...> => <consequent...>` where literals may be plain (x),
    negated (~x), or deontic ([O]x, [P]x, [O]~x, ~[O]x). A leading [O]/[P] in the consequent
    makes the rule prescriptive/permissive; multiple [O] consequents form a compensation
    (contrary-to-duty) chain; no deontic operator => a constitutive ('counts-as') rule."""
    lines = []
    for law in db.get("laws", []) or []:
        ante = ", ".join(law.get("antecedent", []) or [])
        cons = ", ".join(law["consequent"])
        lines.append(f"{law['label']}: {ante} => {cons}")
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
        res = self._solve(facts, _DEONTIC_PREDICATES)

        def lits(symbols):  # 1-ary verdicts only (skip internal obligation(Rule,X,N) form)
            return [s.arguments[0] for s in symbols if len(s.arguments) == 1]

        obligations, prohibitions = [], []
        for c in lits(res["obligation"]):
            if c.name == "non" and len(c.arguments) == 1:
                prohibitions.append(str(c.arguments[0]))   # O(~X) -> X prohibited
            else:
                obligations.append(str(c))                 # O(X)  -> X obligatory
        return {
            "obligations": sorted(set(obligations)),
            "prohibitions": sorted(set(prohibitions)),
            "permissions": sorted({str(c) for c in lits(res["permission"])}),
            "violations": sorted({str(c) for c in lits(res["violation"])}),
        }


if __name__ == "__main__":
    r = LegislativeReasoner()
    print("[theory ASP]\n" + r.theory_asp + "\n")
    for state in (["cube"], ["cube", "in_cell(4)"], ["cube", "in_cell(1)"]):
        print(f"state {state} -> {r.assess(state)}")
