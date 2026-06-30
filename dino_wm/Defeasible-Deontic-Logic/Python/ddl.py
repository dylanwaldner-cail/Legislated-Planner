import clingo
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # Defeasible-Deontic-Logic/ (this file lives in Python/)
sys.path.insert(0, str(_ROOT / "Python"))        # so `import parser` resolves to the DDL parser
import parser

ctl = clingo.Control(["0", "--warn=no-atom-undefined"])

engine_files = [
    "language.asp",
    "Basic/language.asp",
    "Deontic/language.asp",
    "Deontic/defeasible-ab.asp",
    # "Deontic/deontic.asp",
    "Deontic/deontic-comp.asp"
]

rule_files = [
    # "Examples/ambiguity.dl"
    "Examples/exceptions.dl"   # repointed: deontic-test.dl is not in this checkout
]

for file in engine_files:
    ctl.load(str(_ROOT / file))

for f in rule_files:
    p = parser.DDLParser()
    with open(_ROOT / f, "r") as theory:
        content = theory.read()

    print(content)

    p.parse(content)   # (was double-read: the first parse got an already-consumed file -> "")

    with open("simple.lp", "w") as output_file:
        output_file.write(p.get_output())

    ctl.add("base", [], p.get_output())

# ctl.load("Examples/output.lp")   # removed: file not present in this checkout
# ctl.load("Deontic/debug.lp")

ctl.ground([("base", ())])

modelNo = 1

for model in ctl.solve(yield_=True):
    print(f"\nModel: {modelNo}")
    modelNo += 1
    for symbol in model.symbols(shown=True):
        print(symbol)
        # if symbol.name == "obligation":
        #     print (f"--> There is an obligation for {symbol.arguments[0]}")
        # if symbol.name == "refuted":
        #     if symbol.arguments[0].name == "non":
        #         print (f"~{symbol.arguments[0].arguments[0]}")

times = ctl.statistics['summary']['times']
print(f"\nTotal: {times['total']:.3f}")
print(f"CPU:   {times['cpu']:.3f}")

print(f"Atoms: {ctl.statistics['problem']['lp']['atoms']:.0f}")

print(f"Rules: {ctl.statistics['problem']['lp']['rules']:.0f}") 