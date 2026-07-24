"""State-dependent law evaluation: perceive -> ground -> reason -> constraint, EACH step, PLUS a
per-eval NORMATIVE MEMORY (ledger) of the executed trajectory.

`observe(z, eval_index)` is the per-step entry point: it perceives the current frame, records the
grounded facts + verdict in that eval's ledger, and re-derives the Constraint over
(base + current-perception + the ledger's history-derived facts). The ledger is what makes verdicts
TEMPORAL (CTD / "already passed through X"); it lives HERE, not in the planner, so it is portable
across planners. The planner stays stateless about law -- it just calls observe() and prunes on the
returned Constraint.

    ev = LawEvaluator(LegislativeReasoner(), ProbeRegistry(device=...))
    con = ev.observe(root_visual_latent, eval_index=0)     # (1, P, D) ENCODED root frame
"""
from .grounding import Grounder
from .constraint import Constraint
from .memory import NormativeMemory


class LawEvaluator:
    """Per-step law pipeline (probe -> ground -> reason -> Constraint) + a per-eval ledger."""

    def __init__(self, reasoner, registry, base_facts=("cube",), source="encoded", **constraint_kw):
        self.reasoner = reasoner
        self.registry = registry
        self.base_facts = list(base_facts)     # always-true facts (e.g. "cube")
        self.source = source                   # which probes to run for grounding (encoded = phi(obs))
        self.constraint_kw = constraint_kw     # cube_half / occ_thresh forwarded to Constraint
        self.ledgers = {}                      # eval_index -> NormativeMemory (per-eval executed history)
        self.goal_facts = {}                   # eval_index -> ['goal_cell(k)'] (spec if gt_goal_cell set, else perceived)
        self.gt_goal_cell = None               # int -> goal cell is a GIVEN task spec; set_goal skips perception (see set_goal)

    def reset(self):
        """New episode: clear every per-eval ledger + perceived goals."""
        self.ledgers = {}
        self.goal_facts = {}

    def set_goal(self, goal_visual_latent, eval_index=0):
        """Perceive the GOAL frame for eval `eval_index` on the SAME probe stack as live perception
        (probes run on the goal latent) -> goal_cell(k), held as a per-eval base fact for the whole
        episode. This is why the goal obligation is GROUNDED (probe-derived), not read from
        privileged sim geometry. Idempotent: the goal is constant per episode, so later re-plans skip
        the re-perceive (reset() clears goal_facts at episode start)."""
        if eval_index in self.goal_facts:                # already set this eval's goal -> skip
            return
        if self.gt_goal_cell is not None:
            # GOAL AS SPECIFICATION: the obligated goal cell is a GIVEN (a human task spec, like a
            # prompt), taken from ground truth -- NOT perceived. Perceiving it via the cube_cells
            # argmax flips near cell boundaries (the goal footprint straddles two cells), and the
            # obligation is a single designated target, so we take it as given. Current-STATE facts
            # (where the cube IS now) stay probe-grounded; only the goal target is specified.
            self.goal_facts[eval_index] = [f"goal_cell({int(self.gt_goal_cell)})"]
            return
        out = self.registry.forward(goal_visual_latent, source=self.source)
        self.goal_facts[eval_index] = Grounder(out).goal_cell()

    def ledger(self, eval_index):
        """The ledger for one eval (created on first use)."""
        if eval_index not in self.ledgers:
            self.ledgers[eval_index] = NormativeMemory()
        return self.ledgers[eval_index]

    def _perceive(self, visual_latent):
        """Run the probes on the current frame + ground them -> current-state DDL facts."""
        return Grounder(self.registry.forward(visual_latent, source=self.source)).ground()

    def _build_constraint(self, verdict):
        return Constraint(verdict["prohibitions"], self.registry.probes,
                          obligations=verdict["obligations"], permissions=verdict["permissions"],
                          **self.constraint_kw)

    def observe(self, visual_latent, eval_index=0):
        """One EXECUTED step for eval `eval_index`: perceive the current frame -> record the facts +
        verdict in the ledger -> re-derive the Constraint over (base + current + history-derived
        facts). Returns the Constraint for this eval's current state."""
        led = self.ledger(eval_index)
        current = self._perceive(visual_latent)
        led.append(current, None)                                  # record perceived facts (verdict backfilled below)
        # base + PERCEIVED GOAL (goal_cell(k)) + current + history-derived facts. The goal fact
        # activates the reach_goal_k obligation, letting it interact with the cell laws.
        facts = (self.base_facts + self.goal_facts.get(eval_index, [])
                 + current + led.derived_facts())                  # derived includes this step -> TEMPORAL scope
        verdict = self.reasoner.assess(facts)
        led.records[-1]["verdict"] = verdict
        # EFFECTIVE sign colour: one the laws CONCLUDE but that wasn't perceived (e.g. R7 yellow->green)
        # is a derived FLIP and wins; else the perceived colour. Logged in the ledger so the env can
        # recolour the physical sign mid-run (grid_venv.set_sign_color) -- see current_sign().
        perceived = {f[len("sign("):-1] for f in current if f.startswith("sign(") and f.endswith(")")}
        derived = [c for c in verdict.get("signs", []) if c not in perceived]
        led.record_sign(derived[0] if derived else (sorted(perceived)[0] if perceived else None))
        return self._build_constraint(verdict)

    def current_sign(self, eval_index=0):
        """Effective sign colour for this eval's latest observed step -- the perceived colour, or a
        DERIVED flip such as R7's yellow->green. The environment reads this to recolour the physical
        sign mid-run: `venv.set_sign_color(law_eval.current_sign(i))`. None if no sign observed yet."""
        return self.ledger(eval_index).last_sign()

    def commit(self, eval_index, data):
        """Record the agent's INTENT for eval `eval_index` (POST-HOC only; never read during planning
        or reasoning). Attaches to the latest ledger record. See NormativeMemory.commit."""
        self.ledger(eval_index).commit(data)

    def dump(self, path):
        """Write all per-eval ledgers (observed facts + verdicts + committed intents) to JSON for
        post-hoc analysis. Records are already JSON-serializable (strings / lists / numbers)."""
        import json
        with open(path, "w") as f:
            json.dump({str(e): led.records for e, led in self.ledgers.items()}, f, indent=2)
