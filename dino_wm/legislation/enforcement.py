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
import time
import numpy as np
from .grounding import Grounder
from .constraint import Constraint
from .memory import NormativeMemory
from probes.probe_cube_cells import CUBE_HALF, swept_cells   # GT swept-footprint taint (matches abidance metric)


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
        self._gt_cube = {}                     # eval_index -> GT cube xy (authority input for the sign; see set_gt_cube)
        self.timing = self._zero_timing()      # per-episode wall-clock split of observe() (see Q5 runtime report)

    @staticmethod
    def _zero_timing():
        """Fine-grained wall-clock accumulators for the LEGISLATION reason path (observe()), summed over
        every eval x re-plan step in an episode. Splits what runtime_breakdown lumps as legislation_reason_s
        into: probe_s (perception forward), ground_s (symbol grounding + swept-taint geometry), logic_s
        (clingo DDL assess), build_s (Constraint construction). n_observe = # observe() calls (for per-call
        means). The prune path (constraint.violations) is timed separately in the planner (_t_prune)."""
        return {"probe_s": 0.0, "ground_s": 0.0, "logic_s": 0.0, "build_s": 0.0, "n_observe": 0}

    def reset(self):
        """New episode: clear every per-eval ledger + perceived goals + runtime accounting."""
        self.ledgers = {}
        self.goal_facts = {}
        self.timing = self._zero_timing()

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

    def set_gt_cube(self, positions):
        """AUTHORITY input: the env's GROUND-TRUTH cube xy per eval (env-local), pushed by the MPC loop
        each step BEFORE planning. observe() grounds the SIGN's constitutive predicates (occupies ->
        in_yellow_cell / visited) from this, NOT the probe -- the sign is an external control signal and
        must not be flippable by perception error. `positions`: (N,>=2) array-like (state cube xy)."""
        import numpy as np
        p = np.asarray(positions, dtype=float)
        self._gt_cube = {i: p[i] for i in range(p.shape[0])}

    def _perceive(self, visual_latent, gt_cube_xy=None):
        """Run the probes on the current frame + ground them -> current-state DDL facts. gt_cube_xy (the
        env's ground-truth cube position, optional) grounds the SIGN's occupies() predicate on TRUTH;
        all other facts (in_cell, sign colour, ...) stay probe-derived."""
        _t = time.perf_counter()
        out = self.registry.forward(visual_latent, source=self.source)   # PROBES: perception forward pass
        self.timing["probe_s"] += time.perf_counter() - _t
        _t = time.perf_counter()
        facts = Grounder(out, gt_cube_xy=gt_cube_xy).ground()            # GROUNDING: probe reads -> DDL facts (geometry)
        self.timing["ground_s"] += time.perf_counter() - _t
        return facts

    def _build_constraint(self, verdict):
        return Constraint(verdict["prohibitions"], self.registry.probes,
                          obligations=verdict["obligations"], permissions=verdict["permissions"],
                          **self.constraint_kw)

    def observe(self, visual_latent, eval_index=0):
        """One EXECUTED step for eval `eval_index`: perceive the current frame -> record the facts +
        verdict in the ledger -> re-derive the Constraint over (base + current + history-derived
        facts). Returns the Constraint for this eval's current state."""
        led = self.ledger(eval_index)
        gt_xy = self._gt_cube.get(eval_index)
        current = self._perceive(visual_latent, gt_xy)             # sign occupancy from GT (authority): rest FOOTPRINT
        # SWEPT taint: passed_through(c) for every cell the TRUE footprint crossed on the transit that just
        # completed (previous executed rest -> this rest). Fed into visited() (memory.derived_facts) so a
        # mid-stroke drive-THROUGH cell 4 taints the history even when the cube rests clear -- matching the
        # swept-footprint abidance metric (the rest-only occupies would miss it). GT authority, NOT the probe.
        # The spawn's own escape transit is grandfathered PER-CELL (drop cells the spawn footprint already
        # occupied) -- matches the frame-0 grandfather in the metric / constraint (skip_current).
        if gt_xy is not None and led.records:
            _t = time.perf_counter()
            prev = led.records[-1]
            prev_xy = prev.get("gt_xy")
            if prev_xy is not None:
                grand = ({f[len("occupies("):-1] for f in prev["facts"] if f.startswith("occupies(")}
                         if len(led.records) == 1 else set())      # only the spawn->first-rest transit
                swept = np.where(swept_cells(np.asarray(prev_xy, float),
                                             np.asarray(gt_xy, float), CUBE_HALF))[0]
                for c in swept:
                    if str(int(c)) in grand:
                        continue                                   # spawn cell's one escape stroke
                    f = f"passed_through({int(c)})"
                    if f not in current:
                        current.append(f)
            self.timing["ground_s"] += time.perf_counter() - _t    # swept-taint geometry counts as grounding
        led.append(current, None, gt_xy=gt_xy)                     # record RAW perceived facts (verdict backfilled below)
        self.timing["n_observe"] += 1

        # SIGN LATCH: the yellow->green/red flip is a ONE-SHOT resolution (reaching the yellow cell). Once it
        # has resolved to a terminal colour, FREEZE it -- feed the latched colour to the reasoner instead of
        # the freshly-perceived sign, so a later frame (a probe misread, or a corner-graze re-firing R7b)
        # cannot re-open a settled green/red. The RAW perceived sign stays in the recorded facts (above) for
        # debugging; only what the laws SEE here is overridden. See NormativeMemory.resolved_sign.
        sign_facts = current
        if led.resolved_sign is not None:
            sign_facts = [f for f in current if not f.startswith("sign(")] + [f"sign({led.resolved_sign})"]

        # base + PERCEIVED GOAL (goal_cell(k)) + current + history-derived facts. The goal fact
        # activates the reach_goal_k obligation, letting it interact with the cell laws.
        facts = (self.base_facts + self.goal_facts.get(eval_index, [])
                 + sign_facts + led.derived_facts())               # derived includes this step -> TEMPORAL scope
        _t = time.perf_counter()
        verdict = self.reasoner.assess(facts)                      # LOGIC: clingo DDL solve (memoized per fact-set)
        self.timing["logic_s"] += time.perf_counter() - _t
        led.records[-1]["verdict"] = verdict

        # EFFECTIVE sign colour: one the laws CONCLUDE but that wasn't perceived (e.g. R7 yellow->green)
        # is a derived FLIP and wins; else the perceived colour. Logged in the ledger so the env can
        # recolour the physical sign mid-run (grid_venv.set_sign_color) -- see current_sign(). Once
        # resolved_sign is latched, sign_facts carries it, so this stays pinned to the resolved colour.
        perceived = {f[len("sign("):-1] for f in sign_facts if f.startswith("sign(") and f.endswith(")")}
        derived = [c for c in verdict.get("signs", []) if c not in perceived]
        eff_sign = derived[0] if derived else (sorted(perceived)[0] if perceived else None)
        led.record_sign(eff_sign)
        # The GOVERNING sign this step -- the latched/derived colour the verdict was ACTUALLY assessed
        # under (not the raw perceived sign(...) left in records[-1]["facts"]). intrusions() reports it so
        # a cell-4 entry can be read against the colour that decided its legality. The raw perceived sign
        # stays in the recorded facts for perception debugging.
        led.records[-1]["effective_sign"] = eff_sign
        _t = time.perf_counter()
        con = self._build_constraint(verdict)                      # BUILD: assemble the per-candidate Constraint
        self.timing["build_s"] += time.perf_counter() - _t
        return con

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
        """Write all per-eval ledgers to JSON for post-hoc analysis. Per eval:
          `records`      : one entry per executed step -- the observed `facts` (incl. in_cell(C),
                           visited(C), sign(colour), yellow_cell(C)) + the reasoner `verdict`
                           (prohibitions/obligations/permissions/violations + provable `signs`) +
                           any committed intent. Read the facts across steps to see WHEN the cube
                           entered the illegal cell relative to reaching a yellow cell.
          `sign_history` : the EFFECTIVE-sign change-history (perceived, or the DDL-derived flip such
                           as R7 yellow->green / R7b yellow->red) -- the actual flip decisions rendered
                           back to the env, deduped to changes. Empty when no sign was ever observed.
          `intrusions`   : cell-4 entries split by permission (NormativeMemory.intrusions) -- per-frame
                           gt_in/probe_in/sign/prohibited, plus gt_banned (GT in 4 while PROHIBITED = a
                           real violation) vs gt_permitted (GT in 4 under a live permission, e.g. green).
                           The before/after-permission cut: read gt_banned for true law abidance.
        Everything is already JSON-serializable (strings / lists / numbers)."""
        import json
        out = {str(e): {"records": led.records, "sign_history": led.signs,
                        "intrusions": led.intrusions(4)}          # cell-4 entries split by before/after permission
               for e, led in self.ledgers.items()}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
