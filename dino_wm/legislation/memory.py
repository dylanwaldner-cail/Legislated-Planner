"""Per-eval NORMATIVE MEMORY: the ledger of the executed trajectory in deontic terms.

One StepRecord per real (executed) step -- the FACTS perceived + the VERDICT they produced. This is
the persistent state temporal / contrary-to-duty laws read ("already passed through 4", "crossed
twice", "having entered X you must now Y"). It lives in the legislation layer, NOT the planner, so
it is portable across planners. FACTS ONLY -- no latents; the ledger is the derived, symbolic record,
and re-grounding always uses the live probe on the current frame.
"""


class NormativeMemory:
    """Append-only ledger of executed steps for ONE eval."""

    def __init__(self):
        self.records = []   # list of {"step": int, "facts": [str], "verdict": dict}
        self.signs = []     # chronological EFFECTIVE sign colours; last element == current sign

    def append(self, facts, verdict):
        self.records.append({"step": len(self.records), "facts": list(facts), "verdict": verdict})

    def record_sign(self, color):
        """Log the EFFECTIVE sign colour for this step: the perceived colour, or a DERIVED flip
        (e.g. R7 yellow->green) when the laws conclude a new colour. Deduped against the previous
        entry, so `signs` is the change-history and its LAST element is always the current sign --
        that is what the environment renders (grid_venv.set_sign_color). No-op on None/unchanged."""
        if color and (not self.signs or self.signs[-1] != color):
            self.signs.append(color)

    def last_sign(self):
        """Current effective sign colour (last one recorded), or None if never set."""
        return self.signs[-1] if self.signs else None

    def commit(self, data):
        """Attach the agent's INTENT (chosen action + predicted outcome) to the latest step record.
        POST-HOC ONLY -- never read during planning. It lets offline analysis compare what the agent
        INTENDED / PREDICTED at step t against the ACTUAL outcome recorded (as observed facts) at
        step t+1: foreseeability, side-effect attribution, knowing-vs-accidental violation."""
        if self.records:
            self.records[-1]["committed"] = data

    def last_facts(self):
        return self.records[-1]["facts"] if self.records else []

    def derived_facts(self):
        """Sticky / aggregate facts implied by the WHOLE history so far -- what turns a state-only
        verdict into a TEMPORAL one. EXTEND this method to add temporal predicates (counts, decay,
        CTD triggers...). Currently:
          visited(cube,C) : the cube was in cell C at some past-or-current step (union of in_cell).
        Backward-compatible: laws that don't reference these predicates simply ignore them."""
        visited = set()
        for rec in self.records:
            for f in rec["facts"]:
                if f.startswith("in_cell("):
                    visited.add("visited(" + f[len("in_cell("):])   # in_cell(cube,4) -> visited(cube,4)
        return sorted(visited)

    def __len__(self):
        return len(self.records)
