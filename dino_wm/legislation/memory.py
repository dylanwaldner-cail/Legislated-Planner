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
        self.resolved_sign = None   # first TERMINAL (green/red) effective sign -> LATCHED thereafter (one-shot)

    def append(self, facts, verdict, gt_xy=None):
        """gt_xy: the executed GROUND-TRUTH cube xy at this rest step (authority). Stored so the SWEPT
        taint can sweep the footprint between consecutive executed rests (enforcement.observe)."""
        self.records.append({"step": len(self.records), "facts": list(facts), "verdict": verdict,
                             "gt_xy": (list(map(float, gt_xy)) if gt_xy is not None else None)})

    def record_sign(self, color):
        """Log the EFFECTIVE sign colour for this step: the perceived colour, or a DERIVED flip
        (e.g. R7 yellow->green) when the laws conclude a new colour. Deduped against the previous
        entry, so `signs` is the change-history and its LAST element is always the current sign --
        that is what the environment renders (grid_venv.set_sign_color). No-op on None/unchanged.

        LATCH: green/red is a ONE-SHOT resolution (reaching the yellow cell while the sign is yellow ->
        green if clean, red if the trajectory already visited the forbidden cell). Once concluded it must
        PERSIST -- enforcement.observe feeds resolved_sign back to the reasoner on every later step so a
        probe misread or a re-graze re-firing R7b can't overturn a settled sign. white/yellow are
        non-terminal (the pre-resolution states) and never latch."""
        if color and (not self.signs or self.signs[-1] != color):
            self.signs.append(color)
        if color in ("green", "red") and self.resolved_sign is None:
            self.resolved_sign = color

    def clear_resolved_sign(self, color=None):
        """AUTHORITY RESET: unlatch the one-shot green/red resolution so a newly perceived colour is
        believed again. Records `color` as the new effective sign if given.

        This is the ONE legitimate way past the latch, and the distinction is deliberate. The latch
        exists to stop PERCEPTION from re-opening a settled verdict -- a probe misread, or a corner
        graze re-firing R7b. It was written on the assumption that perception is the only thing that
        can change the sign. An EXOGENOUS recolour by the environment is a different kind of event:
        the authority that set the sign green has itself reset it, so there is no settled verdict left
        to protect. Only the sign-flip schedule (planning/sign_control.py) may call this -- never a
        grounder, and never a rule."""
        self.resolved_sign = None
        if color:
            self.record_sign(color)

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
          visited(C) : the cube's TRUE FOOTPRINT was in cell C at some point after spawn -- the union over
                       records[1:] of TWO ground-truth facts, both folded into the same visited(C):
                         occupies(C)       : REST-frame footprint (grounding.occupies, sampled 1/observe)
                         passed_through(C) : the SWEPT transit between rests (enforcement.observe), so a
                                             footprint that crosses C purely MID-STROKE and rests clear is
                                             caught too. This is the majority channel in practice -- on
                                             aug20/aug28 a few hundred steps per run carry
                                             passed_through(4) with no occupies(4).
                       Both are the AUTHORITY's ground truth (center +/- cube_half -- the same geometry as
                       the swept abidance metric, so taint and metric agree on "entered 4"), NOT the probe
                       in_cell: the sign taint (R7b visited(4)->red) must not be triggerable by perception
                       error (see grounding.occupies). visited is MONOTONE -- a set union that only grows,
                       so a taint is permanent for the episode (what makes R7b a terminal sanction).
                       Recomputed from records on every call (not cached), so enforcement.observe() must
                       append the current step BEFORE calling this or visited lags a step.
                       Frame 0 is GRANDFATHERED -- the spawn cell must not
                       taint the trajectory, matching the frame-0 spawn grandfather in constraint.py
                       (skip_current) and planning_metrics.py ("drop frame 0 (spawn); count frames 1..").
                       NB this records[1:] slice grandfathers the spawn FRAME; the spawn's own escape
                       TRANSIT is grandfathered separately, per-cell, in enforcement.observe(). Both are
                       load-bearing -- they cover different events.
        Backward-compatible: laws that don't reference these predicates simply ignore them."""
        visited = set()
        for rec in self.records[1:]:                                 # frame-0 spawn grandfather
            for f in rec["facts"]:
                if f.startswith("occupies("):
                    visited.add("visited(" + f[len("occupies("):])  # GT rest-footprint occupies(4) -> visited(4)
                elif f.startswith("passed_through("):
                    visited.add("visited(" + f[len("passed_through("):])  # GT SWEPT transit through 4 -> visited(4)
        # NB start_cell(C) is deliberately NOT derived here. It was, briefly, off record 0's probe
        # in_cell -- but in_cell is MULTILABEL, so a cube spawning across a boundary emitted TWO
        # start_cell facts, the injected law concluded two return duties, and the agent went home to
        # the wrong cell. Like goal_cell, the start cell is a task SPECIFICATION rather than a
        # perception, so it is set from the benchmark's metadata in LawEvaluator.set_start().
        return sorted(visited)

    def intrusions(self, cell=4):
        """Structured record of every executed step where the cube RESTED in `cell`, TAGGED with whether
        that cell was actually PROHIBITED in the verdict at that step -- so "entered while banned" (a real
        violation) is trivially separable from "entered under a live permission" (e.g. the green light,
        where green_sign_permits_center defeats [O]~in_cell(4)). This is the before/after-permission cut.

        REST FRAMES ONLY -- and that is a real limit, not a detail. The scan below keys off occupies(cell)
        and in_cell(cell), both rest-sampled, so a stroke that SWEEPS `cell` mid-transit and comes to rest
        outside it is skipped entirely (it has passed_through(cell) but neither of the two). Those steps
        still taint visited() and still count against the swept abidance metric. Do NOT read this as the
        episode's full intrusion history, and do NOT use gt_in/probe_in as the probe-vs-GT divergence on
        entering the cell -- it is the divergence on RESTING there, over the subset of steps where at
        least one of the two fired. For the swept channel there is no probe counterpart to compare against
        at all: passed_through is GT-only by construction.

        Per frame: `gt_in` (GT authority occupies(cell)), `probe_in` (perception in_cell(cell)), the
        governing `sign` (the effective/latched colour the verdict was assessed under -- read this to see
        the colour that decided the entry's legality), `sign_raw` (the raw perceived sign that step, for
        perception debugging), and `prohibited` (was in_cell(cell) in the verdict's prohibitions THAT
        step). Spawn (record 0) is grandfathered out (matches visited()/skip_current). The headline counts
        key off GT occupancy (the authority): `gt_banned` = real violations; `gt_permitted` = legal passes."""
        occ, inc = f"occupies({cell})", f"in_cell({cell})"
        frames = []
        for rec in self.records[1:]:                                  # frame-0 spawn grandfather
            gt, probe = occ in rec["facts"], inc in rec["facts"]
            if not (gt or probe):
                continue
            prohib = (rec.get("verdict") or {}).get("prohibitions") or []
            raw = next((f[len("sign("):-1] for f in rec["facts"] if f.startswith("sign(")), None)
            eff = rec.get("effective_sign", raw)                       # GOVERNING colour (latched/derived), falls back to raw
            frames.append({"step": rec["step"], "sign": eff, "sign_raw": raw,
                           "gt_in": gt, "probe_in": probe, "prohibited": inc in prohib})
        return {
            "cell": cell,
            "frames": frames,
            "gt_banned": sum(1 for f in frames if f["gt_in"] and f["prohibited"]),      # REAL violations
            "gt_permitted": sum(1 for f in frames if f["gt_in"] and not f["prohibited"]),  # legal (e.g. green)
        }

    def __len__(self):
        return len(self.records)
