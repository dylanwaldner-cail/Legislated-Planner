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
          visited(C) : the cube's TRUE FOOTPRINT overlapped cell C at some rest step AFTER spawn (union of
                       the GT `occupies` fact over records[1:]). Derived from the AUTHORITY's ground-truth
                       FOOTPRINT occupancy (grounding.occupies, center +/- cube_half -- same geometry as the
                       swept abidance metric, so taint and metric agree on "entered 4"), NOT the probe
                       in_cell -- the sign taint (R7b visited(4)->red) must not be triggerable by perception
                       error (see grounding.occupies). NOTE: occupies is sampled at REST frames (1/observe),
                       so a footprint that sweeps 4 purely MID-STROKE and rests clear is not yet caught here
                       -- see enforcement TODO if full swept-transit taint is wanted. Frame 0 is
                       GRANDFATHERED -- the spawn cell must not
                       taint the trajectory, matching the frame-0 spawn grandfather in constraint.py
                       (skip_current) and planning_metrics.py ("drop frame 0 (spawn); count frames 1..").
        Backward-compatible: laws that don't reference these predicates simply ignore them."""
        visited = set()
        for rec in self.records[1:]:                                 # frame-0 spawn grandfather
            for f in rec["facts"]:
                if f.startswith("occupies("):
                    visited.add("visited(" + f[len("occupies("):])  # GT rest-footprint occupies(4) -> visited(4)
                elif f.startswith("passed_through("):
                    visited.add("visited(" + f[len("passed_through("):])  # GT SWEPT transit through 4 -> visited(4)
        return sorted(visited)

    def intrusions(self, cell=4):
        """Structured record of every executed step where the cube was in `cell`, TAGGED with whether
        that cell was actually PROHIBITED in the verdict at that step -- so "entered while banned" (a real
        violation) is trivially separable from "entered under a live permission" (e.g. the green light,
        where green_sign_permits_center defeats [O]~in_cell(4)). This is the before/after-permission cut.

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
