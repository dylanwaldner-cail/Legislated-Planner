"""Sign-colour control for the MPC rollout — extracted from planning/mpc.py (behaviour-preserving).

`SignController` owns the octagonal rule sign's colour across ONE MPC episode and folds together the two
ways it can change:

  1. EXOGENOUS flip schedule (conf `sign_flip`): the sign becomes `color` at MPC step `frame`; before
     that it holds `base_color`. Recolouring the exec env BEFORE step (frame-1)'s rollout puts the new
     colour in that step's executed frame, which becomes the next step's observation -> the legislation
     re-perceives it exactly at `frame`. A scripted stimulus, not a decision.

  2. DDL RENDER-BACK: once full_lawset's R7/R7b conclude a colour for an eval (yellow -> green when the
     trajectory is clean, yellow -> red when it already visited the forbidden cell), that DERIVED sign is
     rendered back to that eval's env so the flip is PHYSICAL and PERSISTS: the encoded sign_color probe
     re-perceives it next step, so "green permits centre / red freezes" holds after the cube leaves the
     yellow cell. Per-eval, so parallel evals can hold different signs.

The controller is INERT unless a sign_flip schedule is configured AND the env can recolour (grid env
only). Standard law_eval / oracle runs leave `sign_flip.frame=null`, so `active` is False and nothing
here allocates or touches a sign -- the sign machinery is entirely behind the sign_flip flag.
"""
from __future__ import annotations


class SignController:
    """Per-episode sign-colour state (exogenous flip schedule + DDL render-back latch).

    sign_flip : the conf `sign_flip` mapping (or None). Reads `frame` (int MPC step | None -> inactive),
        `color` (flip target, default yellow), `base_color` (pre-flip colour, default white).
    env       : the exec env (needs `set_sign_color`, which accepts a single colour or a per-eval list).
    law_fn    : the sub-planner's LawEvaluator (holds each eval's reasoner-derived effective sign) or None.
    n_evals   : batch size, for the per-eval colour vector.
    """

    def __init__(self, sign_flip, env, law_fn, n_evals):
        self.env = env
        self.law_fn = law_fn
        self.frame = None
        self._colors = None          # per-eval effective-sign vector; None => inactive (no allocation)
        if sign_flip and sign_flip.get("frame") is not None and hasattr(env, "set_sign_color"):
            self.frame = int(sign_flip["frame"])
            self.flip_color = sign_flip.get("color", "yellow")
            self.base_color = sign_flip.get("base_color", "white")
            initial = self.flip_color if self.frame == 0 else self.base_color
            env.set_sign_color(initial)                      # pre-loop colour (frame 0 flips immediately)
            self._colors = [initial] * n_evals

    @property
    def active(self) -> bool:
        """True iff a sign_flip schedule is configured on a recolourable env."""
        return self._colors is not None

    def on_step_pre_roll(self, iteration, n_evals):
        """Recolour BEFORE this step's exec rollout so the executed frame carries the new colour and the
        NEXT step re-perceives it. (1) exogenous: at iter == frame-1 flip the whole sign to `flip_color`;
        (2) DDL render-back: latch any eval whose reasoner just concluded green/red (R7/R7b) -- only
        green/red override, so evals still on white/yellow keep the exogenous colour. No-op if inactive."""
        if not self.active:
            return
        if iteration == self.frame - 1:                      # exogenous flip -> perceived at step `frame`
            self.env.set_sign_color(self.flip_color)
            self._colors = [self.flip_color] * n_evals
        if self.law_fn is not None:                          # DDL render-back (per-eval latch)
            changed = False
            for e in range(n_evals):
                cs = self.law_fn.current_sign(e)
                if cs in ("green", "red") and self._colors[e] != cs:
                    self._colors[e] = cs
                    changed = True
            if changed:
                self.env.set_sign_color(list(self._colors))
