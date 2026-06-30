"""Aimed-contact CEM planner (experimental).

Subclasses the stock CEMPlanner WITHOUT touching its optimization logic. The only
change is the action reparametrization: instead of letting the CEM choose the stroke
START freely (a 4-D box where most samples MISS the cube -- start beside/ahead of it),
we let it optimize ONLY the push displacement and DERIVE the start so the blade always
sits behind the cube and pushes through it. Misses become geometrically unrepresentable.

Formalism (see also the planner-objective notes):

    a = [s, d]            start s in R^2, displacement d in R^2 (grid meters); end = s + d
    ĉ_t = probe(o_t)      cube estimate from the CURRENT obs only (sensors, no sim state)
    dir(d) = d / ||d||
    Phi_ĉ(d) = [ ĉ_t - b·dir(d) ,  d ]          # b = aim_back standoff (m)

    d*  = argmin_d  J( Phi_ĉ(d) ; o_t, o_g )     # CEM searches d; start is a function of d
    a*  = Phi_ĉ(d*)

This collapses the search from a 4-D box onto a 2-D push manifold M(o_t) = { Phi_ĉ(d) },
a strict subset of the old near-cube trust-region box: not "start NEAR the cube" but
"start exactly BEHIND it relative to the push." Contact is guaranteed whenever
||d|| > b; shorter pushes just under-reach (the objective down-weights them), they are
NOT the sideways-miss pathology. Reachability is orthogonal: if ĉ_t - b·dir(d) is
unreachable for the arm the executor stalls and that eval stays frozen -- unchanged here.

ĉ_t is taken from the MPC warm-start seed action (which is itself probe(current obs);
see planning/mpc.py), with a probe-on-obs fallback if no seed is provided. Either way the
anchor is sensors-only. All probe / WM objects are inherited from the base planner.
"""
import torch

from .cem import CEMPlanner
from utils import move_to_device


class AimedContactCEMPlanner(CEMPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # standoff b (m): how far BEHIND the cube the derived start sits. Matches the data
        # sampler's back_range (0.08-0.12); ~0.10 puts the blade just behind the cube.
        self.aim_back = float(kwargs.get("aim_back", 0.10))
        # below this push magnitude (m) the direction is undefined -> treat as a hold
        # (start == cube, ~no push) instead of dividing by ~0.
        self.aim_eps = float(kwargs.get("aim_min_disp", 0.01))
        # keep-off-barrier: grid extent (m, env-local, centered at origin) and the edge
        # band within which a cube counts as "on the barrier". Mirror env/isaaclab/
        # grid_metadata.py (GRID_HALF=0.20) and stroke_sampler._keep_off_barrier
        # (_BARRIER_EPS=0.03). Config-driven so planning/ stays decoupled from the sim env.
        self.aim_grid_half = float(kwargs.get("aim_grid_half", 0.20))
        self.aim_barrier_eps = float(kwargs.get("aim_barrier_eps", 0.03))

    # ----- anchor + normalization caches (set once per plan() call) --------------------
    def _anchor_cube(self, obs_0, actions):
        """ĉ_t in METERS, (n_evals, 2), device. Prefer the MPC warm-start seed (its start
        coords ARE probe(current obs)); else run the probe on obs_0 directly. Sensors only."""
        if actions is not None and actions.shape[1] >= 1:
            cube_norm = actions[:, 0, :2].to(self.device).float()
            return cube_norm * self._astd[:2] + self._amean[:2]
        probe = getattr(self.objective_fn, "position_probe", None)
        if probe is None:
            raise RuntimeError(
                "AimedContactCEMPlanner needs a cube anchor: pass a warm-start seed action "
                "(MPC does this) or use a probe objective so position_probe is available."
            )
        trans = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        with torch.no_grad():
            z = self.wm.encode_obs({"visual": trans["visual"], "proprio": trans["proprio"]})
            return probe(z["visual"][:, -1]).to(self.device).float()  # (n_evals, 2) meters

    def _derive(self, action, cube_m):
        """Phi: given `action` (normalized, (...,4)) and the cube anchor `cube_m` (meters,
        broadcastable to action[..., :2]), keep-off-barrier the push, then set the START to
        ĉ - b·dir(push). Writes BOTH the (possibly flipped) push and the derived start back,
        each clamped to the WM's training action range."""
        amean, astd = self._amean, self._astd
        disp_m = action[..., 2:4] * astd[2:4] + amean[2:4]                 # denormalize push
        # keep-off-barrier: when the cube sits on a clamp edge, FLIP any push component that
        # points further OUTWARD (sign matches the cube's), so we never drive it into the
        # barrier (where it pins/skips). Mirrors stroke_sampler._keep_off_barrier; only fires
        # on outward samples, so genuine inward pushes toward an edge-goal are untouched.
        on_edge = cube_m.abs() >= (self.aim_grid_half - self.aim_barrier_eps)
        flip = on_edge & (torch.sign(disp_m) == torch.sign(cube_m))
        disp_m = torch.where(flip, -disp_m, disp_m)
        n = disp_m.norm(dim=-1, keepdim=True)
        dirv = torch.where(n > self.aim_eps, disp_m / n.clamp_min(self.aim_eps),
                           torch.zeros_like(disp_m))                       # hold when push ~ 0
        start_m = cube_m - self.aim_back * dirv                            # behind the cube
        start_norm = torch.clamp((start_m - amean[:2]) / astd[:2], self._aim_lo[:2], self._aim_hi[:2])
        disp_norm = torch.clamp((disp_m - amean[2:4]) / astd[2:4], self._aim_lo[2:4], self._aim_hi[2:4])
        out = action.clone()
        out[..., :2] = start_norm
        out[..., 2:4] = disp_norm
        return out

    # ----- the single overridden hook (called in-loop by CEMPlanner.plan) --------------
    def _constrain_start(self, action, traj, start_lo, start_hi):
        """Override: derive the behind-cube start from each sample's push direction
        (ignores the base near-cube box; start_lo/start_hi unused)."""
        return self._derive(action, self._cube_hat_m[traj])  # cube (2,) broadcasts over samples

    def plan(self, obs_0, obs_g, actions=None):
        # cache normalization constants + action bounds, then the per-eval cube anchor.
        self._amean = self.preprocessor.action_mean.to(self.device).reshape(-1).float()
        self._astd = self.preprocessor.action_std.to(self.device).reshape(-1).float()
        self._aim_lo, self._aim_hi = self._action_bounds()
        self._cube_hat_m = self._anchor_cube(obs_0, actions)              # (n_evals, 2) meters

        mu, valid = super().plan(obs_0, obs_g, actions)
        # the optimized mu is the MEAN of on-manifold samples (start ~ behind cube) but not
        # exactly on M; re-derive its start from its own optimized push so the EXECUTED
        # stroke is exactly behind-cube. cube_hat as (n_evals,1,2) to broadcast over horizon.
        mu = self._derive(mu, self._cube_hat_m[:, None, :])
        return mu, valid
