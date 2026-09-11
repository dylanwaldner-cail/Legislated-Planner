"""Positive-obligation -> planner target: the small switch that makes an obligation SELECT a goal image.

A positive obligation ([O]in_cell(k), or [O]in_yellow_cell -> {3,5}) is an ACHIEVEMENT goal, not a
prohibition, so it maps onto the planner OBJECTIVE, not the pruning constraint. The planner already
steers toward a goal image: goal image -> encode -> position probe -> target xy -> RRT steers there.
This just swaps WHICH goal image feeds that same pipeline: when an obligation is live, feed the
obligated cell's centered goal image (from the goal-cell bank, scripts/gen_goal_cell_bank.py) instead
of the task goal. Goal-image-as-subgoal has precedent in hierarchical latent world models (HWM).

Nothing here is new machinery -- it is one substitution of the objective's target, re-evaluated each
MPC step (so it tracks a sign flip / a discharged obligation automatically). Discharge is temporal:
once the cube has been in an obligated cell (visited(k) in the ledger) the obligation is satisfied and
the target reverts to the real goal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from probes.probe_cube_position import gm

# `utils.move_to_device` is imported LAZILY inside encode() on purpose. Under Isaac Sim the kit
# python has .../pip_prebundle/cv2 on sys.path, so a top-level `from utils import ...` resolves to
# cv2's `utils` package instead of the repo's utils.py and raises ImportError. encode() is the only
# consumer (it needs the WM + preprocessor), and the sim oracle never calls it -- it uses
# pos_from_states() -- so deferring the import keeps this module importable in the oracle process.

_YELLOW_CELLS = frozenset(k for k in range(gm.N_CELLS)
                          if gm.CELL_COLORS[k // gm.N_COLS][k % gm.N_COLS] == "yellow")


class GoalBank:
    """The 9 centered goal images (one per cell) + their encoded target, and the obligation->cell switch.

    Load the bank dir (scripts/gen_goal_cell_bank.py output); call encode() once with the planner's
    wm / preprocessor / position_probe to get, per cell k: z[k] (goal latent) and pos[k] (probed target
    xy, derived EXACTLY like the task goal so RRT's goal_tol compares like-with-like)."""

    def __init__(self, bank_dir):
        d = Path(bank_dir)
        self.states = torch.load(d / "states.pth").float()                 # (9, 31)
        self.proprio = torch.load(d / "proprio.pth").float()               # (9, 18)
        self.n = self.states.shape[0]
        self.visual = torch.stack([torch.load(d / "obses" / f"cell_{k:02d}.pth")
                                   for k in range(self.n)])                 # (9, H, W, 3) uint8
        self.z = None          # (9, P, D) encoded goal latents (per cell) -- filled by encode()
        self.pos = None        # (9, 2) probed target xy                    -- filled by encode()

    @torch.no_grad()
    def encode(self, wm, preprocessor, position_probe, device):
        """Encode all 9 goal images the SAME way the task goal is encoded (obs T=1 -> encode -> [:,-1]),
        then read the target xy off the position probe. Idempotent; run once per episode/planner."""
        from utils import move_to_device                                # lazy: see the note at the top
        obs = {"visual": self.visual.unsqueeze(1),                         # (9, 1, H, W, 3)
               "proprio": self.proprio.unsqueeze(1)}                       # (9, 1, 18)
        trans = move_to_device(preprocessor.transform_obs(obs), device)
        self.z = wm.encode_obs(trans)["visual"][:, -1]                     # (9, P, D)
        self.pos = position_probe(self.z).detach().cpu().numpy()           # (9, 2)
        return self

    def pos_from_states(self):
        """ORACLE counterpart of encode(): take each cell's target xy from the bank's GROUND-TRUTH goal
        states instead of reading it off the position probe. Same role as encode() -- fill self.pos so
        waypoint() can measure distances -- but derived the way the sim oracle derives every other
        quantity, so the obligation channel is exact rather than probe-limited. Leaves self.z unset
        (the oracle plans in state space and never needs the goal latents)."""
        self.pos = self.states[:, 18:20].numpy().astype(float)             # (9, 2) true cube xy per cell
        return self

    @staticmethod
    def obligation_targets(obligations):
        """Each positive POSITIONAL obligation -> the SET of cells that satisfies it, as a LIST of sets
        (one per obligation) so the disjunction (any-one-satisfies) is preserved for discharge:
          in_cell(k) / passed_through(k) -> {k};   in_yellow_cell -> the yellow cells {3,5}.
        Non-positional obligations (off_grid, moving...) map to no target and are skipped. exit_cell is
        the ONE reparative duty handled elsewhere (exit_obligations + waypoint), not here."""
        targets = []
        for name, args in (obligations or []):
            if name in ("in_cell", "passed_through") and args:
                try:
                    targets.append({int(args[-1])})
                except (ValueError, TypeError):
                    pass
            elif name == "in_yellow_cell":
                targets.append(set(_YELLOW_CELLS))
        return targets

    @staticmethod
    def obligated_cells(obligations):
        """Flattened union of every obligated cell -- for display/logging only (the disjunction
        structure needed for steering/discharge lives in obligation_targets)."""
        out = set()
        for t in GoalBank.obligation_targets(obligations):
            out |= t
        return out

    @staticmethod
    def exit_obligations(obligations):
        """Each exit_cell(k) -> the cell int k the cube must LEAVE (the reparative CTD duty in R10's
        center-cell chain). 'exit_cell(k)' means 'be in any cell but k', so it maps to the COMPLEMENT of
        {k}; waypoint() steers to the nearest such cell while the cube is still in k."""
        out = []
        for name, args in (obligations or []):
            if name == "exit_cell" and args:
                try:
                    out.append(int(args[-1]))
                except (ValueError, TypeError):
                    pass
        return out

    @staticmethod
    def return_obligations(obligations):
        """Each return_cell(k) -> the cell k the cube must GET BACK TO (the injected white-sign duty).

        Deliberately NOT handled as an achievement in_cell(k) duty. waypoint() discharges those on
        `tset & visited`, and the START cell is in visited from frame 0 by construction -- so an
        [O]in_cell(start) would be skipped as 'already discharged' the instant it fired, the goal would
        never switch, and the experiment would report a null result for a plumbing reason rather than a
        normative one. Like exit_cell(k) this is REPARATIVE: live exactly while the reasoner says it is,
        with no temporal discharge test here, recomputed from the live verdict every MPC step."""
        out = []
        for name, args in (obligations or []):
            if name == "return_cell" and args:
                try:
                    out.append(int(args[-1]))
                except (ValueError, TypeError):
                    pass
        return out

    def waypoint(self, obligations, visited_cells, cur_pos, goal_cell):
        """The obligated WAYPOINT cell to steer toward, or None (use the real goal). Handles each
        obligation's disjunction SEPARATELY -- an ACHIEVEMENT obligation (in_cell/passed_through/
        in_yellow_cell) is skipped when:
          - the real GOAL cell already satisfies it (goal in its set) -> the objective handles it, no
            waypoint (so a live goal obligation in_cell(goal) never overrides the real goal), or
          - the cube has already been in a satisfying cell (visited -> discharged, temporal).
        A REPARATIVE exit_cell(k) duty is different: it means 'get out of k'. It is live only while the
        cube is currently IN k (else discharged), and it steers to the NEAREST cell != k, OVERRIDING the
        real goal -- so the agent vacates the forbidden center before resuming. (An exit duty must not be
        skipped on 'goal already satisfies it', or it would never fire when the goal is a normal cell.)
        Of all live candidates, steer to the single NEAREST cell (min probed-target distance from the
        current cube). Swap `min` for random.choice to pick a random yellow cell."""
        visited = set(visited_cells)
        cur = np.asarray(cur_pos)
        cands = []
        for tset in self.obligation_targets(obligations):
            if goal_cell in tset or (tset & visited):
                continue                                          # goal satisfies it, or already discharged
            cands.append(min(tset, key=lambda k: float(np.linalg.norm(cur - self.pos[k]))))
        # REPARATIVE EXIT: steer to the nearest cell != k, from the CURRENT position (probe-read for the
        # WM agent, ground truth for the sim oracle). No discharge test here on purpose: the duty is live
        # exactly while the reasoner says it is -- R10's compensation fires only while in_cell(k) holds,
        # and in_cell is FOOTPRINT-based. This previously gated on `which_cell(cur) == k`, a CENTRE-based
        # test, which silently declined to steer for any footprint GRAZE of k: measured on
        # results/aug15/sign_color, 147 of the 149 exit_cell(4) obligations had the centre outside cell 4
        # (and all 149 were grazes), so the reparative channel was ~99% inert. Nothing is persisted --
        # the target is recomputed from the live verdict each MPC step, so it disappears as soon as the
        # footprint clears k, and re-fires by itself if perception error keeps the cube inside.
        for k in self.exit_obligations(obligations):
            others = [c for c in range(self.n) if c != k]          # nearest LEGAL cell (k is the illegal one)
            cands.append(min(others, key=lambda c: float(np.linalg.norm(cur - self.pos[c]))))
        # REPARATIVE RETURN: steer straight back to k. No discharge test and no 'goal already satisfies
        # it' skip -- both would silence it (the start cell is always visited, and under the injected
        # lawset the episode goal is swapped TO the start cell, so `goal_cell in tset` would also hold).
        # It falls out of `cands` by itself once the reasoner stops concluding the duty.
        for k in self.return_obligations(obligations):
            cands.append(k)
        if not cands:
            return None
        return min(cands, key=lambda k: float(np.linalg.norm(cur - self.pos[k])))


def visited_cells_from_ledger(ledger):
    """Ledger -> set of cell ints the cube has occupied (visited(k) facts). Encoded-executed history only."""
    out = set()
    for f in ledger.derived_facts():                      # 'visited(3)' -> 3
        if f.startswith("visited(") and f.endswith(")"):
            try:
                out.add(int(f[len("visited("):-1]))
            except ValueError:
                pass
    return out
