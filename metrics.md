# Metrics — how the reported numbers are computed

Definitions that are NOT obvious from the code and that change a published number. One section per
metric. If a metric's population/denominator was a decision, the decision is recorded here with its
consequence, not just its outcome.

---

## WM + probe error (Q2, `tab:wm_err_social`)

**Quantity.** `wm_pred_err_steps[eval][t]` from `eval_metrics.json`: the L2 distance in metres between
the position probe's one-step prediction and the executed ground-truth cube position. It is
`|probe∘WM − truth|`, so it contains the probe's readout error as well as the world model's — the two
are not separable in this number (findings.md:1029).

**Population: EXECUTED ACTIONS ONLY.** A sample counts iff the agent actually committed a stroke on
that step. Two exclusions, applied in this order:

1. `t >= n_steps[eval]` — dropped. After the episode reaches the goal it is held there
   (success-hold); those frames are padding, not steps taken.
2. The committed action is a HOLD — dropped. Test is `hypot(dx, dy) == 0` on
   `executed_actions.npy` (shape `(evals, T, 4)`, `[x_start, y_start, dx, dy]`), which is written
   alongside `eval_metrics.json` by every `plan.py` run.

**Do NOT use `runtime_breakdown.per_step[].frozen` for this.** The flag means "the verdict concluded
`[F]moving`", NOT "no action was executed", and whether that suppresses the action depends on the
agent's BIO stance. Measured on results/no_yaw/sign_change (n=400/agent):

| agent | frozen steps that DID execute | HOLD steps the flag MISSED | mean cube move, frozen vs normal |
|---|---|---|---|
| social  | 0   | 133 | 0.019 m vs 0.108 m — obeys the freeze |
| deviant | 168 | 14  | 0.059 m vs 0.118 m — partially ignores it |
| off     | 231 | 0   | 0.104 m vs 0.128 m — ignores it entirely |

So the flag would discard 168 real deviant actions and keep 133 social non-actions. The
zero-delta test is the correct one and it reduces per-agent to exactly the intended rule: none from
social while frozen, only the executed ones from deviant, all of off (off never HOLDs — 0 of 1220).

**Pooling.** All three agents (social + deviant + off/realistic) pooled, grouped by task pair. `All`
is the pooled row over every sample, NOT the mean of the eight task means (the pairs have unequal n:
477–601).

**Reported columns.** Mean, median, IQR (= p75 − p25, a single number), p90, p99.

**Result on results/no_yaw/sign_change** (n=4,416): All row mean 0.0306, median 0.0190, IQR 0.0200,
p90 0.0670, p99 0.1900 m. Cell width is 0.133 m and the cube is 0.09 m across, so p90 is ~half a
cell and p99 exceeds a full cell.

**⚠️ Not reproducible from the yaw-era table.** The published Jurix table (n=5,008, mean 0.0382,
p90 0.1078) predates this definition and could not be reproduced: six filter variants on
results/final/sign_change bracketed it (4,687 / 5,230 / 5,320) without matching. The rule above was
chosen on its merits and stated, rather than reverse-engineered to hit 5,008 — do not treat the old
row as a target.

---

## Runtime cost (Q4, `tab:runtime`)

**Two different denominators, one per panel.** They measure different things and must not be mixed.

**Panel (a) — per DEONTIC DECISION.** Denominator is `runtime_breakdown.leg_n_observe`, the engine's
own exact verdict count (4720 social / 3620 deviant on results/no_yaw/sign_change). This is the unit
the reasoning is billed against: `observe()` fires once per MPC loop and produces one verdict, so
"ms per verdict" is what a deployment would budget for. Numerators are the fine keys
`leg_probe_s + leg_ground_s` (perception/grounding), `leg_logic_s` (clingo), `leg_build_s`
(constraint build). CIs are 10,000-resample bootstraps over BATCHES, since `runtime_breakdown` is
recorded per batch, not per step.

**Panel (b) — per EXECUTED ACTION.** A step counts iff a stroke was actually committed
(`hypot(dx,dy) > 0` in `executed_actions.npy`, within `n_steps`) -- the same rule as the WM error
table above. **The NUMERATOR is restricted to those same steps**, taken from
`runtime_breakdown.per_step[]` (`reason_s`, `prune_s`, `total_s`; rrt = total - reason - prune), NOT
from the batch totals. Using batch totals over an executed-action denominator was the first version
of this table and it is WRONG: it spreads the cost of steps that committed nothing across the steps
that did, and since a frozen step costs 50.0s against 5.3s for the rest, it inflated social from
12.5s to 55.5s per action. Result: social 12,518 ms, deviant 27,528 ms.

**Why not one denominator for both.** Reasoning fires once per verdict whether or not an action
follows, so billing it per executed action would inflate it by the verdict-to-action ratio (3.07x
for social). Planning cost is what it costs to actually move the robot, so it belongs per action.
This is why "Symbolic reasoning" legitimately reads 16.6ms in panel (a) and 20ms in panel (b): same
work, different unit. Say so in the column headers, or a reader will read it as an inconsistency.

**⚠️ Panel (b) covers 1137 of 1537 social executed actions (1259 of 1659 deviant).** `per_step`
begins at step index 1, so **step 0 of every episode is never timed** -- exactly 400 actions per
mode, 26% of social's. The reported means are over the instrumented steps only. Whether step 0 is
cheaper (fresh tree) or dearer than a mid-episode step is NOT known from the artifacts.

**Sanity check on the absolute numbers.** Summing `per_step.total_s` over ALL social steps gives
~23.7h against 26.1h of wall clock for that arm. Of it, the 1137 timed executed actions account for
only ~4.0h: the rest is frozen steps and post-success holds. That is the finding, not a discrepancy
-- a productive action is cheap and the expensive steps are the ones that commit nothing.

**⚠️ Dropped from panel (a): the clingo cache split.** The published table carried
`per solve (43.4%) 26.18ms` and `per cache hit (56.6%) 0.0002ms`. Those memoisation counters are NOT
in `runtime_breakdown` -- only the amortised `leg_logic_s` is -- so they cannot be regenerated from
the run artifacts. The rows were removed rather than carried over stale. To restore them, the solver
would have to log hit/miss counts per batch.

**⚠️ `scripts/runtime_table.py` does NOT produce this table.** Its `COLUMNS` are hardcoded to
yaw-era paths (results/final/base, results/final/social_cushion, results/aug20/sign_change) and its
columns are Q1-geometric / Q2-cushion / Q3-full-lawset, not social/deviant. It also uses a THIRD
denominator (productive actions = within `n_steps` and not `[F]moving`). Repointing it is not enough;
it would need its column scheme rewritten.
