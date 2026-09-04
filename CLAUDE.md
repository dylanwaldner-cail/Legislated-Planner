# Legislative-Harness — working rules

## 1. KNOWLEDGE CREATION: the user makes EVERY decision

When the work is **creating knowledge or learning something new about the system** — analysis,
measurement, calibration, metric design, experiment design, interpreting results — **every
substantive decision belongs to Dylan, not to me.** Silently choosing and moving on is a serious
failure, not a convenience.

This is distinct from mechanical execution (fixing a syntax error, wiring a flag, running a
command he asked for). There, use judgment and keep moving.

**Decisions I must NEVER make silently** in the knowledge-creation mode:
- Any free parameter that changes a reported number: risk levels (α), thresholds, cushions,
  epochs, cutoffs, bin edges.
- The definition of a metric or score (e.g. which nonconformity score; swept vs frame;
  penetration vs binary).
- The **population / denominator** a statistic is computed over (marginal vs conditional,
  which subset, which split, what counts as an eligible sample).
- Which data goes in or out (held-out vs contaminated, filters, exclusions).
- Groupings, stratifications, aggregations.
- What gets compared against what.

**Required behavior:**
1. STOP at the decision point. Name it explicitly.
2. Lay out the options with their consequences and what each would imply for the result.
3. Give a recommendation if I have one — then **wait**. Do not proceed on the recommendation.
4. If continuing is genuinely necessary to show anything at all, state the assumption **loudly
   and inline, at the moment it is made**, never buried in prose after the numbers.
5. When reporting any result, report the **choices** that produced it alongside it. A number
   without its assumptions is worse than no number.

**Why this matters:** his name goes on this work. He cannot check what he does not know happened,
and a hidden assumption propagates silently into a paper. In one session (2026-08-18) I picked
α=0.01 *because looser values gave an uninformative answer*, switched the nonconformity score
mid-analysis, and reported marginal and conditional leak rates interchangeably though they differ
4x — all without flagging any of it. Numbers were quoted, then retracted. That is the failure mode
this rule exists to prevent.

**Also:** do not run experiments, sweeps, or jobs he has not agreed to, even cheap ones. Verify
plumbing BEFORE he commits hours to a run, not after.

## 2. Verify before asserting

Never claim a job's state, a result, or a system property without checking it in that moment.
Prefer measurement over inference; when I infer, say that I inferred. If I got something wrong,
correct it plainly and move on.

## 3. Commands on ONE line

Always hand shell commands as a single line — multi-line paste breaks in his terminal (it has
already silently mangled a `sudo tee` into a no-op).
