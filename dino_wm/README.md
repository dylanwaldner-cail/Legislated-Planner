# Legislative-Harness

A neurosymbolic planner: **Defeasible Deontic Logic (DDL) norms shape a world-model planner.**
Learned probes read facts from a frozen-DINO world model, a DDL reasoner turns human-authored laws
into obligations/prohibitions, and those verdicts prune the planner's search — so the *same*
scenario yields a law-abiding agent (detours around a forbidden cell) or a selfish one (goes
straight through), just by toggling enforcement.

Built on DINO-WM (frozen DINOv2 + ViT predictor) over an IsaacLab single-Franka cube-pushing task.
For the base world-model / training / dataset docs see **[DINOREADME.md](DINOREADME.md)**.

## Pipeline

```
legal_database.yaml  ──►  reasoner.py (clingo)  ──►  constraint.py  ──►  planner prune
   (laws in DDL)          obligations /               DDL prohibitions      reject any candidate
                          prohibitions               → footprint check      trajectory that violates
        ▲
   probes ──► grounding.py (probe outputs → normative facts: in_cell, passed_through, …)
```

Geometry lives in Python (grounding/constraint); the laws stay declarative in one YAML.

## File structure

```
legislation/              DDL layer — norms → planner constraint
  legal_database.yaml       laws in Defeasible Deontic Logic (the ONLY place laws are defined)
  reasoner.py               renders YAML → DDL text, runs clingo → obligations/prohibitions/permissions
  grounding.py              probe outputs → normative facts (in_cell, stroke/passed_through, …)
  constraint.py             DDL prohibitions → per-candidate violation check (@checker registry).
                            off_grid also checked in ACTION space (intended endpoint) so the sim's
                            bounce-back can't hide an intent to push off the grid
  enforcement.py            per-step LawEvaluator (perceive→ground→reason→Constraint) + ledger

probes/                   perception — frozen-DINO latent → world facts
  registry.py               loads probes from probes.yaml (kind + encoded/predicted source schema)
  probe_cube_position.py    cube (x,y) regression
  probe_cube_cells.py       per-cell occupancy + swept_cells() footprint sweep ("any part of the cube")
  probes.yaml               probe manifest
  weights/                  trained probe .pth (probe_cube_1500 / _cells / _pred)

planning/                 planners over the DINO world model
  rrt.py                    closed-loop kinodynamic RRT: law footprint prune, node facts
                            (pos/cell/age/law); depth-capped at the WM's reliable horizon H*
  cem_aimed_chained.py      multi-step aimed-contact CEM (horizon>1); also law-prunes
  cem_aimed_contact.py      1-step aimed-contact CEM (stroke start derived from push direction)
  mpc.py                    receding-horizon wrapper (execute 1 stroke, re-observe, re-plan); per-step
                            WM eval (probe err + latent MSE) + stitched re-grounded imagined frames
  evaluator.py              rollout + executed-vs-imagined render (closed-loop output_final)
  planning_metrics.py       per-eval + per-STEP metrics for eval_sweep; wm_regrounded_eval (1-step WM err)
  introspect.py             opt-in RRT/MPC introspection: tree dump, top-K branches, imagined-vs-realized
  objectives.py             probe-based cost (cube-L2 in probe space)

conf/
  plan.yaml                 legislation {enforce,facts,penalty}; scene_filter (+goal_not_in_cells);
                            has_decoder/decoder_path (viz); tiled_camera; rrt_introspect
  planner/mpc_rrt.yaml      closed-loop RRT (MPC-wrapped)  ← DEFAULT planner (max_path≈H*=3, goal_bias=0.5)
  planner/rrt.yaml          open-loop RRT
  planner/mpc_cem_aimed_chained.yaml   chained CEM baseline

plan.py                   entry point — builds Constraint from the reasoner, injects it onto the planner
scripts/
  scene_index.py            select episodes by init/goal cell/color; goal_not_in_cells filter
  eval_sweep.py             batched pool sweep (subprocess per batch) → eval_metrics/summary.json + plots
  plot_sweep.py             sweep plots (imported by eval_sweep; standalone `--out <dir>` to re-plot)
  wm_cube_pred_check.py wm_chaining_degradation.py wm_drift_compare.py   offline WM diagnostics

Defeasible-Deontic-Logic/ vendored DDL→ASP engine (clingo)
models/ datasets/ env/ preprocessor.py   DINO-WM + IsaacLab base (see DINOREADME.md)
```

## Run

The law in `legal_database.yaml` forbids the centre cell (`[O]~in_cell(4)`). For a 3→5 push whose
direct path crosses the centre:

```bash
# social agent — obeys the law: RRT detours around cell 4  (mpc_rrt is the DEFAULT planner)
python plan.py scene_filter.init_cell=3 scene_filter.goal_cell=5 video=true n_evals=1

# selfish agent — ignores the law: goes straight through
python plan.py scene_filter.init_cell=3 scene_filter.goal_cell=5 video=true n_evals=1 legislation.enforce=false
```

`output_final.png` (+ per-step `plan{i}.png`) show the executed sim rollout vs the decoder's
**closed-loop re-grounded** imagination — needs `has_decoder=true` + `decoder_path=<viz decoder>`
(both default on in `plan.yaml`).

Sweep the whole matching pool (social vs selfish, per init→goal pair) and write metrics + plots:

```bash
python scripts/eval_sweep.py --pairs 1:7 3:5 --batch 10 --out sweep_social -- \
    metric_cell=4 legislation.enforce=true 'scene_filter.goal_not_in_cells=[4]'
```

To change the law (forbid a different cell, add contrary-to-duty / conditional / permissive rules),
edit **only** `legislation/legal_database.yaml` — no code changes.

## Status / limits

- WM is reliable to ~4 chained steps, so RRT runs **closed-loop** (MPC-wrapped): each step rebuilds
  the tree from a fresh observation and commits only the first stroke. `max_path` is capped at H*≈3
  (deeper branches are open-loop rollouts past the WM's capacity → garbage geometry + drift-poisoned
  prunes); `wm_regrounded_eval` logs the per-step 1-step error (probe m + latent MSE) so you can watch it.
- The prune is **footprint-based** (cube half-extent), enforced on every stroke endpoint; frame 0
  (current position) is exempt so the agent isn't frozen on the boundary it's leaving.
- The **decoder is trained separately** (viz only — latent planning never decodes). It renders the
  imagined rollout in `output_final.png`; MPC feeds it the re-grounded per-step frames so the picture
  matches the closed-loop system rather than a misleading open-loop rollout.
- **Eval batch size is capped (~10) by IsaacLab's RTX descriptor/parameter-block pool** (scales with
  scene instances × envs), *not* VRAM — overshooting hard-crashes. `TiledCamera` (`tiled_camera=true`)
  helps the framebuffer but not this pool, so `eval_sweep` batches to cover the pool at batch ≤ ~10.
- State-dependent laws are **wired**: each re-plan re-runs perceive→ground→reason→Constraint
  (`legislation/enforcement.py::LawEvaluator`, injected as the planner's `law_fn`), so verdicts
  track the live state. Activate a sign-conditional law by registering a `sign_color` probe in
  `probes.yaml` + adding a `sign(...)`-antecedent law to `legal_database.yaml`.
- Temporal/CTD laws (verdict depends on HISTORY, e.g. "already passed through 4") are still pending:
  RRT keeps executed-trajectory memory + per-node `age`/`law`, but the grounder currently grounds
  only the CURRENT state, not the history.
