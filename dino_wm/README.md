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
  constraint.py             DDL prohibitions → per-candidate violation check (extensible @checker registry)

probes/                   perception — frozen-DINO latent → world facts
  registry.py               loads probes from probes.yaml (kind + encoded/predicted source schema)
  probe_cube_position.py    cube (x,y) regression
  probe_cube_cells.py       per-cell occupancy + swept_cells() footprint sweep ("any part of the cube")
  probes.yaml               probe manifest

planning/                 planners over the DINO world model
  rrt.py                    closed-loop kinodynamic RRT: law footprint prune, node facts
                            (pos/cell/age/law), executed-trajectory memory
  cem_aimed_chained.py      multi-step aimed-contact CEM (horizon>1); also law-prunes
  cem_aimed_contact.py      1-step aimed-contact CEM (stroke start derived from push direction)
  mpc.py                    receding-horizon wrapper (execute 1 stroke, re-observe, re-plan)
  objectives.py             probe-based cost (cube-L2 in probe space)

conf/
  plan.yaml                 legislation: {enforce, facts, violation_penalty}; scene_filter
  planner/mpc_rrt.yaml      closed-loop RRT (MPC-wrapped)  ← main config
  planner/rrt.yaml          open-loop RRT
  planner/mpc_cem_aimed_chained.yaml   chained CEM baseline

plan.py                   entry point — builds Constraint from the reasoner, injects it onto the planner
scripts/scene_index.py    select dataset episodes by init/goal cell/color for a specific law test

Defeasible-Deontic-Logic/ vendored DDL→ASP engine (clingo)
models/ datasets/ env/ preprocessor.py   DINO-WM + IsaacLab base (see DINOREADME.md)
```

## Run

The law in `legal_database.yaml` forbids the centre cell (`[O]~in_cell(4)`). For a 3→5 push whose
direct path crosses the centre:

```bash
# social agent — obeys the law: RRT detours around cell 4
python plan.py planner=mpc_rrt scene_filter.init_cell=3 scene_filter.goal_cell=5 video=true n_evals=1

# selfish agent — ignores the law: goes straight through
python plan.py planner=mpc_rrt scene_filter.init_cell=3 scene_filter.goal_cell=5 video=true n_evals=1 legislation.enforce=false
```

To change the law (forbid a different cell, add contrary-to-duty / conditional / permissive rules),
edit **only** `legislation/legal_database.yaml` — no code changes.

## Status / limits

- WM is reliable to ~4 chained steps, so RRT runs **closed-loop** (MPC-wrapped): each step rebuilds
  the tree from a fresh observation and commits only the first stroke.
- The prune is **footprint-based** (cube half-extent), enforced on every stroke endpoint; frame 0
  (current position) is exempt so the agent isn't frozen on the boundary it's leaving.
- State-dependent laws are **wired**: each re-plan re-runs perceive→ground→reason→Constraint
  (`legislation/enforcement.py::LawEvaluator`, injected as the planner's `law_fn`), so verdicts
  track the live state. Activate a sign-conditional law by registering a `sign_color` probe in
  `probes.yaml` + adding a `sign(...)`-antecedent law to `legal_database.yaml`.
- Temporal/CTD laws (verdict depends on HISTORY, e.g. "already passed through 4") are still pending:
  RRT keeps executed-trajectory memory + per-node `age`/`law`, but the grounder currently grounds
  only the CURRENT state, not the history.
