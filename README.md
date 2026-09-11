# Legislated Planner

Research code for **law as control**: human-authored legal norms, written in Defeasible Deontic
Logic (DDL), are compiled into constraints on a world-model-based robot motion planner. A learned
probe reads facts off a frozen-DINO world model, a clingo-backed DDL reasoner turns those facts and
the lawset into obligations, prohibitions and permissions, and the verdicts prune the planner's
search *before* an illegal action is executed.

The same scenario yields a **social** agent (obeys — detours around a forbidden cell), a **deviant**
agent (prices violations — breaks the law only when the detour costs more), or a **realistic** agent
(ignores the law) purely by switching the enforcement mode.

The task is single-arm cube-pushing across a 3×3 grid in IsaacLab.

## Layout

Everything lives under [`dino_wm/`](dino_wm/):

| path | what |
|---|---|
| [`dino_wm/README.md`](dino_wm/README.md) | **start here** — setup, dependencies, pipeline, full file map |
| `dino_wm/legislation/` | the DDL layer: `legal_database.yaml` (the only place laws are defined), reasoner, grounding, constraint |
| `dino_wm/planning/` | planners (RRT, CEM, MPC) and the objectives they optimize |
| `dino_wm/probes/` | learned readouts from world-model latents (cube position, cell occupancy, sign color) |
| `dino_wm/env/` | the IsaacLab cube-pushing environment and grid metadata |
| `dino_wm/Defeasible-Deontic-Logic/` | the DDL-in-ASP solver the reasoner calls |
| `dino_wm/DINOREADME.md` | upstream DINO-WM world-model training and dataset docs |

## Requirements

IsaacLab / Isaac Sim (a hard dependency, installed separately — it is not vendored here) and
`clingo` for the deontic reasoning. Full instructions, version pins and known traps are in
[`dino_wm/README.md`](dino_wm/README.md).

## Paper

*Legislating World-Model-Based Planning with Legal Reasoning.* Dylan Waldner, Yiannis Kantaros,
Guido Governatori, Risto Miikkulainen, Amir Banifatemi. Under review.

## License

Academic Public License — free for teaching, academic research and non-profit use. Commercial use
requires a commercial license from Cognizant Technology Solutions Corp. See [LICENSE](LICENSE).
