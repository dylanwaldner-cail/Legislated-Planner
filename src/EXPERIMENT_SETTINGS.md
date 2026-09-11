# Experiment settings

Canonical config for the law-eval / sign / cushion runs. Values marked **default** are now in
`conf/plan.yaml` or `scripts/eval_sweep.py` and don't need passing.

## Fixed across every experiment

| setting | value | where |
|---|---|---|
| seed | `99` | **default** (`plan.yaml`, `eval_sweep --seed`) |
| batch size | `10` | **default** (`eval_sweep --batch`) — capped by the RTX descriptor pool, not VRAM |
| planner | `mpc_rrt` | **default** |
| metric_cell | `4` | **default** |
| has_decoder | `false` | **default** |
| yellow_cells | `[3, 5]` | **default** |

## The stack (swap all four together)

| | locked-yaw (current) | pre-lock |
|---|---|---|
| `model_name` | `wm_5k_no_yaw` **default** | `wm_5k` |
| `model_epoch` | `30` **default** | `30` |
| `objective.pos_probe_path` | `probes/weights/cube_pos_encoded_no_yaw.pth` **default** | `probes/weights/cube_pos_encoded.pth` |
| `data_path` | `data/isaaclab_stroke_5k_no_yaw_merged` **default** | `data/isaaclab_stroke_5k` |
| sign probe | `probe_sign_color_no_yaw.pth` (via `DINOWM_SIGN_PROBE`) | `probe_sign_color.pth` |
| eval set | `data/law_eval_center_no_yaw_400` | `data/law_eval_center_400` |
| goal bank | `data/goal_cell_bank_no_yaw` | `data/goal_cell_bank` |

## Independent variables — never default these

- `legislation.constraint_margin` (cushion δ) — stays `0.0`
- `legislation.mode` — `social` / `deviant` / `off`
- `legislation.active_lawsets` — `[geometric_laws]` vs `[full_lawset]`
- `sign_flip.frame` / `sign_flip.color`
- `rule_injection.frame`

## Must be passed explicitly

- **`legislation.goal_bank`** — must be an **absolute** path (`/workspace/src/data/goal_cell_bank_no_yaw`).
  Hydra chdirs into the run dir, so a relative path raises `FileNotFoundError`. `null` silently disables
  obligation enforcement rather than erroring.
- **`CUDA_VISIBLE_DEVICES=<n>` with `device=cuda:0`** — not `device=cuda:<n>`.
  Container→host GPU map: `0`→2, `1`→3, `2`→6, `3`→7.

## Canonical call

```
CUDA_VISIBLE_DEVICES=3 ./IsaacLab/isaaclab.sh -p scripts/eval_sweep.py --law_eval /workspace/src/data/law_eval_center_no_yaw_400 --modes social --batch 10 --seed 99 --out results/no_yaw/cushion/delta_0.04 --frame 1 --color yellow -- planner=mpc_rrt metric_cell=4 legislation.constraint_margin=0.04 legislation.active_lawsets=[full_lawset] legislation.goal_bank=/workspace/src/data/goal_cell_bank_no_yaw device=cuda:0
```

## Sizes / runtime

400 episodes = 8 pairs × 5 batches × 10. One social arm ≈ 35–40 h at current pace.
Oracle needs `--patience 50`; without it a stalled tree-build runs all 512 rounds (~10× slower).
