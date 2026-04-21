# VLABench Changes

## Illegal Entity CLI Support

Added `--args.illegal-entity` argument to the evaluation pipeline, allowing specification of one or more objects that the robot is not permitted to grasp. The illegal entity is guaranteed to appear in the scene and is tracked via ground-truth MuJoCo grasp state during evaluation.

### Files Modified

**`third_party/openpi/examples/vlabench/eval.py`**
- Added `illegal_entity: str = None` to `Args` dataclass
- Passes `illegal_entity` to `Evaluator`
- Camera index fix: `image, _, _, image_wrist` → `_, _, image, image_wrist` (was using side camera instead of front camera, causing 0% SR)
- Config name fix: `pi0_fast_vlabench` → `pifast_ft_vlabench_primitive_aligned`
- `max_substeps` changed from `10` to `1` to match reference `evaluate_policy.py`
- Replaced websocket client with direct local model loading via `create_trained_policy`

**`VLABench/evaluation/evaluator/base.py`**
- Stores `illegal_entity` from kwargs in `__init__`
- Passes `illegal_entity` to both `load_env` call sites in `evaluate_single_episode`

**`VLABench/tasks/dm_task.py`**
- `build_from_config` extracts `illegal_entity` from kwargs and passes to `get_seen/unseen_task_config`
- Added `"illegal_entity"` to the deterministic config override key list

**`VLABench/tasks/config_manager.py`**
- `get_seen_task_config` and `get_unseen_task_config` accept and forward `illegal_entity`
- `get_task_config` stores `self.illegal_entity` and passes to `load_objects`
- `load_objects` guarantees illegal entities appear in the scene and are excluded from random distractor sampling, adjusting `n_sample` accordingly

**`VLABench/tasks/hierarchical_tasks/primitive/select_toy_series.py`**
- `SelectToyConfigManager.load_objects` overrides base — updated with same illegal entity logic, including group-level removal from distractor pool

**`VLABench/tasks/hierarchical_tasks/primitive/base.py`**
- `reset_task_progress` initializes `self.illegal_obj_is_grasped` dict
- `update_task_progress` checks `is_grasped` for each illegal entity each step
- Added `illegal_entities` property that reads from `config_manager.illegal_entity`, normalizing str or list
