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
- Removed verbose debug prints (state/image shape, instruction, illegal entity) from inference path

**`VLABench/evaluation/evaluator/base.py`**
- Imports `LegislativeHarness` and `LegislativeModule`
- On init, instantiates `self.legis_harness` and `self.legis_module`
- CLI `--args.illegal-entity` entries are registered as structured laws into `self.legis_module.laws` under the `"Symbolic"` schema (NL: `"Don't grab the {entity}"`)
- `load_env` calls now pass `legis_module=self.legis_module` instead of `illegal_entity`
- Per-step loop now:
  - Calls `legis_harness.action_filter(physics, task, law, action)` for each law
  - Accumulates newly detected illegal grasps into `illegal_objs`
  - If `illegal_objs` is non-empty, overrides instruction with `"Raise your gripper straight up in the air."`
  - Calls `legis_harness.update_history(action)` after filtering

**`VLABench/tasks/dm_task.py`**
- `build_from_config` now reads `legis_module` from kwargs and calls `legis_module.get_illegal_objects()` to derive `illegal_entity`, rather than reading `illegal_entity` directly from kwargs
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

**`VLABench/tasks/hierarchical_tasks/primitive/add_condiment_series.py`**
- `load_objects` signature updated to accept `illegal_entity=None` (harness compatibility, no logic change)

**`VLABench/tasks/hierarchical_tasks/primitive/insert_flower_series.py`**
- `load_objects` signature updated to accept `illegal_entity=None` (harness compatibility, no logic change)

---

## Legislative Harness Integration

Introduced a `LegislativeHarness` + `LegislativeModule` system as the central mechanism for constraint enforcement. The raw `illegal_entity` string previously passed through the stack is now registered as a structured law and enforced via the harness at inference time.

**`legislative_harness/__init__.py`**
- Renamed import: `legislative_module.py` → `module.py` (`LegislativeModule` now lives in `module.py`)

**`legislative_harness/legislative_module.py` → deleted**
- Removed old `LegislativeModule` with PDDL-keyed law schema (`"PDDL": {"Object", "Predicate", "Consequence"}`)
- Replaced by `module.py` using a `"Symbolic"` key schema with the same fields

**`legislative_harness/harness.py`**
- `is_illegal_action`, `is_closed`, `is_closing`, `action_filter` all refactored from Libero/MuJoCo-gym API to dm_control API
  - Now accept `physics` (dm_control Physics) and `task` (VLABench task object) instead of a wrapped `env`
  - `is_closed`: reads gripper qpos via `physics.bind(robot.gripper.joints)`; threshold tightened 0.1 → 0.02 rad
  - `is_closing`: handles both 1D and 2D action arrays; checks last 2 dims (both gripper fingers) instead of only last dim; threshold tightened 0 → 0.02
  - `is_illegal_action`: checks `task.entities` for object existence; uses `illegal_entity.is_grasped(physics, task.robot)` for contact detection; returns `(bool, obj_name)` tuple instead of bare `bool`
  - `action_filter`: on illegal action, replaces arm joints with `task.robot.get_qpos(physics)` and opens gripper to 0.04 instead of zeroing deltas; returns `(trajectory, illegal_entity_name)` tuple

---

## KV Cache / Logit-Lens Instrumentation

**`third_party/openpi/src/openpi/models/pi0_fast.py`**
- Added `_DEBUG_WEIGHTS = {}` module-level dict for weight introspection
- Imports harness utils: tokenizer, KV cache metrics, logit-lens helpers, etc.
- `sample_actions`: captures `out` (previously discarded as `_`) from the LLM prefill call to expose `all_hidden_states`
- Added `jax.debug.callback` harness block that runs outside the JIT boundary each replan step:
  - Extracts and decodes token IDs from prefix embeddings
  - Runs logit-lens over all 18 layers at target token positions
  - Computes KV cache drift metrics (m1, m3, m5, m6a, m6b) relative to a stored baseline from replan step 1
  - Writes per-replan analysis to `kv_output.txt`

**`third_party/openpi/src/openpi/policies/policy_config.py`**
- After model load, extracts `embed_matrix` and `attn_vec_einsum` weights from the PaliGemma LLM via `nnx.state` and stores them into `_DEBUG_WEIGHTS` before the JIT boundary
