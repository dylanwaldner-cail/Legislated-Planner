# Changes

libero/libero/libero/bddl\_files/libero\_goal:
Copied open\_the\_middle\_drawer\_of\_the\_cabinet.bddl to open\_the\_bottom\_drawer\_of\_the\_cabinet.bddl
Edited open\_the\_bottom\_drawer\_of\_the\_cabinet.bddl to specify bottom drawer as prompt, obj of interest, and goal state

Added custom\_task.bddl with language "Open the top drawer and put the bowl inside", with akita\_black\_bowl\_1 and wooden\_cabinet\_1\_top\_region as objects of interest

libero/libero/libero/benchmark/libero\_suite\_task\_map.py:
Added the bottom drawer task to the libero goal list (at the top)

libero/libero/libero/init\_files/libero\_goal:
Copied open\_the\_middle\_drawer\_of\_the\_cabinet.pruned\_init to open\_the\_bottom\_drawer\_of\_the\_cabinet.pruned\_init

lerobot/src/lerobot/envs/libero.py:
Added `illegal_obj` parameter to `LiberoEnv.__init__`, `_make_env_fns`, and `create_libero_envs`
Changed `prompt_override` default from `None` to `''`
Added `get_init_state()` and `get_sim_state()` helper methods to `LiberoEnv`
Added drawer override hook in `reset()` for custom init state experimentation
Added debug printing of initial state type and contents

lerobot/src/lerobot/scripts/lerobot\_eval.py:
Imported `LegislativeHarness` and `LegislativeModule`
Added action filter in rollout loop using `legis\_harn.action\_filter`
Added end-of-episode state diff logging to identify changed joints

lerobot/src/lerobot/policies/pi0/modeling\_pi0.py:
Imported `LegislativeHarness` and `LegislativeModule`
