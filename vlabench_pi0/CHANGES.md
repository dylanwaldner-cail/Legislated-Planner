## VLABench Changes

### Overview
VLABench (https://github.com/OpenMOSS/VLABench) is used as an evaluation environment for the π0-fast model fine-tuned on primitive manipulation tasks. The original evaluation pipeline uses a server/client architecture where the policy runs as a separate server process and the environment communicates with it over a websocket. We modified the pipeline to load model weights directly in-process, which is necessary for legislative harness intervention (action filtering, activation steering, KV cache access).

### Files Modified

**`vlabench_pi0/VLABench/third_party/openpi/examples/vlabench/eval.py`**
The main evaluation script. Removed the websocket client instantiation and replaced it with direct model loading via `openpi.policies.policy_config.create_trained_policy`. The `Pi0` policy class is unchanged — it still calls `self.model.infer()`, but now against a local model object rather than a remote server. Also added direct norm stats loading from the checkpoint's `assets/` directory to bypass the asset_id lookup in the training config, which pointed to a different path than the downloaded checkpoint structure. Added `pathlib` import.

Key changes:
- **Camera index fix**: Changed `image, _, _, image_wrist = obs["rgb"]` to `_, _, image, image_wrist = obs["rgb"]`. The original `eval.py` used camera index 0 (side view) while the canonical `OpenPiPolicy` in `VLABench/evaluation/model/policy/openpi.py` used index 2 (front view). This was the primary cause of 0% success rate — the model was receiving the wrong camera view entirely.
- **`max_substeps`**: Changed from `10` to `1` to match the `evaluate_policy.py` reference script.
- **Instruction stripping**: Added `.replace("_seen", "").replace("_unseen", "")` to clean evaluation track suffixes from instructions, since the training data did not include these suffixes.
- **Timing instrumentation**: Added inference time logging around `model.infer()` call.

**`vlabench_pi0/VLABench/third_party/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py`**
Added `ping_timeout=None` to the websocket connection call to prevent timeout errors during long inference runs.

