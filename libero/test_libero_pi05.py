import time
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1" # for local runs
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

def quat2axisangle(quat):
    quat = torch.tensor(quat, dtype=torch.float32).unsqueeze(0)
    w = quat[:, 3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    result = torch.zeros((1, 3))
    if den > 1e-10:
        angle = 2.0 * torch.acos(w)
        axis = quat[:, :3] / den.unsqueeze(1)
        result = axis * angle.unsqueeze(1)
    return result.squeeze(0).numpy()

# ── 1. Load π0.5 policy ──────────────────────────────────────────────────────
model_id = "lerobot/pi05_libero_finetuned"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[info] using device: {device}")

policy = PI05Policy.from_pretrained(model_id).to(device).eval()
preprocess, postprocess = make_pre_post_processors(
	policy.config,
	model_id,
	preprocessor_overrides={"device_processor": {"device": str(device)}},
	)

# ── 2. Load LIBERO environment ────────────────────────────────────────────────
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_goal"]()

task_id = 1
task = task_suite.get_task(task_id)
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] task: {task_description}")

env_args = {
	"bddl_file_name": task_bddl_file,
	"camera_heights": 224,
	"camera_widths": 224,
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
obs = env.reset()

init_states = task_suite.get_task_init_states(task_id)
obs = env.set_init_state(init_states[0])

# ── 4. Run π0.5 for N steps ──────────────────────────────────────────────────
n_steps = 50
num_actions = 20
print(f"[info] running policy for {n_steps} steps...")

for a in range(num_actions):
	before_img = obs["agentview_image"].copy()
	start_time = time.time()
	for step in range(n_steps):

		state = np.concatenate([
		    obs["robot0_eef_pos"],              # 3
		    quat2axisangle(obs["robot0_eef_quat"]),  # 3 (not 4!)
		    obs["robot0_gripper_qpos"],         # 2 (not [:1]!)
		])  # = 8 dims, correct

		frame = {
		    "observation.images.image": torch.flip(
			torch.from_numpy(obs["agentview_image"].transpose(2, 0, 1)).float() / 255.0,
			dims=[1, 2]  # flip H and W
		    ),
		    "observation.images.image2": torch.flip(
			torch.from_numpy(obs["robot0_eye_in_hand_image"].transpose(2, 0, 1)).float() / 255.0,
			dims=[1, 2]
		    ),
		    "observation.images.empty_camera_0": torch.zeros(3, 224, 224),
		    "observation.state": torch.from_numpy(state).float(),
		    "task": task_description,
		}
		batch = preprocess(frame)

		with torch.inference_mode():
			pred_action = policy.select_action(batch)
			pred_action = postprocess(pred_action)

			action = pred_action.cpu().numpy().flatten()[:7]  # LIBERO expects 7-dim action
			obs, reward, done, info = env.step(action)

			if done:
				print("[info] task completed!")
				break

	print(f"Action {a+1} Done, took ", time.time() - start_time, "Seconds.")

	# ── 5. Save AFTER image and compare ──────────────────────────────────────────
	after_img = obs["agentview_image"].copy()

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	axes[0].imshow(before_img)
	axes[0].set_title("Before (step 0)")
	axes[0].axis("off")

	axes[1].imshow(after_img)
	axes[1].set_title(f"After ({step+1} steps)")
	axes[1].axis("off")

	plt.suptitle(f"π0.5 on LIBERO: '{task_description}'")
	plt.tight_layout()
	plt.savefig(f"pi05_before_after_action_{a}.png", dpi=150)
	print(f"[info] saved to pi05_before_after_action_{a}.png")

env.close()
