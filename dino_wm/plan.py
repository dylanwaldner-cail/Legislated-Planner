import multiprocessing as mp
try:
  mp.set_start_method("spawn", force=True)
except RuntimeError:
  pass  # already set (e.g., test reruns)

import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

# Harness Start ---
from hydra.utils import get_original_cwd, to_absolute_path
# Harness End ---

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    import submitit  # lazy: only needed for SLURM auto-launch from train.py
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]
        ### HARNESS EDIT ### cap how deep into an episode a segment can start (offset in [0, max_offset]).
        # Small => init near the episode start (arm at rest, cubes freshly placed) = clean teleport.
        # Large/None => can start mid-trajectory (mid-grasp), which teleports poorly. Default 50.
        self.max_offset = cfg_dict.get("max_offset", 50)

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        ### HARNESS EDIT ### decouple goal_H from MPC horizon/cadence (goal_H = goal distance only)
        # orig (MPC branch): sub_planner.horizon = n_taken_actions = goal_H -> collapses MPC to open-loop
        if isinstance(self.planner, MPCPlanner):
            pass  # keep configured sub_planner.horizon / n_taken_actions
        else:
            self.planner.horizon = cfg_dict["goal_H"]
        ### END HARNESS EDIT ###

        ### HARNESS EDIT ### RIGHT-ROBOT FREEZE: test the LEFT robot only.
        # Hand the GD optimizer each eval's recorded expert action sequence (gt_actions,
        # already aligned to the per-eval episode+frames sampled in prepare_targets). The
        # optimizer then overwrites the right-arm dims with the expert's, so the right robot
        # replays its real trajectory (instead of flopping) while only the left arm is planned.
        # Gated by cfg `freeze_right` (default True); set freeze_right=false for joint planning.
        if cfg_dict.get("freeze_right", True) and self.gt_actions is not None:
            _gd = self.planner.sub_planner if isinstance(self.planner, MPCPlanner) else self.planner
            _gd.freeze_right_gt = self.gt_actions
            print("[freeze_right] right robot replays its recorded expert trajectory; planning LEFT arm only")
        ### END HARNESS EDIT ###

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        # === HARNESS EDIT: fixed init/goal sourced from train_probe yaml ===
        # Single source of truth with train_probe.py (which uses the same
        # INIT/GOAL constants). We still call sample_traj_segment_from_dset
        # to populate env_info -- env.update_env(env_info) is what gives each
        # parallel env worker its concrete maze instance; without it, env.prepare
        # silently fails on workers that never got initialized.
        elif self.goal_source == "fixed":
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            train_cfg_path = os.path.join(
                get_original_cwd(), "conf", "train_probe_point_maze.yaml"
            )
            train_cfg = OmegaConf.load(train_cfg_path)
            init_state_arr = np.array(
                OmegaConf.to_container(train_cfg.probe.init_state, resolve=True),
                dtype=np.float32,
            )
            goal_state_arr = np.array(
                OmegaConf.to_container(train_cfg.probe.goal_state, resolve=True),
                dtype=np.float32,
            )
            # Tile to (n_evals, state_dim) so every parallel eval shares the
            # same task. Differences across the 5 evals then come only from
            # CEM's per-eval random seed, not from task variation.
            fixed_init = np.tile(init_state_arr, (self.n_evals, 1))
            fixed_goal = np.tile(goal_state_arr, (self.n_evals, 1))

            obs_0, state_0 = self.env.prepare(self.eval_seed, fixed_init)
            obs_g, state_g = self.env.prepare(self.eval_seed, fixed_goal)

            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = fixed_init
            self.state_g = fixed_goal
            self.gt_actions = None
        # === END HARNESS EDIT ===
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            ### HARNESS EDIT ### teleport to dataset init/goal states (2 renders) instead of replaying all frameskip*goal_H steps
            # We only keep the first/last frame anyway, and the dataset gives us both
            # endpoint states, so teleport+render each (mirrors the random_state branch)
            # rather than simulating the whole segment (~10 min at goal_H=39).
            #
            # --- original (replayed every step, rendering them all) ---
            # exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # rollout_obses, rollout_states = self.env.rollout(
            #     self.eval_seed, init_state, exec_actions.numpy()
            # )
            # self.obs_0 = {k: np.expand_dims(arr[:, 0], axis=1) for k, arr in rollout_obses.items()}
            # self.obs_g = {k: np.expand_dims(arr[:, -1], axis=1) for k, arr in rollout_obses.items()}
            # self.state_g = rollout_states[:, -1]
            goal_state = np.array([x[-1] for x in states])
            obs_0_env, _ = self.env.prepare(self.eval_seed, init_state)
            obs_g_env, _ = self.env.prepare(self.eval_seed, goal_state)
            self.obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0_env.items()}
            self.obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g_env.items()}
            self.state_g = goal_state  # (b, d)
            ### END HARNESS EDIT ###
            self.state_0 = init_state  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, min(max_offset, self.max_offset))  ### HARNESS EDIT ### cap start depth (default 50)
            ### HARNESS EDIT ### report which val episode + frame range each eval's init/goal came from
            print(
                f"  [eval {i}] episode {traj_id}: frames {offset}..{offset + traj_len - 1} "
                f"(init=frame {offset}, goal=frame {offset + traj_len - 1})"
            )
            ### END HARNESS EDIT ###
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        ### HARNESS EDIT ### reuse MPC's cached executed frames for the final video/metrics (no full re-roll)
        precomputed_env = None
        if getattr(self.planner, "executed_obses", None) is not None:
            precomputed_env = (self.planner.executed_obses, self.planner.executed_states)
        ### END HARNESS EDIT ###
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final",
            full_video=True,  ### HARNESS EDIT ### one video spanning the whole trajectory
            precomputed_env=precomputed_env,  ### HARNESS EDIT ###
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        # weights_only=False: checkpoints store full pickled nn.Module objects
        # (e.g. ViTPredictor), not plain state-dicts. Safe here because these
        # are our own locally-trained checkpoints. Do NOT use for untrusted files.
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    ### HARNESS EDIT ### single device for WM + sim/renderer (override with device=cuda:7); was hardcoded cuda:0
    sim_device = cfg_dict.get("device") or "cuda:0"
    device = torch.device(sim_device if torch.cuda.is_available() else "cpu")
    ### END HARNESS EDIT ###
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = os.path.join(get_original_cwd(), f"checkpoints/outputs/{cfg_dict['model_name']}/")

    print("cwd:", os.getcwd())
    print("model_path:", model_path)
    print("abs model_path:", os.path.abspath(model_path))
    print("exists:", os.path.exists(os.path.join(model_path, "hydra.yaml")))

    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    # Optionally override the dataset data_path baked into the training config.
    # The training run stores an absolute path (e.g. /newdata2/...) that may not
    # exist when planning in a container where the repo is mounted elsewhere.
    data_path_override = cfg_dict.get("data_path")
    if data_path_override:
        with open_dict(model_cfg):
            model_cfg.env.dataset.data_path = data_path_override
        print(f"Overriding dataset data_path -> {data_path_override}")

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # IsaacLab path: single process, one GPU, n_evals batched as the IsaacLab
    # num_envs dimension. No subprocesses (Isaac Sim is one app per process).
    if model_cfg.env.name.startswith("isaaclab_"):
        from env.isaaclab.grid_venv import GridVectorEnv
        kwargs = dict(model_cfg.env.kwargs)
        kwargs.pop("num_envs", None)
        ### HARNESS EDIT ### pin the sim/renderer GPU to the same device as the WM
        kwargs.pop("device", None)
        env = GridVectorEnv(num_envs=cfg_dict["n_evals"], device=sim_device, **kwargs)
        ### END HARNESS EDIT ###
        ### HARNESS EDIT ### Ctrl+C -> force clean exit (Kit ignores SIGINT and hangs, leaking GPU mem)
        import signal
        from env.isaaclab.app_launcher import close_or_exit as _close_or_exit
        signal.signal(signal.SIGINT, lambda *_a: _close_or_exit(env))
        ### END HARNESS EDIT ###
    # use dummy vector env for wall and deformable envs
    elif model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    ### HARNESS EDIT ### force clean exit for isaaclab; Kit's teardown hangs otherwise and leaks GPU mem.
    # close_or_exit = env.close() + os._exit(0) with a watchdog that force-exits if close hangs. Does not return.
    if model_cfg.env.name.startswith("isaaclab_"):
        from env.isaaclab.app_launcher import close_or_exit
        close_or_exit(env)
    ### END HARNESS EDIT ###
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
