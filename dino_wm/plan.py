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

        ### HARNESS EDIT ### scene_filter: enumerate matching (episode, init, goal) segments over the
        # SAME valid-split episodes the sampler uses (self.dset), so pool indices line up with
        # self.dset[i]. (Building from the full states.pth would mis-index the valid TrajSubset.)
        self.scene_pool = None
        self.law_eval_size = None
        _goal_source = cfg_dict.get("goal_source")
        _sf = {k: v for k, v in (cfg_dict.get("scene_filter") or {}).items() if v is not None}
        _scene_offset = cfg_dict.get("scene_offset")
        # CHERRY-PICK (scene_ids): explicit GLOBAL pool indices override the contiguous offset slice.
        _scene_ids = cfg_dict.get("scene_ids")
        self.scene_ids = [int(i) for i in _scene_ids] if _scene_ids else None
        if self.scene_ids is not None:
            cfg_dict["n_evals"] = len(self.scene_ids)
        if _goal_source == "law_eval":
            # LAW-EVAL BENCHMARK: the pool is the pre-generated scenarios in goal_file_path/states.pth,
            # NOT the dset scene_pool. Read its size early so the last batch's n_evals is clamped
            # (mutually exclusive with the scene_filter/scene_offset dset pool path below).
            _M = int(torch.load(Path(cfg_dict["goal_file_path"]) / "states.pth").shape[0])
            self.law_eval_size = _M
            if self.scene_ids is not None:
                _bad = [i for i in self.scene_ids if not (0 <= i < _M)]
                if _bad:
                    raise ValueError(f"scene_ids {_bad} out of range for law_eval size {_M}")
            else:
                _off = int(_scene_offset or 0)
                _avail = max(0, _M - _off)
                if _avail == 0:
                    raise ValueError(f"scene_offset {_off} >= law_eval size {_M} ({cfg_dict['goal_file_path']} exhausted)")
                cfg_dict["n_evals"] = min(cfg_dict["n_evals"], _avail)
        elif _sf or _scene_offset is not None:          # build the pool for filtered runs OR pool sweeps
            from scripts.scene_index import select_pairs_from_states
            _base = getattr(self.dset, "dataset", self.dset)             # TrajSubset -> base dataset
            _idxs = list(getattr(self.dset, "indices", range(len(self.dset))))
            _states = _base.states[_idxs].numpy()
            _seq = np.asarray(_base.seq_lengths)[_idxs]
            self.scene_pool = select_pairs_from_states(_states, _seq, cfg_dict["goal_H"], **_sf)
            print(f"[scene_filter] {len(self.scene_pool)} matching (episode,init,goal) for {_sf or 'ALL moving segments'}")
            if not self.scene_pool:
                raise ValueError(f"scene_filter {_sf} matched 0 segments (goal_H={cfg_dict['goal_H']}); loosen it.")
            # POOL SWEEP: run the DETERMINISTIC slice pool[offset:offset+n_evals] (one eval per distinct
            # segment, no shuffle). Clamp n_evals so the last batch doesn't over-run the pool.
            if self.scene_ids is not None:
                _bad = [i for i in self.scene_ids if not (0 <= i < len(self.scene_pool))]
                if _bad:
                    raise ValueError(f"scene_ids {_bad} out of range for pool size {len(self.scene_pool)}")
            elif _scene_offset is not None:
                _avail = max(0, len(self.scene_pool) - int(_scene_offset))
                if _avail == 0:
                    raise ValueError(f"scene_offset {_scene_offset} >= pool size {len(self.scene_pool)} (pool exhausted)")
                cfg_dict["n_evals"] = min(cfg_dict["n_evals"], _avail)

        # DETERMINISM (fair cross-technique comparison): seed torch/np/python-random from cfg seed so
        # the planner's sampling + any shuffles are reproducible. (Train/valid split is already fixed
        # at seed 42, and scene_offset makes segment selection deterministic.)
        random.seed(cfg_dict["seed"]); np.random.seed(cfg_dict["seed"])
        torch.manual_seed(cfg_dict["seed"]); torch.cuda.manual_seed_all(cfg_dict["seed"])

        # Per-eval env seeds. Under a pool sweep, index by the GLOBAL pool position (scene_offset + n)
        # so a given segment gets the SAME env seed regardless of --batch -> the same episode is set
        # up identically across techniques and batchings.
        # scene_ids: seed each picked scene by its GLOBAL index so the re-run reproduces the exact
        # env setup it had in the full sweep (seed*id+1). Else index by the contiguous slice position.
        if self.scene_ids is not None:
            self.eval_seed = [cfg_dict["seed"] * i + 1 for i in self.scene_ids]
        else:
            _off0 = int(_scene_offset) if _scene_offset is not None else 0
            self.eval_seed = [cfg_dict["seed"] * (_off0 + n) + 1 for n in range(cfg_dict["n_evals"])]
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
            # Optional raw action range -> planners clamp to the training range
            # (None for datasets without it, keeping the legacy [-1,1] clamp).
            action_min=getattr(self.dset, "action_min", None),
            action_max=getattr(self.dset, "action_max", None),
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
        self.evaluator.video = self.cfg_dict.get("video", False)  ### HARNESS EDIT ### gate per-step (smooth) video capture
        self.evaluator.gt_actions = self.gt_actions  ### HARNESS EDIT ### GT (dataset) actions for cem_debug scoring
        self.evaluator.cem_debug = self.cfg_dict.get("cem_debug", False)  ### per-solve CEM score-spread + GT-vs-winner debug
        self._rrt_introspect_cfg = self.cfg_dict.get("rrt_introspect")  ### wired onto the planner below

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

        ### HARNESS EDIT ### hand the introspection config to the (MPC) planner. OFF unless
        # rrt_introspect.enabled=true. Attribute lookup in mpc.py, so a no-op for other planners.
        if self._rrt_introspect_cfg is not None:
            self.planner.introspect_cfg = self._rrt_introspect_cfg

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        ### HARNESS EDIT ### decouple goal_H from MPC horizon/cadence (goal_H = goal distance only)
        # orig (MPC branch): sub_planner.horizon = n_taken_actions = goal_H -> collapses MPC to open-loop
        if isinstance(self.planner, MPCPlanner):
            pass  # keep configured sub_planner.horizon / n_taken_actions
        else:
            self.planner.horizon = cfg_dict["goal_H"]
        ### END HARNESS EDIT ###

        # (Single-robot: no second arm to freeze — the planner optimizes all 7
        # action dims. The two-arm freeze-right glue was removed.)

        ### HARNESS EDIT ### legislation: build the law-derived constraint + inject it into the
        # planner. mode = social (prune -> never break the law) | deviant (don't prune -> minimize
        # violations, strictly preferring fewer-violation goal paths) | off (no constraint = rational).
        leg = self.cfg_dict.get("legislation") or {}
        mode = leg.get("mode", None)
        if mode is None:                                   # back-compat: enforce=true/false -> social/off
            mode = "social" if leg.get("enforce", False) else "off"
        mode = str(mode).lower()
        if mode not in ("social", "deviant", "off"):
            raise ValueError(f"legislation.mode must be social|deviant|off, got {mode!r}")
        target = getattr(self.planner, "sub_planner", self.planner)  # MPC -> sub_planner
        target.mode = mode                                 # how the planner USES the constraint (prune vs rank)
        self.planner.sign_flip = self.cfg_dict.get("sign_flip")   # exogenous sign-flip schedule (MPC loop reads it)
        # OBSERVE-ONLY OFF: the SIGN is an EXTERNAL world authority adjudicated on GROUND TRUTH (R7/R7b),
        # so when an exogenous sign exists (sign_flip configured) it must flip for the realistic agent too.
        # Build the LawEvaluator in off as well -- it perceives, flips the sign, and dumps the ledger (so
        # off gets the SAME gt_banned abidance metric as social/deviant) -- but the planner IGNORES the
        # verdict entirely (no prune / reweight / obligation-steer; gated on mode=="off" in rrt.py). With
        # no sign_flip, off stays truly rational (no evaluator, no ledger). See the mode gates in rrt.py.
        _sf = self.cfg_dict.get("sign_flip") or {}
        _observe_off = (mode == "off") and (_sf.get("frame") is not None)
        if mode in ("social", "deviant") or _observe_off:
            from probes.registry import ProbeRegistry
            from legislation.reasoner import LegislativeReasoner
            from legislation.constraint import Constraint
            from legislation.enforcement import LawEvaluator
            from probes.probe_cube_cells import CUBE_HALF
            reg = ProbeRegistry(device=self.device)
            # SINGLE PROBE SOURCE (no mismatch): tie the legislation cube_position probe to the SAME
            # probe the planner navigates with (objective.pos_probe_path). Perception used for law
            # enforcement then CANNOT differ from perception used for planning -- one CLI knob controls
            # both. (probes.yaml's cube_position path is only a fallback default when no objective probe.)
            _pos_path = (self.cfg_dict.get("objective") or {}).get("pos_probe_path")
            if _pos_path:
                reg.set_probe("cube_position", _pos_path)
                print(f"[legislation] cube_position tied to planner probe -> {_pos_path}")
            base_facts = list(leg.get("facts", ["cube"]))
            # Static yellow-cell designations (full_lawset R5/R7/R7b guard on yellow_cell(Y)). Kept as a
            # plain int-list config key (not literal facts) so Hydra never has to parse `yellow_cell(3)`
            # -- the parens break its override grammar. Grounding stays otherwise probe-derived.
            base_facts += [f"yellow_cell({int(c)})" for c in (leg.get("yellow_cells") or [])]
            _db = leg.get("db_path")                       # null -> geometry 1 (legal_database.yaml)
            _lawsets = leg.get("active_lawsets")            # null -> the db's own active_lawsets default
            _lawsets = list(_lawsets) if _lawsets else None
            _rkw = {}
            if _db:
                _rkw["db_path"] = _db
            if _lawsets is not None:
                _rkw["active_lawsets"] = _lawsets           # which law CATEGORY(ies) to enforce
            reasoner = LegislativeReasoner(**_rkw)
            # CUSHION (robust constraint tightening): inflate the footprint the PLANNER checks against by
            # `constraint_margin` delta (m) to stay clear despite WM/probe perception error. PLANNER-ONLY
            # -- the metric (planning_metrics) scores reality with the true CUBE_HALF, so the delta=0 vs
            # delta>0 ablation is honest. Default 0.0. See methods.md (cushion ablation).
            _margin = float(leg.get("constraint_margin", 0.0))
            _ch = CUBE_HALF + _margin
            # per-step law: perceive -> ground -> reason -> Constraint, re-run each re-plan so the
            # verdict tracks the live state (sign colour, cells already visited, ...).
            evaluator = LawEvaluator(reasoner, reg, base_facts=base_facts, cube_half=_ch)
            penalty = float(leg.get("violation_penalty", 1e6))
            target.law_fn = evaluator
            # POSITIVE-OBLIGATION GOAL BANK (opt-in): a path to scripts/gen_goal_cell_bank.py output.
            # When set, a live positive obligation ([O]in_cell(k) / [O]in_yellow_cell) retargets the
            # planner objective to that cell's centered goal image (a waypoint); null -> obligations are
            # parsed but not acted on (current behaviour). See legislation/goal_bank.py.
            target.goal_bank_path = leg.get("goal_bank")
            # GOAL AS SPECIFICATION: on a law_eval benchmark run, hand the evaluator the pair's
            # GROUND-TRUTH goal cell (metadata.json) so the reach-goal obligation targets the DESIGNATED
            # cell, not a probe-perceived one (cube_cells argmax flips near cell boundaries). The goal
            # cell is a task spec (like a prompt), not a perception; current-STATE facts stay grounded.
            if self.cfg_dict.get("goal_source") == "law_eval" and self.cfg_dict.get("goal_file_path"):
                import json as _json
                _mdp = Path(self.cfg_dict["goal_file_path"]) / "metadata.json"
                if _mdp.exists():
                    _gc = (_json.loads(_mdp.read_text()) or {}).get("goal_cell")
                    if _gc is not None:
                        evaluator.gt_goal_cell = int(_gc)
                        print(f"[law_eval] GT goal_cell = {int(_gc)} (task spec, not perceived)")
            target.violation_penalty = penalty
            # initial/static constraint from base facts only -- the setup verdict, and the fallback
            # for planners that don't re-evaluate per step (e.g. the chained CEM). RRT overwrites
            # target.constraint each step via law_fn.
            target.constraint = Constraint.from_reasoner(reasoner, base_facts, reg.probes, cube_half=_ch)
            _obs_note = "  [OBSERVE-ONLY: sign+ledger, planner ignores verdict]" if _observe_off else ""
            print(f"[legislation] mode={mode}{_obs_note} | base facts {base_facts} | cushion δ={_margin:.3f}m "
                  f"| initial {target.constraint} | violation_penalty={penalty:g}")
            _law_meta = {"mode": mode, "observe_only_off": _observe_off,
                         "db_path": _db or "legislation/legal_database.yaml",
                         "active_lawsets": list(reasoner.active_lawsets) if reasoner.active_lawsets else None,
                         "cushion_margin": _margin}
        else:
            print("[legislation] mode=off (rational agent) -- no constraint")
            _law_meta = {"mode": "off", "db_path": None, "active_lawsets": None, "cushion_margin": 0.0}
        ### END HARNESS EDIT ###
        # PROVENANCE: mark the DATASET + resolved LAW SET for this plan run (manifest.json beside
        # eval_metrics.json). Best-effort — never break planning over it.
        try:
            import provenance
            provenance.write(".", "plan.py", extra={
                "dataset": {"data_path": self.cfg_dict.get("data_path"),
                            "goal_source": self.cfg_dict.get("goal_source"),
                            "goal_file_path": (str(self.cfg_dict.get("goal_file_path"))
                                               if self.cfg_dict.get("goal_file_path") else None)},
                "laws": _law_meta,
                "model": {"name": self.cfg_dict.get("model_name"), "epoch": self.cfg_dict.get("model_epoch")},
            })
        except Exception as _pe:
            print("[provenance] plan manifest skipped:", _pe)

        self.dump_targets()

    @staticmethod
    def _add_time_dim(obs):
        """Insert a singleton time axis: each obs array (b, ...) -> (b, 1, ...)."""
        return {k: np.expand_dims(v, axis=1) for k, v in obs.items()}

    def prepare_targets(self):
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

            self.obs_0 = self._add_time_dim(obs_0)
            self.obs_g = self._add_time_dim(obs_g)
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

            self.obs_0 = self._add_time_dim(obs_0)
            self.obs_g = self._add_time_dim(obs_g)
            self.state_0 = fixed_init
            self.state_g = fixed_goal
            self.gt_actions = None
        # === END HARNESS EDIT ===
        elif self.goal_source == "law_eval":
            ### HARNESS EDIT ### LAW-EVAL BENCHMARK: teleport to the pre-generated (init, goal) cube
            # scenarios from scripts/gen_law_eval_set.py (states.pth (M,2,31)). Slice
            # [scene_offset : +n_evals] so eval_sweep-style batching walks the whole set. Re-renders
            # from the saved states (deterministic; the saved obses/ images are for inspection).
            _d = Path(self.cfg_dict["goal_file_path"])
            _st = torch.load(_d / "states.pth").float().numpy()          # (M, 2, 31)  [init, goal]
            if getattr(self, "scene_ids", None) is not None:             # cherry-picked indices
                _pick = self.scene_ids
            else:
                _off = int(self.cfg_dict.get("scene_offset") or 0)
                _pick = range(_off, _off + self.n_evals)
            init_state, goal_state = _st[list(_pick), 0], _st[list(_pick), 1]   # (b, 31) each
            # sign-flip: set the PRE-FLIP baseline colour before the first perception so MPC step 0
            # (and the goal frame) is grounded at a known sign, not the sim's default. See conf sign_flip.
            _sf = self.cfg_dict.get("sign_flip")
            if _sf and _sf.get("frame") is not None and hasattr(self.env, "set_sign_color"):
                self.env.set_sign_color(_sf.get("color", "yellow") if int(_sf["frame"]) == 0
                                        else _sf.get("base_color", "white"))
            obs_0_env, _ = self.env.prepare(self.eval_seed, init_state)
            obs_g_env, _ = self.env.prepare(self.eval_seed, goal_state)
            self.obs_0 = self._add_time_dim(obs_0_env)
            self.obs_g = self._add_time_dim(obs_g_env)
            self.state_0, self.state_g = init_state, goal_state
            self.gt_actions = None
            self.law_eval_size = int(_st.shape[0])                       # for eval_sweep pool sizing
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
            self.obs_0 = self._add_time_dim(obs_0_env)
            self.obs_g = self._add_time_dim(obs_g_env)
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
        ### HARNESS EDIT ### require the cube to CHANGE CELL between init and goal. Otherwise
        # the goal is a no-op: the cube-position objective has ~zero gradient (nothing to push
        # toward) and GD can't shape the random action init into a push -> jittery, goal-less
        # rollouts. Retry (traj, offset) until the cube changes cell; keep the max-displacement
        # candidate as a fallback if none is found within the cap. Toggle with
        # planning cfg goal_require_cube_move=false.
        from env.isaaclab.grid_metadata import cell_labels_from_states_single
        require_move = self.cfg_dict.get("goal_require_cube_move", True)
        MAX_TRIES = 300

        # scene_filter (plan.yaml): planning_main pre-enumerates matching (episode, init, goal)
        # segments via scripts/scene_index -> robust, deterministic selection of explicit/rare
        # cell pairs that random sampling misses. Only the goal segment uses it (not the
        # traj_len=2 env_info probe that random_state/fixed also call).
        pool = getattr(self, "scene_pool", None)
        use_pool = pool is not None and traj_len == self.frameskip * self.goal_H + 1

        def seg_at(traj_id, offset):
            obs, act, state, e_info = self.dset[traj_id]
            state_np = state.numpy()
            s0, sg = state_np[offset], state_np[offset + traj_len - 1]
            return {
                "traj_id": traj_id, "offset": offset, "obs": obs, "act": act,
                "state": state_np, "e_info": e_info,
                "c0": int(cell_labels_from_states_single(s0)),
                "cg": int(cell_labels_from_states_single(sg)),
                "cube_l2": float(np.linalg.norm(sg[18:20] - s0[18:20])),
            }

        picks = None
        if use_pool:
            _off = self.cfg_dict.get("scene_offset")
            if getattr(self, "scene_ids", None) is not None:  # cherry-picked global indices
                picks = [list(pool)[i] for i in self.scene_ids]
            elif _off is not None:                          # pool sweep: deterministic slice, one seg per eval
                picks = list(pool)[int(_off):]              # n_evals already clamped in __init__ to fit this slice
            else:
                picks = list(pool); random.shuffle(picks)

        for i in range(self.n_evals):
            if use_pool:
                e, f0, _fg = picks[i % len(picks)]          # pre-filtered explicit-cell segment
                seg = seg_at(e, f0)
            else:
                best = None  # highest-cube_l2 candidate, the require_move fallback
                for _try in range(MAX_TRIES):
                    max_offset = -1
                    while max_offset < 0:
                        traj_id = random.randint(0, len(self.dset) - 1)
                        obs, act, state, e_info = self.dset[traj_id]
                        max_offset = obs["visual"].shape[0] - traj_len
                    state_np = state.numpy()
                    offset = random.randint(0, min(max_offset, self.max_offset))
                    s0, sg = state_np[offset], state_np[offset + traj_len - 1]
                    seg = {"traj_id": traj_id, "offset": offset, "obs": obs, "act": act,
                           "state": state_np, "e_info": e_info,
                           "c0": int(cell_labels_from_states_single(s0)),
                           "cg": int(cell_labels_from_states_single(sg)),
                           "cube_l2": float(np.linalg.norm(sg[18:20] - s0[18:20]))}
                    if best is None or seg["cube_l2"] > best["cube_l2"]:
                        best = seg
                    moved = seg["c0"] != seg["cg"] and seg["c0"] != -1 and seg["cg"] != -1
                    if not require_move or moved:
                        break
                else:
                    seg = best
                    print(f"  [eval {i}] WARN: no cell-changing segment in {MAX_TRIES} tries; "
                          f"using max-displacement fallback")

            o = seg["offset"]
            print(f"  [eval {i}] episode {seg['traj_id']}: frames {o}..{o + traj_len - 1}  "
                  f"cube cell {seg['c0']}->{seg['cg']}  cube_l2={seg['cube_l2']:.3f} m")
            observations.append({key: arr[o:o + traj_len] for key, arr in seg["obs"].items()})
            states.append(seg["state"][o:o + traj_len])
            actions.append(seg["act"][o:o + self.frameskip * self.goal_H])
            env_info.append(seg["e_info"])
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
        ### HARNESS EDIT ### post-hoc: dump the per-eval normative ledger (observed facts + verdicts +
        # committed intents) to the output folder. Analysis only; does not affect planning.
        _leg = getattr(getattr(self.planner, "sub_planner", self.planner), "law_fn", None)
        if _leg is not None:
            _leg.dump("normative_ledger.json")
            print("Dumped normative ledger to", os.path.abspath("normative_ledger.json"))
        ### HARNESS EDIT ### reuse MPC's cached executed frames for the final video/metrics (no full re-roll)
        precomputed_env = None
        if getattr(self.planner, "executed_obses", None) is not None:
            precomputed_env = (self.planner.executed_obses, self.planner.executed_states)
        ### END HARNESS EDIT ###
        logs, successes, _, e_states = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final",
            full_video=True,  ### HARNESS EDIT ### one video spanning the whole trajectory
            precomputed_env=precomputed_env,  ### HARNESS EDIT ###
            ### HARNESS EDIT ### closed-loop imagined row (re-grounded per-step frames) if MPC produced them
            precomputed_imagined=getattr(self.planner, "imagined_regrounded", None),
        )
        ### HARNESS EDIT ### dump per-eval metrics for scripts/eval_sweep.py. Computation lives in
        # planning/planning_metrics.py to keep this workspace lean. Never let the dump crash a run.
        try:
            from planning.planning_metrics import build_eval_metrics
            _tgt = getattr(self.planner, "sub_planner", self.planner)
            # RUNTIME split (from the RRT's accumulated timers; see planning/rrt.py reset):
            #   RRT search = _t_plan - _t_reason - _t_prune ; LEGISLATION = reason (DDL observe/set_goal)
            #   + prune (constraint.violations). off/rational -> reason=prune=0 (no law attached).
            _tp = float(getattr(_tgt, "_t_plan", 0.0))
            _trs = float(getattr(_tgt, "_t_reason", 0.0)); _tpr = float(getattr(_tgt, "_t_prune", 0.0))
            _rb = {"plan_total_s": round(_tp, 2), "rrt_s": round(_tp - _trs - _tpr, 2),
                   "legislation_s": round(_trs + _tpr, 2),
                   "legislation_reason_s": round(_trs, 2), "legislation_prune_s": round(_tpr, 2)}
            # FINE split of legislation_reason_s (probe/ground/logic/build), from the LawEvaluator's per-
            # episode accumulators -- lets Q5 show the clingo DDL logic is a negligible slice. off/rational
            # has no law_fn -> the dict is absent and these keys are simply omitted.
            _tim = getattr(getattr(_tgt, "law_fn", None), "timing", None)
            if _tim:
                _rb.update({"leg_probe_s": round(_tim.get("probe_s", 0.0), 3),
                            "leg_ground_s": round(_tim.get("ground_s", 0.0), 3),
                            "leg_logic_s": round(_tim.get("logic_s", 0.0), 3),
                            "leg_build_s": round(_tim.get("build_s", 0.0), 3),
                            "leg_n_observe": int(_tim.get("n_observe", 0))})
            _metrics = build_eval_metrics(
                e_states=e_states, action_len=action_len,
                last_metrics=getattr(self.evaluator, "last_metrics", {}),
                constraint=getattr(_tgt, "constraint", None),
                scene_filter=self.cfg_dict.get("scene_filter"),
                metric_cell=self.cfg_dict.get("metric_cell"),
                scene_offset=self.cfg_dict.get("scene_offset"),
                pool_size=(self.law_eval_size if self.law_eval_size is not None
                           else (len(self.scene_pool) if self.scene_pool is not None else None)),
                n_evals=self.n_evals, seed=self.cfg_dict["seed"],
                wm_pred_err=getattr(self.planner, "wm_pred_err_mean", None),
                wm_latent_err=getattr(self.planner, "wm_latent_err_mean", None),
                wm_pred_err_steps=getattr(self.planner, "wm_pred_err_steps", None),
                wm_pred_xy_steps=getattr(self.planner, "wm_pred_xy_steps", None),
                wm_real_xy_steps=getattr(self.planner, "wm_real_xy_steps", None),
                wm_probe_start_xy_steps=getattr(self.planner, "wm_probe_start_xy_steps", None),
                wm_latent_err_steps=getattr(self.planner, "wm_latent_err_steps", None),
                runtime_breakdown=_rb, goal_states=getattr(self, "state_g", None))
            with open("eval_metrics.json", "w") as _f:
                json.dump(_metrics, _f, indent=2)
            print("Dumped eval metrics to", os.path.abspath("eval_metrics.json"))
            # DENORMALIZED committed strokes per eval (b, T, 4) -> replay via env.rollout(seed, init_state, actions)
            _strokes = getattr(self.planner, "executed_strokes", None)
            if _strokes:
                import numpy as _np
                _np.save("executed_actions.npy", _np.concatenate(_strokes, axis=1))
                print("Saved executed actions to", os.path.abspath("executed_actions.npy"))
        except Exception as _ex:  # noqa: BLE001
            print("[eval_metrics] dump failed:", _ex)
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
            ckpt = torch.load(decoder_path, weights_only=False)  # our ckpt; torch>=2.6 defaults weights_only=True
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path, weights_only=False)
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
    ### HARNESS EDIT ### resolve the run dir robustly: training now writes to
    # outputs/<model_name>/ (hydra.yaml + checkpoints/ directly under it). Try that
    # first; fall back to the legacy checkpoints/outputs/<model_name> symlink layout.
    _cwd = get_original_cwd()
    model_path = os.path.join(_cwd, "outputs", cfg_dict["model_name"])
    if not os.path.exists(os.path.join(model_path, "hydra.yaml")):
        model_path = os.path.join(_cwd, "checkpoints", "outputs", cfg_dict["model_name"])
    model_path = model_path + "/"
    ### END HARNESS EDIT ###

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

    # Optionally attach a standalone-trained decoder so the evaluator can render the
    # IMAGINED rollout (decode_obs on the WM's predicted latents). The WM's saved config
    # has has_decoder=False; these plan-time overrides flip it on and point load_model()
    # at the decoder checkpoint. Mirrors the data_path override above; decoder_path is
    # resolved relative to the repo root inside load_model().
    has_decoder_override = cfg_dict.get("has_decoder")
    if has_decoder_override is not None:
        with open_dict(model_cfg):
            model_cfg.has_decoder = bool(has_decoder_override)
        print(f"Overriding has_decoder -> {bool(has_decoder_override)}")
    decoder_path_override = cfg_dict.get("decoder_path")
    if decoder_path_override is not None:
        with open_dict(model_cfg):
            model_cfg.env.decoder_path = decoder_path_override
        print(f"Overriding decoder_path -> {decoder_path_override}")

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    ### HARNESS EDIT ### clamp n_evals to the scene pool BEFORE building the env, so the IsaacLab
    # env's num_envs matches the number of init states prepare_targets will produce. Without this a
    # pool smaller than --batch (e.g. an init:goal pair with few matching segments) builds an
    # N-env sim but only M<N init states -> write_joint_state size mismatch. PlanWorkspace
    # re-clamps identically (this just moves the clamp ahead of env construction).
    _sf = {k: v for k, v in (cfg_dict.get("scene_filter") or {}).items() if v is not None}
    _soff = cfg_dict.get("scene_offset")
    _sids = cfg_dict.get("scene_ids")
    if _sids:
        # CHERRY-PICK: n_evals is exactly the number of picked indices (bounds checked in PlanWorkspace).
        cfg_dict["n_evals"] = len(_sids)
        print(f"[scene_ids] n_evals -> {cfg_dict['n_evals']} (explicit indices {list(_sids)})")
    elif cfg_dict.get("goal_source") == "law_eval":
        # law-eval benchmark: clamp n_evals to the (init,goal) SET SIZE (states.pth (M,2,31)), sliced
        # by scene_offset -- env num_envs must match the states prepare_targets teleports to.
        _m = int(torch.load(Path(cfg_dict["goal_file_path"]) / "states.pth").shape[0])
        _off = int(_soff or 0)
        cfg_dict["n_evals"] = min(cfg_dict["n_evals"], max(1, _m - _off))
        print(f"[law_eval] n_evals -> {cfg_dict['n_evals']} (set size {_m}, offset {_off})")
    elif _soff is not None:
        from scripts.scene_index import select_pairs_from_states
        _base = getattr(dset, "dataset", dset)
        _idxs = list(getattr(dset, "indices", range(len(dset))))
        _pool = select_pairs_from_states(_base.states[_idxs].numpy(),
                                         np.asarray(_base.seq_lengths)[_idxs],
                                         cfg_dict["goal_H"], **_sf)
        _avail = max(0, len(_pool) - int(_soff))
        if _avail == 0:
            raise ValueError(f"scene_offset {_soff} >= pool size {len(_pool)} (pool exhausted)")
        cfg_dict["n_evals"] = min(cfg_dict["n_evals"], _avail)
        print(f"[pool clamp] n_evals -> {cfg_dict['n_evals']} (pool {len(_pool)}, offset {_soff})")
    ### END HARNESS EDIT ###

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
        env = GridVectorEnv(num_envs=cfg_dict["n_evals"], device=sim_device,
                            tiled_camera=bool(cfg_dict.get("tiled_camera", False)), **kwargs)
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
