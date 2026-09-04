"""Collect single-robot PLANAR-STROKE rollouts from the DinoWMGrid-Single env
into dino_wm .pth format.

One stroke = one recorded timestep. Each episode is a FRESH random cube spawn
followed by T high-level push strokes; each stroke
    action = [x_start, y_start, dx, dy]   (env-local grid-plane meters; end = start + disp)
is executed by GridWrapperSingle.execute_stroke (StrokeExecutor + the env's IK
term) over many internal sim steps, and we record obs/state at the stroke
boundary only -- with the arm PARKED at a fixed home pose (retract phase), so the
recorded frames show only the cube changing. The WM trains as a one-step dynamics
model over strokes (num_hist=3, num_pred=1, frameskip=1 -> one stroke per frame).

Stroke sampling (env.isaaclab.stroke_sampler.StrokeSampler) is MIXED: each stroke is
independent and, per --aimed_frac (default 0.6), either AIMS at the cube (start behind
it, push through -> teaches dynamics) or is UNIFORM over the workspace (covers the
planner's action space, incl. strokes that miss, keeping CEM/GD samples in-dist). The
DINO-WM deformable dataset is pure-uniform, but that relies on granular material filling
the workspace; our small cube would make uniform ~90% no-ops, so we add the aimed
regime to recover contact density. Off-grid drift is contained by the wrapper's hard
cube clamp. (See memory: deformable-datagen-tricks.)

Writes ${output_dir}/:
  states.pth        (E, T, 31)
  actions.pth       (E, T, 4)        [x_start, y_start, dx, dy]
  proprio.pth       (E, T, 18)
  cell_labels.pth   (E, T) int64     (single cube)
  seq_lengths.pth   (E,) int64
  sign_colors.pth   (E,) int64       (per-episode sign color, indexes SIGN_PALETTE)
  metadata.json     collection params (provenance)
  obses/episode_NNNNN.pth   (T, H, W, 3) uint8

The .pth arrays are re-saved every --save_every episodes and on Ctrl-C, so an
interrupted run leaves a valid dataset of everything collected so far.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import provenance
import numpy as np
import torch

from env.isaaclab.grid_wrapper_single import ACTION_DIM, GridWrapperSingle, PROPRIO_DIM
from env.isaaclab.grid_metadata import cell_labels_from_states_single
from env.isaaclab.stroke_sampler import StrokeSampler
from env.isaaclab.app_launcher import close_or_exit

IMG_HW = 224
EP_PAD = 5  # episode index zero-pad width (episode_00000.pth); must match the loader
_CUBE_XY = slice(18, 20)  # cube (x,y) within the 31-D state [arm9, jvel9, cube13]

# Stop-sign colors, randomized per episode (visual diversity + sets up the
# future sign-as-rule signal). Index recorded per episode in sign_colors.pth.
SIGN_PALETTE = [
    ("white", (1.0, 1.0, 1.0)),
    ("red", (1.0, 0.0, 0.0)),
    ("yellow", (1.0, 1.0, 0.0)),
    ("green", (0.0, 0.8, 0.0)),
]


def run_video_preview(args):
    """Render every internal sim step of one episode into a smooth mp4 (single env),
    so the arm/cube dynamics can be eyeballed before committing to a full collect.
    Writes no dataset."""
    import imageio
    env = GridWrapperSingle(
        task_id=args.task_id, num_envs=1, device=args.device,
        render_mode=args.render_mode, spp=args.spp, stroke_max_steps=args.stroke_max_steps,
        fast_stroke_render=False,  # render every step (frame_sink forces it anyway)
        lock_cube_yaw=args.lock_cube_yaw,
    )
    env.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    sampler = StrokeSampler(rng, aimed_frac=args.aimed_frac, push_max=args.push_max,
                            start_margin=args.start_margin,
                            aim_push_range=(args.aim_push_min, args.aim_push_max),
                            aim_offset_sd=args.aim_offset_sd)
    try:
        obs, state = env.reset()
        env.set_sign_color(SIGN_PALETTE[int(rng.randint(0, len(SIGN_PALETTE)))][1])
        obs, state = env._scene_outputs()
        mode = sampler.reset_episode()
        frames = [np.asarray(obs["visual"]).copy()]  # initial frame (N,H,W,3); N=1 here
        for t in range(args.video_strokes):
            obs, state = env.execute_stroke(sampler.sample(state[0, _CUBE_XY]), frame_sink=frames)
        out = Path(args.video_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out), fps=args.video_fps)
        for f in frames:
            fr = np.asarray(f)[0, ..., :3]   # env 0 (frame_sink holds per-env (N,H,W,3))
            if fr.dtype != np.uint8:
                fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8) if fr.max() <= 1.0 else fr.astype(np.uint8)
            writer.append_data(fr)
        writer.close()
        print(f"[video] {mode} episode, {args.video_strokes} strokes, {len(frames)} frames "
              f"@ {args.video_fps}fps -> {out.resolve()}")
    finally:
        close_or_exit(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-Single-v0")
    ap.add_argument("--num_episodes", type=int, default=5000,
                    help="5000 eps x 20 strokes = 100k strokes, ALL aimed at the default --aimed_frac "
                    "1.0, with planner-like near-misses supplied by --aim_offset_sd rather than by a "
                    "uniform regime. These defaults reproduce data/isaaclab_stroke_5k (the set wm_5k "
                    "was trained on); see its metadata.json.")
    ap.add_argument("--episode_len", type=int, default=20,
                    help="STROKES per trajectory (each is a full macro-step / many "
                    "sim steps). >= num_hist+num_pred (=4) or the slicer drops it; "
                    "20 gives 20-4+1=17 windows each.")
    _DEF_OUT = os.environ.get("DATASET_DIR", "./data") + "/isaaclab_single_stroke"
    ap.add_argument("--output_dir", default=_DEF_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0",
                    help="Must be 'cuda:0' for this Isaac Sim build. Mask with CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--render_mode", default="PathTracing",
                    choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--spp", type=int, default=128,
                    help="Samples-per-pixel for PathTracing (128 fine for training; "
                    "256+ only for demo videos).")
    ap.add_argument("--aimed_frac", type=float, default=1.0,
                    help="Fraction of strokes that AIM at the cube (start behind it, push through "
                    "-> teaches dynamics) vs UNIFORM strokes (blind action-space coverage). 1.0 = every "
                    "stroke aimed: miss coverage comes from --aim_offset_sd instead, which is "
                    "distribution-matched to the planner's own aim error rather than blind. "
                    "0.0 = pure-uniform (deformable parity).")
    ap.add_argument("--push_max", type=float, default=0.08,
                    help="UNIFORM strokes: per-axis push displacement bound (m): dx,dy ~ U(-push_max, "
                    "push_max). ~20%% of the grid span, matching the deformable short-push ratio. "
                    "Short pushes = small per-step cube motion = easier one-step dynamics.")
    ap.add_argument("--aim_push_min", type=float, default=0.05,
                    help="AIMED strokes: MIN cube-travel per push (m). MUST match planner.push_min in "
                    "conf/planner/mpc_rrt.yaml (currently 0.05) or the planner queries the WM off its "
                    "training band.")
    ap.add_argument("--aim_push_max", type=float, default=0.09,
                    help="AIMED strokes: MAX cube-travel per push (m). MUST match planner.push_max in "
                    "conf/planner/mpc_rrt.yaml (currently 0.09). A 0.03-0.13 band was drafted for a bigger "
                    "WM but adopted by neither the data nor the planner; do not re-widen one without the "
                    "other.")
    ap.add_argument("--aim_offset_sd", type=float, default=0.025,
                    help="AIMED strokes: lateral aim-offset std (m). Aims at a FALSE point ~N(0,sd) "
                    "perpendicular to the push, mimicking the planner aiming at its probe estimate "
                    "(off by the perception error ~0.03-0.05 m). |offset|<cube_half(0.045) still "
                    "contacts, larger grazes->misses. 0.0 = exact-contact. Paired with --aimed_frac 1.0 "
                    "this REPLACES blind uniform coverage with distribution-matched near-misses.")
    ap.add_argument("--start_margin", type=float, default=0.06,
                    help="Contact point sampled uniformly in +/-(GRID_HALF + start_margin) per "
                    "axis, so the pusher can get behind a cube sitting at/just past a grid edge.")
    ap.add_argument("--stroke_max_steps", type=int, default=320,
                    help="Cap on internal IK sim steps per stroke (4 phases x ~80: "
                    "approach, descend, push, retract-to-home).")
    ap.add_argument("--no_fast_render", action="store_true",
                    help="Disable the mid-stroke render skip (path-trace every internal "
                    "step). For A/B timing vs the default fast path; much slower at PathTracing.")
    # YAW LOCK. Spawns the cube axis-aligned and holds it there (solver clamp: zero max angular
    # velocity + heavy angular damping, plus a per-step re-pin). Abidance is scored with an
    # axis-aligned square footprint, so a yaw-locked dataset makes that model exact instead of an
    # upper bound -- but the world model then has to be RETRAINED on it, since a model trained on
    # yaw-varying data is out of distribution on locked rollouts (measured: +41% episode-mean
    # prediction error, +27% planning steps, results/yawlock 2026-09-01).
    # `default=None` on purpose, NOT False: None defers to DINOWM_LOCK_CUBE_YAW so the env-var path
    # keeps working, whereas a False default would silently override it. Passing the flag records
    # the choice in BOTH metadata.json and manifest.json, which an env var does not.
    ap.add_argument("--lock_cube_yaw", action="store_true", default=None,
                    help="Lock the cube's yaw to 0 (axis-aligned) for the whole collection. "
                    "Unset -> DINOWM_LOCK_CUBE_YAW, else off.")
    ap.add_argument("--num_envs", type=int, default=10,
                    help="Parallel envs collected per batch (GPU-batched physics + the "
                    "vectorized StrokeExecutor). Each env runs an independent episode; "
                    "~Nx throughput on top of the fast render skip.")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted collection in --output_dir instead of "
                         "overwriting it: preloads the existing episodes, keeps counting from "
                         "there, and OFFSETS the seed by the resume count (reusing --seed "
                         "unchanged would deterministically re-draw the same episodes). Orphan "
                         "obs files written past the last array checkpoint are dropped.")
    ap.add_argument("--save_every", type=int, default=25,
                    help="Re-save the .pth arrays roughly every N episodes (crash safety).")
    ap.add_argument("--video", action="store_true",
                    help="PREVIEW mode: render EVERY internal sim step (not just stroke "
                    "boundaries) for one episode and write a smooth dynamics mp4, then exit "
                    "WITHOUT writing a dataset. Use to eyeball the arm/cube before a full collect.")
    ap.add_argument("--video_out", default="./stroke_preview.mp4", help="output mp4 path for --video")
    ap.add_argument("--video_strokes", type=int, default=10, help="number of strokes to roll for --video")
    ap.add_argument("--video_fps", type=int, default=25,
                    help="mp4 frame rate for --video. 25 = real-time (1 frame = decimation*dt "
                    "= 0.04s sim). Bump higher (e.g. 50) to speed up for demos — pure playback "
                    "speed, no effect on the dataset (boundary-only) or physics.")
    args = ap.parse_args()

    # Resolve the yaw lock EXACTLY as GridWrapperSingle does (flag, else env var, else off) so
    # the directory name reflects what is actually collected -- setting DINOWM_LOCK_CUBE_YAW
    # without the flag would otherwise write yaw-locked data into the default unlocked path.
    # Only the DEFAULT is renamed: an explicit --output_dir is always honoured verbatim.
    args.lock_cube_yaw = (bool(int(os.environ.get("DINOWM_LOCK_CUBE_YAW", "0")))
                          if args.lock_cube_yaw is None else bool(args.lock_cube_yaw))
    if args.lock_cube_yaw and args.output_dir == _DEF_OUT:
        args.output_dir += "_no_yaw"

    if args.video:
        run_video_preview(args)
        return

    out = Path(args.output_dir)
    (out / "obses").mkdir(parents=True, exist_ok=True)
    # BEFORE collecting, not after: the tail call is on the success path only, so an interrupted run
    # left no manifest at all (this is why data/isaaclab_stroke_5k has none). Written here, a dataset
    # carries its full command + parsed args even if the collection is killed part way.
    provenance.write(out, __file__, args=args, repo=_REPO_ROOT)

    N = args.num_envs
    env = GridWrapperSingle(
        task_id=args.task_id, num_envs=N, device=args.device,
        render_mode=args.render_mode, spp=args.spp, stroke_max_steps=args.stroke_max_steps,
        fast_stroke_render=not args.no_fast_render,
        lock_cube_yaw=args.lock_cube_yaw,
    )
    # ---- RESUME: continue an interrupted collection instead of overwriting it ----------------
    # Episodes already on disk are counted FIRST, because the seed depends on that count.
    _resume_n = 0
    if args.resume:
        _sl = out / "seq_lengths.pth"
        if _sl.exists():
            _resume_n = int(torch.load(_sl).numel())
            _n_obs = len(list((out / "obses").glob("episode_*.pth")))
            # The .pth arrays are checkpointed every --save_every, but obses are written per
            # episode, so a kill between checkpoints leaves MORE obses than array rows. Trust the
            # arrays and drop the orphans, or the two would be misaligned by episode index.
            if _n_obs > _resume_n:
                for _f in sorted((out / "obses").glob("episode_*.pth"))[_resume_n:]:
                    _f.unlink()
                print(f"[collect] resume: dropped {_n_obs - _resume_n} orphan obs files past the "
                      f"last array checkpoint")
            print(f"[collect] RESUME from {_resume_n} episodes in {out}")
        else:
            print(f"[collect] --resume: nothing at {out}, starting fresh")

    # SEED MUST DIFFER FROM THE ORIGINAL RUN. env.seed()/RandomState() are deterministic, so
    # resuming with args.seed would replay the exact same cube spawns and strokes and silently
    # duplicate the episodes already collected. Offsetting by the resume count gives a fresh
    # stream while staying reproducible (same --seed + same resume point => same continuation).
    _seed = args.seed + _resume_n
    if _resume_n:
        print(f"[collect] seeding continuation with {args.seed}+{_resume_n}={_seed} "
              f"(NOT {args.seed}, which would re-draw the episodes already on disk)")
    env.seed(_seed)
    rng = np.random.RandomState(_seed)
    # One sampler per env (all identical mixed aimed/uniform; the StrokeExecutor itself
    # is already vectorized across envs). Each draws independent strokes.
    samplers = [StrokeSampler(rng, aimed_frac=args.aimed_frac, push_max=args.push_max,
                              start_margin=args.start_margin,
                              aim_push_range=(args.aim_push_min, args.aim_push_max),
                              aim_offset_sd=args.aim_offset_sd)
                for _ in range(N)]
    T = args.episode_len

    all_states, all_actions, all_proprio, all_cells, all_lens, all_sign = [], [], [], [], [], []

    if _resume_n:                       # preload so save_arrays() rewrites the FULL set, not just the tail
        all_states.extend(torch.load(out / "states.pth").numpy())
        all_actions.extend(torch.load(out / "actions.pth").numpy())
        all_proprio.extend(torch.load(out / "proprio.pth").numpy())
        all_cells.extend(torch.load(out / "cell_labels.pth").numpy())
        all_lens.extend(torch.load(out / "seq_lengths.pth").tolist())
        all_sign.extend(torch.load(out / "sign_colors.pth").tolist())
        print(f"[collect] resume: preloaded {len(all_states)} episodes into memory")

    def save_arrays():
        """Persist everything collected so far (valid dataset at any checkpoint)."""
        n = len(all_states)
        if n == 0:
            return
        torch.save(torch.from_numpy(np.stack(all_states)), out / "states.pth")
        torch.save(torch.from_numpy(np.stack(all_actions)), out / "actions.pth")
        torch.save(torch.from_numpy(np.stack(all_proprio)), out / "proprio.pth")
        torch.save(torch.from_numpy(np.stack(all_cells)), out / "cell_labels.pth")
        torch.save(torch.tensor(all_lens, dtype=torch.int64), out / "seq_lengths.pth")
        torch.save(torch.tensor(all_sign, dtype=torch.int64), out / "sign_colors.pth")
        meta = {
            "num_episodes": n, "episode_len": T, "seed": args.seed,
            # A --resume run is NOT reproducible from `seed` alone: episodes [0, resumed_from)
            # came from `seed`, the rest from `effective_seed` (= seed + resumed_from, offset so
            # the continuation does not re-draw the same episodes). Record both or the dataset
            # cannot be regenerated.
            "effective_seed": _seed, "resumed_from_episode": _resume_n,
            "state_dim": env.state_dim, "action_dim": ACTION_DIM, "proprio_dim": PROPRIO_DIM,
            "action_repr": "planar_stroke_start_disp_grid_meters",  # [x_start, y_start, dx, dy]
            "img_hw": IMG_HW, "sampling": "mixed", "aimed_frac": args.aimed_frac,
            "push_max": args.push_max, "start_margin": args.start_margin,
            "aim_push_range": [args.aim_push_min, args.aim_push_max],
            "aim_offset_sd": args.aim_offset_sd,
            # The single most consequential physics switch in the set: a yaw-locked dataset is not
            # interchangeable with an unlocked one (a WM trained on one is out of distribution on
            # the other), so it is recorded here rather than left to manifest.json alone.
            "lock_cube_yaw": bool(args.lock_cube_yaw),
            "num_envs": args.num_envs, "stroke_max_steps": args.stroke_max_steps, "task_id": args.task_id,
            "render_mode": args.render_mode, "spp": args.spp, "ep_pad": EP_PAD,
            "sign_palette": [name for name, _ in SIGN_PALETTE],  # sign_colors.pth indexes this
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    target = args.num_episodes
    # Resume continues toward the SAME total: only the remaining episodes are collected.
    n_batches = (max(0, target - _resume_n) + N - 1) // N  # ceil; last batch may be partial
    ep_count = _resume_n                # episode file numbering continues where it left off
    last_saved = _resume_n
    t0 = time.perf_counter()
    try:
        for b in range(n_batches):
            obs, state = env.reset()  # (N,...) fresh random cube per env
            # Per-env stop-sign color, then refresh obs so recorded frames show it.
            sign_idx = [int(rng.randint(0, len(SIGN_PALETTE))) for _ in range(N)]
            env.set_sign_color([SIGN_PALETTE[s][1] for s in sign_idx])
            obs, state = env._scene_outputs()

            ep_states = np.zeros((N, T, env.state_dim), dtype=np.float32)
            ep_actions = np.zeros((N, T, ACTION_DIM), dtype=np.float32)  # (N,T,4) strokes
            ep_proprio = np.zeros((N, T, PROPRIO_DIM), dtype=np.float32)
            ep_vis = np.zeros((N, T, IMG_HW, IMG_HW, 3), dtype=np.uint8)

            for t in range(T):
                # Pre-step convention: record the obs/state each stroke is taken FROM.
                ep_states[:, t] = state
                ep_proprio[:, t] = obs["proprio"]
                ep_vis[:, t] = obs["visual"]

                strokes = np.stack([samplers[i].sample(state[i, _CUBE_XY]) for i in range(N)])  # (N,4)
                ep_actions[:, t] = strokes

                obs, _, _, info = env.step(strokes)  # ONE full stroke per env (vectorized macro-step)
                state = info["state"]

            # Commit each env's episode (trim the last batch to `target`).
            for i in range(N):
                if ep_count >= target:
                    break
                all_states.append(ep_states[i])
                all_actions.append(ep_actions[i])
                all_proprio.append(ep_proprio[i])
                all_cells.append(cell_labels_from_states_single(ep_states[i]))  # (T,)
                all_lens.append(T)
                all_sign.append(sign_idx[i])
                torch.save(torch.from_numpy(ep_vis[i]), out / "obses" / f"episode_{ep_count:0{EP_PAD}d}.pth")
                ep_count += 1

            elapsed = time.perf_counter() - t0
            rate = ep_count / elapsed if elapsed > 0 else 0.0          # episodes / sec
            eta = (target - ep_count) / rate if rate > 0 else 0.0      # seconds to go
            finish = time.strftime("%a %H:%M", time.localtime(time.time() + eta))  # projected wall-clock done
            print(f"[collect] {ep_count}/{target} eps (batch {b+1}/{n_batches})  "
                  f"elapsed {elapsed/3600:.2f}h  ETA {eta/3600:.2f}h  ({rate:.2f} ep/s, done ~{finish})")
            if ep_count - last_saved >= args.save_every:
                save_arrays()
                last_saved = ep_count
                print(f"[collect] checkpoint saved ({ep_count} episodes)")

        save_arrays()
        print(f"[collect] DONE: wrote {len(all_states)} episodes (T={T}) to {out}")
        provenance.write(out, __file__, args=args, repo=_REPO_ROOT)
    except KeyboardInterrupt:
        print(f"\n[collect] interrupted — saving {len(all_states)} episodes collected so far")
        save_arrays()
    finally:
        close_or_exit(env)


if __name__ == "__main__":
    main()
