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
    )
    env.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    sampler = StrokeSampler(rng, aimed_frac=args.aimed_frac, push_max=args.push_max,
                            start_margin=args.start_margin)
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
    ap.add_argument("--num_episodes", type=int, default=1500,
                    help="MIXED sampling (--aimed_frac): ~60%% of strokes aim at the cube, so contact "
                    "density is high (unlike pure-uniform). 1500 eps x 20 strokes x 0.6 ~= 18k contact "
                    "frames, comparable to the old contact-only run, plus uniform misses for planner "
                    "action-space coverage.")
    ap.add_argument("--episode_len", type=int, default=20,
                    help="STROKES per trajectory (each is a full macro-step / many "
                    "sim steps). >= num_hist+num_pred (=4) or the slicer drops it; "
                    "20 gives 20-4+1=17 windows each.")
    ap.add_argument("--output_dir", default=os.environ.get("DATASET_DIR", "./data") + "/isaaclab_single_stroke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0",
                    help="Must be 'cuda:0' for this Isaac Sim build. Mask with CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--render_mode", default="PathTracing",
                    choices=("PathTracing", "RaytracedLighting"))
    ap.add_argument("--spp", type=int, default=128,
                    help="Samples-per-pixel for PathTracing (128 fine for training; "
                    "256+ only for demo videos).")
    ap.add_argument("--aimed_frac", type=float, default=0.6,
                    help="Fraction of strokes that AIM at the cube (start behind it, push through "
                    "-> teaches dynamics) vs UNIFORM strokes (cover the planner's action space, incl. "
                    "misses, matching the planner's miss-heavy query distribution). 0.6 = 60%% aimed / "
                    "40%% uniform. 0.0 = pure-uniform (deformable parity).")
    ap.add_argument("--push_max", type=float, default=0.08,
                    help="UNIFORM strokes: per-axis push displacement bound (m): dx,dy ~ U(-push_max, "
                    "push_max). ~20%% of the grid span, matching the deformable short-push ratio. "
                    "Short pushes = small per-step cube motion = easier one-step dynamics.")
    ap.add_argument("--start_margin", type=float, default=0.06,
                    help="Contact point sampled uniformly in +/-(GRID_HALF + start_margin) per "
                    "axis, so the pusher can get behind a cube sitting at/just past a grid edge.")
    ap.add_argument("--stroke_max_steps", type=int, default=320,
                    help="Cap on internal IK sim steps per stroke (4 phases x ~80: "
                    "approach, descend, push, retract-to-home).")
    ap.add_argument("--no_fast_render", action="store_true",
                    help="Disable the mid-stroke render skip (path-trace every internal "
                    "step). For A/B timing vs the default fast path; much slower at PathTracing.")
    ap.add_argument("--num_envs", type=int, default=10,
                    help="Parallel envs collected per batch (GPU-batched physics + the "
                    "vectorized StrokeExecutor). Each env runs an independent episode; "
                    "~Nx throughput on top of the fast render skip.")
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

    if args.video:
        run_video_preview(args)
        return

    out = Path(args.output_dir)
    (out / "obses").mkdir(parents=True, exist_ok=True)

    N = args.num_envs
    env = GridWrapperSingle(
        task_id=args.task_id, num_envs=N, device=args.device,
        render_mode=args.render_mode, spp=args.spp, stroke_max_steps=args.stroke_max_steps,
        fast_stroke_render=not args.no_fast_render,
    )
    env.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    # One sampler per env (all identical mixed aimed/uniform; the StrokeExecutor itself
    # is already vectorized across envs). Each draws independent strokes.
    samplers = [StrokeSampler(rng, aimed_frac=args.aimed_frac, push_max=args.push_max,
                              start_margin=args.start_margin)
                for _ in range(N)]
    T = args.episode_len

    all_states, all_actions, all_proprio, all_cells, all_lens, all_sign = [], [], [], [], [], []

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
            "state_dim": env.state_dim, "action_dim": ACTION_DIM, "proprio_dim": PROPRIO_DIM,
            "action_repr": "planar_stroke_start_disp_grid_meters",  # [x_start, y_start, dx, dy]
            "img_hw": IMG_HW, "sampling": "mixed", "aimed_frac": args.aimed_frac,
            "push_max": args.push_max, "start_margin": args.start_margin,
            "num_envs": args.num_envs, "stroke_max_steps": args.stroke_max_steps, "task_id": args.task_id,
            "render_mode": args.render_mode, "spp": args.spp, "ep_pad": EP_PAD,
            "sign_palette": [name for name, _ in SIGN_PALETTE],  # sign_colors.pth indexes this
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    target = args.num_episodes
    n_batches = (target + N - 1) // N  # ceil; last batch may be partially committed
    ep_count = 0
    last_saved = 0
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
            eta = elapsed / ep_count * (target - ep_count)
            print(f"[collect] {ep_count}/{target} episodes (batch {b+1}/{n_batches})  "
                  f"{elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m")
            if ep_count - last_saved >= args.save_every:
                save_arrays()
                last_saved = ep_count
                print(f"[collect] checkpoint saved ({ep_count} episodes)")

        save_arrays()
        print(f"[collect] DONE: wrote {len(all_states)} episodes (T={T}) to {out}")
    except KeyboardInterrupt:
        print(f"\n[collect] interrupted — saving {len(all_states)} episodes collected so far")
        save_arrays()
    finally:
        close_or_exit(env)


if __name__ == "__main__":
    main()
