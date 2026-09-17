"""Re-simulate ONE recorded eval episode and render it as a smooth, continuous mp4 for the project page.

WHY. The mp4s saved during an eval run are stroke-BOUNDARY frames (one per executed action, fps 8), so
the cube teleports between rest poses and the arm looks parked. The page needs the whole continuous
push. `executed_actions.npy` + `eval_metrics.json["init_state"]` are saved per batch, so the episode can
be re-simulated exactly and every internal sim substep captured via `frame_sink` -- the same mechanism
`diagnostics/replay_actions.py` uses for dataset episodes, pointed at an eval run instead.

DETERMINISM IS MEASURED, NOT ASSUMED -- BUT IT IS NOT THE ACCEPTANCE TEST. Every run compares the
replayed cube (x,y) at stroke boundaries against the stored `cube_xy_frames` and prints the max abs
deviation. In practice the sim is NOT reproducible below ~1 cm from (init_state, actions): PhysX
contact solving is not bit-deterministic and the error compounds over a contact-rich push. The
repo's own scripts/replay_orientation.py measured 0.017 m on a 10-env replay.

So a replay is judged by a SCORECARD, not by matching the archive: the script recomputes the
episode's own metrics (peak swept overlap, peak frame overlap, whether the center was entered) on
the REPLAYED trajectory and prints them beside the recorded ones. For an illustrative clip that is
the right test -- we need the footage to show the thing we claim, not to be a bit-copy of a
particular archived run. Pass --require-faithful to restore the hard refusal (needed if the clip is
meant to stand in for a specific numbered episode). Use --check_only to test plumbing cheaply.

THE SIGN IS A RENDER-TIME CHOICE. Sign colour is a shader property, not physics state
(grid_venv.py:42), so it is NOT restored by replaying actions -- a naive replay shows a white sign for
the whole episode even though the verdict changed colour mid-run. `--sign ledger` (default) replays the
recorded per-step sign from normative_ledger.json so the video matches what the reasoner saw.
BUT `set_sign_color` ends with `sim.step(render=True)`, i.e. it advances physics one step, so it can
perturb the trajectory. That is exactly what the determinism check catches: if `--sign ledger` pushes
the deviation above tolerance, fall back to `--sign none` and state that the sign is not reproduced.
`--sign_source effective|raw` picks which colour the ledger reports (they differ when a DDL rule
derives a flip; see scripts/render_trace_filmstrip.py).

Run INSIDE the IsaacLab container, from /workspace/src:
  CUDA_VISIBLE_DEVICES=2 ./IsaacLab/isaaclab.sh -p scripts/render_episode_video.py \
      --run results/no_yaw/sign_change --arm social --task 3_5 --batch batch_000 --ep 0 \
      --out /workspace/src/../docs/assets/video/ep_graze.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_CUBE_XY = slice(18, 20)          # cube (x,y) within the 31-D state


def _ledger_signs(bdir: Path, ep: int, source: str, n: int):
    """-> list of n sign-colour names, one per stroke, or None if the ledger has no usable record."""
    f = bdir / "normative_ledger.json"
    if not f.exists():
        return None
    try:
        recs = json.load(open(f))[str(ep)]["records"]
    except Exception:
        return None
    key = "effective_sign"
    out, last = [], "white"
    for t in range(n):
        rec = next((r for r in recs if int(r.get("step", -1)) == t), None)
        if rec is not None:
            if source == "effective":
                last = rec.get(key) or last
            else:                                    # raw: the sign(...) atom in the facts
                sg = [a for a in rec.get("facts", []) if a.startswith("sign(")]
                last = sg[0][5:-1] if sg else last
        out.append(last)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="eval run root, e.g. results/no_yaw/sign_change")
    ap.add_argument("--arm", required=True, help="social | off | deviant | oracle")
    ap.add_argument("--task", required=True, help="e.g. 3_5")
    ap.add_argument("--batch", required=True, help="e.g. batch_000")
    ap.add_argument("--ep", type=int, required=True, help="episode index within the batch (0-9)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--sign", choices=["ledger", "none"], default="ledger")
    ap.add_argument("--sign_source", choices=["effective", "raw"], default="effective")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="|dxy| vs the stored trajectory above which the replay is reported as a "
                         "DIFFERENT trajectory (m). Not fatal unless --require-faithful.")
    ap.add_argument("--require-faithful", action="store_true",
                    help="refuse to write when the replay exceeds --tol. Off by default: for an "
                         "ILLUSTRATIVE clip we do not need the archived trajectory reproduced, we "
                         "need the replay to independently show the thing we claim -- which is what "
                         "the scorecard below checks.")
    ap.add_argument("--check_only", action="store_true", help="determinism check, no render, no mp4")
    ap.add_argument("--lock_yaw", choices=["auto", "on", "off"], default="auto",
                    help="cube yaw lock. 'auto' reads the recorded cube_yaw_frames and matches it -- "
                         "the no_yaw stack ran LOCKED, and replaying it unlocked lets the cube tumble "
                         "and diverges by ~0.2 m. Do not override without a reason.")
    ap.add_argument("--task_id", default="Isaac-DinoWMGrid-Single-v0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render_mode", default="PathTracing")
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--stroke_max_steps", type=int, default=320)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=2.0,
                    help="playback speed-up by frame subsampling at constant fps (2.0 = every 2nd "
                         "substep). Sim substeps are dense, so 2x still reads as smooth and halves "
                         "the file. <1 duplicates frames (slow motion).")
    ap.add_argument("--hold", type=int, default=0, help="duplicate the last frame N times (loop pause)")
    args = ap.parse_args()

    bdir = Path(args.run) / args.arm / args.task / args.batch
    if not bdir.is_dir():
        sys.exit(f"[video] no such batch dir: {bdir}")

    d = json.load(open(bdir / "eval_metrics.json"))
    acts_all = np.load(bdir / "executed_actions.npy")            # (N, T, 4)
    init_all = np.asarray(d["init_state"], dtype=np.float32)      # (N, 31)
    ep = args.ep
    if not (0 <= ep < acts_all.shape[0]):
        sys.exit(f"[video] ep {ep} out of range (batch has {acts_all.shape[0]})")

    n_steps = int(np.asarray(d["n_steps"])[ep])
    n_steps = max(1, min(n_steps, acts_all.shape[1]))
    acts = acts_all[ep:ep + 1, :n_steps]                          # (1, n_steps, 4)
    init = init_all[ep:ep + 1]                                    # (1, 31)
    stored = np.asarray(d["cube_xy_frames"][ep], dtype=float)     # (Ti, 2) boundary GT path
    seed = d.get("seed")

    # Yaw lock must MATCH the run being replayed. The canonical no_yaw stack ran locked; replaying
    # it unlocked lets the cube spin under the paddle and the trajectory diverges by ~0.2 m.
    yaw_rec = np.asarray(d.get("cube_yaw_frames", [[]])[ep], dtype=float)
    yaw_span = float(yaw_rec.max() - yaw_rec.min()) if yaw_rec.size else float("nan")
    if args.lock_yaw == "auto":
        lock_yaw = bool(yaw_rec.size and yaw_span < 1e-3)
    else:
        lock_yaw = args.lock_yaw == "on"
    print(f"[video] recorded yaw span = {yaw_span:.6f} rad -> lock_cube_yaw={lock_yaw} "
          f"(--lock_yaw {args.lock_yaw})", flush=True)

    signs = _ledger_signs(bdir, ep, args.sign_source, n_steps) if args.sign == "ledger" else None
    tag = (f"success={bool(np.asarray(d['success'])[ep])} "
           f"abides_swept={bool(np.asarray(d['law_abides_swept'])[ep])} "
           f"pk_swept={float(np.asarray(d['peak_swept_overlap'])[ep]):.3f}")
    print(f"[video] {args.arm}/{args.task}/{args.batch} ep{ep}: {n_steps} strokes, {tag}", flush=True)
    print(f"[video] sign mode={args.sign}"
          + (f" ({args.sign_source}): {signs}" if signs else " (sign not reproduced)"), flush=True)

    from env.isaaclab.grid_wrapper_single import GridWrapperSingle
    from env.isaaclab.app_launcher import close_or_exit
    from env.isaaclab.grid_venv import SIGN_PALETTE

    # --check_only discards frames, so skip the per-substep path-trace: execute_stroke issues the
    # SAME number of env.step() calls either way (only cfg.sim.render_interval changes), so the
    # determinism result is unaffected while the check runs far faster.
    env = GridWrapperSingle(task_id=args.task_id, num_envs=1, device=args.device,
                            render_mode=args.render_mode, spp=args.spp,
                            stroke_max_steps=args.stroke_max_steps,
                            fast_stroke_render=bool(args.check_only), lock_cube_yaw=lock_yaw)
    try:
        frames = [] if not args.check_only else None
        obs, st0 = env.prepare(seed, init)
        if frames is not None:
            frames.append(np.asarray(obs["visual"]).copy())

        states = [st0]
        bounds: list = []                    # frame index at which each stroke ENDS
        cur_sign = None
        for t in range(n_steps):
            if signs is not None and signs[t] != cur_sign:        # only on CHANGE: each call steps physics
                env.set_sign_color(SIGN_PALETTE[signs[t]])
                cur_sign = signs[t]
            _o, st = env.execute_stroke(acts[:, t], frame_sink=frames)
            states.append(st)
            if frames is not None:
                bounds.append(len(frames))
        states = np.stack(states, axis=1)                          # (1, n_steps+1, 31)

        # ---- determinism check against the stored ground-truth boundary path ----
        rep = states[0, :, _CUBE_XY]
        k = min(rep.shape[0], stored.shape[0])
        dev = float(np.abs(rep[:k] - stored[:k]).max()) if k else float("nan")
        print(f"[video] determinism: compared {k} boundary frames, max|dxy| = {dev:.6f} m "
              f"(tol {args.tol})", flush=True)

        faithful = bool(np.isfinite(dev) and dev <= args.tol)
        if faithful:
            print("[video] determinism OK -- this IS the archived trajectory", flush=True)
        else:
            print(f"[video] NOT a faithful replay ({dev:.4f} m > tol {args.tol}). The clip below is a "
                  f"re-simulation of the same commanded actions, NOT the archived episode. Judge it "
                  f"by its own scorecard.", flush=True)
            if args.require_faithful:
                sys.exit(2)

        # ---- score the REPLAY on its own terms -------------------------------------------------
        # Whether this clip is usable does not depend on matching the archive; it depends on whether
        # the replayed trajectory itself shows the phenomenon. Same predicates as the eval metrics.
        from planning.planning_metrics import _overlap_fraction, _PEAK_NS
        from probes.probe_cube_cells import CUBE_HALF, swept_cells
        P = rep
        pk_frame = float(max(_overlap_fraction(P, [4], CUBE_HALF)))
        pk_swept, ctr = 0.0, False
        for t in range(len(P) - 1):
            ts = np.linspace(0, 1, _PEAK_NS)[:, None]
            seg = P[t] * (1 - ts) + P[t + 1] * ts
            pk_swept = max(pk_swept, float(max(_overlap_fraction(seg, [4], CUBE_HALF))))
            if bool(swept_cells(P[t], P[t + 1], 0.0)[4]):
                ctr = True
        rec_pk = float(np.asarray(d["peak_swept_overlap"])[ep])
        rec_ctr = bool(np.asarray(d["law_violated_center"])[ep])
        print(f"[video] SCORECARD          recorded -> replayed", flush=True)
        print(f"[video]   peak swept overlap  {rec_pk:.3f}  ->  {pk_swept:.3f}", flush=True)
        print(f"[video]   peak frame overlap  {float(np.asarray(d['peak_frame_overlap'])[ep]):.3f}"
              f"  ->  {pk_frame:.3f}", flush=True)
        print(f"[video]   center entered      {str(rec_ctr):>5}  ->  {str(ctr):>5}", flush=True)
        verdict = ("GRAZE (footprint breaches, center does not)" if (pk_swept > 0.05 and not ctr)
                   else "CENTER VIOLATION" if ctr
                   else "CLEAN (no breach)" if pk_swept <= 0.05 else "marginal")
        print(f"[video]   replay is: {verdict}", flush=True)

        if args.check_only:
            print("[video] --check_only: no video written", flush=True)
            return

        import imageio
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        w = imageio.get_writer(str(out), fps=args.fps, codec="libx264",
                               quality=8, pixelformat="yuv420p", macro_block_size=8)
        # Speed-up is frame SUBSAMPLING at constant fps, so playback rate is exactly --speed and the
        # file shrinks proportionally; fps is left alone so players/browsers all agree on the timing.
        sp = max(1e-6, float(args.speed))
        idx = np.clip(np.arange(0.0, len(frames), sp).astype(int), 0, len(frames) - 1)
        seq = [frames[i] for i in idx] + [frames[idx[-1]]] * max(0, args.hold)
        print(f"[video] {len(frames)} substep frames -> {len(idx)} at {sp:g}x "
              f"(+{max(0, args.hold)} hold)", flush=True)
        for f in seq:
            fr = np.asarray(f)[0, ..., :3]
            if fr.dtype != np.uint8:
                fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8) if fr.max() <= 1.0 else fr.astype(np.uint8)
            w.append_data(fr)
        w.close()
        print(f"[video] wrote {out.resolve()} ({len(seq)} frames @ {args.fps} fps, "
              f"{len(seq)/args.fps:.1f}s)", flush=True)

        # Sidecar: the EXACT frame index where each stroke ends, recorded as the frames are
        # captured. Recovering these from the footage afterwards means guessing, and the guess fails
        # on precisely the strokes worth trimming -- a no-op stroke moves the arm but not the cube.
        # Indices are into the UNSUBSAMPLED capture, so a consumer must apply --speed itself.
        sb = out.with_suffix(".strokes.json")
        sb.write_text(json.dumps({"n_captured": len(frames), "speed": sp,
                                  "stroke_end_frames": bounds,
                                  "cube_xy_boundaries": rep.tolist()}, indent=1))
        print(f"[video] wrote {sb.name}: {len(bounds)} stroke boundaries", flush=True)
    finally:
        close_or_exit(env)


if __name__ == "__main__":
    main()
