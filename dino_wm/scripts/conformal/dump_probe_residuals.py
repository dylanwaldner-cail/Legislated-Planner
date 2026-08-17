"""ONE pass over the offline stroke dataset -> the calibration artifact every conformal script reads.

WHAT THIS PRODUCES
------------------
    data/conformal/residuals_<tag>.npz

with one row per (episode, stroke transition) and, for each, the four planar points the
conformal geometry needs:

    probe_start   probe on the ENCODED latent of frame t      (what the planner believed it saw)
    probe_end     probe on the WM-PREDICTED latent of frame t+1 (what the planner believed would happen)
    gt_start      states.pth cube xy at frame t                (truth)
    gt_end        states.pth cube xy at frame t+1              (truth)

plus the bookkeeping needed to slice and group afterwards: episode id, stroke index, the raw
action, the start cell, and an `in_wm_train` / `in_probe_train` membership flag.

WHY THE OFFLINE DATASET AND NOT EVAL ROLLOUTS
---------------------------------------------
The eval trees do not log the probe's START reading (only `results/aug15/sign_color` does),
and the saved rollout videos are lossy h264, so reconstructing the start from them would put a
measurable reconstruction error inside a number we intend to report as a guarantee. The
offline dataset sidesteps both problems: `obses/` holds the ORIGINAL uint8 frames the encoder
was trained and evaluated on, so the probe reads exactly the pixels it would have read. No
sim, no planner, no IsaacLab -- just encoder + WM + probe forward passes.

THE TWO WAYS THIS CAN SILENTLY LIE (read before trusting the output)
---------------------------------------------------------------------
1. ACTION NORMALISATION. The WM consumes NORMALISED actions -- see planning/rrt.py:135, which
   does `(strokes - amean) / astd` before `wm.rollout`. `actions.pth` on disk is in RAW METRES.
   We therefore load normalisation stats from the same Preprocessor the planner uses. Feeding
   raw metres would produce plausible-looking garbage, so this is the first thing to check if
   residuals look wrong.
2. HISTORY WINDOW. The WM is trained with `num_hist=3` (conf/train.yaml:53), so a faithful
   one-step prediction needs the same 3-frame context the planner supplies, not a single
   frame. For t < num_hist-1 we left-pad by repeating the first frame; those rows are flagged
   `padded_history=True` so you can drop them if you want strict parity.

A third, non-silent caveat: this dataset's action distribution (uniform/aimed sampling with
deliberate misses) is NOT the deployment distribution (RRT's goal-biased, cone-steered,
pruner-filtered proposals). Marginal conformal calibrated here will not transfer as-is --
use mondrian_conformal.py, whose group-conditional quantiles survive the reweighting, and
validate empirical coverage against a real run that logs probe start.

USAGE
-----
    # smoke test on 20 episodes first -- always do this before the full pass
    python scripts/conformal/dump_probe_residuals.py --limit 20 --tag smoke

    # full pass (~5000 episodes; GPU minutes, CPU hours)
    python scripts/conformal/dump_probe_residuals.py --tag 5k_shift025
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from env.isaaclab import grid_metadata as gm          # noqa: E402
from scripts.conformal.common import CONFORMAL_DIR    # noqa: E402

#: cube (x, y) inside the 31-D single-arm state -- mirrors planning/planning_metrics.py:26
_CUBE_XY = slice(18, 20)


def _git_sha() -> str:
    """Record the code version in the artifact so a number can always be traced to a commit."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _build_model_and_probe(args, device):
    """Load the WM + position probe through the SAME code paths deployment uses.

    We deliberately reuse plan.py's `load_model` and probes.registry rather than
    reimplementing the forward pass: any reimplementation would be free to drift from the
    pruner's actual perception, and a drifted calibration is worse than none.
    """
    import hydra
    from omegaconf import OmegaConf

    from plan import load_model                      # noqa: WPS433 (deferred: heavy import)
    from preprocessor import Preprocessor
    from probes.registry import ProbeRegistry

    ckpt_dir = Path(args.ckpt_base_path) / "outputs" / args.model_name
    train_cfg = OmegaConf.load(ckpt_dir / "hydra.yaml")
    train_cfg.has_decoder = False                    # decoder is viz-only; never needed here
    model_ckpt = ckpt_dir / "checkpoints" / f"model_{args.model_epoch}.pth"

    model = load_model(model_ckpt, train_cfg, num_action_repeat=1, device=device)
    model.eval()

    reg = ProbeRegistry(device=device)
    reg.set_probe("cube_position", args.probe_path)  # tie to the planner's probe, one knob
    probe = reg["cube_position"]
    return model, probe, train_cfg


def _write(out_dir, args, rows, num_hist, n_done, partial: bool):
    """Write (or rewrite) the residual npz. Called at every checkpoint and at the end.

    `n_episodes_done` in the metadata records how far the pass actually got, so a partial dump
    is self-describing rather than silently looking like a complete one.
    """
    meta = {
        "dataset": str(args.dataset), "norm_from": str(args.norm_from),
        "n_episodes_done": int(n_done), "n_episodes_requested": (args.limit or "all"),
        "partial": bool(partial),
        "model_name": args.model_name, "model_epoch": str(args.model_epoch),
        "probe_path": str(args.probe_path), "num_hist": num_hist,
        "device": str(args.device), "git_sha": _git_sha(),
        "cube_half_note": ("scores are computed downstream in common.py using "
                           "probes.probe_cube_cells.CUBE_HALF"),
        "split_note": ("in_wm_train is exact (torch randperm seed 42); in_probe_train_guess is a "
                       "GUESS -- the probe .pth records no split. Both are RECORDED not applied, "
                       "and both are recomputable offline from `episode`."),
    }
    out = Path(out_dir) / f"residuals_{args.tag}.npz"
    # NB the temp name must END in .npz -- np.savez_compressed silently appends '.npz' to any
    # path that doesn't, which would leave the real write at a different path than we rename.
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta_json=json.dumps(meta),
                        **{k: np.asarray(v) for k, v in rows.items()})
    tmp.replace(out)                      # atomic: a killed write never corrupts a good dump
    tag = "checkpoint" if partial else "FINAL"
    print(f"[dump] {tag}: wrote {out}  ({len(rows['episode'])} strokes from {n_done} episodes)",
          flush=True)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/isaaclab_stroke_5k_shift025",
                    help="dataset root holding obses/, actions.pth, states.pth")
    ap.add_argument("--norm_from", default="data/isaaclab_stroke_5k_shift025",
                    help="dataset whose action/proprio stats normalise the inputs. MUST stay the "
                         "TRAINING set even when --dataset is a fresh calibration set: the WM was "
                         "trained against those statistics, so normalising by a different set's "
                         "mean/std silently feeds it out-of-distribution inputs.")
    ap.add_argument("--tag", default="5k_shift025", help="suffix of the output npz")
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (smoke test)")
    ap.add_argument("--batch", type=int, default=16, help="episodes encoded per forward pass")
    ap.add_argument("--device", default="cuda:0", help="use 'cpu' to stay off a busy GPU")
    ap.add_argument("--ckpt_base_path", default=".")
    ap.add_argument("--model_name", default="wm_5k")
    ap.add_argument("--model_epoch", default="30")
    ap.add_argument("--probe_path", default="probes/weights/cube_pos_encoded.pth",
                    help="MUST match the run being calibrated (objective.pos_probe_path)")
    # Split membership is recorded, never enforced -- see the contamination note below.
    ap.add_argument("--wm_train_fraction", type=float, default=0.9,
                    help="mirrors isaaclab_grid_dset.py:249 split_traj_datasets(...)")
    ap.add_argument("--wm_split_seed", type=int, default=42)
    ap.add_argument("--probe_val_frac", type=float, default=0.2,
                    help="mirrors probe_cube_position.py --val_frac")
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(CONFORMAL_DIR),
                    help="where the npz lands (default data/conformal/)")
    ap.add_argument("--context_frames", type=int, default=1,
                    help="obs history frames fed to the WM. Default 1 to MATCH DEPLOYMENT "
                         "(plan.py:407 gives the planner a singleton time axis), not the "
                         "num_hist=3 the WM was trained with. Set 0 to fall back to num_hist.")
    ap.add_argument("--save_every", type=int, default=100,
                    help="checkpoint the npz every N episodes. The encode pass costs hours, so "
                         "the artifact is rewritten periodically and on Ctrl-C: killing the run "
                         "leaves a SHORTER but fully valid dump, never an empty one.")
    args = ap.parse_args()

    # FAIL FAST on an unwritable output directory. This pass takes hours; discovering at the
    # final save() that data/ is root-owned throws all of it away. Check before any compute.
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe_file = out_dir / ".write_test"
        probe_file.touch()
        probe_file.unlink()
    except OSError as exc:
        raise SystemExit(
            f"[dump] cannot write to {out_dir}: {exc}\n"
            f"       data/ is root-owned. Either create it once with\n"
            f"         sudo mkdir -p {out_dir} && sudo chown $(id -u):$(id -g) {out_dir}\n"
            f"       or point somewhere writable with --out_dir."
        )

    device = torch.device(args.device)
    root = _REPO / args.dataset

    actions_raw = torch.load(root / "actions.pth").float()        # (E, T, 4) RAW METRES
    states = torch.load(root / "states.pth").float()              # (E, T, 31)
    seq_lengths = torch.load(root / "seq_lengths.pth")            # (E,)
    proprio = torch.load(root / "proprio.pth").float()            # (E, T, 18)
    n_eps = actions_raw.shape[0] if args.limit is None else min(args.limit, actions_raw.shape[0])

    model, probe, train_cfg = _build_model_and_probe(args, device)

    # CONTEXT DEPTH: match DEPLOYMENT, not training.
    #
    # The WM was trained with num_hist=3, but the planner feeds it a SINGLE frame:
    # plan.py:407 `_add_time_dim` expands (b, ...) -> (b, 1, ...), and the MPC loop re-observes
    # each step rather than carrying a frame buffer. We are calibrating the deployed system, so
    # the dump must use the deployed context.
    #
    # Measured 2026-08-17 (240 strokes): the two are a wash -- 1-frame gives mean 3.13 / p90
    # 5.54 cm, 3-frame gives 3.14 / 5.66. The task is effectively Markovian (frames are rest
    # states, no momentum to infer), so history buys nothing. Kept configurable anyway, because
    # that equivalence is a property of THIS task and would not survive a dynamics change.
    num_hist = (int(args.context_frames) if args.context_frames
                else int(getattr(train_cfg, "num_hist", 3)))

    # ---------------------------------------------------------------------------------
    # Input normalisation, computed EXACTLY as the dataset does it.
    #
    # The WM was trained on normalised actions AND normalised proprio
    # (isaaclab_grid_dset.py:110-111, with normalize_action=true in the wm_5k config).
    # Feeding either one raw produces confident nonsense, so we reuse the dataset's own
    # `_mean_std` helper rather than reimplementing it.
    #
    # NOTE the stats are over the FULL dataset, never the --limit subset: otherwise a smoke
    # test would normalise differently from the full run and the two would not be comparable.
    # ---------------------------------------------------------------------------------
    from datasets.isaaclab_grid_dset import IsaacLabSingleDataset as _DS  # noqa: WPS433

    # Stats come from --norm_from (the TRAINING set), NOT from --dataset. When calibrating on a
    # freshly collected set, its own mean/std differ slightly from the training set's; using
    # them would shift every action the WM sees and corrupt the predictions in a way that looks
    # like model error rather than a preprocessing bug.
    norm_root = _REPO / args.norm_from
    if norm_root.resolve() == root.resolve():
        n_act, n_pro, n_len = actions_raw, proprio, seq_lengths
    else:
        n_act = torch.load(norm_root / "actions.pth").float()
        n_pro = torch.load(norm_root / "proprio.pth").float()
        n_len = torch.load(norm_root / "seq_lengths.pth")
    act_mean, act_std = _DS._mean_std(n_act, n_len)
    pro_mean, pro_std = _DS._mean_std(n_pro, n_len)
    print(f"[dump] normalising with stats from {args.norm_from} "
          f"(action_std={act_std.numpy().round(4)}, num_hist={num_hist})")

    # ---------------------------------------------------------------------------------
    # Split MEMBERSHIP, recorded as flags rather than applied as a filter.
    #
    # The WM (seed 42, by trajectory, 90/10) and the probe (seed 0, by episode, 80/20) were
    # trained on DIFFERENT subsets. Rather than guess which one the calibration should honour,
    # we record both and let the analysis scripts slice. That also makes the contamination
    # check possible: compare residuals on trained-on vs held-out episodes. If they match, the
    # heads did not memorise and the full 5k is usable (far more data per Mondrian group); if
    # they diverge, subset to the intersection. Decide from the data, not from assumption.
    # ---------------------------------------------------------------------------------
    n_total = int(actions_raw.shape[0])

    # A FRESH dataset (one the models never saw) is held out by construction, so the split
    # flags below do not apply -- and worse, computing them over the new episode indices would
    # stamp confident-looking labels that mean nothing. Detect that case and mark everything
    # held out.
    is_training_set = (_REPO / args.norm_from).resolve() == root.resolve()
    if not is_training_set:
        print(f"[dump] {args.dataset} != training set -> every episode marked HELD OUT "
              f"(no contamination possible)")

    # WM split -- reproduced EXACTLY as datasets/traj_dset.py does it. This must use a torch
    # Generator (line 113/133 there), NOT numpy: torch.randperm(seed=42) and
    # np.random.RandomState(42).permutation give completely different orderings, so a numpy
    # stand-in would mislabel every episode while looking perfectly plausible. Note also the
    # int() truncation (not round()) and that TRAIN takes the leading slice.
    _g = torch.Generator().manual_seed(args.wm_split_seed)
    wm_perm = torch.randperm(n_total, generator=_g).tolist()
    n_wm_train = int(args.wm_train_fraction * n_total)
    wm_train_eps = set(wm_perm[:n_wm_train])

    # Probe split -- UNKNOWN, and this flag is only a guess.
    # probes/probe_cube_position.py splits by episode with np.random.RandomState(--seed) and
    # --val_frac, but the trained cube_pos_encoded.pth records NO split metadata (no val_eps,
    # no seed, no val_frac -- only kind/source/pool_grid/encoder), and no training command was
    # logged. So we reproduce the DEFAULTS and label the result a guess. Do not treat
    # `in_probe_train_guess` as ground truth; see the bimodality diagnostic in
    # error_decomposition.py for an empirical way to detect probe memorisation instead.
    rng_pr = np.random.RandomState(args.probe_split_seed)
    pr_perm = rng_pr.permutation(np.arange(n_total))
    n_pr_val = max(1, int(round(n_total * args.probe_val_frac)))
    probe_val_eps = set(pr_perm[:n_pr_val].tolist())

    rows: dict[str, list] = {k: [] for k in (
        "episode", "stroke", "probe_start", "probe_end", "gt_start", "gt_end",
        "action_raw", "start_cell", "padded_history", "in_wm_train", "in_probe_train_guess")}
    last_saved = 0

    for e0 in range(0, n_eps, args.batch):
        eps = list(range(e0, min(e0 + args.batch, n_eps)))
        # obses are stored one .pth per episode: (T, 224, 224, 3) uint8
        vids = [torch.load(root / "obses" / f"episode_{e:05d}.pth") for e in eps]

        for bi, e in enumerate(eps):
            T = int(seq_lengths[e])
            vid = vids[bi][:T].float() / 255.0                     # (T,H,W,3) in [0,1]
            vid = vid.permute(0, 3, 1, 2) * 2.0 - 1.0              # (T,3,H,W) Normalize(.5,.5)
            vid = vid.to(device)                                   # matches probe_cube_position.py:143

            act = ((actions_raw[e, :T] - act_mean) / act_std).to(device)     # raw m -> normalised
            prop = ((proprio[e, :T] - pro_mean) / pro_std).to(device)        # proprio too

            # Encode every frame of the episode ONCE. Consecutive strokes share num_hist-1 of
            # their context frames, so re-encoding per stroke would triple the DINO cost for
            # identical results. `encode_obs` honours a precomputed 'visual_cached' latent
            # (visual_world_model.py:125), which is exactly this optimisation.
            vis_cached = model.encode_obs({"visual": vid.unsqueeze(0),
                                           "proprio": prop.unsqueeze(0)})["visual"]  # (1,T,P,D)

            for t in range(T - 1):
                # --- history window: [t-num_hist+1 .. t], left-padded by repeating frame 0 ---
                lo = t - num_hist + 1
                padded = lo < 0
                idx = [max(0, i) for i in range(lo, t + 1)]
                # Both keys on purpose: rollout() reads obs_0['visual'].shape[1] directly
                # (visual_world_model.py:298) just to count the initial frames, while
                # encode_obs() prefers 'visual_cached' and skips the DINO pass (line 125).
                # Handing the cached latent to both gives the right count AND the skip.
                obs_0 = {"visual": vis_cached[:, idx],             # shape[1] only -- never encoded
                         "visual_cached": vis_cached[:, idx],      # (1, num_hist, P, D)
                         "proprio": prop[idx].unsqueeze(0)}        # (1, num_hist, 18)

                # act must be EXACTLY num_hist long -- one action per history frame, where
                # frame i carries the action taken FROM state i. So the last entry is a_t, the
                # stroke whose outcome we want.
                #
                # Do NOT append an extra action here. rollout() runs a final predict() AFTER
                # consuming every action (visual_world_model.py:311), so an act of length
                # num_hist already yields num_hist+1 frames whose LAST frame is the one-step
                # landing. Passing num_hist+1 actions (duplicating a_t) returns num_hist+2
                # frames and makes [:, -1] a TWO-step prediction -- measured at ~5.9 cm mean
                # error versus 3.5 cm for the correct one-step frame.
                act_win = act[idx].unsqueeze(0)                    # (1, num_hist, 4)

                z_obs, _ = model.rollout(obs_0=obs_0, act=act_win)
                z_vis = z_obs["visual"]                            # (1, num_hist+1, P, D)

                # index num_hist-1 = the CURRENT encoded observation (last history frame);
                # index -1 (== num_hist) = the one-step WM prediction.
                p_start = probe(z_vis[:, num_hist - 1]).squeeze(0).cpu().numpy()
                p_end = probe(z_vis[:, -1]).squeeze(0).cpu().numpy()

                g_start = states[e, t, _CUBE_XY].numpy()
                g_end = states[e, t + 1, _CUBE_XY].numpy()

                rows["episode"].append(e)
                rows["stroke"].append(t)
                rows["probe_start"].append(p_start)
                rows["probe_end"].append(p_end)
                rows["gt_start"].append(g_start)
                rows["gt_end"].append(g_end)
                rows["action_raw"].append(actions_raw[e, t].numpy())
                rows["start_cell"].append(int(gm.which_cell(g_start)))
                rows["padded_history"].append(bool(padded))
                rows["in_wm_train"].append(is_training_set and (e in wm_train_eps))
                rows["in_probe_train_guess"].append(is_training_set and (e not in probe_val_eps))

        print(f"[dump] episodes {eps[0]}..{eps[-1]} done ({len(rows['episode'])} strokes)", flush=True)

        # Checkpoint. The npz is rewritten whole each time (cheap next to the encode pass), so
        # an interrupted run always leaves a valid, if shorter, artifact rather than nothing.
        if (eps[-1] + 1 - last_saved) >= args.save_every:
            _write(out_dir, args, rows, num_hist, eps[-1] + 1, partial=True)
            last_saved = eps[-1] + 1

    _write(out_dir, args, rows, num_hist, n_eps, partial=False)


if __name__ == "__main__":
    main()
