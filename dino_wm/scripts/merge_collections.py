"""Merge two or more collect_isaaclab_grid_data.py output dirs into one dataset.

WHY. Collection is not GPU-bound (measured ~2% utilisation on an A100) but the per-process env count
is capped by the RTX descriptor pool, not VRAM -- overshooting --num_envs hard-crashes. So the way to
use more of the machine is several PROCESSES on separate GPUs, each at a safe --num_envs, merged
afterwards. This does the merge.

THE ONE THING THAT MUST BE TRUE: the parts must have DIFFERENT EFFECTIVE SEEDS. The collector builds
ONE `np.random.RandomState(effective_seed)` per run and streams it through every episode -- the seed
is per-RUN, not per-episode -- so two runs launched with the same effective seed replay the same cube
spawns and the same strokes, and merging them gives a dataset that is half duplicates while looking
twice as large. Each part records `effective_seed` (= seed + resumed_from); this compares those and
REFUSES on a collision.

WHAT THIS IS NOT: the merged set is a valid sample of the same distribution, but it is NOT the
dataset a single sequential run of the same size would have produced. Part B opens a fresh RNG stream
rather than continuing part A's, and continuing it exactly is unreachable -- StrokeSampler branches on
the cube's physical position (stroke_sampler.py:94 vs :96) and those branches call rng.normal vs
rng.uniform, which consume different numbers of underlying draws, so the stream advance per episode
depends on physics. There are also two generators to keep in step (numpy for strokes, torch for cube
spawns). Nothing downstream depends on which draw you got.

    python scripts/merge_collections.py OUT_DIR PART_A PART_B [PART_C ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import provenance

EP_PAD = 5                                   # episode_NNNNN.pth -- must match the collector
ARRAYS = ["states", "actions", "proprio", "cell_labels", "seq_lengths", "sign_colors"]


def _eff_seed(meta):
    """The seed this part's single RNG stream was built from (seed + resumed_from when resumed)."""
    return int(meta.get("effective_seed", meta.get("seed", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--move", action="store_true",
                    help="hard-link episode files instead of copying (same filesystem only; saves "
                    "the full dataset size in disk and most of the wall-clock)")
    a = ap.parse_args()
    out = Path(a.out)
    parts = [Path(p) for p in a.parts]
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty -- refusing to merge into it")

    metas, counts = [], []
    for p in parts:
        m = json.loads((p / "metadata.json").read_text())
        n = len(torch.load(p / "seq_lengths.pth"))
        n_obs = len(list((p / "obses").glob("episode_*.pth")))
        if n_obs != n:
            raise SystemExit(f"{p}: {n} seq_lengths but {n_obs} obs files -- part is mid-write, "
                             "stop the collector before merging")
        metas.append(m); counts.append(n)
        print(f"[part] {p}  {n} episodes  seed={m.get('seed')}  "
              f"eff_seed={_eff_seed(m)}  lock_cube_yaw={m.get('lock_cube_yaw')}")

    # 1) DISTINCT RNG STREAMS -- the failure this script exists to prevent
    seeds = [_eff_seed(m) for m in metas]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if seeds[i] == seeds[j]:
                raise SystemExit(
                    f"REFUSING: {parts[i].name} and {parts[j].name} share effective seed "
                    f"{seeds[i]}. One RandomState per run means identical seeds replay identical "
                    f"episodes -- these parts hold the SAME data, and merging would duplicate it "
                    f"while appearing to double the set. Re-collect one with a different --seed.")

    # 2) COLLECTION PARAMS must agree, or the merged set is not one distribution
    KEYS = ["episode_len", "lock_cube_yaw", "aimed_frac", "push_max", "aim_push_range",
            "aim_offset_sd", "start_margin", "stroke_max_steps", "task_id", "render_mode", "spp",
            "img_hw", "action_repr", "state_dim", "action_dim", "proprio_dim"]
    for k in KEYS:
        vals = {json.dumps(m.get(k), sort_keys=True) for m in metas}
        if len(vals) > 1:
            raise SystemExit(f"REFUSING: parts disagree on '{k}': {vals}. Merging different "
                             "collection settings silently mixes two distributions.")

    (out / "obses").mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        cat = torch.cat([torch.load(p / f"{name}.pth") for p in parts], dim=0)
        torch.save(cat, out / f"{name}.pth")
        print(f"[merge] {name}.pth -> {tuple(cat.shape)}")

    off = 0
    for p, n in zip(parts, counts):
        for i, f in enumerate(sorted((p / "obses").glob("episode_*.pth"))):
            dst = out / "obses" / f"episode_{off + i:0{EP_PAD}d}.pth"
            if a.move:
                dst.hardlink_to(f)
            else:
                shutil.copy(f, dst)
        off += n
        print(f"[merge] {p.name}: {n} episodes -> indices {off - n}..{off - 1}")

    total = sum(counts)
    assert len(list((out / "obses").glob("episode_*.pth"))) == total
    assert len(torch.load(out / "seq_lengths.pth")) == total
    meta = dict(metas[0])
    meta.update({"num_episodes": total,
                 "_merged_from": [{"dir": str(p), "n": n, "seed": m.get("seed"),
                                   "effective_seed": _eff_seed(m)}
                                  for p, n, m in zip(parts, counts, metas)],
                 "_note": "merged by scripts/merge_collections.py; effective seeds verified distinct"})
    meta.pop("seed", None); meta.pop("effective_seed", None); meta.pop("resumed_from_episode", None)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    provenance.write(out, __file__, args=a, repo=_REPO_ROOT)
    print(f"\n[merge] DONE: {total} episodes -> {out}")
    print(f"        effective seeds (verified distinct): "
          f"{', '.join(f'{p.name}:{sd}' for p, sd in zip(parts, seeds))}")


if __name__ == "__main__":
    main()
