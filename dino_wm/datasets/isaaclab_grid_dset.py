"""Single-robot DinoWM grid dataset loader.

On-disk schema (scripts/collect_isaaclab_grid_data.py):
  states.pth        (E, T, 31)        [arm jpos(9), jvel(9), cube(13)]
  actions.pth       (E, T, 4)         planar stroke [x_start, y_start, dx, dy] (end = start + disp)
  proprio.pth       (E, T, 18)        [arm jpos(9), jvel(9)]
  cell_labels.pth   (E, T) int64      single cube cell id (optional)
  seq_lengths.pth   (E,) int64
  sign_colors.pth   (E,) int64        per-episode sign color, indexes metadata sign_palette (optional)
  obses/episode_NNNNN.pth   (T, H, W, 3) uint8   single camera (zero-pad = metadata ep_pad)

Single robot / single camera: action 4-D (one stroke = one frame), proprio 18-D,
state 31-D. frameskip MUST be 1 for this stroke task (asserted in the loader), so
TrajSlicerDataset's action-concat is the identity and the model sees a 4-D action.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset, split_traj_datasets


def _resolve_data_dir(data_path):
    """Resolve the dataset dir robustly. The config's absolute path uses the host
    repo prefix (ckpt_base_path); under the container the repo is mounted at a
    different prefix, so the host path won't exist. Fall back to
    <repo_root>/data/<name> (repo_root inferred from this file's location), which
    is correct in whatever environment train.py runs in."""
    p = Path(data_path)
    if (p / "states.pth").is_file():
        return p
    repo_root = Path(__file__).resolve().parent.parent  # <repo>/datasets/.. -> <repo>
    alt = repo_root / "data" / p.name
    if (alt / "states.pth").is_file():
        return alt
    raise FileNotFoundError(
        f"dataset not found — tried '{p}' and '{alt}' (neither has states.pth). "
        f"Override with env.dataset.data_path=<dir containing states.pth>."
    )


class IsaacLabSingleDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        preload: bool = True,
    ):
        p = _resolve_data_dir(data_path)
        print(f"[dataset] resolved data dir: {p}")
        self.data_path = p
        self.transform = transform
        meta = json.loads((p / "metadata.json").read_text()) if (p / "metadata.json").exists() else {}
        self.ep_pad = int(meta.get("ep_pad", 5))

        states = torch.load(p / "states.pth").float()
        actions = torch.load(p / "actions.pth").float() / action_scale
        proprios = torch.load(p / "proprio.pth").float()
        seq_lengths = torch.load(p / "seq_lengths.pth")
        cell_path, sign_path = p / "cell_labels.pth", p / "sign_colors.pth"
        cell_labels = torch.load(cell_path) if cell_path.exists() else None
        sign_colors = torch.load(sign_path) if sign_path.exists() else None

        n = n_rollout if n_rollout else len(states)
        self.states = states[:n]
        self.actions = actions[:n]
        self.proprios = proprios[:n]
        self.seq_lengths = seq_lengths[:n]
        self.cell_labels = cell_labels[:n] if cell_labels is not None else None
        self.sign_colors = sign_colors[:n] if sign_colors is not None else None
        print(f"Loaded {n} single-robot IsaacLab grid rollouts from {p}")

        self.action_dim = self.actions.shape[-1]   # 7
        self.state_dim = self.states.shape[-1]      # 31
        self.proprio_dim = self.proprios.shape[-1]  # 18

        if normalize_action:
            self.action_mean, self.action_std = self._mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self._mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self._mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean, self.action_std = torch.zeros(self.action_dim), torch.ones(self.action_dim)
            self.state_mean, self.state_std = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.proprio_mean, self.proprio_std = torch.zeros(self.proprio_dim), torch.ones(self.proprio_dim)

        # Raw per-dim action RANGE over valid frames (planner clamp bounds). Must
        # be computed BEFORE normalization below, while self.actions is still raw.
        _flat_a = torch.vstack(
            [self.actions[i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))]
        )
        self.action_min = _flat_a.min(dim=0).values  # (action_dim,)
        self.action_max = _flat_a.max(dim=0).values

        # actions + proprio are model inputs -> normalized; states left raw (used
        # for eval / cell labels, not fed to the model).
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

        # Preload all episode image stacks into RAM (uint8) so training reads from
        # memory, not disk. The old per-window torch.load re-read a ~9MB episode
        # file for each of ~41 windows -> disk-bound, ~20min/epoch. With workers
        # this big tensor is copy-on-write shared (loaded once in the main proc).
        self.obses = None
        if preload:
            files = sorted((p / "obses").glob("episode_*.pth"))[:n]
            assert len(files) == n, f"{len(files)} obs files for {n} episodes"
            first = torch.load(files[0])  # (T, H, W, 3) uint8
            self.obses = torch.empty((n, *first.shape), dtype=torch.uint8)
            self.obses[0] = first
            for i in range(1, n):
                self.obses[i] = torch.load(files[i])
                if (i + 1) % 250 == 0:
                    print(f"[dataset] preloaded {i + 1}/{n} episode image stacks")
            print(f"[dataset] preloaded all obses into RAM ({self.obses.numel() / 1e9:.1f} GB uint8)")

    @staticmethod
    def _mean_std(data, traj_lengths):
        flat = torch.vstack([data[i, : traj_lengths[i]] for i in range(len(traj_lengths))])
        mean, std = flat.mean(dim=0), flat.std(dim=0)
        # Constant dims (e.g. the always-open gripper action, held finger joints)
        # have std=0 -> (x-mean)/0 = NaN, which poisons the whole loss. Set those
        # stds to 1 so the dim normalizes to a harmless constant 0 instead.
        std[std < 1e-6] = 1.0
        return mean, std

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        return torch.cat(
            [self.actions[i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))], dim=0
        )

    def get_frames(self, idx, frames):
        if self.obses is not None:               # from RAM (preloaded)
            image = self.obses[idx]
        else:                                     # fallback: from disk per call
            image = torch.load(self.data_path / "obses" / f"episode_{idx:0{self.ep_pad}d}.pth")
        image = rearrange(image[frames].float() / 255.0, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        info = {}
        if self.cell_labels is not None:
            info["cell_labels"] = self.cell_labels[idx, frames]
        if self.sign_colors is not None:
            n = len(frames) if hasattr(frames, "__len__") else len(list(frames))
            info["sign_color"] = self.sign_colors[idx].repeat(n)  # per-episode -> per-frame
        return obs, self.actions[idx, frames], self.states[idx, frames], info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


class _FastSlicer(TrajSlicerDataset):
    """Same slice list / dims as TrajSlicerDataset, but transforms ONLY the
    num_frames frames each window actually uses. The parent calls
    dataset[i] (the whole episode) and then slices, so it float()/Normalize's all
    ~60 frames per window (~15x wasted CPU). Here we index the in-RAM arrays and
    transform just the windowed frames -> num_workers=0 is fast, no /dev/shm needed."""

    def __getitem__(self, idx):
        i, start, end = (int(v) for v in self.slices[idx])
        sub = self.dataset                       # TrajSubset (train/val split)
        real = int(sub.indices[i])               # base-dataset episode index
        base = sub.dataset                        # IsaacLabSingleDataset
        fr = list(range(start, end, self.frameskip))  # the num_frames used frames

        if base.obses is not None:                # RAM (preloaded)
            img = base.obses[real][fr]
        else:                                     # fallback: load file, take fr
            full = torch.load(base.data_path / "obses" / f"episode_{real:0{base.ep_pad}d}.pth")
            img = full[fr]
        img = rearrange(img.float() / 255.0, "T H W C -> T C H W")
        if base.transform:
            img = base.transform(img)

        obs = {"visual": img, "proprio": base.proprios[real][fr]}
        act = base.actions[real][start:end]                       # (frameskip*num_frames, A)
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)
        state = base.states[real][fr]
        return obs, act, state


def load_isaaclab_single_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/isaaclab_single_stroke",
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    # The planar-stroke action is one stroke per frame; the WM/planner pipeline
    # only collapses correctly (and the planner action bounds only tile correctly)
    # at frameskip=1. Fail loudly here rather than silently mis-tiling a 4-D stroke.
    assert frameskip == 1, (
        f"isaaclab stroke task requires frameskip=1 (one stroke per frame), got {frameskip}. "
        f"Set conf/train.yaml frameskip: 1."
    )
    dset = IsaacLabSingleDataset(
        n_rollout=n_rollout, transform=transform, data_path=data_path,
        normalize_action=normalize_action,
    )
    train, val = split_traj_datasets(dset, train_fraction=split_ratio, random_seed=42)
    nf = num_hist + num_pred
    train_s = _FastSlicer(train, nf, frameskip)
    val_s = _FastSlicer(val, nf, frameskip)
    return {"train": train_s, "valid": val_s}, {"train": train, "valid": val}
