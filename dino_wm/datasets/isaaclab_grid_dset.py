"""DinoWM grid dataset loader.

On-disk schema (scripts/collect_isaaclab_grid_data.py):
  states.pth         (E, T, 62)
  actions_left.pth   (E, T, 7)        actions_right.pth   (E, T, 7)
  proprio_left.pth   (E, T, 18)       proprio_right.pth   (E, T, 18)
  cell_labels.pth    (E, T, 2) int64  (optional; per-cube: red, blue)
  seq_lengths.pth    (E,) int64
  obses/left/episode_NNN.pth   (T, H, W, 3) uint8
  obses/right/episode_NNN.pth  (T, H, W, 3) uint8

cooperative=True: 14-D action + 36-D proprio per sample. False not yet wired.
"""
from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from .traj_dset import TrajDataset, get_train_val_sliced


class IsaacLabGridDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        cooperative: bool = True,
        camera: str = "left",
    ):
        if not cooperative:
            raise NotImplementedError("Non-cooperative path not yet wired up.")
        if camera not in ("left", "right"):
            raise ValueError(
                f"camera must be 'left' or 'right' (per-robot OTS views), got {camera!r}"
            )

        p = Path(data_path)
        self.data_path = p
        self.transform = transform
        self.camera = camera

        states = torch.load(p / "states.pth").float()
        actions = torch.cat([
            torch.load(p / "actions_left.pth").float() / action_scale,
            torch.load(p / "actions_right.pth").float() / action_scale,
        ], dim=-1)
        proprios = torch.cat([
            torch.load(p / "proprio_left.pth").float(),
            torch.load(p / "proprio_right.pth").float(),
        ], dim=-1)
        seq_lengths = torch.load(p / "seq_lengths.pth")
        cell_path = p / "cell_labels.pth"
        cell_labels = torch.load(cell_path) if cell_path.exists() else None

        n = n_rollout if n_rollout else len(states)
        self.states = states[:n]
        self.actions = actions[:n]
        self.proprios = proprios[:n]
        self.seq_lengths = seq_lengths[:n]
        self.cell_labels = cell_labels[:n] if cell_labels is not None else None
        print(f"Loaded {n} IsaacLab grid rollouts (cooperative=True, camera={camera})")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self._mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self._mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self._mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean, self.action_std = torch.zeros(self.action_dim), torch.ones(self.action_dim)
            self.state_mean, self.state_std = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.proprio_mean, self.proprio_std = torch.zeros(self.proprio_dim), torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    @staticmethod
    def _mean_std(data, traj_lengths):
        flat = torch.vstack([data[i, : traj_lengths[i]] for i in range(len(traj_lengths))])
        return flat.mean(dim=0), flat.std(dim=0)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([self.actions[i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))], dim=0)

    def get_frames(self, idx, frames):
        image = torch.load(self.data_path / "obses" / self.camera / f"episode_{idx:03d}.pth")
        image = rearrange(image[frames].float() / 255.0, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        info = {"cell_labels": self.cell_labels[idx, frames]} if self.cell_labels is not None else {}
        return obs, self.actions[idx, frames], self.states[idx, frames], info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_isaaclab_grid_slice_train_val(
    transform, n_rollout=None, data_path="data/isaaclab_grid",
    normalize_action=False, cooperative=True, camera="left",
    split_ratio=0.9, num_hist=0, num_pred=0, frameskip=0,
):
    dset = IsaacLabGridDataset(
        n_rollout=n_rollout, transform=transform, data_path=data_path,
        normalize_action=normalize_action, cooperative=cooperative, camera=camera,
    )
    train, val, train_s, val_s = get_train_val_sliced(
        traj_dataset=dset, train_fraction=split_ratio,
        num_frames=num_hist + num_pred, frameskip=frameskip,
    )
    return {"train": train_s, "valid": val_s}, {"train": train, "valid": val}
