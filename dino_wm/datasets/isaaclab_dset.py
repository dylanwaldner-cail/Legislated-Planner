"""IsaacLab stub dataset loader.

Mirrors datasets/point_maze_dset.py's on-disk format and __getitem__ contract
so the rest of the dino_wm pipeline is identical.
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange

from .traj_dset import TrajDataset, get_train_val_sliced


class IsaacLabStubDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float() / action_scale
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth")

        n = n_rollout if n_rollout else len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states.clone()
        print(f"Loaded {n} IsaacLab stub rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self._mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self._mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self._mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    @staticmethod
    def _mean_std(data, traj_lengths):
        all_data = torch.vstack([data[i, : traj_lengths[i]] for i in range(len(traj_lengths))])
        return all_data.mean(dim=0), all_data.std(dim=0)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([self.actions[i, : self.seq_lengths[i]] for i in range(len(self.seq_lengths))], dim=0)

    def get_frames(self, idx, frames):
        image = torch.load(self.data_path / "obses" / f"episode_{idx:03d}.pth")
        image = image[frames].float() / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        return obs, self.actions[idx, frames], self.states[idx, frames], {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_isaaclab_stub_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/isaaclab_stub",
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = IsaacLabStubDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    return {"train": train_slices, "valid": val_slices}, {"train": dset_train, "valid": dset_val}
