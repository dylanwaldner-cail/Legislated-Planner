import os
import logging
import warnings
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, open_dict
import time

from einops import rearrange, repeat
from custom_resolvers import replace_slash
from utils import cfg_to_dict, seed
from plan import load_model

from preprocessor import Preprocessor
from utils import move_to_device

from legislative_harness.probe import LegislativeProbe
from legislative_harness.utils import is_illegal_state, eval_probe 

from torch.nn import BCELoss

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="train_probe_point_maze")
def main(cfg: OmegaConf):

    # match repo pattern
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()

    cfg_dict = cfg_to_dict(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    seed(cfg.training.seed)

    # load world model
    repo_root = Path(get_original_cwd())
    model_path = repo_root / cfg.ckpt_base_path / "outputs" / cfg.model_name

    model_ckpt = model_path / "checkpoints" / "model_latest.pth"

    train_cfg = OmegaConf.load(model_path / "hydra.yaml")

    _, traj_dset = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )

    train_dset = traj_dset["train"]
    val_dset = traj_dset["valid"]

    data_preprocessor = Preprocessor(
        action_mean=train_dset.action_mean,
        action_std=train_dset.action_std,
        state_mean=train_dset.state_mean,
        state_std=train_dset.state_std,
        proprio_mean=train_dset.proprio_mean,
        proprio_std=train_dset.proprio_std,
        transform=train_dset.transform,
    )

    wm = load_model(
        model_ckpt=model_ckpt,
        train_cfg=train_cfg,
        num_action_repeat=train_cfg.num_action_repeat,
        device=device,
    )

    wm.eval()

    num_episodes = cfg.data.num_episodes
    num_samples = cfg.training.num_samples

    probe = LegislativeProbe(input_dim=cfg.probe.input_dim).to(device)

    probe_optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=cfg.probe.lr
    )

    illegal_region = OmegaConf.to_container(cfg.probe.illegal_region, resolve=True)

    probe.train()

    log.info("Setup complete.")

    log_every = 50
    running_loss = 0.0
    running_acc = 0.0
    running_illegal_rate = 0.0
    running_illegal_acc = 0.0
    running_illegal_count = 0
    running_time = 0.0
    running_episode_count = 0

    num_train = len(train_dset)
    num_val = len(val_dset) // 2
    num_test = len(val_dset) - num_val

    train_episodes = list(range(num_train))
    val_episodes = list(range(num_val))
    test_episodes = list(range(num_val, num_val + num_test))

    # before the training loop
    '''
    total_illegal = sum(is_illegal_state(train_dset.get_frames(ep, list(range(train_dset.get_seq_length(ep))))[2], illegal_region).sum().item() for ep in train_episodes)
    total_frames = sum(train_dset.get_seq_length(ep) for ep in train_episodes)
    fixed_pos_weight = torch.tensor((total_frames - total_illegal) / (total_illegal + 1e-8)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=fixed_pos_weight)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    '''
    for episode in range(num_train):
        start_time = time.time()

        frames = list(range(train_dset.get_seq_length(episode)))
        obs, action, states, _ = train_dset.get_frames(episode, frames)

        illegal_states = is_illegal_state(states, illegal_region).float().to(device)

        # Weigh the illegal states equally in the loss
        num_pos = illegal_states.sum()
        num_neg = illegal_states.numel() - num_pos
        pos_weight = num_neg / (num_pos + 1e-8)

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # returning to loss reweighting loss_fn = torch.nn.BCEWithLogitsLoss()

        obs = {
            "visual": obs["visual"].unsqueeze(0).to(device),
            "proprio": obs["proprio"].unsqueeze(0).to(device),
        }

        with torch.no_grad():
            z = wm.encode_obs(obs)

        z_visual = z["visual"].squeeze(0)        # [T, 196, 384]
        z_proprio = z["proprio"].squeeze(0)      # [T, 10]
        z_input = torch.cat([
            z_visual.mean(dim=1),                # [T, 384]
            z_proprio                            # [T, 10]
        ], dim=-1)                               # [T, 394]

        illegal_mask = is_illegal_state(states, illegal_region).bool()

        logits = probe(z_input).squeeze()
        loss = loss_fn(logits, illegal_states)

        probe_optimizer.zero_grad()
        loss.backward()
        probe_optimizer.step()

        # metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            acc = (preds == illegal_states).float().mean()
            illegal_rate = illegal_states.mean()

            illegal_mask = (illegal_states == 1)
            if illegal_mask.sum() > 0:
                illegal_acc = (preds[illegal_mask] == illegal_states[illegal_mask]).float().mean()
                running_illegal_acc += illegal_acc.item()
                running_illegal_count += 1

        elapsed = time.time() - start_time

        running_loss += loss.item()
        running_acc += acc.item()
        running_illegal_rate += illegal_rate.item()
        running_time += elapsed

        if (episode + 1) % log_every == 0:
            avg_illegal_acc = (
                running_illegal_acc / running_illegal_count
                if running_illegal_count > 0 else 0.0
            )
            val_metrics = eval_probe(
                probe=probe,
                wm=wm,
                dset=val_dset,
                eval_episodes=val_episodes,
                illegal_region=illegal_region,
                loss_fn=torch.nn.BCEWithLogitsLoss(),
                device=device,
            )

            log.info(
                f"episode={episode + 1}/{len(train_episodes)} "
                f"loss={running_loss / log_every:.4f} "
                f"acc={running_acc / log_every:.4f} "
                f"illegal_rate={running_illegal_rate / log_every:.4f} "
                f"illegal_acc={avg_illegal_acc:.4f} "
                f"time/ep={running_time / log_every:.4f}s "
                f"val_loss={val_metrics['eval_loss']:.4f} "
                f"val_acc={val_metrics['eval_acc']:.4f} "
                f"val_illegal_rate={val_metrics['eval_illegal_rate']:.4f} "
                f"val_illegal_acc={val_metrics['eval_illegal_acc']:.4f}"
            )
            running_loss = 0.0
            running_acc = 0.0
            running_illegal_rate = 0.0
            running_illegal_acc = 0.0
            running_illegal_count = 0
            running_time = 0.0
            running_episode_count = 0


    test_metrics = eval_probe(
            probe=probe,
            wm=wm,
            dset=val_dset,
            eval_episodes=test_episodes,
            illegal_region=illegal_region,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            device=device,
        )

    log.info(
            f"Test Results "
            f"test_loss={test_metrics['eval_loss']:.4f} "
            f"test_acc={test_metrics['eval_acc']:.4f} "
            f"test_illegal_rate={test_metrics['eval_illegal_rate']:.4f} "
            f"test_illegal_acc={test_metrics['eval_illegal_acc']:.4f}"
        )

    log.info(f"Learning Rate: {cfg.probe.lr}")



if __name__ == "__main__":
    main()
