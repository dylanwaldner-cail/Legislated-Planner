import torch
import hashlib
import h5py     
import numpy as np

def is_illegal_state(states, illegal_region):
    x_min, x_max = illegal_region["x_min"], illegal_region["x_max"]
    y_min, y_max = illegal_region["y_min"], illegal_region["y_max"]

    xy = states[:, :2]  # [T, 2]

    illegal = (
        (xy[:, 0] >= x_min) &
        (xy[:, 0] <= x_max) &
        (xy[:, 1] >= y_min) &
        (xy[:, 1] <= y_max)
    )

    return illegal.float()

def eval_probe(probe, wm, dset, eval_episodes, illegal_region, loss_fn, device):
    probe.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_illegal_rate = 0.0
    total_illegal_acc = 0.0
    illegal_count = 0

    with torch.no_grad():
        for ep in eval_episodes:
            frames = list(range(dset.get_seq_length(ep)))
            obs, action, states, _ = dset.get_frames(ep, frames)

            labels = is_illegal_state(states, illegal_region).float().to(device)

            obs = {
                "visual": obs["visual"].unsqueeze(0).to(device),
                "proprio": obs["proprio"].unsqueeze(0).to(device),
            }

            z = wm.encode_obs(obs)

            z_visual = z["visual"].squeeze(0)        # [T, 196, 384]
            z_proprio = z["proprio"].squeeze(0)      # [T, 10]
            z_input = torch.cat([
                z_visual.mean(dim=1),                # [T, 384]
                z_proprio                            # [T, 10]
            ], dim=-1)    

            logits = probe(z_input).squeeze()
            loss = loss_fn(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            acc = (preds == labels).float().mean()
            illegal_rate = labels.mean()

            illegal_mask = labels == 1
            if illegal_mask.sum() > 0:
                illegal_acc = (preds[illegal_mask] == labels[illegal_mask]).float().mean()
                total_illegal_acc += illegal_acc.item()
                illegal_count += 1

            total_loss += loss.item()
            total_acc += acc.item()
            total_illegal_rate += illegal_rate.item()

    n = len(eval_episodes)
    probe.train()

    return {
        "eval_loss": total_loss / n,
        "eval_acc": total_acc / n,
        "eval_illegal_rate": total_illegal_rate / n,
        "eval_illegal_acc": total_illegal_acc / illegal_count if illegal_count > 0 else 0.0,
    }

def append_step(path, z_input, action, illegal_label, illegal_region, pixel_mse, div_visual, div_proprio, div, start_env_state, new_env_state, iteration, traj_idx, step, seed_val):
    z_hash = hashlib.md5(z_input.cpu().numpy().tobytes()).hexdigest()

    with h5py.File(path, "a") as f:
        if z_hash in f:
            return  # duplicate, skip

        grp = f.create_group(z_hash)
        grp.create_dataset("z_input",         data=z_input.cpu().numpy())
        grp.create_dataset("action",          data=action)
        grp.create_dataset("illegal_label",   data=np.array(illegal_label.item()))
        grp.create_dataset("pixel_mse",       data=np.array(pixel_mse))
        grp.create_dataset("div_visual",      data=np.array(div_visual))
        grp.create_dataset("div_proprio",     data=np.array(div_proprio))
        grp.create_dataset("div",             data=np.array(div))
        grp.create_dataset("start_env_state", data=start_env_state)
        grp.create_dataset("new_env_state",   data=new_env_state) # After rollout, new state for next step
        grp.create_dataset("illegal_region",  data=np.array([
            illegal_region["x_min"], illegal_region["x_max"],
            illegal_region["y_min"], illegal_region["y_max"],
        ]))
        grp.create_dataset("iteration",       data=np.array(iteration))
        grp.create_dataset("traj_idx",        data=np.array(traj_idx))
        grp.create_dataset("step",            data=np.array(step))
        grp.create_dataset("seed_val",        data=np.array(seed_val))
