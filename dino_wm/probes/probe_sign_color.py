"""Probe whether the stop-sign COLOR is decodable from the WM's DINOv2 latent.

This is the cheap de-risking check for the sign-as-rule plan: if an MLP can't
read the sign color off the (frozen) DINOv2 patch latent, no linear probe will
either, and the legislation stack needs a representation fix before being built.

The WM freezes its encoder (conf/train.yaml: train_encoder=False), so the WM's
visual latent == vanilla dinov2_vits14 patch tokens. We encode collected frames
with that same frozen encoder + the same preprocessing the WM uses
(Normalize(0.5,0.5) -> x*2-1), pool the patch grid, and train an MLP to classify
the per-episode sign color (labels from sign_colors.pth).

Reports held-out (split BY EPISODE, so no frame leakage) frame-level and
episode-level (majority-vote) accuracy + a confusion matrix. Read it as:
  ~>=95%  -> color cleanly present; build with confidence.
  middling-> entangled but usable (your gate is a nonlinear probe).
  ~chance (25%) -> DINOv2 ate the color; fix representation first.

No sim needed — run with the container python (torch + cached dinov2 hub):
    python probe_sign_color.py --data_dir data/isaaclab_grid_smoke_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent   # repo root (this file lives in probe/)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.dino import DinoV2Encoder

_DEFAULT_NAMES = ["white", "red", "yellow", "green"]  # matches collect SIGN_PALETTE order


def _spatial_pool_grid(tokens, grid=4):
    """(B, P, D) patch tokens -> (B, grid*grid*D). P must be a perfect square
    (16x16 for vits14@224). 4x4 keeps coarse location so a small fixed-location
    sign isn't diluted away by global mean-pooling."""
    b, p, d = tokens.shape
    s = int(round(p ** 0.5))
    assert s * s == p, f"patch count {p} not a square"
    g = tokens.reshape(b, s, s, d).permute(0, 3, 1, 2)        # (B, D, s, s)
    g = F.adaptive_avg_pool2d(g, grid)                         # (B, D, grid, grid)
    return g.reshape(b, -1)                                    # (B, grid*grid*D)


@torch.no_grad()
def encode_dataset(data_dir, encoder, device, frame_stride, batch_size):
    """Encode every (strided) frame -> pooled latent. Returns X (N,D), y (N,),
    ep (N,) episode index, and the list of color names."""
    p = Path(data_dir)
    labels = torch.load(p / "sign_colors.pth").numpy().astype(np.int64)  # (E,)
    files = sorted((p / "obses").glob("episode_*.pth"))
    assert len(files) == len(labels), f"{len(files)} obs files vs {len(labels)} labels"
    meta = json.loads((p / "metadata.json").read_text()) if (p / "metadata.json").exists() else {}
    names = meta.get("sign_palette", _DEFAULT_NAMES)

    feats, ys, eps = [], [], []
    for ei, f in enumerate(files):
        vid = torch.load(f)                       # (T,H,W,3) uint8
        sel = vid[::frame_stride].float() / 255.0
        x = sel.permute(0, 3, 1, 2)               # (n,3,H,W)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0                          # Normalize(0.5,0.5) == WM preprocessing
        for i in range(0, x.shape[0], batch_size):
            toks = encoder(x[i:i + batch_size].to(device))   # (b,P,D)
            feats.append(_spatial_pool_grid(toks).cpu())
        n = x.shape[0]
        ys.append(np.full(n, labels[ei], dtype=np.int64))
        eps.append(np.full(n, ei, dtype=np.int64))
        if (ei + 1) % 10 == 0:
            print(f"[encode] {ei + 1}/{len(files)} episodes")
    X = torch.cat(feats).numpy().astype(np.float32)
    y = np.concatenate(ys)
    ep = np.concatenate(eps)
    print(f"[encode] {X.shape[0]} frames, latent dim {X.shape[1]}")
    return X, y, ep, names


class MLP(nn.Module):
    def __init__(self, d_in, n_cls, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/isaaclab_stroke_1500")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame_stride", type=int, default=1, help="use every Nth frame per episode")
    ap.add_argument("--enc_batch", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2, help="fraction of EPISODES held out")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="CONTROL: scramble the color labels across episodes (break the "
                    "image<->color link). Real signal -> accuracy collapses to ~chance; "
                    "if it stays high there's a leak/bug.")
    ap.add_argument("--save_path", default=None,
                    help="if set, save the trained probe (+ train-set norm stats) as a "
                         "registry-format .pth (kind=classification, source=encoded) that "
                         "probes/registry.py + grounding.sign_color() can load.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad_(False)

    X, y, ep, names = encode_dataset(args.data_dir, encoder, device, args.frame_stride, args.enc_batch)
    n_cls = len(names)
    print(f"[data] class counts (frames): "
          + ", ".join(f"{names[c]}={int((y == c).sum())}" for c in range(n_cls)))

    if args.shuffle_labels:
        # Reassign each episode a label via a permutation of the per-episode labels
        # (keeps class balance, destroys the image<->color correspondence).
        uniq = np.unique(ep)
        orig = np.array([y[ep == e][0] for e in uniq])
        shuffled = np.random.RandomState(args.seed + 1).permutation(orig)
        remap = {int(e): int(shuffled[i]) for i, e in enumerate(uniq)}
        y = np.array([remap[int(e)] for e in ep], dtype=np.int64)
        print("[CONTROL] labels shuffled across episodes — expect ~chance "
              f"({1.0 / n_cls:.2f}); anything high means a leak/bug.")

    # --- split BY EPISODE (no frame leakage) ---
    ep_ids = np.unique(ep)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ep_ids)
    n_val = max(1, int(round(len(ep_ids) * args.val_frac)))
    val_eps = set(ep_ids[:n_val].tolist())
    test_mask = np.array([e in val_eps for e in ep])
    train_mask = ~test_mask
    print(f"[split] {len(ep_ids) - n_val} train / {n_val} test episodes "
          f"({train_mask.sum()} / {test_mask.sum()} frames)")

    # --- standardize on train stats ---
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).to(device)
    tr = torch.from_numpy(np.where(train_mask)[0])
    te = torch.from_numpy(np.where(test_mask)[0])
    # Standardize on TRAIN frames only (integer-indexed, not np-bool) so no test
    # statistics leak into the normalization.
    mu = Xt[tr].mean(0, keepdim=True)
    sd = Xt[tr].std(0, keepdim=True) + 1e-6
    Xn = ((Xt - mu) / sd).to(device)

    probe = MLP(X.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    # pred/ycpu/te_ep are all in the same ascending test-frame order (np.where and
    # bool-mask indexing both ascend), so frame acc and the episode grouping align.
    def evaluate():
        probe.eval()
        with torch.no_grad():
            pred = probe(Xn[te]).argmax(1)
        pcpu, ycpu = pred.cpu().numpy(), yt[te].cpu().numpy()
        acc = float((pcpu == ycpu).mean())
        # Balanced accuracy = mean per-class recall. Immune to a probe inflating
        # plain accuracy by favoring the majority class on a skewed test set.
        recalls = [float((pcpu[ycpu == c] == c).mean()) for c in range(n_cls) if (ycpu == c).any()]
        bal = float(np.mean(recalls)) if recalls else 0.0
        # episode-level majority vote (the runtime-relevant metric)
        te_ep = ep[test_mask]
        ep_correct = []
        for e in np.unique(te_ep):
            m = te_ep == e
            maj = np.bincount(pcpu[m], minlength=n_cls).argmax()
            ep_correct.append(maj == ycpu[m][0])
        return acc, bal, float(np.mean(ep_correct)), pcpu, ycpu

    te_counts = np.bincount(y[test_mask], minlength=n_cls)
    print("[split] test-frame class counts: "
          + ", ".join(f"{names[c]}={int(te_counts[c])}" for c in range(n_cls))
          + "  (skew here is why we also report balanced acc)")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        probe.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(probe(Xn[idx]), yt[idx])
            loss.backward()
            opt.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            acc, bal, ep_acc, _, _ = evaluate()
            best = max(best, acc)
            print(f"[probe] epoch {epoch:3d}  frame-acc {acc:.3f}  bal-acc {bal:.3f}  episode-acc {ep_acc:.3f}")

    acc, bal, ep_acc, pcpu, ycpu = evaluate()
    print(f"\n[RESULT] frame-acc {acc:.3f} | balanced-acc {bal:.3f} | episode-acc {ep_acc:.3f}  "
          f"(chance {1.0 / n_cls:.2f})")
    print(f"         balanced-acc is the inflation-proof one; episode-acc is the runtime metric.")
    print(f"         best test frame-acc during training = {max(best, acc):.3f}  "
          f"(optimistic — peeks at test to pick the epoch)")
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, pdt in zip(ycpu, pcpu):
        cm[t, pdt] += 1
    print("[confusion] rows=true, cols=pred  (" + ", ".join(names) + ")")
    for c in range(n_cls):
        print(f"  {names[c]:>7s} " + " ".join(f"{cm[c, j]:5d}" for j in range(n_cls)))

    if args.save_path:
        # Registry-format checkpoint (mirrors probe_cube_position). The registry rebuilds the
        # SHARED MLP (probe_cube_position.MLP) + _spatial_pool_grid(pool_grid) and softmaxes the
        # logits (kind=classification); the local MLP here is arch-identical so state_dict loads.
        # `names` is the class order -> grounding.sign_color emits sign(<names[argmax]>). pool_grid=4
        # matches the _spatial_pool_grid default used above (latent dim 4*4*D).
        torch.save({
            "kind": "classification", "source": "encoded",
            "state_dict": probe.state_dict(),
            "x_mu": mu.numpy(), "x_sd": sd.numpy(),
            "pool_grid": 4, "d_in": int(X.shape[1]), "out_dim": int(n_cls), "hidden": 256,
            "names": list(names),
        }, args.save_path)
        print(f"[save] sign-color probe + norm stats -> {args.save_path}")


if __name__ == "__main__":
    main()
