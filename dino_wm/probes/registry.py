"""Load and run MANY probes on a WM latent.

Each probe is a small head over the frozen-DINO latent that reads ONE aspect of the scene
(cube position, per-cell occupancy, sign colour, ...). This registry loads a set of them from
a manifest (probes.yaml) and runs them on a latent, returning a named dict of outputs. It is
the perception layer that feeds (later) the legislation grounding + the planner objective.

Each probe .pth carries:
  kind:    'regression' | 'multilabel' | 'classification' | 'yaw_mod90_sincos4'
           (how to post-process the head)
  source:  'encoded' | 'predicted'                          (which latent it was TRAINED on)
The `source` is the key to handling latent-prediction probes: an 'encoded' probe must be
applied to phi(obs) (encoded real frame), a 'predicted' probe to a WM-rolled latent. The
registry groups by source (`by_source`, or forward(..., source=...)) so the caller routes the
right latent to each. (Old probes without these fields default to regression/encoded.)

    from probes.registry import ProbeRegistry
    reg = ProbeRegistry(device="cuda:0")
    reg.forward(enc_tokens, source="encoded")     # {'cube_position': (B,2) m, ...}
    reg.forward(pred_tokens, source="predicted")  # predicted-latent probes only
"""
import sys
from pathlib import Path

import numpy as np
import torch
import os

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from probes.probe_cube_position import MLP, _spatial_pool_grid  # shared head + pooling

_DEFAULT_MANIFEST = _HERE / "probes.yaml"


class Probe:
    """One loaded probe head. Applies pooled DINO tokens -> output, per `kind`."""

    def __init__(self, name, path, device="cpu"):
        self.name = name
        p = torch.load(path, map_location=device, weights_only=False)
        self.kind = p.get("kind") or ("regression" if "y_mu" in p else None)
        if self.kind is None:
            raise ValueError(f"probe '{name}' ({path}): cannot infer kind; add 'kind' to the .pth")
        self.source = p.get("source", "encoded")          # 'encoded' -> phi(obs); 'predicted' -> WM rollout
        self.pool_grid = int(p["pool_grid"])
        self.out_dim = int(p.get("out_dim") or np.asarray(p["y_mu"]).shape[-1])
        self.mlp = MLP(int(p["d_in"]), d_out=self.out_dim, hidden=int(p.get("hidden", 256))).to(device).eval()
        self.mlp.load_state_dict(p["state_dict"])
        self.x_mu = torch.as_tensor(p["x_mu"], device=device)
        self.x_sd = torch.as_tensor(p["x_sd"], device=device)
        self.y_mu = torch.as_tensor(p["y_mu"], device=device) if "y_mu" in p else None
        self.y_sd = torch.as_tensor(p["y_sd"], device=device) if "y_sd" in p else None
        self.meta = {k: v for k, v in p.items() if k != "state_dict"}
        self.device = device

    @torch.no_grad()
    def __call__(self, tokens):
        """tokens (B, P, D) -> output. regression: (B,out) in original units; multilabel:
        (B,out) per-element probabilities in [0,1]; classification: (B,out) class probs."""
        X = _spatial_pool_grid(tokens, self.pool_grid)
        z = self.mlp((X - self.x_mu.to(X.dtype)) / self.x_sd.to(X.dtype))
        if self.kind == "regression":
            return z * self.y_sd.to(X.dtype) + self.y_mu.to(X.dtype)
        if self.kind == "multilabel":
            return torch.sigmoid(z)
        if self.kind == "classification":
            return torch.softmax(z, dim=-1)
        if self.kind == "yaw_mod90_sincos4":
            # (B,2) = (sin 4t, cos 4t) on the unit circle. Returned RAW: there is no y_mu/y_sd to undo
            # (the target was never standardised), and the caller decodes with atan2(s,c)/4. The 4t
            # encoding folds the square cube's 90deg symmetry, so the decode lands in (-45,45] deg.
            return z
        raise ValueError(f"unknown probe kind: {self.kind}")


class ProbeRegistry:
    """A bundle of probes loaded from a manifest; run them on one latent, grouped by source."""

    def __init__(self, manifest=_DEFAULT_MANIFEST, device="cpu", root=_REPO):
        self.device = device
        self.root = Path(root)
        with open(manifest, "r", encoding="utf-8") as f:   # YAML has non-ascii (em dashes); container default is ascii
            spec = yaml.safe_load(f) or {}
        self.probes = {}
        for entry in spec.get("probes", []) or []:
            if not entry.get("enabled", True):
                continue
            path = Path(entry["path"])
            if not path.is_absolute():
                path = Path(root) / path
            self.probes[entry["name"]] = Probe(entry["name"], path, device=device)
        # PER-RUN OPT-IN for the orientation probe. Left OUT of probes.yaml deliberately: enabling it
        # there would switch every run in the repo from the axis-aligned body model to the oriented
        # one at once. With this unset the constraint sees no "cube_yaw" probe and behaves exactly as
        # it always has (verified: yaw=0 reproduces the axis-aligned verdicts bit for bit).
        _yp = os.environ.get("DINOWM_YAW_PROBE", "").strip()
        if _yp:
            pth = Path(_yp) if Path(_yp).is_absolute() else Path(root) / _yp
            self.probes["cube_yaw"] = Probe("cube_yaw", pth, device=device)
            print(f"[registry] cube_yaw ENABLED from DINOWM_YAW_PROBE={pth} -- enforcement will use "
                  f"the true oriented footprint instead of the axis-aligned one")

    def set_probe(self, name, path):
        """Force probe `name` to (re)load from `path` (absolute or repo-relative), replacing the
        manifest entry. TIES the legislation cube_position probe to the planner's probe
        (objective.pos_probe_path) so perception used for law enforcement physically cannot differ
        from perception used for planning -- one CLI knob, no silent mismatch (see plan.py). The
        probe's kind/source are read from its own .pth."""
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        self.probes[name] = Probe(name, p, device=self.device)

    def by_source(self, source):
        return {n: p for n, p in self.probes.items() if p.source == source}

    @torch.no_grad()
    def forward(self, tokens, source=None):
        """Run probes on `tokens`. If `source` is given ('encoded'/'predicted'), run only those
        probes — so the caller feeds the matching latent (phi(obs) vs WM rollout) to each group."""
        items = self.probes if source is None else self.by_source(source)
        return {name: probe(tokens) for name, probe in items.items()}

    def __getitem__(self, name):
        return self.probes[name]

    def names(self):
        return list(self.probes)
