"""Run provenance — one manifest recording HOW an artifact was produced.

Every data/result-producing script (collection, precompute, probe training, eval, generators)
calls `provenance.write(out, __file__, args, extra=...)` so each output directory carries a
machine-readable record of: the exact command, ALL parsed arguments, the git commit, the time,
cwd, and host. Goal: months later `cat <dir>/manifest.json` tells you exactly what was run and
how to reproduce it, without spelunking scrollback.

Design notes:
  * Provenance must NEVER crash a run — every step is best-effort (git failures -> None).
  * `extra` carries whatever the script wants alongside the call record (e.g. result metrics).
  * Non-JSON values (Path, numpy scalars/arrays) are stringified via default=str; convert
    arrays to plain lists/floats in `extra` if you want them queryable rather than stringified.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent   # provenance.py lives at the repo root (next to .git)


def git_commit(repo=None):
    """Short SHA, suffixed '-dirty' if the working tree has uncommitted changes. None if not a git repo.
    Defaults to THIS file's dir (the repo root) rather than cwd, so it still works when the caller has
    chdir'd elsewhere (e.g. Hydra apps like plan.py run inside their output dir)."""
    repo = str(repo) if repo else str(_REPO_ROOT)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=repo, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=repo, stderr=subprocess.DEVNULL).decode().strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def _quote(a):
    return a if all(c.isalnum() or c in "._-/=:" for c in a) and a else "'" + a.replace("'", "'\\''") + "'"


def build(script, args=None, extra=None, repo=None):
    """Provenance dict: what ran, how, when, from which commit. `args` is an argparse.Namespace,
    a dict, or None; `extra` is merged in at the top level (e.g. {'results': {...}})."""
    if args is None:
        argd = {}
    elif isinstance(args, dict):
        argd = dict(args)
    else:
        argd = vars(args)
    argd = {k: (str(v) if isinstance(v, Path) else v) for k, v in argd.items()}
    cmd = " ".join(_quote(a) for a in ([sys.executable] + sys.argv))
    m = {
        "script": str(script),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "argv": list(sys.argv),
        "args": argd,
        "git_commit": git_commit(repo),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
    }
    if extra:
        m.update(extra)
    return m


def write(out, script, args=None, extra=None, repo=None, filename="manifest.json"):
    """Write the provenance manifest as JSON. `out` is a file path ending in .json (written as-is)
    or a directory (manifest written to <out>/<filename>). Returns the Path written (or None on
    failure — provenance never raises into the caller)."""
    try:
        m = build(script, args=args, extra=extra, repo=repo)
        out = Path(out)
        if out.suffix.lower() == ".json":
            path = out
        else:
            path = out / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(m, f, indent=2, default=str)
        return path
    except Exception as e:  # provenance is best-effort; never break the run over it
        print(f"[provenance] WARNING: could not write manifest to {out}: {e}")
        return None
