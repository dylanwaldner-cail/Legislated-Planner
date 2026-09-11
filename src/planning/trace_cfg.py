"""OPT-IN provenance tracing. All default OFF, so every existing run is byte-identical.

Four things the ledger cannot currently express, each gated separately:
  DINOWM_TRACE_SUBFRAME=1  sub-frame cube xy + YAW per internal sim step (stride
                           DINOWM_TRACE_STRIDE, default 1). Boundary states hold two endpoints
                           per stroke, so `swept` linearises the path and the axis-aligned body
                           model has no yaw to read -- this records both.
  DINOWM_TRACE_TREE=1      every RRT node (pos, cell, parent, cumulative violations).
  DINOWM_TRACE_CAND=1      per EXTEND, the winner under the live rule AND the winner ignoring
                           legality -- enough to re-rank offline at another deviant_lambda or
                           under another pruning rule, without the 64x cost of every candidate.
  DINOWM_TRACE=1           turns on all three.

Sizes measured on a 120-batch sign_change run: subframe ~32 MB, tree ~42 MB, cand ~300 MB.
"""
from __future__ import annotations

import os


def _on(name: str) -> bool:
    return bool(int(os.environ.get(name, "0") or "0")) or \
           bool(int(os.environ.get("DINOWM_TRACE", "0") or "0"))


def trace_subframe() -> bool:
    return _on("DINOWM_TRACE_SUBFRAME")


def trace_tree() -> bool:
    return _on("DINOWM_TRACE_TREE")


def trace_candidates() -> bool:
    return _on("DINOWM_TRACE_CAND")


def trace_stride() -> int:
    return max(1, int(os.environ.get("DINOWM_TRACE_STRIDE", "1") or "1"))
