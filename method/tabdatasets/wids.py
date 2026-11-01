"""wids dataset module: 31 pure order constraints Feature(i+1) <= Feature(i)
for i in {33, 35, ..., 93} (0-indexed over x columns) — the simplest
constraint family (min-repair pattern). Demonstrates the plugin pattern:
a new dataset needs only a map + finalize + judge kwargs."""
from __future__ import annotations

import numpy as np
import torch

EPS_FLOAT = 1e-6
# pairs (hi_idx, lo_idx): x[hi_idx] <= x[lo_idx]  (from samples/wids.py)
ORDER_PAIRS = [(i + 1, i) for i in range(33, 94, 2)]


class WidsFeasibleMap:
    """Free = mutable non-cat features. Forward: box clamp -> BPDA round ->
    min-repairs for every order pair (each pair is disjoint, single pass)."""

    def __init__(self, meta, device):
        self.device = device
        self.names = meta["feature"].tolist()
        mutable = meta["mutable"].to_numpy(dtype=bool)
        noncat = np.asarray(meta["type"] != "cat")
        self.free_mask = torch.tensor(mutable & noncat, device=device)
        self.nf_free = int((mutable & noncat).sum())
        self.lo = torch.tensor(meta["min"].to_numpy(dtype=np.float32),
                               device=device)
        self.hi = torch.tensor(meta["max"].to_numpy(dtype=np.float32),
                               device=device)
        rng = (self.hi - self.lo).clamp_min(1e-6)
        self.margin = EPS_FLOAT * rng
        self.half = 0.5 * rng
        is_int = np.asarray(meta["type"] != "real")
        self.is_int = torch.tensor(is_int, device=device)
        self.pairs = [(torch.tensor(a, device=device),
                       torch.tensor(b, device=device))
                      for a, b in ORDER_PAIRS]

    def __call__(self, x0: torch.Tensor, delta_free: torch.Tensor):
        x = x0.clone()
        d = delta_free * self.half[self.free_mask].unsqueeze(0)
        v = x[:, self.free_mask] + d
        v = torch.max(torch.min(v, (self.hi - self.margin)[self.free_mask]),
                      (self.lo + self.margin)[self.free_mask])
        v = v + (torch.round(v) - v).detach() * self.is_int[self.free_mask]
        x[:, self.free_mask] = v
        # order repairs AFTER rounding (pairs disjoint -> single pass)
        for a, b in self.pairs:
            x[:, a] = torch.minimum(x[:, a].clone(), x[:, b].clone())
        return x


def finalize_wids_f64(x_clean64, x_opt32, meta):
    out = np.array(x_clean64, dtype=np.float64, copy=True)
    o = np.asarray(x_opt32, dtype=np.float64)
    mut = meta["mutable"].to_numpy(dtype=bool)
    out[:, mut] = o[:, mut]
    is_int = np.array([t != "real" for t in meta["type"].tolist()])
    out[:, is_int] = np.round(out[:, is_int])
    lo = meta["min"].to_numpy(dtype=np.float64)
    hi = meta["max"].to_numpy(dtype=np.float64)
    for j in range(out.shape[1]):
        if not is_int[j]:
            m = 1e-9 * max(hi[j] - lo[j], 1.0)
            out[:, j] = np.clip(out[:, j], lo[j] + m, hi[j] - m)
    # repairs AFTER clipping (clipping at box edges must not break orders)
    for a, b in ORDER_PAIRS:
        out[:, a] = np.minimum(out[:, a], out[:, b])
    out[:, ~mut] = np.asarray(x_clean64)[:, ~mut]
    return out


# wids has 108 x-columns: the official checker widens the box tolerance by
# 1000x for exactly this dataset (constraints_checker.py hard-codes it)
JUDGE_KWARGS = {"box_tol_scale": 1000}
