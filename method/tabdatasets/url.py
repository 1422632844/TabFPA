"""url dataset module: integer-linear + implication + capacity constraints,
all enforced by construction. Moved verbatim from exp_url.py (g1 fix:
x1 <= x0, i.e. length_hostname <= length_url)."""
from __future__ import annotations

import numpy as np
import torch

EPS_FLOAT = 1e-6
IMPLS = [(21, 3), (23, 13), (19, 25), (2, 25), (28, 25), (31, 26)]
SUM2 = [3] + list(range(4, 18))
SUM5 = [(20, 3.0), (21, 4.0), (23, 2.0)]
LOWER = [(20, 3.0), (21, 4.0), (2, 4.0), (23, 2.0)]  # x0 >= w*xi - 1


class UrlFeasibleMap:
    """x' = F(x, delta): boxes + integrality + all url relations hold.
    feasible() is the exact per-step guard (capacity vs x0 box)."""

    def __init__(self, meta, device):
        self.device = device
        self.names = meta["feature"].tolist()
        self.nf = len(self.names)
        self.nf_free = int(np.asarray(meta["mutable"].to_numpy(dtype=bool) &
                                      (np.asarray(meta["type"] != "cat"))
                                      ).sum())
        self.types = meta["type"].tolist()
        self.mutable = meta["mutable"].astype(bool).tolist()
        self.lo = torch.tensor(meta["min"].to_numpy(dtype=np.float32),
                               device=device)
        self.hi = torch.tensor(meta["max"].to_numpy(dtype=np.float32),
                               device=device)
        rng = (self.hi - self.lo).clamp_min(1e-6)
        self.margin = EPS_FLOAT * rng
        self.half = 0.5 * rng
        self.is_int = torch.tensor([t != "real" for t in self.types],
                                   device=device)
        self.free_mask = torch.tensor(self.mutable, device=device) & ~torch.tensor(
            [t == "cat" for t in self.types], device=device)

    def __call__(self, x0: torch.Tensor, delta_free: torch.Tensor):
        x = x0.clone()
        d = delta_free * self.half[self.free_mask].unsqueeze(0)
        lo, hi = self.lo, self.hi
        v = x[:, self.free_mask] + d
        v = torch.max(torch.min(v, (hi - self.margin)[self.free_mask]),
                      (lo + self.margin)[self.free_mask])
        v = v + (torch.round(v) - v).detach() * self.is_int[self.free_mask]
        x[:, self.free_mask] = v
        # implications: b <= 0 -> a = 0 (ints, <=0 means ==0)
        for a, b in IMPLS:
            x[:, a] = torch.where(x[:, b] <= 0, torch.zeros_like(x[:, a]),
                                  x[:, a])
        # g12 (clone operands: min would save column views that later
        # in-place writes invalidate)
        x[:, 38] = torch.minimum(x[:, 38].clone(), x[:, 37].clone())
        # capacity lower bounds on x0 (integer-valued after rounding)
        need = torch.maximum(x[:, 1].clone(), x[:, 1] * 0.0 + 1.0)  # g1
        need = torch.maximum(need, x[:, SUM2].sum(dim=1) + 3.0 * x[:, 19])
        s5 = sum(w * x[:, i] for i, w in SUM5)
        need = torch.maximum(need, s5)
        for i, w in LOWER:
            need = torch.maximum(need, w * x[:, i] - 1.0)
        x[:, 0] = torch.maximum(x[:, 0].clone(), need)
        x[:, 0] = x[:, 0] + (torch.round(x[:, 0]) - x[:, 0]).detach()
        return x

    def feasible(self, x: torch.Tensor) -> torch.Tensor:
        ok = ((x >= self.lo - 1e-4) & (x <= self.hi + 1e-4)).all(dim=1)
        ok &= (x[:, self.is_int] == torch.round(x[:, self.is_int])).all(dim=1)
        ok &= x[:, 0] >= x[:, 1] - 1e-4
        ok &= x[:, SUM2].sum(dim=1) + 3.0 * x[:, 19] <= x[:, 0] + 1e-4
        ok &= sum(w * x[:, i] for i, w in SUM5) <= x[:, 0] + 1e-4
        for i, w in LOWER:
            ok &= w * x[:, i] <= x[:, 0] + 1.0 + 1e-4
        for a, b in IMPLS:
            ok &= ~((x[:, a] > 0) & (x[:, b] <= 0))
        ok &= x[:, 38] <= x[:, 37] + 1e-4
        return ok


def finalize_url_f64(x_clean64, x_opt32, meta):
    """float64 rebuild: ints rounded, relations re-enforced in float64."""
    out = np.array(x_clean64, dtype=np.float64, copy=True)
    o = np.asarray(x_opt32, dtype=np.float64)
    mut = meta["mutable"].to_numpy(dtype=bool)
    out[:, mut] = o[:, mut]
    is_int = np.array([t != "real" for t in meta["type"].tolist()])
    out[:, is_int] = np.round(out[:, is_int])
    for a, b in IMPLS:
        out[out[:, b] <= 0, a] = 0.0
    out[:, 38] = np.minimum(out[:, 38], out[:, 37])
    need = np.maximum.reduce([
        np.maximum(out[:, 1], 1.0),
        out[:, SUM2].sum(1) + 3.0 * out[:, 19],
        sum(w * out[:, i] for i, w in SUM5),
        *[w * out[:, i] - 1.0 for i, w in LOWER],
    ])
    out[:, 0] = np.maximum(out[:, 0], need)
    out[:, 0] = np.round(out[:, 0])
    lo = meta["min"].to_numpy(dtype=np.float64)
    hi = meta["max"].to_numpy(dtype=np.float64)
    for j in range(out.shape[1]):
        if not is_int[j]:
            m = 1e-9 * max(hi[j] - lo[j], 1.0)
            out[:, j] = np.clip(out[:, j], lo[j] + m, hi[j] - m)
    out[:, ~mut] = np.asarray(x_clean64)[:, ~mut]
    return out


JUDGE_KWARGS = {}
