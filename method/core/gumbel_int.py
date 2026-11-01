"""ST-Gumbel proposals over discrete grids of a FeasibleMap's free
features — the discrete half of the heterogeneous feasible-region
parameterization (the legacy story: cat + low-cardinality int columns
optimized via Gumbel over their value grid, real + high-cardinality int
via continuous delta; see legacy/data.py and docs/paper_draft.md §3).

2026-09-05 extension: CONTINUOUS free features can also adopt a Gumbel
grid (cont_levels > 0), defined directly in delta space — a uniform
linspace over [-cont_reach, +cont_reach] (delta units; one delta unit
costs ~0.5 in transform L2, so reach 1.0 spans the whole eps ball for a
single feature). Rationale: the step ablation showed the optimizer needs
only 100-200 steps, and Gumbel jumps traverse grids in one step —
discretizing everything lets the same few-step budget drive the whole
search ("度过得更快"). The QP direction is preserved as a logits prior
(bias toward the grid level nearest the QP delta) instead of being
discarded.

Integration contract (keeps feasibility strictly by construction):
  - For each adopted column the attacker holds logits over that column's
    grid. Int grids live in FEATURE space (sampled integer v maps to
    delta slot (v - x0)/half; clamp+round reproduces v exactly). Cont
    grids live in DELTA space (x0_term=0, half_term=1 so the sampled
    grid value IS the delta slot).
  - propose(tau) draws hard one-hots via straight-through Gumbel; the
    FeasibleMap's clamp/round/repairs still run unchanged downstream —
    the output remains feasible by construction whatever is sampled.
  - Gradients flow to the logits through the soft one-hot; non-adopted
    columns keep their ordinary PGD gradients.
"""
from __future__ import annotations

import numpy as np
import torch


def _free_slots(fmap, meta):
    """[(free_slot, name, x_column_index)] for the map's free features."""
    names = meta["feature"].tolist()
    if hasattr(fmap, "FREE"):                       # lcld-style named map
        return [(s, f, names.index(f)) for s, f in enumerate(fmap.FREE)]
    mask = fmap.free_mask.detach().cpu().numpy()    # mask-style map
    cols = np.nonzero(mask)[0]
    return [(s, names[c], int(c)) for s, c in enumerate(cols)]


class GumbelInt:
    """Logits over per-column grids for the adopted free columns; int
    columns whose grid exceeds max_card stay on BPDA (the legacy
    high-cardinality split). cont_levels > 0 additionally adopts the
    real-typed free columns on a delta-space grid."""

    def __init__(self, fmap, meta, x0, device, max_card=64,
                 cont_levels=0, cont_reach=1.0,
                 tau0=1.0, tau_min=0.1):
        self.device = device
        self.tau0, self.tau_min = tau0, tau_min
        types = meta["type"].tolist()
        mins = meta["min"].to_numpy(dtype=np.float64)
        maxs = meta["max"].to_numpy(dtype=np.float64)
        slots, grids, xcols, halves, x0_zero = [], [], [], [], []
        for slot, name, j in _free_slots(fmap, meta):
            if types[j] == "real":
                if cont_levels < 2:
                    continue
                grid = np.linspace(-cont_reach, cont_reach, cont_levels)
                half, zero_x0 = 1.0, True    # delta-space grid
            else:
                low, high = int(np.ceil(mins[j])), int(np.floor(maxs[j]))
                if not (1 <= high - low + 1 <= max_card):
                    continue
                grid = np.arange(low, high + 1)
                half, zero_x0 = 0.5 * max(maxs[j] - mins[j], 1e-6), False
            slots.append(slot)
            grids.append(grid)
            xcols.append(j)
            halves.append(half)
            x0_zero.append(zero_x0)
        self.cols = slots
        self.n = len(slots)
        if not self.n:
            self.theta = None
            return
        x0_np = x0.detach().cpu().numpy() if torch.is_tensor(x0) \
            else np.asarray(x0)
        # delta = (v - x0_term) / half ; cont grids pass v through as-is
        x0_terms = np.where(x0_zero, 0.0,
                            x0_np[:, xcols]).astype(np.float32)
        self.x0_cols = torch.as_tensor(x0_terms, device=device)
        self.half = torch.as_tensor(halves, dtype=torch.float32,
                                    device=device)
        pad = max(len(grid) for grid in grids)
        vals = np.zeros((self.n, pad), dtype=np.float64)  # pad slots are
        for i, grid in enumerate(grids):                 # softmax-masked
            vals[i, :len(grid)] = grid
        self.values = torch.as_tensor(vals, dtype=torch.float32,
                                      device=device)
        self.theta = torch.zeros(x0_np.shape[0], self.n, pad,
                                 device=device, requires_grad=True)
        valid = torch.zeros(self.n, pad, dtype=torch.bool, device=device)
        for i, grid in enumerate(grids):
            valid[i, :len(grid)] = True
        self.pad_mask = ~valid

    def subset(self, idx, bias=None):
        """Fresh-logits copy restricted to the active rows (restart
        compaction). bias: (n_act, nf_free) delta — restart-0 seeds get a
        logits prior toward the grid level nearest their QP/init delta
        (strength 4), so the continuous direction survives the discrete
        parameterization instead of being discarded."""
        out = GumbelInt.__new__(GumbelInt)
        out.device = self.device
        out.tau0, out.tau_min = self.tau0, self.tau_min
        out.cols, out.n = self.cols, self.n
        out.x0_cols = self.x0_cols[idx]
        out.half = self.half
        out.values = self.values
        out.pad_mask = self.pad_mask
        theta = torch.zeros(int(idx.shape[0]), self.n,
                            self.values.shape[1], device=self.device)
        if bias is not None:
            # target delta-slot values for the adopted columns
            tgt = torch.as_tensor(np.asarray(bias, dtype=np.float32),
                                  device=self.device)[:, self.cols]
            v_tgt = tgt * out.half + out.x0_cols      # back to grid space
            gap = (out.values.unsqueeze(0) - v_tgt.unsqueeze(-1)).abs()
            theta = -4.0 * gap    # pad slots are re-masked in propose()
        out.theta = theta.requires_grad_(True)
        return out

    def tau(self, step, steps):
        return float(self.tau0 + (self.tau_min - self.tau0)
                     * (step / max(steps - 1, 1)))

    def propose(self, tau):
        """(N, n) delta-slot values with straight-through gradients."""
        logits = self.theta.masked_fill(self.pad_mask.unsqueeze(0), -1e9)
        u = torch.rand(logits.shape, device=self.device).clamp(
            1e-12, 1.0 - 1e-7)
        e = -torch.log(u)              # Exp(1) sample, in [1e-7, 27.6]
        g = -torch.log(e.clamp_min(1e-12))
        y_soft = ((logits + g) / tau).softmax(dim=-1)
        idx = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
        oh = y_hard - y_soft.detach() + y_soft
        v = (oh * self.values.unsqueeze(0)).sum(dim=-1)
        return (v - self.x0_cols) / self.half
