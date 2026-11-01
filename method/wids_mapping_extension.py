"""Discrete diagnosis grid and completion for the WiDS-mapping scenario."""
import numpy as np
import torch
from core.data import DEVICE
from core.feasible_integer_moves import JointIntegerMoves
from core.wids_completion import WidsCompletion
from core.wids_mapping_constraint import DIAG, SYSTEM, load_intervals
from core.domain_constants import ORDER_PAIRS


class DiagGrid:
    """Per-row ST-Gumbel grid over (legal) diagnosis codes within eps."""

    def __init__(self, x, meta, scaler, eps, arm="constraint"):
        names = meta["feature"].tolist()
        self.dcol = names.index(DIAG)
        self.scol = names.index(SYSTEM)
        num_idx = list(scaler.num_idx)
        k = num_idx.index(self.dcol)
        self.scale = float(scaler.x_max.numpy().ravel()[k]
                           - scaler.x_min.numpy().ravel()[k])
        intervals = load_intervals()
        codes_by_sys = {}
        for s, runs in intervals.items():
            cc = []
            for lo, hi in runs:
                cc.extend(round(float(v), 2)
                          for v in np.arange(lo, hi + 0.005, 0.01))
            codes_by_sys[s] = np.array(cc)
        all_codes = np.unique(np.concatenate(list(codes_by_sys.values())))

        n = len(x)
        grids, s0 = [], x[:, self.scol].astype(int)
        for r in range(n):
            pool = codes_by_sys[int(s0[r])] if arm == "constraint" else all_codes
            cost = np.abs(pool - x[r, self.dcol]) / self.scale
            grids.append(pool[cost <= eps * 0.999])
        kmax = max(len(g) for g in grids)
        self.values = np.full((n, kmax), x[:, self.dcol][:, None])
        self.mask = np.zeros((n, kmax), dtype=bool)
        for r, g in enumerate(grids):
            self.values[r, :len(g)] = g
            self.mask[r, :len(g)] = True
        # every row keeps its clean code (cost 0) in the grid
        assert all(np.isclose(x[r, self.dcol], self.values[r]).any()
                   for r in range(n))
        cost = np.abs(self.values - x[:, self.dcol][:, None]) / self.scale
        prior = np.where(self.mask, -40.0 * cost + 2.0 *
                         (np.isclose(self.values, x[:, self.dcol][:, None])), -1e9)
        self.theta = torch.tensor(prior, dtype=torch.float32, device=DEVICE,
                                  requires_grad=True)

    def sample(self, tau):
        logits = self.theta.masked_fill(
            ~torch.as_tensor(self.mask, device=DEVICE), -1e9)
        u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
        g = -torch.log(-torch.log(u))
        soft = ((logits + g) / tau).softmax(-1)
        idx = soft.argmax(-1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
        w = hard - soft.detach() + soft
        return (w * torch.as_tensor(self.values, dtype=torch.float32,
                                    device=DEVICE)).sum(-1)


class WidsMappingMoves(JointIntegerMoves):
    """Base joint integer moves with the diagnosis column carved out of the
    continuous coordinates (it is a grid action now)."""

    def __init__(self, meta, scaler, cons_affine, dcol, eps=.594):
        super().__init__(meta, scaler, cons_affine, eps)
        self.dcol = dcol
        self.reals = self.reals[self.reals != dcol]
        assert self.dcol not in self.ints
        pair_cols = {c for a, b in ORDER_PAIRS for c in (a, b)}
        assert self.dcol not in pair_cols


class WidsMappingCompletion(WidsCompletion):
    """WiDS completion with the diagnosis column FIXED at the proposed code
    (charged to the budget; legality checked against the empirical map)."""

    def __init__(self, meta, scaler, eps, dcol, scol):
        super().__init__(meta, scaler, eps)
        self.dcol, self.scol = dcol, scol
        assert dcol in self.real
        self.real = self.real[self.real != dcol]
        self.fixed = np.sort(np.append(self.discrete, dcol))
        # recompute the pair position maps for the FILTERED real list
        pos = {j: k for k, j in enumerate(self.real)}
        self.rr = [(pos[a], pos[b]) for a, b in ORDER_PAIRS
                   if a in pos and b in pos]
        self.rd = [(pos[a], b) for a, b in ORDER_PAIRS if a in pos and b not in pos]
        self.dr = [(a, pos[b]) for a, b in ORDER_PAIRS if a not in pos and b in pos]
        self.dd = [(a, b) for a, b in ORDER_PAIRS if a not in pos and b not in pos]
        intervals = load_intervals()
        pairs = sorted((round(float(v), 2), s)
                       for s, runs in intervals.items()
                       for lo, hi in runs
                       for v in np.arange(lo, hi + 0.005, 0.01))
        self.codes_all = np.array([c for c, _ in pairs])
        self.sys_all = np.array([s for _, s in pairs])
        assert len(np.unique(self.codes_all)) == len(self.codes_all)

    def solve(self, clean, proposed):
        x0 = np.asarray(clean, dtype=np.float64)
        p = np.asarray(proposed, dtype=np.float64)
        n = len(p)
        rr = self.real
        sr = self.scale[rr]
        lower = np.broadcast_to(self.lo[rr] / sr, (n, len(rr))).copy()
        upper = np.broadcast_to(self.hi[rr] / sr, (n, len(rr))).copy()
        for a, b in self.rd:
            upper[:, a] = np.minimum(upper[:, a], p[:, b] / sr[a])
        for a, b in self.dr:
            lower[:, b] = np.maximum(lower[:, b], p[:, a] / sr[b])
        disc = self.fixed
        valid = np.isfinite(p).all(1)
        valid &= ((p[:, disc] >= self.lo[disc]) &
                  (p[:, disc] <= self.hi[disc])).all(1)
        ints = np.flatnonzero(self.types == "int")
        valid &= (p[:, ints] == np.round(p[:, ints])).all(1)
        # diagnosis: exact legal code of the row's clean body system
        dv = p[:, self.dcol].round(2)
        pos = np.clip(np.searchsorted(self.codes_all, dv), 0,
                      len(self.codes_all) - 1)
        valid &= (self.codes_all[pos] == dv) & \
                 (self.sys_all[pos] == x0[:, self.scol].astype(int))
        cats = self.types == "cat"
        valid &= (p[:, cats] == x0[:, cats]).all(1)
        for a, b in self.dd:
            valid &= p[:, a] <= p[:, b]
        valid &= (lower <= upper).all(1)
        for a, b in self.rr:
            valid &= lower[:, a] * sr[a] <= upper[:, b] * sr[b]
        discrete_cost = (((p[:, ints] - x0[:, ints]) /
                          self.scale[ints]) ** 2).sum(1)
        discrete_cost += ((p[:, self.dcol] - x0[:, self.dcol]) /
                          self.scale[self.dcol]) ** 2
        radius2 = self.eps ** 2 - discrete_cost
        centre = x0[:, rr] / sr
        desired = p[:, rr] / sr
        min_c = self._poly(centre, lower, upper)
        min_cost = ((min_c - centre) ** 2).sum(1)
        valid &= min_cost <= radius2 + 1e-13
        sol = self._poly(desired, lower, upper)
        over = ((sol - centre) ** 2).sum(1) > radius2
        active = valid & over
        if active.any():
            idx = np.flatnonzero(active)
            left = np.zeros(len(idx))
            right = np.ones(len(idx))
            for _ in range(36):
                alpha = (left + right) * .5
                target = centre[idx] + alpha[:, None] * (desired[idx] - centre[idx])
                q = self._poly(target, lower[idx], upper[idx])
                inside = ((q - centre[idx]) ** 2).sum(1) <= radius2[idx]
                left = np.where(inside, alpha, left)
                right = np.where(inside, alpha, right)
            sol[idx] = self._poly(centre[idx] + left[:, None] *
                                  (desired[idx] - centre[idx]),
                                  lower[idx], upper[idx])
        out = p.copy()
        out[:, rr] = sol * sr
        for a, b in ORDER_PAIRS:
            if self.types[a] == "real" and a != self.dcol:
                out[:, a] = np.minimum(out[:, a], out[:, b])
        valid &= (((out[:, self.numeric] - x0[:, self.numeric]) /
                   self.scale[self.numeric]) ** 2).sum(1) <= self.eps ** 2 + 1e-10
        out[~valid] = x0[~valid]
        self.calls += 1
        self.rows += n
        self.rejected += int((~valid).sum())
        return out, valid
