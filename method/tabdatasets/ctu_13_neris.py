"""ctu_13_neris dataset module (758 features; 432 mutable, 1 immutable cat).

Constraints (tabularbench/datasets/samples/ctu_13_neris.py):
  g1/g2 : sum(proto families, side) == sum(bytes families, side), side in {s,d}
          -> equality-substitution: one MUTABLE member of the LHS proto
          families becomes the derived slack (bytes_in_* are immutable).
  packet: bytes_out_sum_{e}_{p} / pkts_out_sum_{e}_{p} <= 1500 (SafeDivision:
          auto-true when pkts == 0) -> conditional box, applied only where
          pkts > 0; e in {s,d}, last family member (OTHER aggregate) excluded.
  orders: {bytes_out,pkts_out,duration}_{max,min}_{e}_{p} <= ..._sum_{e}_{p}
          and min <= max, for 18 ports x 2 directions -> min-repairs.

Feature membership and fold order replicate the constraint construction
(startswith over x-column order, left fold) so the exact per-step guard
evaluates the same arithmetic the checker will apply.
"""
from __future__ import annotations

import numpy as np
import torch

EPS_FLOAT = 1e-6
PROTO_FAMS = ["icmp_sum_{e}_", "udp_sum_{e}_", "tcp_sum_{e}_"]
BYTES_FAMS = ["bytes_in_sum_{e}_", "bytes_out_sum_{e}_"]
PKTS_FAMS = ["pkts_out_sum_{e}_"]
ORDER_FAMS = ["bytes_out", "pkts_out", "duration"]
PORTS = ["1", "3", "8", "10", "21", "22", "25", "53", "80", "110", "123",
         "135", "138", "161", "443", "445", "993", "OTHER"]


def _family(names, prefix):
    return [i for i, n in enumerate(names) if n.startswith(prefix)]


class CtuFeasibleMap:
    def __init__(self, meta, device):
        self.device = device
        self.names = meta["feature"].tolist()
        names = self.names
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

        # equality sides (side s and d), slack = first mutable proto member
        self.eq = []
        for side in ("s", "d"):
            lhs = [i for f in PROTO_FAMS
                   for i in _family(names, f.format(e=side))]
            rhs = [i for f in BYTES_FAMS
                   for i in _family(names, f.format(e=side))]
            slack = next((i for i in lhs if mutable[i]), None)
            self.eq.append(dict(lhs=lhs, rhs=rhs, slack=slack))
        # conditional boxes: bytes_out <= 1500 * pkts_out where pkts > 0
        self.pk_pairs = []
        for side in ("s", "d"):
            bo = _family(names, f"bytes_out_sum_{side}_")
            po = _family(names, f"pkts_out_sum_{side}_")
            for b, p in list(zip(bo, po))[:-1]:      # exclude last (OTHER)
                self.pk_pairs.append((b, p))
        # order triples: (max, sum), (min, sum), (min, max)
        self.order_pairs = []
        for fam in ORDER_FAMS:
            for port in PORTS:
                for side in ("d", "s"):
                    mx = names.index(f"{fam}_max_{side}_{port}")
                    mn = names.index(f"{fam}_min_{side}_{port}")
                    sm = names.index(f"{fam}_sum_{side}_{port}")
                    self.order_pairs += [(mx, sm), (mn, sm), (mn, mx)]

    def __call__(self, x0: torch.Tensor, delta_free: torch.Tensor):
        x = x0.clone()
        d = delta_free * self.half[self.free_mask].unsqueeze(0)
        v = x[:, self.free_mask] + d
        v = torch.max(torch.min(v, (self.hi - self.margin)[self.free_mask]),
                      (self.lo + self.margin)[self.free_mask])
        v = v + (torch.round(v) - v).detach() * self.is_int[self.free_mask]
        x[:, self.free_mask] = v

        # conditional packet-size boxes (after rounding pkts)
        for b, p in self.pk_pairs:
            cap = 1500.0 * x[:, p]
            x[:, b] = torch.where(x[:, p] > 0,
                                  torch.minimum(x[:, b].clone(), cap),
                                  x[:, b])
        # order repairs
        for a, b in self.order_pairs:
            x[:, a] = torch.minimum(x[:, a].clone(), x[:, b].clone())

        # equality: slack absorbs (float64 internally, exact for int values)
        for eq in self.eq:
            if eq["slack"] is None:
                continue
            lhs_rest = [i for i in eq["lhs"] if i != eq["slack"]]
            with torch.no_grad():
                lhs_sum = x[:, lhs_rest].double().sum(dim=1)
                rhs_sum = x[:, eq["rhs"]].double().sum(dim=1)
                sl = rhs_sum - lhs_sum
                lo_s = self.lo[eq["slack"]].double() + 1e-3
                hi_s = self.hi[eq["slack"]].double() - 1e-3
                sl = torch.clamp(sl, lo_s, hi_s)
                x[:, eq["slack"]] = sl.float()
        return x

    def feasible(self, x: torch.Tensor) -> torch.Tensor:
        """Exact per-step guard (float32 folds replicate checker arithmetic)."""
        ok = ((x >= self.lo - 1e-4) & (x <= self.hi + 1e-4)).all(dim=1)
        ok &= (x[:, self.is_int] == torch.round(x[:, self.is_int])).all(dim=1)
        for eq in self.eq:
            lhs = x[:, eq["lhs"]].sum(dim=1)       # f32 fold, any-order ~ judge
            rhs = x[:, eq["rhs"]].sum(dim=1)
            ok &= lhs <= rhs + 5e-4                # slack clamped rows dropped
            ok &= rhs <= lhs + 5e-4
        for b, p in self.pk_pairs:
            bad = (x[:, p] > 0) & (x[:, b] > 1500.0 * x[:, p] + 1e-4)
            ok &= ~bad
        for a, b in self.order_pairs:
            ok &= x[:, a] <= x[:, b] + 1e-4
        return ok


def finalize_ctu_f64(x_clean64, x_opt32, meta):
    """float64 rebuild; slack recomputed exactly in float64."""
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

    fmap = CtuFeasibleMap(meta, "cpu")
    for b, p in fmap.pk_pairs:
        cap = 1500.0 * out[:, p]
        out[:, b] = np.where(out[:, p] > 0, np.minimum(out[:, b], cap),
                             out[:, b])
    for a, b in fmap.order_pairs:
        out[:, a] = np.minimum(out[:, a], out[:, b])
    for eq in fmap.eq:
        if eq["slack"] is None:
            continue
        lhs_rest = [i for i in eq["lhs"] if i != eq["slack"]]
        sl = out[:, eq["rhs"]].sum(1) - out[:, lhs_rest].sum(1)
        lo_s = float(fmap.lo[eq["slack"]]) + 1e-3
        hi_s = float(fmap.hi[eq["slack"]]) - 1e-3
        out[:, eq["slack"]] = np.clip(sl, lo_s, hi_s)
        out[:, eq["slack"]] = np.round(out[:, eq["slack"]]) \
            if is_int[eq["slack"]] else out[:, eq["slack"]]
    return out


JUDGE_KWARGS = {}
