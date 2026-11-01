"""WiDS-only continuous completion with fixed discrete coordinates.

Nearest point in scaled continuous coordinates, subject to disjoint order
constraints, boxes and the residual L2 ball. This is a convex QCQP, not an
attack-direction QP. The forward solver uses a scalar dual bisection and exact
box/pair projections. The torch bridge uses BPDA, NOT implicit differentiation.
"""
import numpy as np
import torch
from tabdatasets.wids import ORDER_PAIRS


class WidsCompletion:
    def __init__(self, meta, scaler, eps):
        self.eps = eps
        self.types = np.asarray(meta["type"])
        self.real = np.flatnonzero(self.types == "real")
        self.discrete = np.flatnonzero(self.types != "real")
        self.numeric = np.asarray(scaler.num_idx)
        self.lo = meta["min"].to_numpy(dtype=float)
        self.hi = meta["max"].to_numpy(dtype=float)
        self.scale = np.ones(len(meta))
        self.scale[self.numeric] = (scaler.x_max - scaler.x_min).cpu().numpy()
        self.scale = np.maximum(self.scale, 1e-12)
        pos = {j: k for k, j in enumerate(self.real)}
        self.rr = [(pos[a], pos[b]) for a,b in ORDER_PAIRS
                   if a in pos and b in pos]
        self.rd = [(pos[a],b) for a,b in ORDER_PAIRS if a in pos and b not in pos]
        self.dr = [(a,pos[b]) for a,b in ORDER_PAIRS if a not in pos and b in pos]
        self.dd = [(a,b) for a,b in ORDER_PAIRS if a not in pos and b not in pos]
        self.calls = self.rows = self.rejected = 0

    def _poly(self, target, lower, upper):
        """Projection in normalized coordinates onto disjoint boxes/orders."""
        rr = self.real
        r = self.scale[rr]
        out = np.clip(target, lower, upper)
        if self.rr:
            a,b = np.asarray(self.rr).T
            av = out[:,a]*r[a]
            bv = out[:,b]*r[b]
            bad = av > bv
            # On the active equality, both raw coordinates equal t.
            common = (target[:,a]/r[a] + target[:,b]/r[b]) / (
                      1/r[a]**2 + 1/r[b]**2)
            common = np.maximum(common, np.maximum(lower[:,a]*r[a],lower[:,b]*r[b]))
            common = np.minimum(common, np.minimum(upper[:,a]*r[a],upper[:,b]*r[b]))
            out[:,a] = np.where(bad, common/r[a], out[:,a])
            out[:,b] = np.where(bad, common/r[b], out[:,b])
        return out

    def solve(self, clean, proposed):
        x0 = np.asarray(clean, dtype=np.float64)
        p = np.asarray(proposed, dtype=np.float64)
        n = len(p); rr = self.real; sr = self.scale[rr]
        lower = np.broadcast_to(self.lo[rr]/sr,(n,len(rr))).copy()
        upper = np.broadcast_to(self.hi[rr]/sr,(n,len(rr))).copy()
        for a,b in self.rd:
            upper[:,a] = np.minimum(upper[:,a],p[:,b]/sr[a])
        for a,b in self.dr:
            lower[:,b] = np.maximum(lower[:,b],p[:,a]/sr[b])
        disc = self.discrete
        valid = np.isfinite(p).all(1)
        valid &= ((p[:,disc] >= self.lo[disc]) &
                  (p[:,disc] <= self.hi[disc]) &
                  (p[:,disc] == np.round(p[:,disc]))).all(1)
        # L2=0.5 freezes categories under the fitted full one-hot encoding.
        cats = self.types == "cat"
        valid &= (p[:,cats] == x0[:,cats]).all(1)
        for a,b in self.dd:
            valid &= p[:,a] <= p[:,b]
        valid &= (lower <= upper).all(1)
        for a,b in self.rr:
            valid &= lower[:,a]*sr[a] <= upper[:,b]*sr[b]
        ints = np.flatnonzero(self.types == "int")
        discrete_cost = (((p[:,ints]-x0[:,ints])/self.scale[ints])**2).sum(1)
        radius2 = self.eps**2 - discrete_cost
        centre = x0[:,rr]/sr
        desired = p[:,rr]/sr
        min_c = self._poly(centre,lower,upper)
        min_cost = ((min_c-centre)**2).sum(1)
        valid &= min_cost <= radius2 + 1e-13
        sol = self._poly(desired,lower,upper)
        over = ((sol-centre)**2).sum(1) > radius2
        active = valid & over
        if active.any():
            idx = np.flatnonzero(active)
            left = np.zeros(len(idx)); right = np.ones(len(idx))
            for _ in range(36):
                alpha = (left+right)*.5
                target = centre[idx] + alpha[:,None]*(desired[idx]-centre[idx])
                q = self._poly(target,lower[idx],upper[idx])
                inside = ((q-centre[idx])**2).sum(1) <= radius2[idx]
                left = np.where(inside,alpha,left)
                right = np.where(inside,right,alpha)
            sol[idx] = self._poly(centre[idx]+left[:,None]*(desired[idx]-centre[idx]),
                                  lower[idx],upper[idx])
        out = p.copy()
        out[:,rr] = sol*sr
        # Reconcile only floating arithmetic of real-real equalities.
        for a,b in ORDER_PAIRS:
            if self.types[a] == "real":
                out[:,a] = np.minimum(out[:,a],out[:,b])
        valid &= (((out[:,self.numeric]-x0[:,self.numeric]) /
                   self.scale[self.numeric])**2).sum(1) <= self.eps**2+1e-10
        out[~valid] = x0[~valid]
        self.calls += 1; self.rows += n; self.rejected += int((~valid).sum())
        return out, valid

    def torch(self, clean, proposed):
        out, valid = self.solve(clean.detach().cpu().numpy(),
                                proposed.detach().cpu().numpy())
        hard = torch.as_tensor(out,device=proposed.device,dtype=proposed.dtype)
        # Discrete decision stays fixed in every feasible completion.
        return proposed + (hard-proposed).detach(), torch.as_tensor(
            valid,device=proposed.device)
