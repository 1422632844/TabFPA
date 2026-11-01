"""WiDS apache_3j mapping-constrained attack: the dependent-order Gumbel
first instance (gap G8). The diagnosis column leaves the continuous
coordinates and becomes a per-row ST-Gumbel grid over LEGAL codes:

  constraint arm: codes of the row's own body system (fiber of the frozen
                  clean bodysystem) within the eps ball;
  control arm:    all observed codes within the eps ball (same machinery,
                  only the legal set differs — isolates the constraint's
                  effect on our own method).

Every candidate the optimizer can express satisfies the mapping exactly;
the shared judge (with the injected constraint) verifies, not assumes.

Loop skeleton = scripts/feasible_gumbel_ablation.run (released protocol),
with: eps parametrized (0.6), reals exclude the diagnosis column, one
extra theta (code logits), completion keeps the chosen code fixed and
charges its jump to the budget before the residual-ball solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TABULARBENCH_ROOT", str(_REPO_ROOT / "tabularbench"))
os.environ.setdefault("TABULARBENCH_DATASET_DIR", str(_REPO_ROOT / "tabularbench/datasets"))
os.environ.setdefault("TABULARBENCH_TARGET_MODEL_DIR", str(_REPO_ROOT / "tabularbench/target_model"))
if str(_REPO_ROOT / "tabularbench") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tabularbench"))

from core.data import select_subset, setup_components
from core.feasible_integer_moves import JointIntegerMoves
from core.wids_completion import WidsCompletion
from core.wids_mapping_constraint import (
    DIAG, SYSTEM, extended_relation_constraints, load_intervals,
)
from core.judge import BatchJudge
from models import load_victim
from scripts.feasible_gumbel_ablation import AttackLogits
from scripts.wids_gumbel_ablation import DEVICE, Meter
from tabdatasets.wids import ORDER_PAIRS
from tabularbench.constraints.constraints_checker import ConstraintChecker

ROOT = Path(__file__).resolve().parent


class PlainMoves(JointIntegerMoves):
    """arm='continuous': the original released parameterization — diagnosis
    stays a free continuous real (no grid). Unconstrained reference only:
    the mapping constraint is NOT judged/enforced in this arm."""


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


def run_mapping(model_name, meta, scaler, cons_native, model, adapter, judge,
                x, seed, eps=0.6, steps=200, logits_lr=0.12, arm="constraint"):
    import copy
    cons_ext = copy.copy(cons_native)
    if arm == "constraint":
        cons_ext.relation_constraints = extended_relation_constraints(
            cons_native)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    n = len(x)
    current = x.copy()
    names = meta["feature"].tolist()
    dcol, scol = names.index(DIAG), names.index(SYSTEM)
    orders = SimpleNamespace(relation_constraints=list(
        cons_native.relation_constraints))
    grid = arm in ("constraint", "control")
    if grid:
        moves = WidsMappingMoves(meta, scaler, orders, dcol, eps * .99)
        diag = DiagGrid(x, meta, scaler, eps, arm)
    else:
        moves = PlainMoves(meta, scaler, orders, eps * .99)
        diag = None
    x0 = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    scale = torch.tensor(moves.scale, dtype=torch.float32, device=DEVICE)
    values, mask = moves.candidates(x, current)
    prior = -40 * np.abs(values) * np.sqrt(moves.a)[None, :, None]
    theta = torch.tensor(np.concatenate(
        [np.full((n, 1), 2.), prior.reshape(n, -1)], axis=1),
        dtype=torch.float32, device=DEVICE, requires_grad=True)
    d = torch.zeros((n, len(moves.reals)), device=DEVICE, requires_grad=True)
    params = [{"params": [d], "lr": .06},
              {"params": [theta], "lr": logits_lr}]
    if grid:
        params.append({"params": [diag.theta], "lr": logits_lr})
    opt = torch.optim.Adam(params)
    y = np.ones(n, dtype=np.int64)
    scoring = AttackLogits(adapter, model_name)
    meter = Meter(adapter, model, judge, x, y, DEVICE, 256, 128)
    completion = WidsMappingCompletion(meta, scaler, eps * .99, dcol, scol) \
        if grid else WidsCompletion(meta, scaler, eps * .99)
    A = np.concatenate([moves.L, moves.E], axis=0)
    bias = np.concatenate([moves.lb, moves.eb])
    TA = torch.tensor(A, dtype=torch.float32, device=DEVICE)
    TB = torch.tensor(bias, dtype=torch.float32, device=DEVICE)
    norm = (TA.abs() * scale[None]).max(1).values.clamp_min(1e-9)
    iseq = torch.arange(len(A), device=DEVICE) >= len(moves.L)
    lo_t = torch.tensor(moves.lo[moves.reals], dtype=torch.float32, device=DEVICE)
    hi_t = torch.tensor(moves.hi[moves.reals], dtype=torch.float32, device=DEVICE)
    num_idx_t = torch.as_tensor(list(scaler.num_idx), device=DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        logits = meter.score(x0, np.arange(n))
        best_margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    for step in range(steps):
        ids = np.flatnonzero(~meter.ok & (meter.fwd < 254))
        if not len(ids):
            break
        requested = current.copy()
        requested[:, moves.reals] = x[:, moves.reals]
        anchor, possible = completion.solve(x, requested)
        assert possible.all(), "Minimum-cost continuous anchor failed"
        values, mask = moves.candidates(x, anchor)
        tau = 1. - .9 * step / max(steps - 1, 1)
        integer_point, straight, counts = moves.sample(anchor, values, mask,
                                                       theta, tau)
        if grid:
            code = diag.sample(tau)
            integer_point[:, dcol] = code.detach().cpu().numpy()
        valid = judge.constraint_ok(x[ids], integer_point[ids])
        cost = (((integer_point[ids] - x[ids]) / moves.scale) ** 2).sum(1)
        assert valid.all() and (cost <= (eps * .99) ** 2 + 1e-10).all(), \
            "Joint move + code invalid before completion"
        raw = torch.tensor(current, dtype=torch.float32, device=DEVICE)
        raw[:, moves.ints] = straight
        raw[:, moves.reals] = (x0[:, moves.reals] + d * scale[moves.reals]).clamp(
            lo_t, hi_t)
        proposal = integer_point[ids].copy()
        proposal[:, moves.reals] = raw[ids][:, moves.reals].detach().cpu().numpy()
        if grid:
            raw[:, dcol] = code
            proposal[:, dcol] = code.detach().cpu().numpy()[ids]
        sol, allowed = completion.solve(x[ids], proposal)
        ii = ids[allowed]
        if not len(ii):
            continue
        pp = sol[allowed]
        assert judge.constraint_ok(x[ii], pp).all()
        hard = torch.tensor(pp, dtype=torch.float32, device=DEVICE)
        candidate = raw[ii] + (hard - raw[ii]).detach()
        logits = meter.score(candidate, ii)
        meter.bwd[ii] += 1
        attack = (logits[:, 1] - logits[:, 0]).mean()
        dz = (raw[ii] - x0[ii]) / scale
        budget = torch.relu(dz[:, num_idx_t].norm(dim=1) - .9 * eps
                            ).square().mean()
        residual = (raw[ii] @ TA.T + TB) / norm
        relation = torch.where(iseq[None], residual,
                               torch.relu(residual)).square().sum(1).mean()
        loss = attack + 4 * budget + 4 * relation
        opt.zero_grad()
        loss.backward()
        assert theta.grad is not None
        if grid:
            assert diag.theta.grad is not None \
                and torch.isfinite(diag.theta.grad).all()
        opt.step()
        with torch.no_grad():
            meter.accept(pp, ii, logits, np.ones(len(ii), bool))
            margins = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            improve = margins < best_margin[ii]
            current[ii[improve]] = pp[improve]
            best_margin[ii[improve]] = margins[improve]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    ok, dist, _, _ = judge(x, y, meter.out)
    np.testing.assert_array_equal(ok, meter.ok)
    assert ConstraintChecker(cons_ext, tolerance=0.).check_constraints(
        x, meter.out).all()
    assert (dist <= eps).all() and meter.fwd.max() <= 256
    scoring.close()
    row = dict(arm=arm, model=model_name, seed=seed, n=n,
               accepted=int(ok.sum()), asr=float(ok.mean()),
               forward_mean=float(meter.fwd.mean()),
               forward_max=int(meter.fwd.max()),
               seconds=seconds, distance_max=float(dist.max()),
               completion_rejected=completion.rejected,
               diag_grid_max=(int(diag.mask.sum(1).max()) if grid else 0),
               diag_changed=int((meter.out[:, dcol] != x[:, dcol]).sum()),
               diag_farband=(int((((meter.out[:, dcol] - x[:, dcol]).__abs__()
                                  / diag.scale) > .05).sum()) if grid else 0))
    return meter, row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["constraint", "control", "continuous"],
                    default="constraint")
    ap.add_argument("--models", nargs="+",
                    default=["tabnet_default", "torchrln_default",
                             "tabtransformer_default", "stg_default",
                             "vime_default"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wids_mapping")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        ds, x_test, y_test, scaler, model_cpu, cons_native = \
            setup_components("wids", model_name, "cpu")
        x_pool, y_pool = select_subset(x_test, y_test, cons_native, model_cpu,
                                       max(args.n, 8), critical=True)
        x_pool = np.asarray(x_pool)[:args.n]
        _, _, _, _, _, _, adapter = load_victim("wids", model_name, DEVICE)
        import copy
        cons_judge = copy.copy(cons_native)
        if args.arm == "constraint":
            cons_judge.relation_constraints = extended_relation_constraints(
                cons_native)
        judge = BatchJudge(model_cpu, cons_judge, scaler, args.eps,
                           box_tol_scale=1000, rel_tol=0.)
        meta = ds.get_metadata(only_x=True).reset_index(drop=True)
        for seed in args.seeds:
            meter, row = run_mapping(model_name, meta, scaler, cons_native,
                                     model_cpu, adapter, judge, x_pool, seed,
                                     eps=args.eps, steps=args.steps,
                                     arm=args.arm)
            rows.append(row)
            print(json.dumps(row), flush=True)
            key = f"{args.arm}__{model_name}__s{seed}"
            np.savez_compressed(args.out / f"{key}.npz",
                                adv=meter.out, ok=meter.ok, clean=x_pool,
                                fwd=meter.fwd, bwd=meter.bwd)
        del adapter, judge, model_cpu
        torch.cuda.empty_cache()
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    print("PILOT_COMPLETE", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
