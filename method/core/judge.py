"""Batched replica of the official judging: misclassification & distance &
all four constraint families (relations, box, integrality, exact immutability).
Validated to agree with the official ObjectiveCalculator per-sample on the
CAA smoke run (agree = 1.0000)."""
from __future__ import annotations

import numpy as np
import torch

from tabularbench.constraints.constraints_backend_executor import (
    ConstraintsExecutor,
)
from tabularbench.constraints.numpy_backend import NumpyBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from tabularbench.utils.datatypes import to_torch_number
from tabularbench.constraints.constraints import get_feature_min_max

# Absolute tolerance on the SUM of relation residuals, including inequalities.
# This is a declared local relaxation, not the unmodified official checker.
# Feature scales differ; float32 machine epsilon does not prove this harmless.
EQ_TOL = 1e-12


def cat_z_mask(scaler, n_zcols):
    """Boolean mask over the SCALER OUTPUT columns: True = one-hot block of
    a categorical feature. TabScaler layout = [scaled numericals (num_idx
    order) | one-hot blocks (cat_idx order)], so the cat columns are exactly
    the tail starting at len(num_idx) (probe-verified on lcld/wids: flipping
    the first cat col changes z cols [len(num_idx), ...])."""
    n_num = len(getattr(scaler, "num_idx", []))
    mask = np.zeros(n_zcols, dtype=bool)
    mask[n_num:] = True
    return mask


class BatchJudge:
    """x_clean and x_adv must be passed in ONE dtype; a float64/float32 mix
    is rejected. The mixed comparison replicated the released-code precision
    mismatch (voided LCLD/CTU reproductions) and is removed. Acceptance uses
    symmetric float64; the legacy line and float32 archive audits cast BOTH
    sides to float32.
    box_tol_scale: the checker multiplies the float32-eps box tolerance by
    1000 for 108-column datasets (wids); other datasets use 1."""

    def __init__(self, model_eval, constraints_o, scaler, eps, norm="L2",
                 box_tol_scale=1, rel_tol=0.0, num_cols_mask=None):
        self.constraints = constraints_o
        self.proba = model_eval.predict_proba
        self.scaler = scaler
        self.eps = eps
        self.norm = norm
        # norm="Linf" is the type-aware variant of docs/L0_LINF_PROTOCOL.md:
        # per-column |dz| <= eps over NUMERICAL z columns only — cat one-hot
        # columns are exempt (a category switch has no magnitude semantics,
        # mirroring CaFA's treatment of categoricals outside the eps-ball).
        # num_cols_mask: bool array over z columns, True = numerical column
        # (the complement of cat_z_mask). None = all columns (plain Linf).
        self.num_cols_mask = num_cols_mask
        # Applied to the aggregate relation residual, not relative error.
        self.rel_tol = rel_tol
        self.executor = ConstraintsExecutor(
            AndConstraint(constraints_o.relation_constraints),
            backend=NumpyBackend(),
            feature_names=constraints_o.feature_names,
        )
        self.int_mask = np.asarray(constraints_o.feature_types) != "real"
        self.imm_mask = ~np.asarray(constraints_o.mutable_features, dtype=bool)
        self.box_tol = np.finfo(np.float32).eps * box_tol_scale

    def constraint_ok(self, x_clean, x_adv):
        x_clean, x_adv = np.asarray(x_clean), np.asarray(x_adv)
        if x_clean.dtype != x_adv.dtype:
            raise TypeError(
                f"x_clean {x_clean.dtype} vs x_adv {x_adv.dtype}: judge both "
                "sides in one dtype (mixed-precision convention removed)")
        rel = np.asarray(self.executor.execute(x_adv)) <= self.rel_tol
        lb, ub = get_feature_min_max(self.constraints, x_clean)
        xl_ok = np.min((lb - self.box_tol) <= x_adv, axis=1)
        xu_ok = np.min((ub + self.box_tol) >= x_adv, axis=1)
        box = xl_ok & xu_ok
        typ = np.min(x_adv[:, self.int_mask]
                     == np.round(x_adv[:, self.int_mask]), axis=1) \
            if self.int_mask.any() else np.ones(len(x_adv), dtype=bool)
        mut = np.min(x_clean[:, self.imm_mask] == x_adv[:, self.imm_mask],
                     axis=1) if self.imm_mask.any() else np.ones(
                        len(x_adv), dtype=bool)
        return rel & box & typ & mut

    def __call__(self, x_clean, y, x_adv):
        x_clean, x_adv = np.asarray(x_clean), np.asarray(x_adv)
        if x_clean.dtype != x_adv.dtype:
            raise TypeError(
                f"x_clean {x_clean.dtype} vs x_adv {x_adv.dtype}: judge both "
                "sides in one dtype (mixed-precision convention removed)")
        n = len(x_clean)
        out = np.zeros(n, dtype=bool)
        dist_all = np.zeros(n)
        viol_all = np.zeros(n)
        margin = np.zeros(n)
        for s in range(0, n, 256):
            xc, xa, yy = x_clean[s:s + 256], x_adv[s:s + 256], y[s:s + 256]
            yy = yy.astype(np.int64)   # wids labels load as float
            proba = np.asarray(self.proba(xa))
            mis = proba.argmax(1) != yy
            zc = self.scaler.transform(to_torch_number(xc)).numpy()
            za = self.scaler.transform(to_torch_number(xa)).numpy()
            if self.norm == "L2":
                dist = np.linalg.norm(za - zc, axis=1)
            elif self.num_cols_mask is not None:
                dist = np.abs((za - zc)[:, self.num_cols_mask]).max(axis=1)
            else:
                dist = np.abs(za - zc).max(1)
            cons = self.constraint_ok(xc, xa)
            out[s:s + 256] = mis & (dist <= self.eps) & cons
            dist_all[s:s + 256] = dist
            viol_all[s:s + 256] = (~cons).astype(np.float64)
            p = np.clip(proba[np.arange(len(yy)), yy], 1e-12, 1)
            margin[s:s + 256] = np.log(p)
        return out, dist_all, viol_all, margin


L0_REAL_TOL = 1e-5


def edit_counts(x_clean, x_adv, meta, free_cols):
    """Return action count, exact mutable-field L0, effective mutable L0.
    Action budget excludes derived fields. Integer/category edits are exact;
    real edits use normalized tolerance to distinguish float/map nudges.
    Exact semantic L0 is reported separately and includes derived changes.
    """
    xc, xa = np.asarray(x_clean), np.asarray(x_adv)
    mutable = meta["mutable"].to_numpy(dtype=bool)
    types = np.asarray(meta["type"].tolist())
    ranges = np.maximum(meta["max"].to_numpy(dtype=float) -
                        meta["min"].to_numpy(dtype=float), 1e-9)
    exact = xa != xc
    effective = np.where(types == "real",
                         np.abs(xa - xc) / ranges > L0_REAL_TOL, exact)
    action = np.zeros(len(types), dtype=bool)
    action[free_cols] = True
    action |= (types == "cat") & mutable
    action &= mutable
    return (effective[:, action].sum(1), exact[:, mutable].sum(1),
            effective[:, mutable].sum(1))
