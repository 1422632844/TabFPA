"""TabFPA equality repair, auditing and external benchmark component loading."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from benchmark_adapter import TB_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def snap_equalities_f64(x64, constraints_o, max_iter=8):
    """Zero-tolerance EqualConstraints (feature == expr(features)) fail on
    float64 ulp noise whenever the attacker's op order differs from the
    checker's. Newton-snap each EqualConstraint's LHS feature along the
    violation (direction learned by a one-ulp probe; overshoot flips the
    sign) until the official executor reports exactly 0 for every row.

    The compiled expression is affine in the LHS feature, so this converges
    in 1-2 steps. Returns a repaired copy.
    """
    from tabularbench.constraints.constraints_backend_executor import (
        ConstraintsExecutor, )
    from tabularbench.constraints.numpy_backend import NumpyBackend

    names = list(constraints_o.feature_names)
    out = np.array(x64, dtype=np.float64, copy=True)
    for c in constraints_o.relation_constraints:
        if type(c).__name__ != "EqualConstraint":
            continue
        fid = getattr(c.left_operand, "feature_id", None)
        if fid is None or fid not in names:
            continue
        col = names.index(fid)
        ex = ConstraintsExecutor(c, backend=NumpyBackend(),
                                 feature_names=constraints_o.feature_names)
        v = np.asarray(ex.execute(out)).ravel().astype(np.float64)
        act = v > 0
        if not act.any():
            continue
        # learn the violation direction with a one-ulp probe
        probe = out.copy()
        step = np.spacing(np.abs(out[:, col]))
        step[step == 0] = 1e-12
        probe[:, col] += step
        v2 = np.asarray(ex.execute(probe)).ravel().astype(np.float64)
        sgn = np.where(v2 < v, 1.0, -1.0)
        sgn[~act] = 0.0
        for _ in range(max_iter):
            idx = np.where(act)[0]
            if idx.size == 0:
                break
            trial = out.copy()
            trial[idx, col] += sgn[idx] * v[idx]
            v2 = np.asarray(ex.execute(trial)).ravel().astype(np.float64)
            fixed = idx[v2[idx] <= 0]
            out[fixed, col] = trial[fixed, col]
            act[fixed] = False
            worse = idx[v2[idx] > v[idx]]
            sgn[worse] *= -1.0
            v[worse] = v2[worse]
    return out


def trap_rows_f64(x64, constraints_o):
    """Strict-equality violations involving ONLY immutable features.
    An immutable LHS alone is insufficient: a mutable RHS may repair it."""
    from tabularbench.constraints.constraints_backend_executor import (
        ConstraintsExecutor, )
    from tabularbench.constraints.numpy_backend import NumpyBackend

    names = list(constraints_o.feature_names)
    imm = ~np.asarray(constraints_o.mutable_features, dtype=bool)
    from tabularbench.constraints.relation_constraint import ConstraintsNode, Feature

    def referenced_columns(node):
        if isinstance(node, Feature):
            fid = node.feature_id
            return {names.index(fid) if isinstance(fid, str) else int(fid)}
        cols = set()
        if isinstance(node, ConstraintsNode):
            for value in vars(node).values():
                for child in value if isinstance(value, (list, tuple)) else [value]:
                    cols.update(referenced_columns(child))
        return cols

    trap = np.zeros(len(x64), dtype=bool)
    for c in constraints_o.relation_constraints:
        if type(c).__name__ != "EqualConstraint":
            continue
        cols = referenced_columns(c)
        if not cols or not imm[list(cols)].all():
            continue
        ex = ConstraintsExecutor(c, backend=NumpyBackend(),
                                 feature_names=constraints_o.feature_names)
        v = np.asarray(ex.execute(np.asarray(x64, dtype=np.float64))).ravel()
        trap |= v > 0
    return trap


def setup_components(dataset_name, model_name, device):
    """(ds, x_test, y_test, scaler, model_eval, constraints) — identical to
    tasks/run_smoke_test.py:setup_components."""
    from benchmark_adapter import load_model_and_weights
    from benchmark_adapter import get_dataset
    from tabularbench.models.tab_scaler import TabScaler
    from tabularbench.utils.datatypes import to_torch_number

    ds = get_dataset(dataset_name)
    x, y = ds.get_x_y()
    i_test = ds.get_splits()["test"]
    x_test, y_test = x.iloc[i_test].to_numpy(), y[i_test]
    metadata = ds.get_metadata(only_x=True)

    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(to_torch_number(x), x_type=metadata["type"])

    model_arch = model_name.split("_")[0]
    model_training = "_".join(model_name.split("_")[1:])
    model_eval = load_model_and_weights(
        ds.name, model_arch, model_training, metadata, scaler, device
    )
    constraints_o = ds.get_constraints()
    return ds, x_test, y_test, scaler, model_eval, constraints_o
