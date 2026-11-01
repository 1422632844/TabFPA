"""lcld_v2_iid dataset module: FeasibleMap (feasibility retraction cascade)
+ float64 finalize. Moved verbatim from exp_feasible.py."""
from __future__ import annotations

import numpy as np
import torch

EPS_FLOAT = 1e-6  # relative box margin (checker tolerance is float32 eps)


class FeasibleMap:
    """Dataset-specific substitution and order-repair map.
    Box margins/recomputation need not preserve x at delta=0. Final decoded
    candidates require constraint and budget checks; integers use BPDA."""

    FREE = ["loan_amnt", "annual_inc", "dti", "revol_bal", "revol_util",
            "emp_length", "open_acc", "total_acc", "mort_acc"]

    def __init__(self, meta, device):
        self.device = device
        self.j = {n: i for i, n in enumerate(meta["feature"].tolist())}
        self.mm = {r.feature: (float(r.min), float(r.max))
                   for r in meta.itertuples()}
        self.nf = len(self.FREE)
        self.nf_free = self.nf  # unified attribute name

    def _col(self, x, name):
        return x[:, self.j[name]]

    def __call__(self, x0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = x0.clone()
        mm, jj = self.mm, self.j

        def clamp01(v, name):
            lo, hi = mm[name]
            m = EPS_FLOAT * max(hi - lo, 1.0)
            return v.clamp(lo + m, hi - m)

        half = []
        for f in self.FREE:
            lo, hi = mm[f]
            half.append(0.5 * (hi - lo))
        half = torch.tensor(half, device=self.device)

        # annual_inc (free real)
        annual = clamp01(self._col(x, "annual_inc")
                         + delta[:, self.FREE.index("annual_inc")] * half[1],
                         "annual_inc")
        x[:, jj["annual_inc"]] = annual

        # loan_amnt with effective bounds from g1/g5/box
        # int_rate/term are immutable: read from constant x0 so the pow
        # backward saves a stable tensor, not a view of the mutated x
        i_r = x0[:, jj["int_rate"]] / 1200.0
        n_t = x0[:, jj["term"]]
        pow_ = (1.0 + i_r) ** n_t
        k_ann = (i_r * pow_) / (pow_ - 1.0)
        r_lo, r_hi = mm["ratio_loan_amnt_annual_inc"]
        i_lo, i_hi = mm["installment"]
        lo = torch.max(torch.stack(
            [torch.full_like(annual, 1000.0 + EPS_FLOAT * 39000.0),
             r_lo * (1 + 1e-4) * annual,
             (i_lo * (1 + 1e-4)) / k_ann], dim=1), dim=1).values
        hi = torch.min(torch.stack(
            [torch.full_like(annual, 40000.0 - EPS_FLOAT * 39000.0),
             r_hi * (1 - 1e-4) * annual,
             (i_hi * (1 - 1e-4)) / k_ann], dim=1), dim=1).values
        loan = (self._col(x, "loan_amnt")
                + delta[:, self.FREE.index("loan_amnt")] * half[0]).clamp(lo, hi)
        x[:, jj["loan_amnt"]] = loan
        x[:, jj["installment"]] = k_ann * loan                    # g1 exact
        x[:, jj["ratio_loan_amnt_annual_inc"]] = loan / annual    # g5 exact

        # open/total ints + g2 + g6
        tot = bpda_round(clamp01(
            self._col(x, "total_acc")
            + delta[:, self.FREE.index("total_acc")] * half[7], "total_acc"))
        opn = bpda_round(clamp01(
            self._col(x, "open_acc")
            + delta[:, self.FREE.index("open_acc")] * half[6], "open_acc"))
        opn = torch.minimum(opn, tot)                             # g2
        ot_lo = mm["ratio_open_acc_total_acc"][0]
        opn = torch.maximum(
            opn, torch.ceil((ot_lo - 1e-9) * tot - 1e-6).clamp_min(1.0))
        x[:, jj["total_acc"]] = tot
        x[:, jj["open_acc"]] = opn
        x[:, jj["ratio_open_acc_total_acc"]] = opn / tot           # g6 exact

        # remaining independent features
        for fi, f in enumerate(self.FREE):
            if f in ("loan_amnt", "annual_inc", "open_acc", "total_acc"):
                continue
            v = self._col(x, f) + delta[:, fi] * half[fi]
            v = clamp01(v, f)
            if f in ("emp_length", "mort_acc"):
                v = bpda_round(v)
            x[:, jj[f]] = v
        return x


def bpda_round(t: torch.Tensor) -> torch.Tensor:
    return t + (torch.round(t) - t).detach()


def finalize_f64(x_clean64, x_opt32, meta, ratio_arith="f32"):
    """Rebuild the adversarial sample for the official checker:
    immutable columns keep clean float64 bits (exact equality); ints
    rounded and never clipped afterwards; g1 (installment) recomputed with
    the exact annuity formula (0.01 tol); g5/g6 ratios recomputed to match
    the checker bit-for-bit.

    ratio_arith: "f32" — ratios computed in float32 arithmetic from float32
    operands (matches the official float32 recompute of an f32-cast adv;
    the historical default). "f64" — ratios computed in float64 from f64
    operands, for a float64-native attack submission (MOEVA-style): the
    checker's float64 recompute then matches exactly and the zero-tolerance
    relation check passes."""
    names = meta["feature"].tolist()
    j = {n: i for i, n in enumerate(names)}
    out = np.array(x_clean64, dtype=np.float64, copy=True)
    x32 = np.asarray(x_opt32, dtype=np.float64)
    for f in FeasibleMap.FREE:
        c = j[f]
        unchanged = x32[:, c] == np.asarray(x_clean64)[:, c].astype(np.float32)
        out[:, c] = np.where(unchanged, out[:, c], x32[:, c])
    # mutable categorical columns follow the attack sample (the L0/Linf
    # protocol exempts cats from the eps-ball, so they carry discrete flips;
    # frozen historically only because the L2 metric priced a flip at
    # sqrt(2) > eps — docs/L0_LINF_PROTOCOL.md §3)
    mut = meta["mutable"].to_numpy(dtype=bool)
    is_cat = np.asarray(meta["type"].tolist()) == "cat"
    for c in np.nonzero(mut & is_cat)[0]:
        out[:, c] = x32[:, c]
    int_free = ("emp_length", "open_acc", "total_acc", "mort_acc")
    for f in int_free:
        out[:, j[f]] = np.round(out[:, j[f]])
    mm = {r.feature: (float(r.min), float(r.max)) for r in meta.itertuples()}
    for f in FeasibleMap.FREE:
        if f in int_free or f in ("loan_amnt", "annual_inc"):
            continue
        lo_f, hi_f = mm[f]
        out[:, j[f]] = np.clip(out[:, j[f]], lo_f + 1e-9 * max(hi_f - lo_f, 1),
                               hi_f - 1e-9 * max(hi_f - lo_f, 1))
    i_r = out[:, j["int_rate"]] / 1200.0
    n_t = out[:, j["term"]]
    pow_ = (1.0 + i_r) ** n_t
    k_ann = (i_r * pow_) / (pow_ - 1.0)
    r_lo, r_hi = mm["ratio_loan_amnt_annual_inc"]
    i_lo, i_hi = mm["installment"]
    annual = out[:, j["annual_inc"]]
    loan_bound_lo = np.maximum(np.maximum(
        1000.0 + 1e-2, r_lo * (1 + 1e-5) * annual),
        (i_lo * (1 + 1e-5)) / k_ann)
    loan_bound_hi = np.minimum(np.minimum(
        40000.0 - 1e-2, r_hi * (1 - 1e-5) * annual),
        (i_hi * (1 - 1e-5)) / k_ann)
    loan = np.clip(out[:, j["loan_amnt"]], loan_bound_lo, loan_bound_hi)
    out[:, j["loan_amnt"]] = loan
    out[:, j["installment"]] = k_ann * loan
    if ratio_arith == "f64":
        # g5 in pure f64: checker recompute matches bit-for-bit
        out[:, j["ratio_loan_amnt_annual_inc"]] = loan / annual
        # g6: strict lower bound (no epsilon slack) so ratio >= r_lo exactly
        tot = out[:, j["total_acc"]]
        opn = np.clip(np.minimum(out[:, j["open_acc"]], tot),
                      np.ceil(mm["ratio_open_acc_total_acc"][0] * tot), None)
        opn = np.maximum(opn, 1.0)
        out[:, j["open_acc"]] = opn
        out[:, j["ratio_open_acc_total_acc"]] = opn / tot
        out[:, ~mut] = np.asarray(x_clean64)[:, ~mut]
        return out
    loan32 = loan.astype(np.float32)
    annual32 = annual.astype(np.float32)
    out[:, j["ratio_loan_amnt_annual_inc"]] = (
        loan32 / annual32).astype(np.float64)
    tot = out[:, j["total_acc"]]
    opn = np.clip(np.minimum(out[:, j["open_acc"]], tot),
                  np.ceil(mm["ratio_open_acc_total_acc"][0] * tot - 1e-9),
                  None)
    opn = np.maximum(opn, 1.0)
    out[:, j["open_acc"]] = opn
    out[:, j["ratio_open_acc_total_acc"]] = (
        opn.astype(np.float32) / tot.astype(np.float32)).astype(np.float64)
    out[:, ~mut] = np.asarray(x_clean64)[:, ~mut]
    return out


JUDGE_KWARGS = {}  # default box tolerance
