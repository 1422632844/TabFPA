"""Per-row linearized direction direction initialization + true-model line search.

Mechanism: linearize the victim at the clean point (autograd through
FeasibleMap + scaler + wrapper), solve the linearized margin-maximization
numerically under the eps-ball, the per-row delta-box and the coupling encoded
in J = d z / d delta, then walk the ray {s * d*} on the TRUE model and keep
the best-margin budget-feasible candidate as the PGD seed. See
docs/REPORT.md §7.2.2 and docs/qp_direction_init.png.

Public API:
  chain_grads(fmap, scaler, wrapper, x0, y=None) -> g, a0
  jacobian_z(fmap, scaler, x0)            -> J (N, n_cols, nf_free)
  delta_bounds(fmap, meta, x0)            -> l, u (N, nf_free)
  solve_qp_rows(g, J, l, u, eps)          -> d_star (N, nf_free) numpy
  line_search(fmap, scaler, wrapper, x0, z0, d_star, eps_target, y=None,
              n_s=25)                     -> best_delta, best_margin,
                                             best_x, flip_mask

All margins are ATTACK margins a = logit[1-y] - logit[y] (>0 = flipped),
so the direction is correct for any label mix. y=None reproduces the
historical y==0 convention. Before 2026-09-05 this module hardcoded the
y==0 polarity, silently pointing the QP seed the WRONG way under the
critical (y==1) protocols (probe_qp_polarity.py: mirrored seed alone
flipped 37/127 vime rows).

wrapper must accept RAW features and emit 2-column logits (the
models.VictimAdapter convention, probe-verified for all victims).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

EPSF = 1e-6  # same relative box margin as the FeasibleMap clamps


def _two_col(out):
    return out if out.shape[1] == 2 else torch.cat([-out, out], dim=1)


def attack_sign(y):
    """s per row: attack margin a = s * (logit1 - logit0)."""
    if y is None:
        return 1.0
    return torch.where(y == 0, 1.0, -1.0)


def chain_grads(fmap, scaler, wrapper, x0, y=None):
    """g = d(attack margin)/d(delta) at delta=0 through the full chain;
    a0 = clean attack margin (<0 for correctly classified rows)."""
    s = attack_sign(y)
    d = torch.zeros(x0.shape[0], fmap.nf_free, device=x0.device,
                    requires_grad=True)
    out = _two_col(wrapper(fmap(x0, d)))
    g = torch.autograd.grad(((out[:, 1] - out[:, 0]) * s).sum(), d)[0].detach()
    with torch.no_grad():
        lg0 = _two_col(wrapper(x0))
        a0 = s * (lg0[:, 1] - lg0[:, 0])
    return g, a0


def jacobian_z(fmap, scaler, x0):
    """J[i, c, :] = d z_c / d delta at delta=0 (one backward per transformed
    column). Carries the distance-space coupling (installment, ratios, ...)."""
    d = torch.zeros(x0.shape[0], fmap.nf_free, device=x0.device,
                    requires_grad=True)
    z = scaler.transform(fmap(x0, d))
    cols = []
    for c in range(z.shape[1]):
        gr = torch.autograd.grad(z[:, c].sum(), d,
                                 retain_graph=(c < z.shape[1] - 1))[0]
        cols.append(gr.detach())
    return torch.stack(cols, dim=1)


def delta_bounds(fmap, meta, x0):
    """Per-row effective delta-box bounds [l, u] matching the map's clamps.

    lcld-style maps expose .FREE/.j/.mm — extract exact bounds incl. the
    loan g1/g5-implied range (evaluated at delta_annual=0; the true map
    re-imposes feasibility downstream). Unknown maps: uniform +-3 (the map's
    own clamps and the line-search distance gate keep candidates sound).
    """
    nf = fmap.nf_free
    N = x0.shape[0]
    l = torch.full((N, nf), -3.0, device=x0.device)
    u = torch.full((N, nf), 3.0, device=x0.device)
    if not all(hasattr(fmap, a) for a in ("FREE", "j", "mm")):
        return l, u
    mm, j = fmap.mm, fmap.j
    for fi, f in enumerate(fmap.FREE):
        if f == "loan_amnt" or f not in mm:
            continue
        lo, hi = mm[f]
        m = EPSF * max(hi - lo, 1.0)
        half = 0.5 * (hi - lo)
        l[:, fi] = ((lo + m) - x0[:, j[f]]) / half
        u[:, fi] = ((hi - m) - x0[:, j[f]]) / half
    if "loan_amnt" in mm and all(
            c in mm for c in ("ratio_loan_amnt_annual_inc", "installment")) \
            and "annual_inc" in j and "int_rate" in j and "term" in j:
        annual = x0[:, j["annual_inc"]]
        i_r = x0[:, j["int_rate"]] / 1200.0
        n_t = x0[:, j["term"]]
        pow_ = (1.0 + i_r) ** n_t
        k_ann = (i_r * pow_) / (pow_ - 1.0)
        r_lo, r_hi = mm["ratio_loan_amnt_annual_inc"]
        i_lo, i_hi = mm["installment"]
        lo_raw = torch.maximum(torch.maximum(
            torch.full_like(annual, mm["loan_amnt"][0] + EPSF * 39000.0),
            r_lo * (1 + 1e-4) * annual), (i_lo * (1 + 1e-4)) / k_ann)
        hi_raw = torch.minimum(torch.minimum(
            torch.full_like(annual, mm["loan_amnt"][1] - EPSF * 39000.0),
            r_hi * (1 - 1e-4) * annual), (i_hi * (1 - 1e-4)) / k_ann)
        half_loan = 0.5 * (mm["loan_amnt"][1] - mm["loan_amnt"][0])
        fi = fmap.FREE.index("loan_amnt")
        l[:, fi] = (lo_raw - x0[:, j["loan_amnt"]]) / half_loan
        u[:, fi] = (hi_raw - x0[:, j["loan_amnt"]]) / half_loan
    return l, u


def _qp_constraints(Ji, eps, linf):
    """SLSQP inequality constraints for the distance ball.
    L2:  eps^2 - d' (J'J) d >= 0 (single quadratic).
    Linf (type-aware, docs/L0_LINF_PROTOCOL.md): |(J d)_c| <= eps for every
    CONSTRAINED z row c (numerical columns; cat one-hot rows excluded via
    the mask) — one vectorized linear constraint block."""
    if not linf:
        Mi = Ji.T @ Ji
        return [{"type": "ineq",
                 "fun": lambda d, Mi=Mi: eps ** 2 - d @ Mi @ d,
                 "jac": lambda d, Mi=Mi: -2.0 * (Mi @ d)}]
    A = Ji

    def fun(d):
        r = eps - A @ d
        return np.concatenate([r, eps + A @ d])

    def jac(d):
        return np.concatenate([-A, A])
    return [{"type": "ineq", "fun": fun, "jac": jac}]


def _qp_fallback(gi, Ji, li, ui, eps, linf):
    """Budget-scaled gradient direction when SLSQP fails."""
    if linf:
        denom = float(np.max(np.abs(Ji @ gi))) if Ji.size else 0.0
        denom = max(denom, 1e-12)
        return np.clip(gi * (eps / denom), li, ui)
    Mi = Ji.T @ Ji
    denom = float(np.sqrt(max(gi @ Mi @ gi, 1e-24)))
    return np.clip(gi * (eps / denom), li, ui)


def solve_qp_rows(g, J, l, u, eps, norm="L2", jmask=None):
    """Numerical per-row solution of  max g.d  s.t. ||J d||_2<=eps (norm="L2")
    or |(J d)_c| <= eps on masked z rows (norm="Linf", type-aware variant),
    l<=d<=u.

    Convex (linear objective + one quadratic constraint, or purely linear
    under Linf): SLSQP is a numerical solver; its success flag is not a robustness certificate. Rows with
    ~zero gradient keep d=0; solver failures fall back to the budget-scaled
    gradient direction. jmask: bool over J's z rows, True = constrained
    (numerical) column; None = all rows."""
    linf = norm.lower() == "linf"
    if linf and jmask is not None:
        J = J[:, jmask]
    gN = g.detach().cpu().numpy().astype(np.float64)
    JN = J.detach().cpu().numpy().astype(np.float64)
    lN = l.detach().cpu().numpy().astype(np.float64)
    uN = u.detach().cpu().numpy().astype(np.float64)
    N, nf = gN.shape
    out = np.zeros((N, nf))
    n_fail = 0
    for i in range(N):
        gi, Ji = gN[i], JN[i]
        li, ui = lN[i], uN[i]
        Mi = Ji.T @ Ji
        if not np.isfinite(Mi).all() or np.linalg.norm(gi) < 1e-12:
            continue
        res = minimize(
            lambda d, gi=gi: -(gi @ d), np.zeros(nf),
            jac=lambda d, gi=gi: -gi,
            bounds=list(zip(li, ui)),
            constraints=_qp_constraints(Ji, eps, linf),
            method="SLSQP", options={"maxiter": 100, "ftol": 1e-12})
        if res.success and np.isfinite(res.x).all():
            out[i] = res.x
        else:
            n_fail += 1
            out[i] = _qp_fallback(gi, Ji, li, ui, eps, linf)
    return out, n_fail


@torch.no_grad()
def line_search(fmap, scaler, wrapper, x0, z0, d_star, eps_target, y=None,
                n_s=25, norm="L2", num_cols_mask=None):
    """Walk s in [1.0, 0.05] along s*d_star on the TRUE model; keep the
    best-ATTACK-margin candidate whose exact transform-space distance is
    within eps_target. flip = attack margin > 0 (a true misclassification
    for the row's own label). norm="Linf" + num_cols_mask (torch bool over
    z cols, True = numerical/constrained) uses the type-aware distance."""
    d0 = torch.as_tensor(d_star, dtype=torch.float32, device=x0.device)
    N, nf = d0.shape
    s = attack_sign(y)
    best_m = torch.full((N,), -1e30, device=x0.device)
    best_d = torch.zeros(N, nf, device=x0.device)
    best_x = x0.clone()
    flip = torch.zeros(N, dtype=torch.bool, device=x0.device)
    for si in np.linspace(1.0, 0.05, n_s):
        delta = d0 * float(si)
        xr = fmap(x0, delta)
        dz = scaler.transform(xr) - z0
        if norm.lower() == "linf":
            if num_cols_mask is not None:
                dz = dz[:, num_cols_mask]
            dist = dz.abs().max(dim=1).values
        else:
            dist = dz.norm(dim=1)
        lg = _two_col(wrapper(xr))
        m = s * (lg[:, 1] - lg[:, 0])
        m = torch.where(dist <= eps_target, m, torch.full_like(m, -1e30))
        upd = m > best_m
        best_m[upd] = m[upd]
        best_d[upd] = delta[upd]
        best_x[upd] = xr[upd]
        flip |= m > 0
    return best_d, best_m, best_x, flip


def _solve_subqp(gi, Ji, li, ui, eps, linf=False):
    """max g.d on a fixed support (columns preselected)."""
    Mi = Ji.T @ Ji
    if not np.isfinite(Mi).all() or np.linalg.norm(gi) < 1e-12:
        return np.zeros(len(gi))
    res = minimize(
        lambda d: -(gi @ d), np.zeros(len(gi)), jac=lambda d: -gi,
        bounds=list(zip(li, ui)),
        constraints=_qp_constraints(Ji, eps, linf),
        method="SLSQP", options={"maxiter": 100, "ftol": 1e-12})
    if res.success and np.isfinite(res.x).all():
        return res.x
    return _qp_fallback(gi, Ji, li, ui, eps, linf)


def solve_qp_rows_sparse(g, J, l, u, eps, k, max_enumerate=2000,
                         norm="L2", jmask=None):
    """Per-row L0-CONSTRAINED direction: max g.d s.t. distance ball
    (L2: ||J d||_2 <= eps; Linf type-aware: |(J d)_c| <= eps on masked
    rows), l<=d<=u, ||d||_0 <= k. Exact support enumeration when the subset
    count is small (lcld-scale); aligned-greedy forward selection otherwise (url/wids/ctu-scale). Returns (d_star, n_exact).

    The L0 cardinality is counted on FREE deltas — the attacker's actual
    field edits; derived columns follow by construction (docs/
    L0_QP_ANALYSIS.md §2)."""
    linf = norm.lower() == "linf"
    if linf and jmask is not None:
        J = J[:, jmask]
    gN = g.detach().cpu().numpy().astype(np.float64)
    JN = J.detach().cpu().numpy().astype(np.float64)
    lN = l.detach().cpu().numpy().astype(np.float64)
    uN = u.detach().cpu().numpy().astype(np.float64)
    N, nf = gN.shape
    out = np.zeros((N, nf))
    from itertools import combinations
    from math import comb
    k = min(k, nf)
    n_exact = 0
    # count arithmetically — materializing combinations() OOMs the host at
    # C(58,9) ~ 1.9e10 tuples (wids k=10 hit this: rc=137)
    n_sub = sum(comb(nf, j) for j in range(1, k + 1))
    exact = n_sub <= max_enumerate
    for i in range(N):
        gi, Ji, li, ui = gN[i], JN[i], lN[i], uN[i]
        col_cost = np.linalg.norm(Ji, axis=0)          # distance cost per
        align = np.abs(gi) / np.maximum(col_cost, 1e-12)  # unit of budget
        best_val, best_d = 0.0, np.zeros(nf)
        if exact:
            n_exact += 1
            for kk in range(1, k + 1):
                for S in combinations(range(nf), kk):
                    S = list(S)
                    dS = _solve_subqp(gi[S], Ji[:, S], li[S], ui[S], eps, linf)
                    val = gi[S] @ dS
                    if val > best_val:
                        best_val, best_d = val, np.zeros(nf)
                        best_d[S] = dS
        else:
            S = []
            while len(S) < k:
                cand = [j for j in np.argsort(-align) if j not in S]
                if not cand:
                    break
                improved = False
                for j in cand[:max(6, k + 2)]:
                    trial = S + [j]
                    dT = _solve_subqp(gi[trial], Ji[:, trial],
                                      li[trial], ui[trial], eps, linf)
                    val = gi[trial] @ dT
                    if val > best_val * (1 + 1e-9):
                        best_val, S = val, trial
                        improved = True
                        break
                if not improved:
                    break
            if S:
                dS = _solve_subqp(gi[S], Ji[:, S], li[S], ui[S], eps, linf)
                best_d[S] = dS
        out[i] = best_d
    return out, n_exact
