"""WiDS mapping-constrained attack, w_reinit edition (default main method).

Same machinery as wids_mapping_extension.run_mapping (DiagGrid fiber head +
WidsMappingMoves/Completion), upgraded to the universal schedule:
kappa40 init + logits lr 0.48 + in-place FULL reset at round 100 for the
grid head and move logits (kappa0), d->0, current->clean, Adam rebuild.
Single 200-round run, 256-forward cap, no supplementary search.

Full-pool official protocol: pool = legal class-1 test rows (mapping judge,
clean data has 0 violations so the pool equals the native one), per-model
clean-correct, original order; batch 128; per-batch seed = seed*1000003+start;
RA = (N-S)/D. eps = 0.5 (campaign protocol; the 2026-09-08 experiment used 0.6).
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
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
from core.data import setup_components
from core.feasible_integer_moves import JointIntegerMoves
from core.wids_completion import WidsCompletion
from core.wids_mapping_constraint import DIAG, SYSTEM, extended_relation_constraints
from core.judge import BatchJudge
from models import load_victim
from scripts.feasible_gumbel_ablation import AttackLogits
from scripts.wids_gumbel_ablation import DEVICE, Meter
from tabdatasets.wids import ORDER_PAIRS
from tabularbench.constraints.constraints_checker import ConstraintChecker

from wids_mapping_extension import DiagGrid, WidsMappingMoves, WidsMappingCompletion

ROOT = Path(__file__).resolve().parent


def run_mapping_wreinit(model_name, meta, scaler, cons_native, model, adapter,
                        judge, cons_ext, x, seed, eps=0.5, steps=200,
                        logits_lr=0.48, reinit_round=100, reinit_kappa=0.,
                        arm="constraint", progress=None):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    n = len(x); current = x.copy()
    names = meta["feature"].tolist()
    dcol, scol = names.index(DIAG), names.index(SYSTEM)
    orders = SimpleNamespace(relation_constraints=list(cons_native.relation_constraints))
    moves = WidsMappingMoves(meta, scaler, orders, dcol, eps * .99)
    diag = DiagGrid(x, meta, scaler, eps, arm)
    x0 = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    scale = torch.tensor(moves.scale, dtype=torch.float32, device=DEVICE)
    values, mask = moves.candidates(x, current)
    initial_lengths = np.abs(values) * np.sqrt(moves.a)[None, :, None]
    prior = -40 * initial_lengths
    theta = torch.tensor(np.concatenate(
        [np.full((n, 1), 2.), prior.reshape(n, -1)], axis=1),
        dtype=torch.float32, device=DEVICE, requires_grad=True)
    d = torch.zeros((n, len(moves.reals)), device=DEVICE, requires_grad=True)
    params = [{"params": [d], "lr": .06},
              {"params": [theta], "lr": logits_lr},
              {"params": [diag.theta], "lr": logits_lr}]
    opt = torch.optim.Adam(params)
    y = np.ones(n, dtype=np.int64)
    scoring = AttackLogits(adapter, model_name)
    meter = Meter(adapter, model, judge, x, y, DEVICE, 256, 128)
    completion = WidsMappingCompletion(meta, scaler, eps * .99, dcol, scol)
    A = np.concatenate([moves.L, moves.E], axis=0)
    bias = np.concatenate([moves.lb, moves.eb])
    TA = torch.tensor(A, dtype=torch.float32, device=DEVICE)
    TB = torch.tensor(bias, dtype=torch.float32, device=DEVICE)
    norm = (TA.abs() * scale[None]).max(1).values.clamp_min(1e-9)
    iseq = torch.arange(len(A), device=DEVICE) >= len(moves.L)
    lo_t = torch.tensor(moves.lo[moves.reals], dtype=torch.float32, device=DEVICE)
    hi_t = torch.tensor(moves.hi[moves.reals], dtype=torch.float32, device=DEVICE)
    num_idx_t = torch.as_tensor(list(scaler.num_idx), device=DEVICE)
    pregated = 0
    start = time.perf_counter()
    with torch.no_grad():
        logits = meter.score(x0, np.arange(n))
        best_margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    for step in range(steps):
        ids = np.flatnonzero(~meter.ok & (meter.fwd < 254))
        if not len(ids):
            break
        if progress and step % 25 == 0:
            progress(step, int(meter.ok.sum()))
        if step == reinit_round:
            with torch.no_grad():
                theta.data[:, 1:] = torch.tensor(
                    -float(reinit_kappa) * initial_lengths.reshape(n, -1),
                    dtype=torch.float32, device=DEVICE)
                stay_mask = np.isclose(diag.values, x[:, dcol][:, None])
                diag.theta.data.copy_(torch.tensor(
                    np.where(diag.mask, 2.0 * stay_mask, -1e9),
                    dtype=torch.float32, device=DEVICE))
                d.data.zero_()
            current = x.copy()
            opt = torch.optim.Adam(params)
        requested = current.copy()
        requested[:, moves.reals] = x[:, moves.reals]
        anchor, possible = completion.solve(x, requested)
        assert possible.all(), "Minimum-cost continuous anchor failed"
        values, mask = moves.candidates(x, anchor)
        if step >= reinit_round:
            tau = 1. - .9 * (step - reinit_round) / max(steps - 1 - reinit_round, 1)
        else:
            tau = 1. - .9 * step / max(steps - 1, 1)
        integer_point, straight, counts = moves.sample(anchor, values, mask, theta, tau)
        code = diag.sample(tau)
        integer_point[:, dcol] = code.detach().cpu().numpy()
        # joint budget gate: the integer move and the code jump share one L2
        # ball; rows whose combined draw exceeds the internal radius skip the
        # round (no victim query) -- same semantics as completion rejection.
        cost_all = (((integer_point[ids] - x[ids]) / moves.scale) ** 2).sum(1)
        legal_all = judge.constraint_ok(x[ids], integer_point[ids])
        keep = (cost_all <= (eps * .99) ** 2 + 1e-10) & legal_all
        pregated += int((~keep).sum())
        ids = ids[keep]
        if not len(ids):
            continue
        valid = judge.constraint_ok(x[ids], integer_point[ids])
        cost = (((integer_point[ids] - x[ids]) / moves.scale) ** 2).sum(1)
        assert valid.all() and (cost <= (eps * .99) ** 2 + 1e-10).all(), \
            "Joint move + code invalid before completion"
        raw = torch.tensor(current, dtype=torch.float32, device=DEVICE)
        raw[:, moves.ints] = straight
        raw[:, moves.reals] = (x0[:, moves.reals] + d * scale[moves.reals]).clamp(lo_t, hi_t)
        proposal = integer_point[ids].copy()
        proposal[:, moves.reals] = raw[ids][:, moves.reals].detach().cpu().numpy()
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
        budget = torch.relu(dz[:, num_idx_t].norm(dim=1) - .9 * eps).square().mean()
        residual = (raw[ii] @ TA.T + TB) / norm
        relation = torch.where(iseq[None], residual, torch.relu(residual)).square().sum(1).mean()
        loss = attack + 4 * budget + 4 * relation
        opt.zero_grad(); loss.backward()
        assert theta.grad is not None and torch.isfinite(theta.grad).all()
        assert diag.theta.grad is not None and torch.isfinite(diag.theta.grad).all()
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
    assert ConstraintChecker(cons_ext, tolerance=0.).check_constraints(x, meter.out).all()
    assert (dist <= eps).all() and meter.fwd.max() <= 256
    scoring.close()
    row = dict(arm=arm, model=model_name, seed=seed, n=n,
               accepted=int(ok.sum()), asr=float(ok.mean()),
               forward_mean=float(meter.fwd.mean()), forward_max=int(meter.fwd.max()),
               seconds=seconds, distance_max=float(dist.max()),
               completion_rejected=completion.rejected,
               pregated_rows=pregated,
               diag_grid_max=int(diag.mask.sum(1).max()),
               diag_changed=int((meter.out[:, dcol] != x[:, dcol]).sum()),
               schedule="w_reinit(k40,reset@100->k0,lr.48)")
    return meter, row


def run_full(model_name, seeds, eps=0.5, batch=128, out=ROOT.parent / "runs/wids_mapping_full_wreinit"):
    odir = Path(out) / model_name
    odir.mkdir(parents=True, exist_ok=True)
    ds, x_test, y_test, scaler, model_cpu, cons_native = setup_components("wids", model_name, "cpu")
    xall = np.asarray(x_test)[np.asarray(y_test) == 1]
    legal = np.concatenate([BatchJudge(model_cpu, cons_native, scaler, eps, box_tol_scale=1000, rel_tol=0.)
                            .constraint_ok(xall[i:i + 128], xall[i:i + 128]) for i in range(0, len(xall), 128)])
    valid_x = xall[legal]; den = len(valid_x)
    pred = np.concatenate([model_cpu.predict_proba(valid_x[i:i + 128]).argmax(1) for i in range(0, len(valid_x), 128)])
    x = valid_x[pred == 1]; total = len(x)
    assert total > 0
    cons_ext = copy.copy(cons_native)
    cons_ext.relation_constraints = extended_relation_constraints(cons_native)
    judge = BatchJudge(model_cpu, cons_ext, scaler, eps, box_tol_scale=1000, rel_tol=0.)
    meta = ds.get_metadata(only_x=True).reset_index(drop=True)
    _, _, _, _, _, _, adapter = load_victim("wids", model_name, DEVICE)
    print(json.dumps(dict(status="pool", model=model_name, valid_class1=den, clean_correct=total)), flush=True)
    rows = []
    for seed in seeds:
        S = 0
        for st in range(0, total, batch):
            xx = x[st:st + batch]
            meter, row = run_mapping_wreinit(model_name, meta, scaler, cons_native,
                                             model_cpu, adapter, judge, cons_ext, xx,
                                             seed * 1000003 + st, eps=eps)
            S += row["accepted"]
            np.savez_compressed(odir / f"wreinit_s{seed}_b{st}.npz",
                                adv=meter.out, ok=meter.ok, clean=xx, fwd=meter.fwd, bwd=meter.bwd,
                                indices=np.arange(st, st + len(xx)))
            print(json.dumps(dict(status="batch_committed", model=model_name, seed=seed,
                                  offset=st, accepted=row["accepted"], n=row["n"])), flush=True)
            del meter; torch.cuda.empty_cache()
        print(json.dumps(dict(status="scene_complete", model=model_name, seed=seed,
                              S=S, N=total, D=den, RA=round(100.0 * (total - S) / den, 2))), flush=True)
        rows.append(dict(model=model_name, seed=seed, S=S, N=total, D=den,
                         RA=round(100.0 * (total - S) / den, 2)))
    (odir / "results.json").write_text(json.dumps(rows, indent=2))
    del adapter, judge, model_cpu
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["tabnet_default", "torchrln_default",
                                                    "tabtransformer_default", "stg_default", "vime_default"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--smoke", action="store_true", help="single seed, first 128 rows only")
    ap.add_argument("--out", default=None, help="output root (default runs/wids_mapping_full_wreinit)")
    a = ap.parse_args()
    seeds = [0] if a.smoke else a.seeds
    for m in a.models:
        if a.smoke:
            _smoke(m, a)
        else:
            run_full(m, seeds, eps=a.eps, batch=a.batch, out=a.out)
    print("ALL_DONE", flush=True)


def _smoke(model_name, a):
    ds, x_test, y_test, scaler, model_cpu, cons_native = setup_components("wids", model_name, "cpu")
    xall = np.asarray(x_test)[np.asarray(y_test) == 1]
    legal = np.concatenate([BatchJudge(model_cpu, cons_native, scaler, a.eps, box_tol_scale=1000, rel_tol=0.)
                            .constraint_ok(xall[i:i + 128], xall[i:i + 128]) for i in range(0, len(xall), 128)])
    valid_x = xall[legal]
    pred = np.concatenate([model_cpu.predict_proba(valid_x[i:i + 128]).argmax(1) for i in range(0, len(valid_x), 128)])
    x = valid_x[pred == 1][:128]
    cons_ext = copy.copy(cons_native)
    cons_ext.relation_constraints = extended_relation_constraints(cons_native)
    judge = BatchJudge(model_cpu, cons_ext, scaler, a.eps, box_tol_scale=1000, rel_tol=0.)
    meta = ds.get_metadata(only_x=True).reset_index(drop=True)
    _, _, _, _, _, _, adapter = load_victim("wids", model_name, DEVICE)
    meter, row = run_mapping_wreinit(model_name, meta, scaler, cons_native, model_cpu,
                                     adapter, judge, cons_ext, x, 0, eps=a.eps)
    print(json.dumps(dict(status="smoke_done", **row)), flush=True)


if __name__ == "__main__":
    main()
