"""Phase-2B step 2 (2026-09-30): 996 config x five models on CORRECT-sampling
f0draw pools (official-effect draw: filter_correct=False, then clean-correct
subset). Supersedes lcld_tune100_validation_20260929 (wrong sampling order --
clean-correct BEFORE the draw; those numbers must not be cited).

Config = the frozen 996 route: FIXED + ARMS['d06_l12'] from lcld_tune100
(ballv2 + se/dir gates + chain BPDA, logits_lr=.12, d_lr=.06, 100+100).
Kernel hash pinned via lcld_tune100.EXPECTED_KERNEL. One model per invocation,
five seeds, resumable (existing json+npz verified and skipped).
Supports the five default and the four Madry models (tabnet_madry has an empty
clean-correct pool and is pool-only). Outputs: runs/lcld996_f0draw_20260930/
(default) and runs/lcld996_f0draw_madry/ (Madry)."""
import os, sys, json, hashlib, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TABULARBENCH_ROOT", str(ROOT / "tabularbench"))
os.environ.setdefault("TABULARBENCH_DATASET_DIR", str(ROOT / "tabularbench/datasets"))
os.environ.setdefault("TABULARBENCH_TARGET_MODEL_DIR", str(ROOT / "tabularbench/target_model"))
sys.path[:0] = [str(ROOT), str(ROOT / "method"), str(ROOT / "tabularbench")]
os.environ['WK_AUDIT_TOL'] = '1e-12'
assert not os.getenv('WK_PATIENCE')
import argparse
import numpy as np
import torch
import lcld_tune100 as t
from lcld_extension import LcldMoves, LcldCompletion
from core.judge import BatchJudge
from tabularbench.constraints.constraints_checker import ConstraintChecker
import run_experiments as rx

ARM = 'd06_l12'
DEFAULT_MODELS = ["torchrln_default", "stg_default", "vime_default",
                  "tabnet_default", "tabtransformer_default"]
# tabnet_madry is excluded from this attack entry: the official Madry checkpoint
# classifies no drawn row as positive, so its clean-correct pool is empty and
# the forward-accounting assertions cannot run. The pool itself is still built
# by lcld_f0draw_all.py for provenance.
MADRY_MODELS = ["torchrln_madry", "stg_madry", "vime_madry", "tabtransformer_madry"]
MODELS = DEFAULT_MODELS + MADRY_MODELS


def out_root(model):
    return ROOT / ('runs/lcld996_f0draw_madry' if model.endswith('_madry')
                   else 'runs/lcld996_f0draw_20260930')
wk = t.wk
CONFIG = {**t.FIXED, **t.ARMS[ARM]}


def pool_path(model):
    return ROOT / f'reference/caa_pools/lcld_v2_iid__{model}_f0draw_cc.npz'


def run_one(model, seed, meta, scaler, cons, cpu, adapter, judge):
    dest = out_root(model) / f'{model}_seed{seed}_{ARM}'
    kernel_hash = hashlib.sha256(Path(wk.__file__).read_bytes()).hexdigest()
    assert kernel_hash == t.EXPECTED_KERNEL, kernel_hash
    pp = pool_path(model)
    pool_hash = hashlib.sha256(pp.read_bytes()).hexdigest()
    if dest.with_suffix('.json').exists():
        prev = json.loads(dest.with_suffix('.json').read_text())
        assert prev['config'] == CONFIG and prev['kernel_sha256'] == kernel_hash
        assert prev['pool_sha256'] == pool_hash
        assert prev['sampling'] == 'f0draw_cc'
        return prev
    x = np.load(pp)['clean']
    n = len(x)
    ok = np.zeros(n, bool); fwd = np.zeros(n, int)
    adv = x.copy(); best = x.copy(); visits = np.zeros(n, int)
    margins = np.zeros(n); chunks = []
    for start in range(0, n, 128):
        chunk = x[start:start + 128]
        meter, row = wk.run('lcld_v2_iid', model, meta, scaler, cons,
                            cpu, adapter, judge, chunk, seed * 1000003 + start,
                            moves_cls=LcldMoves, completion_ctor=LcldCompletion,
                            **CONFIG)
        sl = slice(start, start + len(chunk))
        ok[sl] = meter.ok; fwd[sl] = meter.fwd; adv[sl] = meter.out
        best[sl] = meter.best_point; visits[sl] = meter.optimization_visits
        margins[sl] = meter.final_best_margin; chunks.append(row)
        assert not meter.rescue_queries.any()
    assert fwd.max() <= 256 and visits.max() <= 200
    summary = dict(model=model, arm=ARM, seed=seed, S=int(ok.sum()), n=n,
                   sampling='f0draw_cc',
                   config=CONFIG, kernel_sha256=kernel_hash,
                   pool_sha256=pool_hash,
                   fwd_max=int(fwd.max()), fwd_mean=float(fwd.mean()),
                   relation_max_seen=max(c_['relation_max_seen'] for c_ in chunks),
                   seconds=sum(c_['seconds'] for c_ in chunks))
    np.savez_compressed(dest.with_suffix('.npz'), ok=ok, fwd=fwd, adv=adv,
                        best_point=best, optimization_visits=visits,
                        final_best_margin=margins)
    dest.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def verify(model, seeds, meta, scaler, cons, cpu, judge):
    pool = np.load(pool_path(model))['clean']
    assert (cpu.predict_proba(pool).argmax(1) == 1).all()
    strict = ConstraintChecker(cons, tolerance=1e-12)
    rows = []
    for seed in seeds:
        z = np.load((out_root(model) / f'{model}_seed{seed}_{ARM}').with_suffix('.npz'))
        adv = z['adv']
        actual, dist, _, _ = judge(pool, np.ones(len(pool), dtype=int), adv)
        actual &= np.asarray(strict.check_constraints(pool, adv), bool)
        np.testing.assert_array_equal(actual, z['ok'])
        assert np.isfinite(adv).all() and (z['fwd'] + 2 <= 256).all()
        numeric = np.asarray(scaler.num_idx)
        moves = LcldMoves(meta, scaler, cons)
        cost = (((adv[:, numeric] - pool[:, numeric]) / moves.scale[numeric]) ** 2).sum(1)
        assert (cost <= .495 ** 2 + 1e-10).all() and (dist <= .495 + 1e-6).all()
        immutable = ~np.asarray(meta.mutable, bool)
        np.testing.assert_array_equal(adv[:, immutable], pool[:, immutable])
        rows.append(dict(seed=seed, S=int(actual.sum()),
                         fwd_max=int(z['fwd'].max())))
    result = dict(model=model, arm=ARM, verified=True, rows=rows,
                  mean_S=sum(r['S'] for r in rows) / len(rows))
    (out_root(model) / f'verification_{model}_{ARM}.json').write_text(json.dumps(result, indent=2))
    print('VERIFIED', json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=MODELS)
    a = ap.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    out = out_root(a.model)
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(arm=ARM, config=CONFIG, kernel_sha256=t.EXPECTED_KERNEL,
                sampling='f0draw_cc (official filter_correct=False draw, then clean-correct subset)',
                supersedes='runs/lcld_tune100_validation_20260929 (wrong sampling order)',
                seeds=list(range(5)))
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    xall, _, meta, scaler, cons = rx.dataset_data('lcld_v2_iid')
    cpu, adapter, _ = rx.victims('lcld_v2_iid', a.model, meta, scaler, cons, xall[:32])
    judge = BatchJudge(cpu, cons, scaler, .5, box_tol_scale=1, rel_tol=1e-7)
    for seed in range(5):
        run_one(a.model, seed, meta, scaler, cons, cpu, adapter, judge)
    verify(a.model, range(5), meta, scaler, cons, cpu, judge)
    print('MODEL_DONE', a.model, flush=True)


if __name__ == "__main__":
    main()
