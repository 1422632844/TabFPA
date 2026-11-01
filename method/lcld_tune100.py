"""LOCAL WRAPPER / AUDIT: parameter-only tuning of frozen B+chain 100+100.

No new optimizer, loss, direction search, restart, or kernel edits.
Development is on the fixed evaluation pool; report that selection explicitly.
"""
import os, sys, json, hashlib, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'method')]
os.environ['WK_AUDIT_TOL'] = '1e-12'
assert not os.getenv('WK_PATIENCE'), 'Fixed restart100 must not be overridden'
import numpy as np
import torch
import run_experiments as rx
import lcld_schedule_kernel as wk
from lcld_extension import LcldMoves, LcldCompletion
from core.judge import BatchJudge
from tabularbench.constraints.constraints_checker import ConstraintChecker

OUT = ROOT / 'runs/lcld_tune100_20260929'
EXPECTED_KERNEL = '23e81796bc2be109a55ea98fc2925dadb9fd9514497710996a937574e39dc2f4'
FIXED = dict(steps=200, kappa=40., reinit_round=100, reinit_kappa=0.,
             cont_param='ballv2', ball_sched='se', reinit_dir=True,
             dir_gate=25, se_gate=25, repair_grad='chain', rel_w=0.,
             logits_lr=.48, d_lr=.06, rho_scale=1.,
             dfo_start=None, extra_search=0, stall_restart=False)
ARMS = {
    'd08_l48': dict(d_lr=.08),
    'd12_l48': dict(d_lr=.12),
    'd18_l48': dict(d_lr=.18),
    'd08_l24': dict(d_lr=.08, logits_lr=.24),
    'd12_l24': dict(d_lr=.12, logits_lr=.24),
    'd18_l24': dict(d_lr=.18, logits_lr=.24),
    'd04_l48': dict(d_lr=.04),
    'd06_l12': dict(logits_lr=.12),
    'gate10': dict(dir_gate=10, se_gate=10),
    'gate50': dict(dir_gate=50, se_gate=50),
}

def path(model, arm, seed):
    return OUT / f'{model}_seed{seed}_{arm}'

def read(model, arm, seed):
    return json.loads(path(model, arm, seed).with_suffix('.json').read_text())

def run_one(model, arm, seed, meta, scaler, cons, cpu, adapter, judge):
    config = {**FIXED, **ARMS[arm]}
    dest = path(model, arm, seed)
    kernel_hash = hashlib.sha256(Path(wk.__file__).read_bytes()).hexdigest()
    assert kernel_hash == EXPECTED_KERNEL
    pool_path = ROOT / f'reference/caa_pools/lcld_v2_iid__{model}.npz'
    pool_hash = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    if dest.with_suffix('.json').exists():
        prev = read(model, arm, seed)
        assert prev['config'] == config and prev['kernel_sha256'] == kernel_hash
        assert prev['pool_sha256'] == pool_hash
        assert dest.with_suffix('.npz').exists()
        return prev
    x = np.load(pool_path)['clean']
    assert len(x) == 1000
    ok = np.zeros(len(x), bool); fwd = np.zeros(len(x), int)
    adv = x.copy(); best = x.copy(); visits = np.zeros(len(x), int)
    margins = np.zeros(len(x)); chunks = []
    for start in range(0, len(x), 128):
        chunk = x[start:start+128]
        meter, row = wk.run('lcld_v2_iid', model, meta, scaler, cons,
            cpu, adapter, judge, chunk, seed*1000003+start,
            moves_cls=LcldMoves, completion_ctor=LcldCompletion, **config)
        sl = slice(start, start+len(chunk))
        ok[sl] = meter.ok; fwd[sl] = meter.fwd; adv[sl] = meter.out
        best[sl] = meter.best_point; visits[sl] = meter.optimization_visits
        margins[sl] = meter.final_best_margin; chunks.append(row)
        assert not meter.rescue_queries.any()
    assert fwd.max() <= 256 and visits.max() <= 200
    summary = dict(model=model, arm=arm, seed=seed, S=int(ok.sum()), n=len(x),
        config=config, kernel_sha256=kernel_hash, pool_sha256=pool_hash,
        fwd_max=int(fwd.max()), fwd_mean=float(fwd.mean()),
        relation_max_seen=max(c['relation_max_seen'] for c in chunks),
        seconds=sum(c['seconds'] for c in chunks), chunks=chunks)
    np.savez_compressed(dest.with_suffix('.npz'), ok=ok, fwd=fwd, adv=adv,
        best_point=best, optimization_visits=visits, final_best_margin=margins)
    dest.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='chunks'}), flush=True)
    return summary

def verify(model, arm, seeds, meta, scaler, cons, cpu, judge):
    pool = np.load(ROOT / f'reference/caa_pools/lcld_v2_iid__{model}.npz')['clean']
    assert (cpu.predict_proba(pool).argmax(1) == 1).all()
    strict = ConstraintChecker(cons, tolerance=1e-12)
    moves = LcldMoves(meta, scaler, cons)
    rows=[]
    for seed in seeds:
        z = np.load(path(model, arm, seed).with_suffix('.npz'))
        adv = z['adv']
        actual, dist, _, _ = judge(pool, np.ones(len(pool), dtype=int), adv)
        actual &= np.asarray(strict.check_constraints(pool, adv), bool)
        np.testing.assert_array_equal(actual, z['ok'])
        assert np.isfinite(adv).all() and (z['fwd']+2 <= 256).all()
        numeric=np.asarray(scaler.num_idx)
        cost=(((adv[:,numeric]-pool[:,numeric])/moves.scale[numeric])**2).sum(1)
        assert (cost <= .495**2+1e-10).all() and (dist <= .495+1e-6).all()
        immutable=~np.asarray(meta.mutable, bool)
        np.testing.assert_array_equal(adv[:,immutable], pool[:,immutable])
        rows.append(dict(seed=seed, S=int(actual.sum()), fwd_max=int(z['fwd'].max()),
            including_verification_max=int(z['fwd'].max()+2),
            float64_l2_max=float(np.sqrt(cost).max())))
    result=dict(model=model, arm=arm, verified=True, rows=rows,
                mean_S=sum(r['S'] for r in rows)/len(rows))
    (OUT/f'verification_{model}_{arm}.json').write_text(json.dumps(result,indent=2))
    print('VERIFIED', json.dumps(result), flush=True)
    return result

def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=dict(fixed=FIXED, arms=ARMS, kernel_sha256=EXPECTED_KERNEL,
        development='Full fixed pool seed0; top three by S, then forward mean, then name',
        promotion='Complete seeds0..4 for selected configurations; require mean_S>996',
        protocol='100+100 gradient rounds, no DFO, original feasibility and numeric tolerances')
    mp=OUT/'plan.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest
    else: mp.write_text(json.dumps(manifest,indent=2))
    xall,_,meta,scaler,cons=rx.dataset_data('lcld_v2_iid')
    def victim(model):
        cpu,adapter,_=rx.victims('lcld_v2_iid',model,meta,scaler,cons,xall[:32])
        return cpu,adapter,BatchJudge(cpu,cons,scaler,.5,box_tol_scale=1,rel_tol=1e-7)
    cpu,adapter,judge=victim('tabnet_default')
    results=[]
    for arm in ARMS:
        results.append(run_one('tabnet_default',arm,0,meta,scaler,cons,cpu,adapter,judge))
    ranked=sorted(results,key=lambda r:(-r['S'],r['fwd_mean'],r['arm']))
    selected=[r['arm'] for r in ranked[:3]]
    print('PROMOTE_TO_FIVE_SEEDS',json.dumps(selected),flush=True)
    verified=[]
    for arm in selected:
        for seed in range(1,5):run_one('tabnet_default',arm,seed,meta,scaler,cons,cpu,adapter,judge)
        verified.append(verify('tabnet_default',arm,range(5),meta,scaler,cons,cpu,judge))
    passed=sorted([r for r in verified if r['mean_S']>996],key=lambda r:(-r['mean_S'],r['arm']))
    chosen=passed[0]['arm'] if passed else None
    promotion=dict(seed0_ranking=[{k:r[k] for k in ['arm','S','fwd_mean']} for r in ranked],
        five_seed_results=verified,selected=chosen,development_pool_reused=True)
    (OUT/'promotion.json').write_text(json.dumps(promotion,indent=2))
    del cpu,adapter,judge;gc.collect();torch.cuda.empty_cache()
    if chosen:
        for model in ['vime_default','tabtransformer_default','torchrln_default','stg_default']:
            cpu,adapter,judge=victim(model)
            run_one(model,chosen,0,meta,scaler,cons,cpu,adapter,judge)
            verify(model,chosen,[0],meta,scaler,cons,cpu,judge)
            del cpu,adapter,judge;gc.collect();torch.cuda.empty_cache()
    print('TUNE100_COMPLETE',flush=True)

if __name__=='__main__':main()
