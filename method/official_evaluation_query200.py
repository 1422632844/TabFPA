"""LOCAL WRAPPER / AUDIT: parameterized full-pool evaluation of the official kernel."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'method'), str(ROOT)]
import runtime
from tabularbench.constraints.constraints_checker import ConstraintChecker
from benchmark_adapter import get_local_model_path
from tabularbench.models.aliases import ALIASES as ARCH_ALIASES
from tabularbench.dataloaders.aliases import ALIASES as TRAINING_ALIASES
import widen_kernel_query200 as kernel
from official_defaults import resolve


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(p, data):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def event(**kw):
    print(json.dumps(kw), flush=True)


class Evaluation:
    def __init__(self, dataset, name, out):
        assert dataset in ('url', 'wids'), 'This audited full-pool entry supports URL and native WiDS.'
        assert not any(os.environ.get(k) for k in ('WK_PATIENCE','WK_AUDIT_TOL','WIDS_APACHE_MAP'))
        self.dataset, self.name, self.out = dataset, name, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        event(event='loading_model', dataset=dataset, model=name)
        ds, self.meta, self.scaler, self.model, self.cons, self.adapter, self.judge = runtime.components(dataset, name)
        frame, labels = ds.get_x_y()
        all_x, labels = frame.to_numpy(), np.asarray(labels)
        indices = np.asarray(ds.get_splits()['test'])
        indices = indices[labels[indices] == 1]
        legal = np.concatenate([self.judge.constraint_ok(all_x[indices[i:i+128]], all_x[indices[i:i+128]])
                                for i in range(0, len(indices), 128)])
        indices = indices[legal]
        self.D = len(indices)
        pred = np.concatenate([self.model.predict_proba(all_x[indices[i:i+128]]).argmax(1)
                               for i in range(0, len(indices), 128)])
        self.indices = indices[pred == 1]
        self.clean = all_x[self.indices]
        self.N = len(self.clean)
        assert self.D == {'url':1143, 'wids':617}[dataset]
        self.sources = {}
        for module in list(sys.modules.values()):
            p = getattr(module, '__file__', None)
            if p and (str(ROOT) in str(p) or getattr(module, '__name__', '').startswith('tabularbench')) and str(p).endswith('.py') and '/runs/' not in str(p):
                self.sources[str(Path(p).resolve())] = sha(p)
        self.sources[str(ROOT/'method/official_defaults.json')] = sha(ROOT/'method/official_defaults.json')
        arch, training = name.split('_', 1)
        weight_path = get_local_model_path(dataset, ARCH_ALIASES.get(arch, arch), TRAINING_ALIASES.get(training, training))
        if not weight_path:
            raise FileNotFoundError('Provide local pretrained checkpoints via TABULARBENCH_TARGET_MODEL_DIR')
        weights = Path(weight_path)
        for p in weights.iterdir():
            if p.is_file(): self.sources[str(p)] = sha(p)
        pool = dict(D=self.D, N=self.N, indices_sha256=array_sha(self.indices), clean_sha256=array_sha(self.clean))
        pp = self.out/'pool.npz'
        if pp.exists():
            with np.load(pp, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['indices'], self.indices)
                np.testing.assert_array_equal(z['clean'], self.clean)
        else:
            np.savez_compressed(pp, clean=self.clean, indices=self.indices, D=self.D, N=self.N)
        save(self.out/'manifest.json', dict(dataset=dataset, model=name, pool=pool,
             source_sha256=self.sources, batch=128, eps=.5, query_cap=200, initial_clean_queries=1, query_counting="attack scoring plus CPU certification; excludes initial clean and offline final audits", constraint_tolerance=0.,
             seed_rule='seed * 1000003 + batch offset', selection='full native legal positive test pool; per-model clean-correct subset; no truncation'))
        event(event='pool_verified', dataset=dataset, model=name, **pool)
        self.checker = ConstraintChecker(self.cons, tolerance=0.)

    def verify_sources(self):
        for p,h in self.sources.items(): assert sha(p) == h, 'Source/checkpoint changed: ' + p

    def run(self, seed, parameters=None, tag='official'):
        assert isinstance(seed, int) and 0 <= seed <= 4
        assert tag.replace('_','').isalnum()
        kw = resolve(self.dataset, self.name) if parameters is None else parameters.copy()
        assert set(kw) == set(resolve(self.dataset,self.name))
        assert 200 <= kw['steps'] <= 250 and kw['state_policy'] == 'released' and kw['cont_param'] == 'plain'
        path = self.out/f'{tag}__s{seed}.json'
        if path.exists():
            r = json.loads(path.read_text())
            assert r.get('attack_query_cap') == 200 and r.get('attack_queries_max', 999) <= 200, 'Existing result lacks the required query-budget audit; use a fresh output directory.'
            assert r['parameters'] == kw and r['seed'] == seed
            assert sha(path.with_suffix('.npz')) == r['artifact_sha256']
            self.verify_sources()
            event(event='reuse_complete', model=self.name, seed=seed, tag=tag, RA=r['RA'])
            return r
        arrays = {k:[] for k in ('adv','ok','fwd','bwd','visits')}
        batches = []
        for offset in range(0,self.N,128):
            x = self.clean[offset:offset+128]
            event(event='batch_start', model=self.name, tag=tag, seed=seed, offset=offset)
            meter,row = kernel.run(self.dataset,self.name,self.meta,self.scaler,self.cons,self.model,
                                   self.adapter,self.judge,x,seed*1000003+offset,query_budget=200,**kw)
            assert np.asarray(self.checker.check_constraints(x,meter.out)).all()
            ok,dist,_,_ = self.judge(x,np.ones(len(x),dtype=np.int64),meter.out)
            np.testing.assert_array_equal(ok,meter.ok)
            assert np.isfinite(meter.out).all() and (dist <= .5).all()
            assert (meter.fwd <= 201).all() and (meter.optimization_visits <= kw['steps']).all()
            for k,v in zip(arrays,(meter.out,meter.ok,meter.fwd,meter.bwd,meter.optimization_visits)):
                arrays[k].append(v.copy())
            batches.append(row)
            event(event='batch_complete', model=self.name, tag=tag, seed=seed, offset=offset,
                  S=int(meter.ok.sum()), seconds=row['seconds'])
        arrays = {k:np.concatenate(v) for k,v in arrays.items()}
        S = int(arrays['ok'].sum())
        r = dict(dataset=self.dataset,model=self.name,seed=seed,tag=tag,parameters=kw,
                 D=self.D,N=self.N,S=S,RA=100.*(self.N-S)/self.D,
                 forward_mean=float(arrays['fwd'].mean()),forward_max=int(arrays['fwd'].max()),
                 attack_query_cap=200, attack_queries_max=int((arrays['fwd']-1).max()),
                 seconds=sum(b['seconds'] for b in batches),batches=batches,
                 constraints_checked=True,budget_checked=True,source_sha256=self.sources)
        self.verify_sources()
        np.savez_compressed(path.with_suffix('.npz'),**arrays,indices=self.indices)
        r['artifact_sha256']=sha(path.with_suffix('.npz'))
        save(path,r)
        event(event='run_complete', **{k:r[k] for k in ('model','tag','seed','D','N','S','RA')})
        return r
