"""CTU fixed-discrete LR comparison; no extra parameter search."""
import os,sys,json,hashlib,time,gc,argparse,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scripts import url_wids_gumbel_tuning as shared
from core.conditional_completion import ConditionalCompletion
from core.judge import BatchJudge
from models import VictimAdapter,DEFAULT_MODELS
from tabdatasets.ctu_13_neris import CtuFeasibleMap
from tabularbench.benchmark.model_utils import load_model_and_weights
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.constraints.constraints import get_constraints_from_metadata
from tabularbench.constraints.constraints_checker import ConstraintChecker
from tabularbench.datasets.samples.ctu_13_neris import get_relation_constraints
from tabularbench.datasets.dataset import CsvDataSource
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ['TABULARBENCH_DATASET_DIR'])/'ctu_13_neris'
OUT=ROOT/'runs/ctu_gumbel_lr'
ARGS=shared.ARGS
CONFIGS={k:shared.CONFIGS[k] for k in ['current','logits_lr_012']}
RESULTS=[];ARCHIVE={}
class Policy:
    def __init__(self, x0, meta, scaler):
        self.x0 = x0
        self.types = np.asarray(meta["type"])
        mutable=np.asarray(meta["mutable"],bool)
        self.ints = np.flatnonzero((self.types == "int") & mutable)
        self.reals = np.flatnonzero((self.types == "real") & mutable)
        self.scale = torch.ones(x0.shape[1], device=x0.device)
        self.scale[scaler.num_idx] = (scaler.x_max - scaler.x_min).to(x0.device)
        self.scale = self.scale.clamp_min(1e-9)
        self.lo = torch.tensor(meta["min"].to_numpy(), dtype=x0.dtype, device=x0.device)
        self.hi = torch.tensor(meta["max"].to_numpy(), dtype=x0.dtype, device=x0.device)
        radius = torch.floor(ARGS.eps * self.scale[self.ints])
        offset = torch.round(radius[:, None] * torch.linspace(-1, 1, 31, device=x0.device))
        self.values = x0[:, self.ints, None] + offset[None]
        unique = torch.ones_like(offset, dtype=torch.bool)
        unique[:, 1:] = offset[:, 1:] != offset[:, :-1]
        self.mask = unique[None] & (self.values >= self.lo[self.ints, None]) & (
            self.values <= self.hi[self.ints, None])
        gap = (self.values - x0[:, self.ints, None]) / self.scale[self.ints, None]
        self.theta = (-40 * gap.abs() + 2 * (self.values == x0[:, self.ints, None])).detach().requires_grad_(True)
        self.d = torch.zeros_like(x0[:, self.reals], requires_grad=True)

    def sample(self, step, steps, cfg):
        tau = cfg["tau_start"] + (cfg["tau_end"]-cfg["tau_start"]) * step/max(steps-1, 1)
        u = torch.rand_like(self.theta).clamp(1e-7, 1-1e-7)
        noise = -torch.log(-torch.log(u))
        soft = ((self.theta.masked_fill(~self.mask, -1e9)+noise)/tau).softmax(-1)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.)
        weights = hard-soft.detach()+soft
        p = self.x0.clone()
        p[:, self.ints] = (weights*self.values).sum(-1)
        p[:, self.reals] = (self.x0[:, self.reals]+self.d*self.scale[self.reals]).clamp(
            self.lo[self.reals], self.hi[self.reals])
        return p



class CtuCompletion(ConditionalCompletion):
    def _convex(self,x0,p):
        out,valid=super()._convex(x0,p)
        real=set(self.real)
        # Reconstruct tied active order constraints exactly in raw float64.
        # Only ULP-sized roundoff is corrected; the strict checker is unchanged.
        for i,j in [(1,2),(0,1),(1,2)]:
            for g in self.groups:
                a,b=g[i],g[j]
                gap=out[:,a]-out[:,b]
                tol=8*np.finfo(float).eps*np.maximum(1,np.maximum(abs(out[:,a]),abs(out[:,b])))
                tiny=(gap>0)&(gap<=tol)
                if a in real:out[tiny,a]=out[tiny,b]
                elif b in real:out[tiny,b]=out[tiny,a]
        return out,valid

def data():
    csv=DATA/'ctu_13_neris.csv'
    names=pd.read_csv(csv,nrows=0).columns.drop('is_botnet').tolist()
    meta=pd.read_csv(DATA/'ctu_13_neris_metadata.csv')
    meta=meta[meta.feature.isin(names)].reset_index(drop=True)
    assert meta.feature.tolist()==names
    lo=np.full(len(names),np.inf);hi=-lo.copy()
    xs=[];ix=[];cursor=0
    # Exact full-data float32 min/max scaler, computed in chunks.
    for part in pd.read_csv(csv,chunksize=2048,low_memory=False):
        values=part[names].to_numpy(dtype=np.float32)
        lo=np.minimum(lo,values.min(0));hi=np.maximum(hi,values.max(0))
        ids=np.arange(cursor,cursor+len(part))
        keep=(ids>=143046)&(part.is_botnet.to_numpy()==1)
        xs.append(part.loc[keep,names].to_numpy(dtype=np.float64));ix.append(ids[keep])
        cursor+=len(part)
    assert not (meta.type=='cat').any()
    scaler=TabScaler(num_scaler='min_max',one_hot_encode=True)
    scaler.fit(torch.tensor(np.stack([lo,hi]),dtype=torch.float32),x_type=meta.type)
    cons=get_constraints_from_metadata(meta,get_relation_constraints(CsvDataSource(str(DATA/'ctu_13_neris_metadata.csv'))),names)
    assert np.asarray(cons.lower_bounds,dtype=float).shape==(len(names),)
    print(json.dumps(dict(event='data',rows=cursor,positive_test=sum(map(len,xs)),
                         fields=len(names),mutable_int=int(((meta.type=='int')&meta.mutable).sum()),
                         mutable_real=int(((meta.type=='real')&meta.mutable).sum()),
                         rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
    return np.concatenate(xs),np.concatenate(ix),meta,scaler,cons

def victim(model_name,meta,scaler,x):
    arch,training=model_name.split('_',1)
    cpu=load_model_and_weights('ctu_13_neris',arch,training,meta,scaler,'cpu')
    gpu=load_model_and_weights('ctu_13_neris',arch,training,meta,scaler,'cuda')
    gpu.wrapper_model.eval();gpu.wrapper_model.requires_grad_(False)
    adapter=VictimAdapter(gpu.wrapper_model,scaler,False).cuda()
    probe=x[:32]
    with torch.no_grad():out=adapter(torch.tensor(probe,dtype=torch.float32,device='cuda')).cpu().numpy()
    prob=cpu.predict_proba(probe)
    assert (out.argmax(1)==prob.argmax(1)).all()
    judge=BatchJudge(cpu,cons,scaler,.5,rel_tol=0.)
    return cpu,adapter,judge

def penalty_parts(meta,scale):
    fmap=CtuFeasibleMap(meta,'cpu');a=[];eq=[]
    for left,right in fmap.order_pairs:
        v=np.zeros(len(meta));v[left]=1;v[right]-=1;a.append(v);eq.append(False)
    for e in fmap.eq:
        v=np.zeros(len(meta));v[e['lhs']]+=1;v[e['rhs']]-=1;a.append(v);eq.append(True)
    A=torch.tensor(np.array(a),dtype=torch.float32,device='cuda')
    norm=(A.abs()*scale[None]).max(1).values.clamp_min(1e-9)
    pk=np.asarray(fmap.pk_pairs)
    return A,norm,torch.tensor(eq,device='cuda'),pk

def reference_check(x,meta,scaler,judge):
    c=CtuCompletion('ctu_13_neris',meta,scaler,judge,.495)
    rng=np.random.RandomState(123)
    proposed=x[:3].copy()
    proposed[:,c.real]=np.clip(proposed[:,c.real]+rng.normal(size=(3,len(c.real)))*c.scale[c.real]*.4,
                              c.lo[c.real],c.hi[c.real])
    out,valid=c.solve(x[:3],proposed);assert valid.all()
    np.testing.assert_array_equal(out[:,c.fixed],proposed[:,c.fixed])
    errors=[]
    for i in range(3):
        r=c.real;sr=c.scale[r];centre=x[i]
        desired=(proposed[i,r]-centre[r])/sr
        def rebuild(u):
            z=proposed[i].copy();z[r]=centre[r]+u*sr;return z
        def orders(u):
            z=rebuild(u);v=z[c.groups]
            return ((v[:,1:]-v[:,:-1])/np.maximum(c.scale[c.groups[:,1:]],c.scale[c.groups[:,:-1]])).ravel()
        # Constant rows arise when all three duration fields are integer.
        jac=np.zeros((len(c.groups)*2,len(r)))
        pos={j:k for k,j in enumerate(r)}
        for gi,g in enumerate(c.groups):
            for j in range(2):
                den=max(c.scale[g[j]],c.scale[g[j+1]])
                if g[j] in pos:jac[gi*2+j,pos[g[j]]]-=c.scale[g[j]]/den
                if g[j+1] in pos:jac[gi*2+j,pos[g[j+1]]]+=c.scale[g[j+1]]/den
        fit=minimize(lambda u:((u-desired)**2).sum(),np.zeros(len(r)),jac=lambda u:2*(u-desired),
             bounds=list(zip((c.lo[r]-centre[r])/sr,(c.hi[r]-centre[r])/sr)),
             constraints=[dict(type='ineq',fun=lambda u:.495**2-u@u,jac=lambda u:-2*u),
                          dict(type='ineq',fun=orders,jac=lambda u:jac)],
             method='SLSQP',options=dict(maxiter=300,ftol=1e-11))
        assert fit.success,fit.message
        ours=(((out[i,r]-proposed[i,r])/sr)**2).sum()
        errors.append(abs(ours-fit.fun))
    assert max(errors)<1e-6,errors
    bad=x[:1].copy()
    # Perturb a mutable protocol count to violate a fixed-integer equality.
    j=next(j for j,n in enumerate(c.names) if n.startswith('tcp_sum_') and c.mutable[j] and c.hi[j]>bad[0,j]+1)
    bad[0,j]+=1
    rejected,valid=c.solve(x[:1],bad)
    assert not valid.any();np.testing.assert_array_equal(rejected,x[:1])
    print(json.dumps(dict(event='completion_reference_pass',max_objective_error=max(errors))),flush=True)

def run(model_name,x,meta,scaler,model,adapter,judge,cfg,seed,steps=200):
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    y=np.ones(len(x),dtype=np.int64)
    meter=shared.base.Meter(adapter,model,judge,x,y,'cuda',ARGS.queries,ARGS.batch)
    x0=torch.tensor(x,dtype=torch.float32,device='cuda')
    p=Policy(x0,meta,scaler)
    opt=torch.optim.Adam([{'params':[p.theta],'lr':cfg['logits_lr']},
                          {'params':[p.d],'lr':.06}])
    c=ConditionalCompletion('ctu_13_neris',meta,scaler,judge,.495)
    A,norm,eq,pk=penalty_parts(meta,p.scale)
    imm=~meta.mutable.to_numpy(bool)
    integer_proposals=0;integer_valid=0;equality_invalid=0
    eq_checks=[shared.ConstraintsExecutor(e,shared.NumpyBackend(),cons.feature_names) for e in cons.relation_constraints[:2]]
    start=time.perf_counter()
    for step in range(steps):
        ids=np.flatnonzero(~meter.ok&(meter.fwd<254)&(meter.bwd<steps))
        if not len(ids):break
        raw=p.sample(step,steps,cfg)[ids]
        rr=raw.detach().cpu().numpy().astype(np.float64)
        # Preserve immutable originals in their stored precision.
        rr[:,imm]=x[ids][:,imm]
        equality_invalid+=int(np.logical_or.reduce([np.asarray(e.execute(rr))>0 for e in eq_checks]).sum())
        sol,allowed=c.solve(x[ids],rr)
        hard=torch.tensor(sol,dtype=raw.dtype,device='cuda')
        candidate=raw+(hard-raw).detach()
        logits=meter.score(candidate,ids);meter.bwd[ids]+=1
        atk=(logits[:,1]-logits[:,0]).mean()
        dz=(raw-x0[ids])/p.scale
        budget=torch.relu(dz[:,scaler.num_idx].norm(dim=1)-.45).square().mean()
        residual=(raw@A.T)/norm
        linear=torch.where(eq[None],residual,torch.relu(residual)).square().sum(1)
        bo=raw[:,pk[:,0]];po=raw[:,pk[:,1]]
        ratio=torch.where(po!=0,bo/torch.where(po!=0,po,torch.ones_like(po)),torch.zeros_like(po))
        packet=torch.relu(ratio/1500.-1).square().sum(1)
        loss=atk+4*budget+4*(linear+packet).mean()
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            meter.raw_valid+=int(meter.valid(rr,ids).sum())
            meter.eval_valid+=int(meter.accept(sol,ids,logits,allowed).sum());meter.seen+=len(ids)
            np.testing.assert_array_equal(sol[allowed][:,c.fixed],rr[allowed][:,c.fixed])
            np.testing.assert_array_equal(sol[:,imm],x[ids][:,imm])
            changed=(rr[:,p.ints]!=x[ids][:,p.ints]).any(1)
            integer_proposals+=int(changed.sum());integer_valid+=int((changed&allowed).sum())
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    ok,dist,viol,_=judge(x,y,meter.out)
    official=ConstraintChecker(cons,tolerance=0.).check_constraints(x,meter.out)
    np.testing.assert_array_equal(ok,meter.ok)
    assert official.all() and (dist<=.5).all() and meter.fwd.max()<=256
    return meter,dict(model=model_name,n=len(x),seed=seed,accepted=int(ok.sum()),asr=float(ok.mean()),
          forward_mean=float(meter.fwd.mean()),forward_max=int(meter.fwd.max()),
          backward_mean=float(meter.bwd.mean()),seconds=seconds,completion_rows=c.rows,
          completion_rejected=c.rejected,raw_valid=meter.raw_valid,eval_valid=meter.eval_valid,
          integer_changed_proposals=integer_proposals,integer_changed_completed=integer_valid,
          equality_invalid_proposals=equality_invalid,
          successful_integer_changed=int(((meter.out[:,p.ints]!=x[:,p.ints]).any(1)&ok).sum()),
          distance_max=float(dist.max()),audit_forward_per_sample=1,
          rss_peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          gpu_peak_mb=torch.cuda.max_memory_allocated()/1024**2)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    xall,indices,meta,scaler,cons=data()
    # Same strict executable checker as previous experiments.
    class NoModel:
        def predict_proba(self,x):raise RuntimeError('not used for constraint filtering')
    initial=BatchJudge(NoModel(),cons,scaler,.5,rel_tol=0.)
    mask=np.concatenate([initial.constraint_ok(xall[i:i+128],xall[i:i+128]) for i in range(0,len(xall),128)])
    denominator=int(mask.sum());xall=xall[mask];indices=indices[mask]
    order=np.random.RandomState(20260906).permutation(len(indices))
    xall=xall[order];indices=indices[order]
    models=['torchrln_default'] if args.smoke else DEFAULT_MODELS
    if not args.smoke:
        OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
        sources=[Path(__file__),ROOT/'scripts/url_wids_gumbel_tuning.py',ROOT/'core/conditional_completion.py',
                 ROOT/'core/judge.py',ROOT/'tabdatasets/ctu_13_neris.py',
                 Path(os.environ['TABULARBENCH_ROOT'])/'tabularbench/datasets/samples/ctu_13_neris.py',
                 DATA/'ctu_13_neris_metadata.csv']
        manifest=dict(dataset='ctu_13_neris',models=models,configs=CONFIGS,n=100,seeds=[0,1,2],
           pool='Official test rows index>=143046; class1; strict clean valid; per-target clean correct; shuffled seed20260906, first100.',
           valid_class1_test=denominator,query_cap=256,steps=200,eps=.5,completion_eps=.495,
           batch=128,threads=2,blas_threads=1,device='cuda:0',parameter_selection=False,
           adaptation='Mutable-only policy; immutable float64 originals; continuous duration order/ball completion with all integers fixed; affine equality/order plus native SafeDivision penalty; actual static native bounds.',
           sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for model_name in models:
        model,adapter,judge=victim(model_name,meta,scaler,xall)
        pred=np.concatenate([model.predict_proba(xall[i:i+128]).argmax(1) for i in range(0,len(xall),128)])
        eligible=np.flatnonzero(pred==1);chosen=eligible[:8 if args.smoke else 100]
        assert len(chosen)==(8 if args.smoke else 100)
        x=xall[chosen];ids=indices[chosen]
        print(json.dumps(dict(event='pool',model=model_name,n=len(x),eligible=len(eligible),valid_class1=denominator)),flush=True)
        if args.smoke:
            reference_check(x,meta,scaler,judge)
        for seed in ([0] if args.smoke else [0,1,2]):
            for setting,cfg in CONFIGS.items():
                meter,row=run(model_name,x,meta,scaler,model,adapter,judge,cfg,seed,6 if args.smoke else 200)
                row['setting']=setting
                print(json.dumps(row),flush=True)
                if not args.smoke:
                    RESULTS.append(row);prefix=model_name+'__'+setting+'__s'+str(seed)
                    ARCHIVE[model_name+'__indices']=ids;ARCHIVE[model_name+'__clean']=x
                    for key,value in [('adv',meter.out),('ok',meter.ok),('forward',meter.fwd),('backward',meter.bwd)]:
                        ARCHIVE[prefix+'__'+key]=value
                    (OUT/'results.json').write_text(json.dumps(RESULTS,indent=2))
        if not args.smoke:np.savez_compressed(OUT/'adversarial.npz',**ARCHIVE)
        del model,adapter,judge;gc.collect();torch.cuda.empty_cache()
    print('SMOKE_COMPLETE' if args.smoke else 'EXPERIMENT_COMPLETE',flush=True)
