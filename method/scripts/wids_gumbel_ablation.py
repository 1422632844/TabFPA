"""WiDS matched-pool Gumbel/QP ordering experiment. No CEM imports or calls.

Main Gumbel = all integer fields on bounded per-record grids, continuous
fields optimized by Adam. A separate legacy-map control uses the existing
GumbelInt + WidsFeasibleMap. Conditional completion fixes integer choices;
its backward pass is explicitly BPDA. All results share per-target pools.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core.data import DEVICE
from core.judge import BatchJudge, edit_counts
from core.qp_init import chain_grads, jacobian_z, delta_bounds, solve_qp_rows, line_search
from core.gumbel_int import GumbelInt
from core.wids_completion import WidsCompletion
from models import load_victim, DEFAULT_MODELS
from tabdatasets.wids import WidsFeasibleMap, ORDER_PAIRS


class Policy:
    def __init__(self,x0,meta,scaler,eps,initial=None,legacy=False):
        self.x0=x0; self.meta=meta; self.legacy=legacy
        self.n,self.nf=x0.shape
        self.types=np.asarray(meta["type"])
        self.ints=np.flatnonzero(self.types=="int")
        self.reals=np.flatnonzero(self.types=="real")
        self.scale=torch.ones(self.nf,device=x0.device)
        self.scale[scaler.num_idx]=(scaler.x_max-scaler.x_min).to(x0.device)
        self.scale=self.scale.clamp_min(1e-9)
        self.lo=torch.tensor(meta["min"].to_numpy(),dtype=torch.float32,device=x0.device)
        self.hi=torch.tensor(meta["max"].to_numpy(),dtype=torch.float32,device=x0.device)
        self.fmap=WidsFeasibleMap(meta,x0.device)
        self.free=torch.nonzero(self.fmap.free_mask).squeeze(1)
        if legacy:
            self.d=torch.zeros(self.n,len(self.free),device=x0.device,requires_grad=True)
            self.g=GumbelInt(self.fmap,meta,x0,x0.device,cont_levels=21)
            self.params=[self.d,self.g.theta]
        else:
            radius=torch.floor(eps*self.scale[self.ints])
            offset=torch.round(radius[:,None]*torch.linspace(-1,1,31,device=x0.device))
            self.values=x0[:,self.ints,None]+offset[None]
            unique=torch.ones_like(offset,dtype=torch.bool)
            unique[:,1:]=offset[:,1:]!=offset[:,:-1]
            self.mask=unique[None] & (self.values>=self.lo[self.ints,None]) & (
                                     self.values<=self.hi[self.ints,None])
            target=x0 if initial is None else torch.tensor(initial,device=x0.device,dtype=x0.dtype)
            gap=(self.values-target[:,self.ints,None])/self.scale[self.ints,None]
            own=self.values==x0[:,self.ints,None]
            theta=-40*gap.abs()+2*own
            self.theta=theta.detach().requires_grad_(True)
            self.d=((target[:,self.reals]-x0[:,self.reals])/self.scale[self.reals]).detach().requires_grad_(True)
            self.params=[self.theta,self.d]

    def sample(self,step,steps):
        tau=1.-.9*step/max(steps-1,1)
        if self.legacy:
            d=self.d.clone()
            if self.g.n:
                d[:,self.g.cols]=self.g.propose(tau)
            raw=self.x0.clone()
            v=self.x0[:,self.free]+d*self.fmap.half[self.free]
            v=v.clamp(self.lo[self.free],self.hi[self.free])
            it=torch.tensor(self.types[self.free.cpu().numpy()]!="real",device=v.device)
            v=v+(torch.round(v)-v).detach()*it
            raw[:,self.free]=v
            return self.fmap(self.x0,d),raw
        u=torch.rand_like(self.theta).clamp(1e-7,1-1e-7)
        noise=-torch.log(-torch.log(u))
        soft=((self.theta.masked_fill(~self.mask,-1e9)+noise)/tau).softmax(-1)
        hard=torch.zeros_like(soft).scatter_(-1,soft.argmax(-1,keepdim=True),1.)
        weights=hard-soft.detach()+soft
        p=self.x0.clone()
        p[:,self.ints]=(weights*self.values).sum(-1)
        p[:,self.reals]=(self.x0[:,self.reals]+self.d*self.scale[self.reals]).clamp(
                         self.lo[self.reals],self.hi[self.reals])
        return p,p


class Meter:
    def __init__(self,adapter,model,judge,x0,y,device,budget,batch):
        self.adapter=adapter;self.model=model;self.judge=judge
        self.x0=x0;self.y=y;self.device=device;self.budget=budget;self.batch=batch
        self.fwd=np.zeros(len(x0),dtype=int);self.bwd=np.zeros(len(x0),dtype=int)
        self.out=x0.copy();self.ok=np.zeros(len(x0),dtype=bool)
        self.best_raw=x0.copy();self.best_margin=np.full(len(x0),-np.inf)
        self.trace=[];self.seen=self.raw_valid=self.eval_valid=0
        self.rewritten=self.raw_changed=0
        self.start=time.perf_counter()
    def score(self,x,ids):
        self.fwd[ids]+=1
        return torch.cat([self.adapter(x[s:s+self.batch]) for s in range(0,len(x),self.batch)])
    def valid(self,p,ids):
        cons=self.judge.constraint_ok(self.x0[ids],p)
        with torch.no_grad():
            xc=torch.as_tensor(self.x0[ids],dtype=torch.float32,device=self.device)
            xx=torch.as_tensor(p,dtype=torch.float32,device=self.device)
            dist=(self.judge.scaler.transform(xx)-self.judge.scaler.transform(xc)).norm(dim=1).cpu().numpy()
        return cons & (dist <= self.judge.eps*.99 + 1e-6)
    def accept(self,p,ids,logits,allowed=None):
        pred=logits.detach().argmax(1).cpu().numpy()
        valid=self.valid(p,ids)
        if allowed is not None:valid &= allowed
        hit=(pred != self.y[ids]) & valid & ~self.ok[ids] & (self.fwd[ids]<self.budget)
        if hit.any():
            ii=ids[hit];self.fwd[ii]+=1
            verified,_,_,_=self.judge(self.x0[ii],self.y[ii],p[hit].astype(np.float64))
            self.out[ii[verified]]=p[hit][verified]
            self.ok[ii[verified]]=True
        return valid


def qp_seed(x,y,meta,scaler,adapter,model,judge,args):
    meter=Meter(adapter,model,judge,x,y,DEVICE,args.queries,args.batch)
    x0=torch.tensor(x,dtype=torch.float32,device=DEVICE)
    yt=torch.tensor(y,dtype=torch.long,device=DEVICE)
    fmap=WidsFeasibleMap(meta,DEVICE)
    ids=np.arange(len(x))
    def wrap(xx):return meter.score(xx,ids)
    t=time.perf_counter()
    g,_=chain_grads(fmap,scaler,wrap,x0,yt);meter.bwd+=1
    J=jacobian_z(fmap,scaler,x0)
    lb,ub=delta_bounds(fmap,meta,x0)
    d,fail=solve_qp_rows(g,J,lb,ub,args.eps)
    z0=scaler.transform(x0).detach()
    _,_,bx,_=line_search(fmap,scaler,wrap,x0,z0,d,args.eps*.99,yt)
    p=bx.cpu().numpy().astype(np.float64)
    with torch.no_grad():lg=meter.score(bx,ids)
    meter.accept(p,ids,lg)
    elapsed=time.perf_counter()-t
    return dict(x=p,out=meter.out,ok=meter.ok,fwd=meter.fwd,bwd=meter.bwd,
                seconds=elapsed,solver_fallbacks=fail)


def run_variant(name,seed,x,y,meta,scaler,adapter,model,judge,qp,args):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    meter=Meter(adapter,model,judge,x,y,DEVICE,args.queries,args.batch)
    x0=torch.tensor(x,dtype=torch.float32,device=DEVICE)
    y0=torch.tensor(y,dtype=torch.long,device=DEVICE)
    fixed=None
    if name=="qp_gumbel":
        fixed=qp["x"]
        meter.out=qp["out"].copy();meter.ok=qp["ok"].copy()
        meter.fwd=qp["fwd"].copy();meter.bwd=qp["bwd"].copy()
    policy=Policy(x0,meta,scaler,args.eps,initial=fixed,legacy=name=="legacy_gumbel")
    if name in ("continuous_inner","frozen_gumbel_inner"):
        policy.theta.requires_grad_(False)
        policy.params=[policy.d]
    opt=torch.optim.Adam(policy.params,lr=args.lr)
    complete=WidsCompletion(meta,scaler,args.eps*.99)
    qseconds=qp["seconds"] if name=="qp_gumbel" else 0.
    start=time.perf_counter()
    for step in range(args.steps):
        ids=np.flatnonzero(~meter.ok & (meter.fwd < args.queries-2) &
                           (meter.bwd < args.steps))
        if not len(ids):break
        p_all,raw_all=policy.sample(step,args.steps)
        p=p_all[ids];raw=raw_all[ids]
        if name=="continuous_inner":
            p=p.clone()
            p[:,policy.ints]=x0[ids][:,policy.ints]
            raw=p
        allowed=np.ones(len(ids),dtype=bool)
        if name in ("gumbel_inner","continuous_inner","frozen_gumbel_inner"):
            p,allowed_t=complete.torch(x0[ids],p)
            allowed=allowed_t.cpu().numpy()
        lg=meter.score(p,ids);meter.bwd[ids]+=1
        ar=torch.arange(len(ids),device=DEVICE)
        atk=(lg[ar,y0[ids]]-lg[ar,1-y0[ids]]).mean()
        # Shared proposal penalties; projection placement is the treatment.
        dz=(raw-x0[ids])/policy.scale
        dist=dz[:,scaler.num_idx].norm(dim=1)
        budget=torch.relu(dist-.9*args.eps).square().mean()
        a,b=np.asarray(ORDER_PAIRS).T
        order=torch.relu((raw[:,a]-raw[:,b])/
                         torch.maximum(policy.scale[a],policy.scale[b])).square().sum(1).mean()
        loss=atk+4*budget+4*order
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            pp=p.detach().cpu().numpy().astype(np.float64)
            rr=raw.detach().cpu().numpy().astype(np.float64)
            margin=(lg[ar,1-y0[ids]]-lg[ar,y0[ids]]).cpu().numpy()
            # Terminal completion uses the best unprojected-margin proposal.
            update=margin>meter.best_margin[ids]
            meter.best_margin[ids[update]]=margin[update]
            meter.best_raw[ids[update]]=rr[update]
            rv=meter.valid(rr,ids)
            ev=meter.accept(pp,ids,lg,allowed)
            ints=policy.types=="int"
            meter.raw_valid+=int(rv.sum());meter.eval_valid+=int(ev.sum())
            meter.seen+=len(ids)
            meter.rewritten+=int(((rr[:,ints]!=pp[:,ints]).any(1) & allowed).sum())
            meter.trace.append(dict(step=step+1,accepted=int(meter.ok.sum()),
                         forward_total=int(meter.fwd.sum()),backward_total=int(meter.bwd.sum())))
    terminal_gain=0
    if name=="gumbel_terminal":
        ids=np.flatnonzero(~meter.ok & (meter.fwd<args.queries-1))
        if len(ids):
            projected,valid=complete.solve(x[ids],meter.best_raw[ids])
            feasible_ids=ids[valid]
            if len(feasible_ids):
                xx=torch.tensor(projected[valid],dtype=torch.float32,device=DEVICE)
                with torch.no_grad():lg=meter.score(xx,feasible_ids)
                before=int(meter.ok.sum())
                meter.accept(xx.cpu().numpy().astype(np.float64),feasible_ids,lg)
                terminal_gain=int(meter.ok.sum())-before
    if torch.cuda.is_available():torch.cuda.synchronize()
    seconds=time.perf_counter()-start+qseconds
    return meter,dict(seconds=seconds,qp_accounted_seconds=qseconds,
                      completion_calls=complete.calls,completion_rows=complete.rows,
                      completion_rejected=complete.rejected,terminal_gain=terminal_gain)


def check_and_record(meter,extra,name,seed,model_name,x,y,meta,scaler,judge,args):
    # Independent FINAL AUDIT: these queries are not used by the attack.
    ok,dist,viol,_=judge(x,y,meter.out)
    from tabularbench.constraints.constraints_checker import ConstraintChecker
    official=ConstraintChecker(judge.constraints,tolerance=0).check_constraints(x,meter.out).astype(bool)
    np.testing.assert_array_equal(official,viol==0)
    np.testing.assert_array_equal(ok,meter.ok)
    assert (meter.fwd <= args.queries).all()
    assert (meter.bwd <= args.steps).all()
    free=np.flatnonzero(np.asarray(meta["type"])!="cat").tolist()
    action,semantic,_=edit_counts(x,meter.out,meta,free)
    return dict(model=model_name,method=name,seed=seed,n=len(x),batch_size=args.batch,
          accepted=int(ok.sum()),asr=float(ok.mean()),
          forward_per_sample=float(meter.fwd.mean()),forward_max=int(meter.fwd.max()),
          backward_per_sample=float(meter.bwd.mean()),
          audit_forward_per_sample=1,
          candidate_n=meter.seen,raw_feasible_n=meter.raw_valid,
          evaluated_feasible_n=meter.eval_valid,
          integer_rewritten_n=meter.rewritten,
          action_mean_success=float(action[ok].mean()) if ok.any() else None,
          semantic_mean_success=float(semantic[ok].mean()) if ok.any() else None,
          distance_max=float(dist.max()),invalid_submitted=int(viol.sum()),
          trace=meter.trace,**extra)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=0,help="0 = all per-target eligible critical rows")
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--queries",type=int,default=256)
    p.add_argument("--eps",type=float,default=.5)
    p.add_argument("--lr",type=float,default=.06)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--models",nargs="+",default=DEFAULT_MODELS)
    p.add_argument("--out",default="runs/wids_no_cem_ablation")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path("core/wids_completion.py"),Path("core/qp_init.py"),
             Path("core/gumbel_int.py"),Path("core/judge.py"),Path("models/__init__.py")]
    manifest=dict(config=vars(args),device=DEVICE,cem=False,
      protocol="per-target class1 & clean-correct & strict-clean-valid; no L0 cap",
      proposal="main: all-int per-record grids(max31), clean-centred prior; real Adam; cats fixed",
      backward="completion uses BPDA identity, not implicit QCQP differentiation",
      qp_cache="QP deterministic per model; reused across seeds; full measured setup charged to qp_gumbel",
      sources={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
    results=[];archive={};completed=set()
    if args.resume:
        previous=json.loads((out/"manifest.json").read_text())
        previous_rows=json.loads((out/"results.json").read_text())
        with np.load(out/"adversarial.npz") as saved:
            archive={k:saved[k] for k in saved.files}
        completed={m for m in args.models
                   if all(m+"__frozen_gumbel_inner__s"+str(seed)+"__ok" in archive
                          for seed in args.seeds)}
        results=[dict(r,batch_size=r.get("batch_size",previous["config"]["batch"]))
                 for r in previous_rows if r["model"] in completed]
        manifest["previous_run"]=previous
        manifest["resumed_completed_models"]=sorted(completed)
        manifest["discarded_uncheckpointed_records"]=len(previous_rows)-len(results)
        (out/"results.json").write_text(json.dumps(results,indent=2))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    for model_name in args.models:
        if model_name in completed:
            print(json.dumps(dict(event="resume_skip",model=model_name)),flush=True)
            continue
        ds,xt,yt,scaler,model,cons,adapter=load_victim("wids",model_name,DEVICE)
        meta=ds.get_metadata(only_x=True).reset_index(drop=True)
        judge=BatchJudge(model,cons,scaler,args.eps,box_tol_scale=1000,rel_tol=0.)
        pred=np.asarray(model.predict_proba(xt)).argmax(1)
        eligible=(yt==1)&(pred==yt)&judge.constraint_ok(xt,xt)
        idx=np.flatnonzero(eligible)
        if args.n:idx=idx[:args.n]
        x=xt[idx].astype(np.float64);y=yt[idx].astype(np.int64)
        assert len(x)>0
        with torch.no_grad():
            xp=torch.tensor(x[:32],dtype=torch.float32,device=DEVICE)
            np.testing.assert_array_equal(adapter(xp).argmax(1).cpu().numpy(),
                                          np.asarray(model.predict_proba(x[:32])).argmax(1))
        archive[model_name+"__clean"]=x;archive[model_name+"__labels"]=y
        archive[model_name+"__test_indices"]=idx
        print(json.dumps(dict(event="model_start",model=model_name,n=len(x))),flush=True)
        qp=qp_seed(x,y,meta,scaler,adapter,model,judge,args)
        qm=Meter(adapter,model,judge,x,y,DEVICE,args.queries,args.batch)
        qm.out=qp["out"];qm.ok=qp["ok"];qm.fwd=qp["fwd"];qm.bwd=qp["bwd"]
        r=check_and_record(qm,dict(seconds=qp["seconds"],solver_fallbacks=qp["solver_fallbacks"]),
                           "qp_only",None,model_name,x,y,meta,scaler,judge,args)
        results.append(r)
        archive[model_name+"__qp_only__adv"]=qm.out
        archive[model_name+"__qp_only__ok"]=qm.ok
        for seed in args.seeds:
            for name in ["gumbel","qp_gumbel","gumbel_terminal","gumbel_inner","legacy_gumbel","continuous_inner","frozen_gumbel_inner"]:
                meter,extra=run_variant(name,seed,x,y,meta,scaler,adapter,model,judge,qp,args)
                r=check_and_record(meter,extra,name,seed,model_name,x,y,meta,scaler,judge,args)
                results.append(r)
                key=model_name+"__"+name+"__s"+str(seed)
                archive[key+"__adv"]=meter.out;archive[key+"__ok"]=meter.ok
                archive[key+"__forward"]=meter.fwd;archive[key+"__backward"]=meter.bwd
                (out/"results.json").write_text(json.dumps(results,indent=2))
                print(json.dumps({k:v for k,v in r.items() if k!="trace"}),flush=True)
        np.savez_compressed(out/"adversarial.npz",**archive)
        print(json.dumps(dict(event="model_done",model=model_name)),flush=True)
        del adapter,model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    (out/"results.json").write_text(json.dumps(results,indent=2))
    print("EXPERIMENT_COMPLETE",flush=True)

if __name__=="__main__":main()
