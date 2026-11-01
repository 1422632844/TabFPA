"""Shared runtime for the released TabFPA experiments; no experiment driver."""
import os
import json
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core.data import DEVICE
from core.conditional_completion import ConditionalCompletion
from core.wids_completion import WidsCompletion
from core.judge import BatchJudge
from models import load_victim
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.constraints.constraints import get_constraints_from_metadata
from tabularbench.datasets.samples.ctu_13_neris import get_relation_constraints
from tabularbench.datasets.dataset import CsvDataSource

EPS = 0.5
DATA = Path(os.environ['TABULARBENCH_DATASET_DIR']) / 'ctu_13_neris'



class AttackLogits(torch.nn.Module):
    def __init__(self,adapter,model_name):
        super().__init__();self.adapter=adapter;self.value=None;self.handle=None;self.checked=False
        arch=model_name.split('_')[0]
        self.arch=arch
        if arch in ['torchrln','vime']:
            targets=[m for name,m in adapter.named_modules() if name.endswith('output_layer')]
            assert len(targets)==1
            self.handle=targets[0].register_forward_hook(lambda m,i,o:setattr(self,'value',o))
        elif arch=='tabtransformer':
            targets=[m.mlp for m in adapter.modules() if type(m).__name__=='MLP']
            assert len(targets)==1
            self.handle=targets[0].register_forward_hook(lambda m,i,o:setattr(self,'value',o))
        elif arch=='stg':
            targets=[m for m in adapter.modules() if type(m).__name__=='STGClassificationModel']
            assert len(targets)==1
            self.handle=targets[0].register_forward_hook(lambda m,i,o:setattr(self,'value',o['logits']))
        else:assert arch=='tabnet'
    def forward(self,x):
        ordinary=self.adapter(x)
        logits=ordinary if self.arch=='tabnet' else self.value
        assert logits is not None and logits.shape[1]==2
        if not self.checked:
            if self.arch!='tabnet':
                torch.testing.assert_close(logits.softmax(1),ordinary,atol=1e-6,rtol=1e-5)
            assert (logits.argmax(1)==ordinary.argmax(1)).all()
            self.checked=True
        return logits
    def close(self):
        if self.handle is not None:self.handle.remove()
        self.value=None


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


def ctu_data():
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


def completion_for(dataset, meta, scaler, judge):
    if dataset == "wids":
        return WidsCompletion(meta, scaler, EPS*.99)
    return ConditionalCompletion(dataset, meta, scaler, judge, EPS*.99)


def components(dataset, model_name):
    ds,xt,yt,scaler,model,cons,adapter = load_victim(dataset,model_name,DEVICE)
    meta=ds.get_metadata(only_x=True).reset_index(drop=True)
    judge=BatchJudge(model,cons,scaler,EPS,box_tol_scale=1000 if dataset=="wids" else 1,rel_tol=0.)
    return ds,meta,scaler,model,cons,adapter,judge
