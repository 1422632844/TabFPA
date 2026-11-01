"""Feasibility-preserving joint-integer Gumbel ablation; old results untouched."""
import os,sys,time,json,hashlib,gc,argparse,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts import ctu_gumbel_lr as ctu
from scripts import url_wids_gumbel_tuning as old
from core.feasible_integer_moves import JointIntegerMoves
from tabularbench.constraints.constraints_checker import ConstraintChecker
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/feasible_gumbel_final'
RESULTS=[];ARCHIVE={};CTU=None


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


def components(dataset,model):
    global CTU
    if dataset=='ctu_13_neris':
        if CTU is None:CTU=ctu.data()
        _,_,meta,scaler,cons=CTU;ctu.cons=cons
        with np.load(ROOT/'reference/pools.npz') as z:
            prefix=dataset+'__'+model
            x=z[prefix+'__clean'].copy();indices=z[prefix+'__indices'].copy()
        victim,adapter,judge=ctu.victim(model,meta,scaler,x)
        return meta,scaler,cons,victim,adapter,judge,x,indices
    _,meta,scaler,victim,cons,adapter,judge=old.components(dataset,model)
    with np.load(ROOT/'reference/pools.npz') as z:
        prefix=dataset+'__'+model
        x=z[prefix+'__clean'].copy();indices=z[prefix+'__indices'].copy()
    return meta,scaler,cons,victim,adapter,judge,x,indices

def run(dataset,model_name,meta,scaler,cons,model,adapter,judge,x,seed,frozen=False,steps=200,probability=False,state_policy="released",logits_lr=.06):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    n=len(x);current=x.copy()
    moves=JointIntegerMoves(meta,scaler,cons)
    x0=torch.tensor(x,dtype=torch.float32,device='cuda')
    scale=torch.tensor(moves.scale,dtype=torch.float32,device='cuda')
    values,mask=moves.candidates(x,current)
    prior=-40*np.abs(values)*np.sqrt(moves.a)[None,:,None]
    theta=torch.tensor(np.concatenate([np.full((n,1),2.),prior.reshape(n,-1)],axis=1),
                       dtype=torch.float32,device='cuda',requires_grad=not frozen)
    initial_theta=theta.detach().clone()
    d=torch.zeros((n,len(moves.reals)),device='cuda',requires_grad=True)
    params=[{'params':[d],'lr':.06}]
    if not frozen:params.append({'params':[theta],'lr':logits_lr})
    optimizer=torch.optim.Adam(params)
    y=np.ones(n,dtype=np.int64)
    scoring=adapter if probability else AttackLogits(adapter,model_name)
    meter=old.base.Meter(scoring,model,judge,x,y,'cuda',256,128)
    completion=ctu.CtuCompletion(dataset,meta,scaler,judge,.495) if dataset=='ctu_13_neris' else old.completion_for(dataset,meta,scaler,judge)
    A=np.concatenate([moves.L,moves.E],axis=0)
    bias=np.concatenate([moves.lb,moves.eb])
    TA=torch.tensor(A,dtype=torch.float32,device='cuda')
    TB=torch.tensor(bias,dtype=torch.float32,device='cuda')
    norm=(TA.abs()*scale[None]).max(1).values.clamp_min(1e-9)
    iseq=torch.arange(len(A),device='cuda')>=len(moves.L)
    start=time.perf_counter()
    with torch.no_grad():
        logits=meter.score(x0,np.arange(n))
        best_margin=(logits[:,1]-logits[:,0]).cpu().numpy()
    proposed=0;witness_valid=0;model_evaluated=0;action_count=0;state_updates=0;anchor_rows=0
    for step in range(steps):
        ids=np.flatnonzero(~meter.ok&(meter.fwd<254))
        if not len(ids):break
        anchor=current
        if state_policy=="released":
            requested=current.copy();requested[:,moves.reals]=x[:,moves.reals]
            anchor,possible=completion.solve(x,requested)
            assert possible.all(),"Minimum-cost continuous anchor failed"
            anchor_rows+=n
        values,mask=moves.candidates(x,anchor)
        integer_point,straight,counts=moves.sample(anchor,values,mask,theta,1.-.9*step/max(steps-1,1))
        valid=judge.constraint_ok(x[ids],integer_point[ids])
        cost=(((integer_point[ids]-x[ids])/moves.scale)**2).sum(1)
        assert valid.all() and (cost<=.495**2+1e-10).all(),'Joint move invalid before QP'
        proposed+=len(ids);witness_valid+=int(valid.sum());action_count+=int(counts[ids].sum())
        raw=torch.tensor(current,dtype=torch.float32,device='cuda')
        raw[:,moves.ints]=straight
        raw[:,moves.reals]=(x0[:,moves.reals]+d*scale[moves.reals]).clamp(
            torch.tensor(moves.lo[moves.reals],dtype=torch.float32,device='cuda'),
            torch.tensor(moves.hi[moves.reals],dtype=torch.float32,device='cuda'))
        proposal=integer_point[ids].copy()
        proposal[:,moves.reals]=raw[ids][:,moves.reals].detach().cpu().numpy()
        sol,allowed=completion.solve(x[ids],proposal)
        # Failed completions never query the victim. A feasible witness exists;
        # failures here are numerical solver outcomes, not illegal discrete draws.
        ii=ids[allowed]
        if not len(ii):continue
        pp=sol[allowed]
        assert judge.constraint_ok(x[ii],pp).all()
        original_ints=np.flatnonzero(moves.types=='int')
        np.testing.assert_array_equal(pp[:,original_ints],proposal[allowed][:,original_ints])
        hard=torch.tensor(pp,dtype=torch.float32,device='cuda')
        candidate=raw[ii]+(hard-raw[ii]).detach()
        logits=meter.score(candidate,ii);meter.bwd[ii]+=1;model_evaluated+=len(ii)
        attack=(logits[:,1]-logits[:,0]).mean()
        dz=(raw[ii]-x0[ii])/scale
        budget=torch.relu(dz[:,scaler.num_idx].norm(dim=1)-.45).square().mean()
        residual=(raw[ii]@TA.T+TB)/norm
        relation=torch.where(iseq[None],residual,torch.relu(residual)).square().sum(1)
        for b,p,cap in moves.packets:
            den=raw[ii,p];num=raw[ii,b]
            ratio=torch.where(den!=0,num/torch.where(den!=0,den,torch.ones_like(den)),torch.zeros_like(den))
            relation=relation+torch.relu(ratio/cap-1).square()
        loss=attack+4*budget+4*relation.mean()
        optimizer.zero_grad();loss.backward()
        if not frozen:assert theta.grad is not None and torch.isfinite(theta.grad).all()
        optimizer.step()
        with torch.no_grad():
            meter.accept(pp,ii,logits,np.ones(len(ii),bool))
            margins=(logits[:,1]-logits[:,0]).cpu().numpy()
            improve=margins<best_margin[ii]
            if state_policy=='walk':improve=np.ones(len(ii),dtype=bool)
            current[ii[improve]]=pp[improve];best_margin[ii[improve]]=margins[improve]
            state_updates+=int(improve.sum())
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    ok,dist,_,_=judge(x,y,meter.out)
    np.testing.assert_array_equal(ok,meter.ok)
    assert ConstraintChecker(cons,tolerance=0.).check_constraints(x,meter.out).all()
    assert (dist<=.5).all() and meter.fwd.max()<=256
    theta_change=float((theta.detach()-initial_theta).abs().max())
    if frozen:assert theta_change==0
    if not probability:scoring.close()
    return meter,dict(dataset=dataset,model=model_name,variant=('released_frozen' if frozen else 'released_learned') if state_policy=='released' else ('walk_learned' if state_policy=='walk' else ('probability_margin' if probability else ('frozen' if frozen else 'learned'))),
        state_policy=state_policy,
        seed=seed,n=n,accepted=int(ok.sum()),asr=float(ok.mean()),
        forward_mean=float(meter.fwd.mean()),forward_max=int(meter.fwd.max()),
        backward_mean=float(meter.bwd.mean()),seconds=seconds,
        joint_proposals=proposed,joint_witness_valid=witness_valid,budget_anchor_rows=anchor_rows,
        conditional_rejected=completion.rejected,conditional_rows=completion.rows,
        model_evaluated_candidates=model_evaluated,mean_available_actions=action_count/max(proposed,1),
        integer_kernel_directions=moves.nd,state_updates=state_updates,theta_max_change=theta_change,
        successful_integer_changed=int((((meter.out[:,moves.ints]-x[:,moves.ints])!=0).any(1)&ok).sum()),
        audit_forward_per_sample=1,rss_peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        gpu_peak_mb=torch.cuda.max_memory_allocated()/1024**2,distance_max=float(dist.max()))

def smoke():
    for dataset in ['ctu_13_neris','url','wids']:
        meta,scaler,cons,model,adapter,judge,x,indices=components(dataset,'torchrln_default')
        x=x[:8]
        moves=JointIntegerMoves(meta,scaler,cons)
        values,mask=moves.candidates(x,x)
        rng=np.random.RandomState(123);checks=[]
        for i in range(len(x)):
            slots=np.argwhere(mask[i])
            for j,k in slots[rng.choice(len(slots),min(16,len(slots)),replace=False)]:
                v=x[i]+moves.D[j]*values[i,j,k]
                assert judge.constraint_ok(x[i:i+1],v[None])[0]
                assert (((v-x[i])/moves.scale)**2).sum()<=.495**2+1e-10
                checks.append(v)
        print(json.dumps(dict(event='move_smoke',dataset=dataset,directions=moves.nd,
                              valid_joint_points=len(checks),max_action_count=int(mask.sum((1,2)).max()))),flush=True)
        for frozen in [False,True]:
            _,row=run(dataset,'torchrln_default',meta,scaler,cons,model,adapter,judge,x,0,frozen,6)
            print(json.dumps(row),flush=True)
        del adapter,model,judge;gc.collect();torch.cuda.empty_cache()
    print('SMOKE_COMPLETE',flush=True)



def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--datasets',nargs='+',choices=['ctu_13_neris','url','wids'],
                        default=['ctu_13_neris','url','wids'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.smoke:return smoke()
    OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists(),'Use a new output directory'
    sources=[Path(__file__),ROOT/'core/feasible_integer_moves.py',ROOT/'scripts/ctu_gumbel_lr.py',
             ROOT/'scripts/url_wids_gumbel_tuning.py',ROOT/'core/conditional_completion.py',
             ROOT/'core/wids_completion.py',ROOT/'core/judge.py']
    native=Path(os.environ['TABULARBENCH_ROOT'])/'tabularbench/models/tabsurvey'
    sources.extend(native/p for p in ['mlp_rln.py','vime.py','tabtransformer.py','stg.py','tabnet.py','stg_lib/models.py'])
    manifest=dict(version='joint_integer_minimum_continuous_anchor',
       datasets=args.datasets,seeds=args.seeds,
       pool='CTU existing per-model n100 test pools; URL/WiDS existing per-model n64 confirmation pools.',
       models='CTU all five; URL/WiDS RLN and TabTransformer.',
       variants='Learned on all evaluated models; frozen-logits control on RLN and TabTransformer.',
       query_cap=256,steps=200,logits_lr=.06,continuous_lr=.06,tau=[1,.1],
       eps=.5,completion_eps=.495,batch=128,device='cuda:0',torch_threads=2,
       sampler='Joint integer kernel moves, bounded using minimum continuous completion cost; best-margin feasible incumbent.',
       action_magnitudes='Each direction +/-1 and 10/50/100 percent of feasible line interval; deduplicated; explicit stay.',
       objective='True pre-softmax logit margin; same target weights and predictions.',
       gradient='Full joint move in ST backward; identity BPDA for continuous completion.',
       accounting='Initial victim score and successful-candidate verification included; final audit +1 per sample outside attack.',
       sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for dataset in args.datasets:
        models=ctu.DEFAULT_MODELS if dataset=='ctu_13_neris' else old.MODELS
        for model_name in models:
            meta,scaler,cons,model,adapter,judge,x,indices=components(dataset,model_name)
            variants=[False,True] if model_name in old.MODELS else [False]
            for seed in args.seeds:
                for frozen in variants:
                    meter,row=run(dataset,model_name,meta,scaler,cons,model,adapter,judge,x,seed,frozen)
                    RESULTS.append(row);print(json.dumps(row),flush=True)
                    prefix=dataset+'__'+model_name
                    ARCHIVE[prefix+'__indices']=indices;ARCHIVE[prefix+'__clean']=x
                    key=prefix+'__'+row['variant']+'__s'+str(seed)
                    for label,value in [('adv',meter.out),('ok',meter.ok),('forward',meter.fwd),('backward',meter.bwd)]:
                        ARCHIVE[key+'__'+label]=value
                    (OUT/'results.json').write_text(json.dumps(RESULTS,indent=2))
            np.savez_compressed(OUT/'adversarial.npz',**ARCHIVE)
            del adapter,model,judge;gc.collect();torch.cuda.empty_cache()
    print('EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
