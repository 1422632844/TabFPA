"""Widening schedule: progressive slot-tier unlock inside ONE 200-round run.

Same loop as fixed_kernel (bit-exact reduction when tier_rounds=None).
Each integer-move direction exposes 8 magnitude slots (|step| ascending):
  negative slots 0..3 = [-1, 10%, 50%, full]; positive slots 4..7 likewise.
Tiers: T0={0,4} (+-1), T1+= {1,5} (10%), T2+= {2,6} (50%), T3+= {3,7} (full).
At each unlock round the newly available slots get theta reset to the neutral
value (0., i.e. the kappa=0 init) so they earn a fair exploration phase --
passive unmasking alone repeats E04's failed mid-run switch. No restarts,
no extra steps, no extra budget: same 200 steps, same 256-forward cap."""
import numpy as np
import torch,time,resource
from scripts.feasible_gumbel_ablation import AttackLogits,JointIntegerMoves,old,ctu,ConstraintChecker

SLOT_TIERS=[np.array([0,4]),np.array([1,5]),np.array([2,6]),np.array([3,7])]

def run(dataset,model_name,meta,scaler,cons,model,adapter,judge,x,seed,frozen=False,steps=200,probability=False,state_policy="released",logits_lr=.48, kappa=40., moves_cls=None, completion_ctor=None, progress=None, tier_rounds=None, tau_sawtooth=True, reinit_round=None, reinit_kappa=0., kappa_anneal=None, l0_cap_frac=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    n=len(x);current=x.copy()
    moves=(moves_cls or JointIntegerMoves)(meta,scaler,cons)
    x0=torch.tensor(x,dtype=torch.float32,device='cuda')
    scale=torch.tensor(moves.scale,dtype=torch.float32,device='cuda')
    values,mask=moves.candidates(x,current)
    initial_lengths=np.abs(values)*np.sqrt(moves.a)[None,:,None]
    prior=-float(kappa)*initial_lengths
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
    certification_calls=np.zeros(n,dtype=np.int64)
    optimization_visits=np.zeros(n,dtype=np.int64)
    completion=completion_ctor(dataset,meta,scaler,judge) if completion_ctor else (ctu.CtuCompletion(dataset,meta,scaler,judge,.495) if dataset=='ctu_13_neris' else old.completion_for(dataset,meta,scaler,judge))
    A=np.concatenate([moves.L,moves.E],axis=0)
    bias=np.concatenate([moves.lb,moves.eb])
    TA=torch.tensor(A,dtype=torch.float32,device='cuda')
    TB=torch.tensor(bias,dtype=torch.float32,device='cuda')
    norm=(TA.abs()*scale[None]).max(1).values.clamp_min(1e-9)
    iseq=torch.arange(len(A),device='cuda')>=len(moves.L)
    # ---- widening machinery (inert when tier_rounds is None) ----
    tier_bool=[];unlock_cols={}
    IL=None;k_prev=float(kappa)
    # L0 protocol: hard cap on changed features (ints exact-diff, reals
    # |dz|>0.01 of range); violating candidates are never scored, never
    # accepted, and never enter the walked state -- same gate pattern as
    # infeasible completions. l0 stats are recorded for EVERY arm.
    is_int_col=np.array([t!='real' for t in moves.types])
    real_cols=np.flatnonzero(~is_int_col)
    int_cols=np.flatnonzero(is_int_col)
    rsc=moves.scale[real_cols]
    def l0_count(p,rows):
        xc=x[rows]
        c=(p[:,int_cols]!=xc[:,int_cols]).sum(1)
        if len(real_cols):
            dz=np.abs(p[:,real_cols]-xc[:,real_cols])/rsc
            c=c+(dz>0.01).sum(1)
        return c
    l0k=int(x.shape[1]*l0_cap_frac) if l0_cap_frac is not None else None
    if kappa_anneal is not None:
        # active prior tracking: theta = learned_offset - kappa(t)*len, so the
        # prior component anneals linearly while learning continues on top.
        IL=torch.tensor(initial_lengths.reshape(n,-1),dtype=torch.float32,device='cuda')
    if tier_rounds is not None:
        assert len(tier_rounds)<=len(SLOT_TIERS)-1 and list(tier_rounds)==sorted(tier_rounds)
        for r in tier_rounds:assert 0<r<steps
        for t in range(len(SLOT_TIERS)):
            b=np.zeros(8,bool);b[np.concatenate(SLOT_TIERS[:t+1])]=True
            tier_bool.append(b)
        for i,r in enumerate(tier_rounds):
            slots=SLOT_TIERS[i+1]
            unlock_cols[r]=(1+np.concatenate([np.arange(moves.nd)*8+s for s in slots])).astype(np.int64)
        phase_bounds=[0]+list(tier_rounds)+[steps]
    start=time.perf_counter()
    with torch.no_grad():
        logits=meter.score(x0,np.arange(n))
        best_margin=(logits[:,1]-logits[:,0]).cpu().numpy()
    initial_margin=best_margin.copy()
    proposed=0;witness_valid=0;model_evaluated=0;action_count=0;state_updates=0;anchor_rows=0
    for step in range(steps):
        if progress and step % 25 == 0: progress(step,int(meter.ok.sum()))
        if reinit_round is not None and step==reinit_round:
            # Within-run soft restart for the neutral phase: accepted rows are
            # already locked in the meter; the rest re-run from clean under
            # the kappa=0 prior with fresh Adam -- E07's restart folded into
            # the SAME 200-step budget (no extra rounds, no extra queries).
            with torch.no_grad():
                theta.data[:,1:]=torch.tensor(-float(reinit_kappa)*initial_lengths.reshape(n,-1),dtype=torch.float32,device='cuda')
                d.data.zero_()
            current=x.copy()
            optimizer=torch.optim.Adam(params)
        if tier_rounds is not None and step in unlock_cols:
            with torch.no_grad():theta.data[:,unlock_cols[step]]=0.
        if kappa_anneal is not None:
            s0,s1=kappa_anneal
            if step<=s0:k_t=float(kappa)
            elif step>=s1:k_t=0.
            else:k_t=float(kappa)*(1.-(step-s0)/(s1-s0))
            with torch.no_grad():
                theta.data[:,1:]+=-(k_t-k_prev)*IL
            k_prev=k_t
        ids=np.flatnonzero(~meter.ok&(meter.fwd<254))
        if not len(ids):break
        optimization_visits[ids]+=1
        anchor=current
        if state_policy=="released":
            requested=current.copy();requested[:,moves.reals]=x[:,moves.reals]
            anchor,possible=completion.solve(x,requested)
            assert possible.all(),"Minimum-cost continuous anchor failed"
            anchor_rows+=n
        values,mask=moves.candidates(x,anchor)
        if tier_rounds is not None:
            tier_idx=sum(step>=r for r in tier_rounds)
            mask=mask&tier_bool[tier_idx][None,None,:]
            if tau_sawtooth:
                s0,s1=phase_bounds[tier_idx],phase_bounds[tier_idx+1]
                tau=1.-.9*(step-s0)/max(s1-1-s0,1)
            else:
                if reinit_round is not None and step>=reinit_round:
                    tau=1.-.9*(step-reinit_round)/max(steps-1-reinit_round,1)
                else:tau=1.-.9*step/max(steps-1,1)
        else:
            if reinit_round is not None and step>=reinit_round:
                tau=1.-.9*(step-reinit_round)/max(steps-1-reinit_round,1)
            else:tau=1.-.9*step/max(steps-1,1)
        integer_point,straight,counts=moves.sample(anchor,values,mask,theta,tau)
        valid=judge.constraint_ok(x[ids],integer_point[ids])
        cost=(((integer_point[ids]-x[ids])/moves.scale)**2).sum(1)
        okj=valid&(cost<=.495**2+1e-10)
        if not okj.all():
            # transient sampler corner (boundary rounding on compiled move bounds):
            # drop offending rows this round, mirroring completion-infeasible rows
            # below; outputs remain subject to every final audit.
            bad=~okj
            import sys as _s
            print('WARN joint_move_drop step=%s n=%d maxdelta=%.3e'%(step,int(bad.sum()),float(np.abs(integer_point[ids][bad]-x[ids][bad]).max())),file=_s.stderr,flush=True)
            ids=ids[~bad]
            if not len(ids): continue
        proposed+=len(ids);witness_valid+=int(okj.sum());action_count+=int(counts[ids].sum())
        raw=torch.tensor(current,dtype=torch.float32,device='cuda')
        raw[:,moves.ints]=straight
        raw[:,moves.reals]=(x0[:,moves.reals]+d*scale[moves.reals]).clamp(
            torch.tensor(moves.lo[moves.reals],dtype=torch.float32,device='cuda'),
            torch.tensor(moves.hi[moves.reals],dtype=torch.float32,device='cuda'))
        proposal=integer_point[ids].copy()
        proposal[:,moves.reals]=raw[ids][:,moves.reals].detach().cpu().numpy()
        sol,allowed=completion.solve(x[ids],proposal)
        ii=ids[allowed]
        if not len(ii):continue
        pp=sol[allowed];prop=proposal[allowed]
        if l0k is not None:
            okl0=l0_count(pp,ii)<=l0k
            ii=ii[okl0];pp=pp[okl0];prop=prop[okl0]
            if not len(ii):continue
        assert judge.constraint_ok(x[ii],pp).all()
        original_ints=np.flatnonzero(moves.types=='int')
        np.testing.assert_array_equal(pp[:,original_ints],prop[:,original_ints])
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
            before_cert=meter.fwd.copy()
            meter.accept(pp,ii,logits,np.ones(len(ii),bool))
            certification_calls+=meter.fwd-before_cert
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
    np.testing.assert_array_equal(meter.fwd,1+meter.bwd+certification_calls)
    assert (optimization_visits<=steps).all()
    meter.certification_calls=certification_calls
    meter.optimization_visits=optimization_visits
    meter.initial_margin=initial_margin
    meter.final_best_margin=best_margin.copy()
    final_l0=l0_count(meter.out,np.arange(n))
    meter.final_l0=final_l0
    theta_change=float((theta.detach()-initial_theta).abs().max())
    if frozen:assert theta_change==0
    if not probability:scoring.close()
    return meter,dict(dataset=dataset,model=model_name,variant='widen' if tier_rounds is not None else ('widen_reinit' if reinit_round is not None else 'fixed'),
        state_policy=state_policy,tier_rounds=list(tier_rounds) if tier_rounds is not None else None,
        tau_sawtooth=bool(tau_sawtooth),reinit_round=reinit_round,reinit_kappa=reinit_kappa,
        kappa_anneal=list(kappa_anneal) if kappa_anneal is not None else None,
        l0_cap=(int(x.shape[1]*l0_cap_frac) if l0_cap_frac is not None else None),
        l0_mean_changed=float(final_l0.mean()),l0_max_changed=int(final_l0.max()),
        l0_mean_changed_successful=float(final_l0[ok].mean()) if ok.any() else None,
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
