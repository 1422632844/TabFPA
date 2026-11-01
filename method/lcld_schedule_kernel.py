"""Widening schedule: progressive slot-tier unlock inside ONE 200-round run.

Same loop as fixed_kernel (bit-exact reduction when tier_rounds=None).
Each integer-move direction exposes 8 magnitude slots (|step| ascending):
  negative slots 0..3 = [-1, 10%, 50%, full]; positive slots 4..7 likewise.
Tiers: T0={0,4} (+-1), T1+= {1,5} (10%), T2+= {2,6} (50%), T3+= {3,7} (full).
At each unlock round the newly available slots get theta reset to the neutral
value (0., i.e. the kappa=0 init) so they earn a fair exploration phase --
the slots are activated without adding steps to the run. No restarts,
no extra steps, no extra budget: same 200 steps, same 256-forward cap."""
import numpy as np
import torch,time,resource
from runtime import AttackLogits, Meter, CtuCompletion, completion_for
from core.feasible_integer_moves import JointIntegerMoves
from tabularbench.constraints.constraints_checker import ConstraintChecker

TRACE=__import__('os').environ.get('WK_TRACE')=='1'
SLOT_TIERS=[np.array([0,4]),np.array([1,5]),np.array([2,6]),np.array([3,7])]

def run(dataset,model_name,meta,scaler,cons,model,adapter,judge,x,seed,frozen=False,steps=200,probability=False,state_policy="released",logits_lr=.48, kappa=40., moves_cls=None, completion_ctor=None, progress=None, tier_rounds=None, tau_sawtooth=True, reinit_round=None, reinit_kappa=0., kappa_anneal=None, l0_cap_frac=None, cont_param='plain', rho_scale=1., ball_hinge=False, ball_sched=None, reinit_dir=False, dir_gate=None, se_gate=None, d_lr=.06, hinge_w=4., hinge_th=.45, rel_w=0., repair_grad='identity', dfo_start=None, extra_search=0, stall_restart=False):
    # rel_w weights the relation surrogate; the released configuration uses 0.
    # cont_param: continuous reparameterization for the real-valued block.
    #   'plain'    unbounded d + box clamp + budget hinge (official baseline)
    #   'ball'     v1 stereographic ball map dz=r_t*2u/(1+|u|^2), hinge off
    #   'ballv2'   radial clamp dz=r_t*u/max(1,|u|) -- ENDPOINT FIX: sphere
    #              pinned for every |u|>=1 (not a measure-zero ring), radial
    #              gradient never flips sign (no fold-back past |u|=1)
    #   'balltanh' smooth saturation dz=r_t*tanh(|u|)/|u|*u, monotone radius
    # r_t = sqrt(max(.495^2 - c_disc,0))*rho_scale, c_disc = this round's
    # detached integer jump cost; at rho_scale=1 the joint cost stays
    # <= .495^2 by construction (same guarantee as 'plain'+completion).
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    n=len(x);current=x.copy()
    search_rng=np.random.RandomState(seed+71039)
    global_best=x.copy()
    last_restart=np.zeros(n,dtype=np.int64)
    milestones=[]
    rescue_queries=np.zeros(n,dtype=np.int64)
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
    params=[{'params':[d],'lr':d_lr}]
    if not frozen:params.append({'params':[theta],'lr':logits_lr})
    optimizer=torch.optim.Adam(params)
    y=np.ones(n,dtype=np.int64)
    scoring=adapter if probability else AttackLogits(adapter,model_name)
    meter=Meter(scoring,model,judge,x,y,'cuda',256,128)
    certification_calls=np.zeros(n,dtype=np.int64)
    optimization_visits=np.zeros(n,dtype=np.int64)
    completion=completion_ctor(dataset,meta,scaler,judge) if completion_ctor else (CtuCompletion(dataset,meta,scaler,judge,.495) if dataset=='ctu_13_neris' else completion_for(dataset,meta,scaler,judge))
    A=np.concatenate([moves.L,moves.E],axis=0)
    bias=np.concatenate([moves.lb,moves.eb])
    TA=torch.tensor(A,dtype=torch.float32,device='cuda')
    TB=torch.tensor(bias,dtype=torch.float32,device='cuda')
    norm=(TA.abs()*scale[None]).max(1).values.clamp_min(1e-9)
    iseq=torch.arange(len(A),device='cuda')>=len(moves.L)
    lo_t=torch.tensor(moves.lo[moves.reals],dtype=torch.float32,device='cuda')
    hi_t=torch.tensor(moves.hi[moves.reals],dtype=torch.float32,device='cuda')
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
    relation_max_seen=0.
    start=time.perf_counter()
    with torch.no_grad():
        logits=meter.score(x0,np.arange(n))
        best_margin=(logits[:,1]-logits[:,0]).cpu().numpy()
    initial_margin=best_margin.copy()
    proposed=0;witness_valid=0;model_evaluated=0;action_count=0;state_updates=0;anchor_rows=0
    # adaptive restart: graduated patience-based trigger
    # phase 1 (0-49):  kappa=40 initial search
    # phase 2 (50-99): kappa=40 re-roll noise (cheap fix first)
    # phase 3 (100-149): kappa=0  switch distribution (escalate)
    # phase 4 (150-199): kappa=0 re-roll noise again
    patience=__import__('os').environ.get('WK_PATIENCE')
    patience=int(patience) if patience else None
    last_progress_step=0;stall_count=0;adaptive_kappa=float(kappa)
    last_improve=np.zeros(n,dtype=np.int64)  # per-row last margin-improvement step (gate signal)
    _prev_step_improve=0
    for step in range(steps):
        if progress and step % 25 == 0: progress(step,int(meter.ok.sum()))
        if stall_restart and step>=50 and step%25==0:
            rr=np.flatnonzero(~meter.ok & (step-last_improve>=25) & (step-last_restart>=25))
            if len(rr):
                with torch.no_grad():
                    theta[rr,0]=2.
                    theta[rr,1:]=0.
                    g=torch.randn((len(rr),d.shape[1]),device=d.device)
                    g=g/g.norm(dim=1,keepdim=True).clamp_min(1e-9)
                    d[rr]=g*(.2+.8*torch.rand((len(rr),1),device=d.device))
                    for param in [d,theta]:
                        for key in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
                            if key in optimizer.state.get(param,{}):optimizer.state[param][key][rr]=0.
                current[rr]=(x if step%75==0 else global_best)[rr]
                last_restart[rr]=step
        _trigger=False
        if patience is not None:
            if _prev_step_improve>0:
                last_progress_step=step;stall_count=0
            _prev_step_improve=0
            if step-last_progress_step>=patience and step>0 and step%patience==0:
                _trigger=True;last_progress_step=step;stall_count+=1
                if stall_count==1:
                    adaptive_kappa=float(kappa)      # re-roll noise, same kappa
                else:
                    adaptive_kappa=0.                # switch to kappa=0
        elif reinit_round is not None and step==reinit_round:
            _trigger=True;adaptive_kappa=float(reinit_kappa)
        if _trigger:
            # Within-run soft restart for the neutral phase: accepted rows are
            # already locked in the meter; the rest re-run from clean under
            # the kappa=0 prior with fresh Adam -- E07's restart folded into
            # the SAME 200-step budget (no extra rounds, no extra queries).
            with torch.no_grad():
                theta.data[:,1:]=torch.tensor(-adaptive_kappa*initial_lengths.reshape(n,-1),dtype=torch.float32,device='cuda')
                if reinit_dir and cont_param!='plain':
                    # direction restart: random unit direction instead of zero --
                    # the map pins the fresh intent straight onto the sphere,
                    # giving stalled rows a new exploration phase (in-ball).
                    # dir_gate=K: restart ONLY rows stalled >= K steps (no
                    # margin improvement); progressing rows KEEP their d (not
                    # zeroed) so slow-but-steady rows lose no progress.
                    g=torch.randn_like(d)
                    g=g/g.norm(dim=1,keepdim=True).clamp_min(1e-9)
                    if dir_gate is None:
                        d.data=g
                    else:
                        stall=torch.as_tensor((step-last_improve)>=dir_gate,device='cuda').unsqueeze(1)
                        d.data=torch.where(stall,g,d.data)
                else:
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
        ids=np.flatnonzero(~meter.ok&(meter.fwd<252))
        if TRACE:print('TRACE step',step,'ids',len(ids),'ok',int(meter.ok.sum()),'fwdmax',int(meter.fwd.max()),flush=True)
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
        assert valid.all() and (cost<=.495**2+1e-10).all(),'Joint move invalid before QP'
        proposed+=len(ids);witness_valid+=int(valid.sum());action_count+=int(counts[ids].sum())
        raw=torch.tensor(current,dtype=torch.float32,device='cuda')
        raw[:,moves.ints]=straight
        if cont_param=='plain':
            raw[:,moves.reals]=(x0[:,moves.reals]+d*scale[moves.reals]).clamp(lo_t,hi_t)
        else:
            # residual-ball reparam around this round's integer jump
            with torch.no_grad():
                ip=torch.as_tensor(integer_point[:,moves.ints],dtype=torch.float32,device='cuda')
                disc=(((ip-x0[:,moves.ints])/scale[moves.ints])**2).sum(1)
                r_t=torch.clamp(.495**2-disc,min=0.).sqrt()*rho_scale
                if ball_sched=='se':
                    # in-ball shrink-expand: sweep effective radius r_t -> 0.3*r_t
                    # every 50 steps (4 ray-sampling cycles in 200) -- the
                    # feasible mirror of A's overshoot-and-repair cycle.
                    # se_gate=K: cycle only rows stalled >= K steps (no margin
                    # improvement); progressing rows keep the full radius.
                    frac=(step%50)/50.
                    sef=0.3+0.7*(1.-frac)
                    if se_gate is not None:
                        stall=torch.as_tensor(((step-last_improve)>=se_gate).astype(np.float32),device='cuda')
                        r_t=r_t*(stall*sef+(1.-stall))
                    else:
                        r_t=r_t*sef
            un=d.norm(dim=1,keepdim=True)
            if cont_param=='ball':
                dzr=2.*d/(1.+un**2)*r_t[:,None]
            elif cont_param=='ballv2':
                dzr=d/un.clamp_min(1.)*r_t[:,None]
            elif cont_param=='balltanh':
                dzr=(torch.tanh(un)/un.clamp_min(1e-12))*d*r_t[:,None]
            else:
                raise ValueError(cont_param)
            raw[:,moves.reals]=(x0[:,moves.reals]+dzr*scale[moves.reals]).clamp(lo_t,hi_t)
        if dfo_start is not None and step>=dfo_start:
            # Search independent real coordinates; repair and acceptance stay exact.
            assert dataset=='lcld_v2_iid'
            jj=completion.j; ff=completion.free
            kk=step-dfo_start
            if kk%2==0:
                search_anchor=global_best.copy()
                direction=np.zeros((n,len(ff)))
                if (kk//2)%6<5:
                    direction[:,(kk//2)%6]=1.
                else:
                    direction=search_rng.normal(size=direction.shape)
                    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
                search_delta=direction*(.4*(.5**((kk//12)%5)))
            pp_search=search_anchor.copy()
            pp_search[:,ff]+=(-1. if kk%2 else 1.)*search_delta*moves.scale[ff]
            pp_search[:,ff]=np.clip(pp_search[:,ff],moves.lo[ff],moves.hi[ff])
            ir=x[:,jj['int_rate']]/1200.;power=(1+ir)**x[:,jj['term']]
            pp_search[:,jj['installment']]=ir*power/(power-1)*pp_search[:,jj['loan_amnt']]
            pp_search[:,jj['ratio_loan_amnt_annual_inc']]=pp_search[:,jj['loan_amnt']]/pp_search[:,jj['annual_inc']]
            # DFO suffix searches around the best state, preserving its integers.
            integer_point=search_anchor.copy()
            raw=raw+(torch.as_tensor(pp_search,dtype=raw.dtype,device=raw.device)-raw).detach()
        proposal=integer_point[ids].copy()
        proposal[:,moves.reals]=raw[ids][:,moves.reals].detach().cpu().numpy()
        sol,allowed=completion.solve(x[ids],proposal)
        ii=ids[allowed]
        if TRACE:print('TRACE step',step,'allowed',int(allowed.sum()),'of',len(ids),flush=True)
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
        if repair_grad=='identity' or dataset!='lcld_v2_iid':
            candidate=raw[ii]+(hard-raw[ii]).detach()
        else:
            jj=completion.j
            # Linearized analytic manifold at the actual repaired point.
            # This is a BPDA approximation, NOT exact differentiation of SLSQP.
            cols=np.concatenate([completion.free,moves.ints])
            J=torch.zeros((len(ii),x.shape[1],len(cols)),device=hard.device,dtype=hard.dtype)
            for cc,col in enumerate(cols):J[:,col,cc]=1.
            loan=hard[:,jj['loan_amnt']];annual=hard[:,jj['annual_inc']]
            ir=hard[:,jj['int_rate']]/1200.;power=(1+ir)**hard[:,jj['term']]
            ci={int(v):i for i,v in enumerate(cols)}
            J[:,jj['installment'],ci[jj['loan_amnt']]]=ir*power/(power-1)*scale[jj['loan_amnt']]/scale[jj['installment']]
            J[:,jj['ratio_loan_amnt_annual_inc'],ci[jj['loan_amnt']]]=scale[jj['loan_amnt']]/annual/scale[jj['ratio_loan_amnt_annual_inc']]
            J[:,jj['ratio_loan_amnt_annual_inc'],ci[jj['annual_inc']]]=-loan*scale[jj['annual_inc']]/annual.square()/scale[jj['ratio_loan_amnt_annual_inc']]
            op=hard[:,jj['open_acc']];tot=hard[:,jj['total_acc']]
            J[:,jj['ratio_open_acc_total_acc'],ci[jj['open_acc']]]=scale[jj['open_acc']]/tot/scale[jj['ratio_open_acc_total_acc']]
            J[:,jj['ratio_open_acc_total_acc'],ci[jj['total_acc']]]=-op*scale[jj['total_acc']]/tot.square()/scale[jj['ratio_open_acc_total_acc']]
            delta=(raw[ii]-raw[ii].detach())/scale
            if repair_grad=='chain':
                mapped=torch.bmm(J,delta[:,cols,None]).squeeze(2)
            elif repair_grad=='tangent':
                # Closest tangent projection in the same scaled L2 metric.
                # Inequality active sets and curvature are deliberately omitted.
                rhs=torch.bmm(J.transpose(1,2),delta[:,:,None])
                coeff=torch.linalg.solve(torch.bmm(J.transpose(1,2),J),rhs)
                mapped=torch.bmm(J,coeff).squeeze(2)
            else:raise ValueError(repair_grad)
            candidate=hard+mapped*scale
        logits=meter.score(candidate,ii);meter.bwd[ii]+=1;model_evaluated+=len(ii)
        attack=(logits[:,1]-logits[:,0]).mean()
        dz=(raw[ii]-x0[ii])/scale
        budget=torch.relu(dz[:,scaler.num_idx].norm(dim=1)-hinge_th).square().mean()
        residual=(raw[ii]@TA.T+TB)/norm
        relation=torch.where(iseq[None],residual,torch.relu(residual)).square().sum(1)
        for b,p,cap in moves.packets:
            den=raw[ii,p];num=raw[ii,b]
            ratio=torch.where(den!=0,num/torch.where(den!=0,den,torch.ones_like(den)),torch.zeros_like(den))
            relation=relation+torch.relu(ratio/cap-1).square()
        relation_max_seen=max(relation_max_seen,float(relation.detach().max().cpu()))
        # ball arms drop the budget hinge (|dz|<=r_t by construction, hinge idle)
        if cont_param=='plain' or ball_hinge:
            loss=attack+hinge_w*budget+rel_w*relation.mean()
        else:
            loss=attack+rel_w*relation.mean()
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
            global_best[ii[improve]]=pp[improve]
            last_improve[ii[improve]]=step
            _prev_step_improve=int(improve.sum());state_updates+=_prev_step_improve
    # Optional rescue queries occupy the unused query allowance, not extra
    # optimizer steps. Full accounting includes initial score and final audits.
    if extra_search:
        assert dataset=='lcld_v2_iid'
        ff=completion.free;jj=completion.j
        for k in range(extra_search):
            ids=np.flatnonzero(~meter.ok&(meter.fwd<252))
            if not len(ids):break
            if k%2==0:
                rescue_anchor=global_best.copy()
                direction=np.zeros((n,len(ff)))
                if (k//2)%6<5:direction[:,(k//2)%6]=1.
                else:
                    direction=search_rng.normal(size=direction.shape)
                    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
                rescue_delta=direction*(.4*(.5**((k//12)%5)))
            proposal=rescue_anchor[ids].copy()
            proposal[:,ff]+=(-1. if k%2 else 1.)*rescue_delta[ids]*moves.scale[ff]
            proposal[:,ff]=np.clip(proposal[:,ff],moves.lo[ff],moves.hi[ff])
            ir=x[ids,jj['int_rate']]/1200.;power=(1+ir)**x[ids,jj['term']]
            proposal[:,jj['installment']]=ir*power/(power-1)*proposal[:,jj['loan_amnt']]
            proposal[:,jj['ratio_loan_amnt_annual_inc']]=proposal[:,jj['loan_amnt']]/proposal[:,jj['annual_inc']]
            sol,allowed=completion.solve(x[ids],proposal)
            ii=ids[allowed];pp=sol[allowed]
            if not len(ii):continue
            with torch.no_grad():
                logits=meter.score(torch.as_tensor(pp,dtype=torch.float32,device='cuda'),ii)
                rescue_queries[ii]+=1
                before_cert=meter.fwd.copy()
                meter.accept(pp,ii,logits,np.ones(len(ii),bool))
                certification_calls+=meter.fwd-before_cert
                margins=(logits[:,1]-logits[:,0]).cpu().numpy()
                improve=margins<best_margin[ii]
                current[ii[improve]]=pp[improve];best_margin[ii[improve]]=margins[improve]
                global_best[ii[improve]]=pp[improve]
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    audit_calls=np.ones(n,dtype=np.int64)
    ok,dist,_,_=judge(x,y,meter.out)
    np.testing.assert_array_equal(ok,meter.ok)
    # official zero-tolerance checker as post-filter (ulp-boundary rows in CAA
    # pools can pass the judge yet fail tolerance=0; demote instead of crash)
    _wk_tol=__import__('os').environ.get('WK_AUDIT_TOL')
    ok_official=np.asarray(ConstraintChecker(cons,tolerance=float(_wk_tol) if _wk_tol else 0.).check_constraints(x,meter.out),bool)
    _viol=meter.ok&~ok_official
    if _viol.any():
        meter.out[_viol]=x[_viol];meter.ok[_viol]=False
        audit_calls+=1
        ok,dist,_,_=judge(x,y,meter.out)
        np.testing.assert_array_equal(ok,meter.ok)
    assert (dist<=.5).all() and meter.fwd.max()<=256
    np.testing.assert_array_equal(meter.fwd,1+meter.bwd+certification_calls+rescue_queries)
    meter.fwd+=audit_calls
    assert meter.fwd.max()<=256
    meter.audit_calls=audit_calls
    meter.rescue_queries=rescue_queries
    meter.best_point=global_best.copy()
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
        state_policy=state_policy,cont_param=cont_param,rho_scale=rho_scale,ball_hinge=ball_hinge,
        ball_sched=ball_sched,reinit_dir=reinit_dir,dir_gate=dir_gate,se_gate=se_gate,
        tier_rounds=list(tier_rounds) if tier_rounds is not None else None,
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
        rel_w=rel_w,relation_max_seen=relation_max_seen,repair_grad=repair_grad,dfo_start=dfo_start,extra_search=extra_search,stall_restart=stall_restart,audit_forward_per_sample=int(audit_calls.max()),rss_peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        gpu_peak_mb=torch.cuda.max_memory_allocated()/1024**2,distance_max=float(dist.max()))
