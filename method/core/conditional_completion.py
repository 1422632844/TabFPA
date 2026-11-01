"""Fixed-discrete conditional completion for the four evaluated datasets.

WiDS retains its exact pair/ball solver. URL and CTU use convex box/order
completion in continuous coordinates. LCLD uses a local SLSQP nonlinear
completion because its loan/annual-income equality is not a convex QP.
No completion may alter a chosen integer or an immutable field.
"""
import numpy as np
from scipy.optimize import minimize
from core.wids_completion import WidsCompletion
from tabdatasets.ctu_13_neris import PORTS


class ConditionalCompletion:
    def __init__(self,dataset,meta,scaler,judge,eps):
        self.dataset=dataset;self.meta=meta;self.judge=judge;self.eps=eps
        self.types=np.asarray(meta["type"]);self.mutable=np.asarray(meta["mutable"],bool)
        self.real=np.flatnonzero(self.mutable & (self.types=="real"))
        self.fixed=np.flatnonzero(~self.mutable | (self.types!="real"))
        self.numeric=np.asarray(scaler.num_idx)
        self.scale=np.ones(len(meta))
        self.scale[self.numeric]=(scaler.x_max-scaler.x_min).cpu().numpy()
        self.scale=np.maximum(self.scale,1e-12)
        self.lo=meta["min"].to_numpy(float);self.hi=meta["max"].to_numpy(float)
        self.names=meta["feature"].tolist();self.j={n:i for i,n in enumerate(self.names)}
        self.calls=self.rows=self.rejected=self.solver_calls=self.solver_failed=0
        self.groups=[]
        if dataset=="wids":
            self.wids=WidsCompletion(meta,scaler,eps)
            self.kind="convex_pair_ball_QCQP"
        elif dataset=="ctu_13_neris":
            self.groups=np.array([[self.j[f"duration_{k}_{side}_{port}"] for k in ["min","max","sum"]]
                                 for port in PORTS for side in ["d","s"]])
            self.kind="convex_order_ball_QCQP_fixed_discrete"
        elif dataset=="url":
            self.kind="convex_box_ball_QCQP_fixed_discrete"
        elif dataset=="lcld_v2_iid":
            self.free=np.array([self.j[n] for n in ["loan_amnt","annual_inc","dti","revol_bal","revol_util"]])
            self.kind="nonlinear_continuous_completion_SLSQP"
        else:raise ValueError(dataset)

    def cost(self,x0,p):
        return (((p[:,self.numeric]-x0[:,self.numeric])/self.scale[self.numeric])**2).sum(1)

    def _poly(self,target,lower,upper):
        out=np.clip(target,lower,upper)
        if len(self.groups)==0:return out
        g=self.groups;s=self.scale[g]
        t=target[:,g]*s;lo=lower[:,g]*s;hi=upper[:,g]*s
        best=np.full(t.shape[:2],np.inf);chosen=out[:,g]*s
        # All contiguous partitions of a three-element isotonic chain.
        for parts in [((0,1),(1,2),(2,3)),((0,2),(2,3)),((0,1),(1,3)),((0,3),)]:
            q=np.empty_like(t);valid=np.ones(t.shape[:2],bool)
            for a,b in parts:
                w=1/s[:,a:b]**2
                v=(t[:,:,a:b]*w).sum(2)/w.sum(1)
                lb=lo[:,:,a:b].max(2);ub=hi[:,:,a:b].min(2)
                valid &= lb<=ub
                q[:,:,a:b]=np.clip(v,lb,ub)[:,:,None]
            valid &= (q[:,:,:-1]<=q[:,:,1:]).all(2)
            obj=(((q-t)/s)**2).sum(2)
            take=valid & (obj<best)
            chosen=np.where(take[:,:,None],q,chosen);best=np.where(take,obj,best)
        out[:,g]=chosen/s
        return out

    def _convex(self,x0,p):
        n=len(p)
        lower=np.broadcast_to(self.lo/self.scale,p.shape).copy()
        upper=np.broadcast_to(self.hi/self.scale,p.shape).copy()
        lower[:,self.fixed]=p[:,self.fixed]/self.scale[self.fixed]
        upper[:,self.fixed]=lower[:,self.fixed]
        # The fixed part includes all integer choices, so projecting cannot fix
        # a discrete-only equality or implication; the final guard rejects it.
        centre=x0/self.scale;desired=p/self.scale
        nearest_clean=self._poly(centre,lower,upper)
        min_cost=((nearest_clean-centre)[:,self.numeric]**2).sum(1)
        valid=np.isfinite(p).all(1) & (min_cost<=self.eps**2+1e-13)
        sol=self._poly(desired,lower,upper)
        over=((sol-centre)[:,self.numeric]**2).sum(1)>self.eps**2
        ids=np.flatnonzero(valid & over)
        if len(ids):
            left=np.zeros(len(ids));right=np.ones(len(ids))
            for _ in range(36):
                alpha=(left+right)/2
                q=self._poly(centre[ids]+alpha[:,None]*(desired[ids]-centre[ids]),
                             lower[ids],upper[ids])
                inside=((q-centre[ids])[:,self.numeric]**2).sum(1)<=self.eps**2
                left=np.where(inside,alpha,left);right=np.where(inside,right,alpha)
            sol[ids]=self._poly(centre[ids]+left[:,None]*(desired[ids]-centre[ids]),
                                lower[ids],upper[ids])
        out=sol*self.scale
        out[:,self.fixed]=p[:,self.fixed]
        return out,valid

    def _lcld(self,x0,p):
        out=p.copy();valid=np.ones(len(p),bool)
        j=self.j;f=self.free;sf=self.scale[f];r=self.real;sr=self.scale[r]
        jr={v:i for i,v in enumerate(r)}
        il=j["installment"];ratio=j["ratio_loan_amnt_annual_inc"]
        rat_ot=j["ratio_open_acc_total_acc"]
        for row in range(len(p)):
            original=x0[row];proposal=p[row];base=proposal.copy()
            op=proposal[j["open_acc"]];tot=proposal[j["total_acc"]]
            if tot<=0 or op>tot:
                valid[row]=False;continue
            base[rat_ot]=op/tot
            if not self.lo[rat_ot]<=base[rat_ot]<=self.hi[rat_ot]:
                valid[row]=False;continue
            ir=original[j["int_rate"]]/1200.;term=original[j["term"]]
            power=(1+ir)**term;k=ir*power/(power-1)
            lower=(self.lo[f]-original[f])/sf;upper=(self.hi[f]-original[f])/sf
            lower[0]=max(lower[0],(self.lo[il]/k-original[f[0]])/sf[0])
            upper[0]=min(upper[0],(self.hi[il]/k-original[f[0]])/sf[0])
            if np.any(lower>upper):valid[row]=False;continue
            def build(u):
                v=base.copy();v[f]=original[f]+sf*u
                loan,annual=v[f[:2]]
                v[il]=k*loan;v[ratio]=loan/annual
                jac=np.zeros((len(r),len(f)))
                for pos,col in enumerate(f):jac[jr[col],pos]=sf[pos]/self.scale[col]
                jac[jr[il],0]=k*sf[0]/self.scale[il]
                jac[jr[ratio],0]=sf[0]/annual/self.scale[ratio]
                jac[jr[ratio],1]=-loan*sf[1]/annual**2/self.scale[ratio]
                return v,jac
            def objective(u):
                v,J=build(u);d=(v[r]-proposal[r])/sr
                return d@d,2*J.T@d
            def constraints(u):
                v,J=build(u);dz=(v-original)/self.scale
                loan,annual=v[f[:2]]
                a=(loan-self.lo[ratio]*annual)/self.scale[f[0]]
                b=(self.hi[ratio]*annual-loan)/self.scale[f[0]]
                return np.array([self.eps**2-dz[self.numeric]@dz[self.numeric],a,b])
            def constraint_jac(u):
                v,J=build(u);d=(v[r]-original[r])/sr
                A=np.zeros((3,len(f)));A[0]=-2*J.T@d
                A[1,:2]=[1,-self.lo[ratio]*sf[1]/sf[0]]
                A[2,:2]=[-1,self.hi[ratio]*sf[1]/sf[0]]
                return A
            u0=np.clip(np.zeros(len(f)),lower,upper)
            # Integer and derived open/total ratio cost cannot be repaired
            # by changing independent continuous fields.
            fixed_cost=(((base[self.fixed]-original[self.fixed])/self.scale[self.fixed])**2).sum()
            fixed_cost+=((base[rat_ot]-original[rat_ot])/self.scale[rat_ot])**2
            if fixed_cost>self.eps**2:
                valid[row]=False;continue
            self.solver_calls+=1
            sol=minimize(objective,u0,jac=True,bounds=list(zip(lower,upper)),
                constraints={"type":"ineq","fun":constraints,"jac":constraint_jac},
                method="SLSQP",options={"maxiter":30,"ftol":1e-9,"disp":False})
            if not sol.success:
                self.solver_failed+=1;valid[row]=False;continue
            out[row]=build(sol.x)[0]
        return out,valid

    def solve(self,x0,proposed):
        p=np.asarray(proposed,dtype=np.float64).copy()
        p[:,~self.mutable]=x0[:,~self.mutable]
        cats=self.types=="cat";p[:,cats]=x0[:,cats]
        if self.dataset=="wids":
            out,valid=self.wids.solve(x0,p)
        elif self.dataset=="lcld_v2_iid":
            out,valid=self._lcld(x0,p)
        else:
            out,valid=self._convex(x0,p)
        valid &= self.judge.constraint_ok(x0,out)
        valid &= self.cost(x0,out)<=self.eps**2+1e-10
        valid &= (out[:,self.fixed]==p[:,self.fixed]).all(1)
        out[~valid]=x0[~valid]
        self.calls+=1;self.rows+=len(p);self.rejected+=int((~valid).sum())
        return out,valid
