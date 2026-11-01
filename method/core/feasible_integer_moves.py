"""Joint integer moves that preserve affine equalities.

Scope: affine equalities/inequalities and Feature/Feature SafeDivision <=
constant, plus OR constraints proven tautological by declared bounds.
Unsupported nonlinear relations raise instead of silently weakening legality.
"""
import numpy as np
import torch
from tabularbench.constraints.relation_constraint import (
    Feature,Constant,MathOperation,ManySum,EqualConstraint,LessEqualConstraint,
    OrConstraint,SafeDivision)

class JointIntegerMoves:
    def __init__(self,meta,scaler,cons,eps=.495):
        self.names=meta.feature.tolist();self.nf=len(meta)
        self.types=meta.type.to_numpy();self.mutable=meta.mutable.to_numpy(bool)
        self.lo=meta['min'].to_numpy(float);self.hi=meta['max'].to_numpy(float)
        self.scale=np.ones(self.nf)
        self.scale[scaler.num_idx]=scaler.min_max_scale.detach().cpu().numpy()
        observed=np.zeros(self.nf)
        observed[scaler.num_idx]=(scaler.x_max-scaler.x_min).detach().cpu().numpy()
        self.ints=np.flatnonzero(self.mutable&(self.types=='int')&(observed>0)&(self.hi>self.lo))
        self.reals=np.flatnonzero(self.mutable&(self.types=='real'))
        self.eps=eps;self.packets=[]
        E=[];eb=[];L=[];lb=[]
        for relation in cons.relation_constraints:
            if isinstance(relation,OrConstraint):
                tautology=False
                for branch in relation.operands:
                    if isinstance(branch,LessEqualConstraint):
                        a,b=self.affine_difference(branch)
                        maximum=(np.maximum(a,0)*self.hi+np.minimum(a,0)*self.lo).sum()+b
                        tautology|=maximum<=0
                if not tautology:raise ValueError('OR is not proven tautological')
                continue
            if isinstance(relation,LessEqualConstraint) and isinstance(relation.left_operand,SafeDivision):
                d=relation.left_operand
                assert isinstance(d.dividend,Feature) and isinstance(d.divisor,Feature)
                assert isinstance(d.fill_value,Constant) and d.fill_value.constant==0
                assert isinstance(relation.right_operand,Constant)
                bound=float(relation.right_operand.constant);assert bound>=0
                self.packets.append((self.column(d.dividend),self.column(d.divisor),bound))
                continue
            a,b=self.affine_difference(relation)
            if isinstance(relation,EqualConstraint):
                assert relation.tolerance is None
                E.append(a);eb.append(b)
            elif isinstance(relation,LessEqualConstraint):L.append(a);lb.append(b)
            else:raise ValueError(type(relation).__name__)
        self.E=np.asarray(E).reshape(-1,self.nf);self.eb=np.asarray(eb)
        self.L=np.asarray(L).reshape(-1,self.nf);self.lb=np.asarray(lb)
        # Group proportional nonzero equality columns. For CTU all nonzero
        # columns have one +/-1 entry, so star moves span its integer kernel.
        groups={};directions=[]
        for j in self.ints:
            col=self.E[:,j]
            if not np.any(col):
                v=np.zeros(self.nf,dtype=np.int64);v[j]=1;directions.append(v)
            else:
                nz=np.flatnonzero(col);key=tuple(col/col[nz[0]])
                groups.setdefault(key,[]).append(j)
        for columns in groups.values():
            pivot=max(columns,key=lambda j:self.scale[j])
            for j in columns:
                if j==pivot:continue
                row=np.flatnonzero(self.E[:,pivot])[0]
                aa=self.E[row,j];bb=self.E[row,pivot]
                assert aa==int(aa) and bb==int(bb)
                g=np.gcd(abs(int(aa)),abs(int(bb)))
                v=np.zeros(self.nf,dtype=np.int64);v[j]=int(bb)//g;v[pivot]=-int(aa)//g
                directions.append(v)
        self.D=np.asarray(directions,dtype=np.int64)
        assert len(self.D)>0
        assert not np.any(self.E@self.D.T)
        expected=len(self.ints)-(np.linalg.matrix_rank(self.E[:,self.ints]) if len(self.E) else 0)
        assert np.linalg.matrix_rank(self.D[:,self.ints])==expected,'Move basis incomplete for these equalities'
        self.nd=len(self.D);self.slopes=self.L@self.D.T
        self.support=[np.flatnonzero(d) for d in self.D]
        self.affected=[np.flatnonzero(self.slopes[:,j]) for j in range(self.nd)]
        self.packet_support=[]
        for b,p,cap in self.packets:
            ids=np.flatnonzero((self.D[:,b]!=0)|(self.D[:,p]!=0))
            self.packet_support.append((b,p,cap,ids))
        self.a=((self.D/self.scale)**2).sum(1)
        self.TD=None

    def column(self,node):
        return self.names.index(node.feature_id) if isinstance(node.feature_id,str) else int(node.feature_id)

    def affine(self,node):
        if isinstance(node,Feature):
            a=np.zeros(self.nf);a[self.column(node)]=1;return a,0.
        if isinstance(node,Constant):return np.zeros(self.nf),float(node.constant)
        if isinstance(node,ManySum):
            terms=[self.affine(n) for n in node.operands]
            return sum(t[0] for t in terms),sum(t[1] for t in terms)
        if isinstance(node,MathOperation):
            a,b=self.affine(node.left_operand);c,d=self.affine(node.right_operand)
            if node.operator=='+':return a+c,b+d
            if node.operator=='-':return a-c,b-d
            if node.operator=='*':
                if not np.any(a):return c*b,d*b
                if not np.any(c):return a*d,b*d
            raise ValueError('Non-affine expression')
        raise ValueError(type(node).__name__)

    def affine_difference(self,node):
        a,b=self.affine(node.left_operand);c,d=self.affine(node.right_operand)
        return a-c,b-d

    def candidates(self,clean,current):
        n=len(clean)
        z=(current-clean)/self.scale
        b=2*(z/self.scale)@self.D.T
        cost=(z*z).sum(1)-self.eps**2
        disc=np.maximum(b*b-4*cost[:,None]*self.a,0)
        low=(-b-np.sqrt(disc))/(2*self.a)
        high=(-b+np.sqrt(disc))/(2*self.a)
        residual=-(current@self.L.T+self.lb)
        for j,cols in enumerate(self.support):
            v=self.D[j,cols]
            bl=(self.lo[cols]-current[:,cols])/v
            bh=(self.hi[cols]-current[:,cols])/v
            low[:,j]=np.maximum(low[:,j],np.minimum(bl,bh).max(1))
            high[:,j]=np.minimum(high[:,j],np.maximum(bl,bh).min(1))
            rows=self.affected[j];sl=self.slopes[rows,j]
            if len(rows):
                t=residual[:,rows]/sl
                if (sl>0).any():high[:,j]=np.minimum(high[:,j],t[:,sl>0].min(1))
                if (sl<0).any():low[:,j]=np.maximum(low[:,j],t[:,sl<0].max(1))
        low=np.ceil(low);high=np.floor(high)
        low=np.minimum(low,0);high=np.maximum(high,0)
        # Four magnitudes in each direction; includes exact +/-1.
        positive=np.stack([np.minimum(high,1),np.floor(high*.1),np.floor(high*.5),high],axis=2)
        negative=np.stack([np.maximum(low,-1),np.ceil(low*.1),np.ceil(low*.5),low],axis=2)
        values=np.concatenate([negative,positive],axis=2).astype(np.int64)
        mask=values!=0
        for k in range(1,8):mask[:,:,k]&=~(values[:,:,k,None]==values[:,:,:k]).any(2)
        for bi,pi,cap,ids in self.packet_support:
            if not len(ids):continue
            num=current[:,bi,None,None]+values[:,ids]*self.D[ids,bi][None,:,None]
            den=current[:,pi,None,None]+values[:,ids]*self.D[ids,pi][None,:,None]
            mask[:,ids]&=(den==0)|(num<=cap*den)
        # Each action was bounded jointly using all moved coordinates and
        # the distance from the original clean sample, never the last step.
        return values,mask

    def sample(self,current,values,mask,theta,tau):
        device=theta.device
        if self.TD is None:self.TD=torch.tensor(self.D[:,self.ints],dtype=torch.float32,device=device)
        n=len(current)
        valid=np.concatenate([np.ones((n,1),bool),mask.reshape(n,-1)],axis=1)
        noise=-torch.log(-torch.log(torch.rand_like(theta).clamp(1e-7,1-1e-7)))
        soft=((theta.masked_fill(~torch.as_tensor(valid,device=device),-1e9)+noise)/tau).softmax(-1)
        choice=soft.argmax(1)
        hard=torch.zeros_like(soft).scatter_(1,choice[:,None],1.)
        weights=hard-soft.detach()+soft
        coefficients=(weights[:,1:].reshape(n,self.nd,8)*torch.as_tensor(values,dtype=torch.float32,device=device)).sum(2)
        soft_int=torch.as_tensor(current[:,self.ints],dtype=torch.float32,device=device)+coefficients@self.TD
        raw=current.copy()
        which=choice.detach().cpu().numpy()-1;active=np.flatnonzero(which>=0)
        dirs=which[active]//8;slots=which[active]%8
        steps=values[active,dirs,slots]
        raw[active]+=self.D[dirs]*steps[:,None]
        exact=torch.as_tensor(raw[:,self.ints],dtype=torch.float32,device=device)
        straight=soft_int+(exact-soft_int).detach()
        return raw,straight,valid.sum(1)
