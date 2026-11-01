"""Small validation-only, shared-hyperparameter study on URL and WiDS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts import wids_gumbel_ablation as base
from core.conditional_completion import ConditionalCompletion
from tabularbench.constraints.constraints_backend_executor import ConstraintsExecutor
from tabularbench.constraints.numpy_backend import NumpyBackend
from tabularbench.constraints.relation_constraint import LessEqualConstraint, OrConstraint, Constant, Feature
from tabularbench.constraints.constraints_checker import ConstraintChecker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/url_wids_gumbel_tuning"
MODELS = ["torchrln_default", "tabtransformer_default"]
DATASETS = ["url", "wids"]
CONFIGS = {
    "current": dict(tau_start=1., tau_end=.1, logits_lr=.06),
    "tau_end_03": dict(tau_start=1., tau_end=.3, logits_lr=.06),
    "tau_fixed_05": dict(tau_start=.5, tau_end=.5, logits_lr=.06),
    "logits_lr_003": dict(tau_start=1., tau_end=.1, logits_lr=.03),
    "logits_lr_012": dict(tau_start=1., tau_end=.1, logits_lr=.12),
}
ARGS = SimpleNamespace(eps=.5, queries=256, steps=200, lr=.06, batch=128)
ARCHIVE = {}
RESULTS = []


class Policy:
    def __init__(self, x0, meta, scaler):
        self.x0 = x0
        self.types = np.asarray(meta["type"])
        assert meta["mutable"].all()
        self.ints = np.flatnonzero(self.types == "int")
        self.reals = np.flatnonzero(self.types == "real")
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


def url_linear_relations(cons, meta, scale):
    # Current native URL OR constraints have a nonnegative integer antecedent
    # written as 0 <= a. Within metadata bounds this disjunct is always true.
    # Preserve official executable semantics, not the stronger source comments.
    A, b = [], []
    z = np.zeros((1, len(meta)))
    eye = np.eye(len(meta))
    for relation in cons.relation_constraints:
        if isinstance(relation, OrConstraint):
            first = relation.operands[0]
            assert isinstance(first, LessEqualConstraint)
            assert isinstance(first.left_operand, Constant) and first.left_operand.constant == 0
            assert isinstance(first.right_operand, Feature)
            col = first.right_operand.feature_id
            assert meta.iloc[col]["min"] >= 0
            continue
        assert isinstance(relation, LessEqualConstraint)
        left = ConstraintsExecutor(relation.left_operand, NumpyBackend(), cons.feature_names)
        right = ConstraintsExecutor(relation.right_operand, NumpyBackend(), cons.feature_names)
        bias = np.asarray(left.execute(z)-right.execute(z)).reshape(-1)[0]
        coeff = np.asarray(left.execute(eye)-right.execute(eye)).reshape(-1)-bias
        A.append(coeff); b.append(bias)
    A = torch.tensor(np.asarray(A), dtype=torch.float32, device=base.DEVICE)
    b = torch.tensor(b, dtype=torch.float32, device=base.DEVICE)
    norm = (A.abs()*scale[None]).max(1).values.clamp_min(1e-9)
    return A, b, norm


def completion_for(dataset, meta, scaler, judge):
    if dataset == "wids":
        return base.WidsCompletion(meta, scaler, ARGS.eps*.99)
    return ConditionalCompletion(dataset, meta, scaler, judge, ARGS.eps*.99)


def run(dataset, model_name, x, y, meta, scaler, model, cons, adapter, judge, cfg, seed, steps=200):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    meter = base.Meter(adapter, model, judge, x, y, base.DEVICE, ARGS.queries, ARGS.batch)
    x0 = torch.tensor(x, dtype=torch.float32, device=base.DEVICE)
    y0 = torch.tensor(y, dtype=torch.long, device=base.DEVICE)
    policy = Policy(x0, meta, scaler)
    if cfg["logits_lr"] == .06:
        opt = torch.optim.Adam([policy.theta, policy.d], lr=.06)
    else:
        opt = torch.optim.Adam([{"params":[policy.theta], "lr":cfg["logits_lr"]},
                                {"params":[policy.d], "lr":.06}])
    complete = completion_for(dataset, meta, scaler, judge)
    linear = url_linear_relations(cons, meta, policy.scale) if dataset == "url" else None
    start = time.perf_counter()
    for step in range(steps):
        ids = np.flatnonzero(~meter.ok & (meter.fwd < ARGS.queries-2) & (meter.bwd < steps))
        if not len(ids): break
        raw = policy.sample(step, steps, cfg)[ids]
        sol, allowed = complete.solve(x0[ids].detach().cpu().numpy(), raw.detach().cpu().numpy())
        hard = torch.as_tensor(sol, dtype=raw.dtype, device=base.DEVICE)
        candidate = raw + (hard-raw).detach()
        logits = meter.score(candidate, ids); meter.bwd[ids] += 1
        ar = torch.arange(len(ids), device=base.DEVICE)
        atk = (logits[ar,y0[ids]]-logits[ar,1-y0[ids]]).mean()
        dz = (raw-x0[ids])/policy.scale
        dist = dz[:,scaler.num_idx].norm(dim=1)
        budget = torch.relu(dist-.9*ARGS.eps).square().mean()
        if dataset == "wids":
            a,b = np.asarray(base.ORDER_PAIRS).T
            relation = torch.relu((raw[:,a]-raw[:,b])/
                                 torch.maximum(policy.scale[a],policy.scale[b])).square().sum(1).mean()
        else:
            A,b,norm = linear
            relation = torch.relu((raw@A.T+b)/norm).square().sum(1).mean()
        loss = atk+4*budget+4*relation
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pp = candidate.detach().cpu().numpy().astype(np.float64)
            rr = raw.detach().cpu().numpy().astype(np.float64)
            valid_raw = meter.valid(rr,ids)
            valid_eval = meter.accept(pp,ids,logits,allowed)
            meter.seen += len(ids)
            meter.raw_valid += int(valid_raw.sum())
            meter.eval_valid += int(valid_eval.sum())
            assert not ((rr[:,policy.ints] != pp[:,policy.ints]).any(1) & allowed).any()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    ok, distance, violation, _ = judge(x,y,meter.out)
    official = ConstraintChecker(cons,tolerance=0.).check_constraints(x,meter.out)
    np.testing.assert_array_equal(ok,meter.ok)
    np.testing.assert_array_equal(official,violation==0)
    assert meter.fwd.max() <= ARGS.queries and meter.bwd.max() <= steps
    assert official.all()
    return meter,dict(n=len(x),accepted=int(ok.sum()),asr=float(ok.mean()),
                      forward_mean=float(meter.fwd.mean()),forward_max=int(meter.fwd.max()),
                      backward_mean=float(meter.bwd.mean()),seconds=seconds,
                      audit_forward_per_sample=1,completion_rows=complete.rows,
                      completion_rejected=complete.rejected,raw_valid=meter.raw_valid,
                      eval_valid=meter.eval_valid,distance_max=float(distance.max()),
                      max_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)


def pools(ds, model, judge):
    X,Y = ds.get_x_y()
    val = np.asarray(ds.get_splits()["val"])
    assert not np.intersect1d(val,ds.get_splits()["test"]).size
    x = X.iloc[val].to_numpy(); y = np.asarray(Y)[val].astype(np.int64)
    valid = (y==1) & judge.constraint_ok(x,x)
    candidates = val[valid]
    rng = np.random.RandomState(20260906)
    candidates = candidates[rng.permutation(len(candidates))]
    halves = np.array_split(candidates,2)
    out = {}
    for phase, indices in zip(["development","confirmation"],halves):
        xx = X.iloc[indices].to_numpy(); yy = np.asarray(Y)[indices].astype(np.int64)
        pred = np.concatenate([model.predict_proba(xx[i:i+128]).argmax(1) for i in range(0,len(xx),128)])
        eligible = indices[pred==yy]
        assert len(eligible)>=64
        chosen = eligible[:64]
        out[phase] = (X.iloc[chosen].to_numpy().astype(np.float64),
                      np.asarray(Y)[chosen].astype(np.int64),chosen)
    assert not np.intersect1d(out["development"][2],out["confirmation"][2]).size
    return out


def components(dataset, model_name):
    ds,xt,yt,scaler,model,cons,adapter = base.load_victim(dataset,model_name,base.DEVICE)
    meta=ds.get_metadata(only_x=True).reset_index(drop=True)
    judge=base.BatchJudge(model,cons,scaler,ARGS.eps,box_tol_scale=1000 if dataset=="wids" else 1,rel_tol=0.)
    return ds,meta,scaler,model,cons,adapter,judge


def smoke():
    from scipy.optimize import minimize
    for dataset in DATASETS:
        ds,meta,scaler,model,cons,adapter,judge=components(dataset,"torchrln_default")
        x,y,_=pools(ds,model,judge)["development"]
        x,y=x[:8],y[:8]
        if dataset=="url":
            completion=completion_for(dataset,meta,scaler,judge)
            rng=np.random.RandomState(123)
            proposed=x.copy()
            rr=np.flatnonzero(np.asarray(meta["type"])=="real")
            scale=completion.scale[rr]
            proposed[:,rr]=np.clip(x[:,rr]+rng.normal(size=(len(x),len(rr)))*scale*.7,
                                   meta["min"].to_numpy()[rr],meta["max"].to_numpy()[rr])
            solution,valid=completion.solve(x,proposed)
            assert valid.all()
            fixed=np.flatnonzero(np.asarray(meta["type"])!="real")
            np.testing.assert_array_equal(solution[:,fixed],proposed[:,fixed])
            maxerr=0.
            for i in range(3):
                target=(proposed[i,rr]-x[i,rr])/scale
                lo=(meta["min"].to_numpy()[rr]-x[i,rr])/scale
                hi=(meta["max"].to_numpy()[rr]-x[i,rr])/scale
                fit=minimize(lambda u: np.sum((u-target)**2),np.zeros(len(rr)),
                             jac=lambda u:2*(u-target),bounds=list(zip(lo,hi)),
                             constraints={"type":"ineq","fun":lambda u:.495**2-np.dot(u,u),
                                          "jac":lambda u:-2*u},
                             method="SLSQP",options={"ftol":1e-12,"maxiter":100})
                assert fit.success
                ours=np.sum(((solution[i,rr]-proposed[i,rr])/scale)**2)
                maxerr=max(maxerr,abs(ours-fit.fun))
            assert maxerr<1e-7
            bad=x.copy();bad[:,1]=meta.iloc[1]["max"];bad[:,0]=meta.iloc[0]["min"]
            rejected,bvalid=completion.solve(x,bad)
            assert not bvalid.any()
            np.testing.assert_array_equal(rejected,x)
            print(json.dumps(dict(event="url_solver_reference_pass",max_objective_error=maxerr)),flush=True)
        meter,row=run(dataset,"torchrln_default",x,y,meta,scaler,model,cons,adapter,judge,CONFIGS["current"],0,6)
        if dataset=="wids":
            oldargs=SimpleNamespace(**vars(ARGS));oldargs.steps=6
            old,extra=base.run_variant("gumbel_inner",0,x,y,meta,scaler,adapter,model,judge,None,oldargs)
            for key in ["out","ok","fwd","bwd"]:
                np.testing.assert_array_equal(getattr(old,key),getattr(meter,key))
            print("WIDS_BASELINE_EXACT_PARITY",flush=True)
        print(json.dumps(dict(event="smoke_attack_pass",dataset=dataset,**row)),flush=True)
        del adapter,model
        torch.cuda.empty_cache()
    print("SMOKE_COMPLETE",flush=True)


def persist(phase,dataset,model_name,setting,seed,meter,row,indices,x,y):
    row.update(phase=phase,dataset=dataset,model=model_name,setting=setting,seed=seed)
    RESULTS.append(row)
    p=phase+"__"+dataset+"__"+model_name
    ARCHIVE[p+"__indices"]=indices;ARCHIVE[p+"__clean"]=x;ARCHIVE[p+"__labels"]=y
    k=p+"__"+setting+"__s"+str(seed)
    ARCHIVE[k+"__adv"]=meter.out;ARCHIVE[k+"__ok"]=meter.ok
    ARCHIVE[k+"__forward"]=meter.fwd;ARCHIVE[k+"__backward"]=meter.bwd
    (OUT/"results.json").write_text(json.dumps(RESULTS,indent=2))
    print(json.dumps(row),flush=True)


def select_setting():
    summary={}
    for setting in CONFIGS:
        rows=[r for r in RESULTS if r["phase"]=="development" and r["setting"]==setting]
        assert len(rows)==len(DATASETS)*len(MODELS)
        summary[setting]=dict(mean_asr=float(np.mean([r["asr"] for r in rows])),
                              mean_forward=float(np.mean([r["forward_mean"] for r in rows])),
                              dataset_asr={d:float(np.mean([r["asr"] for r in rows if r["dataset"]==d])) for d in DATASETS})
    # Always confirm the best altered setting against baseline; no claim of
    # improvement is made from development ranking, even if baseline wins.
    choices=[k for k in CONFIGS if k!="current"]
    chosen=sorted(choices,key=lambda k:(-summary[k]["mean_asr"],summary[k]["mean_forward"],k))[0]
    result=dict(selected=chosen,configs=CONFIGS,development_scores=summary,
                rule="One common changed setting ranked by macro ASR over both datasets and both models; query mean breaks ties. Confirm regardless of gain.")
    (OUT/"selection.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event="selection",**result)),flush=True)
    return chosen



def expand_if_confirmed(chosen):
    decision = json.loads((OUT/"selection.json").read_text())
    scores = {}
    for dataset in DATASETS:
        scores[dataset] = {}
        for setting in ["current",chosen]:
            rows = [r for r in RESULTS if r["phase"]=="confirmation" and
                    r["dataset"]==dataset and r["setting"]==setting]
            assert len(rows)==len(MODELS)*3
            scores[dataset][setting] = float(np.mean([r["asr"] for r in rows]))
    gains = {d:scores[d][chosen]-scores[d]["current"] for d in DATASETS}
    passed = all(v>=-1e-12 for v in gains.values()) and np.mean(list(gains.values()))>1e-12
    decision.update(confirmation_scores=scores,confirmation_asr_gains=gains,
                    full_test_authorized_by_gate=bool(passed),
                    gate="Each dataset confirmation mean ASR must not decrease; macro mean must strictly increase.")
    (OUT/"selection.json").write_text(json.dumps(decision,indent=2))
    print(json.dumps(dict(event="confirmation_gate",**decision)),flush=True)
    if not passed:
        print("NO_CONFIRMED_CROSS_DATASET_GAIN",flush=True)
        return
    for dataset in DATASETS:
        for model_name in base.DEFAULT_MODELS:
            ds,meta,scaler,model,cons,adapter,judge=components(dataset,model_name)
            X,Y=ds.get_x_y()
            indices=np.asarray(ds.get_splits()["test"])
            xx=X.iloc[indices].to_numpy()
            yy=np.asarray(Y)[indices].astype(np.int64)
            valid=(yy==1)&judge.constraint_ok(xx,xx)
            indices=indices[valid];xx=xx[valid];yy=yy[valid]
            denominator=len(indices)
            pred=np.concatenate([model.predict_proba(xx[i:i+128]).argmax(1) for i in range(0,len(xx),128)])
            indices=indices[pred==yy]
            x=X.iloc[indices].to_numpy().astype(np.float64)
            y=np.asarray(Y)[indices].astype(np.int64)
            print(json.dumps(dict(event="full_pool",dataset=dataset,model=model_name,
                                  clean_correct=len(x),valid_class1=denominator)),flush=True)
            for seed in [0,1,2]:
                for setting in ["current",chosen]:
                    meter,row=run(dataset,model_name,x,y,meta,scaler,model,cons,adapter,judge,CONFIGS[setting],seed)
                    row.update(robust_acc=(len(x)-row["accepted"])/denominator,
                               clean_acc=len(x)/denominator,robust_acc_denominator=denominator)
                    persist("full_test",dataset,model_name,setting,seed,meter,row,indices,x,y)
            np.savez_compressed(OUT/"adversarial.npz",**ARCHIVE)
            del adapter,model
            torch.cuda.empty_cache()
    print("FULL_TEST_COMPLETE",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert base.DEVICE=="cuda"
    if args.smoke:
        smoke();return
    OUT.mkdir(exist_ok=True)
    assert not (OUT/"results.json").exists()
    sources=[Path(__file__),ROOT/"core/conditional_completion.py",ROOT/"core/wids_completion.py",
             ROOT/"scripts/wids_gumbel_ablation.py",ROOT/"core/judge.py",
             Path(os.environ["TABULARBENCH_ROOT"])/"tabularbench/datasets/samples/url.py"]
    manifest=dict(datasets=DATASETS,models=MODELS,configs=CONFIGS,n_per_model_phase=64,
                  development_seeds=[0],confirmation_seeds=[0,1,2],split_seed=20260906,
                  split="Official validation set only; class1 and strict valid; shuffled into disjoint halves before model correctness filtering; first64 eligible per half.",
                  selection="One shared setting for both datasets and both models; macro ASR then mean forwards.",
                  query_cap=256,steps=200,eps=.5,completion_eps=.495,real_lr=.06,
                  prior_distance=40,prior_clean_bonus=2,grid_slots=31,
                  penalty_weights=[4,4],budget_soft_fraction=.9,
                  device="cuda:0",batch=128,cpu_threads=2,full_test_run="Conditional on independent confirmation",
                  expansion_gate="Per-dataset mean ASR nondecreasing and macro mean strictly increasing on confirmation; then both datasets, all five models, all eligible test records, seeds 0/1/2, current and selected.",
                  sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    for phase in ["development","confirmation"]:
        chosen=None if phase=="development" else select_setting()
        settings=list(CONFIGS) if phase=="development" else ["current",chosen]
        seeds=[0] if phase=="development" else [0,1,2]
        for dataset in DATASETS:
            for model_name in MODELS:
                ds,meta,scaler,model,cons,adapter,judge=components(dataset,model_name)
                x,y,indices=pools(ds,model,judge)[phase]
                print(json.dumps(dict(event="pool",phase=phase,dataset=dataset,model=model_name,n=len(x))),flush=True)
                for seed in seeds:
                    for setting in settings:
                        meter,row=run(dataset,model_name,x,y,meta,scaler,model,cons,adapter,judge,CONFIGS[setting],seed)
                        persist(phase,dataset,model_name,setting,seed,meter,row,indices,x,y)
                np.savez_compressed(OUT/"adversarial.npz",**ARCHIVE)
                del adapter,model
                torch.cuda.empty_cache()
    print("TUNING_COMPLETE",flush=True)
    expand_if_confirmed(chosen)
    print("EXPERIMENT_COMPLETE",flush=True)


if __name__=="__main__":main()
