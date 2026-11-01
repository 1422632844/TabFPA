"""LOCAL WRAPPER / AUDIT: portable, resumable evaluation of the final repaired attack."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("TABULARBENCH_ROOT",str(ROOT/"tabularbench"))
os.environ.setdefault("TABULARBENCH_DATASET_DIR",str(ROOT/"tabularbench/datasets"))
os.environ.setdefault("TABULARBENCH_TARGET_MODEL_DIR",str(ROOT/"tabularbench/target_model"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE","1")
import sys
sys.path[:0]=[str(ROOT/"method"),str(ROOT/"tabularbench")]
import argparse,csv,fcntl,gc,hashlib,io,json,sqlite3,time
import numpy as np
import torch
from scripts import feasible_gumbel_ablation as method
from core.judge import BatchJudge
from models import VictimAdapter,DEFAULT_MODELS
from tabularbench.benchmark.model_utils import load_model_and_weights
from tabularbench.datasets.dataset_factory import get_dataset
from tabularbench.models.tab_scaler import TabScaler

DATASETS=["url","wids","ctu_13_neris","lcld_v2_iid"]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()

def dataset_data(name):
    if name=="ctu_13_neris":
        return method.ctu.data()
    ds=get_dataset(name)
    X,Y=ds.get_x_y()
    meta=ds.get_metadata(only_x=True).reset_index(drop=True)
    scaler=TabScaler(num_scaler="min_max",one_hot_encode=True)
    scaler.fit(torch.tensor(X.to_numpy(),dtype=torch.float32),x_type=meta.type)
    ix=np.asarray(ds.get_splits()["test"])
    ix=ix[np.asarray(Y)[ix]==1]
    return X.iloc[ix].to_numpy(dtype=np.float64),ix,meta,scaler,ds.get_constraints()

def victims(ds,name,meta,scaler,cons,probe):
    arch,training=name.split("_",1)
    cpu=load_model_and_weights(ds,arch,training,meta,scaler,"cpu")
    gpu=load_model_and_weights(ds,arch,training,meta,scaler,"cuda")
    gpu.wrapper_model.eval();gpu.wrapper_model.requires_grad_(False)
    adapter=VictimAdapter(gpu.wrapper_model,scaler,False).cuda()
    judge=BatchJudge(cpu,cons,scaler,.5,box_tol_scale=1000 if ds=="wids" else 1,rel_tol=0.)
    with torch.no_grad():
        actual=adapter(torch.tensor(probe[:32],dtype=torch.float32,device="cuda")).cpu().numpy()
    expected=cpu.predict_proba(probe[:32])
    np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
    return cpu,adapter,judge

def dump_summary(db,out):
    rows=[]
    for key,n,total,denominator,state in db.execute("SELECT key,n,total,denominator,state FROM scenes ORDER BY key"):
        parts=key.split("/")
        batches=[json.loads(r[0]) for r in db.execute("SELECT metrics FROM batches WHERE scene=? ORDER BY offset",(key,))]
        seen=sum(r["n"] for r in batches);success=sum(r["accepted"] for r in batches)
        rows.append(dict(dataset=parts[0],model=parts[1],logits_lr=float(parts[2]),seed=int(parts[3]),
                         status=state,attacked=seen,attack_pool=n,clean_correct=total,denominator=denominator,
                         accepted=success,asr=success/seen if seen else "",
                         robust_acc=(total-success)/denominator if state=="complete" and n==total else "",
                         conditional_robust_acc=1-success/seen if state=="complete" and seen else "",
                         forward_mean=sum(r["n"]*r["forward_mean"] for r in batches)/seen if seen else "",
                         forward_max=max((r["forward_max"] for r in batches),default=0)))
    if rows:
        tmp=out/"summary.csv.tmp"
        with tmp.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        tmp.replace(out/"summary.csv")
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["audit","smoke","full"],default="smoke")
    p.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
    p.add_argument("--models",nargs="+",choices=DEFAULT_MODELS,default=DEFAULT_MODELS)
    p.add_argument("--logits-lrs",nargs="+",type=float,choices=[.06,.12],default=[.06,.12])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--max-batches",type=int,default=0,help="Operational stop after committing this many new batches")
    a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=ROOT/"runs"/a.mode;out.mkdir(parents=True,exist_ok=True)
    lock=(out/"worker.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seeds=a.seeds if a.mode=="full" else [0]
    steps=200 if a.mode=="full" else 6
    config=dict(mode=a.mode,datasets=a.datasets,models=a.models,logits_lrs=a.logits_lrs,seeds=seeds,
                steps=steps,optimization_batch=a.batch,meter_batch=128,query_cap=256,
                continuous_lr=.06,tau=[1,.1],eps=.5,completion_eps=.495,
                pool="official test; strictly valid class1; per-target clean correct; original row order",
                method="joint_integer_minimum_continuous_anchor",
                source_files={str(q.relative_to(ROOT)):digest(q) for q in [Path(__file__),*sorted((ROOT/"method").rglob("*.py")),*sorted((ROOT/"tabularbench/tabularbench").rglob("*.py"))]},
                assets_manifest_sha256=digest(ROOT/"asset_manifest.json"),
                torch_version=torch.__version__,numpy_version=np.__version__,
                device=torch.cuda.get_device_name(0))
    path=out/"manifest.json"
    if path.exists():
        assert json.loads(path.read_text())==config,"Resume configuration/source/environment differs"
    else:path.write_text(json.dumps(config,indent=2))
    db=sqlite3.connect(out/"state.sqlite3")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS scenes(key TEXT PRIMARY KEY,n INT,total INT,denominator INT,state TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS batches(scene TEXT,offset INT,metrics TEXT,artifact BLOB,PRIMARY KEY(scene,offset))")
    db.execute("CREATE TABLE IF NOT EXISTS pools(dataset TEXT,model TEXT,artifact BLOB,PRIMARY KEY(dataset,model))")
    db.commit()
    support={};new_batches=0;skipped=0
    def status(state):
        (out/"status.json").write_text(json.dumps(dict(status=state,new_batches=new_batches,skipped_batches=skipped,time=time.time()),indent=2))
    status("running")
    for ds in a.datasets:
        xall,indices,meta,scaler,cons=dataset_data(ds)
        try:
            moves=method.JointIntegerMoves(meta,scaler,cons)
            support[ds]=dict(attack_supported=True,directions=moves.nd)
        except (ValueError,AssertionError) as e:
            support[ds]=dict(attack_supported=False,reason=type(e).__name__+": "+str(e))
        models_checked=[]
        for name in a.models:
            cpu,adapter,judge=victims(ds,name,meta,scaler,cons,xall)
            legal=np.concatenate([judge.constraint_ok(xall[i:i+128],xall[i:i+128]) for i in range(0,len(xall),128)])
            valid_x=xall[legal];valid_ix=indices[legal];den=len(valid_x)
            pred=np.concatenate([cpu.predict_proba(valid_x[i:i+128]).argmax(1) for i in range(0,len(valid_x),128)])
            x=valid_x[pred==1];ix=valid_ix[pred==1];total=len(x)
            assert total>0
            models_checked.append(dict(model=name,valid_class1=den,clean_correct=total))
            print(json.dumps(dict(event="pool",dataset=ds,model=name,valid_class1=den,clean_correct=total,
                                  attack_supported=support[ds]["attack_supported"])),flush=True)
            if a.mode=="audit" or not support[ds]["attack_supported"]:
                del cpu,adapter,judge;gc.collect();torch.cuda.empty_cache();continue
            if a.mode=="smoke":x=x[:8];ix=ix[:8]
            buf=io.BytesIO();np.savez_compressed(buf,clean=x,indices=ix)
            with db:db.execute("INSERT OR IGNORE INTO pools VALUES(?,?,?)",(ds,name,buf.getvalue()))
            for lr in a.logits_lrs:
                for seed in seeds:
                    key=f"{ds}/{name}/{lr}/{seed}"
                    with db:db.execute("INSERT OR IGNORE INTO scenes VALUES(?,?,?,?,?)",(key,len(x),total,den,"running"))
                    for start in range(0,len(x),a.batch):
                        if db.execute("SELECT 1 FROM batches WHERE scene=? AND offset=?",(key,start)).fetchone():
                            skipped+=1;continue
                        xx=x[start:start+a.batch]
                        meter,row=method.run(ds,name,meta,scaler,cons,cpu,adapter,judge,xx,
                                             seed*1000003+start,steps=steps,logits_lr=lr)
                        row.update(experiment_seed=seed,offset=start,logits_lr=lr)
                        b=io.BytesIO();np.savez_compressed(b,adv=meter.out,ok=meter.ok,forward=meter.fwd,backward=meter.bwd,indices=ix[start:start+a.batch])
                        with db:db.execute("INSERT INTO batches VALUES(?,?,?,?)",(key,start,json.dumps(row),b.getvalue()))
                        new_batches+=1
                        print(json.dumps(dict(event="batch_committed",scene=key,offset=start,n=row["n"],accepted=row["accepted"])),flush=True)
                        del meter
                        if a.max_batches and new_batches>=a.max_batches:
                            dump_summary(db,out);status("paused_after_checkpoint");db.close();return
                    with db:db.execute("UPDATE scenes SET state='complete' WHERE key=?",(key,))
                    dump_summary(db,out)
            del cpu,adapter,judge;gc.collect();torch.cuda.empty_cache()
        support[ds]["models"]=models_checked
        (out/"support.json").write_text(json.dumps(support,indent=2))
    rows=dump_summary(db,out)
    assert db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
    db.close()
    blocked=[d for d,v in support.items() if not v["attack_supported"]]
    status("complete_with_unsupported_datasets" if blocked else "complete")
    print(json.dumps(dict(event="EVALUATION_COMPLETE",unsupported=blocked,scenes=len(rows),
                          skipped_batches=skipped,new_batches=new_batches)),flush=True)

if __name__=="__main__":main()

