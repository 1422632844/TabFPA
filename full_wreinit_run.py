"""TabFPA full-pool runs for URL default, WiDS/CTU Madry and Malware default.

Uses batches of 128, per-batch seed = seed*1000003+offset, and a fixed
200-step schedule with reinitialization at round 100. One target per run."""
import os, sys, io, gc, json, time, sqlite3, fcntl, argparse
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "method"), str(ROOT)]
import experiment_common as common
import malware_extension
import widen_kernel_wreinit as widen_kernel

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["url", "wids", "ctu_13_neris", "malware"])
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n", type=int, default=0, help=">0: limit clean-correct rows for a smoke check (not a full-pool result)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--reinit-round", type=int, default=100)
    a = p.parse_args()
    regime = "madry" if a.dataset in ("wids", "ctu_13_neris") else "default"
    models = [f"{arch}_{regime}" for arch in ("tabnet", "tabtransformer", "torchrln", "vime", "stg")]
    if a.model not in models:
        p.error(f"This entry requires a {regime} target for {a.dataset}; see README for other schedules.")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    out = Path(os.environ.get("WREINIT_OUT_ROOT", str(ROOT))) / f"runs/{a.dataset}_full_wreinit" / a.model
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(mode=f"{a.dataset}_full_wreinit", model=a.model, seeds=a.seeds, steps=200,
                  optimization_batch=a.batch, query_cap=256, eps=.5, continuous_lr=.06,
                  logits_lr=.48, init_kappa=40, reinit_round=a.reinit_round, reinit_kappa=0,
                  pool="same as runs/full: legal class1 test; per-model clean correct; original order")
    (out / "manifest.json").write_text(json.dumps(config, indent=2))
    db = sqlite3.connect(out / "state.sqlite3")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS scenes(key TEXT PRIMARY KEY,n INT,total INT,denominator INT,state TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS batches(scene TEXT,offset INT,metrics TEXT,artifact BLOB,PRIMARY KEY(scene,offset))")
    db.commit()
    def emit(state, **kw):
        row = dict(status=state, time=time.time(), **kw)
        (out / "status.json").write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)
    emit("running", dataset=a.dataset, model=a.model)
    xall, indices, meta, scaler, cons = common.dataset_data(a.dataset)
    cpu, adapter, judge = common.victims(a.dataset, a.model, meta, scaler, cons, xall)
    legal = np.concatenate([judge.constraint_ok(xall[i:i + 128], xall[i:i + 128]) for i in range(0, len(xall), 128)])
    valid_x = xall[legal]; valid_ix = indices[legal]; den = len(valid_x)
    pred = np.concatenate([cpu.predict_proba(valid_x[i:i + 128]).argmax(1) for i in range(0, len(valid_x), 128)])
    x = valid_x[pred == 1]; ix = valid_ix[pred == 1]; total = len(x)
    if a.n:
        x = x[:a.n]; ix = ix[:a.n]; total = len(x)
    if total == 0:
        # model classifies no legal class-1 row as positive: nothing to attack,
        # RA=(N-S)/D=0 by formula; record and finish cleanly.
        emit("pool", dataset=a.dataset, model=a.model, valid_class1=den, clean_correct=0)
        for seed in a.seeds:
            key = f"{a.dataset}/{a.model}/wreinit/{seed}"
            with db:
                db.execute("INSERT OR REPLACE INTO scenes VALUES(?,?,?,?,?)", (key, 0, 0, den, "complete"))
            emit("scene_complete", scene=key, S=0, N=0, D=den, RA=0.0)
        emit("complete", dataset=a.dataset, model=a.model)
        return
    kw = {}
    if a.dataset == "malware":
        kw = dict(moves_cls=malware_extension.MalwareMoves, completion_ctor=malware_extension.MalwareCompletion)
    emit("pool", dataset=a.dataset, model=a.model, valid_class1=den, clean_correct=total)
    for seed in a.seeds:
        key = f"{a.dataset}/{a.model}/wreinit/{seed}"
        with db:
            db.execute("INSERT OR REPLACE INTO scenes VALUES(?,?,?,?,?)", (key, len(x), total, den, "running"))
        for start in range(0, len(x), a.batch):
            if db.execute("SELECT 1 FROM batches WHERE scene=? AND offset=?", (key, start)).fetchone():
                continue
            xx = x[start:start + a.batch]
            meter, row = widen_kernel.run(a.dataset, a.model, meta, scaler, cons, cpu,
                                          adapter, judge, xx, seed * 1000003 + start,
                                          steps=200, logits_lr=.48, kappa=40.,
                                          reinit_round=a.reinit_round, reinit_kappa=0., **kw)
            row.update(experiment_seed=seed, offset=start)
            b = io.BytesIO()
            np.savez_compressed(b, adv=meter.out, ok=meter.ok, forward=meter.fwd,
                                backward=meter.bwd, indices=ix[start:start + a.batch])
            with db:
                db.execute("INSERT OR REPLACE INTO batches VALUES(?,?,?,?)", (key, start, json.dumps(row, default=str), b.getvalue()))
            emit("batch_committed", scene=key, offset=start, n=row["n"], accepted=row["accepted"])
            del meter; gc.collect(); torch.cuda.empty_cache()
        with db:
            db.execute("UPDATE scenes SET state='complete' WHERE key=?", (key,))
        S = sum(int(json.loads(m)["accepted"]) for _, m in
                db.execute("SELECT offset, metrics FROM batches WHERE scene=?", (key,)).fetchall())
        emit("scene_complete", scene=key, S=S, N=total, D=den, RA=round(100.0 * (total - S) / den, 2))
    emit("complete", dataset=a.dataset, model=a.model)

if __name__ == "__main__":
    main()
