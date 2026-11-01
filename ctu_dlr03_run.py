"""TabFPA full-pool CTU default evaluation with continuous learning rate 0.03.

Uses the fixed 200-step schedule, batches of 128, per-batch seeds and
resumable SQLite outputs under runs/ctu_dlr03_full/<model>/."""
import os, sys, io, gc, json, time, sqlite3, fcntl, argparse
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "method"), str(ROOT)]
import experiment_common as common
import widen_kernel_ballmap_dir as widen_kernel  # current official kernel superset; d_lr exposed, loop identical at defaults


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=[f"{arch}_default" for arch in ("tabnet", "tabtransformer", "torchrln", "vime", "stg")])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    out = Path(os.environ.get("WREINIT_OUT_ROOT", str(ROOT))) / "runs/ctu_dlr03_full" / a.model
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(mode="ctu_dlr03_full", model=a.model, seeds=a.seeds, steps=200,
                  optimization_batch=a.batch, query_cap=256, eps=.5,
                  continuous_lr=.06, d_lr=.03, logits_lr=.48, init_kappa=40,
                  reinit_round=100, reinit_kappa=0,
                  note="frozen CTU default continuous learning rate",
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
    emit("running", dataset="ctu_13_neris", model=a.model)
    xall, indices, meta, scaler, cons = common.dataset_data("ctu_13_neris")
    cpu, adapter, judge = common.victims("ctu_13_neris", a.model, meta, scaler, cons, xall)
    legal = np.concatenate([judge.constraint_ok(xall[i:i + 128], xall[i:i + 128]) for i in range(0, len(xall), 128)])
    valid_x = xall[legal]; valid_ix = indices[legal]; den = len(valid_x)
    pred = np.concatenate([cpu.predict_proba(valid_x[i:i + 128]).argmax(1) for i in range(0, len(valid_x), 128)])
    x = valid_x[pred == 1]; ix = valid_ix[pred == 1]; total = len(x)
    if total == 0:
        emit("pool", dataset="ctu_13_neris", model=a.model, valid_class1=den, clean_correct=0)
        for seed in a.seeds:
            key = f"ctu_13_neris/{a.model}/dlr03/{seed}"
            with db:
                db.execute("INSERT OR REPLACE INTO scenes VALUES(?,?,?,?,?)", (key, 0, 0, den, "complete"))
            emit("scene_complete", scene=key, S=0, N=0, D=den, RA=0.0)
        emit("complete", dataset="ctu_13_neris", model=a.model)
        return
    emit("pool", dataset="ctu_13_neris", model=a.model, valid_class1=den, clean_correct=total)
    for seed in a.seeds:
        key = f"ctu_13_neris/{a.model}/dlr03/{seed}"
        with db:
            db.execute("INSERT OR REPLACE INTO scenes VALUES(?,?,?,?,?)", (key, len(x), total, den, "running"))
        for start in range(0, len(x), a.batch):
            if db.execute("SELECT 1 FROM batches WHERE scene=? AND offset=?", (key, start)).fetchone():
                continue
            xx = x[start:start + a.batch]
            meter, row = widen_kernel.run("ctu_13_neris", a.model, meta, scaler, cons, cpu,
                                          adapter, judge, xx, seed * 1000003 + start,
                                          steps=200, logits_lr=.48, kappa=40.,
                                          reinit_round=100, reinit_kappa=0., d_lr=.03)
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
        n_done = db.execute("SELECT COALESCE(SUM(json_extract(metrics,'$.accepted')),0) FROM batches WHERE scene=?", (key,)).fetchone()[0]
        emit("scene_complete", scene=key, S=n_done, N=total, D=den, RA=round(100.0 * (total - n_done) / den, 2))
    emit("complete", dataset="ctu_13_neris", model=a.model)


if __name__ == "__main__":
    main()
