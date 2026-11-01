"""Train XGBoost and LightGBM targets on the official TabularBench splits.

Retains the settings of the research surrogates.py training functions.
Raw features are converted to float32; Malware uses a sparse CSR matrix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "method"))
DATASETS = ("url", "wids", "lcld_v2_iid", "ctu_13_neris", "malware")


def load_data(name):
    import numpy as np
    from benchmark_adapter import get_dataset

    dataset = get_dataset(name)
    assets = [Path(dataset.data_source.path), Path(dataset.metadata_source.path)]
    if name == "malware":
        import h5py
        from scipy.sparse import csr_matrix, vstack
        from sklearn.model_selection import train_test_split

        with h5py.File(assets[0], "r") as handle:
            group = handle["default"]
            row_ids = group["axis1"][:]
            if not np.array_equal(row_ids, np.arange(len(row_ids))):
                raise ValueError("Malware HDF5 row indices must be a 0..n-1 range.")
            features = [s.decode() for s in group["block0_items"][:]]
            labels = [s.decode() for s in group["block1_items"][:]]
            if labels != ["malicious"]:
                raise ValueError("Unexpected Malware HDF5 label layout.")
            y = group["block1_values"][:, 0].astype(np.int64)
            source = group["block0_values"]
            if source.shape != (len(y), len(features)):
                raise ValueError("Malware feature shape does not match labels/columns.")
            blocks = []
            for start in range(0, len(y), 2048):
                chunk = source[start:start + 2048].astype(np.float32)
                chunk[chunk == 0.0] = 0.0
                blocks.append(csr_matrix(chunk))
        x = vstack(blocks, format="csr", dtype=np.float32)
        indices = np.arange(len(y))
        val, test = train_test_split(indices[12308:], test_size=0.5, random_state=42)
        splits = {"train": indices[:12308], "val": val, "test": test}
    else:
        frame, y = dataset.get_x_y()
        features = list(frame.columns)
        splits = dataset.get_splits()
        x = np.ascontiguousarray(frame.to_numpy(dtype=np.float64)).astype(np.float32)
        y = np.asarray(y).astype(np.int64)
    return x, y, splits, features, assets


def train_targets(x_train, y_train, x_val, y_val, seed=0, class_weight=False):
    import inspect
    import xgboost as xgb
    import lightgbm as lgb

    fit_has_esr = "early_stopping_rounds" in inspect.signature(xgb.XGBClassifier.fit).parameters
    parameters = dict(n_estimators=4000, learning_rate=0.05, max_depth=8,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                      n_jobs=8, random_state=seed, eval_metric="logloss")
    if class_weight:
        parameters["scale_pos_weight"] = float((y_train == 0).sum()) / max(1, (y_train == 1).sum())
    if not fit_has_esr:
        parameters["early_stopping_rounds"] = 100
    xgb_model = xgb.XGBClassifier(**parameters)
    start = time.perf_counter()
    fit_options = {"early_stopping_rounds": 100} if fit_has_esr else {}
    xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False, **fit_options)
    print(f"[train] xgboost seconds={time.perf_counter()-start:.1f} "
          f"best_iteration={xgb_model.best_iteration}", flush=True)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, n_jobs=8,
        random_state=seed, verbosity=-1, force_row_wise=True, num_threads=8,
        **({"class_weight": "balanced"} if class_weight else {}))
    start = time.perf_counter()
    lgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"[train] lightgbm seconds={time.perf_counter()-start:.1f} "
          f"best_iteration={lgb_model.best_iteration_}", flush=True)
    return {"xgboost": xgb_model, "lightgbm": lgb_model}


def load_targets(directory):
    """Reload locally trained targets; the LightGBM object is a native Booster."""
    import xgboost as xgb
    import lightgbm as lgb

    directory = Path(directory)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(directory / "xgb.json")
    return {"xgboost": xgb_model,
            "lightgbm": lgb.Booster(model_file=str(directory / "lgbm.txt"))}


def predict_one(model, x):
    import numpy as np

    if type(model).__name__ == "Booster":
        return (model.predict(x) > 0.5).astype(np.int64)
    return np.asarray(model.predict(x)).astype(np.int64)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-weight", action="store_true",
                        help="Use XGBoost scale_pos_weight and LightGBM balanced class weights.")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/tree_targets",
                        help="Output root; a dataset/seed/regime subdirectory is created.")
    args = parser.parse_args()
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32-1.")
    regime = "balanced" if args.class_weight else "unweighted"
    output = args.out.expanduser().resolve() / args.dataset / f"seed{args.seed}" / regime
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new --out root.")

    import numpy as np
    import sklearn
    import xgboost
    import lightgbm
    from sklearn.metrics import accuracy_score, roc_auc_score
    from benchmark_adapter import TB_ROOT

    start = time.perf_counter()
    print(f"[data] loading {args.dataset}", flush=True)
    x, y, splits, features, assets = load_data(args.dataset)
    train, val, test = [np.asarray(splits[k]) for k in ("train", "val", "test")]
    if any(len(np.unique(y[indices])) != 2 for indices in (train, val, test)):
        raise ValueError("Each split must contain both binary classes.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1.")
    print(f"[data] rows={len(y)} features={x.shape[1]} "
          f"train={len(train)} val={len(val)} test={len(test)}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="training", dataset=args.dataset, seed=args.seed, class_weighted=args.class_weight,
        library_versions=dict(xgboost=xgboost.__version__, lightgbm=lightgbm.__version__,
                              sklearn=sklearn.__version__, numpy=np.__version__),
        representation="csr_f32_raw" if args.dataset == "malware" else "dense_f32_raw",
        feature_names=features,
        split_sizes={k: len(v) for k, v in splits.items()},
        split_sha256={k: hashlib.sha256(np.asarray(v, dtype=np.int64).tobytes()).hexdigest()
                      for k, v in splits.items()},
        asset_sha256={str(p): sha256_file(p) for p in assets},
        source_sha256={str(Path(__file__).resolve()): sha256_file(__file__),
                       str(ROOT / "method/benchmark_adapter.py"): sha256_file(ROOT / "method/benchmark_adapter.py")},
        tabularbench_root=str(TB_ROOT), early_stopping_rounds=100,
        training_policy="fit on train; early stopping on val; test used only for reporting",
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    models = train_targets(x[train], y[train], x[val], y[val], args.seed, args.class_weight)
    test_x, test_y = x[test], y[test]
    metrics = {}
    for name, model in models.items():
        probabilities = model.predict_proba(test_x)[:, 1]
        metrics[name] = dict(test_accuracy=float(accuracy_score(test_y, predict_one(model, test_x))),
                             test_roc_auc=float(roc_auc_score(test_y, probabilities)))
    models["xgboost"].save_model(output / "xgb.json")
    models["lightgbm"].booster_.save_model(str(output / "lgbm.txt"))
    # Check the actual files readers will load, including the early-stopping iteration.
    for name, reloaded in load_targets(output).items():
        np.testing.assert_array_equal(predict_one(models[name], test_x), predict_one(reloaded, test_x))
    parameters = {name: model.get_params() for name, model in models.items()}
    # JSON has no numeric NaN; preserve XGBoost's missing-value setting by name.
    if np.isnan(parameters["xgboost"]["missing"]):
        parameters["xgboost"]["missing"] = "NaN"
    manifest.update(status="complete", test_metrics=metrics,
                    hyperparameters=parameters,
                    best_iteration=dict(xgboost=int(models["xgboost"].best_iteration),
                                        lightgbm=int(models["lightgbm"].best_iteration_)),
                    model_sha256={name: sha256_file(output / name) for name in ("xgb.json", "lgbm.txt")},
                    seconds=round(time.perf_counter()-start, 1))
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(dict(output=str(output), test_metrics=metrics, status="complete")), flush=True)


if __name__ == "__main__":
    main()
