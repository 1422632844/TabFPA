"""Build official-draw LCLD pools for default and Madry targets.

Apply class filtering and float32 relations <=1e-9, then a stratified
1000-row draw (random_state=42). Save the draw and each model's clean-correct
subset. Deterministic; refuses to overwrite existing files."""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
import collections
import collections.abc
for _n in ('Sequence', 'Set', 'Mapping', 'MutableMapping', 'Iterable', 'Callable'):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "method")]
from benchmark_adapter import TB_ROOT
from tabularbench.constraints.constraints_backend_executor import ConstraintsExecutor
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from benchmark_adapter import load_model_and_weights  # noqa: E402
from benchmark_adapter import get_dataset  # noqa: E402
from tabularbench.models.tab_scaler import TabScaler  # noqa: E402
from tabularbench.utils.datatypes import to_torch_number  # noqa: E402

def get_x_attack_np_official(x, y, cons, filter_class, subset):
    """Official subset_utils.get_x_attack with filter_correct=False:
    class filter -> f32 relations (<=1e-9) -> stratified split rs=42."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    m = y == filter_class
    x, y = x[m], y[m]
    executor = ConstraintsExecutor(AndConstraint(cons.relation_constraints),
                                   PytorchBackend(),
                                   feature_names=cons.feature_names)
    val = executor.execute(torch.Tensor(x.astype(np.float32)))
    ok = np.asarray(val <= 1e-9).astype(bool)
    if ok.mean() < 1:
        x, y = x[ok], y[ok]
    if subset > 0 and subset < len(y):
        from sklearn.model_selection import train_test_split
        _, x, _, y = train_test_split(x, y, test_size=subset,
                                      stratify=y, random_state=42)
    return x, y


DS, N = "lcld_v2_iid", 1000
# Default models, then the Madry models used by the adversarially-trained
# scenario. tabnet_madry draws an empty clean-correct subset (the official
# Madry checkpoint classifies no drawn row as positive); its pool is still
# built and kept for provenance.
MODELS = ["torchrln_default", "stg_default", "vime_default",
          "tabnet_default", "tabtransformer_default",
          "torchrln_madry", "stg_madry", "vime_madry",
          "tabnet_madry", "tabtransformer_madry"]
OUT = ROOT / "reference/caa_pools"
OUT.mkdir(parents=True, exist_ok=True)

ds = get_dataset(DS)
x, y = ds.get_x_y()
i_test = ds.get_splits()["test"]
x_test = x.iloc[i_test].to_numpy()
y_test = y[i_test]
metadata = ds.get_metadata(only_x=True)
scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
scaler.fit(to_torch_number(x), x_type=metadata["type"])
cons = ds.get_constraints()
x_draw, y_draw = get_x_attack_np_official(x_test, y_test, cons, 1, N)
assert x_draw.shape == (N, x_test.shape[1])

for model in MODELS:
    arch, training = model.split("_")[0], "_".join(model.split("_")[1:])
    m = load_model_and_weights(ds.name, arch, training, metadata, scaler, "cpu")
    pred = m.predict(x_draw)
    correct = np.asarray(pred == y_draw, bool)
    full = OUT / f"{DS}__{model}_f0draw_full.npz"
    cc = OUT / f"{DS}__{model}_f0draw_cc.npz"
    info = dict(dataset=DS, model=model, n_draw=int(len(x_draw)),
                filter_correct=False,
                n_clean_correct=int(correct.sum()),
                n_wrong=int((~correct).sum()),
                sampler="official get_x_attack with filter_correct=False: "
                        "f32 relations + train_test_split(stratify, random_state=42)")
    if cc.exists() and full.exists():
        print("OFFICIALDRAW_EXISTS", json.dumps(info), flush=True)
        continue
    assert not full.exists() and not cc.exists(), "refuse to overwrite: " + model
    np.savez_compressed(full, clean=x_draw, pred=np.asarray(pred),
                        info=np.array(json.dumps(info)))
    np.savez_compressed(cc, clean=x_draw[correct], info=np.array(json.dumps(info)))
    print("OFFICIALDRAW_BUILT", json.dumps(info), flush=True)
    print("sha256_cc", hashlib.sha256(cc.read_bytes()).hexdigest(), flush=True)
    del m
print("F0DRAW_ALL_DONE", flush=True)
