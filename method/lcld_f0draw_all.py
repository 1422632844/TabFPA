"""Phase-2B step 1 (2026-09-30): build official-effect f0draw pools for ALL
five LCLD models (the tabnet one already exists from lcld_officialdraw_tabnet).

Same semantics as lcld_officialdraw_tabnet.py: official effective path with
filter_correct=False (class filter -> f32 relations <=1e-9 -> stratified
train_test_split(test_size=1000, random_state=42)), then classify the draw
with the model and save the clean-correct subset (_f0draw_cc) plus the full
draw (_f0draw_full). Deterministic; refuses to overwrite existing files."""
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
TB = Path(os.environ.get("TABULARBENCH_ROOT", ROOT / "tabularbench"))
sys.path[:0] = [str(ROOT), str(TB), str(ROOT / "method")]
os.environ.setdefault("TABULARBENCH_ROOT", str(TB))
os.environ.setdefault("TABULARBENCH_DATASET_DIR", str(TB / "datasets"))
os.environ.setdefault("TABULARBENCH_TARGET_MODEL_DIR", str(TB / "target_model"))
from lcld_officialdraw_tabnet import get_x_attack_np_official  # noqa: E402
from tabularbench.benchmark.model_utils import load_model_and_weights  # noqa: E402
from tabularbench.datasets.dataset_factory import get_dataset  # noqa: E402
from tabularbench.models.tab_scaler import TabScaler  # noqa: E402
from tabularbench.utils.datatypes import to_torch_number  # noqa: E402

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
