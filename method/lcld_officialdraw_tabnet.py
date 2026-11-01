"""Build the official-effect draw (filter_correct=False) for LCLD/TabNet.

Self-contained: line-for-line port of the tabularbench effective path
(class filter -> f32 relation constraints <=1e-9 -> stratified
train_test_split(test_size=1000, random_state=42)), with NO clean-correct
filter -- the code default the published numbers carry, per
COMPARISON_PROTOCOL_AUDIT_20260925.md SS2. Then classifies the draw with the
TabNet victim and saves:
  (a) reference/caa_pools/lcld_v2_iid__tabnet_default_f0draw_full.npz
      -- the full 1000-row draw plus per-row predictions;
  (b) reference/caa_pools/lcld_v2_iid__tabnet_default_f0draw_cc.npz
      -- the clean-correct subset in the attack-pool format.
Deterministic. Writes two NEW files only; never touches existing pools.
"""
import sys, os, json, hashlib
from pathlib import Path
import numpy as np
import torch
import collections
import collections.abc
for _n in ('Sequence', 'Set', 'Mapping', 'MutableMapping', 'Iterable', 'Callable'):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

ROOT = Path(__file__).resolve().parents[1]
TB = Path(os.environ['TABULARBENCH_ROOT'])
sys.path[:0] = [str(TB), str(ROOT)]
# Offline dataset load exactly like run_experiments.py does (line 6): without
# this the factory falls back to downloading from the HF hub.
os.environ.setdefault("TABULARBENCH_DATASET_DIR", str(ROOT / "tabularbench/datasets"))
from tabularbench.benchmark.model_utils import load_model_and_weights
from tabularbench.constraints.constraints_backend_executor import ConstraintsExecutor
from tabularbench.constraints.pytorch_backend import PytorchBackend
from tabularbench.constraints.relation_constraint import AndConstraint
from tabularbench.datasets.dataset_factory import get_dataset
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.utils.datatypes import to_torch_number

OUT = ROOT / 'reference/caa_pools'
DS, MODEL = 'lcld_v2_iid', 'tabnet_default'
N = 1000


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


def main():
    ds = get_dataset(DS)
    x, y = ds.get_x_y()
    i_test = ds.get_splits()['test']
    x_test = x.iloc[i_test].to_numpy()
    y_test = y[i_test]
    metadata = ds.get_metadata(only_x=True)
    scaler = TabScaler(num_scaler='min_max', one_hot_encode=True)
    scaler.fit(to_torch_number(x), x_type=metadata['type'])
    arch, training = MODEL.split('_')[0], '_'.join(MODEL.split('_')[1:])
    model = load_model_and_weights(ds.name, arch, training, metadata, scaler, 'cpu')
    cons = ds.get_constraints()

    x_draw, y_draw = get_x_attack_np_official(x_test, y_test, cons, 1, N)
    assert x_draw.shape == (N, x_test.shape[1]), x_draw.shape
    pred = model.predict(x_draw)
    correct = np.asarray(pred == y_draw, bool)

    full = OUT / f'{DS}__{MODEL}_f0draw_full.npz'
    cc = OUT / f'{DS}__{MODEL}_f0draw_cc.npz'
    if full.exists() and cc.exists():
        with np.load(full, allow_pickle=False) as zf:
            info = json.loads(zf['info'].item())
        print('OFFICIALDRAW_EXISTS', json.dumps(info), flush=True)
        return
    assert not full.exists() and not cc.exists(), 'refuse to overwrite existing draw files'
    info = dict(dataset=DS, model=MODEL, n_draw=int(len(x_draw)),
                filter_correct=False,
                n_clean_correct=int(correct.sum()),
                n_wrong=int((~correct).sum()),
                sampler='official get_x_attack with filter_correct=False: '
                        'f32 relations + train_test_split(stratify, random_state=42)')
    np.savez_compressed(full, clean=x_draw, pred=np.asarray(pred),
                        info=np.array(json.dumps(info)))
    np.savez_compressed(cc, clean=x_draw[correct], info=np.array(json.dumps(info)))
    print('OFFICIALDRAW_BUILT', json.dumps(info), flush=True)
    print('sha256_full', hashlib.sha256(full.read_bytes()).hexdigest(), flush=True)
    print('sha256_cc', hashlib.sha256(cc.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
