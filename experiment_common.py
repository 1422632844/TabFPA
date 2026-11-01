"""Dataset and victim-model loaders shared by the main experiment entries."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'method')]
import numpy as np
import torch
import runtime
from core.judge import BatchJudge
from models import VictimAdapter
from benchmark_adapter import load_model_and_weights
from benchmark_adapter import get_dataset
from tabularbench.models.tab_scaler import TabScaler


def dataset_data(name):
    if name=="ctu_13_neris":
        return runtime.ctu_data()
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
