"""Dataset registry. Each dataset module exposes:
    - FeasibleMap(meta, device)   -> callable F(x0, delta) with .nf_free
      (optionally .feasible(x) as the exact per-step guard)
    - finalize(x_clean64, x_opt32, meta) -> float64-consistent rebuild
    - JUDGE_KWARGS                -> per-dataset BatchJudge options
Add a new dataset by dropping a module here and registering it below."""
from __future__ import annotations

from . import ctu_13_neris as ctu_mod
from . import lcld as lcld_mod
from . import malware as malware_mod
from . import url as url_mod
from . import wids as wids_mod

REGISTRY = {
    "lcld_v2_iid": dict(
        map=lcld_mod.FeasibleMap,
        finalize=lcld_mod.finalize_f64,
        feasible_fn=None,
        judge_kwargs=lcld_mod.JUDGE_KWARGS,
    ),
    # lcld variants share features and constraints -> reuse the lcld module
    "lcld_iid": dict(map=lcld_mod.FeasibleMap, finalize=lcld_mod.finalize_f64,
                     feasible_fn=None, judge_kwargs=lcld_mod.JUDGE_KWARGS),
    "lcld_time": dict(map=lcld_mod.FeasibleMap, finalize=lcld_mod.finalize_f64,
                      feasible_fn=None, judge_kwargs=lcld_mod.JUDGE_KWARGS),
    "lcld_v2_time": dict(map=lcld_mod.FeasibleMap,
                         finalize=lcld_mod.finalize_f64, feasible_fn=None,
                         judge_kwargs=lcld_mod.JUDGE_KWARGS),
    "lcld_201317_time": dict(map=lcld_mod.FeasibleMap,
                             finalize=lcld_mod.finalize_f64, feasible_fn=None,
                             judge_kwargs=lcld_mod.JUDGE_KWARGS),
    "lcld_201317_ds_time": dict(map=lcld_mod.FeasibleMap,
                                finalize=lcld_mod.finalize_f64,
                                feasible_fn=None,
                                judge_kwargs=lcld_mod.JUDGE_KWARGS),
    "ctu_13_neris": dict(map=ctu_mod.CtuFeasibleMap,
                         finalize=ctu_mod.finalize_ctu_f64,
                         feasible_fn="feasible",
                         judge_kwargs=ctu_mod.JUDGE_KWARGS),
    "malware": dict(  # UNTESTED: no local data (HF download on first use)
        map=malware_mod.MalwareFeasibleMap,
        finalize=malware_mod.finalize_malware_f64,
        feasible_fn="feasible",
        judge_kwargs=malware_mod.JUDGE_KWARGS),
    "url": dict(
        map=url_mod.UrlFeasibleMap,
        finalize=url_mod.finalize_url_f64,
        feasible_fn="feasible",
        judge_kwargs=url_mod.JUDGE_KWARGS,
    ),
    "wids": dict(
        map=wids_mod.WidsFeasibleMap,
        finalize=wids_mod.finalize_wids_f64,
        feasible_fn=None,
        judge_kwargs=wids_mod.JUDGE_KWARGS,
    ),
}

UNADAPTED: dict = {}


def get_dataset_module(name: str) -> dict:
    if name in REGISTRY:
        return REGISTRY[name]
    if name in UNADAPTED:
        raise NotImplementedError(
            f"dataset '{name}' not yet adapted: {UNADAPTED[name]}")
    raise NotImplementedError(
        f"unknown dataset '{name}'. Adapted: {sorted(REGISTRY)}. "
        "To add one: create datasets/<name>.py (wids.py is the minimal "
        "template) and register it in datasets/__init__.py.")
