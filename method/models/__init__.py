"""Victim-model module: load any TabularBench pretrained model and wrap it
with the correct input convention.

Convention facts (probe-verified on all 5 architectures):
- tabnet, stg, torchrln, tabtransformer, vime: wrapper_model EMBEDS the
  scaler transform -> feed RAW features.
The runtime probe re-verifies against predict_proba on every load (16 rows,
cheap) so an unseen architecture is handled automatically.
"""
from __future__ import annotations

import collections
import collections.abc

import numpy as np
import torch

# stg's library uses pre-3.10 collections aliases
for _n in ("Sequence", "Set", "Mapping", "MutableMapping", "Iterable",
           "Callable"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

from core.data import DEVICE, setup_components  # noqa: E402


class VictimAdapter(torch.nn.Module):
    """Applies the arch-specific input convention and expands to 2-col
    logits (binary models emitting a single column become [-o, o])."""

    def __init__(self, wrapper, scaler, transform_first=False):
        super().__init__()
        self.wrapper = wrapper
        self.scaler = scaler
        self.transform_first = transform_first

    def forward(self, x_raw):
        z = self.scaler.transform(x_raw) if self.transform_first else x_raw
        out = self.wrapper(z)
        if out.shape[-1] == 1:
            out = torch.cat([-out, out], dim=1)
        return out


def load_victim(dataset_name, model_name, device=DEVICE):
    """Returns (ds, x_test, y_test, scaler, model_cpu, cons, adapter_gpu);
    adapter_gpu(x_raw) -> 2-column logits with the correct convention."""
    ds, x_test, y_test, scaler, model_cpu, cons = setup_components(
        dataset_name, model_name, "cpu")
    _, _, _, _, model_gpu, _ = setup_components(dataset_name, model_name,
                                                device)
    model_gpu.wrapper_model.eval()
    model_gpu.wrapper_model.requires_grad_(False)  # optimize inputs, not victim weights

    x_probe = x_test[:16].astype(np.float32)
    p16 = np.asarray(model_cpu.predict_proba(x_probe))
    with torch.no_grad():
        xr = torch.as_tensor(x_probe, dtype=torch.float32, device=device)
        try:
            o_raw = model_gpu.wrapper_model(xr).detach().cpu().numpy()
        except Exception:
            o_raw = None                    # arch needs transformed input
        try:
            o_tr = model_gpu.wrapper_model(
                scaler.transform(xr)).detach().cpu().numpy()
        except Exception:
            o_tr = None                     # arch embeds the transform

    def to2(o):
        if o is None:
            return None
        if o.shape[1] == 1:
            o = np.concatenate([-o, o], 1)
        return o / (np.abs(o).max(1, keepdims=True) + 1e-12)

    p16n = p16 / (p16.max(1, keepdims=True) + 1e-12)
    m_raw = float("inf") if o_raw is None else float(np.abs(to2(o_raw) - p16n).mean())
    m_tr = float("inf") if o_tr is None else float(np.abs(to2(o_tr) - p16n).mean())
    transform_first = m_tr < m_raw
    adapter = VictimAdapter(model_gpu.wrapper_model, scaler,
                            transform_first).to(device)
    conv = "transform-then-model" if transform_first else "model-embeds-transform"
    print(f"[victim] {model_name}: convention={conv} "
          f"(mismatch raw={m_raw:.3f} tr={m_tr:.3f})", flush=True)
    return ds, x_test, y_test, scaler, model_cpu, cons, adapter


# officially released pretrained victims (target_model/<dataset>/)
DEFAULT_MODELS = ["tabnet_default", "torchrln_default",
                  "tabtransformer_default", "stg_default", "vime_default"]
