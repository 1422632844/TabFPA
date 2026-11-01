"""Connect TabFPA to an external TabularBench checkout and local assets.

Dataset definitions, splits, constraints and model classes remain upstream.
Only asset paths are adapted; downloads and target training are not performed.
"""
import collections
import collections.abc
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import scipy.integrate as integrate

REPO = Path(__file__).resolve().parents[1]
TB_ROOT = Path(os.environ.get("TABULARBENCH_ROOT", REPO.parent / "tabularbench")).expanduser().resolve()
if not (TB_ROOT / "tabularbench" / "datasets" / "dataset.py").is_file():
    raise ImportError("Set TABULARBENCH_ROOT to an external official TabularBench checkout; see README.md.")
sys.path.insert(0, str(TB_ROOT))
os.environ.setdefault("TABULARBENCH_ROOT", str(TB_ROOT))
os.environ.setdefault("TABULARBENCH_DATASET_DIR", str(TB_ROOT / "datasets"))
os.environ.setdefault("TABULARBENCH_TARGET_MODEL_DIR", str(TB_ROOT / "target_model"))

# Compatibility for upstream STG/lifelines with Python 3.10 and newer SciPy.
for name in ("Sequence", "Set", "Mapping", "MutableMapping", "Iterable", "Callable"):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))
if not hasattr(integrate, "trapz"):
    integrate.trapz = np.trapz


def _dataset_file(repo_id, filename, repo_type="dataset", **kwargs):
    if repo_id != "serval-uni-lu/tabularbench" or repo_type != "dataset":
        raise ValueError("Unexpected upstream dataset source")
    path = Path(os.environ["TABULARBENCH_DATASET_DIR"]).expanduser() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing TabularBench asset: {path}. Obtain it from the upstream project.")
    return str(path)


def get_dataset(name):
    from tabularbench.datasets.dataset import HuggingFaceDataSource
    from tabularbench.datasets.dataset_factory import get_dataset as upstream_dataset

    # Some upstream factories read metadata while constructing constraints.
    with patch("tabularbench.datasets.dataset.hf_hub_download", _dataset_file):
        dataset = upstream_dataset(name)
    for attr in ("data_source", "metadata_source"):
        source = getattr(dataset, attr)
        if isinstance(source, HuggingFaceDataSource):
            source.file_data_source.path = _dataset_file(source.repo_id, source.filename, source.repo_type)
            setattr(dataset, attr, source.file_data_source)
    return dataset


def get_local_model_path(dataset_name, model_arch, training_name):
    from tabularbench.models.aliases import ALIASES as architectures
    from tabularbench.dataloaders.aliases import ALIASES as training

    arch = architectures.get(model_arch, model_arch)
    regime = training.get(training_name, training_name)
    root = Path(os.environ["TABULARBENCH_TARGET_MODEL_DIR"]).expanduser()
    folder = f"{arch}_{dataset_name}_{regime}.model"
    for path in (root / dataset_name / folder, root / folder):
        if path.is_dir():
            return str(path)
    raise FileNotFoundError(f"Missing pretrained target: {root / dataset_name / folder}. See README.md.")


def load_model_and_weights(dataset_name, model_arch, training_name, metadata, scaler, device):
    from tabularbench.models.model_factory import load_model

    model_class = load_model(model_arch)
    return model_class.load_class(
        get_local_model_path(dataset_name, model_arch, training_name),
        x_metadata=metadata, scaler=scaler, force_device=device or None,
    )
