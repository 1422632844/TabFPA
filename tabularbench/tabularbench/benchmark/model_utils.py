import os
from pathlib import Path

import pandas as pd

from tabularbench.dataloaders.aliases import ALIASES as TRAINING_ALIASES
from tabularbench.models.aliases import ALIASES as ARCH_ALIASES
from tabularbench.models.model import Model
from tabularbench.models.model_factory import load_model
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.utils.huggingface import hf_hub_download_folder


def get_local_model_path(
    dataset_name: str, model_arch: str, training_name: str
) -> str:
    model_root = Path(
        os.environ.get("TABULARBENCH_TARGET_MODEL_DIR", "target_model")
    ).expanduser()
    model_folder = f"{model_arch}_{dataset_name}_{training_name}.model"
    candidates = [
        model_root / dataset_name / model_folder,
        model_root / model_folder,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def load_model_and_weights(
    dataset_name: str,
    model_arch: str,
    training_name: str,
    metadata: pd.DataFrame,
    scaler: TabScaler,
    device: str,
) -> Model:

    # Load model
    model_class = load_model(model_arch)
    model_arch = ARCH_ALIASES.get(model_arch, model_arch)
    training_name = TRAINING_ALIASES.get(training_name, training_name)

    weight_path = get_local_model_path(dataset_name, model_arch, training_name)
    if not weight_path:
        weight_path = hf_hub_download_folder(
            "serval-uni-lu/tabularbench/"
            f"{dataset_name}/"
            f"{model_arch}_{dataset_name}_{training_name}.model"
        )

    force_device = device if device != "" else None
    model = model_class.load_class(
        weight_path,
        x_metadata=metadata,
        scaler=scaler,
        force_device=force_device,
    )

    return model
