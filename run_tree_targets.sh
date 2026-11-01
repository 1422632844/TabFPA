#!/bin/bash
# Train the two additional tree targets on CPU; data remains external.
set -euo pipefail
R="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${PYTHON:-python}" -u "$R/targets/train_tree_targets.py" "$@"
