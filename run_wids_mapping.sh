#!/bin/bash
# Additional scenario: WiDS with the added diagnosis-body-system mapping relation,
# L2 radius 0.6, full eligible pools, five seeds (default models).
# Usage: bash run_wids_mapping.sh [--models ...] [--seeds ...]
set -euo pipefail
R="$(cd -- "$(dirname -- "$0")" && pwd)"
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/.cache"
export HF_HOME="$R/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TABULARBENCH_ROOT="${TABULARBENCH_ROOT:-$R/../tabularbench}"
export TABULARBENCH_DATASET_DIR="${TABULARBENCH_DATASET_DIR:-$TABULARBENCH_ROOT/datasets}"
export TABULARBENCH_TARGET_MODEL_DIR="${TABULARBENCH_TARGET_MODEL_DIR:-$TABULARBENCH_ROOT/target_model}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p "$R/tmp"
exec "$PY" -u "$R/method/wids_mapping_wreinit.py" --eps 0.6 \
  --out "$R/runs/wids_mapping_full_wreinit_eps06" "$@"
