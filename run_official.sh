#!/bin/bash
# Official frozen-default entries: native WiDS (default weights) and URL (Madry).
# Usage: bash run_official.sh --dataset wids [--models ...] [--seeds ...]
set -euo pipefail
R="$(cd -- "$(dirname -- "$0")" && pwd)"
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/.cache"
export HF_HOME="$R/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TABULARBENCH_ROOT="$R/tabularbench"
export TABULARBENCH_DATASET_DIR="$R/tabularbench/datasets"
export TABULARBENCH_TARGET_MODEL_DIR="$R/tabularbench/target_model"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
unset WK_PATIENCE WK_AUDIT_TOL WIDS_APACHE_MAP
mkdir -p "$R/tmp"
exec "$PY" -u "$R/run_official.py" "$@"
