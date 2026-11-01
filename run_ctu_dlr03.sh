#!/bin/bash
# CTU standard-L2 entry: w_reinit with the per-cell d_lr=0.03 tuning.
# Usage: bash run_ctu_dlr03.sh --model tabnet_default
set -euo pipefail
R="$(cd -- "$(dirname -- "$0")" && pwd)"
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/.cache"
export HF_HOME="$R/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
unset WK_PATIENCE WK_AUDIT_TOL WIDS_APACHE_MAP
mkdir -p "$R/tmp"
exec "$PY" -u "$R/ctu_dlr03_run.py" "$@"
