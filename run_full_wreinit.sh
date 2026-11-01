#!/bin/bash
# w_reinit full-pool runner: standard L2 on URL (default weights), Madry targets on
# URL/WiDS/CTU, and the Malware scenario. One model per invocation.
# Usage: bash run_full_wreinit.sh --dataset url --model tabnet_default
#        bash run_full_wreinit.sh --dataset url --model tabnet_madry
#        bash run_full_wreinit.sh --dataset malware --model tabnet_default
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
exec "$PY" -u "$R/full_wreinit_run.py" "$@"
