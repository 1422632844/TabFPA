#!/bin/bash
# LCLD entry (default and Madry targets): frozen 996 configuration
# (ballv2 + SE/DIR gates + chain BPDA, 100+100 schedule) on the official
# stratified 1000-row draw (random_state=42), clean-correct subset.
# Step 1 builds the draw pools once; step 2 runs one model per invocation.
# Usage: bash run_lcld_996.sh pools
#        bash run_lcld_996.sh run --model tabnet_default
#        bash run_lcld_996.sh run --model tabtransformer_madry
set -euo pipefail
R="$(cd -- "$(dirname -- "$0")" && pwd)"
PY="${PYTHON:-python}"
MODE="${1:-run}"; shift || true
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/.cache"
export HF_HOME="$R/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TABULARBENCH_ROOT="${TABULARBENCH_ROOT:-$R/tabularbench}"
export TABULARBENCH_DATASET_DIR="${TABULARBENCH_DATASET_DIR:-$R/tabularbench/datasets}"
export TABULARBENCH_TARGET_MODEL_DIR="${TABULARBENCH_TARGET_MODEL_DIR:-$R/tabularbench/target_model}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p "$R/tmp"
case "$MODE" in
  pools) exec "$PY" -u "$R/method/lcld_f0draw_all.py" "$@" ;;
  run)   exec "$PY" -u "$R/method/lcld_996_f0draw_run.py" "$@" ;;
  *) echo "usage: $0 pools | run [--model <name>]" >&2; exit 2 ;;
esac
