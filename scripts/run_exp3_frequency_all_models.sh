#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
DATA_ROOT="$ROOT_DIR/data/ecg_baseline_wander"
PTBXL_ROOT="$DATA_ROOT/raw/PTBXL"

source "$ROOT_DIR/scripts/experiment_models.sh"

cd "$APP_DIR"

for item in "${EXPERIMENT_MODELS[@]}"; do
  IFS="|" read -r name config exp_name model_dir <<< "$item"
  checkpoint="$ROOT_DIR/runs/ecg_baseline_wander/checkpoint/$exp_name/$model_dir/best_pcc_model.pt"
  if [[ "$model_dir" == "fir_filter" || "$model_dir" == "iir_filter" ]]; then
    checkpoint="__classical_filter_no_checkpoint__"
  fi

  if [[ "$checkpoint" != "__classical_filter_no_checkpoint__" && ! -f "$checkpoint" ]]; then
    echo "MISSING checkpoint: $checkpoint" >&2
    exit 1
  fi

  python3 experiment_suite.py exp3-frequency \
    --config "$config" \
    --input-dir "$PTBXL_ROOT/records100" \
    --metadata-csv "$PTBXL_ROOT/ptbxl_database.csv" \
    --checkpoint "$checkpoint" \
    --output-root "$ROOT_DIR/runs/ecg_baseline_wander/controlled_tests/$name" \
    --baseline-kind sinusoidal \
    --alpha-value 0.2 \
    --frequencies-hz 0.05,0.1,0.2,0.3,0.5,0.8,1.0 \
    "$@"
done
