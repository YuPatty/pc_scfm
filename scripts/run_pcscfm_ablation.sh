#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/ecg_baseline_wander}"

cd "$APP_DIR"
python3 experiment_suite.py exp7-ablation \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --output-root "$OUTPUT_ROOT" \
  "$@"
