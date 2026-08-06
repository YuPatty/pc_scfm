#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"

cd "$APP_DIR"

python3 train_supervised.py \
  --config configs/ecg_baseline_wander_mambattention_no_time_attention.yaml \
  "$@"

python3 train_supervised.py \
  --config configs/ecg_baseline_wander_mambattention_no_freq_attention.yaml \
  "$@"
