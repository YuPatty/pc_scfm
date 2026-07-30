#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
DATA_ROOT="$ROOT_DIR/data/ecg_baseline_wander"
PTBXL_ROOT="$DATA_ROOT/raw/PTBXL"
NOISE_DIR="$DATA_ROOT/raw/NSTDB"

CONFIG="${CONFIG:-configs/ecg_baseline_wander_pc_scfm.yaml}"
RECORDS_DIR="${RECORDS_DIR:-$PTBXL_ROOT/records100}"
METADATA_CSV="${METADATA_CSV:-$PTBXL_ROOT/ptbxl_database.csv}"

cd "$APP_DIR"

cmd=(
  python3 preprocess_ecg.py
  --config "$CONFIG"
  --input-dir "$RECORDS_DIR"
  --metadata-csv "$METADATA_CSV"
  --dataset-name ptbxl
)

if find "$NOISE_DIR" -type f -print -quit 2>/dev/null | grep -q .; then
  cmd+=(--noise-dir "$NOISE_DIR")
else
  cmd+=(--baseline-kind random_low_frequency_drift)
fi

"${cmd[@]}" "$@"
